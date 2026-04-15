"""Standalone Telegram bot worker.

Spawned as a detached subprocess by api/routes/telegram.py when a token is
configured. Each message from the user is routed to a PathClaw chat session
(default or selected) and the agent's reply is streamed back to Telegram.

Commands:
    /start           welcome message + help
    /sessions        list all PathClaw sessions
    /session <id>    switch the current Telegram chat to a PathClaw session
    /new <title>     create a new PathClaw session
    /status          show currently-bound session + running jobs
    /help            show this help

Any non-command text is sent to the currently-bound session as a user message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pathclaw.telegram")


PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
STATE_DIR = PATHCLAW_DATA_DIR / "telegram"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
CONFIG_FILE = PATHCLAW_DATA_DIR / "config.json"


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def _allowed_usernames() -> set[str]:
    cfg = _load_config()
    raw = cfg.get("telegram_allowed_usernames") or ""
    return {u.strip().lstrip("@").lower() for u in raw.split(",") if u.strip()}


def _passcode() -> str:
    return (_load_config().get("telegram_passcode") or "").strip()


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_FILE)


def _backend_base() -> str:
    port_file = PATHCLAW_DATA_DIR / "server.port"
    if port_file.exists():
        try:
            return f"http://localhost:{port_file.read_text().strip()}"
        except Exception:
            pass
    return "http://localhost:8101"


# --------------------------------------------------------------------------
# Telegram API helpers (tiny httpx wrapper so we don't need the full SDK)
# --------------------------------------------------------------------------

class TelegramAPI:
    def __init__(self, token: str):
        self.base = f"https://api.telegram.org/bot{token}"
        self.client = httpx.AsyncClient(timeout=60.0)

    async def send(self, chat_id: int, text: str, parse_mode: str | None = "Markdown") -> None:
        # Telegram messages cap at 4096 characters
        chunks = [text[i: i + 3900] for i in range(0, max(len(text), 1), 3900)] or [""]
        for chunk in chunks:
            try:
                await self.client.post(
                    f"{self.base}/sendMessage",
                    json={"chat_id": chat_id, "text": chunk, "parse_mode": parse_mode},
                )
            except Exception:
                # retry once without markdown in case formatting broke it
                await self.client.post(
                    f"{self.base}/sendMessage",
                    json={"chat_id": chat_id, "text": chunk},
                )

    async def get_updates(self, offset: int, timeout: int = 30) -> list[dict]:
        try:
            r = await self.client.get(
                f"{self.base}/getUpdates",
                params={"offset": offset, "timeout": timeout},
            )
            if r.status_code != 200:
                return []
            return r.json().get("result", [])
        except Exception as e:
            log.warning("getUpdates failed: %s", e)
            return []

    async def close(self) -> None:
        await self.client.aclose()


# --------------------------------------------------------------------------
# PathClaw backend helpers
# --------------------------------------------------------------------------

async def _list_sessions(client: httpx.AsyncClient, base: str) -> list[dict]:
    try:
        r = await client.get(f"{base}/api/chat/history", timeout=10.0)
        if r.status_code == 200:
            return r.json().get("chats", [])
    except Exception:
        pass
    return []


async def _send_to_session(client: httpx.AsyncClient, base: str, session_id: str, message: str) -> str:
    """Send message to the PathClaw non-streaming chat endpoint and return the reply text."""
    try:
        r = await client.post(
            f"{base}/api/chat/",
            json={"message": message, "session_id": session_id},
            timeout=600.0,
        )
        if r.status_code != 200:
            return f"(backend returned {r.status_code})"
        data = r.json()
        return data.get("response") or data.get("reply") or "(empty reply)"
    except Exception as e:
        return f"(error contacting backend: {e})"


async def _system_status(client: httpx.AsyncClient, base: str) -> str:
    try:
        r = await client.get(f"{base}/api/status", timeout=10.0)
        d = r.json()
        gpu = d.get("gpu", {})
        gpu_s = f"{gpu.get('name','?')} x{gpu.get('count',0)}" if gpu.get("available") else "CPU only"
        return f"Backend online. GPU: {gpu_s}"
    except Exception as e:
        return f"Backend unreachable: {e}"


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

HELP = (
    "*PathClaw lab bot*\n\n"
    "Ask me anything — I forward your message to the currently-bound session.\n\n"
    "Commands:\n"
    "/sessions — list all sessions\n"
    "/session <id> — switch to a session\n"
    "/new <title> — create a new session\n"
    "/status — show current binding + backend status\n"
    "/help — show this help"
)


async def _handle_message(api: TelegramAPI, client: httpx.AsyncClient, base: str, msg: dict) -> None:
    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    if not chat_id:
        return
    text = (msg.get("text") or "").strip()
    if not text:
        return

    from_user = msg.get("from", {}) or {}
    username = (from_user.get("username") or "").lower()

    state = _load_state()
    bindings = state.setdefault("bindings", {})  # chat_id -> session_id
    authorized = set(state.setdefault("authorized_chats", []))  # chat_ids past the passcode gate
    key = str(chat_id)

    # --- Access control ---
    allowed = _allowed_usernames()
    if allowed and username not in allowed:
        await api.send(chat_id, "This bot is restricted — your Telegram username is not on the allow list. Ask the owner to add @" + (username or "you") + " in PathClaw → Connect Telegram.")
        return

    passcode = _passcode()
    if passcode and chat_id not in authorized:
        # Require the first message (usually /start) to include the passcode
        parts = text.split(maxsplit=1)
        supplied = parts[1].strip() if len(parts) > 1 and parts[0].lower() in ("/start", "/auth") else ""
        if supplied == passcode:
            authorized.add(chat_id)
            state["authorized_chats"] = list(authorized)
            _save_state(state)
            await api.send(chat_id, "Authorized. " + HELP)
            return
        await api.send(chat_id, "This bot is passcode-protected. Send `/start <passcode>` to authorize.")
        return

    # --- Commands ---
    if text == "/start" or text == "/help" or text.startswith("/start "):
        await api.send(chat_id, HELP)
        return

    if text.startswith("/sessions"):
        sessions = await _list_sessions(client, base)
        if not sessions:
            await api.send(chat_id, "No PathClaw sessions yet. Create one with /new <title> or from the web UI.")
            return
        bound = bindings.get(key)
        lines = ["*PathClaw sessions:*"]
        for s in sessions:
            sid = s.get("session_id", "")
            title = s.get("title") or "(untitled)"
            mark = " ← bound" if sid == bound else ""
            lines.append(f"• `{sid}` — {title}{mark}")
        await api.send(chat_id, "\n".join(lines))
        return

    if text.startswith("/session"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await api.send(chat_id, "Usage: /session <session_id>")
            return
        target = parts[1].strip()
        sessions = await _list_sessions(client, base)
        valid = any(s.get("session_id") == target for s in sessions)
        if not valid:
            await api.send(chat_id, f"Unknown session `{target}`. Use /sessions to list.")
            return
        bindings[key] = target
        _save_state(state)
        await api.send(chat_id, f"Bound this chat to session `{target}`.")
        return

    if text.startswith("/new"):
        parts = text.split(maxsplit=1)
        title = parts[1].strip() if len(parts) > 1 else "Telegram session"
        try:
            r = await client.post(f"{base}/api/chat/sessions", json={"title": title}, timeout=10.0)
            if r.status_code == 200:
                sid = r.json().get("session_id")
                bindings[key] = sid
                _save_state(state)
                await api.send(chat_id, f"Created and bound to session `{sid}` — {title}")
            else:
                await api.send(chat_id, f"Failed to create session ({r.status_code})")
        except Exception as e:
            await api.send(chat_id, f"Failed to create session: {e}")
        return

    if text.startswith("/status"):
        status = await _system_status(client, base)
        bound = bindings.get(key, "(not bound — use /sessions)")
        await api.send(chat_id, f"Session: `{bound}`\n{status}")
        return

    # --- Plain message: forward to bound session ---
    bound = bindings.get(key)
    if not bound:
        # Auto-bind to the most recent session if any exists
        sessions = await _list_sessions(client, base)
        if sessions:
            bound = sessions[0].get("session_id")
            bindings[key] = bound
            _save_state(state)
            await api.send(chat_id, f"(auto-bound to session `{bound}`)")
        else:
            await api.send(chat_id, "No PathClaw sessions yet. Create one with /new <title>.")
            return

    await api.send(chat_id, "_thinking..._")
    reply = await _send_to_session(client, base, bound, text)
    await api.send(chat_id, reply)


async def run(token: str) -> None:
    api = TelegramAPI(token)
    client = httpx.AsyncClient()
    base = _backend_base()
    log.info("Telegram bot running. Backend=%s", base)

    # skip any updates that accumulated while the bot was offline
    offset = 0
    initial = await api.get_updates(offset, timeout=0)
    if initial:
        offset = max(u["update_id"] for u in initial) + 1

    try:
        while True:
            updates = await api.get_updates(offset, timeout=30)
            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message")
                if msg:
                    try:
                        await _handle_message(api, client, base, msg)
                    except Exception as e:
                        log.exception("handler failed: %s", e)
    finally:
        await api.close()
        await client.aclose()


def main() -> int:
    token = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
    if not token:
        print("Usage: telegram_bot <token>", file=sys.stderr)
        return 2
    try:
        asyncio.run(run(token))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
