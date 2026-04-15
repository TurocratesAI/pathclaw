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

    async def send(self, chat_id: int, text: str, parse_mode: str | None = "Markdown") -> int | None:
        # Telegram messages cap at 4096 characters; returns the last sent message_id
        chunks = [text[i: i + 3900] for i in range(0, max(len(text), 1), 3900)] or [""]
        last_id: int | None = None
        for chunk in chunks:
            try:
                r = await self.client.post(
                    f"{self.base}/sendMessage",
                    json={"chat_id": chat_id, "text": chunk, "parse_mode": parse_mode},
                )
                if r.status_code == 200:
                    last_id = (r.json().get("result") or {}).get("message_id")
                    continue
            except Exception:
                pass
            # retry once without markdown in case formatting broke it
            try:
                r = await self.client.post(
                    f"{self.base}/sendMessage",
                    json={"chat_id": chat_id, "text": chunk},
                )
                if r.status_code == 200:
                    last_id = (r.json().get("result") or {}).get("message_id")
            except Exception:
                pass
        return last_id

    async def edit(self, chat_id: int, message_id: int, text: str, parse_mode: str | None = None) -> bool:
        if len(text) > 3900:
            text = text[:3897] + "..."
        try:
            r = await self.client.post(
                f"{self.base}/editMessageText",
                json={"chat_id": chat_id, "message_id": message_id, "text": text, "parse_mode": parse_mode},
            )
            return r.status_code == 200
        except Exception:
            return False

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


def _short_args(args: dict | str, limit: int = 120) -> str:
    if isinstance(args, str):
        s = args
    else:
        try:
            s = json.dumps(args, default=str, separators=(",", ":"))
        except Exception:
            s = str(args)
    return s if len(s) <= limit else s[: limit - 1] + "…"


async def _stream_to_session(
    api: "TelegramAPI",
    chat_id: int,
    base: str,
    session_id: str,
    message: str,
) -> None:
    """Open SSE stream to /api/chat/stream and live-edit a Telegram message with
    tool calls and token deltas as events arrive. Flushes at most every
    ~1.2s to stay under Telegram's edit rate limit."""
    header = "_thinking…_"
    msg_id = await api.send(chat_id, header)
    if msg_id is None:
        return

    trace_lines: list[str] = []          # `• tool_name(args)` / `  ↳ result / Nms`
    tool_starts: dict[int, float] = {}   # trace-line idx → monotonic start time
    current_tool_idx: int | None = None
    reply_buffer = ""                    # accumulated token stream
    last_edit = 0.0
    MIN_EDIT_GAP = 1.2                   # seconds between edits
    EDIT_TAIL_MAX = 3600                 # leave headroom under 4096

    def _render() -> str:
        parts = []
        if trace_lines:
            parts.append("\n".join(trace_lines))
        if reply_buffer:
            if parts:
                parts.append("")
            parts.append(reply_buffer.strip())
        text = "\n".join(parts) or "_thinking…_"
        if len(text) > EDIT_TAIL_MAX:
            text = "…" + text[-(EDIT_TAIL_MAX - 1):]
        return text

    async def _maybe_flush(force: bool = False) -> None:
        nonlocal last_edit
        now = asyncio.get_event_loop().time()
        if not force and (now - last_edit) < MIN_EDIT_GAP:
            return
        last_edit = now
        await api.edit(chat_id, msg_id, _render())

    import time as _time
    try:
        async with httpx.AsyncClient(timeout=None) as sc:
            async with sc.stream(
                "POST",
                f"{base}/api/chat/stream",
                json={"message": message, "session_id": session_id},
                timeout=None,
            ) as resp:
                if resp.status_code != 200:
                    await api.edit(chat_id, msg_id, f"(backend returned {resp.status_code})")
                    return
                async for raw in resp.aiter_lines():
                    if not raw or not raw.startswith("data:"):
                        continue
                    try:
                        evt = json.loads(raw[5:].strip())
                    except Exception:
                        continue

                    t = evt.get("type")
                    if t == "tool_start":
                        name = evt.get("name", "?")
                        args = _short_args(evt.get("args", {}))
                        trace_lines.append(f"• {name}({args})")
                        current_tool_idx = len(trace_lines) - 1
                        tool_starts[current_tool_idx] = _time.monotonic()
                        await _maybe_flush()
                    elif t == "tool_result":
                        dur_ms = evt.get("duration_ms", 0)
                        dur = f"{dur_ms/1000:.1f}s" if dur_ms >= 1000 else f"{dur_ms}ms"
                        res = (evt.get("result") or "").strip().splitlines()
                        first = res[0] if res else ""
                        summary = first[:140] + ("…" if len(first) > 140 else "")
                        if current_tool_idx is not None:
                            trace_lines[current_tool_idx] += f"  — {dur}"
                            if summary:
                                trace_lines.append(f"  ↳ {summary}")
                        current_tool_idx = None
                        await _maybe_flush()
                    elif t == "status":
                        # polling updates (wait_for_job) — don't accumulate, just refresh
                        if current_tool_idx is not None and trace_lines:
                            msg = evt.get("message", "")
                            tail = trace_lines[-1]
                            # replace any previous "  … <status>" line instead of piling up
                            if tail.startswith("  … "):
                                trace_lines[-1] = f"  … {msg}"
                            else:
                                trace_lines.append(f"  … {msg}")
                        await _maybe_flush()
                    elif t == "token":
                        reply_buffer += evt.get("content", "")
                        await _maybe_flush()
                    elif t == "code_exec":
                        code = (evt.get("code") or "").strip().splitlines()
                        head = code[0] if code else ""
                        trace_lines.append(f"• run_python: {head[:140]}")
                        current_tool_idx = len(trace_lines) - 1
                        tool_starts[current_tool_idx] = _time.monotonic()
                        await _maybe_flush()
                    elif t == "error":
                        trace_lines.append(f"⚠ {evt.get('message', 'error')}")
                        await _maybe_flush(force=True)
                    elif t == "done":
                        break
    except Exception as e:
        trace_lines.append(f"⚠ stream error: {e}")

    # Final flush — ensure the message reflects the end state
    await api.edit(chat_id, msg_id, _render())


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
            slug = s.get("slug") or ""
            title = s.get("title") or "(untitled)"
            label = f"`{slug}`" if slug else f"`{sid}`"
            mark = " ← bound" if sid == bound else ""
            lines.append(f"• {label} — {title}{mark}")
        lines.append("\n_Use_ `/session <slug-or-id>` _to bind this chat._")
        await api.send(chat_id, "\n".join(lines))
        return

    if text.startswith("/session"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await api.send(chat_id, "Usage: /session <slug-or-id>")
            return
        target = parts[1].strip()
        sessions = await _list_sessions(client, base)
        # Resolve: slug exact, then session_id exact, then session_id prefix (>=4 chars)
        resolved = ""
        for s in sessions:
            if s.get("slug") == target:
                resolved = s.get("session_id", "")
                break
        if not resolved:
            for s in sessions:
                if s.get("session_id") == target:
                    resolved = target
                    break
        if not resolved and len(target) >= 4:
            prefix_hits = [s.get("session_id", "") for s in sessions if s.get("session_id", "").startswith(target)]
            if len(prefix_hits) == 1:
                resolved = prefix_hits[0]
        if not resolved:
            await api.send(chat_id, f"Unknown session `{target}`. Use /sessions to list.")
            return
        bindings[key] = resolved
        _save_state(state)
        await api.send(chat_id, f"Bound this chat to session `{resolved}`.")
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

    await _stream_to_session(api, chat_id, base, bound, text)


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
