"""Telegram bot management — start/stop/status of the bot worker subprocess."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
TG_DIR = PATHCLAW_DATA_DIR / "telegram"
TG_DIR.mkdir(parents=True, exist_ok=True)
PID_FILE = TG_DIR / "bot.pid"
LOG_FILE = TG_DIR / "bot.log"
CFG_FILE = PATHCLAW_DATA_DIR / "config.json"

router = APIRouter()


class StartRequest(BaseModel):
    token: str = ""
    allowed_usernames: str = ""  # comma-separated, @ optional; empty = anyone
    passcode: str = ""           # if set, users must send /start <passcode> to authorize their chat


def _load_token() -> str:
    if not CFG_FILE.exists():
        return ""
    try:
        cfg = json.loads(CFG_FILE.read_text())
        return (cfg.get("telegram_token") or "").strip()
    except Exception:
        return ""


def _alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _current_pid() -> int:
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except Exception:
            return 0
    return 0


@router.get("/status")
async def status():
    pid = _current_pid()
    running = _alive(pid)
    cfg = {}
    if CFG_FILE.exists():
        try:
            cfg = json.loads(CFG_FILE.read_text())
        except Exception:
            pass
    return {
        "running": running,
        "pid": pid if running else 0,
        "token_set": bool(cfg.get("telegram_token")),
        "allowed_usernames": cfg.get("telegram_allowed_usernames", ""),
        "passcode_set": bool(cfg.get("telegram_passcode")),
    }


@router.post("/start")
async def start(req: StartRequest):
    token = req.token.strip() or _load_token()
    if not token:
        raise HTTPException(400, "No Telegram token provided or saved in settings")
    # Stop any running bot first
    pid = _current_pid()
    if _alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    # Persist the token to config so reboots can restart automatically
    cfg = {}
    if CFG_FILE.exists():
        try:
            cfg = json.loads(CFG_FILE.read_text())
        except Exception:
            cfg = {}
    cfg["telegram_token"] = token
    cfg["telegram_allowed_usernames"] = req.allowed_usernames.strip()
    cfg["telegram_passcode"] = req.passcode.strip()
    CFG_FILE.write_text(json.dumps(cfg, indent=2))

    # Spawn the bot worker
    env = dict(os.environ)
    env["TELEGRAM_BOT_TOKEN"] = token
    with open(LOG_FILE, "a") as log_f:
        proc = subprocess.Popen(
            [sys.executable, "-m", "pathclaw.telegram_bot"],
            stdout=log_f,
            stderr=log_f,
            start_new_session=True,
            env=env,
        )
    PID_FILE.write_text(str(proc.pid))
    return {"ok": True, "pid": proc.pid}


@router.post("/stop")
async def stop():
    pid = _current_pid()
    if not _alive(pid):
        return {"ok": True, "running": False}
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        raise HTTPException(500, f"Failed to stop bot: {e}")
    PID_FILE.unlink(missing_ok=True)
    return {"ok": True, "running": False, "pid": pid}


@router.get("/log")
async def log():
    if not LOG_FILE.exists():
        return {"log": ""}
    text = LOG_FILE.read_text()
    return {"log": text[-8000:]}  # tail to 8 KB
