"""Shared ETA annotation for long-running job status responses."""
from __future__ import annotations

from datetime import datetime, timezone


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _format(sec: float) -> str:
    sec = int(max(0, sec))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m {sec % 60}s"
    h, rem = divmod(sec, 3600)
    return f"{h}h {rem // 60}m"


def annotate_eta(status: dict) -> dict:
    """Mutate `status` in place to add `elapsed_seconds`, `eta_seconds`, `eta_human`.

    ETA is computed as `elapsed * (1 - progress) / progress` once progress > 0.02.
    Only populated while status == 'running'.
    """
    if not isinstance(status, dict):
        return status
    state = status.get("status")
    ts = _parse_ts(status.get("started_at")) or _parse_ts(status.get("created_at"))
    if ts is None:
        return status
    now = datetime.now(timezone.utc)
    elapsed = max(0.0, (now - ts).total_seconds())
    status["elapsed_seconds"] = round(elapsed, 1)
    status["elapsed_human"] = _format(elapsed)

    if state == "running":
        progress = float(status.get("progress") or 0.0)
        if progress > 0.02:
            eta = elapsed * (1.0 - progress) / progress
            status["eta_seconds"] = round(eta, 1)
            status["eta_human"] = _format(eta)
    return status
