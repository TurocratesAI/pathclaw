"""Standalone feature extraction worker.

Runs extract_features() in its own Python process so the main FastAPI
server never shares GIL or memory with heavy GPU workloads.

Invoked by routes/features.py via subprocess.Popen. Reads a JSON args
payload from argv[1] and writes progress to ~/.pathclaw/jobs/{job_id}/status.json
which the server polls.

Usage:
    python -m pathclaw.preprocessing._extract_worker '{"job_id":...}'
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()


def _status_path(job_id: str) -> Path:
    d = PATHCLAW_DATA_DIR / "jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "status.json"


def _write_status(job_id: str, data: dict) -> None:
    p = _status_path(job_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(p)


def _load_status(job_id: str) -> dict:
    p = _status_path(job_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


class _StatusProxy(dict):
    """dict-like object that persists to disk on every mutation."""

    def __init__(self, job_id: str, initial: dict):
        super().__init__(initial)
        self._job_id = job_id
        _write_status(job_id, dict(self))

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        _write_status(self._job_id, dict(self))

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        _write_status(self._job_id, dict(self))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: _extract_worker <json_args>", file=sys.stderr)
        return 2

    args = json.loads(sys.argv[1])
    job_id = args["job_id"]

    initial = _load_status(job_id)
    initial.update({
        "job_id": job_id,
        "type": "feature_extraction",
        "status": "running",
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "errors": initial.get("errors", []),
    })
    job = _StatusProxy(job_id, initial)

    try:
        from pathclaw.preprocessing.feature_extraction import extract_features
        result = extract_features(
            dataset_id=args["dataset_id"],
            backbone=args["backbone"],
            batch_size=args.get("batch_size", 256),
            device_str=args.get("device", "auto"),
            job_status=job,
        )
        job["status"] = "completed"
        job["result"] = result
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        errs = list(job.get("errors", []))
        errs.append(str(e))
        job["errors"] = errs
        job["traceback"] = tb
        job["status"] = "failed"
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        print(tb, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
