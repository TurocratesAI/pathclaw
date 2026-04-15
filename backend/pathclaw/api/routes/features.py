"""Feature extraction API routes.

Feature extraction is GPU-heavy and long-running. We launch each job as a
detached subprocess (via ``_extract_worker``) so the FastAPI server never
shares GIL / memory with the extraction process. Progress is read from
``~/.pathclaw/jobs/{job_id}/status.json`` which the worker updates live.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
JOBS_DIR = PATHCLAW_DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()


class FeatureExtractionConfig(BaseModel):
    dataset_id: str
    backbone: str = "uni"       # uni | conch | ctranspath | virchow | virchow2 | gigapath
    batch_size: int = 256
    device: str = "auto"        # auto | cuda | cuda:0 | cuda:1 | cpu


def _status_file(job_id: str) -> Path:
    return JOBS_DIR / job_id / "status.json"


def _read_status(job_id: str) -> dict:
    p = _status_file(job_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _write_status(job_id: str, data: dict) -> None:
    p = _status_file(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(p)


def _is_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _launch_worker(job_id: str, config: dict) -> int:
    """Spawn the extract worker as a detached subprocess. Returns PID."""
    payload = json.dumps({
        "job_id": job_id,
        "dataset_id": config["dataset_id"],
        "backbone": config["backbone"],
        "batch_size": config.get("batch_size", 256),
        "device": config.get("device", "auto"),
    })
    log_path = JOBS_DIR / job_id / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass through the server's PYTHONPATH so `pathclaw` is importable in the child.
    env = dict(os.environ)

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            [sys.executable, "-m", "pathclaw.preprocessing._extract_worker", payload],
            stdout=log_f,
            stderr=log_f,
            start_new_session=True,  # detach from server process group
            env=env,
        )
    return proc.pid


@router.post("/start")
async def start_feature_extraction(config: FeatureExtractionConfig):
    """Launch a feature extraction job as a detached subprocess."""
    job_id = f"feat-{str(uuid.uuid4())[:8]}"

    job = {
        "job_id": job_id,
        "type": "feature_extraction",
        "status": "queued",
        "config": config.model_dump(),
        "progress": 0.0,
        "slides_completed": 0,
        "slides_total": 0,
        "backbone": config.backbone,
        "feature_dim": None,
        "errors": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_status(job_id, job)

    pid = _launch_worker(job_id, config.model_dump())
    job["pid"] = pid
    _write_status(job_id, job)

    return {"job_id": job_id, "status": "queued", "backbone": config.backbone, "pid": pid}


@router.get("/{job_id}")
async def get_feature_job_status(job_id: str):
    """Get feature extraction job status."""
    status = _read_status(job_id)
    if not status:
        raise HTTPException(404, f"Feature job {job_id} not found")

    # Detect crashed workers (pid gone but status still 'running')
    if status.get("status") == "running":
        pid = status.get("pid")
        if pid and not _is_alive(pid):
            status["status"] = "failed"
            errs = list(status.get("errors", []))
            errs.append("worker process exited unexpectedly")
            status["errors"] = errs
            _write_status(job_id, status)

    return status


@router.post("/{job_id}/cancel")
async def cancel_feature_job(job_id: str):
    """Cancel a running feature extraction job by SIGTERMing its worker."""
    status = _read_status(job_id)
    if not status:
        raise HTTPException(404, f"Feature job {job_id} not found")
    pid = status.get("pid")
    if pid and _is_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            raise HTTPException(500, f"Failed to kill worker pid {pid}: {e}")
        status["status"] = "cancelled"
        _write_status(job_id, status)
        return {"job_id": job_id, "status": "cancelled", "pid": pid}
    return {"job_id": job_id, "status": status.get("status", "unknown")}


@router.get("")
async def list_feature_jobs():
    """List all feature extraction jobs by reading status files on disk."""
    jobs = []
    if JOBS_DIR.exists():
        for d in sorted(JOBS_DIR.iterdir()):
            if not d.is_dir() or not d.name.startswith("feat-"):
                continue
            sp = d / "status.json"
            if sp.exists():
                try:
                    jobs.append(json.loads(sp.read_text()))
                except Exception:
                    continue
    return {"jobs": jobs}
