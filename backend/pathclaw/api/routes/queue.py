"""Task queue — serializes GPU-heavy jobs (training / feature extraction / preprocessing)
to prevent race conditions and OOM when multiple sessions submit work.

Simple FIFO with per-resource concurrency caps.
- GPU tasks (training, features): 1 running at a time by default
- CPU tasks (preprocess, eval): pass through (can run concurrently)

Submit: POST /api/queue/submit {task_type, payload, session_id?}
List:   GET  /api/queue
Cancel: DELETE /api/queue/{task_id}
Update: POST /api/queue/{task_id}/cancel
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
QUEUE_PATH = PATHCLAW_DATA_DIR / "queue.json"

router = APIRouter()

TaskType = Literal["training", "features", "preprocess", "evaluation"]
# Tasks in this set are considered GPU-exclusive — at most one runs concurrently.
GPU_EXCLUSIVE = {"training", "features"}

# Concurrency caps per task type.
CONCURRENCY = {
    "training": 1,
    "features": 1,
    "preprocess": 2,
    "evaluation": 2,
}

_LOCK = asyncio.Lock()
_WORKER_STARTED = False


def _detect_gpu_count() -> int:
    """Count visible CUDA devices. 0 if torch unavailable or no GPUs."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _free_mem_gb() -> float:
    """Host RAM available from /proc/meminfo. Returns 0.0 if unreadable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 2)
    except Exception:
        pass
    return 0.0


def _gpu_snapshot() -> list[dict]:
    """Per-GPU name + free/total VRAM in MB. Empty list on non-CUDA hosts."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_b, total_b = torch.cuda.mem_get_info(i)
            out.append({
                "id": i,
                "name": props.name,
                "free_mb": round(free_b / (1024 * 1024)),
                "total_mb": round(total_b / (1024 * 1024)),
            })
        return out
    except Exception:
        return []


def _gpu_slots() -> dict[int, str | None]:
    """Map of gpu_id → running task_id (None if free)."""
    q = _load_queue()
    slots: dict[int, str | None] = {i: None for i in range(max(_detect_gpu_count(), 1))}
    for t in q:
        if t["status"] == "running" and t["task_type"] in GPU_EXCLUSIVE:
            gid = t.get("gpu_id")
            if gid is not None and gid in slots:
                slots[gid] = t["task_id"]
    return slots


def _pick_free_gpu(q: list[dict], pinned: int | None = None) -> int | None:
    """Return a free GPU id, or None if all busy. If `pinned` is set, return it
    only if that specific GPU is free (else None)."""
    gpu_count = _detect_gpu_count()
    if gpu_count == 0:
        return None
    busy = {
        t.get("gpu_id") for t in q
        if t["status"] == "running" and t["task_type"] in GPU_EXCLUSIVE
    }
    if pinned is not None:
        return pinned if pinned not in busy and 0 <= pinned < gpu_count else None
    for i in range(gpu_count):
        if i not in busy:
            return i
    return None


class QueueSubmit(BaseModel):
    task_type: TaskType
    payload: dict = Field(default_factory=dict)
    session_id: str = ""
    note: str = ""


def _load_queue() -> list[dict]:
    if not QUEUE_PATH.exists():
        return []
    try:
        return json.loads(QUEUE_PATH.read_text())
    except Exception:
        return []


def _save_queue(q: list[dict]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.write_text(json.dumps(q, indent=2))


def _running_count(q: list[dict], task_type: str | None = None) -> int:
    if task_type:
        return sum(1 for t in q if t["status"] == "running" and t["task_type"] == task_type)
    return sum(1 for t in q if t["status"] == "running")


def _gpu_running(q: list[dict]) -> int:
    return sum(1 for t in q if t["status"] == "running" and t["task_type"] in GPU_EXCLUSIVE)


async def _external_gpu_running() -> int:
    """Count running training/features jobs NOT dispatched by this queue.
    Lets the queue respect pre-existing workloads (e.g. Telegram-triggered feature extraction)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get("http://localhost:8101/api/jobs/all")
            if r.status_code != 200:
                return 0
            jobs = r.json().get("jobs", [])
            return sum(1 for j in jobs
                       if j.get("status") == "running"
                       and j.get("type") in ("training", "features"))
    except Exception:
        return 0


def _normalize_training_payload(payload: dict) -> dict:
    """Accept flat agent-friendly keys (mammoth_enabled, epochs, lr, ...) and
    reshape them into the nested TrainingConfig the API expects."""
    p = dict(payload)
    if "mammoth_enabled" in p:
        p.setdefault("mammoth", {})["enabled"] = p.pop("mammoth_enabled")
    train_flat = {}
    for k in ("epochs", "lr", "optimizer", "weight_decay", "scheduler", "early_stopping_patience"):
        if k in p:
            train_flat[k] = p.pop(k)
    if train_flat:
        p.setdefault("training", {}).update(train_flat)
    if "eval_strategy" in p:
        p.setdefault("evaluation", {})["strategy"] = p.pop("eval_strategy")
    # Multi-GPU: if the dispatcher pinned this task to a GPU, pass it as cuda:N.
    gid = p.pop("gpu_id", None)
    if gid is not None and p.get("device", "auto") == "auto":
        try:
            p["device"] = f"cuda:{int(gid)}"
        except (TypeError, ValueError):
            pass
    return p


async def _dispatch_one(task: dict) -> tuple[bool, str]:
    """Kick off a single task via the appropriate API. Returns (success, message)."""
    base = "http://localhost:8101"
    path_map = {
        "training": "/api/training/start",
        "features": "/api/features/extract",
        "preprocess": "/api/preprocess/start",
        "evaluation": "/api/eval/start",
    }
    path = path_map[task["task_type"]]
    payload = task["payload"]
    if task["task_type"] == "training":
        payload = _normalize_training_payload(payload)
    elif task["task_type"] == "features":
        payload = dict(payload)
        gid = payload.pop("gpu_id", None)
        if gid is not None and payload.get("device_str", "auto") == "auto":
            try:
                payload["device_str"] = f"cuda:{int(gid)}"
            except (TypeError, ValueError):
                pass
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{base}{path}", json=payload)
            d = r.json()
            if "job_id" in d:
                task["dispatched_job_id"] = d["job_id"]
                return True, f"dispatched as {d['job_id']}"
            return False, f"dispatch error HTTP {r.status_code}: {d.get('detail', d)}"
    except Exception as e:
        return False, f"dispatch exception: {e}"


async def _poll_job_status(task: dict) -> str | None:
    """Poll the underlying job endpoint. Returns status string, 'missing' if job was lost
    (likely due to a server restart), or None on transient error."""
    base = "http://localhost:8101"
    jid = task.get("dispatched_job_id")
    if not jid:
        return None
    status_map = {
        "training": f"/api/training/{jid}",
        "features": f"/api/features/{jid}/status",
        "preprocess": f"/api/preprocess/{jid}",
        "evaluation": f"/api/eval/{jid}",
    }
    path = status_map.get(task["task_type"])
    if not path:
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{base}{path}")
            if r.status_code == 404:
                return "missing"
            if r.status_code != 200:
                return None
            return r.json().get("status")
    except Exception:
        return None


async def _worker_loop():
    """Single background coroutine that schedules queued tasks."""
    while True:
        try:
            async with _LOCK:
                q = _load_queue()
                changed = False

                # Phase 1: update status of running tasks
                for t in q:
                    if t["status"] == "running":
                        status = await _poll_job_status(t)
                        if status in ("completed", "failed", "cancelled"):
                            t["status"] = status
                            t["completed_at"] = datetime.now(timezone.utc).isoformat()
                            changed = True
                        elif status == "missing":
                            # Underlying job lost (server restart). Mark failed so the queue doesn't stall.
                            t["status"] = "failed"
                            t["message"] = "underlying job no longer tracked (server restart?)"
                            t["completed_at"] = datetime.now(timezone.utc).isoformat()
                            changed = True

                # Phase 2: dispatch queued tasks respecting concurrency + per-GPU slots.
                # GPU-exclusive tasks are allocated to a free GPU (multi-GPU aware).
                # CPU tasks (preprocess, eval) use the existing concurrency caps.
                gpu_count = _detect_gpu_count()
                external_gpu_busy = await _external_gpu_running()
                for t in q:
                    if t["status"] != "queued":
                        continue
                    if t["task_type"] in GPU_EXCLUSIVE:
                        if gpu_count == 0:
                            # CPU-only host — fall back to 1-at-a-time legacy behavior.
                            if _gpu_running(q) + external_gpu_busy >= 1:
                                continue
                            gpu_id = None
                        else:
                            pinned = t.get("payload", {}).get("gpu_id")
                            try:
                                pinned = int(pinned) if pinned not in (None, "", "auto") else None
                            except (TypeError, ValueError):
                                pinned = None
                            gpu_id = _pick_free_gpu(q, pinned)
                            if gpu_id is None:
                                continue
                            # External jobs on the same GPU would collide — defer.
                            if external_gpu_busy and gpu_id == 0:
                                continue
                            t["payload"]["gpu_id"] = gpu_id
                    else:
                        cap = CONCURRENCY.get(t["task_type"], 1)
                        if _running_count(q, t["task_type"]) >= cap:
                            continue
                        gpu_id = None
                    ok, msg = await _dispatch_one(t)
                    t["status"] = "running" if ok else "failed"
                    t["started_at"] = datetime.now(timezone.utc).isoformat()
                    t["message"] = msg
                    if gpu_id is not None:
                        t["gpu_id"] = gpu_id
                    changed = True

                if changed:
                    _save_queue(q)
        except Exception as e:
            print(f"[queue worker] error: {e}")
        await asyncio.sleep(5)


def start_worker():
    global _WORKER_STARTED
    if _WORKER_STARTED:
        return
    _WORKER_STARTED = True
    asyncio.create_task(_worker_loop())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/submit")
async def submit_task(req: QueueSubmit):
    """Queue a task for serialized execution."""
    async with _LOCK:
        q = _load_queue()
        task_id = f"q-{str(uuid.uuid4())[:8]}"
        task = {
            "task_id": task_id,
            "task_type": req.task_type,
            "payload": req.payload,
            "session_id": req.session_id,
            "note": req.note,
            "status": "queued",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "completed_at": None,
            "dispatched_job_id": None,
            "message": "",
        }
        q.append(task)
        _save_queue(q)
    start_worker()
    return task


@router.get("")
async def list_tasks(status: str = "", session_id: str = ""):
    """List queued/running/recent tasks (default: all)."""
    q = _load_queue()
    if status:
        q = [t for t in q if t["status"] == status]
    if session_id:
        q = [t for t in q if t.get("session_id") == session_id]
    return {"tasks": q}


@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    """Remove a queued task, or mark a running task as cancelled (does not kill the underlying job)."""
    async with _LOCK:
        q = _load_queue()
        for t in q:
            if t["task_id"] == task_id:
                if t["status"] == "queued":
                    q.remove(t)
                    _save_queue(q)
                    return {"status": "removed"}
                elif t["status"] == "running":
                    t["status"] = "cancelled"
                    t["completed_at"] = datetime.now(timezone.utc).isoformat()
                    _save_queue(q)
                    return {"status": "marked cancelled (underlying job may still be running)"}
                return {"status": f"already {t['status']}"}
    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@router.get("/resources")
async def queue_resources():
    """Host resource snapshot: free RAM, per-GPU VRAM + current slot assignments."""
    q = _load_queue()
    slots = _gpu_slots()
    return {
        "free_mem_gb": _free_mem_gb(),
        "gpus": _gpu_snapshot(),
        "gpu_slots": {str(k): v for k, v in slots.items()},
        "running": _running_count(q),
        "queued": sum(1 for t in q if t["status"] == "queued"),
    }


@router.post("/clear-finished")
async def clear_finished():
    """Drop completed/failed/cancelled entries from the queue."""
    async with _LOCK:
        q = _load_queue()
        kept = [t for t in q if t["status"] in ("queued", "running")]
        _save_queue(kept)
    return {"kept": len(kept)}
