"""Session task-plan — agent's explicit step-by-step checklist.

Models Claude Code's TaskCreate/TaskUpdate pattern so small local LLMs don't
drift on multi-step requests. The agent commits to a plan up front, then drives
one task per turn. The frontend shows the list live and ticks them off as
`task_updated` SSE events arrive.

Storage: ``~/.pathclaw/sessions/<sid>/tasks.json``::

    {
        "session_id": "<sid>",
        "created_at": "...",
        "updated_at": "...",
        "tasks": [
            {
                "id": 1,
                "title": "Download CHOL slides",
                "description": "search_gdc + download_gdc for TCGA-CHOL DX",
                "status": "completed" | "in_progress" | "pending" | "skipped",
                "pause_after": false,
                "created_at": "...",
                "updated_at": "..."
            },
            ...
        ]
    }
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()

router = APIRouter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tasks_path(session_id: str) -> Path:
    if not session_id or "/" in session_id or ".." in session_id:
        raise HTTPException(400, f"Invalid session_id: {session_id!r}")
    p = PATHCLAW_DATA_DIR / "sessions" / session_id / "tasks.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_plan(session_id: str) -> dict:
    """Return the current plan or an empty shell. Also used by chat.py."""
    p = _tasks_path(session_id)
    if not p.exists():
        return {
            "session_id": session_id,
            "created_at": None,
            "updated_at": None,
            "tasks": [],
        }
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"session_id": session_id, "created_at": None, "updated_at": None, "tasks": []}


def save_plan(plan: dict) -> None:
    sid = plan["session_id"]
    _tasks_path(sid).write_text(json.dumps(plan, indent=2))


def render_plan_for_prompt(session_id: str) -> str:
    """One-line-per-task summary injected into the agent's system prompt so it
    always sees the live plan state without needing an extra tool call."""
    plan = load_plan(session_id)
    tasks = plan.get("tasks") or []
    if not tasks:
        return ""
    lines = ["## Active Task Plan (auto-advance)"]
    for t in tasks:
        mark = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]", "skipped": "[-]"}.get(
            t.get("status", "pending"), "[ ]"
        )
        pause = " (pause_after)" if t.get("pause_after") else ""
        title = t.get("title", "?")
        lines.append(f"{mark} {t['id']}. {title}{pause}")
    lines.append(
        "Rule: work the FIRST `[ ]` pending task next. "
        "Flip it to `in_progress` via update_task_status BEFORE calling other tools, "
        "then to `completed` when done. If all tasks are [x], stop and summarize."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TaskStep(BaseModel):
    title: str
    description: str = ""
    pause_after: bool = False


class CreatePlanBody(BaseModel):
    session_id: str
    tasks: list[TaskStep] = Field(default_factory=list)
    replace: bool = True  # drop existing plan by default


class UpdateTaskBody(BaseModel):
    session_id: str
    status: str  # pending | in_progress | completed | skipped


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/tasks")
async def create_plan(body: CreatePlanBody):
    if not body.tasks:
        raise HTTPException(400, "tasks list is empty")
    now = _now_iso()
    if body.replace:
        existing_tasks: list[dict] = []
    else:
        existing_tasks = load_plan(body.session_id).get("tasks", [])
    start_id = (max((t["id"] for t in existing_tasks), default=0)) + 1
    new_tasks = [
        {
            "id": start_id + i,
            "title": s.title,
            "description": s.description,
            "status": "pending",
            "pause_after": s.pause_after,
            "created_at": now,
            "updated_at": now,
        }
        for i, s in enumerate(body.tasks)
    ]
    plan = {
        "session_id": body.session_id,
        "created_at": now,
        "updated_at": now,
        "tasks": existing_tasks + new_tasks,
    }
    save_plan(plan)
    return plan


@router.get("/tasks")
async def get_plan(session_id: str):
    return load_plan(session_id)


@router.patch("/tasks/{task_id}")
async def update_task(task_id: int, body: UpdateTaskBody):
    valid = {"pending", "in_progress", "completed", "skipped"}
    if body.status not in valid:
        raise HTTPException(400, f"status must be one of {sorted(valid)}")
    plan = load_plan(body.session_id)
    hit = False
    now = _now_iso()
    for t in plan["tasks"]:
        if t["id"] == task_id:
            t["status"] = body.status
            t["updated_at"] = now
            hit = True
            break
    if not hit:
        raise HTTPException(404, f"Task {task_id} not in session {body.session_id}")
    plan["updated_at"] = now
    save_plan(plan)
    return plan


@router.delete("/tasks")
async def clear_plan(session_id: str):
    p = _tasks_path(session_id)
    if p.exists():
        p.unlink()
    return {"session_id": session_id, "cleared": True}
