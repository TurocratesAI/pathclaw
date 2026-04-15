"""Evaluation and inference routes."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
EXPERIMENTS_DIR = PATHCLAW_DATA_DIR / "experiments"

router = APIRouter()


class EvalRequest(BaseModel):
    model_path: str
    dataset_id: str
    split: str = "test"
    metrics: list[str] = Field(default_factory=lambda: ["accuracy", "balanced_accuracy", "auroc"])


_eval_jobs: dict[str, dict] = {}


def _run_evaluation(job_id: str, config: dict):
    """Background eval task."""
    job = _eval_jobs[job_id]
    job["status"] = "running"
    try:
        from pathclaw.evaluation.evaluator import evaluate_model
        result = evaluate_model(config, job_status=job)
        job["status"] = "completed"
        job["result"] = result
    except Exception as e:
        job["status"] = "failed"
        job["errors"].append(str(e))

    exp_dir = EXPERIMENTS_DIR / job_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "eval_status.json").write_text(json.dumps(job, indent=2, default=str))


@router.post("/start")
async def start_evaluation(req: EvalRequest, background_tasks: BackgroundTasks):
    """Launch an evaluation job."""
    job_id = f"eval-{str(uuid.uuid4())[:8]}"
    job = {
        "job_id": job_id,
        "type": "evaluation",
        "status": "queued",
        "config": req.model_dump(),
        "progress": 0.0,
        "metrics": {},
        "errors": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _eval_jobs[job_id] = job
    background_tasks.add_task(_run_evaluation, job_id, req.model_dump())
    return {"job_id": job_id, "status": "queued"}


@router.get("/{job_id}/metrics")
async def get_eval_metrics(job_id: str):
    """Get evaluation metrics."""
    if job_id not in _eval_jobs:
        raise HTTPException(status_code=404, detail=f"Eval job {job_id} not found")
    return _eval_jobs[job_id].get("result", {}).get("metrics", {})


@router.get("/{job_id}/plots")
async def get_eval_plots(job_id: str):
    """Get evaluation visualizations."""
    plot_dir = EXPERIMENTS_DIR / job_id / "plots"
    if not plot_dir.exists():
        return {"job_id": job_id, "plots": []}
    return {
        "job_id": job_id,
        "plots": [{"name": p.name, "path": str(p)} for p in sorted(plot_dir.glob("*.png"))],
    }


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

_heatmap_jobs: dict[str, dict] = {}


class HeatmapRequest(BaseModel):
    dataset_id: str
    slide_stem: str


def _run_heatmap(job_id: str, experiment_id: str, dataset_id: str, slide_stem: str):
    """Background heatmap generation task."""
    job = _heatmap_jobs[job_id]
    job["status"] = "running"
    try:
        from pathclaw.evaluation.heatmap import generate_attention_heatmap
        result = generate_attention_heatmap(
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            slide_stem=slide_stem,
            job_status=job,
        )
        job["status"] = "completed"
        job["result"] = result
    except Exception as e:
        import traceback
        job["status"] = "failed"
        job["error"] = f"{type(e).__name__}: {e}"
        job["traceback"] = traceback.format_exc()


@router.post("/{experiment_id}/heatmap")
async def start_heatmap(experiment_id: str, req: HeatmapRequest, background_tasks: BackgroundTasks):
    """Launch attention heatmap generation for a slide."""
    job_id = f"heatmap-{str(uuid.uuid4())[:8]}"
    job = {
        "job_id": job_id,
        "experiment_id": experiment_id,
        "dataset_id": req.dataset_id,
        "slide_stem": req.slide_stem,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _heatmap_jobs[job_id] = job
    background_tasks.add_task(
        _run_heatmap, job_id, experiment_id, req.dataset_id, req.slide_stem
    )
    return {"job_id": job_id, "status": "queued"}


@router.get("/{experiment_id}/heatmap/status/{slide_stem}")
async def heatmap_status(experiment_id: str, slide_stem: str):
    """Check if a heatmap has been generated for a slide."""
    from pathclaw.evaluation.heatmap import list_heatmaps
    available = list_heatmaps(experiment_id)
    if slide_stem in available:
        return {"experiment_id": experiment_id, "slide_stem": slide_stem, "status": "ready"}
    # Check running jobs
    for job in _heatmap_jobs.values():
        if job.get("experiment_id") == experiment_id and job.get("slide_stem") == slide_stem:
            return {"experiment_id": experiment_id, "slide_stem": slide_stem, "status": job["status"]}
    return {"experiment_id": experiment_id, "slide_stem": slide_stem, "status": "not_generated"}


@router.get("/{experiment_id}/heatmap/list")
async def list_heatmaps_endpoint(experiment_id: str):
    """List slides that have generated heatmaps for this experiment."""
    from pathclaw.evaluation.heatmap import list_heatmaps
    return {"experiment_id": experiment_id, "slides": list_heatmaps(experiment_id)}
