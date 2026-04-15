"""WSI preprocessing routes — Otsu segmentation, patching, QC."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
JOBS_DIR = PATHCLAW_DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PreprocessConfig(BaseModel):
    dataset_id: str
    patch_size: int = 256
    stride: int = 256
    magnification: float = 20.0
    otsu_level: int = 1
    min_tissue_pct: float = 0.5
    preview_only: bool = False
    preview_count: int = 3


class PreprocessJobStatus(BaseModel):
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    progress: float  # 0.0 - 1.0
    slides_processed: int
    slides_total: int
    patches_extracted: int
    errors: list[str]


# ---------------------------------------------------------------------------
# Job tracking (in-memory for MVP, Redis/DB later)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}


def _run_preprocessing(job_id: str, config: dict):
    """Background task for WSI preprocessing."""
    job = _jobs[job_id]
    job["status"] = "running"

    try:
        from pathclaw.preprocessing.pipeline import preprocess_dataset
        result = preprocess_dataset(
            dataset_id=config["dataset_id"],
            patch_size=config["patch_size"],
            stride=config["stride"],
            magnification=config["magnification"],
            otsu_level=config["otsu_level"],
            min_tissue_pct=config["min_tissue_pct"],
            preview_only=config.get("preview_only", False),
            preview_count=config.get("preview_count", 3),
            job_status=job,  # Pass reference for progress updates
        )
        job["status"] = "completed"
        job["result"] = result
    except Exception as e:
        job["status"] = "failed"
        job["errors"].append(str(e))

    # Save job metadata
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "status.json").write_text(json.dumps(job, indent=2, default=str))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/start")
async def start_preprocessing(config: PreprocessConfig, background_tasks: BackgroundTasks):
    """Launch a WSI preprocessing job."""
    job_id = f"prep-{str(uuid.uuid4())[:8]}"

    job = {
        "job_id": job_id,
        "type": "preprocessing",
        "status": "queued",
        "config": config.model_dump(),
        "progress": 0.0,
        "slides_processed": 0,
        "slides_total": 0,
        "patches_extracted": 0,
        "errors": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _jobs[job_id] = job

    background_tasks.add_task(_run_preprocessing, job_id, config.model_dump())

    return {"job_id": job_id, "status": "queued"}


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """Get preprocessing job status."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _jobs[job_id]


@router.get("/preview/{dataset_id}")
async def get_preview_by_dataset(dataset_id: str):
    """Get preprocessing preview images by dataset ID."""
    preview_dir = PATHCLAW_DATA_DIR / "preprocessed" / dataset_id / "previews"
    if not preview_dir.exists():
        return {"dataset_id": dataset_id, "previews": []}
    from fastapi.responses import FileResponse as _FR  # noqa — just checking availability
    previews = []
    for img in sorted(preview_dir.glob("*.png")):
        previews.append({
            "slide": img.stem,
            "url": f"/api/preprocess/preview-img/{dataset_id}/{img.name}",
        })
    return {"dataset_id": dataset_id, "previews": previews}


@router.get("/preview-img/{dataset_id}/{filename}")
async def get_preview_image(dataset_id: str, filename: str):
    """Serve a preprocessing preview PNG."""
    from fastapi.responses import FileResponse
    img_path = PATHCLAW_DATA_DIR / "preprocessed" / dataset_id / "previews" / filename
    if not img_path.exists() or img_path.suffix != ".png":
        raise HTTPException(status_code=404, detail=f"Preview {filename} not found")
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/{job_id}/preview")
async def get_preview(job_id: str):
    """Get preprocessing preview images."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs[job_id]
    # Previews are written to the preprocessed output dir, not the jobs dir
    dataset_id = job["config"]["dataset_id"]
    preview_dir = PATHCLAW_DATA_DIR / "preprocessed" / dataset_id / "previews"
    if not preview_dir.exists():
        return {"job_id": job_id, "previews": []}

    previews = []
    for img in sorted(preview_dir.glob("*.png")):
        previews.append({
            "filename": img.name,
            "path": str(img),
        })

    return {"job_id": job_id, "previews": previews}
