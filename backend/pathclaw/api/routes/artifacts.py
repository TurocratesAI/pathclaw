"""Artifact storage routes."""

from __future__ import annotations

import os
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
EXPERIMENTS_DIR = PATHCLAW_DATA_DIR / "experiments"

router = APIRouter()


@router.get("")
async def list_artifacts(session_id: str = ""):
    """List all experiment artifacts, optionally filtered by session."""
    artifacts = []
    if EXPERIMENTS_DIR.exists():
        for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp_files = [f.name for f in exp_dir.iterdir() if f.is_file()]
            # Read session from config.json if present
            exp_session = ""
            if "config.json" in exp_files:
                try:
                    cfg = json.loads((exp_dir / "config.json").read_text())
                    exp_session = cfg.get("session_id", "")
                except Exception:
                    pass
            if session_id and exp_session != session_id:
                continue
            artifacts.append({
                "experiment_id": exp_dir.name,
                "files": exp_files,
                "has_model": "model.pth" in exp_files,
                "has_config": "config.json" in exp_files,
                "has_metrics": "metrics.json" in exp_files,
            })
    return {"artifacts": artifacts}


@router.get("/{experiment_id}/export")
async def export_experiment(experiment_id: str):
    """Download the full experiment (model.pth + config.json + metrics.json + history + plots + logs) as a zip."""
    from fastapi.responses import StreamingResponse
    import io, zipfile
    exp_dir = EXPERIMENTS_DIR / experiment_id
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    if not exp_dir.resolve().is_relative_to(EXPERIMENTS_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in exp_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(exp_dir))
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{experiment_id}.zip"'},
    )


@router.get("/{experiment_id}/{filename}")
async def download_artifact(experiment_id: str, filename: str):
    """Download a specific artifact file."""
    file_path = EXPERIMENTS_DIR / experiment_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {experiment_id}/{filename}")
    if not file_path.resolve().is_relative_to(EXPERIMENTS_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse(file_path, filename=filename)
