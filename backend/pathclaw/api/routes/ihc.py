"""IHC scoring API.

Two paths:
- Rule-based: `POST /api/ihc/score` — synchronous small-cohort scoring, or
  asynchronous via the queue for cohorts > 10 slides.
- Learned: `POST /api/ihc/patch-labels` — build per-patch supervision for a
  downstream MIL / regressor.

Preset rules: `GET /api/ihc/rules`.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pathclaw.ihc import (
    build_ihc_patch_labels,
    get_rule,
    list_rules,
    score_dataset,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job tracker (matches the pattern used by training/eval routes).
_ihc_jobs: dict[str, dict] = {}


@router.get("/rules")
async def list_ihc_rules():
    return {"rules": list_rules()}


class ScoreBody(BaseModel):
    dataset_id: str
    rule: str                         # preset name, e.g. "ki67_pi"
    rule_override: dict | None = None
    max_slides: int | None = None
    use_cellpose: bool = True
    session_id: str = ""


@router.post("/score")
async def start_score(body: ScoreBody):
    """Score a registered dataset. Runs synchronously for now — large cohorts
    should call with small `max_slides` or use the queue (Phase-2)."""
    try:
        _ = get_rule(body.rule, override=body.rule_override)  # validate early
    except KeyError as e:
        raise HTTPException(400, str(e))
    job_id = f"ihc-{uuid.uuid4().hex[:8]}"
    _ihc_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "progress": 0.0,
        "rule": body.rule,
        "dataset_id": body.dataset_id,
        "session_id": body.session_id,
    }
    try:
        summary = score_dataset(
            dataset_id=body.dataset_id,
            rule=body.rule,
            rule_override=body.rule_override,
            max_slides=body.max_slides,
            use_cellpose=body.use_cellpose,
            session_id=body.session_id,
        )
        _ihc_jobs[job_id].update(status="completed", progress=1.0, summary=summary)
        return {"job_id": job_id, **summary}
    except Exception as e:
        logger.exception("IHC score failed")
        _ihc_jobs[job_id].update(status="failed", error=str(e))
        raise HTTPException(500, str(e))


class PatchLabelBody(BaseModel):
    dataset_id: str
    rule: str
    rule_override: dict | None = None
    patches_per_slide: int | None = None
    out_name: str | None = None
    session_id: str = ""


@router.post("/patch-labels")
async def build_patch_labels(body: PatchLabelBody):
    try:
        _ = get_rule(body.rule, override=body.rule_override)
    except KeyError as e:
        raise HTTPException(400, str(e))
    try:
        summary = build_ihc_patch_labels(
            dataset_id=body.dataset_id,
            rule=body.rule,
            rule_override=body.rule_override,
            patches_per_slide=body.patches_per_slide,
            out_name=body.out_name,
        )
        return summary
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.exception("build_ihc_patch_labels failed")
        raise HTTPException(500, str(e))


@router.get("/{job_id}")
async def get_score_job(job_id: str):
    if job_id not in _ihc_jobs:
        raise HTTPException(404, f"IHC job {job_id} not found")
    return _ihc_jobs[job_id]
