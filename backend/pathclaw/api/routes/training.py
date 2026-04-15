"""MIL/MAMMOTH training routes."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
JOBS_DIR = PATHCLAW_DATA_DIR / "jobs"
EXPERIMENTS_DIR = PATHCLAW_DATA_DIR / "experiments"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Backbone → expected feature_dim mapping
BACKBONE_DIMS: dict[str, int] = {
    "uni": 1024,
    "conch": 512,
    "ctranspath": 768,
    "virchow": 1280,
    "virchow2": 2560,
    "gigapath": 1536,
}


class MammothConfig(BaseModel):
    enabled: bool = True
    num_experts: int = 30
    num_slots: int = 10
    num_heads: int = 16
    share_lora_weights: bool = True
    auto_rank: bool = True
    dropout: float = 0.1
    rank: int = 0              # 0 = auto (used when auto_rank=True)
    temperature: float = 1.0   # routing temperature for expert selection


class TrainingHyperparams(BaseModel):
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"    # adam | adamw | sgd | radam
    scheduler: str = "cosine"  # cosine | step | plateau | none
    early_stopping_patience: int = 0  # 0 = disabled


class EvalConfig(BaseModel):
    strategy: str = "holdout"  # holdout | 3-fold-cv | 5-fold-cv | 10-fold-cv
    metrics: list[str] = Field(default_factory=lambda: ["accuracy", "balanced_accuracy", "auroc"])


class TrainingConfig(BaseModel):
    task: str
    dataset_id: str
    task_type: str = "mil"     # mil | segmentation
    label_column: str = ""
    feature_backbone: str = "uni"
    feature_dim: int = 1024
    embed_dim: int = 512
    attn_dim: int = 128        # ABMIL/CLAM attention hidden dim
    mil_method: str = "abmil"  # abmil | meanpool | transmil | clam | dsmil | rrtmil | wikg
    num_classes: int = 2
    device: str = "auto"       # auto | cuda | cuda:0 | cuda:1 | cpu
    mammoth: MammothConfig = Field(default_factory=MammothConfig)
    training: TrainingHyperparams = Field(default_factory=TrainingHyperparams)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    # Segmentation-specific fields
    seg_model: str = "seg_unet"              # seg_unet | hovernet | cellpose
    num_seg_classes: int = 2                 # number of segmentation classes
    patch_size: int = 256                    # patch size in pixels
    batch_size: int = 8                      # batch size for segmentation training
    pretrained_encoder: bool = True          # use pretrained encoder for HoVer-Net
    cellpose_model_type: str = "cyto3"       # cellpose model type
    cellpose_diameter: float = 30.0          # expected cell diameter in pixels
    session_id: str = ""                     # session that launched this experiment


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

_training_jobs: dict[str, dict] = {}


def _run_training(job_id: str, config: dict):
    """Background task for MIL or segmentation training."""
    job = _training_jobs[job_id]
    job["status"] = "running"

    try:
        task_type = config.get("task_type", "mil")
        if task_type == "segmentation":
            from pathclaw.training.seg_trainer import train_segmentation_model
            config["job_id"] = job_id
            result = train_segmentation_model(config, job_status=job)
        elif task_type == "patch_classification":
            from pathclaw.training.patch_trainer import train_patch_model
            config["_job_id"] = job_id
            result = train_patch_model(config, job_status=job)
        else:
            from pathclaw.training.trainer import train_mil_model
            result = train_mil_model(config, job_status=job)
        job["status"] = "completed"
        job["result"] = result
    except Exception as e:
        import traceback
        job["status"] = "failed"
        job["errors"].append(str(e))
        job["traceback"] = traceback.format_exc()

    # Save experiment record
    exp_dir = EXPERIMENTS_DIR / job_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    (exp_dir / "status.json").write_text(json.dumps(job, indent=2, default=str))
    if "result" in job and "metrics" in job.get("result", {}):
        (exp_dir / "metrics.json").write_text(json.dumps(job["result"]["metrics"], indent=2))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Launch a MIL or segmentation training job."""

    # Patch-classification path: per-patch labels + simple MLP, no MIL plumbing
    if config.task_type == "patch_classification":
        job_id = f"patch-{str(uuid.uuid4())[:8]}"
        job = {
            "job_id": job_id,
            "type": "patch_classification",
            "status": "queued",
            "config": config.model_dump(),
            "epoch": 0,
            "progress": 0.0,
            "metrics": {},
            "errors": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _training_jobs[job_id] = job
        background_tasks.add_task(_run_training, job_id, config.model_dump())
        return {
            "job_id": job_id,
            "status": "queued",
            "task_type": "patch_classification",
        }

    # Segmentation path: skip MIL-specific validation
    if config.task_type == "segmentation":
        job_id = f"seg-{str(uuid.uuid4())[:8]}"
        job = {
            "job_id": job_id,
            "type": "segmentation",
            "status": "queued",
            "config": config.model_dump(),
            "epoch": 0,
            "progress": 0.0,
            "metrics": {},
            "errors": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _training_jobs[job_id] = job
        background_tasks.add_task(_run_training, job_id, config.model_dump())
        return {
            "job_id": job_id,
            "status": "queued",
            "task_type": "segmentation",
            "seg_model": config.seg_model,
        }

    # --- MIL path (original logic below) ---

    # Validate backbone ↔ feature_dim consistency
    expected_dim = BACKBONE_DIMS.get(config.feature_backbone.lower())
    if expected_dim is not None and config.feature_dim != expected_dim:
        raise HTTPException(
            status_code=422,
            detail=(
                f"feature_dim={config.feature_dim} does not match backbone "
                f"'{config.feature_backbone}' (expected {expected_dim}). "
                f"Set feature_dim={expected_dim} or choose a different backbone."
            ),
        )

    # Pre-flight MAMMOTH import check
    if config.mammoth.enabled:
        try:
            import mammoth  # noqa: F401
        except ImportError:
            raise HTTPException(
                status_code=422,
                detail="mammoth-moe is not installed. Run: pip install mammoth-moe einops",
            )

    # Build labels.json from the dataset's label CSV if not already present
    dataset_meta_path = PATHCLAW_DATA_DIR / "datasets" / config.dataset_id / "meta.json"
    labels_json_path = PATHCLAW_DATA_DIR / "datasets" / config.dataset_id / "labels.json"
    if not labels_json_path.exists():
        if not dataset_meta_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{config.dataset_id}' not registered. Call register_dataset first.",
            )
        dataset_meta = json.loads(dataset_meta_path.read_text())
        label_file = dataset_meta.get("label_file")
        if not label_file:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No label CSV found in dataset '{config.dataset_id}'. "
                    f"Add a CSV file with a column named '{config.label_column}' "
                    f"where each row matches a slide filename (without extension)."
                ),
            )
        try:
            import pandas as pd
            df = pd.read_csv(label_file)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Cannot read label file '{label_file}': {e}")

        # Build candidate slide stems: registered WSI files first, then pre-extracted .pt features
        slide_stems = {Path(s["filename"]).stem for s in dataset_meta.get("slides", [])}
        # Include full filenames (with extension) for CSVs that store them that way.
        slide_names = {s["filename"] for s in dataset_meta.get("slides", [])}
        from pathclaw.preprocessing.feature_extraction import resolve_features_dir
        feature_dir = resolve_features_dir(config.dataset_id, config.feature_backbone)
        feature_stems = {p.stem for p in feature_dir.glob("*.pt")} if feature_dir.exists() else set()
        # For TCGA files: feature stems look like "TCGA-XX-XXXX.uuid" but CSV has "TCGA-XX-XXXX".
        # Build a prefix map: csv_value → feature_stem for prefix-based matching.
        prefix_map: dict[str, str] = {}
        for fstem in feature_stems:
            prefix_map[fstem] = fstem                    # exact match
            dot_idx = fstem.find(".")
            if dot_idx > 0:
                prefix_map[fstem[:dot_idx]] = fstem      # prefix match (barcode only)

        # Pick reference set for column detection: feature stems if no WSI slides registered
        ref_stems = slide_stems if slide_stems else set(prefix_map.keys())
        ref_all = ref_stems | slide_names   # match both with and without extension

        id_col = None
        for col in df.columns:
            col_vals = set(df[col].astype(str))
            overlap = ref_all & col_vals
            if len(overlap) / max(len(ref_stems), 1) > 0.3:
                id_col = col
                break
        if id_col is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Cannot find a column in '{label_file}' whose values match slide filenames. "
                    f"Ensure the CSV has a slide-ID column (e.g. 'slide_id') with values matching "
                    f"slide filenames (without extension)."
                ),
            )

        if config.label_column not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Column '{config.label_column}' not found in '{label_file}'. "
                    f"Available columns: {list(df.columns)}"
                ),
            )

        # Map label values → integer class indices.
        # Keys use the full feature stem (to match trainer's f.stem lookup) when available.
        unique_labels = sorted(df[config.label_column].dropna().unique(), key=str)
        label_to_int = {str(lbl): i for i, lbl in enumerate(unique_labels)}
        label_map: dict[str, int] = {}
        # Slide extensions to strip if the CSV column stores full filenames
        _slide_exts = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".vsi", ".scn", ".bif", ".qptiff"}
        for _, row in df.iterrows():
            csv_id = str(row[id_col])
            lbl_val = row.get(config.label_column)
            if pd.isna(lbl_val):
                continue
            int_label = label_to_int[str(lbl_val)]
            # Normalize csv_id: strip slide extension if present, so it matches feature stems
            stem_id = csv_id
            for ext in _slide_exts:
                if stem_id.lower().endswith(ext):
                    stem_id = stem_id[: -len(ext)]
                    break
            resolved = prefix_map.get(stem_id, stem_id)
            label_map[resolved] = int_label

        labels_json_path.write_text(json.dumps(label_map, indent=2))

    job_id = f"train-{str(uuid.uuid4())[:8]}"
    config_dict = config.model_dump()

    job = {
        "job_id": job_id,
        "type": "training",
        "status": "queued",
        "config": config_dict,
        "progress": 0.0,
        "epoch": 0,
        "total_epochs": config.training.epochs,
        "metrics": {},
        "errors": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _training_jobs[job_id] = job

    background_tasks.add_task(_run_training, job_id, config_dict)

    return {"job_id": job_id, "status": "queued", "config": config_dict}


@router.get("/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status and metrics (includes history if available)."""
    if job_id in _training_jobs:
        data = dict(_training_jobs[job_id])
    else:
        status_path = EXPERIMENTS_DIR / job_id / "status.json"
        if not status_path.exists():
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
        data = json.loads(status_path.read_text())

    # Attach history from history.json if not already present
    if "history" not in data:
        history_path = EXPERIMENTS_DIR / job_id / "history.json"
        if history_path.exists():
            data["history"] = json.loads(history_path.read_text())

    return data


@router.get("/{job_id}/plots")
async def list_training_plots(job_id: str):
    """List PNG plots generated during training."""
    plot_dir = EXPERIMENTS_DIR / job_id / "plots"
    if not plot_dir.exists():
        return {"job_id": job_id, "plots": []}
    return {
        "job_id": job_id,
        "plots": [{"name": p.name} for p in sorted(plot_dir.glob("*.png"))],
    }


@router.get("/{job_id}/plots/{filename}")
async def get_training_plot(job_id: str, filename: str):
    """Serve a training plot PNG file."""
    from fastapi.responses import FileResponse
    plot_path = EXPERIMENTS_DIR / job_id / "plots" / filename
    if not plot_path.exists() or not plot_path.suffix == ".png":
        raise HTTPException(status_code=404, detail=f"Plot {filename} not found")
    return FileResponse(str(plot_path), media_type="image/png")


@router.get("/{job_id}/logs")
async def get_training_logs(job_id: str):
    """Get training logs."""
    log_path = EXPERIMENTS_DIR / job_id / "train.log"
    if not log_path.exists():
        return {"job_id": job_id, "logs": "No logs available yet."}
    return {"job_id": job_id, "logs": log_path.read_text()[-5000:]}  # Last 5KB


@router.get("")
async def list_training_jobs():
    """List all training jobs."""
    return {"jobs": list(_training_jobs.values())}


# ---------------------------------------------------------------------------
# LoRA fine-tuning endpoints
# ---------------------------------------------------------------------------

class LoRAConfig(BaseModel):
    backbone: str = "uni"                    # backbone to fine-tune
    dataset_id: str
    label_column: str = ""                   # label column in labels.json
    num_classes: int = 2
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list[str] = Field(default_factory=list)  # empty = auto-detect
    epochs: int = 20
    lr: float = 5e-5
    batch_size: int = 64
    max_patches_per_slide: int = 200
    device: str = "auto"
    merge_adapter: bool = False              # merge LoRA into base after training


_lora_jobs: dict[str, dict] = {}


def _run_lora(job_id: str, config: dict):
    job = _lora_jobs[job_id]
    job["status"] = "running"
    try:
        from pathclaw.training.lora_finetuner import finetune_backbone_lora
        config["job_id"] = job_id
        result = finetune_backbone_lora(config, job_status=job)
        job["status"] = "completed"
        job["result"] = result
    except Exception as e:
        import traceback
        job["status"] = "failed"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()

    exp_dir = EXPERIMENTS_DIR / job_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "lora_status.json").write_text(json.dumps(job, indent=2, default=str))


@router.post("/lora/start")
async def start_lora(config: LoRAConfig, background_tasks: BackgroundTasks):
    """Launch LoRA fine-tuning of a foundation model backbone."""
    job_id = f"lora-{str(uuid.uuid4())[:8]}"
    job = {
        "job_id": job_id,
        "type": "lora",
        "status": "queued",
        "config": config.model_dump(),
        "epoch": 0,
        "progress": 0.0,
        "metrics": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _lora_jobs[job_id] = job
    background_tasks.add_task(_run_lora, job_id, config.model_dump())
    return {
        "job_id": job_id,
        "status": "queued",
        "backbone": config.backbone,
        "lora_rank": config.lora_rank,
    }


@router.get("/lora/{job_id}")
async def get_lora_status(job_id: str):
    """Get LoRA fine-tuning job status."""
    if job_id in _lora_jobs:
        return _lora_jobs[job_id]
    status_path = EXPERIMENTS_DIR / job_id / "lora_status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail=f"LoRA job {job_id} not found")
    return json.loads(status_path.read_text())
