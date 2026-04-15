"""Config introspection endpoints — power the dual-mode config UI."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class RegisterBackbone(BaseModel):
    id: str
    hf_model_id: str
    timm_model: str = ""
    dim: int
    patch_px: int = 224
    magnification: int = 20
    gated: bool = False


@router.post("/backbones/register")
async def register_backbone(body: RegisterBackbone):
    """Register a custom HuggingFace foundation model as a new backbone.

    The entry is persisted to ~/.pathclaw/backbones/custom_registry.json and is
    immediately available to feature extraction runs.
    """
    if not body.id or not body.hf_model_id:
        raise HTTPException(400, detail="id and hf_model_id are required.")
    import re
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", body.id):
        raise HTTPException(400, detail=f"Invalid backbone id: {body.id!r} (use [A-Za-z0-9_-]).")
    from pathclaw.preprocessing.feature_extraction import register_custom_backbone
    manifest = register_custom_backbone(
        entry_id=body.id,
        hf_model_id=body.hf_model_id,
        timm_model=body.timm_model or body.hf_model_id,
        dim=body.dim,
        patch_px=body.patch_px,
        magnification=body.magnification,
        gated=body.gated,
    )
    return {"status": "ok", "id": body.id, "manifest": manifest}


@router.get("/mil-methods")
async def list_mil_methods():
    """Return all available MIL methods with descriptions and implementation status."""
    from pathclaw.training.mammoth_configs import MIL_METHODS
    return {
        "methods": [
            {
                "id": k,
                "name": v["full_name"],
                "description": v["description"],
                "best_for": v["best_for"],
                "implemented": v["implemented"],
                "ref": v.get("ref", ""),
                "mammoth_gain_avg": v.get("mammoth_gain_avg"),
            }
            for k, v in MIL_METHODS.items()
        ]
    }


@router.get("/backbones")
async def list_backbones():
    """Return all available feature backbones (built-in + user-registered)."""
    from pathclaw.training.mammoth_configs import BACKBONES
    from pathclaw.preprocessing.feature_extraction import list_backbones as _list_fe_backbones

    # Start with the curated mammoth_configs entries (descriptions + gating info)
    out = [
        {
            "id": k,
            "dim": v["dim"],
            "hf_id": v["hf_id"],
            "magnification": v["magnification"],
            "gated": v["gated"],
            "description": v["description"],
            "custom": False,
        }
        for k, v in BACKBONES.items()
    ]
    existing_ids = {b["id"] for b in out}
    # Overlay user-registered custom backbones (from feature_extraction registry)
    for bid, manifest in _list_fe_backbones().items():
        if not manifest.get("custom"):
            continue
        if bid in existing_ids:
            continue  # user id collides with built-in — keep built-in description
        out.append({
            "id": bid,
            "dim": manifest.get("dim"),
            "hf_id": manifest.get("hf_model_id", ""),
            "magnification": manifest.get("magnification", 20),
            "gated": manifest.get("gated", False),
            "description": "User-registered backbone",
            "custom": True,
        })
    return {"backbones": out}


@router.get("/mammoth-params")
async def get_mammoth_params():
    """Return the full MAMMOTH parameter schema with ranges and descriptions."""
    from pathclaw.training.mammoth_configs import MAMMOTH_PARAMS
    return {"params": MAMMOTH_PARAMS}


@router.get("/preprocess-params")
async def get_preprocess_params():
    """Return preprocessing parameter schema."""
    return {
        "params": {
            "patch_size": {
                "type": "int", "default": 256, "min": 128, "max": 1024, "step": 128,
                "description": "Patch size in pixels at the target magnification. 256px at 20x is standard.",
            },
            "stride": {
                "type": "int", "default": 256, "min": 64, "max": 1024, "step": 64,
                "description": "Stride between patch centres. Equal to patch_size → non-overlapping (default). "
                               "Smaller → overlapping patches (more coverage, more compute).",
            },
            "magnification": {
                "type": "float", "default": 20.0, "options": [5.0, 10.0, 20.0, 40.0],
                "description": "Target magnification level. All Mahmood Lab backbones expect 20x.",
            },
            "otsu_level": {
                "type": "int", "default": 1, "min": 0, "max": 3,
                "description": "OpenSlide pyramid level for computing the Otsu tissue mask. "
                               "Level 1 (~4× downsampled) is fast. Level 0 is pixel-accurate but slow.",
            },
            "min_tissue_pct": {
                "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                "description": "Minimum fraction of patch area that must be tissue. "
                               "0.5 = discard patches with < 50% tissue.",
            },
            "preview_only": {
                "type": "bool", "default": False,
                "description": "Process only 3 slides for QC preview before the full run.",
            },
        }
    }


@router.get("/training-params")
async def get_training_params():
    """Return training hyperparameter schema."""
    return {
        "params": {
            "epochs": {
                "type": "int", "default": 100, "min": 10, "max": 500,
                "description": "Number of training epochs.",
            },
            "lr": {
                "type": "float", "default": 1e-4, "min": 1e-6, "max": 1e-2,
                "description": "Initial learning rate.",
            },
            "weight_decay": {
                "type": "float", "default": 1e-5, "min": 0.0, "max": 1e-2,
                "description": "L2 regularisation weight decay.",
            },
            "optimizer": {
                "type": "enum",
                "default": "adam",
                "options": [
                    {"value": "adam",  "label": "Adam (default)"},
                    {"value": "adamw", "label": "AdamW (better with large models)"},
                    {"value": "sgd",   "label": "SGD + momentum"},
                    {"value": "radam", "label": "RAdam (rectified Adam)"},
                ],
                "description": "Gradient descent optimizer.",
            },
            "scheduler": {
                "type": "enum",
                "default": "cosine",
                "options": [
                    {"value": "cosine",  "label": "Cosine Annealing (default)"},
                    {"value": "step",    "label": "Step LR (decays at 1/3 intervals)"},
                    {"value": "plateau", "label": "Reduce on Plateau (adapts to val acc)"},
                    {"value": "none",    "label": "None (constant LR)"},
                ],
                "description": "Learning rate schedule.",
            },
            "early_stopping_patience": {
                "type": "int", "default": 0, "min": 0, "max": 100,
                "description": "Stop if val accuracy doesn't improve for N epochs. 0 = disabled.",
            },
        }
    }


@router.get("/defaults")
async def get_defaults(mode: str = "beginner", task_type: str = "subtyping", dataset_size: int = 200):
    """Return recommended config defaults.

    Args:
        mode: 'beginner' (minimal fields) or 'advanced' (all fields).
        task_type: 'subtyping', 'molecular', or 'grading'.
        dataset_size: Approximate number of slides (informs MAMMOTH recommendations).
    """
    from pathclaw.training.mammoth_configs import get_recommended_config
    cfg = get_recommended_config(task_type=task_type, dataset_size=dataset_size)

    if mode == "beginner":
        # Surface only the most important decisions
        return {
            "mode": "beginner",
            "config": {
                "mil_method": cfg["mil_method"],
                "feature_backbone": cfg["feature_backbone"],
                "mammoth_enabled": cfg["mammoth"]["enabled"],
                "epochs": cfg["training"]["epochs"],
                "num_classes": cfg["num_classes"],
                "eval_strategy": cfg["evaluation"]["strategy"],
            },
            "description": (
                f"Smart defaults for {task_type} with ~{dataset_size} slides. "
                f"MAMMOTH is {'enabled' if cfg['mammoth']['enabled'] else 'disabled'} "
                f"(recommended for this dataset size)."
            ),
        }
    else:
        return {"mode": "advanced", "config": cfg}
