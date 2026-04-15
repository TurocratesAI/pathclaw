"""Attention heatmap generation from trained MIL models.

For each slide, runs a forward pass with return_attention=True to extract
per-patch attention scores, maps them onto patch coordinates, and saves:
  - {experiment_id}/heatmaps/{slide_stem}.json  — coords + scores (for tile serving)
  - {experiment_id}/heatmaps/{slide_stem}_thumb.png  — low-res preview thumbnail

The heatmap JSON has the format:
    {
        "slide_stem": str,
        "patch_size": int,
        "patches": [{"x": int, "y": int, "score": float}, ...]
    }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()


def _find_coords(dataset_id: str, slide_stem: str) -> list[dict]:
    """Try multiple locations for patch coordinates."""
    # Location 1: feature_extraction.py expects {preprocessed}/{dataset_id}/{slide_stem}/coords.json
    p1 = PATHCLAW_DATA_DIR / "preprocessed" / dataset_id / slide_stem / "coords.json"
    if p1.exists():
        data = json.loads(p1.read_text())
        return data if isinstance(data, list) else data.get("patches", [])

    # Location 2: pipeline.py saves {preprocessed}/{dataset_id}/patches/{slide_stem}_coords.json
    p2 = PATHCLAW_DATA_DIR / "preprocessed" / dataset_id / "patches" / f"{slide_stem}_coords.json"
    if p2.exists():
        data = json.loads(p2.read_text())
        return data if isinstance(data, list) else data.get("patches", [])

    raise FileNotFoundError(
        f"No coords found for slide '{slide_stem}' in dataset '{dataset_id}'. "
        f"Run preprocessing first."
    )


def _load_model(experiment_id: str) -> tuple[torch.nn.Module, dict]:
    """Load a trained MIL model from an experiment directory."""
    from pathclaw.training.trainer import create_model

    exp_dir = PATHCLAW_DATA_DIR / "experiments" / experiment_id
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment {experiment_id} not found (no config.json)")

    config = json.loads(config_path.read_text())

    # Find best checkpoint (holdout = model.pth, k-fold = fold0_model.pth)
    ckpt_path = exp_dir / "model.pth"
    if not ckpt_path.exists():
        folds = sorted(exp_dir.glob("fold*_model.pth"))
        if folds:
            ckpt_path = folds[0]
        else:
            raise FileNotFoundError(
                f"No model checkpoint found in {exp_dir}. "
                f"Run training first."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, config


def generate_attention_heatmap(
    experiment_id: str,
    dataset_id: str,
    slide_stem: str,
    job_status: Optional[dict] = None,
) -> dict:
    """Generate attention heatmap for a single slide.

    Returns a dict with the path to the saved JSON file.
    """
    out_dir = PATHCLAW_DATA_DIR / "experiments" / experiment_id / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{slide_stem}.json"

    if out_json.exists():
        return {"status": "cached", "path": str(out_json)}

    if job_status:
        job_status["status"] = "loading_model"

    model, config = _load_model(experiment_id)
    device = next(model.parameters()).device

    # Load features (prefer per-backbone subdir, fall back to legacy flat layout)
    backbone = config.get("feature_backbone", "uni")
    from pathclaw.preprocessing.feature_extraction import resolve_features_dir
    feat_path = resolve_features_dir(dataset_id, backbone) / f"{slide_stem}.pt"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"No feature file for slide '{slide_stem}' in dataset '{dataset_id}'. "
            f"Run feature extraction first."
        )
    features = torch.load(feat_path, map_location=device, weights_only=True)  # (N, D)
    if features.dim() == 1:
        features = features.unsqueeze(0)

    if job_status:
        job_status["status"] = "running_forward_pass"

    # Forward pass with attention
    with torch.no_grad():
        result = model(features, return_attention=True)

    if isinstance(result, tuple):
        _, attn = result
    else:
        # Model doesn't support return_attention — use uniform
        N = features.shape[0]
        attn = torch.full((N,), 1.0 / N, device=device)

    attn_np = attn.float().cpu().numpy()

    # Normalize to [0, 1]
    a_min, a_max = attn_np.min(), attn_np.max()
    if a_max > a_min:
        attn_norm = (attn_np - a_min) / (a_max - a_min)
    else:
        attn_norm = attn_np

    # Load coordinates
    if job_status:
        job_status["status"] = "loading_coords"

    coords = _find_coords(dataset_id, slide_stem)
    N_feat = len(attn_norm)
    N_coords = len(coords)
    N = min(N_feat, N_coords)

    if N_feat != N_coords:
        logger.warning(
            f"Feature count ({N_feat}) ≠ coord count ({N_coords}) for {slide_stem}. "
            f"Using first {N} entries."
        )

    patch_size = coords[0].get("patch_size", 256) if coords else 256
    patches_out = [
        {
            "x": int(coords[i]["x"]),
            "y": int(coords[i]["y"]),
            "score": float(attn_norm[i]),
        }
        for i in range(N)
    ]

    heatmap_data = {
        "slide_stem": slide_stem,
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "patch_size": patch_size,
        "patch_count": N,
        "patches": patches_out,
    }

    out_json.write_text(json.dumps(heatmap_data))
    logger.info(f"Heatmap saved → {out_json} ({N} patches)")

    # Generate thumbnail
    _save_thumbnail(heatmap_data, dataset_id, slide_stem, out_dir)

    if job_status:
        job_status["status"] = "done"
        job_status["path"] = str(out_json)

    return {"status": "done", "path": str(out_json), "patch_count": N}


def _save_thumbnail(
    heatmap_data: dict,
    dataset_id: str,
    slide_stem: str,
    out_dir: Path,
    thumb_size: int = 512,
) -> None:
    """Save a low-res PNG thumbnail of the heatmap overlay."""
    try:
        import numpy as np
        from PIL import Image

        patches = heatmap_data["patches"]
        if not patches:
            return

        xs = [p["x"] for p in patches]
        ys = [p["y"] for p in patches]
        ps = heatmap_data["patch_size"]

        w = max(xs) + ps
        h = max(ys) + ps

        scale = min(thumb_size / w, thumb_size / h)
        tw = max(1, int(w * scale))
        th = max(1, int(h * scale))

        # RGBA canvas (transparent background)
        canvas = np.zeros((th, tw, 4), dtype=np.uint8)

        # Jet colormap manually (R, G, B)
        def _jet(v: float) -> tuple[int, int, int]:
            v = max(0.0, min(1.0, v))
            r = min(1.0, max(0.0, 1.5 - abs(4 * v - 3.0)))
            g = min(1.0, max(0.0, 1.5 - abs(4 * v - 2.0)))
            b = min(1.0, max(0.0, 1.5 - abs(4 * v - 1.0)))
            return int(r * 255), int(g * 255), int(b * 255)

        for p in patches:
            x0 = int(p["x"] * scale)
            y0 = int(p["y"] * scale)
            x1 = min(tw, int((p["x"] + ps) * scale))
            y1 = min(th, int((p["y"] + ps) * scale))
            score = p["score"]
            r, g, b = _jet(score)
            alpha = int(128 + 100 * score)  # higher score = more opaque
            canvas[y0:y1, x0:x1] = [r, g, b, alpha]

        img = Image.fromarray(canvas, mode="RGBA")
        thumb_path = out_dir / f"{slide_stem}_thumb.png"
        img.save(thumb_path)
        logger.info(f"Heatmap thumbnail saved → {thumb_path}")
    except Exception as e:
        logger.warning(f"Thumbnail generation failed for {slide_stem}: {e}")


def list_heatmaps(experiment_id: str) -> list[str]:
    """Return slide stems that have generated heatmaps."""
    heatmap_dir = PATHCLAW_DATA_DIR / "experiments" / experiment_id / "heatmaps"
    if not heatmap_dir.exists():
        return []
    return [p.stem for p in heatmap_dir.glob("*.json")]


def get_heatmap_data(experiment_id: str, slide_stem: str) -> dict:
    """Load heatmap JSON for a slide."""
    p = PATHCLAW_DATA_DIR / "experiments" / experiment_id / "heatmaps" / f"{slide_stem}.json"
    if not p.exists():
        raise FileNotFoundError(f"No heatmap for {slide_stem} in experiment {experiment_id}")
    return json.loads(p.read_text())
