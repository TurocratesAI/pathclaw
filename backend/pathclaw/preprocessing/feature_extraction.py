"""Foundation model feature extraction for WSI patches.

Loads a pre-trained backbone via HuggingFace Hub + timm, reads patch
coordinates from preprocessing output, and extracts per-slide feature
tensors (N × D) saved as .pt files.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
logger = logging.getLogger("pathclaw.features")


def resolve_features_dir(dataset_id: str, backbone: str = "") -> Path:
    """Resolve features directory for a dataset, preferring per-backbone subdirs.

    New layout: features/{dataset_id}/{backbone}/{slide_stem}.pt
    Legacy:     features/{dataset_id}/{slide_stem}.pt  (used when backbone subdir missing)
    """
    base = PATHCLAW_DATA_DIR / "features" / dataset_id
    if backbone:
        sub = base / backbone
        if sub.exists() and any(sub.glob("*.pt")):
            return sub
    # Fallback: if legacy flat layout has .pt files, use it
    if base.exists() and any(base.glob("*.pt")):
        return base
    # Default to new layout path (may not exist yet)
    return base / backbone if backbone else base

# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

BACKBONE_REGISTRY: dict[str, dict] = {
    "uni": {
        "hf_model_id": "MahmoodLab/UNI",
        "timm_model": "vit_large_patch16_224",
        "dim": 1024,
        "magnification": 20,
        "patch_px": 224,
        "gated": True,
    },
    "conch": {
        "hf_model_id": "MahmoodLab/CONCH",
        "timm_model": "conch",          # loaded via huggingface_hub directly
        "dim": 512,
        "magnification": 20,
        "patch_px": 224,
        "gated": True,
    },
    "ctranspath": {
        "hf_model_id": "X-zhangyang/CTransPath",
        "timm_model": "swin_tiny_patch4_window7_224",
        "dim": 768,
        "magnification": 20,
        "patch_px": 224,
        "gated": False,
    },
    "virchow": {
        "hf_model_id": "paige-ai/Virchow",
        "timm_model": "vit_huge_patch14_224",
        "dim": 1280,
        "magnification": 20,
        "patch_px": 224,
        "gated": True,
    },
    "virchow2": {
        "hf_model_id": "paige-ai/Virchow2",
        "timm_model": "vit_huge_patch14_224",
        "dim": 2560,
        "magnification": 20,
        "patch_px": 224,
        "gated": True,
    },
    "gigapath": {
        "hf_model_id": "prov-gigapath/prov-gigapath",
        "timm_model": "gigapath",
        "dim": 1536,
        "magnification": 20,
        "patch_px": 256,
        "gated": True,
    },
}


# Merge user-registered backbones at import time.
# Layout: ~/.pathclaw/backbones/custom_registry.json ->
# {"<id>": {"hf_model_id": ..., "timm_model": ..., "dim": int, "magnification": int,
#           "patch_px": int, "gated": bool}, ...}
_CUSTOM_REG_PATH = PATHCLAW_DATA_DIR / "backbones" / "custom_registry.json"


def _load_custom_backbones() -> dict[str, dict]:
    if not _CUSTOM_REG_PATH.exists():
        return {}
    try:
        return json.loads(_CUSTOM_REG_PATH.read_text())
    except Exception as e:
        logger.warning(f"Failed to parse custom backbone registry: {e}")
        return {}


def _persist_custom_backbone(entry_id: str, manifest: dict) -> None:
    _CUSTOM_REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_custom_backbones()
    existing[entry_id] = manifest
    _CUSTOM_REG_PATH.write_text(json.dumps(existing, indent=2))


def refresh_custom_backbones() -> None:
    """Re-merge custom backbones into BACKBONE_REGISTRY (call after registration)."""
    for k, v in _load_custom_backbones().items():
        BACKBONE_REGISTRY[k] = v


def list_backbones() -> dict[str, dict]:
    """Return the merged registry with a `custom: bool` flag per entry."""
    refresh_custom_backbones()
    custom_ids = set(_load_custom_backbones().keys())
    return {
        k: {**v, "custom": k in custom_ids}
        for k, v in BACKBONE_REGISTRY.items()
    }


def register_custom_backbone(
    entry_id: str,
    hf_model_id: str,
    timm_model: str,
    dim: int,
    patch_px: int = 224,
    magnification: int = 20,
    gated: bool = False,
) -> dict:
    manifest = {
        "hf_model_id": hf_model_id,
        "timm_model": timm_model,
        "dim": int(dim),
        "patch_px": int(patch_px),
        "magnification": int(magnification),
        "gated": bool(gated),
    }
    _persist_custom_backbone(entry_id, manifest)
    refresh_custom_backbones()
    return manifest


# Load user-registered backbones at import time.
refresh_custom_backbones()


# ---------------------------------------------------------------------------
# ImageNet normalisation (used by most ViT backbones)
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_transform(patch_px: int):
    """Standard ImageNet normalisation transform (no PIL / torchvision required)."""
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    def transform(img_array: np.ndarray) -> torch.Tensor:
        # img_array: (H, W, 3) uint8  →  (3, H, W) float32  →  normalised
        t = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        t = (t - mean) / std
        return t

    return transform


# ---------------------------------------------------------------------------
# Backbone loader
# ---------------------------------------------------------------------------

def _load_backbone(backbone_name: str, device: torch.device) -> torch.nn.Module:
    """Load the backbone model from HuggingFace Hub via timm or direct HF loading."""
    import timm
    from huggingface_hub import snapshot_download

    info = BACKBONE_REGISTRY[backbone_name]
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")

    logger.info(f"Loading backbone '{backbone_name}' from {info['hf_model_id']} …")

    # Download model weights / config from HF Hub.
    # Restrict to config + weight files so gated repos don't 403 on README/gitattributes fetches.
    weight_patterns = ["config.json", "*.safetensors", "*.bin", "*.pth", "*.pt"]
    try:
        model_dir = snapshot_download(
            repo_id=info["hf_model_id"],
            token=hf_token,
            local_dir=str(PATHCLAW_DATA_DIR / "backbones" / backbone_name),
            allow_patterns=weight_patterns,
        )
    except Exception as fetch_err:
        # If HF denied access but files are already cached, use the cache (offline mode).
        logger.warning(f"HF fetch failed for {info['hf_model_id']}: {fetch_err}. Falling back to local cache.")
        model_dir = snapshot_download(
            repo_id=info["hf_model_id"],
            local_dir=str(PATHCLAW_DATA_DIR / "backbones" / backbone_name),
            allow_patterns=weight_patterns,
            local_files_only=True,
        )

    # Read config.json for model kwargs (UNI, Virchow, GigaPath need init_values, mlp_ratio, reg_tokens, etc.)
    cfg_path = Path(model_dir) / "config.json"
    model_kwargs: dict = {"pretrained": False, "num_classes": 0}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        # HF timm configs nest the kwargs under "model_args"; older configs put them at the top level.
        src = cfg.get("model_args") or cfg
        for k in ("img_size", "init_values", "dynamic_img_size", "global_pool",
                  "mlp_ratio", "reg_tokens", "no_embed_class", "class_token"):
            if k in src:
                model_kwargs[k] = src[k]

    # Virchow / Virchow2 use SwiGLU-Packed MLP with SiLU activation — can't be stored in JSON config.
    if backbone_name in ("virchow", "virchow2"):
        from timm.layers import SwiGLUPacked
        model_kwargs["mlp_layer"] = SwiGLUPacked
        model_kwargs["act_layer"] = torch.nn.SiLU

    try:
        model = timm.create_model(info["timm_model"], **model_kwargs)

        # Load weights — prefer .pth/.pt, fall back to HuggingFace .bin
        ckpt_candidates = (
            list(Path(model_dir).glob("*.pth"))
            + list(Path(model_dir).glob("*.pt"))
            + list(Path(model_dir).glob("pytorch_model.bin"))
            + list(Path(model_dir).glob("model.safetensors"))
            + list(Path(model_dir).glob("*.bin"))
        )
        if ckpt_candidates:
            ckpt_path = ckpt_candidates[0]
            if ckpt_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                state = load_file(str(ckpt_path), device="cpu")
            else:
                state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            # Unwrap nested checkpoint formats
            for key in ("model", "state_dict", "teacher"):
                if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logger.warning(f"{len(missing)} missing keys (e.g. {missing[:3]})")
            if unexpected:
                logger.warning(f"{len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
            logger.info(f"Loaded weights from {ckpt_path.name}")
        else:
            logger.warning(f"No checkpoint file found in {model_dir} — using random weights (debugging only)")
    except Exception as e:
        logger.warning(f"timm load failed ({e}), re-raising …")
        raise

    model.eval()

    # Convert to fp16 BEFORE moving to GPU to avoid double memory allocation
    if device.type == "cuda":
        model = model.half()

    model = model.to(device)

    return model


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def _auto_batch_size(device: torch.device, backbone: str, default: int) -> int:
    """Pick the largest batch size that fits in GPU memory.

    MUST be called AFTER the model is loaded onto the GPU so that free
    memory already accounts for model weights.  We target 50% of the
    remaining free VRAM for batch data + activations.
    """
    if device.type != "cuda":
        return min(default, 64)
    try:
        idx = device.index or 0
        free_mb = torch.cuda.mem_get_info(idx)[0] / (1024 ** 2)
    except Exception:
        return default

    # Per-patch GPU cost estimate (MB): input tensor + intermediate activations
    # ViT-L  (UNI/CTransPath):  ~0.8 MB/patch
    # ViT-H  (Virchow/Virchow2/GigaPath): ~2.5 MB/patch (larger hidden dim + SwiGLU)
    if "virchow" in backbone or "gigapath" in backbone:
        per_patch_mb = 2.5
    else:
        per_patch_mb = 0.8

    usable_mb = free_mb * 0.50  # conservative: leave 50% headroom for activation peaks
    optimal = int(usable_mb / per_patch_mb)
    # Clamp to reasonable range
    bs = max(32, min(optimal, 512))
    logger.info(f"Auto batch size: {bs} (free GPU: {free_mb:.0f} MB, per-patch: {per_patch_mb} MB)")
    return bs


def extract_features(
    dataset_id: str,
    backbone: str = "uni",
    batch_size: int = 0,  # 0 = auto-detect based on GPU memory
    device_str: str = "auto",
    job_status: Optional[dict] = None,
) -> dict:
    """Extract patch features for all slides in a dataset.

    Reads patch coordinates from preprocessing output
    (``~/.pathclaw/preprocessed/{dataset_id}/{slide_stem}/coords.json``),
    reads the corresponding WSI patches, and saves per-slide feature
    tensors to ``~/.pathclaw/features/{dataset_id}/{slide_stem}.pt``.

    Skips slides that already have a .pt file (resume-safe).

    Performance: I/O workers are scaled to CPU count, batch size is
    auto-tuned to GPU memory, and a prefetch pipeline overlaps disk
    reads with GPU inference.
    """
    backbone = backbone.lower()
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Available: {sorted(BACKBONE_REGISTRY)}"
        )

    info = BACKBONE_REGISTRY[backbone]

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Load backbone model FIRST — auto batch size must measure free memory
    # after model weights are on the GPU, not before.
    model = _load_backbone(backbone, device)
    transform = _get_transform(info["patch_px"])

    # Auto-tune batch size to remaining free GPU memory (after model loaded)
    if batch_size <= 0:
        batch_size = _auto_batch_size(device, backbone, 256)
    logger.info(f"Using batch_size={batch_size} on {device}")

    # Output directory — namespace by backbone so multiple FMs can coexist
    features_dir = PATHCLAW_DATA_DIR / "features" / dataset_id / backbone
    features_dir.mkdir(parents=True, exist_ok=True)

    # Find preprocessed slides
    preprocessed_dir = PATHCLAW_DATA_DIR / "preprocessed" / dataset_id
    if not preprocessed_dir.exists():
        raise FileNotFoundError(
            f"No preprocessing output at {preprocessed_dir}. "
            f"Run wsi-preprocess first."
        )

    # Support both layout formats:
    # New (pipeline.py): preprocessed_dir/patches/{slide_stem}_coords.json
    # Old (legacy):      preprocessed_dir/{slide_stem}/coords.json
    patches_dir = preprocessed_dir / "patches"
    if patches_dir.is_dir():
        coords_files = sorted(patches_dir.glob("*_coords.json"))
    else:
        coords_files = []
        for slide_dir in sorted(preprocessed_dir.iterdir()):
            if slide_dir.is_dir() and (slide_dir / "coords.json").exists():
                coords_files.append(slide_dir / "coords.json")

    if not coords_files:
        raise FileNotFoundError(f"No coords files found in {preprocessed_dir}")

    total = len(coords_files)
    completed = 0
    errors: list[str] = []

    for coords_file in coords_files:
        # Derive slide stem from filename
        if patches_dir.is_dir():
            slide_stem = coords_file.stem[: -len("_coords")]  # strip _coords suffix
        else:
            slide_stem = coords_file.parent.name

        out_path = features_dir / f"{slide_stem}.pt"

        if out_path.exists():
            logger.info(f"Skipping {slide_stem} (already extracted)")
            completed += 1
            if job_status:
                job_status["slides_completed"] = completed
                job_status["progress"] = completed / total
            continue

        # Load patch coordinates
        raw = json.loads(coords_file.read_text())
        # New format: dict with 'patches' key containing list of {x, y, patch_size, level}
        # Old format: raw list of {x, y, patch_size, level}
        if isinstance(raw, dict):
            coords = raw.get("patches", [])
            # Slide path embedded in coords file takes precedence
            embedded_slide = raw.get("slide")
        else:
            coords = raw
            embedded_slide = None

        if not coords:
            logger.warning(f"{slide_stem}: empty coords — skipping")
            continue

        # Resolve the original WSI path
        if embedded_slide and Path(embedded_slide).exists():
            slide_path = Path(embedded_slide)
        else:
            # Fall back to dataset meta.json
            dataset_meta_path = PATHCLAW_DATA_DIR / "datasets" / dataset_id / "meta.json"
            slide_path = None
            if dataset_meta_path.exists():
                meta = json.loads(dataset_meta_path.read_text())
                for slide_info in meta.get("slides", []):
                    if Path(slide_info["path"]).stem == slide_stem:
                        slide_path = Path(slide_info["path"])
                        break

        if slide_path is None or not slide_path.exists():
            logger.warning(f"Original slide not found for {slide_stem} — skipping")
            errors.append(f"{slide_stem}: slide file not found")
            continue

        try:
            features_for_slide = _extract_slide_features(
                slide_path=slide_path,
                coords=coords,
                model=model,
                transform=transform,
                patch_px=info["patch_px"],
                batch_size=batch_size,
                device=device,
                fp16=(device.type == "cuda"),
                backbone_name=backbone,
            )

            torch.save(features_for_slide, out_path)  # already on CPU
            logger.info(
                f"Extracted {features_for_slide.shape[0]} patches × "
                f"{features_for_slide.shape[1]} dims → {out_path.name}"
            )
        except Exception as e:
            logger.error(f"Failed to extract {slide_stem}: {e}")
            errors.append(f"{slide_stem}: {e}")
            # Free fragmented GPU memory after OOM
            torch.cuda.empty_cache()

        completed += 1
        if job_status:
            job_status["slides_completed"] = completed
            job_status["slides_total"] = total
            job_status["progress"] = completed / total
            job_status["backbone"] = backbone
            job_status["feature_dim"] = info["dim"]
            job_status["errors"] = errors  # keep errors visible via API

    return {
        "slides_processed": completed,
        "slides_total": total,
        "feature_dim": info["dim"],
        "backbone": backbone,
        "output_dir": str(features_dir),
        "errors": errors,
    }


def _pool_virchow2(tokens: torch.Tensor) -> torch.Tensor:
    """Virchow2 pooling: concat(CLS token, mean of patch tokens).

    Virchow2 uses 1 CLS + 4 register tokens + 256 patch tokens = 261 tokens total.
    The official paper pooling is concat(cls, mean(patch_tokens)) → 2×1280 = 2560-d.
    """
    cls_tok = tokens[:, 0]              # (B, 1280)
    patch_toks = tokens[:, 5:]          # skip CLS + 4 reg tokens → (B, 256, 1280)
    mean_patch = patch_toks.mean(dim=1)  # (B, 1280)
    return torch.cat([cls_tok, mean_patch], dim=-1)  # (B, 2560)


def _extract_slide_features(
    slide_path: Path,
    coords: list[dict],
    model: torch.nn.Module,
    transform,
    patch_px: int,
    batch_size: int,
    device: torch.device,
    fp16: bool,
    backbone_name: str = "",
) -> torch.Tensor:
    """Read patches from the WSI and extract features in batches.

    Patch reading is parallelized with a thread pool (openslide releases the
    GIL during read_region), while GPU inference runs on the main thread.
    A prefetch pipeline overlaps I/O of the next batch with GPU work on the
    current batch.
    """
    import PIL.Image
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        import openslide
        slide = openslide.OpenSlide(str(slide_path))
        use_openslide = True
    except Exception:
        use_openslide = False
        slide_img = PIL.Image.open(str(slide_path))

    num_io_workers = min(16, max(4, (os.cpu_count() or 8) // 4))

    def _read_one(c: dict) -> torch.Tensor:
        x, y = c["x"], c["y"]
        ps = c["patch_size"]
        level = c.get("level", 0)
        try:
            if use_openslide:
                region = slide.read_region((x, y), level, (ps, ps)).convert("RGB")
            else:
                region = slide_img.crop((x, y, x + ps, y + ps)).convert("RGB")
            if ps != patch_px:
                region = region.resize((patch_px, patch_px), PIL.Image.BILINEAR)
            return transform(np.array(region))
        except Exception:
            return torch.zeros(3, patch_px, patch_px)

    def _read_batch(batch_coords: list[dict]) -> torch.Tensor:
        """Read a batch of patches using thread pool, return stacked tensor."""
        with ThreadPoolExecutor(max_workers=num_io_workers) as pool:
            tensors = list(pool.map(_read_one, batch_coords))
        return torch.stack(tensors)

    all_features: list[torch.Tensor] = []

    # Build batch slices
    batches = [
        coords[i: i + batch_size]
        for i in range(0, len(coords), batch_size)
    ]

    # Prefetch first batch
    from concurrent.futures import ThreadPoolExecutor as _TPE
    prefetch_pool = _TPE(max_workers=1)
    pending_future = prefetch_pool.submit(_read_batch, batches[0]) if batches else None

    for idx, batch_coords in enumerate(batches):
        # Wait for prefetched batch
        batch_tensor = pending_future.result()

        # Start prefetching next batch while GPU runs
        if idx + 1 < len(batches):
            pending_future = prefetch_pool.submit(_read_batch, batches[idx + 1])

        batch_tensor = batch_tensor.to(device)
        if fp16:
            batch_tensor = batch_tensor.half()

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=fp16):
            feats = model(batch_tensor)
            if backbone_name == "virchow2":
                feats = _pool_virchow2(feats)

        all_features.append(feats.float().cpu())
        del batch_tensor, feats

    prefetch_pool.shutdown(wait=False)

    if use_openslide:
        slide.close()

    if not all_features:
        raise RuntimeError("No features extracted — all patches failed")

    return torch.cat(all_features, dim=0)  # (N, D) on CPU
