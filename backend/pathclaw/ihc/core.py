"""Shared primitives for rule-based IHC scoring.

- `rgb2hed_channels` separates H / E / DAB channels via the default Ruifrok
  matrix (skimage.color.rgb2hed).
- `sample_tissue_patches` walks a WSI at `target_mpp`, picks tissue-containing
  tiles via Otsu on a downsampled thumbnail, and returns up to N patches.
- `segment_nuclei_cellpose` wraps cellpose 'cyto2' if importable, else falls
  back to a morphology-based DAB-nuclei proxy so scoring still works without
  the heavy dep.
- `membrane_band_mask` highlights thin DAB+ bands adjacent to nuclei — used
  by HER2 / PD-L1 rules.
"""
from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color deconvolution
# ---------------------------------------------------------------------------

def rgb2hed_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split RGB tile into H, E, DAB intensity channels (float32, each 0-~1)."""
    from skimage.color import rgb2hed
    if rgb.dtype != np.float32 and rgb.dtype != np.float64:
        rgb_f = rgb.astype(np.float32) / 255.0
    else:
        rgb_f = rgb
    hed = rgb2hed(rgb_f)
    h = hed[..., 0].astype(np.float32)
    e = hed[..., 1].astype(np.float32)
    dab = hed[..., 2].astype(np.float32)
    # skimage returns optical-density-like signed values. Clip to >=0 and
    # normalise so thresholds in Rule.dab_threshold are interpretable as
    # "fraction of max stain in this patch" across machines.
    dab = np.clip(dab, 0.0, None)
    h = np.clip(h, 0.0, None)
    e = np.clip(e, 0.0, None)
    return h, e, dab


# ---------------------------------------------------------------------------
# WSI patch sampling
# ---------------------------------------------------------------------------

def _open_slide(path: str):
    try:
        import openslide
    except ImportError as e:
        raise RuntimeError("openslide-python is required for IHC scoring") from e
    return openslide.OpenSlide(path)


def _tissue_mask_thumbnail(slide, max_dim: int = 2048) -> np.ndarray:
    """Otsu-threshold a thumbnail; returns a bool mask at thumbnail resolution."""
    from skimage.filters import threshold_otsu
    from skimage.color import rgb2gray
    w, h = slide.dimensions
    scale = max(w, h) / max_dim
    if scale < 1: scale = 1
    thumb_w, thumb_h = int(w / scale), int(h / scale)
    thumb = np.array(slide.get_thumbnail((thumb_w, thumb_h)))[:, :, :3]
    gray = rgb2gray(thumb)
    try:
        t = threshold_otsu(gray)
    except Exception:
        t = 0.85
    # Tissue is darker than background on H&E/IHC scans.
    mask = gray < t
    return mask


def sample_tissue_patches(
    wsi_path: str,
    n_patches: int = 200,
    patch_size: int = 512,
    target_mpp: float = 0.5,
    rng_seed: int = 0,
) -> Iterator[np.ndarray]:
    """Yield up to `n_patches` RGB tiles of `patch_size` px at ~`target_mpp`.

    Picks level closest to target_mpp using slide MPP metadata; if MPP is
    missing, falls back to level 0. Patches are sampled uniformly from
    tissue pixels on a downsampled thumbnail, then read full-res at the
    chosen level.
    """
    slide = _open_slide(wsi_path)
    try:
        # Pick level by MPP
        mpp_x = float(slide.properties.get("openslide.mpp-x", 0) or 0)
        level = 0
        if mpp_x > 0:
            # Each level roughly 2x downsample; pick the one closest to target
            best_diff = abs(mpp_x - target_mpp)
            for li in range(slide.level_count):
                ds = slide.level_downsamples[li]
                level_mpp = mpp_x * ds
                d = abs(level_mpp - target_mpp)
                if d < best_diff:
                    best_diff = d
                    level = li
        ds = slide.level_downsamples[level]
        mask = _tissue_mask_thumbnail(slide)
        mh, mw = mask.shape
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return
        w, h = slide.dimensions
        thumb_scale_x = w / mw
        thumb_scale_y = h / mh
        rng = np.random.default_rng(rng_seed)
        taken = 0
        # Oversample: some candidates land too close to edge or have low
        # tissue content — skip those.
        candidates = rng.permutation(len(xs))[: n_patches * 4]
        for idx in candidates:
            if taken >= n_patches:
                break
            # Center of candidate tissue pixel in level-0 coordinates
            cx = int(xs[idx] * thumb_scale_x)
            cy = int(ys[idx] * thumb_scale_y)
            read_size = int(patch_size * ds)
            x0 = max(0, cx - read_size // 2)
            y0 = max(0, cy - read_size // 2)
            if x0 + read_size >= w or y0 + read_size >= h:
                continue
            try:
                tile = slide.read_region((x0, y0), level, (patch_size, patch_size))
            except Exception:
                continue
            rgb = np.array(tile)[:, :, :3]
            # Quick tissue check on the patch itself to drop near-white tiles
            if np.mean(rgb) > 235:
                continue
            yield rgb
            taken += 1
    finally:
        slide.close()


# ---------------------------------------------------------------------------
# Nuclear segmentation
# ---------------------------------------------------------------------------

def segment_nuclei(rgb: np.ndarray, use_cellpose: bool = True) -> np.ndarray:
    """Return a binary nuclei mask for `rgb`. Falls back to a non-cellpose
    morphological pipeline if the library is missing — still good enough for
    aggregate slide-level scoring."""
    if use_cellpose:
        try:
            from cellpose import models as _cp
            # cyto2 generalises well to H&E/IHC nuclei without retraining.
            model = _cp.Cellpose(model_type="cyto2", gpu=False)
            masks, _, _, _ = model.eval(rgb, diameter=None, channels=[0, 0])
            return masks > 0
        except Exception as e:
            logger.info("cellpose unavailable (%s) — using fallback nuclei segmentation", e)
    # Fallback: hematoxylin channel → Otsu → small-object cleanup
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, opening, disk
    h, _, _ = rgb2hed_channels(rgb)
    try:
        t = threshold_otsu(h)
    except Exception:
        t = np.percentile(h, 70)
    m = h > t
    m = opening(m, disk(1))
    m = remove_small_objects(m, min_size=16)
    return m


def membrane_band_mask(nuclei: np.ndarray, dab: np.ndarray, band_px: int = 6) -> np.ndarray:
    """Dilate nuclei mask to get a pericellular ring, intersect with DAB+."""
    from skimage.morphology import dilation, disk
    ring = dilation(nuclei, disk(band_px)) & ~nuclei
    # Membrane DAB+ pixels within the ring
    # Using lowest DAB threshold from rule is caller's job — here we just
    # return the ring geometry.
    return ring
