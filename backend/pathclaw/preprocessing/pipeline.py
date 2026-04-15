"""WSI preprocessing pipeline — Otsu segmentation, patching, QC."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()


def otsu_tissue_mask(
    slide_path: str,
    level: int = 1,
    kernel_size: int = 7,
    max_thumb_px: int = 4096,
) -> tuple[np.ndarray, float]:
    """Generate a binary tissue mask using Otsu thresholding.

    Returns:
        mask: Binary tissue mask (True = tissue)
        tissue_pct: Percentage of area that is tissue
    """
    # Read a size-capped thumbnail for Otsu. `level` is accepted for backward
    # compatibility but is ignored — on large WSIs the requested level's full
    # dimensions can be gigapixels, which is both slow and wasteful for Otsu.
    try:
        import openslide
        slide = openslide.OpenSlide(slide_path)
        thumbnail = slide.get_thumbnail((max_thumb_px, max_thumb_px)).convert("RGB")
        slide.close()
    except Exception:
        thumbnail = Image.open(slide_path).convert("RGB")
        thumbnail.thumbnail((max_thumb_px, max_thumb_px))

    img_array = np.array(thumbnail)

    # Convert to grayscale
    gray = np.mean(img_array, axis=2).astype(np.uint8)

    # Otsu thresholding
    from skimage.filters import threshold_otsu
    try:
        threshold = threshold_otsu(gray)
    except ValueError:
        # If image is uniform, assume no tissue
        return np.zeros(gray.shape, dtype=bool), 0.0

    # Tissue is darker than background
    mask = gray < threshold

    # Optional morphological cleanup
    if kernel_size > 0:
        from skimage.morphology import binary_closing, disk
        mask = binary_closing(mask, disk(kernel_size))

    tissue_pct = float(np.sum(mask)) / mask.size * 100
    return mask, tissue_pct


def extract_patches(
    slide_path: str,
    tissue_mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 256,
    magnification: float = 20.0,
    min_tissue_pct: float = 0.5,
) -> list[dict]:
    """Extract patches from tissue regions of a WSI.
    
    Returns list of patch metadata dicts with coordinates.
    """
    try:
        import openslide
        slide = openslide.OpenSlide(slide_path)

        base_mag = float(slide.properties.get("openslide.objective-power", 40))
        downsample = base_mag / magnification
        level = slide.get_best_level_for_downsample(downsample)

        slide_w, slide_h = slide.level_dimensions[0]
        slide.close()
    except ImportError:
        return []

    mask_h, mask_w = tissue_mask.shape
    if mask_h == 0 or mask_w == 0 or slide_w <= patch_size or slide_h <= patch_size:
        return []

    scale_x = slide_w / mask_w
    scale_y = slide_h / mask_h

    xs = np.arange(0, slide_w - patch_size, stride, dtype=np.int64)
    ys = np.arange(0, slide_h - patch_size, stride, dtype=np.int64)
    if xs.size == 0 or ys.size == 0:
        return []

    # Integral image over the tissue mask → O(1) rectangle-sum lookups.
    mask_i32 = tissue_mask.astype(np.int32, copy=False)
    integral = np.pad(np.cumsum(np.cumsum(mask_i32, axis=0), axis=1), ((1, 0), (1, 0)))

    mx0 = np.clip((xs / scale_x).astype(np.int64), 0, mask_w)
    my0 = np.clip((ys / scale_y).astype(np.int64), 0, mask_h)
    mx1 = np.clip(((xs + patch_size) / scale_x).astype(np.int64), 0, mask_w)
    my1 = np.clip(((ys + patch_size) / scale_y).astype(np.int64), 0, mask_h)

    # Broadcast to (Ny, Nx)
    my0_c, my1_c = my0[:, None], my1[:, None]
    mx0_r, mx1_r = mx0[None, :], mx1[None, :]

    region_sum = (
        integral[my1_c, mx1_r]
        - integral[my0_c, mx1_r]
        - integral[my1_c, mx0_r]
        + integral[my0_c, mx0_r]
    )
    area = np.maximum((my1 - my0)[:, None] * (mx1 - mx0)[None, :], 1)
    ratios = region_sum.astype(np.float32) / area

    keep = np.argwhere(ratios >= min_tissue_pct)
    if keep.size == 0:
        return []

    xs_i64, ys_i64 = xs, ys
    return [
        {
            "x": int(xs_i64[xi]),
            "y": int(ys_i64[yi]),
            "patch_size": patch_size,
            "level": int(level),
            "tissue_pct": round(float(ratios[yi, xi]) * 100, 1),
        }
        for yi, xi in keep
    ]


def save_preview(
    slide_path: str,
    tissue_mask: np.ndarray,
    output_path: str,
    patches: Optional[list[dict]] = None,
):
    """Save a preview image showing tissue mask overlay."""
    try:
        import openslide
        slide = openslide.OpenSlide(slide_path)
        # Get a thumbnail
        thumb_size = (800, 800)
        thumbnail = slide.get_thumbnail(thumb_size)
        slide.close()
    except Exception:
        thumbnail = Image.open(slide_path).convert("RGB")
        thumbnail.thumbnail((800, 800))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(thumbnail)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(tissue_mask, cmap="Greens", alpha=0.7)
    axes[1].set_title(f"Tissue Mask ({np.sum(tissue_mask) / tissue_mask.size * 100:.1f}% tissue)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def preprocess_dataset(
    dataset_id: str,
    patch_size: int = 256,
    stride: int = 256,
    magnification: float = 20.0,
    otsu_level: int = 1,
    min_tissue_pct: float = 0.5,
    preview_only: bool = False,
    preview_count: int = 3,
    job_status: Optional[dict] = None,
) -> dict:
    """Run the full preprocessing pipeline on a dataset."""
    # Load dataset metadata
    datasets_dir = PATHCLAW_DATA_DIR / "datasets"
    meta_path = datasets_dir / dataset_id / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_id} not found")
    
    meta = json.loads(meta_path.read_text())
    slides = meta.get("slides", [])
    
    if job_status:
        job_status["slides_total"] = len(slides) if not preview_only else min(preview_count, len(slides))
    
    # Output directory
    output_dir = PATHCLAW_DATA_DIR / "preprocessed" / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    
    results = {
        "dataset_id": dataset_id,
        "slides_processed": 0,
        "total_patches": 0,
        "slide_results": [],
        "errors": [],
    }
    
    slides_to_process = slides[:preview_count] if preview_only else slides
    
    for i, slide_info in enumerate(slides_to_process):
        slide_path = slide_info["path"]
        slide_name = Path(slide_path).stem
        
        try:
            # 1. Otsu segmentation
            mask, tissue_pct = otsu_tissue_mask(slide_path, level=otsu_level)
            
            # 2. Extract patches
            patches = extract_patches(
                slide_path, mask,
                patch_size=patch_size,
                stride=stride,
                magnification=magnification,
                min_tissue_pct=min_tissue_pct,
            )
            
            # 3. Save preview
            preview_path = preview_dir / f"{slide_name}_preview.png"
            save_preview(slide_path, mask, str(preview_path), patches)
            
            # 4. Save patch coordinates
            coords_path = patches_dir / f"{slide_name}_coords.json"
            coords_path.write_text(json.dumps({
                "slide": slide_path,
                "patch_count": len(patches),
                "tissue_pct": tissue_pct,
                "config": {
                    "patch_size": patch_size,
                    "stride": stride,
                    "magnification": magnification,
                    "min_tissue_pct": min_tissue_pct,
                },
                "patches": patches,
            }, indent=2))
            
            slide_result = {
                "slide": slide_name,
                "tissue_pct": round(tissue_pct, 1),
                "patch_count": len(patches),
                "preview": str(preview_path),
            }
            results["slide_results"].append(slide_result)
            results["slides_processed"] += 1
            results["total_patches"] += len(patches)
            
            if job_status:
                job_status["slides_processed"] = results["slides_processed"]
                job_status["patches_extracted"] = results["total_patches"]
                job_status["progress"] = (i + 1) / len(slides_to_process)
                
        except Exception as e:
            error_msg = f"Error processing {slide_name}: {e}"
            results["errors"].append(error_msg)
            if job_status:
                job_status["errors"].append(error_msg)
    
    # Save results summary
    (output_dir / "preprocess_summary.json").write_text(json.dumps(results, indent=2))
    
    return results
