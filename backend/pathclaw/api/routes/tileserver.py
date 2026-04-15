"""DZI tile server for whole slide images using openslide-python.

Serves DeepZoom Image (DZI) tiles directly from WSI files so the frontend
can render them with OpenSeadragon without any external tile server.

Endpoints:
    GET /api/tiles/{dataset_id}/{slide_stem}/info
    GET /api/tiles/{dataset_id}/{slide_stem}/thumbnail
    GET /api/tiles/{dataset_id}/{slide_stem}/dzi.dzi
    GET /api/tiles/{dataset_id}/{slide_stem}/{level}/{col}_{row}.jpeg
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, Response

_slide_executor = ThreadPoolExecutor(max_workers=4)

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
DATASETS_DIR = PATHCLAW_DATA_DIR / "datasets"

TILE_SIZE = 256
TILE_OVERLAP = 1
TILE_FORMAT = "jpeg"

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _open_slide(path: str):
    """Open a slide and cache the handle (LRU, max 8 open slides)."""
    try:
        import openslide
        return openslide.OpenSlide(path)
    except Exception as e:
        raise RuntimeError(f"Cannot open slide {path}: {e}") from e


def _get_slide_path(dataset_id: str, slide_stem: str) -> str:
    """Resolve the full path of a slide file from dataset metadata."""
    meta_path = DATASETS_DIR / dataset_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    meta = json.loads(meta_path.read_text())
    slides = meta.get("slides", [])

    # Match by stem (filename without extension)
    for slide in slides:
        p = Path(slide["path"])
        if p.stem == slide_stem or p.name == slide_stem:
            if Path(slide["path"]).exists():
                return slide["path"]
            raise HTTPException(
                status_code=404,
                detail=f"Slide file not found at {slide['path']}. Was it moved?"
            )

    raise HTTPException(
        status_code=404,
        detail=f"Slide '{slide_stem}' not found in dataset {dataset_id}"
    )


def _dzi_level_count(width: int, height: int) -> int:
    """Number of DZI levels = ceil(log2(max(w, h))) + 1."""
    return int(math.ceil(math.log2(max(width, height)))) + 1


def _level_dimensions(width: int, height: int, dzi_level: int) -> tuple[int, int]:
    """Width and height of the image at a given DZI level."""
    max_level = _dzi_level_count(width, height) - 1
    scale = 2 ** (max_level - dzi_level)
    return max(1, math.ceil(width / scale)), max(1, math.ceil(height / scale))


def _slide_to_pil(slide, region_x: int, region_y: int, region_w: int, region_h: int,
                  out_w: int, out_h: int):
    """Read a region from the slide and resize to (out_w, out_h).

    Uses the best matching openslide level to avoid reading massive
    full-resolution regions when only a low-res tile is needed.
    Clamps coordinates to slide bounds and fills excess with white.
    """
    from PIL import Image

    slide_w, slide_h = slide.dimensions

    # Pick the best openslide level: largest downsample that is <= our target downsample
    target_downsample = max(region_w / max(out_w, 1), region_h / max(out_h, 1))
    best_level = 0
    for lvl in range(slide.level_count):
        if slide.level_downsamples[lvl] <= target_downsample * 1.01:
            best_level = lvl
        else:
            break
    ds = slide.level_downsamples[best_level]

    # Clamp to slide bounds
    read_x = max(0, min(region_x, slide_w - 1))
    read_y = max(0, min(region_y, slide_h - 1))
    read_w = max(1, min(region_w, slide_w - read_x))
    read_h = max(1, min(region_h, slide_h - read_y))

    # Read at the chosen level — openslide wants level-0 coordinates for
    # the top-left corner but the size in level-N pixels.
    level_read_w = max(1, int(math.ceil(read_w / ds)))
    level_read_h = max(1, int(math.ceil(read_h / ds)))

    region = slide.read_region((read_x, read_y), best_level, (level_read_w, level_read_h))
    region = region.convert("RGB")

    # If clamped, we may need to pad
    if read_w < region_w or read_h < region_h:
        pad_w = max(1, int(math.ceil(region_w / ds)))
        pad_h = max(1, int(math.ceil(region_h / ds)))
        padded = Image.new("RGB", (pad_w, pad_h), (255, 255, 255))
        padded.paste(region, (0, 0))
        region = padded

    if region.size != (out_w, out_h):
        region = region.resize((out_w, out_h), Image.LANCZOS)

    return region


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

async def _open_slide_async(path: str):
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(_slide_executor, _open_slide, path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/{slide_stem}/info")
async def slide_info(dataset_id: str, slide_stem: str):
    """Return slide metadata: dimensions, level count, mpp, properties."""
    path = _get_slide_path(dataset_id, slide_stem)
    slide = await _open_slide_async(path)

    w, h = slide.dimensions
    props = dict(slide.properties)
    mpp_x = props.get("openslide.mpp-x", props.get("tiff.XResolution", None))
    mpp_y = props.get("openslide.mpp-y", props.get("tiff.YResolution", None))

    return {
        "dataset_id": dataset_id,
        "slide_stem": slide_stem,
        "width": w,
        "height": h,
        "level_count": slide.level_count,
        "level_dimensions": list(slide.level_dimensions),
        "dzi_levels": _dzi_level_count(w, h),
        "mpp_x": float(mpp_x) if mpp_x else None,
        "mpp_y": float(mpp_y) if mpp_y else None,
    }


@router.get("/{dataset_id}/{slide_stem}/thumbnail")
async def slide_thumbnail(dataset_id: str, slide_stem: str, size: int = 512):
    """Return a JPEG thumbnail of the slide at the given max dimension."""
    path = _get_slide_path(dataset_id, slide_stem)
    slide = await _open_slide_async(path)

    size = min(max(64, size), 2048)
    thumb = slide.get_thumbnail((size, size))
    thumb = thumb.convert("RGB")

    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{dataset_id}/{slide_stem}/dzi.dzi")
async def dzi_descriptor(dataset_id: str, slide_stem: str):
    """Return the DZI XML descriptor for this slide."""
    path = _get_slide_path(dataset_id, slide_stem)
    slide = await _open_slide_async(path)

    w, h = slide.dimensions
    tile_url_base = f"/api/tiles/{dataset_id}/{slide_stem}/"

    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
        f'Format="{TILE_FORMAT}" '
        f'Overlap="{TILE_OVERLAP}" '
        f'TileSize="{TILE_SIZE}">'
        f'<Size Width="{w}" Height="{h}"/>'
        '</Image>'
    )

    return Response(
        content=xml,
        media_type="application/xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{dataset_id}/{slide_stem}/heatmap/{experiment_id}/dzi.dzi")
async def heatmap_dzi_descriptor(dataset_id: str, slide_stem: str, experiment_id: str):
    """Return a DZI XML descriptor for the heatmap overlay (same dimensions as slide)."""
    path = _get_slide_path(dataset_id, slide_stem)
    slide = await _open_slide_async(path)

    w, h = slide.dimensions
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
        'Format="png" '
        f'Overlap="{TILE_OVERLAP}" '
        f'TileSize="{TILE_SIZE}">'
        f'<Size Width="{w}" Height="{h}"/>'
        '</Image>'
    )
    return Response(
        content=xml,
        media_type="application/xml",
        headers={"Cache-Control": "public, max-age=300"},
    )


@router.get("/{dataset_id}/{slide_stem}/heatmap/{experiment_id}/{dzi_level}/{tile_col}_{tile_row}.png")
async def heatmap_tile(
    dataset_id: str,
    slide_stem: str,
    experiment_id: str,
    dzi_level: int,
    tile_col: int,
    tile_row: int,
):
    """Return a semi-transparent PNG heatmap tile for OpenSeadragon overlay."""
    import numpy as np
    from pathclaw.evaluation.heatmap import get_heatmap_data
    from PIL import Image as _PIL

    path = _get_slide_path(dataset_id, slide_stem)
    slide = await _open_slide_async(path)

    try:
        hmap = get_heatmap_data(experiment_id, slide_stem)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    slide_w, slide_h = slide.dimensions
    max_level = _dzi_level_count(slide_w, slide_h) - 1
    scale = 2 ** (max_level - dzi_level)

    tile_x_dzi = tile_col * TILE_SIZE - (TILE_OVERLAP if tile_col > 0 else 0)
    tile_y_dzi = tile_row * TILE_SIZE - (TILE_OVERLAP if tile_row > 0 else 0)
    level_w, level_h = _level_dimensions(slide_w, slide_h, dzi_level)
    tile_x_end = min(tile_x_dzi + TILE_SIZE + 2 * TILE_OVERLAP, level_w)
    tile_y_end = min(tile_y_dzi + TILE_SIZE + 2 * TILE_OVERLAP, level_h)

    out_w = tile_x_end - tile_x_dzi
    out_h = tile_y_end - tile_y_dzi
    if out_w <= 0 or out_h <= 0:
        raise HTTPException(status_code=404, detail="Tile out of bounds")

    # Map tile bounds back to level-0 slide coordinates
    tile_slide_x = int(tile_x_dzi * scale)
    tile_slide_y = int(tile_y_dzi * scale)
    tile_slide_w = int(out_w * scale)
    tile_slide_h = int(out_h * scale)

    # RGBA canvas
    canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    patch_size = hmap.get("patch_size", 256)

    def _jet(v: float) -> tuple[int, int, int]:
        v = max(0.0, min(1.0, v))
        r = min(1.0, max(0.0, 1.5 - abs(4 * v - 3.0)))
        g = min(1.0, max(0.0, 1.5 - abs(4 * v - 2.0)))
        b = min(1.0, max(0.0, 1.5 - abs(4 * v - 1.0)))
        return int(r * 255), int(g * 255), int(b * 255)

    for p in hmap["patches"]:
        px, py = p["x"], p["y"]
        # Check if this patch overlaps the tile
        if px + patch_size <= tile_slide_x or px >= tile_slide_x + tile_slide_w:
            continue
        if py + patch_size <= tile_slide_y or py >= tile_slide_y + tile_slide_h:
            continue

        score = p["score"]
        red, green, blue = _jet(score)
        alpha = int(80 + 130 * score)  # 80–210 alpha range

        # Map patch coords to tile pixel coords
        x0 = max(0, int((px - tile_slide_x) / scale))
        y0 = max(0, int((py - tile_slide_y) / scale))
        x1 = min(out_w, int((px + patch_size - tile_slide_x) / scale))
        y1 = min(out_h, int((py + patch_size - tile_slide_y) / scale))

        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = [red, green, blue, alpha]

    img = _PIL.fromarray(canvas, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=300"},
    )


@router.get("/{dataset_id}/{slide_stem}/{dzi_level}/{tile_col}_{tile_row}.jpeg")
async def dzi_tile(
    dataset_id: str,
    slide_stem: str,
    dzi_level: int,
    tile_col: int,
    tile_row: int,
):
    """Return a single DZI tile as JPEG."""
    path = _get_slide_path(dataset_id, slide_stem)
    slide = await _open_slide_async(path)

    slide_w, slide_h = slide.dimensions
    max_level = _dzi_level_count(slide_w, slide_h) - 1

    if dzi_level < 0 or dzi_level > max_level:
        raise HTTPException(status_code=404, detail=f"Invalid DZI level {dzi_level}")

    # Scale factor: how many slide pixels per DZI pixel at this level
    scale = 2 ** (max_level - dzi_level)

    # Tile origin in DZI-level coordinates (with overlap)
    tile_x_dzi = tile_col * TILE_SIZE - (TILE_OVERLAP if tile_col > 0 else 0)
    tile_y_dzi = tile_row * TILE_SIZE - (TILE_OVERLAP if tile_row > 0 else 0)

    # Tile end in DZI-level coordinates
    level_w, level_h = _level_dimensions(slide_w, slide_h, dzi_level)
    tile_x_end = min(tile_x_dzi + TILE_SIZE + 2 * TILE_OVERLAP, level_w)
    tile_y_end = min(tile_y_dzi + TILE_SIZE + 2 * TILE_OVERLAP, level_h)

    out_w = tile_x_end - tile_x_dzi
    out_h = tile_y_end - tile_y_dzi

    if out_w <= 0 or out_h <= 0:
        raise HTTPException(status_code=404, detail="Tile out of bounds")

    # Map DZI coordinates back to level-0 slide coordinates
    region_x = int(tile_x_dzi * scale)
    region_y = int(tile_y_dzi * scale)
    region_w = int(out_w * scale)
    region_h = int(out_h * scale)

    try:
        img = _slide_to_pil(slide, region_x, region_y, region_w, region_h, out_w, out_h)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tile read error: {e}")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/{dataset_id}/{slide_stem}/dzi_files/{dzi_level}/{tile_col}_{tile_row}.jpeg")
async def dzi_tile_compat(
    dataset_id: str,
    slide_stem: str,
    dzi_level: int,
    tile_col: int,
    tile_row: int,
):
    """OpenSeadragon-compatible tile URL (dzi_files/ prefix)."""
    return await dzi_tile(dataset_id, slide_stem, dzi_level, tile_col, tile_row)


# ---------------------------------------------------------------------------
# GeoJSON annotation overlay
# ---------------------------------------------------------------------------

def _geojson_path(dataset_id: str, slide_stem: str) -> Path:
    return DATASETS_DIR / dataset_id / "geojson" / f"{slide_stem}.geojson"


@router.post("/{dataset_id}/{slide_stem}/geojson")
async def upload_geojson(dataset_id: str, slide_stem: str, file: UploadFile = File(...)):
    """Store a GeoJSON annotation file for a slide (e.g. exported from QuPath)."""
    raw = await file.read()
    try:
        data = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if data.get("type") not in ("FeatureCollection", "Feature", "GeometryCollection"):
        raise HTTPException(status_code=400, detail="Expected a GeoJSON FeatureCollection")
    dst = _geojson_path(dataset_id, slide_stem)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(raw)
    n = len(data.get("features", [data] if data.get("type") == "Feature" else []))
    return {"status": "ok", "features": n, "slide_stem": slide_stem}


@router.get("/{dataset_id}/{slide_stem}/geojson")
async def get_geojson(dataset_id: str, slide_stem: str):
    """Return stored GeoJSON for a slide."""
    p = _geojson_path(dataset_id, slide_stem)
    if not p.exists():
        raise HTTPException(status_code=404, detail="No GeoJSON found for this slide")
    return JSONResponse(content=json.loads(p.read_text()))


@router.delete("/{dataset_id}/{slide_stem}/geojson")
async def delete_geojson(dataset_id: str, slide_stem: str):
    """Remove stored GeoJSON annotation for a slide."""
    p = _geojson_path(dataset_id, slide_stem)
    if p.exists():
        p.unlink()
    return {"status": "ok"}
