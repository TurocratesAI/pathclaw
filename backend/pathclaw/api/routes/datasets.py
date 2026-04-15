"""Dataset management routes."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
DATASETS_DIR = PATHCLAW_DATA_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

# Supported slide extensions
SLIDE_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".vsi", ".scn", ".bif", ".qptiff"}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DatasetCreate(BaseModel):
    name: str
    path: str
    description: str = ""
    session_id: str = ""


class DatasetInfo(BaseModel):
    id: str
    name: str
    path: str
    description: str
    slide_count: int
    total_size_mb: float
    label_file: Optional[str] = None
    created_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_slides(root: Path) -> list[dict]:
    """Scan a directory recursively for whole-slide image files."""
    slides = []
    for ext in SLIDE_EXTENSIONS:
        for p in root.rglob(f"*{ext}"):
            slides.append({
                "filename": p.name,
                "path": str(p),
                "size_mb": round(p.stat().st_size / (1024**2), 2),
                "extension": ext,
            })
    return sorted(slides, key=lambda s: s["filename"])


def _find_label_files(root: Path) -> list[str]:
    """Look for CSV/TSV files that might contain labels."""
    label_files = []
    for ext in (".csv", ".tsv", ".xlsx"):
        for p in root.rglob(f"*{ext}"):
            label_files.append(str(p))
    return label_files


def _load_dataset_meta(dataset_id: str) -> dict:
    meta_path = DATASETS_DIR / dataset_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return json.loads(meta_path.read_text())


def _scan_cohort_siblings(slides_path_str: str) -> Optional[dict]:
    """Scan the cohort root (parent of the slides dir) for MAF / clinical / labels.

    Returns None if the cohort dir has no recognisable genomics sibling. This keeps
    self-registered non-TCGA datasets clean while auto-lighting up TCGA downloads.
    """
    try:
        slides_path = Path(slides_path_str)
        cohort = slides_path.parent
        if not cohort.exists():
            return None
    except Exception:
        return None

    out = {"cohort_root": str(cohort), "labels": []}
    found_any = False
    for sub in ("maf", "clinical", "expression", "cnv"):
        d = cohort / sub
        if not d.exists():
            continue
        files = [f for f in d.rglob("*") if f.is_file()]
        if not files:
            continue
        found_any = True
        total = sum(f.stat().st_size for f in files)
        out[sub] = {
            "count": len(files),
            "size_mb": round(total / (1024**2), 1),
            "sample": sorted([f.name for f in files])[:5],
            "path": str(d),
        }
    for csv in sorted(cohort.glob("*labels*.csv")):
        if csv.is_file():
            found_any = True
            out["labels"].append({
                "name": csv.name,
                "path": str(csv),
                "size_kb": round(csv.stat().st_size / 1024, 1),
            })
    return out if found_any else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("")
async def list_datasets(session_id: str = ""):
    """List all registered datasets, optionally filtered by session."""
    datasets = []
    features_root = PATHCLAW_DATA_DIR / "features"
    if DATASETS_DIR.exists():
        for d in DATASETS_DIR.iterdir():
            if d.is_dir() and (d / "meta.json").exists():
                meta = json.loads((d / "meta.json").read_text())
                if session_id and meta.get("session_id", "") != session_id:
                    continue
                feat_dir = features_root / meta["id"]
                # Count .pt files across backbone subdirs (new layout) and flat layout (legacy).
                meta["feature_count"] = len(list(feat_dir.rglob("*.pt"))) if feat_dir.exists() else 0
                genomics = _scan_cohort_siblings(meta.get("path", ""))
                if genomics:
                    meta["genomics"] = genomics
                datasets.append(meta)
    return {"datasets": datasets}


@router.post("")
async def register_dataset(req: DatasetCreate):
    """Register a new dataset from a folder path (or a single WSI file path)."""
    data_path = Path(req.path).expanduser()
    if not data_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.path}")

    # If user gives a single file path, use its parent directory but only index that file
    single_file = None
    if data_path.is_file():
        if data_path.suffix.lower() not in SLIDE_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Not a recognised WSI format: {data_path.suffix}")
        single_file = data_path
        data_path = data_path.parent

    # Use slugified name as ID so feature dirs match (e.g. tcga_luad_virchow2).
    # Fall back to UUID suffix only on collision.
    import re
    slug = re.sub(r"[^a-z0-9_-]", "_", req.name.lower().strip())[:64] or str(uuid.uuid4())[:8]
    if (DATASETS_DIR / slug).exists():
        slug = f"{slug}_{str(uuid.uuid4())[:4]}"
    dataset_id = slug

    if single_file:
        slides = [{
            "filename": single_file.name,
            "path": str(single_file),
            "size_mb": round(single_file.stat().st_size / (1024**2), 2),
            "extension": single_file.suffix.lower(),
        }]
    else:
        slides = _scan_slides(data_path)
    label_files = _find_label_files(data_path)
    total_size = sum(s["size_mb"] for s in slides)

    meta = {
        "id": dataset_id,
        "name": req.name,
        "path": str(data_path),
        "description": req.description,
        "slide_count": len(slides),
        "total_size_mb": round(total_size, 2),
        "label_file": label_files[0] if label_files else None,
        "label_files_found": label_files,
        "slides": slides,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_id": req.session_id or "",
    }

    # Save metadata
    meta_dir = DATASETS_DIR / dataset_id
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return meta


def _safe_csv_path(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    data_root = PATHCLAW_DATA_DIR.resolve()
    if data_root not in p.parents and p != data_root:
        raise HTTPException(status_code=403, detail="Path must be within PATHCLAW_DATA_DIR")
    if p.suffix.lower() not in (".csv", ".tsv"):
        raise HTTPException(status_code=400, detail="Only .csv / .tsv files allowed")
    return p


@router.get("/csv")
async def read_csv(path: str):
    """Read a CSV/TSV file under PATHCLAW_DATA_DIR as text for the UI viewer."""
    p = _safe_csv_path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return {
        "path": str(p),
        "content": p.read_text(),
        "size_kb": round(p.stat().st_size / 1024, 1),
    }


class CsvWrite(BaseModel):
    path: str
    content: str


@router.post("/csv")
async def write_csv(req: CsvWrite):
    """Create or overwrite a CSV/TSV file under PATHCLAW_DATA_DIR."""
    p = _safe_csv_path(req.path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(req.content)
    return {"path": str(p), "bytes": len(req.content.encode("utf-8"))}


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset details."""
    meta = _load_dataset_meta(dataset_id)
    genomics = _scan_cohort_siblings(meta.get("path", ""))
    if genomics:
        meta["genomics"] = genomics
    return meta


@router.get("/{dataset_id}/slides")
async def list_slides(dataset_id: str):
    """List slides in a dataset."""
    meta = _load_dataset_meta(dataset_id)
    return {"dataset_id": dataset_id, "slides": meta.get("slides", [])}


@router.get("/{dataset_id}/profile")
async def profile_dataset(dataset_id: str):
    """Profile a dataset — class balance, quality, etc."""
    meta = _load_dataset_meta(dataset_id)

    profile = {
        "dataset_id": dataset_id,
        "name": meta["name"],
        "slide_count": meta["slide_count"],
        "total_size_mb": meta["total_size_mb"],
        "formats": {},
        "size_stats": {},
    }

    slides = meta.get("slides", [])
    if slides:
        # Format distribution
        for s in slides:
            ext = s["extension"]
            profile["formats"][ext] = profile["formats"].get(ext, 0) + 1

        # Size statistics
        sizes = [s["size_mb"] for s in slides]
        profile["size_stats"] = {
            "min_mb": round(min(sizes), 2),
            "max_mb": round(max(sizes), 2),
            "mean_mb": round(sum(sizes) / len(sizes), 2),
            "median_mb": round(sorted(sizes)[len(sizes) // 2], 2),
        }

    # Label analysis (if label file exists)
    if meta.get("label_file"):
        try:
            import pandas as pd
            df = pd.read_csv(meta["label_file"])
            profile["label_analysis"] = {
                "columns": list(df.columns),
                "rows": len(df),
                "missing_per_column": {c: int(df[c].isna().sum()) for c in df.columns},
            }
            # Try to detect label columns (low cardinality)
            label_candidates = []
            for col in df.columns:
                if df[col].dtype == "object" and 2 <= df[col].nunique() <= 50:
                    label_candidates.append({
                        "column": col,
                        "unique_values": df[col].nunique(),
                        "distribution": df[col].value_counts().to_dict(),
                    })
            profile["label_candidates"] = label_candidates
        except Exception as e:
            profile["label_analysis_error"] = str(e)

    return profile
