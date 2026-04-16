"""Slide-level IHC scoring dispatcher.

`score_slide(wsi_path, rule)` samples tissue patches, runs compartment
segmentation, and aggregates per-patch measurements according to
`rule.aggregation`. Returns a dict with the slide score, interpretation
label, per-patch raw values, and tissue QC.

`score_dataset(dataset_id, marker, rule_override)` batches the above over
every slide in a registered dataset — writes a CSV under the dataset dir
and returns a summary.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    membrane_band_mask,
    rgb2hed_channels,
    sample_tissue_patches,
    segment_nuclei,
)
from .rules import Rule, get_rule

logger = logging.getLogger(__name__)

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()


# ---------------------------------------------------------------------------
# Per-patch measurements
# ---------------------------------------------------------------------------

def _patch_measure(rgb: np.ndarray, rule: Rule, use_cellpose: bool) -> dict:
    """One patch → {positive_cells, total_cells, mean_dab, membrane_fraction}."""
    _, _, dab = rgb2hed_channels(rgb)
    nuclei = segment_nuclei(rgb, use_cellpose=use_cellpose)
    n_total = int(nuclei.sum() > 0)  # guard for empty masks
    out = {
        "total_cells": 0,
        "positive_cells": 0,
        "mean_dab": float(dab.mean()),
        "membrane_fraction": 0.0,
        "per_cell_intensities": [],
    }
    # Label connected components so we can score each cell individually.
    from skimage.measure import label, regionprops
    lbl = label(nuclei)
    regions = regionprops(lbl)
    out["total_cells"] = len(regions)

    # Determine threshold
    thresh = rule.dab_threshold
    thr_single = thresh if isinstance(thresh, (int, float)) else min(thresh)

    if rule.compartment == "nuclear":
        pos = 0
        intensities = []
        for r in regions:
            ys, xs = r.coords[:, 0], r.coords[:, 1]
            cell_dab = dab[ys, xs].mean()
            intensities.append(float(cell_dab))
            if cell_dab >= thr_single:
                pos += 1
        out["positive_cells"] = pos
        out["per_cell_intensities"] = intensities
    elif rule.compartment == "membrane":
        ring = membrane_band_mask(nuclei, dab)
        pos = 0
        ring_dab_per_cell = []
        from skimage.morphology import dilation, disk
        for r in regions:
            # Per-cell ring by dilating that cell's label mask
            cell_mask = (lbl == r.label)
            cell_ring = dilation(cell_mask, disk(6)) & ~cell_mask & ring
            if cell_ring.sum() == 0:
                ring_dab_per_cell.append(0.0)
                continue
            ring_mean = float(dab[cell_ring].mean())
            ring_dab_per_cell.append(ring_mean)
            if ring_mean >= thr_single:
                pos += 1
        out["positive_cells"] = pos
        out["per_cell_intensities"] = ring_dab_per_cell
        out["membrane_fraction"] = float((ring & (dab >= thr_single)).sum() / max(ring.sum(), 1))
    else:  # cytoplasm
        mean_dab = float(dab.mean())
        out["positive_cells"] = int(mean_dab >= thr_single)
        out["total_cells"] = 1
        out["per_cell_intensities"] = [mean_dab]
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(patches: list[dict], rule: Rule) -> dict:
    total_cells = sum(p["total_cells"] for p in patches)
    positive_cells = sum(p["positive_cells"] for p in patches)
    all_intensities: list[float] = []
    for p in patches:
        all_intensities.extend(p.get("per_cell_intensities", []))

    agg = rule.aggregation
    score: float = 0.0
    extras: dict[str, Any] = {}

    if agg == "percent_positive" or agg == "tps":
        score = 100.0 * positive_cells / max(total_cells, 1)
    elif agg == "mean_intensity":
        score = float(np.mean(all_intensities)) if all_intensities else 0.0
    elif agg == "allred":
        # Proportion: 0=0%, 1=<1%, 2=1-10%, 3=10-33%, 4=33-66%, 5=>66%.
        # Intensity: 0=none, 1=weak, 2=moderate, 3=strong — via rule.dab_threshold (3-band).
        frac = positive_cells / max(total_cells, 1)
        if frac == 0:       prop = 0
        elif frac < 0.01:   prop = 1
        elif frac < 0.10:   prop = 2
        elif frac < 0.33:   prop = 3
        elif frac < 0.66:   prop = 4
        else:               prop = 5
        # Intensity: mean of per-cell intensities of positive cells, mapped to 0-3 by rule bands.
        pos_intensities = [v for v in all_intensities if v >= (rule.dab_threshold[0] if isinstance(rule.dab_threshold, list) else rule.dab_threshold)]
        mean_pos = float(np.mean(pos_intensities)) if pos_intensities else 0.0
        bands = rule.dab_threshold if isinstance(rule.dab_threshold, list) else [rule.dab_threshold]
        if not pos_intensities:        inten = 0
        elif mean_pos < bands[1] if len(bands) > 1 else 0.2: inten = 1
        elif mean_pos < (bands[2] if len(bands) > 2 else 0.35): inten = 2
        else:                           inten = 3
        score = float(prop + inten)
        extras["proportion_score"] = prop
        extras["intensity_score"] = inten
        extras["positive_fraction"] = float(frac)
    elif agg == "her2_score":
        # 0/1+/2+/3+ from membrane DAB bands + completeness.
        bands = rule.dab_threshold if isinstance(rule.dab_threshold, list) else [0.10, 0.20, 0.35]
        mean_membrane = np.mean([p["membrane_fraction"] for p in patches]) if patches else 0.0
        mean_pos_intensity = float(np.mean(all_intensities)) if all_intensities else 0.0
        # Heuristic: use highest band the average positive cell exceeds;
        # gate 3+ on completeness fraction ≥ 10%.
        if mean_pos_intensity < bands[0]:
            score = 0.0
        elif mean_pos_intensity < bands[1]:
            score = 1.0
        elif mean_pos_intensity < bands[2] or mean_membrane < 0.10:
            score = 2.0
        else:
            score = 3.0
        extras["mean_membrane_fraction"] = float(mean_membrane)
        extras["mean_positive_intensity"] = mean_pos_intensity
    else:
        raise ValueError(f"Unknown aggregation: {agg}")

    return {
        "score": float(score),
        "total_cells_analyzed": int(total_cells),
        "positive_cells": int(positive_cells),
        "n_patches_used": len(patches),
        "aggregation": agg,
        **extras,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_slide(
    wsi_path: str,
    rule: str | Rule,
    rule_override: dict | None = None,
    use_cellpose: bool = True,
) -> dict:
    """Score one WSI. `rule` is a preset name (e.g. 'ki67_pi') or a Rule object."""
    if isinstance(rule, str):
        rule_obj = get_rule(rule, override=rule_override)
    else:
        rule_obj = rule
    t0 = time.time()
    patches: list[dict] = []
    n = 0
    for rgb in sample_tissue_patches(
        wsi_path,
        n_patches=rule_obj.patches_per_slide,
        patch_size=rule_obj.patch_size,
        target_mpp=rule_obj.target_mpp,
    ):
        try:
            m = _patch_measure(rgb, rule_obj, use_cellpose=use_cellpose)
            patches.append(m)
            n += 1
        except Exception as e:
            logger.warning("patch measure failed: %s", e)

    if not patches:
        return {
            "wsi_path": wsi_path,
            "rule": rule_obj.name,
            "score": None,
            "label": "no tissue patches analysable",
            "n_patches_used": 0,
            "elapsed_sec": round(time.time() - t0, 1),
        }

    agg = _aggregate(patches, rule_obj)
    label = rule_obj.interpret(agg["score"]) if rule_obj.interpret else ""
    return {
        "wsi_path": wsi_path,
        "rule": rule_obj.name,
        "marker": rule_obj.marker,
        "compartment": rule_obj.compartment,
        "label": label,
        "elapsed_sec": round(time.time() - t0, 1),
        **agg,
    }


def score_dataset(
    dataset_id: str,
    rule: str,
    rule_override: dict | None = None,
    max_slides: int | None = None,
    use_cellpose: bool = True,
    session_id: str = "",
) -> dict:
    """Score every slide in a registered dataset. Writes CSV next to meta.json."""
    datasets_dir = PATHCLAW_DATA_DIR / "datasets" / dataset_id
    meta_path = datasets_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_id} not found at {datasets_dir}")
    meta = json.loads(meta_path.read_text())
    slides_root = Path(meta.get("path") or meta.get("slides_path", ""))
    exts = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".vsi", ".scn", ".bif", ".qptiff")
    wsis = sorted([p for p in slides_root.rglob("*") if p.suffix.lower() in exts])
    if max_slides:
        wsis = wsis[:max_slides]
    rule_obj = get_rule(rule, override=rule_override)
    results: list[dict] = []
    for i, p in enumerate(wsis, 1):
        logger.info("IHC score %d/%d: %s", i, len(wsis), p.name)
        try:
            r = score_slide(str(p), rule_obj, use_cellpose=use_cellpose)
        except Exception as e:
            r = {"wsi_path": str(p), "rule": rule_obj.name, "error": str(e)}
        r["slide_id"] = p.stem
        results.append(r)

    out_csv = datasets_dir / f"ihc_{rule_obj.name}.csv"
    if results:
        keys = sorted({k for r in results for k in r.keys()})
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                r_clean = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
                w.writerow(r_clean)

    valid = [r for r in results if r.get("score") is not None]
    summary = {
        "dataset_id": dataset_id,
        "rule": rule_obj.name,
        "n_slides": len(wsis),
        "n_scored": len(valid),
        "n_failed": len(results) - len(valid),
        "csv_path": str(out_csv),
    }
    if valid:
        scores = [r["score"] for r in valid]
        summary["mean_score"] = float(np.mean(scores))
        summary["median_score"] = float(np.median(scores))
        summary["min_score"] = float(np.min(scores))
        summary["max_score"] = float(np.max(scores))
    return summary
