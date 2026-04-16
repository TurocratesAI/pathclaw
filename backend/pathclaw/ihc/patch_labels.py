"""Rule-based patch labels → training set for the learned IHC path.

Each tissue patch gets a continuous per-patch score (e.g. Ki-67 proliferation
fraction, membrane DAB intensity) computed by the rule. The output is a CSV
compatible with PathClaw's existing dataset registry — so the learned path is
just another dataset you can feed into feature extraction + an MIL regressor.

This lets the same rule teach a lightweight per-patch head without labeling
cells by hand.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

from .core import sample_tissue_patches, rgb2hed_channels, segment_nuclei, membrane_band_mask
from .rules import Rule, get_rule

logger = logging.getLogger(__name__)

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()


def _patch_score(rgb, rule: Rule) -> float:
    """One scalar per patch, appropriate for training a regressor."""
    import numpy as np
    _, _, dab = rgb2hed_channels(rgb)
    if rule.compartment == "cytoplasm":
        return float(dab.mean())
    nuclei = segment_nuclei(rgb, use_cellpose=False)  # speed over precision for bulk labeling
    if rule.compartment == "nuclear":
        if nuclei.sum() == 0:
            return 0.0
        dab_in_nuclei = dab[nuclei]
        thr = rule.dab_threshold if isinstance(rule.dab_threshold, (int, float)) else min(rule.dab_threshold)
        return float((dab_in_nuclei > thr).mean())   # fraction of nuclear pixels above threshold
    # membrane
    ring = membrane_band_mask(nuclei, dab)
    if ring.sum() == 0:
        return 0.0
    thr = rule.dab_threshold if isinstance(rule.dab_threshold, (int, float)) else min(rule.dab_threshold)
    return float((dab[ring] > thr).mean())


def build_ihc_patch_labels(
    dataset_id: str,
    rule: str,
    rule_override: dict | None = None,
    patches_per_slide: int | None = None,
    out_name: str | None = None,
) -> dict:
    """Write a per-patch label CSV: slide_id, patch_idx, label (continuous score).

    Consumers (feature extractor, MIL trainer) can read this CSV to attach
    per-patch supervision. Slide-level labels — mean over patches — are also
    included as a separate CSV `<out_name>_slide.csv`.
    """
    import numpy as np
    rule_obj = get_rule(rule, override=rule_override)
    if patches_per_slide is not None:
        rule_obj.patches_per_slide = patches_per_slide

    datasets_dir = PATHCLAW_DATA_DIR / "datasets" / dataset_id
    meta_path = datasets_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_id} not found at {datasets_dir}")
    meta = json.loads(meta_path.read_text())
    slides_root = Path(meta.get("path") or meta.get("slides_path", ""))
    exts = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".vsi", ".scn", ".bif", ".qptiff")
    wsis = sorted([p for p in slides_root.rglob("*") if p.suffix.lower() in exts])

    out_name = out_name or f"patchlabels_{rule_obj.name}"
    patch_csv = datasets_dir / f"{out_name}.csv"
    slide_csv = datasets_dir / f"{out_name}_slide.csv"

    with patch_csv.open("w", newline="") as pf, slide_csv.open("w", newline="") as sf:
        pw = csv.writer(pf); pw.writerow(["slide_id", "patch_idx", "label"])
        sw = csv.writer(sf); sw.writerow(["slide_id", "label", "n_patches"])
        n_patches_total = 0
        for wsi in wsis:
            slide_id = wsi.stem
            patch_scores: list[float] = []
            for i, rgb in enumerate(sample_tissue_patches(
                str(wsi),
                n_patches=rule_obj.patches_per_slide,
                patch_size=rule_obj.patch_size,
                target_mpp=rule_obj.target_mpp,
            )):
                try:
                    s = _patch_score(rgb, rule_obj)
                except Exception as e:
                    logger.warning("patch %d of %s failed: %s", i, slide_id, e)
                    continue
                pw.writerow([slide_id, i, f"{s:.6f}"])
                patch_scores.append(s)
                n_patches_total += 1
            if patch_scores:
                sw.writerow([slide_id, f"{float(np.mean(patch_scores)):.6f}", len(patch_scores)])
            else:
                sw.writerow([slide_id, "", 0])

    return {
        "dataset_id": dataset_id,
        "rule": rule_obj.name,
        "n_slides": len(wsis),
        "n_patch_labels": n_patches_total,
        "patch_csv": str(patch_csv),
        "slide_csv": str(slide_csv),
        "note": "Pass slide_csv as label_file to start_training for slide-level regression; "
                "use patch_csv with a patch-level head.",
    }
