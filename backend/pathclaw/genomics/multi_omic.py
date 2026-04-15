"""Multi-omic label builder — combine genomic, clinical, and model prediction data.

Merges data from multiple sources (MAF mutations, gene expression, clinical
attributes, MIL model predictions) into a unified matrix for multi-label
training or downstream analysis.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pathclaw.genomics")

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()


def _extract_patient_barcode(identifier: str) -> str:
    """Extract TCGA patient barcode (TCGA-XX-XXXX) from various formats."""
    parts = identifier.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return identifier


def _load_maf_labels(source: dict) -> dict[str, dict[str, str]]:
    """Load mutation-based features from MAF files.

    source: {"type": "maf", "path": "dir_or_file", "features": ["EGFR", "TP53", ...]}
    Returns: {patient: {EGFR_mut: "0"/"1", TP53_mut: "0"/"1", ...}}
    """
    from pathclaw.genomics.parsers import _load_all_mafs, NONSYNONYMOUS_CLASSES

    path = Path(source["path"]).expanduser()
    genes = [g.upper() for g in source.get("features", [])]

    if path.is_file():
        rows = _load_all_mafs(str(path.parent))
    else:
        rows = _load_all_mafs(str(path))

    # Track which patients have mutations in which genes
    patient_genes: dict[str, set[str]] = defaultdict(set)
    all_patients: set[str] = set()

    for r in rows:
        sample = r.get("Tumor_Sample_Barcode", "")
        patient = _extract_patient_barcode(sample)
        all_patients.add(patient)
        hugo = r.get("Hugo_Symbol", "").upper()
        if genes and hugo in genes:
            patient_genes[patient].add(hugo)
        elif not genes and r.get("Variant_Classification", "") in NONSYNONYMOUS_CLASSES:
            patient_genes[patient].add(hugo)

    # If no specific genes requested, use top-N most mutated
    if not genes:
        from collections import Counter
        gene_freq = Counter()
        for pgs in patient_genes.values():
            gene_freq.update(pgs)
        genes = [g for g, _ in gene_freq.most_common(20)]

    result: dict[str, dict[str, str]] = {}
    for patient in all_patients:
        pmuts = patient_genes.get(patient, set())
        result[patient] = {f"{g}_mut": "1" if g in pmuts else "0" for g in genes}

    return result


def _load_clinical_labels(source: dict) -> dict[str, dict[str, str]]:
    """Load clinical features from TSV/CSV files.

    source: {"type": "clinical", "path": "file.tsv", "features": ["age", "stage", ...]}
    """
    path = Path(source["path"]).expanduser()
    features = [f.lower() for f in source.get("features", [])]

    result: dict[str, dict[str, str]] = {}

    try:
        delimiter = "\t" if path.suffix in (".tsv", ".txt") else ","
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                # Find patient
                patient = None
                for col in ["bcr_patient_barcode", "case_id", "patient_id", "submitter_id"]:
                    if col in row and row[col]:
                        patient = _extract_patient_barcode(row[col])
                        break
                if not patient:
                    continue

                entry: dict[str, str] = {}
                for key, val in row.items():
                    kl = key.lower()
                    if features and kl not in features:
                        continue
                    if val and val not in ("[Not Available]", "[Not Applicable]", "NA", ""):
                        entry[kl] = val

                if entry:
                    result[patient] = entry
    except Exception as e:
        logger.warning(f"Error loading clinical data from {path}: {e}")

    return result


def _load_predictions(source: dict) -> dict[str, dict[str, str]]:
    """Load MIL model predictions as features.

    source: {"type": "model_predictions", "path": "predictions.csv", "features": ["pred_class", "confidence"]}
    """
    path = Path(source["path"]).expanduser()
    features = source.get("features", [])

    result: dict[str, dict[str, str]] = {}
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try slide_stem or patient_barcode
                patient = None
                for col in ["patient_barcode", "slide_stem", "slide_filename"]:
                    if col in row and row[col]:
                        patient = _extract_patient_barcode(row[col])
                        break
                if not patient:
                    continue

                entry: dict[str, str] = {}
                for key, val in row.items():
                    if features and key not in features:
                        continue
                    if val:
                        entry[f"pred_{key}"] = val
                if entry:
                    result[patient] = entry
    except Exception as e:
        logger.warning(f"Error loading predictions from {path}: {e}")

    return result


def _load_labels_csv(source: dict) -> dict[str, dict[str, str]]:
    """Load an existing labels.csv as a data source.

    source: {"type": "labels", "path": "labels.csv"}
    """
    path = Path(source["path"]).expanduser()
    result: dict[str, dict[str, str]] = {}
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient = row.get("patient_barcode", "")
                if not patient:
                    continue
                result[patient] = {
                    "label": row.get("label", ""),
                    "label_name": row.get("label_name", ""),
                }
    except Exception as e:
        logger.warning(f"Error loading labels from {path}: {e}")
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


_LOADERS = {
    "maf": _load_maf_labels,
    "clinical": _load_clinical_labels,
    "model_predictions": _load_predictions,
    "labels": _load_labels_csv,
}


def build_multi_omic_labels(
    dataset_id: str,
    sources: list[dict],
    output_path: Optional[str] = None,
) -> str:
    """Build a combined multi-omic label/feature matrix from multiple sources.

    Args:
        dataset_id: Dataset ID for output path default
        sources: List of source specs, each with:
            - type: "maf" | "clinical" | "model_predictions" | "labels"
            - path: file or directory path
            - features: list of feature names to extract (optional)
        output_path: Where to save the combined CSV

    Returns:
        Summary text with matrix dimensions and missing data stats.
    """
    if not sources:
        return "Error: no sources provided"

    # Load data from each source
    all_data: dict[str, dict[str, str]] = defaultdict(dict)
    source_stats: list[dict] = []
    all_columns: list[str] = []

    for i, source in enumerate(sources):
        stype = source.get("type", "")
        loader = _LOADERS.get(stype)
        if not loader:
            source_stats.append({"type": stype, "error": f"Unknown source type: {stype}"})
            continue

        try:
            data = loader(source)
            cols = set()
            for patient_data in data.values():
                cols.update(patient_data.keys())
            cols_sorted = sorted(cols)
            all_columns.extend(cols_sorted)

            for patient, features in data.items():
                all_data[patient].update(features)

            source_stats.append({
                "type": stype,
                "path": source.get("path", ""),
                "patients": len(data),
                "features": len(cols_sorted),
                "feature_names": cols_sorted,
            })
        except Exception as e:
            source_stats.append({"type": stype, "error": str(e)})

    if not all_data:
        return "Error: no data loaded from any source"

    # Deduplicate columns while preserving order
    seen = set()
    unique_columns = []
    for c in all_columns:
        if c not in seen:
            seen.add(c)
            unique_columns.append(c)

    # Write combined CSV
    if output_path:
        out_path = Path(output_path).expanduser()
    else:
        out_path = PATHCLAW_DATA_DIR / "datasets" / dataset_id / "multi_omic_labels.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["patient_barcode"] + unique_columns
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for patient in sorted(all_data.keys()):
            row = {"patient_barcode": patient, **all_data[patient]}
            writer.writerow(row)

    # Compute missing data stats
    total_cells = len(all_data) * len(unique_columns)
    filled = sum(
        1 for patient_data in all_data.values()
        for col in unique_columns
        if col in patient_data and patient_data[col]
    )
    missing_pct = (1 - filled / max(total_cells, 1)) * 100

    # Summary
    lines = [
        "## Multi-Omic Matrix",
        f"- **Patients**: {len(all_data)}",
        f"- **Features**: {len(unique_columns)}",
        f"- **Matrix size**: {len(all_data)} × {len(unique_columns)}",
        f"- **Missing data**: {missing_pct:.1f}%",
        f"- **Output**: {out_path}",
        "",
        "### Sources",
    ]
    for s in source_stats:
        if "error" in s:
            lines.append(f"  - {s['type']}: ❌ {s['error']}")
        else:
            lines.append(f"  - {s['type']} ({s['path']}): {s['patients']} patients, {s['features']} features")

    lines.append("\n### Features")
    for col in unique_columns:
        # Count non-missing
        filled_col = sum(1 for pd in all_data.values() if col in pd and pd[col])
        lines.append(f"  - {col}: {filled_col}/{len(all_data)} ({filled_col/len(all_data)*100:.0f}%)")

    return "\n".join(lines)
