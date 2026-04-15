"""Extract slide-level labels from genomic data for MIL training.

Handles TCGA barcode resolution, MSI/mutation/TMB/clinical label
extraction, patient deduplication, and labels.csv generation.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from pathclaw.genomics.parsers import _load_all_mafs, NONSYNONYMOUS_CLASSES

logger = logging.getLogger("pathclaw.genomics")

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()


# ---------------------------------------------------------------------------
# TCGA barcode utilities
# ---------------------------------------------------------------------------


def _extract_patient_barcode(identifier: str) -> str:
    """Extract TCGA patient barcode (TCGA-XX-XXXX) from various formats.

    Handles:
      - Full sample barcode: TCGA-2E-A9G8-01A-11D-A405-09 → TCGA-2E-A9G8
      - Slide filename: TCGA-2E-A9G8-01Z-00-DX1.DCD3E31B-... → TCGA-2E-A9G8
      - Short barcode: TCGA-2E-A9G8 → TCGA-2E-A9G8
    """
    parts = identifier.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return identifier


def _build_slide_map(dataset_id: str) -> dict[str, list[dict]]:
    """Build a map from patient barcode → list of slide info dicts.

    Returns: {patient_barcode: [{stem, path, dx_rank}, ...]}
    """
    meta_path = PATHCLAW_DATA_DIR / "datasets" / dataset_id / "meta.json"
    if not meta_path.exists():
        return {}

    meta = json.loads(meta_path.read_text())
    patient_slides: dict[str, list[dict]] = defaultdict(list)

    for slide in meta.get("slides", []):
        stem = Path(slide["path"]).stem
        patient = _extract_patient_barcode(stem)
        # Rank DX slides: DX1=1, DX2=2, TS1=10, etc.
        dx_rank = 99
        for part in stem.split("-"):
            if part.startswith("DX"):
                try:
                    dx_rank = int(part[2:])
                except ValueError:
                    dx_rank = 1
            elif part.startswith("TS"):
                dx_rank = 10
        patient_slides[patient].append({
            "stem": stem,
            "path": slide["path"],
            "dx_rank": dx_rank,
        })

    return dict(patient_slides)


def _pick_best_slide(slides: list[dict]) -> dict:
    """Pick the best slide for a patient (prefer DX1)."""
    return min(slides, key=lambda s: s["dx_rank"])


# ---------------------------------------------------------------------------
# Label extractors
# ---------------------------------------------------------------------------


def _extract_msi_labels(
    genomic_dir: Path,
    label_spec: dict,
) -> dict[str, tuple[int, str]]:
    """Extract MSI status labels from clinical XML or MAF files.

    Returns: {patient_barcode: (label_int, label_name)}
    """
    labels: dict[str, tuple[int, str]] = {}

    # Strategy 1: Look for clinical XML files with msi_status field
    xml_files = list(genomic_dir.rglob("*.xml"))
    parse_errors = 0
    for xml_path in xml_files:
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            patient_barcode = None
            msi_status = None

            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                text = (elem.text or "").strip()
                if tag == "bcr_patient_barcode" and text:
                    patient_barcode = text
                if "msi" in tag.lower() and text:
                    msi_status = text

            if patient_barcode and msi_status:
                msi_lower = msi_status.lower()
                if "msi-h" in msi_lower or msi_lower == "msi-h":
                    labels[patient_barcode] = (1, "MSI-H")
                elif "mss" in msi_lower or "msi-l" in msi_lower or "stable" in msi_lower:
                    labels[patient_barcode] = (0, "MSS")
        except (ET.ParseError, Exception) as e:
            parse_errors += 1
            if parse_errors <= 5:
                logger.warning(f"Failed to parse XML {xml_path.name}: {e}")
            continue

    if parse_errors > 0:
        logger.warning(f"MSI label extraction: {parse_errors} XML files failed to parse out of total")

    # Strategy 2: TMB-based classification from MAF files (fallback)
    if not labels:
        tmb_threshold = label_spec.get("tmb_threshold", 10.0)
        exome_size = label_spec.get("exome_size_mb", 30.0)
        rows = _load_all_mafs(str(genomic_dir))
        sample_counts: dict[str, int] = defaultdict(int)
        for r in rows:
            if r.get("Variant_Classification", "") in NONSYNONYMOUS_CLASSES:
                sample = r.get("Tumor_Sample_Barcode", "")
                patient = _extract_patient_barcode(sample)
                sample_counts[patient] += 1

        for patient, count in sample_counts.items():
            tmb = count / exome_size
            if tmb >= tmb_threshold:
                labels[patient] = (1, "MSI-H")
            else:
                labels[patient] = (0, "MSS")

    return labels


def _extract_mutation_labels(
    genomic_dir: Path,
    label_spec: dict,
) -> dict[str, tuple[int, str]]:
    """Extract mutation status labels (mutant vs wild-type for a gene).

    label_spec: {"gene": "EGFR"}
    Returns: {patient_barcode: (1, "EGFR-mutant") or (0, "EGFR-wildtype")}
    """
    gene = label_spec.get("gene", "").upper()
    if not gene:
        return {}

    rows = _load_all_mafs(str(genomic_dir))
    all_patients: set[str] = set()
    mutated_patients: set[str] = set()

    for r in rows:
        sample = r.get("Tumor_Sample_Barcode", "")
        patient = _extract_patient_barcode(sample)
        all_patients.add(patient)
        if r.get("Hugo_Symbol", "").upper() == gene:
            mutated_patients.add(patient)

    labels: dict[str, tuple[int, str]] = {}
    for patient in all_patients:
        if patient in mutated_patients:
            labels[patient] = (1, f"{gene}-mutant")
        else:
            labels[patient] = (0, f"{gene}-wildtype")

    return labels


def _extract_tmb_labels(
    genomic_dir: Path,
    label_spec: dict,
) -> dict[str, tuple[int, str]]:
    """Classify patients by TMB (high vs low).

    label_spec: {"threshold": 10}
    Returns: {patient_barcode: (1, "TMB-High") or (0, "TMB-Low")}
    """
    threshold = label_spec.get("threshold", 10.0)
    exome_size = label_spec.get("exome_size_mb", 30.0)

    rows = _load_all_mafs(str(genomic_dir))
    sample_counts: dict[str, int] = defaultdict(int)
    all_patients: set[str] = set()

    for r in rows:
        sample = r.get("Tumor_Sample_Barcode", "")
        patient = _extract_patient_barcode(sample)
        all_patients.add(patient)
        if r.get("Variant_Classification", "") in NONSYNONYMOUS_CLASSES:
            sample_counts[patient] += 1

    labels: dict[str, tuple[int, str]] = {}
    for patient in all_patients:
        tmb = sample_counts.get(patient, 0) / exome_size
        if tmb >= threshold:
            labels[patient] = (1, "TMB-High")
        else:
            labels[patient] = (0, "TMB-Low")

    return labels


def _extract_clinical_field_labels(
    genomic_dir: Path,
    label_spec: dict,
) -> dict[str, tuple[int, str]]:
    """Extract labels from a clinical data field.

    label_spec: {"field": "histological_type", "mapping": {"serous": 0, "endometrioid": 1}}
    If no mapping provided, auto-assigns integers to unique values.
    """
    field_name = label_spec.get("field", "").lower()
    mapping = label_spec.get("mapping", {})
    if not field_name:
        return {}

    labels: dict[str, tuple[int, str]] = {}

    # Search clinical XMLs
    xml_files = list(genomic_dir.rglob("*.xml"))
    parse_errors = 0
    for xml_path in xml_files:
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            patient_barcode = None
            field_value = None

            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                text = (elem.text or "").strip()
                if tag == "bcr_patient_barcode" and text:
                    patient_barcode = text
                if tag.lower() == field_name and text:
                    field_value = text

            if patient_barcode and field_value:
                if mapping:
                    # Try exact match, then lowercase match
                    label_int = mapping.get(field_value, mapping.get(field_value.lower()))
                    if label_int is not None:
                        labels[patient_barcode] = (label_int, field_value)
                else:
                    labels[patient_barcode] = (-1, field_value)  # placeholder, auto-map later
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 5:
                logger.warning(f"Failed to parse XML {xml_path.name}: {e}")
            continue

    if parse_errors > 0:
        logger.warning(f"Clinical field extraction: {parse_errors} XML files failed to parse")

    # Search TSV/CSV files
    for tsv_path in genomic_dir.rglob("*.tsv"):
        try:
            with open(tsv_path, "r") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    # Find patient barcode column
                    patient = None
                    for col in ["bcr_patient_barcode", "case_id", "patient_id", "submitter_id"]:
                        if col in row and row[col]:
                            patient = _extract_patient_barcode(row[col])
                            break
                    if not patient:
                        continue
                    # Find target field
                    val = row.get(field_name, "")
                    if not val:
                        # Case-insensitive search
                        for k, v in row.items():
                            if k.lower() == field_name and v:
                                val = v
                                break
                    if patient and val:
                        if mapping:
                            label_int = mapping.get(val, mapping.get(val.lower()))
                            if label_int is not None:
                                labels[patient] = (label_int, val)
                        else:
                            labels[patient] = (-1, val)
        except Exception:
            continue

    # Auto-map if no mapping was provided
    if not mapping and labels:
        unique_vals = sorted(set(v[1] for v in labels.values()))
        auto_map = {v: i for i, v in enumerate(unique_vals)}
        labels = {k: (auto_map[v[1]], v[1]) for k, v in labels.items()}

    return labels


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_labels(
    genomic_dir: str,
    dataset_id: str,
    label_type: str,
    label_spec: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> str:
    """Extract slide-level labels from genomic files.

    Args:
        genomic_dir: Directory containing MAF/VCF/clinical files
        dataset_id: Dataset ID to match slides against
        label_type: msi_status | mutation_status | tmb_class | clinical_field
        label_spec: Type-specific parameters (e.g. {"gene": "EGFR"})
        output_path: Where to save labels.csv

    Returns:
        Summary text with label statistics.
    """
    genomic_dir = Path(genomic_dir).expanduser()
    if label_spec is None:
        label_spec = {}

    if not genomic_dir.exists():
        return f"Error: genomic directory not found: {genomic_dir}"

    # Step 1: Extract patient-level labels
    extractors = {
        "msi_status": _extract_msi_labels,
        "mutation_status": _extract_mutation_labels,
        "tmb_class": _extract_tmb_labels,
        "clinical_field": _extract_clinical_field_labels,
    }

    extractor = extractors.get(label_type)
    if not extractor:
        return f"Unknown label_type: {label_type}. Available: {', '.join(extractors.keys())}"

    patient_labels = extractor(genomic_dir, label_spec)
    if not patient_labels:
        return f"No labels extracted for label_type={label_type}. Check if genomic files exist in {genomic_dir}"

    # Step 2: Build slide map from dataset
    slide_map = _build_slide_map(dataset_id)
    if not slide_map:
        return f"Error: could not load slide map for dataset {dataset_id}. Is it registered?"

    # Step 3: Match patients to slides (with deduplication)
    matched: list[dict] = []
    unmatched_patients = 0
    for patient, (label_int, label_name) in patient_labels.items():
        slides = slide_map.get(patient)
        if not slides:
            unmatched_patients += 1
            continue
        best = _pick_best_slide(slides)
        matched.append({
            "slide_filename": Path(best["path"]).name,
            "slide_stem": best["stem"],
            "patient_barcode": patient,
            "label": label_int,
            "label_name": label_name,
        })

    if not matched:
        return (
            f"No slides matched to labels. "
            f"Patients with labels: {len(patient_labels)}, "
            f"Patients with slides: {len(slide_map)}"
        )

    # Step 4: Write labels.csv
    if output_path is None:
        output_path = str(PATHCLAW_DATA_DIR / "datasets" / dataset_id / "labels.csv")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_filename", "slide_stem", "patient_barcode", "label", "label_name"])
        writer.writeheader()
        for row in sorted(matched, key=lambda r: r["slide_stem"]):
            writer.writerow(row)

    # Step 5: Generate summary
    class_dist = Counter(r["label_name"] for r in matched)
    lines = [
        f"## Label Extraction: {label_type}",
        f"- **Patients with labels**: {len(patient_labels)}",
        f"- **Patients with slides**: {len(slide_map)}",
        f"- **Matched slides**: {len(matched)}",
        f"- **Unmatched patients** (no slides): {unmatched_patients}",
        f"- **Output**: {out_path}",
        "",
        "### Class Distribution",
    ]
    for name, count in class_dist.most_common():
        pct = count / len(matched) * 100
        lines.append(f"  {name}: {count} ({pct:.1f}%)")

    return "\n".join(lines)
