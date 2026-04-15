"""Survival analysis — Kaplan-Meier, log-rank, Cox regression.

Extracts survival data from TCGA clinical files, merges with slide labels,
and runs standard survival analyses with optional stratification.
"""

from __future__ import annotations

import csv
import json
import logging
import os

import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger("pathclaw.genomics")

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()

# TCGA clinical XML survival fields
SURVIVAL_FIELDS = {
    "days_to_death", "days_to_last_followup", "days_to_last_known_alive",
    "vital_status", "days_to_birth",
}


# ---------------------------------------------------------------------------
# Survival data extraction
# ---------------------------------------------------------------------------


def _extract_survival_from_xml(xml_path: Path) -> Optional[dict]:
    """Extract survival fields from a single TCGA clinical XML file."""
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception:
        return None

    patient_barcode = None
    fields: dict[str, str] = {}

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        text = (elem.text or "").strip()
        if not text:
            continue

        if tag == "bcr_patient_barcode":
            patient_barcode = text
        tag_lower = tag.lower()
        if tag_lower in SURVIVAL_FIELDS:
            fields[tag_lower] = text
        # Also capture age, stage, grade
        if tag_lower in ("age_at_diagnosis", "age_at_initial_pathologic_diagnosis"):
            fields["age"] = text
        if "stage" in tag_lower and "pathologic" in tag_lower:
            fields["stage"] = text
        if "grade" in tag_lower and "histologic" in tag_lower:
            fields["grade"] = text

    if not patient_barcode:
        return None

    return {"patient_barcode": patient_barcode, **fields}


def _extract_survival_from_tsv(tsv_path: Path) -> list[dict]:
    """Extract survival data from a TSV clinical file."""
    results = []
    try:
        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                # Find patient barcode
                patient = None
                for col in ["bcr_patient_barcode", "case_id", "patient_id", "submitter_id"]:
                    if col in row and row[col]:
                        patient = row[col]
                        break
                if not patient:
                    continue

                entry = {"patient_barcode": patient}
                for field in SURVIVAL_FIELDS:
                    for key in row:
                        if key.lower() == field:
                            val = row[key].strip()
                            if val and val not in ("[Not Available]", "[Not Applicable]", "NA"):
                                entry[field] = val
                            break
                # Age, stage
                for key in row:
                    kl = key.lower()
                    if "age" in kl and "diagnosis" in kl:
                        val = row[key].strip()
                        if val and val not in ("[Not Available]", "NA"):
                            entry["age"] = val
                    if "stage" in kl and "pathologic" in kl:
                        val = row[key].strip()
                        if val and val not in ("[Not Available]", "NA"):
                            entry["stage"] = val

                results.append(entry)
    except Exception:
        pass
    return results


def extract_survival_data(
    clinical_dir: str,
) -> dict[str, dict]:
    """Extract survival data from all clinical files in a directory.

    Returns: {patient_barcode: {vital_status, days_to_death, days_to_last_followup, ...}}
    """
    clinical_dir = Path(clinical_dir).expanduser()
    patients: dict[str, dict] = {}

    # Parse XML files
    for xml_path in clinical_dir.rglob("*.xml"):
        entry = _extract_survival_from_xml(xml_path)
        if entry:
            barcode = entry.pop("patient_barcode")
            patients[barcode] = entry

    # Parse TSV files (may supplement or override XML)
    for tsv_path in clinical_dir.rglob("*.tsv"):
        for entry in _extract_survival_from_tsv(tsv_path):
            barcode = entry.pop("patient_barcode")
            if barcode not in patients:
                patients[barcode] = entry
            else:
                # Merge — don't overwrite existing values
                for k, v in entry.items():
                    if k not in patients[barcode]:
                        patients[barcode][k] = v

    return patients


def _compute_os(entry: dict) -> Optional[tuple[float, int]]:
    """Compute overall survival time and event from clinical fields.

    Returns: (os_time_days, os_event) where event=1 for death, 0 for censored.
    """
    vital = entry.get("vital_status", "").lower()

    if vital == "dead":
        dtd = entry.get("days_to_death", "")
        if dtd:
            try:
                return (float(dtd), 1)
            except ValueError:
                pass

    # Alive or unknown vital status — censored
    for field in ["days_to_last_followup", "days_to_last_known_alive"]:
        val = entry.get(field, "")
        if val:
            try:
                return (float(val), 0)
            except ValueError:
                continue

    return None


# ---------------------------------------------------------------------------
# Survival analysis
# ---------------------------------------------------------------------------


def run_survival_analysis(
    clinical_dir: str,
    dataset_id: Optional[str] = None,
    labels_path: Optional[str] = None,
    group_column: str = "label_name",
    output_dir: Optional[str] = None,
) -> str:
    """Run Kaplan-Meier survival analysis with optional stratification.

    Args:
        clinical_dir: Directory with clinical XML/TSV files
        dataset_id: Dataset ID (to auto-find labels.csv)
        labels_path: Path to labels.csv for stratification
        group_column: Column in labels.csv to stratify by
        output_dir: Where to save KM plot and results

    Returns:
        Summary text with survival statistics.
    """
    # Extract survival data
    survival_data = extract_survival_data(clinical_dir)
    if not survival_data:
        return f"No survival data found in {clinical_dir}"

    # Compute OS for each patient
    os_data: dict[str, tuple[float, int]] = {}
    for patient, entry in survival_data.items():
        os_result = _compute_os(entry)
        if os_result:
            os_data[patient] = os_result

    if not os_data:
        return "Could not compute overall survival for any patient. Check clinical fields."

    # Load labels for stratification if available
    labels: dict[str, str] = {}
    if labels_path:
        label_file = Path(labels_path).expanduser()
    elif dataset_id:
        label_file = PATHCLAW_DATA_DIR / "datasets" / dataset_id / "labels.csv"
    else:
        label_file = None

    if label_file and label_file.exists():
        with open(label_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient = row.get("patient_barcode", "")
                group = row.get(group_column, "")
                if patient and group:
                    labels[patient] = group

    # Output directory
    if output_dir:
        out_dir = Path(output_dir).expanduser()
    elif dataset_id:
        out_dir = PATHCLAW_DATA_DIR / "analysis" / dataset_id
    else:
        out_dir = PATHCLAW_DATA_DIR / "analysis" / "survival"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to use lifelines for proper KM analysis
    try:
        return _run_km_lifelines(os_data, labels, group_column, out_dir)
    except ImportError:
        return _run_km_basic(os_data, labels, group_column, out_dir)


def _run_km_lifelines(
    os_data: dict[str, tuple[float, int]],
    labels: dict[str, str],
    group_column: str,
    out_dir: Path,
) -> str:
    """Run KM analysis using lifelines library."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    lines = ["## Survival Analysis (Kaplan-Meier)"]
    lines.append(f"- **Patients with survival data**: {len(os_data)}")

    if labels:
        # Stratified analysis
        groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for patient, (time, event) in os_data.items():
            group = labels.get(patient, "Unknown")
            if group != "Unknown":
                groups[group].append((time, event))

        lines.append(f"- **Patients with labels**: {sum(len(v) for v in groups.values())}")
        lines.append(f"- **Groups**: {', '.join(f'{g} (n={len(v)})' for g, v in sorted(groups.items()))}")
        lines.append("")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
        except ImportError:
            ax = None

        group_stats: dict[str, dict] = {}
        for group_name, data in sorted(groups.items()):
            times = np.array([d[0] for d in data])
            events = np.array([d[1] for d in data])

            kmf = KaplanMeierFitter()
            kmf.fit(times, events, label=group_name)

            median = kmf.median_survival_time_
            n_events = int(events.sum())
            group_stats[group_name] = {
                "n": len(data),
                "events": n_events,
                "censored": len(data) - n_events,
                "median_survival_days": float(median) if not np.isinf(median) else None,
            }

            if ax is not None:
                kmf.plot_survival_function(ax=ax)

        # Log-rank test between groups (pairwise if >2)
        group_names = sorted(groups.keys())
        if len(group_names) == 2:
            g1_data = groups[group_names[0]]
            g2_data = groups[group_names[1]]
            result = logrank_test(
                durations_A=np.array([d[0] for d in g1_data]),
                durations_B=np.array([d[0] for d in g2_data]),
                event_observed_A=np.array([d[1] for d in g1_data]),
                event_observed_B=np.array([d[1] for d in g2_data]),
            )
            lines.append("### Log-rank Test")
            lines.append(f"  p-value: **{result.p_value:.2e}**")
            lines.append(f"  test statistic: {result.test_statistic:.2f}")
            significance = "significant" if result.p_value < 0.05 else "not significant"
            lines.append(f"  → {'Statistically ' + significance} (α=0.05)")
        elif len(group_names) > 2:
            from lifelines.statistics import multivariate_logrank_test
            all_times = []
            all_events = []
            all_groups = []
            for gname, data in groups.items():
                for time, event in data:
                    all_times.append(time)
                    all_events.append(event)
                    all_groups.append(gname)
            result = multivariate_logrank_test(
                np.array(all_times), np.array(all_groups), np.array(all_events),
            )
            lines.append("### Multivariate Log-rank Test")
            lines.append(f"  p-value: **{result.p_value:.2e}**")
            lines.append(f"  test statistic: {result.test_statistic:.2f}")

        lines.append("\n### Group Statistics")
        for gname, stats in sorted(group_stats.items()):
            median_str = f"{stats['median_survival_days']:.0f} days" if stats['median_survival_days'] else "not reached"
            lines.append(
                f"  **{gname}**: n={stats['n']}, events={stats['events']}, "
                f"censored={stats['censored']}, median OS={median_str}"
            )

        if ax is not None:
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Survival Probability")
            ax.set_title(f"Kaplan-Meier Survival Curves by {group_column}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_path = out_dir / "km_plot.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            lines.append(f"\n### Plot saved to: {plot_path}")

        # Save results JSON
        results = {"groups": group_stats, "total_patients": len(os_data)}
        results_path = out_dir / "survival_results.json"
        results_path.write_text(json.dumps(results, indent=2))
        lines.append(f"### Results saved to: {results_path}")

    else:
        # Unstratified — overall survival
        times = [d[0] for d in os_data.values()]
        events = [d[1] for d in os_data.values()]

        kmf = KaplanMeierFitter()
        kmf.fit(np.array(times), np.array(events), label="Overall")

        median = kmf.median_survival_time_
        lines.append(f"- **Events (deaths)**: {sum(events)}")
        lines.append(f"- **Censored**: {len(events) - sum(events)}")
        median_str = f"{median:.0f} days" if not np.isinf(median) else "not reached"
        lines.append(f"- **Median OS**: {median_str}")
        lines.append("\n*Provide labels.csv or dataset_id for stratified analysis*")

    return "\n".join(lines)


def _run_km_basic(
    os_data: dict[str, tuple[float, int]],
    labels: dict[str, str],
    group_column: str,
    out_dir: Path,
) -> str:
    """Basic survival summary without lifelines (fallback)."""
    lines = [
        "## Survival Data Summary",
        f"- **Patients with survival data**: {len(os_data)}",
    ]

    events = sum(1 for _, e in os_data.values() if e == 1)
    censored = len(os_data) - events
    times = [t for t, _ in os_data.values()]

    lines.append(f"- **Events (deaths)**: {events}")
    lines.append(f"- **Censored**: {censored}")
    lines.append(f"- **Follow-up range**: {min(times):.0f} – {max(times):.0f} days")
    lines.append(f"- **Median follow-up**: {sorted(times)[len(times)//2]:.0f} days")

    if labels:
        groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for patient, (time, event) in os_data.items():
            group = labels.get(patient, "Unknown")
            if group != "Unknown":
                groups[group].append((time, event))

        lines.append(f"\n### Stratification by {group_column}")
        for gname, data in sorted(groups.items()):
            g_events = sum(1 for _, e in data if e == 1)
            g_times = [t for t, _ in data]
            lines.append(
                f"  **{gname}**: n={len(data)}, events={g_events}, "
                f"median follow-up={sorted(g_times)[len(g_times)//2]:.0f} days"
            )

    lines.append("\n⚠ Install `lifelines` for full Kaplan-Meier analysis: `pip install lifelines`")

    # Save summary
    results_path = out_dir / "survival_summary.json"
    results_path.write_text(json.dumps({
        "total_patients": len(os_data),
        "events": events,
        "censored": censored,
    }, indent=2))
    lines.append(f"### Summary saved to: {results_path}")

    return "\n".join(lines)
