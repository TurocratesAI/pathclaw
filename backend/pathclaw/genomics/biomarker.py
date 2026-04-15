"""Biomarker discovery — differential analysis and attention-gene correlation.

Identifies genomic features (mutations, expression) that differ between
groups defined by MIL model predictions or known labels.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from pathclaw.genomics.parsers import _load_all_mafs, NONSYNONYMOUS_CLASSES

logger = logging.getLogger("pathclaw.genomics")

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()


def _extract_patient_barcode(identifier: str) -> str:
    """Extract TCGA patient barcode (TCGA-XX-XXXX)."""
    parts = identifier.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return identifier


def _load_groups(
    labels_path: str,
    group_column: str = "label_name",
) -> dict[str, str]:
    """Load patient → group mapping from labels.csv."""
    groups: dict[str, str] = {}
    path = Path(labels_path).expanduser()
    if not path.exists():
        return groups
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient = row.get("patient_barcode", "")
            group = row.get(group_column, "")
            if patient and group:
                groups[patient] = group
    return groups


def mutation_enrichment(
    maf_dir: str,
    labels_path: str,
    group_column: str = "label_name",
    min_total_mutations: int = 5,
    top_n: int = 30,
) -> str:
    """Find genes differentially mutated between groups.

    Computes mutation frequency per gene per group and identifies genes
    with the largest frequency difference (enrichment) between groups.

    Args:
        maf_dir: Directory with MAF files
        labels_path: Path to labels.csv with group assignments
        group_column: Column to stratify by
        min_total_mutations: Min mutations across all samples to consider a gene
        top_n: Number of top enriched genes to report

    Returns:
        Summary with enriched genes, frequencies per group, and fold change.
    """
    groups = _load_groups(labels_path, group_column)
    if not groups:
        return f"Error: no groups loaded from {labels_path} (column: {group_column})"

    rows = _load_all_mafs(str(Path(maf_dir).expanduser()))
    if not rows:
        return f"No MAF data in {maf_dir}"

    # Count mutations per gene per group
    # gene -> group -> set of patients with mutation
    gene_group_patients: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    group_sizes: dict[str, int] = Counter(groups.values())

    for r in rows:
        sample = _extract_patient_barcode(r.get("Tumor_Sample_Barcode", ""))
        hugo = r.get("Hugo_Symbol", "").strip()
        vclass = r.get("Variant_Classification", "")

        if sample not in groups or not hugo:
            continue
        if vclass not in NONSYNONYMOUS_CLASSES:
            continue

        group = groups[sample]
        gene_group_patients[hugo][group].add(sample)

    if not gene_group_patients:
        return "No mutations matched to labeled patients."

    group_names = sorted(group_sizes.keys())
    if len(group_names) < 2:
        return f"Need at least 2 groups for enrichment, found: {group_names}"

    # Compute enrichment for each gene
    enrichment: list[dict] = []
    for gene, group_pats in gene_group_patients.items():
        total_muts = sum(len(pats) for pats in group_pats.values())
        if total_muts < min_total_mutations:
            continue

        freqs = {}
        for gname in group_names:
            n_mutated = len(group_pats.get(gname, set()))
            n_total = group_sizes[gname]
            freqs[gname] = n_mutated / max(n_total, 1)

        # Compute max absolute difference between any two groups
        freq_vals = list(freqs.values())
        max_diff = max(freq_vals) - min(freq_vals)

        enrichment.append({
            "gene": gene,
            "total_mutations": total_muts,
            "frequencies": freqs,
            "max_diff": max_diff,
        })

    enrichment.sort(key=lambda e: e["max_diff"], reverse=True)

    # Format output
    lines = [
        "## Mutation Enrichment Analysis",
        f"- **Groups**: {', '.join(f'{g} (n={group_sizes[g]})' for g in group_names)}",
        f"- **Genes tested**: {len(enrichment)}",
        "",
        f"### Top {min(top_n, len(enrichment))} Differentially Mutated Genes",
        "",
    ]

    header = f"| Gene | {'  |  '.join(group_names)} | Diff |"
    sep = "|" + "|".join(["---"] * (len(group_names) + 2)) + "|"
    lines.append(header)
    lines.append(sep)

    for e in enrichment[:top_n]:
        freq_strs = [f"{e['frequencies'].get(g, 0):.0%}" for g in group_names]
        lines.append(f"| {e['gene']} | {'  |  '.join(freq_strs)} | {e['max_diff']:.0%} |")

    # Try Fisher's exact test for top genes (if scipy available)
    try:
        from scipy.stats import fisher_exact

        lines.append("\n### Statistical Tests (Fisher's exact, top genes)")
        for e in enrichment[:10]:
            if len(group_names) == 2:
                g1, g2 = group_names
                a = len(gene_group_patients[e["gene"]].get(g1, set()))
                b = group_sizes[g1] - a
                c = len(gene_group_patients[e["gene"]].get(g2, set()))
                d = group_sizes[g2] - c
                _, pval = fisher_exact([[a, b], [c, d]])
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                lines.append(f"  {e['gene']}: p={pval:.2e} {sig}")
    except ImportError:
        lines.append("\n*Install scipy for Fisher's exact test: `pip install scipy`*")

    return "\n".join(lines)


def attention_gene_correlation(
    dataset_id: str,
    experiment_id: str,
    maf_dir: str,
    gene_list: Optional[list[str]] = None,
    top_n: int = 20,
) -> str:
    """Correlate MIL attention scores with genomic features.

    Identifies whether high-attention patches come from regions with specific
    mutations. Requires: attention scores from evaluation + MAF files.

    This is an advanced analysis that links morphological features (what the
    model finds important) with molecular features (which genes are mutated).

    Args:
        dataset_id: Dataset ID
        experiment_id: Experiment ID (for attention scores)
        maf_dir: Directory with MAF files
        gene_list: Specific genes to correlate (default: top mutated)
        top_n: Number of genes to analyze

    Returns:
        Summary of attention-mutation correlations.
    """
    # Load attention scores
    exp_dir = PATHCLAW_DATA_DIR / "experiments" / experiment_id
    attention_dir = exp_dir / "attention_scores"

    if not attention_dir.exists():
        # Check in eval results
        eval_dirs = list(exp_dir.glob("eval_*"))
        for ed in eval_dirs:
            ad = ed / "attention_scores"
            if ad.exists():
                attention_dir = ad
                break

    if not attention_dir.exists():
        return (
            f"No attention scores found for experiment {experiment_id}. "
            f"Run evaluation with attention saving enabled first."
        )

    # Load MAF data
    rows = _load_all_mafs(str(Path(maf_dir).expanduser()))
    if not rows:
        return f"No MAF data in {maf_dir}"

    # Get mutation status per patient per gene
    patient_genes: dict[str, set[str]] = defaultdict(set)
    all_patients: set[str] = set()
    gene_counts: Counter = Counter()

    for r in rows:
        sample = _extract_patient_barcode(r.get("Tumor_Sample_Barcode", ""))
        hugo = r.get("Hugo_Symbol", "").strip()
        vclass = r.get("Variant_Classification", "")
        if vclass in NONSYNONYMOUS_CLASSES and hugo:
            patient_genes[sample].add(hugo)
            all_patients.add(sample)
            gene_counts[hugo] += 1

    # Determine genes to analyze
    if gene_list:
        target_genes = [g.upper() for g in gene_list]
    else:
        target_genes = [g for g, _ in gene_counts.most_common(top_n)]

    # Load attention scores and compute mean attention per slide
    import torch
    slide_attention: dict[str, float] = {}

    for attn_file in attention_dir.glob("*.pt"):
        slide_stem = attn_file.stem
        patient = _extract_patient_barcode(slide_stem)
        try:
            attn = torch.load(attn_file, map_location="cpu", weights_only=True)
            if isinstance(attn, torch.Tensor):
                slide_attention[patient] = float(attn.mean())
        except Exception:
            continue

    if not slide_attention:
        return "No attention score files could be loaded."

    # For each gene, compare mean attention between mutated vs wild-type
    lines = [
        "## Attention-Mutation Correlation",
        f"- **Slides with attention scores**: {len(slide_attention)}",
        f"- **Patients with mutations**: {len(patient_genes)}",
        f"- **Genes analyzed**: {len(target_genes)}",
        "",
        "### Mean Attention by Mutation Status",
        "| Gene | Mutant (n) | WT (n) | Mutant Attn | WT Attn | Ratio |",
        "|------|-----------|--------|-------------|---------|-------|",
    ]

    for gene in target_genes:
        mutant_attns = []
        wt_attns = []
        for patient, attn in slide_attention.items():
            if gene in patient_genes.get(patient, set()):
                mutant_attns.append(attn)
            elif patient in all_patients:
                wt_attns.append(attn)

        if not mutant_attns or not wt_attns:
            continue

        import statistics
        mut_mean = statistics.mean(mutant_attns)
        wt_mean = statistics.mean(wt_attns)
        ratio = mut_mean / max(wt_mean, 1e-8)

        lines.append(
            f"| {gene} | {len(mutant_attns)} | {len(wt_attns)} | "
            f"{mut_mean:.4f} | {wt_mean:.4f} | {ratio:.2f}x |"
        )

    if len(lines) <= 8:  # Only headers, no data rows
        lines.append("*No genes had both mutant and wild-type samples with attention scores.*")

    return "\n".join(lines)


def biomarker_discovery(
    maf_dir: str,
    labels_path: str,
    dataset_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    analysis_type: str = "mutation_enrichment",
    gene_list: Optional[list[str]] = None,
    group_column: str = "label_name",
) -> str:
    """Unified biomarker discovery entry point.

    Args:
        maf_dir: Directory with MAF files
        labels_path: Path to labels.csv
        dataset_id: Dataset ID (for attention correlation)
        experiment_id: Experiment ID (for attention correlation)
        analysis_type: mutation_enrichment | attention_correlation
        gene_list: Specific genes to analyze
        group_column: Stratification column in labels.csv

    Returns:
        Analysis results.
    """
    if analysis_type == "mutation_enrichment":
        return mutation_enrichment(
            maf_dir=maf_dir,
            labels_path=labels_path,
            group_column=group_column,
        )
    elif analysis_type == "attention_correlation":
        if not dataset_id or not experiment_id:
            return "Error: dataset_id and experiment_id required for attention correlation"
        return attention_gene_correlation(
            dataset_id=dataset_id,
            experiment_id=experiment_id,
            maf_dir=maf_dir,
            gene_list=gene_list,
        )
    else:
        return f"Unknown analysis_type: {analysis_type}. Available: mutation_enrichment, attention_correlation"
