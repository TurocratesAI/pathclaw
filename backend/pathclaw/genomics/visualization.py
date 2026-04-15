"""Genomics visualizations — oncoplot/waterfall mutation landscape.

Generates a mutation landscape plot (oncoplot) from MAF data showing
the top-N mutated genes across all samples, with variant classification
color-coding.
"""

from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from pathclaw.genomics.parsers import _load_all_mafs, NONSYNONYMOUS_CLASSES

logger = logging.getLogger("pathclaw.genomics")

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()

# Variant classification color scheme (matches cBioPortal convention)
VARIANT_COLORS = {
    "Missense_Mutation": "#2ca02c",       # green
    "Nonsense_Mutation": "#1f77b4",       # blue
    "Frame_Shift_Del": "#d62728",         # red
    "Frame_Shift_Ins": "#ff7f0e",         # orange
    "Splice_Site": "#9467bd",             # purple
    "In_Frame_Del": "#e377c2",            # pink
    "In_Frame_Ins": "#bcbd22",            # yellow-green
    "Translation_Start_Site": "#17becf",  # cyan
    "Nonstop_Mutation": "#8c564b",        # brown
    "Multi_Hit": "#7f7f7f",              # gray (multiple mutations)
}


def _extract_patient_barcode(identifier: str) -> str:
    """Extract TCGA patient barcode (TCGA-XX-XXXX)."""
    parts = identifier.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return identifier


def generate_oncoplot(
    maf_dir: str,
    top_n: int = 20,
    output_path: Optional[str] = None,
    title: str = "Mutation Landscape",
    min_frequency: float = 0.0,
) -> str:
    """Generate an oncoplot (mutation landscape) from MAF files.

    Args:
        maf_dir: Directory containing MAF files
        top_n: Number of top mutated genes to show (default: 20)
        output_path: Where to save the plot (default: auto)
        title: Plot title
        min_frequency: Min mutation frequency to include a gene (0-1)

    Returns:
        Summary text with plot path and mutation stats.
    """
    maf_dir_path = Path(maf_dir).expanduser()
    if not maf_dir_path.exists():
        return f"Error: MAF directory not found: {maf_dir}"

    rows = _load_all_mafs(str(maf_dir_path))
    if not rows:
        return f"No MAF data found in {maf_dir}"

    # Build gene × sample × variant_class matrix
    # gene -> sample -> [variant_classes]
    gene_sample_vars: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    all_samples: set[str] = set()

    for r in rows:
        hugo = r.get("Hugo_Symbol", "").strip()
        vclass = r.get("Variant_Classification", "")
        sample = _extract_patient_barcode(r.get("Tumor_Sample_Barcode", ""))
        if not hugo or not sample:
            continue
        if vclass in NONSYNONYMOUS_CLASSES:
            gene_sample_vars[hugo][sample].append(vclass)
            all_samples.add(sample)

    total_samples = len(all_samples)
    if total_samples == 0:
        return "No nonsynonymous mutations found in MAF files."

    # Compute gene frequencies
    gene_freq: dict[str, float] = {}
    for gene, sample_vars in gene_sample_vars.items():
        freq = len(sample_vars) / total_samples
        if freq >= min_frequency:
            gene_freq[gene] = freq

    # Sort by frequency, take top N
    top_genes = sorted(gene_freq, key=gene_freq.get, reverse=True)[:top_n]

    if not top_genes:
        return f"No genes above min_frequency={min_frequency:.0%}"

    # Determine output path
    if output_path:
        out_path = Path(output_path).expanduser()
    else:
        out_path = PATHCLAW_DATA_DIR / "analysis" / "oncoplot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to generate the plot
    try:
        _render_oncoplot(
            gene_sample_vars=gene_sample_vars,
            top_genes=top_genes,
            all_samples=sorted(all_samples),
            gene_freq=gene_freq,
            out_path=out_path,
            title=title,
        )
        plot_msg = f"Plot saved to: {out_path}"
    except ImportError:
        plot_msg = "matplotlib not available — plot not generated"
    except Exception as e:
        plot_msg = f"Plot generation failed: {e}"

    # Generate text summary
    lines = [
        f"## Oncoplot: {title}",
        f"- **Total samples**: {total_samples}",
        f"- **Genes shown**: {len(top_genes)}",
        f"- {plot_msg}",
        "",
        "### Top Mutated Genes",
    ]
    for gene in top_genes:
        freq = gene_freq[gene]
        n_samples = len(gene_sample_vars[gene])
        # Most common variant class
        all_vcs = []
        for sample_vars in gene_sample_vars[gene].values():
            all_vcs.extend(sample_vars)
        top_vc = Counter(all_vcs).most_common(1)[0][0] if all_vcs else "?"
        lines.append(f"  **{gene}**: {n_samples}/{total_samples} ({freq:.0%}) — mostly {top_vc}")

    return "\n".join(lines)


def _render_oncoplot(
    gene_sample_vars: dict[str, dict[str, list[str]]],
    top_genes: list[str],
    all_samples: list[str],
    gene_freq: dict[str, float],
    out_path: Path,
    title: str,
) -> None:
    """Render the oncoplot using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    n_genes = len(top_genes)
    n_samples = len(all_samples)

    # Sort samples by total mutation count (most mutated first)
    sample_mut_count = Counter()
    for gene in top_genes:
        for sample in gene_sample_vars[gene]:
            if sample in all_samples:
                sample_mut_count[sample] += 1
    sorted_samples = [s for s, _ in sample_mut_count.most_common()] + [
        s for s in all_samples if s not in sample_mut_count
    ]

    sample_idx = {s: i for i, s in enumerate(sorted_samples)}

    fig_width = max(12, n_samples * 0.05)
    fig_height = max(4, n_genes * 0.4 + 2)
    fig, (ax_bar, ax_main) = plt.subplots(
        2, 1, figsize=(fig_width, fig_height),
        height_ratios=[1, n_genes],
        sharex=True,
    )

    # Top bar: mutation count per sample
    sample_counts = [sample_mut_count.get(s, 0) for s in sorted_samples]
    ax_bar.bar(range(n_samples), sample_counts, color="#333333", width=1.0, edgecolor="none")
    ax_bar.set_ylabel("# Mutations")
    ax_bar.set_title(title)
    ax_bar.set_xlim(-0.5, n_samples - 0.5)

    # Main heatmap
    for gi, gene in enumerate(top_genes):
        for sample, vclasses in gene_sample_vars[gene].items():
            if sample not in sample_idx:
                continue
            si = sample_idx[sample]
            # Use the most "severe" variant class for color
            if not vclasses:
                continue
            vc = vclasses[0] if len(vclasses) == 1 else "Multi_Hit" if len(set(vclasses)) > 1 else vclasses[0]
            color = VARIANT_COLORS.get(vc, "#aaaaaa")
            ax_main.add_patch(plt.Rectangle((si - 0.45, gi - 0.4), 0.9, 0.8, color=color, linewidth=0))

    # Background gray for unmutated
    for gi in range(n_genes):
        ax_main.add_patch(plt.Rectangle((-0.5, gi - 0.45), n_samples, 0.9,
                                         color="#f0f0f0", linewidth=0, zorder=0))

    ax_main.set_yticks(range(n_genes))
    ax_main.set_yticklabels([f"{g} ({gene_freq[g]:.0%})" for g in top_genes], fontsize=8)
    ax_main.set_ylim(-0.5, n_genes - 0.5)
    ax_main.set_xlim(-0.5, n_samples - 0.5)
    ax_main.invert_yaxis()
    ax_main.set_xlabel(f"Samples (n={n_samples})")

    # Remove x ticks (too many samples)
    ax_main.set_xticks([])

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=vc.replace("_", " "))
                      for vc, c in VARIANT_COLORS.items() if vc != "Multi_Hit"]
    legend_patches.append(mpatches.Patch(color=VARIANT_COLORS["Multi_Hit"], label="Multi Hit"))
    ax_main.legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   fontsize=7, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Oncoplot saved to {out_path}")
