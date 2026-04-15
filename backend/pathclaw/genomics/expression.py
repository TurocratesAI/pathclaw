"""Gene expression data parsing and analysis.

Handles GDC/TCGA gene expression files (HTSeq counts, STAR counts, FPKM),
computes per-gene statistics, and supports differential expression analysis.
"""

from __future__ import annotations

import csv
import gzip
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pathclaw.genomics")

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()


def _extract_patient_barcode(identifier: str) -> str:
    """Extract TCGA patient barcode (TCGA-XX-XXXX)."""
    parts = identifier.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return identifier


def _detect_expression_format(path: Path) -> str:
    """Detect expression file format from header/content."""
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt") as f:
            first_line = f.readline().strip()
            if "\t" in first_line:
                cols = first_line.split("\t")
                if "gene_id" in cols[0].lower() or "ensembl" in cols[0].lower():
                    return "star_counts" if len(cols) >= 4 else "htseq_counts"
                if "fpkm" in first_line.lower():
                    return "fpkm"
            # Two-column format: gene_id \t count
            parts = first_line.split("\t")
            if len(parts) == 2:
                return "htseq_counts"
    except Exception:
        pass
    return "unknown"


def _parse_expression_file(
    path: Path,
    format_type: str = "auto",
) -> dict[str, dict]:
    """Parse a single expression file.

    Returns: {gene_id: {gene_name, unstranded, stranded_first, stranded_second, tpm_unstranded, fpkm_unstranded}}
    """
    if format_type == "auto":
        format_type = _detect_expression_format(path)

    opener = gzip.open if path.suffix == ".gz" else open
    genes: dict[str, dict] = {}

    try:
        with opener(path, "rt") as f:
            reader = csv.reader(f, delimiter="\t")

            if format_type in ("star_counts",):
                # STAR counts format: gene_id, gene_name, gene_type, unstranded, stranded_first, stranded_second, tpm_unstranded, fpkm_unstranded
                header = next(reader, None)
                if header is None:
                    return {}

                # Map column names
                col_map = {h.lower().strip(): i for i, h in enumerate(header)}

                for row in reader:
                    if len(row) < 2:
                        continue
                    gene_id = row[0].strip()
                    # Skip header rows or N_ special rows
                    if gene_id.startswith("N_") or gene_id == "gene_id":
                        continue

                    entry: dict = {"gene_id": gene_id}
                    if "gene_name" in col_map and len(row) > col_map["gene_name"]:
                        entry["gene_name"] = row[col_map["gene_name"]]
                    if "gene_type" in col_map and len(row) > col_map["gene_type"]:
                        entry["gene_type"] = row[col_map["gene_type"]]

                    # Try to get count columns
                    for col_name in ("unstranded", "stranded_first", "stranded_second",
                                     "tpm_unstranded", "fpkm_unstranded"):
                        if col_name in col_map and len(row) > col_map[col_name]:
                            try:
                                entry[col_name] = float(row[col_map[col_name]])
                            except (ValueError, TypeError):
                                pass

                    genes[gene_id] = entry

            elif format_type in ("htseq_counts",):
                # Simple two-column: gene_id \t count
                for row in reader:
                    if len(row) < 2:
                        continue
                    gene_id = row[0].strip()
                    if gene_id.startswith("__"):  # __no_feature, __ambiguous, etc.
                        continue
                    try:
                        genes[gene_id] = {"gene_id": gene_id, "count": float(row[1])}
                    except ValueError:
                        continue

            else:
                # Try generic TSV
                header = next(reader, None)
                if header:
                    for row in reader:
                        if len(row) >= 2:
                            genes[row[0]] = dict(zip(header, row))

    except Exception as e:
        logger.warning(f"Error parsing {path}: {e}")

    return genes


def parse_gene_expression(
    file_path: str,
    query: str = "summary",
    gene_list: Optional[list[str]] = None,
    limit: int = 50,
) -> str:
    """Parse a gene expression file and return summary or gene-specific data.

    Args:
        file_path: Path to expression file (.tsv, .tsv.gz, .counts, etc.)
        query: "summary" for overview, gene name for specific gene, "top" for highest expressed
        gene_list: List of genes to extract (alternative to single query)
        limit: Max genes to return in detail mode

    Returns:
        Formatted summary text.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        return f"Error: file not found: {path}"

    genes = _parse_expression_file(path)
    if not genes:
        return f"No gene expression data parsed from {path}"

    fmt = _detect_expression_format(path)

    if gene_list and query == "summary":
        # gene_list takes precedence over default summary
        query = None

    if query == "summary":
        # Get the count/expression column
        count_key = None
        sample_gene = next(iter(genes.values()))
        for k in ("unstranded", "count", "fpkm_unstranded", "tpm_unstranded"):
            if k in sample_gene:
                count_key = k
                break

        total_genes = len(genes)
        protein_coding = sum(1 for g in genes.values() if g.get("gene_type") == "protein_coding")

        lines = [
            f"## Gene Expression: {path.name}",
            f"- **Format**: {fmt}",
            f"- **Total genes/features**: {total_genes}",
        ]
        if protein_coding:
            lines.append(f"- **Protein-coding**: {protein_coding}")

        if count_key:
            values = [g.get(count_key, 0) for g in genes.values() if isinstance(g.get(count_key), (int, float))]
            if values:
                nonzero = sum(1 for v in values if v > 0)
                lines.append(f"- **Measured column**: {count_key}")
                lines.append(f"- **Non-zero genes**: {nonzero}/{len(values)}")

                # Top expressed genes
                sorted_genes = sorted(genes.values(), key=lambda g: g.get(count_key, 0), reverse=True)
                lines.append(f"\n### Top {min(20, len(sorted_genes))} Expressed Genes")
                for g in sorted_genes[:20]:
                    name = g.get("gene_name", g.get("gene_id", "?"))
                    val = g.get(count_key, 0)
                    lines.append(f"  {name}: {val:,.0f}")

        # Available columns
        cols = list(sample_gene.keys())
        lines.append(f"\n### Available columns: {', '.join(cols)}")

        return "\n".join(lines)

    elif gene_list:
        # Query specific genes
        lines = [f"## Gene Expression Query: {len(gene_list)} genes"]
        for gene_query in gene_list[:limit]:
            found = False
            for gid, gdata in genes.items():
                gname = gdata.get("gene_name", "")
                if gname.upper() == gene_query.upper() or gid.upper().startswith(gene_query.upper()):
                    lines.append(f"\n### {gname or gid}")
                    for k, v in gdata.items():
                        if k not in ("gene_id",):
                            lines.append(f"  {k}: {v}")
                    found = True
                    break
            if not found:
                lines.append(f"\n### {gene_query}: not found")
        return "\n".join(lines)

    else:
        # Query a specific gene
        for gid, gdata in genes.items():
            gname = gdata.get("gene_name", "")
            if gname.upper() == query.upper() or gid.upper().startswith(query.upper()):
                lines = [f"## {gname or gid}"]
                for k, v in gdata.items():
                    lines.append(f"  {k}: {v}")
                return "\n".join(lines)

        return f"Gene '{query}' not found. Available gene count: {len(genes)}"


def compute_cohort_expression(
    expression_dir: str,
    gene_list: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Compute per-gene statistics across a cohort of expression files.

    Args:
        expression_dir: Directory with expression files
        gene_list: Genes to analyze (default: all protein-coding)
        output_path: Where to save the expression matrix

    Returns:
        Summary with mean/std expression per gene, sample count.
    """
    expr_dir = Path(expression_dir).expanduser()
    if not expr_dir.exists():
        return f"Error: directory not found: {expr_dir}"

    # Find expression files
    expr_files = (
        list(expr_dir.rglob("*.counts"))
        + list(expr_dir.rglob("*counts.tsv*"))
        + list(expr_dir.rglob("*.tsv.gz"))
        + list(expr_dir.rglob("*.tsv"))
    )
    # Deduplicate
    expr_files = list({str(f): f for f in expr_files}.values())

    if not expr_files:
        return f"No expression files found in {expr_dir}"

    # Parse all files and aggregate
    gene_values: dict[str, list[float]] = defaultdict(list)
    gene_names: dict[str, str] = {}  # gene_id -> gene_name
    n_samples = 0

    for efile in expr_files[:500]:  # cap at 500 files
        genes = _parse_expression_file(efile)
        if not genes:
            continue
        n_samples += 1

        # Determine count column
        sample_gene = next(iter(genes.values()))
        count_key = None
        for k in ("unstranded", "count", "tpm_unstranded"):
            if k in sample_gene:
                count_key = k
                break
        if not count_key:
            continue

        for gid, gdata in genes.items():
            gname = gdata.get("gene_name", "")
            if gene_list:
                if gname.upper() not in [g.upper() for g in gene_list] and gid not in gene_list:
                    continue
            elif gdata.get("gene_type") and gdata["gene_type"] != "protein_coding":
                continue  # Only protein-coding by default

            val = gdata.get(count_key, 0)
            if isinstance(val, (int, float)):
                gene_values[gid].append(val)
                if gname:
                    gene_names[gid] = gname

    if not gene_values:
        return "No expression data aggregated across files."

    # Compute stats
    import statistics

    gene_stats: list[dict] = []
    for gid, vals in gene_values.items():
        if len(vals) < 2:
            continue
        mean_val = statistics.mean(vals)
        std_val = statistics.stdev(vals)
        gene_stats.append({
            "gene_id": gid,
            "gene_name": gene_names.get(gid, ""),
            "mean": mean_val,
            "std": std_val,
            "n_samples": len(vals),
            "nonzero": sum(1 for v in vals if v > 0),
        })

    gene_stats.sort(key=lambda g: g["mean"], reverse=True)

    lines = [
        "## Cohort Expression Summary",
        f"- **Samples**: {n_samples}",
        f"- **Genes analyzed**: {len(gene_stats)}",
    ]

    if gene_list:
        lines.append("\n### Queried Genes")
        for gs in gene_stats[:len(gene_list)]:
            name = gs["gene_name"] or gs["gene_id"]
            lines.append(
                f"  **{name}**: mean={gs['mean']:,.1f}, std={gs['std']:,.1f}, "
                f"non-zero={gs['nonzero']}/{gs['n_samples']}"
            )
    else:
        lines.append("\n### Top 20 Expressed Genes (by mean)")
        for gs in gene_stats[:20]:
            name = gs["gene_name"] or gs["gene_id"]
            lines.append(f"  **{name}**: mean={gs['mean']:,.1f}, std={gs['std']:,.1f}")

    # Save matrix if requested
    if output_path:
        out_p = Path(output_path).expanduser()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["gene_id", "gene_name", "mean", "std", "n_samples", "nonzero"])
            writer.writeheader()
            for gs in gene_stats:
                writer.writerow(gs)
        lines.append(f"\n### Full stats saved to: {out_p}")

    return "\n".join(lines)
