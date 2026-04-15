"""Genomic file parsers for MAF, VCF, and clinical XML.

Provides structured parsing with summary/query modes so the LLM agent
does not need to write ad-hoc pandas code for common genomic operations.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger("pathclaw.genomics")

# ---------------------------------------------------------------------------
# MAF standard columns (GDC Masked Somatic Mutation)
# ---------------------------------------------------------------------------

MAF_KEY_COLUMNS = [
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Variant_Classification",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode",
    "HGVSp_Short",
]

NONSYNONYMOUS_CLASSES = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "Nonstop_Mutation",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Translation_Start_Site",
}


def _open_maybe_gz(path: str | Path):
    """Open a file, transparently handling .gz."""
    path = Path(path)
    if path.suffix == ".gz" or str(path).endswith(".maf.gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


# ===================================================================
# MAF PARSER
# ===================================================================


def parse_maf(
    path: str | Path,
    query: str = "summary",
    sample_id: Optional[str] = None,
    limit: int = 50,
) -> str:
    """Parse a MAF file and return structured text.

    query modes:
      "summary" — overview stats (variant counts, top genes, TMB, etc.)
      gene name — filter to that gene (e.g. "TP53", "EGFR")
      "variants" — show first `limit` variant rows
    """
    path = Path(path)
    if not path.exists():
        return f"Error: file not found: {path}"

    rows: list[dict] = []
    header: list[str] = []

    with _open_maybe_gz(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            if not header:
                header = line.strip().split("\t")
                continue
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            row = dict(zip(header, fields))
            if sample_id and row.get("Tumor_Sample_Barcode", "") != sample_id:
                if not row.get("Tumor_Sample_Barcode", "").startswith(sample_id):
                    continue
            rows.append(row)

    if not rows:
        return f"MAF file parsed but 0 variants found (file: {path.name})"

    # --- Summary mode ---
    if query == "summary":
        return _maf_summary(rows, path.name)

    # --- Gene query mode ---
    if query != "variants":
        gene = query.upper()
        gene_rows = [r for r in rows if r.get("Hugo_Symbol", "").upper() == gene]
        if not gene_rows:
            return f"No variants found for gene {gene} in {path.name} ({len(rows)} total variants)"
        return _maf_gene_detail(gene_rows, gene, path.name, limit)

    # --- Variants listing mode ---
    return _maf_variants_table(rows[:limit], path.name, len(rows))


def _maf_summary(rows: list[dict], filename: str) -> str:
    """Generate MAF summary statistics."""
    total = len(rows)
    genes = Counter(r.get("Hugo_Symbol", "Unknown") for r in rows)
    samples = set(r.get("Tumor_Sample_Barcode", "") for r in rows)
    var_class = Counter(r.get("Variant_Classification", "Unknown") for r in rows)
    var_type = Counter(r.get("Variant_Type", "Unknown") for r in rows)

    # TMB per sample (nonsynonymous / 30 Mb)
    sample_nonsyn: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("Variant_Classification", "") in NONSYNONYMOUS_CLASSES:
            sample_nonsyn[r.get("Tumor_Sample_Barcode", "")] += 1

    tmb_values = [count / 30.0 for count in sample_nonsyn.values()] if sample_nonsyn else [0]
    tmb_values.sort()
    median_tmb = tmb_values[len(tmb_values) // 2]

    lines = [
        f"## MAF Summary: {filename}",
        f"- **Total variants**: {total:,}",
        f"- **Unique genes**: {len(genes):,}",
        f"- **Unique samples**: {len(samples):,}",
        "",
        "### Top 20 Mutated Genes",
    ]
    for gene, count in genes.most_common(20):
        pct = count / total * 100
        lines.append(f"  {gene}: {count} ({pct:.1f}%)")

    lines.append("")
    lines.append("### Variant Classification Distribution")
    for vc, count in var_class.most_common():
        lines.append(f"  {vc}: {count}")

    lines.append("")
    lines.append("### Variant Type Distribution")
    for vt, count in var_type.most_common():
        lines.append(f"  {vt}: {count}")

    lines.append("")
    lines.append("### Tumor Mutational Burden (per sample)")
    lines.append(f"  Samples with mutations: {len(sample_nonsyn)}")
    lines.append(f"  Median TMB: {median_tmb:.1f} mut/Mb")
    if tmb_values:
        lines.append(f"  Min TMB: {tmb_values[0]:.1f}, Max TMB: {tmb_values[-1]:.1f}")
        high_tmb = sum(1 for t in tmb_values if t >= 10)
        lines.append(f"  TMB-High (≥10): {high_tmb} samples ({high_tmb/len(tmb_values)*100:.1f}%)")

    return "\n".join(lines)


def _maf_gene_detail(rows: list[dict], gene: str, filename: str, limit: int) -> str:
    """Detail view for a single gene."""
    var_class = Counter(r.get("Variant_Classification", "") for r in rows)
    samples = set(r.get("Tumor_Sample_Barcode", "") for r in rows)
    lines = [
        f"## {gene} mutations in {filename}",
        f"- **Total variants**: {len(rows)}",
        f"- **Affected samples**: {len(samples)}",
        "",
        "### Variant Classification",
    ]
    for vc, count in var_class.most_common():
        lines.append(f"  {vc}: {count}")

    lines.append("")
    lines.append(f"### Top {min(limit, len(rows))} Variants")
    lines.append("| Sample | Classification | HGVSp | Chr:Pos |")
    lines.append("|--------|---------------|-------|---------|")
    for r in rows[:limit]:
        sample = r.get("Tumor_Sample_Barcode", "")[:20]
        vc = r.get("Variant_Classification", "")
        hgvs = r.get("HGVSp_Short", r.get("HGVSp", ""))
        pos = f"{r.get('Chromosome', '')}:{r.get('Start_Position', '')}"
        lines.append(f"| {sample} | {vc} | {hgvs} | {pos} |")

    return "\n".join(lines)


def _maf_variants_table(rows: list[dict], filename: str, total: int) -> str:
    """Table of first N variants."""
    lines = [
        f"## Variants from {filename} (showing {len(rows)} of {total})",
        "| Gene | Classification | Type | Sample | HGVSp | Chr:Pos |",
        "|------|---------------|------|--------|-------|---------|",
    ]
    for r in rows:
        gene = r.get("Hugo_Symbol", "")
        vc = r.get("Variant_Classification", "")
        vt = r.get("Variant_Type", "")
        sample = r.get("Tumor_Sample_Barcode", "")[:20]
        hgvs = r.get("HGVSp_Short", r.get("HGVSp", ""))
        pos = f"{r.get('Chromosome', '')}:{r.get('Start_Position', '')}"
        lines.append(f"| {gene} | {vc} | {vt} | {sample} | {hgvs} | {pos} |")
    return "\n".join(lines)


# ===================================================================
# MAF DIRECTORY OPERATIONS (query_mutations, compute_tmb)
# ===================================================================


_maf_cache: dict[str, list[dict]] = {}


def _clear_maf_cache():
    """Clear the MAF parsing cache (e.g. after new files are downloaded)."""
    _maf_cache.clear()


def _load_all_mafs(maf_dir: str | Path) -> list[dict]:
    """Load all MAF files in a directory into a single list of variant dicts.

    Results are cached by resolved directory path to avoid re-parsing
    hundreds of gzipped files across multiple tool calls.
    """
    maf_dir = Path(maf_dir)
    cache_key = str(maf_dir.resolve())
    if cache_key in _maf_cache:
        logger.debug(f"MAF cache hit for {cache_key} ({len(_maf_cache[cache_key])} rows)")
        return _maf_cache[cache_key]

    rows: list[dict] = []
    maf_files = sorted(maf_dir.glob("*.maf*"))  # .maf and .maf.gz
    if not maf_files:
        # Check subdirectories (GDC downloads may nest files)
        maf_files = sorted(maf_dir.rglob("*.maf*"))

    maf_files = [f for f in maf_files if not f.is_dir()]
    logger.info(f"Loading {len(maf_files)} MAF files from {maf_dir} ...")

    for maf_path in maf_files:
        header: list[str] = []
        try:
            with _open_maybe_gz(maf_path) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    if not header:
                        header = line.strip().split("\t")
                        continue
                    fields = line.strip().split("\t")
                    if len(fields) < len(header):
                        fields.extend([""] * (len(header) - len(fields)))
                    rows.append(dict(zip(header, fields)))
        except Exception as e:
            logger.warning(f"Failed to parse {maf_path.name}: {e}")

    logger.info(f"Loaded {len(rows)} variants from {len(maf_files)} MAF files")
    _maf_cache[cache_key] = rows
    return rows


def query_mutations_impl(
    genomic_dir: str,
    gene: str = "*",
    variant_class: str = "",
    min_frequency: float = 0.0,
    output_format: str = "summary",
) -> str:
    """Query mutation data across a cohort of MAF files."""
    rows = _load_all_mafs(genomic_dir)
    if not rows:
        return f"No MAF variants found in {genomic_dir}"

    all_samples = set(r.get("Tumor_Sample_Barcode", "") for r in rows)
    n_samples = len(all_samples)

    # Filter by gene
    if gene != "*":
        gene_upper = gene.upper()
        rows = [r for r in rows if r.get("Hugo_Symbol", "").upper() == gene_upper]
        if not rows:
            return f"No variants found for gene {gene} across {n_samples} samples"

    # Filter by variant class
    if variant_class:
        rows = [r for r in rows if r.get("Variant_Classification", "") == variant_class]

    # Gene frequency calculation
    gene_samples: dict[str, set] = defaultdict(set)
    for r in rows:
        g = r.get("Hugo_Symbol", "Unknown")
        s = r.get("Tumor_Sample_Barcode", "")
        gene_samples[g].add(s)

    # Filter by frequency
    if min_frequency > 0:
        gene_samples = {
            g: s for g, s in gene_samples.items() if len(s) / n_samples >= min_frequency
        }

    if output_format == "gene_list":
        sorted_genes = sorted(gene_samples.keys(), key=lambda g: len(gene_samples[g]), reverse=True)
        lines = [f"Genes mutated in ≥{min_frequency*100:.0f}% of {n_samples} samples:"]
        for g in sorted_genes:
            freq = len(gene_samples[g]) / n_samples * 100
            lines.append(f"  {g}: {len(gene_samples[g])}/{n_samples} ({freq:.1f}%)")
        return "\n".join(lines)

    if output_format == "table" and gene != "*":
        lines = [f"## {gene} mutation status per sample"]
        lines.append(f"Mutated in {len(gene_samples.get(gene.upper(), set()))}/{n_samples} samples")
        lines.append("")
        lines.append("| Sample | Mutated | Variant | HGVSp |")
        lines.append("|--------|---------|---------|-------|")
        mutated = gene_samples.get(gene.upper(), set())
        for s in sorted(all_samples):
            is_mut = s in mutated
            if is_mut:
                var_rows = [r for r in rows if r.get("Tumor_Sample_Barcode") == s]
                vc = var_rows[0].get("Variant_Classification", "") if var_rows else ""
                hgvs = var_rows[0].get("HGVSp_Short", "") if var_rows else ""
            else:
                vc, hgvs = "", ""
            lines.append(f"| {s[:25]} | {'Yes' if is_mut else 'No'} | {vc} | {hgvs} |")
        return "\n".join(lines[:100])  # cap output

    # Default: summary
    lines = [
        f"## Mutation Query: gene={'all' if gene == '*' else gene}",
        f"- **Total samples**: {n_samples}",
        f"- **Total variants (filtered)**: {len(rows)}",
        f"- **Genes with mutations**: {len(gene_samples)}",
        "",
        "### Top Mutated Genes (by sample frequency)",
    ]
    sorted_genes = sorted(gene_samples.keys(), key=lambda g: len(gene_samples[g]), reverse=True)
    for g in sorted_genes[:30]:
        freq = len(gene_samples[g]) / n_samples * 100
        lines.append(f"  {g}: {len(gene_samples[g])}/{n_samples} ({freq:.1f}%)")

    # Variant classification breakdown
    vc_counts = Counter(r.get("Variant_Classification", "") for r in rows)
    lines.append("")
    lines.append("### Variant Classification")
    for vc, count in vc_counts.most_common():
        lines.append(f"  {vc}: {count}")

    return "\n".join(lines)


def compute_tmb_impl(
    maf_dir: str,
    exome_size_mb: float = 30.0,
    variant_classes: Optional[list[str]] = None,
    thresholds: Optional[dict] = None,
) -> str:
    """Compute TMB from MAF files in a directory."""
    if variant_classes is None:
        variant_classes = list(NONSYNONYMOUS_CLASSES)
    if thresholds is None:
        thresholds = {"low": 0, "medium": 6, "high": 10}

    vc_set = set(variant_classes)
    rows = _load_all_mafs(maf_dir)
    if not rows:
        return f"No MAF variants found in {maf_dir}"

    # Count qualifying variants per sample
    sample_counts: dict[str, int] = defaultdict(int)
    all_samples: set[str] = set()
    for r in rows:
        sample = r.get("Tumor_Sample_Barcode", "")
        all_samples.add(sample)
        if r.get("Variant_Classification", "") in vc_set:
            sample_counts[sample] += 1

    # Compute TMB
    tmb_data: list[tuple[str, float, int]] = []
    for sample in sorted(all_samples):
        count = sample_counts.get(sample, 0)
        tmb = count / exome_size_mb
        tmb_data.append((sample, tmb, count))

    tmb_values = sorted(t[1] for t in tmb_data)
    n = len(tmb_values)

    # Classify
    high_thresh = thresholds.get("high", 10)
    med_thresh = thresholds.get("medium", 6)

    if n == 0:
        return (
            f"## TMB Analysis (0 samples)\n"
            f"- **Exome size**: {exome_size_mb} Mb\n"
            f"- **Variant classes counted**: {', '.join(sorted(vc_set))}\n\n"
            f"No samples found in the provided MAF data."
        )

    n_high = sum(1 for t in tmb_values if t >= high_thresh)
    n_med = sum(1 for t in tmb_values if med_thresh <= t < high_thresh)
    n_low = sum(1 for t in tmb_values if t < med_thresh)

    lines = [
        f"## TMB Analysis ({n} samples)",
        f"- **Exome size**: {exome_size_mb} Mb",
        f"- **Variant classes counted**: {', '.join(sorted(vc_set))}",
        "",
        "### Distribution",
        f"  Median TMB: {tmb_values[n // 2]:.2f} mut/Mb",
        f"  Mean TMB: {sum(tmb_values) / n:.2f} mut/Mb",
        f"  Min: {tmb_values[0]:.2f}, Max: {tmb_values[-1]:.2f}",
    ]
    if n >= 4:
        lines.append(f"  IQR: {tmb_values[n // 4]:.2f} – {tmb_values[3 * n // 4]:.2f}")
    lines += [
        "",
        f"### Classification (thresholds: high≥{high_thresh}, medium≥{med_thresh})",
        f"  TMB-High: {n_high} ({n_high / n * 100:.1f}%)",
        f"  TMB-Medium: {n_med} ({n_med / n * 100:.1f}%)",
        f"  TMB-Low: {n_low} ({n_low / n * 100:.1f}%)",
        "",
        "### Per-Sample TMB (top 20 by TMB)",
        "| Sample | Variants | TMB (mut/Mb) | Class |",
        "|--------|----------|-------------|-------|",
    ]
    tmb_data.sort(key=lambda x: x[1], reverse=True)
    for sample, tmb, count in tmb_data[:20]:
        cls = "High" if tmb >= high_thresh else ("Medium" if tmb >= med_thresh else "Low")
        # Truncate sample barcode for readability
        short = sample[:25] if len(sample) > 25 else sample
        lines.append(f"| {short} | {count} | {tmb:.2f} | {cls} |")

    return "\n".join(lines)


# ===================================================================
# VCF PARSER
# ===================================================================


def parse_vcf(
    path: str | Path,
    query: str = "summary",
    sample_id: Optional[str] = None,
    limit: int = 50,
) -> str:
    """Parse a VCF file and return structured text."""
    path = Path(path)
    if not path.exists():
        return f"Error: file not found: {path}"

    meta_lines: list[str] = []
    header: list[str] = []
    variants: list[dict] = []
    sample_names: list[str] = []

    with _open_maybe_gz(path) as fh:
        for line in fh:
            if line.startswith("##"):
                meta_lines.append(line.strip())
                continue
            if line.startswith("#CHROM"):
                header = line.strip().lstrip("#").split("\t")
                if len(header) > 9:
                    sample_names = header[9:]
                continue
            if not header:
                continue
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            row = dict(zip(header, fields))
            variants.append(row)

    if query == "summary":
        return _vcf_summary(variants, sample_names, meta_lines, path.name)

    # Gene query — VCFs don't have Hugo_Symbol directly, check INFO field
    if query != "variants":
        gene = query.upper()
        gene_variants = [v for v in variants if gene in v.get("INFO", "").upper()]
        if not gene_variants:
            return f"No variants mentioning {gene} in INFO field of {path.name}"
        lines = [f"## Variants matching '{gene}' in {path.name}: {len(gene_variants)}"]
        lines.append("| CHROM | POS | REF | ALT | FILTER | INFO (truncated) |")
        lines.append("|-------|-----|-----|-----|--------|------------------|")
        for v in gene_variants[:limit]:
            info = v.get("INFO", "")[:80]
            lines.append(
                f"| {v.get('CHROM','')} | {v.get('POS','')} | {v.get('REF','')} "
                f"| {v.get('ALT','')} | {v.get('FILTER','')} | {info} |"
            )
        return "\n".join(lines)

    # Variants listing
    lines = [f"## Variants from {path.name} (showing {min(limit, len(variants))} of {len(variants)})"]
    lines.append("| CHROM | POS | REF | ALT | QUAL | FILTER |")
    lines.append("|-------|-----|-----|-----|------|--------|")
    for v in variants[:limit]:
        lines.append(
            f"| {v.get('CHROM','')} | {v.get('POS','')} | {v.get('REF','')} "
            f"| {v.get('ALT','')} | {v.get('QUAL','')} | {v.get('FILTER','')} |"
        )
    return "\n".join(lines)


def _vcf_summary(variants: list[dict], samples: list[str], meta: list[str], filename: str) -> str:
    """Generate VCF summary."""
    chrom_counts = Counter(v.get("CHROM", "") for v in variants)
    filter_counts = Counter(v.get("FILTER", "") for v in variants)

    # Detect variant types from ALT
    type_counts: Counter = Counter()
    for v in variants:
        ref = v.get("REF", "")
        alt = v.get("ALT", "")
        if len(ref) == 1 and len(alt) == 1:
            type_counts["SNP"] += 1
        elif len(ref) > len(alt):
            type_counts["DEL"] += 1
        elif len(ref) < len(alt):
            type_counts["INS"] += 1
        else:
            type_counts["MNV/Complex"] += 1

    lines = [
        f"## VCF Summary: {filename}",
        f"- **Total variants**: {len(variants):,}",
        f"- **Samples**: {len(samples)} ({', '.join(samples[:5])}{'...' if len(samples) > 5 else ''})",
        f"- **Meta lines**: {len(meta)}",
        "",
        "### Variant Types",
    ]
    for vt, count in type_counts.most_common():
        lines.append(f"  {vt}: {count}")

    lines.append("")
    lines.append("### FILTER Distribution")
    for f, count in filter_counts.most_common():
        lines.append(f"  {f or 'PASS'}: {count}")

    lines.append("")
    lines.append("### Chromosome Distribution (top 10)")
    for ch, count in chrom_counts.most_common(10):
        lines.append(f"  {ch}: {count}")

    return "\n".join(lines)


# ===================================================================
# CLINICAL XML PARSER
# ===================================================================


def parse_clinical_xml(
    path: str | Path,
    query: str = "summary",
    sample_id: Optional[str] = None,
    limit: int = 50,
) -> str:
    """Parse a TCGA clinical XML file.

    Uses namespace-agnostic parsing to handle varying TCGA XML schemas.
    """
    path = Path(path)
    if not path.exists():
        return f"Error: file not found: {path}"

    try:
        tree = ET.parse(str(path))
    except ET.ParseError as e:
        return f"Error parsing XML: {e}"

    root = tree.getroot()

    # Extract all leaf elements (namespace-agnostic)
    fields: dict[str, str] = {}
    for elem in root.iter():
        # Strip namespace
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if elem.text and elem.text.strip():
            val = elem.text.strip()
            if tag in fields:
                # Append if duplicate key
                if fields[tag] != val:
                    fields[tag] = f"{fields[tag]}; {val}"
            else:
                fields[tag] = val

    if query != "summary":
        # Search for specific field
        q_lower = query.lower()
        matches = {k: v for k, v in fields.items() if q_lower in k.lower()}
        if not matches:
            return f"No fields matching '{query}' in {path.name}. Available fields ({len(fields)}): {', '.join(sorted(fields.keys())[:30])}"
        lines = [f"## Fields matching '{query}' in {path.name}"]
        for k, v in sorted(matches.items()):
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    # Summary: show all fields grouped by prefix
    lines = [
        f"## Clinical XML Summary: {path.name}",
        f"- **Total fields**: {len(fields)}",
        "",
    ]

    # Group by common prefixes for readability
    key_fields = [
        "bcr_patient_barcode", "bcr_patient_uuid",
        "gender", "vital_status", "days_to_death", "days_to_last_followup",
        "age_at_initial_pathologic_diagnosis", "race", "ethnicity",
        "histological_type", "tumor_stage", "clinical_stage",
        "primary_diagnosis", "site_of_resection_or_biopsy",
        "msi_status", "msi_score", "residual_tumor",
    ]

    lines.append("### Key Clinical Fields")
    found_keys = False
    for kf in key_fields:
        for k, v in fields.items():
            if k.lower() == kf.lower() or k.lower().endswith(f":{kf.lower()}"):
                lines.append(f"  {kf}: {v}")
                found_keys = True
                break
    if not found_keys:
        lines.append("  (none of the standard fields found)")

    lines.append("")
    lines.append(f"### All Fields ({len(fields)})")
    for k, v in sorted(fields.items()):
        lines.append(f"  {k}: {v[:80]}")

    return "\n".join(lines)


# ===================================================================
# AUTO-DETECT + DISPATCH
# ===================================================================


def detect_file_type(path: str | Path) -> str:
    """Detect genomic file type from extension."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".maf") or name.endswith(".maf.gz"):
        return "maf"
    if name.endswith(".vcf") or name.endswith(".vcf.gz"):
        return "vcf"
    if name.endswith(".xml"):
        return "clinical_xml"
    if name.endswith(".tsv") or name.endswith(".csv"):
        return "tsv"
    # Sniff first line
    try:
        with _open_maybe_gz(path) as fh:
            first = ""
            for line in fh:
                if not line.startswith("#"):
                    first = line
                    break
            if "Hugo_Symbol" in first or "Variant_Classification" in first:
                return "maf"
            if first.startswith("CHROM") or "##fileformat=VCF" in first:
                return "vcf"
    except Exception:
        pass
    return "tsv"


def parse_genomic_file(
    file_path: str,
    file_type: str = "auto",
    query: str = "summary",
    sample_id: Optional[str] = None,
    limit: int = 50,
) -> str:
    """Unified entry point: detect type and dispatch to the right parser."""
    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"

    if file_type == "auto":
        file_type = detect_file_type(path)

    if file_type == "maf":
        return parse_maf(path, query=query, sample_id=sample_id, limit=limit)
    elif file_type == "vcf":
        return parse_vcf(path, query=query, sample_id=sample_id, limit=limit)
    elif file_type == "clinical_xml":
        return parse_clinical_xml(path, query=query, sample_id=sample_id, limit=limit)
    elif file_type == "tsv":
        return _parse_tsv(path, query=query, limit=limit)
    else:
        return f"Unknown file type: {file_type}. Supported: maf, vcf, clinical_xml, tsv"


def _parse_tsv(path: Path, query: str = "summary", limit: int = 50) -> str:
    """Generic TSV/CSV parser with summary."""
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    rows: list[dict] = []
    header: list[str] = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if not header:
                header = line.strip().split(sep)
                continue
            fields = line.strip().split(sep)
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            rows.append(dict(zip(header, fields)))

    if not rows:
        return f"TSV file {path.name}: 0 rows, columns: {header}"

    lines = [
        f"## TSV Summary: {path.name}",
        f"- **Rows**: {len(rows):,}",
        f"- **Columns**: {len(header)}",
        f"- **Column names**: {', '.join(header[:20])}{'...' if len(header) > 20 else ''}",
    ]

    if query != "summary":
        # Filter/search
        q_lower = query.lower()
        matching_cols = [h for h in header if q_lower in h.lower()]
        if matching_cols:
            lines.append(f"\n### Values for columns matching '{query}'")
            for col in matching_cols[:5]:
                vals = Counter(r.get(col, "") for r in rows)
                lines.append(f"\n**{col}** ({len(vals)} unique):")
                for v, count in vals.most_common(10):
                    lines.append(f"  {v or '(empty)'}: {count}")
        return "\n".join(lines)

    # Summary: show value distributions for low-cardinality columns
    lines.append("\n### Column Value Distributions (low cardinality)")
    for col in header[:15]:
        vals = set(r.get(col, "") for r in rows)
        if 2 <= len(vals) <= 20:
            val_counts = Counter(r.get(col, "") for r in rows)
            lines.append(f"\n**{col}** ({len(vals)} unique):")
            for v, count in val_counts.most_common(10):
                lines.append(f"  {v or '(empty)'}: {count}")

    return "\n".join(lines)
