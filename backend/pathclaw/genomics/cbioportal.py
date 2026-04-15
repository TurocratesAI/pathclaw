"""cBioPortal REST API client for querying cancer genomics data.

Provides access to mutations, clinical data, CNA, expression, and
MSI scores from public cBioPortal studies (e.g. TCGA pan-cancer).
"""

from __future__ import annotations

import logging
from typing import Optional
from collections import Counter, defaultdict

import httpx

logger = logging.getLogger("pathclaw.genomics")

CBIOPORTAL_BASE = "https://www.cbioportal.org/api"

# Common TCGA study IDs in cBioPortal
TCGA_STUDY_MAP: dict[str, str] = {
    "TCGA-UCEC": "ucec_tcga_pan_can_atlas_2018",
    "TCGA-BRCA": "brca_tcga_pan_can_atlas_2018",
    "TCGA-LUAD": "luad_tcga_pan_can_atlas_2018",
    "TCGA-LUSC": "lusc_tcga_pan_can_atlas_2018",
    "TCGA-COAD": "coadread_tcga_pan_can_atlas_2018",
    "TCGA-READ": "coadread_tcga_pan_can_atlas_2018",
    "TCGA-STAD": "stad_tcga_pan_can_atlas_2018",
    "TCGA-KIRC": "kirc_tcga_pan_can_atlas_2018",
    "TCGA-KIRP": "kirp_tcga_pan_can_atlas_2018",
    "TCGA-KICH": "kich_tcga_pan_can_atlas_2018",
    "TCGA-OV": "ov_tcga_pan_can_atlas_2018",
    "TCGA-PRAD": "prad_tcga_pan_can_atlas_2018",
    "TCGA-BLCA": "blca_tcga_pan_can_atlas_2018",
    "TCGA-SKCM": "skcm_tcga_pan_can_atlas_2018",
    "TCGA-GBM": "gbm_tcga_pan_can_atlas_2018",
    "TCGA-LGG": "lgg_tcga_pan_can_atlas_2018",
    "TCGA-HNSC": "hnsc_tcga_pan_can_atlas_2018",
    "TCGA-THCA": "thca_tcga_pan_can_atlas_2018",
    "TCGA-LIHC": "lihc_tcga_pan_can_atlas_2018",
    "TCGA-PAAD": "paad_tcga_pan_can_atlas_2018",
}


async def _get(endpoint: str, params: Optional[dict] = None, timeout: float = 30.0) -> dict | list:
    """Make a GET request to cBioPortal API."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(
            f"{CBIOPORTAL_BASE}{endpoint}",
            params=params,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


async def _post(endpoint: str, json_body: dict, timeout: float = 60.0) -> dict | list:
    """Make a POST request to cBioPortal API."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{CBIOPORTAL_BASE}{endpoint}",
            json=json_body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


def _resolve_study_id(study_id: str) -> str:
    """Resolve a TCGA project name to a cBioPortal study ID."""
    return TCGA_STUDY_MAP.get(study_id.upper(), study_id)


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------


async def query_clinical(
    study_id: str,
    clinical_attributes: Optional[list[str]] = None,
) -> str:
    """Fetch clinical data for all patients in a study.

    Returns formatted summary of clinical attributes.
    """
    study_id = _resolve_study_id(study_id)

    # Get all clinical data for the study
    try:
        patients = await _get(f"/studies/{study_id}/clinical-data", params={
            "clinicalDataType": "PATIENT",
            "projection": "DETAILED",
        })
    except httpx.HTTPStatusError as e:
        return f"Error querying cBioPortal: {e.response.status_code} — {e.response.text[:200]}"

    if not patients:
        return f"No clinical data found for study {study_id}"

    # Group by attribute
    attr_values: dict[str, list[str]] = defaultdict(list)
    for entry in patients:
        attr_id = entry.get("clinicalAttributeId", "")
        value = entry.get("value", "")
        if clinical_attributes and attr_id not in clinical_attributes:
            continue
        if value and value not in ("NA", "N/A", "[Not Available]", "[Not Evaluated]"):
            attr_values[attr_id].append(value)

    if not attr_values:
        return f"No matching clinical attributes found in {study_id}"

    lines = [f"## Clinical Data: {study_id}", f"- **Patients**: {len(set(e.get('patientId', '') for e in patients))}"]

    # If specific attributes requested, show distribution for each
    if clinical_attributes:
        for attr in clinical_attributes:
            vals = attr_values.get(attr, [])
            if vals:
                dist = Counter(vals).most_common(20)
                lines.append(f"\n### {attr}")
                for val, count in dist:
                    lines.append(f"  {val}: {count}")
            else:
                lines.append(f"\n### {attr}: not found")
    else:
        # Show summary of all attributes (top 20)
        lines.append(f"\n### Available Attributes ({len(attr_values)} total)")
        for attr_id in sorted(attr_values.keys())[:30]:
            vals = attr_values[attr_id]
            unique = len(set(vals))
            lines.append(f"  - **{attr_id}**: {len(vals)} values, {unique} unique")

    return "\n".join(lines)


async def query_mutations(
    study_id: str,
    gene_list: Optional[list[str]] = None,
) -> str:
    """Fetch mutation data for a study, optionally filtered by genes.

    Returns mutation summary with gene frequencies and variant details.
    """
    study_id = _resolve_study_id(study_id)

    # Get molecular profiles (need the mutation profile ID)
    try:
        profiles = await _get(f"/studies/{study_id}/molecular-profiles")
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.status_code} — {e.response.text[:200]}"

    mut_profile = None
    for p in profiles:
        if p.get("molecularAlterationType") == "MUTATION_EXTENDED":
            mut_profile = p["molecularProfileId"]
            break

    if not mut_profile:
        return f"No mutation profile found for {study_id}"

    # Fetch mutations
    try:
        if gene_list:
            # Get entrez gene IDs first
            gene_data = await _post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", json_body=gene_list)
            entrez_ids = [g["entrezGeneId"] for g in gene_data if "entrezGeneId" in g]
            entrez_to_hugo = {g["entrezGeneId"]: g["hugoGeneSymbol"] for g in gene_data if "entrezGeneId" in g}
            if not entrez_ids:
                return f"No valid genes found in cBioPortal for: {gene_list}"

            mutations = await _post(
                f"/molecular-profiles/{mut_profile}/mutations/fetch",
                json_body={
                    "entrezGeneIds": entrez_ids,
                    "sampleListId": f"{study_id}_all",
                },
            )
            # Inject hugo symbol since the API only returns entrezGeneId
            for m in mutations:
                if "hugoGeneSymbol" not in m or not m["hugoGeneSymbol"]:
                    m["hugoGeneSymbol"] = entrez_to_hugo.get(m.get("entrezGeneId"), "?")
        else:
            # Fetch all mutations (can be large — use sample list endpoint)
            mutations = await _get(
                f"/molecular-profiles/{mut_profile}/mutations",
                params={"sampleListId": f"{study_id}_all", "projection": "SUMMARY"},
            )
    except httpx.HTTPStatusError as e:
        return f"Error fetching mutations: {e.response.status_code} — {e.response.text[:200]}"

    if not mutations:
        return f"No mutations found for {study_id}" + (f" (genes: {gene_list})" if gene_list else "")

    # Summarize
    gene_counts: dict[str, int] = Counter()
    sample_counts: dict[str, int] = Counter()
    variant_classes: dict[str, int] = Counter()
    total_samples: set[str] = set()

    for m in mutations:
        gene = m.get("hugoGeneSymbol") or (m.get("gene") or {}).get("hugoGeneSymbol") or "?"
        sample = m.get("sampleId", "")
        vclass = m.get("mutationType", "Unknown")
        gene_counts[gene] += 1
        sample_counts[sample] += 1
        variant_classes[vclass] += 1
        total_samples.add(sample)

    lines = [
        f"## Mutations: {study_id}",
        f"- **Total mutations**: {len(mutations)}",
        f"- **Samples with mutations**: {len(total_samples)}",
        f"- **Unique genes mutated**: {len(gene_counts)}",
        "",
        "### Top Mutated Genes",
    ]
    for gene, count in gene_counts.most_common(20):
        freq = count / max(len(total_samples), 1) * 100
        lines.append(f"  {gene}: {count} mutations ({freq:.1f}% of samples)")

    lines.append("\n### Variant Classification")
    for vc, count in variant_classes.most_common():
        lines.append(f"  {vc}: {count}")

    return "\n".join(lines)


async def query_cna(
    study_id: str,
    gene_list: Optional[list[str]] = None,
) -> str:
    """Fetch copy number alteration data."""
    study_id = _resolve_study_id(study_id)

    try:
        profiles = await _get(f"/studies/{study_id}/molecular-profiles")
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.status_code}"

    cna_profile = None
    for p in profiles:
        if p.get("molecularAlterationType") == "COPY_NUMBER_ALTERATION" and "gistic" not in p.get("molecularProfileId", "").lower():
            cna_profile = p["molecularProfileId"]
            break

    if not cna_profile:
        return f"No CNA profile found for {study_id}"

    if not gene_list:
        return f"CNA profile found: {cna_profile}. Specify gene_list to query specific genes."

    try:
        gene_data = await _post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", json_body=gene_list)
        entrez_ids = [g["entrezGeneId"] for g in gene_data if "entrezGeneId" in g]
        if not entrez_ids:
            return f"No valid genes found in cBioPortal for: {gene_list}"

        cna_data = await _post(
            f"/molecular-profiles/{cna_profile}/discrete-copy-number/fetch",
            json_body={
                "entrezGeneIds": entrez_ids,
                "sampleListId": f"{study_id}_all",
            },
        )
    except httpx.HTTPStatusError as e:
        return f"Error fetching CNA: {e.response.status_code}"

    if not cna_data:
        return f"No CNA data found for genes {gene_list} in {study_id}"

    # CNA values: -2=deep del, -1=shallow del, 0=diploid, 1=gain, 2=amp
    cna_labels = {-2: "Deep Deletion", -1: "Shallow Deletion", 0: "Diploid", 1: "Gain", 2: "Amplification"}
    gene_cna: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for entry in cna_data:
        gene = entry.get("gene", {}).get("hugoGeneSymbol", "?")
        alt = entry.get("alteration", 0)
        gene_cna[gene][cna_labels.get(alt, f"CNA={alt}")] += 1

    lines = [f"## Copy Number Alterations: {study_id}"]
    for gene in gene_list:
        gu = gene.upper()
        if gu in gene_cna:
            lines.append(f"\n### {gu}")
            for label, count in sorted(gene_cna[gu].items()):
                lines.append(f"  {label}: {count}")
        else:
            lines.append(f"\n### {gu}: no CNA data")

    return "\n".join(lines)


async def query_cbioportal(
    study_id: str,
    data_type: str = "clinical",
    gene_list: Optional[list[str]] = None,
    clinical_attributes: Optional[list[str]] = None,
) -> str:
    """Unified cBioPortal query entry point.

    Args:
        study_id: cBioPortal study ID or TCGA project name (e.g. "TCGA-UCEC")
        data_type: mutations | clinical | cna | msi_scores
        gene_list: Genes to query (for mutations/cna)
        clinical_attributes: Clinical attributes to fetch (for clinical type)

    Returns:
        Formatted summary text.
    """
    if data_type == "clinical":
        return await query_clinical(study_id, clinical_attributes)
    elif data_type == "mutations":
        return await query_mutations(study_id, gene_list)
    elif data_type == "cna":
        return await query_cna(study_id, gene_list)
    elif data_type == "msi_scores":
        # MSI scores are stored as clinical attributes
        return await query_clinical(study_id, clinical_attributes=["MSI_SCORE_MANTIS", "MSI_STATUS", "SUBTYPE"])
    else:
        return f"Unknown data_type: {data_type}. Available: clinical, mutations, cna, msi_scores"
