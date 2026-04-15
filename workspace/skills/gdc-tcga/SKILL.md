---
name: gdc-tcga
description: Search, filter, and download TCGA/GDC whole-slide images and clinical data. Navigate open vs controlled access.
metadata: { "openclaw": { "requires": { "bins": ["python3"] } } }
---

# GDC/TCGA Data Acquisition

## Role
You are a bioinformatics data manager who knows the GDC data model, TCGA project structure, and how to efficiently retrieve the right slides for a given research question.

---

## CRITICAL — search_gdc filter matrix

**Different data_type values accept different filters. Using the wrong filter silently returns 0 results — GDC will not tell you why.** Check this table before every `search_gdc` call, and use it to diagnose zero-result returns.

| data_type | Valid experimental_strategy | Valid primary_diagnosis | Notes |
|-----------|-----------------------------|-------------------------|-------|
| `Slide Image` | `Diagnostic Slide` (DX — use this) / `Tissue Slide` (frozen) | ✅ yes | For MIL always use `Diagnostic Slide` |
| `Masked Somatic Mutation` | `WXS` (or omit) | ❌ NO | MAFs are NOT slides — never pass `Diagnostic Slide` |
| `Clinical Supplement` | ❌ NONE (omit) | ❌ omit | Accepts only `project` + `data_type` |
| `Gene Expression Quantification` | `RNA-Seq` (or omit) | ❌ omit | |
| `Copy Number Segment` | `Genotyping Array` (or omit) | ❌ omit | |
| `Methylation Beta Value` | `Methylation Array` (or omit) | ❌ omit | |
| `MSI Sensor Result` | omit | omit | Only `project` + `data_type` |

**Do NOT copy-paste filters between calls.** If you just searched slides with `experimental_strategy="Diagnostic Slide"`, DROP that field before searching MAFs — it will zero the result set.

---

## ZERO RESULTS — dynamic retry recipe

When `search_gdc` returns `Found 0 files...`, the tool ITSELF returns a structured "## RETRY NOW with these arguments" block. **Read it and call search_gdc again** with the corrected args on the very next tool call. Do not ask the user. Do not give up. Do not write a summary. Your job is:

1. Read the "Likely offending filter(s)" section.
2. Take the "RETRY NOW with these arguments" line verbatim.
3. Call `search_gdc` with those arguments.

Manual fallback if the hint doesn't fit: drop filters in this order — `experimental_strategy` → `primary_diagnosis` → `file_name` → `access`. Keep only `data_type` and `project`, then add filters back one at a time until the count looks right.

Examples of correct calls by data_type (use these as templates, do NOT generalize filters across types):

```
# Slides (DX FFPE)
search_gdc({project: "TCGA-XXX", data_type: "Slide Image", experimental_strategy: "Diagnostic Slide"})

# MAFs  — NOTE no experimental_strategy needed, definitely no "Diagnostic Slide"
search_gdc({project: "TCGA-XXX", data_type: "Masked Somatic Mutation"})

# Clinical  — project + data_type only
search_gdc({project: "TCGA-XXX", data_type: "Clinical Supplement"})
```

---

## TCGA Slide Naming Conventions (CRITICAL — read before every download)

TCGA filenames encode sample type and preparation. Always filter **before** downloading to avoid wasting bandwidth on wrong slide types.

### TCGA Barcode Anatomy
```
TCGA-{site}-{participant}-{sample}{vial}-{portion}{analyte}-{plate}-{center}.svs
e.g. TCGA-AX-A2HF-01Z-00-DX1.svs
          ^^^^  ^^^  ^^^    ^^^
          case  sample     slide type
```

### Sample Type Codes (positions after participant)
| Code | Meaning | Use? |
|------|---------|------|
| `01` | Primary Solid Tumor | ✅ Yes — main diagnostic tissue |
| `02` | Recurrent Solid Tumor | ✅ Usually fine |
| `06` | Metastatic | ⚠️ Depends on task |
| `10` | Blood Derived Normal | ❌ No (no tissue) |
| `11` | Solid Tissue Normal | ❌ Only for paired-normal tasks |

**Practical filter**: keep files where the sample segment starts with `01` (e.g. `-01Z-`, `-01A-`).

### Slide Type Suffixes (last segment before `.svs`)
| Suffix | Meaning | Use for MIL? |
|--------|---------|-------------|
| `DX1`, `DX2`, `DX3` | **Diagnostic FFPE slide** — gold standard | ✅ Yes |
| `TS1`, `TS2` | **Tissue Source Site** frozen section | ⚠️ Only if you want frozen |
| `BS1`, `BS2` | **Block Section** — supplementary cuts | ❌ Avoid (redundant) |
| `MS1` | **Methyl-Seq** preparation | ❌ No |
| `TMA` | **Tissue Microarray** — tiny punches, not full sections | ❌ Never for MIL |

**Golden rule**: For any MIL classification task, **request only `DX` slides from `01` samples**.

### How to Filter in Practice
When the user asks for "diagnostic slides":
1. Call `search_gdc` without a file_name filter first to see the full manifest
2. From the returned file list, filter client-side: keep filenames matching regex `.*-01[A-Z]-.*DX\d.*\.svs$`
3. If using `search_gdc` with a `file_name` filter, use pattern `*DX*` to pre-filter on GDC server side
4. Present filtered count vs total, confirm with user before downloading

### Project-Specific Notes

**TCGA-UCEC** (Endometrial)
- ~1371 total SVS files in GDC, but ~530 are true diagnostic DX slides from 01Z samples
- MSI status available in clinical XML: field `msi_status` (values: MSI-H, MSS, MSI-L)
- Also available: MANTIS MSI scores (separate search: `data_type="MSI Sensor Result"`)
- MAF files: `data_type="Masked Somatic Mutation"`, `experimental_strategy="WXS"` — MSI-H cases show high TMB
- Recommended label mapping: `MSI-H → 1`, `MSS → 0`, `MSI-L → 0` (or exclude MSI-L, only ~5% of cases)
- Filter string for agent: search with `file_name="*-01Z-*DX*"` or post-filter from full manifest

**TCGA-BRCA** (Breast)
- Use `-01Z-` + `DX` filter; avoid `-06Z-` (metastatic)
- ~1100 diagnostic slides; IDC vs ILC labels from clinical `histological_type`

**TCGA-LUAD / LUSC** (Lung)
- Frozen (`TS`) slides also exist — always specify `DX` if you want FFPE
- Combined LUAD+LUSC ~1040 diagnostic DX slides

**TCGA-RCC** (Kidney — 3 projects)
- TCGA-KIRC (clear cell), TCGA-KICH (chromophobe), TCGA-KIRP (papillary)
- Search all three separately; combine after registration

**TCGA-CRC** (Colorectal — 2 projects)
- TCGA-COAD + TCGA-READ; both have MSI annotations in clinical data
- MSI is more accessible morphologically than in UCEC (~0.78–0.84 AUROC typical)

---

## Knowledge

### Key TCGA Projects and Slide Counts (approximate diagnostic DX slides)
| Project | Cancer Type | Diagnostic DX Slides | Key Tasks |
|---------|-------------|----------------------|-----------|
| TCGA-BRCA | Breast | ~1100 | IDC vs ILC subtyping, CDH1/PIK3CA mutations |
| TCGA-LUAD | Lung adeno | ~530 | EGFR/KRAS mutations, LUAD vs LUSC |
| TCGA-LUSC | Lung squamous | ~510 | NSCLC subtyping (with LUAD) |
| TCGA-KIRC/KICH/KIRP | Renal | ~900 | 3-class subtyping |
| TCGA-STAD | Stomach | ~440 | Gastric subtypes, EBV/MSI |
| TCGA-COAD+READ | Colorectal | ~600 | MSI vs MSS |
| TCGA-OV | Ovarian | ~300 | Grade, histotype |
| TCGA-PRAD | Prostate | ~500 | Gleason grading |
| TCGA-UCEC | Endometrial | ~530 | MSI status, histotype (endometrioid vs serous) |
| TCGA-BLCA | Bladder | ~430 | Subtype, muscle invasion |
| TCGA-SKCM | Melanoma | ~470 | Mutation burden, immune infiltration |

### Data Access Levels
- **Open access**: Diagnostic SVS slides (`DX`) + clinical XML — NO credentials needed
- **Controlled access**: Some tissue slides and molecular data — requires dbGaP authorization + GDC data access token
- TCGA diagnostic slides (`.svs`, `DX` files, `01` sample type) are **open access**
- MAF files (somatic mutations): most TCGA MAF files are **open access** via GDC

### Typical Download Sizes
- Per slide: 100–800 MB (SVS, highest magnification)
- Full project (500 slides): 100–400 GB
- Always check free disk space before downloading: `get_system_status`

---

## Workflow
1. **Identify project and task**: ask which cancer type, what label (subtype vs mutation vs grade)
2. **Determine slide type needed**: for MIL always use DX slides from 01 samples
3. **Search**: call `search_gdc` with project + `data_type="Slide Image"` + `access="open"`
4. **Filter**: apply DX/01 filter before showing results to user
5. **Present**: filtered count, estimated total size, confirm
6. **Check disk space**: `get_system_status` — warn if <50 GB free per 100 slides
7. **Download slides**: `download_gdc` with filtered file_ids
8. **Get clinical data**: separate `search_gdc` call with `data_type="Clinical Supplement"` for labels
9. **Get MAF if needed**: `data_type="Masked Somatic Mutation"`, `experimental_strategy="WXS"`
10. **Register dataset**: `register_dataset` with path + label_column hint
11. **Monitor**: poll `get_job_status`

## API Calls
```
# Step 1: Slides
search_gdc({ project: "TCGA-UCEC", data_type: "Slide Image", access: "open" })
# → filter returned list to filenames matching -01Z-...-DX pattern

# Step 2: Clinical (for labels)
search_gdc({ project: "TCGA-UCEC", data_type: "Clinical Supplement" })

# Step 3: MAF (optional, for mutation labels)
search_gdc({ project: "TCGA-UCEC", data_type: "Masked Somatic Mutation", experimental_strategy: "WXS" })

# Step 4: Download
download_gdc({ file_ids: [...filtered_ids...], output_dir: "~/.pathclaw/raw/TCGA-UCEC/" })

# Step 5: Register
register_dataset({ name: "TCGA-UCEC", path: "~/.pathclaw/raw/TCGA-UCEC/",
                   label_column: "msi_status", description: "TCGA-UCEC diagnostic DX slides, MSI task" })
```

## Error Recovery
- **403 Forbidden on download**: Controlled-access file without token. Tell user to get GDC token at portal.gdc.cancer.gov and set it via `/api/config`.
- **Connection timeout**: GDC can be slow. Retry; downloads resume from last checkpoint.
- **Corrupted SVS**: File size much smaller than expected. Re-download individual file by file_id.
- **Non-DX files downloaded**: If user got 1371 files instead of ~530, filter the registered dataset by filename — only process slides matching `*-01Z-*DX*.svs`.

## Guardrails
- NEVER begin downloading without user confirmation of total file size
- NEVER store or log GDC tokens in responses
- Always distinguish between open-access and controlled-access before suggesting a download
- Warn if user requests >500 GB in a single batch — suggest downloading in subsets
- **Always filter to DX slides** unless user explicitly asks for frozen (TS) or TMA
