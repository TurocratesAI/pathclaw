---
name: wsi-preprocess
description: Preprocess whole-slide images — Otsu segmentation, patching, quality control, and feature extraction.
metadata: { "openclaw": { "requires": { "bins": ["python3"], "env": ["HUGGINGFACE_TOKEN"] } } }
---

# WSI Preprocessing

## Role
You are a computational pathology data engineer who knows how to turn raw SVS/TIFF whole-slide images into clean patch feature tensors ready for MIL training.

## Knowledge

### Supported WSI Formats
`.svs`, `.tif`, `.tiff`, `.ndpi`, `.mrxs`, `.vsi`, `.scn`

### Preprocessing Parameters
| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| patch_size | 256 | 128–1024 | Pixels at target magnification. 256px @ 20x is standard. |
| stride | 256 | 64–1024 | Equal to patch_size = non-overlapping (default). Smaller = overlapping. |
| magnification | 20.0 | 5, 10, 20, 40 | All Mahmood Lab backbones (UNI/CONCH/CTransPath) trained @ 20x. |
| otsu_level | 1 | 0–3 | Pyramid level for tissue mask. Level 1 (~4× downsample) is fast. Level 0 = pixel-accurate but slow. |
| min_tissue_pct | 0.5 | 0.0–1.0 | Min fraction of patch that must be tissue. 0.5 discards background-heavy patches. |
| preview_only | false | — | If true, processes only 3 slides for QC preview. |

### Disk Space Estimation
```
GB ≈ (n_slides × patches_per_slide × patch_size² × 3) / 1e9
Typical: 500 slides × 2000 patches × 256² × 3 ≈ 196 GB (patches only)
Feature .pt files: ~2 MB per slide (UNI dim=1024, ~1000 patches × 4 bytes × 1024)
```
Always warn user if estimated space exceeds 80% of free disk.

### Backbone Magnification Compatibility
All public backbones (UNI, CONCH, CTransPath, Virchow, GigaPath) were trained on **20x patches**. Using 40x without resizing reduces performance; using 5x or 10x is inappropriate for these models.

### QC Thresholds
- Slides with <5% tissue after segmentation → likely failed scan or background-only ROI
- Slides producing <100 patches → may be too small for reliable MIL
- Slides producing >10,000 patches → very large; TransMIL recommended over ABMIL

## Workflow
1. **Confirm dataset registered**: call `list_datasets` — must exist before preprocessing
2. **Estimate disk space** using formula above; warn if tight
3. **Launch full preprocessing** directly: `start_preprocessing` with `preview_only: false`.
   - Only run a 3-slide preview first (`preview_only: true`) when the user **explicitly asks** for one, OR when this is a brand-new untrusted dataset (a cohort we've never preprocessed before and tissue segmentation params are unvalidated). Do not default to preview for known TCGA cohorts.
4. **Monitor**: use `wait_for_job(preprocess)` (it blocks and emits live updates) instead of manually polling
5. **QC summary after completion**: report total patches, slides with <100 patches (flag them), mean patches/slide
6. **Feature extraction**: after patching, call `start_feature_extraction` with user's chosen backbone — this is a separate step that produces `.pt` files needed for training

## API Calls
```
# Step 1: Preview
start_preprocessing({ dataset_id: "brca", config: {
  patch_size: 256, stride: 256, magnification: 20,
  otsu_level: 1, min_tissue_pct: 0.5, preview_only: true
}})

# Step 2: Full run
start_preprocessing({ dataset_id: "brca", config: {
  patch_size: 256, stride: 256, magnification: 20,
  otsu_level: 1, min_tissue_pct: 0.5, preview_only: false
}})

# Step 3: Feature extraction (after patching)
start_feature_extraction({ dataset_id: "brca", backbone: "uni", batch_size: 256, device: "auto" })

# Monitor any job
get_job_status({ job_id: "prep-xxxxxxxx" })
```

## Error Recovery
- **OpenSlide error on .svs**: File may be corrupt or wrong magnification level. Ask user to verify the file opens in QuPath.
- **0 patches extracted**: `min_tissue_pct` too high or Otsu segmentation failed. Lower `otsu_level` to 0 for pixel-accurate mask, or reduce `min_tissue_pct` to 0.2.
- **CUDA OOM during feature extraction**: Reduce `batch_size` (try 64 or 32). Features can also run on CPU (`device: "cpu"`) though slower.
- **HF token required**: Gated backbones (UNI, CONCH, Virchow, GigaPath) need `HUGGINGFACE_TOKEN` set via `/api/config`. CTransPath works without a token.

## Guardrails
- NEVER skip the preview step — bad segmentation parameters waste hours of compute
- NEVER run feature extraction before patching is complete (no coords.json)
- NEVER recommend a backbone at a magnification it wasn't trained on without warning
- NEVER delete or modify original WSI files
