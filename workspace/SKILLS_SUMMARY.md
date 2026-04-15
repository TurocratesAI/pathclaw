# PathClaw Skills Summary

Always-loaded condensed reference. Full skill details are injected per-turn based on user message keywords.

## Skill Routing Table

| Skill | Domain | Trigger Keywords | Primary Tools |
|-------|--------|-----------------|---------------|
| `dataset-intake` | Data ingestion | upload, register, scan, folder, path, slides | `register_dataset`, `list_datasets` |
| `gdc-tcga` | TCGA data download | tcga, gdc, brca, luad, lusc, rcc, stad, ucec, blca, skcm, prad, crc, coad, read, diagnostic, maf, dx, download, cohort, clinical | `search_gdc`, `download_gdc`, `register_dataset` |
| `data-profiling` | Dataset QA | profile, quality, class balance, distribution, statistics, ready | `get_dataset_profile`, `list_datasets` |
| `data-cleaning` | Label harmonization | clean, standardize, labels, harmonize, mapping, normalize | `get_dataset_profile`, `register_dataset` |
| `data-lifecycle` | Storage management | storage, disk, cleanup, delete, space, retention | `get_system_status` |
| `wsi-preprocess` | Patch extraction | preprocess, patch, segment, otsu, tissue, magnification, tile | `start_preprocessing`, `get_job_status`, `start_feature_extraction` |
| `train-config` | Config design | configure, config, mammoth, backbone, mil method, abmil, epochs, hyperparameter | `list_datasets`, `start_training` |
| `train-exec` | Job execution | train, launch, monitor, progress, loss, val accuracy, checkpoint, nan | `start_training`, `get_job_status`, `get_training_logs` |
| `evaluation` | Metrics computation | evaluate, metrics, auroc, accuracy, confusion, roc, inference | `start_evaluation`, `get_eval_metrics`, `get_eval_plots` |
| `results` | Interpretation | results, performance, compare, recommend, next, why, improve, baseline | `get_eval_metrics`, `list_artifacts` |
| `segmentation` | Seg training | segment, nuclei, cell, mask, iou, dice, hovernet, unet, cellpose, instance | `start_training` (task_type=segmentation) |
| `lora-finetune` | LoRA fine-tuning | lora, fine-tune, adapter, peft, domain adapt, low-rank | `start_lora_finetuning` |
| `genomic-analysis` | Genomic file parsing | maf, vcf, mutation, variant, tmb, somatic, parse maf, oncoplot | `parse_genomic_file`, `query_mutations`, `compute_tmb` |
| `label-engineering` | Label extraction | extract label, msi label, mutation label, labels.csv, barcode, clinical field | `extract_labels_from_genomic`, `parse_genomic_file` |
| `survival-biomarker` | Survival analysis | survival, kaplan, meier, cox, hazard, prognosis, biomarker, log-rank | `run_survival_analysis`, `query_cbioportal`, `build_multi_omic_labels` |

## Pipeline Order
```
dataset-intake  →  (gdc-tcga for downloads)
genomic-analysis → label-engineering  (parse genomic files, then extract labels)
data-profiling  →  data-cleaning  (if issues found)
wsi-preprocess  →  feature extraction (same skill, for MIL)
train-config    →  train-exec
evaluation      →  results  →  survival-biomarker (optional)

Segmentation path:
dataset-intake  →  (add masks to datasets/{id}/masks/)
segmentation skill → start_training(task_type="segmentation", seg_model="seg_unet"|"hovernet"|"cellpose")
```

## Key Facts (always remember)
- All backbones expect **20x magnification** patches
- `feature_dim` must exactly match backbone: uni=1024, conch=512, ctranspath=768, virchow=1280, virchow2=2560, gigapath=1536
- MAMMOTH improves **130/152 configurations** by avg +3.8% — recommend for ≥30 slides
- Skip MAMMOTH for <30 slides; use num_experts=15 for 30–100 slides; 30 for >100
- Evaluation strategy: "holdout" for >200 slides; "5-fold-cv" for <200
- TCGA BRCA subtyping baseline: AUROC ~0.981 (ABMIL+UNI); with MAMMOTH: 0.988
- Gated backbones (UNI, CONCH, Virchow, GigaPath) need HUGGINGFACE_TOKEN set via /api/config
