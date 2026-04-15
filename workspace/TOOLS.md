# PathClaw Tools Reference

All tools call the FastAPI backend. Use tool names — do NOT write curl commands.

## Datasets
- `list_datasets` — list all registered datasets (optional session_id filter)
- `register_dataset(name, path, description?)` — register a dataset from local folder or single WSI file
- `get_dataset_profile(dataset_id)` — quality report: class balance, slide counts, format distribution, label candidates

## GDC/TCGA
- `search_gdc(project, data_type?, experimental_strategy?, workflow_type?, access?)` — search GDC for files
  - data_type: `Slide Image`, `Masked Somatic Mutation`, `Clinical Supplement`, `Gene Expression Quantification`, `Copy Number Segment`, `Methylation Beta Value`
  - workflow_type: `MuTect2 Variant Aggregation and Masking`, `STAR - Counts`, etc.
- `download_gdc(file_ids, output_dir?, project?, gdc_token?, max_concurrent?)` — async download via detached subprocess (survives server restarts)
- `gdc_job_status(job_id)` — check download progress

## Preprocessing & Features
- `start_preprocessing(dataset_id, config)` — launch Otsu + patching job
  - config keys: patch_size, stride, magnification, otsu_level, min_tissue_pct, preview_only
- `start_feature_extraction(dataset_id, backbone, batch_size?, device?)` — extract .pt feature tensors
  - Runs as **detached subprocess** — server stays responsive during extraction
  - backbone: uni (1024-d) | conch (512-d) | ctranspath (768-d) | virchow (1280-d) | virchow2 (2560-d) | gigapath (1536-d)
  - Features saved to `features/{dataset_id}/{backbone}/{slide_stem}.pt` (backbone-namespaced)
  - Resume-safe: skips slides with existing .pt files
- `cancel_feature_job(job_id)` — SIGTERM the extraction worker

## Training
- `start_training(config)` — launch MIL or segmentation training job
  - MIL config: dataset_id, task, mil_method (abmil|meanpool|transmil|clam|dsmil|rrtmil|wikg), feature_backbone, feature_dim, embed_dim, num_classes, mammoth{...}, training{...}, evaluation{...}
  - Segmentation config: task_type="segmentation", seg_model (seg_unet|hovernet|cellpose), num_seg_classes, patch_size, batch_size
- `start_lora_finetuning(backbone, dataset_id, lora_rank?, epochs?, lr?)` — LoRA fine-tune a foundation model backbone
- `get_training_logs(job_id)` — tail training log file (last 5KB)

## Evaluation & Visualization
- `start_evaluation(experiment_id, dataset_id, split?)` — run evaluation on val/test split
- `get_eval_metrics(job_id)` — JSON metrics: AUROC, balanced_accuracy, confusion_matrix, classification_report
- `get_eval_plots(job_id)` — list plot filenames (ROC curves, confusion matrix)
- `generate_heatmap(experiment_id, dataset_id, slide_stem)` — attention heatmap overlay (JSON + thumbnail PNG)
- `compare_experiments(job_ids)` — side-by-side metric comparison across experiments

## Genomics
- `parse_genomic_file(file_path, file_type?, query?, sample_id?, limit?)` — parse MAF/VCF/XML/TSV, returns structured summary
- `query_mutations(genomic_dir, gene?, variant_class?, min_frequency?, output_format?)` — cohort-level mutation queries across MAF files
- `compute_tmb(maf_dir, exome_size_mb?, variant_classes?, thresholds?)` — compute TMB per sample with classification
- `extract_labels_from_genomic(genomic_dir, dataset_id, label_type, label_spec?, output_path?)` — extract slide-level labels from genomic data (MSI, mutations, TMB, clinical fields)
- `query_cbioportal(study_id, data_type?, gene_list?, clinical_attributes?)` — query cBioPortal for mutations, clinical data, CNA, MSI scores
- `run_survival_analysis(clinical_dir, dataset_id?, labels_path?, group_column?)` — Kaplan-Meier survival analysis with log-rank test, KM plot export
- `build_multi_omic_labels(dataset_id, sources, output_path?)` — merge MAF/clinical/predictions into unified matrix
- `generate_oncoplot(maf_dir, top_n?, output_path?, title?, min_frequency?)` — mutation landscape plot from MAF files (PNG + text summary)
- `parse_gene_expression(file_path, query?, gene_list?, limit?)` — parse STAR/HTSeq/FPKM expression files
- `biomarker_discovery(maf_dir, labels_path, analysis_type?, gene_list?)` — differential mutation analysis and attention-gene correlation

## WSI Viewer
- `slide_info(dataset_id, slide_stem)` — slide dimensions, levels, MPP
- `slide_thumbnail(dataset_id, slide_stem, size?)` — JPEG thumbnail
- DZI tiles served via OpenSeadragon at `/api/tiles/{dataset_id}/{slide_stem}/dzi_files/{level}/{col}_{row}.jpeg`
- Heatmap overlay tiles at `/api/tiles/{dataset_id}/{slide_stem}/heatmap/{experiment_id}/{level}/{col}_{row}.png`

## Status & Utilities
- `get_job_status(job_id)` — poll any job (preprocessing, feature, training, eval)
- `wait_for_job(job_id)` — block until job completes with live status updates
- `list_artifacts` — list all experiments with model availability
- `get_system_status` — GPU, storage, backend health
- `run_python(code)` — execute arbitrary Python (pandas, numpy, pathlib pre-imported)
- `search_literature(query, max_results?)` — search PubMed/Semantic Scholar for relevant papers
- `remember_fact(key, value)` / `recall_facts(query?)` — session memory
