# PathClaw v0.3 — Clinical Use Cases & Benchmarks

12 clinically meaningful use cases covering MIL classification, segmentation, LoRA fine-tuning,
and the GDC data acquisition pipeline.

---

## MIL Classification

### 1. Camelyon17 — Lymph Node Metastasis Detection (binary MIL)
- **Task**: Classify whole-slide images as tumor-positive or tumor-negative (sentinel lymph nodes)
- **Data**: Camelyon17 challenge (5 medical centers, 500 WSIs)
- **Setup**: ABMIL + UNI backbone, holdout evaluation
- **Expected AUROC**: ~0.975
- **Interpretation**: High attention scores → suspect metastatic regions; view in Viewer with heatmap overlay
- **Agent prompt**: *"Train an ABMIL model on my Camelyon17 dataset using UNI features to detect lymph node metastasis"*

### 2. TCGA-BRCA — Breast Cancer Subtyping (binary MIL)
- **Task**: IDC (invasive ductal carcinoma) vs ILC (invasive lobular carcinoma) classification
- **Data**: TCGA-BRCA, open-access diagnostic slides (n≈1000)
- **Setup**: TransMIL + MAMMOTH, 5-fold-cv, feature_backbone=uni
- **Expected AUROC**: ~0.988 (MAMMOTH), ~0.981 (baseline ABMIL+UNI)
- **MAMMOTH note**: Use num_experts=30, num_slots=10 for datasets >200 slides
- **Agent prompt**: *"Search TCGA for BRCA diagnostic slides, download them, and train TransMIL with MAMMOTH for IDC vs ILC subtyping"*

### 3. TCGA-NSCLC — Lung Cancer Subtyping (binary MIL)
- **Task**: LUAD (adenocarcinoma) vs LUSC (squamous cell carcinoma)
- **Data**: TCGA-LUAD + TCGA-LUSC (n≈1000 combined)
- **Setup**: CLAM-SB + UNI, holdout
- **Expected AUROC**: ~0.982
- **Reference**: CLAM benchmark (Lu et al., Nature BME 2021): 0.964 AUC (ResNet features); UNI substantially improves this
- **Agent prompt**: *"Download TCGA-LUAD and TCGA-LUSC slides and run CLAM subtyping with UNI features"*

### 4. TCGA-RCC — Kidney Cancer Subtyping (3-class MIL)
- **Task**: CCRCC vs CHRCC vs PRCC (clear cell, chromophobe, papillary)
- **Data**: TCGA-KIRC + TCGA-KICH + TCGA-KIRP
- **Setup**: WIKG + UNI, 5-fold-cv, num_classes=3
- **Expected AUROC (macro OvR)**: ~0.996
- **Agent prompt**: *"Run 3-class kidney cancer subtyping on TCGA-RCC using WIKG and UNI"*

### 5. EGFR Mutation Prediction in LUAD (molecular biomarker MIL)
- **Task**: Predict EGFR mutation status from H&E slides (LUAD only)
- **Data**: TCGA-LUAD with mutation annotation (EGFR column from MAF file)
- **Setup**: RRTMIL + Virchow2, holdout, MAMMOTH enabled
- **Expected AUROC**: ~0.72–0.78 (harder task — molecular phenotype from morphology)
- **Note**: Download both Slide Image AND Masked Somatic Mutation files from GDC for same cases
- **Agent prompt**: *"Train RRTMIL with Virchow2 features to predict EGFR mutation status in LUAD from TCGA"*

### 5b. TCGA-UCEC — MSI Prediction in Endometrioid Carcinoma (priority use case)
- **Task**: Predict microsatellite instability (MSI-H vs MSS) from H&E slides in endometrioid endometrial cancer
- **Data**: TCGA-UCEC (n≈530 WSIs, open-access diagnostic slides); MSI labels from GDC clinical XML (`msi_status` field) or MANTIS scores
- **Setup**: ABMIL + UNI, holdout; optionally MAMMOTH (num_experts=30 for n>200)
- **Expected AUROC**: ~0.76–0.82 (MSI is a morphology-accessible biomarker in endometrial cancer — higher than EGFR in LUAD)
- **MAF note**: GDC also provides Masked Somatic Mutation MAF files for UCEC; high TMB in MSI-H cases is confirmatory but not needed for the H&E task
- **Data acquisition**:
  1. `search_gdc(project="TCGA-UCEC", data_type="Slide Image", access="open")`
  2. `search_gdc(project="TCGA-UCEC", data_type="Clinical Supplement")` — contains MSI status
  3. (Optional) `search_gdc(project="TCGA-UCEC", data_type="Masked Somatic Mutation")` — MAF files
- **Label column**: `msi_status` → map `"MSI-H"→1, "MSS"→0, "MSI-L"→0`
- **Agent prompt**: *"Download TCGA-UCEC slides and clinical data from GDC, then train ABMIL with UNI to predict MSI status in endometrioid carcinoma"*

---

## Segmentation

### 6. Camelyon17 — Tumor Region Segmentation (semantic)
- **Task**: Pixel-level binary classification of tumor vs non-tumor tissue
- **Data**: Camelyon17 slides + annotation masks (XML → binary PNG conversion required)
- **Setup**: SegUNet, num_seg_classes=2, epochs=50, batch_size=8
- **Expected mean IoU**: ~0.81
- **Mask format**: Binary PNG, 0=background, 1=tumor; at `datasets/{id}/masks/{slide_stem}/{patch_idx:06d}.png`
- **Agent prompt**: *"Train a UNet segmentation model on Camelyon17 for tumor region detection"*

### 7. PanNuke — Multi-tissue Nucleus Segmentation (instance)
- **Task**: Instance segmentation of 5 nuclei types (neoplastic, inflammatory, connective, epithelial, dead)
- **Data**: PanNuke dataset (7901 image patches, 19 tissue types)
- **Setup**: HoVer-Net, num_seg_classes=6, epochs=100, batch_size=4
- **Expected mPQ**: ~0.65 (panoptic quality)
- **Mask format**: Instance mask PNG (each unique nonzero value = one nucleus), stored per patch
- **Agent prompt**: *"Train HoVer-Net on PanNuke for multi-class nucleus instance segmentation"*

### 8. CoNSeP — Colorectal Nuclei Segmentation (instance)
- **Task**: 4-class nuclei segmentation (epithelial, inflammatory, spindle-shaped, miscellaneous)
- **Data**: CoNSeP dataset (41 image tiles from colorectal adenocarcinoma)
- **Setup**: HoVer-Net, num_seg_classes=5, pretrained_encoder=True
- **Expected Dice**: ~0.71 (multi-class mean)
- **Agent prompt**: *"Run HoVer-Net training on CoNSeP colorectal nucleus segmentation dataset"*

### 9. Rapid Cell Detection (Cellpose, no annotations)
- **Task**: Detect and outline individual cells in histology patches without training
- **Data**: Any preprocessed dataset with patches
- **Setup**: `seg_model=cellpose`, `cellpose_model_type=cyto3`, no masks needed
- **Expected mAP@50**: ~0.72 on H&E tissue patches (pre-trained)
- **Agent prompt**: *"Run Cellpose on my BRCA dataset patches to detect cells without needing annotation masks"*

---

## LoRA Fine-tuning

### 10. Virchow2 LoRA on TCGA-BRCA
- **Task**: Adapt Virchow2 to BRCA-specific features, then retrain ABMIL
- **Setup**: LoRA rank=8, alpha=16, epochs=20, lr=5e-5 → re-extract → ABMIL
- **Expected improvement**: AUROC +0.003–0.008 over standard Virchow2 features
- **Why it works**: Virchow2 was trained on general pathology; LoRA adapts attention layers to BRCA-specific texture patterns
- **Agent prompt**: *"Fine-tune Virchow2 with LoRA on my BRCA dataset, then re-extract features and retrain ABMIL"*

### 11. UNI LoRA on TCGA-NSCLC
- **Task**: Adapt UNI to lung cancer tissue for LUAD vs LUSC subtyping
- **Setup**: LoRA rank=16, epochs=15 (larger rank for more challenging task)
- **Expected improvement**: AUROC +0.004–0.010 over baseline UNI
- **Agent prompt**: *"Use LoRA to fine-tune the UNI backbone on NSCLC slides before retraining my CLAM classifier"*

---

## End-to-End Pipeline

### 12. GDC Acquisition → Full Pipeline (TCGA-BRCA IDC vs ILC)
Complete autonomous workflow from data download to final evaluation:

```
1. search_gdc(project="TCGA-BRCA", data_type="Slide Image", access="open", limit=100)
2. download_gdc(file_ids=[...])
3. register_dataset(name="TCGA-BRCA", path="~/.pathclaw/raw/TCGA-BRCA/")
4. start_preprocessing(dataset_id="...", patch_size=256, magnification=20.0)
5. start_feature_extraction(dataset_id="...", backbone="uni")
6. start_training(task="IDC vs ILC", dataset_id="...", label_column="histological_type",
                  mil_method="transmil", mammoth_enabled=True)
7. start_evaluation(model_path="...", dataset_id="...")
8. generate_heatmap(experiment_id="...", dataset_id="...", slide_stem="...")
```

Expected wall time (A100 GPU): ~45 min total for 100 slides

---

## LLM Provider Comparison (same BRCA workflow)

| Provider | Model | Chat | Tools | Streaming | Latency (first token) |
|----------|-------|------|-------|-----------|----------------------|
| Ollama | qwen3:8b | ✓ | ✓ | ✓ | ~0.3s |
| Anthropic | claude-sonnet-4-20250514 | ✓ | ✓ | ✓ | ~0.8s |
| OpenAI | gpt-4o | ✓ | ✓ | ✓ | ~0.9s |
| Google | gemini-2.5-flash | ✓ | ✓ | ✓ | ~0.6s |

Configure via Settings (onboarding modal) or ask: *"Switch to Claude Sonnet"*
