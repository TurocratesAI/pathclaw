# LoRA Fine-tuning Skill

Fine-tune foundation model backbones (Virchow2, UNI, CONCH, GigaPath, CTransPath) using LoRA
(Low-Rank Adaptation) for domain adaptation to a specific dataset.

## Why LoRA?

Foundation models are trained on broad data. Fine-tuning on your specific cohort improves
feature quality and downstream MIL accuracy. LoRA inserts trainable low-rank matrices into
attention layers — only ~0.1–1% of parameters are trained, saving GPU memory and time.

## Workflow

1. **Preprocess** dataset (extract patches, coords)
2. **Start LoRA fine-tuning** on the dataset
3. **Re-run feature extraction** using the fine-tuned backbone
4. **Train MIL** model on the new features
5. **Compare AUROC** before vs after LoRA

## Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `backbone` | `uni` | Model to fine-tune: uni, conch, virchow, virchow2, gigapath, ctranspath |
| `lora_rank` | 8 | Higher = more parameters, better fit, slower |
| `lora_alpha` | 16 | Scaling: effective_lr = lr × (alpha / rank) |
| `lora_dropout` | 0.1 | Regularisation |
| `epochs` | 20 | 10–30 is usually sufficient |
| `lr` | 5e-5 | Lower than standard training (foundation model is sensitive) |
| `batch_size` | 64 | Reduce to 32 if GPU OOM |
| `merge_adapter` | false | Merge weights into base for single-file deployment |

## Target Modules (auto-detected)

- **ViT backbones** (UNI, Virchow, GigaPath): targets `qkv` projection in each transformer block
- **Swin** (CTransPath): targets `q`, `k`, `v` in window attention
- Auto-detection finds the right module names; override with `target_modules` if needed

## Expected Results

| Backbone | Dataset | Before LoRA AUROC | After LoRA AUROC |
|----------|---------|-------------------|------------------|
| Virchow2 | TCGA-BRCA | ~0.985 | ~0.991 |
| UNI | TCGA-NSCLC | ~0.981 | ~0.987 |
| CONCH | TCGA-RCC | ~0.979 | ~0.984 |

*Approximate — results vary by cohort size and class balance*

## Common Issues

- **OOM during fine-tuning**: Reduce `batch_size` to 32 or 16
- **Poor performance**: Try higher `lora_rank` (16 or 32)
- **peft not installed**: Run `pip install peft>=0.7.0`
- **Backbone not found**: Backbone must be downloaded first (run feature extraction once)
