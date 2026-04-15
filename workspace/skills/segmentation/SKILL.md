# Segmentation Skill

PathClaw supports three segmentation model types:

## Models

| Model | Type | Description |
|-------|------|-------------|
| `seg_unet` | Semantic | Lightweight UNet trained from scratch. Outputs per-pixel class labels. |
| `hovernet` | Instance | HoVer-Net with ResNet-50 encoder. Predicts nuclear pixel map + H/V distance maps for instance separation. |
| `cellpose` | Instance (inference only) | Pre-trained Cellpose — no training needed. Runs directly on patches. |

## Data Requirements

Segmentation requires **ground truth mask PNG files**:

```
~/.pathclaw/datasets/{dataset_id}/masks/
  {slide_stem}/
    000000.png    # patch index 0 mask
    000001.png    # patch index 1 mask
    ...
```

**Mask format**:
- Semantic (UNet): grayscale PNG, pixel value = class index (0 = background, 1 = class 1, ...)
- Instance (HoVer-Net): grayscale PNG, each unique non-zero value = one nucleus instance
- Cellpose: uses same instance mask format for evaluation

Patch indices match the order in the preprocessing coords.json file.

## Training Configuration

```
Task type: segmentation
Model options: seg_unet, hovernet, cellpose
Number of seg classes: 2 (binary) or more for multi-class
Epochs: 50 (recommended, fewer for quick tests)
Batch size: 8 (reduce to 4 if GPU OOM)
```

## Metrics

- **IoU (Intersection over Union)**: per class and mean
- **Dice coefficient**: per class and mean
- **Pixel accuracy**: fraction of correctly classified pixels
- **Average Precision @0.5 IoU** (Cellpose only)

## Typical Workflows

### Tumor region segmentation (Camelyon17)
1. Register dataset with Camelyon17 slides + binary tumor masks
2. Run: `start_training(task_type="segmentation", seg_model="seg_unet", num_seg_classes=2)`
3. View side-by-side predictions in Workspace > Plots

### Nuclei instance segmentation (PanNuke)
1. Register dataset with PanNuke slides + instance masks
2. Run: `start_training(task_type="segmentation", seg_model="hovernet", num_seg_classes=6)`

### Quick cell detection (no annotations needed)
1. Register dataset
2. Run Cellpose: `start_training(task_type="segmentation", seg_model="cellpose")`
   - No mask files needed — Cellpose is pre-trained
   - Reports Average Precision vs provided masks if available

## Common Issues

- **No mask files found**: Masks must be in the exact path format above
- **GPU OOM with HoVer-Net**: Reduce batch_size to 4 or 2; HoVer-Net has ResNet-50 backbone
- **Low IoU**: Check that mask class indices match your num_seg_classes exactly
- **Cellpose missing**: Run `pip install cellpose` in the backend environment
