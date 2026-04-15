"""LoRA fine-tuning of foundation model backbones (Virchow2, UNI, etc.) for PathClaw.

Uses HuggingFace `peft` to inject low-rank adapters into transformer attention
layers (q, k, v projections) and fine-tunes on patch-level classification using
slide-level labels propagated to all patches (weak supervision).

Workflow:
  1. Load backbone from the pre-downloaded HF snapshot
  2. Wrap with LoraConfig targeting attention QKV projections
  3. Fine-tune on patch images from the dataset using slide labels
  4. Save adapter weights (~10–50MB) — NOT the full model
  5. Optionally merge adapter into base for single-file inference
  6. Re-extract features using the fine-tuned model for downstream MIL training

Outputs saved to:
  ~/.pathclaw/experiments/{job_id}/
    lora_adapter/          — peft adapter weights + config
    metrics.json           — train/val loss + accuracy per epoch
    history.json           — full training history
    config.json            — training config
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()


# ---------------------------------------------------------------------------
# Dataset: patch-level images with slide-level labels (weakly supervised)
# ---------------------------------------------------------------------------

class _PatchDataset(torch.utils.data.Dataset):
    """Yields patch images with their slide-level label.

    Reads coordinates from preprocessing output and loads patches directly
    from WSI files via OpenSlide.
    """

    def __init__(
        self,
        dataset_id: str,
        slide_stems: list[str],
        label_map: dict[str, int],
        backbone_info: dict,
        max_patches_per_slide: int = 200,
    ) -> None:
        import json as _json

        self.backbone_info = backbone_info
        self.samples: list[tuple[str, dict, int]] = []  # (slide_path, coord, label)

        preprocessed_dir = PATHCLAW_DATA_DIR / "preprocessed" / dataset_id
        meta_path = PATHCLAW_DATA_DIR / "datasets" / dataset_id / "meta.json"

        slide_path_map: dict[str, str] = {}
        if meta_path.exists():
            meta = _json.loads(meta_path.read_text())
            for s in meta.get("slides", []):
                stem = Path(s["path"]).stem
                slide_path_map[stem] = s["path"]

        for stem in slide_stems:
            if stem not in label_map:
                continue
            label = label_map[stem]
            slide_path = slide_path_map.get(stem)
            if not slide_path or not Path(slide_path).exists():
                continue

            # Load coords
            coords_path = preprocessed_dir / stem / "coords.json"
            if not coords_path.exists():
                coords_path = preprocessed_dir / "patches" / f"{stem}_coords.json"
            if not coords_path.exists():
                continue

            raw = _json.loads(coords_path.read_text())
            coords = raw if isinstance(raw, list) else raw.get("patches", [])
            if not coords:
                continue

            # Sample patches uniformly (avoid loading all for large slides)
            step = max(1, len(coords) // max_patches_per_slide)
            for i in range(0, len(coords), step):
                self.samples.append((slide_path, coords[i], label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import numpy as np

        slide_path, coord, label = self.samples[idx]
        patch_px = self.backbone_info["patch_px"]

        try:
            import openslide
            slide = openslide.OpenSlide(slide_path)
            x, y = coord["x"], coord["y"]
            ps = coord.get("patch_size", 256)
            level = coord.get("level", 0)
            region = slide.read_region((x, y), level, (ps, ps)).convert("RGB")
            slide.close()
            if region.size != (patch_px, patch_px):
                region = region.resize((patch_px, patch_px))
            img = np.array(region, dtype=np.float32) / 255.0
        except Exception:
            img = np.ones((patch_px, patch_px, 3), dtype=np.float32) * 0.5

        # ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
        t = torch.from_numpy(img)
        t = (t - mean) / std
        return t.permute(2, 0, 1).float(), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# LoRA fine-tuning
# ---------------------------------------------------------------------------

def finetune_backbone_lora(config: dict, job_status: Optional[dict] = None) -> dict:
    """Fine-tune a foundation model backbone with LoRA adapters.

    Config keys:
      backbone (str)           — backbone name: uni, conch, virchow, virchow2, gigapath, ctranspath
      dataset_id (str)         — dataset with preprocessed patches
      label_column (str)       — label column name in labels.json
      job_id (str)             — experiment job identifier
      lora_rank (int)          — LoRA rank r (default 8)
      lora_alpha (int)         — LoRA scaling factor alpha (default 16)
      lora_dropout (float)     — dropout in LoRA layers (default 0.1)
      target_modules (list)    — attention modules to adapt (default ["qkv"])
      epochs (int)             — training epochs (default 20)
      lr (float)               — learning rate (default 5e-5)
      batch_size (int)         — patches per batch (default 64)
      max_patches_per_slide (int) — patches sampled per slide (default 200)
      device (str)             — auto | cuda | cpu
      merge_adapter (bool)     — merge LoRA weights into base after training (default False)
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError(
            "peft is not installed. Run: pip install peft>=0.7.0"
        )

    from pathclaw.preprocessing.feature_extraction import (
        BACKBONE_REGISTRY, _load_backbone, _get_transform,
    )

    backbone_name = config.get("backbone", "uni")
    dataset_id = config["dataset_id"]
    job_id = config.get("job_id", "lora_job")
    lora_rank = config.get("lora_rank", 8)
    lora_alpha = config.get("lora_alpha", 16)
    lora_dropout = config.get("lora_dropout", 0.1)
    epochs = config.get("epochs", 20)
    lr = config.get("lr", 5e-5)
    batch_size = config.get("batch_size", 64)
    max_patches = config.get("max_patches_per_slide", 200)
    merge_adapter = config.get("merge_adapter", False)

    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Available: {sorted(BACKBONE_REGISTRY)}"
        )

    info = BACKBONE_REGISTRY[backbone_name]
    exp_dir = PATHCLAW_DATA_DIR / "experiments" / job_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    device_str = config.get("device", "auto")
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) else
        ("cpu" if device_str == "auto" else device_str)
    )

    if job_status:
        job_status["status"] = "loading_backbone"

    # Load base model
    model = _load_backbone(backbone_name, device)
    model = model.float()  # LoRA requires fp32
    model.train()

    # Determine target modules (ViT uses "qkv" projection in each block)
    target_modules = config.get("target_modules", _infer_target_modules(model))
    logger.info(f"LoRA target modules: {target_modules}")

    # Inject LoRA adapters
    # LoRA for feature extraction = FEATURE_EXTRACTION task (no classification head)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    # Add a thin classification head for patch-level supervision
    num_classes = config.get("num_classes", 2)
    clf_head = nn.Linear(info["dim"], num_classes).to(device)

    # Load labels
    labels_path = PATHCLAW_DATA_DIR / "datasets" / dataset_id / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"No labels.json for dataset '{dataset_id}'. "
            f"Run a MIL training job first (it creates labels.json)."
        )
    label_map: dict[str, int] = json.loads(labels_path.read_text())

    # Build dataset
    slide_stems = list(label_map.keys())
    n = len(slide_stems)
    split = max(1, int(n * 0.8))
    train_stems = slide_stems[:split]
    val_stems = slide_stems[split:]

    if job_status:
        job_status["status"] = "building_dataset"

    train_ds = _PatchDataset(dataset_id, train_stems, label_map, info, max_patches)
    val_ds = _PatchDataset(dataset_id, val_stems, label_map, info, max_patches)

    if len(train_ds) == 0:
        raise FileNotFoundError(
            f"No training patches found for dataset '{dataset_id}'. "
            f"Ensure preprocessing has been run and slide files are accessible."
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(clf_head.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    adapter_path = exp_dir / "lora_adapter"

    for epoch in range(epochs):
        model.train(); clf_head.train()
        train_losses, train_correct, train_total = [], 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                feats = model(imgs)  # (B, D)
                logits = clf_head(feats)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += len(labels)

        scheduler.step()
        avg_train = sum(train_losses) / max(len(train_losses), 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation
        model.eval(); clf_head.eval()
        val_losses, val_correct, val_total = [], 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = model(imgs)
                logits = clf_head(feats)
                loss = F.cross_entropy(logits, labels)
                val_losses.append(loss.item())
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)

        avg_val = sum(val_losses) / max(len(val_losses), 1)
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            # Save best adapter weights
            model.save_pretrained(str(adapter_path))
            torch.save(clf_head.state_dict(), exp_dir / "clf_head.pth")

        if job_status:
            job_status["epoch"] = epoch + 1
            job_status["progress"] = (epoch + 1) / epochs
            job_status["metrics"] = {
                "train_loss": avg_train,
                "val_loss": avg_val,
                "val_accuracy": val_acc,
            }

        logger.info(
            f"Epoch {epoch + 1}/{epochs} — "
            f"train_loss: {avg_train:.4f}  val_loss: {avg_val:.4f}  "
            f"val_acc: {val_acc:.4f}"
        )

    # Merge adapter if requested
    if merge_adapter:
        logger.info("Merging LoRA adapter into base model …")
        merged = model.merge_and_unload()
        merged_path = exp_dir / "merged_model.pth"
        torch.save(merged.state_dict(), merged_path)
        logger.info(f"Merged model saved to {merged_path}")

    metrics = {
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": history["val_acc"][-1] if history["val_acc"] else 0.0,
        "epochs_trained": epochs,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "backbone": backbone_name,
        "adapter_path": str(adapter_path),
    }

    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (exp_dir / "history.json").write_text(json.dumps(history, indent=2))

    return {"metrics": metrics, "history": history, "experiment_dir": str(exp_dir)}


def _infer_target_modules(model: nn.Module) -> list[str]:
    """Detect which module names are suitable LoRA targets (QKV attention projections)."""
    candidates = {"qkv", "q_proj", "k_proj", "v_proj", "query", "key", "value",
                  "to_q", "to_k", "to_v", "in_proj"}
    found = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            found.add(leaf)
    if not found:
        # Fallback: target all Linear layers in attention blocks
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "attn" in name.lower():
                found.add(name.split(".")[-1])
    return list(found) if found else ["qkv"]


def load_lora_backbone(
    backbone_name: str,
    adapter_path: str,
    device: torch.device,
) -> nn.Module:
    """Load a backbone with a saved LoRA adapter for inference / re-extraction."""
    from peft import PeftModel
    from pathclaw.preprocessing.feature_extraction import _load_backbone

    base_model = _load_backbone(backbone_name, device).float()
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model
