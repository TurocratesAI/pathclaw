"""Patch-level trainer — trains a PatchMLP on per-patch labels.

Input labels come from ~/.pathclaw/datasets/<dataset_id>/patch_labels.csv with
columns `slide_id,patch_idx,label` (label is an integer class id). Features are
read from the same .pt caches used by the MIL trainer (one file per slide,
shape (N, D)).

Scope for v1: classification only, holdout split by slide (not by patch — so
the same slide never appears in both train and val).
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from .models import PATCH_MODEL_REGISTRY

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
logger = logging.getLogger("pathclaw.patch_training")


def _load_patch_labels(dataset_id: str) -> dict[str, dict[int, int]]:
    """Read patch_labels.csv → {slide_id: {patch_idx: label}}."""
    csv_path = PATHCLAW_DATA_DIR / "datasets" / dataset_id / "patch_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No patch_labels.csv at {csv_path}. "
            f"Create one with columns 'slide_id,patch_idx,label' (label = integer class id)."
        )
    out: dict[str, dict[int, int]] = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        required = {"slide_id", "patch_idx", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"patch_labels.csv must have columns {sorted(required)}, got {reader.fieldnames}")
        for row in reader:
            sid = row["slide_id"].strip()
            try:
                idx = int(row["patch_idx"])
                lbl = int(row["label"])
            except ValueError:
                continue
            out.setdefault(sid, {})[idx] = lbl
    return out


def _gather_patches(
    feature_files: list[Path],
    labels_by_slide: dict[str, dict[int, int]],
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, int]]]:
    """Concatenate features and labels into flat tensors. Returns (X, y, origins)."""
    feats: list[torch.Tensor] = []
    labs: list[int] = []
    origins: list[tuple[str, int]] = []
    for f in feature_files:
        sid = f.stem
        if sid not in labels_by_slide:
            continue
        labmap = labels_by_slide[sid]
        try:
            tensor = torch.load(f, map_location="cpu", weights_only=True)
        except Exception as e:
            logger.warning(f"Skipping {f.name}: {e}")
            continue
        if tensor.dim() != 2:
            logger.warning(f"Skipping {f.name}: expected (N, D), got {tuple(tensor.shape)}")
            continue
        for idx, lbl in labmap.items():
            if 0 <= idx < tensor.shape[0]:
                feats.append(tensor[idx].to(dtype))
                labs.append(lbl)
                origins.append((sid, idx))
    if not feats:
        raise ValueError("No (slide, patch) pairs matched patch_labels.csv with available features.")
    X = torch.stack(feats)
    y = torch.tensor(labs, dtype=torch.long)
    return X, y, origins


def train_patch_model(config: dict, job_status: Optional[dict] = None) -> dict:
    """Train a patch-level model (currently: PatchMLP)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_id = config["dataset_id"]
    backbone = config.get("feature_backbone", config.get("backbone", "uni"))

    from pathclaw.preprocessing.feature_extraction import resolve_features_dir
    features_dir = resolve_features_dir(dataset_id, backbone)
    if not features_dir.exists():
        raise FileNotFoundError(f"Features not found at {features_dir}. Run feature extraction first.")
    feature_files = sorted(features_dir.glob("*.pt"))
    if not feature_files:
        raise FileNotFoundError(f"No .pt feature files in {features_dir}")

    labels_by_slide = _load_patch_labels(dataset_id)
    if job_status is not None:
        job_status["message"] = f"Loaded patch labels for {len(labels_by_slide)} slides"

    # Hold out whole slides for val so no patch leak
    slide_ids = [f.stem for f in feature_files if f.stem in labels_by_slide]
    if len(slide_ids) < 2:
        raise ValueError(f"Need at least 2 labeled slides for a holdout split, got {len(slide_ids)}.")
    val_frac = float(config.get("evaluation", {}).get("val_frac", 0.2))
    train_slides, val_slides = train_test_split(slide_ids, test_size=val_frac, random_state=42)

    def _subset(files):
        return [f for f in feature_files if f.stem in set(files)]

    X_train, y_train, _ = _gather_patches(_subset(train_slides), labels_by_slide)
    X_val, y_val, val_origins = _gather_patches(_subset(val_slides), labels_by_slide)

    num_classes = int(config.get("num_classes", int(max(int(y_train.max()), int(y_val.max())) + 1)))
    in_dim = int(config.get("feature_dim", X_train.shape[1]))
    if X_train.shape[1] != in_dim:
        logger.warning(f"feature_dim mismatch: config says {in_dim}, features have {X_train.shape[1]}. Using actual.")
        in_dim = X_train.shape[1]

    model_name = config.get("patch_model", "patch_mlp").lower()
    model_cls = PATCH_MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"Unknown patch model {model_name!r}. Available: {sorted(PATCH_MODEL_REGISTRY)}")
    model = model_cls(
        in_dim=in_dim,
        embed_dim=int(config.get("embed_dim", 512)),
        num_classes=num_classes,
        dropout=float(config.get("dropout", 0.1)),
        num_layers=int(config.get("num_layers", 2)),
    ).to(device)

    train_cfg = config.get("training", {})
    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 512))
    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 1e-5))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    # Class-balance sanity
    class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
    logger.info(f"Patch trainer: {len(X_train)} train, {len(X_val)} val; classes={class_counts.tolist()}")

    history: list[dict] = []
    X_val_dev = X_val.to(device).float()
    y_val_dev = y_val.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train))
        running_loss = 0.0
        running_correct = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train[idx].to(device).float()
            yb = y_train[idx].to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += float(loss.item()) * len(idx)
            running_correct += int((logits.argmax(dim=-1) == yb).sum().item())
        train_loss = running_loss / len(perm)
        train_acc = running_correct / len(perm)

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_val_dev)
            val_loss = float(criterion(logits, y_val_dev).item())
            val_preds = logits.argmax(dim=-1)
            val_acc = float((val_preds == y_val_dev).float().mean().item())
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        if job_status is not None:
            job_status["progress"] = epoch / epochs
            job_status["metrics"] = {"epoch": epoch, "val_acc": val_acc, "val_loss": val_loss}
        logger.info(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f} acc={train_acc:.3f}  val_loss={val_loss:.4f} acc={val_acc:.3f}")

    # Final predictions + artifacts
    job_id = config.get("_job_id") or f"patch-{os.urandom(4).hex()}"
    out_dir = PATHCLAW_DATA_DIR / "experiments" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_dev).float().cpu()
    val_probs = torch.softmax(val_logits, dim=-1).numpy()
    val_pred_int = val_probs.argmax(axis=1)

    preds_path = out_dir / "predictions.csv"
    with preds_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        header = ["slide_id", "patch_idx", "y_true", "y_pred"] + [f"prob_{i}" for i in range(num_classes)]
        w.writerow(header)
        for (sid, pidx), y_t, y_p, probs in zip(val_origins, y_val.numpy(), val_pred_int, val_probs):
            w.writerow([sid, pidx, int(y_t), int(y_p)] + [float(p) for p in probs])

    # Confusion matrix plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for yt, yp in zip(y_val.numpy(), val_pred_int):
            cm[int(yt), int(yp)] += 1
        fig, ax = plt.subplots(figsize=(4 + num_classes * 0.5, 4 + num_classes * 0.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Patch confusion matrix")
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(plots_dir / "patch_confusion.png", dpi=120)
        plt.close(fig)

        # Loss/acc curves
        epochs_x = [h["epoch"] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(epochs_x, [h["train_loss"] for h in history], label="train")
        ax1.plot(epochs_x, [h["val_loss"] for h in history], label="val")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.set_title("Loss")
        ax2.plot(epochs_x, [h["train_acc"] for h in history], label="train")
        ax2.plot(epochs_x, [h["val_acc"] for h in history], label="val")
        ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.legend(); ax2.set_title("Accuracy")
        fig.tight_layout()
        fig.savefig(plots_dir / "loss_curves.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    metrics = {
        "best_val_acc": max(h["val_acc"] for h in history),
        "final_val_acc": history[-1]["val_acc"],
        "final_val_loss": history[-1]["val_loss"],
        "n_train_patches": int(len(X_train)),
        "n_val_patches": int(len(X_val)),
        "num_classes": num_classes,
        "class_counts_train": class_counts.tolist(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")

    return {"metrics": metrics, "history": history, "out_dir": str(out_dir)}
