"""Evaluation and metrics computation."""

from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
logger = logging.getLogger("pathclaw.evaluation")


def evaluate_model(config: dict, job_status: Optional[dict] = None) -> dict:
    """Evaluate a trained MIL model on a test set."""
    import torch
    from pathclaw.training.trainer import create_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = config.get("model_path", "")
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load the config from the experiment directory
    exp_dir = Path(model_path).parent
    exp_config_path = exp_dir / "config.json"
    if exp_config_path.exists():
        train_config = json.loads(exp_config_path.read_text())
    else:
        train_config = config
    
    # Create and load model
    model = create_model(train_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Load test features — filter by split if split.json exists
    dataset_id = config.get("dataset_id", train_config.get("dataset_id"))
    backbone = train_config.get("feature_backbone", config.get("feature_backbone", "uni"))
    from pathclaw.preprocessing.feature_extraction import resolve_features_dir
    features_dir = resolve_features_dir(dataset_id, backbone)
    all_feature_files = sorted(features_dir.glob("*.pt"))

    split_file = exp_dir / "split.json"
    requested_split = config.get("split", "val")
    if split_file.exists():
        split_info = json.loads(split_file.read_text())
        split_stems = set(split_info.get(requested_split, split_info.get("val", [])))
        feature_files = [f for f in all_feature_files if f.stem in split_stems]
        if not feature_files:
            logger.warning(
                f"Split '{requested_split}' not found in split.json — evaluating on all features."
            )
            feature_files = all_feature_files
        else:
            logger.info(f"Evaluating on '{requested_split}' split: {len(feature_files)} slides")
    else:
        logger.warning("No split.json found — evaluating on all features (may include training data).")
        feature_files = all_feature_files
    
    # Load labels
    labels_path = PATHCLAW_DATA_DIR / "datasets" / dataset_id / "labels.json"
    if labels_path.exists():
        label_map = json.loads(labels_path.read_text())
    else:
        raise FileNotFoundError("labels.json required for evaluation")
    
    # Run inference
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for i, f in enumerate(feature_files):
            features = torch.load(f, map_location=device, weights_only=True)
            if features.dim() == 2:
                features = features.unsqueeze(0)
            
            label = label_map.get(f.stem, 0)
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            
            all_preds.append(pred)
            all_targets.append(label)
            all_probs.append(probs.cpu().numpy().flatten())
            
            if job_status:
                job_status["progress"] = (i + 1) / len(feature_files)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        confusion_matrix, classification_report,
    )
    
    metrics = {
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(all_targets, all_preds)),
        "num_samples": len(all_targets),
        "num_classes": len(np.unique(all_targets)),
    }
    
    # AUROC (if binary or multi-class with probabilities)
    try:
        from sklearn.metrics import roc_auc_score
        if metrics["num_classes"] == 2:
            metrics["auroc"] = float(roc_auc_score(all_targets, all_probs[:, 1]))
        else:
            metrics["auroc"] = float(roc_auc_score(
                all_targets, all_probs, multi_class="ovr", average="macro"
            ))
    except Exception as e:
        logger.warning(f"Could not compute AUROC: {e}")
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Classification report
    report = classification_report(all_targets, all_preds, output_dict=True)
    metrics["classification_report"] = report
    
    # Save metrics and generate plots
    job_id = job_status["job_id"] if job_status else "manual-eval"
    eval_dir = PATHCLAW_DATA_DIR / "experiments" / job_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    
    _save_eval_plots(metrics, all_targets, all_preds, all_probs, eval_dir)
    
    return {"metrics": metrics}


def _save_eval_plots(
    metrics: dict,
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
):
    """Generate evaluation visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Acc: {metrics['accuracy']:.3f})")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    
    # ROC curve — binary or one-vs-rest multi-class
    if "auroc" in metrics:
        from sklearn.metrics import roc_curve, auc
        n_cls = metrics.get("num_classes", 2)
        COLORS = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
                  "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#a78bfa"]
        fig, ax = plt.subplots(figsize=(8, 6))

        if n_cls == 2:
            fpr, tpr, _ = roc_curve(targets, probs[:, 1])
            ax.plot(fpr, tpr, color=COLORS[0], linewidth=2,
                    label=f"AUROC = {metrics['auroc']:.3f}")
        else:
            # One-vs-rest curve per class
            for cls_i in range(n_cls):
                fpr, tpr, _ = roc_curve((targets == cls_i).astype(int), probs[:, cls_i])
                cls_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=COLORS[cls_i % len(COLORS)], linewidth=2,
                        label=f"Class {cls_i} (AUC={cls_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve" + (" (OvR)" if n_cls > 2 else ""))
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "roc_curve.png", dpi=150)
        plt.close()
