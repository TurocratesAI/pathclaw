"""MIL training engine — all advertised config fields implemented."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split

from .models import MODEL_REGISTRY

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()

logger = logging.getLogger("pathclaw.training")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(config: dict) -> nn.Module:
    """Create a MIL model from a training config dict."""
    mil_method = config.get("mil_method", "abmil").lower()
    model_cls = MODEL_REGISTRY.get(mil_method)
    if model_cls is None:
        raise ValueError(
            f"Unknown MIL method: '{mil_method}'. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    mammoth_cfg = config.get("mammoth", {})
    moe_args = mammoth_cfg if mammoth_cfg.get("enabled", False) else None
    plugins = config.get("plugins") or []

    kwargs: dict = dict(
        in_dim=config.get("feature_dim", 1024),
        embed_dim=config.get("embed_dim", 512),
        num_classes=config.get("num_classes", 2),
        moe_args=moe_args,
        plugins=plugins,
    )
    # ABMIL / CLAM accept attn_dim
    if mil_method in ("abmil", "clam"):
        kwargs["attn_dim"] = config.get("attn_dim", 128)

    return model_cls(**kwargs)


# ---------------------------------------------------------------------------
# Optimizer & scheduler factories
# ---------------------------------------------------------------------------

def _build_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    lr = train_cfg.get("lr", 1e-4)
    wd = train_cfg.get("weight_decay", 1e-5)
    name = train_cfg.get("optimizer", "adam").lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif name == "radam":
        return torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: '{name}'. Choose from: adam, adamw, sgd, radam")


def _build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: dict, epochs: int
) -> Optional[object]:
    name = train_cfg.get("scheduler", "cosine").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, mode="max"
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: '{name}'. Choose from: cosine, step, plateau, none")


# ---------------------------------------------------------------------------
# Training loop for a single train/val split
# ---------------------------------------------------------------------------

def _train_one_split(
    config: dict,
    train_files: list,
    val_files: list,
    label_map: dict,
    exp_dir: Path,
    job_status: Optional[dict] = None,
    fold: Optional[int] = None,
) -> dict:
    device_cfg = config.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    try:
        model = create_model(config).to(device)
    except RuntimeError as e:
        if "CUDA" in str(e) and device.type == "cuda":
            logger.warning(f"CUDA error ({e}), falling back to CPU.")
            device = torch.device("cpu")
            model = create_model(config).to(device)
        else:
            raise
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"{'[Fold %d] ' % fold if fold is not None else ''}"
        f"Model: {config.get('mil_method', 'abmil')} | "
        f"MAMMOTH: {config.get('mammoth', {}).get('enabled', False)} | "
        f"Params: {n_params:,} | Device: {device}"
    )

    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 100)
    patience = train_cfg.get("early_stopping_patience", 0)  # 0 = disabled

    optimizer = _build_optimizer(model, train_cfg)
    scheduler = _build_scheduler(optimizer, train_cfg, epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_predictions: list[dict] = []  # filled when a new best epoch is reached

    fold_tag = f"fold{fold}_" if fold is not None else ""
    ckpt_path = exp_dir / f"{fold_tag}model.pth"

    n_train = len(train_files)

    # ---- Feature cache (big speedup) ----
    # Loading a .pt file from disk per step dominates the loop. Pre-load all
    # features into CPU RAM once, then only pay device transfer per step.
    # Default cap = 50% of free RAM, capped at 128 GB. Stored as bf16 when
    # autocast is on (matches compute dtype → no conversion cost, halves memory).
    try:
        _avail_kb = 0
        with open("/proc/meminfo") as _mi:
            for line in _mi:
                if line.startswith("MemAvailable:"):
                    _avail_kb = int(line.split()[1])
                    break
        _auto_cap_gb = max(4.0, min(128.0, (_avail_kb / 1024 / 1024) * 0.5))
    except Exception:
        _auto_cap_gb = 32.0
    _cache_cap_gb = float(train_cfg.get("feature_cache_gb", _auto_cap_gb))
    _cache_dtype = torch.bfloat16 if (bool(train_cfg.get("amp", True)) and device.type == "cuda") else None
    _feature_cache: dict[str, torch.Tensor] = {}
    if _cache_cap_gb > 0:
        total_bytes = 0
        cap_bytes = int(_cache_cap_gb * (1024 ** 3))
        all_files = list(train_files) + list(val_files)
        for f in all_files:
            try:
                t = torch.load(f, map_location="cpu", weights_only=True)
            except Exception as e:
                logger.warning(f"Cache skip {f.name}: {e}")
                continue
            if t.dim() == 2:
                t = t.unsqueeze(0)
            if _cache_dtype is not None and t.dtype != _cache_dtype:
                t = t.to(_cache_dtype)
            total_bytes += t.numel() * t.element_size()
            if total_bytes > cap_bytes:
                logger.info(
                    f"Feature cache exceeded {_cache_cap_gb:.1f} GB at {len(_feature_cache)} "
                    f"slides — remaining slides will stream from disk."
                )
                break
            _feature_cache[str(f)] = t
        logger.info(
            f"{'[Fold %d] ' % fold if fold is not None else ''}"
            f"Cached {len(_feature_cache)}/{len(all_files)} feature tensors "
            f"in RAM ({total_bytes/1024**3:.2f} GB, dtype={_cache_dtype or 'native'})"
        )

    # Mixed-precision (bf16) — big speedup on Ada-gen GPUs for attention.
    _use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    _amp_dtype = torch.bfloat16 if _use_amp else torch.float32
    if _use_amp:
        logger.info(
            f"{'[Fold %d] ' % fold if fold is not None else ''}"
            f"bfloat16 autocast enabled"
        )

    def _load_features(f) -> torch.Tensor:
        t = _feature_cache.get(str(f))
        if t is None:
            t = torch.load(f, map_location="cpu", weights_only=True)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            if _cache_dtype is not None and t.dtype != _cache_dtype:
                t = t.to(_cache_dtype)
        return t

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses: list[float] = []
        _log_every = max(1, n_train // 8)  # ~8 progress pings per epoch
        if epoch == 0:
            logger.info(
                f"{'[Fold %d] ' % fold if fold is not None else ''}"
                f"Starting epoch 1/{epochs} · {n_train} train slides · {len(val_files)} val slides"
            )
        for i, f in enumerate(train_files):
            try:
                features = _load_features(f).to(device, non_blocking=True)
            except RuntimeError:
                # Large slide OOM on GPU — fall back to CPU for this slide
                try:
                    features = _load_features(f)
                except Exception as load_err:
                    size_mb = f.stat().st_size // 1024 ** 2
                    logger.error(f"Cannot load {f.name} ({size_mb} MB): {load_err}. Skipping.")
                    continue
            label = label_map.get(f.stem, 0)
            target = torch.tensor([label], dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type=device.type, dtype=_amp_dtype, enabled=_use_amp):
                    out = model(features)
                    if isinstance(out, tuple):
                        logits, inst_loss = out
                        loss = criterion(logits, target) + 0.3 * inst_loss
                    else:
                        logits = out
                        loss = criterion(logits, target)
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    size_mb = f.stat().st_size // 1024 ** 2
                    logger.warning(
                        f"Skipping {f.stem}: CUDA OOM ({size_mb} MB, {features.shape[1]} patches). "
                        f"Reduce batch size or switch to CPU."
                    )
                    continue
                raise
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Intra-epoch progress — update shared status + log periodically so
            # users see activity during long first epochs on large WSIs.
            if job_status is not None:
                frac = (epoch + (i + 1) / max(n_train, 1)) / max(epochs, 1)
                job_status["progress"] = min(frac, 0.999)
                job_status["step"] = i + 1
                job_status["step_total"] = n_train
            if (i + 1) % _log_every == 0:
                logger.info(
                    f"{'[Fold %d] ' % fold if fold is not None else ''}"
                    f"Epoch {epoch+1}/{epochs} · slide {i+1}/{n_train} · running loss {float(np.mean(train_losses)):.4f}"
                )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                pass  # stepped after validation
            else:
                scheduler.step()

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # --- Validate ---
        model.eval()
        val_losses: list[float] = []
        correct = 0
        total = 0
        all_labels: list[int] = []
        all_probs: list[list[float]] = []
        all_slides: list[str] = []
        all_preds: list[int] = []
        with torch.no_grad():
            for f in val_files:
                features = _load_features(f).to(device, non_blocking=True)
                label = label_map.get(f.stem, 0)
                target = torch.tensor([label], dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, dtype=_amp_dtype, enabled=_use_amp):
                    out = model(features)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = criterion(logits, target)
                val_losses.append(loss.item())
                probs = torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy().tolist()
                all_probs.append(probs)
                all_labels.append(label)
                all_slides.append(f.stem)
                pred = int(logits.argmax(dim=1).item())
                all_preds.append(pred)
                correct += int(pred == label)
                total += 1

        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_acc = correct / total if total > 0 else 0.0

        # AUROC (binary or macro-OVR multiclass)
        val_auroc = float("nan")
        try:
            from sklearn.metrics import roc_auc_score as _roc_auc
            _y = np.array(all_labels)
            _p = np.array(all_probs)
            if len(set(all_labels)) < 2:
                val_auroc = float("nan")
            elif _p.shape[1] == 2:
                val_auroc = float(_roc_auc(_y, _p[:, 1]))
            else:
                val_auroc = float(_roc_auc(_y, _p, multi_class="ovr", average="macro"))
        except Exception as _e:
            logger.warning(f"AUROC compute skipped: {_e}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history.setdefault("val_auroc", []).append(val_auroc)

        # Checkpoint best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            # Snapshot this epoch's predictions for predictions.csv
            best_predictions = [
                {"slide_id": s, "y_true": int(y), "y_pred": int(p), "probs": list(pr)}
                for s, y, p, pr in zip(all_slides, all_labels, all_preds, all_probs)
            ]
        else:
            patience_counter += 1

        # Update shared job status
        if job_status:
            job_status["epoch"] = epoch + 1
            job_status["progress"] = (epoch + 1) / epochs
            job_status["metrics"] = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc,
                "val_auroc": val_auroc,
                "best_val_accuracy": best_val_acc,
                "best_epoch": best_epoch,
            }
            if fold is not None:
                job_status["metrics"]["fold"] = fold

        _auroc_str = f", auroc={val_auroc:.4f}" if val_auroc == val_auroc else ""  # NaN check
        logger.info(
            f"{'[Fold %d] ' % fold if fold is not None else ''}"
            f"Epoch {epoch+1}/{epochs} complete — "
            f"train={avg_train_loss:.4f}, val={avg_val_loss:.4f}, acc={val_acc:.4f}{_auroc_str}"
        )

        # Early stopping
        if patience > 0 and patience_counter >= patience:
            logger.info(
                f"{'[Fold %d] ' % fold if fold is not None else ''}"
                f"Early stopping at epoch {epoch+1} (patience={patience})"
            )
            break

    # Dump predictions.csv for the best epoch (enables retro plots: ROC, PR, calibration, …)
    if best_predictions:
        import csv as _csv
        num_classes = len(best_predictions[0]["probs"])
        preds_path = exp_dir / f"{fold_tag}predictions.csv"
        with preds_path.open("w", newline="") as _fh:
            _w = _csv.writer(_fh)
            _w.writerow(["slide_id", "y_true", "y_pred"] + [f"prob_{i}" for i in range(num_classes)])
            for r in best_predictions:
                _w.writerow([r["slide_id"], r["y_true"], r["y_pred"]] + [float(p) for p in r["probs"]])

    return {
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else 0.0,
        "total_epochs": len(history["train_loss"]),
        "model_params": n_params,
        "mil_method": config.get("mil_method", "abmil"),
        "mammoth_enabled": config.get("mammoth", {}).get("enabled", False),
        "checkpoint": str(ckpt_path),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------

def train_mil_model(config: dict, job_status: Optional[dict] = None) -> dict:
    """Train a MIL model.

    Dispatches to k-fold CV or stratified holdout based on
    config['evaluation']['strategy'].

    Returns a dict with metrics (and fold-aggregate stats if k-fold).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_id = config["dataset_id"]
    backbone = config.get("feature_backbone", config.get("backbone", "uni"))
    from pathclaw.preprocessing.feature_extraction import resolve_features_dir
    features_dir = resolve_features_dir(dataset_id, backbone)

    if not features_dir.exists():
        raise FileNotFoundError(
            f"Features not found at {features_dir}. "
            f"Run feature extraction first (wsi-preprocess skill → extract features)."
        )

    feature_files = sorted(features_dir.glob("*.pt"))
    if not feature_files:
        raise FileNotFoundError(f"No .pt feature files in {features_dir}")

    # Validate feature files are readable (sample check on first file)
    try:
        _test = torch.load(feature_files[0], map_location="cpu", weights_only=True)
        if _test.dim() < 2:
            raise ValueError(
                f"Feature file {feature_files[0].name} has unexpected shape {_test.shape}. "
                f"Expected (N, D) patch embeddings."
            )
        del _test
    except Exception as e:
        raise RuntimeError(
            f"Cannot read feature file {feature_files[0].name}: {e}. "
            f"Re-run feature extraction to regenerate."
        ) from e

    # Load label mapping
    labels_path = PATHCLAW_DATA_DIR / "datasets" / dataset_id / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"No labels.json found at {labels_path}. "
            f"Create a JSON file mapping slide stem names to integer class labels, e.g. "
            f'{{"slide_001": 0, "slide_002": 1}}. '
            f"Use the data-profiling skill to analyse your label file and generate this mapping."
        )
    label_map: dict = json.loads(labels_path.read_text())

    # Validate label coverage and check for class imbalance
    labeled = [f for f in feature_files if f.stem in label_map]
    unlabeled = [f.stem for f in feature_files if f.stem not in label_map]
    if unlabeled:
        logger.warning(
            f"{len(unlabeled)} feature files have no label: {', '.join(unlabeled[:5])}"
            f"{'...' if len(unlabeled) > 5 else ''}. They will be skipped."
        )
    if not labeled:
        raise ValueError(
            f"No feature files have matching labels in labels.json. "
            f"Ensure slide stem names in labels.json match the .pt filenames in {features_dir}."
        )
    from collections import Counter as _Counter
    label_counts = _Counter(label_map.get(f.stem, None) for f in labeled)
    if None in label_counts:
        del label_counts[None]
    if label_counts:
        counts = list(label_counts.values())
        ratio = max(counts) / max(min(counts), 1)
        if ratio > 10:
            logger.warning(
                f"Severe class imbalance detected: {dict(label_counts)}. "
                f"Max/min ratio = {ratio:.1f}x. Consider oversampling or weighted loss."
            )

    # Experiment output dir
    job_id = job_status["job_id"] if job_status else "manual"
    exp_dir = PATHCLAW_DATA_DIR / "experiments" / job_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # File-based logging. Force INFO so logger.info() lines reach train.log
    # (the root logger defaults to WARNING, which otherwise filters them out).
    log_path = exp_dir / "train.log"
    _fh = logging.FileHandler(log_path)
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_fh)
    _prev_level = logger.level
    logger.setLevel(logging.INFO)

    try:
        eval_strategy = config.get("evaluation", {}).get("strategy", "holdout")
        if "-fold-cv" in eval_strategy:
            result = _train_kfold(config, feature_files, label_map, exp_dir, eval_strategy, job_status)
        else:
            result = _train_holdout(config, feature_files, label_map, exp_dir, job_status)

        # Save outputs
        (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
        (exp_dir / "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
        if "history" in result:
            (exp_dir / "history.json").write_text(json.dumps(result["history"], indent=2))

        try:
            _save_training_plots(result.get("history", {}), exp_dir)
        except Exception as plot_err:
            logger.warning(f"Plot generation skipped: {plot_err}")
        logger.info(f"Training complete. Best val acc: {result['metrics'].get('best_val_accuracy', 0):.4f}")
        return result

    finally:
        logger.removeHandler(_fh)
        _fh.close()
        logger.setLevel(_prev_level)


# ---------------------------------------------------------------------------
# Holdout split training
# ---------------------------------------------------------------------------

def _train_holdout(
    config: dict,
    feature_files: list,
    label_map: dict,
    exp_dir: Path,
    job_status: Optional[dict] = None,
) -> dict:
    labels_arr = [label_map.get(f.stem, 0) for f in feature_files]

    try:
        train_files, val_files, _, _ = train_test_split(
            feature_files, labels_arr,
            test_size=0.2, random_state=42, stratify=labels_arr,
        )
    except ValueError:
        # Fallback when stratification fails (too few samples per class)
        logger.warning("Stratified split failed — falling back to sequential 80/20 split.")
        n = int(0.8 * len(feature_files))
        train_files, val_files = feature_files[:n], feature_files[n:]

    # Save split for evaluation
    split_info = {
        "train": [f.stem for f in train_files],
        "val": [f.stem for f in val_files],
    }
    (exp_dir / "split.json").write_text(json.dumps(split_info, indent=2))

    fold_result = _train_one_split(config, train_files, val_files, label_map, exp_dir, job_status)
    history = fold_result.pop("history")
    return {"metrics": fold_result, "history": history, "experiment_dir": str(exp_dir)}


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------

def _train_kfold(
    config: dict,
    feature_files: list,
    label_map: dict,
    exp_dir: Path,
    strategy: str,
    job_status: Optional[dict] = None,
) -> dict:
    n_folds = int(strategy.split("-")[0])
    labels_arr = np.array([label_map.get(f.stem, 0) for f in feature_files])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics: list[dict] = []
    all_history: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(feature_files, labels_arr)):
        logger.info(f"Starting fold {fold_idx + 1}/{n_folds}")
        train_f = [feature_files[i] for i in train_idx]
        val_f = [feature_files[i] for i in val_idx]

        # Save fold split
        fold_split = {
            "train": [f.stem for f in train_f],
            "val": [f.stem for f in val_f],
        }
        (exp_dir / f"split_fold{fold_idx}.json").write_text(json.dumps(fold_split, indent=2))

        fold_result = _train_one_split(
            config, train_f, val_f, label_map, exp_dir, job_status, fold=fold_idx
        )
        history = fold_result.pop("history")
        fold_metrics.append(fold_result)
        all_history.append(history)

    # Aggregate metrics across folds
    agg = _aggregate_fold_metrics(fold_metrics)
    agg["n_folds"] = n_folds
    agg["per_fold"] = fold_metrics

    return {"metrics": agg, "history": all_history[0], "experiment_dir": str(exp_dir)}


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Compute mean ± std for numeric metrics across folds."""
    keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float)) and k != "n_folds"]
    agg: dict = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        agg[k] = round(float(np.mean(vals)), 6)
        agg[f"{k}_std"] = round(float(np.std(vals)), 6)
    return agg


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _save_training_plots(history: dict, exp_dir: Path) -> None:
    if not history or not history.get("train_loss"):
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train Loss", color="#3b82f6")
    ax.plot(history["val_loss"], label="Val Loss", color="#ef4444")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curves.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["val_acc"], label="Val Accuracy", color="#22c55e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_curve.png", dpi=150)
    plt.close()
