"""Retro-plot builder: build PNG plots for a completed experiment from its
predictions.csv + history.json. Built-in kinds cover the usual diagnostics;
`custom` runs a user-supplied matplotlib snippet in a restricted namespace
that has `history`, `metrics`, `predictions` (pandas DataFrame), and
`num_classes` pre-loaded. The snippet must assign `fig`."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()

_BUILTIN = {"roc_curve", "pr_curve", "per_class_auroc", "calibration",
            "confusion_matrix", "prediction_histogram"}


def _load_experiment(experiment_id: str) -> tuple[Path, pd.DataFrame, dict, dict]:
    exp_dir = PATHCLAW_DATA_DIR / "experiments" / experiment_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment {experiment_id!r} not found at {exp_dir}")
    preds_path = exp_dir / "predictions.csv"
    # Fall back to fold0 predictions if split naming was used
    if not preds_path.exists():
        fold_preds = sorted(exp_dir.glob("fold*_predictions.csv"))
        if fold_preds:
            preds_path = fold_preds[0]
    if not preds_path.exists():
        raise FileNotFoundError(
            f"No predictions.csv in {exp_dir}. Train the model first (predictions are "
            f"dumped at the end of training)."
        )
    preds = pd.read_csv(preds_path)

    history: dict = {}
    hist_path = exp_dir / "history.json"
    if hist_path.exists():
        try:
            history = json.loads(hist_path.read_text())
        except Exception:
            history = {}

    metrics: dict = {}
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = {}
    return exp_dir, preds, history, metrics


def _prob_matrix(preds: pd.DataFrame) -> tuple[np.ndarray, int]:
    prob_cols = [c for c in preds.columns if c.startswith("prob_")]
    prob_cols.sort(key=lambda s: int(s.split("_", 1)[1]))
    if not prob_cols:
        raise ValueError("predictions.csv has no prob_* columns")
    probs = preds[prob_cols].to_numpy(dtype=float)
    return probs, len(prob_cols)


def _roc(preds: pd.DataFrame, title: str = "") -> plt.Figure:
    from sklearn.metrics import roc_curve, auc
    y = preds["y_true"].to_numpy()
    probs, n = _prob_matrix(preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    if n == 2:
        fpr, tpr, _ = roc_curve(y, probs[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f"AUC={auc(fpr, tpr):.3f}")
    else:
        for c in range(n):
            fpr, tpr, _ = roc_curve((y == c).astype(int), probs[:, c])
            ax.plot(fpr, tpr, lw=1.5, label=f"class {c}: AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title or "ROC curve"); ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _pr(preds: pd.DataFrame, title: str = "") -> plt.Figure:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y = preds["y_true"].to_numpy()
    probs, n = _prob_matrix(preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    if n == 2:
        prec, rec, _ = precision_recall_curve(y, probs[:, 1])
        ap = average_precision_score(y, probs[:, 1])
        ax.plot(rec, prec, lw=2, label=f"AP={ap:.3f}")
    else:
        for c in range(n):
            yc = (y == c).astype(int)
            prec, rec, _ = precision_recall_curve(yc, probs[:, c])
            ap = average_precision_score(yc, probs[:, c])
            ax.plot(rec, prec, lw=1.5, label=f"class {c}: AP={ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title or "Precision-Recall curve"); ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def _per_class_auroc(preds: pd.DataFrame, title: str = "") -> plt.Figure:
    from sklearn.metrics import roc_auc_score
    y = preds["y_true"].to_numpy()
    probs, n = _prob_matrix(preds)
    aucs = []
    for c in range(n):
        yc = (y == c).astype(int)
        if yc.sum() == 0 or yc.sum() == len(yc):
            aucs.append(float("nan"))
        else:
            aucs.append(float(roc_auc_score(yc, probs[:, c])))
    fig, ax = plt.subplots(figsize=(max(4, n * 0.7), 4))
    ax.bar(range(n), aucs, color="#3b82f6")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, ls="--", color="#888", lw=1)
    ax.set_xticks(range(n)); ax.set_xticklabels([f"class {i}" for i in range(n)])
    ax.set_ylabel("AUROC"); ax.set_title(title or "Per-class AUROC (OVR)")
    for i, a in enumerate(aucs):
        if a == a:
            ax.text(i, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def _calibration(preds: pd.DataFrame, title: str = "") -> plt.Figure:
    y = preds["y_true"].to_numpy()
    probs, n = _prob_matrix(preds)
    # Use top-class confidence as the calibration signal
    top_probs = probs.max(axis=1)
    top_pred = probs.argmax(axis=1)
    correct = (top_pred == y).astype(int)
    bins = np.linspace(0, 1, 11)
    bin_ids = np.clip(np.digitize(top_probs, bins) - 1, 0, 9)
    fig, ax = plt.subplots(figsize=(5, 5))
    mids, accs = [], []
    for b in range(10):
        mask = bin_ids == b
        if mask.sum() > 0:
            mids.append((bins[b] + bins[b + 1]) / 2)
            accs.append(correct[mask].mean())
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect")
    ax.plot(mids, accs, "o-", lw=2, color="#ef4444", label="empirical")
    ax.set_xlabel("Confidence (top class)"); ax.set_ylabel("Accuracy")
    ax.set_title(title or "Reliability diagram"); ax.legend()
    fig.tight_layout()
    return fig


def _confusion(preds: pd.DataFrame, title: str = "") -> plt.Figure:
    _, n = _prob_matrix(preds)
    cm = np.zeros((n, n), dtype=int)
    for yt, yp in zip(preds["y_true"], preds["y_pred"]):
        cm[int(yt), int(yp)] += 1
    fig, ax = plt.subplots(figsize=(4 + n * 0.5, 4 + n * 0.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title or "Confusion matrix")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    fig.colorbar(im, ax=ax); fig.tight_layout()
    return fig


def _pred_hist(preds: pd.DataFrame, title: str = "") -> plt.Figure:
    probs, n = _prob_matrix(preds)
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in range(n):
        ax.hist(probs[:, c], bins=20, alpha=0.5, label=f"class {c}")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Count")
    ax.set_title(title or "Prediction histogram"); ax.legend()
    fig.tight_layout()
    return fig


_BUILTIN_FNS = {
    "roc_curve": _roc,
    "pr_curve": _pr,
    "per_class_auroc": _per_class_auroc,
    "calibration": _calibration,
    "confusion_matrix": _confusion,
    "prediction_histogram": _pred_hist,
}


def make_plot(experiment_id: str, kind: str, spec: str = "", title: str = "") -> dict[str, Any]:
    if kind not in _BUILTIN and kind != "custom":
        raise ValueError(f"Unknown kind {kind!r}. Expected one of {sorted(_BUILTIN | {'custom'})}")

    exp_dir, preds, history, metrics = _load_experiment(experiment_id)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    if kind in _BUILTIN_FNS:
        fig = _BUILTIN_FNS[kind](preds, title=title)
    else:
        if not spec.strip():
            raise ValueError("kind='custom' requires a non-empty `spec` matplotlib snippet that sets `fig`.")
        _, num_classes = _prob_matrix(preds)
        ns: dict[str, Any] = {
            "history": history, "metrics": metrics, "predictions": preds,
            "num_classes": num_classes, "plt": plt, "np": np, "pd": pd,
            "title": title, "fig": None,
        }
        exec(compile(spec, "<make_plot:custom>", "exec"), ns, ns)
        fig = ns.get("fig")
        if fig is None or not isinstance(fig, plt.Figure):
            raise ValueError("custom spec did not assign `fig = <matplotlib Figure>`")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = plots_dir / f"{kind}_{ts}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return {"path": str(out), "name": out.name, "kind": kind}
