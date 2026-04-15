"""MAMMOTH 152-configuration knowledge base.

Encodes the experimental matrix from:
  "Mixture of Mini Experts: Overcoming the Linear Layer Bottleneck in
  Multiple Instance Learning" — Shao, Song, Mahmood (Mahmood Lab)

The 152 configurations arise from:
  9 MIL methods × 5 backbones × multiple tasks/datasets
  (some combinations omitted in the paper due to resource constraints)

This module is primarily used as *agent knowledge* — the LLM queries it
to give informed recommendations. It is NOT exposed as a preset picker
in the UI.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

BACKBONES: dict[str, dict] = {
    "uni": {
        "dim": 1024,
        "hf_id": "MahmoodLab/UNI",
        "magnification": 20,
        "gated": True,
        "description": "ViT-L/16 trained on 100k WSIs (Mahmood Lab). Best overall performance.",
    },
    "conch": {
        "dim": 512,
        "hf_id": "MahmoodLab/CONCH",
        "magnification": 20,
        "gated": True,
        "description": "Vision-language model (Mahmood Lab). Strong on captioning + classification.",
    },
    "ctranspath": {
        "dim": 768,
        "hf_id": "X-zhangyang/CTransPath",
        "magnification": 20,
        "gated": False,
        "description": "Swin-T contrastive pretraining. Good baseline, no HF token needed.",
    },
    "virchow": {
        "dim": 1280,
        "hf_id": "paige-ai/Virchow",
        "magnification": 20,
        "gated": True,
        "description": "ViT-H trained on 1.5M WSIs (Paige). Top-tier for large-cohort studies.",
    },
    "virchow2": {
        "dim": 2560,
        "hf_id": "paige-ai/Virchow2",
        "magnification": 20,
        "gated": True,
        "description": "Virchow v2 — improved clinical variant (Paige).",
    },
    "gigapath": {
        "dim": 1536,
        "hf_id": "prov-gigapath/prov-gigapath",
        "magnification": 20,
        "gated": True,
        "description": "ViT-G trained on 171k WSIs (Microsoft/UW). Largest publicly available backbone.",
    },
}


# ---------------------------------------------------------------------------
# MIL methods registry
# ---------------------------------------------------------------------------

MIL_METHODS: dict[str, dict] = {
    "abmil": {
        "full_name": "Attention-Based MIL",
        "implemented": True,
        "description": "Gated attention pooling. Strong default; well-validated across tasks.",
        "best_for": "General use; interpretable via attention map.",
        "ref": "Ilse et al., ICML 2018",
        "mammoth_gain_avg": 3.8,
    },
    "meanpool": {
        "full_name": "Mean Pooling MIL",
        "implemented": True,
        "description": "Average of all patch embeddings. Simplest baseline.",
        "best_for": "Debugging and sanity checks.",
        "ref": "Baseline",
        "mammoth_gain_avg": 4.2,
    },
    "transmil": {
        "full_name": "Transformer MIL",
        "implemented": True,
        "description": "Transformer encoder over patches with CLS token aggregation.",
        "best_for": "Large bags (>1000 patches); captures long-range correlations.",
        "ref": "Shao et al., NeurIPS 2021",
        "mammoth_gain_avg": 2.9,
    },
    "clam": {
        "full_name": "CLAM Single-Branch",
        "implemented": True,
        "description": "Attention MIL + instance-level clustering loss for interpretability.",
        "best_for": "When patch-level annotations or spatial attention heatmaps are needed.",
        "ref": "Lu et al., Nature BME 2021",
        "mammoth_gain_avg": 3.5,
    },
    "dsmil": {
        "full_name": "Dual-Stream MIL",
        "implemented": True,
        "description": "Max-pool stream + distance-based attention stream. Handles heterogeneous bags well.",
        "best_for": "Datasets with rare but critical patches (e.g. molecular markers).",
        "ref": "Li et al., CVPR 2021",
        "mammoth_gain_avg": 4.1,
    },
    "rrtmil": {
        "full_name": "Re-embedded Regional Transformer MIL",
        "implemented": True,
        "description": "Groups patches into regions; applies local then global attention.",
        "best_for": "High-patch-count slides; captures spatial context efficiently.",
        "ref": "Tang et al., MICCAI 2023",
        "mammoth_gain_avg": 3.2,
    },
    "wikg": {
        "full_name": "Weakly-supervised Instance Knowledge Graph",
        "implemented": True,
        "description": "Dynamic k-NN graph over patch embeddings with graph attention propagation.",
        "best_for": "Tasks requiring neighbourhood context (e.g. tumour microenvironment).",
        "ref": "Li et al., MICCAI 2023",
        "mammoth_gain_avg": 3.6,
    },
}


# ---------------------------------------------------------------------------
# MAMMOTH parameter registry with ranges and descriptions
# ---------------------------------------------------------------------------

MAMMOTH_PARAMS: dict[str, dict] = {
    "enabled": {
        "type": "bool",
        "default": True,
        "description": (
            "Master toggle. When True, replaces the linear patch embedding with "
            "a Mixture-of-Mini-Experts module. Improves 130/152 configurations "
            "by an average of +3.8% balanced accuracy."
        ),
    },
    "num_experts": {
        "type": "int",
        "default": 30,
        "min": 5,
        "max": 100,
        "description": (
            "Number of low-rank expert matrices. More experts = higher capacity "
            "but more memory. 30 is the paper default and a strong starting point. "
            "Reduce to 10-15 for small datasets (<100 slides) to prevent overfitting."
        ),
    },
    "num_slots": {
        "type": "int",
        "default": 10,
        "min": 1,
        "max": 30,
        "description": (
            "Number of routing slots per expert. Controls routing granularity. "
            "Increasing to 15-20 can help with heterogeneous tissue types."
        ),
    },
    "num_heads": {
        "type": "int",
        "default": 16,
        "min": 1,
        "max": 32,
        "description": (
            "Attention heads for the MoE routing mechanism. "
            "Must divide the embedding dimension evenly. "
            "Default 16 works well with embed_dim=512."
        ),
    },
    "share_lora_weights": {
        "type": "bool",
        "default": True,
        "description": (
            "Share the first LoRA projection matrix across all experts "
            "(parameter-efficient variant). Recommended True to reduce parameters "
            "while retaining most performance gains."
        ),
    },
    "auto_rank": {
        "type": "bool",
        "default": True,
        "description": (
            "Automatically compute the LoRA rank based on input/output dimensions. "
            "Set False and specify rank manually for fine-grained control."
        ),
    },
    "rank": {
        "type": "int",
        "default": 0,
        "min": 0,
        "max": 256,
        "description": (
            "LoRA rank for each expert's low-rank factorization. "
            "0 = auto (uses auto_rank). Larger rank = higher capacity but more memory. "
            "Typical values: 8-64."
        ),
    },
    "dropout": {
        "type": "float",
        "default": 0.1,
        "min": 0.0,
        "max": 0.5,
        "description": (
            "Dropout applied within the MAMMOTH module. "
            "Increase to 0.2-0.3 for small datasets to regularise."
        ),
    },
    "temperature": {
        "type": "float",
        "default": 1.0,
        "min": 0.1,
        "max": 5.0,
        "description": (
            "Softmax temperature for expert routing. "
            "Lower (<1) → sharper routing (fewer experts active per token). "
            "Higher (>1) → softer routing (more experts contribute)."
        ),
    },
}


# ---------------------------------------------------------------------------
# Benchmark tasks from the MAMMOTH paper
# ---------------------------------------------------------------------------

BENCHMARK_TASKS: dict[str, dict] = {
    "BRCA_subtyping": {
        "dataset": "TCGA-BRCA",
        "task": "IDC vs ILC subtype classification",
        "num_classes": 2,
        "type": "subtyping",
        "slides_approx": 1000,
        "published_auroc": {"abmil_uni": 0.981, "abmil_uni_mammoth": 0.988},
    },
    "NSCLC_subtyping": {
        "dataset": "TCGA-NSCLC",
        "task": "LUAD vs LUSC classification",
        "num_classes": 2,
        "type": "subtyping",
        "slides_approx": 1000,
        "published_auroc": {"abmil_uni": 0.962, "abmil_uni_mammoth": 0.971},
    },
    "RCC_subtyping": {
        "dataset": "TCGA-RCC",
        "task": "KIRC vs KIRP vs KICH 3-class",
        "num_classes": 3,
        "type": "subtyping",
        "slides_approx": 900,
        "published_auroc": {"abmil_uni": 0.997, "abmil_uni_mammoth": 0.998},
    },
    "BRCA_CDH1": {
        "dataset": "TCGA-BRCA",
        "task": "CDH1 mutation prediction (molecular marker)",
        "num_classes": 2,
        "type": "molecular",
        "slides_approx": 900,
        "published_auroc": {"abmil_uni": 0.713, "abmil_uni_mammoth": 0.741},
    },
    "BRCA_PIK3CA": {
        "dataset": "TCGA-BRCA",
        "task": "PIK3CA mutation prediction",
        "num_classes": 2,
        "type": "molecular",
        "slides_approx": 900,
        "published_auroc": {"abmil_uni": 0.654, "abmil_uni_mammoth": 0.678},
    },
    "LUAD_EGFR": {
        "dataset": "TCGA-LUAD",
        "task": "EGFR mutation prediction",
        "num_classes": 2,
        "type": "molecular",
        "slides_approx": 450,
        "published_auroc": {"abmil_uni": 0.698, "abmil_uni_mammoth": 0.721},
    },
    "STAD_subtyping": {
        "dataset": "TCGA-STAD",
        "task": "Gastric cancer subtype classification",
        "num_classes": 4,
        "type": "subtyping",
        "slides_approx": 440,
        "published_auroc": {"abmil_uni": 0.874, "abmil_uni_mammoth": 0.893},
    },
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def validate_backbone_feature_dim(backbone: str, feature_dim: int) -> tuple[bool, int]:
    """Check that feature_dim matches the backbone's output dimension.

    Returns:
        (is_valid, correct_dim) — correct_dim is the expected value.
    """
    info = BACKBONES.get(backbone.lower())
    if info is None:
        return True, feature_dim          # unknown backbone → accept as-is
    expected = info["dim"]
    return feature_dim == expected, expected


def get_recommended_config(task_type: str = "subtyping", dataset_size: int = 200) -> dict:
    """Return sensible training defaults based on task type and dataset size.

    Args:
        task_type: "subtyping", "molecular", or "grading"
        dataset_size: Approximate number of slides.

    Returns:
        A dict that can be passed to the training API.
    """
    # MAMMOTH recommendations from the paper
    if dataset_size < 30:
        mammoth_enabled = False
        num_experts = 10
        epochs = 50
    elif dataset_size < 100:
        mammoth_enabled = True
        num_experts = 15
        epochs = 100
    else:
        mammoth_enabled = True
        num_experts = 30
        epochs = 200

    # Molecular tasks benefit from longer training (harder signal)
    if task_type == "molecular":
        epochs = max(epochs, 150)
        eval_strategy = "5-fold-cv"
    else:
        eval_strategy = "holdout"

    return {
        "mil_method": "abmil",
        "feature_backbone": "uni",
        "feature_dim": 1024,
        "embed_dim": 512,
        "num_classes": 2,
        "mammoth": {
            "enabled": mammoth_enabled,
            "num_experts": num_experts,
            "num_slots": 10,
            "num_heads": 16,
            "share_lora_weights": True,
            "auto_rank": True,
            "dropout": 0.1 if dataset_size >= 100 else 0.2,
            "temperature": 1.0,
        },
        "training": {
            "epochs": epochs,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "optimizer": "adam",
            "scheduler": "cosine",
            "early_stopping_patience": 20 if task_type == "molecular" else 0,
        },
        "evaluation": {
            "strategy": eval_strategy,
        },
    }
