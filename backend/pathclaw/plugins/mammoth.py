"""MAMMOTH patch-embed plugin — MoE-LoRA slot-based patch projection.

Replaces a single Linear(in_dim → embed_dim) in MIL models. With keep_slots=True
the output is (batch, num_experts * num_slots, embed_dim) — a fixed-size slot
sequence that all MIL attention heads operate over.
"""
from __future__ import annotations

import torch.nn as nn


def build(in_dim: int, embed_dim: int, config: dict) -> nn.Module:
    try:
        from mammoth import Mammoth
    except ImportError as exc:
        raise ImportError(
            "mammoth-moe is not installed. Run: pip install mammoth-moe einops"
        ) from exc
    return Mammoth(
        input_dim=in_dim,
        dim=embed_dim,
        num_experts=int(config.get("num_experts", 30)),
        num_slots=int(config.get("num_slots", 10)),
        num_heads=int(config.get("num_heads", 16)),
        share_lora_weights=bool(config.get("share_lora_weights", True)),
        auto_rank=bool(config.get("auto_rank", True)),
        dropout=float(config.get("dropout", 0.1)),
        keep_slots=True,
    )
