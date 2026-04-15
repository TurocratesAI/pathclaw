"""Patch-level MLP classifier — runs per-patch on FM features.

Unlike MIL (slide-level pooling), this model predicts an independent label for
each patch. The same UNI/CONCH/Virchow/etc. feature caches feed it directly.
Typical use cases: patch-level tissue-type classification, or cell-detection
where each patch has a binary "cell present" label.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        num_layers: int = 2,
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        prev = in_dim
        for _i in range(num_layers - 1):
            layers += [nn.Linear(prev, embed_dim), nn.GELU(), nn.Dropout(dropout)]
            prev = embed_dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accepts (B, D) or (B, N, D). Returns same leading dims + num_classes.
        return self.net(x)
