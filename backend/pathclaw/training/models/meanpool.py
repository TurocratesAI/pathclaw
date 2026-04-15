"""Mean-pooling MIL baseline with optional MAMMOTH patch embedding."""

from __future__ import annotations

import torch
import torch.nn as nn

from ._utils import make_patch_embed


class MeanPoolMIL(nn.Module):
    """Simple mean-pooling MIL baseline.

    Aggregates patch embeddings by averaging, then classifies the bag-level
    representation. Useful as a sanity-check baseline.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_classes: int,
        moe_args: dict | None = None,
        plugins: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self.patch_embed = make_patch_embed(in_dim, embed_dim, moe_args, plugins=plugins)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.patch_embed(x)          # (B, N, embed_dim)
        bag = torch.mean(h, dim=1)       # (B, embed_dim)
        logits = self.classifier(bag)
        if return_attention:
            N = h.shape[1]
            uniform = torch.full((N,), 1.0 / N, device=x.device)
            return logits, uniform
        return logits
