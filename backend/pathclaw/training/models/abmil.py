"""Attention-Based MIL (ABMIL) with optional MAMMOTH patch embedding.

Reference: Ilse et al., ICML 2018 — "Attention-based Deep Multiple Instance Learning"
Gated attention variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._utils import make_patch_embed


class ABMIL(nn.Module):
    """Gated Attention-Based MIL.

    Args:
        in_dim: Input feature dimension (must match the feature backbone).
        embed_dim: Hidden embedding dimension.
        num_classes: Number of output classes.
        moe_args: If provided and enabled=True, replaces the linear patch
            embedding with a MAMMOTH MoE module.
        attn_dim: Hidden dimension of the gated attention mechanism.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_classes: int,
        moe_args: dict | None = None,
        plugins: list[dict] | None = None,
        attn_dim: int = 128,
    ) -> None:
        super().__init__()
        self.patch_embed = make_patch_embed(in_dim, embed_dim, moe_args, plugins=plugins)

        # Gated attention: A = softmax(w^T (tanh(Vh) ⊙ sigmoid(Uh)))
        self.attention_V = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attn_dim, 1)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (N, D)  or  (B, N, D)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # → (1, N, D)

        h = self.patch_embed(x)               # (B, N, embed_dim)

        A_V = self.attention_V(h)             # (B, N, attn_dim)
        A_U = self.attention_U(h)             # (B, N, attn_dim)
        A = self.attention_w(A_V * A_U)       # (B, N, 1)
        A = torch.softmax(A, dim=1)           # (B, N, 1)

        M = torch.sum(A * h, dim=1)           # (B, embed_dim)
        logits = self.classifier(M)
        if return_attention:
            return logits, A.squeeze(-1).squeeze(0)  # (N,)
        return logits
