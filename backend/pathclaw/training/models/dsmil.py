"""DSMIL — Dual-Stream Multiple Instance Learning.

Reference: Li et al., CVPR 2021
"Dual-stream Multiple Instance Learning Network for Whole Slide Image
Classification with Self-supervised Contrastive Learning"

Two streams:
  - Max-pool stream: selects the most critical patch as a query
  - Distance-based attention stream: uses the critical instance as a query
    to compute attention over all patches
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._utils import make_patch_embed


class DSMIL(nn.Module):
    """Dual-Stream MIL.

    Args:
        in_dim: Input feature dimension.
        embed_dim: Embedding dimension.
        num_classes: Number of output classes.
        moe_args: Optional MAMMOTH config.
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_classes: int,
        moe_args: dict | None = None,
        plugins: list[dict] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = make_patch_embed(in_dim, embed_dim, moe_args, plugins=plugins)
        self.dropout = nn.Dropout(dropout)

        # Instance-level classifier (max-pool stream)
        self.instance_classifier = nn.Linear(embed_dim, num_classes)

        # Attention with critical-instance query
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)

        # Bag-level classifier applied to both streams
        self.bag_classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, N, _ = x.shape
        h = self.dropout(self.patch_embed(x))          # (B, N, embed_dim)

        # --- Max-pool stream: find the most critical patch ---
        inst_logits = self.instance_classifier(h)      # (B, N, num_classes)
        inst_scores = inst_logits.max(dim=-1).values   # (B, N)
        critical_idx = inst_scores.argmax(dim=1)       # (B,)
        critical = h[torch.arange(B), critical_idx]    # (B, embed_dim)

        # --- Attention stream: attend over patches using critical as query ---
        Q = self.query_proj(critical).unsqueeze(1)     # (B, 1, embed_dim)
        K = self.key_proj(h)                           # (B, N, embed_dim)
        scale = h.shape[-1] ** 0.5
        A = torch.bmm(Q, K.transpose(1, 2)) / scale   # (B, 1, N)
        A = torch.softmax(A, dim=-1)
        attn_bag = torch.bmm(A, h).squeeze(1)          # (B, embed_dim)

        # Concatenate both streams
        fused = torch.cat([critical, attn_bag], dim=-1)  # (B, embed_dim*2)
        logits = self.bag_classifier(fused)
        if return_attention:
            return logits, A.squeeze(1).squeeze(0)  # (N,)
        return logits
