"""RRTMIL — Re-embedded Regional Transformer for MIL.

Reference: Tang et al., MICCAI 2023
"Multiple Instance Learning Framework with Masked Hard Instance Mining
for Whole Slide Image Classification" (RRTMIL variant)

Key idea: Group patches into regions, apply local transformer within
each region, then apply global attention across region summaries.
"""

from __future__ import annotations


import torch
import torch.nn as nn

from ._utils import make_patch_embed


class RRTMIL(nn.Module):
    """Re-embedded Regional Transformer MIL.

    Patches are grouped into non-overlapping regions. A local transformer
    processes each region; a global gated-attention aggregator then
    combines region-level representations into a bag-level prediction.

    Args:
        in_dim: Input feature dimension.
        embed_dim: Embedding dimension.
        num_classes: Number of output classes.
        moe_args: Optional MAMMOTH config.
        region_size: Number of patches per region (default 16).
        num_heads: Transformer heads for local encoder (default 4).
        attn_dim: Gated-attention hidden dim (default 128).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_classes: int,
        moe_args: dict | None = None,
        plugins: list[dict] | None = None,
        region_size: int = 16,
        num_heads: int = 4,
        attn_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.region_size = region_size

        self.patch_embed = make_patch_embed(in_dim, embed_dim, moe_args, plugins=plugins)

        # Local transformer encoder (within regions)
        local_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(local_layer, num_layers=1)

        # Gated attention over region summaries
        self.attn_V = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B = x.shape[0]
        h = self.patch_embed(x)                           # (B, T, embed_dim)
        # T = N when using a standard linear; T = num_experts*num_slots when MAMMOTH is active.

        # Pad T to be divisible by region_size
        R = self.region_size
        T = h.shape[1]
        pad = (R - T % R) % R
        if pad:
            h = torch.cat([h, h[:, -pad:]], dim=1)        # repeat-pad
        N_pad = h.shape[1]
        num_regions = N_pad // R

        # Reshape into regions: (B * num_regions, R, embed_dim)
        h_reg = h.view(B * num_regions, R, -1)
        h_reg = self.local_encoder(h_reg)                 # local attention
        region_rep = h_reg.mean(dim=1)                    # (B*num_regions, embed_dim)
        region_rep = region_rep.view(B, num_regions, -1)  # (B, num_regions, embed_dim)

        # Global gated attention over regions
        A_V = self.attn_V(region_rep)
        A_U = self.attn_U(region_rep)
        A = torch.softmax(self.attn_w(A_V * A_U), dim=1)  # (B, num_regions, 1)

        bag = torch.sum(A * region_rep, dim=1)             # (B, embed_dim)
        bag = self.norm(bag)
        logits = self.classifier(bag)
        if return_attention:
            # Expand region-level scores to patch-level (each patch gets its region score)
            attn_region = A.squeeze(-1).squeeze(0)         # (num_regions,)
            attn_patch = attn_region.repeat_interleave(R)  # (N_pad,)
            attn_patch = attn_patch[:T]                    # trim padding to original T
            return logits, attn_patch
        return logits
