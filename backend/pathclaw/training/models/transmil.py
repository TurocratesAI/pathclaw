"""TransMIL — Transformer-based Correlated Multiple Instance Learning.

Reference: Shao et al., NeurIPS 2021
"TransMIL: Transformer based Correlated Multiple Instance Learning for
Whole Slide Image Classification"
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._utils import make_patch_embed


class _SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** 2


class TransMIL(nn.Module):
    """TransMIL with positional encoding and 2-layer transformer encoder.

    Args:
        in_dim: Input feature dimension.
        embed_dim: Transformer model dimension.
        num_classes: Number of output classes.
        moe_args: Optional MAMMOTH config for patch embedding.
        num_heads: Number of transformer attention heads (default 8).
        num_layers: Number of transformer encoder layers (default 2).
        dropout: Dropout for transformer layers (default 0.1).
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_classes: int,
        moe_args: dict | None = None,
        plugins: list[dict] | None = None,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = make_patch_embed(in_dim, embed_dim, moe_args, plugins=plugins)

        # [CLS] token — learnable, prepended to the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(0)  # → (1, N, D)

        B, N, _ = x.shape
        h = self.patch_embed(x)                           # (B, N, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, embed_dim)
        h = torch.cat([cls, h], dim=1)                    # (B, N+1, embed_dim)

        h = self.transformer(h)                           # (B, N+1, embed_dim)
        cls_out = self.norm(h[:, 0])                      # (B, embed_dim)

        logits = self.classifier(cls_out)
        if return_attention:
            # Proxy: cosine similarity between CLS output and each patch token
            import torch.nn.functional as F
            patch_h = h[:, 1:]                            # (B, N, embed_dim)
            cls_vec = cls_out.unsqueeze(1)                # (B, 1, embed_dim)
            sim = F.cosine_similarity(cls_vec, patch_h, dim=-1)  # (B, N)
            attn = torch.softmax(sim, dim=-1)
            return logits, attn.squeeze(0)                # (N,)
        return logits
