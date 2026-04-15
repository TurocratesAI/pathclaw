"""WIKG — Weakly-supervised Instance Knowledge Graph for MIL.

Reference: Li et al., MICCAI 2023
"Dynamic Graph Representation Learning for Weakly Supervised
Whole Slide Image Classification"

Key idea: Build a dynamic k-NN graph over patch embeddings, apply graph
attention to aggregate neighbourhood information, then pool to bag level.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._utils import make_patch_embed


class WIKG(nn.Module):
    """Weakly-supervised Instance Knowledge Graph MIL.

    Constructs a k-nearest-neighbour graph in embedding space and uses
    graph attention convolution to propagate contextual information
    between neighbouring patches before bag-level pooling.

    Args:
        in_dim: Input feature dimension.
        embed_dim: Embedding dimension.
        num_classes: Number of output classes.
        moe_args: Optional MAMMOTH config.
        k_neighbors: Number of neighbours in the k-NN graph (default 8).
        attn_dim: Gated-attention hidden dimension (default 128).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_classes: int,
        moe_args: dict | None = None,
        plugins: list[dict] | None = None,
        k_neighbors: int = 8,
        attn_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.k = k_neighbors

        self.patch_embed = make_patch_embed(in_dim, embed_dim, moe_args, plugins=plugins)
        self.dropout = nn.Dropout(dropout)

        # Graph attention projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)

        # Gated attention for final bag-level pooling
        self.attn_V = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def _build_knn_graph(self, h: torch.Tensor) -> torch.Tensor:
        """Build k-NN adjacency from cosine similarity. Returns (B, N, N) sparse-like."""
        # h: (B, N, D)
        h_norm = F.normalize(h, dim=-1)
        sim = torch.bmm(h_norm, h_norm.transpose(1, 2))   # (B, N, N)
        # Zero out self-connections
        B, N, _ = sim.shape
        eye = torch.eye(N, device=h.device).unsqueeze(0).bool()
        sim = sim.masked_fill(eye, -1.0)
        # Keep top-k neighbours per node
        _, top_idx = torch.topk(sim, self.k, dim=-1)      # (B, N, k)
        mask = torch.zeros_like(sim, dtype=torch.bool)
        mask.scatter_(-1, top_idx, True)
        # Sparse adjacency: keep only top-k edges
        adj = sim.masked_fill(~mask, float("-inf"))
        adj = torch.softmax(adj, dim=-1)
        adj = torch.nan_to_num(adj, nan=0.0)               # handle all-inf rows
        return adj

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        h = self.dropout(self.patch_embed(x))              # (B, N, embed_dim)

        # Graph attention propagation
        adj = self._build_knn_graph(h)                     # (B, N, N)
        Q = self.W_q(h)
        V = self.W_v(h)
        # Aggregate neighbour values weighted by adjacency
        h_agg = torch.bmm(adj, V)                          # (B, N, embed_dim)
        h = self.layer_norm(h + h_agg)                     # residual + norm

        # Gated attention for bag pooling
        A_V = self.attn_V(h)
        A_U = self.attn_U(h)
        A = torch.softmax(self.attn_w(A_V * A_U), dim=1)  # (B, N, 1)
        bag = torch.sum(A * h, dim=1)                      # (B, embed_dim)

        logits = self.classifier(bag)
        if return_attention:
            return logits, A.squeeze(-1).squeeze(0)  # (N,)
        return logits
