"""CLAM — Clustering-constrained Attention Multiple Instance Learning (single-branch).

Reference: Lu et al., Nature Biomedical Engineering 2021
"Data-efficient and weakly supervised computational pathology on whole-slide images"

Single-branch variant (CLAM-SB): one attention branch + instance-level clustering loss.
The clustering loss is computed here but weighting is controlled by cluster_weight.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._utils import make_patch_embed


class CLAM_SB(nn.Module):
    """CLAM Single-Branch.

    Args:
        in_dim: Input feature dimension.
        embed_dim: Hidden dimension.
        num_classes: Number of output classes.
        moe_args: Optional MAMMOTH config.
        attn_dim: Attention hidden dimension (default 128).
        k_sample: Number of top/bottom patches sampled for instance loss (default 8).
        dropout: Dropout on the attention pathway (default 0.25).
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_classes: int,
        moe_args: dict | None = None,
        plugins: list[dict] | None = None,
        attn_dim: int = 128,
        k_sample: int = 8,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.k_sample = k_sample
        self.num_classes = num_classes

        self.patch_embed = make_patch_embed(in_dim, embed_dim, moe_args, plugins=plugins)
        self.dropout = nn.Dropout(dropout)

        # Gated attention
        self.attention_V = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attn_dim, 1)

        # Per-class instance classifiers (for clustering loss)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(embed_dim, 2) for _ in range(num_classes)]
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def _instance_loss(
        self,
        h: torch.Tensor,
        A: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Smooth-SVM instance-level clustering loss."""
        c = label.item()
        top_k = min(self.k_sample, h.shape[1])

        A_squeezed = A.squeeze(-1)                            # (B, N)
        _, top_idx = torch.topk(A_squeezed, top_k, dim=1)    # top-attending patches
        _, bot_idx = torch.topk(-A_squeezed, top_k, dim=1)   # bottom-attending patches

        top_h = h[:, top_idx.squeeze(), :]                    # (B, k, D)
        bot_h = h[:, bot_idx.squeeze(), :]

        inst_clf = self.instance_classifiers[c]

        top_logits = inst_clf(top_h.squeeze(0))               # (k, 2)
        bot_logits = inst_clf(bot_h.squeeze(0))

        top_labels = torch.ones(top_k, dtype=torch.long, device=h.device)
        bot_labels = torch.zeros(top_k, dtype=torch.long, device=h.device)

        loss = (
            F.cross_entropy(top_logits, top_labels)
            + F.cross_entropy(bot_logits, bot_labels)
        ) / 2
        return loss

    def forward(
        self,
        x: torch.Tensor,
        label: torch.Tensor | None = None,
        instance_loss_weight: float = 0.3,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        h = self.dropout(self.patch_embed(x))       # (B, N, embed_dim)

        A_V = self.attention_V(h)
        A_U = self.attention_U(h)
        A = self.attention_w(A_V * A_U)             # (B, N, 1)
        A = torch.softmax(A, dim=1)

        M = torch.sum(A * h, dim=1)                 # (B, embed_dim)
        logits = self.classifier(M)

        if return_attention:
            return logits, A.squeeze(-1).squeeze(0)  # (N,)

        if label is not None and self.training:
            inst_loss = self._instance_loss(h, A, label)
            return logits, inst_loss

        return logits
