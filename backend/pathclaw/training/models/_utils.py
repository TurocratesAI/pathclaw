"""Shared utilities for MIL models — dispatches patch-embed to the plugin registry."""

from __future__ import annotations

import torch.nn as nn

from pathclaw.plugins import build_patch_embed


def make_patch_embed(
    in_dim: int,
    embed_dim: int,
    moe_args: dict | None = None,
    plugins: list[dict] | None = None,
) -> nn.Module:
    """Return a patch-embed module from the plugin registry.

    Compatibility:
    - `plugins`: new-style list of {"id": str, "config": dict} entries.
    - `moe_args`: legacy `config.mammoth` dict. If it has `enabled: true` it's
      auto-translated into a `{"id": "mammoth", "config": ...}` plugin entry.

    If no patch-embed plugin resolves, returns a plain nn.Linear baseline (the
    same fallback the old make_patch_embed used).
    """
    return build_patch_embed(in_dim, embed_dim, plugins, legacy_mammoth=moe_args)
