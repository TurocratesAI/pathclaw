"""Cellpose nuclei/cell segmentation wrapper.

Cellpose is a pretrained generalist segmentation model. We expose it as a
`method` plugin: the agent or user picks a slide directory, configures the
knobs, and we run inference per-patch writing PNG masks + instance counts.

Config schema (what gemma can edit):

    model_type        — "cpsam" (SAM backbone, default), "nuclei", "cyto3"
    diameter          — target object diameter in px (None = auto-estimate)
    flow_threshold    — 0–1; lower = stricter flow consistency
    cellprob_threshold — cell-probability threshold; raise to reject dim objects
    niter             — iterations for dynamics (raise for large/crowded cells)
    channels          — [cyto_channel, nucleus_channel]; 0=grayscale, 1=R, 2=G, 3=B
    min_size          — reject masks smaller than this (px)
    tile_norm         — per-tile normalization (recommended for WSI patches)
    gpu               — use CUDA if available
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONFIG: dict[str, Any] = {
    "model_type": "cpsam",
    "diameter": None,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "niter": 0,
    "channels": [0, 0],
    "min_size": 15,
    "tile_norm": True,
    "gpu": True,
}


def _resolve_model(model_type: str, gpu: bool):
    """Instantiate a Cellpose model from one of the supported type strings."""
    from cellpose import models
    import torch

    use_gpu = bool(gpu and torch.cuda.is_available())
    if model_type == "cpsam":
        return models.CellposeModel(gpu=use_gpu)
    return models.CellposeModel(gpu=use_gpu, model_type=model_type)


def run_on_images(images: list[np.ndarray], config: dict | None = None) -> dict[str, Any]:
    """Run cellpose on a list of RGB uint8 patches.

    Returns {"masks": list[ndarray], "num_objects": list[int], "flows": ...}.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    model = _resolve_model(cfg["model_type"], cfg["gpu"])
    masks, flows, _styles = model.eval(
        images,
        diameter=cfg["diameter"],
        flow_threshold=cfg["flow_threshold"],
        cellprob_threshold=cfg["cellprob_threshold"],
        niter=cfg["niter"],
        channels=cfg["channels"],
        min_size=cfg["min_size"],
        normalize={"tile_norm_blocksize": 128} if cfg["tile_norm"] else True,
    )
    counts = [int(m.max()) for m in masks]
    return {"masks": masks, "num_objects": counts, "flows": flows}


def build(in_dim: int, embed_dim: int, config: dict | None = None):
    """Plugin-contract shim.

    Cellpose isn't a patch-embed — this is here so the plugin registry can
    construct *something* callable for the smoke test. The callable just
    returns a lazy wrapper that exposes `run(images)`.
    """
    import torch.nn as nn

    cfg = {**DEFAULT_CONFIG, **(config or {})}

    class CellposeRunner(nn.Module):
        def __init__(self, config: dict):
            super().__init__()
            self.config = config
            self._model = None

        def ensure(self):
            if self._model is None:
                self._model = _resolve_model(self.config["model_type"], self.config["gpu"])
            return self._model

        def forward(self, x):
            # Segmentation model — a patch-embed forward doesn't apply. We
            # intentionally fail loudly if someone plugs this into an MIL pipe.
            raise RuntimeError(
                "Cellpose is a segmentation method, not a patch embed. Call "
                ".run(images) with RGB uint8 patches instead."
            )

        def run(self, images: list[np.ndarray]):
            return run_on_images(images, self.config)

    return CellposeRunner(cfg)
