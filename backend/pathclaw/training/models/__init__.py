"""MIL model registry and shared utilities."""

from __future__ import annotations

import torch.nn as nn

from ._utils import make_patch_embed
from .abmil import ABMIL
from .meanpool import MeanPoolMIL
from .transmil import TransMIL
from .clam import CLAM_SB
from .dsmil import DSMIL
from .rrtmil import RRTMIL
from .wikg import WIKG
from .seg_unet import SegUNet
from .seg_hovernet import HoVerNet
from .patch_mlp import PatchMLP

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "abmil": ABMIL,
    "meanpool": MeanPoolMIL,
    "transmil": TransMIL,
    "clam": CLAM_SB,
    "dsmil": DSMIL,
    "rrtmil": RRTMIL,
    "wikg": WIKG,
}

SEG_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "seg_unet": SegUNet,
    "hovernet": HoVerNet,
    # "cellpose" is handled separately (pre-trained inference-only wrapper)
}

PATCH_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "patch_mlp": PatchMLP,
}


__all__ = [
    "MODEL_REGISTRY",
    "SEG_MODEL_REGISTRY",
    "PATCH_MODEL_REGISTRY",
    "make_patch_embed",
    "ABMIL",
    "MeanPoolMIL",
    "TransMIL",
    "CLAM_SB",
    "DSMIL",
    "RRTMIL",
    "WIKG",
    "SegUNet",
    "HoVerNet",
    "PatchMLP",
]
