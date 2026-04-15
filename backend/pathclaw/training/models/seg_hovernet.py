"""HoVer-Net instance segmentation model.

Reference: Graham et al., Medical Image Analysis 2019 —
"HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in
Multi-Tissue Histology Images"

Architecture:
  - Shared ResNet-50 encoder (pretrained from torchvision)
  - Three output branches:
      NP  — Nuclear Pixel (binary): is this pixel inside a nucleus?
      HV  — Horizontal-Vertical distance maps: distance to nucleus centroid
      NC  — Nuclear Class (optional multi-class nucleus type classification)
  - Post-processing: watershed on distance maps to separate touching nuclei

Output dict:
    {
        "np":   (B, 2, H, W)   — nuclear pixel logits
        "hv":   (B, 2, H, W)   — horizontal & vertical distance maps
        "nc":   (B, num_types, H, W)  — per-type logits (only if num_types > 1)
    }
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Decoder block shared by all branches
# ---------------------------------------------------------------------------

class _BranchDecoder(nn.Module):
    """4× upsampling decoder head for one HoVer-Net branch."""

    def __init__(self, enc_chs: list[int], out_ch: int) -> None:
        super().__init__()
        # Dense residual blocks (DRB) simplified to standard conv blocks
        self.up4 = self._make_up(enc_chs[3] + enc_chs[2], 256)
        self.up3 = self._make_up(256 + enc_chs[1], 128)
        self.up2 = self._make_up(128 + enc_chs[0], 64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 1),
        )

    @staticmethod
    def _make_up(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats: [f1(B,C1,H/2,W/2), f2(B,C2,H/4,W/4), f3(B,C3,H/8,W/8), f4(B,C4,H/16,W/16)]
        f1, f2, f3, f4 = feats

        x = self.up4(torch.cat([f4, f3], dim=1))          # H/8
        x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, f2], dim=1))            # H/4
        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, f1], dim=1))            # H/2
        x = F.interpolate(x, size=(f1.shape[2] * 2, f1.shape[3] * 2),
                          mode='bilinear', align_corners=False)
        return self.head(x)                                 # H


# ---------------------------------------------------------------------------
# HoVer-Net
# ---------------------------------------------------------------------------

class HoVerNet(nn.Module):
    """HoVer-Net for simultaneous nucleus segmentation and classification.

    Args:
        num_types: Number of nucleus types (default 1 = binary, no NC branch).
        pretrained_encoder: Use ImageNet-pretrained ResNet-50 as encoder.
    """

    # ResNet-50 layer output channels
    _ENC_CHS = [256, 512, 1024, 2048]

    def __init__(
        self,
        num_types: int = 1,
        pretrained_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.num_types = num_types

        # Encoder: ResNet-50 stem + 4 stages
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained_encoder else None)
        except ImportError:
            # Fallback for older torchvision
            from torchvision.models import resnet50
            backbone = resnet50(pretrained=pretrained_encoder)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # (B, 256,  H/4,  W/4)
        self.layer2 = backbone.layer2  # (B, 512,  H/8,  W/8)
        self.layer3 = backbone.layer3  # (B, 1024, H/16, W/16)
        self.layer4 = backbone.layer4  # (B, 2048, H/32, W/32)

        # Reduce encoder channels to fit decoders
        self.reduce = nn.ModuleList([
            nn.Conv2d(self._ENC_CHS[0], 128, 1),
            nn.Conv2d(self._ENC_CHS[1], 256, 1),
            nn.Conv2d(self._ENC_CHS[2], 512, 1),
            nn.Conv2d(self._ENC_CHS[3], 512, 1),
        ])
        reduced = [128, 256, 512, 512]

        self.np_branch = _BranchDecoder(reduced, out_ch=2)         # 2-class nuclear pixel
        self.hv_branch = _BranchDecoder(reduced, out_ch=2)         # H, V distance maps
        if num_types > 1:
            self.nc_branch = _BranchDecoder(reduced, out_ch=num_types)
        else:
            self.nc_branch = None

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        feats = [self.reduce[i](f) for i, f in enumerate([f1, f2, f3, f4])]
        return feats

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self._encode(x)
        out = {
            "np": self.np_branch(feats),
            "hv": self.hv_branch(feats),
        }
        if self.nc_branch is not None:
            out["nc"] = self.nc_branch(feats)
        return out
