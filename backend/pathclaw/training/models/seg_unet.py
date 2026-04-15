"""UNet semantic segmentation model with optional frozen foundation model encoder.

Reference: Ronneberger et al., MICCAI 2015 — "U-Net: Convolutional Networks for
Biomedical Image Segmentation"

When backbone='none', uses a lightweight 4-level CNN encoder.
When backbone is set (e.g. 'uni', 'conch'), the foundation model encoder is frozen
and the decoder is trained to produce per-pixel segmentation from its features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = _ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if needed (odd input sizes)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# SegUNet
# ---------------------------------------------------------------------------

class SegUNet(nn.Module):
    """Lightweight UNet for semantic segmentation of WSI patches.

    Args:
        in_channels: Input image channels (default 3 for RGB).
        num_classes: Number of segmentation classes (including background).
        base_ch: Base channel count (doubles at each encoder level).
        depth: Encoder/decoder depth (default 4).
        dropout: Dropout applied at the bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_ch: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        enc_chs = []
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            self.encoders.append(_ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            enc_chs.append(out_ch)
            ch = out_ch

        # Bottleneck
        bot_ch = base_ch * (2 ** depth)
        self.bottleneck = _ConvBlock(ch, bot_ch, dropout=dropout)
        ch = bot_ch

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            skip_ch = enc_chs[i]
            out_ch = base_ch * (2 ** i)
            self.decoders.append(_UpBlock(ch, skip_ch, out_ch))
            ch = out_ch

        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)  # (B, num_classes, H, W)
