"""
PyTorch U-Net 
-------------------------

Features
- Classic U-Net (Ronneberger et al., 2015)


Notes
-----
* We use padding=1 for 3x3 convolutions so spatial sizes are preserved across convs.
* Skip connections are concatenations along the channel dimension (dim=1).
"""
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Helper blocks
# -----------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2

    This is the basic unit used in both the encoder and decoder.
    Padding=1 keeps H and W the same after each 3x3 convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv.

    Reduces H and W by 2 using MaxPool2d, then expands feature channels with DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Upscaling then DoubleConv.

    Two upsampling options:
    - bilinear: use F.interpolate (fast, memory-light). We follow with a 1x1 conv to
      reduce channels of the skip-connection input if needed.
    - transposed conv: learnable upsampling that can also learn to reduce checkerboards.

    After upsampling, we concatenate the skip feature from the encoder (same spatial size)
    along the channel dimension, then apply DoubleConv to fuse them.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            # When concatenating, channels become (in_channels from up) + (skip channels).
            # We don't know skip channels here, so we'll reduce upsampled channels by half
            # so that after concat the DoubleConv sees roughly a manageable number.
            # Conventionally in U-Net: in_channels is 2 * out_channels at each up step.
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            in_channels = in_channels // 2  # after reduce
        else:
            # Transposed convolution doubles spatial resolution and reduces channels.
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.reduce = nn.Identity()
            in_channels = in_channels // 2

        # DoubleConv will receive concatenated [up(x), skip] -> channels = in_channels + skip_channels
        self.conv = DoubleConv(in_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)            # upsample decoder feature
        x = self.reduce(x)        # ensure channel alignment with design

        # In rare cases, due to odd sizes, upsampled tensor may be off by 1px.
        # We pad/crop to match spatial size of the skip tensor exactly.
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)

        # Concatenate along channel dimension: [N, C_up, H, W] + [N, C_skip, H, W] -> [N, C_up+C_skip, H, W]
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """Final 1x1 convolution to map features -> class logits per pixel."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -----------------------------------------------------------------------------
# Classic U-Net
# -----------------------------------------------------------------------------
class UNet(nn.Module):
    """Classic U-Net for semantic segmentation.

    Architecture (channels shown for base_channels=64):
    Encoder:   3 -> 64 -> 128 -> 256 -> 512 -> 1024 (bottleneck)
    Decoder:   1024 -> 512 -> 256 -> 128 -> 64 -> num_classes

    Args
    ----
    in_channels:   input channels (e.g., 3 for RGB)
    num_classes:   number of output classes (logits per pixel)
    base_channels: channel width multiplier at the first level
    bilinear:      if True use bilinear upsampling; otherwise use transposed conv
    """
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 64, bilinear: bool = True):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.up1 = Up(base_channels * 16, base_channels * 8, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path: capture context at multiple scales
        x1 = self.inc(x)          # [N, C, H, W]
        x2 = self.down1(x1)       # [N, 2C, H/2, W/2]
        x3 = self.down2(x2)       # [N, 4C, H/4, W/4]
        x4 = self.down3(x3)       # [N, 8C, H/8, W/8]
        x5 = self.down4(x4)       # [N,16C, H/16, W/16]  (bottleneck)

        # Decoder path: progressively upsample and fuse with encoder features (skip connections)
        d1 = self.up1(x5, x4)     # [N, 8C, H/8,  W/8]
        d2 = self.up2(d1, x3)     # [N, 4C, H/4,  W/4]
        d3 = self.up3(d2, x2)     # [N, 2C, H/2,  W/2]
        d4 = self.up4(d3, x1)     # [N,  C, H,    W]

        logits = self.outc(d4)    # [N, num_classes, H, W]
        return logits


