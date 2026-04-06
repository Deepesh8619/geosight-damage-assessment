"""
CBAM: Convolutional Block Attention Module.

WHY THIS IMPROVES ACCURACY:
----------------------------
Standard U-Net treats all pixels and channels equally. But in satellite images:
  - Some channels carry more information (RGB vs near-infrared)
  - Some spatial regions matter more (building edges vs open ground)

CBAM adds two attention mechanisms:
  1. Channel Attention: "Which feature channels are most important?"
     → Learns that edge-detection channels matter more than color channels
  2. Spatial Attention: "Which pixel locations should I focus on?"
     → Learns to focus on building boundaries, not empty sky

Published: Woo et al., ECCV 2018 — "CBAM: Convolutional Block Attention Module"
Expected improvement: +3-5% IoU on segmentation tasks

This is plugged into the U-Net decoder between each upsampling block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    "Which feature channels are most important for this input?"

    Takes feature map (B, C, H, W), produces channel weight (B, C, 1, 1).

    Algorithm:
      1. Global average pool → (B, C, 1, 1) — "what's the average activation?"
      2. Global max pool → (B, C, 1, 1) — "what's the strongest activation?"
      3. Both go through a shared 2-layer MLP: C → C/r → C
      4. Sum the two MLP outputs
      5. Sigmoid → weight per channel (0 to 1)
      6. Multiply original features by these weights

    Why both avg AND max pooling?
      Avg pool captures the "typical" importance of each channel
      Max pool captures the "peak" importance
      Together they give a more complete picture
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Global pooling: (B, C, H, W) → (B, C)
        avg_pool = x.mean(dim=(2, 3))                    # average across spatial dims
        max_pool = x.amax(dim=(2, 3))                    # max across spatial dims

        # Shared MLP
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        # Combine and apply sigmoid
        weights = torch.sigmoid(avg_out + max_out)        # (B, C)
        weights = weights.unsqueeze(-1).unsqueeze(-1)     # (B, C, 1, 1)

        return x * weights


class SpatialAttention(nn.Module):
    """
    "Which pixel locations should I focus on?"

    Takes feature map (B, C, H, W), produces spatial weight (B, 1, H, W).

    Algorithm:
      1. Channel-wise average → (B, 1, H, W) — "how active is this location on average?"
      2. Channel-wise max → (B, 1, H, W) — "how active is the most active channel here?"
      3. Concatenate → (B, 2, H, W)
      4. 7×7 convolution → (B, 1, H, W)
      5. Sigmoid → spatial weight map (0 to 1)
      6. Multiply original features by this spatial map

    7×7 kernel because building edges need neighborhood context to be identified.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)     # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)     # (B, 1, H, W)

        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        weights  = torch.sigmoid(self.conv(combined))     # (B, 1, H, W)

        return x * weights


class CBAM(nn.Module):
    """
    Full CBAM module: Channel Attention → Spatial Attention (sequential).

    Usage:
        cbam = CBAM(channels=64)
        features = cbam(features)   # same shape in and out, but attention-weighted
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)     # first: re-weight channels
        x = self.spatial_attn(x)     # then: re-weight spatial locations
        return x


class AttentionUNetDecoder(nn.Module):
    """
    U-Net decoder block with CBAM attention.

    Standard decoder: upsample → concat skip → conv → conv
    Attention decoder: upsample → concat skip → CBAM → conv → conv

    The CBAM is inserted AFTER the skip connection so it can learn
    which parts of the skip features are relevant for the current
    decoding level.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attention = CBAM(in_channels + skip_channels)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch (if input isn't perfectly divisible)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

        x = torch.cat([x, skip], dim=1)     # concat along channel dim
        x = self.attention(x)                # CBAM attention on fused features
        x = self.conv_block(x)              # standard convolutions
        return x
