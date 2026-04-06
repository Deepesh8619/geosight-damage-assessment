"""
Siamese U-Net for Building Damage Classification.

WHY THIS EXISTS (vs the old 6-channel concatenation):
------------------------------------------------------
The old DamageClassificationModel naively concatenated pre+post images into
6 channels and fed them to a single encoder. This has two problems:

  1. The encoder cannot use ImageNet pretrained weights (designed for 3ch)
  2. It treats pre and post as a single fused signal — there's no explicit
     mechanism to compare "what was" vs "what is now"

The Siamese architecture fixes both:
  - Two encoders share weights, each processes 3-channel images independently
  - Both encoders start from ImageNet pretrained weights (better features)
  - Feature difference maps are computed at each encoder level
  - The decoder sees both the original features AND the change signal

This is the approach used by the top-3 solutions in the xView2 competition
(2019), which achieved F1 scores of 0.80+ vs ~0.65 for naive concatenation.

Architecture:
                Pre-image          Post-image
                    │                   │
              ┌─────▼─────┐       ┌─────▼─────┐
              │  Encoder   │◄────►│  Encoder   │  (shared weights)
              │  ResNet34  │       │  ResNet34  │
              └─┬───┬───┬─┘       └─┬───┬───┬─┘
                │   │   │           │   │   │
                │   │   │    diff   │   │   │
            f1_pre f2  f3      f1_post f2  f3
                │   │   │           │   │   │
                └───┼───┼─── ⊖ ────┼───┘   │
                    │   │  change   │       │
                    │   │  features │       │
              ┌─────▼───▼───▼──────▼───────▼──┐
              │         U-Net Decoder          │
              │   (sees pre + post + change)   │
              └───────────────┬────────────────┘
                              │
                    5-class damage map
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from loguru import logger


DAMAGE_CLASSES = {
    0: "background",
    1: "no-damage",
    2: "minor-damage",
    3: "major-damage",
    4: "destroyed",
}

NUM_DAMAGE_CLASSES = len(DAMAGE_CLASSES)


class FeatureDifferenceModule(nn.Module):
    """
    Computes multi-scale difference features between pre and post encoder outputs.

    At each encoder level, produces:
      - Absolute difference:  |f_pre - f_post|  (captures magnitude of change)
      - Element-wise product: f_pre * f_post     (captures correlation/similarity)

    These are concatenated and projected to the original channel count.
    """

    def __init__(self, channel_sizes: List[int]):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 3, ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            )
            for ch in channel_sizes
        ])

    def forward(
        self, pre_features: List[torch.Tensor], post_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            pre_features:  list of (B, C_i, H_i, W_i) encoder feature maps
            post_features: list of (B, C_i, H_i, W_i) encoder feature maps

        Returns:
            fused: list of (B, C_i, H_i, W_i) — same shapes, change-aware
        """
        fused = []
        for i, (f_pre, f_post, proj) in enumerate(
            zip(pre_features, post_features, self.projections)
        ):
            diff     = torch.abs(f_pre - f_post)
            combined = torch.cat([f_pre, f_post, diff], dim=1)  # 3*C channels
            fused.append(proj(combined))
        return fused


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for per-pixel building damage classification.

    Uses a shared-weight encoder (pretrained ResNet34) to independently
    extract features from pre and post-disaster images, computes
    multi-scale difference features, and decodes to a 5-class damage map.

    Inputs:
        pre_image:  (B, 3, H, W) — pre-disaster satellite image
        post_image: (B, 3, H, W) — post-disaster satellite image

    Output:
        logits: (B, 5, H, W) — per-pixel damage class logits
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        num_classes: int = NUM_DAMAGE_CLASSES,
        decoder_channels: Tuple = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        self.num_classes = num_classes

        # Create a full U-Net just to extract the encoder and decoder architecture
        _template = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=True,
        )

        # Shared encoder — both pre and post images go through the same weights
        self.encoder = _template.encoder

        # Get encoder output channel sizes for the difference module
        # ResNet34 encoder outputs: [64, 64, 128, 256, 512] (5 levels)
        dummy = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            encoder_outs = self.encoder(dummy)
        self.encoder_channels = [f.shape[1] for f in encoder_outs]

        # Feature difference module
        self.diff_module = FeatureDifferenceModule(self.encoder_channels)

        # Decoder — takes change-fused features and produces damage map
        self.decoder = _template.decoder
        self.segmentation_head = _template.segmentation_head

        del _template

        logger.info(
            f"SiameseUNet: encoder={encoder_name}, "
            f"pretrained={encoder_weights is not None}, "
            f"encoder_channels={self.encoder_channels}, "
            f"classes={num_classes}, params={self.n_parameters:,}"
        )

    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pre_image:  (B, 3, H, W)
            post_image: (B, 3, H, W)

        Returns:
            logits: (B, num_classes, H, W)
        """
        # Shared encoder extracts features at multiple scales
        pre_features  = self.encoder(pre_image)
        post_features = self.encoder(post_image)

        # Compute change-aware features at each scale
        fused_features = self.diff_module(pre_features, post_features)

        # Decode to damage map
        # smp's UnetDecoder.forward() expects a single list of feature tensors
        decoder_output = self.decoder(fused_features)
        logits = self.segmentation_head(decoder_output)

        return logits

    def forward_concatenated(self, x: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible forward that accepts (B, 6, H, W) input.
        Splits channels 0-2 as pre, 3-5 as post.
        """
        pre  = x[:, :3, :, :]
        post = x[:, 3:, :, :]
        return self.forward(pre, post)

    def predict(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Returns class index map (B, H, W) as int64."""
        with torch.no_grad():
            logits = self.forward(pre, post)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Returns softmax probability map (B, num_classes, H, W)."""
        with torch.no_grad():
            logits = self.forward(pre, post)
            return F.softmax(logits, dim=1)

    def predict_with_confidence(
        self, pre: torch.Tensor, post: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns both class prediction and per-pixel confidence.

        Why confidence matters:
          In disaster response, false positives waste rescue teams.
          A confidence map lets responders prioritise high-certainty
          destroyed buildings over uncertain ones.

        Returns:
            classes:    (B, H, W) int64 — predicted damage class
            confidence: (B, H, W) float32 — probability of predicted class (0-1)
        """
        with torch.no_grad():
            probs      = self.predict_proba(pre, post)
            confidence = probs.max(dim=1).values
            classes    = probs.argmax(dim=1)
            return classes, confidence

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_siamese_optimizer(
    model: SiameseUNet,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Differential LR: encoder at 0.1x (pretrained), rest at 1x."""
    encoder_params = list(model.encoder.parameters())
    other_params = [
        p for n, p in model.named_parameters()
        if "encoder" not in n
    ]
    return torch.optim.AdamW([
        {"params": encoder_params, "lr": lr * 0.1},
        {"params": other_params,   "lr": lr},
    ], weight_decay=weight_decay)
