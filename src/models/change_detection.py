"""
Damage Classification Model (Project 5 component).

A U-Net that takes a 6-channel input (pre + post satellite image concatenated)
and outputs a per-pixel 5-class damage map:
  0: background
  1: no-damage
  2: minor-damage
  3: major-damage
  4: destroyed

This is a change-aware segmentation model — by seeing both pre and post,
it learns to distinguish damage from pre-existing structures.
"""

from typing import List, Optional, Tuple

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


class DamageClassificationModel(nn.Module):
    """
    Dual-image U-Net for per-pixel building damage classification.

    The pre and post images are concatenated channel-wise as input.
    This allows the encoder to learn cross-temporal features that
    capture what changed (and by how much) between the two dates.

    Input:  (B, 6, H, W)  — pre (3ch) + post (3ch)
    Output: (B, 5, H, W)  — raw logits per class
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = None,   # no pretrained for 6ch input
        in_channels: int = 6,
        num_classes: int = NUM_DAMAGE_CLASSES,
        decoder_channels: Tuple = (256, 128, 64, 32, 16),
    ):
        super().__init__()

        self.num_classes = num_classes

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=True,
        )

        # Re-initialise the first conv layer when in_channels != 3
        # to use averaged ImageNet weights for the first 3 channels
        if in_channels != 3 and encoder_weights == "imagenet":
            self._adapt_first_conv(encoder_name, in_channels)

        logger.info(
            f"DamageClassificationModel: encoder={encoder_name}, "
            f"in_channels={in_channels}, classes={num_classes}"
        )

    def _adapt_first_conv(self, encoder_name: str, in_channels: int):
        """
        Adapt pretrained encoder's first conv to accept in_channels.
        New channels are initialised as the mean of the 3 original channels.
        """
        try:
            import segmentation_models_pytorch as smp
            pretrained = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=self.num_classes,
            )
            old_weight = pretrained.encoder.conv1.weight.data   # (64, 3, 7, 7)
            new_weight = old_weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            self.model.encoder.conv1.weight.data = new_weight
            logger.debug("Adapted first conv weights from ImageNet pretrained (mean init).")
        except Exception as e:
            logger.warning(f"Could not adapt first conv weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) — channels 0-2 are pre, channels 3-5 are post

        Returns:
            logits: (B, num_classes, H, W)
        """
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class index map (B, H, W) as int64."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probability map (B, num_classes, H, W)."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class WeightedFocalLoss(nn.Module):
    """
    Focal loss with per-class weights.
    Addresses class imbalance (most pixels are background or no-damage).
    gamma > 0 down-weights easy examples, focusing training on hard cases.
    """

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W)
            targets: (B, H, W) int64
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt       = torch.exp(-ce_loss)
        focal    = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


class CombinedDamageLoss(nn.Module):
    """
    Focal + Lovász-softmax loss combination.
    Lovász directly optimises the intersection-over-union metric.
    Falls back to cross-entropy if lovász is not installed.
    """

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        focal_weight: float = 0.6,
        lovasz_weight: float = 0.4,
    ):
        super().__init__()
        self.focal_weight  = focal_weight
        self.lovasz_weight = lovasz_weight
        self.focal = WeightedFocalLoss(class_weights=class_weights)

        try:
            from segmentation_models_pytorch.losses import LovaszLoss
            self.lovasz = LovaszLoss(mode="multiclass")
            self._use_lovasz = True
        except ImportError:
            logger.warning("LovaszLoss not available; using CrossEntropy instead.")
            self.lovasz      = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights) if class_weights else None
            )
            self._use_lovasz = False

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.focal_weight * self.focal(logits, targets)
        if self._use_lovasz:
            loss += self.lovasz_weight * self.lovasz(logits, targets)
        else:
            loss += self.lovasz_weight * self.lovasz(logits, targets)
        return loss


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_damage_optimizer(
    model: DamageClassificationModel,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
