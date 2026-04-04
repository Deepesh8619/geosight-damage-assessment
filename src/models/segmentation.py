"""
Building Footprint Segmentation Model (Project 3 component).

U-Net with a pretrained ResNet34 encoder trained to segment
building footprints from single-date satellite imagery.
This model runs on pre-disaster images and produces a binary mask
that feeds into the damage assessment pipeline.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from loguru import logger


class BuildingSegmentationModel(nn.Module):
    """
    U-Net with ResNet34 encoder for binary building segmentation.

    Input:  (B, C, H, W) float32 satellite image tile
    Output: (B, 1, H, W) raw logits (apply sigmoid for probabilities)
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,              # raw logits; sigmoid applied in loss/inference
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=True,
        )

        logger.info(
            f"BuildingSegmentationModel: encoder={encoder_name}, "
            f"in_channels={in_channels}, pretrained={encoder_weights is not None}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_mask(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Returns binary mask (0/1) from logits."""
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.sigmoid(logits)
            return (probs > threshold).float()

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_f   = probs.view(-1)
        targets_f = targets.view(-1).float()

        intersection = (probs_f * targets_f).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (
            probs_f.sum() + targets_f.sum() + self.smooth
        )


class BceDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss — standard for satellite segmentation tasks.
    BCE handles pixel-level accuracy; Dice handles class imbalance.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_f = targets.float()
        return (
            self.bce_weight  * self.bce(logits, targets_f) +
            self.dice_weight * self.dice(logits, targets_f)
        )


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def build_segmentation_optimizer(
    model: BuildingSegmentationModel,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Separate learning rates for encoder (lower) and decoder (higher).
    Common practice when fine-tuning pretrained encoders.
    """
    encoder_params = list(model.model.encoder.parameters())
    decoder_params = [
        p for name, p in model.named_parameters()
        if "encoder" not in name
    ]
    return torch.optim.AdamW([
        {"params": encoder_params, "lr": lr * 0.1},
        {"params": decoder_params, "lr": lr},
    ], weight_decay=weight_decay)


def build_lr_scheduler(optimizer, epochs: int, warmup_epochs: int = 5):
    """Cosine annealing with linear warmup."""
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )
    warmup   = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine   = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
