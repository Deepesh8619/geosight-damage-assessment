"""
Test Time Augmentation (TTA).

WHY THIS EXISTS:
----------------
During training, we flip/rotate images randomly. The model learns to handle
all orientations. But at INFERENCE time, we only feed the original orientation.

TTA exploits this: feed the image in multiple orientations, get predictions
for each, then AVERAGE all predictions. The average is more accurate than
any single prediction because random errors cancel out.

It's like asking 8 doctors instead of 1 — the consensus is more reliable.

Expected improvement: +2-3% IoU with ZERO retraining.
Cost: 8x slower inference (but still seconds, not minutes).

Technique: Geometric ensemble (flip + rotate variations)
Published: Standard practice in satellite image segmentation competitions
"""

from typing import List, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


# The 8 geometric transforms (all combinations of flip + rotate90)
TTA_TRANSFORMS = [
    {"name": "original",    "fn": lambda x: x,                                    "inv": lambda x: x},
    {"name": "hflip",       "fn": lambda x: torch.flip(x, [-1]),                  "inv": lambda x: torch.flip(x, [-1])},
    {"name": "vflip",       "fn": lambda x: torch.flip(x, [-2]),                  "inv": lambda x: torch.flip(x, [-2])},
    {"name": "hvflip",      "fn": lambda x: torch.flip(x, [-1, -2]),              "inv": lambda x: torch.flip(x, [-1, -2])},
    {"name": "rot90",       "fn": lambda x: torch.rot90(x, 1, [-2, -1]),          "inv": lambda x: torch.rot90(x, 3, [-2, -1])},
    {"name": "rot180",      "fn": lambda x: torch.rot90(x, 2, [-2, -1]),          "inv": lambda x: torch.rot90(x, 2, [-2, -1])},
    {"name": "rot270",      "fn": lambda x: torch.rot90(x, 3, [-2, -1]),          "inv": lambda x: torch.rot90(x, 1, [-2, -1])},
    {"name": "rot90_hflip", "fn": lambda x: torch.flip(torch.rot90(x, 1, [-2, -1]), [-1]),
                            "inv": lambda x: torch.rot90(torch.flip(x, [-1]), 3, [-2, -1])},
]


class TTAPredictor:
    """
    Wraps any segmentation model with Test Time Augmentation.

    Usage:
        tta = TTAPredictor(model, n_augments=8)
        prediction = tta.predict(image)   # averaged over 8 orientations
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_augments: int = 8,
        mode: str = "segmentation",   # "segmentation" | "classification"
    ):
        """
        Args:
            model: trained model (in eval mode)
            n_augments: how many TTA variants (1=none, 4=flips only, 8=all)
            mode: "segmentation" for per-pixel, "classification" for per-image
        """
        self.model = model
        self.transforms = TTA_TRANSFORMS[:n_augments]
        self.mode = mode
        logger.info(f"TTA initialized with {n_augments} augmentations")

    @torch.no_grad()
    def predict_segmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run TTA for binary segmentation model.

        Args:
            x: (B, C, H, W) input tensor

        Returns:
            averaged_probs: (B, 1, H, W) averaged probability map
        """
        self.model.eval()
        accumulated = torch.zeros_like(self.model(x))

        for t in self.transforms:
            # Apply forward transform to input
            x_aug = t["fn"](x)

            # Get prediction
            logits = self.model(x_aug)
            probs  = torch.sigmoid(logits)

            # Apply INVERSE transform to prediction (undo the augmentation)
            probs_original_space = t["inv"](probs)

            accumulated += probs_original_space

        # Average over all augmentations
        return accumulated / len(self.transforms)

    @torch.no_grad()
    def predict_damage(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        model_type: str = "siamese",
    ) -> torch.Tensor:
        """
        Run TTA for multi-class damage model (Siamese or concatenated).

        Args:
            pre:  (B, 3, H, W)
            post: (B, 3, H, W)
            model_type: "siamese" or "concatenated"

        Returns:
            averaged_probs: (B, num_classes, H, W) averaged probability map
        """
        self.model.eval()

        # Get output shape from a dummy forward pass
        if model_type == "siamese":
            dummy_out = self.model(pre, post)
        else:
            dummy_out = self.model(torch.cat([pre, post], dim=1))

        accumulated = torch.zeros_like(dummy_out)

        for t in self.transforms:
            pre_aug  = t["fn"](pre)
            post_aug = t["fn"](post)

            if model_type == "siamese":
                logits = self.model(pre_aug, post_aug)
            else:
                logits = self.model(torch.cat([pre_aug, post_aug], dim=1))

            probs = F.softmax(logits, dim=1)
            probs_original = t["inv"](probs)
            accumulated += probs_original

        return accumulated / len(self.transforms)

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> tuple:
        """
        Use TTA disagreement as uncertainty measure.

        If all 8 augmentations agree → high confidence.
        If they disagree → uncertain region (model is not robust there).

        Returns:
            mean_probs: (B, 1, H, W) average prediction
            uncertainty: (B, 1, H, W) standard deviation across augmentations
        """
        self.model.eval()
        all_preds = []

        for t in self.transforms:
            x_aug  = t["fn"](x)
            logits = self.model(x_aug)
            probs  = torch.sigmoid(logits)
            probs_orig = t["inv"](probs)
            all_preds.append(probs_orig)

        stacked = torch.stack(all_preds, dim=0)   # (N_aug, B, 1, H, W)
        mean    = stacked.mean(dim=0)
        std     = stacked.std(dim=0)               # high std = uncertain

        return mean, std
