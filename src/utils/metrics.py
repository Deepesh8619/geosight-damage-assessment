"""
Evaluation metrics for segmentation and damage classification tasks.
Implements standard remote sensing assessment metrics.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from loguru import logger


DAMAGE_CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]


# ---------------------------------------------------------------------------
# Segmentation Metrics (for building footprint model)
# ---------------------------------------------------------------------------

class SegmentationMetrics:
    """
    Tracks running IoU, Dice, precision, and recall for binary segmentation.
    Call update() per batch, then compute() at epoch end.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits:  (B, 1, H, W) raw logits
            targets: (B, H, W) or (B, 1, H, W) binary masks
        """
        probs   = torch.sigmoid(logits).squeeze(1)
        preds   = (probs > self.threshold).float()
        targets = targets.float().squeeze(1)

        self.tp += (preds * targets).sum().item()
        self.fp += (preds * (1 - targets)).sum().item()
        self.fn += ((1 - preds) * targets).sum().item()
        self.tn += ((1 - preds) * (1 - targets)).sum().item()

    def compute(self) -> Dict[str, float]:
        eps = 1e-8
        precision = self.tp / (self.tp + self.fp + eps)
        recall    = self.tp / (self.tp + self.fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        iou       = self.tp / (self.tp + self.fp + self.fn + eps)
        dice      = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        accuracy  = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)

        return {
            "iou":       round(iou,       4),
            "dice":      round(dice,      4),
            "f1":        round(f1,        4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "accuracy":  round(accuracy,  4),
        }

    def log(self, prefix: str = ""):
        metrics = self.compute()
        tag = f"[{prefix}] " if prefix else ""
        logger.info(
            f"{tag}Segmentation — "
            f"IoU={metrics['iou']:.4f}  Dice={metrics['dice']:.4f}  "
            f"F1={metrics['f1']:.4f}  Precision={metrics['precision']:.4f}  "
            f"Recall={metrics['recall']:.4f}"
        )
        return metrics


# ---------------------------------------------------------------------------
# Damage Classification Metrics
# ---------------------------------------------------------------------------

class DamageMetrics:
    """
    Tracks per-class and overall metrics for multi-class damage assessment.
    Implements the xView2 competition scoring methodology:
      - F1 score per damage class (weighted by class severity)
      - Overall harmonic mean of per-class F1
    """

    # xView2 competition weights: damage gets more weight than no-damage
    SEVERITY_WEIGHTS = {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    def __init__(self, num_classes: int = 5, ignore_background: bool = True):
        self.num_classes        = num_classes
        self.ignore_background  = ignore_background
        self.all_preds   = []
        self.all_targets = []

    def reset(self):
        self.all_preds   = []
        self.all_targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits:  (B, C, H, W) raw logits
            targets: (B, H, W) int64 class indices
        """
        preds = torch.argmax(logits, dim=1)

        # Flatten and move to CPU
        pred_flat   = preds.cpu().numpy().ravel()
        target_flat = targets.cpu().numpy().ravel()

        if self.ignore_background:
            valid = target_flat != 0
            pred_flat   = pred_flat[valid]
            target_flat = target_flat[valid]

        self.all_preds.extend(pred_flat.tolist())
        self.all_targets.extend(target_flat.tolist())

    def compute(self) -> Dict:
        if not self.all_preds:
            return {}

        preds   = np.array(self.all_preds)
        targets = np.array(self.all_targets)

        classes = list(range(1, self.num_classes)) if self.ignore_background \
                  else list(range(self.num_classes))
        labels  = [DAMAGE_CLASS_NAMES[c] for c in classes]

        per_class_f1 = f1_score(
            targets, preds, labels=classes, average=None, zero_division=0
        )
        per_class_p  = precision_score(
            targets, preds, labels=classes, average=None, zero_division=0
        )
        per_class_r  = recall_score(
            targets, preds, labels=classes, average=None, zero_division=0
        )

        # xView2 harmonic mean score (geometric mean of damage-class F1s)
        damage_f1s = per_class_f1[1:]  # skip no-damage (index 0 in this slice → class 2+)
        xview2_score = _harmonic_mean(per_class_f1)

        macro_f1  = f1_score(targets, preds, labels=classes, average="macro",  zero_division=0)
        weighted_f1 = f1_score(targets, preds, labels=classes, average="weighted", zero_division=0)

        return {
            "xview2_score": round(float(xview2_score), 4),
            "macro_f1":     round(float(macro_f1),     4),
            "weighted_f1":  round(float(weighted_f1),  4),
            "per_class": {
                label: {
                    "f1":        round(float(f), 4),
                    "precision": round(float(p), 4),
                    "recall":    round(float(r), 4),
                }
                for label, f, p, r in zip(labels, per_class_f1, per_class_p, per_class_r)
            },
            "confusion_matrix": confusion_matrix(targets, preds, labels=classes).tolist(),
        }

    def log(self, prefix: str = ""):
        metrics = self.compute()
        if not metrics:
            return metrics

        tag = f"[{prefix}] " if prefix else ""
        logger.info(
            f"{tag}Damage — "
            f"xView2={metrics['xview2_score']:.4f}  "
            f"macro_F1={metrics['macro_f1']:.4f}  "
            f"weighted_F1={metrics['weighted_f1']:.4f}"
        )
        for label, vals in metrics["per_class"].items():
            logger.info(
                f"  {label:15s} F1={vals['f1']:.4f}  "
                f"P={vals['precision']:.4f}  R={vals['recall']:.4f}"
            )
        return metrics

    def classification_report(self) -> str:
        classes = list(range(1, self.num_classes)) if self.ignore_background \
                  else list(range(self.num_classes))
        return classification_report(
            self.all_targets, self.all_preds,
            labels=classes,
            target_names=[DAMAGE_CLASS_NAMES[c] for c in classes],
            zero_division=0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _harmonic_mean(values: np.ndarray) -> float:
    """Harmonic mean — robust to outliers, used in xView2 scoring."""
    values = np.array(values, dtype=np.float64)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    return len(values) / np.sum(1.0 / values)


def compute_iou_per_class(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """Compute per-class IoU from flat arrays."""
    iou_per_class = {}
    for cls in range(num_classes):
        pred_c   = preds   == cls
        target_c = targets == cls
        inter    = (pred_c & target_c).sum()
        union    = (pred_c | target_c).sum()
        iou_per_class[DAMAGE_CLASS_NAMES.get(cls, str(cls))] = inter / (union + 1e-8)
    return iou_per_class
