"""
Preprocessing and data augmentation for satellite imagery.
Handles normalization, mask creation from GeoJSON polygons,
and training-time augmentations using Albumentations.
"""

from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger


# ImageNet stats as a baseline for satellite imagery
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Sentinel-2 (10m bands: B2, B3, B4, B8) normalization constants
SENTINEL2_MEAN = (0.1014, 0.1074, 0.1103, 0.2280)
SENTINEL2_STD  = (0.0450, 0.0460, 0.0570, 0.0800)


def build_augmentation_pipeline(
    phase: str = "train",
    image_size: int = 512,
    use_sar: bool = False,
) -> A.Compose:
    """
    Build albumentations augmentation pipeline.

    Spatial augmentations are applied identically to both pre and post images
    and all masks via the `additional_targets` mechanism.

    Args:
        phase: "train" | "val" | "test"
        image_size: expected spatial size (square tiles)
        use_sar: if True, skip color-based augmentations (not applicable to SAR)
    """
    if phase == "train":
        spatial = [
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.4
            ),
        ]
        color = [] if use_sar else [
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
        ]
        transforms = spatial + color
    else:
        transforms = [
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size,
                border_mode=cv2.BORDER_REFLECT, p=1.0
            ),
        ]

    transforms += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    return A.Compose(
        transforms,
        additional_targets={
            "post_image":    "image",
            "building_mask": "mask",
            "damage_mask":   "mask",
        },
    )


class ImagePreprocessor:
    """
    Stateless image preprocessing helper.
    Converts numpy arrays into model-ready tensors.
    """

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize HxWxC float32 image."""
        return (image - self.mean) / self.std

    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert HxWxC numpy array to CxHxW torch tensor."""
        return torch.from_numpy(image.transpose(2, 0, 1))

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Full preprocess: normalize then convert to tensor."""
        img = image.astype(np.float32) / 255.0
        img = self.normalize(img)
        return self.to_tensor(img)

    def preprocess_pair(
        self, pre: np.ndarray, post: np.ndarray
    ) -> torch.Tensor:
        """
        Preprocess a pre/post image pair and concatenate along channel axis.
        Output tensor: (6, H, W) for 3-channel inputs.
        """
        pre_t  = self.preprocess(pre)
        post_t = self.preprocess(post)
        return torch.cat([pre_t, post_t], dim=0)


class XBDTransform:
    """
    Applies albumentations pipeline to an xBD sample dict.
    Ensures spatial transforms are applied consistently to both images and masks.
    """

    def __init__(self, phase: str = "train", image_size: int = 512):
        self.aug = build_augmentation_pipeline(phase=phase, image_size=image_size)

    def __call__(self, sample: Dict) -> Dict:
        pre_img  = sample["pre_image"]    # (H, W, 3) uint8
        post_img = sample["post_image"]
        bld_mask = sample["building_mask"]
        dmg_mask = sample["damage_mask"]

        result = self.aug(
            image=pre_img,
            post_image=post_img,
            building_mask=bld_mask,
            damage_mask=dmg_mask,
        )

        return {
            "pre_image":     result["image"],            # (3, H, W) float tensor
            "post_image":    result["post_image"],
            "building_mask": result["building_mask"].long(),
            "damage_mask":   result["damage_mask"].long(),
            "event":         sample.get("event", ""),
        }


def compute_dataset_statistics(
    image_paths: List[str],
    n_bands: int = 3,
    sample_fraction: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-band mean and std over a dataset for custom normalization.
    Samples a fraction of pixels to keep it tractable.

    Returns:
        mean: (n_bands,) array
        std:  (n_bands,) array
    """
    from PIL import Image

    sums    = np.zeros(n_bands, dtype=np.float64)
    sq_sums = np.zeros(n_bands, dtype=np.float64)
    count   = 0

    for path in image_paths:
        img = np.array(Image.open(path)).astype(np.float64) / 255.0
        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        h, w, c = img.shape
        n_sample = max(1, int(h * w * sample_fraction))
        flat_idx = np.random.choice(h * w, size=n_sample, replace=False)

        for b in range(min(c, n_bands)):
            band_flat = img[:, :, b].ravel()[flat_idx]
            sums[b]    += band_flat.sum()
            sq_sums[b] += (band_flat ** 2).sum()
            count       += len(flat_idx)

    mean = (sums    / count).astype(np.float32)
    std  = (np.sqrt(sq_sums / count - mean ** 2)).astype(np.float32)

    logger.info(f"Dataset stats — mean: {mean}, std: {std}")
    return mean, std
