"""
DamageAssessmentPipeline — the glue that connects all three projects.

Flow:
  1. [Pipeline/Tiling]    Tile large pre/post GeoTIFF scenes
  2. [Segmentation]       Detect building footprints from pre image
  3. [Change Detection]   Classify damage level from pre+post pair
  4. [Fusion]             Mask damage predictions with building footprints
  5. [Vectorisation]      Convert pixel-level output to building-level GeoJSON
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from .segmentation import BuildingSegmentationModel
from .change_detection import DamageClassificationModel, DAMAGE_CLASSES


class DamageAssessmentPipeline:
    """
    Orchestrates building segmentation and damage classification for
    a single pre/post satellite image pair.

    Can run on CPU or GPU; automatically selects available device.
    """

    def __init__(
        self,
        seg_checkpoint: Optional[str] = None,
        dmg_checkpoint: Optional[str] = None,
        seg_config: Optional[Dict] = None,
        dmg_config: Optional[Dict] = None,
        device: Optional[str] = None,
        seg_threshold: float = 0.5,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seg_threshold = seg_threshold

        seg_cfg = seg_config or {}
        self.seg_model = BuildingSegmentationModel(
            encoder_name=seg_cfg.get("encoder", "resnet34"),
            encoder_weights=seg_cfg.get("encoder_weights", "imagenet"),
            in_channels=seg_cfg.get("in_channels", 3),
        ).to(self.device)

        dmg_cfg = dmg_config or {}
        self.dmg_model = DamageClassificationModel(
            encoder_name=dmg_cfg.get("encoder", "resnet34"),
            in_channels=dmg_cfg.get("in_channels", 6),
            num_classes=dmg_cfg.get("classes", 5),
        ).to(self.device)

        if seg_checkpoint:
            self._load_checkpoint(self.seg_model, seg_checkpoint)
        if dmg_checkpoint:
            self._load_checkpoint(self.dmg_model, dmg_checkpoint)

        self.seg_model.eval()
        self.dmg_model.eval()

        logger.info(
            f"DamageAssessmentPipeline ready on {self.device} | "
            f"seg_params={self.seg_model.n_parameters:,} | "
            f"dmg_params={self.dmg_model.n_parameters:,}"
        )

    def _load_checkpoint(self, model: torch.nn.Module, path: str):
        state = torch.load(path, map_location=self.device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        logger.info(f"Loaded checkpoint: {path}")

    @torch.no_grad()
    def run_on_tiles(
        self,
        pre_tiles: List[np.ndarray],
        post_tiles: List[np.ndarray],
        batch_size: int = 4,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Run both models on lists of (H, W, C) numpy tile arrays.

        Returns:
            building_masks: list of (H, W) float32 arrays (0-1 building probability)
            damage_maps:    list of (H, W) int64 arrays (damage class index)
        """
        building_masks = []
        damage_maps    = []

        n = len(pre_tiles)
        for start in range(0, n, batch_size):
            pre_batch  = pre_tiles[start : start + batch_size]
            post_batch = post_tiles[start : start + batch_size]

            # Convert to tensors: (B, C, H, W)
            pre_t  = self._tiles_to_tensor(pre_batch)
            post_t = self._tiles_to_tensor(post_batch)
            pair_t = torch.cat([pre_t, post_t], dim=1)   # (B, 6, H, W)

            # Building segmentation on pre-disaster image
            seg_logits = self.seg_model(pre_t)
            seg_probs  = torch.sigmoid(seg_logits).squeeze(1)  # (B, H, W)

            # Damage classification on pre+post pair
            dmg_logits = self.dmg_model(pair_t)
            dmg_preds  = torch.argmax(dmg_logits, dim=1)       # (B, H, W)

            # Mask damage predictions with building footprint
            # Pixels outside buildings are forced to class 0 (background)
            building_binary = (seg_probs > self.seg_threshold).long()
            dmg_masked      = dmg_preds * building_binary

            building_masks.extend(seg_probs.cpu().numpy())
            damage_maps.extend(dmg_masked.cpu().numpy())

        return building_masks, damage_maps

    def _tiles_to_tensor(self, tiles: List[np.ndarray]) -> torch.Tensor:
        """
        Convert list of (H, W, C) float32 [0,1] tiles to (B, C, H, W) tensor.
        Applies ImageNet normalization.
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        batch = np.stack(tiles, axis=0)                     # (B, H, W, C)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2) # (B, C, H, W)
        batch = batch.float().to(self.device)
        batch = (batch - mean) / std
        return batch

    def assess_full_scene(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: int = 4,
    ) -> Dict:
        """
        End-to-end damage assessment on full-size (potentially large) image arrays.

        Args:
            pre_image:  (H, W, 3) uint8 pre-disaster image
            post_image: (H, W, 3) uint8 post-disaster image

        Returns:
            dict with keys:
              - building_prob_map: (H, W) float32 — building probability
              - damage_map:        (H, W) int64   — per-pixel damage class
              - damage_rgb:        (H, W, 3) uint8 — colour-coded visualisation
              - stats:             dict with per-class building counts and areas
        """
        from ..pipeline.tiling import RasterTiler
        from ..utils.geo_utils import polygonize_damage_map

        h, w = pre_image.shape[:2]
        tiler = RasterTiler(tile_size=tile_size, overlap=overlap)

        # Normalize images to float [0,1] for model input
        pre_f  = pre_image.astype(np.float32)  / 255.0
        post_f = post_image.astype(np.float32) / 255.0

        pre_tiles,  pre_specs  = tiler.tile_image(pre_f)
        post_tiles, _          = tiler.tile_image(post_f)

        building_masks, damage_maps = self.run_on_tiles(pre_tiles, post_tiles, batch_size)

        building_prob_map = tiler.reassemble(building_masks, pre_specs, h, w, n_classes=1)
        damage_map        = tiler.reassemble(
            [d.astype(np.float32) for d in damage_maps],
            pre_specs, h, w, n_classes=1
        ).astype(np.int64)

        damage_rgb = self._colorize_damage(damage_map)
        stats      = self._compute_stats(damage_map, building_prob_map)

        return {
            "building_prob_map": building_prob_map,
            "damage_map":        damage_map,
            "damage_rgb":        damage_rgb,
            "stats":             stats,
        }

    @staticmethod
    def _colorize_damage(damage_map: np.ndarray) -> np.ndarray:
        """Convert integer damage map to RGB visualisation."""
        COLOR_MAP = {
            0: (0,   0,   0),    # background — black
            1: (0,   200, 0),    # no-damage  — green
            2: (255, 255, 0),    # minor      — yellow
            3: (255, 140, 0),    # major      — orange
            4: (220, 0,   0),    # destroyed  — red
        }
        rgb = np.zeros((*damage_map.shape, 3), dtype=np.uint8)
        for cls_idx, color in COLOR_MAP.items():
            mask = damage_map == cls_idx
            rgb[mask] = color
        return rgb

    @staticmethod
    def _compute_stats(
        damage_map: np.ndarray,
        building_prob: np.ndarray,
        pixel_area_m2: float = 0.5,   # assume 0.5m GSD by default
    ) -> Dict:
        """Compute per-class building counts and area estimates."""
        building_mask = building_prob > 0.5
        total_building_pixels = building_mask.sum()

        stats = {"total_building_area_m2": int(total_building_pixels * pixel_area_m2)}
        for cls_idx, cls_name in DAMAGE_CLASSES.items():
            if cls_idx == 0:
                continue
            n_pixels = int(((damage_map == cls_idx) & building_mask).sum())
            stats[cls_name] = {
                "pixels":  n_pixels,
                "area_m2": n_pixels * pixel_area_m2,
                "pct": (n_pixels / max(total_building_pixels, 1)) * 100,
            }

        return stats
