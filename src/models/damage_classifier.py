"""
DamageAssessmentPipeline V2 — upgraded with all new modules.

Flow:
  1. [Tiling]              Tile large pre/post scenes (Dask-based)
  2. [Segmentation]        Detect building footprints (pre-image U-Net)
  3. [Siamese Change Det]  Classify damage (shared-encoder U-Net)
  4. [Fusion]              Mask damage with building footprints
  5. [Disaster Classifier] Predict disaster type from damage patterns
  6. [Spatial Analysis]    Epicentre, clusters, gradients, radial profile
  7. [Impact Analysis]     Population, economic loss, shelter, severity
  8. [Vectorisation]       Output GeoJSON + GeoTIFF + report
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from .segmentation import BuildingSegmentationModel
from .siamese_unet import SiameseUNet
from .change_detection import DamageClassificationModel, DAMAGE_CLASSES
from .disaster_classifier import DisasterTypeClassifier, DISASTER_TYPES


class DamageAssessmentPipeline:
    """
    Orchestrates the full damage assessment:
      - Building segmentation (pre-image)
      - Damage classification (pre+post Siamese)
      - Disaster type prediction (scene-level)
      - Confidence mapping

    Supports both the old 6-channel model and the new Siamese architecture.
    """

    def __init__(
        self,
        seg_checkpoint: Optional[str] = None,
        dmg_checkpoint: Optional[str] = None,
        disaster_checkpoint: Optional[str] = None,
        seg_config: Optional[Dict] = None,
        dmg_config: Optional[Dict] = None,
        device: Optional[str] = None,
        seg_threshold: float = 0.5,
        use_siamese: bool = True,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seg_threshold = seg_threshold
        self.use_siamese = use_siamese

        # --- Building segmentation model ---
        seg_cfg = seg_config or {}
        self.seg_model = BuildingSegmentationModel(
            encoder_name=seg_cfg.get("encoder", "resnet34"),
            encoder_weights=seg_cfg.get("encoder_weights", "imagenet"),
            in_channels=seg_cfg.get("in_channels", 3),
        ).to(self.device)

        # --- Damage classification model ---
        dmg_cfg = dmg_config or {}
        if use_siamese:
            self.dmg_model = SiameseUNet(
                encoder_name=dmg_cfg.get("encoder", "resnet34"),
                encoder_weights=dmg_cfg.get("encoder_weights", "imagenet"),
                num_classes=dmg_cfg.get("classes", 5),
            ).to(self.device)
        else:
            self.dmg_model = DamageClassificationModel(
                encoder_name=dmg_cfg.get("encoder", "resnet34"),
                in_channels=dmg_cfg.get("in_channels", 6),
                num_classes=dmg_cfg.get("classes", 5),
            ).to(self.device)

        # --- Disaster type classifier ---
        self.disaster_model = DisasterTypeClassifier().to(self.device)

        # Load checkpoints
        if seg_checkpoint:
            self._load_checkpoint(self.seg_model, seg_checkpoint)
        if dmg_checkpoint:
            self._load_checkpoint(self.dmg_model, dmg_checkpoint)
        if disaster_checkpoint:
            self._load_checkpoint(self.disaster_model, disaster_checkpoint)

        self.seg_model.eval()
        self.dmg_model.eval()
        self.disaster_model.eval()

        logger.info(
            f"DamageAssessmentPipeline V2 on {self.device} | "
            f"siamese={use_siamese} | "
            f"seg={self.seg_model.n_parameters:,} | "
            f"dmg={self.dmg_model.n_parameters:,} | "
            f"disaster={self.disaster_model.n_parameters:,}"
        )

    def _load_checkpoint(self, model: torch.nn.Module, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
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
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Run all models on tile batches.

        Returns:
            building_masks:  list of (H, W) float32 — building probability
            damage_maps:     list of (H, W) int64   — damage class
            confidence_maps: list of (H, W) float32 — per-pixel confidence
            damage_probs:    list of (5, H, W) float32 — class probabilities
        """
        building_masks  = []
        damage_maps     = []
        confidence_maps = []
        all_damage_probs = []

        n = len(pre_tiles)
        for start in range(0, n, batch_size):
            pre_batch  = pre_tiles[start : start + batch_size]
            post_batch = post_tiles[start : start + batch_size]

            pre_t  = self._tiles_to_tensor(pre_batch)
            post_t = self._tiles_to_tensor(post_batch)

            # Building segmentation
            seg_logits = self.seg_model(pre_t)
            seg_probs  = torch.sigmoid(seg_logits).squeeze(1)  # (B, H, W)

            # Damage classification
            if self.use_siamese:
                dmg_logits = self.dmg_model(pre_t, post_t)
            else:
                pair_t = torch.cat([pre_t, post_t], dim=1)
                dmg_logits = self.dmg_model(pair_t)

            dmg_probs  = F.softmax(dmg_logits, dim=1)       # (B, 5, H, W)
            confidence = dmg_probs.max(dim=1).values         # (B, H, W)
            dmg_preds  = dmg_probs.argmax(dim=1)             # (B, H, W)

            # Mask: non-building pixels → class 0
            building_binary = (seg_probs > self.seg_threshold).long()
            dmg_masked      = dmg_preds * building_binary

            building_masks.extend(seg_probs.cpu().numpy())
            damage_maps.extend(dmg_masked.cpu().numpy())
            confidence_maps.extend(confidence.cpu().numpy())
            all_damage_probs.extend(dmg_probs.cpu().numpy())

        return building_masks, damage_maps, confidence_maps, all_damage_probs

    @torch.no_grad()
    def predict_disaster_type(
        self,
        post_tiles: List[np.ndarray],
        damage_probs: List[np.ndarray],
    ) -> Dict:
        """
        Predict disaster type from the overall damage pattern.
        Aggregates tile-level features into a scene-level prediction.
        """
        # Use center tiles (most representative) for disaster classification
        n = len(post_tiles)
        if n == 0:
            return {"type": "unknown", "confidence": 0.0, "all_probs": {}}

        # Sample up to 4 tiles from the center of the scene
        indices = [n // 4, n // 2, 3 * n // 4, n - 1]
        indices = list(set(min(i, n - 1) for i in indices))

        sampled_post  = [post_tiles[i]  for i in indices]
        sampled_probs = [damage_probs[i] for i in indices]

        post_t  = self._tiles_to_tensor(sampled_post)
        probs_t = torch.from_numpy(np.stack(sampled_probs)).float().to(self.device)

        return self.disaster_model.predict(post_t, probs_t)

    def _tiles_to_tensor(self, tiles: List[np.ndarray]) -> torch.Tensor:
        """Convert (H,W,C) float32 [0,1] tiles to normalized (B,C,H,W) tensor."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        batch = np.stack(tiles, axis=0)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2)
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
        End-to-end damage assessment on full-size image arrays.

        Returns dict with:
          - building_prob_map: (H, W) float32
          - damage_map:        (H, W) int64
          - confidence_map:    (H, W) float32
          - damage_rgb:        (H, W, 3) uint8
          - disaster_type:     dict with type, confidence, probabilities
          - spatial_analysis:  dict with epicentre, clusters, gradients
          - impact_report:     dict with population, economic, shelter estimates
          - stats:             basic per-class pixel counts
        """
        from ..pipeline.tiling import RasterTiler
        from ..utils.spatial_analysis import SpatialDamageAnalyzer
        from ..utils.impact_analysis import (
            HumanitarianImpactAnalyzer,
            generate_priority_zones,
        )

        h, w = pre_image.shape[:2]
        tiler = RasterTiler(tile_size=tile_size, overlap=overlap)

        # Normalize to float [0,1]
        pre_f  = pre_image.astype(np.float32) / 255.0
        post_f = post_image.astype(np.float32) / 255.0

        pre_tiles,  pre_specs = tiler.tile_image(pre_f)
        post_tiles, _         = tiler.tile_image(post_f)

        # ---- Run models ----
        building_masks, damage_maps, confidence_maps, damage_probs = \
            self.run_on_tiles(pre_tiles, post_tiles, batch_size)

        # ---- Reassemble tiles ----
        building_prob_map = tiler.reassemble(building_masks, pre_specs, h, w, n_classes=1)
        confidence_map    = tiler.reassemble(confidence_maps, pre_specs, h, w, n_classes=1)
        damage_map        = tiler.reassemble(
            [d.astype(np.float32) for d in damage_maps],
            pre_specs, h, w, n_classes=1
        ).astype(np.int64)

        building_mask = building_prob_map > self.seg_threshold

        # ---- Disaster type prediction ----
        disaster_pred = self.predict_disaster_type(post_tiles, damage_probs)
        disaster_type = disaster_pred["type"] if isinstance(disaster_pred, dict) else "unknown"

        # ---- Spatial analysis ----
        spatial_analyzer = SpatialDamageAnalyzer(pixel_gsd_m=0.5)
        spatial_report = spatial_analyzer.full_analysis(damage_map, building_mask)

        # ---- Humanitarian impact ----
        impact_analyzer = HumanitarianImpactAnalyzer(pixel_area_m2=0.25)
        impact_report = impact_analyzer.analyze(
            damage_map, building_mask,
            disaster_type=disaster_type,
            confidence_map=confidence_map,
        )

        # ---- Priority zones ----
        priority_map = generate_priority_zones(
            damage_map, building_mask, confidence_map
        )

        # ---- Basic stats ----
        stats = self._compute_stats(damage_map, building_prob_map)

        # ---- Visualisation ----
        damage_rgb = self._colorize_damage(damage_map)

        return {
            "building_prob_map": building_prob_map,
            "damage_map":        damage_map,
            "confidence_map":    confidence_map,
            "damage_rgb":        damage_rgb,
            "priority_map":      priority_map,
            "disaster_type":     disaster_pred,
            "spatial_analysis":  spatial_report,
            "impact_report":     impact_report.to_dict(),
            "stats":             stats,
        }

    @staticmethod
    def _colorize_damage(damage_map: np.ndarray) -> np.ndarray:
        COLOR_MAP = {
            0: (0,   0,   0),
            1: (0,   200, 0),
            2: (255, 255, 0),
            3: (255, 140, 0),
            4: (220, 0,   0),
        }
        rgb = np.zeros((*damage_map.shape, 3), dtype=np.uint8)
        for cls_idx, color in COLOR_MAP.items():
            rgb[damage_map == cls_idx] = color
        return rgb

    @staticmethod
    def _compute_stats(
        damage_map: np.ndarray,
        building_prob: np.ndarray,
        pixel_area_m2: float = 0.25,
    ) -> Dict:
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
