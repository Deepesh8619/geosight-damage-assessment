"""
Data ingestion module.
Supports:
  - xBD dataset (xView2 Disaster Damage Assessment Challenge)
  - Generic GeoTIFF pre/post image pairs
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import shape, mapping
from torch.utils.data import Dataset
from loguru import logger


# xBD damage label mapping
XBD_LABEL_MAP = {
    "background":    0,
    "no-damage":     1,
    "minor-damage":  2,
    "major-damage":  3,
    "destroyed":     4,
    "un-classified": 1,  # treat as no-damage
}


class XBDDataset(Dataset):
    """
    PyTorch Dataset for the xBD (xView2) disaster damage dataset.

    Dataset structure expected:
        root/
          images/
            <event>_pre_<id>.png
            <event>_post_<id>.png
          labels/
            <event>_pre_<id>.json
            <event>_post_<id>.json

    Each JSON label file has:
        {
          "features": {
            "xy": [{"type": "Feature", "geometry": {...}, "properties": {...}}, ...]
          },
          "metadata": {"img_name": ..., "geotransform": [...], ...}
        }
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",            # "train" | "test" | "hold"
        task: str = "damage",            # "segmentation" | "damage"
        transform=None,
        channels: int = 3,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.task = task
        self.transform = transform
        self.channels = channels

        self.image_dir = self.root_dir / split / "images"
        self.label_dir = self.root_dir / split / "labels"

        self.samples = self._build_sample_list()
        logger.info(f"XBDDataset [{split}]: {len(self.samples)} pre/post pairs loaded.")

    def _build_sample_list(self) -> List[Dict]:
        samples = []
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        pre_images = sorted(self.image_dir.glob("*_pre_*.png"))
        for pre_path in pre_images:
            stem = pre_path.stem                             # e.g. hurricane-harvey_pre_00000001
            post_stem = stem.replace("_pre_", "_post_")
            post_path = self.image_dir / f"{post_stem}.png"

            if not post_path.exists():
                logger.warning(f"No post image found for {pre_path.name}, skipping.")
                continue

            pre_label = self.label_dir / f"{stem}.json"
            post_label = self.label_dir / f"{post_stem}.json"

            if not pre_label.exists() or not post_label.exists():
                logger.warning(f"Label files missing for {stem}, skipping.")
                continue

            samples.append({
                "pre_image":  str(pre_path),
                "post_image": str(post_path),
                "pre_label":  str(pre_label),
                "post_label": str(post_label),
                "event":      stem.split("_pre_")[0],
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        pre_img  = self._load_image(sample["pre_image"])
        post_img = self._load_image(sample["post_image"])

        pre_label_data  = self._load_label(sample["pre_label"])
        post_label_data = self._load_label(sample["post_label"])

        h, w = pre_img.shape[:2]
        building_mask = self._rasterize_buildings(pre_label_data, h, w)
        damage_mask   = self._rasterize_damage(post_label_data, h, w)

        item = {
            "pre_image":     pre_img,
            "post_image":    post_img,
            "building_mask": building_mask,
            "damage_mask":   damage_mask,
            "event":         sample["event"],
        }

        if self.transform:
            item = self.transform(item)

        return item

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        """Load image as HxWxC uint8 numpy array."""
        from PIL import Image
        img = np.array(Image.open(path).convert("RGB"))
        return img

    @staticmethod
    def _load_label(path: str) -> Dict:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _rasterize_buildings(label_data: Dict, height: int, width: int) -> np.ndarray:
        """Create binary building footprint mask from pre-disaster label."""
        from rasterio.features import rasterize as rio_rasterize
        from affine import Affine

        mask = np.zeros((height, width), dtype=np.uint8)
        features = label_data.get("features", {}).get("xy", [])

        if not features:
            return mask

        geotransform = label_data.get("metadata", {}).get("geotransform", None)
        if geotransform:
            transform = Affine(
                geotransform[1], geotransform[2], geotransform[0],
                geotransform[4], geotransform[5], geotransform[3]
            )
        else:
            transform = Affine(1, 0, 0, 0, 1, 0)

        shapes = [
            (feat["geometry"], 1)
            for feat in features
            if feat.get("geometry") is not None
        ]

        if shapes:
            mask = rio_rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )

        return mask

    @staticmethod
    def _rasterize_damage(label_data: Dict, height: int, width: int) -> np.ndarray:
        """Create multi-class damage mask from post-disaster label."""
        from rasterio.features import rasterize as rio_rasterize
        from affine import Affine

        mask = np.zeros((height, width), dtype=np.uint8)
        features = label_data.get("features", {}).get("xy", [])

        if not features:
            return mask

        geotransform = label_data.get("metadata", {}).get("geotransform", None)
        transform = Affine(1, 0, 0, 0, 1, 0)
        if geotransform:
            transform = Affine(
                geotransform[1], geotransform[2], geotransform[0],
                geotransform[4], geotransform[5], geotransform[3]
            )

        # Rasterize from least severe to most severe so severe overwrites
        for class_label, class_idx in sorted(XBD_LABEL_MAP.items(), key=lambda x: x[1]):
            shapes = [
                (feat["geometry"], class_idx)
                for feat in features
                if feat.get("properties", {}).get("subtype", "").lower() == class_label
                and feat.get("geometry") is not None
            ]
            if shapes:
                class_mask = rio_rasterize(
                    shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
                mask = np.where(class_mask > 0, class_mask, mask)

        return mask


class GeoTiffScene:
    """
    Loader for generic GeoTIFF satellite imagery.
    Handles reprojection, band selection, and metadata extraction.
    Works with Sentinel-2, Landsat, Planet, or any multi-band GeoTIFF.
    """

    def __init__(self, path: str, band_indices: Optional[List[int]] = None):
        self.path = Path(path)
        self.band_indices = band_indices  # 1-based band indices; None = all bands

        with rasterio.open(self.path) as src:
            self.meta     = src.meta.copy()
            self.crs      = src.crs
            self.transform = src.transform
            self.bounds   = src.bounds
            self.count    = src.count
            self.width    = src.width
            self.height   = src.height
            self.nodata   = src.nodata

        logger.debug(
            f"Loaded scene: {self.path.name} | "
            f"{self.width}x{self.height} | {self.count} bands | CRS: {self.crs}"
        )

    def read(
        self,
        window: Optional[rasterio.windows.Window] = None,
        reproject_to_crs: Optional[str] = None,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine]:
        """
        Read imagery data.

        Returns:
            data: ndarray of shape (C, H, W) as float32 in [0, 1]
            transform: affine transform of the returned window
        """
        with rasterio.open(self.path) as src:
            bands = self.band_indices or list(range(1, src.count + 1))

            if window:
                data = src.read(bands, window=window)
                transform = src.window_transform(window)
            else:
                data = src.read(bands)
                transform = src.transform

        data = data.astype(np.float32)

        # Percentile clip and normalize to [0, 1]
        for i in range(data.shape[0]):
            band = data[i]
            valid = band[band != self.nodata] if self.nodata is not None else band.ravel()
            p2, p98 = np.percentile(valid, [2, 98])
            data[i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)

        return data, transform

    def to_rgb_uint8(self, rgb_bands: Tuple[int, int, int] = (1, 2, 3)) -> np.ndarray:
        """Quick export to HxWx3 uint8 for visualization."""
        with rasterio.open(self.path) as src:
            r = src.read(rgb_bands[0]).astype(np.float32)
            g = src.read(rgb_bands[1]).astype(np.float32)
            b = src.read(rgb_bands[2]).astype(np.float32)

        def norm(x):
            p2, p98 = np.percentile(x, [2, 98])
            return np.clip((x - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(np.uint8)

        return np.stack([norm(r), norm(g), norm(b)], axis=-1)
