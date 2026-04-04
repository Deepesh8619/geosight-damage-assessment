"""
GeoSightAssessor — the end-to-end inference engine.

Connects all three sub-projects:
  1. [Pipeline/Tiling]   Scalable ingestion of large satellite scenes
  2. [Segmentation]      Building footprint detection (pre-image)
  3. [Change Detection]  Damage classification (pre+post pair)
  4. [Fusion + Output]   Masked damage map → GeoJSON + GeoTIFF + stats report
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from loguru import logger

from ..models.damage_classifier import DamageAssessmentPipeline
from ..pipeline.tiling import RasterTiler
from ..pipeline.ingestion import GeoTiffScene
from ..utils.geo_utils import polygonize_damage_map, save_geotiff
from ..utils.viz_utils import (
    plot_damage_assessment,
    plot_class_distribution,
    create_leaflet_map,
)


class GeoSightAssessor:
    """
    Public API for satellite-based post-disaster building damage assessment.

    Usage:
        assessor = GeoSightAssessor(
            seg_checkpoint="checkpoints/segmentation/best.pth",
            dmg_checkpoint="checkpoints/damage/best.pth",
        )
        report = assessor.assess(
            pre_image_path="data/raw/hurricane-harvey_pre_00000001.tif",
            post_image_path="data/raw/hurricane-harvey_post_00000001.tif",
            output_dir="data/outputs/harvey_001",
        )
    """

    def __init__(
        self,
        seg_checkpoint: Optional[str] = None,
        dmg_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        tile_size: int = 512,
        tile_overlap: int = 64,
        batch_size: int = 4,
        seg_threshold: float = 0.5,
    ):
        self.tile_size    = tile_size
        self.tile_overlap = tile_overlap
        self.batch_size   = batch_size

        self.pipeline = DamageAssessmentPipeline(
            seg_checkpoint=seg_checkpoint,
            dmg_checkpoint=dmg_checkpoint,
            device=device,
            seg_threshold=seg_threshold,
        )

        self.tiler = RasterTiler(tile_size=tile_size, overlap=tile_overlap)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def assess(
        self,
        pre_image_path: str,
        post_image_path: str,
        output_dir: str,
        event_name: Optional[str] = None,
        save_geotiff_output: bool = True,
        save_geojson_output: bool = True,
        save_leaflet_map: bool = True,
        save_figures: bool = True,
        pixel_area_m2: float = 0.5,
    ) -> Dict:
        """
        Run full damage assessment on a pre/post image pair.

        Accepts both PNG (xBD format) and GeoTIFF inputs.
        GeoTIFF inputs retain geographic metadata in outputs.

        Returns a report dict containing paths to all output files
        and summary statistics.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        event_name = event_name or Path(pre_image_path).stem

        logger.info(f"Starting assessment: {event_name}")
        logger.info(f"  Pre:  {pre_image_path}")
        logger.info(f"  Post: {post_image_path}")

        # 1. Load imagery
        pre_img, post_img, transform, crs = self._load_image_pair(
            pre_image_path, post_image_path
        )
        h, w = pre_img.shape[:2]
        logger.info(f"  Image size: {w}x{h}")

        # 2. Run models (tiling happens inside assess_full_scene)
        result = self.pipeline.assess_full_scene(
            pre_image=pre_img,
            post_image=post_img,
            tile_size=self.tile_size,
            overlap=self.tile_overlap,
            batch_size=self.batch_size,
        )

        building_prob = result["building_prob_map"]
        damage_map    = result["damage_map"]
        stats         = result["stats"]

        logger.info(f"  Assessment complete. Stats: {json.dumps(stats, indent=2)}")

        # 3. Save outputs
        output_files = {}

        if save_geotiff_output:
            out_path = str(Path(output_dir) / f"{event_name}_damage.tif")
            save_geotiff(damage_map.astype(np.uint8), out_path, transform, crs)
            output_files["damage_geotiff"] = out_path

            seg_path = str(Path(output_dir) / f"{event_name}_buildings.tif")
            save_geotiff((building_prob * 255).astype(np.uint8), seg_path, transform, crs)
            output_files["building_geotiff"] = seg_path

        gdf = None
        if save_geojson_output:
            building_mask = building_prob > 0.5
            gdf = polygonize_damage_map(
                damage_map, transform, crs,
                min_area_pixels=10,
                building_mask=building_mask,
            )
            geojson_path = str(Path(output_dir) / f"{event_name}_damage.geojson")
            gdf.to_file(geojson_path, driver="GeoJSON")
            output_files["damage_geojson"] = geojson_path
            logger.info(f"  Saved GeoJSON: {geojson_path} ({len(gdf)} buildings)")

        if save_figures:
            fig_path = str(Path(output_dir) / f"{event_name}_assessment.png")
            plot_damage_assessment(
                pre_img, post_img, building_prob, damage_map,
                title=f"Damage Assessment: {event_name}",
                save_path=fig_path,
            )
            output_files["assessment_figure"] = fig_path

            dist_path = str(Path(output_dir) / f"{event_name}_distribution.png")
            plot_class_distribution(stats, save_path=dist_path)
            output_files["distribution_figure"] = dist_path

        if save_leaflet_map and gdf is not None and not gdf.empty:
            map_path = str(Path(output_dir) / f"{event_name}_map.html")
            create_leaflet_map(gdf, output_path=map_path)
            output_files["leaflet_map"] = map_path

        # 4. Save stats report
        report = {
            "event":       event_name,
            "image_size":  {"width": w, "height": h},
            "statistics":  stats,
            "output_files": output_files,
        }
        report_path = str(Path(output_dir) / f"{event_name}_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        output_files["report_json"] = report_path

        logger.info(f"Assessment complete. Report saved: {report_path}")
        return report

    # ------------------------------------------------------------------
    # Batch processing (for multiple event images)
    # ------------------------------------------------------------------

    def assess_batch(
        self,
        pairs: list,
        output_base_dir: str,
        **kwargs,
    ) -> list:
        """
        Run assessment on a list of (pre_path, post_path) pairs.

        Args:
            pairs: list of (pre_path, post_path) tuples
            output_base_dir: base directory; sub-dirs are created per event

        Returns:
            list of report dicts
        """
        reports = []
        for i, (pre_path, post_path) in enumerate(pairs, 1):
            event_name = Path(pre_path).stem
            output_dir = str(Path(output_base_dir) / event_name)
            logger.info(f"Batch [{i}/{len(pairs)}]: {event_name}")
            try:
                report = self.assess(pre_path, post_path, output_dir, **kwargs)
                reports.append(report)
            except Exception as e:
                logger.error(f"Failed on {event_name}: {e}")
                reports.append({"event": event_name, "error": str(e)})
        return reports

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image_pair(
        pre_path: str,
        post_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, rasterio.transform.Affine, str]:
        """
        Load a pre/post image pair.

        Supports GeoTIFF (with geographic metadata) and PNG/JPEG
        (uses a dummy identity transform).

        Returns:
            pre_img:   (H, W, 3) uint8 numpy array
            post_img:  (H, W, 3) uint8 numpy array
            transform: affine transform
            crs:       coordinate reference system string
        """
        ext = Path(pre_path).suffix.lower()

        if ext in (".tif", ".tiff"):
            with rasterio.open(pre_path) as src:
                pre_data  = src.read([1, 2, 3])   # (3, H, W)
                transform = src.transform
                crs       = str(src.crs) if src.crs else "EPSG:4326"
            with rasterio.open(post_path) as src:
                post_data = src.read([1, 2, 3])

            def _norm(d):
                d = d.astype(np.float32)
                for i in range(3):
                    p2, p98 = np.percentile(d[i], [2, 98])
                    d[i] = np.clip((d[i] - p2) / (p98 - p2 + 1e-8), 0, 1)
                return (d.transpose(1, 2, 0) * 255).astype(np.uint8)

            return _norm(pre_data), _norm(post_data), transform, crs

        else:
            from PIL import Image
            pre_img  = np.array(Image.open(pre_path).convert("RGB"))
            post_img = np.array(Image.open(post_path).convert("RGB"))
            h, w     = pre_img.shape[:2]
            transform = from_bounds(0, 0, w, h, w, h)
            return pre_img, post_img, transform, "EPSG:4326"

    @staticmethod
    def _print_report(report: Dict):
        """Pretty-print a damage assessment report to the console."""
        print("\n" + "=" * 60)
        print(f"  GEOSIGHT DAMAGE ASSESSMENT REPORT")
        print(f"  Event: {report['event']}")
        print("=" * 60)

        stats = report.get("statistics", {})
        total = stats.get("total_building_area_m2", 0)
        print(f"\n  Total building area: {total:,} m²")
        print(f"\n  Damage breakdown:")
        for cls_name in ["no-damage", "minor-damage", "major-damage", "destroyed"]:
            s = stats.get(cls_name, {})
            print(
                f"    {cls_name:15s}: "
                f"{s.get('area_m2', 0):8,.0f} m²  "
                f"({s.get('pct', 0):5.1f}%)"
            )

        print(f"\n  Output files:")
        for k, v in report.get("output_files", {}).items():
            print(f"    {k:25s}: {v}")
        print("=" * 60 + "\n")
