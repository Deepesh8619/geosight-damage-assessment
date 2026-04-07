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
        confidence    = result.get("confidence_map")
        priority      = result.get("priority_map")
        stats         = result["stats"]
        disaster_type = result.get("disaster_type", {})
        spatial       = result.get("spatial_analysis", {})
        impact        = result.get("impact_report", {})
        hotspots      = result.get("dbscan_hotspots", {})
        land_cover    = result.get("land_cover", {})

        logger.info(f"  Assessment complete.")
        logger.info(f"  Disaster type: {disaster_type.get('type', 'unknown')} "
                     f"(confidence: {disaster_type.get('confidence', 0):.2%})")
        logger.info(f"  Severity: {impact.get('severity', {}).get('label', 'N/A')} "
                     f"({impact.get('severity', {}).get('index', 0)}/100)")
        logger.info(f"  Est. displaced: {impact.get('population', {}).get('displaced', 0):,}")
        logger.info(f"  Est. economic loss: ${impact.get('economic', {}).get('loss_usd', 0):,.0f}")

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

        # Save confidence and priority maps as GeoTIFF
        if save_geotiff_output and confidence is not None:
            conf_path = str(Path(output_dir) / f"{event_name}_confidence.tif")
            save_geotiff((confidence * 255).astype(np.uint8), conf_path, transform, crs)
            output_files["confidence_geotiff"] = conf_path

        if save_geotiff_output and priority is not None:
            prio_path = str(Path(output_dir) / f"{event_name}_priority.tif")
            save_geotiff((priority * 255).astype(np.uint8), prio_path, transform, crs)
            output_files["priority_geotiff"] = prio_path

        # 4. Save full report
        report = {
            "event":            event_name,
            "image_size":       {"width": w, "height": h},
            "disaster_type":    disaster_type,
            "impact_report":    impact,
            "spatial_analysis": spatial,
            "dbscan_hotspots":  hotspots,
            "land_cover":       land_cover,
            "statistics":       stats,
            "output_files":     output_files,
        }
        report_path = str(Path(output_dir) / f"{event_name}_report.json")

        # Convert numpy types to native Python for JSON serialization
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj

        with open(report_path, "w") as f:
            json.dump(_convert(report), f, indent=2)
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
        """Pretty-print the full humanitarian damage assessment report."""
        W = 64
        print("\n" + "=" * W)
        print("  GEOSIGHT DAMAGE ASSESSMENT REPORT")
        print(f"  Event: {report['event']}")
        print("=" * W)

        # Disaster type
        dt = report.get("disaster_type", {})
        print(f"\n  DISASTER TYPE: {dt.get('type', 'unknown').upper()}"
              f"  (confidence: {dt.get('confidence', 0):.1%})")

        # Impact summary
        imp = report.get("impact_report", {})
        sev = imp.get("severity", {})
        pop = imp.get("population", {})
        eco = imp.get("economic", {})
        shlt = imp.get("shelter", {})

        print(f"\n  SEVERITY: {sev.get('label', 'N/A').upper()} "
              f"({sev.get('index', 0):.0f}/100)")

        print(f"\n  POPULATION IMPACT:")
        print(f"    Estimated affected:    {pop.get('affected', 0):>8,} people")
        print(f"    Estimated displaced:   {pop.get('displaced', 0):>8,} people")
        print(f"    Estimated casualties:  {pop.get('casualties_low', 0):>8,} - "
              f"{pop.get('casualties_high', 0):,}")

        print(f"\n  ECONOMIC IMPACT:")
        print(f"    Estimated loss:        ${eco.get('loss_usd', 0):>12,.0f}")
        print(f"    Reconstruction cost:   ${eco.get('reconstruction_usd', 0):>12,.0f}")

        print(f"\n  SHELTER NEEDS:")
        print(f"    Emergency shelter:     {shlt.get('needed_m2', 0):>8,.0f} m²")
        print(f"    Tents needed:          {shlt.get('tents_needed', 0):>8,}")

        # Building damage breakdown
        bld = imp.get("buildings", {})
        print(f"\n  BUILDINGS: {bld.get('total', 0):,} total")
        print(f"    No damage:    {bld.get('no_damage', 0):>6,}")
        print(f"    Minor damage: {bld.get('minor_damage', 0):>6,}")
        print(f"    Major damage: {bld.get('major_damage', 0):>6,}")
        print(f"    Destroyed:    {bld.get('destroyed', 0):>6,}")

        # Spatial analysis
        spa = report.get("spatial_analysis", {})
        epi = spa.get("epicentre", {})
        grad = spa.get("gradient", {})
        if epi.get("center"):
            print(f"\n  SPATIAL ANALYSIS:")
            print(f"    Damage epicentre:      pixel ({epi['center'][1]}, {epi['center'][0]})")
            print(f"    Damage radius:         {epi.get('radius_m', 0):.0f} m")
            print(f"    Damage pattern:        {grad.get('pattern', 'N/A')}")
            print(f"    Dominant direction:     {grad.get('dominant_direction', 'N/A')}")
            n_clusters = len(spa.get("clusters", []))
            print(f"    Damage clusters:       {n_clusters}")

        # DBSCAN hotspots
        hs = report.get("dbscan_hotspots", {})
        if hs.get("n_clusters", 0) > 0:
            print(f"\n  DAMAGE HOTSPOTS (DBSCAN unsupervised clustering):")
            print(f"    Clusters found:    {hs['n_clusters']}")
            print(f"    Noise ratio:       {hs.get('noise_ratio', 0):.1%} (isolated damage = likely false positives)")
            for c in hs.get("clusters", [])[:5]:
                print(f"    Hotspot {c['id']}: center=({c['centroid'][0]},{c['centroid'][1]})  "
                      f"size={c['n_points']} px  spread={c.get('spread_pixels', 0):.0f} px")

        # Land cover
        lc = report.get("land_cover", {})
        if lc.get("clusters"):
            print(f"\n  LAND COVER (K-Means unsupervised):")
            for c in lc["clusters"]:
                print(f"    Cluster {c['id']}: {c['coverage_pct']:.1f}% coverage  "
                      f"brightness={c['brightness']:.2f}")

        # Response protocol
        proto = imp.get("response_protocol", {})
        if proto:
            print(f"\n  RESPONSE PROTOCOL ({dt.get('type', 'unknown').upper()}):")
            print(f"    Priority:       {proto.get('search_rescue_priority', 'N/A')}")
            print(f"    Primary hazard: {proto.get('primary_hazard', 'N/A')}")
            print(f"    Survivor location: {proto.get('survivor_location', 'N/A')}")
            print(f"    Time window:    {proto.get('time_critical_window', 'N/A')}")
            equip = proto.get("equipment_needed", [])
            if equip:
                print(f"    Equipment:      {', '.join(equip[:5])}")

        print(f"\n  OUTPUT FILES:")
        for k, v in report.get("output_files", {}).items():
            print(f"    {k:25s}: {v}")
        print("=" * W + "\n")
