"""
CLI entry point for running post-disaster damage assessment.

Usage examples:

  # Single image pair (PNG from xBD)
  python scripts/run_assessment.py \
    --pre  data/raw/xbd/test/images/hurricane-harvey_pre_00000001.png \
    --post data/raw/xbd/test/images/hurricane-harvey_post_00000001.png \
    --seg-checkpoint  checkpoints/segmentation/best.pth \
    --dmg-checkpoint  checkpoints/damage/best.pth \
    --output-dir data/outputs/harvey_001

  # Single GeoTIFF pair (Sentinel-2 or similar)
  python scripts/run_assessment.py \
    --pre  data/raw/pre_event.tif \
    --post data/raw/post_event.tif \
    --output-dir data/outputs/my_event

  # Batch mode (process a directory of pairs)
  python scripts/run_assessment.py \
    --batch-dir data/raw/xbd/test \
    --output-dir data/outputs/batch_results
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="GeoSight: Post-Disaster Building Damage Assessment"
    )
    # Single pair mode
    parser.add_argument("--pre",          default=None, help="Pre-disaster image path (PNG or GeoTIFF)")
    parser.add_argument("--post",         default=None, help="Post-disaster image path (PNG or GeoTIFF)")
    parser.add_argument("--event-name",   default=None, help="Event identifier (inferred from filename if omitted)")

    # Batch mode
    parser.add_argument("--batch-dir",    default=None, help="Directory with images/ sub-dir for batch processing")

    # Model checkpoints
    parser.add_argument("--seg-checkpoint", default=None, help="Building segmentation checkpoint path")
    parser.add_argument("--dmg-checkpoint", default=None, help="Damage classification checkpoint path")

    # Output
    parser.add_argument("--output-dir",   default="data/outputs")

    # Pipeline config
    parser.add_argument("--tile-size",    type=int,   default=512)
    parser.add_argument("--tile-overlap", type=int,   default=64)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--seg-threshold", type=float, default=0.5)
    parser.add_argument("--device",        default=None)

    # Output flags
    parser.add_argument("--no-geotiff",   action="store_true")
    parser.add_argument("--no-geojson",   action="store_true")
    parser.add_argument("--no-figures",   action="store_true")
    parser.add_argument("--no-leaflet",   action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    from src.inference.assessor import GeoSightAssessor

    assessor = GeoSightAssessor(
        seg_checkpoint=args.seg_checkpoint,
        dmg_checkpoint=args.dmg_checkpoint,
        device=args.device,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        batch_size=args.batch_size,
        seg_threshold=args.seg_threshold,
    )

    # ---------------------------------------------------------------- Batch mode
    if args.batch_dir:
        img_dir = Path(args.batch_dir) / "images"
        if not img_dir.exists():
            logger.error(f"images/ directory not found in: {args.batch_dir}")
            sys.exit(1)

        pre_images = sorted(img_dir.glob("*_pre_*.png")) + \
                     sorted(img_dir.glob("*_pre_*.tif"))

        pairs = []
        for pre in pre_images:
            stem = pre.stem
            post_stem = stem.replace("_pre_", "_post_")
            for ext in [".png", ".tif", ".tiff"]:
                post = img_dir / f"{post_stem}{ext}"
                if post.exists():
                    pairs.append((str(pre), str(post)))
                    break

        if not pairs:
            logger.error("No matching pre/post pairs found in batch dir.")
            sys.exit(1)

        logger.info(f"Found {len(pairs)} image pairs. Starting batch assessment...")
        reports = assessor.assess_batch(
            pairs=pairs,
            output_base_dir=args.output_dir,
            save_geotiff_output=not args.no_geotiff,
            save_geojson_output=not args.no_geojson,
            save_figures=       not args.no_figures,
            save_leaflet_map=   not args.no_leaflet,
        )

        # Aggregate summary
        total_buildings_area = 0
        damage_summary = {"no-damage": 0, "minor-damage": 0, "major-damage": 0, "destroyed": 0}
        for rep in reports:
            if "error" in rep:
                continue
            stats = rep.get("statistics", {})
            total_buildings_area += stats.get("total_building_area_m2", 0)
            for k in damage_summary:
                damage_summary[k] += stats.get(k, {}).get("area_m2", 0)

        logger.info("\n" + "=" * 50)
        logger.info(f"BATCH SUMMARY ({len(pairs)} events)")
        logger.info(f"  Total building area:  {total_buildings_area:,.0f} m²")
        for k, v in damage_summary.items():
            pct = (v / max(total_buildings_area, 1)) * 100
            logger.info(f"  {k:15s}: {v:10,.0f} m²  ({pct:.1f}%)")
        logger.info("=" * 50)

        return

    # ---------------------------------------------------------------- Single pair mode
    if not args.pre or not args.post:
        logger.error("Provide --pre and --post image paths (or use --batch-dir)")
        sys.exit(1)

    report = assessor.assess(
        pre_image_path=args.pre,
        post_image_path=args.post,
        output_dir=args.output_dir,
        event_name=args.event_name,
        save_geotiff_output=not args.no_geotiff,
        save_geojson_output=not args.no_geojson,
        save_figures=       not args.no_figures,
        save_leaflet_map=   not args.no_leaflet,
    )

    GeoSightAssessor._print_report(report)


if __name__ == "__main__":
    main()
