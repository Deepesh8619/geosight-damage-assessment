"""
Data download and preparation script.

xBD Dataset (primary — recommended):
  - Register at: https://xview2.org/
  - After approval (~1-2 days), download from the portal
  - Extract to: data/raw/xbd/

SpaceNet 8 (alternative — no registration needed, AWS S3 public bucket):
  - aws s3 ls s3://spacenet-dataset/spacenet/SN8_floods/
  - aws s3 sync s3://spacenet-dataset/spacenet/SN8_floods/ data/raw/spacenet8/

This script:
  1. Downloads the SpaceNet 8 sample data (small, no registration)
  2. Provides instructions for xBD
  3. Generates a synthetic mini-dataset for testing without any download
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from loguru import logger

# ---------------------------------------------------------------------------
# SpaceNet 8 public bucket paths (no auth required)
# ---------------------------------------------------------------------------
SN8_BUCKET = "s3://spacenet-dataset/spacenet/SN8_floods"
SN8_SAMPLE_FILES = [
    "train/PRE-event/10300100AF395C00/PRE-event_10300100AF395C00.tif",
    "train/POST-event/10300100AF395C00/POST-event_10300100AF395C00.tif",
]


def download_spacenet8_sample(output_dir: str = "data/raw/spacenet8"):
    """
    Download a small SpaceNet 8 pre/post image pair.
    Requires AWS CLI installed and configured (even anonymous).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Attempting to download SpaceNet 8 sample via AWS CLI ...")
    for rel_path in SN8_SAMPLE_FILES:
        src = f"{SN8_BUCKET}/{rel_path}"
        dst = str(Path(output_dir) / Path(rel_path).name)
        cmd = ["aws", "s3", "cp", "--no-sign-request", src, dst]
        logger.info(f"  {src}  →  {dst}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"  AWS CLI failed: {result.stderr.strip()}")
            logger.warning("  Tip: install awscli with: pip install awscli")
        else:
            logger.info("  Done.")


# ---------------------------------------------------------------------------
# Synthetic mini-dataset generator (no download required)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    output_dir: str = "data/sample",
    n_images: int = 20,
    image_size: int = 1024,
    split: str = "train",
):
    """
    Generate synthetic pre/post disaster image pairs with damage labels.

    Images are random RGB patches with synthetic building rectangles
    and simulated damage (colour shifts) for post-disaster images.
    Labels are valid xBD-format JSON files.

    Use this to verify the pipeline runs end-to-end without real data.
    """
    img_dir = Path(output_dir) / split / "images"
    lbl_dir = Path(output_dir) / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    DAMAGE_SUBTYPES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    rng = np.random.default_rng(42)

    for i in range(n_images):
        event_id = f"synthetic_{i:05d}"

        # --- Background (satellite texture) ---
        bg = rng.integers(20, 80, (image_size, image_size, 3), dtype=np.uint8)

        # --- Place synthetic buildings ---
        n_buildings = rng.integers(5, 25)
        buildings   = []
        for _ in range(n_buildings):
            bx = int(rng.integers(0, image_size - 80))
            by = int(rng.integers(0, image_size - 80))
            bw = int(rng.integers(20, 80))
            bh = int(rng.integers(20, 80))
            color = rng.integers(120, 220, 3).tolist()
            buildings.append({"x": bx, "y": by, "w": bw, "h": bh, "color": color})

        # --- Pre-disaster image ---
        pre_img = bg.copy()
        for b in buildings:
            pre_img[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] = b["color"]

        # --- Post-disaster image (simulate damage) ---
        post_img = pre_img.copy()
        damage_labels = []
        for b in buildings:
            dmg = rng.choice(DAMAGE_SUBTYPES)
            damage_labels.append(dmg)
            if dmg == "minor-damage":
                post_img[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] = np.clip(
                    np.array(b["color"]) * 0.7, 0, 255
                ).astype(np.uint8)
            elif dmg == "major-damage":
                debris = rng.integers(50, 150, 3).astype(np.uint8)
                post_img[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] = debris
            elif dmg == "destroyed":
                rubble = rng.integers(30, 80, 3).astype(np.uint8)
                post_img[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] = rubble

        # Save images
        Image.fromarray(pre_img).save(str(img_dir / f"{event_id}_pre_00000001.png"))
        Image.fromarray(post_img).save(str(img_dir / f"{event_id}_post_00000001.png"))

        # --- xBD JSON labels ---
        geotransform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]   # identity pixel coords

        def make_polygon(b):
            x, y, w, h = b["x"], b["y"], b["w"], b["h"]
            return {
                "type": "Polygon",
                "coordinates": [[
                    [x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]
                ]],
            }

        pre_features = [
            {
                "type": "Feature",
                "geometry": make_polygon(b),
                "properties": {"uid": f"building_{j}"},
            }
            for j, b in enumerate(buildings)
        ]
        post_features = [
            {
                "type": "Feature",
                "geometry": make_polygon(b),
                "properties": {"uid": f"building_{j}", "subtype": dmg},
            }
            for j, (b, dmg) in enumerate(zip(buildings, damage_labels))
        ]

        pre_label = {
            "features": {"xy": pre_features},
            "metadata": {
                "img_name":     f"{event_id}_pre_00000001.png",
                "geotransform": geotransform,
                "height":       image_size,
                "width":        image_size,
            },
        }
        post_label = {
            "features": {"xy": post_features},
            "metadata": {
                "img_name":     f"{event_id}_post_00000001.png",
                "geotransform": geotransform,
                "height":       image_size,
                "width":        image_size,
            },
        }

        with open(str(lbl_dir / f"{event_id}_pre_00000001.json"),  "w") as f:
            json.dump(pre_label, f)
        with open(str(lbl_dir / f"{event_id}_post_00000001.json"), "w") as f:
            json.dump(post_label, f)

    logger.info(
        f"Generated {n_images} synthetic image pairs → {output_dir}/{split}/"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GeoSight data download utility")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "spacenet8", "instructions"],
        default="synthetic",
        help=(
            "synthetic: generate test data locally (no download)\n"
            "spacenet8: download SpaceNet 8 sample (requires awscli)\n"
            "instructions: print xBD download instructions"
        ),
    )
    parser.add_argument("--output-dir",    default="data/sample",     help="Output directory")
    parser.add_argument("--n-images",      type=int, default=50,       help="Number of synthetic images")
    parser.add_argument("--image-size",    type=int, default=1024,     help="Image size for synthetic data")
    args = parser.parse_args()

    if args.mode == "synthetic":
        generate_synthetic_dataset(
            output_dir=args.output_dir,
            n_images=args.n_images,
            image_size=args.image_size,
            split="train",
        )
        generate_synthetic_dataset(
            output_dir=args.output_dir,
            n_images=max(5, args.n_images // 5),
            image_size=args.image_size,
            split="test",
        )

    elif args.mode == "spacenet8":
        download_spacenet8_sample(output_dir=args.output_dir)

    elif args.mode == "instructions":
        print("""
xBD Dataset (xView2 Challenge) — Download Instructions
=======================================================
1. Go to:    https://xview2.org/
2. Register for a free account
3. After approval, log in and download:
     - train_images_labels_targets.tar.gz
     - test_images_labels_targets.tar.gz
     - hold_images_labels_targets.tar.gz  (optional)
4. Extract to:  data/raw/xbd/
   Expected structure:
     data/raw/xbd/
       train/
         images/   *.png
         labels/   *.json
       test/
         images/
         labels/
5. Run: python scripts/train_segmentation.py --data-dir data/raw/xbd

SpaceNet 8 (no registration, AWS S3 public):
===========================================
  Install awscli: pip install awscli
  Download:
    aws s3 sync s3://spacenet-dataset/spacenet/SN8_floods/train/ data/raw/spacenet8/ --no-sign-request
  Note: SN8 has slightly different label format; use ingestion.GeoTiffScene
        for direct GeoTIFF loading instead of XBDDataset.
""")


if __name__ == "__main__":
    main()
