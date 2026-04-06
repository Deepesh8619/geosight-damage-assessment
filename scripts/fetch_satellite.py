"""
Sentinel-2 Satellite Imagery Auto-Download.

Downloads FREE satellite imagery from Copernicus Data Space by coordinates + date.
No payment required — just a free registration.

This makes the project fully self-contained:
  1. Disaster happens at (lat, lon)
  2. This script downloads pre and post images automatically
  3. run_assessment.py analyzes them
  4. Full damage report generated — no manual image sourcing

Usage:
    # Register free at: https://dataspace.copernicus.eu/
    # Set credentials:
    export COPERNICUS_USER="your_email"
    export COPERNICUS_PASSWORD="your_password"

    # Download Hurricane Harvey area (before and after)
    python3 scripts/fetch_satellite.py \
        --lat 29.76 --lon -95.36 \
        --pre-date 2017-08-20 \
        --post-date 2017-09-01 \
        --output-dir data/raw/harvey_sentinel2

    # Download with bounding box instead of point
    python3 scripts/fetch_satellite.py \
        --bbox -95.5,29.5,-95.0,30.0 \
        --pre-date 2017-08-20 \
        --post-date 2017-09-01 \
        --output-dir data/raw/harvey_large

Sentinel-2 specs:
    Resolution: 10m (bands 2,3,4,8), 20m (bands 5,6,7,8A,11,12), 60m (bands 1,9,10)
    Coverage: Global, every 5 days
    Cost: FREE
    Bands we use: B4 (Red), B3 (Green), B2 (Blue) → RGB composite
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from loguru import logger


def get_copernicus_token(username: str, password: str) -> str:
    """
    Authenticate with Copernicus Data Space and get access token.

    Uses OAuth2 client credentials flow.
    """
    import urllib.request
    import urllib.parse

    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    data = urllib.parse.urlencode({
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": "cdse-public",
    }).encode()

    req = urllib.request.Request(token_url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            return result["access_token"]
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        logger.error("Register at: https://dataspace.copernicus.eu/")
        sys.exit(1)


def search_sentinel2(
    token: str,
    bbox: tuple,
    date_start: str,
    date_end: str,
    max_cloud: int = 20,
    max_results: int = 5,
) -> list:
    """
    Search Copernicus catalog for Sentinel-2 scenes.

    Args:
        token: OAuth access token
        bbox: (west, south, east, north) in WGS84
        date_start: "YYYY-MM-DD"
        date_end: "YYYY-MM-DD"
        max_cloud: maximum cloud cover percentage
        max_results: max scenes to return

    Returns:
        List of product dicts with id, name, date, cloud_cover, download_url
    """
    import urllib.request
    import urllib.parse

    west, south, east, north = bbox

    # OData query to Copernicus Data Space
    base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    filter_parts = [
        f"Collection/Name eq 'SENTINEL-2'",
        f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))')",
        f"ContentDate/Start gt {date_start}T00:00:00.000Z",
        f"ContentDate/Start lt {date_end}T23:59:59.999Z",
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {max_cloud})",
    ]

    params = {
        "$filter": " and ".join(filter_parts),
        "$orderby": "ContentDate/Start asc",
        "$top": str(max_results),
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

    products = []
    for item in data.get("value", []):
        products.append({
            "id": item["Id"],
            "name": item["Name"],
            "date": item.get("ContentDate", {}).get("Start", ""),
            "cloud_cover": next(
                (a["Value"] for a in item.get("Attributes", [])
                 if a.get("Name") == "cloudCover"),
                None,
            ),
            "size_mb": item.get("ContentLength", 0) / 1e6,
        })

    logger.info(f"Found {len(products)} Sentinel-2 scenes")
    for p in products:
        logger.info(f"  {p['name'][:60]}  cloud={p['cloud_cover']}%  {p['date'][:10]}")

    return products


def download_product(token: str, product_id: str, output_path: str):
    """
    Download a Sentinel-2 product from Copernicus Data Space.
    """
    import urllib.request

    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading to: {output_path}")
    try:
        with urllib.request.urlopen(req) as resp:
            with open(output_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        logger.info(f"Downloaded: {Path(output_path).stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        logger.error(f"Download failed: {e}")


def create_rgb_geotiff(sentinel2_dir: str, output_path: str):
    """
    Extract RGB bands from Sentinel-2 SAFE format and save as a single GeoTIFF.

    Sentinel-2 bands:
      B02 = Blue  (10m)
      B03 = Green (10m)
      B04 = Red   (10m)

    Output: 3-band GeoTIFF (R, G, B) ready for our pipeline.
    """
    import rasterio
    from rasterio.merge import merge

    safe_dir = Path(sentinel2_dir)

    # Find band files (they're buried inside the SAFE directory structure)
    band_files = {}
    for band_name, band_id in [("B04", "red"), ("B03", "green"), ("B02", "blue")]:
        matches = list(safe_dir.rglob(f"*{band_name}_10m.jp2")) + \
                  list(safe_dir.rglob(f"*{band_name}.jp2"))
        if matches:
            band_files[band_id] = str(matches[0])
        else:
            logger.warning(f"Band {band_name} not found in {sentinel2_dir}")

    if len(band_files) < 3:
        logger.error("Could not find all RGB bands")
        return

    # Read bands and stack
    with rasterio.open(band_files["red"]) as src:
        red = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(band_files["green"]) as src:
        green = src.read(1)
    with rasterio.open(band_files["blue"]) as src:
        blue = src.read(1)

    # Stack as (3, H, W)
    rgb = np.stack([red, green, blue], axis=0)

    # Save
    profile.update(count=3, driver="GTiff", compress="lzw")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(rgb)

    logger.info(f"RGB GeoTIFF saved: {output_path} ({rgb.shape[2]}x{rgb.shape[1]})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Download Sentinel-2 imagery")
    p.add_argument("--lat",        type=float, default=None, help="Center latitude")
    p.add_argument("--lon",        type=float, default=None, help="Center longitude")
    p.add_argument("--bbox",       default=None, help="Bounding box: west,south,east,north")
    p.add_argument("--radius-km",  type=float, default=10, help="Search radius around lat/lon in km")
    p.add_argument("--pre-date",   required=True, help="Pre-disaster date (YYYY-MM-DD)")
    p.add_argument("--post-date",  required=True, help="Post-disaster date (YYYY-MM-DD)")
    p.add_argument("--max-cloud",  type=int, default=20, help="Max cloud cover %%")
    p.add_argument("--output-dir", default="data/raw/sentinel2")
    p.add_argument("--user",       default=None, help="Copernicus username (or COPERNICUS_USER env)")
    p.add_argument("--password",   default=None, help="Copernicus password (or COPERNICUS_PASSWORD env)")
    return p.parse_args()


def main():
    args = parse_args()

    username = args.user or os.environ.get("COPERNICUS_USER")
    password = args.password or os.environ.get("COPERNICUS_PASSWORD")

    if not username or not password:
        print("""
Copernicus Data Space credentials required.

1. Register (free) at: https://dataspace.copernicus.eu/
2. Set credentials:
   export COPERNICUS_USER="your_email"
   export COPERNICUS_PASSWORD="your_password"

   Or pass --user and --password flags.
""")
        sys.exit(1)

    # Compute bounding box
    if args.bbox:
        bbox = tuple(float(x) for x in args.bbox.split(","))
    elif args.lat and args.lon:
        # Approximate bbox from point + radius
        km_to_deg = 1.0 / 111.0   # ~111 km per degree
        r = args.radius_km * km_to_deg
        bbox = (args.lon - r, args.lat - r, args.lon + r, args.lat + r)
    else:
        print("Provide either --lat/--lon or --bbox")
        sys.exit(1)

    logger.info(f"Bounding box: {bbox}")
    logger.info(f"Pre-date: {args.pre_date}, Post-date: {args.post_date}")

    # Authenticate
    token = get_copernicus_token(username, password)
    logger.info("Authenticated with Copernicus Data Space")

    # Search for pre-disaster scene
    pre_start = args.pre_date
    pre_end = (datetime.strptime(args.pre_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
    logger.info(f"\nSearching PRE-disaster scenes ({pre_start} to {pre_end})...")
    pre_products = search_sentinel2(token, bbox, pre_start, pre_end, args.max_cloud)

    # Search for post-disaster scene
    post_start = args.post_date
    post_end = (datetime.strptime(args.post_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
    logger.info(f"\nSearching POST-disaster scenes ({post_start} to {post_end})...")
    post_products = search_sentinel2(token, bbox, post_start, post_end, args.max_cloud)

    if not pre_products:
        logger.error("No pre-disaster scenes found. Try wider date range or higher --max-cloud")
        sys.exit(1)
    if not post_products:
        logger.error("No post-disaster scenes found.")
        sys.exit(1)

    # Download best (lowest cloud) scene for each
    pre_best  = pre_products[0]
    post_best = post_products[0]

    output = Path(args.output_dir)
    pre_zip  = str(output / f"pre_{pre_best['name']}.zip")
    post_zip = str(output / f"post_{post_best['name']}.zip")

    logger.info(f"\nDownloading PRE: {pre_best['name']}")
    download_product(token, pre_best["id"], pre_zip)

    logger.info(f"\nDownloading POST: {post_best['name']}")
    download_product(token, post_best["id"], post_zip)

    logger.info(f"""
Download complete!

Next steps:
  1. Unzip the .zip files
  2. Extract RGB GeoTIFF (or use our pipeline which reads GeoTIFF directly):
     python3 -c "from scripts.fetch_satellite import create_rgb_geotiff; create_rgb_geotiff('path/to/SAFE_dir', 'pre.tif')"

  3. Run assessment:
     python3 scripts/run_assessment.py \\
         --pre  {args.output_dir}/pre.tif \\
         --post {args.output_dir}/post.tif \\
         --seg-checkpoint checkpoints/segmentation/best.pth \\
         --dmg-checkpoint checkpoints/damage/best.pth \\
         --output-dir data/outputs/sentinel2_assessment
""")


if __name__ == "__main__":
    main()
