"""
Test GeoSight on Indian Satellite Imagery.

This script validates that the model works on Indian data — critical
if you're targeting NDMA (National Disaster Management Authority)
or Indian insurance companies.

Data sources for Indian disaster imagery:

1. ISRO Bhuvan (free, Indian satellite data):
   - URL: https://bhuvan.nrsc.gov.in/
   - Provides: Cartosat, Resourcesat imagery
   - Disaster events: Kerala floods, Cyclone Fani, Uttarakhand landslides

2. Copernicus EMS (free, has Indian activations):
   - URL: https://emergency.copernicus.eu/mapping/
   - Search for: India, flood, cyclone, earthquake
   - Provides: Pre/post grading maps with damage assessment

3. Maxar Open Data (free, recent disasters):
   - URL: https://www.maxar.com/open-data
   - Has: Cyclone events in India

4. Google Earth Engine (free for research):
   - Sentinel-2 imagery over any Indian location

Usage:
    # After training, test on Indian imagery
    python3 scripts/test_indian_imagery.py \
        --pre  data/raw/india/pre_event.tif \
        --post data/raw/india/post_event.tif \
        --seg-checkpoint checkpoints/segmentation/best.pth \
        --dmg-checkpoint checkpoints/damage/best.pth \
        --output-dir data/outputs/india_test

    # Or use Sentinel-2 auto-download for a specific Indian event
    # Example: 2024 Wayanad landslides (Kerala)
    python3 scripts/fetch_satellite.py \
        --lat 11.61 --lon 76.08 \
        --pre-date 2024-07-01 \
        --post-date 2024-08-01 \
        --output-dir data/raw/india_wayanad

    # Example: 2023 Joshimath subsidence (Uttarakhand)
    python3 scripts/fetch_satellite.py \
        --lat 30.55 --lon 79.56 \
        --pre-date 2022-12-01 \
        --post-date 2023-02-01 \
        --output-dir data/raw/india_joshimath
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Indian disaster events of interest for testing
INDIAN_EVENTS = {
    "wayanad_landslides_2024": {
        "lat": 11.61, "lon": 76.08,
        "pre_date": "2024-07-01", "post_date": "2024-08-01",
        "disaster_type": "landslide",
        "description": "Wayanad, Kerala — massive landslides, 400+ casualties",
    },
    "joshimath_subsidence_2023": {
        "lat": 30.55, "lon": 79.56,
        "pre_date": "2022-12-01", "post_date": "2023-02-01",
        "disaster_type": "subsidence",
        "description": "Joshimath, Uttarakhand — land subsidence, buildings cracking",
    },
    "cyclone_biparjoy_2023": {
        "lat": 23.25, "lon": 68.56,
        "pre_date": "2023-06-01", "post_date": "2023-06-20",
        "disaster_type": "hurricane",
        "description": "Cyclone Biparjoy, Gujarat coast",
    },
    "assam_floods_2022": {
        "lat": 26.14, "lon": 91.74,
        "pre_date": "2022-05-01", "post_date": "2022-06-20",
        "disaster_type": "flood",
        "description": "Assam floods — millions displaced annually",
    },
    "turkey_syria_earthquake_2023": {
        "lat": 37.17, "lon": 37.04,
        "pre_date": "2023-01-01", "post_date": "2023-02-10",
        "disaster_type": "earthquake",
        "description": "Turkey-Syria earthquake (for cross-validation with xBD earthquakes)",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Test GeoSight on Indian imagery")
    parser.add_argument("--pre",  default=None, help="Pre-disaster image path")
    parser.add_argument("--post", default=None, help="Post-disaster image path")
    parser.add_argument("--seg-checkpoint", default="checkpoints/segmentation/best.pth")
    parser.add_argument("--dmg-checkpoint", default="checkpoints/damage/best.pth")
    parser.add_argument("--output-dir",     default="data/outputs/india_test")
    parser.add_argument("--event",          default=None,
                        choices=list(INDIAN_EVENTS.keys()),
                        help="Download and test a specific Indian event")
    parser.add_argument("--list-events", action="store_true",
                        help="List available Indian disaster events")
    args = parser.parse_args()

    if args.list_events:
        print("\nAvailable Indian disaster events for testing:\n")
        for name, info in INDIAN_EVENTS.items():
            print(f"  {name}:")
            print(f"    {info['description']}")
            print(f"    Location: ({info['lat']}, {info['lon']})")
            print(f"    Dates: {info['pre_date']} → {info['post_date']}")
            print(f"    Type: {info['disaster_type']}")
            print()
        print("Usage:")
        print("  python3 scripts/test_indian_imagery.py --event wayanad_landslides_2024")
        print("  (requires Copernicus credentials — see scripts/fetch_satellite.py)")
        return

    if args.event:
        event = INDIAN_EVENTS[args.event]
        print(f"\nEvent: {event['description']}")
        print(f"This will download Sentinel-2 imagery from Copernicus (free).")
        print(f"\nRun this first:")
        print(f"  python3 scripts/fetch_satellite.py \\")
        print(f"    --lat {event['lat']} --lon {event['lon']} \\")
        print(f"    --pre-date {event['pre_date']} \\")
        print(f"    --post-date {event['post_date']} \\")
        print(f"    --output-dir data/raw/india_{args.event}")
        print(f"\nThen run assessment:")
        print(f"  python3 scripts/run_assessment.py \\")
        print(f"    --pre data/raw/india_{args.event}/pre.tif \\")
        print(f"    --post data/raw/india_{args.event}/post.tif \\")
        print(f"    --seg-checkpoint {args.seg_checkpoint} \\")
        print(f"    --dmg-checkpoint {args.dmg_checkpoint} \\")
        print(f"    --output-dir {args.output_dir}/{args.event}")
        return

    if not args.pre or not args.post:
        print("Provide --pre and --post image paths, or use --event or --list-events")
        return

    # Run assessment
    from src.inference.assessor import GeoSightAssessor

    assessor = GeoSightAssessor(
        seg_checkpoint=args.seg_checkpoint,
        dmg_checkpoint=args.dmg_checkpoint,
    )

    report = assessor.assess(
        pre_image_path=args.pre,
        post_image_path=args.post,
        output_dir=args.output_dir,
    )

    GeoSightAssessor._print_report(report)

    # Compare with expected disaster type
    predicted_type = report.get("disaster_type", {}).get("type", "unknown")
    print(f"\n  NOTE: Model was trained on xBD (US/global disasters).")
    print(f"  Indian building styles may differ — if results look wrong,")
    print(f"  fine-tune on Indian data for better accuracy.")
    print(f"  ISRO Bhuvan (https://bhuvan.nrsc.gov.in/) has Indian satellite data.")


if __name__ == "__main__":
    main()
