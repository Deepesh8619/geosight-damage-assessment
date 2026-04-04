# GeoSight: Satellite-Based Post-Disaster Building Damage Assessment

A production-grade deep learning pipeline that fuses **building segmentation** and **change detection** to classify building damage from before/after satellite image pairs — at scale.

---

## What It Does

Given a pair of satellite images taken before and after a disaster (hurricane, earthquake, flood, wildfire), GeoSight:

1. **Detects building footprints** in the pre-disaster image using a U-Net segmentation model
2. **Classifies damage** per building by comparing pre/post image pairs with a 6-channel change-detection model
3. **Produces** GeoTIFF outputs, GeoJSON building polygons with damage labels, statistics, and an interactive map

| Input | Output |
|-------|--------|
| Pre-disaster satellite image | Building footprint mask (GeoTIFF) |
| Post-disaster satellite image | Per-pixel damage map (GeoTIFF) |
| | Building polygons with damage class (GeoJSON) |
| | Interactive Leaflet damage map (HTML) |
| | Damage statistics report (JSON) |

### Damage Classes
| Class | Label | Colour |
|-------|-------|--------|
| 1 | No Damage | Green |
| 2 | Minor Damage | Yellow |
| 3 | Major Damage | Orange |
| 4 | Destroyed | Red |

---

## Architecture: Three Projects in One

```
┌────────────────────────────────────────────────────┐
│               Pre + Post Satellite Images           │
└────────────────────────────┬───────────────────────┘
                             │
               ┌─────────────▼────────────┐
               │  PROJECT 7: Dask Tiling  │  ← scalable, out-of-core
               │  RasterTiler + xarray    │    handles multi-GB scenes
               └──────┬──────────┬────────┘
                      │          │
          ┌───────────▼──┐  ┌────▼──────────────┐
          │ PROJECT 3    │  │ PROJECT 5          │
          │ U-Net        │  │ 6-channel U-Net    │
          │ Segmentation │  │ Change Detection   │
          │ (pre-image)  │  │ (pre + post pair)  │
          └───────┬──────┘  └────────┬───────────┘
                  │  building mask   │  damage probabilities
                  └────────┬─────────┘
                           │  FUSION
                    ┌──────▼───────┐
                    │  Masked      │
                    │  Damage Map  │  ← buildings only
                    └──────┬───────┘
                           │
          ┌────────────────┼──────────────────┐
          ▼                ▼                  ▼
       GeoTIFF          GeoJSON          Stats Report
     damage map      building polys    + Leaflet map
```

---

## Project Structure

```
geosight-damage-assessment/
├── config/config.yaml              # All hyperparameters and paths
├── src/
│   ├── pipeline/
│   │   ├── ingestion.py            # xBD dataset loader + GeoTIFF reader
│   │   ├── tiling.py               # Dask-based scalable raster tiler
│   │   └── preprocessing.py        # Normalization + Albumentations augmentation
│   ├── models/
│   │   ├── segmentation.py         # U-Net building segmentation + BCE-Dice loss
│   │   ├── change_detection.py     # 6-ch U-Net damage classifier + Focal loss
│   │   └── damage_classifier.py    # End-to-end fusion pipeline
│   ├── utils/
│   │   ├── geo_utils.py            # rasterize, polygonize, reproject, GeoTIFF I/O
│   │   ├── metrics.py              # IoU, Dice, xView2 score, per-class F1
│   │   └── viz_utils.py            # 4-panel figures, Leaflet maps, training curves
│   └── inference/
│       └── assessor.py             # GeoSightAssessor — public API
├── scripts/
│   ├── download_data.py            # xBD instructions + synthetic data generator
│   ├── train_segmentation.py       # Segmentation training loop
│   ├── train_change_detection.py   # Damage model training loop
│   └── run_assessment.py           # CLI for inference (single or batch)
├── notebooks/demo.py               # End-to-end demo (no data download needed)
└── tests/test_pipeline.py          # Unit + integration tests
```

---

## Quickstart (No Data Download Required)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate synthetic test data (20 images, ~10 seconds)
python scripts/download_data.py --mode synthetic --n-images 20

# 3. Run the demo notebook
jupyter notebook notebooks/demo.py
# or as a script:
python notebooks/demo.py

# 4. Run tests
pytest tests/ -v
```

---

## Training on Real Data

### Option A: xBD Dataset (recommended — real disaster events)
```bash
# Step 1: Register at https://xview2.org/ and download the dataset
# Step 2: Extract to data/raw/xbd/

# Train building segmentation
python scripts/train_segmentation.py \
    --data-dir data/raw/xbd \
    --epochs 100 \
    --batch-size 8

# Train damage classifier (uses seg checkpoint for building-aware training)
python scripts/train_change_detection.py \
    --data-dir data/raw/xbd \
    --epochs 80 \
    --batch-size 4 \
    --seg-checkpoint checkpoints/segmentation/best.pth
```

### Option B: SpaceNet 8 (no registration)
```bash
# Requires AWS CLI
pip install awscli
python scripts/download_data.py --mode spacenet8
```

---

## Inference

```bash
# Single image pair (PNG or GeoTIFF)
python scripts/run_assessment.py \
    --pre  data/raw/xbd/test/images/hurricane-harvey_pre_00000001.png \
    --post data/raw/xbd/test/images/hurricane-harvey_post_00000001.png \
    --seg-checkpoint checkpoints/segmentation/best.pth \
    --dmg-checkpoint checkpoints/damage/best.pth \
    --output-dir data/outputs/harvey_001

# Batch mode (entire test set)
python scripts/run_assessment.py \
    --batch-dir data/raw/xbd/test \
    --output-dir data/outputs/batch_results
```

### Python API
```python
from src.inference.assessor import GeoSightAssessor

assessor = GeoSightAssessor(
    seg_checkpoint="checkpoints/segmentation/best.pth",
    dmg_checkpoint="checkpoints/damage/best.pth",
)
report = assessor.assess(
    pre_image_path="pre.tif",
    post_image_path="post.tif",
    output_dir="data/outputs/my_event",
)
print(report["statistics"])
```

---

## Dataset

| Dataset | Size | Events | Source |
|---------|------|--------|--------|
| **xBD** | ~22k buildings, 19 events | Hurricane, Earthquake, Flood, Wildfire, Tsunami, Volcanic | [xview2.org](https://xview2.org/) — free, registration required |
| SpaceNet 8 | Multi-city flood | Flood | AWS S3 public bucket |
| Synthetic | Generated | Random | `scripts/download_data.py` |

---

## Geospatial Insights This System Can Extract

Once trained and run on real imagery, GeoSight can identify:

### Immediate (Rapid Response)
- **Collapse map**: Exact locations of destroyed buildings for search and rescue prioritisation
- **Damage extent**: % of affected built-up area per neighbourhood / grid cell
- **Safe zones**: Areas with intact buildings usable as evacuation shelters

### Analytical (Recovery Planning)
- **Damage gradient**: Radial damage intensity from epicentre/flood path — reveals disaster geometry
- **Infrastructure vulnerability**: Which building types / ages / materials sustain more damage (if combined with footprint attributes)
- **Socioeconomic correlation**: Overlay damage map with census data → assess equity of impact
- **Insurance loss estimation**: Building-level damage × area × cost/m² → estimated financial loss

### Temporal (Monitoring)
- **Reconstruction tracking**: Run periodically after disaster to monitor which buildings have been rebuilt
- **Urban sprawl risk**: Map encroachment into flood/earthquake risk zones over time
- **Deforestation watch**: Repurpose change detection for vegetation loss monitoring

---

## Skills Demonstrated (JD Mapping)

| Requirement | Demonstrated By |
|-------------|-----------------|
| Satellite imagery processing (multispectral) | `ingestion.py`, `geo_utils.py`, Sentinel-2 compatible |
| GDAL, rasterio, geopandas, xarray | Throughout pipeline |
| PyTorch / deep learning | `segmentation.py`, `change_detection.py` |
| Computer vision: segmentation, classification | U-Net models |
| Scalable / distributed processing | `tiling.py` with Dask |
| Model evaluation & validation | `metrics.py` — IoU, Dice, xView2 score |
| EO analytics product development | `assessor.py` — full product output |
| Documentation & reproducibility | This README, type hints, docstrings |

---

## License
MIT
