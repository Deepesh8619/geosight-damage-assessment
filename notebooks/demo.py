"""
GeoSight Demo Notebook (runnable Python script or convert to Jupyter)
Convert to notebook: jupytext --to notebook notebooks/demo.py

Demonstrates the full pipeline on synthetic data:
  1. Generate synthetic pre/post disaster images
  2. Run building segmentation
  3. Run damage classification
  4. Visualise outputs
  5. Export GeoJSON + interactive map
"""

# %% [markdown]
# # GeoSight: Post-Disaster Building Damage Assessment Demo
# This notebook walks through the full pipeline end-to-end using synthetic data
# (no download required). Replace paths with real xBD or Sentinel-2 images for
# production use.

# %% Setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 120

# %% [markdown]
# ## 1. Generate Synthetic Test Data

# %%
from scripts.download_data import generate_synthetic_dataset

generate_synthetic_dataset(
    output_dir="../data/sample",
    n_images=8,
    image_size=512,
    split="train",
)
print("Synthetic data generated.")

# %% [markdown]
# ## 2. Explore the Dataset

# %%
from src.pipeline.ingestion import XBDDataset
from src.pipeline.preprocessing import XBDTransform

dataset = XBDDataset(
    root_dir="../data/sample",
    split="train",
    transform=XBDTransform(phase="val", image_size=512),
)
print(f"Dataset size: {len(dataset)} samples")

# Load one sample
sample = dataset[0]
print("\nSample keys:", list(sample.keys()))
print("Pre image shape: ", sample["pre_image"].shape)
print("Post image shape:", sample["post_image"].shape)
print("Building mask shape:", sample["building_mask"].shape)
print("Damage mask shape:  ", sample["damage_mask"].shape)

# %% [markdown]
# ## 3. Visualise a Sample

# %%
import torch

def tensor_to_img(t):
    """Denormalize ImageNet-normalised tensor to uint8 numpy."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = t.numpy().transpose(1, 2, 0)
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)

DAMAGE_COLORS = {0: [0,0,0], 1: [0,200,0], 2: [255,255,0], 3: [255,140,0], 4: [220,0,0]}

pre_img   = tensor_to_img(sample["pre_image"])
post_img  = tensor_to_img(sample["post_image"])
bld_mask  = sample["building_mask"].numpy()
dmg_mask  = sample["damage_mask"].numpy()

# Color damage mask
dmg_rgb = np.zeros((*dmg_mask.shape, 3), dtype=np.uint8)
for cls, color in DAMAGE_COLORS.items():
    dmg_rgb[dmg_mask == cls] = color

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].imshow(pre_img);   axes[0].set_title("Pre-Disaster");       axes[0].axis("off")
axes[1].imshow(post_img);  axes[1].set_title("Post-Disaster");      axes[1].axis("off")
axes[2].imshow(bld_mask, cmap="Greens"); axes[2].set_title("Building Mask"); axes[2].axis("off")
axes[3].imshow(dmg_rgb);   axes[3].set_title("Damage Ground Truth"); axes[3].axis("off")
plt.tight_layout()
plt.savefig("../data/outputs/sample_visualization.png", bbox_inches="tight")
plt.show()
print("Sample visualization saved.")

# %% [markdown]
# ## 4. Initialize Models (no checkpoints — random weights for demo)

# %%
import torch
from src.models.damage_classifier import DamageAssessmentPipeline

pipeline = DamageAssessmentPipeline(device="cpu")
print("Pipeline initialized.")
print(f"  Segmentation model parameters: {pipeline.seg_model.n_parameters:,}")
print(f"  Damage model parameters:       {pipeline.dmg_model.n_parameters:,}")

# %% [markdown]
# ## 5. Run the Full Assessment Pipeline

# %%
from PIL import Image
import numpy as np

# Load raw images (pre-normalized)
img_dir   = Path("../data/sample/train/images")
pre_paths = sorted(img_dir.glob("*_pre_*.png"))
pre_path  = str(pre_paths[0])
post_path = pre_path.replace("_pre_", "_post_")

pre_raw  = np.array(Image.open(pre_path).convert("RGB"))
post_raw = np.array(Image.open(post_path).convert("RGB"))

print(f"Image size: {pre_raw.shape}")

# %%
result = pipeline.assess_full_scene(
    pre_image=pre_raw,
    post_image=post_raw,
    tile_size=256,
    overlap=32,
    batch_size=2,
)

print("\nAssessment complete!")
print("Building probability map shape:", result["building_prob_map"].shape)
print("Damage map shape:              ", result["damage_map"].shape)
print("\nStats:")
import json
print(json.dumps(result["stats"], indent=2))

# %% [markdown]
# ## 6. Visualise Assessment Results

# %%
from src.utils.viz_utils import plot_damage_assessment, plot_class_distribution

fig = plot_damage_assessment(
    pre_raw, post_raw,
    result["building_prob_map"],
    result["damage_map"],
    title=f"Demo Damage Assessment — {Path(pre_path).stem}",
    save_path="../data/outputs/demo_assessment.png",
)
plt.show()

fig2 = plot_class_distribution(
    result["stats"],
    title="Damage Class Distribution",
    save_path="../data/outputs/demo_distribution.png",
)
plt.show()

# %% [markdown]
# ## 7. Export GeoJSON and Interactive Leaflet Map

# %%
from rasterio.transform import from_bounds
from src.utils.geo_utils import polygonize_damage_map

h, w = result["damage_map"].shape
transform = from_bounds(0, 0, w, h, w, h)

gdf = polygonize_damage_map(
    result["damage_map"],
    transform,
    crs="EPSG:4326",
    building_mask=result["building_prob_map"] > 0.5,
)

print(f"\nExtracted {len(gdf)} building polygons:")
if not gdf.empty:
    print(gdf["damage_label"].value_counts())
    gdf.to_file("../data/outputs/demo_damage.geojson", driver="GeoJSON")
    print("GeoJSON saved: data/outputs/demo_damage.geojson")

# %%
if not gdf.empty:
    from src.utils.viz_utils import create_leaflet_map
    map_path = create_leaflet_map(
        gdf, output_path="../data/outputs/demo_map.html"
    )
    print(f"Interactive map: {map_path}")
    print("Open in browser to explore building-level damage interactively.")

# %% [markdown]
# ## 8. Scalable Pipeline Demo (Dask tiling on large raster)

# %%
import rasterio
import numpy as np
from rasterio.transform import from_bounds
from src.pipeline.tiling import RasterTiler

# Simulate a large scene by creating a temporary GeoTIFF
import tempfile, os
large_img = np.random.randint(0, 255, (3, 2048, 2048), dtype=np.uint8)

with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
    tmp_tif = f.name

with rasterio.open(
    tmp_tif, "w", driver="GTiff",
    height=2048, width=2048, count=3,
    dtype=np.uint8, crs="EPSG:4326",
    transform=from_bounds(0, 0, 1, 1, 2048, 2048),
) as dst:
    dst.write(large_img)

print(f"Simulated large scene: {tmp_tif}")

tiler = RasterTiler(tile_size=512, overlap=64)
dask_tiles, specs, meta = tiler.tile_geotiff(tmp_tif, normalize=True)

print(f"Tiles: {dask_tiles.shape[0]} (lazy Dask array, shape={dask_tiles.shape})")
print("Memory of full scene if loaded eagerly:", large_img.nbytes / 1e6, "MB")
print("Dask loads only requested tiles → constant memory footprint")

# Compute first 4 tiles (triggers actual I/O)
first_four = dask_tiles[:4].compute()
print(f"\nFirst 4 tiles computed: shape={first_four.shape}, dtype={first_four.dtype}")

os.unlink(tmp_tif)

# %% [markdown]
# ## Summary
#
# This demo showed:
# - **Project 3**: U-Net building segmentation on pre-disaster imagery
# - **Project 5**: 5-class damage classification using pre+post image pairs
# - **Project 7**: Dask-based scalable tiling for large satellite scenes
# - **Fusion**: Building masks constrain damage predictions (no false positives outside buildings)
# - **Outputs**: GeoTIFF, GeoJSON, interactive Leaflet map, statistical reports
#
# **Next steps with real data:**
# 1. Download xBD: `python scripts/download_data.py --mode instructions`
# 2. Generate synthetic: `python scripts/download_data.py --mode synthetic --n-images 200`
# 3. Train segmentation: `python scripts/train_segmentation.py --data-dir data/raw/xbd`
# 4. Train damage model: `python scripts/train_change_detection.py --data-dir data/raw/xbd`
# 5. Assess: `python scripts/run_assessment.py --pre PRE.tif --post POST.tif`
