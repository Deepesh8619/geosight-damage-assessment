# GeoSight — Complete Code Explanation
## Every file, every method, every algorithm

---

## Table of Contents

1. [Project Overview — How Everything Connects](#1-project-overview)
2. [File-by-File Breakdown](#2-file-by-file-breakdown)
3. [Algorithms & Techniques Used](#3-algorithms--techniques)
4. [Skills Demonstrated](#4-skills-demonstrated)
5. [Nothing is Secret](#5-nothing-is-secret)
6. [Where to Make Changes](#6-where-to-make-changes)

---

## 1. Project Overview

### The Data Flow (what happens when you run inference)

```
User provides: pre_disaster.png + post_disaster.png
                    │                    │
                    ▼                    ▼
           ┌── INGESTION ──┐    (src/pipeline/ingestion.py)
           │  Load images   │    Reads PNG/GeoTIFF, converts to numpy arrays
           └───────┬────────┘
                   │
           ┌── TILING ──────┐    (src/pipeline/tiling.py)
           │  Split into     │    1024x1024 → multiple 512x512 tiles
           │  overlapping    │    Overlap = 64px so tile edges blend smoothly
           │  512x512 tiles  │
           └───────┬────────┘
                   │
          ┌── SEGMENTATION ──┐   (src/models/segmentation.py)
          │  U-Net model      │   Input: pre-image tile (3 channels RGB)
          │  finds buildings  │   Output: probability map (0.0 to 1.0 per pixel)
          │  in pre-image     │   "Is this pixel a building? 0.92 = yes"
          └───────┬───────────┘
                  │
          ┌── DAMAGE MODEL ──┐   (src/models/siamese_unet.py)
          │  Siamese U-Net    │   Input: pre tile + post tile (3ch each)
          │  classifies       │   Output: 5-class probability per pixel
          │  damage level     │   "This pixel is 87% destroyed"
          └───────┬───────────┘
                  │
          ┌── FUSION ────────┐   (src/models/damage_classifier.py)
          │  damage × building│   Non-building pixels → forced to background
          │  = masked damage  │   Only buildings get damage labels
          └───────┬───────────┘
                  │
          ┌── REASSEMBLY ───┐   (src/pipeline/tiling.py)
          │  Stitch tiles    │   Cosine-weighted blending in overlap zones
          │  back together   │   No visible seams between tiles
          └───────┬──────────┘
                  │
     ┌── DISASTER CLASSIFIER ──┐  (src/models/disaster_classifier.py)
     │  What type of disaster?  │  Looks at damage PATTERNS (spatial stats)
     │  earthquake / flood /    │  + visual TEXTURE of post-image
     │  hurricane / wildfire    │  Scene-level prediction (not per-pixel)
     └───────────┬──────────────┘
                 │
     ┌── SPATIAL ANALYSIS ─────┐  (src/utils/spatial_analysis.py)
     │  Where is the epicentre? │  Centre-of-mass of severe damage
     │  What direction?         │  Sectoral damage gradient
     │  How many clusters?      │  Connected component analysis
     └───────────┬──────────────┘
                 │
     ┌── IMPACT ANALYSIS ──────┐  (src/utils/impact_analysis.py)
     │  How many displaced?     │  Buildings × occupancy rate × displacement prob
     │  Economic loss?          │  Building area × cost/m² × damage-to-loss ratio
     │  Shelter needs?          │  Displaced × 3.5 m²/person (UNHCR standard)
     └───────────┬──────────────┘
                 │
     ┌── OUTPUT ───────────────┐  (src/inference/assessor.py)
     │  GeoTIFF, GeoJSON,      │  + src/utils/geo_utils.py
     │  Leaflet map, JSON      │  + src/utils/viz_utils.py
     │  report, PNG figures     │
     └─────────────────────────┘
```

---

## 2. File-by-File Breakdown

---

### `src/pipeline/ingestion.py` — Data Loading

**Purpose:** Loads satellite images and their labels from disk into Python.

**Key class: `XBDDataset`** (inherits from PyTorch `Dataset`)

This is a PyTorch Dataset — it tells the training loop "here's how to load sample #N". PyTorch calls `__getitem__(n)` to get one training example.

**Method-by-method:**

```python
__init__(root_dir, split, task, transform)
```
- Sets up directory paths: `root/train/images/` and `root/train/labels/`
- Calls `_build_sample_list()` to find all valid pre/post pairs
- Stores the transform (augmentation pipeline) to apply to each sample

```python
_build_sample_list()
```
- Scans the image directory for all files matching `*_pre_*.png`
- For each pre-image, finds the matching post-image by replacing `_pre_` with `_post_`
- Checks that both label JSON files exist
- Returns a list of dicts: `{pre_image, post_image, pre_label, post_label, event}`
- If any file is missing, skips that pair with a warning

```python
__getitem__(idx)
```
- Called by PyTorch DataLoader to get one sample
- Loads pre and post images as numpy arrays (HxWx3, uint8)
- Loads pre and post JSON labels
- Calls `_rasterize_buildings()` → converts building polygons to a binary mask
- Calls `_rasterize_damage()` → converts damage polygons to a 5-class mask
- Applies the transform (augmentation) if provided
- Returns dict: `{pre_image, post_image, building_mask, damage_mask, event}`

```python
_rasterize_buildings(label_data, height, width)
```
- **What it does:** Converts vector polygons (building outlines) into a pixel mask
- **Algorithm:** Rasterization — "burns" polygon shapes onto a pixel grid
- **How:**
  1. Reads building polygons from the JSON label file
  2. xBD uses WKT (Well-Known Text) format: `"POLYGON ((x1 y1, x2 y2, ...))"` — these are pixel coordinates
  3. Parses WKT to Shapely geometry objects
  4. Uses `rasterio.features.rasterize()` to fill those polygons on a blank mask
  5. Pixel inside any building polygon → 1, outside → 0
- **Why:** Neural networks work with pixel grids, not polygons. We need the "answer sheet" as a grid.

```python
_rasterize_damage(label_data, height, width)
```
- Same as above but produces a 5-class mask instead of binary
- Each building polygon has a `subtype` property: "no-damage", "minor-damage", "major-damage", "destroyed"
- Rasterizes from least severe to most severe, so if polygons overlap, the more severe class wins
- Result: mask where each pixel is 0 (background), 1, 2, 3, or 4

**Key class: `GeoTiffScene`**

- Loads GeoTIFF satellite imagery (Sentinel-2, Planet, Maxar, etc.)
- Handles multi-band imagery (satellite images can have 4-13 bands, not just RGB)
- **Percentile normalization:** Instead of dividing by 255 (which assumes 8-bit), it clips at the 2nd and 98th percentile and stretches to [0,1]. This handles different satellite sensors that have different value ranges.

**Skills used:** PyTorch Dataset API, rasterio, shapely, WKT parsing, rasterization, geospatial I/O

---

### `src/pipeline/tiling.py` — Scalable Raster Processing

**Purpose:** Satellite scenes can be 10,000+ pixels wide. Models only handle 512x512. This module splits large images into tiles and reassembles predictions.

**Key class: `RasterTiler`**

```python
compute_tile_specs(raster_width, raster_height)
```
- Calculates a grid of tiles that covers the entire image
- **Overlap:** Tiles overlap by 64 pixels. Why? Without overlap, the model makes poor predictions at tile edges (no context). Overlap gives every pixel enough surrounding context.
- **Stride:** `tile_size - overlap` = 512 - 64 = 448 pixels between tile starts
- Edge tiles that would extend beyond the image are padded with zeros

```python
tile_image(image)
```
- Takes a full numpy image and cuts it into patches
- Returns list of tile arrays + list of TileSpec metadata (position, padding info)

```python
tile_geotiff(raster_path)
```
- **Dask lazy loading:** For multi-GB GeoTIFF files that don't fit in RAM
- Uses `@dask.delayed` to create lazy computation graphs
- Each tile is read from disk only when needed (not all at once)
- **This is the scalability component** — handles arbitrarily large satellite scenes with constant memory

```python
reassemble(tiles, specs, height, width)
```
- **The inverse of tiling:** Puts prediction tiles back together into one image
- **Cosine window blending:** In overlap zones where two tiles contribute predictions, uses a cosine (Hann) window to weight each tile's contribution
- Center of each tile gets weight ≈ 1.0 (high confidence)
- Edges get weight ≈ 0.0 (low confidence)
- This eliminates visible seams between tiles

```python
_cosine_window(size)
```
- **Algorithm:** 2D Hann window = outer product of two 1D Hann windows
- `np.hanning(512)` creates a bell curve: 0 at edges, 1 at center
- `np.outer(hann, hann)` extends to 2D
- Used as multiplicative weights during tile reassembly

**Skills used:** Dask parallel computing, numpy array slicing, sliding window algorithms, signal processing (Hann window)

---

### `src/pipeline/preprocessing.py` — Data Augmentation

**Purpose:** Make the training data more diverse so the model generalizes better.

```python
build_augmentation_pipeline(phase, image_size)
```

**Training augmentations (applied randomly to each sample):**
- `RandomCrop(512, 512)` — cut a random 512x512 patch from the 1024x1024 image. Each epoch the model sees different crops = more training variety
- `HorizontalFlip(p=0.5)` — 50% chance to mirror left-right. Buildings look the same either way
- `VerticalFlip(p=0.5)` — 50% chance to mirror top-bottom. Satellite images have no "up"
- `RandomRotate90(p=0.5)` — 90° rotation. Same reason
- `ShiftScaleRotate` — small random shift, zoom, rotation. Teaches invariance to minor alignment errors
- `ColorJitter` — randomly change brightness/contrast/saturation. Teaches invariance to lighting conditions
- `GaussianBlur` — slight blur. Simulates atmospheric haze
- `GaussNoise` — add random noise. Simulates sensor noise

**Critical: `additional_targets`**
- When we flip the image, we MUST flip the mask the same way
- Albumentations' `additional_targets` mechanism ensures all spatial transforms are applied identically to pre_image, post_image, building_mask, and damage_mask

**Normalization:**
- `A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)` — subtracts mean and divides by std per channel
- Uses ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Why ImageNet stats for satellite images?** Our encoder (ResNet34) was pretrained on ImageNet. Its neurons expect input in this range. Using different normalization would make the pretrained weights useless.

**`XBDTransform`** — Wrapper that applies the augmentation pipeline to our specific dict format

**Skills used:** Albumentations library, data augmentation theory, ImageNet transfer learning normalization

---

### `src/models/segmentation.py` — Building Detection Model

**Purpose:** Given a satellite image, mark every pixel that is part of a building.

**Architecture: U-Net with ResNet34 encoder**

```
U-Net is shaped like the letter U:

Input image (3, 512, 512)
        │
    ENCODER (ResNet34, going DOWN — shrinks image, extracts features)
        │
    [512, 512] → conv → [256, 256] → conv → [128, 128] → conv → [64, 64] → conv → [32, 32]
        │              │              │              │              │
        │              │              │              │         BOTTLENECK
        │              │              │              │              │
    DECODER (going UP — grows image back, using skip connections)
        │              │              │              │              │
    [512, 512] ← conv ← [256, 256] ← conv ← [128, 128] ← conv ← [64, 64] ← conv ← [32, 32]
        │         ↑              ↑              ↑              ↑
        │         └──────────────┘              │              │
        │         SKIP CONNECTIONS ─────────────┘──────────────┘
        │
    Output mask (1, 512, 512)
```

**Why U-Net?**
- The encoder learns WHAT features exist (edges, textures, shapes)
- The decoder learns WHERE those features are (precise pixel locations)
- Skip connections carry fine spatial details from encoder to decoder (otherwise, spatial precision is lost during downsampling)

**Why ResNet34 encoder?**
- ResNet34 was trained on 1.2 million images (ImageNet). It already knows what edges, corners, textures, and objects look like
- We reuse this knowledge (transfer learning) instead of learning from scratch
- Only fine-tune it on our satellite building data
- `encoder_weights="imagenet"` loads the pretrained weights

**Method: `forward(x)`**
- Input: batch of images (B, 3, 512, 512)
- Output: raw logits (B, 1, 512, 512) — not probabilities yet
- Passes through the full U-Net: encode → bottleneck → decode → 1x1 conv head

**Method: `predict_mask(x, threshold=0.5)`**
- Applies sigmoid to convert logits → probabilities (0-1)
- Thresholds at 0.5: probability > 0.5 → building (1), else → not building (0)

**Loss Functions:**

```python
class DiceLoss:
```
- **What:** Measures overlap between prediction and truth
- **Formula:** `1 - (2 * intersection) / (prediction + truth)`
- **Why:** Handles class imbalance. If only 5% of pixels are buildings, a model predicting "nothing" gets 95% accuracy but 0 Dice. Dice only cares about the building pixels.

```python
class BceDiceLoss:
```
- **Combines two losses:**
  - **BCE (Binary Cross-Entropy):** Standard per-pixel classification loss. "How wrong is each pixel?"
  - **Dice:** "How much do the predicted and true building regions overlap?"
- **Why combine?** BCE gives stable gradients (smooth training). Dice directly optimizes the metric we care about (IoU/overlap). Together they work better than either alone.

```python
build_segmentation_optimizer(model, lr)
```
- **Differential learning rates:**
  - Encoder (pretrained ResNet34): `lr * 0.1` = slow learning. These weights are already good from ImageNet — don't change them too fast
  - Decoder (randomly initialized): `lr * 1.0` = normal learning. These need to learn from scratch
- Uses AdamW optimizer (Adam with weight decay for regularization)

```python
build_lr_scheduler(optimizer, epochs)
```
- **Linear warmup (5 epochs):** Start with very low LR, gradually increase. Prevents destroying pretrained weights at the start
- **Cosine annealing:** After warmup, LR follows a cosine curve from high to near-zero. This is better than fixed LR because the model needs large updates early and fine adjustments later.

**Skills used:** U-Net architecture, transfer learning, ResNet, skip connections, BCE loss, Dice loss, learning rate scheduling, differential LR

---

### `src/models/siamese_unet.py` — Improved Damage Model

**Purpose:** Classify damage by comparing pre and post images. This is the upgraded model (V2).

**Why "Siamese"?**
- Siamese = twins. Two encoders that share the same weights
- Pre-image goes through encoder → features_pre
- Post-image goes through THE SAME encoder → features_post
- Since same weights process both, the features are directly comparable

**Why is this better than the old 6-channel approach?**

```
OLD (change_detection.py):
  Concatenate pre+post → [6 channels] → single encoder
  Problem: encoder has never seen 6-channel input, can't use ImageNet weights

NEW (siamese_unet.py):
  Pre → encoder → features_pre     (uses ImageNet weights, 3 channels)
  Post → encoder → features_post   (same encoder, same weights)
  Difference: |features_pre - features_post|  (explicit change signal)
  Fuse all three → decoder → damage map
```

**`FeatureDifferenceModule`**

At each encoder level (5 levels for ResNet34), this computes:
1. `|f_pre - f_post|` — absolute difference (what changed and by how much)
2. Concatenates: `[f_pre, f_post, |difference|]` → 3× the channels
3. A 1×1 convolution projects back to the original channel count

This explicitly tells the decoder "here's what was there, here's what's there now, and here's what changed."

**`predict_with_confidence`**
- Returns both the class prediction AND how confident the model is
- Confidence = probability of the predicted class (max of softmax)
- High confidence (0.95) = model is sure. Low (0.40) = model is guessing
- Used for rescue prioritization — don't send teams to uncertain predictions

**Skills used:** Siamese networks, feature differencing, multi-scale feature fusion, softmax confidence

---

### `src/models/change_detection.py` — Old Damage Model (V1, kept for compatibility)

**Architecture:** Single U-Net with 6-channel input (3 pre + 3 post concatenated)
- Simpler but less accurate than Siamese
- `encoder_weights=None` because 6 channels can't use 3-channel ImageNet weights

**`WeightedFocalLoss`**
- **Problem:** 90% of pixels are background. Model can get 90% accuracy by predicting "background" everywhere
- **Focal Loss formula:** `(1 - p_t)^gamma * CE_loss`
  - `gamma=2`: When model is confident and correct (p_t close to 1), `(1-p_t)^2` is tiny → loss is tiny (don't waste time on easy pixels)
  - When model is wrong (p_t close to 0), `(1-p_t)^2` is large → loss is large (focus on hard pixels)
- **Class weights:** `[0.1, 1.0, 2.0, 3.0, 4.0]` — destroyed buildings count 40× more than background pixels in the loss. Forces model to care about rare severe damage.

**Skills used:** Focal loss, class weighting for imbalanced datasets, multi-class segmentation

---

### `src/models/disaster_classifier.py` — Disaster Type Prediction

**Purpose:** Look at the damage PATTERN and predict what kind of disaster caused it.

**This is NOT a pixel-level model.** It's a scene-level classifier (one answer per image pair).

**`DamagePatternFeatureExtractor`**

Extracts two types of features:

**1. Statistical features (21 numbers, hand-crafted — no learning):**
- **Class ratios (5):** What % of pixels are in each damage class
- **Spatial concentration (4):** Are damaged pixels clustered or spread out? (variance, max, quantile)
- **Directional gradient (8):** Split image into 4 quadrants, compare damage intensity across them. High asymmetry = directional disaster (flood, wind). Low asymmetry = radial (earthquake)
- **Connectivity (4):** Ratio of destroyed-to-minor damage. Earthquakes have sharp boundaries (destroyed next to no-damage). Floods have gradual transitions.

**Why hand-crafted features?** Domain knowledge that different disasters leave distinct spatial signatures. A CNN might eventually learn these, but explicitly encoding them gives the model a head start and requires much less training data.

**2. Visual features (256 numbers, learned by a small CNN):**
- A lightweight 3-layer CNN looks at the post-disaster image texture
- Rubble (earthquake) looks different from waterlogged buildings (flood) which look different from charred structures (wildfire)
- `AdaptiveAvgPool2d(1)` squishes the spatial dimensions to a single vector

**`DisasterTypeClassifier`**
- Combines statistical + visual features → 277-dimensional vector
- 3-layer MLP (fully connected): 277 → 128 → 64 → 7 (one per disaster type)
- Dropout (0.3) prevents overfitting

**`get_disaster_response_protocol`**
- Pure domain knowledge, no ML
- Maps disaster type to actionable guidance: equipment needed, survivor locations, time windows
- This is what makes the output useful for responders, not just researchers

**`XBD_EVENT_TO_TYPE` mapping**
- Ground truth: which xBD events are which disaster type
- Used to generate training labels for the disaster classifier

**Skills used:** Feature engineering, CNN feature extraction, multi-modal fusion (statistical + visual), MLP classification, domain knowledge encoding

---

### `src/models/damage_classifier.py` — Pipeline Orchestrator

**Purpose:** Connects all models together and runs them in sequence.

**`DamageAssessmentPipeline`**

```python
__init__(seg_checkpoint, dmg_checkpoint, ...)
```
- Creates all three models (segmentation, damage, disaster classifier)
- Loads trained weights from checkpoint files
- Sets all models to `eval()` mode (disables dropout and batch norm training behavior)

```python
run_on_tiles(pre_tiles, post_tiles, batch_size)
```
- Processes tiles in batches (to fit in GPU memory)
- For each batch:
  1. Convert numpy tiles to normalized PyTorch tensors
  2. Run segmentation model → building probabilities
  3. Run damage model → damage class probabilities + confidence
  4. Mask: `damage_prediction × building_mask` — non-building pixels forced to class 0
- Returns building masks, damage maps, confidence maps, and raw probabilities

```python
_tiles_to_tensor(tiles)
```
- Converts list of (H,W,C) float32 numpy arrays to (B,C,H,W) PyTorch tensor
- Applies ImageNet normalization: `(pixel - mean) / std`

```python
assess_full_scene(pre_image, post_image, ...)
```
- The main method that runs everything end-to-end:
  1. Normalize images to float [0,1]
  2. Tile into 512×512 patches
  3. Run models on all tiles
  4. Reassemble tile predictions into full scene
  5. Run disaster type prediction
  6. Run spatial analysis
  7. Run impact analysis
  8. Generate priority zones
  9. Compute statistics
  10. Return everything in a dict

**Skills used:** Model serving, batch inference, pipeline orchestration, GPU memory management

---

### `src/inference/assessor.py` — Public API

**Purpose:** The single entry point that users call. Handles file I/O, format detection, output generation.

**`GeoSightAssessor.assess(pre_path, post_path, output_dir)`**

1. **Load image pair** — detects PNG vs GeoTIFF automatically
   - GeoTIFF: reads bands, extracts CRS and transform (geographic coordinates)
   - PNG: uses identity transform (pixel coordinates only)
   - Percentile normalization for GeoTIFF (satellite sensors have variable ranges)

2. **Run the pipeline** — calls `DamageAssessmentPipeline.assess_full_scene()`

3. **Save outputs:**
   - `save_geotiff()` — damage map and building mask as georeferenced rasters
   - `polygonize_damage_map()` → GeoJSON — converts pixel predictions to building polygons
   - `plot_damage_assessment()` — 4-panel figure
   - `create_leaflet_map()` — interactive HTML map
   - JSON report with all statistics

4. **Print report** — formatted humanitarian impact assessment to console

**Skills used:** File format detection, GeoTIFF I/O, rasterio, vector-raster conversion, report generation

---

### `src/utils/geo_utils.py` — Geospatial Operations

**`polygonize_damage_map(damage_map, transform, crs)`**
- **Algorithm:** Converts raster pixels to vector polygons
- Uses `rasterio.features.shapes()` which traces contiguous regions of same class and returns polygon geometries
- Filters out tiny polygons (< 10 pixels)
- Returns a GeoDataFrame with columns: geometry, damage_class, damage_label, area_m2

**`rasterize_polygons(geodataframe, raster_shape, transform)`**
- The reverse: burns vector polygons back onto a pixel grid
- Uses `rasterio.features.rasterize()`

**`save_geotiff(data, path, transform, crs)`**
- Writes a numpy array as a georeferenced GeoTIFF
- Includes CRS (coordinate system), transform (pixel-to-geographic mapping), and compression

**`reproject_to_wgs84(data, src_transform, src_crs)`**
- Converts raster from one coordinate system to WGS84 (lat/lon)
- Uses `rasterio.warp.reproject()`

**Skills used:** rasterio, geopandas, coordinate reference systems, vector-raster conversion, GeoTIFF I/O

---

### `src/utils/metrics.py` — Evaluation

**`SegmentationMetrics`**
- Tracks running TP/FP/FN/TN counts across batches
- Computes: IoU, Dice, F1, Precision, Recall, Accuracy
- **IoU:** `TP / (TP + FP + FN)` — strictest metric, ignores true negatives
- **Dice:** `2*TP / (2*TP + FP + FN)` — equivalent to F1 score for binary segmentation

**`DamageMetrics`**
- Per-class F1 score for all 4 damage classes
- **xView2 score:** Harmonic mean of per-class F1 (the competition's official metric)
- Harmonic mean penalizes any class being low — you can't get a good score by being great at one class and terrible at another
- Confusion matrix for detailed error analysis

**Skills used:** sklearn metrics, confusion matrices, harmonic mean, competition-style evaluation

---

### `src/utils/impact_analysis.py` — Humanitarian Impact

**`HumanitarianImpactAnalyzer.analyze()`**

**Step 1: Count buildings**
- Uses `scipy.ndimage.label()` — connected component analysis
- Finds contiguous groups of building pixels = individual buildings
- Each connected group gets a unique ID

**Step 2: Assign damage per building**
- For each building (connected component), looks at what damage class its pixels are
- Uses the most severe class present in >20% of the building's pixels
- Why 20% threshold? A building might have 1 destroyed pixel at the edge due to prediction noise — that shouldn't make the whole building "destroyed"

**Step 3: Population estimates**
- `buildings × occupancy_rate × displacement_probability`
- Occupancy rates from UN-HABITAT: urban dense = 8 people/building, suburban = 3.5
- Displacement: 0% for no-damage, 10% for minor, 80% for major, 100% for destroyed

**Step 4: Economic loss**
- `building_area × cost_per_m² × damage_loss_ratio`
- Cost/m² from World Bank: low-income = $250, middle = $600, high = $1500
- Loss ratio from FEMA HAZUS: minor = 10% of value, major = 50%, destroyed = 100%
- Reconstruction cost = loss × 1.3 (30% overhead for labor, logistics)

**Step 5: Shelter needs**
- `displaced_people × 3.5 m²/person` (UNHCR Sphere standards)
- Standard UNHCR tent = 16 m² → divide for tent count

**Step 6: Severity index (0-100)**
- Composite score: 40% destruction ratio + 30% displacement ratio + 30% log(scale)
- Categories: low (<15), moderate (15-35), severe (35-60), catastrophic (60+)

**`generate_priority_zones()`**
- Divides scene into 64×64 pixel grid cells
- Priority score per cell: `destroyed×4 + major×2 + minor×1`
- Weighted by model confidence (uncertain predictions get lower priority)
- Output is a heatmap: bright = go here first

**Skills used:** Connected component analysis (scipy), domain knowledge from FEMA/UNHCR/World Bank, composite scoring

---

### `src/utils/spatial_analysis.py` — Spatial Intelligence

**`find_damage_epicentre()`**
- **Algorithm:** Weighted centre of mass
- Weight map: destroyed pixels = 3.0, major = 1.0
- `scipy.ndimage.center_of_mass(weight_map)` → (y, x) coordinates
- Radius: standard deviation of distances from centre to all severe pixels
- Concentration: 1 - (std_distance / max_distance) → high = tightly clustered damage

**`find_damage_clusters()`**
- **Algorithm:** Connected component labeling
- Dilates severe damage mask (fills small gaps between nearby damage)
- `scipy.ndimage.label()` finds distinct clusters
- Reports centre, area, bounding box, and extent of each cluster
- Multiple clusters suggest: earthquake aftershocks, tornado skip pattern, or flood pockets

**`compute_damage_gradient()`**
- Divides scene into 8 angular sectors from centre (N, NE, E, SE, S, SW, W, NW)
- Computes average damage severity per sector
- If one sector has much more damage → directional disaster (wind, water flow)
- If sectors are similar → radial disaster (earthquake)
- `is_directional` flag: True if gradient magnitude > 0.5

**`compute_radial_profile()`**
- Creates 10 concentric rings from the epicentre
- Measures average damage severity in each ring
- Linear regression of severity vs distance → decay rate
- Steep negative decay = point-source (earthquake, explosion)
- Flat profile = wide-area (hurricane, flooding)

**`find_damage_boundary()`**
- Fills holes in the severe damage mask
- Subtracts eroded version from original → boundary/perimeter
- **Compactness:** `4π × area / perimeter²` — a circle has compactness = 1.0
- High compactness → circular damage (earthquake)
- Low compactness → elongated damage (tornado track, flood channel)

**`compute_spread_metrics()`**
- **Dispersion:** Ratio of convex hull area to actual damaged area. High = damage is scattered. Low = damage is concentrated
- Uses `scipy.spatial.ConvexHull` — smallest convex polygon containing all damage points
- **Coverage:** What fraction of the built-up area has any damage

**Skills used:** Computational geometry, centre of mass, connected components, angular sector analysis, convex hull, linear regression, morphological operations (dilation, erosion)

---

### `src/utils/viz_utils.py` — Visualization

- `plot_damage_assessment()` — 4-panel figure: pre, post, buildings, damage
- `plot_class_distribution()` — bar chart + pie chart of damage classes
- `create_leaflet_map()` — generates HTML with Folium library, color-coded building polygons, interactive tooltips
- `plot_training_curves()` — loss and metric over epochs

**Skills used:** matplotlib, Folium/Leaflet, color mapping, subplot layouts

---

## 3. Algorithms & Techniques

| Algorithm | Where Used | What It Does |
|-----------|-----------|--------------|
| **U-Net** | segmentation.py, siamese_unet.py | Encoder-decoder with skip connections for pixel-level prediction |
| **ResNet34** | All models (encoder) | 34-layer residual network pretrained on ImageNet — extracts visual features |
| **Siamese Network** | siamese_unet.py | Shared-weight twin encoders for image comparison |
| **Transfer Learning** | All models | Reuse ImageNet-trained weights instead of learning from scratch |
| **BCE Loss** | segmentation.py | Binary cross-entropy — standard binary classification loss |
| **Dice Loss** | segmentation.py | Overlap-based loss that handles class imbalance |
| **Focal Loss** | change_detection.py | Down-weights easy pixels, focuses on hard ones |
| **Cosine Annealing** | LR scheduler | Learning rate follows cosine curve: high→low for smooth convergence |
| **Data Augmentation** | preprocessing.py | Random flips, rotations, color jitter to prevent overfitting |
| **Rasterization** | ingestion.py, geo_utils.py | Vector polygon → pixel grid conversion |
| **Polygonization** | geo_utils.py | Pixel grid → vector polygon conversion (reverse) |
| **Connected Components** | impact_analysis.py | Label contiguous pixel regions as individual objects (buildings) |
| **Centre of Mass** | spatial_analysis.py | Find weighted centroid of damage distribution |
| **Convex Hull** | spatial_analysis.py | Smallest convex polygon enclosing all points |
| **Cosine Window** | tiling.py | Smooth blending of overlapping tile predictions |
| **Dask Lazy Evaluation** | tiling.py | Out-of-core processing for large files that don't fit in RAM |

---

## 4. Skills Demonstrated

| Skill Category | Specific Skills | Files |
|---------------|----------------|-------|
| **Deep Learning** | U-Net, ResNet, Siamese networks, transfer learning, loss function design, LR scheduling, gradient clipping | models/ |
| **Computer Vision** | Semantic segmentation, multi-class classification, change detection, data augmentation | models/, preprocessing.py |
| **Geospatial** | rasterio, GDAL, geopandas, GeoTIFF I/O, CRS, rasterization, polygonization, reprojection | geo_utils.py, ingestion.py |
| **Python** | OOP, type hints, dataclasses, generators, context managers, pathlib | Throughout |
| **PyTorch** | Dataset, DataLoader, custom loss functions, model checkpointing, MPS/CUDA device handling | Throughout |
| **Scalability** | Dask parallel processing, out-of-core computation, batched inference, memory management | tiling.py |
| **Domain Knowledge** | Disaster response protocols, FEMA HAZUS methodology, UNHCR standards, damage classification | impact_analysis.py, disaster_classifier.py |
| **Spatial Analysis** | Epicentre detection, directional gradients, radial profiles, morphological operations | spatial_analysis.py |
| **Visualization** | matplotlib, Folium/Leaflet interactive maps, training dashboards | viz_utils.py |
| **Software Engineering** | Modular architecture, config management, CLI scripts, unit tests, git | All |

---

## 5. Nothing is Secret

**There is NO proprietary algorithm in this project.** Everything is built from:

- **U-Net:** Published paper 2015, open-source, used by millions
- **ResNet34:** Published by Microsoft 2015, available in PyTorch/TensorFlow
- **Siamese architecture:** Published concept, used in xView2 winning solutions (open-source on GitHub)
- **Focal Loss:** Published by Facebook AI Research 2017
- **Dice Loss:** Published 2016, standard in medical/satellite segmentation
- **All libraries:** PyTorch, rasterio, geopandas, albumentations — all open-source
- **xBD dataset:** Publicly available from xview2.org
- **Impact formulas:** Published by FEMA, UNHCR, World Bank — public domain

**What IS your competitive advantage (if you commercialize):**
1. The trained model weights (the .pth files) — these are expensive to produce (data + compute)
2. The integration — no one else has all these pieces connected end-to-end
3. Domain expertise — knowing WHAT to compute (impact analysis, spatial patterns)
4. Speed of deployment — your pipeline runs in seconds, manual analysis takes days

---

## 6. Where to Make Changes

| If you want to... | Change this file | Specifically |
|-------------------|-----------------|-------------|
| Add more satellite sources | `ingestion.py` | Add a new loader class like `GeoTiffScene` |
| Change tile size | `config/config.yaml` | `data.tile_size` |
| Change model architecture | `segmentation.py` or `siamese_unet.py` | Change `encoder_name` to "resnet50", "efficientnet-b4", etc. |
| Add SAR support | `preprocessing.py` | Set `use_sar=True`, add SAR-specific normalization |
| Add more damage classes | `change_detection.py` | Change `DAMAGE_CLASSES` dict and `NUM_DAMAGE_CLASSES` |
| Add more disaster types | `disaster_classifier.py` | Add to `DISASTER_TYPES` dict and `XBD_EVENT_TO_TYPE` |
| Change population estimates | `impact_analysis.py` | Modify `OCCUPANCY_RATES` or `BUILDING_COST_PER_M2` |
| Change output format | `assessor.py` | Modify the `assess()` method's output section |
| Add a web API | Create `scripts/api.py` | Wrap `GeoSightAssessor` in FastAPI |
| Change training hyperparameters | `config/config.yaml` | Or pass as CLI flags to training scripts |
| Add new spatial analysis | `spatial_analysis.py` | Add method to `SpatialDamageAnalyzer` class |
