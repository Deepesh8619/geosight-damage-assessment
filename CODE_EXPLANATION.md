# GeoSight — Complete Code Explanation (V2)
## Every file, every line, every algorithm, every "why"

**Total codebase: 8,177 lines of Python across 35 files**

---

## Table of Contents

1. [How Everything Connects — The Big Picture](#1-how-everything-connects)
2. [Pipeline Layer — Getting Data In](#2-pipeline-layer)
   - [ingestion.py — Loading Images & Labels](#ingestionpy)
   - [tiling.py — Splitting Large Images](#tilingpy)
   - [preprocessing.py — Augmentation & Normalization](#preprocessingpy)
3. [Model Layer — The AI Brain](#3-model-layer)
   - [segmentation.py — Finding Buildings](#segmentationpy)
   - [siamese_unet.py — Detecting Damage (V2)](#siamese_unetpy)
   - [change_detection.py — Detecting Damage (V1)](#change_detectionpy)
   - [attention.py — CBAM Attention Module](#attentionpy)
   - [tta.py — Test Time Augmentation](#ttapy)
   - [disaster_classifier.py — What Disaster Happened](#disaster_classifierpy)
   - [damage_classifier.py — Pipeline Orchestrator](#damage_classifierpy)
4. [Analysis Layer — Making Sense of Results](#4-analysis-layer)
   - [spatial_analysis.py — Where & How Damage Spreads](#spatial_analysispy)
   - [impact_analysis.py — Humanitarian Numbers](#impact_analysispy)
   - [ml_analysis.py — Classical ML & Unsupervised Learning](#ml_analysispy)
5. [Utilities Layer — Output & Visualization](#5-utilities-layer)
   - [geo_utils.py — Geospatial Operations](#geo_utilspy)
   - [metrics.py — Measuring Accuracy](#metricspy)
   - [viz_utils.py — Visualization](#viz_utilspy)
6. [Inference Layer — Running Predictions](#6-inference-layer)
   - [assessor.py — The Public API](#assessorpy)
7. [Production Layer — Deployment & Data](#7-production-layer)
   - [api.py — REST API Server](#apipy)
   - [fetch_satellite.py — Auto-Download Imagery](#fetch_satellitepy)
8. [Scripts — Training & CLI](#8-scripts)
   - [train_with_visuals.py — Visual Training Loop](#train_with_visualspy)
   - [run_assessment.py — CLI Entry Point](#run_assessmentpy)
   - [download_data.py — Data Preparation](#download_datapy)
9. [Every Algorithm Used](#9-every-algorithm-used)
10. [Every Skill Demonstrated](#10-every-skill-demonstrated)
11. [Nothing Is Secret](#11-nothing-is-secret)
12. [Where To Make Changes](#12-where-to-make-changes)

---

## 1. How Everything Connects

When you run `python3 scripts/run_assessment.py --pre PRE.png --post POST.png`, here's every step that happens:

```
USER INPUT: pre_disaster.png + post_disaster.png
  │
  ▼
[assessor.py] GeoSightAssessor.assess()
  │  Detects file format (PNG vs GeoTIFF)
  │  Loads both images as numpy arrays
  │  Extracts geographic metadata if GeoTIFF
  │
  ▼
[tiling.py] RasterTiler.tile_image()
  │  Splits 1024×1024 image into 512×512 tiles
  │  Adds 64px overlap between tiles
  │  Pads edge tiles to exactly 512×512
  │
  ▼
[damage_classifier.py] DamageAssessmentPipeline.run_on_tiles()
  │  For each batch of tiles:
  │  │
  │  ├─► [segmentation.py] BuildingSegmentationModel.forward()
  │  │     Pre-image → U-Net (ResNet34 encoder) → building probability per pixel
  │  │     Output: 0.0 (not building) to 1.0 (definitely building)
  │  │
  │  ├─► [siamese_unet.py] SiameseUNet.forward()
  │  │     Pre-image → Shared Encoder → features_pre
  │  │     Post-image → Same Encoder → features_post
  │  │     |features_pre - features_post| → change signal
  │  │     [features_pre + features_post + change] → Decoder → 5-class damage
  │  │     Output: probability for each class per pixel
  │  │
  │  └─► FUSION: damage_prediction × (building_prob > 0.5)
  │       Non-building pixels forced to class 0 (background)
  │
  ▼
[tiling.py] RasterTiler.reassemble()
  │  Stitches tile predictions back into full image
  │  Cosine window blending eliminates tile-edge seams
  │
  ▼
[disaster_classifier.py] DisasterTypeClassifier.predict()
  │  Extracts 21 spatial statistics from damage map
  │  CNN extracts visual texture features from post-image
  │  Combined features → MLP → "earthquake" (73% confidence)
  │
  ▼
[spatial_analysis.py] SpatialDamageAnalyzer.full_analysis()
  │  Finds damage epicentre (weighted centre of mass)
  │  Detects damage clusters (connected components)
  │  Computes directional gradient (8 angular sectors)
  │  Measures radial decay profile (epicentre → outward)
  │  Finds damage zone boundary and compactness
  │
  ▼
[impact_analysis.py] HumanitarianImpactAnalyzer.analyze()
  │  Counts individual buildings (connected components)
  │  Assigns damage class per building (mode of pixels)
  │  Estimates: population affected, displaced, casualties
  │  Estimates: economic loss, reconstruction cost
  │  Estimates: emergency shelter needs (tents)
  │  Computes: severity index (0-100)
  │  Generates: response protocol for the disaster type
  │
  ▼
[geo_utils.py + viz_utils.py] Output Generation
  │  save_geotiff() → damage.tif, buildings.tif, confidence.tif, priority.tif
  │  polygonize_damage_map() → damage.geojson (building polygons with labels)
  │  plot_damage_assessment() → 4-panel figure (PNG)
  │  create_leaflet_map() → interactive map (HTML)
  │  JSON report with everything
  │
  ▼
CONSOLE: Formatted humanitarian damage assessment report
```

---

## 2. Pipeline Layer

### `ingestion.py`

**Purpose:** Load satellite images and their ground-truth labels into Python.

#### `XBD_LABEL_MAP` (line 23-30)

```python
XBD_LABEL_MAP = {
    "background":    0,    # not a building
    "no-damage":     1,    # building exists, undamaged
    "minor-damage":  2,    # cosmetic damage (cracked walls, broken windows)
    "major-damage":  3,    # structural damage (partial collapse, holes in roof)
    "destroyed":     4,    # complete collapse / rubble
    "un-classified": 1,    # treat unknown as no-damage (safe default)
}
```

**Why these specific numbers?** The model outputs a number per pixel. 0-4 are sequential so they can be used directly as class indices in PyTorch's CrossEntropyLoss. "un-classified" maps to 1 (no-damage) because it's safer to underestimate damage than overestimate — a false "destroyed" wastes rescue resources.

#### `XBDDataset.__init__()` (line 55-73)

```python
self.image_dir = self.root_dir / split / "images"
self.label_dir = self.root_dir / split / "labels"
```

**Why `Path` instead of string concatenation?** `Path("data") / "raw" / "xbd"` works on Windows (`data\raw\xbd`) and Mac/Linux (`data/raw/xbd`). String concatenation with `/` would break on Windows.

#### `_build_sample_list()` (line 75-105)

```python
pre_images = sorted(self.image_dir.glob("*_pre_*.png"))
```

**Why `glob` instead of `os.listdir`?** Glob pattern `*_pre_*` filters to only pre-disaster images. `listdir` would return ALL files (pre, post, and any other files), requiring manual filtering. `sorted()` ensures deterministic order across runs (important for reproducible train/val splits).

```python
post_stem = stem.replace("_pre_", "_post_")
```

**Why string replacement?** xBD naming convention: `hurricane-harvey_00000001_pre_disaster.png` becomes `hurricane-harvey_00000001_post_disaster.png`. This is more robust than regex for this specific pattern.

```python
"event": stem.split("_pre_")[0]
```

**Why extract the event name?** Used later for disaster type classification. From `hurricane-harvey_00000001_pre_disaster`, we get `hurricane-harvey_00000001` → mapped to disaster type "hurricane" in the disaster classifier.

#### `_rasterize_buildings()` (line 157-200)

```python
transform = Affine(1, 0, 0, 0, 1, 0)
```

**What is an Affine transform?** It maps pixel coordinates to geographic coordinates. `Affine(1, 0, 0, 0, 1, 0)` is the identity — pixel (x, y) maps to coordinate (x, y). We use this because xBD labels are already in pixel coordinates (building polygon vertices are pixel positions within the 1024×1024 image).

```python
wkt_str = feat.get("wkt")
if wkt_str:
    geom = XBDDataset._parse_wkt_to_shapely(wkt_str)
```

**What is WKT?** Well-Known Text — a standard text format for geometry: `POLYGON ((x1 y1, x2 y2, x3 y3, ...))`. xBD uses this instead of GeoJSON. Shapely's `wkt.loads()` parses it into a geometry object that rasterio can burn onto a pixel grid.

```python
mask = rio_rasterize(shapes, out_shape=(height, width), transform=transform, fill=0)
```

**What is rasterization?** Converting vector shapes (polygons defined by coordinates) into a pixel grid (raster). Each pixel inside any building polygon gets value 1, outside gets value 0. This is necessary because neural networks operate on pixel grids, not vector geometry.

#### `_rasterize_damage()` (line 202-250)

```python
for class_label, class_idx in sorted(XBD_LABEL_MAP.items(), key=lambda x: x[1]):
```

**Why sort by class index and rasterize sequentially?** If two building polygons overlap (one labeled "minor", another "destroyed"), we want the MORE SEVERE class to win. By rasterizing from class 0 → 4, each subsequent class overwrites the previous. So "destroyed" (4) always wins over "minor" (2) in overlap zones.

```python
mask = np.where(class_mask > 0, class_mask, mask)
```

**Logic:** "Where the new class's rasterization has a non-zero value, use it. Otherwise, keep what was there before." This is the overwrite mechanism.

#### `GeoTiffScene.read()` (line 279-310)

```python
p2, p98 = np.percentile(valid, [2, 98])
data[i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
```

**Why percentile normalization instead of divide-by-255?** Satellite sensors have wildly different value ranges. Sentinel-2 might have values 0-10000. Maxar might have 0-2048. Some bands have outlier pixels (clouds = very bright, shadows = very dark). Percentile clipping at 2nd and 98th percentile removes these outliers, then stretches the remaining 96% of values to [0, 1]. This works universally across any satellite sensor.

**Why `+ 1e-8`?** Prevents division by zero if p2 == p98 (happens when a band is constant, e.g., all-black or all-white). 1e-8 is a tiny number (0.00000001) that has negligible effect on the result but prevents a crash.

---

### `tiling.py`

**Purpose:** Satellite scenes can be 10,000+ pixels. Models handle 512×512 max. This module splits and reassembles.

#### `RasterTiler.__init__()` (line 62-71)

```python
self.stride = tile_size - overlap  # 512 - 64 = 448
```

**Why overlap?** Without overlap, the model makes poor predictions at tile edges because it has no context beyond the edge. With 64px overlap, every pixel appears in at least one tile with 64px of context around it. The stride (448) is how far apart tile starting positions are.

#### `compute_tile_specs()` (line 77-108)

```python
actual_w = min(self.tile_size, raster_width - col)
pad_r    = self.tile_size - actual_w
```

**Edge handling:** The last tile in a row might not be full-size (e.g., image is 1024px, last tile starts at 896 and only has 128px of real data). `pad_r` records how much zero-padding is needed to make it 512×512. This padding is removed during reassembly.

#### `tile_geotiff()` (line 148-205) — Dask Lazy Loading

```python
@delayed
def _read_tile(spec: TileSpec) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        data = src.read(band_list, window=spec.window)
```

**What is `@delayed`?** Dask's `delayed` decorator says "don't run this function now — just remember that we need to run it later." The tile is NOT read from disk until someone explicitly calls `.compute()`. This means for a 10GB GeoTIFF, we never load the whole thing into RAM. We only load the specific 512×512 tile we need right now.

**Why `rasterio.open()` inside the function?** Each delayed function runs in its own thread/process. File handles can't be shared across processes, so we open a fresh handle for each tile.

```python
dask_array = da.stack(lazy_tiles, axis=0)  # (N, H, W, C)
```

**What is a Dask Array?** It looks like a numpy array but it's lazy — the data isn't in memory. It's a computation graph: "to get tile 0, read from disk at position (0, 0); to get tile 1, read from position (448, 0)..." Only when you index into it or call `.compute()` does it actually read the data.

#### `reassemble()` (line 211-263) — Tile Stitching

```python
weight_patch = self._cosine_window(self.tile_size)
```

**Why cosine window blending?** In overlap zones, two tiles contribute predictions for the same pixel. Without blending, you'd see visible seams where one tile's prediction abruptly changes to another's. The cosine (Hann) window gives each tile's prediction a weight:
- Center of tile: weight ≈ 1.0 (trust this prediction fully)
- Edge of tile: weight ≈ 0.0 (don't trust edge predictions)

The two tiles' weighted predictions are summed and divided by total weight → smooth blending.

```python
canvas[r0:r1, c0:c1]  += tile[:th, :tw] * w
weights[r0:r1, c0:c1] += w
```

**Accumulation logic:** For each tile, multiply its prediction by the weight map and add to the canvas. Also accumulate the weights. At the end, `canvas / weights` gives the weighted average. In non-overlapping regions, one tile contributes with weight 1.0 → result equals the prediction. In overlapping regions, two tiles contribute → result is their blended average.

#### `_cosine_window()` (line 265-270)

```python
win1d = np.hanning(size)        # [0, 0.01, 0.04, ..., 1.0, ..., 0.04, 0.01, 0]
win2d = np.outer(win1d, win1d)  # 2D bell curve: 0 at corners, 1 at center
```

**Why Hann specifically?** It's the standard window function in signal processing for smooth transitions. It starts and ends at 0 (tile edges have zero weight → no contribution) and peaks at 1 (tile center has full weight). The 2D outer product creates a dome-shaped weight map.

---

### `preprocessing.py`

**Purpose:** Make training data more diverse (augmentation) and model-ready (normalization).

#### `build_augmentation_pipeline()` (line 26-79)

**Training augmentations and why each exists:**

```python
A.RandomCrop(height=512, width=512, p=1.0)
```
**Why:** Original images are 1024×1024. Random 512×512 crops mean each epoch, the model sees different parts of each image. 2,520 images × random crops = effectively infinite training variety. `p=1.0` means always apply (not random).

```python
A.HorizontalFlip(p=0.5)
```
**Why:** Buildings look the same mirrored. But without this augmentation, if the training data happens to have more buildings on the left side, the model would learn a left-bias. Flipping teaches orientation invariance. `p=0.5` means 50% chance per image.

```python
A.VerticalFlip(p=0.5)
```
**Why:** Satellite images have no natural "up" (unlike photos where sky is up). A vertically flipped satellite image is equally valid.

```python
A.RandomRotate90(p=0.5)
```
**Why:** Same reasoning — buildings at 0° look identical to buildings at 90°. The model should be rotation-invariant.

```python
A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.4)
```
**Why:** Teaches tolerance for small misalignments between pre and post images (satellite cameras don't photograph from exactly the same position each time). `shift_limit=0.1` means up to 10% shift. `p=0.4` means 40% chance (not too aggressive).

```python
A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.4)
```
**Why:** Satellite images taken at different times of day, different seasons, or different atmospheric conditions have different brightness/contrast. This simulates those variations. `hue=0.05` is small because satellite imagery shouldn't dramatically change color.

```python
A.GaussianBlur(blur_limit=3, p=0.2)
```
**Why:** Simulates atmospheric haze and sensor blur. `blur_limit=3` means a 3×3 blur kernel (mild).

```python
A.GaussNoise(std_range=(0.02, 0.1), p=0.2)
```
**Why:** Simulates sensor noise. Real satellite images have noise especially in shadows.

**Critical: `additional_targets`**

```python
additional_targets={
    "post_image":    "image",
    "building_mask": "mask",
    "damage_mask":   "mask",
}
```

**Why:** When we flip/rotate the pre-image, we MUST apply the exact same transformation to the post-image and both masks. Without this, a building at position (100, 200) in the image would be labeled at (100, 200) in the mask, but after flipping, the building moves to (100, 824-200) while the mask label stays at (100, 200). The labels would be wrong. `additional_targets` ensures all spatial transforms are synchronized.

#### ImageNet Normalization (line 67-69)

```python
A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

**Why these specific numbers?** They are the mean and standard deviation of the entire ImageNet dataset (1.2 million images). Our ResNet34 encoder was pretrained on ImageNet with these exact normalization values. If we used different values, the pretrained weights would receive inputs in a range they've never seen → the pretrained features become useless. Matching the normalization is essential for transfer learning.

```python
ToTensorV2()
```

**Why:** Converts numpy array (H, W, C) with values 0-255 to PyTorch tensor (C, H, W) with values already normalized. PyTorch requires channels-first format (C, H, W) unlike numpy/OpenCV which use channels-last (H, W, C).

---

## 3. Model Layer

### `segmentation.py`

**Purpose:** Given a satellite image, mark every pixel that is part of a building.

#### The U-Net Architecture

```
Why U-Net and not just a simple CNN?

A simple CNN (like ResNet alone) would output ONE answer for the whole image:
  "This image contains buildings" — not useful, we need pixel-level precision

U-Net outputs a GRID of answers, same size as the input:
  Every pixel gets its own prediction: building (1) or not-building (0)

How U-Net achieves this:

ENCODER (left side of the U):
  512×512 → [conv, conv, pool] → 256×256    Halves spatial size
  256×256 → [conv, conv, pool] → 128×128    Each level learns more abstract features
  128×128 → [conv, conv, pool] → 64×64      Level 1: edges, Level 2: textures
  64×64   → [conv, conv, pool] → 32×32      Level 3: shapes, Level 4: objects
  32×32   → [conv, conv] → 32×32            BOTTLENECK: most compressed representation

DECODER (right side of the U):
  32×32   → [upsample]        → 64×64       Doubles spatial size
  64×64   → [concat skip + conv] → 128×128  Skip connection carries fine details
  128×128 → [concat skip + conv] → 256×256  from encoder to decoder
  256×256 → [concat skip + conv] → 512×512  Without skips: decoder can't recover positions

SKIP CONNECTIONS (the horizontal arrows in the U):
  The encoder at 256×256 knows "there's a texture here" but by the time
  it reaches the 32×32 bottleneck, the precise position is lost.
  Skip connections carry the position information directly to the decoder.
  
  This is why U-Net works so well for segmentation — it has BOTH
  high-level understanding (from bottleneck) AND pixel-precise positions (from skips).
```

#### Why ResNet34 Encoder (line 36-44)

```python
self.model = smp.Unet(
    encoder_name="resnet34",        # WHY: good balance of accuracy vs speed
    encoder_weights="imagenet",      # WHY: pretrained, not random
    in_channels=3,                   # WHY: RGB satellite images
    classes=1,                       # WHY: binary (building / not building)
    activation=None,                 # WHY: raw logits, sigmoid applied in loss
)
```

**Why ResNet34 specifically?**
- ResNet18: too small, underfits satellite complexity
- ResNet34: good balance — 24M parameters, fast on 8GB RAM
- ResNet50: better accuracy but 2× slower, might not fit batch-size 2 in 8GB
- EfficientNet-B4: best accuracy but very slow on CPU/MPS

**Why `activation=None`?** The model outputs raw logits (any real number, can be negative). Sigmoid is applied inside the loss function (BCEWithLogitsLoss), which is numerically more stable than applying sigmoid first and then computing log. Sigmoid(logit) = probability.

#### Loss Functions

##### DiceLoss (line 74-89)

```python
intersection = (probs_f * targets_f).sum()
return 1.0 - (2.0 * intersection + smooth) / (probs_f.sum() + targets_f.sum() + smooth)
```

**Step-by-step what this computes:**

```
probs_f  = [0.9, 0.8, 0.1, 0.2, 0.95, 0.05]  ← model predictions (flattened)
targets_f = [1,   1,   0,   0,   1,    0   ]  ← ground truth

probs_f × targets_f = [0.9, 0.8, 0, 0, 0.95, 0]  ← only counts where truth=1
intersection = 0.9 + 0.8 + 0.95 = 2.65

probs_f.sum() = 0.9+0.8+0.1+0.2+0.95+0.05 = 3.0
targets_f.sum() = 1+1+0+0+1+0 = 3.0

Dice = 1 - (2 × 2.65 + 1) / (3.0 + 3.0 + 1) = 1 - 6.3/7.0 = 0.10

Loss = 0.10 (low = good overlap)
```

**Why `smooth=1.0`?** Without it, if there are zero buildings in an image (targets_f all zeros), both numerator and denominator are 0 → 0/0 → NaN → training crashes. Adding 1.0 to both prevents this edge case.

##### BceDiceLoss (line 92-110)

```python
return self.bce_weight * self.bce(logits, targets_f) + self.dice_weight * self.dice(logits, targets_f)
```

**Why combine two losses?**
- **BCE alone:** Treats every pixel independently. Good gradients, stable training. But doesn't care about overall shape — can get a high score by being right on easy background pixels while being wrong on building edges.
- **Dice alone:** Cares about overall overlap between prediction and truth. Great for handling class imbalance (5% buildings vs 95% background). But gradients can be noisy, especially early in training when predictions are random.
- **Combined (0.5 + 0.5):** BCE provides stable gradients throughout training. Dice ensures the model focuses on actually finding buildings, not just getting background right.

#### Differential Learning Rates (line 117-134)

```python
{"params": encoder_params, "lr": lr * 0.1},   # encoder: slow learning
{"params": decoder_params, "lr": lr},           # decoder: normal learning
```

**Why?** The encoder (ResNet34) comes pretrained from ImageNet. Its weights already represent useful visual features (edges, textures, shapes). If we update them too fast with a high learning rate, we destroy these features. By using 10× lower LR, we fine-tune them gently.

The decoder is randomly initialized — it knows nothing. It needs a higher LR to learn quickly from scratch.

#### Learning Rate Scheduler (line 137-146)

```python
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7)
```

**Linear warmup (epochs 1-5):** LR starts at 1% of target and linearly increases. This prevents "catastrophic forgetting" — if you immediately hit pretrained weights with a full learning rate, the first few batches of noisy gradients can permanently damage the pretrained features.

**Cosine annealing (epochs 6-30):** LR follows a cosine curve from high to near-zero. Unlike fixed LR, this naturally explores the loss landscape broadly early on (high LR = big steps) and then fine-tunes precisely later (low LR = small steps).

```
LR schedule visualization:
  
  0.0001 |     /\
         |    /  \
         |   /    \_
  0.00001|  /       \__
         | /           \_____
  0      |/___________________\
         0  5  10  15  20  25  30  (epochs)
           ↑warmup    cosine→
```

---

### `siamese_unet.py`

**Purpose:** Classify building damage by comparing pre and post images. This is the V2 model (more accurate than V1).

#### Why "Siamese"?

```
The word "Siamese" comes from Siamese twins — two bodies sharing one brain.

Pre-image  ──► Encoder (brain A)  ──► features_pre
Post-image ──► Encoder (brain A)  ──► features_post
                  ↑ SAME weights

Since BOTH images go through IDENTICAL processing, the features are
directly comparable. The difference |features_pre - features_post|
is a pure CHANGE signal.

Old approach (6-channel concatenation):
  [pre + post] ──► Single encoder ──► features
  The encoder has to LEARN that channels 0-2 are "before" and 3-5 are "after"
  It can't use ImageNet weights (designed for 3 channels)
  
Siamese approach:
  Each image goes through a normal 3-channel encoder with ImageNet weights
  The comparison happens EXPLICITLY via feature subtraction
  Much easier to learn, much more accurate
```

#### FeatureDifferenceModule (line 38-73)

```python
diff = torch.abs(f_pre - f_post)
combined = torch.cat([f_pre, f_post, diff], dim=1)  # 3×C channels
fused.append(proj(combined))
```

**What happens at each encoder level:**

```
Encoder level 3 (128 channels, 128×128 spatial):

f_pre  = [what features exist in the pre-disaster image]
         128 channels × 128×128 pixels

f_post = [what features exist in the post-disaster image]
         128 channels × 128×128 pixels

diff   = |f_pre - f_post|
         [where features CHANGED between pre and post]
         A building that collapsed: f_pre had "roof texture" features
         f_post has "rubble texture" features → large difference

combined = [f_pre, f_post, diff]  → 384 channels
         Contains: what was, what is, and what changed

proj(combined) = 1×1 convolution: 384 → 128 channels
         Learned projection that picks the most useful information
         from all three sources
```

**Why absolute value `torch.abs()`?** Without abs, positive and negative differences would cancel out when summed. A building that got brighter (positive) and one that got darker (negative) would average to "no change." Absolute value ensures any change is detected regardless of direction.

#### `predict_with_confidence()` (line 148-172)

```python
probs      = self.predict_proba(pre, post)
confidence = probs.max(dim=1).values   # probability of predicted class
classes    = probs.argmax(dim=1)       # which class has highest probability
```

**Why this matters:** A pixel with probabilities [0.02, 0.03, 0.05, 0.05, 0.85] is predicted "destroyed" with 85% confidence — model is sure. A pixel with [0.15, 0.20, 0.25, 0.25, 0.15] is predicted "major-damage" with only 25% confidence — model is guessing. The confidence map tells rescue teams which predictions to act on.

---

### `attention.py`

**Purpose:** CBAM attention makes the model focus on important features and locations.

#### ChannelAttention (line 27-71)

**Analogy:** You're looking at a satellite image. Your brain has many "detectors" — edge detector, color detector, texture detector, shadow detector. Channel attention is your brain deciding "for this particular image, the edge detector and shadow detector are most important, ignore the color detector."

```python
avg_pool = x.mean(dim=(2, 3))    # (B, C) — average activation per channel
max_pool = x.amax(dim=(2, 3))    # (B, C) — maximum activation per channel
```

**Why both avg AND max?**
- Avg pool: "On average, how active is this channel across the whole image?"
  → Captures which channels are generally relevant
- Max pool: "What's the strongest response this channel had anywhere?"
  → Captures which channels detected SOMETHING important, even if only in one spot

```python
mid = max(channels // reduction, 8)
```

**Why reduce channels?** A 64-channel MLP would have 64×64 = 4,096 parameters per layer. With reduction=16, we have 64→4→64 = 64×4 + 4×64 = 512 parameters. This bottleneck forces the network to learn a compressed representation of channel importance, preventing overfitting. `max(..., 8)` ensures we never go below 8 channels (too small = can't represent enough information).

```python
weights = torch.sigmoid(avg_out + max_out)   # (B, C)
return x * weights                            # element-wise multiplication
```

**Sigmoid output [0, 1]:** A weight of 0.95 means "this channel is very important, keep it." A weight of 0.05 means "this channel is not useful, suppress it." The multiplication scales each channel by its importance.

#### SpatialAttention (line 74-103)

**Analogy:** Now you know WHICH detectors to use. Spatial attention decides WHERE to look. "Focus on the building edges in the center, ignore the empty field on the left."

```python
avg_out = x.mean(dim=1, keepdim=True)     # (B, 1, H, W)
max_out = x.amax(dim=1, keepdim=True)     # (B, 1, H, W)
combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
weights  = torch.sigmoid(self.conv(combined))     # (B, 1, H, W)
```

**Step by step:**
1. Average across all channels at each pixel → "how generally active is this location?"
2. Max across all channels at each pixel → "is ANY feature strongly activated here?"
3. Concatenate both → 2-channel feature map
4. 7×7 convolution → learns spatial patterns (7×7 = large enough to see building shapes)
5. Sigmoid → weight per pixel (0 = ignore this area, 1 = focus here)

**Why 7×7 kernel?** Building edges in satellite imagery at 0.5m resolution span multiple pixels. A 3×3 kernel only sees 1.5m context — too small to recognize a building edge. 7×7 sees 3.5m — enough to identify the transition between roof and ground.

---

### `tta.py`

**Purpose:** Free accuracy boost at inference by running the model on multiple orientations and averaging.

#### TTA_TRANSFORMS (line 31-41)

```python
{"name": "original",  "fn": lambda x: x,                    "inv": lambda x: x},
{"name": "hflip",     "fn": lambda x: torch.flip(x, [-1]),  "inv": lambda x: torch.flip(x, [-1])},
```

**Why does `inv` exist?** When we flip the input, the model's output is also flipped. Before averaging, we need to UN-flip it back to the original orientation. For horizontal flip, the inverse IS another horizontal flip (flip twice = original). For rotation 90°, the inverse is rotation 270°.

```python
accumulated += probs_original_space
return accumulated / len(self.transforms)
```

**Why average?** Each orientation gives a slightly different prediction (the model isn't perfectly rotation-invariant despite training augmentations). Averaging cancels out random errors. If 7 out of 8 orientations say "building" and 1 says "not building," the average probability is ~0.875 — correctly leaning toward building. A single wrong prediction gets diluted.

#### `predict_with_uncertainty()` (line 143-171)

```python
stacked = torch.stack(all_preds, dim=0)   # (8, B, 1, H, W)
mean    = stacked.mean(dim=0)              # average prediction
std     = stacked.std(dim=0)               # disagreement between orientations
```

**Why std = uncertainty?** If all 8 orientations predict probability 0.9 for a pixel, std ≈ 0 → model is consistent → high confidence. If orientations predict [0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.85, 0.15], std ≈ 0.3 → model is inconsistent → low confidence, this pixel is uncertain.

---

### `disaster_classifier.py`

**Purpose:** Predict what TYPE of disaster caused the observed damage pattern.

#### DamagePatternFeatureExtractor — Statistical Features (line 119-184)

**These 21 hand-crafted features encode domain knowledge about how different disasters leave distinct spatial signatures:**

##### Feature group 1: Class ratios (5 numbers)

```python
class_ratios = probs.mean(dim=(1, 2))  # (5,) — average probability per class
```

- Earthquakes: high "destroyed" ratio, sharp boundary with "no-damage"
- Floods: gradual progression from "minor" to "major" to "destroyed"
- Wildfires: extreme "destroyed" in burn area, "no-damage" outside

##### Feature group 2: Spatial concentration (4 numbers)

```python
damage_sum = probs[2:].sum(dim=0)  # sum of minor + major + destroyed
spatial_var = damage_sum.var()      # variance of damage intensity
```

- High variance → damage is concentrated in one area (earthquake, tornado)
- Low variance → damage is spread evenly (hurricane, wide-area flood)

##### Feature group 3: Directional gradient (8 numbers)

```python
q_tl = damage_sum[:h_mid, :w_mid].mean()   # top-left quadrant
q_tr = damage_sum[:h_mid, w_mid:].mean()    # top-right quadrant
grad_h = (q_tr + q_br) - (q_tl + q_bl)      # horizontal gradient
```

- Strong horizontal gradient → damage increases east-to-west (flood flowing downstream, hurricane wind direction)
- Symmetric (low gradient all directions) → earthquake (radial from centre)
- Strong in one direction → tsunami (coastal inundation gradient)

##### Feature group 4: Connectivity (4 numbers)

```python
ratio_destroyed_to_minor = class_ratios[4] / class_ratios[2]
```

- High ratio → sharp transitions (earthquake: intact building next to rubble)
- Low ratio → gradual transitions (flood: water level gradually increased)

**Why hand-craft these instead of letting a CNN learn them?** CNNs CAN learn these patterns, but they need thousands of training examples per disaster type. We only have ~200 samples per type in xBD. Hand-crafted features encode decades of disaster science in a few lines of code — the model gets this knowledge for free without needing to learn it from limited data.

#### `get_disaster_response_protocol()` (line 265-342)

This function maps disaster type to actionable response guidance. This is pure domain knowledge — no ML, just encoding what FEMA, INSARAG, and UNHCR recommend for each disaster type.

```python
"earthquake": {
    "time_critical_window": "72 hours (golden period for trapped survivors)",
    "equipment_needed": ["heavy cranes", "concrete saws", "search dogs", ...],
    "survivor_location": "trapped under rubble — urban SAR teams needed",
}
```

**Why this matters:** The ML model output ("earthquake, severity 87") is useful for researchers. The response protocol ("you have 72 hours, bring cranes and dogs, survivors are under rubble") is useful for the person who actually needs to save lives. This bridges ML → action.

---

### `damage_classifier.py`

**Purpose:** Connects all models into one pipeline.

#### `run_on_tiles()` — The Core Logic (line 82-125)

```python
# Building segmentation
seg_logits = self.seg_model(pre_t)
seg_probs  = torch.sigmoid(seg_logits).squeeze(1)

# Damage classification  
if self.use_siamese:
    dmg_logits = self.dmg_model(pre_t, post_t)
else:
    pair_t = torch.cat([pre_t, post_t], dim=1)
    dmg_logits = self.dmg_model(pair_t)

# FUSION: mask non-buildings
building_binary = (seg_probs > self.seg_threshold).long()
dmg_masked = dmg_preds * building_binary
```

**Why fusion?** Without masking, the damage model might predict "destroyed" for a muddy field that changed color between pre and post. The building mask ensures damage labels ONLY appear on actual buildings. `dmg_preds * building_binary`: wherever building_binary is 0 (not a building), the damage prediction becomes 0 (background) regardless of what the damage model predicted.

---

## 4. Analysis Layer

### `spatial_analysis.py`

#### `find_damage_epicentre()` — Centre of Mass

```python
weight_map[(damage_map == 4) & building_mask] = 3.0  # destroyed
weight_map[(damage_map == 3) & building_mask] = 1.0  # major
cy, cx = ndimage.center_of_mass(weight_map)
```

**Why weight destroyed 3× more than major?** The epicentre is the point of maximum destruction — where the earthquake fault ruptured, where the tornado touched down, where the bomb detonated. "Major damage" extends further from the epicentre than "destroyed." By weighting destroyed pixels more, the centre of mass is pulled toward the true epicentre rather than the wider zone of moderate damage.

#### `compute_radial_profile()` — Distance Decay

```python
coeffs = np.polyfit(dists[valid], sevs[valid], 1)
decay_rate = float(coeffs[0])
```

**What this does:** Fits a straight line (y = mx + b) through the data points (distance, severity). The slope `m` is the decay rate:
- Steep negative slope (m = -0.05) → damage drops off quickly with distance → point-source event (earthquake epicentre, explosion)
- Flat slope (m ≈ 0) → damage is uniform regardless of distance → wide-area event (hurricane, flooding across entire region)

#### `find_damage_boundary()` — Compactness

```python
compactness = (4 * np.pi * area_pixels) / (perimeter_pixels ** 2)
```

**What this formula means:** A perfect circle has the maximum area-to-perimeter ratio: compactness = 1.0. Any other shape has compactness < 1.0. The more irregular/elongated the shape, the lower the compactness.
- Earthquake damage zone: roughly circular (compactness 0.6-0.8)
- Tornado track: long narrow strip (compactness 0.1-0.2)
- Flood zone: follows river valleys, irregular (compactness 0.2-0.4)

This single number instantly tells you the disaster geometry.

---

### `impact_analysis.py`

#### Building Count via Connected Components

```python
from scipy.ndimage import label as scipy_label
labeled, _ = scipy_label(building_mask.astype(np.uint8))
```

**What `scipy_label` does:** Scans the binary building mask and assigns a unique integer to each group of connected pixels. Two building-pixels are "connected" if they touch (including diagonally). The result is a labeled array where building 1 has all pixels = 1, building 2 has all pixels = 2, etc. `labeled.max()` = total number of buildings found.

#### Per-Building Damage Assignment

```python
significant = classes[counts > total * 0.2]
building_damage[bld_id] = int(significant.max())
```

**Why 20% threshold and take max?** Consider a building with 100 pixels: 85 pixels classified as "no-damage" and 15 pixels classified as "destroyed" (at the edge, due to prediction noise). Without the threshold, the building would be labeled "destroyed" based on just 15% of noisy edge pixels. With the 20% threshold, the 15% destroyed-class doesn't qualify, and the building is correctly labeled "no-damage."

But if 25 pixels are "destroyed" (>20%), it qualifies and `.max()` selects it — because we'd rather over-report than under-report for buildings where destruction IS present in significant portions.

#### Economic Loss Formula

```python
bld_value = bld_area * self.cost_m2          # building replacement value
loss += bld_value * DAMAGE_LOSS_RATIO[class_name]  # fraction of value lost
```

**Example:**
- Building: 200 m², middle-income region ($600/m²) → value = $120,000
- Damage class: "major-damage" → loss ratio = 0.50
- Loss = $120,000 × 0.50 = $60,000
- Reconstruction cost = $60,000 × 1.3 = $78,000 (30% overhead for labor, permits, logistics)

These constants come from FEMA HAZUS methodology (US) and World Bank disaster assessment guidelines — the standard references used globally.

#### Severity Index (0-100)

```python
severity = destruction_ratio * 40 + displacement_ratio * 30 + scale_factor * 30
```

**Why these weights?**
- 40% destruction ratio: a disaster that destroys 80% of buildings is more severe than one that destroys 10%, regardless of city size
- 30% displacement ratio: high displacement means people can't stay in their homes → immediate humanitarian need
- 30% scale: `log10(n_buildings)/4` — a disaster affecting 10,000 buildings is more severe than one affecting 10, even at the same destruction ratio. Log scale because the difference between 100 and 200 buildings matters more than between 10,000 and 10,100

---

### `ml_analysis.py`

#### XGBoost Disaster Classifier

```python
self.model = GradientBoostingClassifier(
    n_estimators=100,     # build 100 small decision trees
    max_depth=6,          # each tree has max 6 levels (prevents overfitting)
    learning_rate=0.1,    # each tree's contribution is shrunk by 0.1
    subsample=0.8,        # each tree sees 80% of data (randomness → robustness)
)
```

**How gradient boosting works:**
```
Tree 1: Makes rough predictions. Gets many wrong.
Tree 2: Focuses on the ERRORS of Tree 1. Corrects some.
Tree 3: Focuses on the remaining errors. Corrects more.
...
Tree 100: Focuses on the hardest remaining cases.

Final prediction = Tree1 + 0.1×Tree2 + 0.1×Tree3 + ... + 0.1×Tree100
```

Each tree is weak (max 6 levels), but 100 trees combined are strong. The learning_rate=0.1 shrinks each tree's contribution to prevent any single tree from dominating (regularization).

**Why XGBoost for disaster type but U-Net for segmentation?** XGBoost excels on tabular/structured features (our 21 spatial statistics). U-Net excels on spatial grid data (images). Using the right tool for the right data type is a fundamental ML principle.

#### DBSCAN Hotspot Detection

```python
db = DBSCAN(eps=50.0, min_samples=5, metric="euclidean")
labels = db.fit_predict(points)
```

**Parameters explained:**
- `eps=50.0`: Two damage points within 50 pixels of each other are "neighbors." At 0.5m GSD, this is 25 meters — roughly the width of a city block. Damage points on the same block belong to the same cluster.
- `min_samples=5`: A cluster needs at least 5 points. Fewer = noise (isolated false positives from the model).

```python
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
```

**Why subtract 1?** DBSCAN assigns label -1 to noise points. `set(labels)` includes -1, but we don't want to count noise as a cluster. So we subtract 1 if -1 is present.

**Why subsample to 5,000 points?**
```python
max_points = 5000
if len(ys) > max_points:
    idx = np.random.choice(len(ys), max_points, replace=False)
```

DBSCAN computes pairwise distances between all points: O(n²). For 100,000 points, that's 10 billion distances → slow and memory-hungry. Subsampling to 5,000 points gives 25 million distances — fast enough while still finding the same clusters.

#### PCA (Principal Component Analysis)

```python
flat = image.reshape(C, -1).T    # (N_pixels, C)
pca = PCA(n_components=3)
pca.fit(flat)
```

**What PCA mathematically does:**
1. Center the data: subtract mean of each band
2. Compute covariance matrix: which bands vary together?
3. Find eigenvectors: the directions of maximum variance
4. Project data onto top-k eigenvectors

**Example with Sentinel-2 (13 bands):**
```
Original: 13 numbers per pixel (Red, Green, Blue, NIR, SWIR1, SWIR2, ...)
PCA output: 3 numbers per pixel

PC1 (50% variance): weighted sum of all bands ≈ overall brightness
PC2 (25% variance): NIR minus Red ≈ vegetation index
PC3 (10% variance): SWIR minus NIR ≈ moisture content

Total: 85% of information in 3 numbers instead of 13
The other 10 components are mostly noise or redundant correlations
```

#### K-Means Land Cover

```python
self.model = MiniBatchKMeans(n_clusters=5, batch_size=1024)
cluster_labels = self.model.fit_predict(flat)
```

**Why MiniBatchKMeans instead of regular KMeans?** A 1024×1024×3 image has 1,048,576 pixels. Regular KMeans computes distances from all points to all centroids every iteration — slow. MiniBatchKMeans randomly samples 1024 pixels per iteration — much faster, nearly identical results.

**Why 5 clusters?** Satellite images typically contain: water, vegetation, bare soil, urban/buildings, shadows. 5 clusters naturally separate these. More clusters would split "urban" into "bright urban" and "dark urban" which isn't useful; fewer would merge "water" and "shadow" (both dark).

#### MC Dropout — Bayesian Uncertainty

```python
def _enable_dropout(self):
    for module in self.model.modules():
        if isinstance(module, nn.Dropout):
            module.train()   # re-enable dropout
```

**Why is this surprising?** Normally, `model.eval()` turns OFF all dropout layers (they just pass data through unchanged). We selectively turn dropout back ON while keeping everything else in eval mode. Now each forward pass randomly drops different neurons → different predictions each time.

```python
for _ in range(self.n_samples):
    output = self.model(x)    # each run drops different neurons → different result
    predictions.append(probs)

stacked = torch.stack(predictions, dim=0)
mean_pred = stacked.mean(dim=0)       # best estimate
uncertainty = stacked.std(dim=0)       # how much the runs disagree
```

**The mathematical insight (Gal & Ghahramani, 2016):** Running a model with dropout N times and averaging is mathematically equivalent to sampling from a Bayesian posterior distribution. The mean is the point estimate, and the std is the posterior uncertainty. This turns any dropout-containing network into a Bayesian model — for free.

---

## 5. Utilities Layer

### `geo_utils.py`

#### `polygonize_damage_map()` — Raster to Vector

```python
for geom_dict, value in rasterio.features.shapes(class_mask, mask=class_mask, transform=transform):
    geom = shape(geom_dict)
```

**What `rasterio.features.shapes()` does:** Traces the boundaries of contiguous pixel regions and converts them to polygon geometry. It's the reverse of rasterization — going from pixel grid back to vector shapes. The output is GeoJSON-compatible geometry that can be saved as a .geojson file and opened in any GIS software (QGIS, ArcGIS, Google Earth).

**Why `min_area_pixels=10`?** Tiny 1-2 pixel polygons are usually noise from the model (a single misclassified pixel). Filtering them out cleans the output without losing real buildings (which are always >10 pixels at 0.5m GSD).

### `metrics.py`

#### xView2 Competition Score

```python
xview2_score = _harmonic_mean(per_class_f1)
```

**Why harmonic mean instead of arithmetic mean?** Harmonic mean heavily penalizes any class being low.

```
Arithmetic mean of [0.9, 0.9, 0.1, 0.9] = 0.70 (looks okay!)
Harmonic mean of   [0.9, 0.9, 0.1, 0.9] = 0.30 (reveals the weakness)
```

A model that ignores "minor-damage" (F1=0.1) but does well on everything else would get 0.70 arithmetic but only 0.30 harmonic. The harmonic mean forces the model to be good at ALL classes, not just the easy ones.

---

## 7. Production Layer

### `api.py`

#### `@app.on_event("startup")` — Model Loading

```python
assessor = GeoSightAssessor(
    seg_checkpoint=seg_path,
    dmg_checkpoint=dmg_path,
)
```

**Why load at startup, not per-request?** Loading a model takes ~3 seconds (reading 95MB file, transferring to GPU). If we loaded per-request, every API call would have a 3-second overhead. Loading once at startup means the first request is slow but all subsequent requests are fast (just inference: ~1 second).

#### `async def assess_damage()` — Async Endpoint

```python
with tempfile.TemporaryDirectory() as tmpdir:
    with open(pre_path, "wb") as f:
        f.write(await pre_image.read())
```

**Why `async` and `await`?** FastAPI is asynchronous — it can handle multiple requests simultaneously. While one request is waiting for file I/O (slow), the server can process another request. Without async, the server would block on every file read, handling only one request at a time.

**Why temp directory?** Uploaded files need to be saved to disk (our pipeline reads from file paths, not bytes). Temp directory is automatically deleted when the `with` block exits — no file cleanup needed.

### `fetch_satellite.py`

#### OAuth Authentication

```python
data = urllib.parse.urlencode({
    "grant_type": "password",
    "username": username,
    "password": password,
    "client_id": "cdse-public",
})
```

**What is OAuth2?** Copernicus doesn't accept username/password directly. You send credentials to their auth server, get back a temporary token (valid ~10 minutes), then use that token for all subsequent requests. This is more secure — the token can expire and be revoked.

#### Satellite Search

```python
f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({west} {south},...})')"
```

**What is SRID=4326?** It's the coordinate system identifier for WGS84 (standard GPS coordinates — latitude/longitude). Every coordinate system has an SRID number. 4326 is the most common, used by GPS, Google Maps, and most satellite data.

---

## 9. Every Algorithm Used

| Algorithm | Category | File | What it does |
|-----------|----------|------|-------------|
| **U-Net** | Deep Learning | segmentation.py, siamese_unet.py | Encoder-decoder for pixel-level prediction |
| **ResNet34** | Deep Learning | All models (encoder) | 34-layer residual network for feature extraction |
| **Siamese Network** | Deep Learning | siamese_unet.py | Shared-weight twin encoders for image comparison |
| **Transfer Learning** | Deep Learning | All models | Reuse ImageNet pretrained weights |
| **CBAM Attention** | Deep Learning | attention.py | Channel + spatial attention for selective focus |
| **BCE Loss** | Loss Function | segmentation.py | Per-pixel binary classification loss |
| **Dice Loss** | Loss Function | segmentation.py | Overlap-based loss for class imbalance |
| **Focal Loss** | Loss Function | change_detection.py | Down-weights easy pixels, focuses on hard ones |
| **Gradient Boosting (XGBoost)** | Classical ML | ml_analysis.py | Ensemble of decision trees for tabular classification |
| **DBSCAN** | Unsupervised ML | ml_analysis.py | Density-based spatial clustering without pre-specifying k |
| **PCA** | Dimensionality Reduction | ml_analysis.py | Linear projection to max-variance components |
| **K-Means** | Unsupervised ML | ml_analysis.py | Centroid-based pixel clustering for land cover |
| **MC Dropout** | Bayesian ML | ml_analysis.py | Approximate Bayesian inference via dropout sampling |
| **Test Time Augmentation** | Ensemble | tta.py | Geometric ensemble of flip/rotation predictions |
| **Cosine Annealing** | Optimization | segmentation.py | Learning rate schedule: high→low on cosine curve |
| **AdamW** | Optimization | segmentation.py | Adam optimizer with decoupled weight decay |
| **Connected Components** | Image Processing | impact_analysis.py | Label contiguous pixel regions |
| **Centre of Mass** | Statistics | spatial_analysis.py | Weighted centroid of damage distribution |
| **Convex Hull** | Computational Geometry | spatial_analysis.py | Smallest convex polygon enclosing points |
| **Cosine Window** | Signal Processing | tiling.py | Smooth blending of overlapping tiles |
| **Dask Lazy Evaluation** | Distributed Computing | tiling.py | Out-of-core processing for large files |
| **Percentile Normalization** | Statistics | ingestion.py | Robust normalization using 2nd/98th percentile |
| **Rasterization** | GIS | ingestion.py, geo_utils.py | Vector polygon → pixel grid conversion |
| **Polygonization** | GIS | geo_utils.py | Pixel grid → vector polygon conversion |
| **Morphological Operations** | Image Processing | spatial_analysis.py | Dilation/erosion for boundary detection |
| **Linear Regression** | Statistics | spatial_analysis.py | Fit decay rate of damage vs distance |
| **Harmonic Mean** | Statistics | metrics.py | Penalizes low outliers (xView2 scoring) |

---

## 10. Every Skill Demonstrated

| Skill | Sub-skills | Where |
|-------|-----------|-------|
| **Deep Learning** | U-Net, ResNet, Siamese networks, attention mechanisms, transfer learning, loss function design, learning rate scheduling, gradient clipping, batch normalization | `src/models/` |
| **Classical ML** | Gradient boosting, decision trees, feature importance, ensemble methods | `ml_analysis.py` |
| **Unsupervised ML** | DBSCAN clustering, K-Means, PCA dimensionality reduction | `ml_analysis.py` |
| **Bayesian ML** | MC Dropout uncertainty estimation, posterior approximation, calibration | `ml_analysis.py` |
| **Computer Vision** | Semantic segmentation, multi-class classification, change detection, data augmentation, test time augmentation, attention mechanisms | `src/models/`, `preprocessing.py` |
| **Geospatial** | rasterio, GDAL, geopandas, GeoTIFF I/O, CRS systems, rasterization, polygonization, reprojection, coordinate transforms, satellite imagery | `geo_utils.py`, `ingestion.py` |
| **Remote Sensing** | Multi-spectral image processing, spectral indices, PCA on bands, Sentinel-2 data access, damage assessment methodology | `ml_analysis.py`, `fetch_satellite.py` |
| **Spatial Analysis** | Epicentre detection, directional gradients, radial profiles, compactness, connected components, convex hull, morphological operations | `spatial_analysis.py` |
| **Signal Processing** | Cosine (Hann) window for tile blending | `tiling.py` |
| **Distributed Computing** | Dask lazy evaluation, out-of-core processing, parallel tile I/O | `tiling.py` |
| **API Development** | FastAPI, REST endpoints, async/await, file upload, CORS, auto-docs | `api.py` |
| **Domain Knowledge** | FEMA HAZUS methodology, UNHCR Sphere standards, World Bank damage assessment, disaster response protocols | `impact_analysis.py`, `disaster_classifier.py` |
| **Software Engineering** | OOP, type hints, dataclasses, modular architecture, config management, CLI (argparse), unit tests, git version control | Throughout |
| **Python** | PyTorch Dataset/DataLoader, numpy array operations, context managers, pathlib, generators, decorators | Throughout |
| **Visualization** | matplotlib, Folium/Leaflet interactive maps, training dashboards, color mapping | `viz_utils.py` |

---

## 11. Nothing Is Secret

**Every algorithm is published research. Every library is open source.**

| Component | Source | Year | Access |
|-----------|--------|------|--------|
| U-Net | Ronneberger et al., "U-Net" paper | 2015 | Open |
| ResNet | He et al., Microsoft Research | 2015 | Open |
| Siamese networks | Bromley et al. (concept); xView2 winners (application) | 1993/2019 | Open |
| CBAM | Woo et al., ECCV | 2018 | Open |
| Focal Loss | Lin et al., Facebook AI Research | 2017 | Open |
| Dice Loss | Milletari et al. | 2016 | Open |
| MC Dropout | Gal & Ghahramani | 2016 | Open |
| XGBoost/GBM | Friedman (theory); Chen & Guestrin (implementation) | 2001/2016 | Open |
| DBSCAN | Ester et al. | 1996 | Open |
| PCA | Pearson | 1901 | Public domain |
| K-Means | Lloyd | 1957 | Public domain |
| HAZUS methodology | FEMA | Ongoing | Public domain (US gov) |
| Sphere standards | UNHCR | Ongoing | Public domain |

**Your competitive advantage is NOT secret algorithms. It is:**
1. The trained .pth weights (expensive to produce: data + 24h compute)
2. The end-to-end integration (nobody else connects all these pieces)
3. Domain expertise (knowing what to compute and why)
4. Speed of deployment (your pipeline runs in seconds)

---

## 12. Where To Make Changes

| You want to... | Change this file | What to modify |
|----------------|------------------|----------------|
| Use a bigger encoder (ResNet50, EfficientNet) | `segmentation.py` line 37, `siamese_unet.py` line 99 | Change `encoder_name="resnet50"` |
| Change tile size | `config/config.yaml` or CLI flag `--tile-size` | `data.tile_size` |
| Add SAR (radar) support | `preprocessing.py` line 29 | Set `use_sar=True`, adjust channels |
| Add more damage classes | `change_detection.py` line 25 | Add to `DAMAGE_CLASSES` dict |
| Add more disaster types | `disaster_classifier.py` line 24 | Add to `DISASTER_TYPES` dict |
| Change population estimates | `impact_analysis.py` line 38 | Modify `OCCUPANCY_RATES` |
| Change economic assumptions | `impact_analysis.py` line 45 | Modify `BUILDING_COST_PER_M2` |
| Change DBSCAN sensitivity | `ml_analysis.py` line 176 | Adjust `eps` and `min_samples` |
| Add more TTA augmentations | `tta.py` line 31 | Add entries to `TTA_TRANSFORMS` |
| Change API port | `api.py` line 202 | Change `port=8000` |
| Change output formats | `assessor.py` line 127-165 | Add/remove output generation code |
| Add new spatial metric | `spatial_analysis.py` | Add method to `SpatialDamageAnalyzer` |
| Add new analysis module | `src/utils/` | Create new file, import in `__init__.py` |
| Change training hyperparameters | CLI flags or `config/config.yaml` | `--lr`, `--batch-size`, `--epochs` |
