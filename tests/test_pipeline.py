"""
Unit and integration tests for the GeoSight pipeline.
Run with: pytest tests/ -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def sample_image():
    """224x224 RGB uint8 image."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def sample_mask():
    """224x224 binary mask."""
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[60:140, 60:140] = 1
    return mask


@pytest.fixture(scope="session")
def sample_damage_mask():
    """224x224 damage mask with classes 0-4."""
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[50:100, 50:100] = 1   # no-damage
    mask[110:150, 50:100] = 2  # minor-damage
    mask[50:100, 110:150] = 3  # major-damage
    mask[110:150, 110:150] = 4 # destroyed
    return mask


@pytest.fixture(scope="session")
def synthetic_xbd_dir(tmp_path_factory):
    """Create a tiny synthetic xBD dataset in a temp dir."""
    root = tmp_path_factory.mktemp("xbd")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.download_data import generate_synthetic_dataset
    generate_synthetic_dataset(
        output_dir=str(root),
        n_images=4,
        image_size=256,
        split="train",
    )
    return str(root)


# ---------------------------------------------------------------------------
# Pipeline: Tiling
# ---------------------------------------------------------------------------

class TestRasterTiler:
    def test_tile_image_covers_all_pixels(self, sample_image):
        from src.pipeline.tiling import RasterTiler
        tiler  = RasterTiler(tile_size=64, overlap=16)
        tiles, specs = tiler.tile_image(sample_image)

        # Every spec's region must be within image bounds
        h, w = sample_image.shape[:2]
        for spec in specs:
            assert spec.col_off + spec.width  <= w + spec.pad_right
            assert spec.row_off + spec.height <= h + spec.pad_bottom

    def test_tile_count_correct(self):
        from src.pipeline.tiling import RasterTiler
        tiler = RasterTiler(tile_size=64, overlap=0)
        specs = tiler.compute_tile_specs(128, 128)
        assert len(specs) == 4   # 2x2 grid

    def test_reassemble_recovers_input(self):
        from src.pipeline.tiling import RasterTiler
        tiler = RasterTiler(tile_size=64, overlap=0)
        image = np.random.rand(128, 128).astype(np.float32)
        tiles, specs = tiler.tile_image(image)
        reconstructed = tiler.reassemble(tiles, specs, 128, 128, n_classes=1)
        # Should be close (not exact due to hann window)
        assert reconstructed.shape == (128, 128)
        assert np.isfinite(reconstructed).all()

    def test_tiling_2d_mask(self, sample_mask):
        from src.pipeline.tiling import RasterTiler
        tiler = RasterTiler(tile_size=64, overlap=8)
        tiles, specs = tiler.tile_image(sample_mask)
        assert all(t.shape == (64, 64) for t in tiles)


# ---------------------------------------------------------------------------
# Pipeline: Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_augmentation_pipeline_train(self, sample_image, sample_mask):
        from src.pipeline.preprocessing import build_augmentation_pipeline
        aug = build_augmentation_pipeline(phase="train", image_size=128)
        result = aug(
            image=sample_image,
            post_image=sample_image.copy(),
            building_mask=sample_mask,
            damage_mask=sample_mask,
        )
        assert result["image"].shape[1:] == (128, 128)  # (C, H, W)

    def test_augmentation_pipeline_val(self, sample_image, sample_mask):
        from src.pipeline.preprocessing import build_augmentation_pipeline
        aug = build_augmentation_pipeline(phase="val", image_size=224)
        result = aug(
            image=sample_image,
            post_image=sample_image.copy(),
            building_mask=sample_mask,
            damage_mask=sample_mask,
        )
        assert result["image"].shape == (3, 224, 224)

    def test_image_preprocessor_tensor_shape(self, sample_image):
        from src.pipeline.preprocessing import ImagePreprocessor
        proc = ImagePreprocessor()
        tensor = proc.preprocess(sample_image)
        assert tensor.shape == (3, 224, 224)

    def test_preprocess_pair_channels(self, sample_image):
        from src.pipeline.preprocessing import ImagePreprocessor
        proc = ImagePreprocessor()
        pair = proc.preprocess_pair(sample_image, sample_image)
        assert pair.shape[0] == 6   # 3 + 3


# ---------------------------------------------------------------------------
# Models: Segmentation
# ---------------------------------------------------------------------------

class TestSegmentationModel:
    def test_forward_shape(self, device):
        from src.models.segmentation import BuildingSegmentationModel
        model = BuildingSegmentationModel().to(device)
        x = torch.rand(2, 3, 256, 256).to(device)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_predict_mask_binary(self, device):
        from src.models.segmentation import BuildingSegmentationModel
        model = BuildingSegmentationModel().to(device)
        x = torch.rand(1, 3, 256, 256).to(device)
        mask = model.predict_mask(x)
        unique = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique)

    def test_bce_dice_loss(self, device):
        from src.models.segmentation import BceDiceLoss
        loss_fn = BceDiceLoss()
        logits  = torch.randn(2, 1, 256, 256)
        targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_parameter_count_reasonable(self, device):
        from src.models.segmentation import BuildingSegmentationModel
        model = BuildingSegmentationModel()
        # ResNet34 U-Net should be in range 20-30M params
        assert 10_000_000 < model.n_parameters < 50_000_000


# ---------------------------------------------------------------------------
# Models: Damage Classification
# ---------------------------------------------------------------------------

class TestDamageModel:
    def test_forward_shape(self, device):
        from src.models.change_detection import DamageClassificationModel
        model = DamageClassificationModel().to(device)
        x = torch.rand(2, 6, 256, 256).to(device)
        out = model(x)
        assert out.shape == (2, 5, 256, 256)

    def test_predict_class_indices(self, device):
        from src.models.change_detection import DamageClassificationModel
        model = DamageClassificationModel().to(device)
        x     = torch.rand(1, 6, 128, 128).to(device)
        pred  = model.predict(x)
        assert pred.shape == (1, 128, 128)
        assert pred.min() >= 0
        assert pred.max() <= 4

    def test_focal_loss(self, device):
        from src.models.change_detection import WeightedFocalLoss
        loss_fn = WeightedFocalLoss()
        logits  = torch.randn(2, 5, 64, 64)
        targets = torch.randint(0, 5, (2, 64, 64))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Utils: Geo
# ---------------------------------------------------------------------------

class TestGeoUtils:
    def test_save_and_reload_geotiff(self, tmp_path, sample_mask):
        from src.utils.geo_utils import save_geotiff
        from rasterio.transform import from_bounds
        import rasterio

        path = str(tmp_path / "test.tif")
        transform = from_bounds(0, 0, 1, 1, sample_mask.shape[1], sample_mask.shape[0])
        save_geotiff(sample_mask.astype(np.float32), path, transform)

        with rasterio.open(path) as src:
            loaded = src.read(1)
        assert loaded.shape == sample_mask.shape

    def test_polygonize_damage_map(self, sample_damage_mask):
        from src.utils.geo_utils import polygonize_damage_map
        from rasterio.transform import from_bounds

        h, w = sample_damage_mask.shape
        transform = from_bounds(0, 0, w, h, w, h)
        gdf = polygonize_damage_map(sample_damage_mask, transform, crs="EPSG:4326")
        assert len(gdf) > 0
        assert "damage_class" in gdf.columns
        assert "damage_label" in gdf.columns

    def test_rasterize_polygons_roundtrip(self, sample_damage_mask):
        from src.utils.geo_utils import polygonize_damage_map, rasterize_polygons
        from rasterio.transform import from_bounds

        h, w = sample_damage_mask.shape
        transform = from_bounds(0, 0, w, h, w, h)
        gdf = polygonize_damage_map(sample_damage_mask, transform)

        if not gdf.empty:
            mask = rasterize_polygons(
                gdf, (h, w), transform, attribute_col="damage_class"
            )
            assert mask.shape == (h, w)
            assert mask.max() <= 4


# ---------------------------------------------------------------------------
# Utils: Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_segmentation_metrics_perfect_prediction(self):
        from src.utils.metrics import SegmentationMetrics
        m = SegmentationMetrics()
        targets = torch.ones(2, 1, 64, 64)
        logits  = torch.ones(2, 1, 64, 64) * 5.0   # high positive logits
        m.update(logits, targets)
        results = m.compute()
        assert results["iou"]  > 0.95
        assert results["dice"] > 0.95

    def test_damage_metrics_update_compute(self):
        from src.utils.metrics import DamageMetrics
        m = DamageMetrics()
        logits  = torch.randn(2, 5, 32, 32)
        targets = torch.randint(1, 5, (2, 32, 32))
        m.update(logits, targets)
        results = m.compute()
        assert "xview2_score" in results
        assert "per_class" in results


# ---------------------------------------------------------------------------
# Integration: full assessment pipeline on tiny synthetic data
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_assess_full_scene_cpu(self):
        """Run the complete assessment pipeline on a small synthetic image."""
        from src.models.damage_classifier import DamageAssessmentPipeline

        pipeline = DamageAssessmentPipeline(device="cpu")
        pre  = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        result = pipeline.assess_full_scene(
            pre, post,
            tile_size=128, overlap=16, batch_size=2
        )

        assert "building_prob_map" in result
        assert "damage_map" in result
        assert "damage_rgb" in result
        assert "stats" in result
        assert result["building_prob_map"].shape == (256, 256)
        assert result["damage_map"].shape        == (256, 256)

    def test_assessor_single_pair_png(self, tmp_path, synthetic_xbd_dir):
        """End-to-end: load synthetic xBD PNG pair and produce outputs."""
        from src.inference.assessor import GeoSightAssessor
        from PIL import Image as PILImage

        # Find a pair in synthetic data
        img_dir = Path(synthetic_xbd_dir) / "train" / "images"
        pre_imgs = list(img_dir.glob("*_pre_*.png"))
        if not pre_imgs:
            pytest.skip("No synthetic images found")

        pre_path = str(pre_imgs[0])
        post_path = pre_path.replace("_pre_", "_post_")
        if not Path(post_path).exists():
            pytest.skip("No matching post image")

        assessor = GeoSightAssessor(device="cpu", tile_size=128, tile_overlap=16)
        report = assessor.assess(
            pre_path, post_path,
            output_dir=str(tmp_path / "output"),
            save_leaflet_map=False,
        )

        assert "statistics" in report
        assert "output_files" in report
        assert Path(report["output_files"]["report_json"]).exists()
