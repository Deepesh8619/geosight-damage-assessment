"""
Tests for the V2 modules: Siamese U-Net, Disaster Classifier,
Impact Analysis, and Spatial Analysis.

Run with: pytest tests/test_new_modules.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Siamese U-Net
# ---------------------------------------------------------------------------

class TestSiameseUNet:
    def test_forward_shape(self, device):
        from src.models.siamese_unet import SiameseUNet
        model = SiameseUNet(encoder_name="resnet18").to(device)  # smaller for test speed
        pre  = torch.rand(2, 3, 256, 256).to(device)
        post = torch.rand(2, 3, 256, 256).to(device)
        out = model(pre, post)
        assert out.shape == (2, 5, 256, 256)

    def test_backward_compatible_forward(self, device):
        from src.models.siamese_unet import SiameseUNet
        model = SiameseUNet(encoder_name="resnet18").to(device)
        x = torch.rand(1, 6, 128, 128).to(device)
        out = model.forward_concatenated(x)
        assert out.shape == (1, 5, 128, 128)

    def test_predict_with_confidence(self, device):
        from src.models.siamese_unet import SiameseUNet
        model = SiameseUNet(encoder_name="resnet18").to(device)
        pre  = torch.rand(1, 3, 128, 128).to(device)
        post = torch.rand(1, 3, 128, 128).to(device)
        classes, confidence = model.predict_with_confidence(pre, post)
        assert classes.shape == (1, 128, 128)
        assert confidence.shape == (1, 128, 128)
        assert classes.min() >= 0
        assert classes.max() <= 4
        assert confidence.min() >= 0
        assert confidence.max() <= 1.0

    def test_shared_weights(self, device):
        """Verify that both pre and post pass through the same encoder."""
        from src.models.siamese_unet import SiameseUNet
        model = SiameseUNet(encoder_name="resnet18").to(device)
        x = torch.rand(1, 3, 128, 128).to(device)
        # Same input should produce same features
        f1 = model.encoder(x)
        f2 = model.encoder(x)
        for a, b in zip(f1, f2):
            assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Disaster Type Classifier
# ---------------------------------------------------------------------------

class TestDisasterClassifier:
    def test_forward_shape(self, device):
        from src.models.disaster_classifier import DisasterTypeClassifier
        model = DisasterTypeClassifier(visual_feature_dim=64).to(device)
        post  = torch.rand(2, 3, 128, 128).to(device)
        probs = torch.rand(2, 5, 128, 128).to(device)
        # Normalize probs to sum to 1 along class dim
        probs = probs / probs.sum(dim=1, keepdim=True)
        out = model(post, probs)
        assert out.shape == (2, 7)  # 7 disaster types

    def test_predict_returns_type(self, device):
        from src.models.disaster_classifier import DisasterTypeClassifier, DISASTER_TYPES
        model = DisasterTypeClassifier(visual_feature_dim=64).to(device)
        post  = torch.rand(1, 3, 64, 64).to(device)
        probs = torch.rand(1, 5, 64, 64).to(device)
        probs = probs / probs.sum(dim=1, keepdim=True)
        result = model.predict(post, probs)
        assert "type" in result
        assert result["type"] in DISASTER_TYPES.values()
        assert 0.0 <= result["confidence"] <= 1.0

    def test_event_to_type_mapping(self):
        from src.models.disaster_classifier import get_disaster_type_from_event
        assert get_disaster_type_from_event("hurricane-harvey") is not None
        assert get_disaster_type_from_event("mexico-earthquake") is not None
        assert get_disaster_type_from_event("palu-tsunami") is not None

    def test_response_protocol(self):
        from src.models.disaster_classifier import get_disaster_response_protocol
        proto = get_disaster_response_protocol("earthquake")
        assert "equipment_needed" in proto
        assert "time_critical_window" in proto
        assert "survivor_location" in proto


# ---------------------------------------------------------------------------
# Humanitarian Impact Analysis
# ---------------------------------------------------------------------------

class TestImpactAnalysis:
    @pytest.fixture
    def sample_scene(self):
        """Create a small scene with known damage distribution."""
        damage_map = np.zeros((128, 128), dtype=np.int64)
        building_mask = np.zeros((128, 128), dtype=bool)

        # 4 buildings with different damage levels
        building_mask[10:30, 10:30] = True
        damage_map[10:30, 10:30] = 1     # no damage

        building_mask[10:30, 40:60] = True
        damage_map[10:30, 40:60] = 2     # minor

        building_mask[50:70, 10:30] = True
        damage_map[50:70, 10:30] = 3     # major

        building_mask[50:70, 40:60] = True
        damage_map[50:70, 40:60] = 4     # destroyed

        return damage_map, building_mask

    def test_impact_report_fields(self, sample_scene):
        from src.utils.impact_analysis import HumanitarianImpactAnalyzer
        damage_map, building_mask = sample_scene
        analyzer = HumanitarianImpactAnalyzer()
        report = analyzer.analyze(damage_map, building_mask, disaster_type="earthquake")

        assert report.total_buildings > 0
        assert report.estimated_displaced > 0
        assert report.estimated_economic_loss_usd > 0
        assert report.estimated_tents_needed >= 0
        assert 0 <= report.severity_index <= 100
        assert report.severity_label in ["low", "moderate", "severe", "catastrophic"]

    def test_destroyed_buildings_counted(self, sample_scene):
        from src.utils.impact_analysis import HumanitarianImpactAnalyzer
        damage_map, building_mask = sample_scene
        analyzer = HumanitarianImpactAnalyzer()
        report = analyzer.analyze(damage_map, building_mask)
        assert report.buildings_destroyed >= 1
        assert report.buildings_major_damage >= 1

    def test_zero_damage_scene(self):
        from src.utils.impact_analysis import HumanitarianImpactAnalyzer
        damage_map = np.ones((64, 64), dtype=np.int64)  # all no-damage
        building_mask = np.ones((64, 64), dtype=bool)
        analyzer = HumanitarianImpactAnalyzer()
        report = analyzer.analyze(damage_map, building_mask)
        assert report.estimated_displaced == 0
        assert report.estimated_economic_loss_usd == 0.0

    def test_priority_zones(self, sample_scene):
        from src.utils.impact_analysis import generate_priority_zones
        damage_map, building_mask = sample_scene
        priority = generate_priority_zones(damage_map, building_mask, grid_size=32)
        assert priority.shape == (128, 128)
        # Destroyed area should have higher priority than no-damage
        destroyed_priority = priority[50:70, 40:60].mean()
        nodamage_priority  = priority[10:30, 10:30].mean()
        assert destroyed_priority > nodamage_priority

    def test_report_to_dict(self, sample_scene):
        from src.utils.impact_analysis import HumanitarianImpactAnalyzer
        damage_map, building_mask = sample_scene
        analyzer = HumanitarianImpactAnalyzer()
        report = analyzer.analyze(damage_map, building_mask, disaster_type="flood")
        d = report.to_dict()
        assert "buildings" in d
        assert "population" in d
        assert "economic" in d
        assert "shelter" in d
        assert "severity" in d
        assert d["disaster_type"] == "flood"


# ---------------------------------------------------------------------------
# Spatial Analysis
# ---------------------------------------------------------------------------

class TestSpatialAnalysis:
    @pytest.fixture
    def radial_damage(self):
        """Damage that radiates from center (simulates earthquake)."""
        damage_map = np.zeros((200, 200), dtype=np.int64)
        building_mask = np.ones((200, 200), dtype=bool)
        ys, xs = np.mgrid[0:200, 0:200]
        dist = np.sqrt((ys - 100) ** 2 + (xs - 100) ** 2)
        damage_map[dist < 20] = 4  # destroyed at center
        damage_map[(dist >= 20) & (dist < 40)] = 3  # major
        damage_map[(dist >= 40) & (dist < 60)] = 2  # minor
        damage_map[(dist >= 60) & (dist < 80)] = 1  # no damage
        return damage_map, building_mask

    def test_epicentre_near_center(self, radial_damage):
        from src.utils.spatial_analysis import SpatialDamageAnalyzer
        damage_map, building_mask = radial_damage
        analyzer = SpatialDamageAnalyzer()
        result = analyzer.find_damage_epicentre(damage_map, building_mask)
        cy, cx = result["center"]
        # Epicentre should be near (100, 100)
        assert abs(cy - 100) < 15
        assert abs(cx - 100) < 15

    def test_radial_profile_decays(self, radial_damage):
        from src.utils.spatial_analysis import SpatialDamageAnalyzer
        damage_map, building_mask = radial_damage
        analyzer = SpatialDamageAnalyzer()
        profile = analyzer.compute_radial_profile(
            damage_map, building_mask, center=(100, 100)
        )
        assert profile["decay_rate"] < 0  # severity decreases with distance
        assert "point-source" in profile["pattern"]

    def test_damage_clusters(self, radial_damage):
        from src.utils.spatial_analysis import SpatialDamageAnalyzer
        damage_map, building_mask = radial_damage
        analyzer = SpatialDamageAnalyzer()
        severe = (damage_map >= 3) & building_mask
        clusters = analyzer.find_damage_clusters(severe)
        assert len(clusters) >= 1
        assert clusters[0]["area_m2"] > 0

    def test_gradient_radial_pattern(self, radial_damage):
        from src.utils.spatial_analysis import SpatialDamageAnalyzer
        damage_map, building_mask = radial_damage
        analyzer = SpatialDamageAnalyzer()
        gradient = analyzer.compute_damage_gradient(damage_map, building_mask)
        # Radial damage should NOT be strongly directional
        assert gradient["gradient_magnitude"] < 1.0

    def test_directional_damage(self):
        """Damage concentrated on one side (simulates flood/tsunami)."""
        from src.utils.spatial_analysis import SpatialDamageAnalyzer
        damage_map = np.zeros((200, 200), dtype=np.int64)
        building_mask = np.ones((200, 200), dtype=bool)
        # All damage on the east side
        damage_map[:, 150:] = 4
        damage_map[:, 130:150] = 3

        analyzer = SpatialDamageAnalyzer()
        gradient = analyzer.compute_damage_gradient(damage_map, building_mask)
        assert gradient["is_directional"]
        assert gradient["dominant_direction"] in ["E", "NE", "SE"]

    def test_full_analysis_returns_all_keys(self, radial_damage):
        from src.utils.spatial_analysis import SpatialDamageAnalyzer
        damage_map, building_mask = radial_damage
        analyzer = SpatialDamageAnalyzer()
        report = analyzer.full_analysis(damage_map, building_mask)
        assert "epicentre" in report
        assert "clusters" in report
        assert "gradient" in report
        assert "radial_profile" in report
        assert "boundary" in report
        assert "spread" in report

    def test_boundary_compactness(self, radial_damage):
        from src.utils.spatial_analysis import SpatialDamageAnalyzer
        damage_map, building_mask = radial_damage
        severe = (damage_map >= 3) & building_mask
        analyzer = SpatialDamageAnalyzer()
        boundary = analyzer.find_damage_boundary(severe)
        # Radial damage should be relatively compact (circular)
        assert boundary["compactness"] > 0.3
        assert "circular" in boundary["shape_hint"]


# ---------------------------------------------------------------------------
# Integration: full V2 pipeline on synthetic data
# ---------------------------------------------------------------------------

class TestV2Integration:
    def test_full_pipeline_cpu(self):
        """Run the full V2 pipeline end-to-end on synthetic data."""
        from src.models.damage_classifier import DamageAssessmentPipeline

        pipeline = DamageAssessmentPipeline(device="cpu", use_siamese=True)
        pre  = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        result = pipeline.assess_full_scene(
            pre, post, tile_size=128, overlap=16, batch_size=2
        )

        # Check all V2 outputs exist
        assert "building_prob_map" in result
        assert "damage_map" in result
        assert "confidence_map" in result
        assert "priority_map" in result
        assert "disaster_type" in result
        assert "spatial_analysis" in result
        assert "impact_report" in result
        assert "stats" in result

        # Check shapes
        assert result["building_prob_map"].shape == (256, 256)
        assert result["damage_map"].shape == (256, 256)
        assert result["confidence_map"].shape == (256, 256)

        # Check disaster type prediction has expected fields
        dt = result["disaster_type"]
        assert "type" in dt
        assert "confidence" in dt

        # Check impact report
        imp = result["impact_report"]
        assert "buildings" in imp
        assert "population" in imp
        assert "economic" in imp
        assert "severity" in imp

    def test_old_model_backward_compatible(self):
        """Verify the pipeline still works with use_siamese=False."""
        from src.models.damage_classifier import DamageAssessmentPipeline

        pipeline = DamageAssessmentPipeline(device="cpu", use_siamese=False)
        pre  = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        result = pipeline.assess_full_scene(
            pre, post, tile_size=128, overlap=0, batch_size=1
        )
        assert "damage_map" in result
