"""
Disaster Type Classifier.

WHY THIS EXISTS:
----------------
When our damage model outputs a map showing "40% destroyed, 30% major damage",
a responder's next question is: "What happened here?"

This matters because:
  - Earthquake → survivors trapped UNDER rubble → need heavy machinery, dogs
  - Flood → survivors ON TOP of buildings/trees → need boats, helicopters
  - Hurricane → scattered debris, downed power lines → need chainsaws, line crews
  - Wildfire → burn zones, toxic smoke → need hazmat, burn treatment
  - Tsunami → waterlogged structures, debris fields → need drainage, water rescue

Different disasters also have distinct spatial damage signatures that we can
learn from the data:

  - Earthquake: radial damage pattern from epicentre, liquefaction zones
  - Flood: linear pattern along river/coastline, elevation-dependent
  - Hurricane: wind-band spiral, eyewall ring of max destruction
  - Wildfire: fire-line boundary, follows fuel and wind direction
  - Tsunami: coastal inundation gradient, distance-from-shore decay

This classifier looks at the DISTRIBUTION and SPATIAL PATTERN of damage
across an entire scene and predicts the most likely disaster type.

Architecture:
    Takes the per-pixel damage prediction map (from Siamese U-Net) +
    the pre/post image pair, extracts scene-level features, and
    classifies into one of 7 disaster categories.

    This is a SCENE-LEVEL classifier (one prediction per image pair),
    not a pixel-level one.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# xBD events mapped to disaster types
DISASTER_TYPES = {
    0: "earthquake",
    1: "flood",
    2: "hurricane",
    3: "wildfire",
    4: "tsunami",
    5: "volcanic",
    6: "tornado",
}

NUM_DISASTER_TYPES = len(DISASTER_TYPES)

# xBD event-to-type mapping (ground truth from the dataset)
XBD_EVENT_TO_TYPE = {
    "guatemala-volcano":     "volcanic",
    "hurricane-florence":    "hurricane",
    "hurricane-harvey":      "hurricane",
    "hurricane-matthew":     "hurricane",
    "hurricane-michael":     "hurricane",
    "mexico-earthquake":     "earthquake",
    "midwest-flooding":      "flood",
    "moore-tornado":         "tornado",
    "nepal-flooding":        "flood",
    "palu-tsunami":          "tsunami",
    "portugal-wildfire":     "wildfire",
    "santa-rosa-wildfire":   "wildfire",
    "socal-fire":            "wildfire",
    "sunda-tsunami":         "tsunami",
    "tuscaloosa-tornado":    "tornado",
    "woolsey-fire":          "wildfire",
    "joplin-tornado":        "tornado",
    "lower-puna-volcano":    "volcanic",
    "pinery-bushfire":       "wildfire",
}


class DamagePatternFeatureExtractor(nn.Module):
    """
    Extracts scene-level features from a damage map and image pair.

    Computes two types of features:

    1. Statistical features (no learnable params):
       - Per-class pixel ratios (what % is destroyed vs minor)
       - Spatial autocorrelation (is damage clustered or dispersed?)
       - Damage gradient (directional damage intensity change)
       - Connectivity (are damaged regions contiguous or scattered?)

    2. Visual features (CNN):
       - A lightweight CNN that looks at the post-disaster image
         to capture texture cues (rubble vs waterlogged vs burned)
    """

    def __init__(self, visual_feature_dim: int = 256):
        super().__init__()

        # Lightweight CNN for visual texture features from post-disaster image
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, visual_feature_dim),
            nn.ReLU(inplace=True),
        )

        # Statistical feature dimension:
        #   5 (class ratios) + 4 (spatial stats) + 8 (gradient) + 4 (connectivity)
        self.stat_feature_dim = 21
        self.visual_feature_dim = visual_feature_dim
        self.total_feature_dim = self.stat_feature_dim + visual_feature_dim

    def forward(
        self,
        post_image: torch.Tensor,
        damage_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            post_image:   (B, 3, H, W) — post-disaster image
            damage_probs: (B, 5, H, W) — softmax probabilities from damage model

        Returns:
            features: (B, total_feature_dim) scene-level feature vector
        """
        visual_feats = self.visual_encoder(post_image)
        stat_feats   = self._compute_statistical_features(damage_probs)
        return torch.cat([visual_feats, stat_feats], dim=1)

    @staticmethod
    def _compute_statistical_features(damage_probs: torch.Tensor) -> torch.Tensor:
        """
        Extract hand-crafted spatial statistics from the damage probability map.

        These are domain-knowledge features that encode how different disasters
        leave distinct damage patterns.
        """
        B = damage_probs.shape[0]
        device = damage_probs.device
        features = []

        for b in range(B):
            probs = damage_probs[b]  # (5, H, W)

            # 1. Class ratios — what % of pixels are in each damage class
            class_ratios = probs.mean(dim=(1, 2))  # (5,)

            # 2. Spatial concentration — are damaged pixels clustered or spread?
            #    High variance in damage probability = concentrated damage
            #    Low variance = evenly spread
            damage_sum = probs[2:].sum(dim=0)  # sum of minor + major + destroyed
            spatial_mean = damage_sum.mean()
            spatial_var  = damage_sum.var()
            spatial_max  = damage_sum.max()
            spatial_q90  = torch.quantile(damage_sum.flatten().float(), 0.9)

            spatial_stats = torch.stack([spatial_mean, spatial_var, spatial_max, spatial_q90])

            # 3. Directional gradient — does damage increase in a particular direction?
            #    Earthquakes: radial from centre. Floods: one direction (downstream).
            #    Hurricanes: spiral. Wildfires: fire-line front.
            H, W = damage_sum.shape
            h_mid, w_mid = H // 2, W // 2

            # Split into quadrants and compare damage intensity
            q_tl = damage_sum[:h_mid, :w_mid].mean()
            q_tr = damage_sum[:h_mid, w_mid:].mean()
            q_bl = damage_sum[h_mid:, :w_mid].mean()
            q_br = damage_sum[h_mid:, w_mid:].mean()

            # Gradient: horizontal, vertical, diagonal
            grad_h = (q_tr + q_br) - (q_tl + q_bl)  # left→right
            grad_v = (q_bl + q_br) - (q_tl + q_tr)  # top→bottom
            grad_d1 = q_br - q_tl                     # diagonal
            grad_d2 = q_bl - q_tr                     # anti-diagonal

            # Asymmetry ratios (high = directional, low = symmetric/radial)
            total_q = q_tl + q_tr + q_bl + q_br + 1e-8
            asym_h = torch.abs(grad_h) / total_q
            asym_v = torch.abs(grad_v) / total_q
            asym_d1 = torch.abs(grad_d1) / total_q
            asym_d2 = torch.abs(grad_d2) / total_q

            gradient_feats = torch.stack([
                grad_h, grad_v, grad_d1, grad_d2,
                asym_h, asym_v, asym_d1, asym_d2,
            ])

            # 4. Connectivity — ratio of destroyed-class to minor-class
            #    Earthquakes have sharp destroyed/no-damage boundary
            #    Floods have gradual transition (minor → major → destroyed)
            ratio_destroyed_to_minor = (class_ratios[4] + 1e-8) / (class_ratios[2] + 1e-8)
            ratio_severe_to_light = (class_ratios[3] + class_ratios[4] + 1e-8) / \
                                    (class_ratios[1] + class_ratios[2] + 1e-8)
            damage_entropy = -(probs[1:] * (probs[1:] + 1e-8).log()).sum(dim=0).mean()
            total_damage_ratio = 1.0 - class_ratios[0] - class_ratios[1]

            connectivity_feats = torch.stack([
                ratio_destroyed_to_minor,
                ratio_severe_to_light,
                damage_entropy,
                total_damage_ratio,
            ])

            feat = torch.cat([class_ratios, spatial_stats, gradient_feats, connectivity_feats])
            features.append(feat)

        return torch.stack(features)  # (B, 21)


class DisasterTypeClassifier(nn.Module):
    """
    Scene-level disaster type classifier.

    Takes features from DamagePatternFeatureExtractor and predicts
    which type of disaster caused the observed damage pattern.

    Input:  post-disaster image + damage probability map
    Output: 7-class probability (earthquake, flood, hurricane, wildfire, tsunami, volcanic, tornado)
    """

    def __init__(self, visual_feature_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.feature_extractor = DamagePatternFeatureExtractor(
            visual_feature_dim=visual_feature_dim
        )

        feat_dim = self.feature_extractor.total_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_DISASTER_TYPES),
        )

        logger.info(
            f"DisasterTypeClassifier: feature_dim={feat_dim}, "
            f"classes={NUM_DISASTER_TYPES}, params={self.n_parameters:,}"
        )

    def forward(
        self,
        post_image: torch.Tensor,
        damage_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            post_image:   (B, 3, H, W)
            damage_probs: (B, 5, H, W) — softmax output from damage model

        Returns:
            logits: (B, 7) — disaster type logits
        """
        features = self.feature_extractor(post_image, damage_probs)
        return self.classifier(features)

    def predict(
        self,
        post_image: torch.Tensor,
        damage_probs: torch.Tensor,
    ) -> Dict:
        """
        Returns predicted disaster type with confidence.

        Returns dict with:
            type:       str — most likely disaster type
            confidence: float — probability of that type
            all_probs:  dict — probability for each disaster type
        """
        self.eval()  # BatchNorm requires eval mode for batch_size=1
        with torch.no_grad():
            logits = self.forward(post_image, damage_probs)
            probs  = F.softmax(logits, dim=1)

            # Average probabilities across batch (multiple tiles → single prediction)
            avg_probs = probs.mean(dim=0)  # (num_classes,)
            top_idx   = avg_probs.argmax().item()
            top_conf  = avg_probs[top_idx].item()
            all_probs = {
                DISASTER_TYPES[i]: round(avg_probs[i].item(), 4)
                for i in range(NUM_DISASTER_TYPES)
            }

            return {
                "type":       DISASTER_TYPES[top_idx],
                "confidence": round(top_conf, 4),
                "all_probs":  all_probs,
            }

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_disaster_type_from_event(event_name: str) -> Optional[int]:
    """
    Look up disaster type index from xBD event name.
    Used for generating training labels.
    """
    event_lower = event_name.lower().replace(" ", "-")

    for event_key, dtype in XBD_EVENT_TO_TYPE.items():
        if event_key in event_lower:
            type_to_idx = {v: k for k, v in DISASTER_TYPES.items()}
            return type_to_idx.get(dtype)

    return None


def get_disaster_response_protocol(disaster_type: str) -> Dict:
    """
    Returns actionable response guidance based on disaster type.

    This is the bridge from ML prediction to saving lives:
    the model output gets translated into specific actions
    that emergency managers can execute.
    """
    protocols = {
        "earthquake": {
            "search_rescue_priority": "high",
            "primary_hazard": "structural collapse, aftershocks",
            "equipment_needed": [
                "heavy cranes", "concrete saws", "search dogs",
                "acoustic listening devices", "thermal cameras"
            ],
            "survivor_location": "trapped under rubble — urban SAR teams needed",
            "time_critical_window": "72 hours (golden period for trapped survivors)",
            "secondary_hazards": ["gas leaks", "fire", "landslides", "aftershocks"],
            "infrastructure_priorities": ["hospitals", "bridges", "water mains"],
        },
        "flood": {
            "search_rescue_priority": "high",
            "primary_hazard": "drowning, waterborne disease",
            "equipment_needed": [
                "boats", "helicopters", "water pumps",
                "water purification systems", "sandbags"
            ],
            "survivor_location": "rooftops, elevated ground, upper floors",
            "time_critical_window": "24-48 hours (water levels, hypothermia)",
            "secondary_hazards": ["contaminated water", "electrocution", "structural weakening"],
            "infrastructure_priorities": ["water treatment", "power grid", "roads"],
        },
        "hurricane": {
            "search_rescue_priority": "medium-high",
            "primary_hazard": "wind damage, storm surge, flying debris",
            "equipment_needed": [
                "chainsaws", "generators", "tarps", "line trucks", "boats"
            ],
            "survivor_location": "damaged homes, shelters, flooded areas",
            "time_critical_window": "48-96 hours (storm surge, power outage)",
            "secondary_hazards": ["flooding", "tornadoes", "power outage", "fuel shortage"],
            "infrastructure_priorities": ["power lines", "cell towers", "port facilities"],
        },
        "wildfire": {
            "search_rescue_priority": "medium",
            "primary_hazard": "fire, smoke inhalation, burn injuries",
            "equipment_needed": [
                "fire trucks", "aerial tankers", "hazmat gear",
                "respirators", "burn treatment supplies"
            ],
            "survivor_location": "evacuation zones, cleared areas",
            "time_critical_window": "ongoing until fire contained",
            "secondary_hazards": ["mudslides (post-fire)", "toxic ash", "air quality"],
            "infrastructure_priorities": ["evacuation routes", "water supply", "cell towers"],
        },
        "tsunami": {
            "search_rescue_priority": "high",
            "primary_hazard": "drowning, debris impact, waterlogging",
            "equipment_needed": [
                "boats", "helicopters", "heavy equipment",
                "water pumps", "body recovery teams"
            ],
            "survivor_location": "high ground, upper floors of intact buildings",
            "time_critical_window": "24-72 hours (trapped, injured, exposure)",
            "secondary_hazards": ["contaminated water", "structural collapse", "disease"],
            "infrastructure_priorities": ["coastal roads", "harbours", "medical facilities"],
        },
        "volcanic": {
            "search_rescue_priority": "medium",
            "primary_hazard": "lava, pyroclastic flow, ashfall",
            "equipment_needed": [
                "respirators", "evacuation vehicles", "hazmat suits",
                "air quality monitors", "heavy loaders"
            ],
            "survivor_location": "outside exclusion zone, evacuation centres",
            "time_critical_window": "varies (ongoing eruption monitoring)",
            "secondary_hazards": ["lahars", "toxic gas", "roof collapse from ash weight"],
            "infrastructure_priorities": ["evacuation routes", "airports", "water supply"],
        },
        "tornado": {
            "search_rescue_priority": "high",
            "primary_hazard": "structural collapse, flying debris",
            "equipment_needed": [
                "heavy equipment", "search dogs", "chainsaws",
                "mobile medical units", "temporary shelters"
            ],
            "survivor_location": "basements, interior rooms, under debris",
            "time_critical_window": "24-48 hours (trapped, injured)",
            "secondary_hazards": ["gas leaks", "downed power lines", "secondary tornadoes"],
            "infrastructure_priorities": ["hospitals", "schools", "mobile home parks"],
        },
    }
    return protocols.get(disaster_type, protocols["earthquake"])
