"""
Classical ML & Unsupervised Learning Module.

This module demonstrates ML techniques BEYOND deep learning:
  - XGBoost (gradient boosted trees) for disaster type classification
  - DBSCAN (density-based clustering) for damage hotspot detection
  - PCA (principal component analysis) for multi-spectral dimensionality reduction
  - K-Means clustering for unsupervised land cover segmentation
  - MC Dropout for Bayesian uncertainty estimation

These complement the deep learning models by:
  1. Adding interpretability (XGBoost feature importance shows WHY it classified a disaster type)
  2. Working without labels (DBSCAN, K-Means, PCA are unsupervised)
  3. Providing uncertainty (MC Dropout gives confidence intervals, not point estimates)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from loguru import logger


# ===========================================================================
# 1. XGBoost — Gradient Boosted Trees for Disaster Type Classification
# ===========================================================================

class XGBoostDisasterClassifier:
    """
    Classical ML alternative to the neural network disaster classifier.

    WHY XGBoost IN ADDITION TO THE CNN:
      - XGBoost works on TABULAR features (the 21 spatial statistics we extract)
      - It's interpretable: feature_importance tells you WHICH spatial features
        matter most for classifying disaster type
      - It's fast: trains in seconds on extracted features
      - It ensembles well with the CNN: average both predictions for higher accuracy
      - In interviews: shows you know classical ML, not just deep learning

    Algorithm: Gradient Boosted Decision Trees
      - Builds many small decision trees sequentially
      - Each tree corrects the errors of the previous one
      - Final prediction = sum of all trees' outputs
      - Better than Random Forest for structured/tabular data

    Input: 21-dimensional feature vector from DamagePatternFeatureExtractor
    Output: 7-class disaster type probability
    """

    DISASTER_TYPES = [
        "earthquake", "flood", "hurricane", "wildfire",
        "tsunami", "volcanic", "tornado",
    ]

    def __init__(self, n_estimators: int = 100, max_depth: int = 6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.feature_names = [
            # Class ratios (5)
            "ratio_bg", "ratio_nodmg", "ratio_minor", "ratio_major", "ratio_destroyed",
            # Spatial stats (4)
            "spatial_mean", "spatial_var", "spatial_max", "spatial_q90",
            # Gradient (8)
            "grad_h", "grad_v", "grad_d1", "grad_d2",
            "asym_h", "asym_v", "asym_d1", "asym_d2",
            # Connectivity (4)
            "ratio_dest_minor", "ratio_severe_light", "damage_entropy", "total_damage",
        ]

    def train(self, features: np.ndarray, labels: np.ndarray):
        """
        Train on extracted spatial features.

        Args:
            features: (N, 21) array — one row per scene
            labels: (N,) int array — disaster type index (0-6)
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            logger.error("scikit-learn required for XGBoost classifier")
            return

        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self.model.fit(features, labels)
        logger.info(
            f"XGBoost trained: {len(labels)} samples, "
            f"train accuracy={self.model.score(features, labels):.3f}"
        )

    def predict(self, features: np.ndarray) -> Dict:
        """
        Predict disaster type from spatial features.

        Returns:
            dict with type, confidence, probabilities, and feature_importance
        """
        if self.model is None:
            return {"type": "unknown", "confidence": 0.0}

        if features.ndim == 1:
            features = features.reshape(1, -1)

        probs = self.model.predict_proba(features)[0]
        pred_idx = np.argmax(probs)

        return {
            "type": self.DISASTER_TYPES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_probs": {
                t: float(p) for t, p in zip(self.DISASTER_TYPES, probs)
            },
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        WHY THIS MATTERS:
        Unlike a neural network which is a black box, XGBoost tells you
        exactly which features drove its decision.

        Example output:
          "ratio_destroyed": 0.25  ← most important: how much is destroyed
          "asym_h": 0.18           ← directional asymmetry (wind direction)
          "spatial_var": 0.12      ← how concentrated is damage

        This is INTERPRETABLE ML — crucial for government/insurance clients
        who need to understand WHY the system classified something.
        """
        if self.model is None:
            return {}

        importance = self.model.feature_importances_
        return {
            name: float(imp)
            for name, imp in sorted(
                zip(self.feature_names, importance),
                key=lambda x: -x[1]
            )
        }


# ===========================================================================
# 2. DBSCAN — Unsupervised Damage Hotspot Detection
# ===========================================================================

class DBSCANHotspotDetector:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    WHY DBSCAN INSTEAD OF CONNECTED COMPONENTS:
      - Connected components require a binary threshold (damaged / not damaged)
      - DBSCAN finds clusters of ANY density, handles noise, and doesn't
        require pre-specifying the number of clusters
      - It naturally handles irregular shapes (building clusters, flood corridors)
      - Points that don't belong to any cluster are labeled as NOISE — these
        are isolated damage that may be false positives

    Algorithm:
      1. For each point, count neighbors within radius eps
      2. If count >= min_samples → this is a CORE point
      3. Core points that are neighbors form a cluster
      4. Non-core points near a cluster are BORDER points
      5. Points not near any cluster are NOISE

    Input: pixel coordinates of severely damaged buildings
    Output: cluster assignments, centroids, and noise identification
    """

    def __init__(self, eps: float = 50.0, min_samples: int = 5):
        """
        Args:
            eps: maximum distance (in pixels) between two points to be neighbors
            min_samples: minimum points to form a cluster
        """
        self.eps = eps
        self.min_samples = min_samples

    def detect_hotspots(
        self,
        damage_map: np.ndarray,
        building_mask: np.ndarray,
        min_damage_class: int = 3,
    ) -> Dict:
        """
        Find damage hotspots using DBSCAN clustering.

        Args:
            damage_map: (H, W) int array, classes 0-4
            building_mask: (H, W) bool array
            min_damage_class: minimum class to consider (3=major+destroyed)

        Returns:
            dict with clusters, noise points, and statistics
        """
        from sklearn.cluster import DBSCAN

        # Extract coordinates of severely damaged pixels
        severe_mask = (damage_map >= min_damage_class) & building_mask
        ys, xs = np.where(severe_mask)

        if len(ys) < self.min_samples:
            logger.info("DBSCAN: not enough damage points for clustering")
            return {"n_clusters": 0, "clusters": [], "noise_ratio": 0.0}

        # Subsample if too many points (DBSCAN is O(n²) on distance matrix)
        max_points = 5000
        if len(ys) > max_points:
            idx = np.random.choice(len(ys), max_points, replace=False)
            ys, xs = ys[idx], xs[idx]

        points = np.column_stack([xs, ys])

        # Run DBSCAN
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean")
        labels = db.fit_predict(points)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        # Extract cluster info
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_points = points[labels == cluster_id]
            centroid = cluster_points.mean(axis=0)
            spread = cluster_points.std(axis=0).mean()

            clusters.append({
                "id": cluster_id,
                "centroid": (int(centroid[0]), int(centroid[1])),
                "n_points": len(cluster_points),
                "spread_pixels": float(spread),
                "bbox": (
                    int(cluster_points[:, 1].min()),
                    int(cluster_points[:, 0].min()),
                    int(cluster_points[:, 1].max()),
                    int(cluster_points[:, 0].max()),
                ),
            })

        clusters.sort(key=lambda c: c["n_points"], reverse=True)

        result = {
            "n_clusters": n_clusters,
            "clusters": clusters,
            "noise_points": int(n_noise),
            "noise_ratio": float(n_noise / len(points)) if len(points) > 0 else 0.0,
            "total_points": len(points),
        }

        logger.info(
            f"DBSCAN: {n_clusters} hotspots, "
            f"{n_noise} noise points ({result['noise_ratio']:.1%})"
        )
        return result


# ===========================================================================
# 3. PCA — Principal Component Analysis for Multi-Spectral Imagery
# ===========================================================================

class SpectralPCA:
    """
    PCA for multi-spectral satellite imagery dimensionality reduction.

    WHY PCA FOR SATELLITE DATA:
      Sentinel-2 has 13 bands. Landsat has 11. Planet has 4-8.
      Many bands are correlated (e.g., red and green).

      PCA finds the directions of maximum variance and projects the
      data onto fewer dimensions while keeping most information.

      Example: 13 Sentinel-2 bands → 3 PCA components that capture
      95%+ of the variance. These 3 components are:
        PC1: overall brightness (correlates with all visible bands)
        PC2: vegetation index (red vs near-infrared contrast)
        PC3: moisture/water content

      This is LOSSLESS COMPRESSION of spectral information.

    Algorithm:
      1. Flatten spatial dims: (C, H, W) → (C, H*W)
      2. Center the data (subtract mean per band)
      3. Compute covariance matrix (C × C)
      4. Eigendecomposition → eigenvectors (principal components)
      5. Project data onto top-k eigenvectors
      6. Reshape back to (k, H, W)
    """

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, image: np.ndarray):
        """
        Fit PCA on a multi-band image.

        Args:
            image: (C, H, W) or (H, W, C) multi-band satellite image
        """
        from sklearn.decomposition import PCA

        if image.ndim == 3 and image.shape[0] < image.shape[-1]:
            # Already (C, H, W)
            C, H, W = image.shape
        else:
            # (H, W, C) → (C, H, W)
            H, W, C = image.shape
            image = image.transpose(2, 0, 1)

        # Flatten: (C, H*W) → transpose to (H*W, C) for sklearn
        flat = image.reshape(C, -1).T   # (N_pixels, C)

        pca = PCA(n_components=min(self.n_components, C))
        pca.fit(flat)

        self.components_ = pca.components_
        self.mean_ = pca.mean_
        self.explained_variance_ratio_ = pca.explained_variance_ratio_

        total_var = sum(self.explained_variance_ratio_)
        logger.info(
            f"PCA: {C} bands → {self.n_components} components "
            f"({total_var:.1%} variance explained)"
        )

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply fitted PCA to reduce spectral dimensions.

        Args:
            image: (C, H, W) multi-band image

        Returns:
            reduced: (n_components, H, W) dimensionality-reduced image
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        C, H, W = image.shape
        flat = image.reshape(C, -1).T                          # (N, C)
        centered = flat - self.mean_                            # center
        projected = centered @ self.components_.T               # (N, n_components)
        return projected.T.reshape(self.n_components, H, W)    # (k, H, W)

    def fit_transform(self, image: np.ndarray) -> np.ndarray:
        self.fit(image)
        return self.transform(image)

    def get_component_interpretation(self) -> List[Dict]:
        """
        Interpret what each principal component represents.

        Returns loadings — how much each original band contributes
        to each component. High loading on NIR band → vegetation component.
        """
        if self.components_ is None:
            return []

        return [
            {
                "component": i + 1,
                "variance_explained": float(self.explained_variance_ratio_[i]),
                "loadings": self.components_[i].tolist(),
            }
            for i in range(len(self.explained_variance_ratio_))
        ]


# ===========================================================================
# 4. K-Means — Unsupervised Land Cover Segmentation
# ===========================================================================

class KMeansLandCover:
    """
    K-Means clustering for unsupervised land cover classification.

    WHY K-MEANS:
      Before running the building segmentation model, it helps to know
      what the land cover looks like: urban, vegetation, water, bare soil.

      K-Means groups pixels by color similarity WITHOUT any labels.
      The result is a rough land-cover map that can:
        1. Focus the segmentation model on urban areas only
        2. Provide context for damage assessment (damage in urban vs rural)
        3. Estimate vegetation loss (pre vs post disaster)

    Algorithm:
      1. Initialize k cluster centroids randomly
      2. Assign each pixel to nearest centroid (by color distance)
      3. Recompute centroids as mean of assigned pixels
      4. Repeat 2-3 until convergence
      5. Result: each pixel gets a cluster label (0 to k-1)

    Typical clusters on satellite imagery:
      Cluster 0: Water (dark blue/black)
      Cluster 1: Vegetation (green)
      Cluster 2: Bare soil (brown)
      Cluster 3: Urban/buildings (gray/white)
      Cluster 4: Shadows (dark)
    """

    TYPICAL_LABELS = {
        "water": "dark pixels (low across all bands)",
        "vegetation": "high NIR, low red (vegetation reflects NIR)",
        "bare_soil": "moderate brightness, brownish",
        "urban": "high brightness, gray tones",
        "shadow": "very dark, near-zero",
    }

    def __init__(self, n_clusters: int = 5, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.model = None
        self.cluster_centers_ = None

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Segment image into land cover clusters.

        Args:
            image: (H, W, C) satellite image (uint8 or float)

        Returns:
            labels: (H, W) int array — cluster assignment per pixel
            info: dict with cluster centers and pixel counts
        """
        from sklearn.cluster import MiniBatchKMeans

        H, W, C = image.shape
        flat = image.reshape(-1, C).astype(np.float32)

        # MiniBatchKMeans is faster than KMeans for large images
        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            batch_size=1024,
            random_state=42,
        )
        cluster_labels = self.model.fit_predict(flat)
        self.cluster_centers_ = self.model.cluster_centers_

        labels_2d = cluster_labels.reshape(H, W)

        # Compute per-cluster stats
        info = {
            "n_clusters": self.n_clusters,
            "clusters": [],
        }
        for i in range(self.n_clusters):
            mask = labels_2d == i
            n_pixels = mask.sum()
            center = self.cluster_centers_[i]
            info["clusters"].append({
                "id": i,
                "center_rgb": center[:3].tolist(),
                "n_pixels": int(n_pixels),
                "coverage_pct": float(n_pixels / (H * W) * 100),
                "brightness": float(center.mean()),
            })

        # Sort by brightness (rough: dark=water, bright=urban)
        info["clusters"].sort(key=lambda c: c["brightness"])

        logger.info(
            f"K-Means: {self.n_clusters} clusters | "
            + " | ".join(f"C{c['id']}:{c['coverage_pct']:.1f}%" for c in info["clusters"])
        )
        return labels_2d, info

    def get_urban_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Quick urban area detection: cluster, then pick the brightest cluster
        (urban areas tend to be brightest in optical imagery).

        Returns (H, W) bool mask of likely urban areas.
        """
        labels, info = self.segment(image)
        brightest = info["clusters"][-1]["id"]   # sorted by brightness, last = brightest
        return labels == brightest


# ===========================================================================
# 5. MC Dropout — Bayesian Uncertainty Estimation
# ===========================================================================

class MCDropoutEstimator:
    """
    Monte Carlo Dropout for Bayesian uncertainty estimation.

    WHY THIS MATTERS FOR DISASTER RESPONSE:
      A standard model gives: "This building is destroyed (probability 0.87)"
      MC Dropout gives: "This building is destroyed (probability 0.87 ± 0.12)"

      The ± tells you HOW CERTAIN the model is. A rescue team should visit:
        "destroyed, 0.92 ± 0.03" BEFORE "destroyed, 0.65 ± 0.25"

      The first prediction is confident. The second is a guess.

    Algorithm:
      Normally, dropout is OFF during inference (deterministic prediction).
      MC Dropout keeps dropout ON and runs the model N times on the same input.

      Each run gives a slightly different prediction (because different neurons
      are randomly dropped each time).

      If all N runs agree → model is confident (low uncertainty)
      If runs disagree → model is uncertain (high uncertainty)

      Mathematically, this approximates Bayesian inference — it estimates
      the posterior distribution over predictions, not just a point estimate.

    Published: Gal & Ghahramani, 2016 — "Dropout as a Bayesian Approximation"
    """

    def __init__(self, model: 'torch.nn.Module', n_samples: int = 10):
        """
        Args:
            model: trained model WITH dropout layers
            n_samples: number of forward passes (more = better estimate, slower)
        """
        self.model = model
        self.n_samples = n_samples

    def _enable_dropout(self):
        """
        Enable dropout during inference.
        Normally model.eval() disables dropout. We selectively re-enable it.
        """
        import torch.nn as nn
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()   # re-enable dropout in this layer only

    def estimate_uncertainty(
        self, x: 'torch.Tensor'
    ) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
        """
        Run model N times with dropout enabled, compute statistics.

        Args:
            x: (B, C, H, W) input tensor

        Returns:
            mean_prediction: (B, classes, H, W) — average prediction (use this)
            uncertainty: (B, H, W) — per-pixel uncertainty (std of predictions)
            all_predictions: (N, B, classes, H, W) — all N forward passes
        """
        import torch

        self.model.eval()
        self._enable_dropout()

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(x)
                if output.shape[1] == 1:
                    probs = torch.sigmoid(output)
                else:
                    probs = torch.nn.functional.softmax(output, dim=1)
                predictions.append(probs)

        stacked = torch.stack(predictions, dim=0)   # (N, B, C, H, W)

        mean_pred   = stacked.mean(dim=0)            # average prediction
        uncertainty = stacked.std(dim=0).mean(dim=1)  # std across runs, avg across classes

        # Restore normal eval mode
        self.model.eval()

        return mean_pred, uncertainty, stacked

    def get_confidence_calibration(
        self, x: 'torch.Tensor', targets: 'torch.Tensor'
    ) -> Dict:
        """
        Check if the model's confidence matches its actual accuracy.

        A well-calibrated model:
          - When it says 90% confident → it's correct 90% of the time
          - When it says 50% confident → it's correct 50% of the time

        Returns calibration curve data for plotting.
        """
        import torch

        mean_pred, uncertainty, _ = self.estimate_uncertainty(x)

        if mean_pred.shape[1] == 1:
            # Binary segmentation
            confidences = mean_pred.squeeze(1).cpu().numpy().ravel()
            pred_classes = (confidences > 0.5).astype(int)
        else:
            confidences = mean_pred.max(dim=1).values.cpu().numpy().ravel()
            pred_classes = mean_pred.argmax(dim=1).cpu().numpy().ravel()

        targets_flat = targets.cpu().numpy().ravel()

        # Bin predictions by confidence level
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        calibration = []

        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_accuracy = (pred_classes[mask] == targets_flat[mask]).mean()
            bin_confidence = confidences[mask].mean()
            calibration.append({
                "confidence_bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                "mean_confidence": float(bin_confidence),
                "actual_accuracy": float(bin_accuracy),
                "n_samples": int(mask.sum()),
            })

        return {"calibration_curve": calibration}
