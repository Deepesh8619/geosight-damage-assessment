"""
Spatial Intelligence Module.

WHY THIS EXISTS:
----------------
Disasters are not random. They have SPATIAL STRUCTURE:

  - Earthquakes radiate outward from an epicentre — damage forms concentric rings
  - Floods follow elevation — damage follows river valleys and low-lying areas
  - Hurricanes have a spiral damage pattern with an eyewall of max destruction
  - Wildfires advance along a fire-line front, driven by wind and fuel
  - Tsunamis create a coastal inundation gradient — damage decreases with distance from shore
  - Tornadoes leave a narrow linear track of total destruction

This module analyzes the SPATIAL PATTERN of damage to:

  1. Detect damage clusters and hotspots
  2. Estimate disaster epicentre / origin point
  3. Compute directional damage gradients
  4. Measure damage radial profile (distance decay)
  5. Identify the "destruction corridor" or "impact zone boundary"

These spatial insights help responders:
  - Predict where UNSURVEYED areas are likely damaged (extrapolation)
  - Understand disaster dynamics (what direction did it come from?)
  - Plan logistics (where to set up base camps relative to damage)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from loguru import logger


class SpatialDamageAnalyzer:
    """
    Analyzes the spatial distribution and patterns of damage.
    All coordinates are in pixel space; multiply by GSD for meters.
    """

    def __init__(self, pixel_gsd_m: float = 0.5):
        """
        Args:
            pixel_gsd_m: ground sampling distance in meters/pixel
        """
        self.gsd = pixel_gsd_m

    def full_analysis(
        self,
        damage_map: np.ndarray,
        building_mask: np.ndarray,
    ) -> Dict:
        """
        Run all spatial analyses and return a comprehensive report.

        Args:
            damage_map:    (H, W) int array, classes 0-4
            building_mask: (H, W) bool array

        Returns:
            Dict with all spatial analysis results
        """
        severe_mask = ((damage_map >= 3) & building_mask)

        report = {}

        # 1. Damage epicentre
        report["epicentre"] = self.find_damage_epicentre(damage_map, building_mask)

        # 2. Damage clusters
        report["clusters"] = self.find_damage_clusters(severe_mask)

        # 3. Directional gradient
        report["gradient"] = self.compute_damage_gradient(damage_map, building_mask)

        # 4. Radial profile from epicentre
        if report["epicentre"]["center"] is not None:
            cy, cx = report["epicentre"]["center"]
            report["radial_profile"] = self.compute_radial_profile(
                damage_map, building_mask, center=(cy, cx)
            )
        else:
            report["radial_profile"] = {}

        # 5. Damage boundary
        report["boundary"] = self.find_damage_boundary(severe_mask)

        # 6. Spread metrics
        report["spread"] = self.compute_spread_metrics(damage_map, building_mask)

        return report

    def find_damage_epicentre(
        self,
        damage_map: np.ndarray,
        building_mask: np.ndarray,
    ) -> Dict:
        """
        Find the center of mass of severe damage (the likely epicentre/ground zero).

        For earthquakes, this approximates the surface rupture point.
        For hurricanes, this approximates the eyewall landing point.
        For floods, this is the area of deepest inundation.

        Returns:
            Dict with center coords, radius of severe damage, and confidence
        """
        severe_mask = ((damage_map >= 3) & building_mask).astype(np.float32)

        if severe_mask.sum() == 0:
            return {"center": None, "radius_m": 0, "confidence": 0.0}

        # Weight by severity: destroyed (4) counts more than major (3)
        weight_map = np.zeros_like(damage_map, dtype=np.float32)
        weight_map[(damage_map == 3) & building_mask] = 1.0
        weight_map[(damage_map == 4) & building_mask] = 3.0

        # Center of mass
        cy, cx = ndimage.center_of_mass(weight_map)

        # Radius: standard deviation of distance from center
        ys, xs = np.where(weight_map > 0)
        if len(ys) > 0:
            distances = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2) * self.gsd
            radius = float(np.std(distances))
            max_radius = float(np.max(distances))
        else:
            radius = 0.0
            max_radius = 0.0

        # Confidence: how concentrated is the damage? (lower std = more certain)
        if max_radius > 0:
            concentration = 1.0 - min(1.0, radius / max_radius)
        else:
            concentration = 0.0

        return {
            "center":            (int(cy), int(cx)),
            "center_m":          (cy * self.gsd, cx * self.gsd),
            "radius_m":          round(radius, 1),
            "max_radius_m":      round(max_radius, 1),
            "concentration":     round(concentration, 3),
        }

    def find_damage_clusters(
        self,
        severe_mask: np.ndarray,
        min_cluster_pixels: int = 50,
    ) -> List[Dict]:
        """
        Identify distinct clusters of severe damage.

        Multiple clusters suggest:
          - Earthquake aftershocks (multiple epicentres)
          - Tornado touchdowns (skip pattern)
          - Flood pockets (low-lying areas)

        Returns list of clusters sorted by severity (largest first).
        """
        if severe_mask.sum() == 0:
            return []

        # Dilate slightly to merge nearby damage into single clusters
        structure = np.ones((5, 5))
        dilated = ndimage.binary_dilation(severe_mask, structure=structure, iterations=2)
        labeled, n_clusters = ndimage.label(dilated)

        clusters = []
        for i in range(1, n_clusters + 1):
            cluster_pixels = (labeled == i)
            actual_damage  = cluster_pixels & severe_mask
            n_pixels       = actual_damage.sum()

            if n_pixels < min_cluster_pixels:
                continue

            cy, cx = ndimage.center_of_mass(actual_damage.astype(float))
            ys, xs = np.where(actual_damage)

            # Bounding box
            r_min, r_max = int(ys.min()), int(ys.max())
            c_min, c_max = int(xs.min()), int(xs.max())

            clusters.append({
                "id":       i,
                "center":   (int(cy), int(cx)),
                "n_pixels": int(n_pixels),
                "area_m2":  round(float(n_pixels) * self.gsd ** 2, 1),
                "bbox":     (r_min, c_min, r_max, c_max),
                "extent_m": round(max(r_max - r_min, c_max - c_min) * self.gsd, 1),
            })

        clusters.sort(key=lambda c: c["n_pixels"], reverse=True)
        logger.info(f"Found {len(clusters)} damage clusters")
        return clusters

    def compute_damage_gradient(
        self,
        damage_map: np.ndarray,
        building_mask: np.ndarray,
        n_sectors: int = 8,
    ) -> Dict:
        """
        Compute directional damage gradient.

        Divides the scene into angular sectors from the center and
        computes average damage severity per sector.

        High gradient in one direction suggests:
          - Wind-driven damage (hurricane, tornado)
          - Flow-direction (flood, tsunami)
          - Fire front advancement (wildfire)

        Uniform gradient suggests:
          - Earthquake (radial pattern)
        """
        H, W = damage_map.shape
        cy, cx = H / 2, W / 2

        # Create severity map (0-4 only on buildings)
        severity = damage_map.astype(np.float32)
        severity[~building_mask] = np.nan

        # Create angle map from center
        ys, xs = np.mgrid[0:H, 0:W]
        angles = np.arctan2(ys - cy, xs - cx)  # -pi to pi
        angles = (angles + 2 * np.pi) % (2 * np.pi)  # 0 to 2pi

        sector_size = 2 * np.pi / n_sectors
        sectors = {}
        sector_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

        for i in range(n_sectors):
            a_start = i * sector_size
            a_end   = (i + 1) * sector_size
            sector_mask = (angles >= a_start) & (angles < a_end) & building_mask

            if sector_mask.sum() > 0:
                sector_vals  = severity[sector_mask]
                valid_vals   = sector_vals[~np.isnan(sector_vals)]
                avg_severity = float(valid_vals.mean()) if len(valid_vals) > 0 else 0.0
            else:
                avg_severity = 0.0

            label = sector_labels[i] if i < len(sector_labels) else f"S{i}"
            sectors[label] = round(avg_severity, 3)

        # Find dominant direction (highest damage)
        if sectors:
            max_sector    = max(sectors, key=sectors.get)
            min_sector    = min(sectors, key=sectors.get)
            max_severity  = sectors[max_sector]
            min_severity  = sectors[min_sector]
            gradient_mag  = max_severity - min_severity
            is_directional = gradient_mag > 0.5   # threshold for "clearly directional"
        else:
            max_sector = min_sector = "N"
            gradient_mag = 0.0
            is_directional = False

        return {
            "sectors":         sectors,
            "dominant_direction": max_sector,
            "weakest_direction":  min_sector,
            "gradient_magnitude": round(gradient_mag, 3),
            "is_directional":     is_directional,
            "pattern": "directional" if is_directional else "radial/uniform",
        }

    def compute_radial_profile(
        self,
        damage_map: np.ndarray,
        building_mask: np.ndarray,
        center: Tuple[int, int] = None,
        n_rings: int = 10,
    ) -> Dict:
        """
        Compute how damage severity changes with distance from the epicentre.

        A steep decay suggests a point-source event (earthquake, explosion).
        A flat profile suggests a wide-area event (hurricane, flood).

        Returns:
            Dict with distance bins and average severity per ring
        """
        H, W = damage_map.shape
        if center is None:
            center = (H // 2, W // 2)

        cy, cx = center
        ys, xs = np.mgrid[0:H, 0:W]
        distances = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2) * self.gsd

        max_dist = distances[building_mask].max() if building_mask.any() else 1.0
        ring_width = max_dist / n_rings

        rings = []
        for i in range(n_rings):
            d_min = i * ring_width
            d_max = (i + 1) * ring_width
            ring_mask = (distances >= d_min) & (distances < d_max) & building_mask

            if ring_mask.sum() > 0:
                avg_sev = float(damage_map[ring_mask].mean())
                n_buildings = int(ring_mask.sum())
            else:
                avg_sev = 0.0
                n_buildings = 0

            rings.append({
                "distance_min_m": round(d_min, 1),
                "distance_max_m": round(d_max, 1),
                "avg_severity":   round(avg_sev, 3),
                "n_pixels":       n_buildings,
            })

        # Decay rate: linear fit of severity vs distance
        dists = np.array([r["distance_max_m"] for r in rings])
        sevs  = np.array([r["avg_severity"]   for r in rings])
        valid = sevs > 0
        if valid.sum() >= 2:
            coeffs   = np.polyfit(dists[valid], sevs[valid], 1)
            decay_rate = float(coeffs[0])  # negative = damage decreases with distance
        else:
            decay_rate = 0.0

        return {
            "rings":      rings,
            "decay_rate": round(decay_rate, 6),
            "pattern":    "point-source (rapid decay)" if decay_rate < -0.001 else "wide-area (slow/no decay)",
        }

    def find_damage_boundary(
        self,
        severe_mask: np.ndarray,
    ) -> Dict:
        """
        Find the boundary/perimeter of the damaged area.

        This is the "impact zone" edge — the transition between
        damaged and undamaged territory. Useful for:
          - Setting up evacuation perimeters
          - Positioning relief supply depots at the boundary
          - Estimating total affected area
        """
        if severe_mask.sum() == 0:
            return {"perimeter_m": 0, "area_m2": 0, "compactness": 0}

        # Fill holes and smooth to get the damage envelope
        filled = ndimage.binary_fill_holes(severe_mask)
        filled = ndimage.binary_dilation(filled, iterations=3)
        filled = ndimage.binary_erosion(filled, iterations=3)

        # Perimeter: count edge pixels
        eroded    = ndimage.binary_erosion(filled, iterations=1)
        perimeter = filled.astype(int) - eroded.astype(int)
        perimeter_pixels = perimeter.sum()

        area_pixels = filled.sum()

        # Compactness: how circular is the damage zone?
        # Circle has compactness = 1; elongated/irregular < 1
        if perimeter_pixels > 0:
            compactness = (4 * np.pi * area_pixels) / (perimeter_pixels ** 2)
        else:
            compactness = 0.0

        return {
            "perimeter_m":  round(perimeter_pixels * self.gsd, 1),
            "area_m2":      round(area_pixels * self.gsd ** 2, 1),
            "compactness":  round(float(compactness), 3),
            "shape_hint":   "circular (earthquake/explosion)" if compactness > 0.6
                            else "elongated (flood/tornado/fire-line)" if compactness > 0.2
                            else "irregular (complex event)",
        }

    def compute_spread_metrics(
        self,
        damage_map: np.ndarray,
        building_mask: np.ndarray,
    ) -> Dict:
        """
        Compute metrics about how damage is spatially distributed.

        Returns:
          - dispersion: how scattered is the damage (0=concentrated, 1=dispersed)
          - coverage: what % of the buildable area has any damage
          - intensity_gradient: how sharply damage transitions occur
        """
        any_damage = (damage_map >= 2) & building_mask

        if any_damage.sum() == 0 or building_mask.sum() == 0:
            return {"dispersion": 0, "coverage": 0, "intensity_variance": 0}

        # Coverage
        coverage = float(any_damage.sum()) / float(building_mask.sum())

        # Dispersion: ratio of convex hull area to actual damaged area
        ys, xs = np.where(any_damage)
        if len(ys) >= 3:
            from scipy.spatial import ConvexHull
            try:
                points = np.column_stack([xs, ys])
                hull = ConvexHull(points)
                hull_area = hull.volume  # in 2D, .volume gives area
                actual_area = any_damage.sum()
                dispersion = 1.0 - min(1.0, actual_area / (hull_area + 1e-8))
            except Exception:
                dispersion = 0.5
        else:
            dispersion = 0.0

        # Intensity variance: how much does damage severity vary spatially?
        severity_values = damage_map[building_mask].astype(np.float32)
        intensity_variance = float(np.var(severity_values))

        return {
            "dispersion":         round(dispersion, 3),
            "coverage":           round(coverage, 3),
            "intensity_variance": round(intensity_variance, 3),
        }
