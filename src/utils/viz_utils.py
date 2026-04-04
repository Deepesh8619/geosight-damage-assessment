"""
Visualization utilities for satellite imagery and damage assessment outputs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
import cv2


DAMAGE_COLORS_RGB = {
    0: (0,   0,   0),     # background
    1: (0,   200, 0),     # no-damage  — green
    2: (255, 255, 0),     # minor      — yellow
    3: (255, 140, 0),     # major      — orange
    4: (220, 0,   0),     # destroyed  — red
}

DAMAGE_LABELS = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed",
}


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 HxWx3."""
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255)
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def colorize_damage_map(damage_map: np.ndarray) -> np.ndarray:
    """Convert (H, W) int damage map to (H, W, 3) uint8 RGB image."""
    rgb = np.zeros((*damage_map.shape, 3), dtype=np.uint8)
    for cls_idx, color in DAMAGE_COLORS_RGB.items():
        rgb[damage_map == cls_idx] = color
    return rgb


def plot_prediction_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    title: str = "Building Mask Overlay",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay binary building mask on top of the satellite image.

    Args:
        image: (H, W, 3) uint8 satellite image
        mask:  (H, W) float or binary mask
        alpha: transparency for mask overlay
        title: plot title
        save_path: if provided, saves figure here
    """
    image = _to_uint8(image)
    overlay = image.copy()
    mask_bool = (mask > 0.5) if mask.dtype == np.float32 else mask.astype(bool)

    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool].astype(np.float32) +
        alpha * np.array([0, 200, 100], dtype=np.float32)
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_damage_assessment(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    building_prob: np.ndarray,
    damage_map: np.ndarray,
    title: str = "Post-Disaster Building Damage Assessment",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create the main 4-panel assessment figure:
      [pre-image] [post-image] [building footprints] [damage classification]
    """
    pre_u8   = _to_uint8(pre_image)
    post_u8  = _to_uint8(post_image)
    seg_u8   = (building_prob * 255).astype(np.uint8)
    damage_rgb = colorize_damage_map(damage_map)

    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    panels = [
        (pre_u8,     "Pre-Disaster",           None),
        (post_u8,    "Post-Disaster",           None),
        (seg_u8,     "Building Footprints",     "Greens"),
        (damage_rgb, "Damage Classification",   None),
    ]

    for i, (img, subtitle, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(subtitle, fontsize=11)
        ax.axis("off")

    # Add legend for damage colours
    legend_patches = [
        mpatches.Patch(color=np.array(c) / 255, label=DAMAGE_LABELS[i])
        for i, c in DAMAGE_COLORS_RGB.items() if i > 0
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(legend_patches),
        fontsize=9,
        framealpha=0.8,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved assessment figure: {save_path}")

    return fig


def plot_class_distribution(
    stats: Dict,
    title: str = "Building Damage Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of area per damage class."""
    classes = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    areas   = [stats.get(c, {}).get("area_m2", 0) for c in classes]
    pcts    = [stats.get(c, {}).get("pct",     0) for c in classes]
    colors  = [np.array(DAMAGE_COLORS_RGB[i + 1]) / 255 for i in range(4)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Area bar
    bars = axes[0].bar(classes, areas, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Damage Class")
    axes[0].set_ylabel("Estimated Area (m²)")
    axes[0].set_title("Area by Damage Class")
    for bar, val in zip(bars, areas):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(areas) * 0.01,
            f"{val:,.0f} m²",
            ha="center", va="bottom", fontsize=8,
        )

    # Pie chart
    non_zero = [(pct, c, col) for pct, c, col in zip(pcts, classes, colors) if pct > 0]
    if non_zero:
        pcts_nz, classes_nz, colors_nz = zip(*non_zero)
        axes[1].pie(
            pcts_nz,
            labels=classes_nz,
            colors=colors_nz,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.85,
        )
    axes[1].set_title("Percentage Breakdown")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: List[Dict],
    val_metrics: List[Dict],
    metric_name: str = "iou",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot loss and metric curves for training diagnostics."""
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="Train", color="#2196F3")
    axes[0].plot(epochs, val_losses,   label="Val",   color="#F44336")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    train_m = [m.get(metric_name, 0) for m in train_metrics]
    val_m   = [m.get(metric_name, 0) for m in val_metrics]
    axes[1].plot(epochs, train_m, label="Train", color="#2196F3")
    axes[1].plot(epochs, val_m,   label="Val",   color="#F44336")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_name.upper())
    axes[1].set_title(f"{metric_name.upper()} Curves")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_leaflet_map(
    gdf,
    pre_image_path: Optional[str] = None,
    output_path: str = "data/outputs/map.html",
    center_lat: float = 0.0,
    center_lon: float = 0.0,
) -> str:
    """
    Generate an interactive Leaflet/Folium map showing building damage polygons.

    Returns the path to the saved HTML file.
    """
    import folium
    from folium.plugins import HeatMap

    center = [center_lat, center_lon]
    if not gdf.empty:
        centroid = gdf.to_crs("EPSG:4326").unary_union.centroid
        center   = [centroid.y, centroid.x]

    m = folium.Map(location=center, zoom_start=15, tiles="CartoDB dark_matter")

    COLOR_MAP = {
        1: "#00C800",   # green   — no-damage
        2: "#FFFF00",   # yellow  — minor
        3: "#FF8C00",   # orange  — major
        4: "#DC0000",   # red     — destroyed
    }

    for _, row in gdf.iterrows():
        geom  = row.geometry
        cls   = int(row.damage_class)
        color = COLOR_MAP.get(cls, "#888888")
        label = row.get("damage_label", str(cls))

        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor":   c,
                "color":       c,
                "weight":      1,
                "fillOpacity": 0.6,
            },
            tooltip=folium.Tooltip(f"Damage: {label}<br>Area: {row.get('area_m2', 0):.1f} m²"),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:rgba(0,0,0,0.8); padding:12px; border-radius:8px;
                color:white; font-size:12px;">
      <b>Damage Level</b><br>
      <span style="color:#00C800">&#9646;</span> No Damage<br>
      <span style="color:#FFFF00">&#9646;</span> Minor Damage<br>
      <span style="color:#FF8C00">&#9646;</span> Major Damage<br>
      <span style="color:#DC0000">&#9646;</span> Destroyed
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(output_path)
    print(f"Leaflet map saved: {output_path}")
    return output_path
