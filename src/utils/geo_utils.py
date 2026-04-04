"""
Geospatial utility functions.
Bridges between raster (numpy/GeoTIFF) and vector (GeoJSON/GeoDataFrame) worlds.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.crs import CRS
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from loguru import logger


DAMAGE_CLASS_NAMES = {
    0: "background",
    1: "no-damage",
    2: "minor-damage",
    3: "major-damage",
    4: "destroyed",
}


def rasterize_polygons(
    geodataframe: gpd.GeoDataFrame,
    raster_shape: Tuple[int, int],
    transform: Affine,
    attribute_col: str = "damage_class",
    default_value: int = 1,
) -> np.ndarray:
    """
    Burn GeoDataFrame vector polygons into a raster mask.

    Args:
        geodataframe: GeoDataFrame with polygon geometries
        raster_shape: (height, width) of output raster
        transform: affine transform of the output raster
        attribute_col: column in GDF to use as burn value (None = use default_value)
        default_value: fallback burn value when attribute_col is None or missing

    Returns:
        mask: (H, W) uint8 raster
    """
    if geodataframe.empty:
        return np.zeros(raster_shape, dtype=np.uint8)

    if attribute_col and attribute_col in geodataframe.columns:
        shapes = [
            (geom, int(val))
            for geom, val in zip(geodataframe.geometry, geodataframe[attribute_col])
            if geom is not None and not geom.is_empty
        ]
    else:
        shapes = [
            (geom, default_value)
            for geom in geodataframe.geometry
            if geom is not None and not geom.is_empty
        ]

    if not shapes:
        return np.zeros(raster_shape, dtype=np.uint8)

    mask = rasterio.features.rasterize(
        shapes,
        out_shape=raster_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask


def polygonize_damage_map(
    damage_map: np.ndarray,
    transform: Affine,
    crs: Union[str, CRS] = "EPSG:4326",
    min_area_pixels: int = 10,
    building_mask: Optional[np.ndarray] = None,
) -> gpd.GeoDataFrame:
    """
    Convert a per-pixel damage classification raster into a GeoDataFrame
    of building polygons with damage attributes.

    Args:
        damage_map:       (H, W) int array with class indices 0-4
        transform:        affine transform for the damage_map
        crs:              coordinate reference system of the output
        min_area_pixels:  discard polygons smaller than this
        building_mask:    optional (H, W) bool array; if provided, non-building
                          pixels are excluded before polygonizing

    Returns:
        GeoDataFrame with columns: geometry, damage_class, damage_label, area_m2
    """
    if building_mask is not None:
        damage_map = damage_map.copy()
        damage_map[~building_mask] = 0

    records = []
    for class_idx in range(1, 5):   # skip background (0)
        class_mask = (damage_map == class_idx).astype(np.uint8)
        if class_mask.sum() == 0:
            continue

        for geom_dict, value in rasterio.features.shapes(
            class_mask, mask=class_mask, transform=transform
        ):
            geom = shape(geom_dict)
            if geom.area < min_area_pixels:
                continue
            records.append({
                "geometry":     geom,
                "damage_class": class_idx,
                "damage_label": DAMAGE_CLASS_NAMES[class_idx],
                "area_m2":      geom.area,
            })

    if not records:
        logger.warning("polygonize_damage_map: no buildings found in damage map.")
        return gpd.GeoDataFrame(
            columns=["geometry", "damage_class", "damage_label", "area_m2"],
            crs=crs,
        )

    gdf = gpd.GeoDataFrame(records, crs=crs)
    logger.info(
        f"Polygonized {len(gdf)} building polygons | "
        f"class distribution: {gdf['damage_label'].value_counts().to_dict()}"
    )
    return gdf


def save_geotiff(
    data: np.ndarray,
    output_path: str,
    transform: Affine,
    crs: Union[str, CRS] = "EPSG:4326",
    nodata: Optional[float] = None,
    compress: str = "lzw",
) -> None:
    """
    Save a numpy array as a GeoTIFF.

    Args:
        data:         (H, W) or (C, H, W) float32 / int array
        output_path:  destination file path
        transform:    affine transform
        crs:          coordinate reference system
        nodata:       nodata value
        compress:     compression algorithm
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, ...]   # add band dim → (1, H, W)

    n_bands, h, w = data.shape
    dtype = str(data.dtype)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=h, width=w,
        count=n_bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress=compress,
    ) as dst:
        dst.write(data)

    logger.info(f"Saved GeoTIFF: {output_path} ({w}x{h}, {n_bands} band(s))")


def reproject_to_wgs84(
    data: np.ndarray,
    src_transform: Affine,
    src_crs: Union[str, CRS],
    target_resolution: float = 0.0001,   # degrees (~10m at equator)
) -> Tuple[np.ndarray, Affine]:
    """
    Reproject raster data to WGS84 (EPSG:4326).

    Returns:
        reprojected_data: (H', W') reprojected array
        dst_transform: affine transform for reprojected data
    """
    dst_crs = CRS.from_epsg(4326)
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    transform_out, w, h = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs, data.shape[2], data.shape[1],
        left=src_transform.c,
        top=src_transform.f,
        right=src_transform.c + src_transform.a * data.shape[2],
        bottom=src_transform.f + src_transform.e * data.shape[1],
        resolution=target_resolution,
    )

    reprojected = np.zeros((data.shape[0], h, w), dtype=data.dtype)
    rasterio.warp.reproject(
        source=data,
        destination=reprojected,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=transform_out,
        dst_crs=dst_crs,
        resampling=rasterio.warp.Resampling.nearest,
    )

    return reprojected.squeeze(), transform_out


def compute_geotiff_stats(path: str) -> Dict:
    """
    Compute per-band statistics for a GeoTIFF file.
    Useful for validating data before feeding into the pipeline.
    """
    with rasterio.open(path) as src:
        stats = {
            "path":   path,
            "crs":    str(src.crs),
            "bounds": dict(zip(["left", "bottom", "right", "top"], src.bounds)),
            "width":  src.width,
            "height": src.height,
            "count":  src.count,
            "dtype":  src.meta["dtype"],
            "bands":  [],
        }
        for i in range(1, src.count + 1):
            band = src.read(i).astype(np.float64)
            valid = band[band != src.nodata] if src.nodata is not None else band.ravel()
            stats["bands"].append({
                "band":   i,
                "min":    float(valid.min()),
                "max":    float(valid.max()),
                "mean":   float(valid.mean()),
                "std":    float(valid.std()),
                "nodata": float(src.nodata) if src.nodata is not None else None,
            })
    return stats


def clip_raster_to_bounds(
    src_path: str,
    dst_path: str,
    bounds: Tuple[float, float, float, float],   # (minx, miny, maxx, maxy)
    crs: Union[str, CRS] = "EPSG:4326",
) -> None:
    """Clip a GeoTIFF to a bounding box and save."""
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box

    geom = gpd.GeoDataFrame(
        {"geometry": [box(*bounds)]}, crs=crs
    )

    with rasterio.open(src_path) as src:
        if str(geom.crs) != str(src.crs):
            geom = geom.to_crs(src.crs)

        out_image, out_transform = rio_mask(src, geom.geometry, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({
        "height":    out_image.shape[1],
        "width":     out_image.shape[2],
        "transform": out_transform,
    })

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **out_meta) as dst:
        dst.write(out_image)

    logger.info(f"Clipped raster saved to: {dst_path}")
