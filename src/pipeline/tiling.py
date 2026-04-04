"""
Scalable Raster Tiling Pipeline (Project 7 component).
Uses Dask for parallel, out-of-core processing of large satellite scenes.
Handles overlapping tiles, edge padding, and tile reassembly.
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
import dask
import dask.array as da
from dask import delayed
from loguru import logger


class TileSpec:
    """Metadata for a single tile."""

    def __init__(
        self,
        col_off: int,
        row_off: int,
        width: int,
        height: int,
        tile_row: int,
        tile_col: int,
        pad_right: int = 0,
        pad_bottom: int = 0,
    ):
        self.col_off    = col_off
        self.row_off    = row_off
        self.width      = width
        self.height     = height
        self.tile_row   = tile_row
        self.tile_col   = tile_col
        self.pad_right  = pad_right
        self.pad_bottom = pad_bottom

    @property
    def window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)

    def __repr__(self):
        return (
            f"TileSpec(r={self.tile_row}, c={self.tile_col}, "
            f"offset=({self.col_off},{self.row_off}), size=({self.width}x{self.height}))"
        )


class RasterTiler:
    """
    Tiles a raster into overlapping patches for inference or training,
    then reassembles prediction tiles back into a full scene.

    Uses Dask for parallelism so multi-GB scenes can be processed
    without loading them entirely into RAM.
    """

    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 64,
        num_workers: int = 4,
    ):
        self.tile_size   = tile_size
        self.overlap     = overlap
        self.stride      = tile_size - overlap
        self.num_workers = num_workers

    # ------------------------------------------------------------------
    # Tile generation
    # ------------------------------------------------------------------

    def compute_tile_specs(self, raster_width: int, raster_height: int) -> List[TileSpec]:
        """
        Compute all tile specs for a raster of given dimensions.
        Edge tiles are padded to tile_size.
        """
        specs = []
        row, row_idx = 0, 0

        while row < raster_height:
            col, col_idx = 0, 0
            while col < raster_width:
                actual_w = min(self.tile_size, raster_width  - col)
                actual_h = min(self.tile_size, raster_height - row)
                pad_r    = self.tile_size - actual_w
                pad_b    = self.tile_size - actual_h

                specs.append(TileSpec(
                    col_off=col, row_off=row,
                    width=actual_w, height=actual_h,
                    tile_row=row_idx, tile_col=col_idx,
                    pad_right=pad_r, pad_bottom=pad_b,
                ))
                col += self.stride
                col_idx += 1
            row += self.stride
            row_idx += 1

        logger.debug(
            f"Computed {len(specs)} tiles for {raster_width}x{raster_height} raster "
            f"(tile={self.tile_size}, overlap={self.overlap})"
        )
        return specs

    def tile_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[TileSpec]]:
        """
        Tile a numpy image array (H, W, C) or (H, W) into patches.

        Returns:
            tiles: list of (tile_size, tile_size, C) patches
            specs: list of TileSpec corresponding to each tile
        """
        if image.ndim == 2:
            h, w = image.shape
            is_2d = True
        else:
            h, w, _ = image.shape
            is_2d = False

        specs = self.compute_tile_specs(w, h)
        tiles = []

        for spec in specs:
            if is_2d:
                patch = image[spec.row_off : spec.row_off + spec.height,
                              spec.col_off : spec.col_off + spec.width]
                if spec.pad_right > 0 or spec.pad_bottom > 0:
                    patch = np.pad(patch, ((0, spec.pad_bottom), (0, spec.pad_right)))
            else:
                patch = image[spec.row_off : spec.row_off + spec.height,
                              spec.col_off : spec.col_off + spec.width, :]
                if spec.pad_right > 0 or spec.pad_bottom > 0:
                    patch = np.pad(patch, ((0, spec.pad_bottom), (0, spec.pad_right), (0, 0)))

            tiles.append(patch)

        return tiles, specs

    # ------------------------------------------------------------------
    # Parallel tiling via Dask (for large GeoTIFF files on disk)
    # ------------------------------------------------------------------

    def tile_geotiff(
        self,
        raster_path: str,
        bands: Optional[List[int]] = None,
        normalize: bool = True,
    ) -> Tuple[da.Array, List[TileSpec], Dict]:
        """
        Lazily tile a large GeoTIFF using Dask without loading into RAM.

        Returns:
            dask_array: shape (N_tiles, tile_size, tile_size, C), float32
            specs: tile metadata list
            scene_meta: rasterio metadata dict
        """
        with rasterio.open(raster_path) as src:
            meta    = src.meta.copy()
            height  = src.height
            width   = src.width
            n_bands = len(bands) if bands else src.count

        specs = self.compute_tile_specs(width, height)

        @delayed
        def _read_tile(spec: TileSpec) -> np.ndarray:
            with rasterio.open(raster_path) as src:
                band_list = bands or list(range(1, src.count + 1))
                data = src.read(band_list, window=spec.window)  # (C, H, W)

            data = data.astype(np.float32)

            if normalize:
                for i in range(data.shape[0]):
                    b = data[i]
                    p2, p98 = np.percentile(b, [2, 98])
                    data[i] = np.clip((b - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)

            # Transpose to (H, W, C) and pad if edge tile
            data = np.transpose(data, (1, 2, 0))
            if spec.pad_bottom > 0 or spec.pad_right > 0:
                data = np.pad(data, ((0, spec.pad_bottom), (0, spec.pad_right), (0, 0)))

            return data

        lazy_tiles = [
            da.from_delayed(
                _read_tile(spec),
                shape=(self.tile_size, self.tile_size, n_bands),
                dtype=np.float32,
            )
            for spec in specs
        ]

        dask_array = da.stack(lazy_tiles, axis=0)  # (N, H, W, C)
        logger.info(
            f"Lazy-tiled {Path(raster_path).name}: "
            f"{len(specs)} tiles, shape={dask_array.shape}"
        )
        return dask_array, specs, meta

    # ------------------------------------------------------------------
    # Tile reassembly (stitching predictions back to full image)
    # ------------------------------------------------------------------

    def reassemble(
        self,
        tiles: List[np.ndarray],
        specs: List[TileSpec],
        raster_height: int,
        raster_width: int,
        n_classes: int = 1,
    ) -> np.ndarray:
        """
        Merge overlapping prediction tiles back into a single canvas
        using weighted averaging in overlap regions.

        Args:
            tiles: list of (tile_size, tile_size) or (tile_size, tile_size, C) arrays
            specs: corresponding tile specs
            raster_height, raster_width: full scene dimensions
            n_classes: number of output channels

        Returns:
            output: (H, W) or (H, W, C) merged prediction array
        """
        if n_classes == 1:
            canvas  = np.zeros((raster_height, raster_width),      dtype=np.float32)
            weights = np.zeros((raster_height, raster_width),      dtype=np.float32)
        else:
            canvas  = np.zeros((raster_height, raster_width, n_classes), dtype=np.float32)
            weights = np.zeros((raster_height, raster_width),            dtype=np.float32)

        # Build a weight map that down-weights tile edges (cosine window)
        weight_patch = self._cosine_window(self.tile_size)

        for tile, spec in zip(tiles, specs):
            r0 = spec.row_off
            c0 = spec.col_off
            r1 = min(r0 + spec.height, raster_height)
            c1 = min(c0 + spec.width,  raster_width)
            th = r1 - r0
            tw = c1 - c0

            w = weight_patch[:th, :tw]

            if n_classes == 1:
                canvas[r0:r1, c0:c1]  += tile[:th, :tw] * w
            else:
                canvas[r0:r1, c0:c1]  += tile[:th, :tw, :] * w[:, :, np.newaxis]
            weights[r0:r1, c0:c1] += w

        # Normalize by accumulated weights
        weights = np.where(weights == 0, 1e-8, weights)
        if n_classes == 1:
            return canvas / weights
        else:
            return canvas / weights[:, :, np.newaxis]

    @staticmethod
    def _cosine_window(size: int) -> np.ndarray:
        """1D Hann window extended to 2D for smooth tile blending."""
        win1d = np.hanning(size).astype(np.float32)
        win2d = np.outer(win1d, win1d)
        return win2d

    # ------------------------------------------------------------------
    # Utility: process all tiles in parallel batches with Dask
    # ------------------------------------------------------------------

    def process_tiles_parallel(
        self,
        tiles: List[np.ndarray],
        fn,
        batch_size: int = 8,
    ) -> List[np.ndarray]:
        """
        Apply a function (e.g. model inference) to tiles in parallel batches.

        fn: callable that takes a list of np.ndarray and returns a list of np.ndarray
        """
        results = []
        for i in range(0, len(tiles), batch_size):
            batch   = tiles[i : i + batch_size]
            result  = fn(batch)
            results.extend(result)
        return results
