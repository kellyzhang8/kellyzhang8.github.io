"""Fast builder for GitHub Pages-ready raster PMTiles.

This creates one height PMTiles and one land-type PMTiles for each project-year.
The output is intentionally web-optimized: zoom levels 10, 11, and 12 are generated
from the full point grid, which keeps the GitHub package small and responsive.
For deeper inspection, add 14 to ZOOMS, but the output and build time will grow.
"""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from pmtiles.writer import write
from pmtiles.tile import zxy_to_tileid, TileType, Compression

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
DATA_DIR = ROOT / "data"
HEIGHT_PARQUET = Path("/mnt/data/Predicted_height_ALL_years_from_saved_model.parquet")
LANDTYPE_PARQUET = Path("/mnt/data/Predicted_landtype_ALL_years.parquet")
PROJECTS = ["FDL", "KBIC", "MBCI", "BN"]
YEARS = range(2017, 2025)
ZOOMS = [10, 11, 12, 13, 14]
TILE_SIZE = 256
HEIGHT_MIN = 0.0
HEIGHT_MAX = 25.0

LANDTYPE_COLORS = {
    11: (70, 107, 159, 230), 12: (210, 230, 255, 230),
    51: (178, 153, 104, 230), 52: (210, 180, 140, 230),
    61: (112, 168, 0, 230), 62: (38, 115, 0, 230),
    71: (209, 255, 115, 230), 72: (255, 255, 190, 230),
    81: (230, 230, 0, 230), 82: (255, 204, 102, 230),
    91: (181, 210, 156, 230), 92: (199, 233, 180, 230),
    121: (217, 146, 130, 230), 122: (235, 0, 0, 230),
    130: (179, 179, 179, 230), 140: (204, 204, 204, 230),
    181: (168, 112, 0, 230), 182: (215, 194, 158, 230),
    183: (163, 204, 81, 230), 192: (122, 142, 245, 230),
    200: (250, 250, 250, 230), 210: (20, 20, 20, 230),
}
UNKNOWN_COLOR = (150, 150, 150, 220)


def lonlat_to_tile_pixel(lon, lat, z):
    n = 2 ** z
    xf = (lon + 180.0) / 360.0 * n
    lat_rad = np.radians(lat)
    yf = (1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n
    tx = np.floor(xf).astype(np.int32)
    ty = np.floor(yf).astype(np.int32)
    tx = np.clip(tx, 0, n - 1)
    ty = np.clip(ty, 0, n - 1)
    px = np.floor((xf - tx) * TILE_SIZE).astype(np.int16)
    py = np.floor((yf - ty) * TILE_SIZE).astype(np.int16)
    px = np.clip(px, 0, TILE_SIZE - 1)
    py = np.clip(py, 0, TILE_SIZE - 1)
    return tx, ty, px, py


def png_bytes(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def color_height(avg, mask):
    v = np.clip((avg - HEIGHT_MIN) / (HEIGHT_MAX - HEIGHT_MIN), 0, 1)
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    left = v <= 0.5
    right = ~left
    t = np.zeros_like(v, dtype=np.float32)
    t[left] = v[left] / 0.5
    rgba[..., 0][left] = (44 + (255 - 44) * t[left]).astype(np.uint8)
    rgba[..., 1][left] = (123 + (255 - 123) * t[left]).astype(np.uint8)
    rgba[..., 2][left] = (182 + (191 - 182) * t[left]).astype(np.uint8)
    t[right] = (v[right] - 0.5) / 0.5
    rgba[..., 0][right] = (255 + (215 - 255) * t[right]).astype(np.uint8)
    rgba[..., 1][right] = (255 + (25 - 255) * t[right]).astype(np.uint8)
    rgba[..., 2][right] = (191 + (28 - 191) * t[right]).astype(np.uint8)
    rgba[..., 3] = np.where(mask, 230, 0).astype(np.uint8)
    return rgba


def color_land(arr):
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    for code, color in LANDTYPE_COLORS.items():
        m = arr == code
        if m.any():
            rgba[m] = color
    nonzero = arr != 0
    known = np.isin(arr, list(LANDTYPE_COLORS.keys()))
    rgba[nonzero & (~known)] = UNKNOWN_COLOR
    return rgba


def header_metadata(project, year, layer, bounds):
    minx, miny, maxx, maxy = bounds
    header = {
        "tile_type": TileType.PNG,
        "tile_compression": Compression.NONE,
        "min_lon_e7": int(minx * 1e7), "min_lat_e7": int(miny * 1e7),
        "max_lon_e7": int(maxx * 1e7), "max_lat_e7": int(maxy * 1e7),
        "center_lon_e7": int(((minx + maxx) / 2) * 1e7),
        "center_lat_e7": int(((miny + maxy) / 2) * 1e7),
        "center_zoom": 12,
    }
    metadata = {
        "name": f"{project} {year} {layer}",
        "format": "png",
        "bounds": f"{minx},{miny},{maxx},{maxy}",
        "center": f"{(minx + maxx) / 2},{(miny + maxy) / 2},12",
        "minzoom": str(min(ZOOMS)),
        "maxzoom": str(max(ZOOMS)),
        "type": "overlay",
    }
    return header, metadata


def read_project_year(project, year):
    filters = [("project", "==", project), ("year", "==", year)]
    h = pd.read_parquet(HEIGHT_PARQUET, columns=["x", "y", "height_m"], filters=filters, engine="pyarrow")
    l = pd.read_parquet(LANDTYPE_PARQUET, columns=["Land_type"], filters=filters, engine="pyarrow")
    if len(h) != len(l):
        raise ValueError(f"Row mismatch: {project} {year}")
    h["Land_type"] = l["Land_type"].to_numpy(dtype=np.int16, copy=False)
    return h


def write_project_year(project, year):
    print(f"{project} {year}: reading", flush=True)
    df = read_project_year(project, year)
    lon = df["x"].to_numpy(dtype=np.float64, copy=False)
    lat = df["y"].to_numpy(dtype=np.float64, copy=False)
    height = df["height_m"].to_numpy(dtype=np.float32, copy=False)
    land = df["Land_type"].to_numpy(dtype=np.int16, copy=False)
    bounds = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))

    DATA_DIR.mkdir(exist_ok=True)
    out_h = DATA_DIR / f"{project}_{year}_height.pmtiles"
    out_l = DATA_DIR / f"{project}_{year}_landtype.pmtiles"
    hh, hm = header_metadata(project, year, "height", bounds)
    lh, lm = header_metadata(project, year, "landtype", bounds)

    tile_counter = 0
    with write(str(out_h)) as wh, write(str(out_l)) as wl:
        for z in ZOOMS:
            tx, ty, px, py = lonlat_to_tile_pixel(lon, lat, z)
            # Stable ordering by tile id, then process contiguous groups.
            tile_key = tx.astype(np.int64) * (2 ** z) + ty.astype(np.int64)
            order = np.argsort(tile_key, kind="stable")
            sorted_key = tile_key[order]
            starts = np.r_[0, np.nonzero(sorted_key[1:] != sorted_key[:-1])[0] + 1]
            ends = np.r_[starts[1:], len(order)]
            for s, e in zip(starts, ends):
                idx = order[s:e]
                x = int(tx[idx[0]]); y = int(ty[idx[0]])
                # height tile
                sums = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                counts = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint16)
                np.add.at(sums, (py[idx], px[idx]), height[idx])
                np.add.at(counts, (py[idx], px[idx]), 1)
                mask = counts > 0
                avg = np.zeros_like(sums)
                avg[mask] = sums[mask] / counts[mask]
                wh.write_tile(zxy_to_tileid(z, x, y), png_bytes(color_height(avg, mask)))
                # land tile: last value per pixel; sufficient for visual browsing.
                arr = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int16)
                arr[py[idx], px[idx]] = land[idx]
                wl.write_tile(zxy_to_tileid(z, x, y), png_bytes(color_land(arr)))
                tile_counter += 1
        wh.finalize(hh, hm)
        wl.finalize(lh, lm)
    print(f"{project} {year}: wrote {tile_counter} tiles x 2 layers | {out_h.stat().st_size/1048576:.2f} MB + {out_l.stat().st_size/1048576:.2f} MB", flush=True)


def main():
    for project in PROJECTS:
        for year in YEARS:
            write_project_year(project, year)

if __name__ == "__main__":
    main()
