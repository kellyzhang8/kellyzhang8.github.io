"""Rebuild the visible height raster PMTiles with the same viridis-style colors
used in the manuscript map code.

This overwrites files such as:
    data/MBCI_2024_height.pmtiles

Run from the root of height_landtype_webmap_10m/:
    source .venv/bin/activate
    python scripts/build_height_viridis_pmtiles.py --all --force

Test one project-year first:
    python scripts/build_height_viridis_pmtiles.py --project MBCI --year 2024 --force
"""
from __future__ import annotations

import argparse
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
HEIGHT_PARQUET = RAW_DIR / "Predicted_height_ALL_years_from_saved_model.parquet"

PROJECTS = ["FDL", "KBIC", "MBCI", "BN"]
YEARS = list(range(2017, 2025))
ZOOMS = [10, 11, 12, 13, 14]
TILE_SIZE = 256
HEIGHT_MIN = 0.0
HEIGHT_MAX = 20.0

# Viridis anchor colors, matching the map code's viridis-style palette.
VIRIDIS_STOPS = np.array([
    [68, 1, 84],      # #440154
    [59, 82, 139],    # #3B528B
    [33, 145, 140],   # #21918C
    [94, 201, 98],    # #5EC962
    [253, 231, 37],   # #FDE725
], dtype=np.float32)


def lonlat_to_tile_pixel(lon: np.ndarray, lat: np.ndarray, z: int):
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


def png_bytes(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def color_height_viridis(avg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.clip((avg - HEIGHT_MIN) / (HEIGHT_MAX - HEIGHT_MIN), 0, 1)
    pos = v * (len(VIRIDIS_STOPS) - 1)
    i0 = np.floor(pos).astype(np.int16)
    i1 = np.clip(i0 + 1, 0, len(VIRIDIS_STOPS) - 1)
    t = (pos - i0)[..., None].astype(np.float32)
    rgb = VIRIDIS_STOPS[i0] * (1.0 - t) + VIRIDIS_STOPS[i1] * t

    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(mask, 230, 0).astype(np.uint8)
    return rgba


def header_metadata(project: str, year: int, bounds: tuple[float, float, float, float]):
    minx, miny, maxx, maxy = bounds
    header = {
        "tile_type": TileType.PNG,
        "tile_compression": Compression.NONE,
        "min_lon_e7": int(minx * 1e7),
        "min_lat_e7": int(miny * 1e7),
        "max_lon_e7": int(maxx * 1e7),
        "max_lat_e7": int(maxy * 1e7),
        "center_lon_e7": int(((minx + maxx) / 2) * 1e7),
        "center_lat_e7": int(((miny + maxy) / 2) * 1e7),
        "center_zoom": 12,
    }
    metadata = {
        "name": f"{project} {year} viridis height",
        "format": "png",
        "bounds": f"{minx},{miny},{maxx},{maxy}",
        "center": f"{(minx + maxx) / 2},{(miny + maxy) / 2},12",
        "minzoom": str(min(ZOOMS)),
        "maxzoom": str(max(ZOOMS)),
        "type": "overlay",
        "height_palette": "viridis anchors #440154 #3B528B #21918C #5EC962 #FDE725",
        "height_range_m": f"{HEIGHT_MIN}-{HEIGHT_MAX}",
    }
    return header, metadata


def read_project_year(project: str, year: int) -> pd.DataFrame:
    if not HEIGHT_PARQUET.exists():
        raise FileNotFoundError(f"Missing {HEIGHT_PARQUET}. Put the height parquet in raw/ first.")
    filters = [("project", "==", project), ("year", "==", year)]
    print(f"Reading {project} {year} height ...", flush=True)
    df = pd.read_parquet(
        HEIGHT_PARQUET,
        columns=["x", "y", "height_m"],
        filters=filters,
        engine="pyarrow",
    )
    print(f"Rows: {len(df):,}", flush=True)
    return df


def write_project_year(project: str, year: int, force: bool = False):
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / f"{project}_{year}_height.pmtiles"
    if out_path.exists() and not force:
        print(f"Skipping {out_path.name}; use --force to overwrite.", flush=True)
        return

    df = read_project_year(project, year)
    if df.empty:
        print(f"No rows for {project} {year}; skipped.", flush=True)
        return

    lon = df["x"].to_numpy(dtype=np.float64, copy=False)
    lat = df["y"].to_numpy(dtype=np.float64, copy=False)
    height = df["height_m"].to_numpy(dtype=np.float32, copy=False)
    bounds = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    header, metadata = header_metadata(project, year, bounds)

    print(f"Writing viridis height PMTiles: {out_path.name}", flush=True)
    tile_count = 0
    with write(str(out_path)) as writer:
        for z in ZOOMS:
            tx, ty, px, py = lonlat_to_tile_pixel(lon, lat, z)
            tile_key = tx.astype(np.int64) * (2 ** z) + ty.astype(np.int64)
            order = np.argsort(tile_key, kind="stable")
            sorted_key = tile_key[order]
            starts = np.r_[0, np.nonzero(sorted_key[1:] != sorted_key[:-1])[0] + 1]
            ends = np.r_[starts[1:], len(order)]

            for s, e in zip(starts, ends):
                idx = order[s:e]
                x = int(tx[idx[0]])
                y = int(ty[idx[0]])
                sums = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                counts = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint16)
                np.add.at(sums, (py[idx], px[idx]), height[idx])
                np.add.at(counts, (py[idx], px[idx]), 1)
                mask = counts > 0
                avg = np.zeros_like(sums)
                avg[mask] = sums[mask] / counts[mask]
                writer.write_tile(zxy_to_tileid(z, x, y), png_bytes(color_height_viridis(avg, mask)))
                tile_count += 1

        writer.finalize(header, metadata)

    print(f"Done {project} {year}: {tile_count:,} tiles | {out_path.stat().st_size / 1048576:.2f} MB", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build viridis height display PMTiles.")
    parser.add_argument("--all", action="store_true", help="Build all four projects and all years.")
    parser.add_argument("--project", choices=PROJECTS, help="Build one project, e.g., MBCI.")
    parser.add_argument("--year", type=int, choices=YEARS, help="Build one year, e.g., 2024.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing *_height.pmtiles files.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all:
        targets = [(p, y) for p in PROJECTS for y in YEARS]
    else:
        if args.project is None or args.year is None:
            raise SystemExit("Use either --all, or both --project PROJECT --year YEAR.")
        targets = [(args.project, args.year)]

    for project, year in targets:
        write_project_year(project, year, force=args.force)

    print("All requested viridis height PMTiles are complete.", flush=True)


if __name__ == "__main__":
    main()
