"""Build click-query PMTiles for the height/land-type web map.

What this creates
-----------------
For every project-year, this script creates two hidden raster PMTiles files:

    data/value_pmtiles/FDL_2017_height_values.pmtiles
    data/value_pmtiles/FDL_2017_landtype_values.pmtiles
    ...

The visible web map still uses your existing colored PMTiles. These value PMTiles
are only used when the user clicks the map, so the popup can report the nearest
original 10 m prediction-cell value.

Encoding
--------
height_m is stored as millimeters in RGB:
    mm = round(height_m * 1000)
    R = mm >> 16
    G = mm >> 8
    B = mm
    A = 255 for valid pixels

Land_type is stored in the red channel:
    R = Land_type
    G = 0
    B = 0
    A = 255 for valid pixels

Run from the root of height_landtype_webmap_10m/:

    pip install pandas pyarrow numpy pillow pmtiles
    python scripts/build_click_value_pmtiles.py --all

For testing one smaller case first:

    python scripts/build_click_value_pmtiles.py --project MBCI --year 2024 --force
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
VALUE_DIR = DATA_DIR / "value_pmtiles"

HEIGHT_PARQUET = RAW_DIR / "Predicted_height_ALL_years_from_saved_model.parquet"
LANDTYPE_PARQUET = RAW_DIR / "Predicted_landtype_ALL_years.parquet"

PROJECTS = ["FDL", "KBIC", "MBCI", "BN"]
YEARS = list(range(2017, 2025))

QUERY_Z = 15
TILE_SIZE = 256
HEIGHT_SCALE = 1000.0  # millimeters


def lonlat_to_tile_pixel(lon: np.ndarray, lat: np.ndarray, z: int):
    """Convert lon/lat arrays to Web Mercator z/x/y tile and pixel indices."""
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
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def header_metadata(project: str, year: int, layer: str, bounds: tuple[float, float, float, float]):
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
        "center_zoom": QUERY_Z,
    }
    metadata = {
        "name": f"{project} {year} {layer} click-query values",
        "format": "png",
        "bounds": f"{minx},{miny},{maxx},{maxy}",
        "center": f"{(minx + maxx) / 2},{(miny + maxy) / 2},{QUERY_Z}",
        "minzoom": str(QUERY_Z),
        "maxzoom": str(QUERY_Z),
        "type": "overlay",
        "description": "Hidden value PMTiles for click-query popups. Not intended for visual display.",
    }
    return header, metadata


def encode_height_rgba(height_m: np.ndarray) -> np.ndarray:
    """Return N x 4 uint8 array with height_m encoded as millimeters in RGB."""
    mm = np.rint(np.asarray(height_m, dtype=np.float64) * HEIGHT_SCALE)
    mm = np.nan_to_num(mm, nan=0.0, posinf=0.0, neginf=0.0)
    mm = np.clip(mm, 0, 16_777_215).astype(np.uint32)
    out = np.zeros((len(mm), 4), dtype=np.uint8)
    out[:, 0] = ((mm >> 16) & 255).astype(np.uint8)
    out[:, 1] = ((mm >> 8) & 255).astype(np.uint8)
    out[:, 2] = (mm & 255).astype(np.uint8)
    out[:, 3] = 255
    return out


def encode_land_rgba(land_type: np.ndarray) -> np.ndarray:
    """Return N x 4 uint8 array with Land_type encoded in R."""
    land = pd.to_numeric(pd.Series(land_type), errors="coerce").fillna(0).to_numpy()
    land = np.clip(land, 0, 255).astype(np.uint8)
    out = np.zeros((len(land), 4), dtype=np.uint8)
    out[:, 0] = land
    out[:, 3] = np.where(land > 0, 255, 0).astype(np.uint8)
    return out


def read_project_year(project: str, year: int, verify_keys: bool = False) -> pd.DataFrame:
    if not HEIGHT_PARQUET.exists():
        raise FileNotFoundError(f"Missing {HEIGHT_PARQUET}. Put the height parquet in raw/ first.")
    if not LANDTYPE_PARQUET.exists():
        raise FileNotFoundError(f"Missing {LANDTYPE_PARQUET}. Put the land-type parquet in raw/ first.")

    filters = [("project", "==", project), ("year", "==", year)]
    print(f"Reading {project} {year} ...", flush=True)

    h = pd.read_parquet(
        HEIGHT_PARQUET,
        columns=["project", "year", "x", "y", "height_m"],
        filters=filters,
        engine="pyarrow",
    )

    if verify_keys:
        l = pd.read_parquet(
            LANDTYPE_PARQUET,
            columns=["project", "year", "x", "y", "Land_type"],
            filters=filters,
            engine="pyarrow",
        )
        key_cols = ["project", "year", "x", "y"]
        if len(h) != len(l) or not h[key_cols].reset_index(drop=True).equals(l[key_cols].reset_index(drop=True)):
            print("Key order differs; merging by project/year/x/y ...", flush=True)
            h = h.merge(l, on=key_cols, how="inner", validate="one_to_one")
        else:
            h["Land_type"] = l["Land_type"].to_numpy(copy=False)
    else:
        # The provided height and land-type parquet files have matching row order.
        l = pd.read_parquet(
            LANDTYPE_PARQUET,
            columns=["Land_type"],
            filters=filters,
            engine="pyarrow",
        )
        if len(h) != len(l):
            raise ValueError(f"Row count mismatch for {project} {year}: height={len(h)}, land={len(l)}. Rerun with --verify-keys.")
        h["Land_type"] = l["Land_type"].to_numpy(copy=False)

    print(f"Rows: {len(h):,}", flush=True)
    return h


def write_value_pmtiles(project: str, year: int, force: bool = False, verify_keys: bool = False):
    VALUE_DIR.mkdir(parents=True, exist_ok=True)
    out_h = VALUE_DIR / f"{project}_{year}_height_values.pmtiles"
    out_l = VALUE_DIR / f"{project}_{year}_landtype_values.pmtiles"

    if out_h.exists() and out_l.exists() and not force:
        print(f"Skipping {project} {year}; value PMTiles already exist. Use --force to rebuild.", flush=True)
        return

    df = read_project_year(project, year, verify_keys=verify_keys)
    if df.empty:
        print(f"No rows for {project} {year}; skipped.", flush=True)
        return

    lon = df["x"].to_numpy(dtype=np.float64, copy=False)
    lat = df["y"].to_numpy(dtype=np.float64, copy=False)
    height = df["height_m"].to_numpy(dtype=np.float64, copy=False)
    land = df["Land_type"].to_numpy(copy=False)
    bounds = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))

    tx, ty, px, py = lonlat_to_tile_pixel(lon, lat, QUERY_Z)
    height_rgba_values = encode_height_rgba(height)
    land_rgba_values = encode_land_rgba(land)

    tile_key = tx.astype(np.int64) * (2 ** QUERY_Z) + ty.astype(np.int64)
    order = np.argsort(tile_key, kind="stable")
    sorted_key = tile_key[order]
    starts = np.r_[0, np.nonzero(sorted_key[1:] != sorted_key[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(order)]

    hh, hm = header_metadata(project, year, "height", bounds)
    lh, lm = header_metadata(project, year, "landtype", bounds)

    print(f"Writing {out_h.name} and {out_l.name} ...", flush=True)
    tile_count = 0
    with write(str(out_h)) as wh, write(str(out_l)) as wl:
        for s, e in zip(starts, ends):
            idx = order[s:e]
            x = int(tx[idx[0]])
            y = int(ty[idx[0]])

            height_img = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
            land_img = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)

            yy = py[idx]
            xx = px[idx]
            height_img[yy, xx, :] = height_rgba_values[idx, :]
            land_img[yy, xx, :] = land_rgba_values[idx, :]

            wh.write_tile(zxy_to_tileid(QUERY_Z, x, y), png_bytes(height_img))
            wl.write_tile(zxy_to_tileid(QUERY_Z, x, y), png_bytes(land_img))
            tile_count += 1

        wh.finalize(hh, hm)
        wl.finalize(lh, lm)

    print(
        f"Done {project} {year}: {tile_count:,} z{QUERY_Z} tiles | "
        f"height {out_h.stat().st_size / 1048576:.2f} MB, "
        f"landtype {out_l.stat().st_size / 1048576:.2f} MB",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build click-query value PMTiles.")
    parser.add_argument("--all", action="store_true", help="Build all four projects and all years.")
    parser.add_argument("--project", choices=PROJECTS, help="Build one project, e.g., MBCI.")
    parser.add_argument("--year", type=int, choices=YEARS, help="Build one year, e.g., 2024.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing value PMTiles.")
    parser.add_argument("--verify-keys", action="store_true", help="Read x/y from both parquet files and verify/merge keys. Slower but safest.")
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
        write_value_pmtiles(project, year, force=args.force, verify_keys=args.verify_keys)

    print("All requested click-query value PMTiles are complete.", flush=True)


if __name__ == "__main__":
    main()
