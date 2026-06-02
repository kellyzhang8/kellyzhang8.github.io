"""Build five-class forest land-cover display PMTiles.

This script overwrites/rebuilds the visible land-type raster PMTiles:

    data/FDL_2017_landtype.pmtiles
    data/FDL_2018_landtype.pmtiles
    ...

It does NOT change the click-query value PMTiles. Click popups still read the
original GLC_FCS10 code and then aggregate it in the browser to the five forest
classes.

Five-class aggregation used here:
    51, 52 -> Evergreen broadleaved forest
    71, 72 -> Evergreen needleleaved forest
    61, 62 -> Deciduous broadleaved forest
    81, 82 -> Deciduous needleleaved forest
    91, 92 -> Mixed-leaf forest
    all other codes -> transparent / not displayed

Run from the root of height_landtype_webmap_10m/:

    source .venv/bin/activate
    python scripts/build_grouped_landtype_pmtiles.py --all --force

For testing one case first:

    python scripts/build_grouped_landtype_pmtiles.py --project KBIC --year 2024 --force
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
LANDTYPE_PARQUET = RAW_DIR / "Predicted_landtype_ALL_years.parquet"

PROJECTS = ["FDL", "KBIC", "MBCI", "BN"]
YEARS = list(range(2017, 2025))
ZOOMS = [10, 11, 12, 13, 14]
TILE_SIZE = 256

# group id -> RGBA
GROUP_COLORS = {
    1: (68, 1, 84, 230),      # Evergreen broadleaved forest, #440154
    2: (59, 82, 139, 230),    # Evergreen needleleaved forest, #3B528B
    5: (33, 145, 140, 230),   # Mixed-leaf forest, #21918C
    3: (94, 201, 98, 230),    # Deciduous broadleaved forest, #5EC962
    4: (253, 231, 37, 230),   # Deciduous needleleaved forest, #FDE725
    0: (0, 0, 0, 0),          # Other / non-forest hidden
}

CODE_TO_GROUP = {
    51: 1, 52: 1,
    71: 2, 72: 2,
    61: 3, 62: 3,
    81: 4, 82: 4,
    91: 5, 92: 5,
}


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


def classify_landtype(codes: np.ndarray) -> np.ndarray:
    codes = pd.to_numeric(pd.Series(codes), errors="coerce").fillna(0).to_numpy(dtype=np.int16)
    out = np.zeros(len(codes), dtype=np.uint8)
    for code, group_id in CODE_TO_GROUP.items():
        out[codes == code] = group_id
    return out


def color_group_tile(group_arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    for group_id, color in GROUP_COLORS.items():
        m = (group_arr == group_id) & valid_mask
        if m.any():
            rgba[m] = color
    return rgba


def mode_group_per_pixel(px: np.ndarray, py: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 256x256 group array using per-pixel majority class and a valid mask."""
    counts = np.zeros((6, TILE_SIZE, TILE_SIZE), dtype=np.uint16)
    for group_id in range(6):
        m = groups == group_id
        if m.any():
            np.add.at(counts[group_id], (py[m], px[m]), 1)
    total = counts.sum(axis=0)
    valid = total > 0
    majority = counts.argmax(axis=0).astype(np.uint8)
    return majority, valid


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
        "name": f"{project} {year} five-class forest land cover",
        "format": "png",
        "bounds": f"{minx},{miny},{maxx},{maxy}",
        "center": f"{(minx + maxx) / 2},{(miny + maxy) / 2},12",
        "minzoom": str(min(ZOOMS)),
        "maxzoom": str(max(ZOOMS)),
        "type": "overlay",
        "aggregation": "51/52 evergreen broadleaved; 71/72 evergreen needleleaved; 61/62 deciduous broadleaved; 81/82 deciduous needleleaved; 91/92 mixed-leaf; others non-forest",
    }
    return header, metadata


def read_project_year(project: str, year: int) -> pd.DataFrame:
    if not LANDTYPE_PARQUET.exists():
        raise FileNotFoundError(f"Missing {LANDTYPE_PARQUET}. Put the land-type parquet in raw/ first.")
    filters = [("project", "==", project), ("year", "==", year)]
    print(f"Reading {project} {year} land type ...", flush=True)
    df = pd.read_parquet(
        LANDTYPE_PARQUET,
        columns=["x", "y", "Land_type"],
        filters=filters,
        engine="pyarrow",
    )
    print(f"Rows: {len(df):,}", flush=True)
    return df


def write_project_year(project: str, year: int, force: bool = False):
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / f"{project}_{year}_landtype.pmtiles"
    if out_path.exists() and not force:
        print(f"Skipping {out_path.name}; use --force to overwrite.", flush=True)
        return

    df = read_project_year(project, year)
    if df.empty:
        print(f"No rows for {project} {year}; skipped.", flush=True)
        return

    lon = df["x"].to_numpy(dtype=np.float64, copy=False)
    lat = df["y"].to_numpy(dtype=np.float64, copy=False)
    groups = classify_landtype(df["Land_type"].to_numpy(copy=False))
    bounds = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    header, metadata = header_metadata(project, year, bounds)

    print(f"Writing grouped land type PMTiles: {out_path.name}", flush=True)
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
                majority, valid = mode_group_per_pixel(px[idx], py[idx], groups[idx])
                rgba = color_group_tile(majority, valid)
                writer.write_tile(zxy_to_tileid(z, x, y), png_bytes(rgba))
                tile_count += 1

        writer.finalize(header, metadata)

    print(f"Done {project} {year}: {tile_count:,} tiles | {out_path.stat().st_size / 1048576:.2f} MB", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build five-class forest land-cover display PMTiles.")
    parser.add_argument("--all", action="store_true", help="Build all four projects and all years.")
    parser.add_argument("--project", choices=PROJECTS, help="Build one project, e.g., KBIC.")
    parser.add_argument("--year", type=int, choices=YEARS, help="Build one year, e.g., 2024.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing *_landtype.pmtiles files.")
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

    print("All requested grouped land-type PMTiles are complete.", flush=True)


if __name__ == "__main__":
    main()
