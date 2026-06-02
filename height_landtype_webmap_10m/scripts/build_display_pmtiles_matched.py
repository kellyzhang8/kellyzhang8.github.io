"""Build display PMTiles directly from the raw parquet files.

This is a clean rebuild script for the visible map layers.
It overwrites:
    data/<PROJECT>_<YEAR>_height.pmtiles
    data/<PROJECT>_<YEAR>_landtype.pmtiles

Design choices matched to the requested webmap:
- Height uses a viridis-style palette with range 0–25+ m.
- Land type is aggregated to the five forest classes only.
- Non-forest / non-target classes are transparent.
- Height per display pixel is the mean of source cells in that display pixel.
- Land type per display pixel is the majority class in that display pixel.

Run from the project root:
    source .venv/bin/activate
    python scripts/build_display_pmtiles_matched.py --all --force

Test one project/year first:
    python scripts/build_display_pmtiles_matched.py --project KBIC --year 2024 --force
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
LANDTYPE_PARQUET = RAW_DIR / "Predicted_landtype_ALL_years.parquet"

PROJECTS = ["FDL", "KBIC", "MBCI", "BN"]
YEARS = list(range(2017, 2025))
ZOOMS = [10, 11, 12, 13, 14]
TILE_SIZE = 256
HEIGHT_MIN = 0.0
HEIGHT_MAX = 25.0
ALPHA = 230

VIRIDIS_STOPS = np.array([
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
], dtype=np.float32)

# five aggregated forest groups
CAT_COLORS = {
    1: "#440154",  # evergreen broadleaved
    2: "#3B528B",  # evergreen needleleaved
    5: "#21918C",  # mixed-leaf
    3: "#5EC962",  # deciduous broadleaved
    4: "#FDE725",  # deciduous needleleaved
}
CODE_TO_GROUP = {
    51: 1, 52: 1,
    71: 2, 72: 2,
    91: 5, 92: 5,
    61: 3, 62: 3,
    81: 4, 82: 4,
}


def hex_to_rgba(hex_color: str, alpha: int = ALPHA):
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


GROUP_COLORS = {gid: hex_to_rgba(color) for gid, color in CAT_COLORS.items()}


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


def color_height(avg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.clip((avg - HEIGHT_MIN) / (HEIGHT_MAX - HEIGHT_MIN), 0, 1)
    pos = v * (len(VIRIDIS_STOPS) - 1)
    i0 = np.floor(pos).astype(np.int16)
    i1 = np.clip(i0 + 1, 0, len(VIRIDIS_STOPS) - 1)
    t = (pos - i0)[..., None].astype(np.float32)
    rgb = VIRIDIS_STOPS[i0] * (1.0 - t) + VIRIDIS_STOPS[i1] * t
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(mask, ALPHA, 0).astype(np.uint8)
    return rgba


def classify_landtype(codes: np.ndarray) -> np.ndarray:
    codes = pd.to_numeric(pd.Series(codes), errors="coerce").fillna(0).to_numpy(dtype=np.int16)
    out = np.zeros(len(codes), dtype=np.uint8)
    for code, gid in CODE_TO_GROUP.items():
        out[codes == code] = gid
    return out


def color_land(groups: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    for gid, color in GROUP_COLORS.items():
        m = (groups == gid) & valid
        if m.any():
            rgba[m] = color
    return rgba


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


def read_project_year(project: str, year: int) -> pd.DataFrame:
    if not HEIGHT_PARQUET.exists():
        raise FileNotFoundError(f"Missing {HEIGHT_PARQUET}")
    if not LANDTYPE_PARQUET.exists():
        raise FileNotFoundError(f"Missing {LANDTYPE_PARQUET}")
    filters = [("project", "==", project), ("year", "==", year)]
    h = pd.read_parquet(HEIGHT_PARQUET, columns=["x", "y", "height_m"], filters=filters, engine="pyarrow")
    l = pd.read_parquet(LANDTYPE_PARQUET, columns=["Land_type"], filters=filters, engine="pyarrow")
    if len(h) != len(l):
        raise ValueError(f"Row count mismatch for {project} {year}: height={len(h)}, landtype={len(l)}")
    h["Land_type"] = l["Land_type"].to_numpy(copy=False)
    return h


def write_project_year(project: str, year: int, force: bool = False):
    DATA_DIR.mkdir(exist_ok=True)
    out_h = DATA_DIR / f"{project}_{year}_height.pmtiles"
    out_l = DATA_DIR / f"{project}_{year}_landtype.pmtiles"
    if out_h.exists() and out_l.exists() and not force:
        print(f"Skipping {project} {year}; use --force to overwrite.", flush=True)
        return

    print(f"Reading {project} {year} ...", flush=True)
    df = read_project_year(project, year)
    lon = df["x"].to_numpy(dtype=np.float64, copy=False)
    lat = df["y"].to_numpy(dtype=np.float64, copy=False)
    height = df["height_m"].to_numpy(dtype=np.float32, copy=False)
    groups = classify_landtype(df["Land_type"].to_numpy(copy=False))
    bounds = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    hh, hm = header_metadata(project, year, "height", bounds)
    lh, lm = header_metadata(project, year, "landtype", bounds)

    tile_counter = 0
    with write(str(out_h)) as wh, write(str(out_l)) as wl:
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

                # height: mean value per display pixel
                sums = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                counts = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint16)
                np.add.at(sums, (py[idx], px[idx]), height[idx])
                np.add.at(counts, (py[idx], px[idx]), 1)
                mask = counts > 0
                avg = np.zeros_like(sums)
                avg[mask] = sums[mask] / counts[mask]
                wh.write_tile(zxy_to_tileid(z, x, y), png_bytes(color_height(avg, mask)))

                # land type: majority aggregated class per display pixel
                countstack = np.zeros((6, TILE_SIZE, TILE_SIZE), dtype=np.uint16)
                for gid in (1, 2, 3, 4, 5):
                    m = groups[idx] == gid
                    if m.any():
                        np.add.at(countstack[gid], (py[idx][m], px[idx][m]), 1)
                total = countstack.sum(axis=0)
                valid = total > 0
                majority = countstack.argmax(axis=0).astype(np.uint8)
                wl.write_tile(zxy_to_tileid(z, x, y), png_bytes(color_land(majority, valid)))

                tile_counter += 1
        wh.finalize(hh, hm)
        wl.finalize(lh, lm)

    print(
        f"Done {project} {year}: wrote {tile_counter} tile pairs | "
        f"height {out_h.stat().st_size / 1048576:.2f} MB, "
        f"landtype {out_l.stat().st_size / 1048576:.2f} MB",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build matched display PMTiles from raw parquet.")
    parser.add_argument("--all", action="store_true", help="Build all project-year combinations.")
    parser.add_argument("--project", choices=PROJECTS, help="Build one project.")
    parser.add_argument("--year", type=int, choices=YEARS, help="Build one year.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing PMTiles.")
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

    print("All requested display PMTiles are complete.", flush=True)


if __name__ == "__main__":
    main()
