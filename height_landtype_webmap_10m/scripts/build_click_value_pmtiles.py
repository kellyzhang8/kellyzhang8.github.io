"""Build click-query value PMTiles from raw 10 m raster-center parquet data.

This footprint version is stricter than the old point-value builder:

Old behavior:
    each raw cell center was encoded to one tile pixel;
    clicking could return a nearby encoded pixel.

New behavior:
    each raw x/y point is treated as the CENTER of a 10 m raster cell;
    the value is burned into a 10 m x 10 m footprint in Web Mercator;
    the webpage samples the clicked pixel directly.

Outputs:
    data/value_pmtiles/<PROJECT>_<YEAR>_height_values.pmtiles
    data/value_pmtiles/<PROJECT>_<YEAR>_landtype_values.pmtiles

Run from height_landtype_webmap_10m/:

    source .venv/bin/activate
    python scripts/build_click_value_pmtiles.py --project FDL --year 2017 --force

All:

    python scripts/build_click_value_pmtiles.py --all --force

Notes:
- QUERY_Z defaults to 18 for fine click-query precision. This can create larger
  value PMTiles than the old z15 point builder.
- If full rebuild is too slow/large, use --query-z 17 and set QUERY_Z = 17 in
  index.html too.
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

QUERY_Z_DEFAULT = 18
TILE_SIZE = 256
HEIGHT_SCALE = 1000.0  # meters -> millimeters in RGB
CELL_SIZE_M = 10.0     # raw x/y is the center of a 10 m raster cell
WEBMERCATOR_HALF_WORLD = 20037508.342789244
WEBMERCATOR_WORLD = WEBMERCATOR_HALF_WORLD * 2.0


def lonlat_to_webmercator(lon: np.ndarray, lat: np.ndarray):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lat = np.clip(lat, -85.05112878, 85.05112878)
    x = lon * WEBMERCATOR_HALF_WORLD / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) * WEBMERCATOR_HALF_WORLD / np.pi
    return x, y


def webmercator_to_lonlat(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lon = x / WEBMERCATOR_HALF_WORLD * 180.0
    lat = (2.0 * np.arctan(np.exp(y / WEBMERCATOR_HALF_WORLD * np.pi)) - np.pi / 2.0) * 180.0 / np.pi
    return lon, lat


def mercator_to_global_pixel(x: np.ndarray, y: np.ndarray, z: int):
    scale = (2 ** z) * TILE_SIZE
    gx = (x + WEBMERCATOR_HALF_WORLD) / WEBMERCATOR_WORLD * scale
    gy = (WEBMERCATOR_HALF_WORLD - y) / WEBMERCATOR_WORLD * scale
    return gx, gy


def png_bytes(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def encode_height_one(height_m: float):
    if not np.isfinite(height_m):
        return None
    mm = int(round(float(height_m) * HEIGHT_SCALE))
    mm = max(0, min(16_777_215, mm))
    return np.array([(mm >> 16) & 255, (mm >> 8) & 255, mm & 255, 255], dtype=np.uint8)


def encode_land_one(land_type):
    if pd.isna(land_type):
        return None
    code = int(land_type)
    if code <= 0:
        return None
    code = max(0, min(255, code))
    return np.array([code, 0, 0, 255], dtype=np.uint8)


def header_metadata(project: str, year: int, layer: str, bounds: tuple[float, float, float, float], query_z: int):
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
        "center_zoom": query_z,
    }
    metadata = {
        "name": f"{project} {year} {layer} click-query footprint values",
        "format": "png",
        "bounds": f"{minx},{miny},{maxx},{maxy}",
        "center": f"{(minx + maxx) / 2},{(miny + maxy) / 2},{query_z}",
        "minzoom": str(query_z),
        "maxzoom": str(query_z),
        "type": "overlay",
        "cell_size_m": str(CELL_SIZE_M),
        "description": "Hidden value PMTiles. Raw raster-center points are burned as 10 m cell footprints.",
    }
    return header, metadata


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
    ).reset_index(drop=True)

    if verify_keys:
        l = pd.read_parquet(
            LANDTYPE_PARQUET,
            columns=["project", "year", "x", "y", "Land_type"],
            filters=filters,
            engine="pyarrow",
        ).reset_index(drop=True)
        key_cols = ["project", "year", "x", "y"]
        if len(h) != len(l) or not h[key_cols].equals(l[key_cols]):
            print("Key order differs; merging by project/year/x/y ...", flush=True)
            h = h.merge(l, on=key_cols, how="inner", validate="one_to_one")
        else:
            h["Land_type"] = l["Land_type"].to_numpy(copy=False)
    else:
        l = pd.read_parquet(
            LANDTYPE_PARQUET,
            columns=["Land_type"],
            filters=filters,
            engine="pyarrow",
        ).reset_index(drop=True)
        if len(h) != len(l):
            raise ValueError(
                f"Row count mismatch for {project} {year}: height={len(h)}, landtype={len(l)}. "
                "Rerun with --verify-keys."
            )
        h["Land_type"] = l["Land_type"].to_numpy(copy=False)

    print(f"Rows: {len(h):,}", flush=True)
    return h


def build_rect_index(gx: np.ndarray, gy: np.ndarray, query_z: int):
    """Return rectangle bounds and a sorted tile->row mapping for 10 m footprints."""
    m_per_px = WEBMERCATOR_WORLD / ((2 ** query_z) * TILE_SIZE)
    cell_px = max(1.0, CELL_SIZE_M / m_per_px)
    half = cell_px / 2.0
    max_global = (2 ** query_z) * TILE_SIZE - 1

    x0 = np.floor(gx - half).astype(np.int64)
    x1 = np.ceil(gx + half).astype(np.int64)
    y0 = np.floor(gy - half).astype(np.int64)
    y1 = np.ceil(gy + half).astype(np.int64)

    x0 = np.clip(x0, 0, max_global)
    x1 = np.clip(x1, 0, max_global + 1)
    y0 = np.clip(y0, 0, max_global)
    y1 = np.clip(y1, 0, max_global + 1)

    tx0 = x0 // TILE_SIZE
    tx1 = (x1 - 1) // TILE_SIZE
    ty0 = y0 // TILE_SIZE
    ty1 = (y1 - 1) // TILE_SIZE

    n = len(gx)
    # Most cells touch one tile, some touch two/four near tile borders.
    repeats = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    total = int(repeats.sum())

    tile_keys = np.empty(total, dtype=np.int64)
    row_ids = np.empty(total, dtype=np.int64)

    pos = 0
    world_tiles = 2 ** query_z
    for i in range(n):
        for tx in range(int(tx0[i]), int(tx1[i]) + 1):
            for ty in range(int(ty0[i]), int(ty1[i]) + 1):
                tile_keys[pos] = tx * world_tiles + ty
                row_ids[pos] = i
                pos += 1

    order = np.argsort(tile_keys, kind="stable")
    return x0, x1, y0, y1, tile_keys[order], row_ids[order]


def write_value_pmtiles(project: str, year: int, force: bool = False, verify_keys: bool = False, query_z: int = QUERY_Z_DEFAULT):
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

    x_m, y_m = lonlat_to_webmercator(lon, lat)
    gx, gy = mercator_to_global_pixel(x_m, y_m, query_z)

    bounds = (float(np.nanmin(lon)), float(np.nanmin(lat)), float(np.nanmax(lon)), float(np.nanmax(lat)))
    hh, hm = header_metadata(project, year, "height", bounds, query_z)
    lh, lm = header_metadata(project, year, "landtype", bounds, query_z)

    print(f"Building 10 m footprint tile index at z{query_z} ...", flush=True)
    x0, x1, y0, y1, sorted_tile_keys, sorted_row_ids = build_rect_index(gx, gy, query_z)

    starts = np.r_[0, np.nonzero(sorted_tile_keys[1:] != sorted_tile_keys[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(sorted_tile_keys)]
    world_tiles = 2 ** query_z

    print(f"Writing {out_h.name} and {out_l.name} | tiles: {len(starts):,} ...", flush=True)
    tile_count = 0

    with write(str(out_h)) as wh, write(str(out_l)) as wl:
        for s, e in zip(starts, ends):
            key = int(sorted_tile_keys[s])
            tx = key // world_tiles
            ty = key % world_tiles

            height_img = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
            land_img = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)

            for rid in sorted_row_ids[s:e]:
                rid = int(rid)
                lx0 = max(0, int(x0[rid]) - tx * TILE_SIZE)
                lx1 = min(TILE_SIZE, int(x1[rid]) - tx * TILE_SIZE)
                ly0 = max(0, int(y0[rid]) - ty * TILE_SIZE)
                ly1 = min(TILE_SIZE, int(y1[rid]) - ty * TILE_SIZE)

                if lx1 <= lx0 or ly1 <= ly0:
                    continue

                hc = encode_height_one(float(height[rid]))
                if hc is not None:
                    height_img[ly0:ly1, lx0:lx1, :] = hc

                lc = encode_land_one(land[rid])
                if lc is not None:
                    land_img[ly0:ly1, lx0:lx1, :] = lc

            if height_img[..., 3].max() > 0:
                wh.write_tile(zxy_to_tileid(query_z, tx, ty), png_bytes(height_img))
            if land_img[..., 3].max() > 0:
                wl.write_tile(zxy_to_tileid(query_z, tx, ty), png_bytes(land_img))
            tile_count += 1

        wh.finalize(hh, hm)
        wl.finalize(lh, lm)

    print(
        f"Done {project} {year}: {tile_count:,} z{query_z} footprint tiles | "
        f"height {out_h.stat().st_size / 1048576:.2f} MB, "
        f"landtype {out_l.stat().st_size / 1048576:.2f} MB",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build click-query value PMTiles using 10 m cell footprints.")
    parser.add_argument("--all", action="store_true", help="Build all four projects and all years.")
    parser.add_argument("--project", choices=PROJECTS, help="Build one project, e.g. FDL.")
    parser.add_argument("--year", type=int, choices=YEARS, help="Build one year, e.g. 2017.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing value PMTiles.")
    parser.add_argument("--verify-keys", action="store_true", help="Verify/merge landtype by project/year/x/y. Slower but safest.")
    parser.add_argument("--query-z", type=int, default=QUERY_Z_DEFAULT, help=f"Query zoom. Default: {QUERY_Z_DEFAULT}.")
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
        write_value_pmtiles(
            project,
            year,
            force=args.force,
            verify_keys=args.verify_keys,
            query_z=args.query_z,
        )

    print("All requested click-query footprint value PMTiles are complete.", flush=True)


if __name__ == "__main__":
    main()
