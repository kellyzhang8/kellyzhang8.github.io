"""Build web display PMTiles for forest land-cover type using the same
visualization logic as the Colab manuscript map.

This script is for the VISIBLE landtype layer only. It overwrites:
    data/<PROJECT>_<YEAR>_landtype.pmtiles

It does not change:
    data/<PROJECT>_<YEAR>_height.pmtiles
    data/value_pmtiles/*

Logic:
- Read original raw/Predicted_landtype_ALL_years.parquet
- Keep only the five forest-code groups:
    51/52 -> Evergreen broadleaved
    71/72 -> Evergreen needleleaved
    91/92 -> Mixed-leaf
    61/62 -> Deciduous broadleaved
    81/82 -> Deciduous needleleaved
- Convert lon/lat to EPSG:3857 meters
- Aggregate original 10 m cells to a display grid using dominant class
- Rasterize display cells into PMTiles

Run from height_landtype_webmap_10m/:

    source .venv/bin/activate
    python scripts/build_landtype_display_grid_pmtiles.py --all --force --display-res 50

Test one first:

    python scripts/build_landtype_display_grid_pmtiles.py --project KBIC --year 2022 --force --display-res 50
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
ZOOMS = [8, 9, 10, 11, 12, 13, 14]
TILE_SIZE = 256
WEBMERCATOR_HALF_WORLD = 20037508.342789244
WEBMERCATOR_WORLD = WEBMERCATOR_HALF_WORLD * 2.0
ALPHA = 230

# This order and color palette matches the Colab map legend.
GROUP_LABELS = {
    1: "Evergreen broadleaved",
    2: "Evergreen needleleaved",
    5: "Mixed-leaf",
    3: "Deciduous broadleaved",
    4: "Deciduous needleleaved",
}

CODE_TO_GROUP = {
    51: 1, 52: 1,
    71: 2, 72: 2,
    91: 5, 92: 5,
    61: 3, 62: 3,
    81: 4, 82: 4,
}

GROUP_COLORS = {
    1: (68, 1, 84, ALPHA),       # #440154 Evergreen broadleaved
    2: (59, 82, 139, ALPHA),     # #3B528B Evergreen needleleaved
    5: (33, 145, 140, ALPHA),    # #21918C Mixed-leaf
    3: (94, 201, 98, ALPHA),     # #5EC962 Deciduous broadleaved
    4: (253, 231, 37, ALPHA),    # #FDE725 Deciduous needleleaved
}


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
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def classify_landtype(codes: np.ndarray) -> np.ndarray:
    codes = pd.to_numeric(pd.Series(codes), errors="coerce").fillna(0).to_numpy(dtype=np.int16)
    out = np.zeros(len(codes), dtype=np.uint8)
    for code, group_id in CODE_TO_GROUP.items():
        out[codes == code] = group_id
    return out


def aggregate_to_display_grid(lon: np.ndarray, lat: np.ndarray, groups: np.ndarray, display_res: float):
    """Aggregate raw 10 m cells to display grid in EPSG:3857 using dominant class."""
    x, y = lonlat_to_webmercator(lon, lat)
    ok = np.isfinite(x) & np.isfinite(y) & np.isin(groups, [1, 2, 3, 4, 5])
    if not np.any(ok):
        return pd.DataFrame({"xg": [], "yg": [], "group": []})

    xg = np.round(x[ok] / display_res) * display_res
    yg = np.round(y[ok] / display_res) * display_res
    df = pd.DataFrame({
        "xg": xg.astype(np.float64),
        "yg": yg.astype(np.float64),
        "group": groups[ok].astype(np.uint8),
    })

    cnt = (
        df.groupby(["xg", "yg", "group"], observed=True, sort=False)
          .size()
          .reset_index(name="n")
    )
    idx = cnt.groupby(["xg", "yg"], observed=True)["n"].idxmax()
    out = cnt.loc[idx, ["xg", "yg", "group"]].reset_index(drop=True)
    return out


def read_project_year(project: str, year: int) -> pd.DataFrame:
    if not LANDTYPE_PARQUET.exists():
        raise FileNotFoundError(f"Missing {LANDTYPE_PARQUET}. Put the landtype parquet in raw/ first.")

    filters = [("project", "==", project), ("year", "==", year)]
    print(f"Reading {project} {year} landtype ...", flush=True)
    df = pd.read_parquet(
        LANDTYPE_PARQUET,
        columns=["x", "y", "Land_type"],
        filters=filters,
        engine="pyarrow",
    )
    print(f"Rows: {len(df):,}", flush=True)
    return df


def header_metadata(project: str, year: int, bounds_lonlat: tuple[float, float, float, float], display_res: float):
    minx, miny, maxx, maxy = bounds_lonlat
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
        "name": f"{project} {year} forest landtype display grid",
        "format": "png",
        "bounds": f"{minx},{miny},{maxx},{maxy}",
        "center": f"{(minx + maxx) / 2},{(miny + maxy) / 2},12",
        "minzoom": str(min(ZOOMS)),
        "maxzoom": str(max(ZOOMS)),
        "type": "overlay",
        "display_res_m": str(display_res),
        "legend": "1 evergreen broadleaved; 2 evergreen needleleaved; 5 mixed-leaf; 3 deciduous broadleaved; 4 deciduous needleleaved",
    }
    return header, metadata


def paint_rect(tiles: dict[tuple[int, int], np.ndarray], z: int, gx_center: float, gy_center: float, group_id: int, display_res: float):
    """Paint one display-grid cell as a square into all touched 256x256 tiles."""
    if group_id not in GROUP_COLORS:
        return

    m_per_px = WEBMERCATOR_WORLD / ((2 ** z) * TILE_SIZE)
    cell_px = max(1.0, display_res / m_per_px)
    half = cell_px / 2.0

    x0 = int(np.floor(gx_center - half))
    x1 = int(np.ceil(gx_center + half))
    y0 = int(np.floor(gy_center - half))
    y1 = int(np.ceil(gy_center + half))

    max_global = (2 ** z) * TILE_SIZE - 1
    x0 = max(0, min(max_global, x0))
    x1 = max(0, min(max_global + 1, x1))
    y0 = max(0, min(max_global, y0))
    y1 = max(0, min(max_global + 1, y1))

    if x1 <= x0 or y1 <= y0:
        return

    tx0, tx1 = x0 // TILE_SIZE, (x1 - 1) // TILE_SIZE
    ty0, ty1 = y0 // TILE_SIZE, (y1 - 1) // TILE_SIZE

    color = GROUP_COLORS[group_id]
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            tile = tiles.get((tx, ty))
            if tile is None:
                tile = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
                tiles[(tx, ty)] = tile

            lx0 = max(0, x0 - tx * TILE_SIZE)
            lx1 = min(TILE_SIZE, x1 - tx * TILE_SIZE)
            ly0 = max(0, y0 - ty * TILE_SIZE)
            ly1 = min(TILE_SIZE, y1 - ty * TILE_SIZE)

            if lx1 > lx0 and ly1 > ly0:
                tile[ly0:ly1, lx0:lx1, :] = color


def write_project_year(project: str, year: int, display_res: float, force: bool = False):
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

    display = aggregate_to_display_grid(lon, lat, groups, display_res=display_res)
    print(f"Display cells after {display_res:g} m dominant-class aggregation: {len(display):,}", flush=True)

    if display.empty:
        print(f"No forest display cells for {project} {year}; skipped.", flush=True)
        return

    display_lon, display_lat = webmercator_to_lonlat(display["xg"].to_numpy(), display["yg"].to_numpy())
    bounds = (
        float(np.nanmin(display_lon)),
        float(np.nanmin(display_lat)),
        float(np.nanmax(display_lon)),
        float(np.nanmax(display_lat)),
    )
    header, metadata = header_metadata(project, year, bounds, display_res)

    print(f"Writing {out_path.name} ...", flush=True)
    with write(str(out_path)) as writer:
        tile_count = 0
        xg = display["xg"].to_numpy(dtype=np.float64)
        yg = display["yg"].to_numpy(dtype=np.float64)
        g = display["group"].to_numpy(dtype=np.uint8)

        for z in ZOOMS:
            tiles: dict[tuple[int, int], np.ndarray] = {}
            gx, gy = mercator_to_global_pixel(xg, yg, z)
            for i in range(len(display)):
                paint_rect(tiles, z, float(gx[i]), float(gy[i]), int(g[i]), display_res)

            for (tx, ty), img in tiles.items():
                writer.write_tile(zxy_to_tileid(z, tx, ty), png_bytes(img))
                tile_count += 1

        writer.finalize(header, metadata)

    print(f"Done {project} {year}: {tile_count:,} tiles | {out_path.stat().st_size / 1048576:.2f} MB", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build landtype web PMTiles using Colab-style display-grid dominant class.")
    parser.add_argument("--all", action="store_true", help="Build all four projects and all years.")
    parser.add_argument("--project", choices=PROJECTS, help="Build one project, e.g. KBIC.")
    parser.add_argument("--year", type=int, choices=YEARS, help="Build one year, e.g. 2022.")
    parser.add_argument("--display-res", type=float, default=50.0, help="Display-grid resolution in EPSG:3857 meters. Default: 50.")
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
        write_project_year(project, year, display_res=args.display_res, force=args.force)

    print("All requested landtype display-grid PMTiles are complete.", flush=True)


if __name__ == "__main__":
    main()
