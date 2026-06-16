"""Robust height display PMTiles builder.

This rebuilds the visible Forest height layer directly from raw parquet.
It overwrites:
    data/<PROJECT>_<YEAR>_height.pmtiles

It does NOT touch:
    data/*_landtype.pmtiles
    data/value_pmtiles/*

Settings:
- Zooms: z8-z14, matching the web map raster source.
- Height color scale: 0-20+ m, matching the web legend.
- Color: viridis-style 5-stop palette.
- Per tile pixel value: mean height of raw cells falling into that pixel.
- Tiny transparent display gaps are filled from neighboring pixels so MapLibre
  does not show white seams between raster tiles.

Run from the project root:
    source .venv/bin/activate
    python scripts/build_height_viridis_pmtiles_safe.py --project MBCI --year 2024 --force
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pmtiles.writer import write
from pmtiles.tile import zxy_to_tileid, TileType, Compression
from shapely.geometry import shape
from shapely.ops import unary_union

try:
    from shapely import contains_xy
except Exception:
    contains_xy = None

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
DATA_DIR = ROOT / "data"
HEIGHT_PARQUET = RAW_DIR / "Predicted_height_ALL_years_from_saved_model.parquet"
BOUNDARY_GEOJSON = DATA_DIR / "boundaries.geojson"

PROJECTS = ["FDL", "KBIC", "MBCI", "BN"]
YEARS = list(range(2017, 2025))
ZOOMS = [8, 9, 10, 11, 12, 13, 14]
TILE_SIZE = 256
HEIGHT_MIN = 0.0
HEIGHT_MAX = 20.0
ALPHA = 230
GAP_FILL_ITERATIONS = 256
DISPLAY_RES_DEFAULT = 50.0
WEBMERCATOR_HALF_WORLD = 20037508.342789244
WEBMERCATOR_WORLD = WEBMERCATOR_HALF_WORLD * 2.0

VIRIDIS_STOPS = np.array([
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
], dtype=np.float32)


def lonlat_to_tile_pixel(lon: np.ndarray, lat: np.ndarray, z: int):
    """Return tile x/y and pixel x/y for lon/lat arrays in Web Mercator tiling."""
    n = 2 ** z
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lat = np.clip(lat, -85.05112878, 85.05112878)

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


def tile_pixel_centers_to_lonlat(z: int, tx: int, ty: int):
    scale = (2 ** z) * TILE_SIZE

    px = tx * TILE_SIZE + np.arange(TILE_SIZE, dtype=np.float64) + 0.5
    py = ty * TILE_SIZE + np.arange(TILE_SIZE, dtype=np.float64) + 0.5

    x_merc = px / scale * WEBMERCATOR_WORLD - WEBMERCATOR_HALF_WORLD
    y_merc = WEBMERCATOR_HALF_WORLD - py / scale * WEBMERCATOR_WORLD

    lon_1d, _ = webmercator_to_lonlat(x_merc, np.zeros_like(x_merc))
    _, lat_1d = webmercator_to_lonlat(np.zeros_like(y_merc), y_merc)

    lon2, lat2 = np.meshgrid(lon_1d, lat_1d)
    return lon2, lat2


def png_bytes(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def color_height(avg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Color a 256x256 average-height tile with a 0-20m viridis scale."""
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


def color_height_values(height_m: np.ndarray) -> np.ndarray:
    v = np.clip((height_m - HEIGHT_MIN) / (HEIGHT_MAX - HEIGHT_MIN), 0, 1)
    pos = v * (len(VIRIDIS_STOPS) - 1)
    i0 = np.floor(pos).astype(np.int16)
    i1 = np.clip(i0 + 1, 0, len(VIRIDIS_STOPS) - 1)
    t = (pos - i0)[:, None].astype(np.float32)
    rgb = VIRIDIS_STOPS[i0] * (1.0 - t) + VIRIDIS_STOPS[i1] * t

    rgba = np.zeros((len(height_m), 4), dtype=np.uint8)
    rgba[:, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[:, 3] = ALPHA
    return rgba


def fill_display_gaps(
    rgba: np.ndarray,
    fill_mask: np.ndarray | None = None,
    iterations: int = GAP_FILL_ITERATIONS,
) -> np.ndarray:
    """Fill display-only transparent gaps from neighboring colors."""
    img = rgba.copy()

    for _ in range(iterations):
        alpha = img[..., 3]
        transparent = alpha == 0 if fill_mask is None else (alpha == 0) & fill_mask
        if not transparent.any():
            break

        rgb_sum = np.zeros((*alpha.shape, 3), dtype=np.uint16)
        count = np.zeros(alpha.shape, dtype=np.uint8)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                src_y0 = max(0, -dy)
                src_y1 = TILE_SIZE - max(0, dy)
                src_x0 = max(0, -dx)
                src_x1 = TILE_SIZE - max(0, dx)
                dst_y0 = max(0, dy)
                dst_y1 = TILE_SIZE - max(0, -dy)
                dst_x0 = max(0, dx)
                dst_x1 = TILE_SIZE - max(0, -dx)

                src_alpha = alpha[src_y0:src_y1, src_x0:src_x1]
                valid = src_alpha > 0
                if not valid.any():
                    continue

                rgb_sum[dst_y0:dst_y1, dst_x0:dst_x1, :] += (
                    img[src_y0:src_y1, src_x0:src_x1, :3].astype(np.uint16)
                    * valid[..., None]
                )
                count[dst_y0:dst_y1, dst_x0:dst_x1] += valid.astype(np.uint8)

        fill = transparent & (count > 0)
        if not fill.any():
            break

        img[fill, :3] = (rgb_sum[fill] / count[fill, None]).astype(np.uint8)
        img[fill, 3] = ALPHA

    return img


def read_project_boundary_lonlat(project: str):
    if not BOUNDARY_GEOJSON.exists():
        raise FileNotFoundError(f"Missing {BOUNDARY_GEOJSON}.")

    with open(BOUNDARY_GEOJSON, "r", encoding="utf-8") as f:
        gj = json.load(f)

    geoms = []
    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        values = {str(v).upper() for v in props.values() if v is not None}
        matched = False
        for key in ["project", "PROJECT", "Project", "name", "NAME", "id", "ID"]:
            if key in props and str(props[key]).upper() == project.upper():
                matched = True
                break
        if not matched and project.upper() in values:
            matched = True
        if matched and feat.get("geometry"):
            geoms.append(shape(feat["geometry"]))

    if not geoms:
        raise ValueError(f"No boundary geometry found for project={project} in {BOUNDARY_GEOJSON}")
    return unary_union(geoms).buffer(0)


def tile_boundary_mask(z: int, tx: int, ty: int, boundary_lonlat):
    lon2, lat2 = tile_pixel_centers_to_lonlat(z, tx, ty)
    if contains_xy is not None:
        return contains_xy(boundary_lonlat, lon2, lat2)

    from shapely.geometry import Point
    return np.array(
        [
            boundary_lonlat.contains(Point(float(lon), float(lat)))
            for lon, lat in zip(lon2.ravel(), lat2.ravel())
        ],
        dtype=bool
    ).reshape((TILE_SIZE, TILE_SIZE))


def mask_tile_to_boundary(img: np.ndarray, inside: np.ndarray):
    img[~inside, :] = 0
    return img


def aggregate_to_display_grid(lon: np.ndarray, lat: np.ndarray, height: np.ndarray, display_res: float):
    x, y = lonlat_to_webmercator(lon, lat)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(height)
    if not np.any(ok):
        return pd.DataFrame({"xg": [], "yg": [], "height_m": []})

    xg = np.round(x[ok] / display_res) * display_res
    yg = np.round(y[ok] / display_res) * display_res

    df = pd.DataFrame({
        "xg": xg.astype(np.float64),
        "yg": yg.astype(np.float64),
        "height_m": height[ok].astype(np.float32),
    })

    return (
        df.groupby(["xg", "yg"], observed=True, sort=False)["height_m"]
        .mean()
        .reset_index()
    )


def paint_rect(
    tiles: dict[tuple[int, int], np.ndarray],
    z: int,
    gx_center: float,
    gy_center: float,
    color: np.ndarray,
    display_res: float,
):
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


def header_metadata(project: str, year: int, bounds: tuple[float, float, float, float], display_res: float):
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
        "name": f"{project} {year} forest height",
        "format": "png",
        "bounds": f"{minx},{miny},{maxx},{maxy}",
        "center": f"{(minx + maxx) / 2},{(miny + maxy) / 2},12",
        "minzoom": str(min(ZOOMS)),
        "maxzoom": str(max(ZOOMS)),
        "type": "overlay",
        "height_range_m": f"{HEIGHT_MIN}-{HEIGHT_MAX}",
        "display_res_m": str(display_res),
        "clip": "data/boundaries.geojson",
    }
    return header, metadata


def read_project_year(project: str, year: int) -> pd.DataFrame:
    if not HEIGHT_PARQUET.exists():
        raise FileNotFoundError(f"Missing {HEIGHT_PARQUET}")

    filters = [("project", "==", project), ("year", "==", year)]
    print(f"Reading {project} {year} height raw parquet ...", flush=True)
    df = pd.read_parquet(
        HEIGHT_PARQUET,
        columns=["x", "y", "height_m"],
        filters=filters,
        engine="pyarrow",
    )
    df = df.dropna(subset=["x", "y", "height_m"]).reset_index(drop=True)
    print(f"Valid rows: {len(df):,}", flush=True)
    return df


def write_project_year(project: str, year: int, display_res: float, force: bool = False):
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / f"{project}_{year}_height.pmtiles"
    if out_path.exists() and not force:
        print(f"Skipping {out_path.name}; use --force to overwrite.", flush=True)
        return

    boundary_lonlat = read_project_boundary_lonlat(project)

    df = read_project_year(project, year)
    if df.empty:
        raise ValueError(f"No valid height rows for {project} {year}; not writing empty PMTiles.")

    lon = df["x"].to_numpy(dtype=np.float64, copy=False)
    lat = df["y"].to_numpy(dtype=np.float64, copy=False)
    height = df["height_m"].to_numpy(dtype=np.float32, copy=False)

    display = aggregate_to_display_grid(lon, lat, height, display_res=display_res)
    print(
        f"Display cells after {display_res:g} m mean-height aggregation: {len(display):,}",
        flush=True,
    )

    if display.empty:
        raise ValueError(f"No display cells for {project} {year}; refusing to write empty PMTiles.")

    display_lon, display_lat = webmercator_to_lonlat(
        display["xg"].to_numpy(dtype=np.float64),
        display["yg"].to_numpy(dtype=np.float64),
    )
    bounds = (
        float(np.nanmin(display_lon)),
        float(np.nanmin(display_lat)),
        float(np.nanmax(display_lon)),
        float(np.nanmax(display_lat)),
    )
    header, metadata = header_metadata(project, year, bounds, display_res)

    print(f"Writing {out_path.name} ...", flush=True)
    total_tiles = 0
    with write(str(out_path)) as writer:
        xg = display["xg"].to_numpy(dtype=np.float64)
        yg = display["yg"].to_numpy(dtype=np.float64)
        colors = color_height_values(display["height_m"].to_numpy(dtype=np.float32))

        for z in ZOOMS:
            tiles: dict[tuple[int, int], np.ndarray] = {}
            gx, gy = mercator_to_global_pixel(xg, yg, z)

            for i in range(len(display)):
                paint_rect(
                    tiles,
                    z,
                    float(gx[i]),
                    float(gy[i]),
                    colors[i],
                    display_res,
                )

            z_tiles = 0

            for (tile_x, tile_y), img in tiles.items():
                inside = tile_boundary_mask(z, tile_x, tile_y, boundary_lonlat)
                img = mask_tile_to_boundary(fill_display_gaps(img, fill_mask=inside), inside)
                if img[..., 3].max() == 0:
                    continue
                writer.write_tile(zxy_to_tileid(z, tile_x, tile_y), png_bytes(img))
                z_tiles += 1

            print(f"  z{z}: {z_tiles:,} tiles", flush=True)
            total_tiles += z_tiles

        if total_tiles == 0:
            raise ValueError(f"No tiles written for {project} {year}; refusing to finalize empty PMTiles.")

        writer.finalize(header, metadata)

    print(f"Done {project} {year}: {total_tiles:,} tiles | {out_path.stat().st_size / 1048576:.2f} MB", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build robust visible height PMTiles from raw parquet.")
    parser.add_argument("--all", action="store_true", help="Build all project-year combinations.")
    parser.add_argument("--project", choices=PROJECTS, help="Build one project.")
    parser.add_argument("--year", type=int, choices=YEARS, help="Build one year.")
    parser.add_argument("--display-res", type=float, default=DISPLAY_RES_DEFAULT, help=f"Display-grid resolution in EPSG:3857 meters. Default: {DISPLAY_RES_DEFAULT:g}.")
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
        write_project_year(project, year, display_res=args.display_res, force=args.force)

    print("All requested robust height PMTiles are complete.", flush=True)


if __name__ == "__main__":
    main()
