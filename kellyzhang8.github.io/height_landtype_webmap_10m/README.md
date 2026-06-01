# Height and Land Type Web Map

This folder is ready for GitHub Pages / Netlify deployment.

## What is included

- `index.html`: the MapLibre GL web map.
- `data/*.pmtiles`: web-ready PMTiles raster tiles for four projects, 2017–2024, and two layers:
  - predicted canopy height
  - predicted land cover type
- `data/boundaries.geojson`: simplified project boundaries from the uploaded shapefiles.
- `scripts/build_pmtiles_fast.py`: rebuild script from the original parquet files.
- `scripts/build_missing_pmtiles.py`: resume script for missing PMTiles.

## Deploy on GitHub Pages

1. Create a new GitHub repository.
2. Upload everything in this folder to the repository root.
3. Go to **Settings → Pages**.
4. Set **Source** to `Deploy from a branch`.
5. Choose branch `main` and folder `/root`.
6. Open the GitHub Pages URL after deployment.

## Local preview

Run this inside the folder:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

Do not open `index.html` by double-clicking it, because browser security rules may block local PMTiles loading.

## Data notes

The original uploaded parquet files contain about 55.48 million point-grid records:

- `Predicted_height_ALL_years_from_saved_model.parquet`
- `Predicted_landtype_ALL_years.parquet`

For GitHub Pages, the data were converted to **raster PMTiles** at zoom levels 10, 11, 12, 13, and 14. This keeps the web map small and fast enough for direct hosting. Zoom level 14 is included so the visual map approaches/exceeds the original 10 m raster resolution in all four project latitudes. At zoom levels above 14, MapLibre overzooms the z14 tiles.

If you want finer local detail, edit this line in `scripts/build_pmtiles_fast.py`:

```python
ZOOMS = [10, 11, 12, 13, 14]
```

For example:

```python
ZOOMS = [10, 11, 12, 13, 14]
```

Then place the two original parquet files in `raw/` and run:

```bash
python scripts/build_pmtiles_fast.py
```

Higher zooms will create larger files and take longer to build.

## Rebuild requirements

```bash
pip install pandas pyarrow pillow pmtiles numpy
```

The current packaged web map does not require Python. Python is only needed if you want to rebuild the PMTiles.


## Resolution note

This package is a raster-tile visualization generated from all point-grid records. It is not a vector feature package and does not preserve every 10 m cell as a clickable feature. For 10 m visual inspection, z14 tiles are included.
