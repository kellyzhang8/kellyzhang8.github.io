# Final clean rebuild files for height_landtype_webmap_10m

Copy all files in this package into the root of:

```text
/Users/kelly/Documents/GitHub/kellyzhang8.github.io/height_landtype_webmap_10m
```

This package contains the final versions to avoid mixed old/new PMTiles:

```text
index.html
scripts/build_height_viridis_pmtiles_safe.py
scripts/build_landtype_display_grid_pmtiles_clipped.py
scripts/build_click_value_pmtiles.py
requirements_webmap.txt
run_clean_rebuild.sh
```

## What each final script does

- `build_height_viridis_pmtiles_safe.py`
  - Visible `Forest height` layer
  - reads `raw/Predicted_height_ALL_years_from_saved_model.parquet`
  - writes `data/<PROJECT>_<YEAR>_height.pmtiles`
  - color scale: 0–20+ m
  - zooms: z8–z14

- `build_landtype_display_grid_pmtiles_clipped.py`
  - Visible `Forest land-cover type` layer
  - reads `raw/Predicted_landtype_ALL_years.parquet`
  - writes `data/<PROJECT>_<YEAR>_landtype.pmtiles`
  - display grid: 50 m dominant forest type
  - clips every output tile pixel to `data/boundaries.geojson`
  - zooms: z8–z14

- `build_click_value_pmtiles.py`
  - Hidden click-query value layer
  - writes `data/value_pmtiles/*_values.pmtiles`
  - treats each raw x/y as the center of a 10 m cell footprint
  - query zoom: z18

- `index.html`
  - uses absolute PMTiles URLs
  - visible raster minzoom/maxzoom: 8–14
  - click-query `QUERY_Z = 18`
  - height legend: 0 / 10 / 20+ m

## Install packages if needed

Inside the project folder:

```bash
source .venv/bin/activate
python -m pip install -r requirements_webmap.txt
```

## Clean rebuild everything from raw

Inside the project folder:

```bash
source .venv/bin/activate
bash run_clean_rebuild.sh
```

Expected final counts:

```text
height visible PMTiles:   32
landtype visible PMTiles: 32
click value PMTiles:      64
```

## Test locally

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

Use `Command + Shift + R` or an incognito window to avoid browser caching old PMTiles.
