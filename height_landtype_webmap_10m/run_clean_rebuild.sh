#!/usr/bin/env bash
set -euo pipefail

# Run from the root folder of height_landtype_webmap_10m.
# This deletes old generated PMTiles and rebuilds everything from raw/.

if [ ! -d "raw" ] || [ ! -d "data" ] || [ ! -d "scripts" ]; then
  echo "ERROR: Run this script from the root of height_landtype_webmap_10m." >&2
  exit 1
fi

if [ ! -f "raw/Predicted_height_ALL_years_from_saved_model.parquet" ]; then
  echo "ERROR: Missing raw/Predicted_height_ALL_years_from_saved_model.parquet" >&2
  exit 1
fi

if [ ! -f "raw/Predicted_landtype_ALL_years.parquet" ]; then
  echo "ERROR: Missing raw/Predicted_landtype_ALL_years.parquet" >&2
  exit 1
fi

if [ ! -f "data/boundaries.geojson" ]; then
  echo "ERROR: Missing data/boundaries.geojson" >&2
  exit 1
fi

mkdir -p data/value_pmtiles

stamp=$(date +%Y%m%d_%H%M%S)
backup_dir="$HOME/Desktop/height_landtype_generated_backup_$stamp"
mkdir -p "$backup_dir"

# Back up existing generated files only.
find data -maxdepth 1 \( -name '*_height.pmtiles' -o -name '*_landtype.pmtiles' \) -exec cp {} "$backup_dir" \; || true
find data/value_pmtiles -maxdepth 1 -name '*_values.pmtiles' -exec cp {} "$backup_dir" \; || true

echo "Backed up existing generated PMTiles to: $backup_dir"

echo "Deleting old generated PMTiles..."
rm -f data/*_height.pmtiles
rm -f data/*_landtype.pmtiles
rm -f data/value_pmtiles/*_values.pmtiles

echo "Rebuilding visible height layer: 0-20 m, z8-z14..."
python scripts/build_height_viridis_pmtiles_safe.py --all --force

echo "Rebuilding boundary-clipped landtype layer: display_res=50 m, z8-z14..."
python scripts/build_landtype_display_grid_pmtiles_clipped.py --all --force --display-res 50

echo "Rebuilding click-query value PMTiles: 10 m footprint, z18..."
python scripts/build_click_value_pmtiles.py --all --force --query-z 18

echo "Counts:"
echo -n "height visible PMTiles:   "
find data -maxdepth 1 -name '*_height.pmtiles' | wc -l
echo -n "landtype visible PMTiles: "
find data -maxdepth 1 -name '*_landtype.pmtiles' | wc -l
echo -n "click value PMTiles:      "
find data/value_pmtiles -maxdepth 1 -name '*_values.pmtiles' | wc -l

echo "Files smaller than 10 KB, if any:"
find data -name '*.pmtiles' -size -10k -print
find data/value_pmtiles -name '*.pmtiles' -size -10k -print

echo "Done. Test with: python -m http.server 8000"
