# UI clean patch

This patch only changes `index.html`.

Changes:
- Removes the descriptive paragraph under the title.
- Changes the Layer option from `Forest land-cover type (5 classes)` to `Forest land-cover type`.
- Hides the popup close button so the popup line reads `Project: MBCI` without the overlapping `x`.
- Removes the `Loaded: data/...pmtiles` status message.

No raster/value PMTiles need to be regenerated.

Use:
1. Copy this `index.html` into `height_landtype_webmap_10m/` and overwrite the old file.
2. Test locally with `python -m http.server 8000`.
3. Commit and push `index.html` to GitHub.
