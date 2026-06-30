# gee_export_all.py
# Exports Sentinel-2 and Sentinel-1 to Google Drive > PS2_BAH2026 folder.
# Project: aidsync-4e460
#
# HOW TO USE:
#   1. Run this script: python gee_export_all.py
#   2. Monitor tasks:   https://code.earthengine.google.com  → Tasks tab (top right)
#   3. Wait for all 9 tasks to show COMPLETED (green) — takes 20-40 min
#   4. Open Google Drive → folder PS2_BAH2026 → download all 9 GeoTIFFs
#   5. Move files:
#        *_S1_sar.tif   → ps2_cloud/data/raw/sentinel1/
#        *_S2_*.tif     → ps2_cloud/data/raw/sentinel2/

import ee

ee.Initialize(project='aidsync-4e460')
print("GEE initialized OK\n")

# ── AOIs matched to actual LISS-IV scene coverage ───────────────────────
AOIs = {
    'bengaluru': ee.Geometry.Rectangle([76.699, 12.979, 77.550, 13.746]),
    'punjab':    ee.Geometry.Rectangle([74.757, 30.222, 75.723, 30.991]),
    'meghalaya': ee.Geometry.Rectangle([91.228, 24.892, 92.189, 25.645]),
}

# ── Date ranges matched to your LISS-IV scenes ──────────────────────────
cloudy_dates = {
    'bengaluru': ('2026-05-01', '2026-06-30'),
    'punjab':    ('2025-06-01', '2025-07-31'),
    'meghalaya': ('2025-06-01', '2025-07-31'),
}

clear_dates = {
    'bengaluru': ('2025-11-01', '2026-02-28'),
    'punjab':    ('2025-10-01', '2026-01-31'),
    'meghalaya': ('2025-11-01', '2026-02-28'),
}

tasks_started = []

for name, aoi in AOIs.items():
    print(f"Setting up exports for: {name.upper()}")

    # ── Sentinel-2 CLOUDY ─────────────────────────────────────────────
    s2_cloudy = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(*cloudy_dates[name])
        .filterBounds(aoi)
        .filter(ee.Filter.gt('CLOUDY_PIXEL_PERCENTAGE', 30))
        .sort('CLOUDY_PIXEL_PERCENTAGE', False)
        .first()
        .select(['B3', 'B4', 'B8'])
        .divide(10000)
        .toFloat())

    # ── Sentinel-2 CLEAR ──────────────────────────────────────────────
    s2_clear = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(*clear_dates[name])
        .filterBounds(aoi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
        .sort('CLOUDY_PIXEL_PERCENTAGE')
        .first()
        .select(['B3', 'B4', 'B8'])
        .divide(10000)
        .toFloat())

    # ── Sentinel-1 SAR ────────────────────────────────────────────────
    # Already calibrated + terrain corrected in dB scale
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(*cloudy_dates[name])
        .filterBounds(aoi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select(['VV', 'VH'])
        .mean()
        .toFloat())

    # ── Export to Google Drive ────────────────────────────────────────
    for img, suffix in [
        (s2_cloudy, 'S2_cloudy'),
        (s2_clear,  'S2_clear'),
        (s1,        'S1_sar'),
    ]:
        desc = f'{name}_{suffix}'
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=desc,
            folder='PS2_BAH2026',
            fileNamePrefix=desc,
            region=aoi,
            scale=10,
            crs='EPSG:4326',
            maxPixels=1e9,
            fileFormat='GeoTIFF',
        )
        task.start()
        tasks_started.append(desc)
        print(f"  ✓ Task started: {desc}")

print(f"\n{'='*50}")
print(f"All {len(tasks_started)} export tasks started.")
print(f"{'='*50}")
print()
print("NEXT STEPS:")
print("1. Open: https://code.earthengine.google.com")
print("2. Click 'Tasks' tab (top right of the page)")
print("3. Wait for all 9 tasks to show COMPLETED (turns green)")
print("   Each task takes 10-30 minutes")
print()
print("4. Open Google Drive: https://drive.google.com")
print("   Go to folder: PS2_BAH2026")
print("   Download all 9 GeoTIFF files to your PC")
print()
print("5. Move downloaded files:")
print("   bengaluru_S1_sar.tif  →  ps2_cloud/data/raw/sentinel1/")
print("   punjab_S1_sar.tif     →  ps2_cloud/data/raw/sentinel1/")
print("   meghalaya_S1_sar.tif  →  ps2_cloud/data/raw/sentinel1/")
print("   *_S2_cloudy.tif       →  ps2_cloud/data/raw/sentinel2/")
print("   *_S2_clear.tif        →  ps2_cloud/data/raw/sentinel2/")
print()
print("Files that will appear in Drive:")
for t in tasks_started:
    print(f"   {t}.tif")
