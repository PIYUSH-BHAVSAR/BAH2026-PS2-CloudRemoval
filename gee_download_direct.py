# gee_download_direct.py
# Downloads Sentinel-2 and Sentinel-1 data DIRECTLY to your PC.
# No Google Drive needed. Files saved straight to ps2_cloud/data/raw/
# Project: aidsync-4e460

import ee
import requests
import os

ee.Initialize(project='aidsync-4e460')
print("GEE initialized OK\n")

# ── Output folders ───────────────────────────────────────────────────────
os.makedirs('ps2_cloud/data/raw/sentinel1', exist_ok=True)
os.makedirs('ps2_cloud/data/raw/sentinel2', exist_ok=True)

# ── AOIs ─────────────────────────────────────────────────────────────────
AOIs = {
    'bengaluru': ee.Geometry.Rectangle([76.699, 12.979, 77.550, 13.746]),
    'punjab':    ee.Geometry.Rectangle([74.757, 30.222, 75.723, 30.991]),
    'meghalaya': ee.Geometry.Rectangle([91.228, 24.892, 92.189, 25.645]),
}

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

def download_image(image, aoi, out_path, description):
    """Download a GEE image directly to a local GeoTIFF."""
    if os.path.exists(out_path):
        print(f"  Already exists, skipping: {out_path}")
        return

    print(f"  Downloading {description} ...", end=' ', flush=True)
    try:
        url = image.getDownloadURL({
            'region':  aoi,
            'scale':   10,
            'crs':     'EPSG:4326',
            'format':  'GEO_TIFF',
        })
        r = requests.get(url, timeout=300)
        if r.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(r.content)
            size_mb = len(r.content) / 1024 / 1024
            print(f"OK ({size_mb:.1f} MB) → {out_path}")
        else:
            print(f"FAILED (HTTP {r.status_code})")
    except Exception as e:
        print(f"ERROR: {e}")


for name, aoi in AOIs.items():
    print(f"\n{'='*50}")
    print(f"AOI: {name.upper()}")
    print('='*50)

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

    download_image(
        s2_cloudy, aoi,
        f'ps2_cloud/data/raw/sentinel2/{name}_S2_cloudy.tif',
        f'{name} S2 cloudy'
    )

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

    download_image(
        s2_clear, aoi,
        f'ps2_cloud/data/raw/sentinel2/{name}_S2_clear.tif',
        f'{name} S2 clear'
    )

    # ── Sentinel-1 SAR ────────────────────────────────────────────────
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(*cloudy_dates[name])
        .filterBounds(aoi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select(['VV', 'VH'])
        .mean()
        .toFloat())

    download_image(
        s1, aoi,
        f'ps2_cloud/data/raw/sentinel1/{name}_S1_sar.tif',
        f'{name} S1 SAR'
    )

print("\n\nAll downloads complete.")
print("Files saved to:")
print("  ps2_cloud/data/raw/sentinel2/  (6 files)")
print("  ps2_cloud/data/raw/sentinel1/  (3 files)")
