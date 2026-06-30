# download_dem.py
# Downloads SRTM 1-arc-second DEM tiles from OpenTopography (no login needed)
# then merges + clips to each AOI bounding box.

import requests, zipfile, os, struct
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds as tfb
from rasterio.crs import CRS
from rasterio.windows import from_bounds as wfb

os.makedirs('ps2_cloud/data/raw/dem/tmp', exist_ok=True)

def download_srtm_tile(lat, lon, tmp_dir='ps2_cloud/data/raw/dem/tmp'):
    """
    Download SRTM GL1 (1 arc-second, ~30m) tile from OpenTopography AWS mirror.
    lat, lon = bottom-left integer corner. e.g. lat=12, lon=76 → N12E076
    Returns path to extracted .tif, or None on failure.
    """
    ns   = 'N' if lat >= 0 else 'S'
    ew   = 'E' if lon >= 0 else 'W'
    tile = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"

    tif_path = os.path.join(tmp_dir, f"{tile}.tif")
    if os.path.exists(tif_path):
        print(f"  Cached: {tile}")
        return tif_path

    # OpenTopography AWS public bucket — reliable, no auth
    url = f"https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm/{tile}.tif"
    print(f"  Downloading {tile} ...", end=' ', flush=True)
    r = requests.get(url, timeout=120)
    if r.status_code == 200:
        with open(tif_path, 'wb') as f:
            f.write(r.content)
        print(f"OK ({len(r.content)//1024} KB)")
        return tif_path
    else:
        print(f"FAILED (HTTP {r.status_code})")
        return None

def make_dem(bounds, output_path):
    """
    Download all SRTM tiles covering bounds, merge, clip, save as float32 GeoTIFF.
    bounds = (west, south, east, north)
    """
    west, south, east, north = bounds

    # Collect integer-degree tile corners that overlap the bounds
    tile_paths = []
    for lat in range(int(south), int(north) + 1):
        for lon in range(int(west), int(east) + 1):
            p = download_srtm_tile(lat, lon)
            if p:
                tile_paths.append(p)

    if not tile_paths:
        raise RuntimeError(f"No tiles downloaded for bounds {bounds}. Check internet connection.")

    # Open all tiles
    src_files = [rasterio.open(p) for p in tile_paths]

    # Merge into one array
    merged, merge_transform = merge(src_files)

    # Clip to exact AOI bounds using pixel window
    win = wfb(west, south, east, north, merge_transform)
    row_start = max(0, int(win.row_off))
    row_stop  = min(merged.shape[1], int(win.row_off + win.height) + 1)
    col_start = max(0, int(win.col_off))
    col_stop  = min(merged.shape[2], int(win.col_off + win.width)  + 1)

    clipped = merged[:, row_start:row_stop, col_start:col_stop].astype(np.float32)

    # Replace SRTM nodata (-32768) with 0
    clipped[clipped < -1000] = 0.0

    out_transform = tfb(west, south, east, north, clipped.shape[2], clipped.shape[1])

    out_meta = src_files[0].meta.copy()
    out_meta.update({
        'driver':    'GTiff',
        'dtype':     'float32',
        'nodata':    None,
        'height':    clipped.shape[1],
        'width':     clipped.shape[2],
        'transform': out_transform,
        'crs':       CRS.from_epsg(4326),
        'count':     1,
    })

    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(clipped)

    for s in src_files:
        s.close()

    h, w = clipped.shape[1], clipped.shape[2]
    print(f"  → Saved: {output_path}  ({h}×{w} px, "
          f"min={clipped.min():.0f}m max={clipped.max():.0f}m)")


# ── Run for all 3 AOIs ───────────────────────────────────────────────────

print("\nBengaluru DEM...")
make_dem((76.699, 12.979, 77.550, 13.746),
          'ps2_cloud/data/raw/dem/dem_bengaluru.tif')

print("\nPunjab DEM...")
make_dem((74.757, 30.222, 75.723, 30.991),
          'ps2_cloud/data/raw/dem/dem_punjab.tif')

print("\nMeghalaya DEM...")
make_dem((91.228, 24.892, 92.189, 25.645),
          'ps2_cloud/data/raw/dem/dem_meghalaya.tif')

print("\nAll 3 DEMs done.")
