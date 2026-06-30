# debug_preprocess.py
# Diagnose the two issues before running full preprocess
import rasterio
import os

print("="*60)
print("DIAGNOSTIC 1 — LISS-IV raster CRS and bounds")
print("="*60)

scenes = {
    'bengaluru_cloudy': 'ps2_cloud/data/raw/liss4/bengaluru_cloudy/BAND2.tif',
    'bengaluru_clear':  'ps2_cloud/data/raw/liss4/bengaluru_clear/BAND2.tif',
    'punjab_cloudy':    'ps2_cloud/data/raw/liss4/punjab_cloudy/BAND2.tif',
    'meghalaya_cloudy': 'ps2_cloud/data/raw/liss4/meghalaya_cloudy/BAND2.tif',
}

for name, path in scenes.items():
    with rasterio.open(path) as src:
        print(f"\n{name}:")
        print(f"  CRS        : {src.crs}")
        print(f"  Shape      : {src.height} x {src.width} px")
        print(f"  Bounds     : {src.bounds}")
        print(f"  Transform  : {src.transform}")
        print(f"  dtype      : {src.dtypes[0]}")
        print(f"  nodata     : {src.nodata}")

print("\n" + "="*60)
print("DIAGNOSTIC 2 — BAND_META.txt content")
print("="*60)

meta_path = 'ps2_cloud/data/raw/liss4/bengaluru_cloudy/BAND_META.txt'
print(f"\nFile: {meta_path}")
print("-"*40)
with open(meta_path, encoding='utf-8', errors='ignore') as f:
    content = f.read()
print(content[:3000])  # print first 3000 chars
