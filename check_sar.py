# check_sar.py
# Verify the Sentinel-1 SAR files are valid and check their content
import rasterio
import numpy as np

sar_files = {
    'bengaluru': 'ps2_cloud/data/raw/sentinel1/bengaluru_S1_sar.tif',
    'punjab':    'ps2_cloud/data/raw/sentinel1/punjab_S1_sar.tif',
    'meghalaya': 'ps2_cloud/data/raw/sentinel1/meghalaya_S1_sar.tif',
}

for name, path in sar_files.items():
    print(f"\n{'='*50}")
    print(f"SAR: {name}")
    print(f"{'='*50}")
    with rasterio.open(path) as src:
        print(f"  CRS       : {src.crs}")
        print(f"  Shape     : {src.height} x {src.width} px")
        print(f"  Bounds    : {src.bounds}")
        print(f"  Bands     : {src.count}  ({src.descriptions})")
        print(f"  dtype     : {src.dtypes}")
        print(f"  Transform : {src.transform}")

        data = src.read().astype(np.float32)

    print(f"  Band stats:")
    band_names = ['VV', 'VH']
    for i in range(data.shape[0]):
        b = data[i]
        valid = b[b != 0]
        if len(valid) == 0:
            print(f"    Band {i+1} ({band_names[i] if i < 2 else '?'}): ALL ZEROS — no data!")
            continue
        print(f"    Band {i+1} ({band_names[i] if i < 2 else '?'}): "
              f"min={b.min():.2f}  max={b.max():.2f}  "
              f"mean={valid.mean():.2f}  "
              f"zeros={((b==0).mean()*100):.1f}%")

        # Check if values are in dB range (GEE exports in dB)
        # Typical: VV in [-25, 5] dB, VH in [-30, 0] dB
        if b.min() < -50 or b.max() > 20:
            print(f"    ⚠ Values outside expected dB range [-50, 20] — check units")
        elif b.min() > 0:
            print(f"    ⚠ All positive values — may be linear scale not dB")
        else:
            print(f"    ✓ Values look like dB scale")

print("\n\nSummary:")
print("GEE S1_GRD exports are in dB scale (sigma0 log10).")
print("Expected ranges: VV=[-25, 5] dB, VH=[-30, 0] dB")
print("If values outside these — normalization clips to [0,1] but no data is lost.")
