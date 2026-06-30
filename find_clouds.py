# find_clouds.py
import rasterio
import numpy as np

with rasterio.open('ps2_cloud/data/raw/liss4/bengaluru_cloudy/BAND2.tif') as src:
    # Read at 1/20 resolution for fast scanning
    h20 = src.height // 20
    w20 = src.width  // 20
    b2 = src.read(1, out_shape=(h20, w20)).astype(np.float32)
    transform = src.transform
    print(f"Full scene: {src.height} x {src.width}")
    print(f"Bounds: {src.bounds}")

with rasterio.open('ps2_cloud/data/raw/liss4/bengaluru_cloudy/BAND4.tif') as src:
    b4 = src.read(1, out_shape=(h20, w20)).astype(np.float32)  # NIR

# Gain from metadata: B2 gain=0.05083, B4 gain=0.03079
b2_refl = b2 * 0.05083 / 1023.0 * 1023.0  # just use raw DN for ratio
# Cloud indicator: high green, low NIR ratio = cloud
# OR simply: high brightness in visible bands
brightness = b2.astype(np.float32)
nonzero = brightness > 0

# Normalize brightness within the valid scene area
valid = brightness[nonzero]
p50 = np.percentile(valid, 50)
p90 = np.percentile(valid, 90)
p10 = np.percentile(valid, 10)
print(f"\nBrightness stats (raw DN):")
print(f"  p10={p10:.0f}  p50={p50:.0f}  p90={p90:.0f}  max={brightness.max():.0f}")

# Cloud = pixels brighter than 80th percentile AND in valid area
# (clouds are relatively bright compared to vegetation in visible bands)
threshold = np.percentile(valid, 75)
cloud_candidate = (brightness > threshold) & nonzero
print(f"  Using threshold: DN > {threshold:.0f}  (75th percentile)")
print(f"  Cloud-candidate pixels: {cloud_candidate.mean()*100:.1f}%")

# Find 30km × 30km window (= 1500×1500 px at 5m, or 75×75 at 1/20 res)
win = 75  # 30km / (5m × 20) = 75 pixels at 1/20 res
best_r, best_c, best_score = 0, 0, 0

for r in range(0, h20 - win, 10):
    for c in range(0, w20 - win, 10):
        patch = cloud_candidate[r:r+win, c:c+win]
        score = patch.mean()
        if score > best_score:
            best_score = score
            best_r, best_c = r, c

# Convert back to full-res pixel
full_r = best_r * 20 + (win * 20) // 2
full_c = best_c * 20 + (win * 20) // 2

utm_x = transform.c + full_c * transform.a
utm_y = transform.f + full_r * transform.e

print(f"\nBest cloudy 30km window:")
print(f"  Center pixel (full res): row={full_r}, col={full_c}")
print(f"  UTM center: X={utm_x:.0f}, Y={utm_y:.0f}")
print(f"  Cloud score: {best_score*100:.1f}%")
print(f"\nUpdate preprocess.py bengaluru crop_utm to:")
print(f"  ({utm_x-15000:.0f}, {utm_y-15000:.0f}, {utm_x+15000:.0f}, {utm_y+15000:.0f})")

# Also check the image to understand the cloud DN range
print(f"\nFor reference — what the visible clouds look like in this crop:")
patch_b2 = b2[best_r:best_r+win, best_c:best_c+win]
print(f"  B2 DN in best window: min={patch_b2.min():.0f} max={patch_b2.max():.0f} mean={patch_b2.mean():.0f}")
