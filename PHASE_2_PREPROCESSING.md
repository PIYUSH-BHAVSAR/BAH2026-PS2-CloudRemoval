# PHASE 2 — Preprocessing
## PS2 | BAH 2026
### Goal: Raw downloads → co-registered, normalized, masked 11-channel stacks ready for training

---

## What This Phase Produces

```
INPUT  (from Phase 1)                OUTPUT (going into Phase 3)
──────────────────────────────────   ──────────────────────────────────────
data/raw/liss4/*/BAND{2,3,4}.tif  →  data/processed/{aoi}_stack.npy
data/raw/sentinel2/*_S2_*.tif     →  data/processed/{aoi}_clear.npy
data/raw/sentinel1/*_S1_sar.tif   →  (both are float32 numpy arrays)
data/raw/dem/dem_*.tif
```

The stack is an **11-channel float32 numpy array** per AOI:

```
Channel  Content                           Source
───────  ────────────────────────────────  ──────────────────────
0        Green band (cloudy scene)         LISS-IV / S2
1        Red band (cloudy scene)           LISS-IV / S2
2        NIR band (cloudy scene)           LISS-IV / S2
3        Cloud+shadow mask (normalized)    s2cloudless + solar geometry
4        SAR VV (normalized dB)            Sentinel-1 via GEE
5        SAR VH (normalized dB)            Sentinel-1 via GEE
6        Green band (clear reference)      LISS-IV / S2
7        Red band (clear reference)        LISS-IV / S2
8        NIR band (clear reference)        LISS-IV / S2
9        Change detection mask (NDVI)      computed
10       DEM elevation (normalized)        SRTM
```

---

## Full Preprocessing Script

Save as `scripts/preprocess/run_all_preprocessing.py`.
Run once per AOI after Phase 1 is complete.

```python
# scripts/preprocess/run_all_preprocessing.py
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Load and normalize optical imagery
# ═══════════════════════════════════════════════════════════════

def load_optical(path):
    """
    Load a 3-band optical GeoTIFF and return float32 array in [0, 1].

    For GEE Sentinel-2 exports: already divided by 10000 during export,
    so values are already in [0,1] — just clip to remove any artifacts.

    For raw LISS-IV: run parse_liss4_meta + dn_to_reflectance_liss4 first
    (see PHASE_1_DATA_COLLECTION.md), then save as float32 GeoTIFF before
    calling this function.
    """
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)   # C x H x W
        meta = src.meta.copy()
    return np.clip(data, 0, 1), meta


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Load and normalize SAR
# ═══════════════════════════════════════════════════════════════

def load_sar(path):
    """
    Load SAR GeoTIFF (GEE export, already in dB scale).
    Normalize VV and VH bands to [0, 1] using known physical ranges.

    VV range in dB: [-25,  5] — covers urban and vegetation backscatter
    VH range in dB: [-30,  0] — volume scattering from vegetation
    """
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)   # 2 x H x W (VV, VH)
    vv_norm = np.clip((data[0] - (-25)) / (5   - (-25)), 0, 1)
    vh_norm = np.clip((data[1] - (-30)) / (0   - (-30)), 0, 1)
    return np.stack([vv_norm, vh_norm], axis=0)   # 2 x H x W


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Co-register all sources to the optical reference grid
# ═══════════════════════════════════════════════════════════════

def coregister(source_path, reference_path, output_path,
               method=Resampling.bilinear):
    """
    Reproject source image to EXACTLY match reference image
    (same CRS, same transform, same pixel grid).

    CRITICAL: must be done before stacking — a single pixel misalignment
    at 5.8m LISS-IV resolution = 5.8m error in SAR-guided reconstruction.

    method: bilinear for continuous data (SAR, DEM, optical)
            nearest  for discrete data (masks) — preserves exact class values
    """
    with rasterio.open(reference_path) as ref:
        ref_meta   = ref.meta.copy()
        ref_crs    = ref.crs
        ref_tr     = ref.transform
        ref_h      = ref.height
        ref_w      = ref.width

    with rasterio.open(source_path) as src:
        src_data = src.read().astype(np.float32)
        src_crs  = src.crs
        src_tr   = src.transform

    n_bands = src_data.shape[0]
    out = np.zeros((n_bands, ref_h, ref_w), dtype=np.float32)

    for b in range(n_bands):
        reproject(
            source=src_data[b],       destination=out[b],
            src_transform=src_tr,     src_crs=src_crs,
            dst_transform=ref_tr,     dst_crs=ref_crs,
            resampling=method,
        )

    out_meta = ref_meta.copy()
    out_meta.update({'count': n_bands, 'dtype': 'float32'})
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(out)

    print(f"  Co-registered: {os.path.basename(source_path)}")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Cloud + shadow mask (s2cloudless + solar geometry)
# ═══════════════════════════════════════════════════════════════

def make_cloud_mask(cloudy_rgb_path, sun_azimuth_deg=135.0,
                    sun_elevation_deg=50.0, threshold=0.4):
    """
    Generate cloud+shadow mask using s2cloudless neural network detector
    plus solar-geometry-based shadow estimation.

    Why s2cloudless instead of a brightness threshold:
      - Brightness threshold misses thin cloud, cirrus, and haze
      - In NER monsoon scenes, those can be 20-40% of masked pixels
      - s2cloudless uses a trained neural network probability map
      - Takes ~3 minutes per scene — worth every second

    3-band workaround:
      GEE exports only B3/B4/B8. s2cloudless needs 10 bands.
      We fill missing bands with NIR (conservative proxy).
      This is still far more accurate than any threshold rule.

    Shadow detection:
      Shadows are NOT flagged by s2cloudless — they look like dark terrain.
      We estimate shadow location using the sun's azimuth and elevation.
      Confirmation: candidate shadow pixels must also be dark in NIR.

    Returns:
      combined_mask: 1 x H x W, float32
        0.0 = clear pixel
        0.5 = shadow (original value 1, divided by 2 to normalize)
        1.0 = cloud  (original value 2, divided by 2 to normalize)
    """
    from s2cloudless import S2PixelCloudDetector

    with rasterio.open(cloudy_rgb_path) as src:
        rgb = src.read().astype(np.float32)   # 3 x H x W, [0,1]

    H, W = rgb.shape[1], rgb.shape[2]
    nir = rgb[2:3]   # 1 x H x W

    # Build 10-band input for s2cloudless
    # Order: B1,B2,B4,B5,B8,B8A,B9,B10,B11,B12
    # Available: B3(Green)=ch0, B4(Red)=ch1, B8(NIR)=ch2
    s2_10band = np.concatenate([
        nir,         # B1  proxy (NIR)
        rgb[0:1],    # B2  proxy (Green)
        rgb[1:2],    # B4  (Red — actual)
        nir,         # B5  proxy (NIR)
        nir,         # B8  (NIR — actual)
        nir,         # B8A proxy
        nir,         # B9  proxy
        nir,         # B10 proxy
        rgb[0:1],    # B11 proxy (Green)
        rgb[0:1],    # B12 proxy (Green)
    ], axis=0)   # 10 x H x W

    # s2cloudless input format: N x H x W x C
    inp = s2_10band.transpose(1, 2, 0)[np.newaxis]   # 1 x H x W x 10

    detector = S2PixelCloudDetector(
        threshold=threshold,
        average_over=4,       # smoothing kernel radius
        dilation_size=2,      # expand cloud edges by 2px
        all_bands=False,
    )
    cloud_prob = detector.get_cloud_probability_maps(inp)[0]   # H x W
    cloud_mask = (cloud_prob > threshold).astype(np.uint8)     # 0 or 1

    # ── Shadow mask via solar geometry ──────────────────────────────────
    # Shadow is cast in the direction OPPOSITE to sun azimuth
    shadow_az   = (sun_azimuth_deg + 180) % 360
    # Pixel offset: assume cloud height ~2km, tan(elevation) = h/d → d = h/tan(e)
    dist_px     = int(20 / np.tan(np.radians(max(sun_elevation_deg, 5))))
    dx = int(dist_px * np.sin(np.radians(shadow_az)))
    dy = int(dist_px * np.cos(np.radians(shadow_az)))

    shadow_candidate = np.zeros_like(cloud_mask)
    for i in range(H):
        for j in range(W):
            si, sj = i + dy, j + dx
            if 0 <= si < H and 0 <= sj < W:
                shadow_candidate[i, j] = cloud_mask[si, sj]

    # Confirm shadow: candidate must also be dark in NIR (<0.15)
    # (shadows strongly absorb NIR; dark soil/water won't be shadow candidates)
    dark_nir    = (rgb[2] < 0.15).astype(np.uint8)
    shadow_mask = (shadow_candidate & dark_nir).astype(np.uint8)

    # Combine and normalize
    combined = np.zeros_like(cloud_mask)
    combined[shadow_mask == 1] = 1   # shadow = 1
    combined[cloud_mask  == 1] = 2   # cloud = 2 (overwrites shadow)
    combined_norm = (combined / 2.0).astype(np.float32)   # → [0, 0.5, 1.0]

    cloud_pct  = (cloud_mask  > 0).mean() * 100
    shadow_pct = (shadow_mask > 0).mean() * 100
    print(f"  Cloud: {cloud_pct:.1f}%  Shadow: {shadow_pct:.1f}%")

    return combined_norm[np.newaxis]   # 1 x H x W


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Change detection mask (NDVI-based)
# ═══════════════════════════════════════════════════════════════

def make_change_mask(cloudy_rgb, clear_rgb, threshold=0.2):
    """
    Detect pixels where land surface has changed between the clear reference
    date and the cloudy scene date.

    Changed pixels should NOT use the temporal reference for reconstruction —
    the model must rely on SAR and spatial context instead.

    threshold=0.2: 0.2 NDVI units = significant vegetation change
      Tune to 0.15 for conservative (flag more pixels as changed)
      Tune to 0.25 if too many false positives (e.g., seasonal agriculture)

    Returns: 1 x H x W, float32, 1=changed (avoid temporal), 0=stable (use temporal)
    """
    eps = 1e-8
    ndvi_cloudy = (cloudy_rgb[2] - cloudy_rgb[1]) / (cloudy_rgb[2] + cloudy_rgb[1] + eps)
    ndvi_clear  = (clear_rgb[2]  - clear_rgb[1])  / (clear_rgb[2]  + clear_rgb[1]  + eps)
    diff        = np.abs(ndvi_cloudy - ndvi_clear)
    change_mask = (diff > threshold).astype(np.float32)

    changed_pct = change_mask.mean() * 100
    print(f"  Change detection: {changed_pct:.1f}% of pixels flagged as changed")
    return change_mask[np.newaxis]   # 1 x H x W


# ═══════════════════════════════════════════════════════════════
# STEP 6 — DEM normalization
# ═══════════════════════════════════════════════════════════════

def load_dem(dem_path, reference_path):
    """
    Co-register DEM to optical grid and normalize elevation to [0,1].
    India elevation: ~-100m (coastal) to ~3000m (Himalayan foothills).
    """
    coreg_path = dem_path.replace('.tif', '_coreg.tif')
    coregister(dem_path, reference_path, coreg_path, method=Resampling.bilinear)
    with rasterio.open(coreg_path) as src:
        dem = src.read(1).astype(np.float32)
    dem_norm = np.clip((dem - (-100)) / (3000 - (-100)), 0, 1)
    return dem_norm[np.newaxis]   # 1 x H x W


# ═══════════════════════════════════════════════════════════════
# STEP 7 — Stack everything into 11-channel tensor
# ═══════════════════════════════════════════════════════════════

def build_stack(aoi_name,
                processed_dir='ps2_cloud/data/processed',
                raw_dem_dir='ps2_cloud/data/raw/dem',
                sun_azimuth_deg=135.0,
                sun_elevation_deg=50.0):
    """
    Build the complete 11-channel input stack for one AOI.

    CRITICAL ORDER — do not rearrange:
      1. Load optical reference first (defines the target grid)
      2. Co-register SAR to optical grid  ← BEFORE load_sar (file doesn't exist yet)
      3. Load co-registered SAR
      4. Generate masks (cloud, shadow, change) from loaded arrays
      5. Co-register DEM
      6. Resize any remaining shape mismatches
      7. Concatenate and save

    sun_azimuth_deg, sun_elevation_deg:
      Get from image metadata or from GEE:
        img.get('MEAN_SOLAR_AZIMUTH_ANGLE').getInfo()
        img.get('MEAN_SOLAR_ZENITH_ANGLE').getInfo()  (then convert: elev = 90 - zenith)
      Defaults (135°, 50°) are reasonable for India in July.
    """
    print(f"\n=== Building stack: {aoi_name} ===")

    # ── Load optical ────────────────────────────────────────────────────
    cloudy_path = f'{processed_dir}/{aoi_name}_S2_cloudy.tif'
    clear_path  = f'{processed_dir}/{aoi_name}_S2_clear.tif'
    cloudy, meta = load_optical(cloudy_path)
    clear,  _    = load_optical(clear_path)
    H, W = cloudy.shape[1], cloudy.shape[2]
    print(f"  Optical loaded: {cloudy.shape} — {H}×{W} pixels")

    # ── Co-register SAR FIRST, then load ────────────────────────────────
    sar_raw   = f'{processed_dir}/{aoi_name}_S1_sar.tif'
    sar_coreg = f'{processed_dir}/{aoi_name}_S1_sar_coreg.tif'
    coregister(sar_raw, cloudy_path, sar_coreg, method=Resampling.bilinear)
    sar = load_sar(sar_coreg)

    # ── Cloud + shadow mask ──────────────────────────────────────────────
    combined_mask = make_cloud_mask(
        cloudy_path,
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
    )

    # ── Change detection ─────────────────────────────────────────────────
    change_mask = make_change_mask(cloudy, clear)

    # ── DEM (co-registers internally) ───────────────────────────────────
    dem = load_dem(f'{raw_dem_dir}/dem_{aoi_name}.tif', cloudy_path)

    # ── Resize any remaining shape mismatches ───────────────────────────
    import cv2
    def resize_to(arr, h, w):
        if arr.shape[1] == h and arr.shape[2] == w:
            return arr
        return np.stack([cv2.resize(arr[c], (w, h), interpolation=cv2.INTER_LINEAR)
                         for c in range(arr.shape[0])])

    sar           = resize_to(sar,           H, W)
    combined_mask = resize_to(combined_mask, H, W)
    change_mask   = resize_to(change_mask,   H, W)
    dem           = resize_to(dem,           H, W)

    # ── Stack and validate ───────────────────────────────────────────────
    # Channel layout: [cloudy_G, cloudy_R, cloudy_NIR, cloud_mask,
    #                  SAR_VV, SAR_VH, clear_G, clear_R, clear_NIR,
    #                  change_mask, DEM]
    stack = np.concatenate([
        cloudy,         # ch 0-2
        combined_mask,  # ch 3
        sar,            # ch 4-5
        clear,          # ch 6-8
        change_mask,    # ch 9
        dem,            # ch 10
    ], axis=0)

    assert stack.shape[0] == 11, \
        f"Expected 11 channels, got {stack.shape[0]}"

    # Clean any NaN/Inf from reprojection artifacts
    stack = np.nan_to_num(stack, nan=0.0, posinf=1.0, neginf=0.0)

    # ── Save ─────────────────────────────────────────────────────────────
    out_stack = f'{processed_dir}/{aoi_name}_stack.npy'
    out_clear = f'{processed_dir}/{aoi_name}_clear.npy'
    np.save(out_stack, stack)
    np.save(out_clear, clear)

    cloud_pct = (combined_mask > 0.4).mean() * 100   # >0.4 = cloud class
    print(f"  Saved: {out_stack}")
    print(f"  Stack shape: {stack.shape} | Cloud coverage: {cloud_pct:.1f}%")

    return stack, clear


# ═══════════════════════════════════════════════════════════════
# RUN — process all 3 AOIs
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import shutil

    # Copy GEE downloads to processed folder (they're already normalized)
    # If using raw LISS-IV, run dn_to_reflectance first and save as float32 GeoTIFF
    for aoi in ['bengaluru', 'punjab', 'meghalaya']:
        for suffix in ['S2_cloudy', 'S2_clear', 'S1_sar']:
            src = f'ps2_cloud/data/raw/sentinel2/{aoi}_{suffix}.tif' \
                  if 'S2' in suffix else \
                  f'ps2_cloud/data/raw/sentinel1/{aoi}_{suffix}.tif'
            dst = f'ps2_cloud/data/processed/{aoi}_{suffix}.tif'
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                print(f"Copied: {os.path.basename(src)}")

    # Build 11-channel stacks
    for aoi in ['bengaluru', 'punjab', 'meghalaya']:
        build_stack(aoi)

    print("\n=== Preprocessing complete ===")
    print("Output files:")
    for aoi in ['bengaluru', 'punjab', 'meghalaya']:
        s = np.load(f'ps2_cloud/data/processed/{aoi}_stack.npy')
        print(f"  {aoi}_stack.npy: {s.shape}  "
              f"min={s.min():.3f}  max={s.max():.3f}  "
              f"nan={np.isnan(s).sum()}")
```

---

## Co-registration Sanity Check (run after preprocessing)

```python
# Quick visual check — plot cloudy RGB + SAR VV overlay
# If they're aligned, field boundaries in SAR should match optical features
import numpy as np
import matplotlib.pyplot as plt

for aoi in ['bengaluru', 'punjab', 'meghalaya']:
    stack = np.load(f'ps2_cloud/data/processed/{aoi}_stack.npy')

    cloudy_rgb = stack[:3].transpose(1, 2, 0)        # H x W x 3
    sar_vv     = stack[4]                             # H x W
    cloud_mask = stack[3]                             # H x W

    # Crop to 512×512 for quick check
    h, w = min(512, stack.shape[1]), min(512, stack.shape[2])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{aoi} — co-registration check')

    axes[0].imshow(np.clip(cloudy_rgb[:h, :w, [1,0,2]], 0, 1))   # R,G,B order
    axes[0].set_title('Cloudy optical (R-G-B)')
    axes[0].axis('off')

    axes[1].imshow(sar_vv[:h, :w], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('SAR VV (should align with optical)')
    axes[1].axis('off')

    axes[2].imshow(cloud_mask[:h, :w], cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[2].set_title('Cloud+shadow mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'ps2_cloud/outputs/figures/{aoi}_coregistration_check.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {aoi}_coregistration_check.png")

# What to look for:
#   GOOD: SAR VV shows bright values where optical shows buildings/roads
#   BAD:  SAR features are spatially offset from optical features
#   If bad: re-run co-registration, check CRS of both files
```

---

## Phase 2 Checklist

```
PREPROCESSING
  [ ] run_all_preprocessing.py ran for all 3 AOIs without errors
  [ ] SAR co-registration ran BEFORE load_sar (order in build_stack)
  [ ] 3 _stack.npy files saved in data/processed/
  [ ] 3 _clear.npy files saved in data/processed/
  [ ] No NaN values (check printed: nan=0)
  [ ] Values in [0,1] range (check printed: min≥0, max≤1)

CLOUD MASKING
  [ ] s2cloudless ran without errors
  [ ] Cloud coverage printed per AOI (expect 40-90% for cloudy scenes)
  [ ] Shadow coverage printed (expect 5-20%)

CO-REGISTRATION CHECK
  [ ] 3 coregistration_check.png figures saved and visually inspected
  [ ] SAR features align with optical features (no spatial offset)

STACK VALIDATION
  [ ] bengaluru_stack.npy shape: (11, ~7000, ~7000) or similar
  [ ] punjab_stack.npy shape: (11, ~H, ~W)
  [ ] meghalaya_stack.npy shape: (11, ~H, ~W)
  [ ] All channel ranges correct (verify with the print check above)
```

---

## Channel Value Ranges (expected after preprocessing)

| Channel | Content | Expected Range |
|---|---|---|
| 0–2 | Cloudy optical G/R/NIR | [0, 1] |
| 3 | Cloud+shadow mask | {0.0, 0.5, 1.0} |
| 4–5 | SAR VV, VH normalized | [0, 1] |
| 6–8 | Clear reference G/R/NIR | [0, 1] |
| 9 | Change mask | {0.0, 1.0} |
| 10 | DEM elevation | [0, 1] |

---

## Storage After Phase 2

```
data/processed/bengaluru_stack.npy   ~2.8 GB  (11 × 7000 × 7000 × 4 bytes)
data/processed/bengaluru_clear.npy   ~0.8 GB  (3  × 7000 × 7000 × 4 bytes)
(same for punjab and meghalaya)
────────────────────────────────────────────────────────────────
TOTAL new                            ~11 GB
```

> Tip: if RAM is tight when loading these for tiling, use `np.load(..., mmap_mode='r')`
> to memory-map the file rather than loading the full array into RAM.

---

**→ Continue to PHASE_3_MODEL.md**
