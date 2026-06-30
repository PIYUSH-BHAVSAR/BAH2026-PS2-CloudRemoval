# preprocess.py
# Phase 2 — Full preprocessing pipeline for PS2 BAH 2026
# Inputs  : Raw LISS-IV bands, Sentinel-1 SAR, Sentinel-2, DEM
# Outputs : {aoi}_stack.npy (11-channel input) + {aoi}_clear.npy (3-channel target)
#
# Run: python preprocess.py
# Time: ~10-20 minutes per AOI (dominated by s2cloudless on large arrays)

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import warnings
warnings.filterwarnings('ignore')

# ── Output directory ──────────────────────────────────────────────────────
os.makedirs('ps2_cloud/data/processed', exist_ok=True)
os.makedirs('ps2_cloud/outputs/figures', exist_ok=True)

# ── AOI configuration ─────────────────────────────────────────────────────
# Crop bounds in UTM meters — matching actual LISS-IV raster CRS
# Bengaluru/Punjab: EPSG:32643 (UTM Zone 43N)
# Meghalaya:        EPSG:32646 (UTM Zone 46N)
#
# Crop = intersection of cloudy + clear scene bounds, inset slightly
# Bengaluru cloudy: left=684278, bottom=1436267, right=773973, top=1520187
# Bengaluru clear:  left=685998, bottom=1436212, right=775768, top=1520282
# Intersection:     left=685998, bottom=1436267, right=773973, top=1520187
# Inset 500m each side to avoid edge artifacts

AOI_CONFIG = {
    'bengaluru': {
        'liss4_cloudy_dir': 'ps2_cloud/data/raw/liss4/bengaluru_cloudy',
        'liss4_clear_dir':  'ps2_cloud/data/raw/liss4/bengaluru_clear',
        'sar_path':  'ps2_cloud/data/raw/sentinel1/bengaluru_S1_sar.tif',
        's2_cloudy': 'ps2_cloud/data/raw/sentinel2/bengaluru_S2_cloudy.tif',
        'dem_path':  'ps2_cloud/data/raw/dem/dem_bengaluru.tif',
        # 30km × 30km crop centered on scene — fits in 16GB RAM
        # Full scene: left=684278 bottom=1436267 right=773973 top=1520187
        # Center: ~729125, 1478227  → ±15km each side
        'crop_utm': (738029, 1490000, 768029, 1520000),  # 30km × 30km, cloudy area (find_clouds.py)
        'liss4_crs': 'EPSG:32643',
    },
    'punjab': {
        'liss4_cloudy_dir': 'ps2_cloud/data/raw/liss4/punjab_cloudy',
        'liss4_clear_dir':  'ps2_cloud/data/raw/liss4/punjab_clear',
        'sar_path':  'ps2_cloud/data/raw/sentinel1/punjab_S1_sar.tif',
        's2_cloudy': 'ps2_cloud/data/raw/sentinel2/punjab_S2_cloudy.tif',
        'dem_path':  'ps2_cloud/data/raw/dem/dem_punjab.tif',
        # Full scene: left=476893 bottom=3343597 right=569048 top=3428632
        # Center: ~522970, 3386114 → ±15km
        'crop_utm': (490000, 3398000, 520000, 3428000),  # 30km × 30km, upper area (cloudy)
        'liss4_crs': 'EPSG:32643',
    },
    'meghalaya': {
        'liss4_cloudy_dir': 'ps2_cloud/data/raw/liss4/meghalaya_cloudy',
        'liss4_clear_dir':  'ps2_cloud/data/raw/liss4/meghalaya_clear',
        'sar_path':  'ps2_cloud/data/raw/sentinel1/meghalaya_S1_sar.tif',
        's2_cloudy': 'ps2_cloud/data/raw/sentinel2/meghalaya_S2_cloudy.tif',
        'dem_path':  'ps2_cloud/data/raw/dem/dem_meghalaya.tif',
        # Full scene: left=325958 bottom=2754102 right=418083 top=2836432
        # Center: ~372020, 2795267 → ±15km
        'crop_utm': (340000, 2806000, 370000, 2836000),  # 30km × 30km, upper area (cloudy)
        'liss4_crs': 'EPSG:32646',
    },
}

# ── ESUN constants (ResourceSat-2 Data Users Handbook) ───────────────────
ESUN = {'2': 1848.0, '3': 1549.0, '4': 1044.0}  # W/m²/µm, Band2=G, Band3=R, Band4=NIR


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Parse BAND_META.txt
# ═══════════════════════════════════════════════════════════════════════════

def parse_band_meta(meta_path):
    """
    Parse BAND_META.txt from Bhoonidhi LISS-IV download.
    Actual format: flat key=value file, band-specific keys prefixed with B2/B3/B4.
    e.g. B2_Lmax=52.0000, B3_Lmin=0.0000, SunElevationAtCenter=62.26
    Note: sun azimuth key has ISRO typo: SunAziumthAtCenter (not SunAzimuthAtCenter)
    """
    params = {}
    with open(meta_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, val = line.split('=', 1)
            params[key.strip()] = val.strip()

    sun_elev = float(params.get('SunElevationAtCenter', 62.0))
    # ISRO typo in the key name — handle both spellings
    sun_az   = float(params.get('SunAziumthAtCenter',
                    params.get('SunAzimuthAtCenter', 135.0)))

    # Earth-sun distance from acquisition date
    try:
        import ephem, datetime
        date_str = params.get('DateOfPass', '01-JAN-2025')
        acq_date = datetime.datetime.strptime(date_str, '%d-%b-%Y')
        obs = ephem.Observer()
        obs.date = acq_date
        sun = ephem.Sun(); sun.compute(obs)
        earth_sun_dist = float(sun.earth_distance)
    except Exception:
        earth_sun_dist = 1.0

    # Build per-band calibration — keys are B2_Lmax, B3_Lmin etc.
    result = {}
    for band_id in ['2', '3', '4']:
        lmax = float(params.get(f'B{band_id}_Lmax', 52.0))
        lmin = float(params.get(f'B{band_id}_Lmin', 0.0))
        qmax = 1023.0  # 10-bit LISS-IV
        qmin = 0.0
        gain = (lmax - lmin) / (qmax - qmin)
        bias = lmin - gain * qmin
        result[band_id] = {
            'gain':           gain,
            'bias':           bias,
            'esun':           ESUN.get(band_id, 1500.0),
            'sun_elevation':  sun_elev,
            'sun_azimuth':    sun_az,
            'earth_sun_dist': earth_sun_dist,
        }

    print(f"    Parsed: SunElev={sun_elev:.1f}°  SunAz={sun_az:.1f}°  "
          f"d={earth_sun_dist:.4f} AU")
    print(f"    B2: gain={result['2']['gain']:.5f}  B3: gain={result['3']['gain']:.5f}  "
          f"B4: gain={result['4']['gain']:.5f}")

    return result, sun_az, sun_elev


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — DN → Surface Reflectance
# ═══════════════════════════════════════════════════════════════════════════

def dn_to_reflectance(dn_array, gain, bias, esun, sun_elevation, earth_sun_dist):
    """Convert LISS-IV DN → surface reflectance [0, 1]."""
    dn = dn_array.astype(np.float32)
    radiance = dn * gain + bias
    radiance = np.clip(radiance, 0, None)
    zenith_rad = np.radians(90.0 - sun_elevation)
    cos_z = max(np.cos(zenith_rad), 0.01)  # avoid division by zero
    refl = (np.pi * radiance * earth_sun_dist**2) / (esun * cos_z)
    return np.clip(refl, 0.0, 1.0).astype(np.float32)


def load_liss4_scene(scene_dir, meta_params):
    """
    Load and convert all 3 LISS-IV bands to reflectance.
    meta_params: dict from parse_band_meta keyed by band_id '2','3','4'
    Returns: (green, red, nir) as float32 arrays, rasterio meta
    """
    band_map = {'2': 'green', '3': 'red', '4': 'nir'}
    bands = {}
    meta = None

    for band_num, band_name in band_map.items():
        tif_path = os.path.join(scene_dir, f'BAND{band_num}.tif')
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"Missing: {tif_path}")

        with rasterio.open(tif_path) as src:
            dn = src.read(1).astype(np.float32)
            if meta is None:
                meta = src.meta.copy()

        if band_num in meta_params:
            p = meta_params[band_num]
            refl = dn_to_reflectance(dn, p['gain'], p['bias'],
                                      p['esun'], p['sun_elevation'],
                                      p['earth_sun_dist'])
        else:
            print(f"    Warning: fallback normalization for BAND{band_num}")
            refl = np.clip(dn / 1023.0, 0, 1).astype(np.float32)

        bands[band_name] = refl
        print(f"    BAND{band_num} ({band_name}): "
              f"min={refl.min():.4f} max={refl.max():.4f} mean={refl.mean():.4f}")

    return bands, meta


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Crop to overlap bounding box
# ═══════════════════════════════════════════════════════════════════════════

def crop_array_to_bounds(array, src_meta, left, bottom, right, top):
    """
    Crop a numpy array to the given bounding box in the raster's native CRS units.
    For LISS-IV this is UTM meters (not degrees).
    src_meta: rasterio metadata dict
    Returns: (cropped_array, new_transform)
    """
    from rasterio.windows import from_bounds as window_from_bounds

    transform = src_meta['transform']

    # Window in pixel space
    win = window_from_bounds(left, bottom, right, top, transform)
    row_off = max(0, int(win.row_off))
    col_off = max(0, int(win.col_off))
    row_end = min(array.shape[-2], int(np.ceil(win.row_off + win.height)))
    col_end = min(array.shape[-1], int(np.ceil(win.col_off + win.width)))

    if row_end <= row_off or col_end <= col_off:
        raise ValueError(
            f"Crop bounds [{left},{bottom},{right},{top}] produce empty window. "
            f"Raster bounds: {src_meta.get('transform')}. "
            f"Check crop_utm values are in the raster's CRS units (meters for UTM)."
        )

    if array.ndim == 2:
        cropped = array[row_off:row_end, col_off:col_end]
    else:
        cropped = array[:, row_off:row_end, col_off:col_end]

    h = cropped.shape[-2]
    w = cropped.shape[-1]

    # New transform: upper-left corner moves, pixel size stays the same
    px_w = transform.a   # pixel width  (meters)
    px_h = transform.e   # pixel height (negative meters)
    new_left = transform.c + col_off * px_w
    new_top  = transform.f + row_off * px_h

    from rasterio.transform import Affine
    new_transform = Affine(px_w, 0.0, new_left,
                           0.0, px_h, new_top)

    return cropped, new_transform


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Co-register source to reference grid
# ═══════════════════════════════════════════════════════════════════════════

def coregister_to_grid(source_array, src_crs, src_transform,
                        ref_crs, ref_transform, ref_h, ref_w,
                        method=Resampling.bilinear):
    """
    Reproject source array to exactly match the reference pixel grid.
    Handles: different CRS, different resolution, slight geographic offsets.
    
    method: Resampling.bilinear for continuous data
            Resampling.nearest for discrete masks
    """
    if source_array.ndim == 2:
        source_array = source_array[np.newaxis]  # add band dim
        squeeze = True
    else:
        squeeze = False

    n_bands = source_array.shape[0]
    output = np.zeros((n_bands, ref_h, ref_w), dtype=np.float32)

    for b in range(n_bands):
        reproject(
            source=source_array[b].astype(np.float32),
            destination=output[b],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=method,
        )

    if squeeze:
        output = output[0]

    return output


def load_and_coregister(tif_path, ref_crs, ref_transform, ref_h, ref_w,
                         method=Resampling.bilinear, normalize_fn=None):
    """Load a GeoTIFF and co-register it to the reference grid in one step."""
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
        src_crs = src.crs
        src_transform = src.transform

    if normalize_fn:
        data = normalize_fn(data)

    registered = coregister_to_grid(
        data, src_crs, src_transform,
        ref_crs, ref_transform, ref_h, ref_w,
        method=method
    )

    return registered


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — SAR normalization (dB → [0,1])
# ═══════════════════════════════════════════════════════════════════════════

def normalize_sar(sar_db):
    """
    GEE S1_GRD exports are already in dB (sigma0).
    Normalize VV and VH to [0,1].

    Observed ranges from check_sar.py across all 3 AOIs:
      VV: min=-38.91, max=31.11  → use [-40, 35]
      VH: min=-48.27, max=22.17  → use [-50, 25]
    Old range [-25,5] / [-30,0] was too narrow — clipped real urban returns.
    """
    out = np.zeros_like(sar_db)
    out[0] = np.clip((sar_db[0] - (-40.0)) / (35.0 - (-40.0)), 0, 1)  # VV
    out[1] = np.clip((sar_db[1] - (-50.0)) / (25.0 - (-50.0)), 0, 1)  # VH
    return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — DEM normalization
# ═══════════════════════════════════════════════════════════════════════════

def normalize_dem(dem):
    """Normalize elevation to [0,1]. India range: -100m to 3000m."""
    return np.clip((dem.astype(np.float32) - (-100.0)) / (3000.0 - (-100.0)),
                   0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7b — Cloud mask directly from LISS-IV reflectance
# ═══════════════════════════════════════════════════════════════════════════

def generate_cloud_mask_from_liss4(liss4_crop, sun_azimuth, sun_elevation,
                                    cloud_percentile=80):
    """
    Generate cloud+shadow mask directly from LISS-IV reflectance bands.
    
    LISS-IV cloud detection strategy:
      - Clouds are spectrally bright in Green+Red but have LOW NIR response
        (unlike vegetation which is bright in NIR)
      - Cloud indicator = high brightness in visible + NIR ratio < 1.2
      - Use relative thresholding (percentile-based) to handle scene-wide
        radiometric variation
    
    liss4_crop: 3 x H x W array (channels: Green, Red, NIR)
    Returns: 1 x H x W float32, values {0.0=clear, 0.5=shadow, 1.0=cloud}
    """
    green = liss4_crop[0]   # ch 0
    red   = liss4_crop[1]   # ch 1
    nir   = liss4_crop[2]   # ch 2

    # Visible brightness
    brightness = (green + red) / 2.0

    # Cloud = bright in visible AND relatively low NIR
    # (vegetation: nir >> green; clouds: nir ≈ green or nir < green)
    eps = 1e-8
    nir_vis_ratio = nir / (brightness + eps)

    # Adaptive threshold: use scene percentile so it works across all dates
    valid_mask = brightness > brightness.max() * 0.05  # exclude black border
    if valid_mask.sum() > 1000:
        thresh_bright = np.percentile(brightness[valid_mask], cloud_percentile)
    else:
        thresh_bright = brightness.max() * 0.5

    # Cloud pixels: brighter than threshold AND NIR/visible ratio < 1.5
    # (clouds have nir_vis_ratio close to 1; vegetation has ratio > 2)
    cloud_mask = ((brightness > thresh_bright) & (nir_vis_ratio < 1.5)).astype(np.uint8)

    # Morphological cleanup: remove isolated bright pixels (likely noise)
    try:
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN,  kernel)  # remove small noise
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_DILATE, kernel)  # expand edges
    except Exception:
        pass  # skip if cv2 not available

    cloud_pct = cloud_mask.mean() * 100
    print(f"    LISS-IV cloud detection: {cloud_pct:.1f}%  "
          f"(threshold brightness > {thresh_bright:.4f})")

    # Shadow mask using solar geometry
    H, W = cloud_mask.shape
    shadow_az = (sun_azimuth + 180) % 360
    dist_px   = int(20 / np.tan(np.radians(max(sun_elevation, 5))))
    dx = int(dist_px * np.sin(np.radians(shadow_az)))
    dy = int(dist_px * np.cos(np.radians(shadow_az)))

    shadow_cand = np.zeros_like(cloud_mask)
    # Vectorized shift instead of per-pixel loop
    if dy >= 0 and dx >= 0:
        shadow_cand[:H-dy, :W-dx] = cloud_mask[dy:, dx:]
    elif dy >= 0 and dx < 0:
        shadow_cand[:H-dy, -dx:] = cloud_mask[dy:, :W+dx]
    elif dy < 0 and dx >= 0:
        shadow_cand[-dy:, :W-dx] = cloud_mask[:H+dy, dx:]
    else:
        shadow_cand[-dy:, -dx:] = cloud_mask[:H+dy, :W+dx]

    dark_nir    = (nir < nir.max() * 0.3).astype(np.uint8)
    shadow_mask = (shadow_cand & dark_nir).astype(np.uint8)
    shadow_pct  = shadow_mask.mean() * 100
    print(f"    Shadow: {shadow_pct:.1f}%")

    # Combine: cloud=2, shadow=1, clear=0 → divide by 2 → [0, 0.5, 1.0]
    combined = np.zeros_like(cloud_mask, dtype=np.float32)
    combined[shadow_mask == 1] = 1.0
    combined[cloud_mask  == 1] = 2.0
    combined = (combined / 2.0).astype(np.float32)

    return combined[np.newaxis]  # 1 x H x W


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — Cloud mask from S2 (kept for reference, replaced by LISS-IV method)
# ═══════════════════════════════════════════════════════════════════════════

def generate_cloud_shadow_mask(s2_cloudy_path, ref_crs, ref_transform,
                                ref_h, ref_w, sun_azimuth, sun_elevation,
                                threshold=0.35):
    """
    Generate cloud + shadow mask using s2cloudless.
    
    s2cloudless needs 10-band S2 input. GEE exported only 3 bands (B3, B4, B8).
    Strategy: duplicate available bands into the 10 expected slots.
    This is conservative — won't miss obvious clouds but may have minor
    false positives on bright surfaces. Still far better than a threshold.
    
    Returns: combined_mask on reference grid
      0.0 = clear
      0.5 = shadow (encoded as 1/2)
      1.0 = cloud  (encoded as 2/2)
    """
    from s2cloudless import S2PixelCloudDetector

    with rasterio.open(s2_cloudy_path) as src:
        s2 = src.read().astype(np.float32)  # 3 x H x W — already [0,1] from GEE
        src_crs = src.crs
        src_transform = src.transform

    green = s2[0]   # B3
    red   = s2[1]   # B4
    nir   = s2[2]   # B8

    # Build 10-band array for s2cloudless
    # Order: B1,B2,B4,B5,B8,B8A,B9,B10,B11,B12
    s2_10 = np.stack([
        nir,    # B1  proxy
        green,  # B2  proxy
        red,    # B4
        nir,    # B5  proxy
        nir,    # B8
        nir,    # B8A proxy
        nir,    # B9  proxy
        nir,    # B10 proxy
        green,  # B11 proxy
        green,  # B12 proxy
    ], axis=0)  # 10 x H x W

    H, W = s2.shape[1], s2.shape[2]
    inp = s2_10.transpose(1, 2, 0)[np.newaxis]  # 1 x H x W x 10

    print(f"    Running s2cloudless (threshold={threshold}) ...", end=' ', flush=True)
    detector = S2PixelCloudDetector(threshold=threshold, average_over=4,
                                     dilation_size=2, all_bands=False)
    cloud_prob = detector.get_cloud_probability_maps(inp)[0]   # H x W
    cloud_mask = (cloud_prob > threshold).astype(np.uint8)     # 0 or 1
    cloud_pct  = cloud_mask.mean() * 100
    print(f"cloud={cloud_pct:.1f}%")

    # Shadow mask using solar geometry
    shadow_az  = (sun_azimuth + 180) % 360
    shadow_px  = int(20 / np.tan(np.radians(max(sun_elevation, 5))))
    dx = int(shadow_px * np.sin(np.radians(shadow_az)))
    dy = int(shadow_px * np.cos(np.radians(shadow_az)))

    shadow_cand = np.zeros_like(cloud_mask)
    for i in range(H):
        for j in range(W):
            si, sj = i + dy, j + dx
            if 0 <= si < H and 0 <= sj < W:
                shadow_cand[i, j] = cloud_mask[si, sj]

    dark_nir = (nir < 0.15).astype(np.uint8)
    shadow_mask = (shadow_cand & dark_nir).astype(np.uint8)
    shadow_pct  = shadow_mask.mean() * 100
    print(f"    Shadow: {shadow_pct:.1f}%")

    # Combine: cloud=2, shadow=1, clear=0  →  divide by 2 → [0, 0.5, 1.0]
    combined = np.zeros(cloud_mask.shape, dtype=np.float32)
    combined[shadow_mask == 1] = 1.0
    combined[cloud_mask  == 1] = 2.0
    combined = (combined / 2.0).astype(np.float32)

    # Co-register to LISS-IV grid
    registered = coregister_to_grid(
        combined, src_crs, src_transform,
        ref_crs, ref_transform, ref_h, ref_w,
        method=Resampling.nearest
    )

    return registered[np.newaxis]  # 1 x H x W


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — Change detection mask
# ═══════════════════════════════════════════════════════════════════════════

def generate_change_mask(cloudy_nir, cloudy_red, clear_nir, clear_red,
                          threshold=0.20):
    """
    NDVI-based change detection.
    Flags pixels where land cover changed between clear and cloudy dates.
    Changed pixels should NOT use temporal reference for reconstruction.
    threshold=0.20: ~0.2 NDVI units = significant change
    Returns: change_mask 1 x H x W, float32, 0=stable 1=changed
    """
    eps = 1e-8
    ndvi_cloudy = (cloudy_nir - cloudy_red) / (cloudy_nir + cloudy_red + eps)
    ndvi_clear  = (clear_nir  - clear_red)  / (clear_nir  + clear_red  + eps)
    diff = np.abs(ndvi_cloudy - ndvi_clear)
    mask = (diff > threshold).astype(np.float32)
    changed_pct = mask.mean() * 100
    print(f"    Change detection: {changed_pct:.1f}% pixels flagged as changed")
    return mask[np.newaxis]  # 1 x H x W


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9 — Build 11-channel stack
# ═══════════════════════════════════════════════════════════════════════════

def build_stack_for_aoi(aoi_name, cfg):
    """
    Full preprocessing pipeline for one AOI.
    Returns: stack (11 x H x W), target (3 x H x W)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {aoi_name.upper()}")
    print(f"{'='*60}")

    left, bottom, right, top = cfg['crop_utm']
    liss4_crs = CRS.from_epsg(int(cfg['liss4_crs'].split(':')[1]))

    # ── Parse metadata ──────────────────────────────────────────────────
    print("\n[1/9] Parsing LISS-IV metadata ...")
    meta_path = os.path.join(cfg['liss4_cloudy_dir'], 'BAND_META.txt')
    meta_cloudy, sun_az, sun_elev = parse_band_meta(meta_path)

    meta_path_clear = os.path.join(cfg['liss4_clear_dir'], 'BAND_META.txt')
    meta_clear, _, _ = parse_band_meta(meta_path_clear)

    # ── Load LISS-IV cloudy ──────────────────────────────────────────────
    print("\n[2/9] Loading LISS-IV cloudy bands (DN → reflectance) ...")
    cloudy_bands, liss4_meta = load_liss4_scene(cfg['liss4_cloudy_dir'], meta_cloudy)

    # ── Crop LISS-IV cloudy to AOI (UTM meters) ──────────────────────────
    print(f"\n[3/9] Cropping to AOI: "
          f"left={left} bottom={bottom} right={right} top={top} (UTM m) ...")

    liss4_cloudy_full = np.stack([
        cloudy_bands['green'],
        cloudy_bands['red'],
        cloudy_bands['nir'],
    ], axis=0)
    del cloudy_bands

    liss4_cloudy_crop, ref_transform = crop_array_to_bounds(
        liss4_cloudy_full, liss4_meta, left, bottom, right, top
    )
    del liss4_cloudy_full

    ref_h, ref_w = liss4_cloudy_crop.shape[1], liss4_cloudy_crop.shape[2]
    ref_crs = liss4_meta['crs']

    size_km_h = ref_h * 5.0 / 1000
    size_km_w = ref_w * 5.0 / 1000
    print(f"  Reference grid: {ref_h} × {ref_w} px  "
          f"(~{size_km_h:.1f} km × {size_km_w:.1f} km at 5m)")

    # ── Load + co-register LISS-IV clear ────────────────────────────────
    print("\n[4/9] Loading LISS-IV clear + co-registering ...")
    clear_bands, clear_meta = load_liss4_scene(cfg['liss4_clear_dir'], meta_clear)
    liss4_clear_full = np.stack([
        clear_bands['green'],
        clear_bands['red'],
        clear_bands['nir'],
    ], axis=0)
    del clear_bands

    liss4_clear = coregister_to_grid(
        liss4_clear_full,
        clear_meta['crs'], clear_meta['transform'],
        ref_crs, ref_transform, ref_h, ref_w,
        method=Resampling.bilinear
    )
    del liss4_clear_full
    print(f"  Clear co-registered: {liss4_clear.shape}")

    # ── Co-register SAR ──────────────────────────────────────────────────
    print("\n[5/9] Loading + co-registering Sentinel-1 SAR ...")
    sar = load_and_coregister(
        cfg['sar_path'], ref_crs, ref_transform, ref_h, ref_w,
        method=Resampling.bilinear,
        normalize_fn=normalize_sar
    )
    print(f"  SAR: {sar.shape}  VV[{sar[0].min():.3f},{sar[0].max():.3f}]  "
          f"VH[{sar[1].min():.3f},{sar[1].max():.3f}]")

    # ── Co-register DEM ──────────────────────────────────────────────────
    print("\n[6/9] Loading + co-registering DEM ...")
    dem = load_and_coregister(
        cfg['dem_path'], ref_crs, ref_transform, ref_h, ref_w,
        method=Resampling.bilinear,
        normalize_fn=normalize_dem
    )
    if dem.ndim == 3:
        dem = dem[0:1]
    else:
        dem = dem[np.newaxis]
    print(f"  DEM: {dem.shape}  range[{dem.min():.3f},{dem.max():.3f}]")

    # ── Cloud + shadow mask ───────────────────────────────────────────────
    print("\n[7/9] Generating cloud + shadow mask (LISS-IV based) ...")
    # Use LISS-IV cloudy bands directly — more reliable than S2 when S2 export
    # may not perfectly overlap with LISS-IV scene area
    cloud_mask = generate_cloud_mask_from_liss4(
        liss4_cloudy_crop, sun_azimuth=sun_az, sun_elevation=sun_elev
    )
    print(f"  Mask: {cloud_mask.shape}  "
          f"cloud coverage: {(cloud_mask > 0.4).mean()*100:.1f}%")

    # ── Change detection mask ─────────────────────────────────────────────
    print("\n[8/9] Change detection mask ...")
    change_mask = generate_change_mask(
        liss4_cloudy_crop[2], liss4_cloudy_crop[1],  # NIR, Red cloudy
        liss4_clear[2],       liss4_clear[1],         # NIR, Red clear
    )

    # ── Stack 11 channels ────────────────────────────────────────────────
    print("\n[9/9] Building 11-channel stack ...")
    stack = np.concatenate([
        liss4_cloudy_crop,   # ch 0-2
        cloud_mask,          # ch 3
        sar,                 # ch 4-5
        liss4_clear,         # ch 6-8
        change_mask,         # ch 9
        dem,                 # ch 10
    ], axis=0).astype(np.float32)

    assert stack.shape[0] == 11, f"Expected 11 channels, got {stack.shape[0]}"
    stack = np.nan_to_num(stack, nan=0.0, posinf=1.0, neginf=0.0)

    target = liss4_clear.astype(np.float32)

    print(f"\n  Stack : {stack.shape}  ({stack.nbytes/1024**3:.2f} GB)")
    print(f"  Target: {target.shape}")

    return stack, target


# ═══════════════════════════════════════════════════════════════════════════
# STEP 10 — Save + verify
# ═══════════════════════════════════════════════════════════════════════════

def save_and_verify(aoi_name, stack, target):
    """Save stacks and run quick sanity checks."""
    stack_path  = f'ps2_cloud/data/processed/{aoi_name}_stack.npy'
    target_path = f'ps2_cloud/data/processed/{aoi_name}_clear.npy'

    print(f"\n  Saving {stack_path} ...", end=' ', flush=True)
    np.save(stack_path,  stack)
    print(f"OK ({os.path.getsize(stack_path) / 1024**3:.2f} GB)")

    print(f"  Saving {target_path} ...", end=' ', flush=True)
    np.save(target_path, target)
    print(f"OK ({os.path.getsize(target_path) / 1024**3:.2f} GB)")

    # Sanity checks
    print(f"\n  Sanity checks:")
    ok = True

    # No NaN or Inf
    has_nan = np.isnan(stack).any() or np.isnan(target).any()
    has_inf = np.isinf(stack).any() or np.isinf(target).any()
    print(f"    NaN present : {has_nan}  ← should be False")
    print(f"    Inf present : {has_inf}  ← should be False")
    if has_nan or has_inf:
        ok = False

    # Value ranges per channel
    print(f"    Channel ranges:")
    ch_names = ['G_cloudy','R_cloudy','NIR_cloudy','cloud_mask',
                'SAR_VV','SAR_VH','G_clear','R_clear','NIR_clear',
                'change_mask','DEM']
    for i, name in enumerate(ch_names):
        mn, mx, mean = stack[i].min(), stack[i].max(), stack[i].mean()
        flag = '⚠' if (mn < -0.01 or mx > 1.01) else '✓'
        print(f"    {flag} ch{i:02d} {name:12s}: [{mn:.3f}, {mx:.3f}]  mean={mean:.3f}")
        if mn < -0.01 or mx > 1.01:
            ok = False

    # Cloud coverage check
    cloud_cov = (stack[3] > 0.4).mean() * 100
    print(f"\n    Cloud coverage  : {cloud_cov:.1f}%  ← should be > 20% for training data")
    if cloud_cov < 5:
        print(f"    ⚠ Very low cloud coverage — verify this is the CLOUDY scene")
        ok = False

    # Change detection check
    change_pct = stack[9].mean() * 100
    print(f"    Changed pixels  : {change_pct:.1f}%  ← should be < 60%")

    # Verify NDVI looks reasonable
    eps = 1e-8
    ndvi = (stack[2] - stack[1]) / (stack[2] + stack[1] + eps)
    print(f"    Cloudy NDVI     : mean={ndvi.mean():.3f}  ← >0.2 = vegetation present")

    ndvi_clear = (stack[8] - stack[7]) / (stack[8] + stack[7] + eps)
    print(f"    Clear NDVI      : mean={ndvi_clear.mean():.3f}  ← should differ from cloudy")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION PLOT — save RGB + NIR + SAR side by side
# ═══════════════════════════════════════════════════════════════════════════

def save_verification_plot(aoi_name, stack):
    """Save a 5-panel quick-look image for visual verification."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        eps = 1e-8

        def to_rgb(g, r, nir):
            # False color: NIR→R, Red→G, Green→B
            rgb = np.stack([nir, r, g], axis=2)
            return np.clip(rgb * 3.0, 0, 1)  # stretch for visibility

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f'{aoi_name.upper()} — Preprocessing Verification', fontsize=13)

        # Downsample for plot (every 10th pixel)
        s = 10
        cloudy_rgb = to_rgb(stack[0,::s,::s], stack[1,::s,::s], stack[2,::s,::s])
        clear_rgb  = to_rgb(stack[6,::s,::s], stack[7,::s,::s], stack[8,::s,::s])

        axes[0].imshow(cloudy_rgb)
        axes[0].set_title('Cloudy LISS-IV\n(False color: NIR-R-G)')
        axes[0].axis('off')

        axes[1].imshow(clear_rgb)
        axes[1].set_title('Clear LISS-IV\n(Temporal reference)')
        axes[1].axis('off')

        axes[2].imshow(stack[3,::s,::s], cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title(f'Cloud+Shadow Mask\n(cloud={( stack[3] > 0.4).mean()*100:.1f}%)')
        axes[2].axis('off')

        axes[3].imshow(stack[4,::s,::s], cmap='gray', vmin=0, vmax=1)
        axes[3].set_title('SAR VV\n(Co-registered)')
        axes[3].axis('off')

        axes[4].imshow(stack[10,::s,::s], cmap='terrain', vmin=0, vmax=1)
        axes[4].set_title('DEM\n(Normalized elevation)')
        axes[4].axis('off')

        out_path = f'ps2_cloud/outputs/figures/{aoi_name}_preprocess_verify.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Verification plot saved: {out_path}")

    except Exception as e:
        print(f"  Plot failed (non-critical): {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import time

    # Bengaluru + Punjab = training scenes
    # Meghalaya = validation scene (held out entirely)
    AOI_ORDER = ['bengaluru', 'punjab', 'meghalaya']

    results = {}
    total_start = time.time()

    for aoi in AOI_ORDER:
        cfg = AOI_CONFIG[aoi]
        t0 = time.time()

        try:
            stack, target = build_stack_for_aoi(aoi, cfg)
            ok = save_and_verify(aoi, stack, target)
            save_verification_plot(aoi, stack)
            elapsed = time.time() - t0
            results[aoi] = 'OK' if ok else 'WARNINGS'
            print(f"\n  {aoi} done in {elapsed/60:.1f} min — status: {results[aoi]}")
            del stack, target

        except Exception as e:
            import traceback
            print(f"\n  ERROR processing {aoi}: {e}")
            traceback.print_exc()
            results[aoi] = f'ERROR: {e}'

    # ── Summary ────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE  ({total_elapsed/60:.1f} min total)")
    print(f"{'='*60}")
    for aoi, status in results.items():
        icon = '✓' if status == 'OK' else ('⚠' if 'WARN' in status else '✗')
        print(f"  {icon}  {aoi:12s} : {status}")

    print(f"\nOutput files:")
    for aoi in AOI_ORDER:
        for suffix in ['stack', 'clear']:
            p = f'ps2_cloud/data/processed/{aoi}_{suffix}.npy'
            if os.path.exists(p):
                mb = os.path.getsize(p) / 1024**2
                print(f"  {p}  ({mb:.0f} MB)")

    print(f"\nVerification plots: ps2_cloud/outputs/figures/")
    print(f"\nNext: python train.py")
