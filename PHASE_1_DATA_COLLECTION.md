# PHASE 1 — Data Collection
## PS2 | BAH 2026 | Lenovo LOQ RTX 3050 6GB
### Goal: All accounts registered, all raw data downloaded and stored, folder structure ready

---

## What You Are Collecting

```
DATA YOU NEED                    SOURCE              SIZE
─────────────────────────────────────────────────────────────────
LISS-IV cloudy scenes (×3 AOI)  Bhoonidhi           ~600 MB
LISS-IV clear scenes  (×3 AOI)  Bhoonidhi           ~600 MB
Sentinel-2 cloudy     (×3 AOI)  Google Earth Engine ~300 MB
Sentinel-2 clear ref  (×3 AOI)  Google Earth Engine ~300 MB
Sentinel-1 SAR        (×3 AOI)  Google Earth Engine ~180 MB
DEM SRTM              (×3 AOI)  elevation library    ~50 MB
─────────────────────────────────────────────────────────────────
TOTAL RAW                                           ~2.0 GB
```

> **Why GEE for SAR?**
> Raw Sentinel-1 .SAFE files are ~1 GB each and need SNAP for calibration + terrain correction (30 min/scene).
> GEE gives you pre-calibrated, terrain-corrected SAR as a 30–80 MB GeoTIFF export.
> Only LISS-IV must come from Bhoonidhi — everything else goes through GEE.

---

## AOIs (Areas of Interest)

| Name | Bounding Box | Terrain | Why |
|---|---|---|---|
| `bengaluru` | `[77.45, 12.85, 77.75, 13.1]` | Urban | Dense built-up, hard edges |
| `punjab` | `[74.80, 30.50, 75.20, 30.9]` | Agricultural | Regular field patterns, NDVI-critical |
| `meghalaya` | `[91.50, 25.30, 91.90, 25.7]` | Forested | Heavy cloud cover, dense vegetation |

Meghalaya is the **validation scene** — held out entirely from training.

---

## Step 1 — Environment Setup

```bash
# Create conda environment
conda create -n ps2 python=3.10 -y
conda activate ps2

# PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Geospatial + ML stack
pip install rasterio gdal numpy scipy pyproj shapely albumentations
pip install opencv-python scikit-image pillow tqdm wandb matplotlib seaborn
pip install pytorch-msssim lpips s2cloudless streamlit fastapi uvicorn
pip install einops timm earthengine-api ephem xarray rioxarray dask elevation

# Verify GPU — MUST pass before anything else
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
# Expected: CUDA: True | GPU: NVIDIA GeForce RTX 3050 | VRAM: 6.0 GB

# Quick VRAM test — simulates training input
t = torch.zeros(2, 11, 128, 128).cuda()
print('VRAM test OK:', t.shape)
del t
"
```

---

## Step 2 — Create Project Folder Structure

```bash
mkdir -p ps2_cloud/data/raw/{liss4,sentinel1,sentinel2,dem}
mkdir -p ps2_cloud/data/processed
mkdir -p ps2_cloud/data/demo_cases
mkdir -p ps2_cloud/models/{checkpoints,pretrained}
mkdir -p ps2_cloud/scripts/{preprocess,dataset,train,evaluate,inference,viz,download}
mkdir -p ps2_cloud/outputs/figures
mkdir -p ps2_cloud/demo
mkdir -p ps2_cloud/logs
```

Expected after Phase 1 completes:
```
ps2_cloud/
└── data/
    └── raw/
        ├── liss4/
        │   ├── bengaluru_cloudy/   ← BAND2.tif, BAND3.tif, BAND4.tif, BAND_META.txt
        │   ├── bengaluru_clear/
        │   ├── punjab_cloudy/
        │   ├── punjab_clear/
        │   ├── meghalaya_cloudy/
        │   └── meghalaya_clear/
        ├── sentinel1/
        │   ├── bengaluru_S1_sar.tif
        │   ├── punjab_S1_sar.tif
        │   └── meghalaya_S1_sar.tif
        ├── sentinel2/
        │   ├── bengaluru_S2_cloudy.tif
        │   ├── bengaluru_S2_clear.tif
        │   ├── punjab_S2_cloudy.tif
        │   ├── punjab_S2_clear.tif
        │   ├── meghalaya_S2_cloudy.tif
        │   └── meghalaya_S2_clear.tif
        └── dem/
            ├── dem_bengaluru.tif
            ├── dem_punjab.tif
            └── dem_meghalaya.tif
```

---

## Step 3 — Account Registrations (do all simultaneously, Day 1 morning)

### 3.1 Bhoonidhi — LISS-IV
- URL: https://bhoonidhi.nrsc.gov.in
- Register with college email
- **Approval takes 24–48 hours** — this is why you register first, before anything else
- What you will download: ResourceSat-2 → LISS-IV-MX

### 3.2 Google Earth Engine — SAR + Sentinel-2
- URL: https://earthengine.google.com
- Register with college email for free academic access
- Usually approved within 2–6 hours
- **This is your primary data source for SAR and S2**

### 3.3 Copernicus Dataspace — backup only
- URL: https://dataspace.copernicus.eu
- Instant registration
- Keep as backup if GEE has issues — you won't need it if GEE works

### 3.4 WandB — training monitoring
- URL: https://wandb.ai
- Instant registration, free
- Needed on Day 3 when training starts

---

## Step 4 — GEE Data Export (run as soon as GEE is approved)

```python
# scripts/download/gee_export_all.py
# Run once. Exports go to your Google Drive > PS2_BAH2026 folder.
# Check progress in: https://code.earthengine.google.com → Tasks tab

import ee
ee.Authenticate()   # opens browser first time — follow prompts
ee.Initialize()

AOIs = {
    'bengaluru': ee.Geometry.Rectangle([77.45, 12.85, 77.75, 13.1]),
    'punjab':    ee.Geometry.Rectangle([74.80, 30.50, 75.20, 30.9]),
    'meghalaya': ee.Geometry.Rectangle([91.50, 25.30, 91.90, 25.7]),
}

for name, aoi in AOIs.items():

    # ── Sentinel-2 CLOUDY (monsoon season, heavy cloud) ──────────────────
    s2_cloudy = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate('2023-06-01', '2023-09-30')
        .filterBounds(aoi)
        .filter(ee.Filter.gt('CLOUDY_PIXEL_PERCENTAGE', 40))
        .sort('CLOUDY_PIXEL_PERCENTAGE', False)   # most cloudy first
        .first()
        .select(['B3', 'B4', 'B8'])               # Green, Red, NIR
        .divide(10000))                           # DN → reflectance [0,1]

    # ── Sentinel-2 CLEAR (winter, cloud-free reference) ──────────────────
    s2_clear = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate('2022-11-01', '2023-02-28')
        .filterBounds(aoi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
        .sort('CLOUDY_PIXEL_PERCENTAGE')          # least cloudy first
        .first()
        .select(['B3', 'B4', 'B8'])
        .divide(10000))

    # ── Sentinel-1 SAR (pre-calibrated + terrain-corrected in GEE) ───────
    # GEE COPERNICUS/S1_GRD is already in dB scale (sigma0).
    # No SNAP needed. VV and VH in dB — ready to normalize directly.
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate('2023-07-01', '2023-07-31')
        .filterBounds(aoi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select(['VV', 'VH'])
        .mean())                                  # temporal average if multiple passes

    # ── Export all 3 layers per AOI ───────────────────────────────────────
    for img, suffix in [
        (s2_cloudy, 'S2_cloudy'),
        (s2_clear,  'S2_clear'),
        (s1,        'S1_sar'),
    ]:
        task = ee.batch.Export.image.toDrive(
            image=img.toFloat(),
            description=f'{name}_{suffix}',
            folder='PS2_BAH2026',                 # Google Drive folder name
            region=aoi,
            scale=10,                             # 10m resolution
            crs='EPSG:32643',                     # UTM Zone 43N — covers all 3 AOIs
            maxPixels=1e9,
        )
        task.start()
        print(f"Export started: {name}_{suffix}")

print()
print("All 9 exports started.")
print("Monitor at: https://code.earthengine.google.com → Tasks tab")
print("ETA: 10–30 minutes per export")
print("When done: Google Drive > PS2_BAH2026 > download all 9 GeoTIFFs")
```

Expected output files in Google Drive:
```
bengaluru_S2_cloudy.tif   (~50–100 MB)
bengaluru_S2_clear.tif    (~50–100 MB)
bengaluru_S1_sar.tif      (~30–60 MB)
punjab_S2_cloudy.tif      (~50–100 MB)
punjab_S2_clear.tif       (~50–100 MB)
punjab_S1_sar.tif         (~30–60 MB)
meghalaya_S2_cloudy.tif   (~50–100 MB)
meghalaya_S2_clear.tif    (~50–100 MB)
meghalaya_S1_sar.tif      (~30–60 MB)
```

After downloading, move them to `data/raw/sentinel2/` and `data/raw/sentinel1/`.

---

## Step 5 — DEM Download (5 minutes, run immediately)

```bash
pip install elevation
eio selfcheck   # verifies the tool is working

python -c "
import elevation, os
os.makedirs('ps2_cloud/data/raw/dem', exist_ok=True)

# SRTM 1 arc-second (~30m resolution)
elevation.clip(bounds=(77.45, 12.85, 77.75, 13.1),
               output='ps2_cloud/data/raw/dem/dem_bengaluru.tif')

elevation.clip(bounds=(74.80, 30.50, 75.20, 30.9),
               output='ps2_cloud/data/raw/dem/dem_punjab.tif')

elevation.clip(bounds=(91.50, 25.30, 91.90, 25.7),
               output='ps2_cloud/data/raw/dem/dem_meghalaya.tif')

print('DEM download complete — 3 files saved')
"
```

---

## Step 6 — LISS-IV from Bhoonidhi (once account is approved)

```
How to download from Bhoonidhi portal:

1. Login at https://bhoonidhi.nrsc.gov.in
2. Go to: Data Discovery & Download
3. Sensor: ResourceSat-2 → LISS-IV-MX
4. For CLOUDY scene (Bengaluru):
   - Draw AOI: 77.45°E 12.85°N to 77.75°E 13.1°N
   - Date range: 2023-06-01 to 2023-09-30
   - Cloud filter: 40–100%
   - Add to cart → Download
5. For CLEAR scene (Bengaluru):
   - Same AOI, date range: 2022-11-01 to 2023-02-28
   - Cloud filter: 0–10%
   - Add to cart → Download
6. Repeat for Punjab and Meghalaya AOIs

File structure you receive per scene (zip):
  RS2A_LISS4_MX_YYYYMMDD_XXXXXX/
  ├── BAND2.tif          ← Green band (DN, uint16)
  ├── BAND3.tif          ← Red band
  ├── BAND4.tif          ← NIR band
  └── BAND_META.txt      ← gain, bias, sun elevation, date — needed for reflectance
```

**If Bhoonidhi approval is delayed:** Use Sentinel-2 as a LISS-IV proxy for all
development. The pipeline is identical — just point it to the S2 files instead.
Switch to real LISS-IV when approval comes. No code changes needed.

### LISS-IV Metadata Parser (needed for DN → reflectance conversion)

```python
# scripts/preprocess/parse_liss4_meta.py
import ephem, datetime

def parse_liss4_meta(band_meta_path):
    """
    Parse BAND_META.txt from every Bhoonidhi LISS-IV download.
    Returns calibration coefficients for DN → surface reflectance conversion.

    ESUN constants from ResourceSat-2 Data Users Handbook:
      Band2 (Green): 1848.0 W/m²/µm
      Band3 (Red):   1549.0 W/m²/µm
      Band4 (NIR):   1044.0 W/m²/µm
    """
    params = {}
    with open(band_meta_path) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                params[k.strip()] = v.strip()

    esun_table = {'2': 1848.0, '3': 1549.0, '4': 1044.0}

    band_id = params.get('BandID', '2')
    lmax    = float(params.get('Lmax',    52.14))
    lmin    = float(params.get('Lmin',    -1.55))
    qmax    = float(params.get('Qcalmax', 1023))
    qmin    = float(params.get('Qcalmin', 0))

    gain = (lmax - lmin) / (qmax - qmin)
    bias = lmin - gain * qmin

    sun_elev = float(params.get('SunElevationAtCenter', 45.0))

    try:
        date_str = params.get('DateOfPass', '15-JUL-2023')
        acq_date = datetime.datetime.strptime(date_str, '%d-%b-%Y')
        obs = ephem.Observer()
        obs.date = acq_date
        sun = ephem.Sun(); sun.compute(obs)
        earth_sun_dist = float(sun.earth_distance)
    except Exception:
        earth_sun_dist = 1.0   # fallback: 1 AU

    return {
        'band_id':        band_id,
        'gain':           gain,
        'bias':           bias,
        'esun':           float(esun_table.get(band_id, 1500.0)),
        'sun_elevation':  sun_elev,
        'earth_sun_dist': earth_sun_dist,
    }


def dn_to_reflectance_liss4(dn_array, gain, bias, esun,
                             sun_elevation, earth_sun_dist):
    """Convert LISS-IV DN → surface reflectance [0, 1]."""
    import numpy as np
    radiance = dn_array.astype('float32') * gain + bias
    zenith   = np.radians(90 - sun_elevation)
    refl     = (np.pi * radiance * earth_sun_dist**2) / (esun * np.cos(zenith))
    return np.clip(refl, 0, 1).astype('float32')


# Usage — run for each LISS-IV scene:
# import rasterio
# for band_num in ['BAND2', 'BAND3', 'BAND4']:
#     meta = parse_liss4_meta(f'data/raw/liss4/bengaluru_cloudy/{band_num}_META.txt')
#     with rasterio.open(f'data/raw/liss4/bengaluru_cloudy/{band_num}.tif') as src:
#         dn = src.read(1)
#     refl = dn_to_reflectance_liss4(dn, meta['gain'], meta['bias'],
#                                    meta['esun'], meta['sun_elevation'],
#                                    meta['earth_sun_dist'])
#     # Save reflectance band
```

---

## Phase 1 Checklist

```
ACCOUNTS
  [ ] Bhoonidhi registration submitted (takes 24-48 hrs)
  [ ] Google Earth Engine approved and authenticated
  [ ] Copernicus Dataspace registered (backup)
  [ ] WandB account created

ENVIRONMENT
  [ ] conda env 'ps2' created
  [ ] All packages installed without errors
  [ ] CUDA: True printed
  [ ] VRAM test passed (torch.zeros on GPU)

DATA — GEE
  [ ] gee_export_all.py ran — 9 tasks started in GEE Tasks tab
  [ ] All 9 GeoTIFFs appeared in Google Drive PS2_BAH2026 folder
  [ ] Downloaded to data/raw/sentinel1/ and data/raw/sentinel2/

DATA — DEM
  [ ] dem_bengaluru.tif, dem_punjab.tif, dem_meghalaya.tif saved

DATA — LISS-IV
  [ ] Bhoonidhi approved (or S2 proxy confirmed as fallback)
  [ ] 6 LISS-IV scenes downloaded (3 AOI × cloudy + clear)
  [ ] Each scene has BAND2.tif, BAND3.tif, BAND4.tif, BAND_META.txt

FOLDER
  [ ] ps2_cloud/ folder structure created as shown above
  [ ] All raw files in correct subfolders
```

---

## Storage After Phase 1

```
data/raw/sentinel2/    ~600 MB   (6 S2 GeoTIFFs)
data/raw/sentinel1/    ~180 MB   (3 SAR GeoTIFFs)
data/raw/dem/           ~50 MB   (3 DEMs)
data/raw/liss4/       ~1.2 GB   (6 scenes × 3 bands)
─────────────────────────────────────────────────────
TOTAL                  ~2.0 GB
```

**→ Continue to PHASE_2_PREPROCESSING.md**
