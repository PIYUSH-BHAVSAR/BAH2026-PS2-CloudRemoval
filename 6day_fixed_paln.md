# PS2 — Realistic 6-Day Build Plan
## Grounded in Real Data, Real Hardware, Real Constraints
### Bharatiya Antariksh Hackathon 2026 | Lenovo LOQ RTX 3050 6GB

---

## Your Hardware — Honest Assessment

| Component | Spec | Reality for this project |
|---|---|---|
| CPU | Intel i5-12450hx (8 cores) | Good — handles preprocessing, tiling, dataloaders |
| RAM | 16 GB | Tight but workable — watch memory during tiling |
| Storage | 512 GB | Will get tight — budget carefully (see storage plan) |
| GPU | RTX 3050 6GB VRAM | The real constraint — everything must fit in 6GB |
| GPU Bandwidth | 168 GB/s (96-bit bus) | Slow for large batch training — use small batches |

### What 6GB VRAM Actually Means

```
6GB total VRAM breakdown during training:
  Model weights (float16)     → ~450 MB   (CloudRemovalNet with f=32)
  Activations per batch       → ~800 MB   (batch_size=2, tile=128x128)
  Gradients                   → ~450 MB
  Optimizer states (AdamW)    → ~900 MB
  CUDA overhead               → ~400 MB
  PyTorch reserved            → ~300 MB
  ─────────────────────────────────────────
  Total usable                → ~3.3 GB   (out of 6GB)
  Safety margin               → ~2.7 GB   ← this is your headroom

SAFE settings for RTX 3050 6GB:
  tile_size  = 128 (NOT 256 — 256x256 will OOM)
  batch_size = 2   (NOT 8 — 8 will OOM)
  base_filters (f) = 32 (NOT 64 — 64 will OOM)
  mixed_precision = True (MANDATORY — halves memory use)
```

---

## Real Data — Sizes and Download Times

### Data Source 1: LISS-IV from Bhoonidhi

```
What it is:
  Sensor    : ResourceSat-2 LISS-IV-MX
  Bands     : 3 (BAND2=Green, BAND3=Red, BAND4=NIR)
  Resolution: 5.8m
  Swath     : 70km × 70km per scene
  Format    : Individual band GeoTIFF per band + BAND_META.txt
  Revisit   : 24 days systematic (5 days targeted)

Real scene size:
  70km / 5.8m = ~12,069 pixels per side
  One band GeoTIFF = 12069 × 12069 × 2 bytes (10-bit stored in uint16) = ~276 MB
  Three bands = ~830 MB per scene (uncompressed)
  Zip download = ~120-200 MB per scene (compressed)

You need:
  3 AOIs × 2 scenes each (cloudy + clear) = 6 downloads minimum
  Total download = ~720 MB – 1.2 GB
  Bhoonidhi download speed = variable (NRSC servers, often 1-5 MB/s)
  Estimated time = 2-4 hours total

CRITICAL: LISS-IV is Indian territory only as open data.
  Bhoonidhi account needed — approval takes 24-48 hours.
  Register TODAY. This is the first thing to do.
```

### Data Source 2: Sentinel-1 SAR

```
What it is:
  Mode      : IW (Interferometric Wide Swath) GRD
  Bands     : VV + VH polarizations
  Resolution: 10m (after terrain correction)
  Swath     : 250km wide
  Revisit   : 12 days (Sentinel-1C only, as of 2025)

Real scene size:
  GRD .SAFE uncompressed = ~1 GB per scene (as of Sep 2023, no longer compressed)
  After terrain correction to GeoTIFF = ~400-500 MB per scene per AOI

PROBLEM: 1 GB download per scene, plus SNAP preprocessing.
SOLUTION: Use Google Earth Engine (GEE) instead.
  GEE has Sentinel-1 GRD pre-calibrated, pre-terrain-corrected.
  No download needed. Export only your AOI as small GeoTIFF.
  Your AOI export size = ~30-80 MB (much more manageable)

GEE Sentinel-1 export for 70km AOI:
  70km / 10m = 7000 pixels per side
  2 bands (VV, VH) × 7000 × 7000 × 4 bytes = ~392 MB
  But: at 10m resolution, clipped to your LISS-IV AOI only
  Practical export size: 30-80 MB per AOI
```

### Data Source 3: DEM

```
SRTM 1 arc-second DEM:
  Resolution: ~30m
  Download: Python elevation library — instant, automated
  Size per AOI: ~5-15 MB
  Total: negligible

Command: eio clip -o dem_bengaluru.tif --bounds 77.45 12.85 77.75 13.1
```

### Total Storage Budget

```
Raw downloads:
  LISS-IV (6 scenes × 200MB zip)  = 1.2 GB
  GEE exports SAR (3 AOIs)        = 240 MB
  GEE exports S2 (3 AOIs × 2)     = 360 MB
  DEM (3 AOIs)                    = 50 MB

Processed data:
  LISS-IV stacked (unzipped+conv) = 3 GB
  SAR co-registered               = 500 MB
  Masks + change detection        = 300 MB
  11-channel stacks (npy)         = 2 GB
  Training tiles (128×128)        = 1.5 GB

Model + checkpoints:
  Pretrained backbone             = 200 MB
  Training checkpoints (5 saves)  = 500 MB

Outputs + demo:
  Demo .npz files                 = 200 MB
  Figures + attention maps        = 500 MB

TOTAL: ~10 GB
Fits in 512 GB easily. Keep a 50 GB project folder as working space.
```

---

## Why GEE Changes Everything

**Google Earth Engine** is your secret weapon for this project. It eliminates the biggest bottlenecks:

| Task | Without GEE | With GEE |
|---|---|---|
| SAR calibration | Download 1GB .SAFE + SNAP 30min | GEE pre-processes it, export 50MB |
| SAR terrain correction | SNAP graph, 20-30 min per scene | Done automatically in GEE |
| Sentinel-2 cloud mask | Download 800MB S2 scene | Run s2cloudless in GEE, export mask only |
| Change detection | Compute locally on big arrays | Run in GEE, export 10MB result |
| Getting paired cloudy+clear | Manual search + multiple downloads | GEE filter by cloud % and date, export |

**Everything you need from Sentinel-1 and Sentinel-2 can be exported from GEE as small, pre-processed GeoTIFFs.** Only LISS-IV must come from Bhoonidhi directly.

---

## Corrected Model for RTX 3050 6GB

The original plan used `f=64` (base filters). On 6GB VRAM with 256×256 tiles, that OOMs. Here is the corrected architecture:

```python
# CORRECT settings for RTX 3050 6GB:
model = CloudRemovalNet(in_ch=11, out_ch=3, f=32)  # f=32 NOT f=64
tile_size  = 128   # 128x128 NOT 256x256
batch_size = 2     # 2 NOT 8
# Enable mixed precision (MANDATORY)
from torch.cuda.amp import autocast, GradScaler

# Memory calculation for f=32, tile=128, batch=2:
# Model params: ~4.5M parameters × 2 bytes (fp16) = ~9 MB
# Bottleneck feature map: 2 × 512 × 8 × 8 = 65k values × 4 bytes = 0.26 MB
# Attention at bottleneck: N = 8×8 = 64 pixels → 64×64 attention = trivial
# Activations total: ~400 MB
# Gradients: ~400 MB
# Optimizer: ~800 MB
# Total: ~1.7 GB → fits in 6 GB with room to spare
```

**Trade-off:** Smaller model means lower capacity. Will your metrics suffer?

With enough data (even 3 AOIs tiled at 128×128) and 50 epochs, f=32 gives:
- Expected SSIM: 0.70-0.80 (good, not sota)
- Expected NDVI MAE: 0.04-0.07 (acceptable)
- Training time: ~4-6 hours for 50 epochs on RTX 3050

This is competitive for a hackathon. You are not trying to beat SOTA papers, you are trying to show a working, evaluated, well-designed system.

**Alternative for more capacity:** Use `gradient_checkpointing=True` with f=48:
```python
# gradient checkpointing trades compute for memory
# ~30% slower training but lets you use larger model
from torch.utils.checkpoint import checkpoint
# Wrap encoder blocks in checkpoint() calls
```

---

## The 3 Bugs — Fixed Code

### Fix 1 — Attention assert (prevents OOM crash)

```python
# In models/attention.py, CrossModalAttention.forward():
def forward(self, optical, sar, temporal):
    B, C, H, W = optical.shape
    
    # CRITICAL FIX: assert spatial size is at bottleneck level
    # With 4 encoder stages of stride 2 and input 128×128:
    # After enc1: 64×64, enc2: 32×32, enc3: 16×16, enc4: 8×8
    # Bottleneck: 8×8 = N=64 — perfectly fine for full attention
    assert H * W <= 1024, (
        f"Attention spatial size {H}×{W}={H*W} is too large. "
        f"Must be at bottleneck (≤32×32). "
        f"Check encoder downsampling — with tile=128 expect 8×8 here."
    )
    N = H * W
    
    # Delete the dead split_heads block that was here
    opt = optical.flatten(2).transpose(1, 2)   # B x N x C
    sar_ = sar.flatten(2).transpose(1, 2)
    tmp = temporal.flatten(2).transpose(1, 2)
    
    Q = self.to_q(opt).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
    K = self.to_k(sar_).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
    V = self.to_v(tmp).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
    
    attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
    attn = F.softmax(attn, dim=-1)   # dim=-1 NOT dim=-2
    attn_weights = attn.detach()
    attn = self.drop(attn)
    
    out = torch.matmul(attn, V)
    out = out.transpose(1, 2).reshape(B, N, C)
    out = self.norm(self.out(out) + opt)
    out = out.transpose(1, 2).reshape(B, C, H, W)
    
    return out, attn_weights
```

### Fix 2 — Remove double projection

```python
# In models/cloud_removal_net.py:
# DELETE these three lines from __init__:
# self.opt_proj = nn.Conv2d(f*16, f*16, 1)   ← DELETE
# self.sar_proj = nn.Conv2d(f*16, f*16, 1)   ← DELETE
# self.tmp_proj = nn.Conv2d(f*16, f*16, 1)   ← DELETE

# In forward(), REPLACE:
# opt  = self.opt_proj(x)
# sar  = self.sar_proj(x)
# temp = self.tmp_proj(x)
# fused, attn = self.attention(opt, sar, temp)

# WITH:
fused, attn = self.attention(x, x, x)
# CrossModalAttention's own to_q/to_k/to_v handle the projections correctly
# This removes redundant double-projection
```

### Fix 3 — Scene-level train/val split

```python
# In scripts/dataset/cloud_dataset.py:
def tile_and_load(input_stacks, targets,
                   val_scene_indices=[2],   # last scene = val
                   tile_size=128, overlap=16, batch_size=2):
    """
    Split by SCENE not by tile.
    With 3 AOIs: indices 0,1 = train, index 2 = validation.
    
    val_scene_indices=[2] means Meghalaya = val
    (geographically different from Bengaluru + Punjab train)
    
    This prevents data leakage where nearby tiles from the same
    scene appear in both train and val sets.
    """
    train_x, train_y = [], []
    val_x, val_y     = [], []
    stride = tile_size - overlap

    for scene_i, (stack, target) in enumerate(zip(input_stacks, targets)):
        _, H, W = stack.shape
        cloud_ch = stack[3] * 2  # un-normalize mask channel
        is_val = scene_i in val_scene_indices

        for r in range(0, H - tile_size + 1, stride):
            for c in range(0, W - tile_size + 1, stride):
                tile_x = stack[:, r:r+tile_size, c:c+tile_size]
                tile_y = target[:, r:r+tile_size, c:c+tile_size]
                cloud_cov = (cloud_ch[r:r+tile_size, c:c+tile_size] > 0).mean()

                if 0.2 <= cloud_cov <= 0.95:
                    if is_val:
                        val_x.append(tile_x); val_y.append(tile_y)
                    else:
                        train_x.append(tile_x); train_y.append(tile_y)

    print(f"Train tiles: {len(train_x)} (from scenes {[i for i in range(len(input_stacks)) if i not in val_scene_indices]})")
    print(f"Val tiles:   {len(val_x)} (from scenes {val_scene_indices})")

    train_ds = CloudDataset(train_x, train_y, augment=True)
    val_ds   = CloudDataset(val_x,   val_y,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)
    return train_loader, val_loader

# Call like this:
# train_loader, val_loader = tile_and_load(
#     input_stacks=[bengaluru_stack, punjab_stack, meghalaya_stack],
#     targets=[bengaluru_clear, punjab_clear, meghalaya_clear],
#     val_scene_indices=[2],  # Meghalaya held out entirely
#     tile_size=128, batch_size=2
# )
```

### Fix 4 — LISS-IV metadata parser

```python
# scripts/preprocess/parse_liss4_meta.py
def parse_liss4_meta(band_meta_path):
    """
    Parse BAND_META.txt that comes with every Bhoonidhi LISS-IV download.
    
    File format is key=value pairs, one per line.
    Example content:
      OTSProductID = RS2A_LISS4_MX_20230715_123456
      DateOfPass = 15-JUL-2023
      SunElevationAtCenter = 62.3
      BandID = 2
      Lmax = 52.14
      Lmin = -1.55
      Qcalmax = 1023
      Qcalmin = 0
      
    For reflectance conversion we need:
      gain = (Lmax - Lmin) / (Qcalmax - Qcalmin)
      bias = Lmin - gain * Qcalmin
      ESUN = from ResourceSat-2 Data Users Handbook (band-specific constants)
    """
    params = {}
    with open(band_meta_path) as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=', 1)
                params[key.strip()] = val.strip()

    # ESUN values from ResourceSat-2 Data Users Handbook
    # Band2=Green, Band3=Red, Band4=NIR
    esun_table = {'2': 1848.0, '3': 1549.0, '4': 1044.0}

    band_id = params.get('BandID', '2')
    lmax = float(params.get('Lmax', 52.14))
    lmin = float(params.get('Lmin', -1.55))
    qmax = float(params.get('Qcalmax', 1023))
    qmin = float(params.get('Qcalmin', 0))

    gain = (lmax - lmin) / (qmax - qmin)
    bias = lmin - gain * qmin

    sun_elev = float(params.get('SunElevationAtCenter', 45.0))

    # Earth-sun distance using pyephem
    try:
        import ephem, datetime
        date_str = params.get('DateOfPass', '15-JUL-2023')
        acq_date = datetime.datetime.strptime(date_str, '%d-%b-%Y')
        observer = ephem.Observer()
        observer.date = acq_date
        sun = ephem.Sun()
        sun.compute(observer)
        earth_sun_dist = float(sun.earth_distance)
    except Exception:
        earth_sun_dist = 1.0  # fallback: 1 AU

    return {
        'band_id':        band_id,
        'gain':           gain,
        'bias':           bias,
        'esun':           float(esun_table.get(band_id, 1500.0)),
        'sun_elevation':  sun_elev,
        'earth_sun_dist': earth_sun_dist,
    }

# Usage:
# for band in ['BAND2', 'BAND3', 'BAND4']:
#     meta = parse_liss4_meta(f'{scene_folder}/{band}_META.txt')
#     dn_band = rasterio.open(f'{scene_folder}/{band}.tif').read(1)
#     reflectance = dn_to_reflectance_liss4(
#         dn_band, meta['gain'], meta['bias'], meta['esun'],
#         meta['sun_elevation'], meta['earth_sun_dist']
#     )
```

---

## 6-Day Plan — Revised and Realistic

---

### DAY 1 — Setup + Accounts + Data Start
**Total time: 6 hours active | rest running overnight**

#### Morning (2 hours) — Model Lead

```bash
# 1. Create environment
conda create -n ps2 python=3.10 -y && conda activate ps2

# 2. Install all packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install rasterio gdal numpy scipy pyproj shapely albumentations
pip install opencv-python scikit-image pillow tqdm wandb matplotlib seaborn
pip install pytorch-msssim lpips s2cloudless streamlit fastapi uvicorn
pip install einops timm earthengine-api ephem xarray rioxarray dask

# 3. Verify GPU
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
# Should print: CUDA: True, GPU: NVIDIA GeForce RTX 3050, VRAM: ~6.0 GB
"

# 4. Test memory limit IMMEDIATELY
python -c "
import torch
# Simulate what training will use
model_dummy = torch.zeros(2, 11, 128, 128).cuda()  # batch of 2, 11ch, 128x128
print('Input tensor OK:', model_dummy.shape)
del model_dummy
# If this crashes, something else is using VRAM (close Chrome, Discord, etc.)
"
```

#### Morning (2 hours) — Data Lead

```
ACCOUNT REGISTRATIONS (do all simultaneously):

1. Bhoonidhi: https://bhoonidhi.nrsc.gov.in
   → Register → wait for email approval (24-48 hrs)
   → While waiting: learn the interface, bookmark
   
2. Copernicus Dataspace: https://dataspace.copernicus.eu
   → Register (instant) → we will use GEE instead but have this as backup
   
3. Google Earth Engine: https://earthengine.google.com
   → Register with college email for free academic access
   → Usually approved within hours
   → GEE is your PRIMARY source for SAR and S2

4. WandB: https://wandb.ai
   → Register (instant, free) — for training monitoring
```

#### Afternoon (2 hours) — Data Lead

```python
# GEE export script — run this as soon as GEE is approved
# File: scripts/download/gee_export_all.py

import ee
ee.Authenticate()  # first time only
ee.Initialize()

AOIs = {
    'bengaluru': ee.Geometry.Rectangle([77.45, 12.85, 77.75, 13.1]),
    'punjab':    ee.Geometry.Rectangle([74.80, 30.50, 75.20, 30.9]),
    'meghalaya': ee.Geometry.Rectangle([91.50, 25.30, 91.90, 25.7]),
}

for name, aoi in AOIs.items():

    # --- SENTINEL-2 CLOUDY (monsoon) ---
    s2_cloudy = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate('2023-06-01', '2023-09-30')
        .filterBounds(aoi)
        .filter(ee.Filter.gt('CLOUDY_PIXEL_PERCENTAGE', 40))
        .sort('CLOUDY_PIXEL_PERCENTAGE', False)
        .first()
        .select(['B3','B4','B8'])
        .divide(10000))  # → reflectance [0,1]

    # --- SENTINEL-2 CLEAR (winter reference) ---
    s2_clear = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate('2022-11-01', '2023-02-28')
        .filterBounds(aoi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
        .sort('CLOUDY_PIXEL_PERCENTAGE')
        .first()
        .select(['B3','B4','B8'])
        .divide(10000))

    # --- SENTINEL-1 SAR (GEE pre-processed: calibrated + terrain corrected in dB) ---
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate('2023-07-01', '2023-07-31')
        .filterBounds(aoi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select(['VV', 'VH'])
        .mean())  # average if multiple scenes

    for img, suffix in [(s2_cloudy, 'S2_cloudy'), (s2_clear, 'S2_clear'), (s1, 'S1_sar')]:
        ee.batch.Export.image.toDrive(
            image=img.toFloat(),
            description=f'{name}_{suffix}',
            folder='PS2_BAH2026',
            region=aoi,
            scale=10,
            crs='EPSG:32643',
            maxPixels=1e9,
        ).start()
        print(f"Export started: {name}_{suffix}")

print("All exports started. Check GEE Tasks tab.")
print("Downloads will be in your Google Drive > PS2_BAH2026 folder")
print("Estimated time: 10-30 minutes per export")
```

#### Evening — DEM download (5 min)

```bash
# Install and download DEM — instant
pip install elevation && eio selfcheck

python -c "
import elevation
elevation.clip(bounds=(77.45,12.85,77.75,13.1), output='data/raw/dem/dem_bengaluru.tif')
elevation.clip(bounds=(74.80,30.50,75.20,30.9), output='data/raw/dem/dem_punjab.tif')
elevation.clip(bounds=(91.50,25.30,91.90,25.7), output='data/raw/dem/dem_meghalaya.tif')
print('DEM done')
"
```

#### Day 1 Checklist
- [ ] Conda env created, GPU verified, memory test passed
- [ ] All 4 accounts registered (Bhoonidhi, Copernicus, GEE, WandB)
- [ ] GEE exports started (9 exports: 3 AOIs × SAR + S2 cloudy + S2 clear)
- [ ] DEM downloaded (5 min task)
- [ ] Project folder structure created

---

### DAY 2 — Data Download + Preprocessing
**Goal: All data in hand, preprocessed, 11-channel stacks ready**

#### Morning — Check GEE exports

```python
# Check if GEE exports finished (they appear in Google Drive)
# Download them: Drive > PS2_BAH2026 folder > download all 9 GeoTIFFs

# File naming you'll see:
# bengaluru_S2_cloudy.tif    (~50-100 MB)
# bengaluru_S2_clear.tif     (~50-100 MB)
# bengaluru_S1_sar.tif       (~30-60 MB)
# (same for punjab and meghalaya)
```

#### Get LISS-IV from Bhoonidhi (if approved)

```
If Bhoonidhi approved (likely by Day 2):
  1. Go to Data Discovery & Download
  2. Select: ResourceSat-2 → LISS-IV-MX
  3. Draw AOI for Bengaluru, set date range June-Sept 2023
  4. Filter: Cloud Cover 40-100% (for cloudy training scene)
  5. Add to cart → Download (Direct Download if available)
  6. Also download cloud-free scene (Nov-Feb date range)
  7. Repeat for Punjab, Meghalaya

If NOT approved yet:
  Use Sentinel-2 as LISS-IV proxy for ALL development.
  Switch to real LISS-IV when it arrives.
  THE CODE DOES NOT CHANGE — same pipeline, just different input files.
```

#### Preprocessing script (run for each AOI)

```python
# scripts/preprocess/run_all_preprocessing.py
# Run this once per AOI after all data is downloaded

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ── Step 1: Normalize S2/LISS-IV to [0,1] ──────────────────────────────
def load_optical(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        meta = src.meta.copy()
    # GEE S2 export already divided by 10000 → just clip
    return np.clip(data, 0, 1), meta

# ── Step 2: Normalize SAR (GEE outputs are already in dB) ──────────────
def load_sar(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)  # VV, VH in dB
    vv_norm = np.clip((data[0] - (-25)) / (5 - (-25)), 0, 1)
    vh_norm = np.clip((data[1] - (-30)) / (0  - (-30)), 0, 1)
    return np.stack([vv_norm, vh_norm])

# ── Step 3: Co-register everything to S2/LISS4 grid ────────────────────
def coregister(source_path, reference_path, output_path, method=Resampling.bilinear):
    with rasterio.open(reference_path) as ref:
        ref_meta = ref.meta.copy()
        ref_crs, ref_tr, ref_h, ref_w = ref.crs, ref.transform, ref.height, ref.width
    with rasterio.open(source_path) as src:
        src_data = src.read().astype(np.float32)
        src_crs, src_tr = src.crs, src.transform
    out = np.zeros((src_data.shape[0], ref_h, ref_w), np.float32)
    for b in range(src_data.shape[0]):
        reproject(source=src_data[b], destination=out[b],
                  src_transform=src_tr, src_crs=src_crs,
                  dst_transform=ref_tr, dst_crs=ref_crs, resampling=method)
    out_meta = ref_meta.copy()
    out_meta.update({'count': src_data.shape[0], 'dtype': 'float32'})
    with rasterio.open(output_path, 'w', **out_meta) as dst: dst.write(out)

# ── Step 4: Cloud + shadow mask (s2cloudless — proper detection) ────────
def make_cloud_mask(cloudy_rgb_path, sun_azimuth_deg=135.0, sun_elevation_deg=50.0,
                    threshold=0.4):
    """
    Generate cloud mask using s2cloudless + solar-geometry shadow mask.
    
    s2cloudless requires all 10 Sentinel-2 bands. Since GEE exports only
    B3/B4/B8 (3 bands), we use the GEE-exported cloud probability band as a
    direct proxy (export it alongside the scene — see gee_export_all.py).
    
    Fallback (if only 3-band export available): s2cloudless on repeated bands.
    This is still far more accurate than a brightness threshold because it
    uses the trained neural network probability, not a hard rule.
    
    sun_azimuth_deg, sun_elevation_deg: from image metadata or GEE property
      GEE property names: MEAN_SOLAR_AZIMUTH_ANGLE, MEAN_SOLAR_ZENITH_ANGLE
    
    Returns: combined_mask — 1 x H x W, values:
      0.0 = clear
      0.5 = shadow (normalized: original value 1, stored as /2)
      1.0 = cloud  (normalized: original value 2, stored as /2)
    """
    from s2cloudless import S2PixelCloudDetector
    import rasterio

    with rasterio.open(cloudy_rgb_path) as src:
        rgb = src.read().astype(np.float32)  # 3 x H x W, already [0,1]

    # s2cloudless works on 10-band input scaled to [0, 1].
    # With only 3 bands available, tile them into the 10 expected slots.
    # Bands order: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
    # s2cloudless uses: B1,B2,B4,B5,B8,B8A,B9,B10,B11,B12 (10 bands)
    # We have B3(G)=ch0, B4(R)=ch1, B8(NIR)=ch2
    # Fill missing bands with NIR (conservative — won't hurt cloud detection much)
    nir = rgb[2:3]  # 1 x H x W
    s2_10band = np.concatenate([
        nir,          # B1  proxy
        rgb[0:1],     # B2  proxy (Green)
        rgb[1:2],     # B4  (Red)
        nir,          # B5  proxy
        nir,          # B8  (NIR)
        nir,          # B8A proxy
        nir,          # B9  proxy
        nir,          # B10 proxy
        rgb[0:1],     # B11 proxy (Green)
        rgb[0:1],     # B12 proxy (Green)
    ], axis=0)  # 10 x H x W

    H, W = rgb.shape[1], rgb.shape[2]
    # s2cloudless expects: N x H x W x C (batch x height x width x bands)
    inp = s2_10band.transpose(1, 2, 0)[np.newaxis]  # 1 x H x W x 10

    detector = S2PixelCloudDetector(
        threshold=threshold,
        average_over=4,
        dilation_size=2,
        all_bands=False
    )
    cloud_prob = detector.get_cloud_probability_maps(inp)[0]  # H x W
    cloud_mask = (cloud_prob > threshold).astype(np.uint8)    # H x W, 0 or 1

    # Shadow mask using solar geometry
    # Shadow falls in the direction opposite to sun azimuth
    shadow_azimuth = (sun_azimuth_deg + 180) % 360
    # Estimate shadow offset in pixels: cloud height ~2km, tan(elev) gives ground offset
    shadow_dist_px = int(20 / np.tan(np.radians(max(sun_elevation_deg, 5))))
    dx = int(shadow_dist_px * np.sin(np.radians(shadow_azimuth)))
    dy = int(shadow_dist_px * np.cos(np.radians(shadow_azimuth)))

    shadow_candidate = np.zeros_like(cloud_mask)
    for i in range(H):
        for j in range(W):
            si, sj = i + dy, j + dx
            if 0 <= si < H and 0 <= sj < W:
                shadow_candidate[i, j] = cloud_mask[si, sj]

    # Confirm shadow: dark in NIR (shadows strongly absorb NIR)
    dark_nir = (rgb[2] < 0.15).astype(np.uint8)
    shadow_mask = (shadow_candidate & dark_nir).astype(np.uint8)

    # Combine: cloud=2, shadow=1, clear=0 → then divide by 2 to normalize to [0,1]
    combined = np.zeros_like(cloud_mask)
    combined[shadow_mask == 1] = 1   # shadow
    combined[cloud_mask  == 1] = 2   # cloud overwrites shadow
    combined_norm = (combined / 2.0).astype(np.float32)  # [0, 0.5, 1.0]

    cloud_pct  = (cloud_mask  > 0).mean() * 100
    shadow_pct = (shadow_mask > 0).mean() * 100
    print(f"  Cloud: {cloud_pct:.1f}%  Shadow: {shadow_pct:.1f}%")

    return combined_norm[np.newaxis]  # 1 x H x W

# ── Step 5: Change detection mask ──────────────────────────────────────
def make_change_mask(cloudy_rgb, clear_rgb, threshold=0.2):
    eps = 1e-8
    ndvi_c = (cloudy_rgb[2]-cloudy_rgb[1])/(cloudy_rgb[2]+cloudy_rgb[1]+eps)
    ndvi_r = (clear_rgb[2]-clear_rgb[1])/(clear_rgb[2]+clear_rgb[1]+eps)
    diff = np.abs(ndvi_c - ndvi_r)
    return (diff > threshold).astype(np.float32)[np.newaxis]

# ── Step 6: Normalize DEM ────────────────────────────────────────────────
def load_dem(path, ref_path):
    coregister(path, ref_path, path.replace('.tif', '_coreg.tif'),
               method=Resampling.bilinear)
    with rasterio.open(path.replace('.tif','_coreg.tif')) as src:
        dem = src.read(1).astype(np.float32)
    return np.clip((dem - (-100)) / (3000 - (-100)), 0, 1)[np.newaxis]

# ── Step 7: Stack into 11 channels ─────────────────────────────────────
def build_stack(aoi_name, base_dir='data/processed',
                sun_azimuth_deg=135.0, sun_elevation_deg=50.0):
    """
    Build the 11-channel input stack for one AOI.
    
    IMPORTANT ORDER:
      1. Load optical (reference for co-registration)
      2. Co-register SAR to optical grid  ← must happen BEFORE load_sar
      3. Load SAR from co-registered file
      4. Generate masks
      5. Stack everything
    """
    cloudy, meta = load_optical(f'{base_dir}/{aoi_name}_S2_cloudy.tif')
    clear, _     = load_optical(f'{base_dir}/{aoi_name}_S2_clear.tif')

    # ── Co-register SAR FIRST, then load ────────────────────────────────
    # FIX: coregister must run before load_sar — coreg file doesn't exist yet
    coregister(f'{base_dir}/{aoi_name}_S1_sar.tif',
               f'{base_dir}/{aoi_name}_S2_cloudy.tif',
               f'{base_dir}/{aoi_name}_S1_sar_coreg.tif')
    sar = load_sar(f'{base_dir}/{aoi_name}_S1_sar_coreg.tif')

    # ── Cloud + shadow mask (s2cloudless) ───────────────────────────────
    combined_mask = make_cloud_mask(
        f'{base_dir}/{aoi_name}_S2_cloudy.tif',
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
    )

    change_mask  = make_change_mask(cloudy, clear)
    dem          = load_dem(f'data/raw/dem/dem_{aoi_name}.tif',
                            f'{base_dir}/{aoi_name}_S2_cloudy.tif')

    H, W = cloudy.shape[1], cloudy.shape[2]
    # Resize all to same shape
    def resize_to(arr, h, w):
        if arr.shape[1] == h and arr.shape[2] == w: return arr
        import cv2
        return np.stack([cv2.resize(arr[c], (w, h)) for c in range(arr.shape[0])])

    sar           = resize_to(sar,           H, W)
    combined_mask = resize_to(combined_mask, H, W)
    change_mask   = resize_to(change_mask,   H, W)
    dem           = resize_to(dem,           H, W)

    stack = np.concatenate([cloudy, combined_mask, sar, clear, change_mask, dem], axis=0)
    assert stack.shape[0] == 11, f"Expected 11 channels got {stack.shape[0]}"
    stack = np.nan_to_num(stack, nan=0.0, posinf=1.0, neginf=0.0)

    np.save(f'{base_dir}/{aoi_name}_stack.npy', stack)
    np.save(f'{base_dir}/{aoi_name}_clear.npy', clear)

    cloud_pct = (combined_mask > 0.4).mean() * 100   # >0.4 after /2 normalization = was 1 (cloud)
    print(f"{aoi_name}: stack {stack.shape}, cloud {cloud_pct:.1f}%")
    return stack, clear

# Run for all 3 AOIs:
for aoi in ['bengaluru', 'punjab', 'meghalaya']:
    build_stack(aoi)
```

#### Day 2 Checklist
- [ ] All GEE exports downloaded from Google Drive (9 files)
- [ ] LISS-IV download started from Bhoonidhi (or S2 proxy confirmed)
- [ ] Preprocessing script run for all 3 AOIs
- [ ] 3 `_stack.npy` files saved (each ~11 x 12000 x 12000 → crop to AOI)
- [ ] Cloud coverage verified per stack (print statement)
- [ ] Co-registration verified: plot cloudy + SAR overlay, check alignment

---

### DAY 3 — Model Code + First Training Run
**Goal: All model files written, tiling done, first epoch completes without error**

Write all model files with the corrected architecture (f=32, tile=128).

Key parameter changes from original plan:

```python
# config.yaml — CORRECTED for RTX 3050 6GB
data:
  tile_size: 128      # WAS 256 — changed to fit VRAM
  overlap: 16         # WAS 32
  batch_size: 2       # WAS 8 — critical for 6GB
  num_workers: 2      # WAS 4 — LOQ has shared memory bandwidth

model:
  base_filters: 32    # WAS 64 — critical for 6GB
  attention_heads: 4  # WAS 8 — smaller model

training:
  phase1_epochs: 15   # WAS 20 — same wall clock time with small batches
  phase2_epochs: 25   # WAS 30
  phase1_lr: 5.0e-4   # slightly lower for small batch
  phase2_lr: 5.0e-5
  mixed_precision: true  # MANDATORY
```

#### Expected tile count with 128×128 on your 3 AOIs

```
One GEE export at 10m resolution for 70km × 70km AOI:
  70km / 10m = 7000 pixels per side
  Tiles at 128×128, overlap=16, stride=112:
  (7000 - 128) / 112 + 1 = ~61 tiles per row
  61 × 61 = ~3721 tiles per scene
  Filter for 20-95% cloud coverage: keep ~60% = ~2232 tiles
  3 scenes total: ~6696 tiles
  Train (2 scenes): ~4464 tiles
  Val (1 scene, Meghalaya): ~2232 tiles

At batch_size=2: 2232 batches per epoch
Epoch time on RTX 3050: ~8-12 minutes per epoch
50 epochs total: ~7-10 hours
```

#### First run test (do this before starting full training)

```python
# test_forward.py — run this first
import torch
from models.cloud_removal_net import CloudRemovalNet
from models.losses import CombinedLoss

device = 'cuda'
model = CloudRemovalNet(in_ch=11, out_ch=3, f=32).to(device)

# Test forward pass
x = torch.randn(2, 11, 128, 128).to(device)
y = torch.rand(2, 3, 128, 128).to(device)

with torch.cuda.amp.autocast():
    recon, uncert, attn = model(x)
    print("recon:", recon.shape)    # should be 2 x 3 x 128 x 128
    print("uncert:", uncert.shape)  # should be 2 x 3 x 128 x 128
    print("attn:", attn.shape)      # should be 2 x 4 x 64 x 64 (N=8*8=64)
    
    criterion = CombinedLoss()
    loss, ld = criterion(recon, y, uncert)
    print("loss:", loss.item())
    for k, v in ld.items(): print(f"  {k}: {v:.4f}")

# Check VRAM usage
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"VRAM reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
# Should see: ~1.5-2.5 GB used (safe for 6GB)

loss.backward()
print("Backward pass OK")
```

#### Day 3 Checklist
- [ ] All model files written with CORRECTED f=32, tile=128 settings
- [ ] test_forward.py passes without OOM
- [ ] VRAM usage printed: < 4 GB during test forward + backward
- [ ] Tiling done: tile counts printed and verified
- [ ] Phase 1 training STARTED (first 3 epochs confirmed running)
- [ ] WandB dashboard open and logging

---

### DAY 4 — Training Monitoring + Debug
**Goal: Phase 1 complete, Phase 2 running, no crashes overnight**

#### Leave training running overnight (Day 3→4)

Training schedule on RTX 3050:
```
Phase 1 (15 epochs): ~15 × 12 min = 3 hours
Phase 2 (25 epochs): ~25 × 12 min = 5 hours
Total: ~8 hours
Start evening Day 3 → done by morning Day 4
```

#### What to watch on WandB

```
Good training (Phase 1, epochs 1-15):
  total_loss: 2.5 → 0.9     (should drop clearly)
  spectral:   1.2 → 0.3     (drops fastest — highest weight)
  ssim_loss:  0.45 → 0.25   (structural improvement)
  ndvi_mae:   0.12 → 0.06   (critical metric)
  val_ssim:   0.45 → 0.65   (should be > 0.6 by epoch 15)

Bad signs (what to do):
  NaN loss:      add eps=1e-8 everywhere, check data for inf/nan
  Loss stuck:    check dataloader is actually loading diverse tiles
  OOM crash:     reduce batch to 1, or tile_size to 96
  NDVI not moving: increase w_spectral from 2.0 to 3.5
```

#### If OOM on your hardware

```python
# Emergency OOM fix: enable gradient checkpointing
# Add this to CloudRemovalNet.__init__:
from torch.utils.checkpoint import checkpoint_sequential

# And in forward(), wrap encoder:
# Instead of: s1, x = self.enc1(x)
# Use:
def forward(self, x):
    def create_custom_forward(module):
        def custom_forward(*inputs):
            feat = module.conv(inputs[0])
            return feat, module.pool(feat)
        return custom_forward
    
    # This trades 30% speed for ~40% memory saving
    s1, x = checkpoint(create_custom_forward(self.enc1), x, use_reentrant=False)
    # ... etc
```

#### Day 4 Checklist
- [ ] Phase 1 complete (loss < 1.0, SSIM > 0.60)
- [ ] Phase 2 started and running
- [ ] No NaN losses in any run
- [ ] Best checkpoint saved: models/checkpoints/best.pth
- [ ] WandB graphs show clear downward trend

---

### DAY 5 — Evaluation + Visualization
**Goal: All 8 metrics computed, attention maps saved, 3 demo cases ready**

#### Run full evaluation

```python
# evaluate.py
import torch
import numpy as np
from models.cloud_removal_net import CloudRemovalNet
from scripts.evaluate.metrics import Evaluator

device = 'cuda'
model = CloudRemovalNet(in_ch=11, out_ch=3, f=32).to(device)
model.load_state_dict(torch.load('models/checkpoints/best.pth'))
model.eval()

evaluator = Evaluator(device=device)

# Load validation scene (Meghalaya)
val_stack = np.load('data/processed/meghalaya_stack.npy')
val_clear = np.load('data/processed/meghalaya_clear.npy')

# Run tile-by-tile and collect metrics
all_metrics = []
tile_size = 128
stride = tile_size - 16

_, H, W = val_stack.shape
for r in range(0, H - tile_size + 1, stride):
    for c in range(0, W - tile_size + 1, stride):
        x = torch.tensor(val_stack[:, r:r+tile_size, c:c+tile_size]).unsqueeze(0).to(device)
        y = torch.tensor(val_clear[:, r:r+tile_size, c:c+tile_size]).unsqueeze(0).to(device)
        with torch.no_grad():
            recon, uncert, attn = model(x)
        metrics = evaluator.compute_all(recon, y, uncert)
        all_metrics.append(metrics)

# Print final metrics table
print("\n=== VALIDATION METRICS (Meghalaya — held-out scene) ===")
avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
for k, v in sorted(avg.items()):
    print(f"  {k:20s}: {v:.4f}")
```

#### Full-scene inference with tile stitching (add to scripts/inference/reconstruct_full.py)

This MUST exist before Day 6. Without Gaussian-blended stitching, tile seams will be visible
in the demo — a 1-pixel grid artifact every 128 pixels that immediately looks wrong live.

```python
# scripts/inference/reconstruct_full.py
import torch
import numpy as np
from scipy.signal import windows as sig_windows

def reconstruct_scene(model, stack_11ch, tile_size=128, overlap=32, device='cuda'):
    """
    Run tile-by-tile inference and stitch results with Gaussian-weighted overlap blending.
    
    Gaussian weighting: center pixels of each tile contribute more than edges.
    Where tiles overlap, contributions are weighted and averaged.
    Result: seamless full-scene reconstruction with no visible tile boundaries.
    
    Args:
        model      : trained CloudRemovalNet, already in eval() mode
        stack_11ch : numpy array, shape 11 x H x W, normalized [0,1]
        tile_size  : must match what model was trained on (128)
        overlap    : overlap between adjacent tiles (32 recommended)
        device     : 'cuda' or 'cpu'
    
    Returns:
        reconstruction : numpy array, shape 3 x H x W, values [0,1]
        uncertainty    : numpy array, shape 3 x H x W, values [0,1]
    """
    _, H, W = stack_11ch.shape
    stride  = tile_size - overlap

    output_recon   = np.zeros((3, H, W), dtype=np.float32)
    output_uncert  = np.zeros((3, H, W), dtype=np.float32)
    weight_map     = np.zeros((1, H, W), dtype=np.float32)

    # 2-D Gaussian window — center pixels weighted ~4× more than corners
    gauss_1d = sig_windows.gaussian(tile_size, std=tile_size / 4.0).astype(np.float32)
    gauss_2d = np.outer(gauss_1d, gauss_1d)[np.newaxis]  # 1 x T x T

    model.eval()
    with torch.no_grad():
        for r in range(0, H - tile_size + 1, stride):
            for c in range(0, W - tile_size + 1, stride):
                tile = torch.tensor(
                    stack_11ch[:, r:r+tile_size, c:c+tile_size]
                ).unsqueeze(0).to(device)                      # 1 x 11 x T x T

                recon_t, uncert_t, _ = model(tile)
                r_np = recon_t[0].cpu().numpy()                # 3 x T x T
                u_np = uncert_t[0].cpu().numpy()               # 3 x T x T

                output_recon[:,  r:r+tile_size, c:c+tile_size] += r_np * gauss_2d
                output_uncert[:, r:r+tile_size, c:c+tile_size] += u_np * gauss_2d
                weight_map[:,    r:r+tile_size, c:c+tile_size] += gauss_2d

    # Handle right/bottom edge tiles if scene size isn't perfectly divisible
    # (stride may not reach the last tile — pad and run remaining edge)
    last_r = ((H - tile_size) // stride) * stride
    last_c = ((W - tile_size) // stride) * stride

    for r, c in [(H - tile_size, lc) for lc in range(0, W - tile_size + 1, stride)] + \
                [(lr, W - tile_size) for lr in range(0, H - tile_size + 1, stride)]:
        if weight_map[0, r + tile_size//2, c + tile_size//2] > 0:
            continue  # already covered
        tile = torch.tensor(
            stack_11ch[:, r:r+tile_size, c:c+tile_size]
        ).unsqueeze(0).to(device)
        recon_t, uncert_t, _ = model(tile)
        r_np = recon_t[0].cpu().numpy()
        u_np = uncert_t[0].cpu().numpy()
        output_recon[:,  r:r+tile_size, c:c+tile_size] += r_np * gauss_2d
        output_uncert[:, r:r+tile_size, c:c+tile_size] += u_np * gauss_2d
        weight_map[:,    r:r+tile_size, c:c+tile_size] += gauss_2d

    # Normalize by accumulated weights
    eps = 1e-8
    reconstruction = output_recon  / (weight_map + eps)
    uncertainty    = output_uncert / (weight_map + eps)

    # Zero-weight regions (scene edges smaller than tile_size) → set to 0
    zero_mask = (weight_map[0] < eps)
    reconstruction[:, zero_mask] = 0.0
    uncertainty[:,    zero_mask] = 0.0

    return np.clip(reconstruction, 0, 1), np.clip(uncertainty, 0, 1)


# ── Usage ────────────────────────────────────────────────────────────────
# from models.cloud_removal_net import CloudRemovalNet
# import numpy as np, torch
#
# device = 'cuda'
# model = CloudRemovalNet(in_ch=11, out_ch=3, f=32).to(device)
# model.load_state_dict(torch.load('models/checkpoints/final_best.pth'))
# model.eval()
#
# stack = np.load('data/processed/bengaluru_stack.npy')  # 11 x H x W
# recon, uncert = reconstruct_scene(model, stack, tile_size=128, overlap=32, device=device)
# print("Output shape:", recon.shape)   # 3 x H x W
# np.save('outputs/bengaluru_reconstruction.npy', recon)
```

#### Save demo cases and figures

```python
# Pick best 128×128 tiles (high cloud coverage, visually interesting)
# Save as .npz for Streamlit demo
for aoi in ['bengaluru', 'punjab', 'meghalaya']:
    stack = np.load(f'data/processed/{aoi}_stack.npy')
    clear = np.load(f'data/processed/{aoi}_clear.npy')
    
    # Find tile with ~60% cloud coverage
    best_r, best_c, best_cov = 0, 0, 0
    cloud_ch = stack[3] * 2
    for r in range(0, stack.shape[1]-128, 128):
        for c in range(0, stack.shape[2]-128, 128):
            cov = (cloud_ch[r:r+128, c:c+128] > 0).mean()
            if abs(cov - 0.6) < abs(best_cov - 0.6):
                best_r, best_c, best_cov = r, c, cov
    
    demo_x = stack[:, best_r:best_r+128, best_c:best_c+128]
    demo_y = clear[:, best_r:best_r+128, best_c:best_c+128]
    
    np.savez(f'data/demo_cases/{aoi}.npz',
             input_stack=demo_x, ground_truth=demo_y)
    print(f"Saved {aoi} demo — cloud {best_cov*100:.1f}%")
```

#### Day 5 Checklist
- [ ] All 8 metrics computed on Meghalaya validation scene
- [ ] Ablation study run (no SAR / no temporal / full model)
- [ ] 5-panel comparison figures saved for all 3 AOIs
- [ ] Attention visualization saved for all 3 AOIs
- [ ] 3 demo .npz files created and verified
- [ ] Metrics table printed and copied for PPT slide

---

### DAY 6 — Demo + Submission
**Goal: Streamlit app running, PPT updated with real numbers, submitted by 11:59 PM**

#### Morning
- Streamlit app working on all 3 demo cases
- FastAPI health check responding
- All figures verified

#### Afternoon
- Update PPT slides with real metric numbers from Day 5
- Add your actual attention map and comparison figures to slides
- Fill in team names, colleges
- Export PDF

#### Evening
- Form filled on Hack2skill (use the copy-paste text from earlier)
- PDF uploaded
- SUBMIT

#### Final submit checklist
```
□ Team names filled in deck
□ Real metric numbers in evaluation slide
□ Real figures (not placeholder) in architecture/results slides
□ PDF exported and verified < 5 MB
□ Form: Brief about idea (copy from earlier text)
□ Form: Problem being solved (copy from earlier text)
□ Form: Tech stack (copy from earlier text)
□ Form: Hackathon experience (copy from earlier text)
□ PDF uploaded
□ SUBMITTED before 11:59 PM July 1, 2026 IST
□ Screenshot confirmation page
```

---

## Hardware-Specific Tips for Lenovo LOQ

```
1. ALWAYS close Chrome, Discord, background apps before training
   They consume ~500MB–1GB VRAM even idle on integrated graphics

2. Set power mode to Performance before training:
   Windows: Battery icon → Power Mode → Best Performance

3. The LOQ throttles under sustained load — check temperatures
   Target: GPU < 85°C, CPU < 95°C
   If throttling: clean vents, use a laptop cooler

4. Pin memory in DataLoader (already in our config)
   But set num_workers=2 (not 4) — LOQ's RAM is shared with iGPU

5. Mixed precision (AMP) is not optional on 6GB — always use autocast()

6. Don't run training and Streamlit demo simultaneously
   Load demo from CPU: torch.load(..., map_location='cpu')

7. Checkpoint frequently — LOQ can overheat and crash
   Save every 5 epochs, not just best model

8. If you get CUDA initialization error on restart:
   conda activate ps2 → then run again
   Sometimes CUDA needs a fresh Python process
```

---

## Realistic Expectations — What You Will Actually Get

| Metric | Realistic for RTX 3050 6GB | Good SOTA (A100) | Judges expect |
|---|---|---|---|
| SSIM | 0.68 – 0.78 | 0.85 – 0.92 | > 0.65 ✅ |
| PSNR | 24 – 28 dB | 30 – 36 dB | > 22 dB ✅ |
| NDVI MAE | 0.04 – 0.09 | 0.02 – 0.04 | < 0.10 ✅ |
| Edge F1 | 0.55 – 0.70 | 0.75 – 0.88 | > 0.50 ✅ |
| Training time | 8 – 10 hours | 2 – 3 hours | N/A |

**You will not match SOTA numbers on 6GB. That is fine.** ISRO judges at a student hackathon are looking for:
1. Correct problem understanding ✅ (you have this)
2. Physically justified approach ✅ (SAR + temporal + spectral loss)
3. Working system with real evaluation ✅ (8 metrics, ablation)
4. Good presentation of results ✅ (figures, attention maps, demo)

A working system scoring SSIM 0.72 with an honest ablation study beats a team claiming SSIM 0.92 with no working demo every single time.

---

*RTX 3050 6GB | 16GB RAM | i5-12450HX | 512GB | BAH 2026 PS-02*
*Submission deadline: July 1, 2026 11:59 PM IST*