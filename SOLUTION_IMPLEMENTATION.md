# Solution Implementation — PS2 Cloud Removal
## Bharatiya Antariksh Hackathon 2026
## Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery

---

## Team & Hardware

| Item | Detail |
|---|---|
| Problem Statement | PS-02 — Cloud Removal for LISS-IV |
| Hardware | Lenovo LOQ — Intel i5-12450H, 16 GB RAM, RTX 3050 6GB VRAM |
| Framework | Python 3.10, PyTorch 2.x, CUDA 11.8 |
| Total Training Time | ~8 hours on RTX 3050 6GB |

---

## Problem Overview

LISS-IV (ResourceSat-2) is a high-resolution optical sensor at 5.8m resolution. Over tropical and mountainous regions of India — particularly during monsoon season — persistent cloud cover renders large portions of optical imagery unusable. This system reconstructs cloud-free LISS-IV imagery by fusing multiple data sources through a deep learning model.

---

## Phase 1 — Data Collection

### Primary Data: LISS-IV from Bhoonidhi Portal

All LISS-IV scenes downloaded from **https://bhoonidhi.nrsc.gov.in** (ISRO/NRSC).

| AOI | Scene Type | Product ID | Date | Cloud % | Path/Row |
|---|---|---|---|---|---|
| Bengaluru | Cloudy | R2F11JUN2026078587009900064SSANSTUC00GTD | 11-Jun-2026 | ~20% | 99/64 |
| Bengaluru | Clear | R2F18JAN2026076548009900064SSANSTUC00GTD | 18-Jan-2026 | ~0% | 99/64 |
| Punjab | Cloudy | R2F04JUL2025073728009300049SSANSTUC00GTD | 04-Jul-2025 | ~20% | 93/49 |
| Punjab | Clear | R2F25NOV2025075781009300049SSANSTUC00GTD | 25-Nov-2025 | ~0% | 93/49 |
| Meghalaya | Cloudy | R2F17JUL2025073912011000054SSANSTUC00GTD | 17-Jul-2025 | ~19% | 110/54 |
| Meghalaya | Clear | R2F25JAN2026076640011000054SSANSTUC00GTD | 25-Jan-2026 | ~0% | 110/54 |

**LISS-IV specifications:**
- Sensor: ResourceSat-2 LISS-IV-MX (Multispectral)
- Native resolution: 5.8m → resampled to 5.0m in product
- Bands: BAND2 (Green), BAND3 (Red), BAND4 (NIR)
- Format: Individual band GeoTIFF per band + BAND_META.txt
- CRS: UTM Zone 43N (EPSG:32643) for Bengaluru/Punjab; UTM Zone 46N (EPSG:32646) for Meghalaya
- Full scene size: ~16,000 × 17,000 pixels (~84 × 87 km)

**Scene coverage (actual downloaded extent):**
- Bengaluru: 76.699–77.533°E, 12.980–13.745°N (~74 km × 85 km)
- Punjab: 74.758–75.723°E, 30.222–30.991°N (~83 km × 86 km)
- Meghalaya: 91.267–92.184°E, 24.892–25.643°N (~80 km × 83 km)

**AOI choice rationale:**
- Bengaluru: urban terrain — tests reconstruction of built-up areas with hard edges
- Punjab: agricultural terrain — tests NDVI preservation over crop fields (critical for ISRO use case)
- Meghalaya: forested terrain, deep monsoon cloud — most challenging scene

---

### Auxiliary Data: Sentinel-1 SAR + Sentinel-2 + DEM

**Sentinel-1 SAR** — downloaded via Google Earth Engine (GEE), project `aidsync-4e460`
- Collection: `COPERNICUS/S1_GRD` (already calibrated + terrain corrected in dB in GEE)
- Mode: IW (Interferometric Wide Swath), VV + VH polarizations
- Resolution: 10m, EPSG:4326
- Date range: matched to cloudy LISS-IV acquisition months

| File | Size | VV range (dB) | VH range (dB) |
|---|---|---|---|
| bengaluru_S1_sar.tif | 674 MB | −38.9 to +31.1 | −48.3 to +22.2 |
| punjab_S1_sar.tif | 762 MB | −29.0 to +17.3 | −41.5 to +5.9 |
| meghalaya_S1_sar.tif | 751 MB | −30.4 to +15.9 | −46.2 to +6.7 |

**Sentinel-2 optical** — also from GEE (`COPERNICUS/S2_SR_HARMONIZED`)
- Bands: B3 (Green), B4 (Red), B8 (NIR), divided by 10000 on export → [0,1]
- Used for cloud masking via s2cloudless (alternative cloud detection)
- Also used as a spatial reference for SAR co-registration

**DEM** — SRTM GL1 (~30m) from OpenTopography AWS public bucket
- Downloaded using custom script (`download_dem.py`) — no authentication needed
- Tiles: 4 tiles for Bengaluru, 2 for Punjab, 4 for Meghalaya

| File | Size (px) | Elevation range |
|---|---|---|
| dem_bengaluru.tif | 2763×3065 | 627–1338 m (Deccan plateau) |
| dem_punjab.tif | 2770×3479 | 132–279 m (Indo-Gangetic plain) |
| dem_meghalaya.tif | 2712×3460 | −42–1964 m (river valleys to hills) |

---

## Phase 2 — Preprocessing Pipeline

All preprocessing in `preprocess.py`. Output: 11-channel input stacks and 3-channel clear targets.

### Step 1 — LISS-IV Calibration (DN → Surface Reflectance)

Parsed `BAND_META.txt` from each scene. Key calibration formula:

```
radiance    = DN × gain + bias
reflectance = (π × radiance × d²) / (ESUN × cos(zenith))
```

Where:
- `gain = (Lmax − Lmin) / (Qcalmax − Qcalmin)`
- `bias = Lmin − gain × Qcalmin`
- `d` = Earth-Sun distance (AU)
- `ESUN` = exoatmospheric solar irradiance from ResourceSat-2 Data Users Handbook:
  - Band2 (Green): 1848.0 W/m²/µm
  - Band3 (Red): 1549.0 W/m²/µm
  - Band4 (NIR): 1044.0 W/m²/µm

Actual calibration values from BAND_META.txt:
- B2: Lmax=52.0, Lmin=0.0 → gain=0.05083
- B3: Lmax=47.0, Lmin=0.0 → gain=0.04594
- B4: Lmax=31.5, Lmin=0.0 → gain=0.03079

### Step 2 — Spatial Crop to Processing AOI

Full LISS-IV scenes (~84 km × 87 km) exceed 16 GB when stacked as 11-channel float32. Cropped each scene to a 30 km × 30 km window (6001 × 6001 pixels at 5m) centered on the cloudiest area (identified via `find_clouds.py`).

Crop coordinates (UTM meters):
- Bengaluru: (738029, 1490000, 768029, 1520000) — upper-right of scene, ~88% cloud score
- Punjab: (490000, 3398000, 520000, 3428000) — upper portion, monsoon cloud zone
- Meghalaya: (340000, 2806000, 370000, 2836000) — upper portion, dense cloud area

### Step 3 — Co-registration

All sources reprojected to match the LISS-IV cloudy scene grid exactly (same CRS, same pixel size, same transform) using `rasterio.warp.reproject`:
- SAR → bilinear resampling (continuous data)
- DEM → bilinear resampling
- Masks → nearest-neighbour (discrete labels)

This ensures pixel-perfect alignment: channel 0 pixel [100,200] and channel 4 pixel [100,200] represent the exact same ground location.

### Step 4 — SAR Normalization

GEE exports SAR in dB scale (sigma0). Normalized to [0,1]:
- VV: clipped to [−40, +35] dB → normalized
- VH: clipped to [−50, +25] dB → normalized

Ranges widened from standard [−25,5]/[−30,0] to cover actual observed urban backscatter (Bengaluru VV max = +31.1 dB).

### Step 5 — Cloud + Shadow Mask Generation

Used LISS-IV band reflectance directly (more reliable than S2 over large offset areas):
- Cloud detection: pixels with visible brightness > 75th percentile AND NIR/visible ratio < 1.5
- Shadow detection: geometric projection from cloud mask using sun azimuth + elevation (from BAND_META.txt), confirmed by dark NIR criterion (NIR < 30% of scene max)
- Mask values: 0.0 = clear, 0.5 = shadow, 1.0 = cloud (stored as /2 to normalize to [0,1])

Cloud coverage in processed windows:
- Bengaluru: 20.4%
- Punjab: 20.0%
- Meghalaya: 18.6%

### Step 6 — Change Detection Mask

NDVI-based change detection between cloudy (current) and clear (reference) dates:
- NDVI = (NIR − Red) / (NIR + Red)
- Pixels where |NDVI_current − NDVI_reference| > 0.20 are flagged as "changed"
- Changed pixels should not use temporal reference for reconstruction

Change percentages:
- Bengaluru: 21.5% (seasonal vegetation + urban construction)
- Punjab: 3.9% (stable post-harvest fields)
- Meghalaya: 57.2% (high seasonal forest phenology change Jul vs Jan)

### Step 7 — 11-Channel Stack Assembly

```
Channel  Content                    Source               Range
0–2      Cloudy LISS-IV G, R, NIR   LISS-IV reflectance  [0, 1]
3        Cloud + shadow mask        s2cloudless + solar  {0, 0.5, 1}
4–5      SAR VV, VH                 Sentinel-1 GEE       [0, 1]
6–8      Clear reference G, R, NIR  LISS-IV reflectance  [0, 1]
9        Change detection mask      NDVI diff            {0, 1}
10       DEM elevation              SRTM GL1 normalised  [0, 1]
```

Final stack shapes:
- Each AOI: (11, 6001, 6001) = 1511 MB per file
- Each clear target: (3, 6001, 6001) = 412 MB per file
- Total processed data: ~6 GB

---

## Phase 3 — Model Architecture

### Architecture: Cross-Modal Attention U-Net

```
INPUT: B × 11 × 128 × 128

ENCODER (4 stages, stride-2 MaxPool):
  enc1: 11 → 32 ch  | 128×128 → 64×64
  enc2: 32 → 64 ch  |  64×64  → 32×32
  enc3: 64 → 128ch  |  32×32  → 16×16
  enc4:128 → 256ch  |  16×16  →  8×8

BOTTLENECK:
  conv: 256 → 512ch |  8×8 spatial  (N = 64 positions)

CROSS-MODAL ATTENTION (at bottleneck):
  Query  = optical stream  → "what needs reconstructing?"
  Key    = SAR stream      → "what structure exists through cloud?"
  Value  = temporal stream → "what did this look like before?"
  dim=512, heads=4, head_dim=128
  N=64 — trivially small, full N×N attention feasible

DECODER (4 stages, 2× transposed conv):
  dec4: 512+256 → 256ch |  8→16
  dec3: 256+128 → 128ch | 16→32
  dec2: 128+64  →  64ch | 32→64
  dec1:  64+32  →  32ch | 64→128

DUAL OUTPUT HEADS:
  recon_head:  Conv2d(32, 3, 1) → Sigmoid  → cloud-free image [0,1]
  uncert_head: Conv2d(32, 3, 1) → Sigmoid  → per-pixel uncertainty [0,1]

OUTPUT: reconstruction (B×3×128×128), uncertainty (B×3×128×128), attn_weights (B×4×64×64)
```

**Total trainable parameters: 6.46M**

**RTX 3050 6GB memory settings:**
- base_filters f=32 (f=64 OOMs)
- tile_size=128 (256×256 OOMs)
- batch_size=2 (batch 8 OOMs)
- mixed_precision=True (mandatory)

### Loss Function: 5-Term Combined Loss

```
L_total = 1.0 × L1 + 0.5 × SSIM + 2.0 × Spectral + 0.3 × Edge + 0.2 × Uncertainty

L1:          Mean absolute error — baseline pixel accuracy
SSIM:        1 − SSIM structural similarity — texture and contrast
Spectral:    L1(NDVI_pred, NDVI_target) + L1(NDWI_pred, NDWI_target)
             → highest weight because ISRO evaluates spectral indices first
Edge:        L1(Sobel(pred), Sobel(target)) — field boundary sharpness
Uncertainty: L1(uncertainty, |pred − target|) — calibration to actual error
```

### Dataset and Tiling

- Tile size: 128 × 128 pixels (= 640m × 640m at 5m resolution)
- Overlap: 16 pixels (stride = 112)
- Cloud coverage filter: 20–95% per tile
- Total usable tiles: 1693 (538 Bengaluru + 686 Punjab + 469 Meghalaya)
- Train/val split: 85/15 random tile split — all 3 AOIs contribute to both sets
- Train tiles: 1439 | Val tiles: 254

**Augmentations (training only):**
- HorizontalFlip p=0.5
- VerticalFlip p=0.5
- RandomRotate90 p=0.5
- Brightness jitter ±10% on optical channels only (not SAR, masks, DEM)

---

## Phase 4 — Training

### Two-Phase Training Strategy

**Phase 1 (epochs 1–15) — Frozen encoder:**
- Encoder weights frozen, only bottleneck + attention + decoder + heads update
- Learning rate: 5×10⁻⁴ with CosineAnnealing (η_min=10⁻⁵)
- Rationale: new attention module converges faster without fighting encoder gradients
- Optimizer: AdamW (weight_decay=10⁻⁴)

**Phase 2 (epochs 16–40) — Full fine-tune:**
- All weights trainable
- Learning rate: 5×10⁻⁵ with CosineAnnealing (η_min=10⁻⁶)
- Gradient clipping: max norm = 1.0

**Training progression:**

| Epoch | Phase | Train Loss | Val Loss | Val SSIM | Val NDVI MAE |
|---|---|---|---|---|---|
| 1 | 1 | 1.0972 | 0.8851 | 0.1461 | 0.0643 |
| 3 | 1 | 0.3009 | 0.2508 | 0.9809 | 0.0530 |
| 15 | 1 | 0.1861 | 0.2698 | 0.9837 | 0.0599 |
| 20 | 2 | 0.1558 | 0.2012 | 0.9867 | 0.0450 |
| 27 | 2 | 0.1295 | 0.1282 | 0.9897 | 0.0290 |
| 31 | 2 | 0.1219 | 0.1179 | 0.9908 | 0.0274 |
| 35 | 2 | 0.1195 | **0.1102** | **0.9929** | **0.0253** |
| 40 | 2 | 0.1148 | 0.2243 | 0.9860 | 0.0492 |

Best checkpoint saved at epoch 35: `models/checkpoints/final_best.pth`

---

## Phase 5 — Evaluation Results

### Per-AOI Metrics (final_best.pth, epoch 35)

| Metric | Bengaluru (Urban) | Punjab (Agricultural) | Meghalaya (Forested) |
|---|---|---|---|
| SSIM ↑ | **0.9945** | 0.9226 | 0.9827 |
| PSNR ↑ (dB) | 54.33 | **55.41** | 52.67 |
| LPIPS ↓ | 0.0017 | **0.0010** | 0.0019 |
| RMSE Green ↓ | 0.0012 | 0.0025 | 0.0017 |
| RMSE Red ↓ | 0.0016 | 0.0025 | 0.0020 |
| RMSE NIR ↓ | 0.0027 | 0.0031 | 0.0033 |
| NDVI MAE ↓ | **0.0294** | 0.0343 | 0.0426 |
| NDWI MAE ↓ | **0.0250** | 0.0330 | 0.0361 |
| Edge F1 ↑ | **0.2993** | 0.1033 | 0.1309 |
| Coverage ↑ | 100% | 100% | 100% |

**All metrics exceed the judge-expected thresholds:** SSIM > 0.65, PSNR > 22 dB, NDVI MAE < 0.10

### Ablation Study

| Variant | SSIM | NDVI MAE | Edge F1 | PSNR (dB) |
|---|---|---|---|---|
| **Full model** | **0.9934** | **0.0259** | **0.2133** | **54.95** |
| No SAR | 0.9764 | 0.0559 | 0.1822 | 49.62 |
| No temporal | 0.9467 | 0.1032 | 0.0016 | 46.80 |
| No DEM | 0.9736 | 0.0357 | 0.1983 | 51.05 |

**Key findings:**
- Removing temporal reference causes NDVI MAE to jump 4× (0.026 → 0.103) — most important input
- Removing SAR doubles NDVI MAE (0.026 → 0.056) — second most important
- Removing DEM degrades all metrics measurably — elevation context matters

---

## Phase 6 — Demo Application

### Streamlit Interactive Dashboard (`demo/app.py`)

Interactive browser-based demo showing:
- Before/After/Ground-Truth panel (false colour: NIR-R-G)
- NDVI comparison (cloudy, ground truth, reconstruction, error map)
- Uncertainty map (per-pixel confidence)
- Attention visualization (spatial attention weights at bottleneck)
- Live metric cards (SSIM, PSNR, NDVI MAE, Coverage)
- Scene selector: Bengaluru / Punjab / Meghalaya
- Architecture explainer panel

**Run command:**
```bash
streamlit run demo/app.py
```
Opens at: http://localhost:8501

### FastAPI REST Endpoint (`demo/api.py`)

Operational deployment proof:
- `GET /health` — model status
- `GET /demo/{aoi}` — inference on pre-loaded demo case, returns JSON metrics
- `GET /metrics` — full evaluation results and ablation table

**Run command:**
```bash
uvicorn demo.api:app --host 0.0.0.0 --port 8000
```

### Full-Scene Reconstruction (`demo/reconstruct_full.py`)

Tiles the full 30×30 km (6001×6001 px) stack, runs tile-by-tile inference,
stitches with **Gaussian-weighted overlap blending** (no visible seams):
- Gaussian window std = tile_size/4 → center pixels weighted ~16× edge pixels
- Accumulates weighted contributions, divides by weight sum

---

## Complete File Structure

```
D:\projects\BAH2026\
├── train.py                          ← main training entry point
├── evaluate.py                       ← full evaluation + ablation + demo cases
├── preprocess.py                     ← end-to-end preprocessing pipeline
├── config.yaml                       ← all hyperparameters
├── test_forward.py                   ← model sanity check
├── find_clouds.py                    ← find cloudiest 30km crop window
├── download_dem.py                   ← SRTM DEM download
├── extract_liss4.py                  ← unzip Bhoonidhi downloads
├── gee_export_all.py                 ← GEE SAR + S2 export to Drive
├── download_from_drive.py            ← download from Google Drive
│
├── models/
│   ├── __init__.py
│   ├── attention.py                  ← CrossModalAttention
│   ├── blocks.py                     ← EncoderBlock, DecoderBlock
│   ├── cloud_removal_net.py          ← CloudRemovalNet (full model)
│   ├── losses.py                     ← CombinedLoss (5 terms)
│   └── checkpoints/
│       ├── phase1_best.pth
│       └── final_best.pth            ← BEST MODEL (epoch 35)
│
├── scripts/
│   ├── dataset/
│   │   └── cloud_dataset.py          ← CloudDataset + tile_and_load
│   ├── train/
│   │   └── trainer.py                ← two-phase training loop
│   └── evaluate/
│       ├── metrics.py                ← 8-metric Evaluator
│       └── ablation.py               ← ablation study runner
│
├── demo/
│   ├── app.py                        ← Streamlit dashboard
│   ├── api.py                        ← FastAPI REST endpoint
│   └── reconstruct_full.py           ← full-scene tile stitching
│
├── ps2_cloud/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── liss4/                ← 6 scenes × 3 bands + metadata
│   │   │   ├── sentinel1/            ← 3 SAR GeoTIFFs (~700 MB each)
│   │   │   ├── sentinel2/            ← 6 S2 GeoTIFFs
│   │   │   └── dem/                  ← 3 SRTM DEMs
│   │   ├── processed/
│   │   │   ├── bengaluru_stack.npy   ← 11×6001×6001 float32 (1.5 GB)
│   │   │   ├── bengaluru_clear.npy   ← 3×6001×6001 float32 (412 MB)
│   │   │   └── ... (same for punjab, meghalaya)
│   │   └── demo_cases/
│   │       ├── bengaluru.npz         ← 11×128×128 + 3×128×128
│   │       ├── punjab.npz
│   │       └── meghalaya.npz
│   └── outputs/
│       └── figures/
│           ├── bengaluru_result.png
│           ├── punjab_result.png
│           └── meghalaya_result.png
│
└── scenes.md                         ← data collection log with product IDs
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Architecture | U-Net with cross-modal attention | U-Net preserves spatial detail via skip connections; attention enables multi-source fusion at bottleneck |
| Base filters | f=32 | f=64 OOMs on RTX 3050 6GB; f=32 gives 6.46M params with enough capacity |
| Attention position | Bottleneck (8×8=64 positions) | Full N×N attention is O(N²); at N=64 it's trivial; placing it here fuses all three data streams at the highest semantic level |
| Loss spectral weight | 2.0 (highest) | ISRO's primary evaluation criterion is NDVI/NDWI fidelity |
| Train/val split | Random tile split across all 3 AOIs | Only 3 scenes — scene-level split wastes 33% of data; random tile split maximizes training signal |
| Cloud mask | LISS-IV band ratio (not s2cloudless) | S2 export footprint doesn't perfectly overlap LISS-IV crop area; LISS-IV-based detection is direct and reliable |
| SAR normalization | [−40,+35] dB for VV | Actual observed range included urban specular returns up to +31 dB; standard [−25,+5] was too narrow |

---

## Streamlit Demo

**Local deployment:**
```bash
streamlit run demo/app.py
```
Opens at: **http://localhost:8501**

**FastAPI endpoint:**
```bash
uvicorn demo.api:app --host 0.0.0.0 --port 8000
```
API at: **http://localhost:8000**

*Note: Cloud deployment link to be added after Streamlit Community Cloud or HuggingFace Spaces deployment.*

---

## Summary

Built a working, evaluated cloud removal system for LISS-IV satellite imagery in under 6 days on a 6GB laptop GPU:

- **Data:** 6 real LISS-IV scenes from Bhoonidhi + SAR + DEM across 3 terrain types
- **Model:** Cross-Modal Attention U-Net, 6.46M parameters, 40 epochs, ~8 hours training
- **Results:** SSIM 0.99, PSNR 54 dB, NDVI MAE 0.026 — all exceeding judge thresholds
- **Ablation:** Proves SAR and temporal reference each contribute meaningfully
- **Demo:** Interactive Streamlit dashboard + FastAPI endpoint

*BAH 2026 | Submission deadline: July 1, 2026 — 11:59 PM IST*
