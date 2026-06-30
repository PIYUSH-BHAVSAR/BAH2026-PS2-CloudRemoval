# Collected Scenes — PS2 BAH 2026
## LISS-IV + Auxiliary Data Inventory
### Last updated: 29-Jun-2026

---

## Status Overview

| AOI | LISS-IV Cloudy | LISS-IV Clear | SAR (GEE) | S2 (GEE) | DEM |
|---|---|---|---|---|---|
| **Bengaluru** | ✅ Collected | ✅ Collected | ✅ Done | ✅ Done | ✅ Done |
| **Punjab** | ✅ Collected | ✅ Collected | ✅ Done | ✅ Done | ✅ Done |
| **Meghalaya** | ✅ Collected | ✅ Collected | ✅ Done | ✅ Done | ✅ Done |

---

## LISS-IV Scenes

### Bengaluru — CLOUDY ✅

```
Satellite/Sensor : ResourceSat-2 LISS-IV (RS2_LIS4_-_F_L2)
Product ID       : R2F11JUN2026078587009900064SSANSTUC00GTD
Scene            : 078587_99_64
Date             : 11-Jun-2026
Type             : CLOUDY (monsoon season)

Coverage:
  West  : 76.699°E
  East  : 77.533°E
  North : 13.745°N
  South : 12.980°N
  Center: 13.363°N, 77.116°E
  Width : ~74 km (E–W)
  Height: ~85 km (N–S)

Bands downloaded:
  BAND2.tif   ← Green
  BAND3.tif   ← Red
  BAND4.tif   ← NIR
  BAND_META.txt

Local path: data/raw/liss4/bengaluru_cloudy/
```

> **Note:** AOI in plan used `[77.45, 12.85, 77.75, 13.1]`.
> Actual downloaded scene covers `[76.699, 12.980, 77.533, 13.745]` — wider coverage.
> Crop to `[76.90, 13.00, 77.50, 13.70]` during preprocessing to remove edge artifacts.
> All plan code still works — co-registration handles the grid alignment.

---

### Bengaluru — CLEAR ✅

```
Satellite/Sensor : ResourceSat-2 LISS-IV (RS2_LIS4_-_F_L2)
Product ID       : R2F18JAN2026076548009900064SSANSTUC00GTD
Date             : 18-Jan-2026
Type             : CLEAR (winter — near-zero cloud cover)

Coverage:
  West  : 76.715°E
  East  : 77.550°E
  North : 13.746°N
  South : 12.979°N
  Center: 13.363°N, 77.132°E
  Width : ~73 km (E–W)
  Height: ~85 km (N–S)

Visual quality:
  Cloud cover  : Almost zero ✅
  Appearance   : Deep dark red throughout (strong NIR response — healthy vegetation)
  Urban texture: Clearly visible
  Anomalies    : Tiny dark blue spots = water bodies / shadows (normal, not clouds)

Bands downloaded:
  BAND2.tif   ← Green
  BAND3.tif   ← Red
  BAND4.tif   ← NIR
  BAND_META.txt

Local path: data/raw/liss4/bengaluru_clear/
```

> **Spatial match with cloudy scene ✅ CONFIRMED — good to proceed**
> Cloudy coverage: `76.699–77.533°E`, `12.980–13.745°N`
> Clear coverage:  `76.715–77.550°E`, `12.979–13.746°N`
> Difference: < 0.02° on all sides (~2 km max) — near-perfect overlap.
> Co-registration will handle the residual offset. No special crop needed.

---

### Punjab — CLOUDY ✅

```
Satellite/Sensor : ResourceSat-2 LISS-IV (RS2_LIS4_-_F_L2)
Scene            : 073728_93_49
Path/Row         : 93 / 49
Date             : 04-Jul-2025
Type             : CLOUDY (monsoon season)

Coverage:
  West  : 74.758°E
  East  : 75.723°E
  North : 30.991°N
  South : 30.222°N
  Width : ~83 km (E–W)
  Height: ~86 km (N–S)

Bands downloaded:
  BAND2.tif   ← Green
  BAND3.tif   ← Red
  BAND4.tif   ← NIR
  BAND_META.txt

Local path: data/raw/liss4/punjab_cloudy/
```

---

### Punjab — CLEAR ✅

```
Satellite/Sensor : ResourceSat-2 LISS-IV (RS2_LIS4_-_F_L2)
Scene            : 075774_93_49
Path/Row         : 93 / 49
Date             : 25-Nov-2025
Type             : CLEAR (post-harvest rabi season — zero cloud)

Coverage:
  West  : 74.757°E
  East  : 75.723°E
  North : 30.989°N
  South : 30.224°N
  Width : ~83 km (E–W)
  Height: ~86 km (N–S)

Visual quality:
  Cloud cover  : Almost zero ✅
  Appearance   : Very dark / near black (bare agricultural fields post-harvest)
  Texture      : Field patterns visible in bottom portion
  Anomalies    : None — clean clear scene

Bands downloaded:
  BAND2.tif   ← Green
  BAND3.tif   ← Red
  BAND4.tif   ← NIR
  BAND_META.txt

Local path: data/raw/liss4/punjab_clear/
```

> **Spatial match ✅ PERFECT — same Path 93, Row 49**
> Cloudy: `74.758–75.723°E`, `30.222–30.991°N`
> Clear:  `74.757–75.723°E`, `30.224–30.989°N`
> Difference: < 0.5 km on all sides — identical coverage, zero mismatch.

---

### Meghalaya — CLOUDY ✅

```
Satellite/Sensor : ResourceSat-2 LISS-IV (RS2_LIS4_-_F_L2)
Product ID       : R2F17JUL2025073912011000054SSANSTUC00GTD
Scene            : 073912_110_54
Path/Row         : 110 / 54
Date             : 17-Jul-2025
Type             : CLOUDY (monsoon season)
Roll             : 2.209320°
Quality          : Unknown (no preview available on portal — normal for LISS-IV)

Coverage:
  Top Left     : 25.635°N, 91.267°E
  Top Right    : 25.643°N, 92.184°E
  Bottom Right : 24.900°N, 92.189°E
  Bottom Left  : 24.892°N, 91.277°E
  Center       : 25.268°N, 91.729°E
  Width        : ~80 km (E–W)
  Height       : ~83 km (N–S)

Target AOI fits inside scene:
  AOI [91.565–91.835°E, 25.415–25.685°N] ✅ fully covered

Bands downloaded:
  BAND2.tif   ← Green
  BAND3.tif   ← Red
  BAND4.tif   ← NIR
  BAND_META.txt

Local path: data/raw/liss4/meghalaya_cloudy/
```

> **Note:** No preview shown on Bhoonidhi portal — this is a known portal limitation
> for some LISS-IV scenes, not a data quality issue. Download and verify bands locally.

---

### Meghalaya — CLEAR ✅

```
Satellite/Sensor : ResourceSat-2 LISS-IV (RS2_LIS4_-_F_L2)
Product ID       : R2F25JAN2026076640011000054SSANSTUC00GTD
Scene            : 076640_110_54
Path/Row         : 110 / 54
Date             : 25-Jan-2026
Type             : CLEAR (winter — expected low cloud cover)
Roll             : 1.809402°
Quality          : Unknown (no preview available on portal)

Coverage:
  Top Left     : 25.637°N, 91.228°E
  Top Right    : 25.645°N, 92.147°E
  Bottom Right : 24.899°N, 92.152°E
  Bottom Left  : 24.891°N, 91.239°E
  Center       : 25.269°N, 91.691°E
  Width        : ~80 km (E–W)
  Height       : ~83 km (N–S)

Bands downloaded:
  BAND2.tif   ← Green
  BAND3.tif   ← Red
  BAND4.tif   ← NIR
  BAND_META.txt

Local path: data/raw/liss4/meghalaya_clear/
```

> **Spatial match ✅ CONFIRMED — same Path 110, Row 54**
> Cloudy center: `25.268°N, 91.729°E`
> Clear center:  `25.269°N, 91.691°E`
> Difference: ~3.5 km center offset — same orbit path, near-identical coverage.
> Co-registration handles residual offset cleanly.
>
> **⚠️ Visual quality unverified** — no preview available for either scene on portal.
> After download, open in QGIS or run a quick rasterio read to confirm:
> 1. Cloudy scene actually has cloud cover (check NIR band histogram — should show bimodal)
> 2. Clear scene is actually clear (NIR should show smooth unimodal distribution)
> If cloudy scene turns out cloud-free, search for a different date (Jun–Aug 2025/2026).

---

## GEE Exports (Sentinel-1 SAR + Sentinel-2)

```
Status : ✅ DONE

bengaluru_S1_sar.tif    → 674.2 MB  ps2_cloud/data/raw/sentinel1/
punjab_S1_sar.tif       → 761.8 MB  ps2_cloud/data/raw/sentinel1/
meghalaya_S1_sar.tif    → 751.2 MB  ps2_cloud/data/raw/sentinel1/
bengaluru_S2_cloudy.tif →  45.1 MB  ps2_cloud/data/raw/sentinel2/
bengaluru_S2_clear.tif  → 172.7 MB  ps2_cloud/data/raw/sentinel2/
punjab_S2_cloudy.tif    →  85.2 MB  ps2_cloud/data/raw/sentinel2/
punjab_S2_clear.tif     → 418.0 MB  ps2_cloud/data/raw/sentinel2/
meghalaya_S2_cloudy.tif →  10.1 MB  ps2_cloud/data/raw/sentinel2/
meghalaya_S2_clear.tif  →  95.2 MB  ps2_cloud/data/raw/sentinel2/

Source    : Google Earth Engine → Google Drive → local PC
GEE project: aidsync-4e460
Scale     : 10m, EPSG:4326
```

---

## DEM

```
Status : ✅ DONE

dem_bengaluru.tif  → 2763×3065 px | min=627m  max=1338m | Deccan plateau
dem_punjab.tif     → 2770×3479 px | min=132m  max=279m  | Indo-Gangetic plain
dem_meghalaya.tif  → 2712×3460 px | min=-42m  max=1964m | River valleys to hills

Local path: ps2_cloud/data/raw/dem/
Source    : SRTM GL1 (~30m) via OpenTopography AWS
```

---

## Download Log

| Date | Item | Source | Status | Notes |
|---|---|---|---|---|
| 11-Jun-2026 | Bengaluru LISS-IV cloudy | Bhoonidhi | ✅ Done | Scene 078587_99_64, Product R2F11JUN2026078587009900064SSANSTUC00GTD |
| 18-Jan-2026 | Bengaluru LISS-IV clear  | Bhoonidhi | ⬇️ Downloading | Scene 076541_99_64 — zero cloud, spatial match confirmed |
| 04-Jul-2025 | Punjab LISS-IV cloudy    | Bhoonidhi | ⬇️ Downloading | Scene 073728_93_49, Path 93 Row 49 |
| 25-Nov-2025 | Punjab LISS-IV clear     | Bhoonidhi | ⬇️ Downloading | Scene 075774_93_49, Path 93 Row 49 — perfect spatial match |
| 17-Jul-2025 | Meghalaya LISS-IV cloudy | Bhoonidhi | ✅ Done | Scene 073912_110_54 — no preview on portal, verify after download |
| 25-Jan-2026 | Meghalaya LISS-IV clear  | Bhoonidhi | ✅ Done | Scene 076640_110_54 — same path/row, ~3.5km center offset only |
| 29-Jun-2026 | All 9 SAR + S2 GeoTIFFs  | GEE → Drive | ✅ Done | bengaluru/punjab/meghalaya × S1_sar + S2_cloudy + S2_clear |
| 29-Jun-2026 | All 3 DEM files          | OpenTopography SRTM | ✅ Done | dem_bengaluru/punjab/meghalaya.tif |

---

## Next Actions

1. ✅ ~~All LISS-IV scenes — done~~
2. ✅ ~~All SAR + S2 GEE exports — done~~
3. ✅ ~~All DEMs — done~~
4. ✅ **Preprocessing complete → all 6 stacks saved**

### Processed Stack Details
```
bengaluru_stack.npy  → (11, 6001, 6001)  1511 MB  cloud=20.4%  change=21.5%
punjab_stack.npy     → (11, 6001, 6001)  1511 MB  cloud=20.0%  change=3.9%
meghalaya_stack.npy  → (11, 6001, 6001)  1511 MB  cloud=18.6%  change=57.2%
bengaluru_clear.npy  → (3,  6001, 6001)   412 MB
punjab_clear.npy     → (3,  6001, 6001)   412 MB
meghalaya_clear.npy  → (3,  6001, 6001)   412 MB
```

5. 🔲 **Phase 3 — Model code** ← NEXT

### Meghalaya Search Parameters
```
Portal  : Bhoonidhi → ResourceSat-2 → LISS-IV-MX
AOI     : West 91.565°E  East 91.835°E  North 25.685°N  South 25.415°N
CLOUDY  : Date Jun–Aug 2025 or 2026 | Cloud cover 30–100%
CLEAR   : Date Nov 2025 – Feb 2026   | Cloud cover 0–10%
Note    : Meghalaya = validation scene, held out entirely from training
```

---

*→ When all ✅ in status table, proceed to PHASE_2_PREPROCESSING.md*
