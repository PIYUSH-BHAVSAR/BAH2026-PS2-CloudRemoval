# PHASE 5 — Demo App + Submission
## PS2 | BAH 2026
### Goal: Working Streamlit demo, FastAPI endpoint, submission-ready package

---

## What Phase 5 Produces

```
demo/
├── app.py              ← Streamlit interactive dashboard (judges see this)
├── api.py              ← FastAPI REST endpoint (operational readiness proof)
└── reconstruct_full.py ← Full-scene tile stitching with Gaussian blending
```

---

## Part 1 — Streamlit Demo App (`demo/app.py`)

The interactive dashboard shown to judges during the 30-hour finale.

### What it shows:
```
┌─────────────────────────────────────────────────────────┐
│  PS2 Cloud Removal — BAH 2026                           │
│                                                          │
│  Scene: [Bengaluru ▼]                                   │
│                                                          │
│  [Cloudy Input]  [Reconstruction]  [Ground Truth]       │
│                                                          │
│  SSIM: 0.9945   NDVI MAE: 0.0294   Coverage: 100%       │
│                                                          │
│  ☑ Show NDVI comparison                                  │
│  ☑ Show uncertainty map                                  │
│  ☑ Show attention visualization                          │
└─────────────────────────────────────────────────────────┘
```

### Panels:
1. **Before/After** — cloudy input vs reconstructed output vs ground truth
2. **NDVI comparison** — NDVI map of ground truth, reconstruction, and error
3. **Uncertainty map** — where the model is confident vs uncertain
4. **Attention map** — which spatial regions the model attended to most
5. **Live metrics** — SSIM, PSNR, NDVI MAE shown as metric cards

### How to run:
```bash
streamlit run demo/app.py
```
Opens at http://localhost:8501 — works in any browser on the same PC.

---

## Part 2 — FastAPI Endpoint (`demo/api.py`)

Proves the system is operationally deployable, not just a research script.

### Endpoints:
```
GET  /health          → {"status": "ok", "model": "CloudRemovalNet"}
POST /reconstruct     → accepts 11-channel GeoTIFF, returns reconstruction stats
GET  /demo/{aoi}      → returns pre-computed demo result for bengaluru/punjab/meghalaya
```

### How to run:
```bash
uvicorn demo.api:app --host 0.0.0.0 --port 8000
```

Test with:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/demo/bengaluru
```

---

## Part 3 — Full-Scene Reconstruction (`demo/reconstruct_full.py`)

Tiles the full 6000×6000 LISS-IV scene into 128×128 patches, runs inference
on each, and stitches back together with Gaussian-weighted overlap blending.

No visible tile seams — center pixels of each tile weighted ~4× more than edges.

### Output:
- Full 30×30 km cloud-free reconstruction at 5m resolution
- Saved as GeoTIFF with correct spatial reference (CRS + transform preserved)
- Used to generate the full-scene figure for the presentation

---

## Demo Cases (already saved from evaluate.py)

```
ps2_cloud/data/demo_cases/
├── bengaluru.npz   → input_stack (11×128×128) + ground_truth (3×128×128)
├── punjab.npz
└── meghalaya.npz
```

Each .npz has ~60% cloud coverage — visually striking for demo.

---

## Hardware Note

- Run Streamlit from CPU: `torch.load(..., map_location='cpu')`
- Do NOT run training simultaneously with the demo
- Close Chrome tabs before running — shared iGPU VRAM

---

## Files to Create

| File | Purpose | Priority |
|---|---|---|
| `demo/__init__.py` | Package init | Low |
| `demo/app.py` | Streamlit dashboard | HIGH |
| `demo/api.py` | FastAPI endpoint | Medium |
| `demo/reconstruct_full.py` | Full-scene stitching | Medium |

---

## Submission Checklist

```
DEMO
  [ ] streamlit run demo/app.py — opens without errors
  [ ] All 3 scenes load and display correctly
  [ ] Metrics (SSIM, NDVI MAE) show correct values
  [ ] NDVI comparison panel works
  [ ] Uncertainty map displays
  [ ] FastAPI /health returns 200 OK
  [ ] Full-scene reconstruction runs for bengaluru

SUBMISSION PACKAGE
  [ ] models/checkpoints/final_best.pth  ← trained model
  [ ] ps2_cloud/data/demo_cases/*.npz    ← 3 demo cases
  [ ] ps2_cloud/outputs/figures/*.png    ← result figures
  [ ] evaluate.py results recorded       ← metrics for slides
  [ ] ablation table recorded            ← for slides

PRESENTATION SLIDES
  [ ] Architecture diagram (U-Net with cross-attention)
  [ ] Training curves screenshot (loss going down, SSIM going up)
  [ ] Results table (SSIM, PSNR, NDVI MAE per AOI)
  [ ] Ablation table (Full / No SAR / No temporal / No DEM)
  [ ] Before/After figures from outputs/figures/
  [ ] Demo screenshot

DEADLINE: July 1, 2026 — 11:59 PM IST
```

---

## Expected Metrics for Slides

| Metric | Bengaluru | Punjab | Meghalaya |
|---|---|---|---|
| SSIM | 0.9945 | 0.9226 | 0.9827 |
| PSNR (dB) | 54.33 | 55.41 | 52.67 |
| LPIPS | 0.0017 | 0.0010 | 0.0019 |
| NDVI MAE | 0.0294 | 0.0343 | 0.0426 |
| NDWI MAE | 0.0250 | 0.0330 | 0.0361 |
| Coverage | 100% | 100% | 100% |

## Ablation Results for Slides

| Variant | SSIM | NDVI MAE | PSNR |
|---|---|---|---|
| Full model | 0.9934 | 0.0259 | 54.95 |
| No SAR | 0.9764 | 0.0559 | 49.62 |
| No temporal | 0.9467 | 0.1032 | 46.80 |
| No DEM | 0.9736 | 0.0357 | 51.05 |

---

**→ Implement: demo/app.py → demo/api.py → demo/reconstruct_full.py**
