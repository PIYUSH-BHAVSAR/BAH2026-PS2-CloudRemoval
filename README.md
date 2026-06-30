# PS2 — Cloud Removal for LISS-IV Satellite Imagery
## Bharatiya Antariksh Hackathon 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cloudremoval.streamlit.app/)

**🚀 Live Demo: https://cloudremoval.streamlit.app/**

Generative AI-based cloud removal and reconstruction for LISS-IV (ResourceSat-2) satellite imagery using Cross-Modal Attention U-Net.

## Demo

```bash
streamlit run demo/app.py
```

## Results

| Metric | Bengaluru | Punjab | Meghalaya |
|---|---|---|---|
| SSIM | 0.9945 | 0.9226 | 0.9827 |
| PSNR (dB) | 54.33 | 55.41 | 52.67 |
| NDVI MAE | 0.0294 | 0.0343 | 0.0426 |

## Architecture

Cross-Modal Attention U-Net — 6.46M parameters
- Input: 11 channels (LISS-IV + cloud mask + SAR + temporal reference + change mask + DEM)
- Cross-modal attention at bottleneck fuses optical, SAR, and temporal streams
- Output: cloud-free reconstruction + per-pixel uncertainty map

## Setup

```bash
pip install -r requirements.txt
streamlit run demo/app.py
```

Model weights and demo cases auto-download from Google Drive on first run.

## Data Sources

- LISS-IV: Bhoonidhi portal (ISRO/NRSC)
- Sentinel-1 SAR: Google Earth Engine
- DEM: SRTM GL1 via OpenTopography

See `SOLUTION_IMPLEMENTATION.md` for full details.
