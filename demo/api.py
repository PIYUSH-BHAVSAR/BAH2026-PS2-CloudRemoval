# demo/api.py
# FastAPI REST endpoint for cloud removal inference.
# Run: uvicorn demo.api:app --host 0.0.0.0 --port 8000
# Test: curl http://localhost:8000/health

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(
    title="PS2 Cloud Removal API",
    description="Generative AI cloud removal for LISS-IV satellite imagery — BAH 2026",
    version="1.0.0",
)

model      = None
demo_cache = {}


@app.on_event("startup")
def load_model():
    global model
    from models.cloud_removal_net import CloudRemovalNet
    model = CloudRemovalNet(in_ch=11, out_ch=3, f=32)
    ckpt  = 'models/checkpoints/final_best.pth'
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    print(f"Model loaded: {ckpt}")

    # Pre-load demo cases
    for aoi in ['bengaluru', 'punjab', 'meghalaya']:
        p = f'ps2_cloud/data/demo_cases/{aoi}.npz'
        if os.path.exists(p):
            demo_cache[aoi] = np.load(p)
    print(f"Demo cases loaded: {list(demo_cache.keys())}")


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   "CloudRemovalNet",
        "version": "1.0.0",
        "params":  "6.46M",
        "device":  "cpu",
    }


@app.get("/demo/{aoi}")
def demo_inference(aoi: str):
    """
    Run inference on a pre-loaded demo case.
    aoi: one of bengaluru, punjab, meghalaya
    Returns reconstruction statistics.
    """
    if aoi not in demo_cache:
        raise HTTPException(
            status_code=404,
            detail=f"AOI '{aoi}' not found. Available: {list(demo_cache.keys())}"
        )

    data  = demo_cache[aoi]
    inp   = data['input_stack']    # 11 x 128 x 128
    gt    = data['ground_truth']   # 3  x 128 x 128

    x = torch.tensor(inp).unsqueeze(0)
    with torch.no_grad():
        recon, uncert, _ = model(x)

    r_np  = recon[0].numpy()
    u_np  = uncert[0].numpy()
    eps   = 1e-8

    # NDVI
    ndvi_r  = (r_np[2]  - r_np[1])  / (r_np[2]  + r_np[1]  + eps)
    ndvi_gt = (gt[2]    - gt[1])    / (gt[2]    + gt[1]    + eps)
    ndvi_mae= float(np.abs(ndvi_r - ndvi_gt).mean())

    # Cloud coverage in input
    cloud_pct = float((inp[3] > 0.4).mean() * 100)

    return JSONResponse({
        "aoi":               aoi,
        "status":            "success",
        "reconstruction":    "cloud-free LISS-IV generated",
        "input_cloud_pct":   round(cloud_pct, 1),
        "mean_uncertainty":  round(float(u_np.mean()), 4),
        "coverage_pct":      round(float((u_np < 0.3).mean() * 100), 1),
        "ndvi_mae":          round(ndvi_mae, 4),
        "output_shape":      list(r_np.shape),
        "model":             "CloudRemovalNet (Cross-Modal Attention U-Net)",
        "inputs_used":       ["LISS-IV optical", "Sentinel-1 SAR", "temporal reference", "DEM"],
    })


@app.get("/metrics")
def get_metrics():
    """Return evaluated metrics from training run."""
    return {
        "model": "CloudRemovalNet f=32 (6.46M params)",
        "training": {"epochs": 40, "phase1": 15, "phase2": 25},
        "results": {
            "bengaluru": {"ssim": 0.9945, "psnr": 54.33, "ndvi_mae": 0.0294, "lpips": 0.0017},
            "punjab":    {"ssim": 0.9226, "psnr": 55.41, "ndvi_mae": 0.0343, "lpips": 0.0010},
            "meghalaya": {"ssim": 0.9827, "psnr": 52.67, "ndvi_mae": 0.0426, "lpips": 0.0019},
        },
        "ablation": {
            "full_model":  {"ssim": 0.9934, "ndvi_mae": 0.0259, "psnr": 54.95},
            "no_sar":      {"ssim": 0.9764, "ndvi_mae": 0.0559, "psnr": 49.62},
            "no_temporal": {"ssim": 0.9467, "ndvi_mae": 0.1032, "psnr": 46.80},
            "no_dem":      {"ssim": 0.9736, "ndvi_mae": 0.0357, "psnr": 51.05},
        },
    }
