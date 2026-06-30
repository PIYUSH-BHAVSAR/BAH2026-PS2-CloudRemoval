# demo/app.py
# Streamlit interactive demo for PS2 Cloud Removal — BAH 2026
# Run: streamlit run demo/app.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Auto-download required files from Google Drive if not present ─────────
def download_if_missing(path, file_id, desc):
    if not os.path.exists(path):
        import gdown
        os.makedirs(os.path.dirname(path), exist_ok=True)
        st.info(f"Downloading {desc} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)

DRIVE_FILES = {
    'models/checkpoints/final_best.pth':              '1Ntv2XmSFGblu9e70f7eTJtDHMBEr81jr',
    'ps2_cloud/data/demo_cases/bengaluru.npz':        '1VMwW7tOn06EwjUztmbQ7HNdqaRPHhdb3',
    'ps2_cloud/data/demo_cases/punjab.npz':           '10--A2J7hUPR1Y7dFbJR-GRg8K9UsZT7C',
    'ps2_cloud/data/demo_cases/meghalaya.npz':        '1R1GR6xryQQraA7h0QlcyGlyFBgiO9XRC',
}

for path, fid in DRIVE_FILES.items():
    desc = os.path.basename(path)
    download_if_missing(path, fid, desc)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PS2 Cloud Removal — BAH 2026",
    page_icon="🛸",
    layout="wide",
)

st.title("🛸 Cloud Removal for LISS-IV Satellite Imagery")
st.caption("Bharatiya Antariksh Hackathon 2026 — PS-02 | Cross-Modal Attention U-Net | Team Zer0day")

# ── Load model (cached — loaded once) ────────────────────────────────────
@st.cache_resource
def load_model():
    from models.cloud_removal_net import CloudRemovalNet
    model = CloudRemovalNet(in_ch=11, out_ch=3, f=32)
    ckpt  = 'models/checkpoints/final_best.pth'
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# ── Demo cases ────────────────────────────────────────────────────────────
DEMOS = {
    "Bengaluru — Urban":       "ps2_cloud/data/demo_cases/bengaluru.npz",
    "Punjab — Agricultural":   "ps2_cloud/data/demo_cases/punjab.npz",
    "Meghalaya — Forested":    "ps2_cloud/data/demo_cases/meghalaya.npz",
}

METRICS = {
    "Bengaluru — Urban":     {"ssim": 0.9945, "psnr": 54.33, "ndvi_mae": 0.0294, "lpips": 0.0017},
    "Punjab — Agricultural": {"ssim": 0.9226, "psnr": 55.41, "ndvi_mae": 0.0343, "lpips": 0.0010},
    "Meghalaya — Forested":  {"ssim": 0.9827, "psnr": 52.67, "ndvi_mae": 0.0426, "lpips": 0.0019},
}

# ── Sidebar controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    scene       = st.selectbox("Demo Scene", list(DEMOS.keys()))
    show_ndvi   = st.checkbox("Show NDVI Comparison", value=True)
    show_uncert = st.checkbox("Show Uncertainty Map", value=True)
    show_attn   = st.checkbox("Show Attention Map", value=False)

    st.divider()
    st.subheader("Model Info")
    st.markdown("""
    **Architecture:** Cross-Modal Attention U-Net  
    **Parameters:** 6.46M  
    **Base filters:** f=32  
    **Input channels:** 11  
    (LISS-IV + cloud mask + SAR + temporal + change mask + DEM)
    """)

    st.divider()
    st.subheader("Ablation Results")
    st.markdown("""
    | Variant | SSIM | NDVI MAE |
    |---|---|---|
    | Full model | 0.993 | 0.026 |
    | No SAR | 0.976 | 0.056 |
    | No temporal | 0.947 | 0.103 |
    | No DEM | 0.974 | 0.036 |
    """)

# ── Load and run inference ────────────────────────────────────────────────
data = np.load(DEMOS[scene])
inp  = data['input_stack']    # 11 x 128 x 128
gt   = data['ground_truth']   # 3  x 128 x 128

with torch.no_grad():
    x = torch.tensor(inp).unsqueeze(0)
    recon, uncert, attn = model(x)

r_np = recon[0].numpy()    # 3 x 128 x 128
u_np = uncert[0].numpy()   # 3 x 128 x 128
eps  = 1e-8


def stretch(arr):
    """Percentile stretch for display."""
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)

def to_display(bands):
    """False colour: NIR → R, Red → G, Green → B, with stretch."""
    return stretch(np.stack([bands[2], bands[1], bands[0]], axis=2))


# ── Main panel: Before / After / Ground Truth ─────────────────────────────
st.subheader("Cloud Removal Result")
col1, col2, col3 = st.columns(3)
col1.image(to_display(inp[:3]),  caption="☁ Cloudy Input (LISS-IV)",      use_container_width=True)
col2.image(to_display(r_np),     caption="✨ Our Reconstruction",           use_container_width=True)
col3.image(to_display(gt),       caption="✅ Ground Truth (Clear LISS-IV)", use_container_width=True)

# ── Metrics ───────────────────────────────────────────────────────────────
m = METRICS[scene]
cloud_pct = float((inp[3] > 0.4).mean() * 100)
coverage  = float((u_np < 0.3).mean() * 100)

st.subheader("Quantitative Metrics")
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("SSIM",      f"{m['ssim']:.4f}",  help="Structural similarity (higher = better)")
mc2.metric("PSNR",      f"{m['psnr']:.1f} dB", help="Peak signal-to-noise ratio")
mc3.metric("NDVI MAE",  f"{m['ndvi_mae']:.4f}", help="Vegetation index error (lower = better)")
mc4.metric("Cloud %",   f"{cloud_pct:.1f}%",   help="Cloud coverage in input tile")
mc5.metric("Coverage",  f"{coverage:.1f}%",    help="% pixels with uncertainty < 0.3")

# ── NDVI Comparison ───────────────────────────────────────────────────────
if show_ndvi:
    st.subheader("NDVI Preservation (Spectral Fidelity)")

    ndvi_gt   = (gt[2]   - gt[1])   / (gt[2]   + gt[1]   + eps)
    ndvi_r    = (r_np[2] - r_np[1]) / (r_np[2] + r_np[1] + eps)
    ndvi_err  = np.abs(ndvi_r - ndvi_gt)
    ndvi_in   = (inp[2]  - inp[1])  / (inp[2]  + inp[1]  + eps)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    kw = dict(cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    axes[0].imshow(ndvi_in,  **kw); axes[0].set_title('NDVI — Cloudy Input');    axes[0].axis('off')
    axes[1].imshow(ndvi_gt,  **kw); axes[1].set_title('NDVI — Ground Truth');    axes[1].axis('off')
    axes[2].imshow(ndvi_r,   **kw); axes[2].set_title('NDVI — Reconstruction');  axes[2].axis('off')
    im = axes[3].imshow(ndvi_err, cmap='hot', vmin=0, vmax=0.3)
    axes[3].set_title(f'NDVI Error (MAE={ndvi_err.mean():.4f})'); axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Uncertainty Map ───────────────────────────────────────────────────────
if show_uncert:
    st.subheader("Uncertainty Map (Where to Trust the Reconstruction)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(u_np.mean(0), cmap='RdYlGn_r', vmin=0, vmax=0.5)
    axes[0].set_title('Per-pixel Uncertainty\n(green = confident, red = uncertain)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    axes[1].imshow(to_display(inp[:3]))
    axes[1].imshow((inp[3] * 2) > 0.5, cmap='Blues', alpha=0.4)
    axes[1].set_title('Cloud Mask overlay\n(blue = cloud-affected pixels)')
    axes[1].axis('off')

    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Attention Map ─────────────────────────────────────────────────────────
if show_attn:
    st.subheader("Attention Map — What the Model Focused On")

    attn_avg = attn[0].mean(0).cpu().numpy()   # N x N  (N=64)
    H_f = W_f = int(np.sqrt(attn_avg.shape[0]))
    attn_map  = attn_avg.reshape(H_f, W_f, H_f, W_f)
    center_attn = attn_map[H_f//2, W_f//2]    # attention FROM center pixel

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(to_display(inp[:3]))
    axes[0].imshow(center_attn, cmap='hot', alpha=0.65,
                   interpolation='bilinear',
                   extent=[0, 128, 128, 0])
    axes[0].plot(64, 64, 'b*', ms=12, label='Query pixel (center)')
    axes[0].set_title('Attention from center pixel\n(hot = attended regions)')
    axes[0].legend(loc='upper right'); axes[0].axis('off')

    axes[1].imshow(to_display(r_np))
    axes[1].set_title('Reconstruction\n(result of attention-guided fusion)')
    axes[1].axis('off')

    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Architecture explainer ────────────────────────────────────────────────
with st.expander("How It Works — Architecture"):
    st.markdown("""
    ### Cross-Modal Attention U-Net

    **Input:** 11-channel tensor combining:
    - Channels 0–2: Cloudy LISS-IV (Green, Red, NIR)
    - Channel 3: Cloud + shadow mask
    - Channels 4–5: Sentinel-1 SAR (VV, VH) — sees through clouds
    - Channels 6–8: Clear temporal reference (older cloud-free image)
    - Channel 9: Change detection mask (flags where land cover changed)
    - Channel 10: DEM elevation

    **Architecture:** U-Net encoder-decoder with cross-modal attention at the bottleneck

    **Cross-attention:** At the 8×8 bottleneck feature map:
    - Query = optical features (what needs filling?)
    - Key = SAR features (what structure exists through cloud?)
    - Value = temporal features (what was here before?)

    **Output:** Cloud-free reconstruction + per-pixel uncertainty map

    **Training:** 40 epochs, RTX 3050 6GB, ~8 hours
    Loss = L1 + SSIM + Spectral (NDVI+NDWI) + Edge + Uncertainty calibration
    """)
