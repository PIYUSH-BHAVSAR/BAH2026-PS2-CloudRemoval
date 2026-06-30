"""
evaluate.py — full evaluation on all 3 AOIs after training
Run: python evaluate.py

Computes:
  - All 8 metrics (SSIM, PSNR, LPIPS, RMSE per band, NDVI MAE, NDWI MAE, Edge F1, Coverage)
  - Ablation study (Full / No SAR / No temporal / No DEM)
  - Saves demo .npz cases for Streamlit app
  - Saves comparison figures
"""
import sys, os
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.cloud_removal_net import CloudRemovalNet
from scripts.evaluate.metrics import Evaluator
from scripts.evaluate.ablation import run_ablation
from scripts.dataset.cloud_dataset import CloudDataset
from torch.utils.data import DataLoader

os.makedirs('ps2_cloud/data/demo_cases', exist_ok=True)
os.makedirs('ps2_cloud/outputs/figures', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Load model ────────────────────────────────────────────────────────────
model = CloudRemovalNet(in_ch=11, out_ch=3, f=32).to(device)
ckpt  = 'models/checkpoints/final_best.pth'
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
print(f"Loaded: {ckpt}\n")

evaluator = Evaluator(device=device)

TILE  = 128
STRIDE= TILE - 16

all_aoi_metrics = {}

for aoi in ['bengaluru', 'punjab', 'meghalaya']:
    print(f"{'='*55}")
    print(f"Evaluating: {aoi.upper()}")
    print(f"{'='*55}")

    stack = np.load(f'ps2_cloud/data/processed/{aoi}_stack.npy')
    clear = np.load(f'ps2_cloud/data/processed/{aoi}_clear.npy')
    _, H, W = stack.shape

    metrics_list = []

    with torch.no_grad():
        for r in range(0, H - TILE + 1, STRIDE):
            for c in range(0, W - TILE + 1, STRIDE):
                x = torch.tensor(stack[:, r:r+TILE, c:c+TILE]).unsqueeze(0).to(device)
                y = torch.tensor(clear[:, r:r+TILE, c:c+TILE]).unsqueeze(0).to(device)
                recon, uncert, _ = model(x)
                metrics_list.append(evaluator.compute_all(recon, y, uncert))

    avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    all_aoi_metrics[aoi] = avg

    print(f"  Tiles evaluated: {len(metrics_list)}")
    for k, v in sorted(avg.items()):
        print(f"  {k:20s}: {v:.4f}")
    print()

# ── Summary table ─────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("SUMMARY — All AOIs")
print(f"{'='*55}")
keys = ['ssim', 'psnr', 'lpips', 'ndvi_mae', 'ndwi_mae', 'edge_f1']
print(f"{'Metric':<15}", end='')
for aoi in ['bengaluru', 'punjab', 'meghalaya']:
    print(f"  {aoi:>12}", end='')
print()
print("─" * 55)
for k in keys:
    print(f"{k:<15}", end='')
    for aoi in ['bengaluru', 'punjab', 'meghalaya']:
        print(f"  {all_aoi_metrics[aoi][k]:>12.4f}", end='')
    print()

# ── Ablation study on all tiles ───────────────────────────────────────────
print(f"\n{'='*55}")
print("ABLATION STUDY")
print(f"{'='*55}")

all_tiles_x, all_tiles_y = [], []
for aoi in ['bengaluru', 'punjab', 'meghalaya']:
    stack = np.load(f'ps2_cloud/data/processed/{aoi}_stack.npy')
    clear = np.load(f'ps2_cloud/data/processed/{aoi}_clear.npy')
    cloud_ch = stack[3] * 2
    _, H, W = stack.shape
    for r in range(0, H - TILE + 1, STRIDE):
        for c in range(0, W - TILE + 1, STRIDE):
            cov = (cloud_ch[r:r+TILE, c:c+TILE] > 0).mean()
            if 0.2 <= cov <= 0.95:
                all_tiles_x.append(stack[:, r:r+TILE, c:c+TILE])
                all_tiles_y.append(clear[:, r:r+TILE, c:c+TILE])

abl_loader = DataLoader(
    CloudDataset(all_tiles_x, all_tiles_y, augment=False),
    batch_size=2, shuffle=False, num_workers=0
)
run_ablation(model, abl_loader, evaluator, device)

# ── Save demo cases ───────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("Saving demo cases...")
for aoi in ['bengaluru', 'punjab', 'meghalaya']:
    stack = np.load(f'ps2_cloud/data/processed/{aoi}_stack.npy')
    clear = np.load(f'ps2_cloud/data/processed/{aoi}_clear.npy')
    cloud_ch = stack[3] * 2
    best_r, best_c, best_cov = 0, 0, 0
    for r in range(0, stack.shape[1] - TILE, TILE):
        for c in range(0, stack.shape[2] - TILE, TILE):
            cov = (cloud_ch[r:r+TILE, c:c+TILE] > 0).mean()
            if abs(cov - 0.6) < abs(best_cov - 0.6):
                best_r, best_c, best_cov = r, c, cov
    demo_x = stack[:, best_r:best_r+TILE, best_c:best_c+TILE]
    demo_y = clear[:, best_r:best_r+TILE, best_c:best_c+TILE]
    out = f'ps2_cloud/data/demo_cases/{aoi}.npz'
    np.savez(out, input_stack=demo_x, ground_truth=demo_y)
    print(f"  Saved {out}  (cloud={best_cov*100:.0f}%)")

# ── Comparison figures ────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("Saving comparison figures...")

def stretch(arr):
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)

for aoi in ['bengaluru', 'punjab', 'meghalaya']:
    data   = np.load(f'ps2_cloud/data/demo_cases/{aoi}.npz')
    inp    = data['input_stack']    # 11 x 128 x 128
    gt     = data['ground_truth']   # 3  x 128 x 128

    with torch.no_grad():
        x     = torch.tensor(inp).unsqueeze(0).to(device)
        recon, uncert, attn = model(x)

    r_np  = recon[0].cpu().numpy()   # 3 x 128 x 128
    u_np  = uncert[0].cpu().numpy()  # 3 x 128 x 128

    def fc(arr):  # false colour: NIR-R-G
        return stretch(np.stack([arr[2], arr[1], arr[0]], axis=2))

    eps     = 1e-8
    ndvi_gt = (gt[2]-gt[1])/(gt[2]+gt[1]+eps)
    ndvi_r  = (r_np[2]-r_np[1])/(r_np[2]+r_np[1]+eps)
    ndvi_err= np.abs(ndvi_r - ndvi_gt)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'{aoi.upper()} — Cloud Removal Result  '
                 f'(SSIM={all_aoi_metrics[aoi]["ssim"]:.4f}  '
                 f'NDVI MAE={all_aoi_metrics[aoi]["ndvi_mae"]:.4f})',
                 fontsize=12)

    axes[0].imshow(fc(inp[:3]));     axes[0].set_title('Cloudy Input');       axes[0].axis('off')
    axes[1].imshow(fc(r_np));        axes[1].set_title('Reconstruction');      axes[1].axis('off')
    axes[2].imshow(fc(gt));          axes[2].set_title('Ground Truth (Clear)');axes[2].axis('off')
    im3 = axes[3].imshow(u_np.mean(0), cmap='RdYlGn_r', vmin=0, vmax=0.5)
    axes[3].set_title('Uncertainty'); axes[3].axis('off'); plt.colorbar(im3, ax=axes[3], fraction=0.046)
    im4 = axes[4].imshow(ndvi_err, cmap='hot', vmin=0, vmax=0.3)
    axes[4].set_title(f'NDVI Error\n(MAE={ndvi_err.mean():.4f})'); axes[4].axis('off')
    plt.colorbar(im4, ax=axes[4], fraction=0.046)

    plt.tight_layout()
    out = f'ps2_cloud/outputs/figures/{aoi}_result.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")

print("\nEvaluation complete.")
print("Next: python demo/app.py  (Streamlit demo)")
