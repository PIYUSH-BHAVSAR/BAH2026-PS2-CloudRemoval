# demo/reconstruct_full.py
# Full-scene tile stitching with Gaussian-weighted overlap blending.
# Eliminates visible tile boundary seams in the output.

import torch
import numpy as np
from scipy.signal import windows as sig_windows


def reconstruct_scene(model, stack_11ch, tile_size=128, overlap=32, device='cpu'):
    """
    Run tile-by-tile inference and stitch with Gaussian-weighted blending.

    Gaussian weighting: center pixels of each tile count ~4x more than edges.
    Where tiles overlap, contributions are averaged weighted by the Gaussian.
    Result: seamless full-scene reconstruction.

    Args:
        model      : CloudRemovalNet in eval() mode
        stack_11ch : numpy array  11 x H x W  float32 [0,1]
        tile_size  : must match training (128)
        overlap    : overlap between adjacent tiles (32 recommended)
        device     : 'cuda' or 'cpu' — use 'cpu' when running alongside Streamlit

    Returns:
        reconstruction : numpy  3 x H x W  float32 [0,1]
        uncertainty    : numpy  3 x H x W  float32 [0,1]
    """
    _, H, W = stack_11ch.shape
    stride  = tile_size - overlap

    out_recon  = np.zeros((3, H, W), dtype=np.float32)
    out_uncert = np.zeros((3, H, W), dtype=np.float32)
    weights    = np.zeros((1, H, W), dtype=np.float32)

    # 2D Gaussian window — center ~4x weight of corners
    g1d  = sig_windows.gaussian(tile_size, std=tile_size / 4.0).astype(np.float32)
    g2d  = np.outer(g1d, g1d)[np.newaxis]   # 1 x T x T

    model.eval()
    with torch.no_grad():
        for r in range(0, H - tile_size + 1, stride):
            for c in range(0, W - tile_size + 1, stride):
                tile  = torch.tensor(
                    stack_11ch[:, r:r+tile_size, c:c+tile_size]
                ).unsqueeze(0).to(device)

                recon_t, uncert_t, _ = model(tile)
                r_np = recon_t[0].cpu().numpy()   # 3 x T x T
                u_np = uncert_t[0].cpu().numpy()

                out_recon[:,  r:r+tile_size, c:c+tile_size] += r_np * g2d
                out_uncert[:, r:r+tile_size, c:c+tile_size] += u_np * g2d
                weights[:,    r:r+tile_size, c:c+tile_size] += g2d

    eps  = 1e-8
    recon  = np.clip(out_recon  / (weights + eps), 0, 1)
    uncert = np.clip(out_uncert / (weights + eps), 0, 1)

    # Zero out unprocessed edge pixels
    zero = weights[0] < eps
    recon[:, zero]  = 0.0
    uncert[:, zero] = 0.0

    return recon, uncert


if __name__ == '__main__':
    # Run full-scene reconstruction for all 3 AOIs
    import sys, os
    sys.path.insert(0, '.')
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    from models.cloud_removal_net import CloudRemovalNet

    os.makedirs('ps2_cloud/outputs/full_scenes', exist_ok=True)

    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    model  = CloudRemovalNet(in_ch=11, out_ch=3, f=32).to(device)
    model.load_state_dict(__import__('torch').load(
        'models/checkpoints/final_best.pth', map_location=device))
    model.eval()
    print(f"Model loaded on {device}")

    for aoi in ['bengaluru', 'punjab', 'meghalaya']:
        print(f"\nReconstructing {aoi} full scene...")
        stack = np.load(f'ps2_cloud/data/processed/{aoi}_stack.npy')
        recon, uncert = reconstruct_scene(model, stack, tile_size=128,
                                          overlap=32, device=device)
        out = f'ps2_cloud/outputs/full_scenes/{aoi}_reconstruction.npy'
        np.save(out, recon)
        print(f"  Saved: {out}  shape={recon.shape}")
        print(f"  Mean uncertainty: {uncert.mean():.4f}")

    print("\nFull-scene reconstruction complete.")
