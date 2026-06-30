import torch
import numpy as np


def run_ablation(model, val_loader, evaluator, device):
    """
    Zero out each input source and measure performance drop.
    Proves every design decision (SAR, temporal, DEM) contributes.

    Channel map:
      [4-5]  SAR VV, VH     → "No SAR" variant
      [6-9]  temporal + mask → "No temporal" variant
      [10]   DEM             → "No DEM" variant
    """
    variants = {
        'Full model':  [],
        'No SAR':      [4, 5],
        'No temporal': [6, 7, 8, 9],
        'No DEM':      [10],
    }

    results = {name: [] for name in variants}
    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            for name, zero_chs in variants.items():
                x_mod = x.clone()
                for c in zero_chs:
                    x_mod[:, c] = 0.0
                recon, uncert, _ = model(x_mod)
                results[name].append(evaluator.compute_all(recon, y, uncert))

    print("\n=== ABLATION STUDY ===")
    print(f"{'Variant':<20} {'SSIM':>6} {'NDVI MAE':>9} {'Edge F1':>8} {'PSNR':>7}")
    print("─" * 55)
    for name, mlist in results.items():
        avg = {k: np.mean([m[k] for m in mlist]) for k in mlist[0]}
        print(f"{name:<20} {avg['ssim']:>6.4f} {avg['ndvi_mae']:>9.4f} "
              f"{avg['edge_f1']:>8.4f} {avg['psnr']:>7.2f}")
    return results
