# make_plots.py
# Generate verification plots from already-processed stacks.
# Run after: pip install matplotlib

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('ps2_cloud/outputs/figures', exist_ok=True)

AOIs = ['bengaluru', 'punjab', 'meghalaya']

for aoi in AOIs:
    stack_path = f'ps2_cloud/data/processed/{aoi}_stack.npy'
    if not os.path.exists(stack_path):
        print(f"Missing: {stack_path} — run preprocess.py first")
        continue

    print(f"Loading {aoi} stack...", end=' ', flush=True)
    stack = np.load(stack_path)   # 11 x H x W
    print(f"shape={stack.shape}")

    s = 10  # downsample factor for plotting (every 10th pixel)

    def to_falsecolor(g, r, nir):
        # False colour: NIR → Red channel, Red → Green, Green → Blue
        # Stretch for visibility (values are low ~0.01-0.09)
        rgb = np.stack([nir, r, g], axis=2)
        p2  = np.percentile(rgb, 2)
        p98 = np.percentile(rgb, 98)
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
        return rgb

    cloudy_fc  = to_falsecolor(stack[0,::s,::s], stack[1,::s,::s], stack[2,::s,::s])
    clear_fc   = to_falsecolor(stack[6,::s,::s], stack[7,::s,::s], stack[8,::s,::s])
    cloud_mask = stack[3, ::s, ::s]
    sar_vv     = stack[4, ::s, ::s]
    dem        = stack[10,::s, ::s]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(
        f'{aoi.upper()} — Preprocessing Verification\n'
        f'cloud={cloud_mask.mean()*100:.1f}%  '
        f'change={stack[9].mean()*100:.1f}%  '
        f'DEM=[{stack[10].min():.2f},{stack[10].max():.2f}]',
        fontsize=12
    )

    axes[0].imshow(cloudy_fc)
    axes[0].set_title('Cloudy LISS-IV\n(False colour: NIR-R-G)')
    axes[0].axis('off')

    axes[1].imshow(clear_fc)
    axes[1].set_title('Clear LISS-IV\n(Temporal reference)')
    axes[1].axis('off')

    im2 = axes[2].imshow(cloud_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[2].set_title(f'Cloud+Shadow Mask\n(cloud={cloud_mask.mean()*100:.1f}%)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    axes[3].imshow(sar_vv, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title(f'SAR VV\n(Co-registered, norm dB)')
    axes[3].axis('off')

    im4 = axes[4].imshow(dem, cmap='terrain', vmin=0, vmax=1)
    axes[4].set_title(f'DEM\n(Normalised elevation)')
    axes[4].axis('off')
    plt.colorbar(im4, ax=axes[4], fraction=0.046)

    out_path = f'ps2_cloud/outputs/figures/{aoi}_verify.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

print("\nAll plots done. Open ps2_cloud/outputs/figures/ to view.")
