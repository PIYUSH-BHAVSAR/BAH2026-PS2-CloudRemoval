"""
train.py — main training entry point for PS2 BAH 2026
Run: python train.py

What this does:
  1. Loads all 3 preprocessed stacks from data/processed/
  2. Tiles them into 128x128 patches, filters by cloud coverage
  3. Builds DataLoaders (85/15 train/val split, all 3 AOIs contribute)
  4. Trains CloudRemovalNet for 40 epochs (15 frozen + 25 full)
  5. Saves best checkpoint to models/checkpoints/final_best.pth
  6. Logs all metrics to WandB

Expected time on RTX 3050 6GB: 8-10 hours
"""
import sys
import os
import yaml
import torch
import numpy as np

sys.path.insert(0, '.')

from models.cloud_removal_net import CloudRemovalNet
from scripts.dataset.cloud_dataset import tile_and_load
from scripts.train.trainer import train


def main():
    # ── Load config ───────────────────────────────────────────────────────────
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")
    if device == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print()

    # ── Load preprocessed stacks ──────────────────────────────────────────────
    print("Loading stacks...")
    stacks  = []
    targets = []

    for aoi in ['bengaluru', 'punjab', 'meghalaya']:
        sp = f'ps2_cloud/data/processed/{aoi}_stack.npy'
        tp = f'ps2_cloud/data/processed/{aoi}_clear.npy'
        if not os.path.exists(sp):
            raise FileNotFoundError(f"Missing: {sp} — run preprocess.py first")
        s = np.load(sp)
        t = np.load(tp)
        stacks.append(s)
        targets.append(t)
        print(f"  {aoi}: stack={s.shape}  clear={t.shape}")

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    print("\nTiling and building DataLoaders...")
    train_loader, val_loader = tile_and_load(
        input_stacks=stacks,
        targets=targets,
        tile_size=config['data']['tile_size'],
        overlap=config['data']['overlap'],
        val_fraction=config['data']['val_fraction'],
        batch_size=config['data']['batch_size'],
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = CloudRemovalNet(
        in_ch=config['model']['in_channels'],
        out_ch=config['model']['out_channels'],
        f=config['model']['base_filters'],
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: CloudRemovalNet  ({params/1e6:.2f}M parameters)")

    # ── Train ─────────────────────────────────────────────────────────────────
    best_ckpt = train(model, train_loader, val_loader, config, device=device)
    print(f"\nBest checkpoint: {best_ckpt}")
    print("Run python evaluate.py to compute all 8 metrics.")


if __name__ == '__main__':
    main()
