import torch
import numpy as np
import random
import albumentations as A
from torch.utils.data import Dataset, DataLoader


def get_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


class CloudDataset(Dataset):
    def __init__(self, inputs, targets, augment=True):
        self.inputs  = inputs   # list of float32 11 x T x T
        self.targets = targets  # list of float32  3 x T x T
        self.augment = augment
        self.aug     = get_augmentations() if augment else None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx].copy()
        y = self.targets[idx].copy()

        if self.augment and self.aug:
            combined = np.concatenate([x, y], 0).transpose(1, 2, 0)  # H x W x 14
            aug_out  = self.aug(image=combined)['image']
            aug_out  = aug_out.transpose(2, 0, 1)                     # 14 x H x W
            x, y     = aug_out[:11], aug_out[11:]

            # Brightness jitter — optical channels only, not SAR/masks/DEM
            if np.random.random() < 0.3:
                factor  = np.random.uniform(0.9, 1.1)
                x[:3]   = np.clip(x[:3]  * factor, 0, 1)   # cloudy optical
                x[6:9]  = np.clip(x[6:9] * factor, 0, 1)   # clear reference
                y        = np.clip(y      * factor, 0, 1)   # ground truth

        return (torch.tensor(np.nan_to_num(x.astype(np.float32))),
                torch.tensor(np.nan_to_num(y.astype(np.float32))))


def tile_and_load(input_stacks, targets,
                  tile_size=128, overlap=16,
                  val_fraction=0.15,
                  batch_size=2):
    """
    Tile all scenes into 128x128 patches and create DataLoaders.

    Uses ALL 3 AOIs for both train and val (random tile split).
    This is better than scene-level split when you only have 3 scenes.

    Tiles are shuffled before splitting so all 3 terrain types appear
    in both train and val sets.

    Only tiles with 20-95% cloud coverage are kept — tiles with no clouds
    are useless for training cloud removal.
    """
    all_x, all_y = [], []
    stride = tile_size - overlap

    for scene_i, (stack, target) in enumerate(zip(input_stacks, targets)):
        _, H, W  = stack.shape
        cloud_ch = stack[3] * 2   # un-normalize: was stored as /2

        tile_count = 0
        for r in range(0, H - tile_size + 1, stride):
            for c in range(0, W - tile_size + 1, stride):
                tile_x    = stack[:,  r:r+tile_size, c:c+tile_size]
                tile_y    = target[:, r:r+tile_size, c:c+tile_size]
                cloud_cov = (cloud_ch[r:r+tile_size, c:c+tile_size] > 0).mean()
                if 0.20 <= cloud_cov <= 0.95:
                    all_x.append(tile_x)
                    all_y.append(tile_y)
                    tile_count += 1

        print(f"  Scene {scene_i}: {tile_count} usable tiles")

    # Shuffle together to mix all 3 AOIs
    combined = list(zip(all_x, all_y))
    random.shuffle(combined)
    all_x, all_y = zip(*combined)
    all_x, all_y = list(all_x), list(all_y)

    split = int(len(all_x) * (1 - val_fraction))
    train_x, train_y = all_x[:split], all_y[:split]
    val_x,   val_y   = all_x[split:], all_y[split:]

    print(f"\nTotal tiles: {len(all_x)}")
    print(f"Train: {len(train_x)}  Val: {len(val_x)}")

    train_ds = CloudDataset(train_x, train_y, augment=True)
    val_ds   = CloudDataset(val_x,   val_y,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=True)

    return train_loader, val_loader
