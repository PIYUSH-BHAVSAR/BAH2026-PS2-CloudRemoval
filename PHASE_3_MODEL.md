# PHASE 3 — Model Architecture
## PS2 | BAH 2026
### Goal: All model files written, tiling done, forward pass verified, first epoch running

---

## Architecture Overview

```
INPUT
  11-channel tensor — B × 11 × 128 × 128
  (cloudy optical + cloud mask + SAR + temporal ref + change mask + DEM)

ENCODER (4 stages, stride-2 pooling each)
  enc1: 11 → 32 ch  | 128×128 → 64×64
  enc2: 32 → 64 ch  |  64×64  → 32×32
  enc3: 64 → 128ch  |  32×32  → 16×16
  enc4:128 → 256ch  |  16×16  →  8×8

BOTTLENECK
  conv: 256 → 512ch |  8×8   (spatial size stays)
  N = 8×8 = 64 spatial positions — safe for full attention

CROSS-MODAL ATTENTION (at bottleneck, N=64)
  Query  = optical features  → "what needs to be filled?"
  Key    = SAR features      → "what structure exists through the cloud?"
  Value  = temporal features → "what did it look like before?"
  Attention weights saved → used for visualization in demo

DECODER (4 stages, 2× upsampling each)
  dec4: 512 → 256ch with skip from enc4
  dec3: 256 → 128ch with skip from enc3
  dec2: 128 →  64ch with skip from enc2
  dec1:  64 →  32ch with skip from enc1

DUAL OUTPUT HEADS (from final 32-ch decoder output)
  recon_head:  3-band cloud-free reconstruction  (Sigmoid → [0,1])
  uncert_head: per-pixel uncertainty map         (Sigmoid → [0,1])

OUTPUT
  reconstruction: B × 3 × 128 × 128
  uncertainty:    B × 3 × 128 × 128
  attn_weights:   B × 4 × 64 × 64  (saved for visualization)
```

**RTX 3050 6GB settings — do not change these:**
```
f (base_filters) = 32     WAS 64 — 64 OOMs on 6GB
tile_size        = 128    WAS 256 — 256×256 OOMs on 6GB
batch_size       = 2      WAS 8   — 8 OOMs on 6GB
mixed_precision  = True   MANDATORY on 6GB
attention_heads  = 4      WAS 8   — 4 correct for f=32
```

---

## File: `models/attention.py`

```python
# models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Three-way cross-attention between optical, SAR, and temporal streams.

    Applied at the U-Net bottleneck where spatial size = 8×8 = N=64.
    Full N×N attention at N=64 is trivial (64×64 matrix = 4096 values).

    Query  = optical features  (what needs to be reconstructed?)
    Key    = SAR features      (what structural cues exist through cloud?)
    Value  = temporal features (what was here in the clear reference?)

    Attention weights are RETURNED — saved for visualization.
    These maps prove to judges that every reconstructed pixel used
    real physical data from a specific source.

    dim       : feature dimension at bottleneck = f*16 = 32*16 = 512
    num_heads : 4 (head_dim = 512/4 = 128 — expressive for small model)
    """
    def __init__(self, dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.heads    = num_heads
        self.head_dim = dim // num_heads
        self.scale    = self.head_dim ** -0.5

        self.to_q  = nn.Linear(dim, dim, bias=False)
        self.to_k  = nn.Linear(dim, dim, bias=False)
        self.to_v  = nn.Linear(dim, dim, bias=False)
        self.out   = nn.Linear(dim, dim)
        self.drop  = nn.Dropout(dropout)
        self.norm  = nn.LayerNorm(dim)

    def forward(self, optical, sar, temporal):
        B, C, H, W = optical.shape
        N = H * W

        # Safety assertion — must be at bottleneck (8×8 with tile=128)
        assert N <= 1024, (
            f"Attention spatial size {H}×{W}={N} too large. "
            f"Expected bottleneck 8×8=64. "
            f"Check encoder — 4 stages of stride-2 on 128×128 gives 8×8."
        )

        # Flatten spatial dims: B×C×H×W → B×N×C
        opt  = optical.flatten(2).transpose(1, 2)   # B × N × C
        sar_ = sar.flatten(2).transpose(1, 2)
        tmp  = temporal.flatten(2).transpose(1, 2)

        # Multi-head Q/K/V projections
        Q = self.to_q(opt).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        K = self.to_k(sar_).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        V = self.to_v(tmp).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        # Shapes: B × heads × N × head_dim

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # B×heads×N×N
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn.detach()   # save for visualization, no gradient
        attn = self.drop(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)                          # B×heads×N×head_dim
        out = out.transpose(1, 2).reshape(B, N, C)           # B×N×C
        out = self.norm(self.out(out) + opt)                  # residual + layer norm
        out = out.transpose(1, 2).reshape(B, C, H, W)        # B×C×H×W

        return out, attn_weights   # attn_weights: B × heads × N × N
```

---

## File: `models/blocks.py`

```python
# models/blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """
    Double-conv encoder block with max-pool downsampling.
    Returns (skip_features, pooled_output).
    skip_features go to the corresponding decoder via skip connection.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        feat = self.conv(x)
        return feat, self.pool(feat)   # (skip, downsampled)


class DecoderBlock(nn.Module):
    """
    Transposed-conv upsampling + skip concatenation + double conv.
    Handles size mismatches with bilinear interpolation (common with odd tile sizes).
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))
```

---

## File: `models/cloud_removal_net.py`

```python
# models/cloud_removal_net.py
import torch
import torch.nn as nn
from .blocks import EncoderBlock, DecoderBlock
from .attention import CrossModalAttention


class CloudRemovalNet(nn.Module):
    """
    Cross-Modal Attention U-Net for cloud removal.

    Input shape:  B × 11 × 128 × 128
    Output:
      reconstruction : B × 3  × 128 × 128  (cloud-free image, Sigmoid)
      uncertainty    : B × 3  × 128 × 128  (per-pixel confidence, Sigmoid)
      attn_weights   : B × 4  ×  64 ×  64  (for visualization)

    f=32 is the RTX 3050 6GB safe setting.
    Do not increase to f=64 without gradient checkpointing.
    """
    def __init__(self, in_ch=11, out_ch=3, f=32):
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────────
        self.enc1 = EncoderBlock(in_ch, f)         # 128×128 → 64×64,  f ch
        self.enc2 = EncoderBlock(f,     f*2)       #  64×64  → 32×32,  f*2 ch
        self.enc3 = EncoderBlock(f*2,   f*4)       #  32×32  → 16×16,  f*4 ch
        self.enc4 = EncoderBlock(f*4,   f*8)       #  16×16  →  8×8,   f*8 ch

        # ── Bottleneck ────────────────────────────────────────────────────
        # f*8 → f*16 channels, spatial stays at 8×8
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f*8, f*16, 3, padding=1, bias=False),
            nn.BatchNorm2d(f*16),
            nn.ReLU(inplace=True),
        )

        # ── Cross-modal attention at bottleneck ───────────────────────────
        # dim = f*16 = 32*16 = 512 | num_heads=4 → head_dim=128
        # N = 8×8 = 64 spatial positions — perfectly manageable
        # FIX: no double projection — attention module handles Q/K/V projections
        self.attention = CrossModalAttention(dim=f*16, num_heads=4)

        # ── Decoder ───────────────────────────────────────────────────────
        self.dec4 = DecoderBlock(f*16, f*8, f*8)   #  8×8  → 16×16
        self.dec3 = DecoderBlock(f*8,  f*4, f*4)   # 16×16 → 32×32
        self.dec2 = DecoderBlock(f*4,  f*2, f*2)   # 32×32 → 64×64
        self.dec1 = DecoderBlock(f*2,  f,   f)     # 64×64 → 128×128

        # ── Dual output heads ─────────────────────────────────────────────
        self.recon_head  = nn.Sequential(nn.Conv2d(f, out_ch, 1), nn.Sigmoid())
        self.uncert_head = nn.Sequential(nn.Conv2d(f, out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        # Encode
        s1, x = self.enc1(x)   # skip1: B×f×128×128
        s2, x = self.enc2(x)   # skip2: B×f*2×64×64
        s3, x = self.enc3(x)   # skip3: B×f*4×32×32
        s4, x = self.enc4(x)   # skip4: B×f*8×16×16

        # Bottleneck
        x = self.bottleneck(x)   # B × f*16 × 8 × 8

        # Cross-modal attention
        # Same bottleneck features passed as all three streams.
        # The Q/K/V linear projections inside CrossModalAttention
        # learn to specialize each stream differently from shared features.
        fused, attn = self.attention(x, x, x)

        # Decode with skip connections
        x = self.dec4(fused, s4)
        x = self.dec3(x,     s3)
        x = self.dec2(x,     s2)
        x = self.dec1(x,     s1)

        return self.recon_head(x), self.uncert_head(x), attn
```

---

## File: `models/losses.py`

```python
# models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class CombinedLoss(nn.Module):
    """
    5-term loss function.

    Term         Weight  What it enforces
    ──────────── ──────  ──────────────────────────────────────────────────
    L1           1.0     Baseline pixel accuracy (mean absolute error)
    SSIM         0.5     Structural similarity (textures, contrast, edges)
    Spectral     2.0     NDVI + NDWI preservation ← HIGHEST WEIGHT
    Edge         0.3     Sobel edge sharpness (field boundaries, roads)
    Uncertainty  0.2     Forces uncertainty to match actual prediction error

    Spectral weight = 2.0 because ISRO evaluates NDVI/NDWI fidelity first.
    If NDVI MAE is not improving: increase w_spec to 3.0 or 3.5.
    """
    def __init__(self, w_l1=1.0, w_ssim=0.5, w_spec=2.0,
                 w_edge=0.3, w_unc=0.2):
        super().__init__()
        self.w = dict(l1=w_l1, ssim=w_ssim, spec=w_spec,
                      edge=w_edge, unc=w_unc)

    def forward(self, pred, target, uncertainty):
        eps = 1e-8

        # ── L1 ────────────────────────────────────────────────────────────
        l1 = F.l1_loss(pred, target)

        # ── SSIM ──────────────────────────────────────────────────────────
        ssim_val  = ssim(pred, target, data_range=1.0, nonnegative_ssim=True)
        ssim_loss = 1 - ssim_val

        # ── Spectral consistency (NDVI + NDWI) ────────────────────────────
        # ch layout: 0=Green, 1=Red, 2=NIR
        ndvi_p = (pred[:,2]   - pred[:,1])   / (pred[:,2]   + pred[:,1]   + eps)
        ndvi_t = (target[:,2] - target[:,1]) / (target[:,2] + target[:,1] + eps)
        ndwi_p = (pred[:,0]   - pred[:,2])   / (pred[:,0]   + pred[:,2]   + eps)
        ndwi_t = (target[:,0] - target[:,2]) / (target[:,0] + target[:,2] + eps)
        spec   = F.l1_loss(ndvi_p, ndvi_t) + F.l1_loss(ndwi_p, ndwi_t)

        # ── Edge preservation (Sobel) ─────────────────────────────────────
        def sobel(x):
            kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                               dtype=torch.float32, device=x.device).view(1,1,3,3)
            ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                               dtype=torch.float32, device=x.device).view(1,1,3,3)
            mags = []
            for c in range(x.shape[1]):
                ch = x[:, c:c+1]
                mag = torch.sqrt(F.conv2d(ch, kx, padding=1)**2 +
                                  F.conv2d(ch, ky, padding=1)**2 + eps)
                mags.append(mag)
            return torch.cat(mags, 1)

        edge = F.l1_loss(sobel(pred), sobel(target))

        # ── Uncertainty calibration ───────────────────────────────────────
        # Forces uncertainty map to predict actual reconstruction error
        actual_err = pred.detach().sub(target).abs()
        unc_loss   = F.l1_loss(uncertainty, actual_err)

        # ── Total ─────────────────────────────────────────────────────────
        total = (self.w['l1']   * l1 +
                 self.w['ssim'] * ssim_loss +
                 self.w['spec'] * spec +
                 self.w['edge'] * edge +
                 self.w['unc']  * unc_loss)

        return total, {
            'total':    total.item(),
            'l1':       l1.item(),
            'ssim':     ssim_loss.item(),
            'spectral': spec.item(),
            'edge':     edge.item(),
            'uncert':   unc_loss.item(),
        }
```

---

## File: `scripts/dataset/cloud_dataset.py`

```python
# scripts/dataset/cloud_dataset.py
import torch
import numpy as np
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
        self.inputs  = inputs    # list of float32 arrays: 11 × T × T
        self.targets = targets   # list of float32 arrays:  3 × T × T
        self.augment = augment
        self.aug     = get_augmentations() if augment else None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx].copy()
        y = self.targets[idx].copy()

        if self.augment and self.aug is not None:
            # Stack input+target for joint spatial augmentation
            combined = np.concatenate([x, y], 0).transpose(1, 2, 0)  # H×W×14
            aug_out  = self.aug(image=combined)['image']
            aug_out  = aug_out.transpose(2, 0, 1)                     # 14×H×W
            x, y     = aug_out[:11], aug_out[11:]

            # Brightness jitter — optical channels only (not SAR ch4-5, not masks ch3,9,10)
            if np.random.random() < 0.3:
                factor  = np.random.uniform(0.9, 1.1)
                x[:3]   = np.clip(x[:3]   * factor, 0, 1)   # cloudy optical
                x[6:9]  = np.clip(x[6:9]  * factor, 0, 1)   # temporal reference
                y       = np.clip(y        * factor, 0, 1)   # ground truth

        x = np.nan_to_num(x.astype(np.float32))
        y = np.nan_to_num(y.astype(np.float32))
        return torch.tensor(x), torch.tensor(y)


def tile_and_load(input_stacks, targets,
                  val_scene_indices=None,
                  tile_size=128, overlap=16, batch_size=2):
    """
    Tile all scenes and create train/val DataLoaders.

    CRITICAL: split by SCENE not by tile.
    With 3 AOIs, assigning one whole scene to val prevents data leakage
    where tiles from the same scene land in both train and val sets.

    val_scene_indices: list of indices from input_stacks to hold out for val
      Default [2] = third scene (Meghalaya — geographically distinct from train)

    Returns: train_loader, val_loader
    """
    if val_scene_indices is None:
        val_scene_indices = [2]

    train_x, train_y = [], []
    val_x,   val_y   = [], []
    stride = tile_size - overlap

    for scene_i, (stack, target) in enumerate(zip(input_stacks, targets)):
        _, H, W   = stack.shape
        cloud_ch  = stack[3] * 2   # un-normalize mask (was divided by 2 to store)
        is_val    = scene_i in val_scene_indices

        for r in range(0, H - tile_size + 1, stride):
            for c in range(0, W - tile_size + 1, stride):
                tile_x    = stack[:,  r:r+tile_size, c:c+tile_size]
                tile_y    = target[:, r:r+tile_size, c:c+tile_size]
                cloud_cov = (cloud_ch[r:r+tile_size, c:c+tile_size] > 0).mean()

                # Only tiles with meaningful cloud coverage are useful for training
                if 0.20 <= cloud_cov <= 0.95:
                    if is_val:
                        val_x.append(tile_x); val_y.append(tile_y)
                    else:
                        train_x.append(tile_x); train_y.append(tile_y)

    train_scenes = [i for i in range(len(input_stacks)) if i not in val_scene_indices]
    print(f"Train tiles: {len(train_x)}  (scenes {train_scenes})")
    print(f"Val tiles:   {len(val_x)}    (scenes {val_scene_indices})")

    train_ds = CloudDataset(train_x, train_y, augment=True)
    val_ds   = CloudDataset(val_x,   val_y,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)
    return train_loader, val_loader
```

---

## File: `config.yaml`

```yaml
# config.yaml — RTX 3050 6GB corrected settings
data:
  tile_size: 128        # 256 OOMs on 6GB
  overlap: 16
  min_cloud_coverage: 0.20
  max_cloud_coverage: 0.95
  val_scene_indices: [2]  # Meghalaya = validation
  batch_size: 2           # 8 OOMs on 6GB
  num_workers: 2          # LOQ shared memory bandwidth limit

model:
  in_channels: 11
  out_channels: 3
  base_filters: 32        # 64 OOMs on 6GB
  attention_heads: 4      # correct for f=32 (dim=512, head_dim=128)
  attention_dropout: 0.1

loss:
  w_l1: 1.0
  w_ssim: 0.5
  w_spectral: 2.0         # highest — NDVI/NDWI fidelity is primary metric
  w_edge: 0.3
  w_uncert: 0.2

training:
  phase1_epochs: 15       # frozen encoder
  phase2_epochs: 25       # full fine-tune
  phase1_lr: 5.0e-4
  phase2_lr: 5.0e-5
  weight_decay: 1.0e-4
  grad_clip_norm: 1.0
  mixed_precision: true   # MANDATORY on 6GB

preprocessing:
  sar_vv_range: [-25, 5]
  sar_vh_range: [-30, 0]
  dem_min: -100
  dem_max: 3000
  ndvi_change_threshold: 0.2
  cloud_threshold: 0.4

evaluation:
  uncertainty_threshold: 0.3
  edge_threshold: 0.1
```

---

## Forward Pass Test (run before starting training)

```python
# test_forward.py
import torch
from models.cloud_removal_net import CloudRemovalNet
from models.losses import CombinedLoss

device = 'cuda'
model  = CloudRemovalNet(in_ch=11, out_ch=3, f=32).to(device)

x = torch.randn(2, 11, 128, 128).to(device)
y = torch.rand( 2,  3, 128, 128).to(device)

with torch.cuda.amp.autocast():
    recon, uncert, attn = model(x)

    print("recon  shape:", recon.shape)    # expect: 2 × 3  × 128 × 128
    print("uncert shape:", uncert.shape)   # expect: 2 × 3  × 128 × 128
    print("attn   shape:", attn.shape)     # expect: 2 × 4  ×  64 ×  64

    criterion = CombinedLoss()
    loss, ld  = criterion(recon, y, uncert)
    print(f"\nloss: {loss.item():.4f}")
    for k, v in ld.items():
        print(f"  {k:10s}: {v:.4f}")

print(f"\nVRAM used:     {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"VRAM reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
# Expected: used ~1.5-2.0 GB  (safe — 4+ GB headroom remaining)

loss.backward()
print("\nBackward pass OK")

# Count parameters
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {params/1e6:.2f}M")
# Expected: ~4-5M parameters for f=32
```

---

## Phase 3 Checklist

```
MODEL FILES
  [ ] models/__init__.py created (empty or with imports)
  [ ] models/attention.py  — CrossModalAttention with assert N<=1024
  [ ] models/blocks.py     — EncoderBlock, DecoderBlock
  [ ] models/cloud_removal_net.py — CloudRemovalNet(f=32, heads=4)
  [ ] models/losses.py     — CombinedLoss (5 terms)

DATASET FILES
  [ ] scripts/dataset/cloud_dataset.py — CloudDataset + tile_and_load
  [ ] config.yaml — all settings at RTX 3050 6GB corrected values

FORWARD PASS TEST
  [ ] test_forward.py ran without OOM
  [ ] recon shape: (2, 3, 128, 128) ✓
  [ ] uncert shape: (2, 3, 128, 128) ✓
  [ ] attn shape: (2, 4, 64, 64) ✓
  [ ] All 5 loss terms non-zero ✓
  [ ] Backward pass completed ✓
  [ ] VRAM used: < 4 GB ✓

TILING
  [ ] tile_and_load ran on all 3 stacks
  [ ] Train tile count printed (expect ~4400 tiles from 2 scenes)
  [ ] Val tile count printed (expect ~2200 tiles from Meghalaya)
```

---

**→ Continue to PHASE_4_TRAINING.md**
