import torch
import torch.nn as nn
from .blocks import EncoderBlock, DecoderBlock
from .attention import CrossModalAttention


class CloudRemovalNet(nn.Module):
    """
    Cross-Modal Attention U-Net for LISS-IV cloud removal.

    Input:   B x 11 x 128 x 128
    Output:
      reconstruction : B x 3  x 128 x 128  (cloud-free, Sigmoid [0,1])
      uncertainty    : B x 3  x 128 x 128  (per-pixel confidence, Sigmoid)
      attn_weights   : B x 4  x  64 x  64  (for visualization)

    f=32 is the RTX 3050 6GB safe setting (f=64 OOMs).
    """
    def __init__(self, in_ch=11, out_ch=3, f=32):
        super().__init__()

        # Encoder: 128 → 64 → 32 → 16 → 8
        self.enc1 = EncoderBlock(in_ch, f)       # 128×128 → 64×64
        self.enc2 = EncoderBlock(f,     f*2)     #  64×64  → 32×32
        self.enc3 = EncoderBlock(f*2,   f*4)     #  32×32  → 16×16
        self.enc4 = EncoderBlock(f*4,   f*8)     #  16×16  →  8×8

        # Bottleneck: 8×8 spatial, f*8 → f*16 channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f*8, f*16, 3, padding=1, bias=False),
            nn.BatchNorm2d(f*16), nn.ReLU(inplace=True),
        )

        # Cross-modal attention at bottleneck (N = 8×8 = 64 — trivially small)
        # dim = f*16 = 512, heads = 4, head_dim = 128
        # No double projection — attention's own Q/K/V layers handle it
        self.attention = CrossModalAttention(dim=f*16, num_heads=4)

        # Decoder: 8 → 16 → 32 → 64 → 128
        self.dec4 = DecoderBlock(f*16, f*8, f*8)
        self.dec3 = DecoderBlock(f*8,  f*4, f*4)
        self.dec2 = DecoderBlock(f*4,  f*2, f*2)
        self.dec1 = DecoderBlock(f*2,  f,   f)

        # Dual output heads
        self.recon_head  = nn.Sequential(nn.Conv2d(f, out_ch, 1), nn.Sigmoid())
        self.uncert_head = nn.Sequential(nn.Conv2d(f, out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        s1, x = self.enc1(x)   # skip1: B x f    x 128 x 128 (before pool: 64x64)
        s2, x = self.enc2(x)   # skip2: B x f*2  x  64 x  64
        s3, x = self.enc3(x)   # skip3: B x f*4  x  32 x  32
        s4, x = self.enc4(x)   # skip4: B x f*8  x  16 x  16

        x = self.bottleneck(x)  # B x f*16 x 8 x 8

        # Pass same bottleneck features as all three streams.
        # Q/K/V projections inside CrossModalAttention specialize them.
        fused, attn = self.attention(x, x, x)

        x = self.dec4(fused, s4)
        x = self.dec3(x,     s3)
        x = self.dec2(x,     s2)
        x = self.dec1(x,     s1)

        return self.recon_head(x), self.uncert_head(x), attn
