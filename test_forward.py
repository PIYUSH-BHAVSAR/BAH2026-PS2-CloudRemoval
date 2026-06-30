import sys
sys.path.insert(0, '.')

import torch
from models.cloud_removal_net import CloudRemovalNet
from models.losses import CombinedLoss

print("Testing forward pass on RTX 3050 6GB...")
print("="*50)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

model = CloudRemovalNet(in_ch=11, out_ch=3, f=32).to(device)

# Count parameters
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {params/1e6:.2f}M")
print()

# Dummy batch
x = torch.randn(2, 11, 128, 128).to(device)
y = torch.rand( 2,  3, 128, 128).to(device)

# Forward pass with mixed precision
with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
    recon, uncert, attn = model(x)

    print(f"recon  shape: {recon.shape}   ← expect (2, 3, 128, 128)")
    print(f"uncert shape: {uncert.shape}  ← expect (2, 3, 128, 128)")
    print(f"attn   shape: {attn.shape}    ← expect (2, 4, 64, 64)")
    print()

    # Value ranges
    print(f"recon  range: [{recon.min().item():.4f}, {recon.max().item():.4f}]  ← expect [0,1]")
    print(f"uncert range: [{uncert.min().item():.4f}, {uncert.max().item():.4f}] ← expect [0,1]")
    print()

    # Loss
    criterion = CombinedLoss()
    loss, ld  = criterion(recon, y, uncert)
    print(f"Total loss: {loss.item():.4f}")
    for k, v in ld.items():
        status = '✓' if v > 0 else '✗ ZERO — check!'
        print(f"  {status} {k:10s}: {v:.4f}")
    print()

if device == 'cuda':
    print(f"VRAM used:     {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"VRAM reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"VRAM free:     {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())/1e9:.2f} GB")
    print()

# Backward pass
loss.backward()
print("Backward pass: OK")
print()

# Shape assertions
assert recon.shape  == (2, 3, 128, 128), f"Wrong recon shape: {recon.shape}"
assert uncert.shape == (2, 3, 128, 128), f"Wrong uncert shape: {uncert.shape}"
assert attn.shape   == (2, 4, 64, 64),  f"Wrong attn shape: {attn.shape}"
assert all(v > 0 for v in ld.values()),  "Some loss terms are zero"

print("="*50)
print("ALL CHECKS PASSED — ready for training")
