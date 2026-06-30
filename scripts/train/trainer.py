import os
import torch
import numpy as np
from torch.amp import autocast, GradScaler
import wandb


def train(model, train_loader, val_loader, config, device='cuda'):
    """
    Two-phase training with mixed precision, cosine LR, WandB logging.
    Phase 1: frozen encoder (15 epochs) — fast convergence of new modules
    Phase 2: full fine-tune (25 epochs) — refine everything at lower LR
    """
    from models.losses import CombinedLoss
    criterion = CombinedLoss(
        w_l1=config['loss']['w_l1'],
        w_ssim=config['loss']['w_ssim'],
        w_spec=config['loss']['w_spectral'],
        w_edge=config['loss']['w_edge'],
        w_unc=config['loss']['w_uncert'],
    )
    scaler   = GradScaler()
    os.makedirs('models/checkpoints', exist_ok=True)

    wandb.init(project='ps2-cloud-removal', config=config, mode='disabled')
    best_val  = float('inf')
    best_ckpt = 'models/checkpoints/final_best.pth'

    p1 = config['training']['phase1_epochs']
    p2 = config['training']['phase2_epochs']

    # ── PHASE 1: frozen encoder ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1: {p1} epochs — encoder frozen")
    print(f"{'='*60}")

    encoder_params = (list(model.enc1.parameters()) +
                      list(model.enc2.parameters()) +
                      list(model.enc3.parameters()) +
                      list(model.enc4.parameters()))
    for p in encoder_params:
        p.requires_grad = False

    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['phase1_lr'],
        weight_decay=config['training']['weight_decay'],
    )
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=p1, eta_min=1e-5)

    for epoch in range(1, p1 + 1):
        model.train()
        losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt1.zero_grad(set_to_none=True)
            with autocast('cuda'):
                recon, uncert, _ = model(x)
                loss, _          = criterion(recon, y, uncert)
            scaler.scale(loss).backward()
            scaler.unscale_(opt1)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config['training']['grad_clip_norm'])
            scaler.step(opt1); scaler.update()
            losses.append(loss.item())

        sch1.step()
        val_loss, m = validate(model, val_loader, criterion, device)
        t_avg = np.mean(losses)

        wandb.log({'epoch': epoch, 'phase': 1,
                   'train_loss': t_avg, 'val_loss': val_loss,
                   'val_ssim': m['ssim'], 'val_ndvi_mae': m['ndvi_mae']})

        print(f"[Ph1 {epoch:02d}/{p1}]  "
              f"train={t_avg:.4f}  val={val_loss:.4f}  "
              f"SSIM={m['ssim']:.4f}  NDVI={m['ndvi_mae']:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'models/checkpoints/phase1_best.pth')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'models/checkpoints/ep{epoch:03d}.pth')

    # ── PHASE 2: full fine-tune ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2: {p2} epochs — all weights trainable")
    print(f"{'='*60}")

    for p in model.parameters():
        p.requires_grad = True

    opt2 = torch.optim.AdamW(model.parameters(),
                              lr=config['training']['phase2_lr'],
                              weight_decay=config['training']['weight_decay'])
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=p2, eta_min=1e-6)

    for epoch in range(p1 + 1, p1 + p2 + 1):
        model.train()
        losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt2.zero_grad(set_to_none=True)
            with autocast('cuda'):
                recon, uncert, _ = model(x)
                loss, _          = criterion(recon, y, uncert)
            scaler.scale(loss).backward()
            scaler.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config['training']['grad_clip_norm'])
            scaler.step(opt2); scaler.update()
            losses.append(loss.item())

        sch2.step()
        val_loss, m = validate(model, val_loader, criterion, device)
        t_avg = np.mean(losses)

        wandb.log({'epoch': epoch, 'phase': 2,
                   'train_loss': t_avg, 'val_loss': val_loss,
                   'val_ssim': m['ssim'], 'val_ndvi_mae': m['ndvi_mae']})

        print(f"[Ph2 {epoch:02d}/{p1+p2}]  "
              f"train={t_avg:.4f}  val={val_loss:.4f}  "
              f"SSIM={m['ssim']:.4f}  NDVI={m['ndvi_mae']:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt)
            print(f"  → Best saved: {best_ckpt}")
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'models/checkpoints/ep{epoch:03d}.pth')

    print(f"\nDone. Best val loss: {best_val:.4f}")
    wandb.finish()
    return best_ckpt


@torch.no_grad()
def validate(model, loader, criterion, device):
    from pytorch_msssim import ssim as ssim_fn
    model.eval()
    total, ss, nd, n = 0, 0, 0, 0
    eps = 1e-8
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        recon, uncert, _ = model(x)
        loss, _ = criterion(recon, y, uncert)
        total  += loss.item()
        ss     += ssim_fn(recon, y, data_range=1.0, nonnegative_ssim=True).item()
        ndvi_p  = (recon[:,2]-recon[:,1])/(recon[:,2]+recon[:,1]+eps)
        ndvi_t  = (y[:,2]-y[:,1])/(y[:,2]+y[:,1]+eps)
        nd     += torch.nn.functional.l1_loss(ndvi_p, ndvi_t).item()
        n      += 1
    return total/n, {'ssim': ss/n, 'ndvi_mae': nd/n}
