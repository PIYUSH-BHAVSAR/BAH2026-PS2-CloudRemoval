import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class CombinedLoss(nn.Module):
    """
    5-term loss for cloud removal quality.

    Term        Weight  Enforces
    ─────────── ──────  ──────────────────────────────────────────
    L1          1.0     Pixel-level accuracy (MAE)
    SSIM        0.5     Structural similarity (texture, contrast)
    Spectral    2.0     NDVI + NDWI index preservation  ← PRIMARY
    Edge        0.3     Sobel edge sharpness (roads, field edges)
    Uncertainty 0.2     Uncertainty calibration to actual error

    Spectral has highest weight — ISRO evaluates NDVI fidelity first.
    If NDVI MAE is not improving during training: raise w_spec to 3.0.
    """
    def __init__(self, w_l1=1.0, w_ssim=0.5, w_spec=2.0,
                 w_edge=0.3, w_unc=0.2):
        super().__init__()
        self.w = dict(l1=w_l1, ssim=w_ssim, spec=w_spec,
                      edge=w_edge, unc=w_unc)

    def forward(self, pred, target, uncertainty):
        eps = 1e-8

        # L1
        l1 = F.l1_loss(pred, target)

        # SSIM (1 - SSIM so it's a minimisable loss)
        ssim_val  = ssim(pred, target, data_range=1.0, nonnegative_ssim=True)
        ssim_loss = 1.0 - ssim_val

        # Spectral: NDVI + NDWI  (ch: 0=Green, 1=Red, 2=NIR)
        ndvi_p = (pred[:,2]   - pred[:,1])   / (pred[:,2]   + pred[:,1]   + eps)
        ndvi_t = (target[:,2] - target[:,1]) / (target[:,2] + target[:,1] + eps)
        ndwi_p = (pred[:,0]   - pred[:,2])   / (pred[:,0]   + pred[:,2]   + eps)
        ndwi_t = (target[:,0] - target[:,2]) / (target[:,0] + target[:,2] + eps)
        spec   = F.l1_loss(ndvi_p, ndvi_t) + F.l1_loss(ndwi_p, ndwi_t)

        # Edge (Sobel gradient magnitude)
        def sobel_mag(x):
            kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                               dtype=torch.float32, device=x.device).view(1,1,3,3)
            ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                               dtype=torch.float32, device=x.device).view(1,1,3,3)
            out = []
            for c in range(x.shape[1]):
                ch  = x[:, c:c+1]
                mag = torch.sqrt(F.conv2d(ch, kx, padding=1)**2 +
                                  F.conv2d(ch, ky, padding=1)**2 + eps)
                out.append(mag)
            return torch.cat(out, 1)

        edge = F.l1_loss(sobel_mag(pred), sobel_mag(target))

        # Uncertainty calibration
        actual_err = pred.detach().sub(target).abs()
        unc_loss   = F.l1_loss(uncertainty, actual_err)

        # Weighted sum
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
