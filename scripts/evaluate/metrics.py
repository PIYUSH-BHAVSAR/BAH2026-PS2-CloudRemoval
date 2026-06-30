import torch
import numpy as np
import torch.nn.functional as F
from pytorch_msssim import ssim
import lpips


class Evaluator:
    """Computes all 8+ evaluation metrics on model output batches."""
    def __init__(self, device='cuda'):
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.device   = device

    @torch.no_grad()
    def compute_all(self, pred, target, uncertainty=None):
        eps = 1e-8
        r   = {}

        r['ssim']  = ssim(pred, target, data_range=1.0, nonnegative_ssim=True).item()

        mse        = F.mse_loss(pred, target)
        r['psnr']  = (20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse + eps))).item()

        r['lpips'] = self.lpips_fn(pred*2-1, target*2-1).mean().item()

        for i, band in enumerate(['green', 'red', 'nir']):
            r[f'rmse_{band}'] = torch.sqrt(F.mse_loss(pred[:,i], target[:,i])).item()

        ndvi_p       = (pred[:,2]-pred[:,1])   / (pred[:,2]+pred[:,1]+eps)
        ndvi_t       = (target[:,2]-target[:,1]) / (target[:,2]+target[:,1]+eps)
        r['ndvi_mae']= F.l1_loss(ndvi_p, ndvi_t).item()

        ndwi_p       = (pred[:,0]-pred[:,2])   / (pred[:,0]+pred[:,2]+eps)
        ndwi_t       = (target[:,0]-target[:,2]) / (target[:,0]+target[:,2]+eps)
        r['ndwi_mae']= F.l1_loss(ndwi_p, ndwi_t).item()

        def sobel(x):
            kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                               dtype=torch.float32, device=x.device).view(1,1,3,3)
            ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                               dtype=torch.float32, device=x.device).view(1,1,3,3)
            return torch.cat([torch.sqrt(F.conv2d(x[:,c:c+1],kx,padding=1)**2 +
                                          F.conv2d(x[:,c:c+1],ky,padding=1)**2+eps)
                               for c in range(x.shape[1])], 1)

        ep  = (sobel(pred)   > 0.1).float()
        et  = (sobel(target) > 0.1).float()
        tp  = (ep*et).sum(); fp = (ep*(1-et)).sum(); fn = ((1-ep)*et).sum()
        pr  = tp/(tp+fp+eps); re = tp/(tp+fn+eps)
        r['edge_f1'] = (2*pr*re/(pr+re+eps)).item()

        if uncertainty is not None:
            r['coverage'] = (uncertainty < 0.3).float().mean().item()

        return r
