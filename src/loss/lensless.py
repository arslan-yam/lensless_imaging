import lpips
from torch import nn
from src.utils.roi import crop_roi


class LenslessLoss(nn.Module):
    def __init__(self, mse_weight=1.0, lpips_weight=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.lpips = lpips.LPIPS(net="vgg")
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def forward(self, reconstruction, lensed, **batch):
        rec = crop_roi(reconstruction)
        target = crop_roi(lensed)
        mse_loss = nn.functional.mse_loss(rec, target)
        lpips_loss = self.lpips(rec, target, normalize=True).mean()
        loss = self.mse_weight * mse_loss + self.lpips_weight * lpips_loss
        return {"loss": loss, "mse_loss": mse_loss, "lpips_loss": lpips_loss}
