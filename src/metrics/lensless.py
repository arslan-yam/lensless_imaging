import lpips
import torch
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from src.metrics.base_metric import BaseMetric
from src.utils.roi import crop_roi


class PSNRMetric(BaseMetric):
    def __call__(self, reconstruction, lensed, **batch):
        with torch.no_grad():
            value = peak_signal_noise_ratio(crop_roi(reconstruction), crop_roi(lensed), data_range=1.0, dim=(1, 2, 3), reduction="elementwise_mean")
        return value.item()


class SSIMMetric(BaseMetric):
    def __call__(self, reconstruction, lensed, **batch):
        with torch.no_grad():
            value = structural_similarity_index_measure(crop_roi(reconstruction), crop_roi(lensed), data_range=1.0)
        return value.item()


class MSEMetric(BaseMetric):
    def __call__(self, reconstruction, lensed, **batch):
        with torch.no_grad():
            value = torch.nn.functional.mse_loss(crop_roi(reconstruction), crop_roi(lensed))
        return value.item()


class LPIPSMetric(BaseMetric):
    def __init__(self, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lpips = lpips.LPIPS(net="vgg").to(device)

    def __call__(self, reconstruction, lensed, **batch):
        with torch.no_grad():
            value = self.lpips(crop_roi(reconstruction), crop_roi(lensed), normalize=True)
        return value.mean().item()
