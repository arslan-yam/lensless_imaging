import torch
from torch import nn
from src.model.admm import LeADMM


class LenslessReconstructor(nn.Module):
    def __init__(self, n_iters=5, admm_trainable=True, pre_processor=None, post_processor=None, mu1=1e-4, mu2=1e-4, mu3=1e-4, tau=2e-4, solver=None):
        super().__init__()
        self.pre_processor = pre_processor
        if solver is None:
            solver = LeADMM(n_iters=n_iters, trainable=admm_trainable, mu1=mu1, mu2=mu2, mu3=mu3, tau=tau)
        self.admm = solver
        self.post_processor = post_processor

    def forward(self, lensless, psf, **batch):
        x = lensless
        if self.pre_processor is not None:
            x = self.pre_processor(x)
        x = self.admm(x, psf)
        if self.post_processor is not None:
            x = self.post_processor(x)
        x = torch.clamp(x, min=0.0)
        x = x / (x.amax(dim=(-3, -2, -1), keepdim=True) + 1e-8)
        return {"reconstruction": x}

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        return result_info
