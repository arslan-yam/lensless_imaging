import math
import torch
from torch import nn


def soft_threshold(x, thresh):
    return torch.sign(x) * torch.clamp(torch.abs(x) - thresh, min=0.0)


class FISTA(nn.Module):
    def __init__(self, n_iters=100, trainable=False, tau=2e-4):
        super().__init__()
        self.n_iters = n_iters
        init = torch.full((n_iters,), math.log(tau))
        if trainable:
            self.log_tau = nn.Parameter(init)
        else:
            self.register_buffer("log_tau", init)

    def operators(self, b, psf):
        h, w = psf.shape[-2:]
        hp, wp = 2 * h, 2 * w
        top, left = (hp - h) // 2, (wp - w) // 2
        psf_pad = b.new_zeros(*psf.shape[:-2], hp, wp)
        psf_pad[..., top:top + h, left:left + w] = psf
        hf = torch.fft.rfft2(torch.fft.ifftshift(psf_pad, dim=(-2, -1)))
        lips = 2*(torch.abs(hf)**2).amax()
        return hf, lips, (top, left, h, w)

    def conv(self, x, Hf):
        return torch.fft.irfft2(torch.fft.rfft2(x) * Hf, s=x.shape[-2:])

    def conv_adj(self, y, Hf):
        return torch.fft.irfft2(torch.fft.rfft2(y) * torch.conj(Hf), s=y.shape[-2:])

    def forward(self, b, psf):
        hf, l, roi = self.operators(b, psf)
        top, left, h, w = roi
        x = b.new_zeros(*b.shape[:-2], 2 * h, 2 * w)
        y = torch.zeros_like(x)
        t = 1.0
        for k in range(self.n_iters):
            tau = torch.exp(self.log_tau[k])
            resid = self.conv(y, hf)[..., top:top + h, left:left + w] - b
            grad = torch.zeros_like(x)
            grad[..., top:top+h, left:left+w] = resid
            grad = 2 * self.conv_adj(grad, hf)
            x_next = torch.clamp(soft_threshold(y - grad / l, tau / l), min=0.0)
            t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
            y = x_next + ((t - 1) / t_next) * (x_next - x)
            x, t = x_next, t_next
        return x[..., top:top + h, left:left + w]
