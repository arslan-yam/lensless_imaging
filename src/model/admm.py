import math
import torch
from torch import nn


def soft_threshold(x, thresh):
    return torch.sign(x) * torch.clamp(torch.abs(x) - thresh, min=0.0)


def forward_diff(x):
    dy = torch.roll(x, -1, dims=-2) - x
    dx = torch.roll(x, -1, dims=-1) - x
    return torch.stack([dy, dx], dim=2)


def adjoint_diff(g):
    gy, gx = g[:, :, 0], g[:, :, 1]
    aty = torch.roll(gy, 1, dims=-2) - gy
    atx = torch.roll(gx, 1, dims=-1) - gx
    return aty + atx


class LeADMM(nn.Module):
    def __init__(self, n_iters=5, trainable=True, mu1=1e-4, mu2=1e-4, mu3=1e-4, tau=2e-4):
        super().__init__()
        self.n_iters = n_iters
        init = torch.tensor([[math.log(mu1), math.log(mu2), math.log(mu3), math.log(tau)] for _ in range(n_iters)])
        if trainable:
            self.log_params = nn.Parameter(init)
        else:
            self.register_buffer("log_params", init)

    def _operators(self, b, psf):
        H, W = psf.shape[-2:]
        Hp, Wp = 2 * H, 2 * W
        top, left = (Hp - H) // 2, (Wp - W) // 2
        psf_pad = b.new_zeros(*psf.shape[:-2], Hp, Wp)
        psf_pad[..., top : top + H, left : left + W] = psf
        Hf = torch.fft.rfft2(torch.fft.ifftshift(psf_pad, dim=(-2, -1)))
        HtH = torch.abs(Hf) ** 2
        m = torch.arange(Hp, device=b.device, dtype=b.dtype)
        n = torch.arange(Wp // 2 + 1, device=b.device, dtype=b.dtype)
        PsiTPsi = (2 - 2 * torch.cos(2 * math.pi * m / Hp)).view(Hp, 1) + (2 - 2 * torch.cos(2 * math.pi * n / Wp)).view(1, Wp // 2 + 1)
        CtC = b.new_zeros(1, 1, Hp, Wp)
        CtC[..., top : top + H, left : left + W] = 1.0
        Ctb = b.new_zeros(*b.shape[:-2], Hp, Wp)
        Ctb[..., top : top + H, left : left + W] = b
        return Hf, HtH, PsiTPsi, CtC, Ctb, (top, left, H, W)

    def _conv(self, x, Hf):
        return torch.fft.irfft2(torch.fft.rfft2(x) * Hf, s=x.shape[-2:])

    def _conv_adj(self, y, Hf):
        return torch.fft.irfft2(torch.fft.rfft2(y) * torch.conj(Hf), s=y.shape[-2:])

    def forward(self, b, psf):
        Hf, HtH, PsiTPsi, CtC, Ctb, roi = self._operators(b, psf)
        top, left, H, W = roi
        x = torch.zeros_like(Ctb)
        v = torch.zeros_like(Ctb)
        w = torch.zeros_like(Ctb)
        a1 = torch.zeros_like(Ctb)
        a3 = torch.zeros_like(Ctb)
        u = torch.zeros_like(forward_diff(x))
        a2 = torch.zeros_like(u)

        for k in range(self.n_iters):
            mu1, mu2, mu3, tau = torch.exp(self.log_params[k])
            denom = mu1 * HtH + mu2 * PsiTPsi + mu3
            Hx = self._conv(x, Hf)
            Psix = forward_diff(x)
            u = soft_threshold(Psix + a2 / mu2, tau)
            v = (a1 + mu1 * Hx + Ctb) / (CtC + mu1)
            w = torch.clamp(a3 / mu3 + x, min=0.0)
            r = ((mu3 * w - a3) + adjoint_diff(mu2 * u - a2) + self._conv_adj(mu1 * v - a1, Hf))
            x = torch.fft.irfft2(torch.fft.rfft2(r) / denom, s=r.shape[-2:])
            Hx = self._conv(x, Hf)
            Psix = forward_diff(x)
            a1 = a1 + mu1 * (Hx - v)
            a2 = a2 + mu2 * (Psix - u)
            a3 = a3 + mu3 * (x - w)

        return x[..., top : top + H, left : left + W]
