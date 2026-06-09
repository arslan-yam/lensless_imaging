import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


def res_blocks(channels, n_blocks):
    return nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])


class DRUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=(32, 64, 128, 256), n_blocks=4):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.head = nn.Conv2d(in_channels, c0, 3, padding=1, bias=False)
        self.enc0 = res_blocks(c0, n_blocks)
        self.down0 = nn.Conv2d(c0, c1, 2, stride=2, bias=False)
        self.enc1 = res_blocks(c1, n_blocks)
        self.down1 = nn.Conv2d(c1, c2, 2, stride=2, bias=False)
        self.enc2 = res_blocks(c2, n_blocks)
        self.down2 = nn.Conv2d(c2, c3, 2, stride=2, bias=False)
        self.body = res_blocks(c3, n_blocks)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2, bias=False)
        self.dec2 = res_blocks(c2, n_blocks)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2, bias=False)
        self.dec1 = res_blocks(c1, n_blocks)
        self.up0 = nn.ConvTranspose2d(c1, c0, 2, stride=2, bias=False)
        self.dec0 = res_blocks(c0, n_blocks)
        self.tail = nn.Conv2d(c0, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        x0 = self.head(x)
        d0 = self.enc0(x0)
        d1 = self.enc1(self.down0(d0))
        d2 = self.enc2(self.down1(d1))
        b = self.body(self.down2(d2))
        u2 = self.dec2(self.up2(b) + d2)
        u1 = self.dec1(self.up1(u2) + d1)
        u0 = self.dec0(self.up0(u1) + d0)
        out = self.tail(u0)
        return out[..., :h, :w]