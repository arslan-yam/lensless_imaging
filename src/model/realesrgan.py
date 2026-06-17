import torch
from torch import nn
from torch.nn import functional as F
from src.model.rrdbnet import RRDBNet
WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


class RealESRGAN(nn.Module):
    def __init__(self, pretrained=True, frozen=True):
        super().__init__()
        self.net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        if pretrained:
            state = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu")
            self.net.load_state_dict(state.get("params_ema", state))
        if frozen:
            for p in self.net.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = x / (x.amax(dim=(-3, -2, -1), keepdim=True) + 1e-8)
        x = F.interpolate(x, size=(h // 4, w // 4), mode="bilinear", align_corners=False)
        out = self.net(x)
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
