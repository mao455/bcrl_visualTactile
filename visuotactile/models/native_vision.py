"""Pinned DINO cache arithmetic, with a causal fixed-size image batch."""
import torch

from torch import nn
DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


class FrozenDinoSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.backbone = torch.hub.load('facebookresearch/dinov2:'+DINO_REVISION,
                'dinov2_vits14', pretrained=True, trust_repo=True).eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).view(1,3,1,1))

    @torch.no_grad()
    def forward(self, rgb):
        if rgb.shape[-3:] != (3,252,448):
            raise ValueError('Expected unresized 448x252 RGB')
        images = rgb.float()/(255. if rgb.dtype == torch.uint8 else 1.)
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError('DINO RGB must be finite [0,1]')
        leading = images.shape[:-3]
        images = (images.reshape(-1,3,252,448)-self.mean)/self.std
        features = self.backbone.forward_features(images)['x_norm_patchtokens']
        return features.transpose(1,2).reshape(*leading,384,18,32).half()


class NativeDinoSpatial(FrozenDinoSpatial):
    """Match the six-image cache kernel by duplicating only the current views.

    DINO has no cross-image attention; padding introduces no future input.
    Normalized camera pixels must round-trip to their original uint8 bytes.
    """
    @torch.no_grad()
    def forward(self, rgb):
        if rgb.shape != (3, 3, 252, 448):
            raise ValueError('Native DINO requires one current three-camera observation')
        if rgb.is_floating_point():
            if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                raise ValueError('Native camera RGB must be finite in [0,1]')
            pixels = (rgb*255).round().to(torch.uint8)
            if not torch.allclose(rgb, pixels.float()/255., atol=1e-7, rtol=0):
                raise ValueError('Native camera RGB is not reversible uint8/255')
            rgb = pixels
        matmul = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=True):
                return super().forward(torch.stack((rgb, rgb)))[0]
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
