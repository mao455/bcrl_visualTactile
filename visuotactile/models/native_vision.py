"""Pinned DINO cache arithmetic, with a causal fixed-size image batch."""
import torch

from .long_horizon import FrozenDinoSpatial


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
