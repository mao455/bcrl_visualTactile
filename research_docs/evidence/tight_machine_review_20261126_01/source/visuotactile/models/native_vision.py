"""Pinned DINO cache arithmetic, with a causal fixed-size image batch."""
import torch

from torch import nn
DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
VISION_BATCH_VERSION = 'current_observation_lane_microbatch_v1'
EXACT_TAIL_PROFILE = dict(version='current_observation_exact_tail_v1')


def validate_vision_microbatch(value):
    if type(value) is not int or value < 1:
        raise ValueError('vision_microbatch must be an explicit positive integer')
    return value


def validate_vision_tail_profile(value):
    """Missing preserves the original fixed-size padded kernel contract."""
    if value is None:
        return None
    if not isinstance(value, dict) or value != EXACT_TAIL_PROFILE:
        raise ValueError('Unknown vision_tail_profile; exact tail requires its explicit version')
    return dict(value)


def vision_lane_budget(lanes, vision_microbatch=2, *, vision_tail_profile=None):
    """Image work for one call; padding is compute, never another observation."""
    size = validate_vision_microbatch(vision_microbatch)
    profile = validate_vision_tail_profile(vision_tail_profile)
    if type(lanes) is not int or lanes < 0:
        raise ValueError('Expected a nonnegative real lane count')
    calls = (lanes + size - 1) // size
    result = dict(version=VISION_BATCH_VERSION, vision_microbatch=size,
        images_per_kernel=3*size, forward_features_calls=calls,
        real_lanes=lanes, padding_lanes=calls*size-lanes,
        real_images=3*lanes, processed_images=3*calls*size)
    if profile is not None:
        full, tail = divmod(lanes, size)
        histogram = {str(3*size): full} if full else {}
        if tail:
            histogram[str(3*tail)] = 1
        result.update(version=profile['version'], vision_tail_profile=profile,
            images_per_kernel=None, max_images_per_kernel=3*size,
            kernel_image_batch_counts=histogram, padding_lanes=0, processed_images=3*lanes)
    return result


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
    """Default to the six-image cache kernel; opt-in lane batches are candidates.

    DINO has no cross-image attention; padding introduces no future input.
    Normalized camera pixels must round-trip to their original uint8 bytes.
    Changing the lane batch changes CUDA kernel shape and needs a real command
    parity gate. It does not change the ordinary single-observation/cache path.
    """
    @torch.no_grad()
    def forward(self, rgb):
        if rgb.shape != (3, 3, 252, 448):
            raise ValueError('Native DINO requires one current three-camera observation')
        return self.forward_lanes(rgb[None])[0]

    @torch.no_grad()
    def forward_lanes(self, rgb, *, vision_microbatch=2, vision_tail_profile=None):
        """Independent current observations in explicitly sized lane groups.

        By default the final lane is duplicated to pad the final group. The
        explicit exact-tail candidate uses the final group's actual size.
        Neither mode pools time or mixes images within an attention sequence.
        """
        size = validate_vision_microbatch(vision_microbatch)
        profile = validate_vision_tail_profile(vision_tail_profile)
        if profile is not None:
            self.last_lane_execution = None
        if rgb.ndim != 5 or rgb.shape[1:] != (3, 3, 252, 448) or len(rgb) < 1:
            raise ValueError('Expected N current three-camera observations')
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
                results = []; kernel_images = []
                for start in range(0, len(rgb), size):
                    group = rgb[start:start+size]
                    count = len(group)
                    if count < size and profile is None:
                        group = torch.cat((group, group[-1:].expand(size-count, -1, -1, -1, -1)))
                    if profile is not None:
                        kernel_images.append(3*len(group))
                    results.append(super().forward(group)[:count])
                result = torch.cat(results)
                if profile is not None:
                    work = vision_lane_budget(len(rgb), size, vision_tail_profile=profile)
                    actual = {}
                    for images in kernel_images:
                        actual[str(images)] = actual.get(str(images), 0)+1
                    if actual != work['kernel_image_batch_counts']:
                        raise RuntimeError('Actual DINO kernel shapes differ from registered tail work')
                    self.last_lane_execution = dict(work, kernel_image_batch_counts=actual)
                return result
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
