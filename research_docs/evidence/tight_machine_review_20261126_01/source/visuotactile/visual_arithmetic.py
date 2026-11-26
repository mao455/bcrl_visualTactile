"""Explicit DINO arithmetic identity, shared by capture, frozen cache and rollout."""
import hashlib
import json
from .models.native_vision import DINO_REVISION,validate_vision_microbatch,validate_vision_tail_profile

def profile(microbatch=2,tail=None):
    size=validate_vision_microbatch(microbatch);tail=validate_vision_tail_profile(tail)
    return dict(version='native_dino_arithmetic_v1',dino_revision=DINO_REVISION,
        vision_microbatch=size,vision_tail_profile=tail,
        tail_policy='duplicate_last_real_lane' if tail is None else 'exact',
        cameras=['fixed_high','fixed_side','wrist_rgbd'],rgb='lossless_uint8_3x252x448',
        normalize_mean=[.485,.456,.406],normalize_std=[.229,.224,.225],
        patches='float16_384x18x32',depth='metric_m_2x18x32',
        matmul_allow_tf32=False,cudnn_enabled=True,cudnn_benchmark=False,
        cudnn_deterministic=False,cudnn_allow_tf32=True,autocast=False,
        frozen=True,training=False,temporal_stride=1)

def validate(value):
    if not isinstance(value,dict) or value!=profile(value.get('vision_microbatch'),value.get('vision_tail_profile')):
        raise ValueError('Exact explicit DINO arithmetic identity required')
    return dict(value)

def digest(value):
    return hashlib.sha256(json.dumps(validate(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()
