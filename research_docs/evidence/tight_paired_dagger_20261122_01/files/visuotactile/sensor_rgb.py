"""Canonical uint8 sensor normalization for exact compact RGB reconstruction."""
import numpy as np
import torch


RGB_NORMALIZATION = dict(version='sensor_uint8_canonical_float32_v1',
    source='actual_camera_uint8', values='numpy_float32_uint8_div_255',
    cuda_implementation='device_lookup_256', cpu_implementation='unchanged_float32_div_255',
    storage='unchanged_exact_uint8_255_roundtrip')


def canonical_rgb_lookup(device):
    """Copy 1 KiB once; no per-frame host transfer or approximate division."""
    values=np.arange(256,dtype=np.float32)/np.float32(255)
    return torch.from_numpy(values).to(device)


def canonical_rgb_u8(rgb,lookup):
    """Map real sensor bytes to the exact floats reconstructed by storage.

    CUDA scalar division may multiply by a rounded reciprocal. Its one-ULP
    differences are not accepted by the lossless recorder. Indexing the same
    256 representable values avoids that arithmetic difference without changing
    a sensor byte or weakening recorder validation.
    """
    if rgb.dtype!=torch.uint8:
        raise ValueError('Canonical sensor RGB requires actual uint8 pixels')
    if lookup.shape!=(256,) or lookup.dtype!=torch.float32 or lookup.device!=rgb.device:
        raise ValueError('Canonical RGB lookup must be float32[256] on the sensor device')
    return lookup[rgb.to(torch.int64)]
