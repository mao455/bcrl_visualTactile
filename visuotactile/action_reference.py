"""Reversible research adapter between native commands and joint references.

The native environment remains the sole owner of delay, smoothing, clipping and
physics.  This module only changes the supervised representation at the
research policy boundary.
"""
from __future__ import annotations

import torch


def _validate(command, target_before, limits):
    if command.shape[-1] != 29 or target_before.shape != command.shape:
        raise ValueError("Expected command and controller target with final dimension 29")
    if limits.shape != (29, 2):
        raise ValueError("Expected 29 joint limits")
    if not torch.isfinite(command).all() or not torch.isfinite(target_before).all():
        raise ValueError("Commands/references and controller targets must be finite")
    if not torch.isfinite(limits).all() or (limits[:, 1] <= limits[:, 0]).any():
        raise ValueError("Joint limit ranges must be finite and positive")


def command_to_reference(command, target_before, limits, *, speed_scale=1.5, control_dt=1 / 60):
    """Convert ``command_clipped_pre_delay`` to an absolute joint reference."""
    limits = torch.as_tensor(limits, dtype=command.dtype, device=command.device)
    _validate(command, target_before, limits)
    if speed_scale <= 0 or control_dt <= 0 or (command.abs() > 1.000001).any():
        raise ValueError("Expected clipped native command and positive native scale/dt")
    result = torch.empty_like(command)
    result[..., :7] = target_before[..., :7] + speed_scale * control_dt * command[..., :7]
    result[..., 7:] = limits[7:, 0] + .5 * (command[..., 7:] + 1.) * (limits[7:, 1] - limits[7:, 0])
    return result


def reference_to_command(reference, target_before, limits, *, speed_scale=1.5, control_dt=1 / 60):
    """Convert predicted references back to the native pre-delay command."""
    limits = torch.as_tensor(limits, dtype=reference.dtype, device=reference.device)
    _validate(reference, target_before, limits)
    result = torch.empty_like(reference)
    scale = speed_scale * control_dt
    if scale <= 0:
        raise ValueError("speed_scale and control_dt must be positive")
    result[..., :7] = (reference[..., :7] - target_before[..., :7]) / scale
    result[..., 7:] = 2. * (reference[..., 7:] - limits[7:, 0]) / (limits[7:, 1] - limits[7:, 0]) - 1.
    return result.clamp(-1., 1.)


def roundtrip_metrics(command, target_before, limits, *, speed_scale=1.5, control_dt=1 / 60):
    reference = command_to_reference(command, target_before, limits,
                                     speed_scale=speed_scale, control_dt=control_dt)
    recovered = reference_to_command(reference, target_before, limits,
                                      speed_scale=speed_scale, control_dt=control_dt)
    error = (recovered - command).abs()
    return {
        "reference": reference,
        "recovered": recovered,
        "max_abs_command_error": float(error.max()) if error.numel() else 0.,
        "mean_abs_command_error": float(error.mean()) if error.numel() else 0.,
        "elements_exceeding_1e_6_error": int((error > 1e-6).sum()),
        "elements": int(error.numel()),
    }
