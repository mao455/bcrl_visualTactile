"""Opt-in, loss-only current visual pose supervision; no policy integration.

Labels never enter a policy frame, action mask, observation builder or controller.
The caller must supply the actual every-tick visual512 readout and the same clock.
This module does not implement training, readout caching or a new checkpoint loader.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import torch
from torch import nn


VERSION = 'visual_relative_pose_aux_v1'
ROUTES = ('detached_readout', 'shared_vision')
POSITION_SCALE_M = .1
LOCAL_CENTER_M = (0., .025, 0.)
OBJECT_URDF_SHA256 = 'f5a7ba576e235d5f5326140f36075658a03bbe6ee57a84cb1cecf27f799071f3'


def auxiliary_contract(route):
    if route not in ROUTES:
        raise ValueError('Explicit detached_readout or shared_vision route required')
    return dict(version=VERSION, readout='last3_rgbd512_every_tick',
        target_frame='current_preaction_palm_body', position='collision_aabb_center',
        local_center_m=list(LOCAL_CENTER_M), orientation='rotation_columns6',
        position_scale_m=POSITION_SCALE_M, gradient_route=route,
        deploy_predictions=False, action_supervision='unchanged_separate_mask')


class VisualPoseHead(nn.Module):
    """The head is explicit and CPU-initialized without consuming policy RNG.

    Move it to the selected device after construction. Unlike torch.manual_seed,
    default_generator.manual_seed affects only the forked CPU RNG, not CUDA RNG.
    """
    def __init__(self, *, seed):
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError('Explicit nonnegative int63 head initialization seed required')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.linear = nn.Linear(512, 9, device='cpu', dtype=torch.float32)

    def forward(self, visual, *, route):
        auxiliary_contract(route)
        if (visual.ndim != 2 or visual.shape[1] != 512 or not visual.is_floating_point()
                or not torch.isfinite(visual).all()):
            raise ValueError('Finite actual per-tick visual512 readout required')
        return self.linear(visual.detach() if route == 'detached_readout' else visual)


@dataclass(frozen=True)
class VisualPoseLabels:
    """Separate loss context. ``valid`` is perception eligibility, never action eligibility."""
    position_m: torch.Tensor
    rotation6: torch.Tensor
    valid: torch.Tensor
    time_s: torch.Tensor


def _rotation_matrix(q):
    """Finite unit scalar-first quaternion to a local-to-world matrix."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def rotation_columns6(rotation):
    if rotation.shape[-2:] != (3, 3):
        raise ValueError('Expected rotation matrices')
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), -1)


@torch.no_grad()
def current_pose_labels(*, observation_time_s, time_before_s, camera_time_s,
                        camera_valid, object_pose_w, palm_pose_w):
    """Construct current labels without action labels or future states.

    A malformed clock fails closed. Missing cameras or invalid current poses
    mask perception loss only; masked targets are explicit finite zero fillers.
    The fixed center is for the separately hash-verified current L-shaped asset.
    """
    stamp = observation_time_s.detach()
    n = len(stamp)
    tensors = (time_before_s, camera_time_s, camera_valid, object_pose_w, palm_pose_w)
    if (stamp.shape != (n,) or n == 0 or not stamp.is_floating_point()
            or any(value.device != stamp.device for value in tensors)
            or time_before_s.shape != (n,) or not torch.equal(stamp, time_before_s)
            or not torch.isfinite(stamp).all()
            or not torch.allclose(stamp[1:]-stamp[:-1], torch.full_like(stamp[1:], 1/60), atol=1e-6, rtol=0)):
        raise ValueError('Pose labels require the exact current pre-action 60Hz clock')
    if (camera_time_s.shape != (n, 3) or camera_valid.shape != (n, 3)
            or camera_valid.dtype != torch.bool or not camera_time_s.is_floating_point()):
        raise ValueError('Three aligned camera clocks and Boolean validity required')
    selected_clock = camera_time_s[camera_valid]
    if (not torch.isfinite(selected_clock).all()
            or not torch.allclose(selected_clock, stamp[:, None].expand(-1, 3)[camera_valid], atol=1e-6, rtol=0)):
        raise ValueError('Valid camera observations must be current, not stale or future')
    if any(pose.shape != (n, 7) or not pose.is_floating_point() for pose in (object_pose_w, palm_pose_w)):
        raise ValueError('Current xyz+wxyz object and palm poses required')
    obj, palm = object_pose_w.detach().double(), palm_pose_w.detach().double()
    valid = camera_valid.all(-1).clone()
    for pose in (obj, palm):
        valid &= torch.isfinite(pose).all(-1) & ((pose[:, 3:].norm(dim=-1)-1).abs() <= 1e-3)
    # Sanitize invalid rows before division/matmul; 0*NaN would still be NaN.
    identity_pose = torch.tensor([0., 0., 0., 1., 0., 0., 0.], device=stamp.device, dtype=torch.float64)
    obj = torch.where(valid[:, None], obj, identity_pose)
    palm = torch.where(valid[:, None], palm, identity_pose)
    obj_r = _rotation_matrix(obj[:, 3:]/obj[:, 3:].norm(dim=-1, keepdim=True))
    palm_inverse = _rotation_matrix(palm[:, 3:]/palm[:, 3:].norm(dim=-1, keepdim=True)).transpose(-1, -2)
    center = obj.new_tensor(LOCAL_CENTER_M)
    center_w = obj[:, :3] + (obj_r @ center[..., None]).squeeze(-1)
    position = (palm_inverse @ (center_w-palm[:, :3])[..., None]).squeeze(-1)
    orientation = rotation_columns6(palm_inverse @ obj_r)
    return VisualPoseLabels(torch.where(valid[:, None], position, 0.),
        torch.where(valid[:, None], orientation, 0.), valid, stamp.clone())


def _tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _checked_center(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Current object asset identity changed')
    links = ET.fromstring(raw).findall('link')
    if len(links) != 1:
        raise ValueError('Single-link current object asset required')
    lower, upper = [], []
    for collider in links[0].findall('collision'):
        box, origin = collider.find('geometry/box'), collider.find('origin')
        if box is None:
            raise ValueError('Audited axis-aligned box collision geometry required')
        xyz = [float(x) for x in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()]
        rpy = [float(x) for x in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
        size = [float(x) for x in box.get('size', '').split()]
        if (any(len(x) != 3 for x in (xyz, rpy, size)) or not all(math.isfinite(x) for x in xyz+rpy+size)
                or any(rpy) or any(x <= 0 for x in size)):
            raise ValueError('Unsupported object collider definition')
        lower.append([a-b/2 for a,b in zip(xyz, size)])
        upper.append([a+b/2 for a,b in zip(xyz, size)])
    if not lower:
        raise ValueError('Object collision geometry is empty')
    center = (torch.tensor(lower, dtype=torch.float64).amin(0) + torch.tensor(upper, dtype=torch.float64).amax(0))/2
    if not torch.allclose(center, torch.tensor(LOCAL_CENTER_M, dtype=torch.float64), atol=1e-12, rtol=0):
        raise ValueError('Current collision-AABB center changed')


def perception_exposure(labels, *, control_source=None, expert_label_valid=None,
                        switch_clean=None, segment_id=None):
    """Authority partitions describe exposure only; never construct action labels.

    ``transition`` includes every expert-authority row without a legal label,
    rather than assuming all such rows are exactly one fixed queue delay.
    """
    n = len(labels.valid)
    values = (control_source, expert_label_valid, switch_clean, segment_id)
    if all(value is None for value in values):
        phases = dict(original=torch.ones_like(labels.valid))
    else:
        if (any(value is None for value in values) or any(value.shape != (n,) for value in values)
                or any(value.device != labels.valid.device for value in values)
                or control_source.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                or expert_label_valid.dtype != torch.bool or switch_clean.dtype != torch.bool
                or segment_id.dtype != torch.int64):
            raise ValueError('Complete aligned recorded authority fields required')
        if (not n or control_source[0] != 0 or not ((control_source == 0) | (control_source == 1)).all()
                or (control_source[1:] < control_source[:-1]).any()
                or (expert_label_valid & ((control_source != 1) | ~switch_clean)).any()
                or (segment_id < 0).any() or (segment_id[1:] < segment_id[:-1]).any()):
            raise ValueError('Illegal recorded student/expert authority or eligibility')
        change = ((control_source[1:] != control_source[:-1])
                  | (expert_label_valid[1:] != expert_label_valid[:-1]))
        if (change & (segment_id[1:] == segment_id[:-1])).any():
            raise ValueError('Authority/eligibility transition lacks a new segment')
        phases = dict(student_prefix=control_source == 0,
                      transition=(control_source == 1) & ~expert_label_valid,
                      legal_expert=expert_label_valid)
    return {key:dict(observed_ticks=int(mask.sum()), perception_ticks=int((mask & labels.valid).sum()),
                     invalid_perception_ticks=int((mask & ~labels.valid).sum())) for key,mask in phases.items()}


def load_pose_labels(entry, *, split, allowed_families, expected_urdf_sha256=OBJECT_URDF_SHA256):
    """Read only admitted train/validation observations, never test payloads.

    ``allowed_families`` must come from the caller's validated split manifest.
    Raw source SHA is manifest provenance here, not a new full-HDF5 hash claim.
    Actual label, clock and camera tensors are hashed in the returned report.
    """
    if (split not in ('train', 'validation') or entry.get('split', split) != split
            or not entry.get('family') or entry['family'] not in set(allowed_families)
            or not isinstance(entry.get('length'), int) or entry['length'] < 1):
        raise ValueError('Explicit admitted train/validation family required before payload access')
    with h5py.File(entry['path'], 'r') as stream:
        metadata = json.loads(stream.attrs['metadata_json'])
        group = stream['episodes'][entry['episode']]
        if (metadata.get('initial_family') != entry['family'] or metadata.get('dataset_split') != split
                or not bool(group.attrs.get('complete', False))):
            raise ValueError('Source family/split/completeness differs')
        _checked_center(metadata['config']['assets']['object_urdf'], expected_urdf_sha256)
        get = lambda key:torch.from_numpy(group[key][:])
        tensors = dict(observation_time_s=get('obs/time_s'), time_before_s=get('transition/time_before_s'),
            camera_time_s=get('obs/camera_time_s'), camera_valid=get('obs/camera_valid'),
            object_pose_w=get('transition/diagnostics_before/object_pose_w'),
            palm_pose_w=get('transition/diagnostics_before/palm_body_pose_w'))
        n = entry['length']
        terminal = get('terminated').bool() | get('truncated').bool()
        if (len(tensors['observation_time_s']) != n or terminal.shape != (n,) or terminal[:-1].any()
                or not torch.equal(get('obs/step_id'), torch.arange(n))):
            raise ValueError('Source crosses an episode boundary or skips a native observation')
        has_authority = 'transition/dagger' in group
        if has_authority != (entry.get('trajectory_kind') == 'expert_takeover_v1'):
            raise ValueError('Manifest trajectory kind differs from recorded authority provenance')
        authority = {name:get('transition/dagger/'+name) for name in
            ('control_source', 'expert_label_valid', 'switch_clean', 'segment_id')} if has_authority else {}
    context = current_pose_labels(**tensors)
    report = dict(version=VERSION, family=entry['family'], split=split, ticks=n,
        valid_perception_ticks=int(context.valid.sum()), invalid_perception_ticks=int((~context.valid).sum()),
        source_manifest_sha256=entry.get('source_sha256'), object_urdf_sha256=expected_urdf_sha256,
        input_tensor_sha256={key:_tensor_sha(value) for key,value in tensors.items()},
        authority_tensor_sha256={key:_tensor_sha(value) for key,value in authority.items()},
        perception_exposure=perception_exposure(context, **authority),
        transition_definition='Recorded expert authority without a legal expert action label; not assumed to be a fixed queue duration.',
        label_sha256={key:_tensor_sha(getattr(context, key)) for key in ('position_m', 'rotation6', 'valid', 'time_s')},
        scope='Current loss-only perception labels. Authority/eligibility masks describe exposure only; no action values are read or action masks changed. Raw HDF5 SHA not rehashed.')
    return context, report


def _validated_rows(prediction, labels, time_s):
    if not isinstance(labels, VisualPoseLabels):
        raise TypeError('Separate VisualPoseLabels loss context required')
    n = len(labels.time_s)
    if (prediction.shape != (n, 9) or not prediction.is_floating_point()
            or labels.position_m.shape != (n, 3) or labels.rotation6.shape != (n, 6)
            or not labels.position_m.is_floating_point() or not labels.rotation6.is_floating_point()
            or labels.valid.shape != (n,) or labels.valid.dtype != torch.bool
            or labels.time_s.shape != (n,) or not labels.time_s.is_floating_point()
            or not torch.isfinite(labels.time_s).all()
            or not torch.equal(time_s.to(labels.time_s.device), labels.time_s)
            or any(x.requires_grad for x in (labels.position_m, labels.rotation6, labels.time_s))):
        raise ValueError('Detached current targets, perception mask and exact readout clock required')
    if any(x.device != labels.time_s.device for x in (labels.position_m, labels.rotation6, labels.valid)):
        raise ValueError('Perception context tensors must share their device')
    mask = labels.valid.to(prediction.device)
    target = torch.cat((labels.position_m, labels.rotation6), -1)[labels.valid].to(prediction)
    selected = prediction[mask]
    if not torch.isfinite(target).all() or not torch.isfinite(selected).all():
        raise ValueError('Finite supervised pose predictions and targets required')
    first, second = target[:, 3:6].double(), target[:, 6:].double()
    if (((first.norm(dim=-1)-1).abs() > 1e-5).any()
            or ((second.norm(dim=-1)-1).abs() > 1e-5).any()
            or ((first*second).sum(-1).abs() > 1e-5).any()):
        raise ValueError('Valid orientation targets must be orthonormal rotation columns')
    return selected, target, mask


def pose_loss(prediction, labels, *, time_s):
    """Equal tick mean within one episode; no action eligibility is constructed."""
    selected, target, mask = _validated_rows(prediction, labels, time_s)
    if len(selected):
        position = ((selected[:, :3]-target[:, :3])/POSITION_SCALE_M).square().sum(-1).mean()
        orientation = (selected[:, 3:]-target[:, 3:]).square().mean()
    else:
        # Selecting first also avoids masked NaNs leaking through zero products.
        position = selected.sum()
        orientation = selected.sum()
    return dict(loss=position+orientation, position=position, orientation6=orientation,
                valid_ticks=int(mask.sum()), invalid_ticks=len(mask)-int(mask.sum()))


@torch.no_grad()
def rotation6d_to_matrix(value, *, epsilon=1e-6):
    """Diagnostic conversion; degenerate rows return zero matrices AND false validity."""
    if value.shape[-1] != 6 or not value.is_floating_point() or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Floating rotation columns and positive finite tolerance required')
    finite = torch.isfinite(value).all(-1)
    safe = torch.where(finite[..., None], value, 0.)
    first, second = safe[..., :3], safe[..., 3:]
    first_norm = first.norm(dim=-1, keepdim=True)
    x = first/first_norm.clamp_min(epsilon)
    perpendicular = second-(x*second).sum(-1, keepdim=True)*x
    second_norm = perpendicular.norm(dim=-1, keepdim=True)
    y = perpendicular/second_norm.clamp_min(epsilon)
    valid = finite & (first_norm[..., 0] > epsilon) & (second_norm[..., 0] > epsilon)
    rotation = torch.stack((x, y, torch.linalg.cross(x, y)), -1)
    return torch.where(valid[..., None, None], rotation, 0.), valid


@torch.no_grad()
def pose_metrics(prediction, labels, *, time_s):
    selected, target, _ = _validated_rows(prediction, labels, time_s)
    predicted_rotation, usable = rotation6d_to_matrix(selected[:, 3:].double())
    target_rotation, target_valid = rotation6d_to_matrix(target[:, 3:].double())
    if not target_valid.all():
        raise ValueError('Valid perception labels contain a degenerate rotation')
    n = len(selected)
    result = dict(valid_ticks=n, invalid_perception_ticks=len(labels.valid)-n,
        valid_orientation_predictions=int(usable.sum()), degenerate_orientation_predictions=int((~usable).sum()),
        position_vector_rmse_mm=None, orientation6_mse=None, orientation_mean_degrees=None,
        orientation_rms_degrees=None)
    if n:
        result.update(position_vector_rmse_mm=float((selected[:, :3].double()-target[:, :3].double()).square().sum(-1).mean().sqrt()*1000),
                      orientation6_mse=float((selected[:, 3:].double()-target[:, 3:].double()).square().mean()))
    if usable.any():
        relative = predicted_rotation[usable].transpose(-1, -2) @ target_rotation[usable]
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)
        angle = torch.rad2deg(torch.acos(cosine))
        result.update(orientation_mean_degrees=float(angle.mean()), orientation_rms_degrees=float(angle.square().mean().sqrt()))
    return result


def parameter_partitions(policy, head):
    """Exclude a registered or standalone head from the original policy clipping."""
    head_parameters = tuple(head.parameters())
    head_ids = {id(p) for p in head_parameters}
    policy_parameters = tuple(policy.parameters())
    overlap = head_ids & {id(p) for p in policy_parameters}
    if not head_ids or (overlap and overlap != head_ids):
        raise ValueError('Auxiliary head must be entirely registered or entirely separate')
    original = tuple(p for p in policy_parameters if id(p) not in head_ids)
    if not original:
        raise ValueError('Original policy parameters are required')
    return original, head_parameters


def clip_auxiliary_gradients(policy, head, *, policy_max_norm=1., head_max_norm=1.):
    """Independent norms prevent detached head gradients from scaling policy updates."""
    if any(not math.isfinite(x) or x <= 0 for x in (policy_max_norm, head_max_norm)):
        raise ValueError('Separate finite positive gradient bounds required')
    original, auxiliary = parameter_partitions(policy, head)
    # Validate both groups before mutating either gradient group.
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in original+auxiliary):
        raise FloatingPointError('Nonfinite policy or auxiliary gradient')
    return dict(policy_norm=nn.utils.clip_grad_norm_(original, policy_max_norm, error_if_nonfinite=True),
                head_norm=nn.utils.clip_grad_norm_(auxiliary, head_max_norm, error_if_nonfinite=True))
