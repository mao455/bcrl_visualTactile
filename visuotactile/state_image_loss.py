"""Offline-only sample/x0 geometry objective, independent of simulator lifetime.

Link-frame FK is in the robot base frame; fingers are not contact points.
Weights are calibrated once on train-only predictions and never on validation.
"""
import torch

from .dp_objectives import masked_objectives
from .kinematics import EncoderKinematics, FINGERS

LOSS_ID = 'position_geometry_sample_v1'
PALM_LOSS_ID = 'palm_command_sample_commit12'
COMPONENTS = ('position', 'rotation', 'fingertip', 'temporal')


def rotation_chord(a, b):
    # ||Ra-Rb||_F^2 / 8 == sin(theta/2)^2, smooth at zero and pi.
    return (a-b).square().sum((-2, -1))/8


class GeometryObjective:
    def __init__(self, metadata, mode=LOSS_ID, commit=12):
        if mode not in (LOSS_ID, PALM_LOSS_ID) or commit != 12:
            raise ValueError('Explicit supported geometry objective/commit required')
        self.mode, self.commit, self.gain = mode, commit, 1.
        self.fk = EncoderKinematics(metadata['config']['assets']['robot_urdf'],
            metadata['joint_order'], links=('iiwa14_link_7', *FINGERS))
        if self.fk.sha256 != metadata['fk_urdf_sha256']:
            raise ValueError('FK asset differs from recorded data')
        if mode == PALM_LOSS_ID:
            self.palm_fk = EncoderKinematics(metadata['config']['assets']['robot_urdf'],
                metadata['joint_order'], links=('iiwa14_link_7',))
        self.weights = {name: 0. for name in (('position', 'rotation') if mode == PALM_LOSS_ID else COMPONENTS)}

    def components(self, prediction, command, valid, scale, offset):
        target = torch.where(valid[..., None], command*scale+offset, 0.)
        base = masked_objectives(prediction, target, valid)['B']
        if self.mode == PALM_LOSS_ID:
            # The same anchor's first contract.commit actions are actually issued.
            mask = valid & (torch.arange(valid.shape[1], device=valid.device)[None] < self.commit)
            if not mask.any():
                zero = prediction.sum()*0.
                return dict(base=base, position=zero, rotation=zero)
            # sample means x0: no SNR division and no DDIM graph. FK uses real
            # target radians, never noisy actions or next-tick measured state.
            q = (prediction[mask].float()-offset.float())/scale.float()
            xyz, rotation = self.palm_fk(q)
            with torch.no_grad(): exact_xyz, exact_rotation = self.palm_fk(command[mask].float())
            return dict(base=base, position=(xyz[:, 0]-exact_xyz[:, 0]).square().sum(-1).mean(),
                rotation=.5*(rotation[:, 0]-exact_rotation[:, 0]).square().sum((-2, -1)).mean())
        # Select valid rows BEFORE FK. Invalid labels cannot affect its graph.
        q = (prediction[valid]-offset)/scale
        truth = command[valid]
        if not len(q):
            raise ValueError('No valid targets')
        xyz, rotation = self.fk(q)
        with torch.no_grad(): exact_xyz, exact_rotation = self.fk(truth)
        position = ((xyz[:, 0]-exact_xyz[:, 0])/.1).square().sum(-1).mean()
        orientation = rotation_chord(rotation[:, 0], exact_rotation[:, 0]).mean()
        finger = ((xyz[:, 1:]-exact_xyz[:, 1:])/.05).square().sum(-1).mean()
        # Match demonstrated differences, not zero velocity; equal Arm/Hand weight.
        pairs = valid[:, 1:] & valid[:, :-1]
        temporal = masked_objectives(prediction[:, 1:]-prediction[:, :-1],
            target[:, 1:]-target[:, :-1], pairs)['B']
        return dict(base=base, position=position, rotation=orientation, fingertip=finger, temporal=temporal)

    def forward(self, model, condition, command, valid, generator):
        if model.noise.config.prediction_type != 'sample' or model.horizon != 24:
            raise ValueError('New geometry objective is explicitly H24 sample only')
        target = torch.where(valid[..., None], command*model.action_scale+model.action_offset, 0.)
        noise = torch.randn(target.shape, device=target.device, generator=generator)
        ticks = torch.randint(100, (len(target),), device=target.device, generator=generator)
        prediction = model.denoiser(model.noise.add_noise(target, noise, ticks), ticks, global_cond=condition)
        values = self.components(prediction, command, valid, model.action_scale, model.action_offset)
        total = values['base']+self.gain*sum(self.weights[k]*values[k] for k in self.weights)
        return total, values, prediction


def calibrated_weights(rows):
    # Each auxiliary term <=5% of base value AND output-gradient norm in the
    # worst of the frozen train batches. Sum bound <=20% by triangle inequality.
    return {key: min(1., *(min(.05*r['loss']['base']/max(r['loss'][key], 1e-12),
        .05*r['gradient']['base']/max(r['gradient'][key], 1e-12)) for r in rows))
        for key in COMPONENTS}


def calibrated_palm_weights(rows):
    """One train-only scale check: each palm term <=5% of base value/gradient.

    Position uses m² (lambda has units m^-2); rotation is dimensionless.
    This rule is frozen before inspecting validation; no repeated tuning.
    """
    return {key: min(min(.05*r['loss']['base']/max(r['loss'][key], 1e-12),
        .05*r['gradient']['base']/max(r['gradient'][key], 1e-12)) for r in rows)
        for key in ('position', 'rotation')}
