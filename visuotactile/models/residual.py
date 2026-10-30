"""Version 1 contact-gated native-command residual with independent parameters."""
import copy
import torch
from torch import nn

from .correction import CorrectionPolicy
from ..history import recurrent_scan


def observed_contact(force, valid):
    return valid.bool() & (force.norm(dim=-1).amax(dim=-1) > .1)


def compose_residual(base, raw, contact, scales):
    if base.shape != raw.shape or base.shape[-1] != 29 or contact.shape != base.shape[:-1]:
        raise ValueError('Residual action/gate shapes disagree')
    if not torch.isfinite(base).all() or not torch.isfinite(raw).all():
        raise ValueError('Nonfinite residual action')
    if scales.shape != (29,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Positive per-joint native command scales required')
    clipped = base.clamp(-1, 1)
    delta = scales * raw.tanh()
    return torch.where(contact[..., None], (clipped + delta).clamp(-1, 1), clipped), delta


class ResidualCorrection(CorrectionPolicy):
    version = 1

    def __init__(self, base_encoder, cfg, scales, *, force_masked=False):
        if (cfg.perception_version, cfg.memory_version, cfg.own_plan_mode, cfg.action_history_mode) != (2, 2, 'none', 'none'):
            raise ValueError('Residual v1 requires frozen B0 v2/no-own-plan, not v3 or B1')
        super().__init__(cfg)
        self.visual = copy.deepcopy(base_encoder)
        self.visual.requires_grad_(False)
        self.visual.temporal.requires_grad_(True)
        self.policy = nn.Sequential(nn.Linear(cfg.hidden+192, 256), nn.SiLU(),
                                    nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 29))
        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)
        self.register_buffer('scales', torch.as_tensor(scales, dtype=torch.float32))
        self.force_masked = force_masked

    def forward_cached(self, features, tactile, tactile_valid, plans, plan_valid, motion, state=None):
        state = {} if state is None else state
        batch, steps = features.shape[:2]
        valid = torch.ones(batch, steps, dtype=torch.bool, device=features.device)
        visual, visual_state = recurrent_scan(self.visual.temporal, features, valid, state.get('visual'))
        if self.force_masked:
            tactile = {name: torch.zeros_like(value) if name in ('force_w_N', 'force_local_N') else value
                       for name, value in tactile.items()}
        touch, tactile_state = self.tactile(tactile, tactile_valid, state.get('tactile'),
                                           return_state=True, return_sequence=True)
        touch = touch[:, 1::2]
        intent = plans.new_zeros(batch*steps, 64)
        flat_plan, flat_valid = plans.flatten(0, 1), plan_valid.flatten(0, 1)
        for cursor in range(plans.shape[2]):
            intent = torch.where(flat_valid[:, cursor, None],
                self.intent(flat_plan[:, cursor].detach(), intent), intent)
        intent, intent_state = recurrent_scan(self.intent_memory, intent.reshape(batch, steps, 64),
                                               valid, state.get('intent'))
        proprio = (motion['proprio']-self.visual.proprio_mean)/self.visual.proprio_std
        sequence = torch.cat((proprio, motion['previous_command'], motion['command_valid'][..., None].float(),
                              motion['actuator_target_rad'], motion['tracking_error_rad']), -1)
        motion_feature, motion_state = recurrent_scan(self.motion, sequence, valid, state.get('motion'))
        raw = self.policy(torch.cat((visual, touch, intent, motion_feature), -1))
        return raw, dict(visual=visual_state, tactile=tactile_state, intent=intent_state, motion=motion_state)


def residual_loss(action, target, delta, contact, regularization=1e-4):
    if not contact.any():
        return action.sum()*0
    error = (action-target).square()
    per_step = .5*error[..., :7].mean(-1)+.5*error[..., 7:].mean(-1)
    penalty = .5*delta[..., :7].square().mean(-1)+.5*delta[..., 7:].square().mean(-1)
    return (per_step[contact]+regularization*penalty[contact]).mean()
