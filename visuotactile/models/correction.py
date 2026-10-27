"""Replaceable correction research interface. Untrained networks cannot execute."""
import torch
from torch import nn
from .nominal import VisualHistoryEncoder
from ..history import recurrent_scan


class TactileEncoder(nn.Module):
    """Five aggregate normal-force vectors, not tactile pixels or pressure."""
    def __init__(self, hidden=64):
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(15,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.temporal = nn.GRUCell(64,hidden)
        self.hidden = hidden

    def forward(self, force_history_w_N, valid, state=None, *, return_state=False):
        if force_history_w_N.ndim != 4 or force_history_w_N.shape[2:] != (5,3):
            raise ValueError("Tactile history must be (B,T,5,3)")
        x = force_history_w_N.sign()*torch.log1p(force_history_w_N.abs())
        x = self.spatial(x.flatten(2))
        _,h = recurrent_scan(self.temporal,x,valid,state)
        return (h,h) if return_state else h


class CorrectionPolicy(nn.Module):
    """Separate visual/tactile/action sequence encoders -> full candidate action.

    Expert actions belong only in an external training loss. The nominal chunk
    must be a causal model prediction; never replace it with demonstrated future.
    No training objective or claimed tactile benefit is implemented here yet.
    """
    def __init__(self, nominal_cfg):
        super().__init__()
        self.visual = VisualHistoryEncoder(nominal_cfg)  # own parameters, no nominal feature sharing
        self.perception_version = nominal_cfg.perception_version
        if self.perception_version == 2:
            from .tactile import FingerTactileEncoder
            self.tactile = FingerTactileEncoder()
        else:
            self.tactile = TactileEncoder()
        self.intent = nn.GRUCell(29,64)
        self.memory_version = nominal_cfg.memory_version
        if self.memory_version == 2:
            self.intent_memory = nn.GRUCell(64,64)
        self.motion = nn.GRUCell(nominal_cfg.proprio_dim+29+1+(58 if self.perception_version == 2 else 0),64)
        self.policy = nn.Sequential(nn.Linear(nominal_cfg.hidden+192,256),nn.SiLU(),
                                    nn.Linear(256,128),nn.SiLU(),nn.Linear(128,29),nn.Tanh())

    def forward(self, visual_history, tactile_history, tactile_valid, nominal_chunk,
                nominal_valid, motion_history, state=None, *, return_state=False):
        state = {} if state is None else state
        intent = nominal_chunk.new_zeros(len(nominal_chunk),64)
        for i in range(nominal_chunk.shape[1]):
            intent = torch.where(nominal_valid[:,i,None],self.intent(nominal_chunk[:,i].detach(),intent),intent)
        if self.memory_version == 2:
            previous = state.get("intent",torch.zeros_like(intent))
            intent = torch.where(nominal_valid.any(-1,keepdim=True),self.intent_memory(intent,previous),previous)
        extra = {"actuator_target_rad","tracking_error_rad"} if self.perception_version == 2 else set()
        if set(motion_history) != {"proprio","previous_command","command_valid","valid"}|extra:
            raise ValueError("Correction motion accepts encoders and causal issued-command history only")
        q = (motion_history["proprio"]-self.visual.proprio_mean)/self.visual.proprio_std
        motion = state.get("motion",q.new_zeros(len(q),64))
        sequence = torch.cat((q,motion_history["previous_command"],motion_history["command_valid"][...,None].float()),-1)
        if self.perception_version == 2:
            sequence = torch.cat((sequence,motion_history["actuator_target_rad"],motion_history["tracking_error_rad"]),-1)
        for i in range(sequence.shape[1]):
            motion = torch.where(motion_history["valid"][:,i,None],self.motion(sequence[:,i],motion),motion)
        visual, visual_state = self.visual(visual_history,state.get("visual"),return_state=True)
        tactile, tactile_state = self.tactile(tactile_history,tactile_valid,state.get("tactile"),return_state=True)
        context = torch.cat((visual,tactile,intent,motion),dim=-1)
        result = self.policy(context)
        next_state = dict(visual=visual_state,tactile=tactile_state,intent=intent,motion=motion)
        return (result,next_state) if return_state else result


def compose_action(nominal, candidate=None, contact=None, *, trained=False, blend=1.0):
    """Blend complete native commands (not radians), then original env handles them.

    Arm velocity-style and hand absolute-position semantics remain in env.step.
    No contact or disabled correction is exactly nominal after the native clip.
    This gate is not a force/torque safety guarantee.
    """
    if not torch.isfinite(nominal).all() or nominal.shape[-1] != 29:
        raise ValueError("Invalid nominal command")
    base = nominal.clamp(-1,1)
    if candidate is None:
        return base
    if not trained:
        raise RuntimeError("Untrained correction is prohibited from execution")
    if not 0 <= blend <= 1 or candidate.shape != nominal.shape or not torch.isfinite(candidate).all():
        raise ValueError("Invalid correction candidate or blend")
    if contact is None or contact.shape != nominal.shape[:-1]:
        raise ValueError("A per-environment observed contact mask is required")
    combined = ((1-blend)*base + blend*candidate).clamp(-1,1)
    return torch.where(contact[...,None],combined,base)
