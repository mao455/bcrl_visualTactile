"""Tactile conditioned command diffusion baseline (B1).

This is deliberately separate from :mod:`models.correction`: B1 is an offline
command BC baseline and its target is the same native expert command used by
B0.  Tactile enters only through the current observation's two physical
samples; no post transition fields or future labels are read.
"""
from torch import nn
import torch
from diffusers import DDPMScheduler, DDIMScheduler

from .nominal import PolicyConfig, VisualHistoryEncoder
from .tactile import FingerTactileEncoder
from .vendor.conditional_unet1d import ConditionalUnet1D
from ..history import recurrent_scan


class TactileCommandDiffusionPolicy(nn.Module):
    """B1: visual/proprio/task context plus causal five-finger tactile state."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        if self.cfg.horizon % 2 or not 1 <= self.cfg.replan_steps <= self.cfg.horizon:
            raise ValueError("Two-level U-Net requires even horizon; replan must fit chunk")
        if self.cfg.memory_version != 2:
            raise ValueError("B1 requires persistent memory version 2")
        self.visual = VisualHistoryEncoder(self.cfg)
        self.tactile = FingerTactileEncoder(hidden=64)
        self.fusion = nn.GRUCell(128 + 64, self.cfg.hidden)
        self.noise = DDPMScheduler(
            num_train_timesteps=self.cfg.diffusion_steps,
            beta_schedule="squaredcos_cap_v2", clip_sample=True,
            prediction_type="epsilon")
        self.sampler = DDIMScheduler.from_config(self.noise.config)
        self.denoiser = ConditionalUnet1D(
            input_dim=29, global_cond_dim=self.cfg.hidden,
            diffusion_step_embed_dim=64, down_dims=(64, 128), kernel_size=3,
            cond_predict_scale=True)

    def encode(self, obs, tactile, tactile_valid, state=None, *, return_state=False,
               return_sequence=False):
        if set(tactile) != {"contact_substeps_w_N", "contact_substeps_local_N",
                             "finger_position_base_m", "finger_rotation_base",
                             "contact_time_s"}:
            raise ValueError("Unexpected B1 tactile fields")
        visual_state = None if state is None else state.get("visual")
        tactile_state = None if state is None else state.get("tactile")
        fusion_state = None if state is None else state.get("fusion")
        visual, visual_state = self.visual(
            obs, visual_state, return_state=True, return_sequence=True)
        force = tactile["contact_substeps_w_N"]
        if force.ndim == 4:
            force = force.unsqueeze(0)
            tactile = {k: (v.unsqueeze(0) if v.ndim >= 1 else v) for k, v in tactile.items()}
            tactile_valid = tactile_valid.unsqueeze(0)
        if force.ndim != 5 or force.shape[2:] != (2, 5, 3):
            raise ValueError("B1 expects tactile force shape (B,T,2,5,3)")
        b, t, physical = force.shape[:3]
        if physical != 2 or tactile_valid.shape != (b, t):
            raise ValueError("B1 expects two physical samples and control validity")
        flat = {}
        for key, value in tactile.items():
            if key == "contact_time_s":
                if value.shape != (b, t, 2):
                    raise ValueError("contact_time_s must be (B,T,2)")
                flat[key] = value.reshape(b, t * 2)
            else:
                flat[key] = value.reshape(b, t * 2, *value.shape[3:])
        physical_valid = tactile_valid.bool().unsqueeze(-1).expand(b, t, 2).reshape(b, t * 2)
        tactile_seq, tactile_state = self.tactile(
            {"force_w_N": flat["contact_substeps_w_N"],
             "force_local_N": flat["contact_substeps_local_N"],
             "position_base_m": flat["finger_position_base_m"],
             "rotation_base": flat["finger_rotation_base"],
             "time_s": flat["contact_time_s"]},
            physical_valid, tactile_state, return_state=True, return_sequence=True)
        if tactile_seq.shape != (b, t * 2, 64):
            raise RuntimeError(f"Unexpected tactile sequence shape {tuple(tactile_seq.shape)}")
        tactile_control = tactile_seq.reshape(b, t, 2, 64)[:, :, -1]
        if visual.shape[:2] != (b, t):
            raise RuntimeError("Visual and tactile control timelines differ")
        features = torch.cat((visual, tactile_control), -1)
        sequence, hidden = recurrent_scan(self.fusion, features,
                                          obs["valid"], fusion_state)
        next_state = {"visual": visual_state, "tactile": tactile_state,
                      "fusion": hidden}
        output = sequence if return_sequence else hidden
        return (output, next_state) if return_state else output

    def loss_from_condition(self, cond, action_target, action_valid):
        if action_target.shape[1:] != (self.cfg.horizon, 29):
            raise ValueError("Action supervision must be (B,H,29)")
        eps = torch.randn_like(action_target)
        step = torch.randint(self.cfg.diffusion_steps, (len(action_target),),
                             device=action_target.device)
        noisy = self.noise.add_noise(action_target, eps, step)
        prediction = self.denoiser(noisy, step, global_cond=cond)
        squared = (prediction - eps).square().mean(-1)
        return (squared * action_valid).sum() / action_valid.sum().clamp_min(1)

    @torch.no_grad()
    def predict_from_condition(self, cond, generator=None):
        x = torch.randn((len(cond), self.cfg.horizon, 29), device=cond.device,
                        generator=generator)
        self.sampler.set_timesteps(self.cfg.inference_steps, device=cond.device)
        for step in self.sampler.timesteps:
            pred = self.denoiser(x, step, global_cond=cond)
            x = self.sampler.step(pred, step, x, eta=0.0).prev_sample
        return x.clamp(-1, 1)

