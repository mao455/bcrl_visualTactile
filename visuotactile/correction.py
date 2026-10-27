"""Causal correction context and NON-CONTROLLING real-data integration probe.

This module intentionally has no trained-controller loader yet. Shadow candidate
commands are diagnostics, never actions or imitation labels. A justified trained
checkpoint/evaluation gate is required before introducing live correction.
"""
import time
import math
import torch
from .history import OnlineHistory, reset_state
from .models.correction import CorrectionPolicy
from .observations import nominal_keys


def remaining_nominal_chunk(plan, cursor):
    """Current and future predicted commands; mask the unpredicted padded tail."""
    horizon = plan.shape[1]
    if cursor.shape != (len(plan),) or (cursor < 0).any() or (cursor >= horizon).any():
        raise ValueError("Current cursor must index an existing predicted nominal command")
    indices = cursor[:,None] + torch.arange(horizon,device=plan.device)
    valid = indices < horizon
    index = indices.clamp(max=horizon-1)[...,None].expand(-1,-1,29)
    return plan.gather(1,index), valid


class CorrectionContext:
    def __init__(self, num_envs, nominal_cfg, force_history_steps=12):
        self.version = nominal_cfg.perception_version
        self.visual = OnlineHistory(num_envs,nominal_cfg.history,nominal_cfg.stride,keys=nominal_keys(self.version))
        extra = ("contact_substeps_local_N","finger_position_base_m","finger_rotation_base","contact_time_s",
                 "actuator_target_rad","tracking_error_rad") if self.version == 2 else ()
        self.feedback = OnlineHistory(num_envs,force_history_steps,keys=(
            "contact_substeps_w_N","contact_valid","proprio","previous_command","command_valid")+extra)
        self.episode_ids = torch.full((num_envs,),-1,dtype=torch.long)
        self.previous_command = torch.zeros(num_envs,29)
        self.command_valid = torch.zeros(num_envs,dtype=torch.bool)
        self.awaiting_command = False

    def observe(self, packet, nominal_plan, current_cursor):
        if self.awaiting_command:
            raise RuntimeError("Commit the actually issued command before the next observation")
        changed = packet["episode_id"].cpu() != self.episode_ids
        self.previous_command[changed] = 0
        self.command_valid[changed] = False
        self.episode_ids = packet["episode_id"].cpu().clone()
        self.visual.append(packet)
        self.feedback.append({**packet,"previous_command":self.previous_command,"command_valid":self.command_valid})
        feedback = self.feedback.batch()
        force = feedback["contact_substeps_w_N"]
        b,t,s,f,c = force.shape
        tactile_valid = (feedback["valid"] & feedback["contact_valid"])[...,None].expand(-1,-1,s).reshape(b,t*s)
        chunk, valid = remaining_nominal_chunk(nominal_plan,current_cursor)
        self.awaiting_command = True
        result = {"visual_history":self.visual.batch(),
                "tactile_history":force.reshape(b,t*s,f,c),
                "tactile_valid":tactile_valid,
                "nominal_chunk":chunk,"nominal_valid":valid,
                "motion_history":{k:feedback[k] for k in ("proprio","previous_command","command_valid","valid")}}
        if self.version == 2:
            result["tactile_history"] = {
                "force_w_N":force.reshape(b,t*s,f,c),
                "force_local_N":feedback["contact_substeps_local_N"].reshape(b,t*s,f,c),
                "position_base_m":feedback["finger_position_base_m"].reshape(b,t*s,f,c),
                "rotation_base":feedback["finger_rotation_base"].reshape(b,t*s,f,3,3),
                "time_s":feedback["contact_time_s"].reshape(b,t*s)}
            result["motion_history"].update({k:feedback[k] for k in ("actuator_target_rad","tracking_error_rad")})
        return result

    def commit(self, command):
        if not self.awaiting_command or command.shape != self.previous_command.shape or not torch.isfinite(command).all():
            raise ValueError("Expected one finite actually issued native command after observe")
        self.previous_command.copy_(command.detach())
        self.command_valid[:] = True
        self.awaiting_command = False


class CorrectionShadow:
    """Untrained network plumbing probe; returns logs and never an execute action."""
    def __init__(self, nominal_controller, num_envs, contact_threshold_N=.1):
        if not math.isfinite(contact_threshold_N) or contact_threshold_N < 0:
            raise ValueError("Contact threshold must be nonnegative")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            self.model = CorrectionPolicy(nominal_controller.model.cfg).eval()
        self.model.visual.proprio_mean.copy_(nominal_controller.model.encoder.proprio_mean)
        self.model.visual.proprio_std.copy_(nominal_controller.model.encoder.proprio_std)
        self.context = CorrectionContext(num_envs,nominal_controller.model.cfg)
        self.threshold = contact_threshold_N
        self.inference_ms = []
        self.memory = None
        self.episode_ids = torch.full((num_envs,),-1,dtype=torch.long)

    @torch.no_grad()
    def observe(self, packet, nominal_controller):
        start = time.perf_counter()
        inputs = self.context.observe(nominal_controller.last_packet,nominal_controller.plan,nominal_controller.cursor-1)
        self.last_inputs = inputs
        if self.model.memory_version == 2:
            changed = packet["episode_id"].cpu() != self.episode_ids
            self.memory = reset_state(self.memory,changed)
            self.episode_ids = packet["episode_id"].cpu().clone()
            inputs = latest_correction_inputs(inputs,packet["contact_substeps_w_N"].shape[1])
            candidate,self.memory = self.model(**inputs,state=self.memory,return_state=True)
        else:
            candidate = self.model(**inputs)
        if not torch.isfinite(candidate).all():
            raise FloatingPointError("Nonfinite correction shadow output")
        contact = packet["contact_w_N"].norm(dim=-1).amax(dim=-1) > self.threshold
        contact &= packet["contact_valid"]
        self.inference_ms.append(1000*(time.perf_counter()-start))
        return {"untrained_candidate_diagnostic":candidate,"contact_mask":contact,
                "nominal_remaining_chunk":inputs["nominal_chunk"],
                "nominal_remaining_valid":inputs["nominal_valid"],
                "previous_issued_command":self.context.previous_command.clone(),
                "previous_command_valid":self.context.command_valid.clone()}


def latest_correction_inputs(inputs,physical_samples=2):
    """Consume only new samples; repeated finite-window history must not be counted twice."""
    result = dict(inputs)
    for key in ("visual_history","motion_history"):
        result[key] = {k:v[:,-1:] for k,v in inputs[key].items()}
    result["tactile_valid"] = inputs["tactile_valid"][:,-physical_samples:]
    tactile = inputs["tactile_history"]
    result["tactile_history"] = ({k:v[:,-physical_samples:] for k,v in tactile.items()}
                                 if isinstance(tactile,dict) else tactile[:,-physical_samples:])
    return result
