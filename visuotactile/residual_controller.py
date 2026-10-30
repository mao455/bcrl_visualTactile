"""Trained residual execution; mixed feedback belongs only to Correction."""
from pathlib import Path
import torch

from .action_history import ExecutionHistory
from .compact import sha256
from .correction import remaining_nominal_chunk
from .history import reset_state
from .models.residual import ResidualCorrection, compose_residual, observed_contact
from .observations import nominal_keys
from .residual_data import BASE_HASH, load_base


class ResidualController:
    def __init__(self, checkpoint, *, bypass=False, device='cuda:0'):
        self.base = load_base(device)
        self.device = torch.device(device)
        self.checkpoint_sha256 = sha256(Path(checkpoint))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (saved.get('residual_checkpoint_version') != 1 or saved.get('base_sha256') != BASE_HASH
                or not saved.get('trained') or saved.get('optimizer_updates', 0) < 1
                or saved.get('input_version') != 'residual_frozen_frame_memory2_v1'
                or saved.get('action_version') != 'gated_additive_native_command_v1'):
            raise ValueError('Untrained or incompatible residual checkpoint cannot execute')
        if saved['base_config'] != vars(self.base.model.cfg):
            raise ValueError('Residual and B0 training contracts differ')
        with torch.random.fork_rng(devices=[]):
            self.model = ResidualCorrection(self.base.model.encoder, self.base.model.cfg,
                                           saved['scales'], force_masked=saved['force_masked'])
        self.model.load_state_dict(saved['model'])
        self.model.to(self.device).requires_grad_(False).eval()
        self.bypass = bypass
        self.history = ExecutionHistory(1)
        self.memory = None
        self.episode_ids = torch.full((1,), -1, dtype=torch.long)
        self.pending = False

    @torch.no_grad()
    def act(self, packet):
        if self.pending:
            raise ValueError('Correction requires native execution acknowledgment before next observation')
        base = self.base.act(packet).to(self.device)
        history = self.history.context(packet)
        changed = packet['episode_id'].cpu() != self.episode_ids
        self.memory = reset_state(self.memory, changed)
        self.episode_ids = packet['episode_id'].cpu().clone()
        current = {name: self.base.last_packet[name].to(self.device)[:, None] for name in nominal_keys(2)}
        current['valid'] = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        features, _ = self.model.visual.encode_frames(current)
        tactile = {target: packet[source].to(self.device) for source, target in (
            ('contact_substeps_w_N', 'force_w_N'), ('contact_substeps_local_N', 'force_local_N'),
            ('finger_position_base_m', 'position_base_m'), ('finger_rotation_base', 'rotation_base'),
            ('contact_time_s', 'time_s'))}
        valid = packet['contact_valid'].to(self.device)[:, None].expand(-1, 2)
        plans, plan_valid = remaining_nominal_chunk(self.base.plan, self.base.cursor-1)
        motion = {name: packet[name].to(self.device)[:, None] for name in (
                  'proprio', 'actuator_target_rad', 'tracking_error_rad')}
        motion.update(previous_command=history['previous_issued_command'].to(self.device)[:, None],
                      command_valid=history['action_history_valid'].to(self.device)[:, None])
        raw, self.memory = self.model.forward_cached(features, tactile, valid,
            plans.to(self.device)[:, None], plan_valid.to(self.device)[:, None], motion, self.memory)
        contact = observed_contact(packet['contact_w_N'], packet['contact_valid']).to(self.device)
        action, delta = compose_residual(base, raw[:, 0], contact, self.model.scales)
        if self.bypass:
            action = base
        self.last = dict(base=base.cpu(), delta=delta.cpu(), raw=raw[:, 0].cpu(), contact=contact.cpu(),
            command=action.cpu(), previous_command=history['previous_issued_command'],
            command_valid=history['action_history_valid'], plans=plans, plan_valid=plan_valid,
            cursor=(self.base.cursor-1).clone(), generation_step=self.base.plan_generation_step.clone(),
            memory_updates=self.base.memory_updates.clone())
        self.last_packet = packet
        self.pending = True
        return action.cpu()

    def acknowledge(self, transition, submitted, *, source='student'):
        if not self.pending or source not in ('student', 'released_expert_rollin'):
            raise ValueError('Missing pending residual action or undeclared execution source')
        if source == 'student' and not torch.equal(submitted.cpu(), self.last['command']):
            raise ValueError('Student command differs from the selected residual/base action')
        self.history.observe(transition, submitted.clamp(-1, 1),
                             self.last_packet['episode_id'], self.last_packet['step_id'])
        self.base.observe_execution(transition)
        self.pending = False
