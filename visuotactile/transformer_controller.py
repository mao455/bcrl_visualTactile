"""G01 actual mixed-command history and10Hz residual with native60Hz feedback."""
from pathlib import Path
import time

import torch

from .action_history import ExecutionHistory
from .compact import sha256
from .long_controller import LongController
from .long_data import proprio_radians
from .models.residual import observed_contact
from .models.transformer_residual import HISTORY_PROTOCOL, TransformerResidual, TransformerResidualConfig, combine
from .train_long import move
from .transformer_data import residual_inputs, tactile_features


class TransformerController:
    def __init__(self, base_path, residual_path, dino, *, bypass=False, device='cuda:0'):
        base_saved = torch.load(base_path,map_location='cpu',weights_only=False)
        saved = torch.load(residual_path,map_location='cpu',weights_only=False)
        if (saved.get('residual_version')!=1 or not saved.get('trained') or saved.get('updates',0)<1
                or saved.get('base_sha256')!=sha256(Path(base_path)) or saved.get('history_protocol')!=HISTORY_PROTOCOL):
            raise ValueError('Untrained or mismatched Transformer residual cannot control')
        self.base = LongController(base_saved,dino,device)
        if self.base.model.cfg.tactile:
            raise ValueError('G01 nominal must be vision-only')
        self.base.model.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            self.model = TransformerResidual(TransformerResidualConfig(**saved['config']))
        self.model.load_state_dict(saved['model'])
        self.model.to(device).eval().requires_grad_(False)
        self.device,self.bypass = device,bypass
        self.episode = None
        self.pending = False
        self.checkpoint_sha256 = sha256(Path(residual_path))

    @torch.no_grad()
    def act(self, packet):
        if self.pending:
            raise ValueError('Missing native execution acknowledgment')
        episode,step = int(packet['episode_id'][0]),int(packet['step_id'][0])
        if episode!=self.episode:
            if step!=0:
                raise ValueError('Residual requires complete episode history')
            self.episode,self.state = episode,None
            self.actual = ExecutionHistory(1)
            self.records = {name:[] for name in ('time','proprio','action','target','tracking','tactile','base','plans','visual')}
        base = self.base.act(packet)
        history = self.actual.context(packet)
        now = packet['time_s'][0].cpu()
        action = torch.cat((history['previous_issued_command'][0],history['previous_delayed_command'][0],
            torch.stack((now-history['previous_command_time_s'][0],now-history['previous_delayed_time_s'][0])).float(),
            history['action_history_valid'][0,None].float()))
        action = torch.where(history['action_history_valid'][0],action,0.)
        tactile = tactile_features(packet['contact_substeps_w_N'][0].cpu(),packet['contact_substeps_local_N'][0].cpu(),
            packet['finger_position_base_m'][0].cpu(),packet['finger_rotation_base'][0].cpu(),
            packet['contact_time_s'][0].cpu(),now,packet['contact_valid'][0].cpu())
        current = dict(time=now,proprio=proprio_radians(packet['proprio'][0].cpu(),self.base.limits),action=action,
            target=packet['actuator_target_rad'][0].cpu(),tracking=packet['tracking_error_rad'][0].cpu(),
            tactile=tactile,base=base[0])
        for name,value in current.items():
            self.records[name].append(value)
        if step%12==0:
            self.records['plans'].append(self.base.plan[0].clone())
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.startswith('cuda')):
                visual = self.base.model.project_frames(self.base.records['patches'][-1][None].to(self.device),
                    self.base.records['depth'][-1][None].to(self.device),
                    self.base.records['camera_valid'][-1][None].to(self.device))
            self.records['visual'].append(visual[0].float().cpu())
        record = {name:torch.stack(values) for name,values in self.records.items()}
        anchors = torch.tensor([step],dtype=torch.long) if step%6==0 else torch.empty(0,dtype=torch.long)
        inputs = move(residual_inputs(record,step,step+1,anchors),self.device)
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        began = time.perf_counter()
        prediction,self.state = self.model(inputs,self.state)
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-began
        if step%6==0:
            self.delta_window = prediction[0].float().cpu()
            self.generation_step = step
        if not self.generation_step<=step<self.generation_step+6:
            raise ValueError('Residual window is stale or corrects an already-executed prefix')
        delta = self.delta_window[step-self.generation_step][None]
        gate = observed_contact(packet['contact_w_N'].cpu(),packet['contact_valid'].cpu())
        command = base if self.bypass else combine(base,delta,gate)
        self.last = dict(base=base,delta=delta,gate=gate,command=command,
            residual_generation_step=self.generation_step,residual_cursor=step-self.generation_step,
            nominal_generation_step=step-step%12,nominal_cursor=step%12,
            residual_forward_s=elapsed,residual_refresh=step%6==0,actual_previous=history['previous_issued_command'],
            nominal_previous=self.base.records['action'][-1][:29][None])
        self.packet = packet
        self.pending = True
        return command

    def acknowledge(self, transition, submitted, *, source='student'):
        if not self.pending or source not in ('student','released_expert_rollin'):
            raise ValueError('Unknown execution source or duplicate acknowledgment')
        if source=='student' and not torch.equal(submitted.cpu(),self.last['command']):
            raise ValueError('Student executed a different residual/base command')
        self.actual.observe(transition,submitted.clamp(-1,1),self.packet['episode_id'],self.packet['step_id'])
        self.base.execution.observe(transition,submitted.clamp(-1,1),self.packet['episode_id'],self.packet['step_id'])
        self.pending = False
