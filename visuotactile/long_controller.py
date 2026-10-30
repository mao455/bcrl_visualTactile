"""Native-tick deployment for candidate2, with actual execution acknowledgments."""
import torch

from .action_history import ExecutionHistory
from .action_reference import reference_to_command
from .long_data import inputs_for, depth_grid, proprio_radians, tactile_values
from .models.long_horizon import LongHorizonConfig, LongHorizonPolicy, DINO_REVISION


class LongController:
    def __init__(self, saved, dino, device='cuda:0'):
        if saved.get('version') != 2 or saved.get('updates',0) < 1:
            raise ValueError('Only trained candidate2 checkpoints may control')
        if saved['vision']['dino_revision'] != DINO_REVISION:
            raise ValueError('DINO source contract differs')
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            self.model = LongHorizonPolicy(LongHorizonConfig(**saved['config']))
        self.model.load_state_dict(saved['model'])
        self.model.to(device).eval()
        self.dino = dino
        self.metadata = saved['metadata']
        self.vision = saved['vision']
        self.limits = torch.tensor(self.metadata['joint_limits_rad'][0])
        self.speed = self.metadata['config']['action']['dof_speed_scale']
        self.dt = self.metadata['control_dt']
        self.generator = torch.Generator(device=device).manual_seed(0)
        self.episode = None
        self.state = None
        self.records = {}
        self.execution = ExecutionHistory(1)
        self.next_step = 0
        self.plan = None
        self.checkpoint_epoch = saved['epoch']
        self.updates = saved['updates']

    @torch.no_grad()
    def act(self, packet):
        episode = int(packet['episode_id'][0])
        step = int(packet['step_id'][0])
        if episode != self.episode:
            if step != 0:
                raise ValueError('Online memory requires a complete native episode prefix')
            self.episode,self.next_step,self.state = episode,0,None
            self.records = {name:[] for name in ('time','proprio','action','patches','depth','camera_time','camera_valid')}
            if self.model.cfg.tactile:
                self.records['tactile'] = []
            self.execution = ExecutionHistory(1)
        if step != self.next_step:
            raise ValueError('Native observation skipped or duplicated')
        history = self.execution.context(packet)
        now = packet['time_s'][0].cpu()
        action = torch.cat((history['previous_issued_command'][0],history['previous_delayed_command'][0],
            torch.stack((now-history['previous_command_time_s'][0],now-history['previous_delayed_time_s'][0])).float(),
            history['action_history_valid'][0,None].float()))
        action = torch.where(history['action_history_valid'][0],action,0.)
        self.records['time'].append(now)
        self.records['proprio'].append(proprio_radians(packet['proprio'][0].cpu(),self.limits))
        self.records['action'].append(action)
        if self.model.cfg.tactile:
            self.records['tactile'].append(tactile_values(packet['contact_substeps_w_N'][0].cpu(),
                packet['contact_substeps_local_N'][0].cpu(),packet['contact_time_s'][0].cpu(),
                now,packet['contact_valid'][0].cpu()))
        if step%12 == 0:
            self.records['patches'].append(self.dino(packet['image'][0,:,:3].to(self.device)).cpu())
            self.records['depth'].append(depth_grid(packet['image'][0,:,3:].cpu(),packet['depth_valid'][0].cpu()))
            self.records['camera_time'].append(packet['camera_time_s'][0].cpu())
            self.records['camera_valid'].append(packet['camera_valid'][0].cpu())
        record = {name:torch.stack(values) for name,values in self.records.items()}
        inputs = {name:value.to(self.device) for name,value in inputs_for(record,step,step+1,
            self.model.cfg.tactile,torch.tensor([step])).items()}
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type == 'cuda'):
            condition,self.state = self.model.encode(inputs,self.state)
            if step%12 == 0:
                self.plan = self.model.predict_from_condition(condition,self.generator).float().cpu()
        self.last_condition = condition.float().cpu()
        self.reference = self.plan[:,step%12]
        command = reference_to_command(self.reference,packet['actuator_target_rad'].cpu(),self.limits,
                                        speed_scale=self.speed,control_dt=self.dt)
        self.previous_command = command
        self.last_packet = packet
        self.next_step += 1
        self.cursor = step%12
        return command

    def observe_execution(self, transition):
        self.execution.observe(transition,self.previous_command,self.last_packet['episode_id'],
                               self.last_packet['step_id'])
