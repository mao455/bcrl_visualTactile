"""Direct native commands and exact execution acknowledgments at every tick."""
import torch

from .action_history import ExecutionHistory
from .native_data import depth_grid, proprio_radians
from .native_data import geometry, object_dimensions, robot_rows
from .models.native_dp import load_policy


class NativeController:
    def __init__(self, saved, metadata, dino=None, device='cuda:0'):
        if saved.get('updates', 0) < 1:
            raise ValueError('An untrained native policy cannot control')
        with torch.random.fork_rng(devices=[]):
            self.model = load_policy(saved, device)
        self.device, self.dino = torch.device(device), dino
        self.limits = torch.tensor(metadata['joint_limits_rad'][0])
        self.dimensions = object_dimensions(metadata) if self.model.group=='S-NativeDP' else None
        self.position_mode = self.model.contract['action']=='absolute_joint_position_target_rad'
        self.generator = torch.Generator(device=device).manual_seed(self.model.contract.get('sampler_seed',7093))
        self.episode, self.memory, self.plan = None, None, None
        self.execution = ExecutionHistory(1)
        self.next_step = 0

    @torch.no_grad()
    def act(self, packet, diagnostics=None):
        episode, step = int(packet['episode_id'][0]), int(packet['step_id'][0])
        if episode != self.episode:
            if step != 0:
                raise ValueError('Deployment requires an episode start')
            self.episode, self.next_step, self.memory, self.plan = episode, 0, None, None
            self.execution = ExecutionHistory(1)
            self.position_previous = None
            self.started = packet['time_s'][0].cpu()
        if step != self.next_step:
            raise ValueError('Native tick skipped or duplicated')
        stamp = packet['time_s'][0].cpu()
        if self.position_mode:
            past=torch.zeros(61)
            if self.position_previous is not None:
                requested,applied,previous_stamp=self.position_previous
                past=torch.cat((requested,applied,(stamp-previous_stamp).float().repeat(2),torch.ones(1)))
        else:
            history = self.execution.context(packet)
            past = torch.cat((history['previous_issued_command'][0], history['previous_delayed_command'][0],
                torch.stack((stamp-history['previous_command_time_s'][0], stamp-history['previous_delayed_time_s'][0])).float(),
                history['action_history_valid'][0, None].float()))
            past = torch.where(history['action_history_valid'][0], past, 0.)
        if self.model.modality:
            from .state_history import controller_proprio
            proprio=controller_proprio(packet,diagnostics,self.limits,'canonical_joint_state_v1')
            if int(packet['task_id'][0])!=0 or bool(packet['goal_valid'][0]):raise ValueError('Unexpected external task command')
        else:proprio = proprio_radians(packet['proprio'][0].cpu(), self.limits)
        robot = robot_rows(proprio[:29], proprio[29:], packet['actuator_target_rad'][0].cpu(), past, stamp-self.started)
        frame = None
        if step % 12 == 0:
            frame = dict(time=packet['camera_time_s'][0].to(self.device), valid=packet['camera_valid'][0].to(self.device))
            if self.model.group == 'S-NativeDP':
                if diagnostics is None:
                    raise ValueError('State baseline requires current allowlisted geometry')
                frame['geometry'] = geometry(diagnostics, self.dimensions)[0].to(self.device)
            elif self.model.group == 'I-HistoryDP':
                frame.update(patches=self.dino(packet['image'][0, :, :3].to(self.device)),
                    depth=depth_grid(packet['image'][0, :, 3:].cpu(), packet['depth_valid'][0].cpu()).to(self.device))
            else:
                frame.update(rgb=packet['image'][0, :, :3].to(self.device),
                    depth=packet['image'][0, :, 3:].to(self.device), depth_valid=packet['depth_valid'][0].to(self.device))
        touch=None
        if self.model.modality=='G10':
            from .native_data import tactile_values
            touch=tactile_values(packet['contact_substeps_w_N'][0].cpu(),packet['contact_substeps_local_N'][0].cpu(),
                packet['contact_time_s'][0].cpu(),stamp,packet['contact_valid'][0].cpu()).to(self.device)
        self.memory = self.model.observe(robot.to(self.device), stamp.to(self.device), step, frame, self.memory,tactile=touch)
        if step % 12 == 0:
            self.last_condition = self.model.condition(self.memory, stamp.to(self.device))
            self.plan = self.model.predict(self.last_condition, self.generator).cpu()
        if self.position_mode:
            from .position_control import bounded_targets
            self.command=bounded_targets(self.plan[:,step%12],self.limits)
        else:
            self.command = self.plan[:, step % 12].clamp(-1, 1)
        self.last_packet = packet
        self.next_step += 1
        return self.command

    def observe_execution(self, transition):
        if self.position_mode:
            self.position_previous=(self.command[0].clone(),transition['position_applied_rad'][0].cpu().clone(),self.last_packet['time_s'][0].cpu())
        else:
            self.execution.observe(transition, self.command, self.last_packet['episode_id'], self.last_packet['step_id'])
