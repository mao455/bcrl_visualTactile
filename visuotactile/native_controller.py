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
        self.last_robot = robot
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


class ResidualController:
    """One frozen DP tick followed by one independent 60Hz residual tick."""
    def __init__(self,saved,metadata,dino=None,device='cuda:0'):
        from .compact import sha256
        from .models.native_dp import TargetResidual
        if saved.get('residual_version')!=TargetResidual.version or saved.get('updates',0)<1:
            raise ValueError('Untrained or incompatible target residual')
        config=saved['configuration']
        if sha256(config['base_checkpoint'])!=config['base_sha256']:raise ValueError('Frozen base identity changed')
        base=torch.load(config['base_checkpoint'],map_location='cpu',weights_only=False)
        if base['contract']!=saved['contract']:raise ValueError('Residual/base contract differs')
        self.base=NativeController(base,metadata,dino,device)
        self.model=self.base.model.eval().requires_grad_(False)
        self.residual=TargetResidual(saved['model'],saved['model']['scales']).to(device)
        self.residual.load_state_dict(saved['model']);self.residual.eval().requires_grad_(False)
        for k in ('robot_mean','robot_std','action_scale','action_offset'):
            if not torch.equal(getattr(self.model,k),getattr(self.residual,k)):raise ValueError('Residual normalizer differs')
        self.device=self.base.device;self.state=None;self.episode=None

    @torch.no_grad()
    def act(self,packet,diagnostics=None):
        from .native_data import residual_visual,remaining_target_plan,residual_batch
        from .position_control import bounded_targets
        episode=int(packet['episode_id'][0]);tick=int(packet['step_id'][0])
        if self.episode!=episode:self.state=None;self.episode=episode
        nominal=self.base.act(packet,diagnostics)
        if tick%12==0:self.visual=residual_visual(self.model,self.base.memory,packet['time_s'][0].to(self.device)).cpu()
        plan=bounded_targets(self.base.plan[0],self.base.limits)[None]
        remaining,valid=remaining_target_plan(plan,tick%12)
        touch={target:packet[source][0].cpu() for source,target in (
            ('contact_substeps_w_N','force_w_N'),('contact_substeps_local_N','force_local_N'),
            ('finger_position_base_m','position_base_m'),('finger_rotation_base','rotation_base'),('contact_time_s','time_s'))}
        record=dict(visual=self.visual,robot=self.base.last_robot[None],plans=remaining,plan_valid=valid[None],
            tactile=touch,tactile_valid=packet['contact_valid'][0].cpu().bool().repeat(2))
        inputs=residual_batch([record],0,1,self.device)
        raw,self.state=self.residual(inputs,self.state)
        gate=packet['contact_valid'][0].bool()&(packet['contact_w_N'][0].norm(dim=-1).amax()>.1)
        command,delta=self.residual.compose(nominal.to(self.device),raw[:,0],gate.reshape(1).to(self.device),self.base.limits.to(self.device))
        self.nominal=nominal;self.delta=delta.cpu();self.contact_gate=bool(gate)
        self.command=command.cpu();self.plan=self.base.plan
        # Actual submitted mixed targets, never private base targets, become
        # strict-past history after the real environment acknowledgement.
        self.base.command=self.command
        return self.command

    def observe_execution(self,transition):self.base.observe_execution(transition)
