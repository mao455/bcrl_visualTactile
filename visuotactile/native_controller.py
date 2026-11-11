"""Direct native commands and exact execution acknowledgments at every tick."""
import time

import torch

from .action_history import ExecutionHistory
from .native_data import depth_grid, proprio_radians
from .native_data import geometry, object_dimensions, robot_rows
from .models.native_dp import load_policy


class InferenceTimings:
    """Observe call wall time and CUDA-stream intervals without adding a wait.

    CUDA events are resolved only after query says they are complete. Existing
    command/plan CPU transfers retain their original synchronization behavior.
    An unresolved event stays missing rather than triggering synchronization.
    """
    def __init__(self, device):
        self.device = torch.device(device)
        self.rows, self.pending = [], []

    def collect(self):
        pending = []
        for start, end, row in self.pending:
            if end.query(): row['cuda_stream_s'] = start.elapsed_time(end)/1000
            else: pending.append((start, end, row))
        self.pending = pending

    def begin(self, stage, tick):
        self.collect()
        start = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        if start is not None: start.record(torch.cuda.current_stream(self.device))
        return stage, tick, time.perf_counter(), start

    def finish(self, token):
        stage, tick, began, start = token
        end = torch.cuda.Event(enable_timing=True) if start is not None else None
        if end is not None: end.record(torch.cuda.current_stream(self.device))
        row = dict(stage=stage, tick=tick, wall_s=time.perf_counter()-began, cuda_stream_s=None)
        self.rows.append(row)
        if end is not None: self.pending.append((start, end, row))

    @staticmethod
    def summarize(rows):
        from .state_motion import distribution
        return {stage: dict(wall_s=distribution([r['wall_s'] for r in rows if r['stage']==stage]),
            cuda_stream_s=distribution([r['cuda_stream_s'] for r in rows if r['stage']==stage and r['cuda_stream_s'] is not None]))
            for stage in sorted({r['stage'] for r in rows})}

    def report(self):
        self.collect()
        return dict(rows=self.rows, summary=self.summarize(self.rows), pending_cuda_events=len(self.pending),
            timing_scope='Observed call wall times and asynchronous current-CUDA-stream elapsed intervals; no added synchronize/wait; stream intervals include CPU submission gaps, not pure kernel time',
            wall_scope='Native planning includes observation, DINO/depth, history, DDIM and required CPU return; native history-only calls may return before queued CUDA work completes',
            excludes='resource waits, environment stepping/rendering, result serialization', control_dt_changed=False)


class NativeController:
    def __init__(self, saved, metadata, dino=None, device='cuda:0'):
        from .state_history import trained_continuation_updates
        if trained_continuation_updates(saved) < 1:
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
        self.timings = InferenceTimings(self.device)

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
            self.timings = InferenceTimings(self.device)
        if step != self.next_step:
            raise ValueError('Native tick skipped or duplicated')
        measured = self.timings.begin('native_planning' if step%12==0 else 'native_history',step)
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
            diffusion_measured = self.timings.begin('ddim_plan',step)
            self.plan = self.model.predict(self.last_condition, self.generator).cpu()
            self.timings.finish(diffusion_measured)
        if self.position_mode:
            from .position_control import bounded_targets
            self.command=bounded_targets(self.plan[:,step%12],self.limits)
        else:
            self.command = self.plan[:, step % 12].clamp(-1, 1)
        self.last_packet = packet
        self.next_step += 1
        self.timings.finish(measured)
        return self.command

    def inference_metrics(self):
        return self.timings.report()

    def observe_execution(self, transition):
        if self.position_mode:
            self.position_previous=(self.command[0].clone(),transition['position_applied_rad'][0].cpu().clone(),self.last_packet['time_s'][0].cpu())
        else:
            self.execution.observe(transition, self.command, self.last_packet['episode_id'], self.last_packet['step_id'])


class ResidualController:
    """One frozen DP tick followed by one independent 60Hz residual tick."""
    def __init__(self,saved,metadata,dino=None,device='cuda:0'):
        from .compact import sha256
        from .models.native_dp import TargetResidual, LegacyTargetResidual, load_residual
        from .state_history import trained_continuation_updates
        if saved.get('residual_version') not in (TargetResidual.version, LegacyTargetResidual.version) or trained_continuation_updates(saved)<1:
            raise ValueError('Untrained or incompatible target residual')
        config=saved['configuration']
        if sha256(config['base_checkpoint'])!=config['base_sha256']:raise ValueError('Frozen base identity changed')
        base=torch.load(config['base_checkpoint'],map_location='cpu',weights_only=False)
        if base['contract']!=saved['contract']:raise ValueError('Residual/base contract differs')
        self.base=NativeController(base,metadata,dino,device)
        self.model=self.base.model.eval().requires_grad_(False)
        self.residual=load_residual(saved,device).requires_grad_(False)
        normalizers=('robot_mean','robot_std','action_scale','action_offset')
        if self.residual.version==TargetResidual.version:
            from pathlib import Path
            path=Path(config.get('residual_statistics_path',''))
            if config.get('residual_normalization_identity')!='u_expert_train166_affine_v1' or not path.is_absolute():
                raise ValueError('New residual requires an explicit common u_expert affine')
            if sha256(path)!=config.get('residual_statistics_sha256'):
                raise ValueError('Residual statistics identity changed')
            statistics=torch.load(path,map_location='cpu',weights_only=False)
            for k in normalizers:
                if not torch.equal(getattr(self.residual,k).detach().cpu(),statistics[k]):
                    raise ValueError('Residual buffer differs from common command statistics')
            # Q nominal keeps its independent reached-configuration affine;
            # every new residual uses command coordinates for plans, deltas,
            # and final u_expert loss. Physical robot observations are shared.
            normalizers=('robot_mean','robot_std')
        for k in normalizers:
            if not torch.equal(getattr(self.model,k),getattr(self.residual,k)):raise ValueError('Residual normalizer differs')
        self.device=self.base.device;self.state=None;self.episode=None
        self.previous_nominal=None
        self.timings=InferenceTimings(self.device)
        self.action_scale_cpu=self.residual.action_scale.detach().cpu()
        self.residual_scales_cpu=self.residual.scales.detach().cpu()

    @torch.no_grad()
    def act(self,packet,diagnostics=None):
        from .native_data import residual_visual,remaining_target_plan,residual_batch
        from .position_control import bounded_targets
        episode=int(packet['episode_id'][0]);tick=int(packet['step_id'][0])
        if self.episode!=episode:
            self.state=None;self.episode=episode;self.previous_nominal=None
        if bool(packet['contact_valid'][0]) and (packet['contact_time_s'][0]>packet['time_s'][0]+1e-6).any():
            raise ValueError('Future tactile sample cannot enter residual feedback')
        if tick==0:self.timings=InferenceTimings(self.device)
        measured=self.timings.begin('combined_controller',tick)
        nominal=self.base.act(packet,diagnostics)
        if tick%12==0:self.visual=residual_visual(self.model,self.base.memory,packet['time_s'][0].to(self.device)).cpu()
        plan=bounded_targets(self.base.plan[0],self.base.limits)[None]
        remaining,valid=remaining_target_plan(plan,tick%12)
        touch={target:packet[source][0].cpu() for source,target in (
            ('contact_substeps_w_N','force_w_N'),('contact_substeps_local_N','force_local_N'),
            ('finger_position_base_m','position_base_m'),('finger_rotation_base','rotation_base'),('contact_time_s','time_s'))}
        record=dict(visual=self.visual,robot=self.base.last_robot[None],plans=remaining,plan_valid=valid[None],
            tactile=touch,tactile_valid=packet['contact_valid'][0].cpu().bool().repeat(2))
        from .models.native_dp import TargetResidual
        if self.residual.version==TargetResidual.version:
            record.update(plan_cursor=torch.tensor([tick%12]),
                previous_nominal=torch.zeros_like(nominal) if self.previous_nominal is None else self.previous_nominal,
                previous_nominal_valid=torch.tensor([self.previous_nominal is not None]))
        inputs=residual_batch([record],0,1,self.device)
        residual_measured=self.timings.begin('residual_forward_compose',tick)
        raw,self.state=self.residual(inputs,self.state)
        gate=packet['contact_valid'][0].bool()&(packet['contact_w_N'][0].norm(dim=-1).amax()>.1)
        command,delta=self.residual.compose(nominal.to(self.device),raw[:,0],gate.reshape(1).to(self.device),self.base.limits.to(self.device))
        self.nominal=nominal;self.delta=delta.cpu();self.contact_gate=bool(gate)
        # This is the base prediction issued at t, separate from the mixed
        # submitted command that the next acknowledgement puts in robot178.
        self.previous_nominal=nominal.detach().cpu().clone()
        self.command=command.cpu();self.plan=self.base.plan
        self.timings.finish(residual_measured)
        proposed_delta=self.delta/self.action_scale_cpu
        admitted_delta=proposed_delta if self.residual.version==TargetResidual.version or bool(gate) else torch.zeros_like(proposed_delta)
        unbounded=nominal+admitted_delta
        self.residual_diagnostics=dict(proposed_rad=proposed_delta,applied_rad=self.command-nominal,
            clip_mask=(unbounded<self.base.limits[:,0])|(unbounded>self.base.limits[:,1]),
            tanh_fraction=self.delta.abs()/self.residual_scales_cpu,
            contact_valid=packet['contact_valid'][0].cpu().bool())
        # Actual submitted mixed targets, never private base targets, become
        # strict-past history after the real environment acknowledgement.
        self.base.command=self.command
        self.timings.finish(measured)
        return self.command

    def observe_execution(self,transition):self.base.observe_execution(transition)

    def inference_metrics(self):
        base=self.base.inference_metrics();residual=self.timings.report()
        rows=base['rows']+residual['rows']
        return dict(base,rows=rows,summary=InferenceTimings.summarize(rows),
            pending_cuda_events=base['pending_cuda_events']+residual['pending_cuda_events'],
            residual_scope='residual_forward_compose includes Transformer/recurrence/composition and original CPU result copies; combined_controller additionally includes nominal and residual input preparation')
