"""Opt-in event replay over the unchanged research physics wrapper."""
import torch
from isaaclab.utils.math import quat_apply_inverse

from .environment import InstrumentedAssemblyEnv


class RecordedReplayEnv(InstrumentedAssemblyEnv):
    def __init__(self, *args, **kwargs):
        self.recorded_events = None
        self.recorded_event = None
        self._recorded_delay_index = None
        self.physics_samples = None
        super().__init__(*args,**kwargs)
        original_write = self.object.write_data_to_sim
        def write_recorded_wrench():
            injecting = (self.recorded_event is not None and self.recorded_events.mode == 'delay_wrench_fixed'
                         and self._wrench_samples is not None)
            saved_force, saved_torque = self._object_forces, self._object_torques
            if injecting:
                substep = len(self._wrench_samples)
                if self.recorded_events.wrench_reconstruction=='native_request':
                    self._object_forces = self.recorded_event['request_force'].to(self.device)
                    self._object_torques = self.recorded_event['request_torque'].to(self.device)
                else:
                    self._object_forces = self.recorded_event['force'][substep:substep+1].to(self.device)
                    self._object_torques = self.recorded_event['torque'][substep:substep+1].to(self.device)
            try:
                original_write()
            finally:
                self._object_forces, self._object_torques = saved_force, saved_torque
        self.object.write_data_to_sim = write_recorded_wrench
        original_update = self.scene.update
        def update_with_physics_trace(dt):
            original_update(dt)
            if dt > 0 and self.physics_samples is not None:
                self.physics_samples.append(dict(joint_position_rad=self.robot.data.joint_pos[:,self._perm_lab_to_canon].clone(),
                    joint_velocity_rad_s=self.robot.data.joint_vel[:,self._perm_lab_to_canon].clone(),
                    object_pose_w=self.object.data.root_state_w[:,:7].clone(),
                    object_velocity=self.object.data.root_state_w[:,7:13].clone()))
        self.scene.update = update_with_physics_trace

    def set_recorded_step(self, events, step):
        if self.num_envs != 1 or str(self.device) != 'cpu' or self.wrench_frame_mode != 'world_hold':
            raise ValueError('Recorded replay is validated only for single CPU physics lane with world_hold')
        if events is not self.recorded_events and events is not None and events.mode=='delay_wrench_fixed' and events.wrench_reconstruction=='native_request':
            for request,body in ((events.request_force,events.force_body),(events.request_torque,events.torque_body)):
                reconstructed=quat_apply_inverse(events.source_quat,request[:,None].expand_as(body))
                if not torch.equal(reconstructed,body):
                    raise ValueError('Source world_hold request does not exactly reconstruct recorded body buffers')
        self.recorded_events = events
        self.recorded_event = events.at(step) if events is not None else None
        self._recorded_delay_index = (self.recorded_event['delay_index'].to(self.device)
            if events is not None and events.mode != 'original' else None)

    def clear_recorded_events(self):
        self.recorded_events = self.recorded_event = self._recorded_delay_index = None
        self.physics_samples = None

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self.physics_samples = []
        if self.recorded_event is not None and self.recorded_events.mode == 'delay_wrench_fixed':
            self.last_transition['sampled_wrench_before_replay'] = dict(
                force_w_N=self._object_forces.clone(),torque_w_Nm=self._object_torques.clone())
            self._object_forces = self.recorded_event['request_force'].to(self.device).clone()
            self._object_torques = self.recorded_event['request_torque'].to(self.device).clone()
            self.last_transition['replayed_wrench_request'] = dict(
                force_w_N=self._object_forces.clone(),torque_w_Nm=self._object_torques.clone())

    def _get_rewards(self):
        reward = super()._get_rewards()
        self.last_transition['physics_samples'] = {name:torch.stack([row[name] for row in self.physics_samples],1)
            for name in self.physics_samples[0]}
        self.physics_samples = None
        return reward
