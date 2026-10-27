"""Opt-in instrumentation; original task, control and RL classes are unchanged.

The original student path changes PreciseAssembly's policy observation space.
Here that switch stays disabled; a cached, separate sensor packet is exposed.
Only contact reporting APIs are added to existing distal rigid bodies.
"""
from contextlib import contextmanager

import torch
from pxr import PhysxSchema, UsdPhysics
from isaaclab.sensors import Camera, ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply
import isaaclab.sim as sim_utils

from isaacsimenvs.tasks.precise_assembly.precise_assembly_env import PreciseAssemblyEnv
from isaacsimenvs.tasks.play.utils.obs_utils import build_student_observations
from isaacsimenvs.tasks.play.utils.scene_utils import (
    JOINT_NAMES_CANONICAL, setup_student_camera, hide_goal_viz_for_student_camera,
)
from visuotactile.kinematics import EncoderKinematics, rotate_force_to_link
from visuotactile.observations import CAMERA_IDS, TaskCommand, empty_plan_context

DISTAL_BODIES = (
    "left_index_DP", "left_middle_DP", "left_ring_DP",
    "left_thumb_DP", "left_pinky_DP",
)


class InstrumentedAssemblyEnv(PreciseAssemblyEnv):
    """Teacher obs unchanged; research packet cached once per returned observation."""

    def __init__(self, cfg, *, perception_version=2, task_command=None, **kwargs):
        if perception_version not in (1, 2):
            raise ValueError("Unsupported perception version")
        self.perception_version = perception_version
        self.task_command = task_command or (TaskCommand(cfg.precise_assembly.problem) if perception_version == 2 else None)
        if cfg.student_obs.camera_update_period_s != 0:
            raise ValueError("Initial adapter requires camera_update_period_s=0: original RGB-D skip cache is unsafe")
        if tuple(cfg.student_obs.proprio_list) != ("joint_pos","joint_vel","prev_action_targets"):
            raise ValueError("Research 58-D encoder projection requires the audited original proprio field order")
        if (cfg.student_obs.use_camera_delay or cfg.student_obs.use_student_obs_delay
                or cfg.student_obs.use_camera_pose_rand or cfg.student_obs.use_depth_aug):
            raise ValueError("Research sensor augmentation/delay is not yet timestamp/RNG validated; original teacher/action DR remains supported")
        if not cfg.student_obs.hide_goal_viz:
            raise ValueError("Research images must hide privileged GoalViz; use rendered regression for visible original scene")
        # Keep all original delay/noise settings, but prevent student policy rerouting.
        cfg.student_obs.enabled = False
        self.research_packet = None
        self.last_transition = None
        self._research_ready = False
        super().__init__(cfg, **kwargs)
        if list(self._arm_joint_ids) != list(range(7)) or list(self._hand_joint_ids) != list(range(7, 29)):
            raise RuntimeError("Native action pipeline assumes arm-first Lab order; discovered mapping differs")
        self.research_episode = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.research_step = torch.zeros_like(self.research_episode)
        self.research_tick = 0
        self._kinematic_reset_poses = {}
        self._research_ready = True
        self._wrench_samples = None
        # Observe the exact local buffers consumed by this installed Lab's
        # write_data_to_sim. Its deprecated world-wrench setter composes into
        # persistent BODY-frame buffers; requested world force is not enough.
        original_write = self.object.write_data_to_sim
        def write_with_wrench_sample():
            if self._wrench_samples is not None:
                permanent = self.object.permanent_wrench_composer
                instant = self.object.instantaneous_wrench_composer
                force_b = torch.zeros_like(permanent.composed_force_as_torch)
                torque_b = torch.zeros_like(permanent.composed_torque_as_torch)
                for composer in (permanent, instant):
                    if composer.active:
                        force_b += composer.composed_force_as_torch
                        torque_b += composer.composed_torque_as_torch
                quat = self.object.data.root_quat_w[:, None].expand(-1, force_b.shape[1], -1).clone()
                self._wrench_samples.append(dict(force_b_N=force_b.clone(), torque_b_Nm=torque_b.clone(),
                    object_quat_wxyz=quat, force_w_N=quat_apply(quat,force_b), torque_w_Nm=quat_apply(quat,torque_b)))
            original_write()
        self.object.write_data_to_sim = write_with_wrench_sample
        if self.perception_version == 2:
            self._fk = EncoderKinematics(cfg.assets.robot_urdf, self.robot.joint_names)
            self._finger_ids = [self.robot.find_bodies(n)[0][0] for n in DISTAL_BODIES]
            if tuple(self.robot.cfg.init_state.rot) != (1., 0., 0., 0.):
                raise ValueError("Current force calibration requires the audited identity base orientation")
            self._fk_samples = []
            # A read-only observer AFTER the original per-physics-step scene update.
            # No replacement of DirectRLEnv.step, physics or actuator processing.
            original_update = self.scene.update
            def update_with_encoder_sample(dt):
                original_update(dt)
                if dt > 0:
                    self._sample_kinematics()
            self.scene.update = update_with_encoder_sample

    @contextmanager
    def _student_enabled(self):
        previous = self.cfg.student_obs.enabled
        self.cfg.student_obs.enabled = True
        try:
            yield
        finally:
            self.cfg.student_obs.enabled = previous

    def _setup_scene(self):
        super()._setup_scene()
        with self._student_enabled():
            hide_goal_viz_for_student_camera(self)
            setup_student_camera(self)
        self.research_cameras = [self.student_camera]
        if self.perception_version == 2:
            if (self.cfg.student_obs.camera_backend != "standard"
                    or self.cfg.student_obs.crop_enabled
                    or self.cfg.student_obs.image_modality != "rgbd"):
                raise ValueError("Perception v2 currently validates uncropped standard RGB-D only")
            camera_cfg = self.student_camera.cfg.copy()
            camera_cfg.prim_path = "/World/envs/env_.*/ResearchLocalCamera"
            camera_cfg.spawn.horizontal_aperture_offset = 0.
            camera_cfg.spawn.vertical_aperture_offset = 0.
            self.local_camera = Camera(camera_cfg)
            self.scene.sensors["research_local_camera"] = self.local_camera
            self.research_cameras.append(self.local_camera)
        self.research_contacts = []
        self.research_contact_paths = []
        for name in DISTAL_BODIES:
            matches = sim_utils.find_matching_prims(f"/World/envs/env_0/Robot/.*/{name}")
            # Some converters put links directly under Robot.
            if not matches:
                matches = sim_utils.find_matching_prims(f"/World/envs/env_0/Robot/{name}")
            rigid = [p for p in matches if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(rigid) != 1:
                raise RuntimeError(f"Expected one existing rigid body for {name}, found {[str(p.GetPath()) for p in rigid]}")
            expr = str(rigid[0].GetPath()).replace("/env_0/", "/env_.*/")
            for prim in sim_utils.find_matching_prims(expr):
                api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                api.CreateThresholdAttr().Set(0.0)
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=expr, update_period=0.0, history_length=int(self.cfg.decimation),
                track_pose=False, debug_vis=False,
            ))
            self.scene.sensors[f"research_contact_{name}"] = sensor
            self.research_contacts.append(sensor)
            self.research_contact_paths.append(expr)

    def _contact_window(self):
        # Sensor history newest-first -> chronological within the last control step.
        return torch.cat([s.data.net_forces_w_history.flip(1) for s in self.research_contacts], dim=2).clone()

    def configure_camera_views(self, local_eye=(-.7, -.35, 1.15), local_target=(0., -.03, .68)):
        """Static calibrations relative to env origin; never follow a true object pose."""
        views = [((.6, -.8, 1.1), (0., .15, .65)), (local_eye, local_target)]
        for camera, (eye, target) in zip(self.research_cameras, views):
            camera.set_world_poses_from_view(torch.tensor([eye], device=self.device)+self.scene.env_origins,
                                            torch.tensor([target], device=self.device)+self.scene.env_origins)

    def _sample_kinematics(self):
        q = self.robot.data.joint_pos.clone()
        pos, rot = self._fk(q)
        self._fk_samples.append((pos.clone(), rot.clone(), q[:,self._perm_lab_to_canon]))
        self._fk_samples = self._fk_samples[-self.cfg.decimation:]

    def _kinematic_packet(self, force):
        if not self._fk_samples:
            self._sample_kinematics()
        samples = ([self._fk_samples[0]]*self.cfg.decimation + self._fk_samples)[-self.cfg.decimation:]
        pos = torch.stack([s[0] for s in samples], 1)
        rot = torch.stack([s[1] for s in samples], 1)
        joint = torch.stack([s[2] for s in samples],1)
        # After auto-reset replace that env's old interval; contact_valid masks it.
        reset = self.research_step == 0
        current_pos, current_rot = self._fk(self.robot.data.joint_pos)
        pos[reset] = current_pos[reset, None]
        rot[reset] = current_rot[reset, None]
        joint[reset] = self.robot.data.joint_pos[reset][:,self._perm_lab_to_canon][:,None]
        return dict(finger_joint_position_rad=joint,finger_position_base_m=pos, finger_rotation_base=rot,
                    contact_substeps_local_N=rotate_force_to_link(force, rot))

    def _camera_packet(self):
        images, depths, valid = [], [], []
        for camera in self.research_cameras:
            # The original global reader has already rendered this instant.
            # Local read consumes that same render, not another simulation step.
            if camera is not self.student_camera:
                camera.update(self.physics_dt, force_recompute=True)
            output = camera.data.output
            rgb = output["rgb"][..., :3].permute(0, 3, 1, 2).float()/255.
            depth = output["distance_to_image_plane"].permute(0, 3, 1, 2).clone()
            mask = torch.isfinite(depth) & (depth > 0)
            metric = torch.where(mask, depth, torch.zeros_like(depth))
            # Version 2 stores metric depth separately from the original normalized reader.
            images.append(torch.cat((rgb, metric), 1))
            depths.append(metric)
            valid.append(mask)
        return dict(image=torch.stack(images, 1), depth_m=torch.stack(depths, 1),
                    depth_valid=torch.stack(valid, 1),
                    camera_valid=torch.ones(self.num_envs, len(images), dtype=torch.bool, device=self.device),
                    camera_time_s=torch.full((self.num_envs, len(images)), self.research_tick*self.step_dt,
                                             dtype=torch.float64, device=self.device),
                    camera_frame=torch.stack([c.frame.clone() for c in self.research_cameras], 1))

    def _diagnostics(self):
        perm = self._perm_lab_to_canon
        result = {
            "joint_position_rad": self.robot.data.joint_pos[:, perm].clone(),
            "joint_velocity_rad_s": self.robot.data.joint_vel[:, perm].clone(),
            "object_pose_w": self.object.data.root_state_w[:, :7].clone(),
            "goal_pose_w": self.goal_viz.data.root_state_w[:, :7].clone(),
            "hole_pose_w": self.hole.data.root_state_w[:, :7].clone(),
            "table_pose_w": self.table.data.root_state_w[:, :7].clone(),
            "successes": self._successes.clone(),
            "retract_phase": self.retract_phase.clone(),
            "retract_succeeded": self.retract_succeeded.clone(),
            "max_goals": self.env_max_goals.clone(),
            "lifted_object": self._lifted_object.clone(),
            "object_initial_z_m": self._object_init_z.clone(),
            "object_velocity_w_m_s_rad_s": self.object.data.root_state_w[:, 7:13].clone(),
            "goal_velocity_w_m_s_rad_s": self.goal_viz.data.root_state_w[:, 7:13].clone(),
            "base_pose_w": self.robot.data.root_state_w[:, :7].clone(),
            "palm_body_pose_w": self.robot.data.body_state_w[:, self._palm_body_id, :7].clone(),
            "native_stage_step": self.episode_length_buf.clone(),
            "is_random_goal": self.is_random_goal_env.clone(),
        }
        if self.perception_version == 2:
            pos, rot = self._fk(self.robot.data.joint_pos)
            result["finger_position_w"] = self.robot.data.body_pos_w[:, self._finger_ids].clone()
            result["finger_quat_wxyz"] = self.robot.data.body_quat_w[:, self._finger_ids].clone()
            result["fk_position_error_m"] = (pos + torch.tensor(self.robot.cfg.init_state.pos, device=self.device)
                + self.scene.env_origins[:, None] - result["finger_position_w"]).norm(dim=-1)
        return result

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        if self._research_ready:
            self.research_episode[env_ids] += 1
            self.research_step[env_ids] = 0
            for name in ("hole", "table"):
                value = getattr(self, name).data.root_pos_w.clone()
                if name not in self._kinematic_reset_poses:
                    self._kinematic_reset_poses[name] = value
                self._kinematic_reset_poses[name][env_ids] = value[env_ids]
        # No cross-episode image cache reuse. period=0 guarantees a fresh read.
        for name in ("_last_student_image_noisy", "_last_student_image_clean"):
            if hasattr(self, name):
                getattr(self, name)[env_ids] = 0

    def _record_action_trace(self, delayed_lab, delay_index, queue_initialized):
        """Read values chosen by the original pipeline; never resample delay/RNG."""
        n = self.num_envs
        steps = self.research_step
        enabled = delay_index is not None
        if enabled:
            length = self._action_queue.shape[1]
            if not hasattr(self,"_research_command_steps"):
                self._research_command_steps = steps[:,None].expand(-1,length).clone()
            self._research_command_steps[queue_initialized] = steps[queue_initialized,None]
            self._research_command_steps = torch.roll(self._research_command_steps,1,1)
            self._research_command_steps[:,0] = steps
            selected_step = self._research_command_steps[torch.arange(n,device=self.device),delay_index]
        else:
            length = 1
            delay_index = torch.zeros_like(steps)
            selected_step = steps.clone()
        self.last_transition["control_trace"] = dict(
            valid=torch.ones(n,dtype=torch.bool,device=self.device),
            delayed_command=delayed_lab[:,self._perm_lab_to_canon].clone(),
            delay_index=delay_index.clone(), selected_command_step=selected_step.clone(),
            effective_lag_steps=steps-selected_step,
            selected_command_time_s=self.last_transition["time_before_s"]-(steps-selected_step).to(torch.float64)*self.step_dt,
            delay_enabled=torch.full((n,),enabled,dtype=torch.bool,device=self.device),
            queue_length=torch.full_like(steps,length))

    def _pre_physics_step(self, actions):
        if actions.shape != (self.num_envs, 29) or not torch.isfinite(actions).all():
            raise ValueError("Expected finite (num_envs,29) canonical actions")
        self.last_transition = {
            "episode_id": self.research_episode.clone(),
            "step_id": self.research_step.clone(),
            "raw_action": actions.clone(),  # legacy alias of submitted_action
            "submitted_action": actions.clone(),
            "command_clipped_pre_delay": actions.clamp(-1,1).clone(),
            "time_before_s": torch.full((self.num_envs,),self.research_tick*self.step_dt,dtype=torch.float64,device=self.device),
            "diagnostics_before": self._diagnostics(),
            "executed_action": actions.clamp(-1, 1).clone(),
            "target_before_rad": self._prev_targets[:, self._perm_lab_to_canon].clone(),
        }
        self._wrench_samples = []
        super()._pre_physics_step(actions.clamp(-1, 1))
        if "control_trace" not in self.last_transition:
            raise RuntimeError("Replay target bypass has no native command trace; unsupported research capture")
        # These are the values selected by the native wrench randomizer after
        # probability gating, lifted/retract gating and decay.  They are
        # telemetry only: PlayEnv has queued these requests in the object's
        # composer. Actual per-substep submissions are observed separately.
        self.last_transition["external_force_after_dr_w_N"] = self._object_forces.clone()
        self.last_transition["external_torque_after_dr_w_Nm"] = self._object_torques.clone()
        self.last_transition["external_force_nonzero"] = self._object_forces.norm(dim=-1).squeeze(-1) > 0
        self.last_transition["external_torque_nonzero"] = self._object_torques.norm(dim=-1).squeeze(-1) > 0
        self.last_transition["external_wrench_lift_gate"] = self._lifted_object.clone()
        self.last_transition["external_wrench_task_gate"] = (~self.retract_phase).clone()
        self.last_transition["torch_rng_after_preprocess"] = torch.random.get_rng_state()[None].expand(self.num_envs, -1).clone()
        self.last_transition["external_force_probability"] = self._random_force_prob.clone()
        self.last_transition["external_torque_probability"] = self._random_torque_prob.clone()
        self.last_transition["external_object_mass_kg"] = self._object_mass.squeeze(-1).clone()
        self.last_transition["external_force_scale"] = torch.full(
            (self.num_envs,), float(self.cfg.domain_randomization.force_scale),
            device=self.device,
        )
        self.last_transition["external_torque_scale"] = torch.full(
            (self.num_envs,), float(self.cfg.domain_randomization.torque_scale),
            device=self.device,
        )
        self.last_transition["target_after_rad"] = self._cur_targets[:, self._perm_lab_to_canon].clone()
        self.research_step += 1
        self.research_tick += 1

    def _get_rewards(self):
        reward = super()._get_rewards()
        for name, expected in self._kinematic_reset_poses.items():
            if not torch.allclose(getattr(self, name).data.root_pos_w, expected, atol=1e-5, rtol=0):
                raise RuntimeError(f"Kinematic {name} moved after reset; abort invalid CPU/render trajectory")
        # After task dones/reward, before DirectRLEnv's auto-reset. No extra task updates.
        self.last_transition.update({
            "diagnostics_after": self._diagnostics(),
            "contact_substeps_after_w_N": self._contact_window(),
            "reward": reward.clone(),
            "time_after_s": torch.full((self.num_envs,),self.research_tick*self.step_dt,dtype=torch.float64,device=self.device),
            "termination_reasons": {name: value.clone() for name, value in self._termination_reasons.items()},
        })
        if len(self._wrench_samples) != self.cfg.decimation:
            raise RuntimeError("Expected one observed PhysX wrench submission per physics substep")
        self.last_transition["applied_wrench"] = {
            key: torch.stack([sample[key] for sample in self._wrench_samples], 1)
            for key in self._wrench_samples[0]
        }
        self.last_transition["applied_wrench"]["time_s"] = (
            self.last_transition["time_before_s"][:,None]
            + torch.arange(self.cfg.decimation,device=self.device,dtype=torch.float64)*self.physics_dt)
        self._wrench_samples = None
        if self.perception_version == 2:
            self.last_transition["kinematics_after"] = self._kinematic_packet(self.last_transition["contact_substeps_after_w_N"])
            offsets = torch.arange(1-self.cfg.decimation,1,device=self.device,dtype=torch.float64)*self.physics_dt
            self.last_transition["contact_time_after_s"] = self.last_transition["time_after_s"][:,None]+offsets
        return reward

    def _get_observations(self):
        teacher = super()._get_observations()
        with self._student_enabled():
            student = build_student_observations(self)
        # 87-D full student proprio retained only for provenance. Nominal uses
        # encoders (58-D): executed target history may contain tactile correction.
        self.research_packet = {
            "image": student["image"].unsqueeze(1).clone(),  # N,V,C,H,W
            "proprio": student["proprio"][:, :58].clone(),
            "full_proprio_diagnostic": student["proprio"].clone(),
            "contact_w_N": self._contact_window()[:, -1].clone(),
            "contact_substeps_w_N": self._contact_window(),
            "contact_valid": (self.research_step > 0).clone(),
            "episode_id": self.research_episode.clone(),
            "step_id": self.research_step.clone(),
            "time_s": torch.full((self.num_envs,), self.research_tick * self.step_dt, device=self.device),
        }
        if self.perception_version == 2:
            self.research_packet.update(self._camera_packet())
            self.research_packet.update(self._kinematic_packet(self.research_packet["contact_substeps_w_N"]))
            self.research_packet.update(self.task_command.packet(self.num_envs, self.device))
            self.research_packet.update(empty_plan_context(self.num_envs, device=self.device))
            self.research_packet["time_s"] = torch.full((self.num_envs,), self.research_tick*self.step_dt,
                                                       dtype=torch.float64, device=self.device)
            offsets = torch.arange(1-self.cfg.decimation, 1, device=self.device, dtype=torch.float64)*self.physics_dt
            self.research_packet["contact_time_s"] = self.research_packet["time_s"][:, None] + offsets
            target = self._cur_targets[:, self._perm_lab_to_canon].clone()
            self.research_packet["actuator_target_rad"] = target
            self.research_packet["tracking_error_rad"] = target - self.robot.data.joint_pos[:, self._perm_lab_to_canon]
        return teacher

    def research_metadata(self):
        metadata = {
            "wrench_trace_version": 3,
            "wrench_target": "object root rigid body; body-frame composer buffers passed to PhysX, position_data=None",
            "wrench_timestamp": "external_*_after_dr is requested world wrench at time_before_s; applied_wrench captures each 120Hz submission and world rotation using that instant's object pose",
            "trace_version": 2,
            "missing_fields": {"terminal_image": "Auto-reset occurs before observation render; no terminal exposure captured",
                               "raw_expert_before_player_clip": "Player API returns a clipped command; original policy mean is not captured"},
            "joint_order": list(JOINT_NAMES_CANONICAL),
            "joint_order_lab": list(self.robot.joint_names),
            "joint_limits_rad": self.robot.data.joint_pos_limits[:,self._perm_lab_to_canon].tolist(),
            "contact_bodies": list(DISTAL_BODIES),
            "contact_paths": self.research_contact_paths,
            "contact_kind": "aggregate normal force, world axes, Newton; NOT pressure/tactile image",
            "physics_dt": self.physics_dt, "control_dt": self.step_dt,
            "student_config": self.cfg.student_obs.to_dict(),
            "nominal_proprio": "canonical joint position normalized by limits (29), encoder velocity rad/s (29)",
            "task_context": "externally selected problem identity; no goal pose or success-driven stage",
        }
        metadata["perception_version"] = self.perception_version
        if self.perception_version == 2:
            metadata.update(task_command=self.task_command.metadata(), depth_encoding="metric_m_zero_invalid_with_mask",
                            fk_urdf_sha256=self._fk.sha256, fk_root=self._fk.root,
                            base_position_env=list(self.robot.cfg.init_state.pos),
                            base_orientation_wxyz=list(self.robot.cfg.init_state.rot),
                            contact_pose_source="URDF FK from joint encoders at each physics update; no body truth input",
                            camera_ids=list(CAMERA_IDS), cameras=[])
            for identity, camera in zip(CAMERA_IDS, self.research_cameras):
                metadata["cameras"].append(dict(id=identity, resolution=[camera.cfg.height, camera.cfg.width],
                    pos_env=(camera.data.pos_w-self.scene.env_origins).tolist(),
                    quat_ros_wxyz=camera.data.quat_w_ros.tolist(), intrinsics=camera.data.intrinsic_matrices.tolist(),
                    clock="global simulated control time after render", rate_hz=1/self.step_dt))
        return metadata
