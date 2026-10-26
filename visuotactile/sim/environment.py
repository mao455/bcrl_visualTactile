"""Opt-in instrumentation; original task, control and RL classes are unchanged.

The original student path changes PreciseAssembly's policy observation space.
Here that switch stays disabled; a cached, separate sensor packet is exposed.
Only contact reporting APIs are added to existing distal rigid bodies.
"""
from contextlib import contextmanager

import torch
from pxr import PhysxSchema, UsdPhysics
from isaaclab.sensors import ContactSensor, ContactSensorCfg
import isaaclab.sim as sim_utils

from isaacsimenvs.tasks.precise_assembly.precise_assembly_env import PreciseAssemblyEnv
from isaacsimenvs.tasks.play.utils.obs_utils import build_student_observations
from isaacsimenvs.tasks.play.utils.scene_utils import (
    JOINT_NAMES_CANONICAL, setup_student_camera, hide_goal_viz_for_student_camera,
)

DISTAL_BODIES = (
    "left_index_DP", "left_middle_DP", "left_ring_DP",
    "left_thumb_DP", "left_pinky_DP",
)


class InstrumentedAssemblyEnv(PreciseAssemblyEnv):
    """Teacher obs unchanged; research packet cached once per returned observation."""

    def __init__(self, cfg, **kwargs):
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

    def _diagnostics(self):
        perm = self._perm_lab_to_canon
        return {
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
        }

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

    def _pre_physics_step(self, actions):
        if actions.shape != (self.num_envs, 29) or not torch.isfinite(actions).all():
            raise ValueError("Expected finite (num_envs,29) canonical actions")
        self.last_transition = {
            "episode_id": self.research_episode.clone(),
            "step_id": self.research_step.clone(),
            "raw_action": actions.clone(),
            "executed_action": actions.clamp(-1, 1).clone(),
            "target_before_rad": self._prev_targets[:, self._perm_lab_to_canon].clone(),
        }
        super()._pre_physics_step(actions.clamp(-1, 1))
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
        })
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
        return teacher

    def research_metadata(self):
        return {
            "joint_order": list(JOINT_NAMES_CANONICAL),
            "contact_bodies": list(DISTAL_BODIES),
            "contact_paths": self.research_contact_paths,
            "contact_kind": "aggregate normal force, world axes, Newton; NOT pressure/tactile image",
            "physics_dt": self.physics_dt, "control_dt": self.step_dt,
            "student_config": self.cfg.student_obs.to_dict(),
            "nominal_proprio": "canonical joint position normalized by limits (29), encoder velocity rad/s (29)",
            "task_context": "externally selected problem identity; no goal pose or success-driven stage",
        }
