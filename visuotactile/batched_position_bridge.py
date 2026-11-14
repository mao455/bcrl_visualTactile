"""Independent existing expert converters for one wave of position episodes.

This adapter does not batch or reimplement native action mathematics.  The
position executor still makes its one existing N-lane delay draw, which is
partitioned between unchanged ``ExpertPositionBridge`` instances.  A completed
lane is never reused in this wave and supplies no further training labels.
"""

import torch

from .position_control import ExpertPositionBridge, POSITION_ACTION


class _LaneEnvironment:
    """Read-through lane views, including targets replaced by the executor."""

    _lane_fields = frozenset(('_prev_targets', '_cur_targets', '_action_queue',
        'episode_length_buf', '_successes'))
    _bounds = frozenset(('_arm_lower', '_arm_upper', '_hand_lower', '_hand_upper'))

    def __init__(self, env, lane):
        self._env, self._lane = env, lane
        self.num_envs = 1

    def __getattr__(self, name):
        env = object.__getattribute__(self, '_env')
        value = getattr(env, name)
        if name in self._lane_fields:
            return value[self._lane:self._lane+1]
        if name in self._bounds and value.ndim > 1:
            if value.shape[0] == self._env.num_envs:
                return value[self._lane:self._lane+1]
            if value.shape[0] != 1:
                raise ValueError('Joint bounds must be shared or indexed by lane')
        return value


class BatchedExpertPositionBridge:
    """Lane-isolated expert queues with the original position-executor API.

    ``tick`` is the common wave clock; each still-active lane has exactly that
    many observations. ``active`` may only change from true to false. Once a
    lane finishes after a select, pass false at the next submit. Its converter,
    queue, phase and clock then remain frozen while its actuator holds the real
    current native target (including any native automatic-reset target).

    Recorder-facing traces retain all serial fields and add ``bridge_active``.
    Inactive trace rows retain their last diagnostics, but are explicitly not
    clean or valid expert labels. The collector must stop recording that lane
    at its actual terminal transition. No dynamic reset/reuse is supported.
    """

    version = 'batched_expert_takeover_position_wave_v1'
    speed_scheme = ExpertPositionBridge.speed_scheme

    def __init__(self, env, takeover_ticks, speed, *, pipeline=None):
        if (isinstance(env.num_envs, bool) or not isinstance(env.num_envs, int)
                or env.num_envs < 1 or env.action_control_mode != POSITION_ACTION):
            raise ValueError('A positive wave of position-controlled episodes is required')
        ticks = list(takeover_ticks)
        if len(ticks) != env.num_envs:
            raise ValueError('One preregistered takeover tick per lane is required')
        self.env, self.num_envs = env, env.num_envs
        self.lane_envs = [_LaneEnvironment(env, lane) for lane in range(self.num_envs)]
        self.bridges = [ExpertPositionBridge(lane_env, tick, speed, pipeline=pipeline)
            for lane_env, tick in zip(self.lane_envs, ticks)]
        self.tick, self.pending = 0, None
        self.active = torch.ones(self.num_envs, dtype=torch.bool, device=env.device)
        self.requested = env._prev_targets[:, env._perm_lab_to_canon].clone()
        self.trace = self._combined_trace(torch.zeros_like(self.active))

    def _initial_trace(self, lane):
        bridge, env = self.bridges[lane], self.lane_envs[lane]
        previous = env._prev_targets[:, env._perm_lab_to_canon].clone()
        zero = torch.zeros_like(previous)
        scalar = lambda value, dtype: torch.tensor([value], dtype=dtype, device=env.device)
        return dict(expert_raw_action=zero.clone(), expert_clipped_action=zero.clone(),
            expert_delayed_action=zero.clone(), expert_source_step=scalar(-1, torch.long),
            expert_delay_index=scalar(-1, torch.long),
            expert_delay_enabled=scalar(bridge.delay_enabled, torch.bool),
            expert_queue_length=scalar(bridge.queue_length, torch.long),
            expert_source_is_expert_issued=scalar(False, torch.bool),
            control_source=scalar(-1, torch.long), segment_id=scalar(-1, torch.long),
            switch_clean=scalar(False, torch.bool), expert_label_valid=scalar(False, torch.bool),
            expert_native_target_rad=previous.clone(), expert_progressed_target_rad=previous.clone(),
            expert_mapped_target_rad=previous.clone(), student_proposed_target_rad=previous.clone(),
            previous_issued=previous.clone(), previous_issued_valid=scalar(False, torch.bool),
            takeover_tick=scalar(bridge.takeover_tick, torch.long))

    def _combined_trace(self, active):
        rows = [bridge.trace if bridge.trace is not None else self._initial_trace(lane)
            for lane, bridge in enumerate(self.bridges)]
        result = {key: torch.cat([row[key] for row in rows], dim=0) for key in rows[0]}
        result['bridge_active'] = active.clone()
        result['expert_label_valid'] &= active
        result['switch_clean'] &= active
        return result

    def schedule_takeover(self, lane, tick):
        if (isinstance(lane, bool) or not isinstance(lane, int) or not 0 <= lane < self.num_envs
                or self.pending is not None or not bool(self.active[lane])):
            raise ValueError('Takeover requires an active lane before the next submit')
        self.bridges[lane].schedule_takeover(tick)

    def submit(self, raw_action, clipped_action, tick, active):
        if (isinstance(tick, bool) or not isinstance(tick, int)
                or tick != self.tick or self.pending is not None):
            raise ValueError('Expert wave must advance exactly once per real control tick')
        active = torch.as_tensor(active, device=self.env.device)
        if active.shape != (self.num_envs,) or active.dtype != torch.bool:
            raise ValueError('A boolean active mask per lane is required')
        if (active & ~self.active).any():
            raise ValueError('Completed lanes cannot reactivate within a wave')
        if raw_action.shape != (self.num_envs, 29) or clipped_action.shape != (self.num_envs, 29):
            raise ValueError('Expert wave requires canonical [N,29] native commands')
        lanes = active.nonzero(as_tuple=False).flatten().tolist()
        # Validate the whole wave before mutating any serial pending state.
        for lane in lanes:
            raw, clipped = raw_action[lane:lane+1], clipped_action[lane:lane+1]
            bridge = self.bridges[lane]
            if not torch.isfinite(raw).all() or not torch.equal(raw.clamp(-1, 1), clipped):
                raise ValueError('Expert raw/clipped action provenance mismatch')
            if bridge.tick != tick or bridge.pending is not None:
                raise ValueError('Active lane clock or pending expert shadow is inconsistent')
        for lane in lanes:
            self.bridges[lane].submit(raw_action[lane:lane+1], clipped_action[lane:lane+1], tick)
        self.active = active.clone()
        self.pending = lanes

    def select(self, env, student_targets, event_draw):
        if self.pending is None or env is not self.env or env.action_control_mode != POSITION_ACTION:
            raise ValueError('Missing expert wave or changed position executor')
        if student_targets.shape != (self.num_envs, 29):
            raise ValueError('Student wave requires canonical [N,29] position targets')
        if event_draw.shape != (self.num_envs,) or event_draw.dtype != torch.long:
            raise ValueError('One native delay draw per lane is required')
        if event_draw.device != env._prev_targets.device:
            raise ValueError('Native delay draws must stay on the environment device')
        if not torch.isfinite(student_targets[self.pending]).all():
            raise ValueError('Active student targets must be finite')
        for lane in self.pending:
            bridge = self.bridges[lane]
            if bridge.delay_enabled and not 0 <= int(event_draw[lane]) < bridge.queue_length:
                raise ValueError('Expert source delay index outside the recorded queue')
        requested = env._prev_targets[:, env._perm_lab_to_canon].clone()
        for lane in self.pending:
            requested[lane:lane+1] = self.bridges[lane].select(self.lane_envs[lane],
                student_targets[lane:lane+1], event_draw[lane:lane+1])
        self.trace = self._combined_trace(self.active)
        self.requested = requested.clone()
        self.pending = None
        self.tick += 1
        return requested
