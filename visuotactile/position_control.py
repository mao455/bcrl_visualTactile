"""Absolute canonical actuator targets: recorded labels and explicit control contract."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .compact import sha256
from .native_data import limits_affine
from .training_resources import atomic_json

POSITION_ACTION = 'absolute_joint_position_target_rad'


def recorded_target_plan(targets, tick):
    """Diagnostic H24 oracle; only commit12 before recorded sequence exhaustion."""
    targets=torch.as_tensor(targets)
    if targets.ndim!=2 or targets.shape[1]!=29 or tick%12 or not 0<=tick<len(targets):
        raise ValueError('Recorded canonical targets and a valid12tick anchor required')
    index=torch.arange(tick,tick+24,device=targets.device).clamp_max(len(targets)-1)
    plan=targets[index]
    if not torch.isfinite(plan).all():raise ValueError('Nonfinite oracle targets')
    return plan.unsqueeze(0)


def overlap_transition(previous, current, elapsed, steps=4):
    """Align targets by native ticks; linearly hand over to the new plan at K.

    Uses only already generated future targets. No shifted cursor or extra delay.
    K=4 weights for new targets are 1/4, 1/2, 3/4, 1. Old/default K=0 is exact.
    """
    if steps not in (0, 4):
        raise ValueError('Only preregistered K=0/4 supported')
    if previous is None or steps == 0:
        return current
    if elapsed != 12 or previous.shape != current.shape or elapsed+steps > current.shape[1]:
        raise ValueError('Plan overlap must align at the next native tick12 anchor')
    output = current.clone()
    weight = torch.arange(1, steps+1, device=current.device, dtype=current.dtype)[None,:,None]/steps
    output[:,:steps] = previous[:,elapsed:elapsed+steps]*(1-weight)+current[:,:steps]*weight
    return output


def contract(horizon=32, position=True, prediction_type='sample'):
    from .state_history import STATE_CONTRACT
    if horizon not in (24, 32):
        raise ValueError('Short-horizon study requires H24 or H32')
    if prediction_type not in ('sample', 'epsilon') or (prediction_type=='epsilon' and (horizon!=24 or not position)):
        raise ValueError('Epsilon comparison is restricted to position H24')
    return dict(STATE_CONTRACT, version=3, horizon=horizon, prediction_type=prediction_type,
        action=POSITION_ACTION if position else STATE_CONTRACT['action'],
        action_contract='position_rad_v1' if position else 'native_command_h32_control_v1',
        history_source=('past_requested_and_applied_position_rad; teacher_forced_in_training'
            if position else STATE_CONTRACT['history_source']))


def image_contract(group, horizon=32, modality=None, supervision=None):
    from .native_data import CONTRACT
    if group not in ('I-OfficialDP','I-HistoryDP') or horizon not in (24,32):
        raise ValueError('Explicit short-horizon ImageDP group required')
    value = dict(CONTRACT,version=3,horizon=horizon,action=POSITION_ACTION,
        action_contract='position_rad_v1',observation=group+'_robot178_rgbd_no_privileged_v1',
        history_source='past_requested_and_applied_position_rad; teacher_forced_in_training')
    if modality is not None:
        if group != 'I-HistoryDP' or horizon != 24 or modality not in ('G00', 'G10'):
            raise ValueError('Explicit H24 visual pair required')
        value.update(version=4, modality=modality, input_builder='canonical_joint_state_v1',
            observation='robot178_dino_rgb_depth_v1'+('_touch32x242' if modality=='G10' else ''),
            tactile_history=242 if modality=='G10' else 0, sampler_seed=7091,
            task_condition='constant externally selected tight_insertion; goal pose absent')
    if supervision is not None:
        if modality is None or supervision not in ('u_expert', 'q_reached'):
            raise ValueError('Explicit visual clean-sample supervision required')
        value.update(version=5, supervision=supervision,
            clean_sample_field=('transition/target_after_rad' if supervision=='u_expert'
                else 'transition/diagnostics_after/joint_position_rad'),
            normalization_identity=supervision+'_train166_affine_v1',
            label_time='same transition: issued target before integration / reached configuration after integration',
            deployment_output='absolute_joint_position_target_rad')
    return value


def bounded_targets(actions, limits):
    if actions.ndim != 2 or actions.shape[-1] != 29 or not torch.isfinite(actions).all():
        raise ValueError('Finite canonical [N,29] absolute radians required')
    if limits.shape[-2:] != (29, 2) or not torch.isfinite(limits).all() or (limits[..., 0] >= limits[..., 1]).any():
        raise ValueError('Invalid actual actuator limits')
    return torch.maximum(torch.minimum(actions, limits[..., 1]), limits[..., 0])


def apply_position_targets(env, actions):
    """Direct targets; only the explicit DAgger bridge converts expert proposals."""
    limits = env.robot.data.joint_pos_limits[:, env._perm_lab_to_canon]
    # Retain the UNUSED native delay RNG draw to align subsequent observation
    # noise/wrench events across the command/position comparison. No delay applied.
    dr = env.cfg.domain_randomization
    event_draw = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    if dr.use_action_delay and dr.action_delay_max > 0:
        event_draw = torch.randint(0, env._action_queue.shape[1], (env.num_envs,), device=env.device)
    bridge = getattr(env, '_dagger_position_bridge', None)
    if bridge is not None:
        actions = bridge.select(env, actions, event_draw)
    clipped = bounded_targets(actions.to(env.device), limits)
    env._cur_targets[:] = clipped[:, env._perm_canon_to_lab]
    env._prev_targets = env._cur_targets.clone()
    return clipped, event_draw


class ExpertPositionBridge:
    """Opt-in shadow expert conversion, followed by the unchanged position actuator.

    The expert queue advances on every real observation, including student ticks.
    Conversion reuses native math on an isolated proxy and the position executor's
    one existing random delay draw. It neither resumes nor mutates the native
    command queue. Every conversion starts from the last *actual* issued target.
    """
    version = 'expert_takeover_position_v1'
    speed_scheme = dict(kind='closed_loop', arm=.5, hand=1., phase='all', hand_target_cap_rad_s=2.)

    def __init__(self, env, takeover_tick, speed, *, pipeline=None):
        from types import SimpleNamespace
        if isinstance(takeover_tick, bool) or not isinstance(takeover_tick, int) or takeover_tick < 1:
            raise ValueError('A preregistered positive student prefix is required')
        if dict(speed) != self.speed_scheme:
            raise ValueError('DAgger requires the existing validated expert speed scheme')
        if env.num_envs != 1 or env.action_control_mode != POSITION_ACTION:
            raise ValueError('DAgger bridge requires one continuous position-controlled episode')
        if pipeline is None:
            from isaacsimenvs.tasks.play.utils.action_utils import apply_action_pipeline
            pipeline = apply_action_pipeline
        self.pipeline = pipeline
        self.takeover_tick, self.speed = takeover_tick, dict(speed)
        dr = env.cfg.domain_randomization
        self.delay_enabled = bool(dr.use_action_delay and dr.action_delay_max > 0)
        self.queue_length = env._action_queue.shape[1] if self.delay_enabled else 1
        self.queue = self.source_steps = None
        self.tick, self.pending, self.trace = 0, None, None
        self.segment, self.previous_phase = -1, None
        # Disable only the proxy delay: the already drawn index selects our
        # continuously maintained queue before calling the exact native mapper.
        names = ('device', 'num_envs', 'step_dt', '_perm_canon_to_lab',
                 '_arm_joint_ids', '_hand_joint_ids', '_arm_lower', '_arm_upper', '_hand_lower', '_hand_upper')
        self.proxy = SimpleNamespace(**{key: getattr(env, key) for key in names},
            cfg=SimpleNamespace(action=env.cfg.action,
                domain_randomization=SimpleNamespace(use_action_delay=False, action_delay_max=0)))

    def submit(self, raw_action, clipped_action, tick):
        if tick != self.tick or self.pending is not None:
            raise ValueError('Expert shadow must advance exactly once per real control tick')
        if raw_action.shape != (1, 29) or clipped_action.shape != (1, 29):
            raise ValueError('Expert shadow requires canonical 29D native commands')
        if not torch.isfinite(raw_action).all() or not torch.equal(raw_action.clamp(-1, 1), clipped_action):
            raise ValueError('Expert raw/clipped action provenance mismatch')
        self.pending = raw_action.detach().clone(), clipped_action.detach().clone()

    def select(self, env, student_target, event_draw):
        from .expert_speed import progress_target, cap_hand_target
        if self.pending is None or env.action_control_mode != POSITION_ACTION:
            raise ValueError('Missing expert shadow or changed position executor')
        raw, clipped = (value.to(env.device) for value in self.pending)
        if self.queue is None:
            self.queue = clipped[:, None].expand(-1, self.queue_length, -1).clone()
            self.source_steps = torch.zeros((1, self.queue_length), device=env.device, dtype=torch.long)
        self.queue = torch.roll(self.queue, 1, 1); self.queue[:, 0] = clipped
        self.source_steps = torch.roll(self.source_steps, 1, 1); self.source_steps[:, 0] = self.tick
        index = event_draw if self.delay_enabled else torch.zeros_like(event_draw)
        if (index < 0).any() or (index >= self.queue_length).any():
            raise ValueError('Expert source delay index outside the recorded queue')
        delayed = self.queue[torch.arange(1, device=env.device), index]
        selected_step = self.source_steps[torch.arange(1, device=env.device), index]
        previous = env._prev_targets[:, env._perm_lab_to_canon].clone()
        self.proxy._prev_targets = env._prev_targets.clone()
        self.proxy._cur_targets = env._cur_targets.clone()
        self.pipeline(self.proxy, delayed)
        native = self.proxy._cur_targets[:, env._perm_lab_to_canon].clone()
        progressed = progress_target(previous, native, self.speed['arm'], self.speed['hand'])
        mapped = cap_hand_target(previous, progressed, self.speed['hand_target_cap_rad_s'], env.step_dt)
        expert = self.tick >= self.takeover_tick
        # Flush every pre-takeover suggestion, not merely whichever slot this
        # random draw selects. The first clean label starts a new H24 segment.
        clean = expert and self.tick >= self.takeover_tick + self.queue_length - 1
        source_is_expert = selected_step >= self.takeover_tick
        valid = clean and bool(source_is_expert.all())
        phase = (expert, valid)
        if phase != self.previous_phase:
            self.segment += 1; self.previous_phase = phase
        requested = mapped if expert else student_target.to(env.device)
        scalar = lambda value, dtype: torch.tensor([value], dtype=dtype, device=env.device)
        self.trace = dict(expert_raw_action=raw, expert_clipped_action=clipped,
            expert_delayed_action=delayed.clone(), expert_source_step=selected_step.clone(),
            expert_delay_index=index.clone(), expert_delay_enabled=scalar(self.delay_enabled, torch.bool),
            expert_queue_length=scalar(self.queue_length, torch.long),
            expert_source_is_expert_issued=source_is_expert.clone(),
            control_source=scalar(int(expert), torch.long), segment_id=scalar(self.segment, torch.long),
            switch_clean=scalar(clean, torch.bool),
            expert_label_valid=scalar(valid, torch.bool),
            expert_native_target_rad=native, expert_progressed_target_rad=progressed,
            expert_mapped_target_rad=mapped, student_proposed_target_rad=student_target.to(env.device).clone(),
            previous_issued=previous, previous_issued_valid=scalar(self.tick > 0, torch.bool),
            takeover_tick=scalar(self.takeover_tick, torch.long))
        self.requested = requested.clone()
        self.pending = None; self.tick += 1
        return requested


def position_robot_history(record, targets):
    """Reuse robot178 slots with declared radian semantics, strictly previous tick."""
    robot = record['robot'].clone()
    robot[:, 116:177] = 0.
    if len(targets) > 1:
        robot[1:, 116:145] = targets[:-1]
        robot[1:, 145:174] = targets[:-1]
        age = (record['time'][1:]-record['time'][:-1]).float()
        robot[1:, 174:176] = age[:, None]
        robot[1:, 176] = 1.
    return robot


def audit(source, output):
    """Read every original target, clock, limit and camera alignment; no reconstruction."""
    data = json.loads((source/'data.json').read_text())
    rows = []; training_targets = []; limits_ref = None; order_ref = None
    for identity, entry in data['entries'].items():
        with h5py.File(entry['path'], 'r') as f:
            g = f['episodes'][entry['episode']]; meta = json.loads(f.attrs['metadata_json'])
            target = g['transition/target_after_rad'][:]; before = g['transition/target_before_rad'][:]
            obs = g['obs/actuator_target_rad'][:]; time = g['obs/time_s'][:]
            limits = np.asarray(meta['joint_limits_rad'])[0]
            n = entry['length']
            assert target.shape == before.shape == obs.shape == (n,29), identity
            assert np.isfinite(target).all() and np.isfinite(before).all(), identity
            assert np.array_equal(before,obs) and np.array_equal(target[:-1],before[1:]), identity
            assert np.array_equal(g['obs/step_id'][:],np.arange(n)), identity
            assert np.allclose(time,g['transition/time_before_s'][:],rtol=0,atol=1e-8),identity
            dt = meta['control_dt']
            assert abs(dt-1/60)<1e-8 and abs(meta['physics_dt']-1/120)<1e-8
            assert np.allclose(g['transition/time_after_s'][:]-time,dt,rtol=0,atol=1e-8),identity
            assert np.all(target>=limits[:,0]-1e-6) and np.all(target<=limits[:,1]+1e-6),identity
            assert g['frames/rgb_u8'].shape==(n,3,3,252,448),identity
            assert np.allclose(g['obs/camera_time_s'][:],time[:,None],rtol=0,atol=1e-8),identity
            assert g['obs/camera_valid'][:].all(),identity
            assert len(meta['joint_order'])==len(meta['joint_order_lab'])==29
            assert set(meta['joint_order'])==set(meta['joint_order_lab'])
            if limits_ref is None:limits_ref=limits;order_ref=meta['joint_order']
            assert np.array_equal(limits,limits_ref) and meta['joint_order']==order_ref
            delay=g['transition/control_trace/delay_index'][:]
            rows.append(dict(identity=identity,steps=n,source_sha256=entry['source_sha256'],
                target_sha256=__import__('hashlib').sha256(target.tobytes()).hexdigest(),
                delayed_ticks=int((delay!=0).sum()),target_changes=int(np.any(target!=before,axis=-1).sum()),
                lower_limit_hits=int((target==limits[:,0]).sum()),upper_limit_hits=int((target==limits[:,1]).sum())))
            if identity in data['splits']['train']:training_targets.append(torch.from_numpy(target))
    assert len(rows)==400 and sum(r['steps'] for r in rows)==81894
    output.mkdir(parents=True,exist_ok=True)
    scale,offset=limits_affine(torch.cat(training_targets))
    torch.save(dict(action_scale=scale,action_offset=offset),output/'position_statistics.pt')
    result=dict(passed=True,episodes=len(rows),steps=sum(r['steps'] for r in rows),rows=rows,
        source_field='transition/target_after_rad',unit='rad',order='canonical',joint_order=order_ref,
        joint_limits_rad=limits_ref.tolist(),statistics_episodes=320,
        statistics_sha256=sha256(output/'position_statistics.pt'),
        direct_recording='InstrumentedAssemblyEnv._pre_physics_step reads _cur_targets after native action processing; PlayEnv._apply_action sends it to set_joint_position_target at both substeps',
        no_label_reconstruction=True,observation_target_is_before=True,visual_alignment_passed=True,
        source_sha256={p:sha256(p) for p in ['visuotactile/sim/environment.py','isaacsimenvs/tasks/play/play_env.py','isaacsimenvs/tasks/play/utils/action_utils.py']})
    atomic_json(output/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.source,a.output);print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
