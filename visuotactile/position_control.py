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

# This describes the reachable *postprocessed* target set, not another pass
# through the native command queue. Historical checkpoints omit this identity.
REACHABLE_POSITION_EXECUTION = dict(version='expert_reachable_postprocessed_v1',
    input_space='postprocessed_absolute_joint_target_rad',
    reference='actual_pre_action_actuator_target_rad', control_dt=1/60,
    dof_speed_scale=1.5, arm_moving_average=.1, hand_moving_average=.1,
    arm_progress=.5, hand_progress=1., hand_target_cap_rad_s=2.,
    roundoff_tolerance_rad=1e-6,
    delay='already_resolved_in_supervision_no_second_queue')


def validate_execution_adapter(value):
    """Missing means legacy; new identities never silently acquire old math."""
    if value is None:
        return None
    if value != REACHABLE_POSITION_EXECUTION:
        raise ValueError('Unknown or changed position execution adapter identity')
    return dict(value)


def validate_execution_environment(adapter, metadata):
    adapter = validate_execution_adapter(adapter)
    if adapter is None:
        return
    action = metadata.get('config', {}).get('action', {})
    if (metadata.get('action_control_mode') != POSITION_ACTION
            or abs(metadata.get('control_dt',0)-adapter['control_dt']) > 1e-10
            or any(action.get(key) != adapter[key] for key in
                   ('dof_speed_scale', 'arm_moving_average', 'hand_moving_average'))):
        raise ValueError('Position execution adapter differs from actual environment')


def processed_expert_target(previous, native, speed, dt):
    """Native-mapped/post-delay radians -> post-speed expert command, once.

    ``native`` is already native arm-delta/hand-affine/EMA/limit processed.
    This function must never receive an existing u_expert label.
    """
    from .expert_speed import progress_target, cap_hand_target
    progressed = progress_target(previous, native, speed['arm'], speed['hand'])
    return progressed, cap_hand_target(previous, progressed, speed.get('hand_target_cap_rad_s'), dt)


def reachable_position_bounds(previous, limits, adapter):
    """Exact image of native actions in [-1,1] at the supplied real history.

    Arm endpoints use p + speed*dt*a, clamp, EMA, clamp, then progression.
    Hand endpoints use the absolute joint limits, EMA, then the 2 rad/s cap.
    These are independent monotone joint maps; endpoint images give the whole
    reachable interval. Delay has already chosen a native action and does not
    change this interval. No new delay or alpha is applied to final labels.
    """
    adapter = validate_execution_adapter(adapter)
    if adapter is None:
        raise ValueError('Explicit reachable execution adapter required')
    if previous.shape[-1] != 29 or not torch.isfinite(previous).all():
        raise ValueError('Finite real previous actuator targets required')
    if limits.shape[-2:] != (29, 2):
        raise ValueError('Canonical actuator limits required')
    p = previous.detach(); limits = limits.detach()
    lo, hi = limits[..., 0], limits[..., 1]
    def endpoint(sign):
        arm_raw = (p[..., :7]+adapter['dof_speed_scale']*adapter['control_dt']*sign).clamp(
            min=lo[..., :7], max=hi[..., :7])
        arm = (adapter['arm_moving_average']*arm_raw+
            (1-adapter['arm_moving_average'])*p[..., :7]).clamp(min=lo[..., :7], max=hi[..., :7])
        hand_raw = lo[..., 7:] if sign < 0 else hi[..., 7:]
        hand = (adapter['hand_moving_average']*hand_raw+
            (1-adapter['hand_moving_average'])*p[..., 7:]).clamp(min=lo[..., 7:], max=hi[..., 7:])
        native = torch.cat((arm, hand), -1)
        speed = dict(arm=adapter['arm_progress'], hand=adapter['hand_progress'],
            hand_target_cap_rad_s=adapter['hand_target_cap_rad_s'])
        return processed_expert_target(p, native, speed, adapter['control_dt'])[1]
    return torch.stack((endpoint(-1), endpoint(1)), -1)


def project_processed_position_targets(target, previous, limits, adapter):
    """Shared differentiable final-command projection for training/deployment.

    Feasible expert postprocessed targets remain bytewise unchanged. Predictions
    outside that same reachable set are clipped; this is not repeated slowing.
    Real previous targets (including t0) come from robot[...,58:87], never nominal.
    """
    if target.shape != previous.shape or target.shape[-1] != 29 or not torch.isfinite(target).all():
        raise ValueError('Matching finite canonical targets/history required')
    bounds = reachable_position_bounds(previous, limits, adapter)
    clipped = torch.maximum(torch.minimum(target, bounds[..., 1]), bounds[..., 0])
    # Recorded CUDA FP32 arithmetic and CPU endpoint reconstruction can differ
    # by an ulp. Preserve those already-processed labels, never apply alpha twice.
    # This fixed numerical allowance is recorded in the execution identity.
    return torch.where((clipped-target).abs() <= adapter['roundoff_tolerance_rad'], target, clipped)


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
        progressed, mapped = processed_expert_target(previous, native, self.speed, env.step_dt)
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


def audit_execution(source, output, rollout_roots=()):
    """CPU-only actual-record audit; never opens independent test trajectories."""
    import hashlib
    import csv
    source,output=Path(source),Path(output)
    data=json.loads((source/'data.json').read_text())
    allowed=list(dict.fromkeys(data['splits']['train']+data['splits']['validation']+
        data['splits'].get('correction_dev',[])))
    if set(allowed)&set(data['splits'].get('test',[])):
        raise ValueError('Execution audit may not read held-out test records')
    adapter=REACHABLE_POSITION_EXECUTION
    rows=[];joint_rows=[];labels=0;max_roundoff=0.;max_mapping_error=0.;mapped_expert_ticks=0;max_substep_error=0.
    changed_expert=0;student_ticks=student_changed=0;max_equivalence=0.
    def examine(g,metadata,identity,kind,source_path,source_sha):
        nonlocal labels,max_roundoff,max_mapping_error,changed_expert,student_ticks,student_changed,max_equivalence,mapped_expert_ticks,max_substep_error
        tr=g['transition']; previous=torch.from_numpy(tr['target_before_rad'][:]); target=torch.from_numpy(tr['target_after_rad'][:])
        limits=torch.tensor(metadata['joint_limits_rad'][0])
        if abs(metadata['control_dt']-adapter['control_dt'])>1e-10 or any(metadata['config']['action'][key]!=adapter[key]
                for key in ('dof_speed_scale','arm_moving_average','hand_moving_average')):
            raise ValueError('Recorded native control settings differ: '+identity)
        mask=torch.from_numpy(tr['dagger/expert_label_valid'][:]).bool() if 'dagger' in tr else torch.ones(len(target),dtype=torch.bool)
        student=(torch.from_numpy(tr['dagger/control_source'][:])==0 if 'dagger' in tr else torch.zeros(len(target),dtype=torch.bool))
        if kind=='autonomous_student':
            mask=torch.zeros_like(mask);student=torch.ones_like(mask)
        bounds=reachable_position_bounds(previous,limits,adapter)
        exact=torch.maximum(torch.minimum(target,bounds[...,1]),bounds[...,0])
        projected=project_processed_position_targets(target,previous,limits,adapter)
        error=(projected-target).abs();roundoff=(exact-target).abs()
        raw=tr.get('dagger/expert_native_target_rad',tr.get('speed_native_target_rad'))
        map_error=0.;equivalence=0.
        if mask.any() and raw is None:
            raise ValueError('Expert supervision is missing its native mapped target: '+identity)
        if raw is not None and mask.any():
            mapped_expert_ticks+=int(mask.sum())
            native=torch.from_numpy(raw[:]);speed=dict(arm=.5,hand=1.,hand_target_cap_rad_s=2.)
            mapped=processed_expert_target(previous,native,speed,adapter['control_dt'])[1]
            map_error=float((mapped[mask]-target[mask]).abs().max())
            as_student=project_processed_position_targets(mapped,previous,limits,adapter)
            equivalence=float((as_student[mask]-mapped[mask]).abs().max())
            max_mapping_error=max(max_mapping_error,map_error);max_equivalence=max(max_equivalence,equivalence)
        substeps=torch.from_numpy(tr['actuator_substep_target_rad'][:])
        substep_error=float((substeps-target[:,None]).abs().max())
        max_substep_error=max(max_substep_error,substep_error)
        count=int(mask.sum());scount=int(student.sum());modified=int((error[mask]>0).any(-1).sum())
        labels+=count;changed_expert+=modified;student_ticks+=scount;student_changed+=int((error[student]>0).any(-1).sum())
        pre_error=float(roundoff[mask].max()) if count else 0.;max_roundoff=max(max_roundoff,pre_error)
        digest=hashlib.sha256()
        for name,value in (('previous',previous),('target',target),('expert_mask',mask),('limits',limits)):
            digest.update(name.encode());digest.update(value.numpy().tobytes())
        row=dict(identity=identity,kind=kind,path=str(source_path),manifest_source_sha256=source_sha,
            numeric_evidence_sha256=digest.hexdigest(),ticks=len(target),expert_ticks=count,
            expert_changed_ticks=modified,expert_before_roundoff_allowance_max_rad=pre_error,
            expert_mapping_max_error_rad=map_error,expert_student_equivalence_max_error_rad=equivalence,
            actual_physics_substep_target_max_error_rad=substep_error,
            student_ticks=scount,student_changed_ticks=int((error[student]>0).any(-1).sum()),
            student_projection_max_rad=float(error[student].max()) if scount else 0.,
            student_arm_target_speed_max_rad_s=float((target-previous)[student,:7].abs().max()/adapter['control_dt']) if scount else None,
            student_hand_target_speed_max_rad_s=float((target-previous)[student,7:].abs().max()/adapter['control_dt']) if scount else None)
        rows.append(row)
        if scount:
            for j,name in enumerate(metadata['joint_order']):
                joint_rows.append(dict(identity=identity,kind=kind,joint=name,joint_index=j,student_ticks=scount,
                    clipped_ticks=int((error[student,j]>0).sum()),projection_rmse_rad=float(error[student,j].square().mean().sqrt()),
                    target_speed_max_rad_s=float((target-previous)[student,j].abs().max()/adapter['control_dt'])))
    for identity in allowed:
        entry=data['entries'][identity]
        with h5py.File(entry['path'],'r') as stream:
            examine(stream['episodes'][entry['episode']],json.loads(stream.attrs['metadata_json']),identity,
                'training_or_development_record',entry['path'],entry.get('source_sha256'))
    for root in rollout_roots:
        for path in sorted(Path(root).glob('*/numeric.h5')):
            with h5py.File(path,'r') as stream:
                examine(stream['episode'],json.loads(stream.attrs['metadata_json']),path.parent.name,
                    'autonomous_student',path,None)
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'execution_rows.json',rows)
    with (output/'execution_joints.csv').open('w',newline='') as stream:
        if joint_rows:
            writer=csv.DictWriter(stream,fieldnames=list(joint_rows[0]));writer.writeheader();writer.writerows(joint_rows)
    files=['visuotactile/position_control.py','visuotactile/expert_speed.py','visuotactile/sim/expert_speed_environment.py',
        'visuotactile/models/native_dp.py','visuotactile/native_controller.py','visuotactile/state_benchmark.py','visuotactile/anchor_continue.py',
        'isaacsimenvs/tasks/play/utils/action_utils.py']
    report=dict(passed=changed_expert==0 and max_mapping_error==0 and max_equivalence==0 and mapped_expert_ticks==labels and max_substep_error==0,
        scope='CPU numeric replay of actual recorded targets/history; no new physics or success claim',
        execution_adapter=adapter,expert_student_applied_equivalent=max_equivalence==0,
        equivalence_scope='Same native-mapped absolute proposal and actual history, one speed mapping, then shared final position target',
        supervision_semantics_preserved=changed_expert==0,expert_valid_ticks=labels,
        mapped_expert_ticks=mapped_expert_ticks,actual_physics_substep_target_max_error_rad=max_substep_error,
        expert_changed_ticks=changed_expert,expert_mapping_max_error_rad=max_mapping_error,
        maximum_fp32_endpoint_roundoff_rad=max_roundoff,training_development_episodes=len(allowed),
        test_episodes_read=0,student_ticks=student_ticks,student_outside_shared_reachable_ticks=student_changed,
        native_control_changed=False,new_simulator_rollouts=0,
        finding='Legacy postprocessed student commands can exceed the expert reachable target set; labels already include speed/delay and must not be slowed twice',
        double_speed_application_found=False,valid_expert_labels_invalidated=False,
        historical_introduction=dict(direct_postprocessed_position_executor='c7fcbb0e2d27e21fa5c818171014413d2b067d0a',
            hand_target_slew_cap='024cdaabf420b26565fb173ea2c917ddff7e98f2',
            dagger_expert_position_bridge='01ce8d6ba785f73e8737f6598f4ceb4e13515b35'),
        sources={p:sha256(p) for p in files},data_manifest=dict(path=str(source/'data.json'),sha256=sha256(source/'data.json')),
        evidence=[dict(path=str(output/name),sha256=sha256(output/name)) for name in ('execution_rows.json','execution_joints.csv')])
    atomic_json(output/'execution_acceptance.json',report)
    return report


def audit_execution_models(roots, output):
    """Real trained rear/cache CPU forwards and separate fit/reg gradients, no step."""
    from .models.native_dp import TargetResidual
    from .native_data import residual_batch
    from .anchor_continue import load_residual_continuation
    import math
    results=[]
    for root in map(Path,roots):
        checkpoint=root/'training/best.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        cfg=saved['configuration'];index=json.loads((root/'nominal/index.json').read_text())
        data=json.loads((root/'state_data/data.json').read_text())
        if index['base_sha256']!=cfg['base_sha256'] or index['data_sha256']!=sha256(root/'state_data/data.json'):
            raise ValueError('Real model audit needs the checkpoint-matched nominal cache')
        statistics=torch.load(cfg['residual_statistics_path'],map_location='cpu',weights_only=False)
        scales=torch.tensor([.2]*7+[.9]*22)
        model=TargetResidual(statistics,scales,execution_adapter=REACHABLE_POSITION_EXECUTION).train()
        adaptation=dict(version='matched_ab_scales_v1',requires_training_before_deployment=True,
            old=saved['model']['scales'].tolist(),new=scales.tolist())
        continuation=load_residual_continuation(model,saved,cfg['residual_statistics_sha256'],scale_adaptation=adaptation)
        strata=data.get('sampling_strata',{})
        samples=[('core',strata.get('core',data['splits']['train'])[0])]
        corrections=strata.get('current_correction',[])+strata.get('historical_correction',[])
        if corrections:samples.append(('recovery_initial',corrections[0]))
        for kind,identity in samples:
            path=root/'nominal'/(identity+'.pt')
            if sha256(path)!=index['episodes'][identity]['sha256']:
                raise ValueError('Real numeric cache changed during model audit')
            record=torch.load(path,map_location='cpu',weights_only=False)
            mask=record.get('supervision_mask',torch.ones(len(record['target']),dtype=torch.bool)).bool()
            first=int(mask.nonzero()[0]);stop=min(first+32,len(mask));state=None
            with torch.no_grad():
                for start in range(0,first,32):
                    prefix=residual_batch([record],start,min(start+32,first),'cpu')
                    _,state=model(prefix,state)
            inputs=residual_batch([record],first,stop,'cpu');raw,_=model(inputs,state)
            previous=inputs['robot'][...,58:87];base=record['base'][None,first:stop];target=record['target'][None,first:stop]
            valid=mask[None,first:stop]&inputs['valid'];limits=record['limits'];gate=record['gate'][None,first:stop]
            action,delta=model.compose(base,raw,gate,limits,previous=previous)
            z=((action-target)*model.action_scale).square()
            fit=((.5*z[...,:7].mean(-1)+.5*z[...,7:].mean(-1))*valid).sum()/valid.sum()
            reg=(1e-4*(.5*delta[...,:7].square().mean(-1)+.5*delta[...,7:].square().mean(-1))*valid).sum()/valid.sum()
            parameters=list(model.parameters())
            fit_grads=torch.autograd.grad(fit,parameters,retain_graph=True,allow_unused=True)
            raw_fit=torch.autograd.grad(fit,raw,retain_graph=True)[0]
            reg_grads=torch.autograd.grad(reg,parameters,allow_unused=True)
            norm=lambda values: math.sqrt(sum(float(g.detach().double().square().sum()) for g in values if g is not None))
            fit_norm=norm(fit_grads);reg_norm=norm(reg_grads)
            pre=(base+delta/model.action_scale).clamp(min=limits[...,0],max=limits[...,1])
            projection=(action-pre).abs()>REACHABLE_POSITION_EXECUTION['roundoff_tolerance_rad']
            label_projection=project_processed_position_targets(target,previous,limits,REACHABLE_POSITION_EXECUTION)
            final_projection=project_processed_position_targets(action,previous,limits,REACHABLE_POSITION_EXECUTION)
            results.append(dict(group=root.name,sample_kind=kind,identity=identity,prefix_ticks_replayed=first,
                evaluated_ticks=stop-first,valid_expert_ticks=int(valid.sum()),checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                frontend_sha256=cfg['base_sha256'],cache=str(path),cache_sha256=sha256(path),continuation=continuation,
                data_fit_loss=float(fit.detach()),regularization_loss=float(reg.detach()),
                data_fit_gradient_l2=fit_norm,regularization_gradient_l2=reg_norm,
                data_fit_raw_output_gradient_l2=norm([raw_fit]),
                raw_output_nonzero_fit_gradient_joints=int((raw_fit[valid].abs()>0).sum()),
                valid_output_joint_population=int(valid.sum())*29,
                projection_active_joint_fraction=float(projection[valid].float().mean()),
                projection_active_tick_fraction=float(projection[valid].any(-1).float().mean()),
                final_projection_idempotent=torch.equal(final_projection,action),
                expert_labels_unchanged=torch.equal(label_projection[valid],target[valid]),
                effective_fit_gradient=math.isfinite(fit_norm) and fit_norm>0,
                finite=math.isfinite(float(fit.detach())) and math.isfinite(fit_norm) and math.isfinite(reg_norm)))
    report=dict(passed=bool(results) and all(r['finite'] and r['effective_fit_gradient'] and r['final_projection_idempotent'] and r['expert_labels_unchanged'] for r in results),
        scope='CPU actual selected Transformer weights, actual matching cache and causal full prefix; no optimizer step or simulator',
        execution_adapter=REACHABLE_POSITION_EXECUTION,optimizer_updates=0,new_gpu_work=False,
        note='A finite/nonzero regularizer gradient cannot substitute for a measured data-fit gradient; clipping fractions and fit gradients are separate',
        results=results,sources={p:sha256(p) for p in ('visuotactile/position_control.py','visuotactile/anchor_continue.py','visuotactile/models/native_dp.py','visuotactile/native_controller.py','visuotactile/state_benchmark.py')})
    atomic_json(Path(output)/'execution_model_acceptance.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execution-audit',action='store_true');p.add_argument('--rollout-root',type=Path,action='append',default=[])
    p.add_argument('--model-root',type=Path,action='append',default=[])
    a=p.parse_args();result=audit_execution(a.source,a.output,a.rollout_root) if a.execution_audit else audit(a.source,a.output)
    if a.model_root:result['model_validation']=audit_execution_models(a.model_root,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
