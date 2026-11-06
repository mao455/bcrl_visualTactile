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


def image_contract(group, horizon=32):
    from .native_data import CONTRACT
    if group not in ('I-OfficialDP','I-HistoryDP') or horizon not in (24,32):
        raise ValueError('Explicit short-horizon ImageDP group required')
    return dict(CONTRACT,version=3,horizon=horizon,action=POSITION_ACTION,
        action_contract='position_rad_v1',observation=group+'_robot178_rgbd_no_privileged_v1',
        history_source='past_requested_and_applied_position_rad; teacher_forced_in_training')


def bounded_targets(actions, limits):
    if actions.ndim != 2 or actions.shape[-1] != 29 or not torch.isfinite(actions).all():
        raise ValueError('Finite canonical [N,29] absolute radians required')
    if limits.shape[-2:] != (29, 2) or not torch.isfinite(limits).all() or (limits[..., 0] >= limits[..., 1]).any():
        raise ValueError('Invalid actual actuator limits')
    return torch.maximum(torch.minimum(actions, limits[..., 1]), limits[..., 0])


def apply_position_targets(env, actions):
    """No velocity conversion, queue, smoothing, rate cap or joint-state writes."""
    limits = env.robot.data.joint_pos_limits[:, env._perm_lab_to_canon]
    clipped = bounded_targets(actions.to(env.device), limits)
    env._cur_targets[:] = clipped[:, env._perm_canon_to_lab]
    env._prev_targets = env._cur_targets.clone()
    # Retain the UNUSED native delay RNG draw to align subsequent observation
    # noise/wrench events across the command/position comparison. No delay applied.
    dr = env.cfg.domain_randomization
    event_draw = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    if dr.use_action_delay and dr.action_delay_max > 0:
        event_draw = torch.randint(0, env._action_queue.shape[1], (env.num_envs,), device=env.device)
    return clipped, event_draw


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
