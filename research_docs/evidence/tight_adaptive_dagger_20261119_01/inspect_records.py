"""CPU diagnostics of closed real records; never alters admission or labels."""
import argparse
import collections
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.tight_balance_20261117.diagnose import phases
from tools.tight_frozen_comparison import table
from tools.tight_frozen_frontend import read, write
from visuotactile.geometry_sampling import STAGES
from visuotactile.grasp_curriculum import holding_metrics


def maximum_difference(a, b):
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def first(mask):
    values = np.flatnonzero(mask)
    return int(values[0]) if len(values) else None


def inspect(roots, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    for root in roots:
        plan = read(root / 'plan.json')
        accepted = read(root / 'correction_entries.json') if (root / 'correction_entries.json').exists() else {}
        for case in plan['cases']:
            directory = root / case['name']
            if not (directory / 'result.json').exists():
                continue
            result = read(directory / 'result.json')
            if not result['legal'] or not result['software_complete']:
                rows.append(dict(case=case['name'], root=str(root), legal=result['legal'], software_complete=result['software_complete']))
                continue
            with h5py.File(directory / 'numeric.h5') as stream:
                group = stream['episode']; transition = group['transition']; observation = group['obs']
                command = transition['target_after_rad'][:]
                before = transition['target_before_rad'][:]
                position = transition['diagnostics_after/joint_position_rad'][:]
                assert np.isfinite(command).all() and np.isfinite(position).all()
                n = len(command); phase = phases(transition)
                legal = np.zeros(n, bool); control = np.zeros(n, bool)
                if 'dagger' in transition:
                    d = transition['dagger']; legal = d['expert_label_valid'][:].reshape(-1).astype(bool)
                    control = d['control_source'][:].reshape(-1).astype(bool)
                    assert not (legal & ~control).any()
                    assert not (legal & ~d['switch_clean'][:].reshape(-1).astype(bool)).any()
                    assert np.array_equal(d['q_reached'][:][legal], position[legal])
                    assert np.array_equal(d['u_expert'][:][legal], command[legal])
                switch = first(control); prefix = switch if switch is not None else n
                continuity = dict(
                    command_max_rad=maximum_difference(before[1:], command[:-1]),
                    joint_max_rad=maximum_difference(transition['diagnostics_before/joint_position_rad'][1:], position[:-1]),
                    time_max_s=maximum_difference(transition['time_before_s'][1:], transition['time_after_s'][:-1]),
                    actuator_feedback_max_rad=maximum_difference(observation['actuator_target_rad'][:], before),
                    observation_time_max_s=maximum_difference(observation['time_s'][:], transition['time_before_s'][:]))
                assert max(continuity.values()) < 1e-6, (case['name'], continuity)
                pose = transition['diagnostics_after/object_pose_w'][:]
                goal = transition['diagnostics_before/goal_pose_w'][:]
                force = transition['object_contact_force_w'][:]
                fingers = (np.linalg.norm(force, axis=-1) > .1).sum(-1).min(-1)
                keypoint = transition['native_keypoints_max_dist'][:].reshape(-1)
                root_distance = np.linalg.norm(pose[:, :3] - goal[:, :3], axis=-1)
                clip = transition['position_clip_mask'][:].reshape(n, -1).any(-1)
                error = command - position
                dagger = result.get('dagger', {})
                trigger = dagger.get('trigger_receipt', {})
                row = dict(case=case['name'], root=str(root), role=case['role'], seed=case['spec']['seed'],
                    legal=True, software_complete=True, success=result['success'], native_subgoals=result['native_subgoals'],
                    stage=case.get('adaptive_stage', 'autonomous'), accepted=case['name'] in accepted,
                    takeover_tick=switch, takeover_phase=switch % 12 if switch is not None else None,
                    first_legal_expert_tick=first(legal), legal_expert_ticks=int(legal.sum()),
                    expert_rescue=dagger.get('expert_takeover_success', False),
                    autonomous_success=bool(result['success'] and switch is None),
                    trigger_state=trigger.get('at_trigger'), steps=n, student_prefix_ticks=prefix,
                    continuity=continuity, student_phase_ticks={s:int((phase[:prefix] == k).sum()) for k,s in enumerate(STAGES)},
                    expert_phase_ticks={s:int(((phase == k) & legal).sum()) for k,s in enumerate(STAGES)},
                    final_student_phase=STAGES[int(phase[prefix-1])],
                    first_two_finger_tick=first(fingers[:prefix] >= 2),
                    first_held_tick=first(np.isin(phase[:prefix], [2,3,4])),
                    first_near_tick=first(phase[:prefix] == 4),
                    student_min_keypoint_m=float(keypoint[:prefix].min()),
                    student_final_keypoint_m=float(keypoint[prefix-1]),
                    student_clipped_ticks=int(clip[:prefix].sum()),
                    student_tracking_arm_rmse_rad=float(np.sqrt(np.mean(error[:prefix,:7] ** 2))),
                    student_tracking_hand_rmse_rad=float(np.sqrt(np.mean(error[:prefix,7:] ** 2))),
                    **{f'student_hold{window}':holding_metrics(pose[:prefix],force[:prefix],window=window) for window in (30,60,120)})
                rows.append(row)
                for tick in range(0,n,12):
                    curves.append(dict(case=case['name'],role=case['role'],seed=case['spec']['seed'],tick=tick,
                        expert=bool(control[tick]),legal_expert=bool(legal[tick]),phase=STAGES[int(phase[tick])],
                        contact_fingers=int(fingers[tick]),object_z_m=float(pose[tick,2]),root_m=float(root_distance[tick]),
                        keypoint_m=float(keypoint[tick]),tracking_rmse_rad=float(np.sqrt(np.mean(error[tick] ** 2)))))
    write(output/'records.json',rows)
    if curves:table(output/'curves.csv',curves)
    summary = dict(closed_cases=len(rows), legal_cases=sum(r['legal'] for r in rows),
        roots=[str(p) for p in roots],
        stages={stage:dict(attempts=len(rr),takeovers=sum(r['takeover_tick'] is not None for r in rr),
            expert_rescues=sum(r['expert_rescue'] for r in rr),accepted=sum(r['accepted'] for r in rr),
            legal_ticks=sum(r['legal_expert_ticks'] for r in rr),
            phases=dict(collections.Counter(r['takeover_phase'] for r in rr if r['takeover_phase'] is not None)))
            for stage in sorted({r['stage'] for r in rows if r['legal']})
            for rr in [[r for r in rows if r['legal'] and r['stage']==stage]]},
        scope='Closed actual records only. Student metrics stop at real takeover. Geometry phases/holding are diagnostics; expert rescue is not autonomous insertion. Zero continuity errors check observable feedback, not hidden-state determinism.')
    write(output/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--roots',nargs='+',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();inspect(args.roots,args.output)
