"""Explain a completed takeover's grasp-window flags without declaring a drop."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from tools.tight_fresh_phases import case
from tools.tight_fresh_stage_audit import audit_numeric
from visuotactile.compact import sha256
from visuotactile.takeover_trigger import make_trigger, stable_windows
from visuotactile.training_resources import atomic_json


def window_components(pose, palm, force, height, completed_tick):
    pose,palm,force=map(np.asarray,(pose,palm,force))
    if pose.ndim!=2 or pose.shape[1:]!=(7,) or palm.shape!=pose.shape or force.shape!=(len(pose),2,5,3):
        raise ValueError('Aligned actual root/palm poses and two five-finger physics samples required')
    if isinstance(completed_tick,bool) or not isinstance(completed_tick,(int,np.integer)) or not 12 <= completed_tick <= len(pose):
        raise ValueError('A complete past12 observation window is required')
    sl = slice(completed_tick-12, completed_tick)
    if not np.isfinite(height) or not all(np.isfinite(value[sl]).all() for value in (pose,palm,force)):
        raise ValueError('Finite actual poses, contacts and height reference required')
    obj, hand = pose[sl], palm[sl]
    inverse = hand[:,3:].copy(); inverse[:,1:] *= -1
    vector = obj[:,:3]-hand[:,:3]; qv = inverse[:,1:]
    cross = 2*np.cross(qv,vector)
    relative = vector+inverse[:,:1]*cross+np.cross(qv,cross)
    q = obj[:,3:]
    rotation = np.concatenate((inverse[:,:1]*q[:,:1]-(qv*q[:,1:]).sum(-1,keepdims=True),
        inverse[:,:1]*q[:,1:]+q[:,:1]*qv+np.cross(qv,q[:,1:])),axis=-1)
    drift = float(np.linalg.norm(relative-relative[0],axis=-1).max())
    angle = float((2*np.arccos(np.clip(np.abs((rotation*rotation[0]).sum(-1)),0,1))).max())
    contact = np.linalg.norm(force[sl],axis=-1)>.1
    margin = float((obj[:,2]-height-.02).min())
    conditions = dict(two_contact_fingers_every_substep=bool((contact.sum(-1)>=2).all()),
        height=bool((obj[:,2]>=height+.02).all()), translation=drift<=.01, rotation=angle<=.2)
    actual,_ = stable_windows(obj,hand,force[sl],height)
    if bool(actual[-1]) != all(conditions.values()):
        raise ValueError('Component explanation differs from the existing stable-window definition')
    return dict(completed_tick=completed_tick, conditions=conditions,
        stable_window=bool(actual[-1]), height_margin_m=margin,
        palm_relative_root_max_translation_m=drift,palm_relative_max_rotation_rad=angle)


def timeline_checks(transition, observation, result, pose, palm, force):
    """Tie completed-count diagnostics to actual native transition/authority rows."""
    n=len(pose);before=transition['time_before_s'][:];after=transition['time_after_s'][:]
    if n<48 or not all(value.shape==(n,) for value in (before,after,observation['time_s'][:])):
        raise ValueError('Complete aligned60Hz observations and settled reference required')
    if not (np.array_equal(observation['time_s'][:],before) and np.array_equal(after[:-1],before[1:])
        and np.isfinite(before).all() and np.isfinite(after).all()
        and np.allclose(after-before,1/60,atol=1e-9,rtol=0)
        and np.array_equal(transition['step_id'][:],np.arange(n))):
        raise ValueError('Recorded completed-count/step/time alignment differs')
    episode=transition['episode_id'][:]
    if episode.shape!=(n,) or not (episode==episode[0]).all():
        raise ValueError('One uninterrupted recorded episode required')
    for key in ('object_pose_w','palm_body_pose_w','goal_pose_w','successes'):
        if not np.array_equal(transition['diagnostics_after'][key][:-1],transition['diagnostics_before'][key][1:]):
            raise ValueError('Adjacent recorded physical/goal/subgoal states differ')
    source=transition['dagger/control_source'][:];student=source==0;count=int(student.sum())
    if not (source.shape==(n,) and np.isin(source,[0,1]).all()
        and np.array_equal(student,np.arange(n)<count) and 48<=count<n):
        raise ValueError('Complete student prefix and actual expert suffix required')
    if result['steps']!=n or result['dagger']['student_ticks']!=count or result['dagger']['expert_ticks']!=n-count:
        raise ValueError('Result authority/length differs from actual numeric rows')
    receipt=result['dagger']['trigger_receipt']
    trigger=make_trigger(receipt['configuration'],receipt['fallback_tick'])
    for tick in range(count):trigger.observe(tick,pose[tick],palm[tick],force[tick])
    if trigger.receipt()!=receipt or trigger.scheduled_tick!=count:
        raise ValueError('Recorded actual trigger receipt or takeover boundary differs')
    return dict(passed=True,control_hz=60,transitions=n,student_last_completed_tick=count,
        expert_first_action_index=count,expert_first_completed_tick=count+1,
        time_and_step_alignment=True,adjacent_observed_state_continuity=True,
        trigger_replay_exact=True,trigger_receipt=receipt)


def run(directory, output):
    if output.exists():raise FileExistsError('Preserve prior audit')
    result=json.loads((directory/'result.json').read_text())
    if not result.get('software_complete') or not result.get('legal'):
        raise ValueError('Complete legal actual trajectory required')
    phases=case(directory);n=phases['authority']['student_ticks']
    if not phases['authority']['expert_ticks']:raise ValueError('Actual expert takeover required')
    label_check,stage_phase=audit_numeric(directory/'numeric.h5',result)
    if stage_phase['authority']!=phases['authority']:
        raise ValueError('Descriptive phases differ from causal trigger/authority phases')
    with h5py.File(directory/'numeric.h5') as stream:
        transition=stream['episode/transition'];after=transition['diagnostics_after']
        pose,palm,force=[group[key][:] for group,key in ((after,'object_pose_w'),
            (after,'palm_body_pose_w'),(transition,'object_contact_force_w'))]
        timeline=timeline_checks(transition,stream['episode/obs'],result,pose,palm,force)
        timeline.update(first_legal_expert_action_index=label_check['first_legal_label_action_index'],
            first_legal_expert_completed_tick=label_check['first_legal_label_action_index']+1,
            legal_expert_ticks=label_check['legal_expert_ticks'])
        height=float(np.median(pose[24:48,2]));snapshots=[]
        stable,_=stable_windows(pose,palm,force,height)
        for tick in range(12,len(pose)+1):
            if window_components(pose,palm,force,height,tick)['stable_window']!=bool(stable[tick-1]):
                raise ValueError('Window component explanation differs from full-history stable result')
        timeline['full_window_component_checks']=len(pose)-11
        ends={n,phases['authority']['student_first_stable_window_end_tick'],
            result['dagger']['trigger_receipt'].get('first_contact_height_window_end_completed_tick')}
        for tick in sorted(t for t in ends if t is not None and t>=12):
            row=window_components(pose,palm,force,height,tick);goal=after['goal_pose_w'][tick-1]
            row.update(goal_root_distance_m=float(np.linalg.norm(pose[tick-1,:3]-goal[:3])),
                goal_quaternion_angle_rad=float(2*np.arccos(np.clip(abs((pose[tick-1,3:]*goal[3:]).sum()),0,1))),
                native_keypoint_max_distance_m=float(transition['native_keypoints_max_dist'][tick-1]),
                native_insertion_tolerance_m=float(transition['native_success_diagnostic/insertion_tolerance_m'][tick-1]))
            snapshots.append(row)
        terminated={key:bool(value[:n].any()) for key,value in transition['termination_reasons'].items()}
        subgoals=after['successes'][:].reshape(-1)
        subgoal_changes=[dict(completed_tick=i+1,subgoals=int(v)) for i,v in enumerate(subgoals)
            if i==0 or v!=subgoals[i-1]]
        before=transition['diagnostics_before'];subgoals_before=before['successes'][:].reshape(-1)
        boundaries=[dict(action_index=int(i),completed_tick=int(i)+1,
            authority='student' if i<n else 'expert',expert_label_valid=bool(transition['dagger/expert_label_valid'][i]),
            native_alignment=bool(transition['native_alignment_condition'][i]),
            subgoals_before=int(subgoals_before[i]),subgoals_after=int(subgoals[i]),
            goal_root_before=before['goal_pose_w'][i].tolist(),goal_root_after=after['goal_pose_w'][i].tolist(),
            goal_root_changed_during_transition=not np.array_equal(before['goal_pose_w'][i],after['goal_pose_w'][i]))
            for i in np.flatnonzero(subgoals!=subgoals_before)]
    if sha256(directory/'numeric.h5')!=phases['numeric_sha256'] or sha256(directory/'result.json')!=phases['result_sha256']:
        raise ValueError('Trajectory/result changed during the audit')
    source_root=Path(__file__).resolve().parents[1]
    atomic_json(output,dict(complete=True,source_case=str(directory.resolve()),phases=phases,
        snapshots=snapshots,student_native_termination_flags=terminated,subgoal_changes=subgoal_changes,
        timeline_checks=timeline,expert_label_alignment=label_check,subgoal_transition_boundaries=boundaries,
        source_sha256=sha256(__file__),source_environment_sha256=sha256(source_root/'visuotactile/sim/environment.py'),
        trigger_sha256=sha256(source_root/'visuotactile/takeover_trigger.py'),
        dependencies_sha256={path:sha256(source_root/path) for path in ('tools/tight_fresh_phases.py','tools/tight_fresh_stage_audit.py')},
        tick_semantics='completed_tick is1-based completed control transitions; action_index is0-based. The last student completed tick N is the pre-action boundary for expert action N, whose completed tick is N+1. Expert labels additionally wait for the recorded delay queue flush.',
        settled_reference='Median recorded object root z at completed ticks25-48; available causally after completed48.',
        subgoal_goal_semantics='diagnostics_after is captured after native reward/goal updates and before automatic episode reset. At a subgoal transition, its goal root can already be the next target; compare before/after explicitly. Native alignment/subgoal completion remains authoritative.',
        pose_semantics='Recorded object.data.root_state_w and goal_viz.data.root_state_w reference origins, not collision-AABB centers. Quaternion angle ignores shape symmetries; native keypoint criteria remain authoritative.',
        interpretation='A failed stable window is a diagnostic flag. Relative motion can include reorientation; this report does not identify deliberate versus accidental motion, slipping, or dropping. Preserved expert completion is not autonomous student success.',
        optimizer_updates=0,simulator_starts=0,test_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.case,args.output)
