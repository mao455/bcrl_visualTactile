"""Read-only native success reconstruction and physical-time motion diagnostics.

Ticks in this report are one-based completed control transitions. This module
never modifies the task's predicates, goals, controller, or simulation state.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .native_data import rotate,multiply_quaternion
from tools.runtime import read, write, sha


def first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]+1) if len(indices) else None


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0)
    return dict(n=len(values), mean=float(values.mean()), **{
        name:float(value) for name,value in zip(('min','p05','p25','median','p75','p95','p99','max'),
            np.quantile(values, [0,.05,.25,.5,.75,.95,.99,1]))})


def keypoint_distance(obj, goal, reward):
    assert reward['fixed_size_keypoint_reward']
    corners = torch.tensor(((1,1,1),(1,1,-1),(-1,-1,1),(-1,-1,-1)), dtype=torch.float32)
    offsets = corners*(.5*reward['keypoint_scale']*torch.tensor(reward['fixed_size'], dtype=torch.float32))
    obj,goal = torch.from_numpy(obj),torch.from_numpy(goal)
    def points(p):
        q=p[:,None,3:].expand(-1,4,-1)
        return p[:,None,:3]+rotate(q,offsets[None].expand(len(p),-1,-1))
    return (points(obj)-points(goal)).norm(dim=-1).max(dim=-1).values.numpy()


def reconstruct(distance, fingertip_mean, config, max_goals):
    """Independent state machine for the configured insertion-only native task."""
    p,t,r=config['precise_assembly'],config['termination'],config['reward']
    assert p['goal_mode']=='preInsertAndFinal' and p['random_goal_fraction']==0
    near=0; successes=0; retract=False; finished=False
    counts=[];success=[];near_counts=[];enter=[]
    for d,ft in zip(distance,fingertip_mean):
        inside=d<=p['insertion_success_tolerance']*r['keypoint_scale']
        near=(near+int(inside))*(int(inside) if t['force_consecutive_near_goal_steps'] else 1)
        reached=near>=t['success_steps'] and not (p['enable_retract'] and retract)
        successes=min(max_goals,successes+int(reached))
        if p['enable_retract']:
            retract |= successes>=max_goals
            finished |= bool(retract and d<=p['retract_success_tolerance']*r['keypoint_scale']
                and ft>p['retract_distance_threshold'])
        else:
            finished=successes>=max_goals
        near_counts.append(near);counts.append(successes);success.append(finished);enter.append(retract)
        if reached and successes<max_goals and not retract:near=0
    return dict(subgoals=np.asarray(counts),complete=np.asarray(success),near_count=np.asarray(near_counts),retract=np.asarray(enter))


def terminal_geometry(obj,hole,fingertips,goal_state,config):
    """Final geometric/retraction predicate only; NEVER replaces native success.

    Uses this environment's exported insertion pose sequence. In particular it
    must not mistake the currently active pre-insertion goal for the final pose.
    """
    assert config['precise_assembly']['goal_mode']=='preInsertAndFinal'
    assert goal_state['_insert_pos_rel'].shape==(2,3)
    hole=torch.from_numpy(hole);q=hole[:,3:]
    final=torch.cat((hole[:,:3]+rotate(q,goal_state['_insert_pos_rel'][-1].expand(len(hole),-1)),
        multiply_quaternion(q,goal_state['_insert_quat_wxyz'][-1].expand(len(hole),-1))),-1).numpy()
    distance=keypoint_distance(obj,final,config['reward'])
    fingers=np.linalg.norm(fingertips-obj[:,None,:3],axis=-1).mean(-1)
    p=config['precise_assembly'];limit=p['retract_success_tolerance']*config['reward']['keypoint_scale']
    mask=(distance<=limit)&(fingers>p['retract_distance_threshold'])
    longest=current=0
    for value in mask:
        current=current+1 if value else 0;longest=max(longest,current)
    return dict(ever_satisfied=bool(mask.any()),first_tick=first(mask),satisfied_ticks=int(mask.sum()),
        longest_consecutive_ticks=longest,minimum_final_keypoint_distance_mm=float(distance.min()*1000),
        closest_final_tick=int(distance.argmin()+1),final_tolerance_m=limit,
        fingertip_mean_threshold_m=p['retract_distance_threshold'],
        scope='Diagnostic only: final native geometric and finger-clear predicates without the ordered subgoal prerequisite',
        changes_native_success_or_termination=False),distance,mask


def analyze(group, metadata, trace_output=None, goal_state=None):
    tr=group['transition']; before=tr['diagnostics_before']; after=tr['diagnostics_after']
    cfg=metadata['config']; dt=float(metadata['control_dt']); assert abs(dt-1/60)<1e-9
    obj=after['object_pose_w'][:]; goal=before['goal_pose_w'][:]
    # The next goal is written inside _get_dones. AFTER goal pose is therefore
    # wrong for evaluating the transition that completes the previous goal.
    distance=keypoint_distance(obj,goal,cfg['reward'])
    ft=np.linalg.norm(after['finger_position_w'][:]-obj[:,None,:3],axis=-1).mean(-1)
    native=tr['termination_reasons/max_successes'][:].astype(bool)
    observed_subgoals=after['successes'][:]
    reconstructed=reconstruct(distance,ft,cfg,int(after['max_goals'][0]))
    terminated=np.logical_or.reduce([tr['termination_reasons/'+k][:].astype(bool)
        for k in tr['termination_reasons']])
    first_complete=first(reconstructed['complete']); terminal_tick=first(terminated)
    force=tr['object_contact_force_w'][:] if 'object_contact_force_w' in tr else tr['contact_substeps_after_w_N'][:]
    contact=(np.linalg.norm(force,axis=-1)>.1).any(axis=(1,2))
    first_contact=first(contact);first_lift=first(after['lifted_object'][:]);first_subgoal=first(observed_subgoals>=1)
    all_goals=first(observed_subgoals>=int(after['max_goals'][0]))
    result=dict(ticks=len(obj),duration_sim_s=len(obj)*dt,first_contact_tick=first_contact,
        contact_scope='object-filtered' if 'object_contact_force_w' in tr else 'unfiltered fingertip force diagnostic',
        first_lift_tick=first_lift,first_subgoal_tick=first_subgoal,all_insertion_goals_tick=all_goals,
        first_complete_tick=first_complete,first_native_success_tick=first(native),termination_tick=terminal_tick,
        continued_after_complete_ticks=(len(obj)-first_complete if first_complete else 0),
        reconstructed_success=bool(reconstructed['complete'].any()),native_success=bool(native.any()),
        subgoal_mismatches=int(np.count_nonzero(reconstructed['subgoals']!=observed_subgoals)),
        success_mismatches=int(np.count_nonzero(reconstructed['complete']!=native)),
        min_keypoint_distance_mm=float(distance.min()*1000),
        near_goal_ticks=int(np.count_nonzero(distance<=cfg['precise_assembly']['insertion_success_tolerance']*cfg['reward']['keypoint_scale'])),
        recorded_keypoint_max_difference_m=float(np.max(np.abs(distance-tr['native_keypoints_max_dist'][:]))) if 'native_keypoints_max_dist' in tr else None,
        terminal_reasons={k:bool(tr['termination_reasons/'+k][-1]) for k in tr['termination_reasons']},
        tick_index='1-based completed control transitions',
        phase_times_sim_s=dict(contact=first_contact*dt if first_contact else None,lift=first_lift*dt if first_lift else None,
            first_subgoal=first_subgoal*dt if first_subgoal else None,all_insertion_goals=all_goals*dt if all_goals else None,
            complete=first_complete*dt if first_complete else None))
    result['passed']=result['subgoal_mismatches']==result['success_mismatches']==0 and (first_complete is None or first_complete==terminal_tick==len(obj))
    velocities=after['joint_velocity_rad_s'][:]; q=after['joint_position_rad'][:]; targets=tr['target_after_rad'][:]
    qv0=before['joint_velocity_rad_s'][0];t0=tr['target_before_rad'][0]
    acceleration=np.diff(np.concatenate((qv0[None],velocities)),axis=0)/dt
    rate=np.diff(np.concatenate((t0[None],targets)),axis=0)/dt
    samples={}
    for part,sl in [('arm',slice(0,7)),('hand',slice(7,29))]:
        for name,value in [('speed_rad_s',velocities),('acceleration_rad_s2',acceleration),('target_rate_rad_s',rate),('tracking_rad',q-targets)]:
            samples[part+'_'+name]=np.abs(value[:,sl]).reshape(-1)
    samples['fingertip_contact_N']=np.linalg.norm(force,axis=-1).reshape(-1)
    result['motion']={k:distribution(v) for k,v in samples.items()}
    extra={}
    if goal_state is not None:
        diagnosis,final_distance,terminal_mask=terminal_geometry(obj,before['hole_pose_w'][:],after['finger_position_w'][:],goal_state,cfg)
        result['terminal_geometry']=diagnosis
        final_stage=before['successes'][:]>=1
        if final_stage.any():
            # Both independent reconstructions refer to the same final goal once
            # the native state machine advances; this checks frame/pose sequence.
            delta=float(np.abs(final_distance[final_stage]-distance[final_stage]).max())
            result['terminal_geometry']['native_final_stage_geometry_max_difference_m']=delta
            assert delta<2e-6,'Exported final goal frame disagrees with active native final goal'
        extra=dict(final_keypoint_distance_m=final_distance,terminal_geometry_only=terminal_mask)
    if trace_output:
        np.savez_compressed(trace_output,keypoint_distance_m=distance,fingertip_mean_m=ft,native_success=native,
            native_subgoals=observed_subgoals,**reconstructed,**extra)
    return result,samples


def audit(source, assessment, output):
    output.mkdir(parents=True,exist_ok=True)
    data=read(source/'data.json'); students=[];experts=[];samples={}
    summary=read(assessment/'summary.json')
    for record in summary['rows']:
        t=record['result']['trials'][0]
        if t['cohort']!='train':continue
        directory=Path(record['output'])/t['identity'];m=read(directory/'metadata.json')
        with h5py.File(directory/'numeric.h5') as f:
            result,_=analyze(f['episode'],m,output/(t['identity']+'.npz'))
        students.append(dict(identity=t['identity'],seed=t['seed'],video=str(directory/'rollout.mp4'),**result))
    for identity,entry in data['entries'].items():
        with h5py.File(entry['path']) as f:
            m=json.loads(f.attrs['metadata_json']);g=f['episodes'][entry['episode']]
            result,values=analyze(g,m)
        experts.append(dict(identity=identity,**result))
        for k,v in values.items():samples.setdefault(k,[]).append(v)
    result=dict(state='completed',students_train4=students,expert_count=len(experts),
        expert_passed=sum(e['passed'] for e in experts),experts=experts,
        expert_duration_sim_s=distribution([e['duration_sim_s'] for e in experts]),
        expert_stage_time_sim_s={k:distribution([e['phase_times_sim_s'][k] for e in experts if e['phase_times_sim_s'][k] is not None])
            for k in experts[0]['phase_times_sim_s']},
        expert_motion={k:distribution(np.concatenate(v)) for k,v in samples.items()},
        checkpoint=read(assessment/'checkpoint.json'),source_manifest_sha256=sha(source/'data.json'),
        counts_are_reconstructed_without_changing_original_results=True)
    write(output/'success_timing_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('experts','expert_motion','checkpoint')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--assessment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.source,a.assessment,a.output)
