"""Read-only phases from actual numeric records; never changes native success."""
import argparse
import json
from pathlib import Path
import numpy as np
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def first(mask):
    found=np.flatnonzero(mask)
    return int(found[0]+1) if len(found) else None


def stable_windows(pose,palm,force,height):
    """Original 12tick contact/relative-pose thresholds, explicit height reference."""
    contact=np.linalg.norm(force,axis=-1)>.1
    two=(contact.sum(-1)>=2).all(-1)
    inverse=palm[:,3:].copy();inverse[:,1:]*=-1
    qv=inverse[:,1:];v=pose[:,:3]-palm[:,:3];cross=2*np.cross(qv,v)
    relative=v+inverse[:,:1]*cross+np.cross(qv,cross)
    q=pose[:,3:]
    relative_q=np.concatenate((inverse[:,:1]*q[:,:1]-(qv*q[:,1:]).sum(-1,keepdims=True),
        inverse[:,:1]*q[:,1:]+q[:,:1]*qv+np.cross(qv,q[:,1:])),axis=-1)
    stable=np.zeros(len(pose),bool)
    for stop in range(12,len(pose)+1):
        start=stop-12
        if not two[start:stop].all() or not (pose[start:stop,2]>=height+.02).all():continue
        drift=np.linalg.norm(relative[start:stop]-relative[start],axis=-1).max()
        cosine=np.clip(np.abs((relative_q[start:stop]*relative_q[start]).sum(-1)),0,1)
        stable[stop-1]=drift<=.01 and (2*np.arccos(cosine)).max()<=.2
    return stable,contact


def case(directory):
    import h5py
    result=json.loads((directory/'result.json').read_text())
    if not result.get('legal'):return dict(case=directory.name,legal=False)
    path=directory/'numeric.h5'
    with h5py.File(path) as f:
        t=f['episode/transition'];d=t['diagnostics_after']
        pose=d['object_pose_w'][:];palm=d['palm_body_pose_w'][:];force=t['object_contact_force_w'][:]
        initial=float(t['diagnostics_before/object_pose_w'][0,2]);settled=float(np.median(pose[24:48,2]))
        native_stable,contact=stable_windows(pose,palm,force,initial)
        relative_stable,_=stable_windows(pose,palm,force,settled)
        # Reset begins in midair. Exclude its fall when reporting later upward movement.
        late=np.arange(len(pose))>=48
        contacts=contact.any(axis=(1,2));two=(contact.sum(-1)>=2).all(-1)
        goal=d['goal_pose_w'][:];distance=np.linalg.norm(pose[:,:3]-goal[:,:3],axis=-1)
        cosine=np.clip(np.abs((pose[:,3:]*goal[:,3:]).sum(-1)),0,1)
        angle=2*np.arccos(cosine)
        tracking=(d['joint_position_rad'][:]-t['target_after_rad'][:])
        post_contact=np.arange(len(pose))>=((first(contacts) or len(pose))-1)
        def rmse(mask,part):
            return float(np.sqrt(np.square(tracking[mask,part]).mean())) if mask.any() else None
        keypoint=t['native_keypoints_max_dist'][:]
        tolerance=t['native_success_diagnostic/insertion_tolerance_m'][:]
        subgoals=d['successes'][:]
        initial_goal=subgoals==0
        return dict(case=directory.name,legal=True,steps=len(pose),success=bool(result['success']),
            reasons=result['reasons'],numeric_sha256=sha256(path),result_sha256=sha256(directory/'result.json'),
            first_contact_tick=first(contacts),first_two_fingers_both_substeps_tick=first(two),
            original_height_stable_grasp=bool(native_stable.any()),settled_height_stable_grasp=bool(relative_stable.any()),
            first_settled_height_stable_window_end_tick=first(relative_stable),
            max_later_rise_over_settled_m=float((pose[late,2]-settled).max()) if late.any() else None,
            later_rise_with_two_fingers_max_m=float((pose[late&two,2]-settled).max()) if (late&two).any() else None,
            native_alignment_first_tick=first(t['native_alignment_condition'][:]),
            native_near_goal_first_tick=first(t['native_success_diagnostic/near_goal'][:]),
            native_first_subgoal_tick=first(subgoals>0),native_subgoals=int(subgoals.max()),
            native_lift=bool(d['lifted_object'][:].any()),
            first_goal_min_center_distance_m=float(distance[initial_goal].min()) if initial_goal.any() else None,
            first_goal_min_quaternion_angle_rad=float(angle[initial_goal].min()) if initial_goal.any() else None,
            first_goal_min_keypoint_gap_m=float((keypoint-tolerance)[initial_goal].min()) if initial_goal.any() else None,
            post_contact_arm_tracking_rmse_rad=rmse(post_contact,slice(0,7)),
            post_contact_hand_tracking_rmse_rad=rmse(post_contact,slice(7,29)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=json.loads((a.rollouts/'plan.json').read_text())
    cases=[case(a.rollouts/c['name']) for c in plan['cases'] if (a.rollouts/c['name']/'result.json').exists()]
    atomic_json(a.output,dict(cases=cases,planned=len(plan['cases']),completed=len(cases),
        source_plan_sha256=sha256(a.rollouts/'plan.json'),tick_index='1-based completed control transitions',
        stable_rule='12 ticks, both physics samples >=2 object-contact fingers above0.1N, >=2cm elevation, palm-relative drift<=1cm/0.2rad',
        settled_reference='Median recorded object center z at completed ticks25-48; descriptive alternative because reset starts in midair',
        limits='Phase diagnostics only, no new success criterion. Goal-angle minima need not coincide with position minima and ignore shape symmetry. Same reset seed is not exact replay.'))

if __name__=='__main__':main()
