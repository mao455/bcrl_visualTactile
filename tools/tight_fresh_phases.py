"""Read-only phases from actual numeric records; never changes native success."""
import argparse
import json
from pathlib import Path
import numpy as np
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

ROOT_POSE_SEMANTICS = ('Object and goal root_state_w reference origins, not collision-AABB centers. '
    'Legacy center_distance keys retain root-origin distances. Quaternion angles ignore shape symmetry.')
KEYPOINT_GAP_SEMANTICS = ('keypoint_gap_m = native maximum keypoint distance minus native insertion '
    'tolerance; positive values are excess beyond tolerance, not the full distance.')
SETTLED_REFERENCE = 'Median recorded object root z at completed ticks25-48; descriptive alternative because reset starts in midair'
LEGACY_SETTLED_REFERENCE = SETTLED_REFERENCE.replace('object root z', 'object center z')


def canonical_settled_reference(value):
    # These two documented strings describe the same historical root-state code.
    # Do not silently normalize other definitions or numerical thresholds.
    return SETTLED_REFERENCE if value == LEGACY_SETTLED_REFERENCE else value


def first(mask):
    found=np.flatnonzero(mask)
    return int(found[0]+1) if len(found) else None


from visuotactile.takeover_trigger import stable_windows


def authority_phases(source, stable, near, alignment, subgoals):
    source=np.asarray(source).reshape(-1)
    student=source==0
    if not np.isin(source,[0,1]).all() or not np.array_equal(student,np.arange(len(source))<student.sum()):
        raise ValueError('Expected one actual student prefix followed by expert authority')
    n=int(student.sum())
    return dict(student_ticks=n,expert_ticks=int((source==1).sum()),
        student_first_stable_window_end_tick=first(stable & student),
        student_stable_window_at_last_tick=bool(stable[n-1]) if n else False,
        student_near_goal_first_tick=first(near & student),
        student_alignment_first_tick=first(alignment & student),
        student_max_subgoals=int(subgoals[student].max()) if n else 0,
        expert_first_stable_window_end_tick=first(stable & ~student),
        scope='Actual control authority. Expert completion and post-switch stable windows are not autonomous student outcomes.')


def grasp_progress(stable,distance,angle,subgoals):
    """Describe what happens after the first actual stable window, before goal change."""
    stable=np.asarray(stable,dtype=bool);found=np.flatnonzero(stable)
    if not len(found):return dict(stable_ticks=0,first_tick=None,last_tick=None,longest_continuous_stable_ticks=0)
    start=int(found[0]);runs=np.split(found,np.flatnonzero(np.diff(found)>1)+1)
    after=(np.arange(len(stable))>=start)&(np.asarray(subgoals)==0)
    selected=np.flatnonzero(after)
    return dict(stable_ticks=int(stable.sum()),first_tick=start+1,last_tick=int(found[-1])+1,
        longest_continuous_stable_ticks=max(map(len,runs)),
        first_stable_goal_distance_m=float(distance[start]),first_stable_goal_angle_rad=float(angle[start]),
        subsequent_first_goal_min_distance_m=float(distance[after].min()) if after.any() else None,
        subsequent_first_goal_min_angle_rad=float(angle[after].min()) if after.any() else None,
        first_goal_last_distance_m=float(distance[selected[-1]]) if len(selected) else None,
        first_goal_last_angle_rad=float(angle[selected[-1]]) if len(selected) else None,
        stable_at_final_tick=bool(stable[-1]))


def case(directory, *, numeric_path=None, result=None, case_name=None):
    import h5py
    result=json.loads((directory/'result.json').read_text()) if result is None else result
    case_name=directory.name if case_name is None else case_name
    if not result.get('legal'):return dict(case=case_name,legal=False)
    path=directory/'numeric.h5' if numeric_path is None else numeric_path
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
        simultaneous={}
        for label,metric in [('closest_center',distance),('smallest_keypoint_gap',keypoint-tolerance)]:
            indices=np.flatnonzero(initial_goal)
            if len(indices):
                tick=int(indices[np.argmin(metric[initial_goal])])
                simultaneous[label]=dict(tick=tick+1,center_distance_m=float(distance[tick]),
                    root_origin_distance_m=float(distance[tick]),
                    quaternion_angle_rad=float(angle[tick]),keypoint_gap_m=float(keypoint[tick]-tolerance[tick]),
                    keypoint_distance_m=float(keypoint[tick]),keypoint_tolerance_m=float(tolerance[tick]),
                    stable_window=bool(relative_stable[tick]),two_contact_fingers_both_substeps=bool(two[tick]))
        source=t['dagger/control_source'][:] if 'dagger/control_source' in t else np.zeros(len(pose),dtype=int)
        authority=authority_phases(source,relative_stable,t['native_success_diagnostic/near_goal'][:],
            t['native_alignment_condition'][:],subgoals)
        return dict(case=case_name,legal=True,steps=len(pose),success=bool(result['success']),
            pose_reference_semantics=ROOT_POSE_SEMANTICS,keypoint_gap_semantics=KEYPOINT_GAP_SEMANTICS,
            authority=authority,first_goal_simultaneous=simultaneous,
            grasp_progress=grasp_progress(relative_stable,distance,angle,subgoals),
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
        settled_reference=SETTLED_REFERENCE,pose_reference_semantics=ROOT_POSE_SEMANTICS,
        keypoint_gap_semantics=KEYPOINT_GAP_SEMANTICS,
        limits='Phase diagnostics only, no new success criterion. Goal-angle minima need not coincide with position minima and ignore shape symmetry. Same reset seed is not exact replay.'))

if __name__=='__main__':main()
