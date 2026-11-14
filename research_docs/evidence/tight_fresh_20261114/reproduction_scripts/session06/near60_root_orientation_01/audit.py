"""Closed A64 numeric-only near60 exclusion review; no live or RGB payload access."""
from pathlib import Path
import hashlib,json,time
import h5py,numpy as np
from tools.tight_fresh_report import audit_trigger_case
from tools.tight_fresh_stage_audit import audit_numeric
from tools.tight_fresh_takeover_pose_audit import window_components
from visuotactile.takeover_trigger import contact_height_windows
from visuotactile.compact import sha256

OUT=Path(__file__).with_name('review.json')
BASE=OUT.parent.parent
CENSUS=BASE/'collection09_A64_review_01/closed_A64.json'
TARGET=BASE/'target_observability_01/closed_target_clock_audit.json'
read=lambda p:json.loads(Path(p).read_text())
assert not OUT.exists(), 'Never replace earlier evidence'
started=time.time();census=read(CENSUS);target=read(TARGET);root=Path(census['root'])
plan=read(root/'plan.json');cases={c['name']:c for c in plan['cases']}
index_raw=(root/'results.json').read_bytes();index={r['case']:r['result'] for r in json.loads(index_raw)}
entries=read(root/'correction_entries.json')
assert census['role_complete'] and len(census['cases'])==64
selected=[r for r in census['cases'] if r['schedule']=='near60']
references=[r for r in census['held_cases'] if r['accepted']]
assert len(selected)==8 and len(references)==6
assert all(cases[r['case']]['role']=='A' for r in selected+references)
rows=[];refs=[]


def longest(mask):
    n=best=0
    for value in mask:n=n+1 if value else 0;best=max(n,best)
    return int(best)


def intervals(mask):
    positions=np.flatnonzero(mask)
    if not len(positions):return []
    chunks=np.split(positions,np.flatnonzero(np.diff(positions)>1)+1)
    return [dict(first_completed_tick=int(c[0]+1),last_completed_tick=int(c[-1]+1),ticks=len(c)) for c in chunks]


for prior in selected+references:
    name=prior['case'];directory=root/name;result=read(directory/'result.json');capture=read(directory/'capture_complete.json')
    assert result==index[name] and result['software_complete'] and result['legal']
    assert capture['status']=='source_verified' and capture['dataset_capture']==result['dataset_capture']
    assert capture['dataset_capture']['verified_source_fields']==185
    near=prior['schedule']=='near60'
    replay=audit_trigger_case(directory,result) if near else prior['trigger_replay']
    labels,phases=audit_numeric(directory/'numeric.h5',result) if near else (prior['labels'],prior['phases'])
    numeric_sha=sha256(directory/'numeric.h5')
    assert numeric_sha==replay['numeric_sha256']
    if not near:
        assert prior['result_sha256']==sha256(directory/'result.json')
        assert prior['capture_receipt_sha256']==sha256(directory/'capture_complete.json')
        assert entries[name]['source_sha256']==capture['dataset_capture']['sha256']
    bound=read(root/f"wave_{result['batch_execution']['wave']:04d}_native_bound.json")
    limit=bound['native_max_episode_length'];assert limit==601 and not bound['task_timeout_changed']
    with h5py.File(directory/'numeric.h5','r') as stream:
        t=stream['episode/transition'];b=t['diagnostics_before'];a=t['diagnostics_after']
        pose=a['object_pose_w'][:];palm=a['palm_body_pose_w'][:];goal=b['goal_pose_w'][:];hole=b['hole_pose_w'][:]
        force=t['object_contact_force_w'][:];source=t['dagger/control_source'][:];student=source==0
        assert len(pose)==result['steps'] and student[:48].all()
        assert np.array_equal(a['goal_pose_w'][:-1],goal[1:])
        assert np.array_equal(a['successes'][:-1],b['successes'][1:])
        clock=a['native_stage_step'][:];before_clock=b['native_stage_step'][:]
        assert np.array_equal(clock[:-1],before_clock[1:])
        height=float(np.median(pose[24:48,2]));held=contact_height_windows(pose,force,height);held[:47]=False
        same=(goal==a['goal_pose_w'][:]).all(-1)&(b['successes'][:]==a['successes'][:])
        held_student=held&student&same
        kp=t['native_keypoints_max_dist'][:]
        distance=np.linalg.norm(pose[:,:3]-goal[:,:3],axis=1)
        hole_distance=np.linalg.norm(pose[:,:3]-hole[:,:3],axis=1)
        # Use precisely the historical audit's wxyz sign-invariant quaternion convention.
        angle=2*np.arccos(np.clip(np.abs((pose[:,3:]*goal[:,3:]).sum(-1)),0,1))
        two=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)>=2
        instant=two&(pose[:,2]>=height+.02)&student&(np.arange(len(pose))>=47)
        native_near=t['native_success_diagnostic/near_goal'][:].astype(bool)
        native_align=t['native_alignment_condition'][:].astype(bool)
        assert np.isfinite(distance).all() and np.isfinite(angle).all()
        def state(i):
            if i is None:return None
            return dict(completed_tick=int(i+1),student_authority=bool(student[i]),held12=bool(held[i]),
                same_goal_pre_post=bool(same[i]),native_goal_before=int(b['successes'][i]),native_goal_after=int(a['successes'][i]),
                object_goal_root_distance_m=float(distance[i]),native_keypoint_max_distance_m=float(kp[i]),
                object_goal_angle_rad=float(angle[i]),object_goal_angle_deg=float(np.degrees(angle[i])),
                object_hole_root_distance_m=float(hole_distance[i]),goal_hole_root_offset_m=float(np.linalg.norm(goal[i,:3]-hole[i,:3])),
                object_minus_goal_w_m=(pose[i,:3]-goal[i,:3]).tolist(),goal_minus_hole_w_m=(goal[i,:3]-hole[i,:3]).tolist(),
                height_above_settled_m=float(pose[i,2]-height),native_stage_step=int(clock[i]),
                remaining_controls_if_no_subgoal=int(limit-clock[i]),remaining_seconds_if_no_subgoal=float((limit-clock[i])/60),
                native_near_goal=bool(native_near[i]),native_alignment=bool(native_align[i]),
                window=window_components(pose,palm,force,height,int(i+1)) if i>=11 else None)
        def minimum(mask,value):
            indices=np.flatnonzero(mask)
            return state(int(indices[np.argmin(value[indices])])) if len(indices) else None
        def first(mask):
            indices=np.flatnonzero(mask)
            return state(int(indices[0])) if len(indices) else None
        common=dict(case=name,schedule=prior['schedule'],steps=len(pose),native_reasons=result['reasons'],
            numeric_sha256=numeric_sha,result_sha256=sha256(directory/'result.json'),
            capture_receipt_sha256=sha256(directory/'capture_complete.json'),source_capture_sha256=result['dataset_capture']['sha256'])
        if near:
            assert not result['accepted_for_training'] and labels['switch_action_index'] is None
            assert student.all() and labels['legal_expert_ticks']==0
            assert not (held_student&(kp<=.06)).any()
            root60=held_student&(distance<=.06)
            row=dict(common,trigger_replay=replay,labels=labels,student_native_lift=phases['student_native_lift'],
                held_student_ticks=int(held_student.sum()),held_keypoint_over60mm_ticks=int((held_student&(kp>.06)).sum()),
                held_keypoint_within60mm_ticks=int((held_student&(kp<=.06)).sum()),
                held_root_within60mm_ticks=int(root60.sum()),held_root_within60mm_wrong_keypoint_ticks=int((root60&(kp>.06)).sum()),
                longest_two_finger_contact_ticks=longest(two&student),longest_instant_contact_and_height_ticks=longest(instant),
                held_intervals=intervals(held_student),held_root60_intervals=intervals(root60),
                first_held=first(held_student),first_held_root60=first(root60),
                best_root_while_held=minimum(held_student,distance),best_keypoint_while_held=minimum(held_student,kp),
                best_root_all_student=minimum(student&same,distance),best_root_instant_contact_height=minimum(instant&same,distance),
                last_held=state(int(np.flatnonzero(held_student)[-1])) if held_student.any() else None,
                terminal=state(len(pose)-1),student_native_subgoal_events=int((a['successes'][:]>b['successes'][:]).sum()),
                held_root60_angle_deg_range=[float(np.degrees(angle[root60]).min()),float(np.degrees(angle[root60]).max())] if root60.any() else None)
            rows.append(row)
        else:
            switch=labels['switch_action_index'];boundary=state(switch-1)
            assert boundary['held12'] and boundary['same_goal_pre_post'] and boundary['student_authority']
            assert result['dagger']['expert_takeover_success'] and result['accepted_for_training']
            reference=dict(common,boundary=boundary,expert_recovery_success=True,accepted=True,
                expert_actions=result['dagger']['expert_ticks'],legal_expert_labels=labels['legal_expert_ticks'],
                near60_would_qualify_at_actual_boundary=boundary['native_keypoint_max_distance_m']<=.06,
                root60_would_qualify_at_actual_boundary=boundary['object_goal_root_distance_m']<=.06,
                meaning='Successful expert continuation of another true student state; not a rescue result for the untriggered near60 case.')
            match=next((r for r in target['cases'] if r['case']==name),None)
            if match is not None:
                assert match['boundary']['completed_student_tick']==switch
                assert match['boundary']['native_stage_step']==boundary['native_stage_step']
                assert abs(match['boundary']['object_goal_root_distance_m']-boundary['object_goal_root_distance_m'])<1e-9
                reference['previous_target_clock_audit_matches']=True
            refs.append(reference)

summary=dict(near60_opportunities=8,legal=8,actual_takeovers=0,accepted=0,
    ever_held=sum(r['held_student_ticks']>0 for r in rows),never_held=sum(r['held_student_ticks']==0 for r in rows),
    held_but_never_keypoint60=sum(r['held_student_ticks']>0 and r['held_keypoint_within60mm_ticks']==0 for r in rows),
    held_root60_but_keypoint_over60=sum(r['held_root_within60mm_wrong_keypoint_ticks']>0 for r in rows),
    total_held_student_ticks=sum(r['held_student_ticks'] for r in rows),
    total_held_root60_student_ticks=sum(r['held_root_within60mm_ticks'] for r in rows),
    student_native_subgoal_events=sum(r['student_native_subgoal_events'] for r in rows),
    successful_held_reference_count=len(refs),reference_root60_boundary=sum(r['root60_would_qualify_at_actual_boundary'] for r in refs),
    reference_keypoint60_boundary=sum(r['near60_would_qualify_at_actual_boundary'] for r in refs))
report=dict(version='closed_A64_held_root_vs_native_keypoint_review_v1',started_utc=started,finished_utc=time.time(),
    root=str(root),summary=summary,near60_cases=rows,successful_held_boundary_references=refs,
    inputs=dict(plan_sha256=sha256(root/'plan.json'),closed_A64_sha256=sha256(CENSUS),
        target_clock_audit_sha256=sha256(TARGET),index_read_snapshot_sha256=hashlib.sha256(index_raw).hexdigest(),
        script_sha256=sha256(__file__)),
    definitions=['Current goal root proximity is neither native keypoint alignment nor proximity to the physical hole mouth; both root/goal/hole offsets are reported.',
        'Every sampled state uses the same completed transition: post-action object, unchanged pre/post current goal/stage, completed native clock.',
        'Held12 uses only the previous12 completed transitions and frozen median completed25–48 height; orientation changes are allowed.',
        'Retrospective minima use later observations only for analysis. A future root-distance trigger must use only the current/past completed state, never future minima or persistence.',
        'Remaining time is conditional on no stage change and no earlier native termination. A real native subgoal resets its stage clock; no extra time is proposed.',
        'The six rescued states are independent references, not matched-state counterfactual recovery for s28600052.',
        'Eight registered near60 cases form the whole selected cohort; no failed case replacement or validation/test payload is used.'],
    recommendation=dict(status='Supports a separate small held-root60 collection candidate; does not validate recovery or justify broad weakening of grasp checks.',
        candidate='Keep held12 and unchanged current goal/stage, replace only keypoint<=60mm with current object-root/current-goal-root<=60mm in an explicit new subset, delay0/no fallback; record keypoint and angle without requiring small orientation error.',
        rationale='One actual near60 prefix is held near the target root with large orientation error; five of six independently successful held rescues also enter within60mm root distance while all six exceed60mm keypoint at takeover.',
        limits='Seven of eight near60 prefixes never hold the object. Root proximity cannot fix those early failures. Six references do not establish a recovery probability or optimal distance/time threshold.',
        unchanged='No modifications or launches; current collection, original native15mm success and clock remain unchanged.'),
    simulator_starts=0,optimizer_updates=0,test_validation_payloads_opened=0,rgb_payloads_opened=0,rgb_hashes_recomputed=0)
with OUT.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(dict(summary=summary,elapsed_seconds=report['finished_utc']-started),indent=2))
