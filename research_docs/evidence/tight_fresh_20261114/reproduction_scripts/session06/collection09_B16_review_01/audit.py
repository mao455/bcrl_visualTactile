"""Closed B wave4 and same-family A wave0 numeric-only audit; never active HDF/RGB."""
from pathlib import Path
from collections import Counter
import hashlib,json,time
import h5py,numpy as np
from tools.tight_fresh_report import audit_trigger_case
from tools.tight_fresh_stage_audit import audit_numeric
from tools.tight_fresh_takeover_pose_audit import window_components
from tools.tight_collection_summary import held_recheck
from visuotactile.compact import sha256
from visuotactile.takeover_trigger import contact_height_windows,stable_windows

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent/'new_family_09'
OUT=HERE/'closed_B16.json'
assert not OUT.exists(), 'Preserve prior result'
started=time.time()
read=lambda p:json.loads(Path(p).read_text())
plan=read(ROOT/'plan.json')
raw=(ROOT/'results.json').read_bytes()
index={r['case']:r['result'] for r in json.loads(raw)}
entry_raw=(ROOT/'correction_entries.json').read_bytes();entries=json.loads(entry_raw)
selected={role:sorted([c for c in plan['cases'] if c['role']==role and 28600000<=c['spec']['seed']<28600016],
    key=lambda c:c['spec']['seed']) for role in ('A','B')}
assert all(len(cs)==16 for cs in selected.values())
assert all(c['name'] in index for cs in selected.values() for c in cs)
for role,cs in selected.items():
    assert all(index[c['name']]['batch_execution']['wave']==(0 if role=='A' else 4) for c in cs)

def first(mask):
    ix=np.flatnonzero(mask);return int(ix[0]+1) if len(ix) else None

def intervals(mask):
    ix=np.flatnonzero(mask)
    if not len(ix):return []
    chunks=np.split(ix,np.flatnonzero(np.diff(ix)>1)+1)
    return [dict(first_completed_tick=int(x[0]+1),last_completed_tick=int(x[-1]+1),ticks=len(x)) for x in chunks]

def audit(c):
    name=c['name'];directory=ROOT/name;result=read(directory/'result.json')
    assert result==index[name] and result['software_complete'] is True
    row=dict(case=name,role=c['role'],seed=c['spec']['seed'],family=c['spec']['family'],schedule=c['expansion_schedule'],
        legal=bool(result.get('legal')),accepted=bool(result.get('accepted_for_training')),success=bool(result.get('success')),
        expert_recovery_success=bool(result.get('dagger',{}).get('expert_takeover_success')),
        student_autonomous_success=bool(result.get('dagger',{}).get('student_autonomous_success')),
        steps=result.get('steps'),native_reasons=result.get('reasons'),result_sha256=sha256(directory/'result.json'),
        initial_observable_sha256=sha256(directory/'initial_observable.json'),
        initial_collision_sha256=sha256(directory/'initial_collision.json'),batch_execution=result['batch_execution'])
    if c['expansion_schedule'] in ('held60','held120'):row['held_recheck']=held_recheck(c,result)
    if not row['legal']:
        assert not row['accepted'] and not row['success'] and result.get('not_started')
        assert not (directory/'numeric.h5').exists()
        row['initial_collision']=read(directory/'initial_collision.json')
        return row
    capture=read(directory/'capture_complete.json')
    assert capture['status']=='source_verified' and capture['dataset_capture']==result['dataset_capture']
    assert capture['dataset_capture']['verified_source_fields']==185
    labels,phases=audit_numeric(directory/'numeric.h5',result)
    replay=audit_trigger_case(directory,result)
    if row['accepted']:
        entry=entries[name]
        assert row['expert_recovery_success'] and row['success']
        assert entry['source_sha256']==capture['dataset_capture']['sha256']
        assert entry['family']==c['spec']['family'] and entry['takeover_tick']==labels['switch_action_index']
        assert all(entry[k]==plan['pairs'][c['role']][k] for k in ('frontend_sha256','backend_sha256'))
    native_bound=read(ROOT/f"wave_{result['batch_execution']['wave']:04d}_native_bound.json")
    assert native_bound['native_max_episode_length']==601 and not native_bound['task_timeout_changed']
    with h5py.File(directory/'numeric.h5','r') as f:
        t=f['episode/transition'];b=t['diagnostics_before'];a=t['diagnostics_after']
        source=t['dagger/control_source'][:];student=source==0;valid=t['dagger/expert_label_valid'][:].astype(bool)
        pose=a['object_pose_w'][:];palm=a['palm_body_pose_w'][:];goal=b['goal_pose_w'][:];hole=b['hole_pose_w'][:]
        force=t['object_contact_force_w'][:];n=len(pose)
        assert n==result['steps'] and np.array_equal(t['step_id'][:],np.arange(n))
        assert np.array_equal(f['episode/obs/time_s'][:],t['time_before_s'][:])
        assert np.allclose(t['time_after_s'][:]-t['time_before_s'][:],1/60,rtol=0,atol=1e-9)
        for key in ('object_pose_w','goal_pose_w','successes','native_stage_step'):
            assert np.array_equal(a[key][:-1],b[key][1:])
        assert t['target_after_rad'].shape==(n,29) and np.isfinite(t['target_after_rad'][:]).all()
        height=float(np.median(pose[24:48,2]));student_reference=bool(student[:48].all())
        held=contact_height_windows(pose,force,height);stable,_=stable_windows(pose,palm,force,height)
        held[:47]=False;stable[:47]=False
        if not student_reference:held[student]=False;stable[student]=False
        two=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)>=2
        any_contact=(np.linalg.norm(force,axis=-1)>.1).any(axis=(1,2))
        kp=t['native_keypoints_max_dist'][:];dist=np.linalg.norm(pose[:,:3]-goal[:,:3],axis=-1)
        angle=2*np.arccos(np.clip(np.abs((pose[:,3:]*goal[:,3:]).sum(-1)),0,1))
        before_stage=b['successes'][:];after_stage=a['successes'][:]
        same=(goal==a['goal_pose_w'][:]).all(-1)&(before_stage==after_stage)
        near=t['native_success_diagnostic/near_goal'][:].astype(bool)
        align=t['native_alignment_condition'][:].astype(bool)
        lift=a['lifted_object'][:].astype(bool);clock=a['native_stage_step'][:]
        stage_events=after_stage>before_stage
        def state(i):
            if i is None:return None
            return dict(completed_tick=int(i+1),authority='student' if student[i] else 'expert',expert_label_valid=bool(valid[i]),
                native_goal_before=int(before_stage[i]),native_goal_after=int(after_stage[i]),same_goal_pre_post=bool(same[i]),
                native_clock_before=int(b['native_stage_step'][i]),native_clock_after=int(clock[i]),
                remaining_controls_if_no_subgoal=int(601-clock[i]),remaining_seconds_if_no_subgoal=float((601-clock[i])/60),
                held12=bool(held[i]) if not student[i] or student_reference else None,
                old_stable12=bool(stable[i]) if not student[i] or student_reference else None,
                native_lift=bool(lift[i]),two_fingers_both_substeps=bool(two[i]),
                object_current_goal_root_distance_m=float(dist[i]),native_keypoint_max_distance_m=float(kp[i]),
                object_current_goal_angle_deg=float(np.degrees(angle[i])),
                object_hole_root_distance_m=float(np.linalg.norm(pose[i,:3]-hole[i,:3])),
                goal_minus_hole_root_w_m=(goal[i,:3]-hole[i,:3]).tolist(),
                height_above_settled_m=float(pose[i,2]-height) if not student[i] or student_reference else None,
                native_near_goal=bool(near[i]),native_alignment=bool(align[i]),
                window=window_components(pose,palm,force,height,int(i+1)) if i>=47 and (not student[i] or student_reference) else None)
        def minimum(mask,value):
            ix=np.flatnonzero(mask);return state(int(ix[np.argmin(value[ix])])) if len(ix) else None
        switch=labels['switch_action_index']
        sfirst=first(student&held);sbest=minimum(student&held&same,kp)
        snapshots={}
        for label,tick in [('first_student_held',sfirst),('first_student_stable',first(student&stable)),
            ('student_last',int(student.sum())),('takeover',switch)]:
            snapshots[label]=state(tick-1) if tick else None
        student_counts=dict(native_lift=bool((student&lift).any()),
            held12=bool((student&held).any()) if student_reference else None,
            old_stable12=bool((student&stable).any()) if student_reference else None,
            held_completed_ticks=int((student&held).sum()) if student_reference else None,
            old_stable_completed_ticks=int((student&stable).sum()) if student_reference else None,
            native_near_goal=bool((student&near).any()),native_alignment=bool((student&align).any()),
            native_subgoal_events=int((student&stage_events).sum()),native_max_subgoals=int(after_stage[student].max()),
            genuine_autonomous_success=row['student_autonomous_success'])
        stages=[]
        for stage in sorted(set(map(int,before_stage))):
            scope=before_stage==stage;v=scope&valid
            stages.append(dict(native_goal_index_before_action=stage,all_controls=int(scope.sum()),
                student_controls=int((scope&student).sum()),expert_controls=int((scope&~student).sum()),legal_expert_controls=int(v.sum()),
                expert_flush_masked=int((scope&~student&~valid).sum()),
                legal_expert_any_contact=int((v&any_contact).sum()),legal_expert_two_fingers=int((v&two).sum()),
                legal_expert_held12=int((v&held).sum()),legal_expert_old_stable12=int((v&stable).sum()),
                legal_expert_native_near_goal=int((v&near).sum()),legal_expert_native_alignment=int((v&align).sum()),
                legal_expert_native_lift=int((v&lift).sum()),native_subgoal_events=int((scope&stage_events).sum()),
                legal_expert_first_completed_tick=first(v),legal_expert_last_completed_tick=int(np.flatnonzero(v)[-1]+1) if v.any() else None,
                best_keypoint_legal_expert=minimum(v&same,kp),goal_minus_hole_at_stage_first_action_w_m=(goal[scope][0,:3]-hole[scope][0,:3]).tolist()))
        row.update(labels=labels,phases=phases,trigger_replay=replay,student=student_counts,
            snapshots=snapshots,best_held_student_keypoint=sbest,best_student_keypoint=minimum(student&same,kp),
            student_held_intervals=intervals(student&held) if student_reference else None,
            student_stable_intervals=intervals(student&stable) if student_reference else None,
            settled_reference_authority='student_only' if student_reference else 'includes_expert_ticks',
            stage_coverage=stages,native_subgoal_transitions=[state(int(i)) for i in np.flatnonzero(stage_events)],
            terminal=state(n-1),expert_held_intervals=intervals(valid&held))
    row.update(numeric_sha256=sha256(directory/'numeric.h5'),capture_receipt_sha256=sha256(directory/'capture_complete.json'),
        capture_source_sha256=capture['dataset_capture']['sha256'],capture_payload_rehashed=False)
    if replay is not None:assert replay['numeric_sha256']==row['numeric_sha256']
    assert sha256(directory/'result.json')==row['result_sha256']
    return row

rows={role:[audit(c) for c in cs] for role,cs in selected.items()}
comparisons=[]
for ca,cb,ra,rb in zip(selected['A'],selected['B'],rows['A'],rows['B']):
    assert ca['spec']['seed']==cb['spec']['seed'] and ca['spec']['family']==cb['spec']['family']
    for key in ('initial_configuration_sha256','source_identity','narrow_reset'):assert ca['spec'][key]==cb['spec'][key]
    ia=read(ROOT/ca['name']/'initial_observable.json');ib=read(ROOT/cb['name']/'initial_observable.json')
    assert ia.keys()==ib.keys()
    fields={}
    for key in ia:
        a,b=np.asarray(ia[key]),np.asarray(ib[key]);assert a.shape==b.shape
        fields[key]=dict(equal=bool(np.array_equal(a,b)),max_abs_difference=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0)
    comparisons.append(dict(seed=ca['spec']['seed'],family=ca['spec']['family'],schedule=ca['expansion_schedule'],
        registered_initial_configuration_sha256=ca['spec']['initial_configuration_sha256'],same_saved_observable_state=all(v['equal'] for v in fields.values()),
        initial_fields=fields,A=dict(case=ra['case'],legal=ra['legal'],accepted=ra['accepted'],student=ra.get('student'),
            takeover=ra.get('snapshots',{}).get('takeover'),first_held=ra.get('snapshots',{}).get('first_student_held')),
        B=dict(case=rb['case'],legal=rb['legal'],accepted=rb['accepted'],student=rb.get('student'),
            takeover=rb.get('snapshots',{}).get('takeover'),first_held=rb.get('snapshots',{}).get('first_student_held'))))

def group(values):
    legal=[r for r in values if r['legal']];window=[r for r in legal if r['student']['held12'] is not None]
    return dict(opportunities=len(values),legal=len(legal),illegal=len(values)-len(legal),
        expert_takeovers=sum(r.get('labels',{}).get('expert_ticks',0)>0 for r in values),
        expert_recoveries=sum(r['expert_recovery_success'] for r in values),accepted=sum(r['accepted'] for r in values),
        accepted_legal_expert_actions=sum(r['labels']['legal_expert_ticks'] for r in values if r['accepted']),
        student_native_lift_cases=sum(r['student']['native_lift'] for r in legal),
        student_held_window_available_cases=len(window),student_ever_held_cases=sum(r['student']['held12'] for r in window),
        student_ever_old_stable_cases=sum(r['student']['old_stable12'] for r in window),
        student_native_near_goal_cases=sum(r['student']['native_near_goal'] for r in legal),
        student_native_alignment_cases=sum(r['student']['native_alignment'] for r in legal),
        student_subgoal_events=sum(r['student']['native_subgoal_events'] for r in legal),
        student_autonomous_successes=sum(r['student_autonomous_success'] for r in values),
        distinct_families=len({r['family'] for r in values}),accepted_distinct_families=len({r['family'] for r in values if r['accepted']}))

report=dict(version='closed_B_first16_with_matched_A_numeric_review_v1',started_unix=started,finished_unix=time.time(),
    root=str(ROOT),scope='Exactly registered B seeds28600000..28600015, closed wave4; same-family closed A wave0 comparison. Active B wave5 and later untouched.',
    cohort_complete=True,whole_collection_complete=False,
    B_summary=group(rows['B']),A_same16_summary=group(rows['A']),
    B_by_schedule={s:group([r for r in rows['B'] if r['schedule']==s]) for s in sorted({r['schedule'] for r in rows['B']})},
    B_held_classifications=dict(Counter(r['held_recheck']['category'] for r in rows['B'] if 'held_recheck' in r)),
    B_cases=rows['B'],A_same16_cases=rows['A'],same_family_observed_comparison=comparisons,
    fixed80_865_case=next(r['case'] for r in rows['B'] if r['seed']==28600012),source_pairs=plan['pairs'],
    evidence=dict(script_sha256=sha256(__file__),plan_sha256=sha256(ROOT/'plan.json'),
        results_read_snapshot_sha256=hashlib.sha256(raw).hexdigest(),results_read_snapshot_rows=len(index),
        accepted_index_read_snapshot_sha256=hashlib.sha256(entry_raw).hexdigest(),
        prior_A64_review_sha256=sha256(HERE.parent/'collection09_A64_review_01/closed_A64.json')),
    definitions=['A saved seed/config identity is not a physical/hidden-state match; compare actual saved initial observable fields separately. RGB payloads are not read, so image equality is not claimed.',
        'Completed transition N is the boundary for expert action index N; its first completed expert transition is N+1. Two queue-flush transitions remain unsupervised.',
        'Held12 is latest12 completed transitions with >=2 fingers above0.1N in both physics samples and height >=median completed25..48+20mm; fixed40 has no student-only reference and is reported unavailable.',
        'Old stable additionally restricts palm-relative translation and orientation. Its absence is neither native drop nor a declaration of unstable holding.',
        'Native lift, collection held12, old stable, native keypoint near/alignment, subgoal and expert recovery are distinct measurements.',
        'Pose/angle uses post-transition object rigid root against the pre-transition current goal root; stage-change flags are explicit. Quaternion angle ignores object symmetries; native keypoints remain authoritative.',
        'Goal0 root is186mm above the physical hole root. Root proximity to goal0 is not proximity to the hole mouth or task alignment.',
        'Stage coverage is assigned by PRE-action native goal index, including transition-to-next-goal action. Contact/held/stable/near columns overlap and must not be added.',
        'Remaining stage time assumes no subgoal reset or earlier termination. Real subgoals reset native601-step clocks; total865 is not a relaxed timeout.',
        'Accepted expert labels are data availability, not optimizer exposure or autonomous successes. This completed slice is not a final whole-collection rate.'],
    simulator_starts=0,optimizer_updates=0,active_numeric_or_capture_files_opened=0,rgb_payloads_opened=0,rgb_hashes_recomputed=0)
with OUT.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:report[k] for k in ('B_summary','A_same16_summary','B_by_schedule','B_held_classifications')},indent=2))
print('elapsed_seconds',report['finished_unix']-started)
