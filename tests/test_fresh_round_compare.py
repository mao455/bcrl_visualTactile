"""Fail-closed provenance and same-observation held-angle diagnostics, CPU only."""
import hashlib
import copy
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

from tools import tight_fresh_round_compare as review


def test_registered_source_drift_is_rejected(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    source = tmp_path / 'source.py'
    source.write_text('value = 1\n')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fixture'], cwd=tmp_path, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tmp_path, text=True).strip()
    plan = dict(source_commit=commit, source_sha256={'source.py': review.sha256(source)})
    assert review.check_sources(plan, tmp_path)['verified_source_files'] == 1
    source.write_text('value = 2\n')
    with pytest.raises(ValueError, match='Frozen file changed'):
        review.check_sources(plan, tmp_path)
    plan['source_commit'] = '0' * 40
    with pytest.raises(ValueError, match='Frozen commit differs'):
        review.check_sources(plan, tmp_path)


def test_angles_use_same_tick_and_active_first_goal(tmp_path, monkeypatch):
    # A later goal has a smaller angle, but must not enter first-goal comparison.
    pose = np.array([[0, 0, 0, 2, 0, 0, 0],
                     [1, 0, 0, 1, 0, 0, 1],
                     [2, 0, 0, 1, 0, 0, 0]], dtype=float)
    goal = np.array([[0, 0, 0, 1, 0, 0, 0]] * 3, dtype=float)
    with h5py.File(tmp_path / 'numeric.h5', 'w') as stream:
        stream['episode/transition/diagnostics_after/object_pose_w'] = pose
        stream['episode/transition/diagnostics_before/goal_pose_w'] = goal
        stream['episode/transition/diagnostics_before/successes'] = [0, 0, 1]
        stream['episode/transition/object_contact_force_w'] = np.zeros((3, 2, 5, 3))
        stream['episode/transition/native_keypoints_max_dist'] = [.3, .1, 0]
        stream['episode/transition/native_success_diagnostic/insertion_tolerance_m'] = [.015] * 3
    monkeypatch.setattr(review, 'causal_contact_height', lambda p, f: (np.ones(3, dtype=bool), 0.), raising=False)
    value = review.held_orientation(tmp_path)
    assert value['eligible_contact_height_windows'] == 2
    assert value['smallest_root_angle']['completed_tick'] == 1
    assert value['smallest_root_angle']['root_angle_rad'] == 0  # Quaternion scale is normalized.
    assert value['smallest_root_angle']['native_keypoint_distance_m'] == .3  # Not another tick's minimum.
    monkeypatch.setattr(review, 'causal_contact_height', lambda p, f: (np.zeros(3, dtype=bool), 0.))
    assert review.held_orientation(tmp_path) == {'eligible_contact_height_windows': 0}


def test_existing_analysis_is_never_overwritten(tmp_path):
    (tmp_path / 'autonomy.json').write_text('{}')
    with pytest.raises(FileExistsError, match='Preserve previous'):
        review.run(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)


def test_explicit_paths_and_json_digest(tmp_path):
    arguments = review.parser().parse_args([
        '--source-root', 'source', '--candidate-root', 'candidate',
        '--source-tree', 'old', '--candidate-tree', 'new', '--output', 'report'])
    assert str(arguments.candidate_tree) == 'new'
    destination = tmp_path / 'report.json'
    review.atomic_json(destination, {'value': .015})
    assert json.loads(destination.read_text()) == {'value': .015}
    assert review.sha256(destination) == hashlib.sha256(destination.read_bytes()).hexdigest()


def batch_plan(count=8):
    cases=[]
    for index in range(count):
        for role in ('A','B'):
            cases.append(dict(name=f'{role}_{index}',role=role,
                spec=dict(identity=str(index),family=f'dev_{index}',seed=index,event_seed=index+100,
                    split='development',collision_gate='live_collision_mesh_bounds_v1',narrow_reset=dict(x=index))))
    return dict(kind='autonomous',test_used=False,cases=cases,pairs={'A':{},'B':{}},
        batch=dict(version='whole_wave_complete_student_v1',lanes=8,vision_microbatch=8,
            seed=1000,neural_batch=True),inference_scheduler_index_mode='host',
        execution_backend=dict(version='tight_execution_backend_v2',physics_device='cpu',
            camera_backend='standard',kit_tasking_threads=16,physics_threads=2,
            renderer_descriptor_sets=360000,renderer_reserved_descriptors=900000))


@pytest.mark.parametrize('change', ['spec','lanes','microbatch','backend','seed','tail','inactive_render','scheduler','duplicate','split'])
def test_explicit_comparison_rejects_mismatched_opportunities_or_execution(change):
    old=batch_plan();new=copy.deepcopy(old)
    if change=='spec':new['cases'][0]['spec']['seed']+=1
    if change=='lanes':new['batch']['lanes']=16
    if change=='microbatch':new['batch']['vision_microbatch']=2
    if change=='backend':new['execution_backend']['physics_device']='cuda:0'
    if change=='seed':new['batch']['seed']+=1
    if change=='tail':new['batch']['vision_tail_profile']={'version':'current_observation_exact_tail_v1'}
    if change=='inactive_render':new['batch']['inactive_render_profile']={'version':'standard_inactive_render_products_v1'}
    if change=='scheduler':new['inference_scheduler_index_mode']='legacy'
    if change=='duplicate':new['cases'][1]['name']=new['cases'][0]['name']
    if change=='split':new['cases'][0]['spec']['split']='train'
    with pytest.raises(ValueError):review.compare_plans(old,new)


def test_new_eight_per_role_and_old_four_defaults():
    assert review.compare_plans(batch_plan(),batch_plan())['opportunities_per_role']=={'A':8,'B':8}
    serial=batch_plan(4);serial.pop('batch');serial.pop('execution_backend')
    assert review.compare_plans(serial,copy.deepcopy(serial))['execution_identity']['mode']=='serial'
    args=review.parser().parse_args(['--source-root','session','--candidate-root','round_08',
        '--source-development','session/development_source08','--candidate-development','round_08/development',
        '--source-tree','old','--candidate-tree','new','--output','review'])
    assert str(args.source_development)=='session/development_source08'


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def write_clock(path,steps=2):
    with h5py.File(path/'numeric.h5','w') as f:
        times=np.arange(steps,dtype=np.float64)/60
        f['episode/obs/time_s']=times
        f['episode/obs/camera_time_s']=np.repeat(times[:,None],3,axis=1)
        f['episode/obs/camera_frame']=np.repeat(np.arange(steps)[:,None],3,axis=1)
        f['episode/obs/camera_valid']=np.ones((steps,3),dtype=bool)
        f['episode/obs/actuator_target_rad']=np.zeros((steps,29))
        f['episode/transition/target_after_rad']=np.zeros((steps,29))
        f['episode/transition/time_before_s']=times
        f['episode/transition/time_after_s']=times+1/60


@pytest.mark.parametrize('fault',['camera','history','dimension','timestamp'])
def test_numeric_real_clock_and_target_continuity(tmp_path,fault):
    write_clock(tmp_path)
    assert review.audit_numeric_clock(tmp_path,2)['camera_clock_verified']
    with h5py.File(tmp_path/'numeric.h5','a') as f:
        if fault=='camera':f['episode/obs/camera_frame'][1]=0
        if fault=='history':f['episode/obs/actuator_target_rad'][1,0]=1
        if fault=='dimension':
            del f['episode/transition/target_after_rad'];f['episode/transition/target_after_rad']=np.zeros((2,28))
        if fault=='timestamp':f['episode/obs/camera_time_s'][1,2]=0
    with pytest.raises(ValueError):review.audit_numeric_clock(tmp_path,2)


def test_observable_equality_is_reported_without_hidden_state_claim(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';cases=[dict(name='A_0')]
    for root in (a,b):write_json(root/'A_0/initial_observable.json',dict(q=[[0.,1.]]))
    assert review.initial_observation_comparison(a,b,cases)['all_cases_equal']
    write_json(b/'A_0/initial_observable.json',dict(q=[[0.,1.0001]]))
    result=review.initial_observation_comparison(a,b,cases)
    assert not result['all_cases_equal'] and result['cases']['A_0']['fields']['q']['max_absolute_difference']>0
    assert 'hidden PhysX' in result['scope']


def audited_group_fixture(tmp_path,monkeypatch):
    root=tmp_path/'resource_session';directory=root/'development_source08';directory.mkdir(parents=True)
    plan=batch_plan();roots={};pairs={}
    for role,stage in [('A','round_05'),('B','round_06')]:
        trained=tmp_path/stage;roots[role]=trained
        front=trained/role/'training/best.pt';back=trained/role/'backend'/role/'training/best.pt'
        front.parent.mkdir(parents=True);back.parent.mkdir(parents=True)
        front.write_bytes(role.encode());back.write_bytes((role+'rear').encode())
        pairs[role]=dict(frontend=str(front),backend=str(back),frontend_sha256=review.sha256(front),
            backend_sha256=review.sha256(back),residual_version='rear_v1')
    plan['pairs']=pairs;plan['resource_root']=str(root);write_json(directory/'plan.json',plan)
    write_json(directory/'run_started.json',dict(plan_sha256=review.sha256(directory/'plan.json')))
    write_json(directory/'result.json',dict(software_complete=True,worker_error=None,cases=16))
    write_json(directory/'initialization.json',dict(lanes=8,execution_backend=plan['execution_backend'],vision_microbatch=8))
    write_json(directory/'execution_backend_observed.json',dict(requested=dict(configuration=plan['execution_backend'],num_envs=8),
        physics_device='cpu',camera_backend='standard',num_envs=8,physics_dt=1/120,control_dt=1/60))
    rows=[]
    for case in plan['cases']:
        role=case['role'];lane=int(case['spec']['identity']);wave=0 if role=='A' else 1
        path=directory/case['name'];pair=pairs[role]
        batch=dict(wave=wave,lane=lane,lanes=8,vision_microbatch=8,physics_seed=1000+wave,event_seed=101000+wave)
        write_json(path/'started.json',dict(spec=case['spec'],batch_execution=batch))
        write_json(path/'metadata.json',dict(batch_execution=batch,policy_checkpoint_sha256=pair['backend_sha256'],
            execution_backend=plan['execution_backend']))
        result=dict(**case['spec'],software_complete=True,legal=True,steps=2,success=False,
            checkpoint_sha256=pair['backend_sha256'],residual_version='rear_v1',residual_ticks=2,
            simulation_step_hz=60,physics_hz=120,plan_hz=5,action_control_mode='absolute_joint_position_target_rad',
            expert_shadow_computed=False,complete_student_policy=pair,batch_execution=batch,native_lift=False,native_subgoals=0,
            native_inference=dict(summary={key:dict(wall_s=dict(count=1 if key=='ddim_plan' else 2)) for key in
                ['dino_rgb','learned_visual_and_history','residual_forward_compose','ddim_plan']}))
        write_json(path/'result.json',result);write_clock(path)
        write_json(path/'initial_observable.json',dict(q=[[0.,1.]]))
        rows.append(dict(case=case['name'],role=role,result=result,timing=dict(total_seconds=5.)))
    write_json(directory/'results.json',rows)
    calls=[]
    def completed(actual,role):
        calls.append((actual,role));assert actual==roots[role];return pairs[role]
    monkeypatch.setattr(review,'completed_pair',completed,raising=False)
    monkeypatch.setattr(review,'check_sources',lambda *a:{})
    monkeypatch.setattr(review,'phase_case',lambda path:dict(case=path.name,legal=True,steps=2,success=False,native_lift=False,
        native_subgoals=0,authority=dict(expert_ticks=0),settled_height_stable_grasp=False,
        native_alignment_first_tick=None,native_near_goal_first_tick=None,reasons=dict(timeout=True)),raising=False)
    monkeypatch.setattr(review,'contact_motion',lambda *a:dict(first_contact_height_window_end_tick=None),raising=False)
    monkeypatch.setattr(review,'held_orientation',lambda *a:{})
    return root,directory,plan,roots,calls,rows


def test_batch_audit_uses_each_pair_training_root_and_real_lane_receipts(tmp_path,monkeypatch):
    root,directory,plan,roots,calls,rows=audited_group_fixture(tmp_path,monkeypatch)
    value=review.audit_group('source',root,tmp_path,directory)
    assert calls==[(roots['A'],'A'),(roots['B'],'B')]
    assert not (root/'A').exists() and value['totals']['A']['opportunities']==8
    assert value['totals']['A']['overlapping_lane_lifetime_seconds']==40
    rejected=next(case for case in plan['cases'] if case['name']=='A_3');path=directory/'A_3'
    result=dict(**rejected['spec'],software_complete=True,legal=False,success=False,not_started=True,
        reason='pre-action collision rejection; no outcome replacement',batch_execution=read_json(path/'started.json')['batch_execution'])
    write_json(path/'result.json',result);(path/'numeric.h5').unlink()
    write_json(path/'initial_observable.json',dict(q=[[0.,1.]]))
    write_json(path/'initial_collision.json',dict(legal=False,version='live_collision_mesh_bounds_v1',minimum_gap_m=-.005))
    next(row for row in rows if row['case']=='A_3')['result']=result
    write_json(directory/'results.json',rows)
    with pytest.raises(ValueError,match='Not complete autonomous'):review.audit_group('source',root,tmp_path,directory)
    explicit=review.audit_group('source',root,tmp_path,directory,allow_illegal=True)
    assert explicit['totals']['A']['opportunities']==8 and explicit['totals']['A']['legal_initializations']==7
    assert explicit['totals']['A']['invalid_initializations']==1
    assert explicit['totals']['A']['termination_reasons']=={'timeout':7}
    assert explicit['totals']['A']['raw_successes_per_registered_opportunity']['denominator']==8
    assert next(row for row in explicit['cases'] if row['case']=='A_3')['initial_rejection']['task_outcome'] is None
    # A single mismatched lane's receipt must invalidate a complete-looking index.
    wrong=directory/'B_7/started.json';receipt=json.loads(wrong.read_text())
    receipt['batch_execution']['lane']=0;write_json(wrong,receipt)
    with pytest.raises(ValueError,match='Per-lane execution'):review.audit_group('source',root,tmp_path,directory,allow_illegal=True)


def test_actual_selected_and_last_checkpoints_are_distinct_cpu_evidence(tmp_path):
    import torch
    front=tmp_path/'A/training/best.pt';front.parent.mkdir(parents=True)
    back=tmp_path/'A/backend/A/training/best.pt';back.parent.mkdir(parents=True);back.write_bytes(b'rear')
    for filename,epoch,updates in [('best.pt',3,306),('last.pt',10,1020)]:
        torch.save(dict(epoch=epoch,updates=updates,configuration=dict(continuation=dict(path='earlier_best'))),front.parent/filename)
    pair=dict(frontend=str(front),backend=str(back),frontend_sha256=review.sha256(front))
    write_json(front.parent/'finished.json',dict(complete=True,checkpoint_sha256=pair['frontend_sha256'],epochs=10,
        updates=1020,target_ticks=543680,anchors_seen=23730))
    result=review.frontend_checkpoint_receipt(pair,'A')
    assert result['selected']['epoch']==3 and result['last']['epoch']==10
    assert result['selected']['updates']==306 and result['last']['updates']==1020
    write_json(front.parent/'finished.json',dict(complete=True,checkpoint_sha256=pair['frontend_sha256'],epochs=10,
        updates=999,target_ticks=1,anchors_seen=1))
    with pytest.raises(ValueError,match='completion binding'):review.frontend_checkpoint_receipt(pair,'A')


def read_json(path):return json.loads(path.read_text())


def test_mismatched_legal_cohorts_keep_full_denominators_and_separate_intersection():
    def row(role,index,legal,success=False):
        value=dict(case=f'{role}_{index}',role=role,legal=legal)
        if legal:value.update(success=success,native_lift=False,settled_height_stable_grasp=False,
            contact_motion=dict(first_contact_height_window_end_tick=None),native_alignment_first_tick=None,
            native_near_goal_first_tick=None,native_subgoals=0,reasons=dict(timeout=not success),steps=2,wall_seconds=3.)
        return value
    source=dict(execution=dict(mode='whole_wave'),cases=[row('A',0,True,True),row('A',1,True),row('A',2,False),
        row('B',0,False)])
    candidate=dict(execution=dict(mode='whole_wave'),cases=[row('A',0,True),row('A',1,False),row('A',2,True,True),
        row('B',0,False)])
    total=review.autonomy_totals(source['cases'][:3],batch=True,explicit=True)
    assert total['autonomous_success_rate_among_legal']==.5
    assert total['raw_successes_per_registered_opportunity']['ratio']==1/3
    result=review.matched_legal_comparison(source,candidate)['by_role']
    assert result['A']['matched_legal_cases']==['A_0']
    assert result['A']['source_only_legal_cases']==['A_1'] and result['A']['candidate_only_legal_cases']==['A_2']
    assert result['A']['source']['autonomous_success_rate_among_legal']==1
    assert result['A']['candidate']['autonomous_success_rate_among_legal']==0
    assert result['B']['matched_legal_opportunities']==0 and result['B']['both_invalid_cases']==['B_0']
    assert result['B']['source']['autonomous_success_rate_among_legal'] is None


@pytest.mark.parametrize('fault',['executed','contradictory_collision','success'])
def test_illegal_initialization_cannot_hide_actions_or_success(tmp_path,fault):
    result=dict(software_complete=True,legal=False,not_started=True,success=False,collision_gate='mesh_v1')
    for name,value in [('initial_collision.json',dict(legal=False,version='mesh_v1')),
                       ('initial_observable.json',dict(q=[[0.]])),('started.json',{}),('result.json',result)]:
        write_json(tmp_path/name,value)
    assert review.invalid_initialization_evidence(tmp_path,result)['task_outcome'] is None
    if fault=='executed':write_clock(tmp_path)
    if fault=='executed':
        with h5py.File(tmp_path/'numeric.h5','a') as stream:stream['episode/transition/step_id']=[0,1]
    if fault=='contradictory_collision':write_json(tmp_path/'initial_collision.json',dict(legal=True,version='mesh_v1'))
    if fault=='success':result['success']=True
    with pytest.raises(ValueError):review.invalid_initialization_evidence(tmp_path,result)


def mixed_source_fixture(tmp_path,monkeypatch):
    groups={name:audited_group_fixture(tmp_path/name,monkeypatch) for name in ('a08','b06','candidate')}
    pairs={(roots[role],role):plan['pairs'][role] for root,directory,plan,roots,_,_ in groups.values() for role in ('A','B')}
    calls=[]
    def completed(root,role):
        calls.append((root,role));return pairs[root,role]
    monkeypatch.setattr(review,'completed_pair',completed,raising=False)
    monkeypatch.setattr(review,'load_helpers',lambda tree:None)
    for name,group in groups.items():
        tree=tmp_path/name/'frozen';tree.mkdir()
        for helper in ('tools/tight_fresh_phases.py','tools/tight_fresh_visual_closeout.py',
                       'tools/tight_fresh_contact_height_coverage.py','tools/tight_fresh_learning_report.py'):
            path=tree/helper;path.parent.mkdir(exist_ok=True);path.write_text('# fixture\n')
        path=tree/'visuotactile/takeover_trigger.py';path.parent.mkdir()
        path.write_text('def stable_windows():\n    return None\n')
    manifest=tmp_path/'sources.json'
    write_json(manifest,dict(version='per_role_development_sources_v1',roles={
        role:dict(root=str(groups[name][0]),development=str(groups[name][1]),tree=str(tmp_path/name/'frozen'))
        for role,name in [('A','a08'),('B','b06')]}))
    return groups,manifest,calls


def reject_fixture_case(group,name):
    root,directory,plan,_,_,rows=group;case=next(case for case in plan['cases'] if case['name']==name)
    path=directory/name;result=dict(**case['spec'],software_complete=True,legal=False,success=False,not_started=True,
        reason='collision',batch_execution=read_json(path/'started.json')['batch_execution'])
    write_json(path/'result.json',result);(path/'numeric.h5').unlink()
    write_json(path/'initial_collision.json',dict(legal=False,version='live_collision_mesh_bounds_v1'))
    next(row for row in rows if row['case']==name)['result']=result;write_json(directory/'results.json',rows)


def test_manifest_compares_only_selected_source_roles_and_preserves_all_original_plans(tmp_path,monkeypatch):
    groups,manifest,calls=mixed_source_fixture(tmp_path,monkeypatch)
    # Distinct admission differences per source role must not become task failures.
    reject_fixture_case(groups['a08'],'A_2');reject_fixture_case(groups['b06'],'B_3')
    reject_fixture_case(groups['candidate'],'A_4')
    # Unselected source role checkpoints are not a dependency of this comparison.
    Path(groups['a08'][2]['pairs']['B']['frontend']).unlink()
    Path(groups['b06'][2]['pairs']['A']['frontend']).unlink()
    before={name:review.sha256(value[1]/'plan.json') for name,value in groups.items()}
    observed=[]
    monkeypatch.setattr(review,'write_learning',lambda out,root,candidate,source,tree,**kw:observed.append((source,candidate)))
    root,directory,plan,_,_,_=groups['candidate'];output=tmp_path/'review'
    review.run(source_manifest=manifest,candidate_root=root,candidate_development=directory,
        candidate_tree=tmp_path/'candidate/frozen',output=output)
    result=read_json(output/'autonomy.json')
    assert result['version']=='per_role_development_cpu_review_v1'
    assert set(result['source_by_role'])=={'A','B'} and 'groups' not in result
    for role,source in [('A','a08'),('B','b06')]:
        evidence=result['source_by_role'][role]
        assert set(evidence['pairs'])=={role} and len(evidence['cases'])==8
        assert evidence['development']==str(groups[source][1])
        assert evidence['totals'][role]['legal_initializations']==7
        assert observed[0][0][role]==groups[source][2]['pairs'][role]
        assert len(result['initial_observation_comparison_by_role'][role]['cases'])==8
    matched=result['matched_legal_comparison']['by_role']
    assert matched['A']['matched_legal_opportunities']==6
    assert matched['A']['candidate_only_legal_cases']==['A_2']
    assert matched['A']['source_only_legal_cases']==['A_4']
    assert matched['B']['matched_legal_opportunities']==7 and matched['B']['candidate_only_legal_cases']==['B_3']
    assert calls==[(groups['a08'][3]['A'],'A'),(groups['b06'][3]['B'],'B'),
                  (groups['candidate'][3]['A'],'A'),(groups['candidate'][3]['B'],'B')]
    assert before=={name:review.sha256(value[1]/'plan.json') for name,value in groups.items()}


def test_source_preflight_requires_no_candidate_and_never_creates_one(tmp_path,monkeypatch):
    groups,manifest,calls=mixed_source_fixture(tmp_path,monkeypatch)
    output=tmp_path/'preflight'
    review.run(source_manifest=manifest,source_preflight=True,output=output)
    report=read_json(output/'source_preflight.json')
    assert report['candidate_audited'] is False and not (output/'autonomy.json').exists()
    assert report['source_by_role']['B']['pairs']=={'B':groups['b06'][2]['pairs']['B']}
    assert calls==[(groups['a08'][3]['A'],'A'),(groups['b06'][3]['B'],'B')]


@pytest.mark.parametrize('explicit',[False,True])
def test_single_source_run_keeps_historical_output_contract(tmp_path,monkeypatch,explicit):
    groups,_,_=mixed_source_fixture(tmp_path,monkeypatch)
    source,candidate=groups['a08'],groups['candidate'];kwargs={}
    if explicit:
        kwargs.update(source_development=source[1],candidate_development=candidate[1])
    else:
        source[1].rename(source[0]/'development');candidate[1].rename(candidate[0]/'development')
    learning=[];monkeypatch.setattr(review,'write_learning',lambda *args,**kw:learning.append(kw))
    output=tmp_path/'legacy_review'
    review.run(source[0],candidate[0],tmp_path/'a08/frozen',tmp_path/'candidate/frozen',output,**kwargs)
    result=read_json(output/'autonomy.json')
    assert result['version']==('explicit_development_cpu_review_v2' if explicit else 'round06_independent_cpu_review_v1')
    assert set(result['groups'])==({'source','candidate'} if explicit else {'source_round05','candidate_round06'})
    assert 'source_by_role' not in result and learning==[dict(explicit=explicit)]


@pytest.mark.parametrize('fault',['wrong_root','outside_root','missing_role','extra_field','relative','version'])
def test_source_manifest_fails_closed_on_ambiguous_bindings(tmp_path,monkeypatch,fault):
    groups,manifest,_=mixed_source_fixture(tmp_path,monkeypatch);value=read_json(manifest)
    if fault=='wrong_root':value['roles']['A']['root']=str(tmp_path)
    if fault=='outside_root':
        root,directory,plan,*_=groups['a08'];plan['resource_root']=str(tmp_path/'a08/frozen')
        write_json(directory/'plan.json',plan);value['roles']['A']['root']=plan['resource_root']
    if fault=='missing_role':value['roles'].pop('B')
    if fault=='extra_field':value['roles']['A']['training_root']=str(tmp_path)
    if fault=='relative':value['roles']['B']['tree']='frozen'
    if fault=='version':value['version']='invented'
    write_json(manifest,value)
    with pytest.raises(ValueError):review.source_manifest_inputs(manifest)


@pytest.mark.parametrize('fault',['spec','execution','wave_order'])
def test_role_comparison_preserves_actual_wave_and_case_provenance(fault):
    old=batch_plan();new=copy.deepcopy(old)
    if fault=='spec':new['cases'][0]['spec']['event_seed']+=1
    if fault=='execution':new['batch']['vision_microbatch']=2
    if fault=='wave_order':new['cases']=sorted(new['cases'],key=lambda row:row['role'],reverse=True)
    with pytest.raises(ValueError):review.compare_role_plans(old,new,'A')


def test_role_comparison_does_not_require_unselected_role_same_specs():
    old=batch_plan();new=copy.deepcopy(old)
    next(case for case in old['cases'] if case['role']=='B')['spec']['event_seed']+=1
    assert review.compare_role_plans(old,new,'A')['opportunities']==8
    with pytest.raises(ValueError):review.compare_role_plans(old,new,'B')


@pytest.mark.parametrize('kwargs',[
    dict(source_root=Path('legacy')),dict(source_tree=Path('legacy')),dict(source_development=Path('legacy')),
    dict(source_preflight=True,candidate_root=Path('candidate'))])
def test_manifest_and_single_source_arguments_cannot_silently_override_each_other(tmp_path,kwargs):
    with pytest.raises(ValueError):review.run(source_manifest=tmp_path/'manifest.json',output=tmp_path/'out',**kwargs)


def test_manifest_cli_and_legacy_missing_argument_guard(tmp_path):
    args=review.parser().parse_args(['--source-manifest','sources.json','--source-preflight','--output','review'])
    assert args.source_root is None and args.source_tree is None and args.source_preflight
    with pytest.raises(ValueError,match='requires source/candidate'):
        review.run(candidate_root=tmp_path,output=tmp_path/'review')


@pytest.mark.parametrize('fault',['checkpoint','index','startup','partial'])
def test_selected_source_role_must_have_exact_completed_evidence(tmp_path,monkeypatch,fault):
    groups,manifest,_=mixed_source_fixture(tmp_path,monkeypatch)
    root,directory,plan,_,_,rows=groups['b06']
    if fault=='checkpoint':Path(plan['pairs']['B']['frontend']).write_bytes(b'changed')
    if fault=='index':rows[-1]['case']='not_registered';write_json(directory/'results.json',rows)
    if fault=='startup':write_json(directory/'run_started.json',dict(plan_sha256='0'*64))
    if fault=='partial':write_json(directory/'result.json',dict(software_complete=False,cases=16))
    with pytest.raises(ValueError):review.run(source_manifest=manifest,source_preflight=True,output=tmp_path/'out')
