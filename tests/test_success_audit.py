import ast
from pathlib import Path

import numpy as np
import torch

from visuotactile.success_audit import reconstruct, keypoint_distance,terminal_geometry
from tools.anchor_interim_eval import random_specs, wilson, initial_signature


def config():
    return dict(precise_assembly=dict(goal_mode='preInsertAndFinal',random_goal_fraction=0,
        insertion_success_tolerance=.01,enable_retract=True,retract_success_tolerance=.005,retract_distance_threshold=.1),
        termination=dict(success_steps=10,force_consecutive_near_goal_steps=False),
        reward=dict(keypoint_scale=1.5,fixed_size_keypoint_reward=True,fixed_size=[.141,.03025,.0271]))


def test_visual_alignment_or_single_subgoal_is_not_full_success():
    c=config();d=np.array([.01]*10+[.03]*3+[.006]*10+[.006]*2)
    fingers=np.array([.05]*23+[.11]*2)
    r=reconstruct(d,fingers,c,2)
    assert r['subgoals'][9]==1 and r['subgoals'][22]==2
    assert not r['complete'][:23].any()
    assert r['complete'][23:].all()


def test_native_accumulation_is_not_silently_changed_to_consecutive():
    source=Path('isaacsimenvs/tasks/play/utils/reward_utils.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='update_near_goal_steps')
    namespace={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'native_counter','exec'),namespace)
    c=config();inside=[True,False]*10;native=torch.tensor([0])
    for near in inside:native=namespace['update_near_goal_steps'](torch.tensor([near]),native,False)
    r=reconstruct(np.where(inside,.01,.03),np.full(20,.05),c,2)
    assert native.item()==10 and r['subgoals'][-1]==1
    c['termination']['force_consecutive_near_goal_steps']=True
    assert reconstruct(np.where(inside,.01,.03),np.full(20,.05),c,2)['subgoals'][-1]==0


def test_retract_requires_both_tighter_geometry_and_fingers_clear():
    c=config();d=np.array([.006]*20+[.008,.006]);f=np.array([.05]*20+[.2,.2])
    r=reconstruct(d,f,c,2)
    assert not r['complete'][:-1].any() and r['complete'][-1]


def test_keypoints_include_orientation_not_just_object_center():
    c=config();obj=np.array([[0,0,0,1,0,0,0]],np.float32);goal=obj.copy()
    assert keypoint_distance(obj,goal,c['reward'])[0]==0
    goal[0,3:]=[0,0,0,1]
    assert keypoint_distance(obj,goal,c['reward'])[0]>.2


def test_random_cohorts_are_fixed_unique_and_exclude_test():
    data=dict(splits=dict(train=[f't{i}' for i in range(320)],validation=[f'v{i}' for i in range(40)],test=[f'x{i}' for i in range(40)]))
    a=random_specs(data,20261105);b=random_specs(data,20261105)
    assert a==b and len(a)==240 and len({r['seed'] for r in a})==240
    assert sum(not r['reserve'] for r in a)==200
    assert len({r['identity'] for r in a[:120]})==120
    assert not {r['identity'] for r in a}&set(data['splits']['test'])
    assert {r['cohort'] for r in a[:100]}=={'train','validation'}


def test_zero_success_ci_and_empty_denominator_are_honest():
    assert wilson(0,0) is None
    low,high=wilson(0,100);assert abs(low)<1e-15 and .0369<high<.0371
    low,high=wilson(10,100);assert .055<low<.056 and .174<high<.175


def test_initial_signature_is_lane_translation_invariant():
    values=dict(base_pose_w=[0,0,0,1,0,0,0],object_pose_w=[.2,.3,.1,1,0,0,0],
        hole_pose_w=[.3,.1,.2,1,0,0,0],joint_position_rad=[0]*29)
    moved={k:list(v) for k,v in values.items()}
    for k in ('base_pose_w','object_pose_w','hole_pose_w'):moved[k][0]+=8
    assert initial_signature(values)==initial_signature(moved)
    moved['joint_position_rad'][0]=.1
    assert initial_signature(values)!=initial_signature(moved)


def test_missing_software_result_does_not_enter_task_denominator(tmp_path,monkeypatch):
    from tools import anchor_interim_eval as module
    from tools.runtime import write,read
    monkeypatch.setattr(module,'CODE',tmp_path)
    (tmp_path/'research_docs').mkdir();(tmp_path/'research_docs/STATE_DP_RESULTS.md').write_text('# Preserved report\n\nPrior evidence.\n')
    root=tmp_path/'run';root.mkdir()
    write(root/'session.json',dict(kit_maximum_starts=243,sampling_seed=20261105,source_commit='test'))
    write(root/'status.json',dict(state='blocked',stage='random_historical_000'))
    write(root/'checkpoint.json',dict(sha256='frozen'))
    write(root/'attempts.json',[dict(name='random_historical_000',exit_code=-6,valid=False)])
    owner=module.RandomEvaluation(root);owner.report();owner.report()
    report=read(root/'summary.json')
    assert report['software_missing']==1 and report['valid']==0 and report['pending_valid']==200
    assert all(r['success_rate'] is None and r['ci95'] is None for r in report['groups'])
    text=(tmp_path/'research_docs/STATE_DP_RESULTS.md').read_text()
    assert text.count('<!-- RANDOM_EVALUATION_START -->')==1 and 'Prior evidence.' in text


def test_final_pose_diagnostic_does_not_promote_skipped_native_stages():
    c=config();obj=np.array([[0,0,.02,1,0,0,0]]*19,np.float32)
    hole=np.array([[0,0,0,1,0,0,0]]*19,np.float32)
    tips=np.repeat(obj[:,None,:3],5,axis=1);tips[:,:,0]+=.12
    state=dict(_insert_pos_rel=torch.tensor([[0,0,.06],[0,0,.02]]),
        _insert_quat_wxyz=torch.tensor([[1.,0,0,0],[1.,0,0,0]]))
    diagnosis,_,mask=terminal_geometry(obj,hole,tips,state,c)
    assert diagnosis['ever_satisfied'] and diagnosis['longest_consecutive_ticks']==19
    assert diagnosis['first_tick']==1 and diagnosis['changes_native_success_or_termination'] is False
    native=reconstruct(np.full(19,.04),np.full(19,.12),c,2)
    assert not native['complete'].any() and native['subgoals'].max()==0
    tips[:,:,0]=.05
    assert not terminal_geometry(obj,hole,tips,state,c)[0]['ever_satisfied']


def test_rollout_result_helpers_import_before_kit_and_keep_latency_schema():
    from visuotactile.state_benchmark import distribution,plan_metrics
    result=distribution([1.,3.,float('nan')])
    assert result['count']==2 and result['p50']==result['mean']==2.
    assert distribution([])['mean'] is None
    assert plan_metrics(np.zeros((2,24,29),np.float32))['arm_overlap_rms_rad']==0


def test_random_owner_clears_pid_on_real_exit(tmp_path):
    from tools.anchor_interim_eval import RandomEvaluation
    from tools.runtime import write,read
    write(tmp_path/'session.json',{});write(tmp_path/'status.json',dict(pid=123,worker_pid=456))
    RandomEvaluation(tmp_path).status(state='blocked',pid=None,worker_pid=None)
    assert read(tmp_path/'status.json')['pid'] is None
