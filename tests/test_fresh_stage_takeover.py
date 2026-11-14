import copy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tools import tight_fresh_stage_takeover as stage
from visuotactile.compact import sha256
from visuotactile.takeover_trigger import StableGraspTrigger, VERSION, CONTACT_HEIGHT_VERSION


def data():
    ids = [f'original{i}' for i in range(30)]
    return dict(normalization_train_ids=ids, splits=dict(train=ids+['correction'], validation=['dev'], test=['test']),
        entries={k: dict(family=k) for k in ids})


@pytest.mark.parametrize('identity', ['dev', 'test', 'correction', 'unknown'])
def test_only_original_training_families(identity):
    with pytest.raises(ValueError, match='original train30'):
        stage.validate_requests(data(), [identity], [0])


def test_opportunities_and_delays_are_explicit():
    stage.validate_requests(data(), ['original1', 'original2'], [0, 12])
    for ids, delays in [(['original1']*2, [0, 12]), (['original1'], []), (['original1'], [24]), (['original1'], [True])]:
        with pytest.raises(ValueError): stage.validate_requests(data(), ids, delays)
    changed = data(); changed['entries']['original2']['family'] = 'original1'
    with pytest.raises(ValueError, match='different original family'):
        stage.validate_requests(changed, ['original1', 'original2'], [0, 12])


def sample(tick, stable):
    pose=np.array([0.,0.,.53 if stable and tick >= 48 else .5,1.,0.,0.,0.])
    palm=pose.copy(); palm[0]=.03
    force=np.zeros((2,5,3));force[:,:2,2]=.2
    return pose,palm,force


def result_and_audit(trigger, switch, success=False, valid=0):
    result=dict(legal=True,software_complete=True,dagger=dict(expert_takeover_success=success and switch is not None,
        student_autonomous_success=success and switch is None,expert_ticks=0 if switch is None else 30,
        expert_valid_ticks=valid))
    audit=dict(category='ended_before_takeover' if switch is None else 'student_stable_then_takeover',
        actual_switch_tick=switch,receipt=trigger.receipt())
    return result,audit


def test_unstable_natural_termination_never_switches_or_creates_labels():
    trigger=StableGraspTrigger(dict(version=VERSION,delay_ticks=0),stage.NEVER_FALLBACK)
    for tick in range(601): assert trigger.observe(tick,*sample(tick,False)) is None
    result,audit=result_and_audit(trigger,None)
    assert stage.classify(result,audit)=='ended_without_student_stable_window'
    result['dagger']['expert_valid_ticks']=1
    with pytest.raises(ValueError,match='expert authority or labels'):stage.classify(result,audit)


@pytest.mark.parametrize('delay',[0,12])
def test_stable_switch_preserves_causal_tick_and_delayed_labels(delay):
    trigger=StableGraspTrigger(dict(version=VERSION,delay_ticks=delay),stage.NEVER_FALLBACK)
    for tick in range(60+delay):
        issued=trigger.observe(tick,*sample(tick,True))
        if tick==59: assert issued==60+delay
    result,audit=result_and_audit(trigger,60+delay,True,28)
    assert stage.classify(result,audit)=='expert_rescue_after_stable'
    audit['actual_switch_tick']+=1
    with pytest.raises(ValueError,match='registered causal'):stage.classify(result,audit)


def test_stable_but_terminates_before_delayed_switch():
    trigger=StableGraspTrigger(dict(version=VERSION,delay_ticks=12),stage.NEVER_FALLBACK)
    for tick in range(62):trigger.observe(tick,*sample(tick,True))
    result,audit=result_and_audit(trigger,None)
    assert stage.classify(result,audit)=='ended_after_stable_before_scheduled_takeover'


def test_bound_guard_handles_subgoal_timeouts_and_rejects_reachable_sentinel():
    env=SimpleNamespace(env_max_goals=np.array([2]),max_episode_length=600)
    assert stage.native_bound_guard(env)['collector_max_control_ticks']==2404
    env.max_episode_length=stage.NEVER_FALLBACK
    with pytest.raises(ValueError,match='strictly exceed'):stage.native_bound_guard(env)


def test_reject_actual_fallback_even_if_expert_rescued():
    trigger=StableGraspTrigger(dict(version=VERSION,delay_ticks=0),stage.NEVER_FALLBACK)
    result,audit=result_and_audit(trigger,601,True,20)
    audit['category']='fallback_without_stable_grasp'
    with pytest.raises(ValueError,match='fallback is forbidden'):stage.classify(result,audit)


@pytest.mark.parametrize('version', [VERSION, CONTACT_HEIGHT_VERSION])
def test_real_prepare_checks_data_hashes_pairs_session_and_preserves_identity(tmp_path,monkeypatch,version):
    source=tmp_path/'source';source.mkdir();session_root=tmp_path/'session';session_root.mkdir()
    (session_root/'session.json').write_text(json.dumps(dict(deadline=10**12)))
    original=tmp_path/'expert.h5'
    with h5py.File(original,'w') as f:
        f.attrs['result_json']=json.dumps(dict(success=True,seed=9,narrow_reset={'x':1}))
    values=data()
    for entry in values['entries'].values():entry.update(path=str(original),source_sha256=sha256(original))
    for role in ('A','B'):
        folder=source/role/'state_data';folder.mkdir(parents=True)
        (folder/'data.json').write_text(json.dumps(values))
    pair=dict(frontend='current.pt',frontend_sha256='front',backend='rear.pt',backend_sha256='rear')
    monkeypatch.setattr(stage,'completed_pair',lambda root,role:dict(pair,role=role))
    output=tmp_path/'collection'
    plan=stage.prepare(source,output,['original1','original2'],[0,12],5,999,session_root=session_root,
        trigger_version=version, first_plan_forensics=True)
    stage.validate_plan(plan)
    assert [c['role'] for c in plan['cases']]==['A','B','B','A']
    assert [c['spec']['event_seed'] for c in plan['cases']]==[999,999,1000,1000]
    assert all(c['spec']['split']=='train' and c['spec']['takeover_tick']==stage.NEVER_FALLBACK for c in plan['cases'])
    assert plan['stage_takeover_contract']==stage.trigger_contract(version)[0]
    assert all(c['dagger']['trigger']['version']==version for c in plan['cases'])
    assert plan['first_plan_forensics'] is True
    assert sha256(output/'session.json')==plan['source_session_sha256']
    before=(output/'plan.json').read_bytes()
    with pytest.raises(FileExistsError):stage.prepare(source,output,['original1'],[0],5,999,session_root=session_root)
    assert (output/'plan.json').read_bytes()==before
    original.write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):
        stage.prepare(source,tmp_path/'new',['original1'],[0],5,999,session_root=session_root)
    assert not (tmp_path/'new').exists()


def test_run_guards_real_environment_before_any_trial_and_restores_dispatch(tmp_path,monkeypatch):
    from tools import tight_fresh_rollouts
    from visuotactile import state_benchmark
    session=tmp_path/'session.json';session.write_text('{}')
    case=dict(name='one',spec=dict(takeover_tick=stage.NEVER_FALLBACK),
        dagger=dict(trigger=dict(version=VERSION,delay_ticks=0)))
    plan=dict(stage_takeover_contract=stage.CONTRACT,never_fallback=stage.NEVER_FALLBACK,
        cases=[case],source_data_sha256={},source_session_sha256=sha256(session))
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    entered=[]
    def original(*args,**kwargs):entered.append(True)
    monkeypatch.setattr(state_benchmark,'trial',original)
    def existing(output):
        state_benchmark.trial(SimpleNamespace(env_max_goals=np.array([2]),
            max_episode_length=stage.NEVER_FALLBACK),None,output,case['spec'])
    monkeypatch.setattr(tight_fresh_rollouts,'run',existing)
    with pytest.raises(ValueError,match='strictly exceed'):stage.run(tmp_path)
    assert not entered and state_benchmark.trial is original
