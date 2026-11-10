"""Collector/owner compatibility and finite denominators without starting Kit."""
import json
from types import SimpleNamespace as NS
import time

import pytest
import torch

from tools import tight_dagger as owner
from visuotactile.position_control import ExpertPositionBridge


def policy(n=3):
    return dict(attempts_per_group=n,gate_attempts=2,minimum_successful_corrections=1,
        minimum_expert_ticks=1,minimum_dp_anchors=1)


def outcome(identity, kind):
    student=kind=='student';flush=kind=='flush'
    valid=0 if student or flush else 20
    success=kind!='timeout'
    return dict(identity=identity,legal=True,software_complete=True,success=success,steps=30,
        reasons=dict(max_successes=success,timeout=not success),dagger=dict(provenance_passed=True,
            position_executor_continuous=True,expert_valid_ticks=valid,
            takeover_triggered=not student,student_autonomous_success=student,
            expert_takeover_success=success and not student,student_ticks=30 if student else 10,
            expert_ticks=0 if student else 20,expert_shadow_queries=30))


def fixture(tmp_path,monkeypatch,kinds):
    from tools import tight_state
    from visuotactile import state_benchmark,camera_validity,narrow_data,anchor_sampling,native_evaluation
    root=tmp_path/'run';root.mkdir();data=root/'state_data';data.mkdir()
    frontend=root/'frontend.pt';backend=root/'backend.pt';expert=root/'expert.pt'
    for path in (frontend,backend,expert):torch.save({},path)
    source=dict(splits=dict(train=['train_a']))
    owner.write(data/'data.json',source)
    specs=[dict(identity=f'attempt_{i}',family='family',source_identity='train_a',cohort='dagger_train',
        split='train',narrow_reset={},seed=100+i,event_seed=200+i,takeover_tick=60) for i in range(len(kinds))]
    owner.write(root/'collection_plan.json',specs)
    owner.write(root/'configuration.json',dict(collection=policy(len(kinds)),
        bridge=dict(version=ExpertPositionBridge.version,expert_speed=ExpertPositionBridge.speed_scheme)))
    owner.write(root/'initial_pair.json',dict(frontend=str(frontend),frontend_sha256=owner.sha(frontend),
        backend=str(backend),backend_sha256=owner.sha(backend),data_root=str(data)))
    owner.write(root/'session.json',dict(deadline=time.time()+100,collection_deadline=time.time()+100,
        reserve_bytes=0,expert_checkpoint=str(expert),expert_sha256=owner.sha(expert)))
    lease=NS(deadline=None);calls=[];loaded=[]
    monkeypatch.setattr(tight_state,'environment',lambda *a,**k:(NS(),NS(),NS(),lease))
    def trial(env,lease,out,spec,source,player,checkpoint,**kwargs):
        assert kwargs['position'] and kwargs['capture_dataset'] and kwargs['expert_checkpoint']==str(expert)
        assert kwargs['dagger']['expert_speed']==ExpertPositionBridge.speed_scheme
        index=len(calls);calls.append(spec['identity']);out.mkdir(parents=True)
        if kinds[index]=='error':raise RuntimeError('controlled collection software error')
        path=out/'data.h5';path.write_bytes(b'opaque fixture')
        result=outcome(spec['identity'],kinds[index]);result['dataset_capture']=dict(path=str(path));return result
    def record(entry,include_geometry):
        loaded.append(entry['episode']);assert include_geometry is False;return {},None,None
    monkeypatch.setattr(state_benchmark,'trial',trial)
    monkeypatch.setattr(narrow_data,'dagger_record',record)
    monkeypatch.setattr(anchor_sampling,'supervision_population',lambda r:dict(supervised_ticks=20,supervised_anchors=2))
    monkeypatch.setattr(camera_validity,'inspect_capture',lambda p:dict(passed=True))
    monkeypatch.setattr(native_evaluation,'exit_completed_kit_worker',lambda r:None)
    return root,calls,loaded


def test_collect_empty_student_mask_is_not_a_training_reader_error(tmp_path,monkeypatch):
    root,calls,loaded=fixture(tmp_path,monkeypatch,['timeout','success','student'])
    owner.collect(root)
    assert calls==['attempt_0','attempt_1','attempt_2'] and loaded==calls[:2]
    summary=owner.read(root/'collection_finished.json')
    assert summary['complete'] and summary['attempts']==3 and summary['accepted_episodes']==1
    assert summary['expert_takeover_successes']==1 and summary['student_autonomous_successes']==1
    assert summary['expert_takeover_timeouts']==1 and summary['unstarted']==0
    assert owner.read(root/'collection_gate.json')['passed']
    with pytest.raises(ValueError,match='already attempted'):owner.collect(root)


def test_collect_software_failure_preserves_started_denominator_and_stops(tmp_path,monkeypatch):
    root,calls,loaded=fixture(tmp_path,monkeypatch,['error','success','success'])
    with pytest.raises(RuntimeError,match='controlled'):owner.collect(root)
    summary=owner.read(root/'collection_finished.json')
    assert not summary['complete'] and summary['attempts']==1 and summary['unstarted']==2
    assert summary['software_or_admission_errors']==1 and summary['unknown_interaction_attempts']==1
    assert calls==['attempt_0'] and loaded==[]
    assert owner.read(root/'collection_results.json')[0]['error']


def test_first_gate_uses_two_budgeted_attempts_without_success_requirement(tmp_path,monkeypatch):
    root,calls,loaded=fixture(tmp_path,monkeypatch,['timeout','timeout','timeout'])
    owner.collect(root)
    assert len(calls)==3 and owner.read(root/'collection_gate.json')['passed']
    assert not owner.read(root/'collection_finished.json')['sufficient']


def test_early_student_success_continues_fixed_plan_and_later_takeovers_pass_gate(tmp_path,monkeypatch):
    root,calls,loaded=fixture(tmp_path,monkeypatch,['student','timeout','success'])
    owner.collect(root)
    assert calls==['attempt_0','attempt_1','attempt_2'] and loaded==calls[1:]
    summary=owner.read(root/'collection_finished.json')
    assert summary['attempts']==3 and summary['unstarted']==0 and summary['student_autonomous_successes']==1
    assert summary['complete'] and summary['sufficient'] and summary['software_or_admission_errors']==0
    gate=owner.read(root/'collection_gate.json')
    assert gate['passed'] and [r['identity'] for r in gate['attempts']]==calls[1:]


def test_insufficient_actual_takeovers_exhausts_only_fixed_plan_without_software_error(tmp_path,monkeypatch):
    root,calls,loaded=fixture(tmp_path,monkeypatch,['student','flush','success'])
    owner.collect(root)
    assert calls==['attempt_0','attempt_1','attempt_2'] and loaded==calls[2:]
    summary=owner.read(root/'collection_finished.json')
    assert summary['complete'] and not summary['sufficient'] and summary['accepted_episodes']==1
    assert summary['attempts']==3 and summary['unstarted']==0 and summary['software_or_admission_errors']==0
    assert summary['insufficient_interface_evidence']
    gate=owner.read(root/'collection_gate.json')
    assert not gate['passed'] and gate['qualified_attempts']==1 and gate['required_attempts']==2


@pytest.mark.parametrize('fault',['label','mapping'])
def test_actual_interface_fault_still_stops_before_remaining_registered_attempts(tmp_path,monkeypatch,fault):
    from visuotactile import narrow_data,state_benchmark
    root,calls,loaded=fixture(tmp_path,monkeypatch,['student','timeout','success'])
    if fault=='label':
        def invalid_label(*args,**kwargs):raise ValueError('expert label differs from actual target')
        monkeypatch.setattr(narrow_data,'dagger_record',invalid_label)
    else:
        trial=state_benchmark.trial
        def invalid_mapping(*args,**kwargs):
            result=trial(*args,**kwargs)
            if result['dagger']['takeover_triggered']:result['dagger']['provenance_passed']=False
            return result
        monkeypatch.setattr(state_benchmark,'trial',invalid_mapping)
    with pytest.raises(ValueError):owner.collect(root)
    summary=owner.read(root/'collection_finished.json')
    assert calls==['attempt_0','attempt_1']
    assert not summary['complete'] and not summary['sufficient'] and summary['attempts']==2 and summary['unstarted']==1
    assert summary['software_or_admission_errors']==1 and not owner.read(root/'collection_gate.json')['passed']


def test_flush_success_is_not_student_autonomous_success():
    rows=[dict(result=outcome('flush','flush'),accepted=False)]
    summary=owner.collection_summary(rows,policy())
    assert summary['mixed_successes']==1 and summary['expert_takeover_successes']==1
    assert summary['student_autonomous_successes']==0 and summary['effective_expert_ticks']==0
