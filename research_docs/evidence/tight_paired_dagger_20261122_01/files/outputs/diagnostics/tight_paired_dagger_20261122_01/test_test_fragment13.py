"""Synthetic CPU-only contingency fixtures; never inspect real evaluation."""
import copy
import json
from pathlib import Path
import threading
import time

import pytest

from outputs.diagnostics.tight_paired_dagger_20261122_01 import test_fragment13 as m


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture64(tmp_path,launched=3):
    test=tmp_path/'test'; test.mkdir()
    stages=[]; hashes={}
    for i in range(64):
        wave=test/f'wave{i:02d}'; wave.mkdir()
        put(wave/'plan.json',dict(index=i,resource_root=str(test)))
        put(wave/'session.json',dict(deadline=m.DEADLINE))
        hashes[str(wave)]=m.digest(wave/'plan.json')
        stages.append(dict(entry='paired_evaluate',args=['worker','--output',str(wave)]))
    job=dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(test),stages=stages)
    jobpath=tmp_path/'job.json'; put(jobpath,job)
    freeze=dict(cohort='test',n=64,job=str(jobpath),job_sha256=m.digest(jobpath),wave_plan_sha256=hashes)
    put(test/'registration.json',freeze)
    commands=[['python','-m','outputs.diagnostics.tight_paired_dagger_20261122_01.eval_startup12',
               'worker','--output',str(test/f'wave{i:02d}')] for i in range(64)]
    put(test/'project_job_started.json',dict(job=job,commands=commands))
    put(test/'project_status.json',dict(stage=launched-1,pid=23,command=commands[launched-1]))
    for i in range(launched):
        (test/f'project_stage_{i:02d}.log').touch()
    # Result content is deliberately not parseable. Census may hash it, but
    # must never read outcome fields to select a suffix.
    for i in range(launched-1):
        (test/f'wave{i:02d}/results.json').write_text('NOT AN OUTCOME JSON')
        put(test/f'wave{i:02d}/process_exit_requested.json',dict(utc=2))
    return test


def test_closed_prefix_failed_wave_pristine_suffix_full_population(tmp_path):
    test=fixture64(tmp_path)
    result=m.lifecycle_census(test)
    assert result['consumed_waves']==3
    assert result['eligible_indices']==list(range(3,64))
    assert len(result['rows'])==64 and result['registered_opportunities']==256
    assert result['rows'][2]['status']=='consumed_or_ambiguous_wave'
    assert result['outcome_values_interpreted'] is False


def test_empty_stage_log_is_consumed(tmp_path):
    result=m.lifecycle_census(fixture64(tmp_path,launched=4))
    assert result['eligible_indices'][0]==4


@pytest.mark.parametrize('artifact',['run_started.json','initial_collision.json','lane/started.json','numeric.h5','unknown.file'])
def test_any_suffix_runtime_or_unknown_artifact_refuses(tmp_path,artifact):
    test=fixture64(tmp_path)
    path=test/'wave04'/artifact; path.parent.mkdir(parents=True,exist_ok=True); path.touch()
    with pytest.raises(ValueError,match='beyond original durable launch prefix'):
        m.lifecycle_census(test)


def test_gapped_launch_logs_refuse(tmp_path):
    test=fixture64(tmp_path)
    (test/'project_stage_01.log').unlink()
    with pytest.raises(ValueError,match='unique prefix'):
        m.lifecycle_census(test)


def test_changed_original_plan_refuses(tmp_path):
    test=fixture64(tmp_path)
    put(test/'wave63/plan.json',dict(changed=True))
    with pytest.raises(ValueError,match='frozen wave changed'):
        m.lifecycle_census(test)


def test_symlink_escape_refuses(tmp_path):
    test=fixture64(tmp_path)
    (test/'wave63/extra').symlink_to(tmp_path/'job.json')
    with pytest.raises(ValueError,match='symlink'):
        m.lifecycle_census(test)


def test_unexplained_empty_suffix_directory_refuses(tmp_path):
    test=fixture64(tmp_path)
    (test/'wave63/empty').mkdir()
    with pytest.raises(ValueError,match='beyond original durable launch prefix'):
        m.lifecycle_census(test)


def test_suffix_fifo_refuses(tmp_path):
    import os
    test=fixture64(tmp_path)
    os.mkfifo(test/'wave63/fifo')
    with pytest.raises(ValueError,match='special entry'):
        m.lifecycle_census(test)


def timeout_fixture(tmp_path):
    owner=tmp_path/'owner'; owner.mkdir()
    test=tmp_path/'test'; wave=test/'wave07'; wave.mkdir(parents=True)
    status=dict(stage=0,pid=13,utc=5,command=['python','-m','worker','--output',str(wave)])
    put(owner/'project_status.json',status)
    return owner,test,wave,status


def test_timeout_stage_change_during_intent_fsync_abandons(tmp_path):
    owner,test,wave,status=timeout_fixture(tmp_path)
    def change():
        put(owner/'project_status.json',dict(status,stage=1))
    row=m.attempt_timeout(owner,test,owner,status,now=lambda:100,after_intent=change)
    assert row['decision']=='abandoned' and row['stage_changed']
    assert not (owner/'STOP').exists()


def test_timeout_readiness_during_intent_fsync_abandons(tmp_path):
    owner,test,wave,status=timeout_fixture(tmp_path)
    row=m.attempt_timeout(owner,test,owner,status,now=lambda:100,
                          after_intent=lambda:put(wave/'initialization.json',dict(seconds=1)))
    assert row['decision']=='abandoned' and row['initialized']
    assert not (owner/'STOP').exists()


def test_timeout_is_durable_and_whole_wave_consumed(tmp_path):
    owner,test,wave,status=timeout_fixture(tmp_path)
    row=m.attempt_timeout(owner,test,owner,status,now=lambda:100)
    assert row['decision']=='stop_committed' and (owner/'STOP').exists()
    intent=m.read(owner/'startup_intent_stage00_wave07.json')
    assert intent['timeout_seconds']==1800 and intent['entire_wave_consumed'] and not intent['rerun']
    with pytest.raises(FileExistsError):
        m.attempt_timeout(owner,test,owner,status,now=lambda:101)


def test_stage_and_readiness_atomic_adapters_share_lock(tmp_path):
    owner,test,wave,status=timeout_fixture(tmp_path)
    entered=threading.Event(); complete=threading.Event()
    target=wave/'initialization.json'
    def save(path,value):
        entered.set(); put(path,value); complete.set()
    wrapped=m.locked_atomic(save,owner,target)
    with m.lifecycle_lock(owner):
        thread=threading.Thread(target=wrapped,args=(target,dict(seconds=1)))
        thread.start()
        assert not entered.wait(.05)
    thread.join(2)
    assert entered.is_set() and complete.is_set()


def test_exact_lease_routing_only(tmp_path):
    calls=[]
    native=lambda root,deadline: calls.append((root,deadline)) or 'native-full-lease'
    factory=m.scoped_lease_factory(native,test=tmp_path/'test',owner=tmp_path/'owner',deadline=m.DEADLINE)
    assert factory(tmp_path/'test',deadline=m.DEADLINE)=='native-full-lease'
    assert calls==[(tmp_path/'owner',m.DEADLINE)]
    with pytest.raises(ValueError):
        factory(tmp_path/'other',deadline=m.DEADLINE)
    with pytest.raises(ValueError):
        factory(tmp_path/'test',deadline=m.DEADLINE+1)


def test_single_worker_cli_and_native_owner(tmp_path,monkeypatch):
    from tools import tight_project
    monkeypatch.setitem(tight_project.ENTRIES,'paired_evaluate_fragment13',('evaluate',m.MODULE,[]))
    job=dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(tmp_path),stages=[
        dict(entry='paired_evaluate_fragment13',args=['worker','--output',str(tmp_path/'test/wave03')])])
    command=tight_project.commands(job,'evaluate')[0]
    assert command[1:]==['-m',m.MODULE,'worker','--output',str(tmp_path/'test/wave03')]
    assert command.count('worker')==1


def test_exclusive_dispatch_no_second_attempt_or_outside_suffix(tmp_path,monkeypatch):
    test=fixture64(tmp_path)
    owner=tmp_path/'owner'; (owner/'dispatches').mkdir(parents=True)
    registry=tmp_path/'registry13.json'; put(registry,dict(inactive_fixture=True))
    monkeypatch.setattr(m,'TEST',test); monkeypatch.setattr(m,'OWNER',owner); monkeypatch.setattr(m,'REGISTRY',registry)
    value=dict(eligible_indices=list(range(3,64)))
    m.dispatch_fence(test/'wave03',value)
    with pytest.raises(FileExistsError):
        m.dispatch_fence(test/'wave03',value)
    with pytest.raises(ValueError,match='reviewed never-launched'):
        m.dispatch_fence(test/'wave02',value)
    assert len(list((owner/'dispatches').iterdir()))==1


def test_dispatch_symlink_alias_refuses_before_resolve(tmp_path,monkeypatch):
    test=fixture64(tmp_path)
    owner=tmp_path/'owner'; (owner/'dispatches').mkdir(parents=True)
    registry=tmp_path/'registry13.json'; put(registry,dict(inactive_fixture=True))
    monkeypatch.setattr(m,'TEST',test); monkeypatch.setattr(m,'OWNER',owner); monkeypatch.setattr(m,'REGISTRY',registry)
    alias=tmp_path/'alias'; alias.symlink_to(test/'wave03',target_is_directory=True)
    with pytest.raises(ValueError,match='literal canonical'):
        m.dispatch_fence(alias,dict(eligible_indices=list(range(3,64))))
    assert not list((owner/'dispatches').iterdir())


def test_attempted_helper_is_never_retried(tmp_path,monkeypatch):
    root=tmp_path/'run'; (root/'cycle_logs_recovery12').mkdir(parents=True)
    monkeypatch.setattr(m,'ROOT',root); monkeypatch.setattr(m,'TEST',root/'evaluation/test')
    monkeypatch.setattr(m,'HERE',tmp_path/'audit')
    (root/'cycle_logs_recovery12/summarize_test.log').touch()
    with pytest.raises(ValueError,match='Ambiguous attempted'):
        m.pending_tail_stages()


def test_partial_helper_output_without_log_refuses(tmp_path,monkeypatch):
    root=tmp_path/'run'; (root/'evaluation/test/summary').mkdir(parents=True)
    monkeypatch.setattr(m,'ROOT',root); monkeypatch.setattr(m,'TEST',root/'evaluation/test')
    monkeypatch.setattr(m,'HERE',tmp_path/'audit')
    with pytest.raises(ValueError,match='partial original'):
        m.pending_tail_stages()


def test_closed_helper_skips_with_hash(tmp_path,monkeypatch):
    root=tmp_path/'run'; (root/'cycle_logs_recovery12').mkdir(parents=True)
    monkeypatch.setattr(m,'ROOT',root); monkeypatch.setattr(m,'TEST',root/'evaluation/test')
    monkeypatch.setattr(m,'HERE',tmp_path/'audit')
    put(root/'cycle_logs_recovery12/summarize_test_exit.json',dict(returncode=0))
    result=m.pending_tail_stages()
    assert result[0]['skip_closed'] is True and result[0]['original_exit_sha256']
    assert result[1]['skip_closed'] is False
