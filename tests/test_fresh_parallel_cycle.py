from pathlib import Path
import threading
import json
import os
import time
from types import SimpleNamespace

import pytest

from tools.tight_fresh_cycle import paired_frontends, parallel_admission
from tools import tight_fresh_cycle as cycle
from visuotactile.training_resources import atomic_json


@pytest.mark.parametrize('failed',[None,'A','B'])
def test_independent_frontends_join_and_preserve_both_outcomes(failed):
    barrier=threading.Barrier(2,timeout=2)
    seen={}
    def execute(name,args,completion,*,update_status):
        role=name[-1]
        assert args==['tools.tight_fresh_train','train','--root',Path('/round'),'--role',role,'--epochs',10]
        assert completion==Path('/round')/role/'training/finished.json'
        assert not update_status
        barrier.wait()
        seen[role]='failed' if role==failed else 'complete'
        if role==failed:raise RuntimeError('saved failed stage')
    result=paired_frontends(execute,Path('/round'),10)
    assert set(seen)=={'A','B'}
    assert result['complete']==(failed is None)
    assert set(result['errors'])==({failed} if failed else set())
    assert result['ended']>=result['started']


@pytest.mark.parametrize('free,ram,age,expected',[
    (22,24,1,True),(21.99,24,1,False),(22,23.99,1,False),
    (22,24,9,False),(22,24,-1,False)])
def test_parallel_joint_launch_admission_keeps_extra_headroom(free,ram,age,expected):
    row=dict(utc_epoch=100-age,gpu_free_mib=free*1024,ram_available_bytes=ram*1024**3)
    assert parallel_admission(row,100)['admitted'] is expected


def test_episode_staging_cannot_change_after_either_learner_starts(tmp_path):
    import json
    from tools.tight_fresh_cycle import configure_episode_io
    config=dict(version='frozen_episode_prefetch_v1',max_bytes=1024<<20,minimum_available_bytes=10<<30)
    for role in ('A','B'):(tmp_path/role).mkdir()
    (tmp_path/'B/training').mkdir()
    with pytest.raises(ValueError):configure_episode_io(tmp_path,config)
    assert not (tmp_path/'A/episode_io.json').exists()
    (tmp_path/'B/training').rmdir()
    assert configure_episode_io(tmp_path,config)==config
    assert json.loads((tmp_path/'A/episode_io.json').read_text())==json.loads((tmp_path/'B/episode_io.json').read_text())==config
    with pytest.raises(ValueError):configure_episode_io(tmp_path,dict(config,max_bytes=512<<20))


def admission_clock(monkeypatch, start=100.):
    clock=[start]
    monkeypatch.setattr(cycle.time,'time',lambda:clock[0])
    monkeypatch.setattr(cycle.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(cycle.time,'sleep',lambda duration:clock.__setitem__(0,clock[0]+duration))
    return clock


def telemetry(path, now, free=22*1024, ram=24*1024**3):
    atomic_json(path,dict(utc_epoch=now,gpu_free_mib=free,ram_available_bytes=ram))


def test_joint_wait_rejects_recent_pre_stage_sample_then_admits_new_release(tmp_path,monkeypatch):
    clock=admission_clock(monkeypatch); resource=tmp_path/'resources.json'; receipt=tmp_path/'gate.json'
    telemetry(resource,clock[0]-.1,free=16389)
    def update(duration):
        clock[0]+=duration
        telemetry(resource,clock[0])
    monkeypatch.setattr(cycle.time,'sleep',update)
    result=cycle.wait_parallel_admission(resource,receipt,deadline=150)
    assert result['admitted'] and result['reason']=='admitted' and result['wait_seconds']==.25
    assert len(result['samples'])==2 and result['samples'][0]['gate_reason']=='waiting_for_post_admission_sample'
    assert result['samples'][1]['sample_after_admission_start']
    assert result['required_free_gpu_mib']==22528 and result['required_available_ram_bytes']==24*1024**3
    assert json.loads(receipt.read_text())==result


@pytest.mark.parametrize('fault',['old_high','gpu','ram','future','unavailable'])
def test_joint_wait_never_relaxes_freshness_or_headroom_and_ends_at_30s(tmp_path,monkeypatch,fault):
    clock=admission_clock(monkeypatch); resource=tmp_path/'resources.json'; receipt=tmp_path/'gate.json'
    if fault!='unavailable':telemetry(resource,clock[0]-.1)
    def update(duration):
        clock[0]+=duration
        if fault in ('old_high','unavailable'):return
        telemetry(resource,clock[0]+1 if fault=='future' else clock[0],
            free=22527 if fault=='gpu' else 22528,ram=(24*1024**3)-1 if fault=='ram' else 24*1024**3)
    monkeypatch.setattr(cycle.time,'sleep',update)
    result=cycle.wait_parallel_admission(resource,receipt,deadline=200)
    assert not result['admitted'] and result['reason']=='wait_exhausted'
    assert result['wait_seconds']==30 and clock[0]==130
    assert all(not row['admitted'] for row in result['samples'])
    assert result['deadline']==200 and result['admission_deadline']==130


def test_joint_wait_deadline_preempts_wait_without_serial_permission(tmp_path,monkeypatch):
    clock=admission_clock(monkeypatch); resource=tmp_path/'resources.json'; receipt=tmp_path/'gate.json'
    telemetry(resource,clock[0]-.1)
    with pytest.raises(TimeoutError,match='Original fresh session'):
        cycle.wait_parallel_admission(resource,receipt,deadline=101)
    saved=json.loads(receipt.read_text())
    assert saved['reason']=='original_session_deadline' and saved['wait_seconds']==1
    assert saved['deadline']==saved['admission_deadline']==101 and not saved['admitted']


@pytest.mark.parametrize('already_stopped',[False,True])
def test_joint_wait_honors_stop_before_read_or_during_wait(tmp_path,monkeypatch,already_stopped):
    clock=admission_clock(monkeypatch); stop=tmp_path/'STOP'; receipt=tmp_path/'gate.json'
    if already_stopped:stop.touch()
    def update(duration):
        clock[0]+=duration;stop.touch()
    monkeypatch.setattr(cycle.time,'sleep',update)
    with pytest.raises(InterruptedError,match='STOP'):
        cycle.wait_parallel_admission(tmp_path/'absent.json',receipt,deadline=200,stop_paths=[stop])
    saved=json.loads(receipt.read_text())
    assert saved['reason']=='requested_stop' and not saved['admitted']
    assert saved['wait_seconds']==(0 if already_stopped else .25)


@pytest.mark.parametrize('parallel',[False,True])
def test_cycle_records_fallback_only_after_bounded_parallel_wait(tmp_path,monkeypatch,parallel):
    clock=admission_clock(monkeypatch);root=tmp_path/'source';root.mkdir()
    collection=tmp_path/'collection';collection.mkdir();output=tmp_path/'next';ledger=tmp_path/'next_stages'
    atomic_json(root/'session.json',dict(deadline=200,resource_policy='full_gpu_v1'))
    telemetry(root/'shared_resources.json',clock[0]-.1,free=16389)
    atomic_json(collection/'result.json',dict(software_complete=True))
    atomic_json(collection/'correction_entries.json',{'valid':{}})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cycle.subprocess,'check_output',lambda *a,**k:'source commit')
    calls=[]
    def launch(command,**kwargs):
        calls.append(command)
        if 'aggregate' in command:
            output.mkdir();(output/'A').mkdir();(output/'B').mkdir()
            atomic_json(output/'prepared.json',dict(complete=True))
        elif 'cache' in command:
            atomic_json(output/'B/cache_index.json',dict(complete=True))
        else:
            assert command[2:4]==['tools.tight_fresh_train','train']
            assert command[command.index('--role')+1]=='A'
            raise RuntimeError('CPU serial dispatch boundary')
        return SimpleNamespace(pid=os.getpid(),returncode=0,poll=lambda:0)
    monkeypatch.setattr(cycle.subprocess,'Popen',launch)
    if not parallel:
        monkeypatch.setattr(cycle,'wait_parallel_admission',lambda *a,**k:pytest.fail('Serial default must not wait'))
    with pytest.raises(RuntimeError,match='CPU serial dispatch boundary'):
        cycle.run(root,collection,output,2,1,4,1,parallel_frontends=parallel)
    assert len(calls)==3
    if parallel:
        saved=json.loads((ledger/'parallel_admission.json').read_text())
        assert saved['frontend_mode']=='serial_fallback' and saved['reason']=='wait_exhausted'
        assert saved['wait_seconds']==30
    else:
        assert not (ledger/'parallel_admission.json').exists() and clock[0]==100
