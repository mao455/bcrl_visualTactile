"""CPU-only fixtures. Never import torch or launch a trainer."""
import json
from pathlib import Path
import pytest
from outputs.diagnostics.tight_paired_dagger_20261122_01 import parallel_rear_proposal as p
from outputs.diagnostics.tight_paired_dagger_20261122_01 import resource_watch_parallel_proposal as w


def registration():
    return dict(max_telemetry_age_s=3.,min_ram_bytes=10*2**30,
        ram_per_worker_bytes=6*2**30,vram_per_worker_mib=1536,
        min_gpu_free_mib=4096,safety_factor=1.25)


def telemetry():
    return dict(utc_epoch=100.,gpu_processes=[],external_gpu_processes=[],
        process_telemetry_errors=[],ram_available_bytes=32*2**30,gpu_free_mib=20000)


def test_memory_chooses_serial_before_child_launch():
    row=telemetry();row['ram_available_bytes']=20*2**30
    assert not p.admission(row,registration(),100.,2)['allowed']
    assert p.admission(row,registration(),100.,1)['allowed']


@pytest.mark.parametrize('field,value',[('utc_epoch',95.),('gpu_free_mib',4000),
    ('ram_available_bytes',9*2**30),('external_gpu_processes',[dict(pid=5)]),
    ('process_telemetry_errors',['read failed']),('gpu_processes',None)])
def test_bad_admission(field,value):
    row=telemetry();row[field]=value
    assert not p.admission(row,registration(),100.,2)['allowed']


def test_job_keeps_two_serial_dp_and_one_rear_controller():
    original=p.serial_job(); job=p.proposed_job()
    assert job['stages'][:2]==original['stages'][:2]
    assert [x['entry'] for x in job['stages']]==['paired_frontend','paired_frontend','paired_rear_parallel']
    for role in 'AB':
        actual=p.stage_commands(p.ROOT/'learning',role)
        source=original['stages'][2+(role=='B')*4:6+(role=='B')*4]
        for cmd,s in zip(actual,source): assert cmd[4:]==s['args']


def test_watcher_registers_only_exact_new_owner():
    rows=w.specs(p.ROOT,p.CONFIG)
    main=next(x for x in rows if x['name']=='main_train')
    assert main['job']==p.JOB and main['module']==p.MODULE
    assert len(rows)==len(w.original_specs(p.ROOT,p.CONFIG))
    assert w.base.PAUSES==(None,'A','B','G00','G10','G01','G11')


class FakeChild:
    counter=100
    def __init__(self,command,**kwargs):
        FakeChild.counter+=1; self.pid=FakeChild.counter;self.command=command
        self.returncode=None;self.polls=0
    def poll(self):
        self.polls+=1
        if self.polls>=3: self.returncode=7 if self.command==['FAIL'] else 0
        return self.returncode


def runner(tmp_path,parallel,commands,**overrides):
    launched=[]; cleaned=[];t=[100.]
    def spawn(cmd,**kwargs):
        assert kwargs['start_new_session'] is True
        assert kwargs['env']['OMP_NUM_THREADS']=='2'
        child=FakeChild(cmd);launched.append(child);return child
    def sleep(_): t[0]+=.1
    kw=dict(spawn=spawn,cleanup=lambda ps:cleaned.extend(ps),now=lambda:t[0],sleep=sleep)
    kw.update(overrides)
    try:
        events=p.run_roles(commands,parallel,120.,[],lambda:True,tmp_path,**kw)
        return events,launched,cleaned
    finally:
        assert set(cleaned)==set(launched)


def test_parallel_preserves_each_own_order(tmp_path):
    events,launched,_=runner(tmp_path,True,dict(A=[['A0'],['A1']],B=[['B0'],['B1']]))
    cmds=[x.command for x in launched]
    assert cmds.index(['A0'])<cmds.index(['A1'])
    assert cmds.index(['B0'])<cmds.index(['B1'])
    assert cmds.index(['B0'])<cmds.index(['A1'])
    assert len([e for e in events if 'returncode' in e])==4


def test_serial_fallback_does_not_overlap_roles(tmp_path):
    _,launched,_=runner(tmp_path,False,dict(A=[['A0'],['A1']],B=[['B0'],['B1']]))
    assert [c.command for c in launched]==[['A0'],['A1'],['B0'],['B1']]


def test_failure_cleans_both_and_stops_new_work(tmp_path):
    with pytest.raises(RuntimeError,match='no retry'):
        runner(tmp_path,True,dict(A=[['FAIL'],['never']],B=[['B0'],['B1']]))
    events=json.loads((tmp_path/'pipeline_events.json').read_text())
    assert not any(e.get('command')==['never'] for e in events)


def test_deadline_with_paused_launches_is_finite(tmp_path):
    t=[0.];spawned=[];cleaned=[]
    def sleep(_):t[0]+=.5
    with pytest.raises(TimeoutError):
        p.run_roles(dict(A=[['A']]),True,1.,[],lambda:False,tmp_path,
            spawn=lambda *a,**k:spawned.append(a),cleanup=lambda x:cleaned.extend(x),now=lambda:t[0],sleep=sleep)
    assert not spawned and not cleaned


def test_stop_never_launches(tmp_path):
    stop=tmp_path/'STOP';stop.touch();launched=[]
    with pytest.raises(TimeoutError):
        p.run_roles(dict(A=[['A']]),True,10.,[stop],lambda:True,tmp_path,
            spawn=lambda *a,**k:launched.append(a),cleanup=lambda _:None,now=lambda:0.,sleep=lambda _:None)
    assert not launched


def test_sources_are_output_only():
    assert p.HERE.is_relative_to(p.PROJECT/'outputs/diagnostics')
    assert p.JOB.parent==p.REGISTRATION.parent==p.HERE


@pytest.mark.parametrize('field,value',[('gpu_free_mib',float('nan')),('gpu_free_mib',float('inf')),
    ('ram_available_bytes',float('nan')),('gpu_free_mib',True),('ram_available_bytes',-1)])
def test_nonfinite_or_invalid_resource_quantity(field,value):
    row=telemetry();row[field]=value
    assert not p.admission(row,registration(),100.,2)['allowed']


def test_invalid_admission_bound_rejected():
    reg=registration();reg['ram_per_worker_bytes']=-1
    assert not p.admission(telemetry(),reg,100.,2)['allowed']



def test_actual_owned_cpu_children_are_both_reaped():
    import subprocess,sys
    children=[]
    try:
        for _ in range(2):
            children.append(subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True))
        p.stop_all(children,grace_s=.25,kill_s=1.)
        assert all(child.poll() is not None for child in children)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill();child.wait(timeout=2)
