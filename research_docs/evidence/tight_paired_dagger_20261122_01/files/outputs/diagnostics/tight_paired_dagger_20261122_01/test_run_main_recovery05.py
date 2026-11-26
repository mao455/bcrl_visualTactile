"""CPU fixtures for unactivated orchestration; no experiment launches or writes."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_main_recovery05 as m


def test_import_and_stage_contract():
    rows = list(m.stages())
    names = [r[0] for r in rows]
    assert len(names) == len(set(names)) == 16
    assert not any('collect_96' in n for n in names)
    assert names.index('storage_gate') < names.index('prepare_learning_recovery05')
    assert names.index('preflight_learning') < names.index('cache_own_observations')
    assert names.index('train_four_components_parallel_rears') < names.index('verify_final_actual_training') < names.index('prepare_development')
    assert names.index('summarize_development') < names.index('prepare_test')
    assert all('--reconstruct-cuda-noise' in args and gpu for name,args,gpu in rows if name.startswith('summarize_'))
    assert rows[-1][0] == 'audit_collection_cuda_noise_recovery05'
    prep = next(args for name,args,gpu in rows if name == 'prepare_learning_recovery05')
    assert prep[1].endswith('collection_recovery05_scope')


def test_storage_includes_actual_prefixes_and_reserve():
    entries = {'a':dict(length=100),'b':dict(length=201)}
    row = m.storage_forecast(entries, 10**12)
    assert row['complete_history_ticks'] == 301
    assert row['required_free_bytes'] == row['estimated_feature_bytes']+47*2**30
    assert row['ready']
    assert not m.storage_forecast(entries,row['required_free_bytes']-1)['ready']
    assert m.storage_forecast(entries,row['required_free_bytes'])['ready']


@pytest.mark.parametrize('value',[0,-1,True,1.5,'10'])
def test_malformed_storage_length_rejected(value):
    with pytest.raises(ValueError):m.storage_forecast({'a':dict(length=value)},10**12)


def watcher():
    row = dict(utc=100.,pid=123,paused=False,reason='no_external_gpu',errors=[],schedule_conflicts=[],external_gpu_processes=[],external_test_python_processes=[],external_test_python_scan_errors=[],active_resource_roots=['/active'],raw_monitor_rows=[{}],registered_workers_excluded=[])
    command=['python','-m',m.MODULE+'coexistence_watch_recovery05','--audit',str(m.WATCH.parent),'--apply']
    proc = SimpleNamespace(pid=123,is_running=lambda:True,cmdline=lambda:command,cwd=lambda:str(m.PROJECT),create_time=lambda:90.)
    return row,proc


def test_exact_live_watcher_accepted():
    row,proc=watcher();m.watcher_ready(row,proc,101.)


def idle_watcher():
    row,proc=watcher()
    row.update(paused=True,reason='no_fresh_owned_telemetry',active_resource_roots=[],raw_monitor_rows=[],registered_workers_excluded=[])
    return row,proc


def test_exact_idle_gap_allows_owner_start_keeps_pause():
    row,proc=idle_watcher();before=copy.deepcopy(row)
    result=m.watcher_ready(row,proc,101.,allow_owner_start=True)
    assert result['starting_without_owned_telemetry'] and result['watcher_remains_paused']
    assert row==before
    with pytest.raises(ValueError):m.watcher_ready(row,proc,101.)


@pytest.mark.parametrize('mutation',['active','raw','worker','external','scan_error','registration_error','conflict','other_pause','malformed_pause'])
def test_startup_exception_never_allows_live_or_unknown_pause(mutation):
    row,proc=idle_watcher()
    if mutation=='active':row['active_resource_roots']=['/live']
    elif mutation=='raw':row['raw_monitor_rows']=[{}]
    elif mutation=='worker':row['registered_workers_excluded']=[dict(pid=1)]
    elif mutation=='external':row['external_gpu_processes']=[dict(pid=1)]
    elif mutation=='scan_error':row['external_test_python_scan_errors']=['denied']
    elif mutation=='registration_error':row['errors']=['missing receipt']
    elif mutation=='conflict':row['schedule_conflicts']=['changed']
    elif mutation=='other_pause':row['reason']='external_gpu_present'
    else:row['paused']=1
    with pytest.raises(ValueError):m.watcher_ready(row,proc,101.,allow_owner_start=True)


def test_standalone_idle_gap_requires_separate_capacity():
    row,proc=idle_watcher();decision=m.watcher_ready(row,proc,101.,require_idle=True)
    assert decision['watcher_remains_paused']
    live,proc=watcher()
    with pytest.raises(ValueError,match='no active owner'):m.watcher_ready(live,proc,101.,require_idle=True)


def independent_capacity():
    return (dict(query_started_utc=99.,utc=100.,gpu_free_mib=5120,ram_available_bytes=12*2**30,
        disk_free_bytes=12*2**30,gpu_compute_pids=[],external_test_scan=dict(processes=[],errors=[])),
        dict(resources=dict(min_gpu_free_mib=4096,min_ram_gib=10),reserve_bytes=12*2**30))


def test_standalone_capacity_adds_resident_allowances():
    row,session=independent_capacity();r=m.standalone_admission(row,session,101.)
    assert r['gpu_required_free_mib']==5120 and r['ram_required_available_bytes']==12*2**30
    assert r['owned_telemetry_claimed'] is False


def test_standalone_snapshot_queries_directly_without_owner_file(driver,monkeypatch):
    import psutil,resource_watch_v4 as watch
    calls=[]
    def query(args,**kwargs):
        calls.append(args)
        assert kwargs['timeout']==5
        return '15000\n' if '--query-gpu=memory.free' in args else ''
    monkeypatch.setattr(m.subprocess,'check_output',query)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=20*2**30))
    monkeypatch.setattr(watch,'external_test_processes',lambda:dict(processes=[],errors=[]))
    row=m.standalone_snapshot()
    assert len(calls)==2 and row['gpu_free_mib']==15000 and row['gpu_compute_pids']==[]
    assert 'not owned telemetry' in row['source']


@pytest.mark.parametrize('mutation',['gpu','ram','disk','resident','external','scan_unknown','nan','negative','stale','slow','bad_reserve'])
def test_standalone_capacity_rejects_insufficient_unknown_or_resident(mutation):
    row,session=independent_capacity()
    if mutation=='gpu':row['gpu_free_mib']-=1
    elif mutation=='ram':row['ram_available_bytes']-=1
    elif mutation=='disk':row['disk_free_bytes']-=1
    elif mutation=='resident':row['gpu_compute_pids']=[42]
    elif mutation=='external':row['external_test_scan']['processes']=[42]
    elif mutation=='scan_unknown':row['external_test_scan']['errors']=['denied']
    elif mutation=='nan':row['gpu_free_mib']=float('nan')
    elif mutation=='negative':row['disk_free_bytes']=-1
    elif mutation=='stale':row['utc']=97.
    elif mutation=='slow':row['query_started_utc']=80.
    else:session['resources']['min_ram_gib']=float('nan')
    with pytest.raises(ValueError):m.standalone_admission(row,session,101.)


@pytest.mark.parametrize('mutation',['stale','future','nan','paused','external','unknown','conflict','argv','cwd','pid_reuse'])
def test_watcher_unknown_or_changed_identity_fails_closed(mutation):
    row,proc=watcher()
    if mutation=='stale':row['utc']=80.
    elif mutation=='future':row['utc']=104.
    elif mutation=='nan':row['utc']=float('nan')
    elif mutation=='paused':row['paused']=True
    elif mutation=='external':row['external_gpu_processes']=[dict(pid=2)]
    elif mutation=='unknown':row['external_test_python_scan_errors']=['denied']
    elif mutation=='conflict':row['schedule_conflicts']=['changed']
    elif mutation=='argv':proc.cmdline=lambda:['python','-m','other']
    elif mutation=='cwd':proc.cwd=lambda:'/tmp'
    else:proc.create_time=lambda:102.
    with pytest.raises(ValueError):m.watcher_ready(row,proc,101.)


@pytest.fixture
def driver(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir();out=tmp_path/'audit_driver'
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'OUT',out)
    (root/'cycle_logs').mkdir();(root/'session.json').write_text(json.dumps(dict(deadline=1000.)))
    monkeypatch.setattr(m.time,'time',lambda:100.)
    d=m.Driver();out.mkdir();return d,root,out


def test_owner_start_gate_accepts_idle_without_capacity_or_schedule_mutation(driver,monkeypatch):
    import psutil,driver_support_v4 as support
    d,root,out=driver;row,proc=idle_watcher();row['utc']=100.
    monkeypatch.setattr(support,'wait_external',lambda *a:None)
    monkeypatch.setattr(psutil,'Process',lambda _:proc)
    old_read=m.read;monkeypatch.setattr(m,'read',lambda p:row if Path(p)==m.WATCH else old_read(p))
    def forbidden():raise AssertionError('Owner startup must not fabricate standalone capacity')
    monkeypatch.setattr(m,'standalone_snapshot',forbidden)
    d.gpu_gate('cache_own_observations')
    receipt=old_read(out/'cache_own_observations_startup_admission.json')
    assert receipt['watcher']['watcher_remains_paused']
    assert row['paused'] is True and not (root/'shared_resources.json').exists()


def test_standalone_gate_uses_independent_capacity_and_no_owner_telemetry(driver,monkeypatch):
    import psutil,driver_support_v4 as support
    d,root,out=driver;row,proc=idle_watcher();capacity,session=independent_capacity()
    (root/'session.json').write_text(json.dumps(session))
    monkeypatch.setattr(support,'wait_external',lambda *a:None)
    monkeypatch.setattr(psutil,'Process',lambda _:proc)
    old_read=m.read;monkeypatch.setattr(m,'read',lambda p:row if Path(p)==m.WATCH else old_read(p))
    monkeypatch.setattr(m,'standalone_snapshot',lambda:capacity)
    d.gpu_gate('summarize_test')
    receipt=old_read(out/'summarize_test_resource_admission.json')
    assert receipt['capacity']['owned_telemetry_claimed'] is False
    assert not list(root.rglob('shared_resources.json'))


def test_standalone_capacity_failure_waits_and_never_admits(driver,monkeypatch):
    import psutil,driver_support_v4 as support
    d,root,out=driver;row,proc=idle_watcher();capacity,session=independent_capacity()
    capacity['gpu_compute_pids']=[19]
    (root/'session.json').write_text(json.dumps(session))
    monkeypatch.setattr(support,'wait_external',lambda *a:None)
    monkeypatch.setattr(psutil,'Process',lambda _:proc)
    old_read=m.read;monkeypatch.setattr(m,'read',lambda p:row if Path(p)==m.WATCH else old_read(p))
    monkeypatch.setattr(m,'standalone_snapshot',lambda:capacity)
    def stopped(_):raise InterruptedError('fixture finite wait ended')
    monkeypatch.setattr(m.time,'sleep',stopped)
    with pytest.raises(InterruptedError):d.gpu_gate('summarize_test')
    assert not (out/'summarize_test_resource_admission.json').exists()
    assert 'resident compute' in old_read(out/'status.json')['reason']


def test_telemetry_only_rebinds_symlink(driver):
    d,root,out=driver;(root/'learning').mkdir();p=root/'learning/shared_resources.json'
    p.write_text('retain')
    with pytest.raises(ValueError):d.telemetry()
    assert p.read_text()=='retain'
    p.unlink();p.symlink_to(root/'owners/main_cache/shared_resources.json');d.telemetry()
    assert p.readlink()==root/'owners/main_train/shared_resources.json'
    assert m.read(out/'training_telemetry_transition.json')['old'].endswith('main_cache/shared_resources.json')


def test_run_failure_keeps_exit_and_never_retries(driver,monkeypatch):
    import driver_support_v4 as support
    from tools import tight_paired_dagger as production
    d,root,out=driver;calls=[]
    monkeypatch.setattr(support,'run_owned',lambda cmd,*a:(calls.append(cmd) or 7))
    monkeypatch.setattr(production,'sources',lambda:{'a':'b'})
    d.source_identity={'a':'b'}
    with pytest.raises(ValueError,match='no retry'):d.run('fake_component',['-m','fake'],False)
    assert len(calls)==1 and calls[0][:3]==['taskset','-c','8-15']
    assert m.read(root/'cycle_logs/fake_component_exit.json')['returncode']==7


def test_no_gpu_child_without_gate(driver,monkeypatch):
    import driver_support_v4 as support
    d,root,out=driver;calls=[]
    monkeypatch.setattr(support,'run_owned',lambda *a:calls.append(a))
    def reject(name):raise ValueError('external work')
    monkeypatch.setattr(d,'gpu_gate',reject)
    with pytest.raises(ValueError,match='external work'):d.run('evaluate_test',['-m','fake'],True)
    assert calls==[] and not (root/'cycle_logs/evaluate_test.log').exists()


@pytest.mark.parametrize('mode',['deadline','stop','source'])
def test_deadline_stop_or_changed_source_prevents_launch(driver,monkeypatch,mode):
    import driver_support_v4 as support
    from tools import tight_paired_dagger as production
    d,root,out=driver;calls=[]
    monkeypatch.setattr(support,'run_owned',lambda *a:calls.append(a))
    if mode=='deadline':d.deadline=99.
    elif mode=='stop':d.stop.touch()
    else:
        d.source_identity={'a':'b'};monkeypatch.setattr(production,'sources',lambda:{'a':'changed'})
    with pytest.raises(ValueError):d.run('fake',['-m','fake'],False)
    assert calls==[]


def test_snapshot_main_preserves47plus1_and_full_stage_order(driver,monkeypatch):
    from tools import tight_paired_dagger as production
    import collection_recovery05_scope as scope
    d,root,out=driver;out.rmdir();calls=[]
    collection=root/'owners/collection_recovery05';collection.mkdir(parents=True)
    (collection/'project_exit.json').write_text(json.dumps(dict(children_reaped=True)))
    previous=m.HERE/'driver_collection_recovery05/exit.json'
    original_read=m.read
    monkeypatch.setattr(m,'read',lambda p:dict(returncode=0) if Path(p)==previous else original_read(p))
    ages={r:{'frontend_epochs':60,'backend_epochs':60,'frontend_updates':7200,'backend_updates':22763} for r in 'AB'}
    (root/'actual_parent_ages.json').write_text(json.dumps(dict(roles=ages)))
    summary=dict(observable_pair_checks=dict(passed=47,unverified=1,failed=0,all_verified=False))
    monkeypatch.setattr(scope,'validated_summary',lambda _:summary)
    monkeypatch.setattr(production,'sources',lambda:{'frozen':'same'})
    (root/'common180_parents').mkdir()
    (root/'common180_parents/training_source_identity.json').write_text(json.dumps(dict(sources={'frozen':'same'})))
    monkeypatch.setattr(d,'run',lambda name,args,gpu:calls.append(name))
    monkeypatch.setattr(d,'storage_gate',lambda:calls.append('storage_gate'))
    d.main()
    assert calls==[r[0] for r in m.stages()]
    registered=original_read(out/'registration.json')
    assert registered['collection_summary']['observable_pair_checks']['all_verified'] is False
    assert original_read(out/'status.json')['stage']=='main_stages_complete'
