"""CPU-only fixtures: no actual collection, aggregate, owner or GPU launches."""
import copy
import json
import sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import wait_then_main_recovery05 as m


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'run';here=tmp_path/'audit';root.mkdir();here.mkdir()
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'HERE',here);monkeypatch.setattr(m,'OUT',here/'wait')
    monkeypatch.setattr(m.time,'time',lambda:100.)
    put(root/'session.json',dict(deadline=1000.))
    (here/'run_collection_recovery05.py').write_text('synthetic frozen source')
    put(here/'driver_collection_recovery05/registration.json',dict(command=m.original_command(root),deadline=1000.,
        driver_sha256=m.digest(here/'run_collection_recovery05.py'),scope='81 previously unassigned opportunities; no retry'))
    return root,here


def closed_fixture(root,here):
    owner=root/'owners/collection_recovery05'
    job=dict(resource_root=str(owner),operation='dagger',stages=[dict(entry='paired_collection_recovery05',args=['--output',str(root/'collection_recovery05')])])
    command=[sys.executable,'-m',m.MODULE,'worker','--output',str(root/'collection_recovery05')]
    put(root/'collection_recovery05_job.json',job)
    put(owner/'project_job_started.json',dict(job=job,commands=[command],utc=10.))
    put(owner/'project_status.json',dict(command=command,stage=0))
    put(owner/'project_exit.json',dict(children_reaped=True,utc=90.))
    put(owner/'project_complete.json',dict(stages=1,utc=89.))
    put(here/'driver_collection_recovery05/exit.json',dict(returncode=0,utc=91.))
    put(here/'driver_collection_recovery05/status.json',dict(stage='collection_suffix_closed',utc=92.))


def test_live_suffix_only_waits(fixture):
    root,here=fixture;assert m.closure(root) is None


def test_exact_closed_receipts_accepted(fixture):
    root,here=fixture;closed_fixture(root,here);assert m.closure(root)['owner_exit']['children_reaped']


@pytest.mark.parametrize('mutation',['command','deadline','driver_sha','failure','exit_nonzero','unreaped','job','worker','times','status','owner_failed','owner_incomplete'])
def test_wrong_or_failed_closure_never_admits(fixture,mutation):
    root,here=fixture;closed_fixture(root,here)
    if mutation in ('command','deadline','driver_sha'):
        p=here/'driver_collection_recovery05/registration.json';d=m.read(p)
        d[dict(command='command',deadline='deadline',driver_sha='driver_sha256')[mutation]]='changed';put(p,d)
    elif mutation=='failure':put(here/'driver_collection_recovery05/failure.json',dict(error='failed'))
    elif mutation=='exit_nonzero':put(here/'driver_collection_recovery05/exit.json',dict(returncode=1))
    elif mutation=='unreaped':put(root/'owners/collection_recovery05/project_exit.json',dict(children_reaped=False))
    elif mutation=='job':
        p=root/'collection_recovery05_job.json';d=m.read(p);d['operation']='train';put(p,d)
    elif mutation=='worker':
        p=root/'owners/collection_recovery05/project_status.json';d=m.read(p);d['command']=['other'];put(p,d)
    elif mutation=='times':put(root/'owners/collection_recovery05/project_exit.json',dict(children_reaped=True,utc=94.))
    elif mutation=='status':put(here/'driver_collection_recovery05/status.json',dict(stage='stopped_with_evidence',utc=92.))
    elif mutation=='owner_failed':put(root/'owners/collection_recovery05/project_failure.json',{})
    else:put(root/'owners/collection_recovery05/project_complete.json',dict(stages=0))
    with pytest.raises(ValueError):m.closure(root)


def test_atomic_terminal_status_race_keeps_waiting(fixture):
    root,here=fixture;closed_fixture(root,here)
    put(here/'driver_collection_recovery05/status.json',dict(stage='collection_running',utc=89.))
    assert m.closure(root) is None


@pytest.mark.parametrize('mutation',['aggregate_only','index_only','one_artifact','symlink'])
def test_partial_existing_aggregate_never_overwritten(fixture,mutation):
    root,here=fixture
    if mutation=='aggregate_only':(root/'collection_aggregate_recovery05').mkdir()
    elif mutation=='index_only':put(root/'collection/recovery05_active_aggregate.json',{})
    elif mutation=='one_artifact':put(root/'collection/results.json',[])
    else:(root/'collection_aggregate_recovery05').symlink_to('/tmp/absent-fixture-target')
    with pytest.raises(ValueError,match='without overwrite'):m.index_state(root)


def published_fixture(root):
    original=root/'collection';aggregate=root/'collection_aggregate_recovery05';aggregate.mkdir()
    recovered=root/'collection_consumed_recovered05';continued=root/'collection_recovery05'
    put(original/'plan.json',dict(frozen=True));put(recovered/'recovered_results.json',[]);put(continued/'results.json',[])
    for name in ('results.json','correction_entries.json','collection_summary.json'):
        put(original/name,{});put(aggregate/name,{})
    binding=dict(source_directories=list(map(str,(original,recovered,continued))),no_initiated_opportunity_reexecuted=True,
        original_plan_sha256=m.digest(original/'plan.json'),recovered_results_sha256=m.digest(recovered/'recovered_results.json'),continued_results_sha256=m.digest(continued/'results.json'))
    put(aggregate/'recovery05_aggregate_binding.json',binding);put(original/'recovery05_aggregate_binding.json',binding)
    put(original/'recovery05_active_aggregate.json',dict(path=str(aggregate),binding_sha256=m.digest(aggregate/'recovery05_aggregate_binding.json'),indexed_files=['results.json','correction_entries.json','collection_summary.json']))
    put(aggregate/'paired_queue_assignments.json',[dict(attempt='original' if i<15 else 'continued') for i in range(96)])
    return dict(opportunities=96,observable_pair_checks=dict(passed=47,unverified=1,failed=0,all_verified=False))


def test_existing_exact_index_validates_without_writer(fixture,monkeypatch):
    import collection_recovery05_scope as scope
    root,here=fixture;summary=published_fixture(root)
    monkeypatch.setattr(scope,'validated_summary',lambda _:summary)
    before={str(p):p.read_bytes() for p in root.rglob('*.json')}
    assert m.index_state(root)=='published' and m.validate_index(root)==summary
    assert before=={str(p):p.read_bytes() for p in root.rglob('*.json')}


@pytest.mark.parametrize('mutation',['pointer','source','binding','attempts','fake_all48'])
def test_published_index_tamper_rejected(fixture,monkeypatch,mutation):
    import collection_recovery05_scope as scope
    root,here=fixture;summary=published_fixture(root);monkeypatch.setattr(scope,'validated_summary',lambda _:summary)
    if mutation=='pointer':put(root/'collection/recovery05_active_aggregate.json',{})
    elif mutation=='source':put(root/'collection_recovery05/results.json',['changed'])
    elif mutation=='binding':put(root/'collection/recovery05_aggregate_binding.json',{})
    elif mutation=='attempts':put(root/'collection_aggregate_recovery05/paired_queue_assignments.json',[])
    else:summary['observable_pair_checks']['passed']=48
    with pytest.raises((ValueError,KeyError)):m.validate_index(root)


def waiter_fixture(root,here,monkeypatch):
    d=m.Waiter();events=[]
    monkeypatch.setattr(m,'frozen_identity',lambda:dict(frozen=True))
    monkeypatch.setattr(m,'verify_identity',lambda f:events.append('verify'))
    monkeypatch.setattr(m,'closure',lambda:dict(closed=True))
    monkeypatch.setattr(m,'validate_index',lambda:dict(opportunities=96))
    monkeypatch.setattr(d,'run',lambda name,args:events.append(name))
    return d,events


@pytest.mark.parametrize('published',[False,True])
def test_wait_aggregate_once_or_validate_existing_then_exact_main(fixture,monkeypatch,published):
    root,here=fixture;d,events=waiter_fixture(root,here,monkeypatch)
    monkeypatch.setattr(m,'index_state',lambda:'published' if published else 'absent')
    d.main()
    assert events==(['verify','verify','reviewed_main'] if published else ['verify','aggregate_and_publish_once','verify','reviewed_main'])
    reg=m.read(here/'wait/registration.json');assert reg['poll_seconds']==30
    command=reg['aggregate_command'];assert command[-1]=='--publish-index' and command.count('--output')==1
    assert reg['main_command']==[sys.executable,str(here/'run_main_recovery05.py')]


def test_wait_polls30_cpu_only_before_closure(fixture,monkeypatch):
    root,here=fixture;d,events=waiter_fixture(root,here,monkeypatch);steps=iter([None,None,dict(closed=True)])
    monkeypatch.setattr(m,'closure',lambda:next(steps));monkeypatch.setattr(m,'index_state',lambda:'published')
    monkeypatch.setattr(m.time,'sleep',lambda seconds:events.append(('sleep',seconds)))
    d.main();assert events[:2]==[('sleep',30),('sleep',30)]
    assert events[2:]==['verify','verify','reviewed_main']


def test_changed_frozen_source_blocks_aggregate_and_main(fixture,monkeypatch):
    root,here=fixture;d,events=waiter_fixture(root,here,monkeypatch)
    def reject(_):raise ValueError('changed during wait')
    monkeypatch.setattr(m,'verify_identity',reject)
    with pytest.raises(ValueError):d.main()
    assert events==[]


def test_launched_aggregate_failure_never_retries_or_starts_main(fixture,monkeypatch):
    root,here=fixture;d,events=waiter_fixture(root,here,monkeypatch);monkeypatch.setattr(m,'index_state',lambda:'absent')
    def fail(name,args):events.append(name);raise ValueError('launched child failed')
    monkeypatch.setattr(d,'run',fail)
    with pytest.raises(ValueError):d.main()
    assert events==['verify','aggregate_and_publish_once']


@pytest.mark.parametrize('mode',['deadline','stop'])
def test_finite_wait_never_launches_on_stop_or_deadline(fixture,monkeypatch,mode):
    root,here=fixture;d,events=waiter_fixture(root,here,monkeypatch)
    if mode=='deadline':d.deadline=99.
    else:d.stop.touch()
    with pytest.raises(ValueError):d.main()
    assert events==[]


def test_owned_helper_failure_records_exit_and_no_retry(fixture,monkeypatch):
    import driver_support_v4 as support
    root,here=fixture;d=m.Waiter();(here/'wait').mkdir();calls=[]
    monkeypatch.setattr(support,'run_owned',lambda *args:(calls.append(args) or 9))
    with pytest.raises(ValueError,match='no retry'):d.run('child',['synthetic'])
    assert len(calls)==1 and m.read(here/'wait/child_exit.json')['returncode']==9


def test_frozen_identity_changed_source_or_file_rejected(fixture,monkeypatch):
    from tools import tight_paired_dagger as production
    root,here=fixture;p=here/'frozen.py';p.write_text('old')
    frozen=dict(production_sha256={'p':'old'},orchestration_and_inputs_sha256={str(p):m.digest(p)})
    monkeypatch.setattr(production,'sources',lambda:{'p':'old'});m.verify_identity(frozen)
    p.write_text('changed')
    with pytest.raises(ValueError,match='Frozen orchestration'):m.verify_identity(frozen)
    monkeypatch.setattr(production,'sources',lambda:{'p':'changed'})
    with pytest.raises(ValueError,match='Production source'):m.verify_identity(frozen)


def test_owned_helper_exception_preserves_failure_without_replacement(fixture,monkeypatch):
    import driver_support_v4 as support
    root,here=fixture;d=m.Waiter();(here/'wait').mkdir();calls=[]
    def fail(*args):calls.append(args);raise InterruptedError('STOP')
    monkeypatch.setattr(support,'run_owned',fail)
    with pytest.raises(InterruptedError):d.run('child',['synthetic'])
    assert len(calls)==1 and 'STOP' in m.read(here/'wait/child_exception.json')['error']
