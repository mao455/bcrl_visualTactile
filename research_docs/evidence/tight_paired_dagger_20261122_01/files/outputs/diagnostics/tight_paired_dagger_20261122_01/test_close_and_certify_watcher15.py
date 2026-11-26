"""CPU temporary lifecycle/path fixtures only. Never invoke stop/certify/main."""
from pathlib import Path
import base64,hashlib,importlib.util,json,os
from types import SimpleNamespace
import pytest,psutil
P=Path(__file__).with_name('close_and_certify_watcher15.py')
spec=importlib.util.spec_from_file_location('closure15_test_helper',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def put(p,row):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(row));return p

@pytest.fixture
def env(tmp_path,monkeypatch):
    base=tmp_path/'project';a=base/'audit';r=base/'run';a.mkdir(parents=True);r.mkdir()
    for key,value in dict(BASE=base,A=a,R=r,OWNER=r/'owners/test_continuation15',DRIVER=a/'driver15').items():monkeypatch.setattr(m,key,value)
    return base,a,r

def absent(pid):raise psutil.NoSuchProcess(pid)

@pytest.mark.parametrize('row',[dict(pid=True,created=1),dict(pid=0,created=1),dict(pid=1,created=float('nan')),dict(pid=1,created=float('inf')),dict(pid=1,created=None)])
def test_invalid_pid_birth_refused(row):
    with pytest.raises(ValueError):m.absent(row,lookup=absent)


def test_exact_birth_or_independently_absent(env):
    assert m.absent(dict(pid=1,created=1),lookup=absent)
    assert not m.absent(dict(pid=1,created=1),lookup=lambda _:SimpleNamespace(create_time=lambda:1))
    assert m.absent(dict(pid=1,created=1),lookup=lambda _:SimpleNamespace(create_time=lambda:2))
    assert m.closed_window(1,1,2,lookup=absent)['birth_source']=='not_captured'
    assert m.closed_window(1,1,2,lookup=lambda _:SimpleNamespace(create_time=lambda:3))['later_reused_birth']==3
    with pytest.raises(ValueError):m.closed_window(1,1,2,lookup=lambda _:SimpleNamespace(create_time=lambda:1.5))


def test_exclusive_marker_refuses_existing_and_dangling_symlink(env):
    _,a,r=env;p=r/'marker';outside=a/'outside'
    p.symlink_to(outside)
    with pytest.raises(ValueError):m.exclusive_bytes(p,b'no',r)
    assert not outside.exists();p.unlink();m.exclusive_bytes(p,b'owned',r)
    with pytest.raises(FileExistsError):m.exclusive_bytes(p,b'rewrite',r)
    assert p.read_bytes()==b'owned'


def test_canonical_rejects_dotdot_and_parent_symlink(env):
    _,a,r=env
    with pytest.raises(ValueError):m.canonical(r/'..'/'audit'/'x',r)
    (r/'link').symlink_to(a,target_is_directory=True)
    with pytest.raises(ValueError):m.canonical(r/'link'/'x',r)


def test_census_external_process_is_not_veto(env):
    p=SimpleNamespace(pid=10,info=dict(cmdline=['python','external_test.py'],uids=SimpleNamespace(real=os.getuid())),create_time=lambda:1)
    assert m.own_census(processes=[p])['matching_processes']==[]
    p.info['cmdline']=['python','-m',m.MODULE,'worker','--output','irrelevant']
    with pytest.raises(ValueError,match='still live'):m.own_census(processes=[p])


def test_exact_tail_helper_census_without_historical_birth(env):
    assert m.own_command(['python','-m','tools.tight_paired_evaluate','summarize','--output',str(m.R/'evaluation/test')])
    assert m.own_command(['python',str(m.A/'audit_collection_recovery05.py'),'noise','--collection',str(m.R/'collection')])
    assert not m.own_command(['python','-m','tools.tight_paired_evaluate','summarize','--output','/another/test'])


def backups(env):
    _,a,r=env;w=a/'watch_shared_capacity15';d=w/'resource_watch_v4_backups_1_123';d.mkdir(parents=True)
    put(w/'resource_watch_v4_exit.json',dict(pid=123,conflicts=[],modified_schedules=0,exact_original_byte_restoration=True,no_process_signals=True))
    target=r/'wave17/concurrency.json';target.parent.mkdir();raw=b'{"original":true}';target.write_bytes(raw)
    row=dict(path=str(target),existed=True,base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
    backup=put(d/(hashlib.sha256(str(target).encode()).hexdigest()+'.json'),row)
    absenttarget=r/'wave18/concurrency.json';absenttarget.parent.mkdir()
    put(d/(hashlib.sha256(str(absenttarget).encode()).hexdigest()+'.json'),dict(path=str(absenttarget),existed=False,base64=None,sha256=None))
    return w,d,target,backup,row


def test_every_dynamic_backup_present_and_absent_exact(env):
    backups(env);_,rows=m.restored_backups(dict(pid=123));assert len(rows)==2 and {r['existed'] for r in rows}=={True,False}

@pytest.mark.parametrize('change',['bytes','dotdot','parent_symlink','backup_dir_symlink','bad_existed'])
def test_invalid_restoration_boundaries_refused(env,change):
    w,d,target,backup,row=backups(env)
    if change=='bytes':target.write_bytes(b'changed')
    elif change=='dotdot':row['path']=str(m.R/'..'/'audit'/'concurrency.json');put(backup,row)
    elif change=='parent_symlink':
        target.unlink();target.parent.rmdir();target.parent.symlink_to(m.A,target_is_directory=True)
    elif change=='backup_dir_symlink':
        moved=m.A/'elsewhere';d.rename(moved);d.symlink_to(moved,target_is_directory=True)
    else:row['existed']='yes';put(backup,row)
    with pytest.raises(ValueError):m.restored_backups(dict(pid=123))


def lifecycle_fixture(env,monkeypatch):
    _,a,r=env;monkeypatch.setattr(psutil,'Process',absent)
    monkeypatch.setattr(m,'service',lambda _:('fixture',dict(MainPID='0',Result='success',ActiveState='inactive',ControlGroup='',InvocationID='one')))
    monkeypatch.setattr(m,'own_census',lambda:dict(matching_processes=[],signals=0))
    reg=put(a/'TEST_CONTINUATION15_REGISTRATION.json',dict(tail_stages=[dict(name='summarize_test',skip_closed=False),dict(name='audit_collection_cuda_noise_recovery05',skip_closed=False)]))
    review=put(a/'ROOT_TEST_CONTINUATION15_REVIEW.json',{})
    actors={m.DRIVER_UNIT:dict(pid=100,created=1,cwd=str(m.BASE),command=['python','-m',m.MODULE,'driver']),
            m.WATCH_UNIT:dict(pid=101,created=1,cwd=str(m.BASE),command=['python','-m',m.WATCH_MODULE,'--apply'])}
    put(a/'TEST_CONTINUATION15_ACTIVATED.json',dict(registry_sha256=m.sha(reg),root_review_sha256=m.sha(review),actors=actors))
    put(m.DRIVER/'driver_exit.json',dict(pid=100,utc=200,returned_after_owned_helpers=True))
    put(m.DRIVER/'closure.json',dict(utc=199,status='original_missing47_queue_and_audits_complete'))
    put(m.DRIVER/'registration.json',dict(activated=True,activated_utc=2))
    put(m.OWNER/'project_job_started.json',dict(pid=102,utc=3))
    put(m.OWNER/'project_exit.json',dict(utc=150,children_reaped=True,worker_reaped=True,monitor_reaped=True))
    put(m.OWNER/'project_complete.json',dict(utc=149,status='finite47_queue_consumed_all',stages=47,all_children_reaped=True))
    put(m.OWNER/'monitor.json',dict(pid=103,started=3));put(m.OWNER/'monitor_exit.json',dict(pid=103,ended=150))
    put(m.OWNER/'project_status.json',dict(pid=163,stage=46,utc=96))
    (m.OWNER/'dispatches').mkdir()
    for stage,index in enumerate(range(17,64)):
        put(m.OWNER/'launches'/f'wave{index:02d}.json',dict(index=index,stage=stage,utc=4+stage*2))
        put(m.OWNER/'stage_exits'/f'wave{index:02d}.json',dict(index=index,stage=stage,utc=5+stage*2,child_reaped=True,process_started=True,queue_aborted=False,returncode=0))
        put(m.OWNER/'dispatches'/f'wave{index:02d}.json',dict(index=index,pid=100+index,utc=4+stage*2,dispatch_attempt=1,rerun=False))
    for name in ('summarize_test','audit_collection_cuda_noise_recovery05'):put(r/'cycle_logs_continuation15'/(name+'_exit.json'),dict(returncode=0,utc=180))
    put(a/'ROOT_LIVE15_IDENTITIES.json',dict(identities={}))


def test_direct_actor_and47_reap_evidence_replaces_sparse_snapshot(env,monkeypatch):
    lifecycle_fixture(env,monkeypatch);result=m.lifecycle()
    assert len(result['closed_windows'])==50 and len(result['helpers'])==2 and result['sampled_exact_births']==[]

@pytest.mark.parametrize('change',['live_driver','wrong_driver','unreaped_stage','missing_stage','monitor_exit','driver_service'])
def test_lifecycle_refuses_incomplete_actual_reaping(env,monkeypatch,change):
    lifecycle_fixture(env,monkeypatch)
    if change=='live_driver':
        def lookup(pid):
            if pid==100:return SimpleNamespace(create_time=lambda:1)
            raise psutil.NoSuchProcess(pid)
        monkeypatch.setattr(psutil,'Process',lookup)
    elif change=='wrong_driver':put(m.DRIVER/'driver_exit.json',dict(pid=999,utc=200,returned_after_owned_helpers=True))
    elif change=='unreaped_stage':
        p=m.OWNER/'stage_exits/wave30.json';row=json.loads(p.read_bytes());row['child_reaped']=False;put(p,row)
    elif change=='missing_stage':(m.OWNER/'stage_exits/wave30.json').unlink()
    elif change=='monitor_exit':put(m.OWNER/'monitor_exit.json',dict(pid=999,ended=150))
    else:monkeypatch.setattr(m,'service',lambda _:('fixture',dict(MainPID='100',Result='success',ActiveState='active',ControlGroup='',InvocationID='one')))
    with pytest.raises(ValueError):m.lifecycle()
