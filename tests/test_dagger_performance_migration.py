"""Measured implementation migration preserves durable work and fixed budgets."""
import copy
import subprocess
import time
from pathlib import Path

import pytest

from tools import tight_dagger as d
from tools.runtime import read, write, sha


def migration(tmp_path,monkeypatch):
    from tools import tight_finish
    code=tmp_path/'code';code.mkdir();(code/'adapter.py').write_text('reviewed implementation\n')
    root=tmp_path/'same_run';root.mkdir();(root/'stages').mkdir()
    profile=dict(first_frontend_round=2,collection={'capture_compression':'lzf'},cache={'transfer_mode':'episode','scheduler_index_mode':'host'},
        process_environment={'frontend':{'RESEARCH_CPUS':'12,13,14,15','RESEARCH_THREADS':2},'cache':{'RESEARCH_CPUS':'12,13','RESEARCH_THREADS':2}})
    cfg=dict(version='successful_takeover_pool_v1',groups={'QG01':{},'G01':{}},rounds=2,collection={'capture_compression':'gzip4_lossless_v1'})
    session=dict(source_head='old-source',started=100.,deadline=time.time()+7200,compute_deadline=time.time()+6600,dagger=cfg)
    write(root/'configuration.json',cfg);write(root/'session.json',session);write(root/'evaluation_plan.json',[{'seed':7}])
    write(root/'registration.json',dict(configuration_sha256=sha(root/'configuration.json'),evaluation_plan_sha256=sha(root/'evaluation_plan.json')))
    write(root/'launch.json',dict(pid=7,start_ticks=12,source_head='old-source',session_sha256=sha(root/'session.json')))
    write(root/'status.json',dict(state='STOPPED',owned_workers_exited=True,source_head='old-source'))
    write(root/'worker_processes.json',[]);write(root/'source_freeze.json',{'adapter.py':sha(code/'adapter.py')})
    write(root/'rounds.json',{'round_01':{'state':'RUNNING'}})
    done=root/'finished.json';write(done,dict(complete=True,updates=1355))
    receipt=root/'stages/r01_QG01_frontend.json'
    write(receipt,dict(args=['--fixed-args'],source_head='old-source',exit_code=0,completion=str(done),completion_sha256=sha(done)))
    write(root/'stages/r01_QG01_frontend_allocation.json',dict(deadline=1.,seconds=10800))
    weights=root/'retained.pt';weights.write_bytes(b'not a reload or an optimizer reset')
    evidence=code/'numeric.json';write(evidence,{'passed':True,'ticks':1297})
    sources={'adapter.py':sha(code/'adapter.py')};links=[dict(path=str(evidence),sha256=sha(evidence))]
    acceptance=code/'performance.json';write(acceptance,dict(passed=True,formal_exposure_unchanged=True,sources=sources,evidence=links,profile=profile))
    control=code/'control.json';write(control,dict(passed=True,expert_student_applied_equivalent=True,supervision_semantics_preserved=True,sources=sources,evidence=links))
    binding=dict(path=str(control),sha256=sha(control))
    write(root/'continuation_request.json',dict(control_acceptance=binding))
    request=dict(version='dagger_throughput_v1',started=session['started'],deadline=session['deadline'],compute_deadline=session['compute_deadline'],
        configuration_sha256=sha(root/'configuration.json'),evaluation_plan_sha256=sha(root/'evaluation_plan.json'),
        acceptance=dict(path=str(acceptance),sha256=sha(acceptance)),profile=profile,control_acceptance=binding)
    write(root/'performance_request.json',request)
    monkeypatch.setattr(d,'CODE',code);monkeypatch.setattr(d,'proc',lambda pid:None)
    monkeypatch.setattr(tight_finish,'matrix_worker_absence',lambda rows:(True,[]))
    def git(*args):
        if args[0]=='status':return ''
        if args[0]=='rev-parse':return 'new-reviewed-source'
        if args[0]=='ls-files':return 'adapter.py'
        raise AssertionError(args)
    monkeypatch.setattr(d,'git',git)
    return root,code,request,weights,receipt


def test_migration_keeps_same_root_deadlines_weights_and_completed_work(tmp_path,monkeypatch):
    root,code,request,weights,receipt=migration(tmp_path,monkeypatch)
    before=sha(weights);old_receipt=sha(receipt);old_allocation=sha(root/'stages/r01_QG01_frontend_allocation.json')
    d.performance_migrate(root)
    session=read(root/'session.json');change=read(root/'performance_revision.json')
    assert all(session[k]==request[k] for k in ('started','deadline','compute_deadline'))
    assert session['source_head']=='new-reviewed-source' and not change['deadline_reset']
    assert sha(weights)==before and sha(receipt)==old_receipt
    assert sha(root/'stages/r01_QG01_frontend_allocation.json')==old_allocation
    assert read(root/'revisions/dagger_throughput_v1/applied.json')['same_root']==str(root)
    assert read(root/'launch.json')['session_sha256']==sha(root/'session.json')
    assert read(root/'registration.json')['configuration_sha256']==sha(root/'configuration.json')
    monkeypatch.setattr(d.subprocess,'Popen',lambda *a,**k:pytest.fail('Completed stage cannot run again'))
    assert d.Stages(root).execute('r01_QG01_frontend',['--fixed-args'],1)['source_head']=='old-source'
    done=read(root/'finished.json');done['updates']+=1;write(root/'finished.json',done)
    with pytest.raises(ValueError,match='Completed receipt changed'):
        d.Stages(root).execute('r01_QG01_frontend',['--fixed-args'],1)


@pytest.mark.parametrize('corruption',['deadline','control','performance','live_owner'])
def test_invalid_migration_never_partially_rewrites_registered_metadata(tmp_path,monkeypatch,corruption):
    root,code,request,_,_=migration(tmp_path,monkeypatch)
    if corruption=='deadline':request['deadline']+=3600
    elif corruption=='control':
        receipt=read(code/'control.json');receipt['passed']=False;write(code/'control.json',receipt)
        request['control_acceptance']['sha256']=sha(code/'control.json')
    elif corruption=='performance':
        receipt=read(code/'performance.json');receipt['passed']=False;write(code/'performance.json',receipt)
        request['acceptance']['sha256']=sha(code/'performance.json')
    else:monkeypatch.setattr(d,'proc',lambda pid:dict(start_ticks=12))
    write(root/'performance_request.json',request)
    names=['session.json','configuration.json','registration.json','launch.json','continuation_request.json','status.json']
    before={name:sha(root/name) for name in names}
    with pytest.raises(ValueError):d.performance_migrate(root)
    assert all(sha(root/name)==digest for name,digest in before.items())
    assert not (root/'revisions/dagger_throughput_v1').exists()


def test_first_round_frontend_keeps_environment_but_second_round_uses_profile(tmp_path,monkeypatch):
    root,_,request,_,_=migration(tmp_path,monkeypatch)
    cfg=read(root/'configuration.json');cfg['performance']=request['profile'];write(root/'configuration.json',cfg)
    monkeypatch.setenv('RESEARCH_CPUS','22,23');monkeypatch.setenv('RESEARCH_THREADS','2')
    assert d.performance_environment(root,'r01_G01_frontend')['RESEARCH_CPUS']=='22,23'
    assert d.performance_environment(root,'r01_QG01_frontend')['RESEARCH_CPUS']=='22,23'
    assert d.performance_environment(root,'r02_G01_frontend')['RESEARCH_CPUS']=='12,13,14,15'
    assert d.performance_environment(root,'r01_G01_predictions')['RESEARCH_CPUS']=='12,13'


def test_prepared_backend_keeps_original_cache_and_training_configuration(tmp_path):
    root=tmp_path/'run';job=root/'round_01/QG01';back=job/'backend';front=job/'dp';back.mkdir(parents=True);(front/'training').mkdir(parents=True)
    (front/'training/best.pt').write_bytes(b'completed frontend')
    write(back/'prepared.json',dict(frontend_sha256=sha(front/'training/best.pt')))
    write(back/'configuration.json',dict(residual_pair={'version':'existing formal configuration'}))
    write(job/'initial_pair.json',{});write(root/'configuration.json',dict(performance={'cache':{'transfer_mode':'episode','scheduler_index_mode':'host'}}))
    before=sha(back/'configuration.json')
    assert d.prepare_backend(root,'QG01')==back and sha(back/'configuration.json')==before


def test_resume_clears_only_owned_backend_root_stop_after_verified_exit(tmp_path,monkeypatch):
    root,_,_,_,_=migration(tmp_path,monkeypatch)
    owned=[root/'STOP',root/'round_01/QG01/backend/STOP',root/'round_01/QG01/backend/QG01/STOP',root/'round_01/QG01/backend/QG01/training/STOP']
    for path in owned:path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    outside=tmp_path/'unrelated/STOP';outside.parent.mkdir();outside.touch()
    called=[]
    def run(path,*,recover):
        called.append((path,recover))
        assert not any(p.exists()for p in owned)
    monkeypatch.setattr(d,'run',run)
    d.resume(root)
    assert called==[(root,True)] and outside.exists()


@pytest.mark.parametrize('remote,prepared_ancestor,changed,rejected',[
    ('old-report',False,'research_docs/STATE_DP_RESULTS.md',False),
    ('new-reviewed-source',True,'',False),
    ('new-report',True,'research_docs/evidence/same_run/summary.json',False),
    ('old-code-change',False,'tools/unreviewed.py',True),
    ('new-code-change',True,'tools/unreviewed.py',True),
])
def test_watcher_reexports_boundary_even_when_old_public_receipt_says_pushed(tmp_path,monkeypatch,remote,prepared_ancestor,changed,rejected):
    from tools import tight_diagnostics
    root,code,request,_,_=migration(tmp_path,monkeypatch)
    deployment=tmp_path/'live';deployment.mkdir();(deployment/'adapter.py').write_bytes((code/'adapter.py').read_bytes())
    request.update(deployment_tree=str(deployment),prepared_source='new-reviewed-source',prepared_parent='old-public',
        boundary_stage='r01_QG01_frontend',owner=dict(pid=7,start_ticks=12),resume_service='test-perf-owner')
    write(root/'performance_request.json',request);write(root/'public_git.json',dict(pushed=True,head='stale-prior-public'))
    events=[]
    def git(*args):
        if args[:2]==('rev-parse','HEAD'):return 'new-reviewed-source'
        if args[:2]==('rev-parse','FETCH_HEAD'):return remote
        if args[:2]==('merge-base','--is-ancestor'):
            if args[2]=='new-reviewed-source' and not prepared_ancestor:
                raise subprocess.CalledProcessError(1,['git',*args])
            assert args[2] in ('new-reviewed-source','old-public') and args[3]==remote
        if args[:2]==('diff','--name-only'):
            assert args[2:]==(('new-reviewed-source' if prepared_ancestor else 'old-public'),remote)
            events.append(('remote_delta',args[2:]));return changed
        events.append(('git',args));return ''
    monkeypatch.setattr(d,'git',git)
    monkeypatch.setattr(tight_diagnostics,'dagger_publish',lambda path:events.append(('publish',path)))
    monkeypatch.setattr(d,'performance_migrate',lambda path:events.append(('migrate',path)))
    def launch(*args,**kwargs):
        events.append(('launch',args[0]))
        write(root/'status.json',dict(state='RUNNING'))
        write(root/'resume.json',dict(pid=99,start_ticks=22))
    monkeypatch.setattr(d.subprocess,'run',launch)
    monkeypatch.setattr(d,'proc',lambda pid:dict(start_ticks=22,state='S') if pid==99 else None)
    if rejected:
        with pytest.raises(ValueError,match='Unrelated remote changes'):d.performance_follow(root)
        assert not any(e[0] in ('publish','migrate','launch') for e in events)
        assert not any(e[0]=='git' and e[1][0]=='merge' for e in events)
        assert read(root/'public_git.json')['head']=='stale-prior-public'
        return
    d.performance_follow(root)
    assert ('remote_delta',('new-reviewed-source' if prepared_ancestor else 'old-public',remote)) in events
    assert ('publish',root)in events
    assert events.index(('publish',root))<events.index(('migrate',root))
    assert read(root/'performance_status.json')['state']=='RUNNING'
    assert read(root/'performance_status.json')['deadline']==request['deadline']
    assert read(root/'performance_status.json')['no_new_run']
