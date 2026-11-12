"""Only independent DP stages overlap; durable ownership remains per worker."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools import tight_dagger as owner


def configuration():
    return dict(version='successful_takeover_pool_v1',matched_protocol={'version':'matched'},performance=dict(
        parallel_frontends=True,first_frontend_round=2,
        process_environment=dict(frontend=dict(RESEARCH_THREADS='2')),
        frontend_process_environment=dict(QG01=dict(RESEARCH_CPUS='16,17'),G01=dict(RESEARCH_CPUS='18,19'))))


def test_parallel_scope_preserves_first_round_and_complete_pair_requirement():
    cfg=configuration()
    assert not owner.parallel_frontends_enabled(cfg,1,['QG01','G01'])
    assert owner.parallel_frontends_enabled(cfg,2,['QG01','G01'])
    with pytest.raises(ValueError):owner.parallel_frontends_enabled(cfg,2,['QG01'])
    assert not owner.parallel_frontends_enabled({},2,['QG01','G01'])


def test_frontend_process_affinity_is_separate_and_scope_limited(tmp_path):
    owner.write(tmp_path/'configuration.json',configuration())
    a=owner.performance_environment(tmp_path,'r02_QG01_frontend')
    b=owner.performance_environment(tmp_path,'r02_G01_frontend')
    assert a['RESEARCH_CPUS']=='16,17' and b['RESEARCH_CPUS']=='18,19'
    assert a['RESEARCH_THREADS']==b['RESEARCH_THREADS']=='2'


def test_two_concurrent_stages_reap_their_own_pid_and_preserve_completed_receipts(tmp_path,monkeypatch):
    from tools import tight_finish
    cfg=configuration();owner.write(tmp_path/'configuration.json',cfg)
    owner.write(tmp_path/'session.json',dict(deadline=time.time()+60,source_head='source'))
    owner.write(tmp_path/'source_freeze.json',{})
    start=threading.Barrier(2);counter=iter((501,502));launch_lock=threading.Lock();launched=[]
    class Worker:
        def __init__(self,args,**kwargs):
            with launch_lock:self.pid=next(counter);launched.append(self.pid)
            self.returncode=None;self.calls=0;start.wait(timeout=3)
        def poll(self):
            self.calls+=1
            if self.calls>=3:self.returncode=0
            return self.returncode
    monkeypatch.setattr(owner.subprocess,'Popen',Worker)
    monkeypatch.setattr(owner,'proc',lambda pid:dict(pid=pid,start_ticks=pid+100))
    monkeypatch.setattr(owner,'kill_owned',lambda worker:None)
    monkeypatch.setattr(tight_finish,'matrix_worker_absence',lambda records:(True,[dict(r) for r in records]))
    real_sleep=time.sleep;monkeypatch.setattr(owner.time,'sleep',lambda seconds:real_sleep(.01))
    stages=owner.Stages(tmp_path)
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(stages.execute,name,['-c','pass'],30) for name in ('r02_QG01_frontend','r02_G01_frontend')]
        for future in futures:future.result()
    records=owner.read(tmp_path/'worker_processes.json');assert len(records)==2
    for record in records:
        reaped=owner.read(tmp_path/(record['phase']+'_reaped.json'))
        assert reaped['workers'][0]['pid']==record['pid']
        assert owner.read(tmp_path/'stages'/(record['phase']+'.json'))['exit_code']==0
    assert owner.read(tmp_path/'active_stage.json')['active_stages']==[]
    # Recovery reads durable stage receipts and never starts a third process.
    assert stages.execute('r02_G01_frontend',['-c','pass'],30)['exit_code']==0
    assert len(launched)==2


def test_common_snapshot_preparations_precede_both_dp_workers(tmp_path,monkeypatch):
    owner.write(tmp_path/'configuration.json',configuration())
    snapshot=tmp_path/'snapshot.json';owner.write(snapshot,dict(snapshot='same'))
    prepared=[];barrier=threading.Barrier(2);calls=[]
    def prepare(root,group,round_index,stages,path,reuse):
        assert path==snapshot and round_index==2
        prepared.append(group);front=root/'round_02'/group/'dp';front.mkdir(parents=True)
        return front
    def train(root,group,round_index,stages,front):
        assert set(prepared)=={'QG01','G01'}
        barrier.wait(timeout=3);calls.append(group);return front
    monkeypatch.setattr(owner,'prepare_frontend_stages',prepare)
    monkeypatch.setattr(owner,'train_frontend_stage',train)
    fronts,errors=owner.train_frontends_parallel(tmp_path,2,['QG01','G01'],None,snapshot,None)
    assert not errors and set(fronts)==set(calls)=={'QG01','G01'}
    result=owner.read(tmp_path/'round_02/parallel_frontends.json')
    assert result['snapshot_sha256']==owner.sha(snapshot)
    assert all(v['complete'] for v in result['groups'].values())


def test_frontend_failure_preserves_other_completed_group_without_backend_substitution(tmp_path,monkeypatch):
    owner.write(tmp_path/'configuration.json',configuration())
    snapshot=tmp_path/'snapshot.json';owner.write(snapshot,{})
    def prepare(root,group,*args):
        front=root/'round_02'/group/'dp';front.mkdir(parents=True);return front
    def train(root,group,*args):
        if group=='QG01':raise ValueError('real input failure')
        owner.write(root/'round_02/G01/dp/completed.json',dict(complete=True))
    monkeypatch.setattr(owner,'prepare_frontend_stages',prepare);monkeypatch.setattr(owner,'train_frontend_stage',train)
    _,errors=owner.train_frontends_parallel(tmp_path,2,['QG01','G01'],None,snapshot,None)
    assert set(errors)=={'QG01'}
    assert owner.read(tmp_path/'round_02/G01/dp/completed.json')['complete']
    assert not (tmp_path/'STOP').exists()


def test_stop_requests_both_workers_save_before_owner_unwinds(tmp_path,monkeypatch):
    owner.write(tmp_path/'configuration.json',configuration())
    snapshot=tmp_path/'snapshot.json';owner.write(snapshot,{})
    def prepare(root,group,*args):
        front=root/'round_02'/group/'dp';front.mkdir(parents=True);return front
    def train(root,group,*args):
        if group=='QG01':raise InterruptedError('STOP at complete boundary')
        deadline=time.time()+3
        while not (root/'STOP').exists():
            assert time.time()<deadline
            time.sleep(.01)
        owner.write(root/'round_02/G01/dp/saved.json',dict(saved=True))
        raise InterruptedError('Other complete boundary saved')
    monkeypatch.setattr(owner,'prepare_frontend_stages',prepare);monkeypatch.setattr(owner,'train_frontend_stage',train)
    with pytest.raises(InterruptedError):owner.train_frontends_parallel(tmp_path,2,['QG01','G01'],None,snapshot,None)
    assert owner.read(tmp_path/'round_02/G01/dp/saved.json')['saved']
