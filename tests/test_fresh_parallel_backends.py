"""CPU orchestration: actual chain overlap, failure isolation and bounded owned cleanup."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tools import tight_fresh_cycle as cycle
from visuotactile.training_resources import atomic_json


@pytest.mark.parametrize('failed',[None,'A','B'])
def test_backend_chains_overlap_keep_order_and_join_independent_peer(tmp_path,failed):
    barrier=threading.Barrier(2,timeout=3)
    offline_entered=threading.Event();release_offline=threading.Event()
    seen={'A':[],'B':[]}
    survivor='B' if failed!='B' else 'A'
    def execute(name,args,completion,*,update_status,stop_paths):
        stage,role=name.rsplit('_',1);stage=stage.removeprefix('backend_')
        assert not update_status and stop_paths==(tmp_path/'STOP',)
        seen[role].append(stage)
        if stage=='prepare':
            assert '--continue-rear' in args
            barrier.wait()
        if stage=='cache' and role==failed:raise RuntimeError('saved failed cache '+role)
        if stage=='offline' and role==survivor:
            offline_entered.set();assert release_offline.wait(3)
        completion.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(completion,dict(complete=True))
    with ThreadPoolExecutor(max_workers=1) as pool:
        result=pool.submit(cycle.paired_backends,execute,tmp_path,15,
            continue_rear=True,stop_paths=(tmp_path/'STOP',))
        try:
            assert offline_entered.wait(3)
            assert not result.done(), 'Returning before the other chain completes would admit evaluation early'
        finally:release_offline.set()
        receipt=result.result(timeout=3)
    assert receipt['complete']==(failed is None)
    assert set(receipt['errors'])==({failed} if failed else set())
    for role in ('A','B'):
        assert seen[role]==(['prepare','cache'] if role==failed else ['prepare','cache','train','offline'])
        assert receipt['outcomes'][role]['complete']==(role!=failed)
        assert (tmp_path/role/'backend'/role/'offline/result.json').exists()==(role!=failed)


class CycleFixture:
    def __init__(self,tmp_path,monkeypatch,*,admit=True,backend_parallel=False,frontend_parallel=False,
            failure=None,stop_after=None):
        self.root=tmp_path/'source';self.output=tmp_path/'next';self.collection=tmp_path/'collection'
        self.ledger=tmp_path/'next_stages';self.calls=[];self.gates=[];self.waits=[]
        self.deadline=time.time()+100
        self.root.mkdir();self.collection.mkdir()
        atomic_json(self.root/'session.json',dict(deadline=self.deadline,resource_policy='full_gpu_v1'))
        atomic_json(self.collection/'result.json',dict(software_complete=True))
        atomic_json(self.collection/'correction_entries.json',{'real':{}})
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cycle.subprocess,'check_output',lambda *a,**k:'frozen-commit')
        barriers={name:threading.Barrier(2,timeout=3) for name in ('frontend','backend_prepare')}
        def emit(path,complete=True):
            path.parent.mkdir(parents=True,exist_ok=True)
            atomic_json(path,dict(complete=complete))
        def gate(resource,receipt_path,*,deadline,stop_paths):
            name=receipt_path.stem;self.gates.append(name)
            assert deadline==self.deadline and resource==self.root/'shared_resources.json'
            assert stop_paths==(self.root/'STOP',self.output/'STOP',self.ledger/'STOP')
            if name=='parallel_backend_admission':
                assert all((self.output/role/'training/finished.json').exists() for role in ('A','B'))
                assert not any(name.startswith('backend_') for name,command in self.calls)
            return dict(admitted=admit,reason='admitted' if admit else 'wait_exhausted',deadline=deadline)
        def ordinary_wait(proc,deadline,**kwargs):
            self.waits.append((proc.stage,deadline,False));return None
        def backend_wait(proc,deadline,stop_paths):
            self.waits.append((proc.stage,deadline,True));return None
        def launch(command,**kwargs):
            module=command[2];role=None
            def argument(key):return command[command.index(key)+1]
            if module=='tools.tight_fresh_dagger':
                name='aggregate' if command[3]=='aggregate' else 'visual_cache'
                completion=self.output/('prepared.json' if name=='aggregate' else 'B/cache_index.json')
            elif module=='tools.tight_fresh_train':
                role=argument('--role');name='frontend_'+role;completion=self.output/role/'training/finished.json'
            elif module=='tools.tight_fresh_backend':
                role=argument('--role');name='backend_prepare_'+role;completion=self.output/role/'backend/prepared.json'
                assert all((self.output/r/'training/finished.json').exists() for r in ('A','B'))
                assert int(argument('--epochs'))==15 and '--continue-rear' in command
            elif module=='tools.tight_finish':
                rear=Path(argument('--root'));role=rear.parent.name;name='backend_cache_'+role
                assert (rear/'prepared.json').exists();completion=rear/'cache_complete.json'
            elif module=='visuotactile.anchor_continue':
                destination=Path(argument('--output'));rear=Path(argument('--root'));role=rear.name
                stage='train' if command[3]=='residual-train' else 'offline';name='backend_'+stage+'_'+role
                predecessor=rear.parent/'cache_complete.json' if stage=='train' else rear/'training/finished.json'
                assert predecessor.exists();completion=destination/('finished.json' if stage=='train' else 'result.json')
            else:
                assert module=='tools.tight_fresh_rollouts'
                assert all((self.output/r/'backend'/r/'offline/result.json').exists() for r in ('A','B'))
                name='development_prepare' if command[3]=='prepare' else 'development'
                completion=self.output/'development'/('plan.json' if name=='development_prepare' else 'result.json')
            self.calls.append((name,command))
            if frontend_parallel and name.startswith('frontend_'):barriers['frontend'].wait()
            if backend_parallel and name.startswith('backend_prepare_'):barriers['backend_prepare'].wait()
            code=1 if name==failure else 0
            if not code:emit(completion)
            if name==stop_after:(self.ledger/'STOP').touch()
            return SimpleNamespace(stage=name,pid=os.getpid(),returncode=code,poll=lambda:code)
        monkeypatch.setattr(cycle,'wait_parallel_admission',gate)
        monkeypatch.setattr(cycle,'wait_owned',ordinary_wait)
        monkeypatch.setattr(cycle,'wait_backend_owned',backend_wait)
        monkeypatch.setattr(cycle.subprocess,'Popen',launch)

    def run(self,**options):
        return cycle.run(self.root,self.collection,self.output,10,15,4,123,continue_rear=True,**options)


@pytest.mark.parametrize('frontends',[False,True])
def test_formal_cycle_admits_after_both_frontends_and_joins_before_development(tmp_path,monkeypatch,frontends):
    fixture=CycleFixture(tmp_path,monkeypatch,backend_parallel=True,frontend_parallel=frontends)
    fixture.run(parallel_backends=True,parallel_frontends=frontends)
    assert fixture.gates==(['parallel_admission'] if frontends else [])+['parallel_backend_admission']
    registration=json.loads((fixture.ledger/'registration.json').read_text())
    assert registration['parallel_backends'] is True
    assert registration['parallel_backend_contract']['deadline']==fixture.deadline
    assert all(deadline==fixture.deadline for _,deadline,_ in fixture.waits)
    assert all(checked==name.startswith('backend_') for name,deadline,checked in fixture.waits)
    receipt=json.loads((fixture.ledger/'finished.json').read_text())
    names=[row['stage'] for row in receipt['stages']]
    assert len(names)==len(set(names))==14
    assert names[-2:]==['development_prepare','development']
    assert json.loads((fixture.ledger/'parallel_backends.json').read_text())['complete']


@pytest.mark.parametrize('enabled',[False,True])
def test_default_and_unadmitted_request_keep_serial_backend_commands(tmp_path,monkeypatch,enabled):
    fixture=CycleFixture(tmp_path,monkeypatch,admit=False)
    fixture.run(parallel_backends=enabled)
    names=[name for name,command in fixture.calls]
    assert names[4:12]==[f'backend_{stage}_{role}' for role in ('A','B') for stage in ('prepare','cache','train','offline')]
    assert fixture.gates==(['parallel_backend_admission'] if enabled else [])
    registration=json.loads((fixture.ledger/'registration.json').read_text())
    assert ('parallel_backends' in registration)==enabled
    assert ('parallel_backend_contract' in registration)==enabled
    assert not (fixture.ledger/'parallel_backends.json').exists()
    if enabled:
        assert json.loads((fixture.ledger/'parallel_backend_admission.json').read_text())['backend_mode']=='serial_fallback'
    else:assert not (fixture.ledger/'parallel_backend_admission.json').exists()


@pytest.mark.parametrize('failed',['A','B'])
def test_formal_failure_preserves_peer_manifest_and_blocks_development(tmp_path,monkeypatch,failed):
    fixture=CycleFixture(tmp_path,monkeypatch,backend_parallel=True,failure='backend_cache_'+failed)
    with pytest.raises(RuntimeError,match='Backend stage failed'):fixture.run(parallel_backends=True)
    survivor='B' if failed=='A' else 'A'
    assert (fixture.output/survivor/'backend'/survivor/'offline/result.json').exists()
    assert not (fixture.output/failed/'backend'/failed/'training').exists()
    assert not (fixture.output/'development').exists()
    assert not (fixture.ledger/'finished.json').exists()
    failure=json.loads((fixture.ledger/'failure.json').read_text())
    rows={row['stage']:row for row in failure['stages']}
    assert rows['backend_cache_'+failed]['returncode']==1
    assert rows['backend_offline_'+survivor]['returncode']==0
    for name in rows:
        assert (fixture.ledger/(name+'.json')).exists()
        assert (fixture.ledger/(name+'_worker.json')).exists()
    assert json.loads((fixture.ledger/'parallel_backends.json').read_text())['outcomes'][survivor]['complete']


def test_frontend_failure_never_admits_or_prepares_any_backend(tmp_path,monkeypatch):
    fixture=CycleFixture(tmp_path,monkeypatch,frontend_parallel=True,failure='frontend_A')
    with pytest.raises(RuntimeError,match='Frontend stage failed'):
        fixture.run(parallel_frontends=True,parallel_backends=True)
    assert fixture.gates==['parallel_admission']
    assert (fixture.output/'B/training/finished.json').exists()
    assert not any(name.startswith('backend_') for name,command in fixture.calls)


def test_optin_serial_fallback_honors_stop_between_backend_stages(tmp_path,monkeypatch):
    fixture=CycleFixture(tmp_path,monkeypatch,admit=False,stop_after='backend_cache_A')
    with pytest.raises(InterruptedError,match='STOP before backend_train_A'):
        fixture.run(parallel_backends=True)
    assert (fixture.output/'A/backend/cache_complete.json').exists()
    assert not any(name=='backend_train_A' or name.endswith('_B') and name.startswith('backend_')
        for name,command in fixture.calls)
    assert (fixture.ledger/'failure.json').exists()


@pytest.mark.parametrize('reason',['requested_stop','original_session_deadline'])
def test_wait_backend_stops_only_owned_cpu_process_group(tmp_path,reason):
    stop=tmp_path/'STOP'
    if reason=='requested_stop':stop.touch()
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
    try:
        deadline=time.time()+(5 if reason=='requested_stop' else -.1)
        assert cycle.wait_backend_owned(process,deadline,(stop,))==reason
        assert process.poll() is not None
    finally:
        if process.poll() is None:cycle.stop_owned(process)


@pytest.mark.parametrize('invalid',[1,'true',None])
def test_parallel_backends_requires_explicit_bool_before_any_registration(tmp_path,invalid):
    with pytest.raises(ValueError,match='explicit boolean'):
        cycle.run(tmp_path,tmp_path,tmp_path/'output',1,1,1,1,parallel_backends=invalid)
    assert not (tmp_path/'output_stages').exists()


def test_parallel_backends_requires_full_gpu_contract(tmp_path,monkeypatch):
    fixture=CycleFixture(tmp_path,monkeypatch)
    atomic_json(fixture.root/'session.json',dict(deadline=fixture.deadline))
    with pytest.raises(ValueError,match='Parallel backends require'):
        fixture.run(parallel_backends=True)
    assert not fixture.ledger.exists() and not fixture.calls


@pytest.mark.parametrize('enabled',[False,True])
def test_cli_optin_reaches_run_without_changing_existing_arguments(tmp_path,monkeypatch,enabled):
    captured={}
    monkeypatch.setattr(cycle,'run',lambda *args,**kwargs:captured.update(args=args,kwargs=kwargs))
    monkeypatch.setattr(sys,'argv',['cycle','--root',str(tmp_path),'--collection',str(tmp_path/'collection'),
        '--output',str(tmp_path/'next')]+(['--parallel-backends'] if enabled else []))
    cycle.main()
    assert captured['kwargs']['parallel_backends'] is enabled
    assert captured['kwargs']['parallel_frontends'] is False
    assert captured['args'][3:]==(10,5,4,28140000)
