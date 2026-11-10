"""CPU-only independent-DAG ownership, bounded probes and failure isolation."""
from pathlib import Path
import math

import pytest

from tools import tight_finish as owner
from tools import tight_diagnostics as publication


def node(kind,stage,deps=(),**extra):
    return dict(kind=kind,stage=stage,depends_on=list(deps),budget=dict(cache_s=20,gate_s=20,probe_s=20,
        training_s=100,offline_s=20,evaluation_s=20),**extra)


def plan():
    return dict(accepted=True,maximum_wall_seconds=1000,max_parallel_jobs=2,nodes={
        'G01':node('residual','command_residual'),'G11':node('residual','command_residual'),
        'Q00':node('frontend','configuration'),'Q01':node('residual','configuration_residual',['Q00']),
        'QG00':node('frontend','configuration'),'QG01':node('residual','configuration_residual',['QG00'])})


def test_dependency_failure_only_blocks_descendants():
    p=plan();states=dict.fromkeys(p['nodes'],'PENDING');states.update(G01='FAILED',Q00='COMPLETE',QG00='FAILED')
    ready,blocked=owner.matrix_ready_nodes(p,states)
    assert ready==['G11','Q01'] and blocked=={'QG01':'Dependency did not complete: QG00'}
    states['QG00']='PENDING';p['nodes']['QG00'].update(admitted=False,reason='measured geometry cost')
    ready,blocked=owner.matrix_ready_nodes(p,states)
    assert blocked['QG00']=='measured geometry cost' and 'Q01' in ready


@pytest.mark.parametrize('bad', ['unbounded','base_training','dependency','missing_phase'])
def test_plan_rejects_unregistered_or_unbounded_work(tmp_path,bad):
    p=plan()
    if bad=='unbounded':p['maximum_wall_seconds']=math.inf
    elif bad=='base_training':p['nodes']['G00']=node('frontend','command')
    elif bad=='dependency':p['nodes']['Q01']['depends_on']=['QG00']
    else:p['nodes']['G01']['budget'].pop('evaluation_s')
    owner.write(tmp_path/'execution_plan.json',p)
    with pytest.raises(ValueError):owner.matrix_execution_plan(tmp_path)


def test_measured_plan_has_no_arbitrary_day_cutoff(tmp_path):
    p=plan();p['maximum_wall_seconds']=90000
    owner.write(tmp_path/'execution_plan.json',p)
    assert owner.matrix_execution_plan(tmp_path)['maximum_wall_seconds']==90000


@pytest.mark.parametrize('case',['geometry_failed','backend_failed','overall_budget','absolute_budget'])
def test_owner_isolates_branch_failure_and_reaps_every_owned_process(tmp_path,monkeypatch,case):
    root=tmp_path/'matrix';root.mkdir();p=plan();now=[0.];children=[];reports=[];prepared=[]
    if case=='overall_budget':p['maximum_wall_seconds']=1
    if case=='absolute_budget':p['absolute_deadline_utc']=1
    physics=tmp_path/'retained_reset.json';owner.write(physics,dict(passed=True));p['physics_acceptance_path']=str(physics)
    owner.write(root/'execution_plan.json',p);owner.write(root/'session.json',{})
    owner.write(root/'configuration.json',dict(method_matrix={}))
    owner.write(root/'shared_resources.json',dict(utc_epoch=0.))
    monkeypatch.setattr(owner,'CODE',tmp_path)
    monkeypatch.setattr(owner.time,'time',lambda:now[0])
    monkeypatch.setattr(owner.time,'sleep',lambda seconds:now.__setitem__(0,now[0]+seconds))
    def git(args,**kwargs):return 'reviewed-source\n' if args[:3]==['git','rev-parse','HEAD'] else ''
    monkeypatch.setattr(owner.subprocess,'check_output',git)
    def prepare(r,group,spec):
        prepared.append(group);job=r/'queue'/group;job.mkdir(parents=True)
        failed=group==('QG00' if case=='geometry_failed' else 'G01')
        owner.write(job/'status.json',dict(state='FAILED' if failed else 'COMPLETE',owned_workers_exited=True))
        owner.write(job/'worker_processes.json',[])
        return job
    monkeypatch.setattr(owner,'matrix_prepare_job',prepare)
    class Child:
        def __init__(self,args,**kwargs):
            self.args=args;self.pid=1000+len(children);self.returncode=None;children.append(self)
            if 'matrix_group_run' in args and case not in ('overall_budget','absolute_budget'):
                group=Path(args[-1]).name
                self.returncode=1 if group==('QG00' if case=='geometry_failed' else 'G01') else 0
        def poll(self):return self.returncode
        def wait(self,timeout=None):self.returncode=0 if self.returncode is None else self.returncode;return self.returncode
        def terminate(self):self.returncode=-15
    monkeypatch.setattr(owner.subprocess,'Popen',Child)
    monkeypatch.setattr(owner,'kill_owned',lambda child:setattr(child,'returncode',-15) if child.poll() is None else None)
    monkeypatch.setattr(owner,'matrix_report',lambda r:reports.append(owner.read(r/'status.json')))
    monkeypatch.setattr(owner,'matrix_update_stages',lambda *args:None)
    monkeypatch.setattr(publication,'matrix_publish',lambda *args:None)
    monkeypatch.setattr(publication,'matrix_commit_publication',lambda *args:None)
    if case in ('overall_budget','absolute_budget'):
        with pytest.raises(TimeoutError):owner.matrix_run(root)
    else:owner.matrix_run(root)
    status=owner.read(root/'status.json')
    assert status['owned_workers_exited'] and all(child.poll() is not None for child in children)
    assert reports[-1]['owned_workers_exited']
    assert not any(any(name in c.args for name in ('formal_test','tools.reset_query_probe','vision_run')) for c in children)
    if case=='geometry_failed':
        assert status['groups']['Q01']=='COMPLETE' and status['groups']['QG01']=='BLOCKED'
        assert prepared.index('Q00')<prepared.index('Q01') and 'QG01' not in prepared
    elif case=='backend_failed':
        assert status['groups']['G11']=='COMPLETE' and status['groups']['Q01']=='COMPLETE'
    else:assert status['state']=='BUDGET_STOP'
    with pytest.raises(ValueError,match='relaunch'):owner.matrix_run(root)


def test_q_probe_retains_formal_schedule_and_only_registered_eight_updates(tmp_path,monkeypatch):
    from visuotactile import anchor_continue
    root=tmp_path/'configuration';root.mkdir();(root/'state_data').mkdir()
    owner.write(tmp_path/'configuration.json',dict(method_matrix=dict(probe_limit_updates=8)))
    v=dict(groups=['Q00','QG00'],modalities={'Q00':'G00'},warm_starts={'Q00':dict(path='old_best.pt',sha256='a'*64)},
        microbatch=4,initialization_seed=1,geometry_profiles={})
    owner.write(root/'session.json',dict(matrix_owner=str(tmp_path),vision_pair=v,cache_deadline=10**12))
    owner.write(root/'budget.json',dict(total_updates=8476,validation_interval=1000,early_stop_minimum_update=4238,
        comparison_updates=[2119,4238,6357,8476]))
    monkeypatch.setattr(owner,'vision_check',lambda r:owner.write(r/'vision_acceptance.json',dict(passed=True)))
    calls=[]
    def fresh(path,data,output,target,**kwargs):
        runtime=owner.read(path/'session.json');calls.append((target,runtime,kwargs))
        if not kwargs.get('smoke'):
            (path/'training.jsonl').write_text('\n'.join('{"seconds_per_update":2}' for _ in range(runtime['probe_limit_updates'])))
    monkeypatch.setattr(anchor_continue,'fresh',fresh)
    owner.matrix_gate(root,group='Q00')
    assert len(calls)==2 and calls[0][0]==1 and calls[1][0]==8476
    assert calls[1][1]['probe_limit_updates']==8 and calls[1][1]['early_stop_minimum_update']==4238
    assert calls[1][1]['warm_start']==v['warm_starts']['Q00']
    assert set(owner.read(root/'throughput_probe.json')['groups'])=={'Q00'}
    with pytest.raises(ValueError,match='duplicate'):owner.matrix_gate(root,group='Q00')


def test_residual_statistics_wrapper_does_not_resolve_back_to_legacy_metadata(tmp_path,monkeypatch):
    import torch
    source=tmp_path/'old';data=source/'state_data';data.mkdir(parents=True)
    splits={name:[name+str(i) for i in range(count)] for name,count in [('train',166),('validation',30),('test',22)]}
    owner.write(data/'data.json',dict(splits=splits));(data/'statistics.pt').write_bytes(b'unchanged historical affine')
    checkpoint=source/'G00/training/best.pt';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'old EMA')
    owner.write(source/'session.json',dict(vision_pair={'groups':['G00','G10']}))
    owner.write(source/'manifest.json',{});owner.write(source/'cache_index.json',{})
    owner.write(source/'evaluation_plan.json',[dict(identity=str(i)) for i in range(10)])
    wrapper=tmp_path/'command_statistics';wrapper.mkdir();(wrapper/'statistics.pt').symlink_to(data/'statistics.pt')
    owner.write(wrapper/'data.json',dict(splits=splits,supervision='u_expert',normalization_identity='u_expert_train166_affine_v1'))
    v=dict(version='target_residual_transformer60_v1',source_root=str(source),groups=['G01'],
        residual_statistics_path=str(wrapper/'statistics.pt'),bases={'G01':dict(group='G00',checkpoint='G00/training/best.pt',sha256=owner.sha(checkpoint))},
        min_gpu_free_mib=6144,min_ram_gib=14,min_disk_gib=60,evaluation_attempts_per_group=10,
        maximum_control_tick_epochs=200,minimum_control_tick_epochs=50,training_hours=None,evaluation_hours=1,cache_hours=1,offline_hours=1,maximum_hours=None)
    saved=dict(contract=dict(modality='G00',horizon=24,commit=12,native_hz=60,action='absolute_joint_position_target_rad'),
        configuration=dict(statistics_sha256=owner.sha(data/'statistics.pt')),updates=100)
    monkeypatch.setattr(torch,'load',lambda *args,**kwargs:saved)
    before=(data/'data.json').read_bytes();root=tmp_path/'new_residual'
    owner.residual_prepare(root,dict(residual_pair=v))
    assert owner.read(root/'configuration.json')['residual_pair']['residual_statistics_path']==str(wrapper/'statistics.pt')
    assert (data/'data.json').read_bytes()==before


def test_mixed_parallel_serial_measurement_pauses_training_only_and_releases_on_exit(tmp_path):
    p=plan();states={'Q00':'RUNNING','G01':'RUNNING'};scheduler={}
    owner.write(tmp_path/'configuration/throughput_probe.json',dict(groups={'Q00':dict(mean_seconds_per_update=2.)}))
    owner.write(tmp_path/'queue/G01/throughput_probe.json',dict(measurement=dict(training_wall_s=4.,supervised_ticks=128)))
    def progress(group,update):
        job=tmp_path/'queue'/group;owner.write(job/'status.json',dict(stage='training'))
        owner.write(job/group/'training/progress.json',dict(update=update,control_ticks=update*128))
    progress('Q00',32);progress('G01',32)
    owner.matrix_concurrency_tick(tmp_path,p,states,scheduler,0.)
    progress('Q00',64);progress('G01',64)
    owner.matrix_concurrency_tick(tmp_path,p,states,scheduler,192.)
    assert owner.read(tmp_path/'queue/G01/concurrency.json')['pause_groups']==['G01']
    progress('Q00',96)
    owner.matrix_concurrency_tick(tmp_path,p,states,scheduler,224.)
    entry=scheduler['pairs']['Q00:G01']
    assert entry['mode']=='serial' and entry['decision_ratio']==pytest.approx(.5)
    assert entry['references']['G01']['unit']=='supervised_control_tick'
    owner.matrix_concurrency_tick(tmp_path,p,states,scheduler,345.)
    assert owner.read(tmp_path/'queue/Q00/concurrency.json')['pause_groups']==['Q00']
    owner.write(tmp_path/'queue/Q00/status.json',dict(stage='offline'))
    owner.matrix_concurrency_tick(tmp_path,p,states,scheduler,346.)
    assert owner.read(tmp_path/'queue/Q00/concurrency.json')['pause_groups']==[]
    assert owner.read(tmp_path/'queue/G01/concurrency.json')['pause_groups']==[]


@pytest.mark.parametrize('evaluation_timeout',[False,True])
def test_budget_limited_q_development_holds_lock_through_timeout_cleanup(tmp_path,monkeypatch,evaluation_timeout):
    import fcntl
    from tools import runtime
    master=tmp_path/'matrix';job=master/'queue/Q00';dest=job/'Q00';(dest/'training').mkdir(parents=True)
    v=dict(groups=['Q00'],microbatch=4,initialization_seed=1,warm_starts={'Q00':dict(path='old.pt',sha256='f'*64)})
    owner.write(master/'session.json',dict(deadline=10**12,source_head='frozen'))
    owner.write(master/'source_freeze.json',{})
    owner.write(master/'configuration/vision_acceptance.json',dict(passed=True))
    owner.write(master/'configuration/smoke_Q00/check/result.json',dict(complete_restore=True))
    owner.write(master/'configuration/throughput_probe.json',dict(groups={'Q00':{}}))
    owner.write(job/'session.json',dict(matrix_owner=str(master),vision_pair=v))
    owner.write(job/'configuration.json',dict(vision_pair=v))
    owner.write(job/'node.json',dict(group='Q00',**node('frontend','configuration')))
    owner.write(job/'status.json',dict(state='PREPARED'))
    owner.write(job/'budget.json',dict(total_updates=8476,validation_interval=1000,early_stop_minimum_update=4238,comparison_updates=[4238,8476]))
    owner.write(job/'evaluation_plan.json',[dict(identity=str(i),spec=dict(cohort='development')) for i in range(10)])
    calls=[];events=[];now=[0.]
    monkeypatch.setattr(owner.time,'time',lambda:now[0])
    original_flock=fcntl.flock
    def flock(file,operation):
        if operation==fcntl.LOCK_UN:events.append('unlock')
        return original_flock(file,operation)
    monkeypatch.setattr(fcntl,'flock',flock)
    class Child:
        def __init__(self,args,**kwargs):
            self.pid=99999991;self.returncode=0;calls.append(args)
            if 'fresh-train' in args:
                owner.write(dest/'training/finished.json',dict(updates=123,stop_reason='fixed_wall_budget',target_reached=False))
                (dest/'training/best.pt').write_bytes(b'validation selected')
            elif 'final' in args:owner.write(dest/'offline/result.json',dict(complete=True))
            elif 'vision_evaluate' in args:
                owner.write(job/'evaluation_complete.json',dict(complete=True))
                if evaluation_timeout:self.returncode=None;now[0]=30.
        def poll(self):return self.returncode
    monkeypatch.setattr(owner.subprocess,'Popen',Child)
    def stop(child):
        if child.poll() is None:events.append('terminate_and_reap');child.returncode=-15
    monkeypatch.setattr(owner,'kill_owned',stop)
    def gpu():events.append('verify_gpu_absent');return set()
    monkeypatch.setattr(runtime,'gpu_pids',gpu)
    if evaluation_timeout:
        with pytest.raises(TimeoutError):owner.matrix_group_run(job)
        assert events.index('terminate_and_reap')<events.index('verify_gpu_absent')<events.index('unlock')
        assert owner.read(job/'status.json')['state']=='BUDGET_STOP'
    else:
        owner.matrix_group_run(job)
        assert owner.read(job/'status.json')['state']=='COMPLETE'
        assert owner.read(job/'status.json')['budget_limited'] is True
    assert owner.read(dest/'training/adaptation_scope.json')['convergence_proven'] is False
    assert owner.read(job/'evaluation_process_reaped.json')['gpu_absent'] is True
    assert len(calls)==3 and any('vision_evaluate' in args for args in calls)


def test_dead_group_owner_does_not_leave_its_separate_session_worker(tmp_path,monkeypatch):
    from tools import runtime
    import signal
    record=dict(pid=99999992,start_ticks=123,process_group=99999992)
    owner.write(tmp_path/'worker_processes.json',[record]);signals=[]
    monkeypatch.setattr(owner,'proc',lambda pid:dict(record) if not signals else None)
    monkeypatch.setattr(owner.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    monkeypatch.setattr(runtime,'gpu_pids',lambda:set())
    class ExitedOwner:
        pid=99999993
        def poll(self):return 1
    receipt=owner.matrix_stop_owner(tmp_path,ExitedOwner())
    assert signals==[(record['pid'],signal.SIGTERM)]
    assert receipt['verified'] and receipt['owner_exited']
