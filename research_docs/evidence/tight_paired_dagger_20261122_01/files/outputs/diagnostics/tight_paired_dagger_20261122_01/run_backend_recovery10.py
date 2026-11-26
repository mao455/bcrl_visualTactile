"""Explicit missing-cache continuation; preserve 09, DP and physical work.

Preparation only writes new audit artifacts. Root reviews and activates once.
Native scientific workers run through an exact scoped pre-entry resource wait.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]))
import run_backend_recovery09 as prior
import run_main_recovery05 as original
import parallel_rear_proposal as pipeline
import parallel_rear_shared06 as shared
import shared_capacity06 as capacity

ROOT,PROJECT,CONFIG=prior.ROOT,prior.PROJECT,prior.CONFIG
MODULE=original.MODULE+'run_backend_recovery10'
WORKER_MODULE=original.MODULE+'parallel_rear_recovery10'
WAIT_MODULE=original.MODULE+'scoped_resource_wait10'
OUT=HERE/'driver_backend_recovery10'
OWNER=ROOT/'owners/backend_recovery10'
JOB=HERE/'proposed_backend_recovery10.json'
REGISTRATION=HERE/'BACKEND_RECOVERY10_REGISTRATION.json'
RUNTIME=HERE/'parallel_rear_recovery10_runtime'
LOGS=ROOT/'cycle_logs_recovery10'
WATCH=HERE/'watch_shared_capacity10'
SNAPSHOT=HERE/'backend_recovery09_preserved_for10'
preserved_dp=prior.preserved_dp


def backend_job():
    return dict(version='tight_project_job_v1',operation='train',resource_root=str(OWNER),
        stages=[dict(entry='paired_backend_recovery10',args=['--root',str(ROOT/'learning'),
            '--registration',str(REGISTRATION)])])


def stages():
    yield 'train_backend_recovery10',['-m',WORKER_MODULE,'project','train','--job',str(JOB)],True
    for name,args,gpu in list(prior.stages())[1:]:
        if name=='verify_final_actual_training':args=[*args[:-1],str(HERE/'presealed_guard_main_recovery10.json')]
        yield name,args,gpu


def native_commands(root=ROOT/'learning'):
    out={}
    for role in 'AB':
        commands=pipeline.stage_commands(root,role)
        for key,index in (('prepare',0),('cache',1),('train',2),('offline',3)):
            if key=='prepare':continue
            out[role+'_'+key]=commands[index][1:]
    return out


def wait_configuration():
    from scoped_resource_wait10 import VERSION
    commands={}
    for key,argv in native_commands().items():
        role,kind=key.split('_');rear=ROOT/'learning'/role/'backend'
        worker=Path(argv[argv.index('--root')+1])
        commands[key]=dict(argv=argv,worker_root=str(worker),lease_roots=[str(rear)],
            telemetry_path=str(OWNER/'shared_resources.json'))
    return dict(version=VERSION,
        wrapper_sha256=original.digest(HERE/'scoped_resource_wait10.py'),
        state_resources_sha256=original.digest(PROJECT/'visuotactile/state_resources.py'),
        original_deadline=original.read(ROOT/'session.json')['deadline'],commands=commands,
        stop_paths=[str(p) for p in (ROOT/'STOP',ROOT/'STOP_CYCLE',ROOT/'learning/STOP',OWNER/'STOP')])


def wrapped_command(key):
    return [sys.executable,'-m',WAIT_MODULE,'--registration',str(REGISTRATION),
        '--command-key',key,'--receipt-dir',str(RUNTIME/('wait_'+key)),
        '--',*native_commands()[key]]


def snapshot_prior():
    original.require(not SNAPSHOT.exists(),'Preserve 10 snapshot identity')
    SNAPSHOT.mkdir()
    shutil.copytree(prior.OWNER,SNAPSHOT/'owner09',symlinks=True)
    shutil.copytree(prior.OUT,SNAPSHOT/'driver09',symlinks=True)
    for name in ('run_backend_recovery09.py','parallel_rear_recovery09.py',
                 'shared_capacity09.py','scoped_resource_wait09.py',
                 'BACKEND_RECOVERY09_REGISTRATION.json','proposed_backend_recovery09.json'):
        shutil.copy2(HERE/name,SNAPSHOT/name)
    discarded={}
    for role in 'AB':
        rear=ROOT/'learning'/role/'backend';dest=SNAPSHOT/role;dest.mkdir()
        for name in ('residual_cache_error.json','cache_progress.json','prepared.json'):
            shutil.copy2(rear/name,dest/name)
        shutil.copy2(rear/role/'nominal/index.json',dest/'nominal_index.json')
        progress=original.read(rear/'cache_progress.json')
        discarded[role]=dict(identity=progress['identity'],last_reported_tick=progress.get('tick'),
            last_reported_conditions=progress.get('conditions'),last_reported_plans=progress.get('plans'),
            last_reported_phase=progress.get('phase'),exact_failure_tick_unlogged=True)
    pipeline.write_new(SNAPSHOT/'derived_discarded_work.json',dict(
        roles=discarded,discarded='Unpersisted learned history only; each missing episode rebuilds from tick0',
        physical_opportunities_repeated=0,optimizer_updates_repeated=0,
        finished_cache_episodes_recomputed=0))


def cache_boundary(root=ROOT,*,role,verify_payload=False,expected=None):
    original.require(role in 'AB' and len(role)==1,'Exact cache role required')
    dest=Path(root)/'learning'/role/'backend'/role/'nominal'
    index=original.read(dest/'index.json')
    source=index if expected is None else expected
    header={k:v for k,v in source.items() if k!='episodes'}
    original.require({k:v for k,v in index.items() if k!='episodes'}==header,
        'Preserved cache contract/header changed')
    rows={}
    for identity,row in source['episodes'].items():
        path=dest/(identity+'.pt')
        original.require(path.is_file() and not path.is_symlink(),'Preserved cache payload absent/aliased')
        original.require(index['episodes'].get(identity)==row,'Preserved indexed episode changed')
        if verify_payload:original.require(original.digest(path)==row['sha256'],'Preserved cache SHA changed')
        rows[identity]=dict(sha256=row['sha256'],size_bytes=path.stat().st_size)
    return dict(index_header=header,episodes=rows,count=len(rows),
        total_bytes=sum(r['size_bytes'] for r in rows.values()))


def prepared_backends(root=ROOT,*,require_unstarted=False):
    """Both existing prepares are immutable and neither rear trainer started."""
    result={}
    for role in 'AB':
        rear=Path(root)/'learning'/role/'backend';prepared=original.read(rear/'prepared.json')
        deployment=original.read(Path(root)/'learning'/role/'training/deployment.json')
        original.require(prepared['role']==role and prepared['epochs']==60,
            'Completed rear prepare identity/budget differs')
        original.require(prepared['source_frontend_sha256']==deployment['checkpoint_sha256'],
            'Completed rear prepare uses a different DP')
        if require_unstarted:
            original.require(not any((rear/role/'training').iterdir()),
                'Preserve any started rear trainer')
        result[role]=dict(prepared_sha256=original.digest(rear/'prepared.json'),
            source_frontend_sha256=prepared['source_frontend_sha256'],train=prepared['train'],
            validation=prepared['validation'],epochs=prepared['epochs'])
    return result


def static_paths():
    return [prior.REGISTRATION,prior.JOB,HERE/'OLD_WATCHER09_CLOSED_FOR_RECOVERY10.json',
        HERE/'BACKEND_RECOVERY09_ACTUAL_TERMINAL_BOUNDARY10.json',
        prior.RUNTIME/'caches/pipeline_events.json',
        prior.LOGS/'train_backend_recovery09_exit.json',
        *sorted(p for p in SNAPSHOT.rglob('*') if p.is_file())]


def scientific_configuration_paths():
    paths=[ROOT/'learning/session.json',ROOT/'learning/frozen_visual_io.json',
        *[ROOT/'learning'/('paired_group_deck_'+r+'.json') for r in 'AB']]
    for role in 'AB':
        rear=ROOT/'learning'/role/'backend'
        paths.extend((rear/'session.json',rear/'configuration.json',rear/'base_identities.json',
            rear/'prepared.json',rear/role/'session.json',rear/role/'configuration.json',
            rear/'state_data/data.json'))
    return paths


def expected_registration():
    from tools.tight_paired_dagger import sources
    old=original.read(prior.REGISTRATION)
    original.require(sources()==old['source_sha256'],'Production source changed since09')
    original.require(time.time()<old['original_deadline']==original.read(ROOT/'session.json')['deadline'],'Original deadline changed/expired')
    original.require(original.read(prior.OWNER/'project_exit.json')['children_reaped'] is True,'09 children not reaped')
    original.require(original.read(prior.OUT/'driver_exit.json')['returned_after_owned_helpers'] is True,'09 driver not closed')
    original.require(original.read(prior.OUT/'status.json')['stage']=='stopped_with_evidence','09 is not terminal failed')
    for path in (OWNER,OUT,RUNTIME,LOGS,ROOT/'evaluation/development',ROOT/'evaluation/test'):
        original.require(not path.exists(),'Preserve attempted continuation/evaluation: '+str(path))
    boundary={}
    for role,count in (('A',174),('B',16)):
        expected=original.read(SNAPSHOT/role/'nominal_index.json')
        boundary[role]=cache_boundary(role=role,verify_payload=True,expected=expected)
        original.require(boundary[role]['count']==count,'Exact completed cache boundary differs')
        dest=ROOT/'learning'/role/'backend'/role/'nominal'
        original.require(original.digest(dest/'index.json')==original.digest(SNAPSHOT/role/'nominal_index.json'),
            'Partial index changed before10')
        original.require({p.name for p in dest.iterdir()}=={'index.json',*(k+'.pt' for k in expected['episodes'])},
            'Unexpected orphan/partial/undocumented cache file')
    prepared=prepared_backends(require_unstarted=True)
    names=('run_backend_recovery10.py','parallel_rear_recovery10.py','shared_capacity10.py','scoped_resource_wait10.py',
        'run_backend_recovery09.py','run_backend_recovery08.py','run_backend_recovery07.py','parallel_rear_proposal.py','parallel_rear_shared06.py','run_main_recovery05.py',
        'shared_capacity06.py','resource_watch_v4.py','presealed_guard.py','audit_collection_recovery05.py',
        'driver_support.py','driver_support_v4.py')
    return dict(version='paired_backend_missing_cache_recovery10_v1',activated=False,created_utc=time.time(),
        root=str(ROOT),owner=str(OWNER),job=str(JOB),logs=str(LOGS),watcher=str(WATCH),
        watcher_module=original.MODULE+'shared_capacity10',original_deadline=old['original_deadline'],
        source_sha256=old['source_sha256'],old_production_sha256=old['old_production_sha256'],
        prior_registration_sha256=original.digest(prior.REGISTRATION),
        orchestration_sha256={str(HERE/n):original.digest(HERE/n) for n in names},
        preserved_receipts_sha256={str(p):original.digest(p) for p in static_paths()},
        scientific_configuration_sha256={str(p):original.digest(p) for p in scientific_configuration_paths()},
        preserved_dp=preserved_dp(verify_payload=True),preserved_cache=boundary,
        preserved_prepared_backends=prepared,
        old_telemetry_target=os.readlink(ROOT/'learning/shared_resources.json'),
        job_sha256=original.digest(JOB),scoped_resource_wait10=wait_configuration(),
        stages=[dict(name=n,command=a,gpu=g) for n,a,g in stages()],
        backend_schedule=deepcopy(old['backend_schedule']),
        scientific_invariants=dict(dp_reruns=0,dp_new_updates=0,collection_retries=0,new_opportunities=0,
            deploy_epoch=60,optimizer_epochs=60,rear_updates_per_role=38400,caches_own_dp=True,
            retained_complete_cache_episodes={'A':174,'B':16},prepare_reruns=0,
            derived_unpersisted_episode_restart_from_tick0=True,
            external_presence_is_veto=False,encoder_frozen=True,sealed_test_families=64),
        scheduling='Native distinct own-DP caches run parallel only if actual combined RAM/VRAM admission permits; otherwise serial. Same rule for rear trainers. Offline serial. No scientific worker/body retries.',
        evaluation_resource_scope='Scoped wait applies only to exact backend cache/train/offline workers. Development/test keep native physical owners and failure propagation; no body retry, outcome-dependent rerun or physical opportunity replacement is added.',
        prior_failure='A09 rejected the native null pause sentinel before a scientific block. B09 was peer-terminated; its residual_cache_error.json is retained08 metadata-race evidence, not a09 failure trace.')


def prepare():
    original.require(not JOB.exists() and not REGISTRATION.exists(),'Preserve10 preparation identity')
    snapshot_prior();pipeline.write_new(JOB,backend_job())
    pipeline.write_new(REGISTRATION,expected_registration())


def validate_registration(*,payload=False,require_absent_owner=False):
    from tools.tight_paired_dagger import sources
    reg=original.read(REGISTRATION)
    original.require(reg['version']=='paired_backend_missing_cache_recovery10_v1' and reg['root']==str(ROOT),'Registry identity differs')
    original.require(reg['original_deadline']==original.read(ROOT/'session.json')['deadline'] and time.time()<reg['original_deadline'],'Original deadline changed/expired')
    original.require(reg['source_sha256']==sources(),'Production source changed')
    for mapping in ('orchestration_sha256','preserved_receipts_sha256','scientific_configuration_sha256'):
        for path,value in reg[mapping].items():original.require(original.digest(path)==value,'Pinned file changed: '+path)
    original.require(original.digest(JOB)==reg['job_sha256'] and original.read(JOB)==backend_job(),'10 job changed')
    original.require(wait_configuration()==reg['scoped_resource_wait10'],'Scoped resource command contract changed')
    original.require(preserved_dp(verify_payload=payload)==reg['preserved_dp'],'Completed DP identities changed')
    for role in 'AB':
        expected=original.read(SNAPSHOT/role/'nominal_index.json')
        original.require(cache_boundary(role=role,verify_payload=payload,expected=expected)==reg['preserved_cache'][role],
            'Preserved cache changed for '+role)
    original.require(prepared_backends(require_unstarted=require_absent_owner)==reg['preserved_prepared_backends'],
        'Completed backend prepare changed')
    original.require(reg['stages']==[dict(name=n,command=a,gpu=g) for n,a,g in stages()],'10 stage list changed')
    if require_absent_owner:
        for p in (OWNER,OUT,RUNTIME,LOGS,ROOT/'evaluation/development',ROOT/'evaluation/test'):
            original.require(not p.exists(),'10/evaluation already attempted')
        for role in 'AB':
            original.require(original.digest(ROOT/'learning'/role/'backend'/role/'nominal/index.json')
                ==original.digest(SNAPSHOT/role/'nominal_index.json'),'Initial index changed')
        original.require(os.readlink(ROOT/'learning/shared_resources.json')==reg['old_telemetry_target'],'Prior telemetry target changed')
    return reg


def atomic_telemetry_transition(path,target,receipt,write,*,expected_old=None):
    path,target=Path(path),Path(target)
    original.require(path.is_symlink(),'Only an existing telemetry link may be redirected')
    old=os.readlink(path)
    original.require(expected_old is None or old==expected_old,'Prior telemetry target changed')
    temporary=path.with_name(path.name+'.recovery10_new')
    original.require(not temporary.exists() and not temporary.is_symlink(),'Preserve transition temporary identity')
    write(receipt,dict(utc=time.time(),old=old,new=str(target),atomic=True))
    temporary.symlink_to(target)
    os.replace(temporary,path)
    return old


def rear_pair(root,registration):
    original.require(Path(root).resolve()==ROOT/'learning' and Path(registration).resolve()==REGISTRATION,'Exact10 root/registry required')
    reg=validate_registration(payload=True);launch=deepcopy(reg['backend_schedule'])
    launch.update(owner=str(OWNER),root=str(ROOT/'learning'),deadline=reg['original_deadline'])
    affinity=set(launch['cpu_affinity']);original.require(affinity<=os.sched_getaffinity(0),'CPU8-15 unavailable');os.sched_setaffinity(0,affinity)
    from resource_watch_v4 import external_test_processes
    deadline=reg['original_deadline'];stops=[ROOT/'STOP',ROOT/'STOP_CYCLE',ROOT/'learning/STOP',OWNER/'STOP']
    RUNTIME.mkdir(exist_ok=False)
    allowance=dict(launch,vram_per_worker_mib=launch['cache_and_offline_vram_per_worker_mib'])
    def ready():
        try:
            row=original.read(OWNER/'shared_resources.json');decision=shared.readiness(row,external_test_processes(),launch,time.time())
            admission=shared.admission(row,allowance,time.time(),1)
            decision.update(allowed=decision['allowed'] and admission['allowed'],launch_capacity=admission,raw_monitor_row=row)
        except (OSError,KeyError,TypeError,ValueError) as exc:decision=dict(allowed=False,reasons=[repr(exc)])
        with (RUNTIME/'readiness.jsonl').open('a') as f:f.write(original.json.dumps(dict(utc=time.time(),decision=decision))+'\n')
        return decision['allowed']
    def await_ready():
        while not ready():
            if time.time()>=deadline or any(p.exists() for p in stops):raise TimeoutError('Finite10 capacity wait')
            time.sleep(.5)
    def decide(name,registration):
        await_ready();row=original.read(OWNER/'shared_resources.json')
        pair=shared.admission(row,registration,time.time(),2);single=shared.admission(row,registration,time.time(),1)
        original.require(pair['allowed'] or single['allowed'],'No capacity for first worker')
        result=dict(parallel=pair['allowed'],pair=pair,single=single,raw_monitor_row=row,registration=registration,utc=time.time())
        pipeline.write_new(RUNTIME/(name+'_admission.json'),result);return result
    with pipeline.termination_signals():
        cache=decide('caches',allowance);prep=RUNTIME/'caches';prep.mkdir()
        pipeline.run_roles({'A':[wrapped_command('A_cache')],'B':[wrapped_command('B_cache')]},cache['parallel'],deadline,stops,ready,prep)
        validate_registration(payload=True)
        sizes={r:sum(p.stat().st_size for p in (ROOT/'learning'/r/'backend'/r/'nominal').glob('*.pt')) for r in 'AB'}
        actual=dict(launch,ram_per_worker_bytes=max(launch['ram_per_worker_bytes'],2*2**30+3*max(sizes.values())))
        allowance=actual;train=decide('rear_trainers',actual);training=RUNTIME/'rear_training';training.mkdir()
        events=pipeline.run_roles({r:[wrapped_command(r+'_train')] for r in 'AB'},train['parallel'],deadline,stops,ready,training)
        allowance=dict(actual,vram_per_worker_mib=launch['cache_and_offline_vram_per_worker_mib'])
        offline=RUNTIME/'serial_offline';offline.mkdir()
        pipeline.run_roles({r:[wrapped_command(r+'_offline')] for r in 'AB'},False,deadline,stops,ready,offline)
        pipeline.write_new(RUNTIME/'complete.json',dict(utc=time.time(),events=len(events),caches_parallel=cache['parallel'],rears_parallel=train['parallel'],offline_serial=True,completed_cache_episodes_recomputed=0))


class Driver(original.Driver):
    def gpu_gate(self,name):
        import psutil
        from shared_capacity10 import startup_row
        gpu,ram=(2048,6*2**30) if name=='train_backend_recovery10' else ((11264,16*2**30) if name.startswith('evaluate_') else (1024,2*2**30))
        while True:
            self.finite()
            try:
                row=original.read(WATCH/'resource_watch_v4_status.json');decision=startup_row(row,psutil.Process(row['pid']),time.time(),allow_gap=True)
                live=capacity.capacity_decision(capacity.live_capacity(),gpu_resident_mib=gpu,ram_resident_bytes=ram)
                original.require(live['ready'],'Actual capacity below allowance plus reserve')
                fresh=original.read(WATCH/'resource_watch_v4_status.json');startup_row(fresh,psutil.Process(fresh['pid']),time.time(),allow_gap=True)
                self.write(OUT/(name+'_shared_capacity06_admission.json'),dict(utc=time.time(),watcher=decision,watcher_status=fresh,capacity=live));return
            except (OSError,KeyError,TypeError,ValueError,psutil.Error,original.subprocess.SubprocessError) as exc:
                self.note('waiting_actual_shared_capacity_before_'+name,reason=repr(exc));time.sleep(5)
    def run(self,name,args,gpu):
        reg=validate_registration();original.require(self.source_identity==reg['source_sha256'],'10 source binding changed')
        if name=='train_backend_recovery10':
            atomic_telemetry_transition(ROOT/'learning/shared_resources.json',OWNER/'shared_resources.json',OUT/'training_telemetry_transition.json',self.write,expected_old=reg['old_telemetry_target'])
            self.write(OUT/'backend_source_identity.json',dict(utc=time.time(),sources=self.source_identity,completed_dp_sources_preserved=str(ROOT/'learning/training_source_identity.json')))
        return super().run(name,args,gpu)
    def main(self):
        reg=validate_registration(payload=True,require_absent_owner=True)
        OUT.mkdir(exist_ok=False);self.logs=LOGS;self.logs.mkdir(exist_ok=False);self.source_identity=reg['source_sha256']
        OWNER.mkdir(exist_ok=False);session=deepcopy(original.read(ROOT/'session.json'));session['resource_root']=str(OWNER)
        self.write(OWNER/'session.json',session);self.write(OWNER/'cuda_allocator_policy.json',dict(version='reclaim_unused_on_low_headroom_v1'))
        self.write(OUT/'registration.json',dict(reg,activated=True,activated_utc=time.time(),registry_sha256=original.digest(REGISTRATION)))
        for name,args,gpu in stages():self.run(name,args,gpu)
        self.note('main_stages_complete',report_required=True,dp_reruns=0,collection_retries=0)
        self.write(OUT/'closure.json',dict(utc=time.time(),stage='main_stages_complete',status='backend_and_remaining_main_stages_complete',report_required=True))


def project():
    from tools import tight_project
    tight_project.ENTRIES['paired_backend_recovery10']=('train',WORKER_MODULE,['rear-pair'])
    sys.argv.pop(1);tight_project.main()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='project':return project()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','rear-pair'])
    p.add_argument('--root',type=Path);p.add_argument('--registration',type=Path,default=REGISTRATION);a=p.parse_args()
    if a.mode=='prepare':return prepare()
    if a.mode=='rear-pair':return rear_pair(a.root,a.registration)
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def interrupted(signum,frame):raise KeyboardInterrupt(signum)
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    original.OUT=OUT;driver=Driver()
    try:driver.main()
    except BaseException as exc:
        if OUT.exists():
            driver.write(OUT/'failure.json',dict(utc=time.time(),error=repr(exc),traceback=traceback.format_exc()));driver.note('stopped_with_evidence',error=repr(exc))
        raise
    finally:
        if OUT.exists():driver.write(OUT/'driver_exit.json',dict(utc=time.time(),pid=os.getpid(),returned_after_owned_helpers=True))


if __name__=='__main__':main()
