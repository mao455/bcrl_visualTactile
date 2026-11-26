"""INACTIVE admission14: reuse the never-started native owner13 exactly once.

Only run-local entry mapping and verified cooperative schedule metadata differ.
Scientific worker/lease/startup watchdog remain delegated to frozen13 source.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse,base64,copy,fcntl,hashlib,json,math,os,signal,sys,threading,time,traceback

from outputs.diagnostics.tight_paired_dagger_20261122_01 import test_fragment13 as f

HERE,ROOT,TEST,OWNER,JOB,DEADLINE=f.HERE,f.ROOT,f.TEST,f.OWNER,f.JOB,f.DEADLINE
MODULE='outputs.diagnostics.tight_paired_dagger_20261122_01.run_test_admission14'
REGISTRY=HERE/'TEST_ADMISSION14_REGISTRATION.json'
REVIEW=HERE/'ROOT_TEST_ADMISSION14_REVIEW.json'
BOUNDARY=HERE/'ROOT_TEST_ADMISSION14_BOUNDARY.json'
OUT=HERE/'driver_test_admission14'
LOGS=ROOT/'cycle_logs_admission14'
WATCH_AUDIT=HERE/'watch_shared_capacity14'
SCHEDULE_LOCK=ROOT/'admission14_schedule.lock'
ADMISSION=dict(gpu_resident_mib=11264,ram_resident_bytes=12*2**30,
               gpu_reserve_mib=4096,ram_reserve_bytes=10*2**30,
               cooperative_early_gpu_margin_mib=1024,cooperative_early_ram_margin_bytes=2*2**30)
read,digest,require,write_new=f.read,f.digest,f.require,f.write_new


@contextmanager
def schedule_lock():
    with SCHEDULE_LOCK.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def pristine_names(wave,*,allow_cooperative=False):
    wave=Path(wave)
    require(not wave.is_symlink() and wave.resolve()==wave,'Literal canonical original wave required')
    entries={p.name:p for p in wave.iterdir()}
    allowed={'plan.json','session.json'}|({'concurrency.json'} if allow_cooperative else set())
    require(set(entries)<=allowed and {'plan.json','session.json'}<=set(entries),
            'Unknown/runtime metadata in unstarted wave')
    require(all(p.is_file() and not p.is_symlink() for p in entries.values()),'Symlink/directory/special wave entry')
    return entries


def assert_owner_unattempted():
    files=('project_job_started.json','project_status.json','project_complete.json','project_failure.json',
           'project_owner.lock','project_monitor.log','monitor.json','shared_resources.json','STOP','STOP_MONITOR')
    require(not any((OWNER/name).exists() or (OWNER/name).is_symlink() for name in files),
            'Native owner13 already attempted or stopped')
    require(not list(OWNER.glob('project_stage_*.log')),'Native owner13 stage log already exists')
    require((OWNER/'dispatches').is_dir() and not (OWNER/'dispatches').is_symlink()
            and not list((OWNER/'dispatches').iterdir()),'Native owner13 dispatch already attempted')


def zero_native_attempts(value13):
    """Only owner13 lightweight lifecycle paths and sealed eligible metadata."""
    assert_owner_unattempted()
    for index in value13['eligible_indices']:
        wave=TEST/f'wave{index:02d}'
        pristine_names(wave)
    return dict(native_owner13_started=False,native_stage_logs=0,dispatch_fences=0,
                eligible_indices=value13['eligible_indices'],native_attempts=0,lifecycle_only=True,
                outcome_values_interpreted=False)


def boundary_gate(value13):
    import psutil
    row=read(BOUNDARY)
    require(row.get('status')=='actual_fragment13_failed_before_native_owner'
            and row.get('deadline')==DEADLINE and row.get('original13_driver_reaped') is True,
            'Root actual13 pre-native failure certificate required')
    require(row.get('no_external_signals') is True,'External processes must remain untouched')
    required=[HERE/'driver_test_fragment13/failure.json',HERE/'driver_test_fragment13/driver_exit.json',
              ROOT/'cycle_logs_fragment13/evaluate_fragment13.log',ROOT/'cycle_logs_fragment13/evaluate_fragment13_exit.json',
              HERE/'WATCHER13_ACTUAL_CLOSED.json']
    require(set(row['artifact_sha256'])>={str(p) for p in required},'Incomplete actual13 boundary artifacts')
    for path,expected in row['artifact_sha256'].items():
        require(digest(path)==expected,'Actual13 boundary changed: '+path)
    exited=read(HERE/'driver_test_fragment13/driver_exit.json')
    require(exited['returned_after_owned_helpers'] is True
            and not (HERE/'driver_test_fragment13/closure.json').exists(), 'Preserve truthful failed13 driver')
    require(read(ROOT/'cycle_logs_fragment13/evaluate_fragment13_exit.json')['returncode']!=0,
            'Only actual failed pre-native helper can be superseded')
    watch=read(HERE/'WATCHER13_ACTUAL_CLOSED.json')
    require({exited['pid'],watch['pid']}<={p['pid'] for p in row['closed_identities']},
            'Actual13 driver/watcher PID identities missing')
    for identity in row['closed_identities']:
        require(type(identity['pid']) is int and identity['pid']>0
                and type(identity['created']) in (int,float) and math.isfinite(identity['created'])
                and identity['created']>0,'Finite positive actual PID/create-time required')
        try:
            proc=psutil.Process(identity['pid'])
            require(abs(proc.create_time()-identity['created'])>.001,'Original13 identity remains live')
        except psutil.NoSuchProcess:
            pass
    require(watch['source_pid_reaped'] is True and watch['conflicts']==[],'Actual watcher13 not restored')
    service=watch['service']
    require(str(service['MainPID'])=='0' and service['ActiveState']=='inactive' and service['Result']=='success',
            'Watcher13 service not inactive/success')
    return row,zero_native_attempts(value13)


def prepare():
    require(not REGISTRY.exists() and not OUT.exists() and not LOGS.exists(),'Preserve any attempted14 identity')
    value13=f.validate_binding()
    boundary,census=boundary_gate(value13)
    names=['run_test_admission14.py','shared_capacity14.py','test_fragment13.py','shared_capacity13.py',
           'run_main_recovery05.py','shared_capacity06.py','resource_watch_v4.py','driver_support.py','driver_support_v4.py']
    row=dict(version='test_admission14_pre_native_only_v1',utc=time.time(),deadline=DEADLINE,
             owner=str(OWNER),job=str(JOB),original13_registry_sha256=digest(f.REGISTRY),
             original13_job_sha256=digest(JOB),boundary_sha256=digest(BOUNDARY),
             original_test_registration_sha256=value13['original_registration_sha256'],
             original13_driver_artifacts=boundary['artifact_sha256'],native_zero_attempt_census=census,
             eligible_indices=value13['eligible_indices'],tail_stages=value13['tail_stages'],
             source_sha256=value13['source_sha256'],frozen_pairs=value13['frozen_pairs'],
             original_cohorts_sha256=value13['original_cohorts_sha256'],admission=ADMISSION,
             orchestration_sha256={str(HERE/name):digest(HERE/name) for name in names},
             original13_driver_remains_failed=True,native_owner13_reused_once=True,
             scientific_changes=0,training_updates=0,no_outcome_retry=True,
             schedule_metadata_policy='Only live watcher14-owned exact paused schedule with exact original backup binding; no other runtime metadata')
    write_new(REGISTRY,row)
    return row


def validate(*,review=True):
    row=read(REGISTRY)
    require(row['version']=='test_admission14_pre_native_only_v1' and row['deadline']==DEADLINE
            and time.time()<DEADLINE and row['admission']==ADMISSION,'Original finite14/admission identity differs')
    require(row['original13_registry_sha256']==digest(f.REGISTRY) and row['original13_job_sha256']==digest(JOB)
            and row['boundary_sha256']==digest(BOUNDARY),'Original13/boundary identities changed')
    value13=f.validate_binding()
    require(row['eligible_indices']==value13['eligible_indices'] and row['tail_stages']==value13['tail_stages']
            and row['source_sha256']==value13['source_sha256'] and row['frozen_pairs']==value13['frozen_pairs'],
            'Original scientific population or native13 binding changed')
    for path,expected in row['orchestration_sha256'].items():
        require(digest(path)==expected,'Reviewed14 source changed: '+path)
    for path,expected in row['original13_driver_artifacts'].items():
        require(digest(path)==expected,'Original13 failed evidence changed: '+path)
    if review:
        accepted=read(REVIEW)
        require(accepted['status']=='source_and_actual_pre_native14_review_passed_inactive'
                and accepted['registration_sha256']==digest(REGISTRY),'Root actual14 review required')
    return row,value13


def paused_schedule_bytes(original,pid):
    from outputs.diagnostics.tight_paired_dagger_20261122_01.resource_watch_v4 import PAUSES
    value={} if original is None else json.loads(original)
    groups=list(value.get('pause_groups',[]))
    for group in PAUSES:
        if group not in groups:groups.append(group)
    value=dict(value,pause_groups=groups,paired_resource_watch_v4_pid=pid)
    return json.dumps(value,indent=2,allow_nan=False).encode()


def verify_live_cooperative_schedule(path):
    """Called while watcher14 schedule lock excludes pause/restore temp files."""
    from outputs.diagnostics.tight_paired_dagger_20261122_01 import shared_capacity14 as watcher
    import psutil
    status=read(WATCH_AUDIT/'resource_watch_v4_status.json')
    watcher.live_identity(status,psutil.Process(status['pid']),time.time())
    require(status['paused'] is True,'Present cooperative pause requires actual watcher pause state')
    backups=list(WATCH_AUDIT.glob(f'resource_watch_v4_backups_*_{status["pid"]}'))
    require(len(backups)==1,'Exact current watcher14 backup directory required')
    backup=backups[0]/(hashlib.sha256(str(path).encode()).hexdigest()+'.json')
    require(backup.is_file() and not backup.is_symlink() and backups[0].resolve()==backups[0],
            'Unsafe cooperative schedule backup')
    row=read(backup)
    require(row['path']==str(path) and row['existed'] is False and row['base64'] is None and row['sha256'] is None,
            'Unstarted suffix schedule must derive from original absence')
    require(path.read_bytes()==paused_schedule_bytes(None,status['pid']),'Unknown or conflicting concurrency metadata')
    return dict(path=str(path),sha256=digest(path),watcher_pid=status['pid'],backup=str(backup),backup_sha256=digest(backup))


def guarded_wave(wave,row):
    entries=pristine_names(wave,allow_cooperative=True)
    freeze=read(TEST/'registration.json')
    require(digest(wave/'plan.json')==freeze['wave_plan_sha256'][str(wave)],'Original sealed plan changed')
    require(digest(wave/'session.json')==f.read(f.REGISTRY)['original_wave_session_sha256'][str(wave/'session.json')],
            'Original wave session changed')
    metadata=verify_live_cooperative_schedule(wave/'concurrency.json') if 'concurrency.json' in entries else None
    return metadata


def dispatch_fence(output,value13):
    raw=Path(output);wave=raw.resolve()
    require(raw==wave and not raw.is_symlink() and wave.parent==TEST
            and wave.name in {f'wave{i:02d}' for i in value13['eligible_indices']},'Exact original sealed suffix wave required')
    row,_=validate()
    with schedule_lock():
        metadata=guarded_wave(wave,row)
        index=int(wave.name[4:])
        write_new(OWNER/'dispatches'/f'wave{index:02d}.json',dict(utc=time.time(),pid=os.getpid(),index=index,wave=str(wave),
             plan_sha256=digest(wave/'plan.json'),registration_sha256=digest(f.REGISTRY),
             admission14_registration_sha256=digest(REGISTRY),cooperative_metadata=metadata,dispatch_attempt=1,rerun=False))


def worker(output):
    validate()
    old=f.dispatch_fence
    f.dispatch_fence=dispatch_fence
    try:
        f.worker(output)
    finally:
        f.dispatch_fence=old


def project():
    row,value13=validate()
    from tools import tight_project
    from visuotactile import training_resources
    require(sys.argv[2:]==['evaluate','--job',str(JOB)],'Exact finite nativeowner13 job command required')
    assert_owner_unattempted()
    with schedule_lock():
        for index in row['eligible_indices']:
            guarded_wave(TEST/f'wave{index:02d}',row)
    old=training_resources.atomic_json
    training_resources.atomic_json=f.locked_atomic(old,OWNER,OWNER/'project_status.json')
    tight_project.ENTRIES['paired_evaluate_fragment13']=('evaluate',MODULE,[])
    done=threading.Event();watch=threading.Thread(target=f.startup_watchdog,args=(done,value13),name='startup-watchdog14-native13')
    watch.start()
    try:
        tight_project.execute(read(JOB),'evaluate')
    finally:
        done.set();watch.join(timeout=5);training_resources.atomic_json=old
        require(not watch.is_alive(),'Native13 startup watchdog not joined')


def driver():
    sys.path.insert(0,str(HERE))
    import run_main_recovery05 as original
    import shared_capacity06 as capacity
    from outputs.diagnostics.tight_paired_dagger_20261122_01 import shared_capacity14 as watcher
    import psutil,subprocess
    row,value13=validate()
    require(not OUT.exists() and not LOGS.exists(),'Attempted14 driver cannot restart')
    # At startup watcher14 may already have paused the metadata; no native
    # owner receipts/fences may exist, then exact guarded suffix replaces the
    # original plan/session-only test. No source or wave bytes are rewritten.
    assert_owner_unattempted()
    OUT.mkdir();LOGS.mkdir();original.OUT=OUT
    class Driver(original.Driver):
        def gpu_gate(self,name):
            gpu,ram=(ADMISSION['gpu_resident_mib'],ADMISSION['ram_resident_bytes']) if name=='evaluate_fragment14' else (1024,2*2**30)
            while True:
                self.finite()
                try:
                    state=read(WATCH_AUDIT/'resource_watch_v4_status.json')
                    identity=watcher.startup_row(state,psutil.Process(state['pid']),time.time(),allow_gap=True)
                    observed=capacity.capacity_decision(capacity.live_capacity(),gpu_resident_mib=gpu,ram_resident_bytes=ram)
                    require(observed['ready'],'Actual capacity below residence plus native reserve')
                    state=read(WATCH_AUDIT/'resource_watch_v4_status.json')
                    watcher.startup_row(state,psutil.Process(state['pid']),time.time(),allow_gap=True)
                    self.write(OUT/(name+'_capacity_admission.json'),dict(utc=time.time(),watcher=identity,capacity=observed,admission=ADMISSION))
                    return
                except (OSError,KeyError,TypeError,ValueError,psutil.Error,subprocess.SubprocessError) as error:
                    self.note('waiting_actual_capacity_before_'+name,reason=repr(error));time.sleep(5)
        def run(self,name,args,gpu):
            validate();return super().run(name,args,gpu)
    instance=Driver();instance.logs=LOGS;instance.source_identity=row['source_sha256']
    write_new(OUT/'registration.json',dict(row,activated=True,activated_utc=time.time()))
    try:
        # The exclusive durable driver-level dispatch fences even a pre-owner
        # helper failure, while the native per-wave13 fences remain authoritative.
        write_new(OUT/'native_owner_dispatch14.json',dict(utc=time.time(),job_sha256=digest(JOB),dispatch_attempt=1,rerun=False))
        instance.run('evaluate_fragment14',['-m',MODULE,'project','evaluate','--job',str(JOB)],True)
        for stage in row['tail_stages']:
            if not stage['skip_closed']:instance.run(stage['name'],stage['args'],True)
        write_new(OUT/'closure.json',dict(utc=time.time(),status='original_missing_suffix_and_audits_complete',
            original13_driver_remains_failed=True,native_owner13_used_once=True,report_required=True))
    except BaseException as error:
        write_new(OUT/'failure.json',dict(utc=time.time(),error=repr(error),traceback=traceback.format_exc()));raise
    finally:
        write_new(OUT/'driver_exit.json',dict(utc=time.time(),pid=os.getpid(),returned_after_owned_helpers=True))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='project':return project()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','driver','worker']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':print(json.dumps(prepare(),indent=2))
    elif args.mode=='worker':
        require(args.output is not None,'Explicit original wave required');worker(args.output)
    else:
        def interrupted(signum,frame):raise KeyboardInterrupt(signum)
        for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
        driver()

if __name__=='__main__':main()
