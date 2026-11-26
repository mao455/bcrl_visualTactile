"""Inactive finite original-wave17..63 queue; import performs no lifecycle I/O.

Run-local orchestration reuses tight_project commands/monitor, frozen13 science,
lease and startup-readiness locking. A reaped software exit consumes its wave;
only the NEXT never-dispatched wave can run. Original failures remain intact.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse,copy,fcntl,hashlib,json,math,os,signal,sys,threading,time,traceback
from outputs.diagnostics.tight_paired_dagger_20261122_01 import test_fragment13 as f
from outputs.diagnostics.tight_paired_dagger_20261122_01 import run_test_admission14 as a14

HERE,ROOT,TEST,DEADLINE=f.HERE,f.ROOT,f.TEST,f.DEADLINE
OWNER=ROOT/'owners/test_continuation15'
JOB=HERE/'test_continuation15_job.json'
MODULE='outputs.diagnostics.tight_paired_dagger_20261122_01.run_test_continuation15'
REGISTRY=HERE/'TEST_CONTINUATION15_REGISTRATION.json'
REVIEW=HERE/'ROOT_TEST_CONTINUATION15_REVIEW.json'
BOUNDARY=HERE/'ROOT_TEST_CONTINUATION15_BOUNDARY.json'
OUT=HERE/'driver_test_continuation15'
LOGS=ROOT/'cycle_logs_continuation15'
WATCH_AUDIT=HERE/'watch_shared_capacity15'
SCHEDULE_LOCK=ROOT/'continuation15_schedule.lock'
ELIGIBLE=list(range(17,64))
ADMISSION=a14.ADMISSION
read,digest,require,write_new=f.read,f.digest,f.require,f.write_new
pristine_names,paused_schedule_bytes=a14.pristine_names,a14.paused_schedule_bytes
POLICY=dict(version='outcome_independent_software_exit_continue_next_v1',registered_indices=ELIGIBLE,
            attempts_per_wave=1,continue_only_after_child_reaped=True,retry_consumed_wave=False,
            global_stop_deadline_monitor_or_infrastructure_error_aborts=True,
            startup_timeout_seconds=f.STARTUP_SECONDS,startup_timeout_consumes_wave=True,
            no_outcome_values_used=True,original_software_unfinished_indices=[12,16])

@contextmanager
def schedule_lock():
    with SCHEDULE_LOCK.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);yield


def boundary_gate():
    import psutil
    row=read(BOUNDARY)
    require(row.get('status')=='actual_admission14_failed_closed_missing_only15_boundary'
            and row.get('deadline')==DEADLINE and row.get('all_admission14_actors_reaped') is True
            and row.get('no_external_signals') is True,'Actual closed14 boundary required')
    require(row['eligible_indices']==ELIGIBLE and row['all_consumed_indices']==list(range(17))
            and row['software_unfinished_indices']==[12,16],'Exact original17 consumed /47 pristine split required')
    required=[HERE/'driver_test_admission14/failure.json',HERE/'driver_test_admission14/driver_exit.json',
              HERE/'WATCHER14_ACTUAL_CLOSED.json',a14.OWNER/'project_failure.json',a14.OWNER/'project_exit.json',
              a14.OWNER/'project_job_started.json',a14.OWNER/'monitor.json',a14.OWNER/'project_status.json']
    require(set(row['artifact_sha256'])>={str(p) for p in required},'Incomplete actual14 failed boundary')
    for path,expected in row['artifact_sha256'].items():require(digest(path)==expected,'Actual14 evidence changed: '+path)
    for identity in row['closed_identities']:
        pid,created=identity['pid'],identity['created']
        require(type(pid) is int and pid>0,'Positive actual PID required')
        try:proc=psutil.Process(pid)
        except psutil.NoSuchProcess:continue
        require(type(created) in (int,float) and math.isfinite(created) and created>0,
                'Unknown historical birth may only bind an absent PID')
        require(abs(proc.create_time()-created)>.001,'Actual14 identity still live')
    watch=read(HERE/'WATCHER14_ACTUAL_CLOSED.json');service=watch['service']
    must_be_closed={read(HERE/'driver_test_admission14/driver_exit.json')['pid'],watch['pid'],
                    read(a14.OWNER/'project_job_started.json')['pid'],read(a14.OWNER/'monitor.json')['pid'],
                    read(a14.OWNER/'project_status.json')['pid']}
    require(must_be_closed<={identity['pid'] for identity in row['closed_identities']},
            'Actual14 driver/watcher/native owner/monitor/worker identities missing')
    require(watch['source_pid_reaped'] is True and watch['conflicts']==[]
            and str(service['MainPID'])=='0' and service['ActiveState']=='inactive' and service['Result']=='success',
            'Actual watcher14 restoration/closure required')
    require(read(a14.OWNER/'project_exit.json')['children_reaped'] is True
            and read(HERE/'driver_test_admission14/driver_exit.json')['returned_after_owned_helpers'] is True,
            'Actual14 native/helper children not reaped')
    return row


def prefix_inventory():
    # Cooperative watcher concurrency bytes are separately backed/restored;
    # all other original prefix artifacts, including empty logs, stay frozen.
    return {str(TEST/f'wave{i:02d}'):{k:v for k,v in f.inventory(TEST/f'wave{i:02d}').items()
            if k!='concurrency.json'} for i in range(17)}


def exact_consumed_census():
    original_logs=[i for i in range(64) if (TEST/f'project_stage_{i:02d}.log').exists()]
    native_logs=[i for i in range(51) if (a14.OWNER/f'project_stage_{i:02d}.log').exists()]
    fences=sorted(p.name for p in (a14.OWNER/'dispatches').iterdir())
    require(original_logs==list(range(13)) and native_logs==list(range(4))
            and fences==[f'wave{i:02d}.json' for i in range(13,17)],'Empty logs/fences count as consumed; exact prefix required')
    for i in ELIGIBLE:pristine_names(TEST/f'wave{i:02d}')
    return dict(consumed_indices=list(range(17)),eligible_indices=ELIGIBLE,original12_stage_logs=original_logs,
                original14_stage_logs=native_logs,original14_dispatch_fences=fences,
                prefix_inventory=prefix_inventory(),lifecycle_only=True,outcome_values_interpreted=False)


def prepare():
    require(not any(p.exists() or p.is_symlink() for p in (REGISTRY,OWNER,JOB,OUT,LOGS)),
            'Preserve any attempted15 identity')
    value13=f.validate_binding();boundary=boundary_gate();census=exact_consumed_census()
    freeze=read(TEST/'registration.json');original=read(freeze['job'])
    session=copy.deepcopy(read(TEST/'session.json'))
    require(session['deadline']==DEADLINE and session['resource_policy']=='full_gpu_v1','Original finite FullLease required')
    session['resource_root']=str(OWNER);session['owned_worker_roots']=[str(TEST/f'wave{i:02d}') for i in ELIGIBLE]
    job=dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(OWNER),
             stages=[dict(entry='paired_evaluate_continuation15',args=original['stages'][i]['args']) for i in ELIGIBLE])
    names=['run_test_continuation15.py','shared_capacity15.py','run_test_admission14.py','shared_capacity14.py',
           'test_fragment13.py','shared_capacity13.py','run_main_recovery05.py','shared_capacity06.py',
           'resource_watch_v4.py','driver_support.py','driver_support_v4.py']
    OWNER.mkdir(parents=True,exist_ok=False);(OWNER/'dispatches').mkdir();(OWNER/'launches').mkdir()
    (OWNER/'stage_exits').mkdir();(OWNER/'startup_stops').mkdir();f.sync_directory(OWNER.parent)
    write_new(OWNER/'session.json',session)
    write_new(OWNER/'cuda_allocator_policy.json',dict(version='reclaim_unused_on_low_headroom_v1'))
    write_new(JOB,job)
    row=dict(version='test_continuation15_missing47_v1',utc=time.time(),deadline=DEADLINE,owner=str(OWNER),job=str(JOB),
         job_sha256=digest(JOB),owner_session_sha256=digest(OWNER/'session.json'),
         original13_registry_sha256=digest(f.REGISTRY),original14_registry_sha256=digest(a14.REGISTRY),
         original_test_registration_sha256=digest(TEST/'registration.json'),boundary_sha256=digest(BOUNDARY),
         original14_artifact_sha256=boundary['artifact_sha256'],census=census,eligible_indices=ELIGIBLE,
         original_wave_session_sha256=value13['original_wave_session_sha256'],
         source_sha256=value13['source_sha256'],frozen_pairs=value13['frozen_pairs'],
         original_cohorts_sha256=value13['original_cohorts_sha256'],tail_stages=value13['tail_stages'],
         admission=ADMISSION,software_error_policy=POLICY,
         orchestration_sha256={str(HERE/name):digest(HERE/name) for name in names},
         scientific_changes=0,training_updates=0,no_outcome_retry=True,
         original12_13_14_failures_preserved=True,external_process_presence_is_not_a_veto=True)
    write_new(REGISTRY,row);return row


def validate(*,review=True):
    row=read(REGISTRY);value13=f.validate_binding()
    require(not OWNER.is_symlink() and OWNER.resolve()==OWNER and not TEST.is_symlink() and TEST.resolve()==TEST,
            'Canonical literal owner15/original test required')
    require(row['version']=='test_continuation15_missing47_v1' and row['deadline']==DEADLINE
            and time.time()<DEADLINE and row['eligible_indices']==ELIGIBLE and row['software_error_policy']==POLICY
            and row['admission']==ADMISSION,'Exact finite47 policy required')
    exact={JOB:row['job_sha256'],OWNER/'session.json':row['owner_session_sha256'],
           f.REGISTRY:row['original13_registry_sha256'],a14.REGISTRY:row['original14_registry_sha256'],
           BOUNDARY:row['boundary_sha256'],TEST/'registration.json':row['original_test_registration_sha256'],
           ROOT/'cohorts.json':row['original_cohorts_sha256']}
    exact.update({Path(p):h for p,h in row['orchestration_sha256'].items()})
    exact.update({Path(p):h for p,h in row['original14_artifact_sha256'].items()})
    for path,expected in exact.items():require(digest(path)==expected,'Bound15 artifact changed: '+str(path))
    require(row['source_sha256']==value13['source_sha256'] and row['frozen_pairs']==value13['frozen_pairs'],
            'Frozen production295/eightweights mapping changed')
    with schedule_lock():
        require(prefix_inventory()==row['census']['prefix_inventory'],'Original consumed prefix changed')
    if review:
        accepted=read(REVIEW)
        require(accepted['status']=='source_and_actual_missing47_review_passed_inactive'
                and accepted['registration_sha256']==digest(REGISTRY),'Root actual15 review required')
    return row,value13


def verify_live_cooperative_schedule(path):
    from outputs.diagnostics.tight_paired_dagger_20261122_01 import shared_capacity15 as watcher
    import psutil
    status=read(WATCH_AUDIT/'resource_watch_v4_status.json')
    watcher.live_identity(status,psutil.Process(status['pid']),time.time())
    require(status['paused'] is True,'Exact current15 pause state required')
    backups=list(WATCH_AUDIT.glob(f'resource_watch_v4_backups_*_{status["pid"]}'))
    require(len(backups)==1,'Exact current watcher15 backup required')
    backup=backups[0]/(hashlib.sha256(str(path).encode()).hexdigest()+'.json')
    require(backup.is_file() and not backup.is_symlink() and backups[0].resolve()==backups[0],'Unsafe schedule backup')
    row=read(backup)
    require(row==dict(path=str(path),existed=False,base64=None,sha256=None),'Suffix schedule must derive from original absence')
    require(path.read_bytes()==paused_schedule_bytes(None,status['pid']),'Unknown cooperative metadata')
    return dict(path=str(path),sha256=digest(path),watcher_pid=status['pid'],backup=str(backup),backup_sha256=digest(backup))


def guarded_wave(wave,row):
    entries=pristine_names(wave,allow_cooperative=True);freeze=read(TEST/'registration.json')
    require(digest(wave/'plan.json')==freeze['wave_plan_sha256'][str(wave)]
            and digest(wave/'session.json')==row['original_wave_session_sha256'][str(wave/'session.json')],
            'Original plan/session changed')
    return verify_live_cooperative_schedule(wave/'concurrency.json') if 'concurrency.json' in entries else None


def dispatch_fence(output,value):
    wave=Path(output)
    require(wave.resolve()==wave and not wave.is_symlink() and wave.parent==TEST
            and wave.name in {f'wave{i:02d}' for i in ELIGIBLE},'Exact original47 wave required')
    row,_=validate()
    with schedule_lock():
        metadata=guarded_wave(wave,row);index=int(wave.name[4:])
        launch=read(OWNER/'launches'/f'wave{index:02d}.json')
        require(launch['index']==index and launch['dispatch_attempt']==1 and launch['registration_sha256']==digest(REGISTRY),
                'Exclusive owner15 launch fence required before worker')
        write_new(OWNER/'dispatches'/f'wave{index:02d}.json',dict(utc=time.time(),pid=os.getpid(),index=index,wave=str(wave),
            plan_sha256=digest(wave/'plan.json'),registration_sha256=digest(REGISTRY),
            original13_registration_sha256=digest(f.REGISTRY),launch_sha256=digest(OWNER/'launches'/f'wave{index:02d}.json'),
            cooperative_metadata=metadata,dispatch_attempt=1,rerun=False))


def worker(output):
    row,_=validate()
    old=(f.OWNER,f.validate_binding,f.dispatch_fence)
    f.OWNER=OWNER;f.validate_binding=lambda:row;f.dispatch_fence=dispatch_fence
    try:f.worker(output)
    finally:f.OWNER,f.validate_binding,f.dispatch_fence=old


def startup_timeout(status,*,now=time.time,after_intent=None):
    """Frozen13 readiness-lock algorithm; scoped abort preserves future waves."""
    stage=status['stage'];index=ELIGIBLE[stage];wave=TEST/f'wave{index:02d}'
    intent=OWNER/f'startup_intent_stage{stage:02d}_wave{index:02d}.json'
    write_new(intent,dict(utc=now(),stage=stage,index=index,status=status,timeout_seconds=f.STARTUP_SECONDS,
                          physical_steps_unknown=True,entire_wave_consumed=True,rerun=False))
    if after_intent:after_intent()
    with f.lifecycle_lock(OWNER):
        current=read(OWNER/'project_status.json');initialized=(wave/'initialization.json').exists()
        abandoned=current!=status or initialized
        if not abandoned:
            write_new(OWNER/'startup_stops'/f'wave{index:02d}.json',dict(utc=now(),intent_sha256=digest(intent),index=index))
        decision=dict(utc=now(),intent_sha256=digest(intent),decision='abandoned' if abandoned else 'scoped_stop_committed',
                      stage_changed=current!=status,initialized=initialized,stop_written=not abandoned,global_stop_written=False)
        write_new(OWNER/f'startup_decision_stage{stage:02d}_wave{index:02d}.json',decision)
    return decision


def startup_watchdog(done):
    seen=set()
    try:
        while not done.wait(.5):
            if not (OWNER/'project_status.json').exists():continue
            status=read(OWNER/'project_status.json');stage=status['stage']
            require(type(stage) is int and 0<=stage<len(ELIGIBLE),'Unexpected15 stage')
            index=ELIGIBLE[stage];wave=TEST/f'wave{index:02d}';begin=wave/'run_started.json'
            require(status['command'][-2:]==['--output',str(wave)],'Unexpected15 command')
            if not begin.exists() or stage in seen or (wave/'initialization.json').exists():continue
            started=read(begin)['utc'];require(type(started) in (int,float) and math.isfinite(started),'Finite startup timestamp required')
            if time.time()-started<f.STARTUP_SECONDS:continue
            seen.add(stage);startup_timeout(status)
    except BaseException as error:
        write_new(OWNER/'startup_watchdog15_error.json',dict(utc=time.time(),error=repr(error)))
        with f.lifecycle_lock(OWNER):(OWNER/'STOP').touch(exist_ok=True)


def record_stage_exit(stage,index,command,returncode,*,startup_stopped=False,child_reaped=True,
                      process_started=True,queue_aborted=False):
    require(child_reaped is True,'No next dispatch before actual child reaping')
    row=dict(utc=time.time(),stage=stage,index=index,command=command,returncode=returncode,
             child_reaped=True,process_started=process_started,queue_aborted=queue_aborted,
             startup_stopped=startup_stopped,software_completed=returncode==0 and not startup_stopped and not queue_aborted,
             consumed=True,rerun=False,next_only_if_pristine=True,policy_sha256=hashlib.sha256(json.dumps(POLICY,sort_keys=True).encode()).hexdigest())
    write_new(OWNER/'stage_exits'/f'wave{index:02d}.json',row);return row


def execute_queue(job):
    """Equivalent native tight_project finite owner, with registered rc policy."""
    import shutil,subprocess
    from tools import tight_project
    from visuotactile.training_resources import atomic_json
    queue=tight_project.commands(job,'evaluate');session=read(OWNER/'session.json')
    require(Path.cwd().resolve()==tight_project.PROJECT and session['deadline']==DEADLINE and time.time()<DEADLINE,
            'Canonical finite native queue required')
    require(shutil.disk_usage(OWNER).free>=session.get('reserve_bytes',12*2**30),'Native disk reserve unmet')
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             PYTHONPATH=os.pathsep.join([str(tight_project.PROJECT),str(tight_project.PROJECT/'rl_games')]))
    child=monitor=None;exits=[];current=None
    with tight_project.termination_signals(),(OWNER/'project_owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (OWNER/'project_job_started.json').exists() and not (OWNER/'STOP_MONITOR').exists(),'Attempted15 owner cannot restart')
        write_new(OWNER/'project_job_started.json',dict(job=job,commands=queue,pid=os.getpid(),utc=time.time(),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),software_error_policy=POLICY))
        try:
            with (OWNER/'project_monitor.log').open('x') as log:
                monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(OWNER)],
                    cwd=tight_project.PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            until=min(DEADLINE,time.time()+30)
            while not (OWNER/'shared_resources.json').exists():
                require(monitor.poll() is None and time.time()<until,'Resource monitor did not start');time.sleep(.2)
            for stage,command in enumerate(queue):
                validate() # A scientific/binding change aborts, rather than consuming future waves.
                index=ELIGIBLE[stage];wave=TEST/f'wave{index:02d}'
                require(monitor.poll() is None and time.time()<DEADLINE and not (OWNER/'STOP').exists(),
                        'Monitor/deadline/global STOP ended queue')
                with schedule_lock():
                    guarded_wave(wave,read(REGISTRY))
                    write_new(OWNER/'launches'/f'wave{index:02d}.json',dict(utc=time.time(),stage=stage,index=index,
                        command=command,registration_sha256=digest(REGISTRY),dispatch_attempt=1,rerun=False))
                current=(stage,index,command);child=None
                with (OWNER/f'project_stage_{stage:02d}.log').open('x') as log:
                    child=subprocess.Popen(command,cwd=tight_project.PROJECT,env=env,stdout=log,
                                           stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(OWNER/'project_status.json',dict(stage=stage,command=command,pid=child.pid,utc=time.time()))
                scoped=OWNER/'startup_stops'/f'wave{index:02d}.json'
                while child.poll() is None:
                    require(monitor.poll() is None and time.time()<DEADLINE and not (OWNER/'STOP').exists(),
                            'Monitor/deadline/global STOP ended queue')
                    if scoped.exists():tight_project.stop(child);break
                    time.sleep(1)
                child.wait() # Explicit reaping precedes durable rc and any NEXT dispatch.
                exits.append(record_stage_exit(stage,index,command,child.returncode,startup_stopped=scoped.exists()))
                if child.returncode or scoped.exists():
                    atomic_json(OWNER/'project_nonzero_stages.json',dict(utc=time.time(),policy=POLICY,
                        stages=[r for r in exits if not r['software_completed']],continued_next_never_launched=True))
            atomic_json(OWNER/'project_complete.json',dict(utc=time.time(),status='finite47_queue_consumed_all',stages=len(queue),
                all_children_reaped=True,software_completed_stages=sum(r['software_completed'] for r in exits),
                software_unfinished_indices=[r['index'] for r in exits if not r['software_completed']],
                all_waves_software_completed=all(r['software_completed'] for r in exits),outcomes_interpreted=False))
        except BaseException as error:
            atomic_json(OWNER/'project_failure.json',dict(error=repr(error),utc=time.time()));raise
        finally:
            child_reaped=child is None
            try:
                tight_project.stop(child)
                child_reaped=child is None or child.poll() is not None
                if current is not None and not (OWNER/'stage_exits'/f'wave{current[1]:02d}.json').exists():
                    if child is not None:child.wait();child_reaped=True
                    record_stage_exit(*current,None if child is None else child.returncode,
                        process_started=child is not None,queue_aborted=True,child_reaped=child_reaped)
            finally:
                # A failed exit-receipt write must never strand the native monitor.
                try:(OWNER/'STOP_MONITOR').touch()
                finally:
                    monitor_reaped=monitor is None
                    try:
                        if monitor:
                            try:monitor.wait(timeout=15)
                            except subprocess.TimeoutExpired:tight_project.stop(monitor)
                            monitor_reaped=monitor.poll() is not None
                    finally:
                        atomic_json(OWNER/'project_exit.json',dict(children_reaped=child_reaped and monitor_reaped,
                            worker_reaped=child_reaped,monitor_reaped=monitor_reaped,utc=time.time()))


def project():
    row,_=validate();from tools import tight_project
    from visuotactile import training_resources
    require(sys.argv[2:]==['evaluate','--job',str(JOB)],'Exact finite owner15 job required')
    require(not (OWNER/'project_job_started.json').exists(),'Attempted15 cannot restart')
    with schedule_lock():
        for index in ELIGIBLE:guarded_wave(TEST/f'wave{index:02d}',row)
    old=training_resources.atomic_json
    training_resources.atomic_json=f.locked_atomic(old,OWNER,OWNER/'project_status.json')
    tight_project.ENTRIES['paired_evaluate_continuation15']=('evaluate',MODULE,[])
    done=threading.Event();watch=threading.Thread(target=startup_watchdog,args=(done,),name='startup-watchdog15')
    watch.start()
    try:execute_queue(read(JOB))
    finally:
        done.set();watch.join(timeout=5);training_resources.atomic_json=old
        require(not watch.is_alive(),'Startup watchdog15 not joined')


def driver():
    sys.path.insert(0,str(HERE));import run_main_recovery05 as original
    import shared_capacity06 as capacity
    from outputs.diagnostics.tight_paired_dagger_20261122_01 import shared_capacity15 as watcher
    import psutil,subprocess
    row,_=validate();require(not OUT.exists() and not LOGS.exists(),'Attempted15 driver cannot restart')
    OUT.mkdir();LOGS.mkdir();original.OUT=OUT
    class Driver(original.Driver):
        def gpu_gate(self,name):
            gpu,ram=(ADMISSION['gpu_resident_mib'],ADMISSION['ram_resident_bytes']) if name=='evaluate_continuation15' else (1024,2*2**30)
            while True:
                self.finite()
                try:
                    state=read(WATCH_AUDIT/'resource_watch_v4_status.json')
                    identity=watcher.startup_row(state,psutil.Process(state['pid']),time.time(),allow_gap=True)
                    observed=capacity.capacity_decision(capacity.live_capacity(),gpu_resident_mib=gpu,ram_resident_bytes=ram)
                    require(observed['ready'],'Actual capacity below residence plus native reserve')
                    state=read(WATCH_AUDIT/'resource_watch_v4_status.json')
                    watcher.startup_row(state,psutil.Process(state['pid']),time.time(),allow_gap=True)
                    self.write(OUT/(name+'_capacity_admission.json'),dict(utc=time.time(),watcher=identity,capacity=observed,admission=ADMISSION));return
                except (OSError,KeyError,TypeError,ValueError,psutil.Error,subprocess.SubprocessError) as error:
                    self.note('waiting_actual_capacity_before_'+name,reason=repr(error));time.sleep(5)
        def run(self,name,args,gpu):validate();return super().run(name,args,gpu)
    instance=Driver();instance.logs=LOGS;instance.source_identity=row['source_sha256']
    write_new(OUT/'registration.json',dict(row,activated=True,activated_utc=time.time()))
    try:
        write_new(OUT/'native_owner_dispatch15.json',dict(utc=time.time(),job_sha256=digest(JOB),dispatch_attempt=1,rerun=False))
        instance.run('evaluate_continuation15',['-m',MODULE,'project','evaluate','--job',str(JOB)],True)
        for stage in row['tail_stages']:
            if not stage['skip_closed']:instance.run(stage['name'],stage['args'],True)
        write_new(OUT/'closure.json',dict(utc=time.time(),status='original_missing47_queue_and_audits_complete',
            queue_consumed_not_all_software_completed=True,original12_13_14_failures_preserved=True,report_required=True))
    except BaseException as error:
        write_new(OUT/'failure.json',dict(utc=time.time(),error=repr(error),traceback=traceback.format_exc()));raise
    finally:write_new(OUT/'driver_exit.json',dict(utc=time.time(),pid=os.getpid(),returned_after_owned_helpers=True))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='project':return project()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','driver','worker']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':print(json.dumps(prepare(),indent=2))
    elif args.mode=='worker':require(args.output is not None,'Explicit original wave required');worker(args.output)
    else:
        def interrupted(signum,frame):raise KeyboardInterrupt(signum)
        for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
        driver()

if __name__=='__main__':main()
