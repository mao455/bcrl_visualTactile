"""INACTIVE missing-only sealed-test contingency; import performs no I/O.

Root must close recovery12 and certify the actual boundary before ``prepare``.
Consumed/ambiguous whole waves are never replayed. Native research math, source
identities, original plans and original deadline remain unchanged.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOT = PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
TEST = ROOT/'evaluation/test'
OWNER = ROOT/'owners/test_fragment13'
OUT = HERE/'driver_test_fragment13'
LOGS = ROOT/'cycle_logs_fragment13'
MODULE = 'outputs.diagnostics.tight_paired_dagger_20261122_01.test_fragment13'
WATCH_MODULE = 'outputs.diagnostics.tight_paired_dagger_20261122_01.shared_capacity13'
REGISTRY = HERE/'TEST_FRAGMENT13_REGISTRATION.json'
REVIEW = HERE/'ROOT_TEST_FRAGMENT13_REVIEW.json'
BOUNDARY = HERE/'ROOT_TEST_FRAGMENT13_BOUNDARY.json'
JOB = HERE/'test_fragment13_job.json'
DEADLINE = 1795684571.2261844
STARTUP_SECONDS = 1800


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sync_directory(path):
    descriptor=os.open(Path(path),os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path, value):
    path=Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def inventory(directory):
    """Lifecycle identities only. Never interpret result contents or HDF5."""
    rows = {}
    for path in sorted(Path(directory).rglob('*')):
        require(not path.is_symlink(), 'Ambiguous symlink in original wave')
        if path.is_dir():
            rows[str(path.relative_to(directory))] = dict(kind='directory',mtime_ns=path.stat().st_mtime_ns)
            continue
        if path.is_file():
            stat = path.stat()
            row = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if path.suffix in ('.json', '.log') and stat.st_size <= 2**20:
                row['sha256'] = digest(path)
            rows[str(path.relative_to(directory))] = row
        else:
            raise ValueError('Unexplained special entry in original wave: '+str(path))
    return rows


def lifecycle_census(test):
    """Call only AFTER root has actually closed every original actor.

    Empty original stage logs count as launches. Any wave artifact beyond the
    original plan/session consumes that wave and all earlier waves. The last
    status must belong to the existing original prefix; unusual gaps refuse.
    """
    require(not Path(test).is_symlink(), 'Unsafe original test owner symlink')
    test = Path(test).resolve()
    freeze = read(test/'registration.json')
    job = read(freeze['job'])
    started = read(test/'project_job_started.json')
    status = read(test/'project_status.json')
    require(freeze['cohort'] == 'test' and freeze['n'] == len(job['stages']) == 64,
            'Exact original64 population required')
    require(digest(freeze['job']) == freeze['job_sha256'] and started['job'] == job,
            'Original job or started binding changed')
    require(Path(job['resource_root']).resolve() == test, 'Original owner differs')
    expected_commands = [
        ['-m','outputs.diagnostics.tight_paired_dagger_20261122_01.eval_startup12',
         'worker','--output',str(test/f'wave{i:02d}')] for i in range(64)]
    require(len(started.get('commands',[])) == 64
            and all(isinstance(command,list) and command[1:] == expected
                    and Path(command[0]).name.startswith('python')
                    for command,expected in zip(started['commands'],expected_commands)),
            'Original frozen12 dispatch command identity changed')
    require(type(status['stage']) is int and 0 <= status['stage'] < 64, 'Invalid original stage')
    logs = [i for i in range(64) if (test/f'project_stage_{i:02d}.log').exists()]
    require(logs and logs == list(range(max(logs)+1)), 'Original stage logs are not a unique prefix')
    require(0 <= max(logs)-status['stage'] <= 1, 'Status does not describe the last durable launch boundary')
    require(status.get('command',[])[1:] == expected_commands[status['stage']], 'Original status command changed')
    require(all(not (test/f'project_stage_{i:02d}.log').is_symlink() for i in logs), 'Unsafe original launch log symlink')
    expected_dirs = {f'wave{i:02d}' for i in range(64)}
    require({p.name for p in test.iterdir() if p.name.startswith('wave')} == expected_dirs,
            'Unexpected original wave paths')
    rows = []
    last = max(logs)
    for i, stage in enumerate(job['stages']):
        wave = test/f'wave{i:02d}'
        require(not wave.is_symlink() and wave.is_dir(), 'Unsafe original wave directory')
        require(stage == dict(entry='paired_evaluate', args=['worker', '--output', str(wave)]),
                'Original wave order/arguments changed')
        require(digest(wave/'plan.json') == freeze['wave_plan_sha256'][str(wave)],
                'Original frozen wave changed')
        payload = inventory(wave)
        require('plan.json' in payload and 'session.json' in payload, 'Original wave identity missing')
        evidence = set(payload)-{'plan.json', 'session.json'}
        if evidence:
            last = max(last, i)
        terminal = 'results.json' in payload and 'process_exit_requested.json' in payload
        rows.append(dict(index=i, wave=str(wave), stage_log_present=i in logs,
                         terminal_receipts_present=terminal, inventory=payload))
    # Runtime evidence beyond the launch-prefix means dispatch history is
    # ambiguous. Refuse rather than interpreting absent logs as zero actions.
    require(last == max(logs), 'Runtime artifacts beyond original durable launch prefix')
    for row in rows:
        i = row['index']
        row.update(consumed=i <= last, eligible_for_dispatch=i > last,
                   status='completed_wave_receipts' if row['terminal_receipts_present'] else
                          'consumed_or_ambiguous_wave' if i <= last else 'provably_never_launched_wave')
        if i > last:
            require(set(row['inventory']) == {'plan.json', 'session.json'}, 'Suffix is not pristine')
    require(last < 63, 'No original unlaunched whole-wave suffix remains')
    return dict(version='test_fragment13_lifecycle_census_v1', lifecycle_only=True,
                outcome_values_interpreted=False, registered_waves=64, registered_opportunities=256,
                consumed_waves=last+1, eligible_indices=list(range(last+1,64)), rows=rows,
                original_registration_sha256=digest(test/'registration.json'),
                original_job_sha256=digest(freeze['job']),
                original_started_sha256=digest(test/'project_job_started.json'),
                original_status_sha256=digest(test/'project_status.json'))


def boundary_check(boundary, *, now=None):
    """Root certificate plus exact durable failures; no process is signalled."""
    import psutil
    now = time.time() if now is None else now
    require(boundary.get('status') == 'actual_original12_boundary_closed_for_missing_only_fragment',
            'Root actual-boundary review required')
    require(boundary.get('deadline') == DEADLINE and now < DEADLINE,
            'Original finite deadline changed/expired')
    require(boundary.get('all_original12_actors_reaped') is True and boundary.get('no_external_signals') is True,
            'Original actors not closed')
    for item in boundary['closed_identities']:
        require(type(item['pid']) is int and item['pid']>0 and type(item['created']) in (int,float)
                and math.isfinite(item['created']), 'PID/create-time required')
        if psutil.pid_exists(item['pid']):
            proc = psutil.Process(item['pid'])
            require(abs(proc.create_time()-item['created']) > .001, 'Original identity is still live')
    required = [TEST/'project_failure.json', TEST/'project_exit.json',
                TEST/'monitor.json',TEST/'project_status.json',TEST/'project_job_started.json',
                HERE/'driver_eval_recovery12/failure.json', HERE/'driver_eval_recovery12/driver_exit.json',
                HERE/'WATCHER12_ACTUAL_CLOSED.json']
    require(set(boundary['artifact_sha256']) >= {str(p) for p in required}, 'Incomplete actual closure binding')
    for path, expected in boundary['artifact_sha256'].items():
        require(digest(path) == expected, 'Actual boundary artifact changed: '+path)
    require(read(TEST/'project_exit.json')['children_reaped'] is True, 'Original test children not reaped')
    require(read(HERE/'driver_eval_recovery12/driver_exit.json')['returned_after_owned_helpers'] is True,
            'Original12 helper closure missing')
    watch = read(HERE/'WATCHER12_ACTUAL_CLOSED.json')
    require(watch['source_pid_reaped'] is True and watch['conflicts'] == [], 'Actual watcher12 not restored')
    service = watch['service']
    require(str(service['MainPID']) == '0' and service['ActiveState'] == 'inactive'
            and service['Result'] == 'success', 'Watcher12 service remains active or failed')
    must_be_closed={read(TEST/'project_job_started.json')['pid'],
                    read(TEST/'monitor.json')['pid'],read(TEST/'project_status.json')['pid'],
                    read(HERE/'driver_eval_recovery12/driver_exit.json')['pid'],watch['pid']}
    require({row['pid'] for row in boundary['closed_identities']} >= must_be_closed,
            'Original owner/monitor/worker/driver/watcher PID identities absent from root boundary certificate')
    require(not (TEST/'project_complete.json').exists()
            and not (HERE/'driver_eval_recovery12/closure.json').exists(), 'Preserve original12 failure truth')


def pending_tail_stages():
    """Skip only actual rc0 tails; refuse attempted or partial failed helpers."""
    result = []
    paths = [
        ('summarize_test', ['-m','tools.tight_paired_evaluate','summarize','--output',str(TEST),'--reconstruct-cuda-noise']),
        ('audit_collection_cuda_noise_recovery05', [str(HERE/'audit_collection_recovery05.py'),'noise',
            '--collection',str(ROOT/'collection'),'--output',str(HERE/'collection_noise_recovery05_actual01'),
            '--reconstruct-cuda-noise'])]
    for name, args in paths:
        exit_path = ROOT/'cycle_logs_recovery12'/(name+'_exit.json')
        log = ROOT/'cycle_logs_recovery12'/(name+'.log')
        if exit_path.exists():
            require(read(exit_path)['returncode'] == 0, 'Attempted failed CUDA helper needs separate reviewed scope')
            result.append(dict(name=name, args=args, skip_closed=True, original_exit_sha256=digest(exit_path)))
        else:
            require(not log.exists(), 'Ambiguous attempted original CUDA helper cannot be replayed')
            payload=TEST/'summary' if name=='summarize_test' else HERE/'collection_noise_recovery05_actual01'
            require(not payload.exists(), 'Preserve unexplained/partial original CUDA helper output')
            result.append(dict(name=name, args=args, skip_closed=False))
    return result


def prepare():
    """MUTATING preparation, intentionally never invoked by source-prep agent."""
    require(not REGISTRY.exists() and not OWNER.exists() and not JOB.exists()
            and not OUT.exists() and not LOGS.exists(), 'Preserve attempted fragment13 identities')
    boundary = read(BOUNDARY)
    boundary_check(boundary)
    census = lifecycle_census(TEST)
    freeze = read(TEST/'registration.json')
    from tools.tight_paired_dagger import sources
    from tools.tight_paired_evaluate import validate_wave, validate_pair
    require(sources() == freeze['source_sha256'], 'Frozen production295 changed')
    require(read(ROOT/'session.json')['deadline'] == freeze['deadline'] == DEADLINE, 'Original deadline changed')
    for role, pair in freeze['pairs'].items():
        validate_pair(pair, 'q_reached' if role.startswith('A') else 'u_expert')
    sessions = {}
    for index in range(64):
        wave = TEST/f'wave{index:02d}'
        plan, registered = validate_wave(wave)
        require(registered == freeze and plan['execution_backend']['kit_tasking_threads'] == 8
                and plan['execution_backend']['physics_threads'] == 2, 'Original native wave configuration changed')
        require(read(wave/'session.json')['deadline'] == DEADLINE, 'Original wave deadline differs')
        sessions[str(wave/'session.json')] = digest(wave/'session.json')
    session = copy.deepcopy(read(TEST/'session.json'))
    require(session['resource_policy'] == 'full_gpu_v1' and session['deadline'] == DEADLINE,
            'Exact native FullLease finite policy required')
    session['resource_root'] = str(OWNER)
    session['owned_worker_roots'] = [str(TEST/f'wave{i:02d}') for i in census['eligible_indices']]
    original_job = read(freeze['job'])
    job = dict(version='tight_project_job_v1', operation='evaluate', resource_root=str(OWNER),
               stages=[dict(entry='paired_evaluate_fragment13', args=original_job['stages'][i]['args'])
                       for i in census['eligible_indices']])
    tails = pending_tail_stages()
    names = ('test_fragment13.py','shared_capacity13.py','run_main_recovery05.py','shared_capacity06.py',
             'resource_watch_v4.py','coexistence_watch_recovery05.py','resource_watch_parallel_proposal.py',
             'driver_support.py','driver_support_v4.py','audit_collection_recovery05.py',
             'audit_collection_noise.py','audit_collection_integrity.py','collection_recovery05_scope.py')
    OWNER.mkdir(parents=True, exist_ok=False)
    sync_directory(OWNER.parent)
    (OWNER/'dispatches').mkdir()
    sync_directory(OWNER)
    write_new(OWNER/'session.json',session)
    write_new(OWNER/'cuda_allocator_policy.json',dict(version='reclaim_unused_on_low_headroom_v1'))
    write_new(JOB,job)
    value = dict(version='test_missing_whole_suffix_fragment13_v1', created_utc=time.time(), root=str(ROOT),
                 deadline=DEADLINE, owner=str(OWNER), test=str(TEST), job=str(JOB), job_sha256=digest(JOB),
                 boundary_sha256=digest(BOUNDARY), census=census, original_registration_sha256=digest(TEST/'registration.json'),
                 original_owner_session_sha256=digest(TEST/'session.json'), fragment_owner_session_sha256=digest(OWNER/'session.json'),
                 original_wave_session_sha256=sessions, source_sha256=freeze['source_sha256'],
                 original_cohorts_sha256=digest(ROOT/'cohorts.json'), frozen_pairs=freeze['pairs'],
                 eligible_indices=census['eligible_indices'], tail_stages=tails, startup_timeout_seconds=STARTUP_SECONDS,
                 orchestration_sha256={str(HERE/name):digest(HERE/name) for name in names},
                 scientific_changes=0, optimizer_updates=0, collection_opportunities=0,
                 external_process_presence_is_not_a_veto=True, original12_remains_failed=True)
    write_new(REGISTRY,value)
    return value


def validate_binding(*, require_review=True):
    value = read(REGISTRY)
    require(not TEST.is_symlink() and not OWNER.is_symlink() and TEST.resolve()==TEST and OWNER.resolve()==OWNER,
            'Original/fragment owner path is not canonical or is a symlink')
    require(value['version'] == 'test_missing_whole_suffix_fragment13_v1'
            and value['deadline'] == DEADLINE == read(ROOT/'session.json')['deadline']
            and time.time() < DEADLINE, 'Fragment13 finite identity differs/expired')
    require(value['eligible_indices'] == value['census']['eligible_indices']
            and value['eligible_indices'] == list(range(value['eligible_indices'][0],64)), 'Whole suffix required')
    for path, expected in value['orchestration_sha256'].items():
        require(digest(path) == expected, 'Reviewed13 source changed: '+path)
    exact = {BOUNDARY:value['boundary_sha256'], JOB:value['job_sha256'],
             TEST/'registration.json':value['original_registration_sha256'],
             TEST/'session.json':value['original_owner_session_sha256'],
             OWNER/'session.json':value['fragment_owner_session_sha256'],
             ROOT/'cohorts.json':value['original_cohorts_sha256']}
    exact.update({TEST/'project_job_started.json':value['census']['original_started_sha256'],
                  TEST/'project_status.json':value['census']['original_status_sha256'],
                  Path(read(TEST/'registration.json')['job']):value['census']['original_job_sha256']})
    exact.update({Path(p):h for p,h in value['original_wave_session_sha256'].items()})
    for path, expected in exact.items():
        require(digest(path) == expected, 'Fragment13 identity changed: '+str(path))
    for row in value['tail_stages']:
        if row['skip_closed']:
            exit_path=ROOT/'cycle_logs_recovery12'/(row['name']+'_exit.json')
            require(digest(exit_path)==row['original_exit_sha256'] and read(exit_path)['returncode']==0,
                    'Previously closed original12 helper receipt changed')
    from tools.tight_paired_dagger import sources
    require(sources() == value['source_sha256'], 'Frozen production295 changed')
    if require_review:
        review = read(REVIEW)
        require(review['status'] == 'source_and_actual_missing_suffix_review_passed_inactive'
                and review['registration_sha256'] == digest(REGISTRY), 'Root actual13 review required')
    return value


@contextmanager
def lifecycle_lock(owner):
    with (Path(owner)/'lifecycle13.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def locked_atomic(original, owner, exact_path):
    def atomic(path, value):
        if Path(path).resolve() == Path(exact_path).resolve():
            with lifecycle_lock(owner):
                result=original(path,value)
                with Path(path).open('rb') as stream:
                    os.fsync(stream.fileno())
                sync_directory(Path(path).parent)
                return result
        return original(path,value)
    return atomic


def attempt_timeout(owner, test, receipt_root, status, *, now=time.time, after_intent=None):
    """Serialize final timeout with owner stage and worker readiness commits.

    Intent is durable BEFORE final recheck. A readiness/stage change during
    fsync writes an abandoned decision and never STOP. All three participants
    use this exact lock, so readiness/stage cannot change between recheck and
    the STOP linearization point.
    """
    owner,test,receipt_root = map(Path,(owner,test,receipt_root))
    stage = status['stage']; index = int(Path(status['command'][-1]).name.removeprefix('wave'))
    wave = test/f'wave{index:02d}'
    intent = receipt_root/f'startup_intent_stage{stage:02d}_wave{index:02d}.json'
    write_new(intent,dict(utc=now(),stage=stage,index=index,status=status,timeout_seconds=STARTUP_SECONDS,
                          physical_steps_unknown=True,entire_wave_consumed=True,rerun=False))
    if after_intent is not None:
        after_intent()
    with lifecycle_lock(owner):
        current = read(owner/'project_status.json')
        initialized = (wave/'initialization.json').exists()
        abandoned = current != status or initialized
        if abandoned:
            decision = dict(utc=now(),intent_sha256=digest(intent),decision='abandoned',
                            stage_changed=current!=status,initialized=initialized,stop_written=False)
        else:
            stop = owner/'STOP'
            if not stop.exists():
                with stop.open('x') as stream:
                    stream.write('Preregistered fragment13 startup timeout; consumed wave, no retry.\n')
                    stream.flush(); os.fsync(stream.fileno())
                sync_directory(owner)
            decision = dict(utc=now(),intent_sha256=digest(intent),decision='stop_committed',
                            stage_changed=False,initialized=False,stop_written=True)
        write_new(receipt_root/f'startup_decision_stage{stage:02d}_wave{index:02d}.json',decision)
    return decision


def startup_watchdog(done, value):
    seen = set()
    try:
        while not done.wait(.5):
            if not (OWNER/'project_status.json').exists():
                continue
            status = read(OWNER/'project_status.json'); stage=status['stage']
            require(type(stage) is int and 0 <= stage < len(value['eligible_indices']), 'Unexpected fragment stage')
            index = value['eligible_indices'][stage]
            require(status['command'][-2:] == ['--output',str(TEST/f'wave{index:02d}')], 'Unexpected fragment command')
            wave = TEST/f'wave{index:02d}'
            begin = wave/'run_started.json'
            if not begin.exists() or stage in seen or (wave/'initialization.json').exists():
                continue
            started = read(begin)['utc']; now=time.time()
            require(type(started) in (int,float) and math.isfinite(started), 'Finite startup timestamp required')
            if now-started < STARTUP_SECONDS:
                continue
            seen.add(stage)
            decision = attempt_timeout(OWNER,TEST,OWNER,status)
            if decision['stop_written']:
                return
    except BaseException as error:
        write_new(OWNER/'startup_watchdog13_error.json',dict(utc=time.time(),error=repr(error)))
        with lifecycle_lock(OWNER):
            (OWNER/'STOP').touch(exist_ok=True)


def dispatch_fence(output, value):
    raw_output=Path(output)
    output=raw_output.resolve()
    require(raw_output==output and not raw_output.is_symlink(),'Wave path must be literal canonical original path')
    require(output.parent == TEST and output.name in {f'wave{i:02d}' for i in value['eligible_indices']},
            'Only reviewed never-launched original suffix waves may dispatch')
    index=int(output.name[4:])
    require(not output.is_symlink() and set(p.name for p in output.iterdir()) == {'plan.json','session.json'},
            'Wave has already launched or gained unexplained artifacts')
    freeze=read(TEST/'registration.json')
    require(digest(output/'plan.json') == freeze['wave_plan_sha256'][str(output)], 'Original wave plan changed')
    write_new(OWNER/'dispatches'/f'wave{index:02d}.json',dict(utc=time.time(),pid=os.getpid(),index=index,
              wave=str(output),plan_sha256=digest(output/'plan.json'),registration_sha256=digest(REGISTRY),
              dispatch_attempt=1,rerun=False))


def scoped_lease_factory(native, *, test, owner, deadline):
    def factory(root, deadline=None):
        require(Path(root).resolve() == Path(test).resolve() and deadline == DEADLINE,
                'Lease mapping escaped exact original test/deadline')
        return native(Path(owner),deadline=deadline)
    require(deadline == DEADLINE, 'Lease adapter finite deadline differs')
    return factory


def worker(output):
    value=validate_binding()
    dispatch_fence(output,value)
    output=Path(output).resolve()
    from tools import tight_state,tight_fresh_batch_run
    from tools.tight_paired_evaluate import worker as native_worker,validate_wave
    from visuotactile import state_resources
    import faulthandler
    plan,freeze=validate_wave(output)
    require(freeze['pairs']==value['frozen_pairs'] and freeze['cohort']=='test', 'Original frozen policies differ')
    old_environment=tight_state.environment
    old_lease=state_resources.lease_for
    old_atomic=tight_fresh_batch_run.atomic_json
    def environment(root,*args,**kwargs):
        require(Path(root).resolve()==output,'Diagnostics escaped exact wave')
        kwargs['startup_diagnostics']=True
        return old_environment(root,*args,**kwargs)
    stream=(output/'boot_signal_tracebacks13.log').open('x')
    faulthandler.register(signal.SIGUSR2,file=stream,all_threads=True,chain=False)
    tight_state.environment=environment
    state_resources.lease_for=scoped_lease_factory(old_lease,test=TEST,owner=OWNER,deadline=DEADLINE)
    tight_fresh_batch_run.atomic_json=locked_atomic(old_atomic,OWNER,output/'initialization.json')
    try:
        native_worker(output)
    finally:
        tight_state.environment=old_environment
        state_resources.lease_for=old_lease
        tight_fresh_batch_run.atomic_json=old_atomic
        stream.flush(); os.fsync(stream.fileno()); stream.close()
        # Native successful Kit execution legitimately calls os._exit.
        # There is no fabricated successful Python finalizer receipt.


def project():
    value=validate_binding()
    from tools import tight_project
    from visuotactile import training_resources
    require(sys.argv[2:] == ['evaluate','--job',str(JOB)], 'Exact finite fragment project command required')
    require(not (OWNER/'project_job_started.json').exists(), 'Attempted fragment owner cannot restart')
    # Recheck every original suffix wave before owner launch; each worker also
    # uses a durable exclusive fence before any native Kit/import work.
    for index in value['eligible_indices']:
        wave=TEST/f'wave{index:02d}'
        require(not wave.is_symlink() and wave.resolve()==wave
                and set(p.name for p in wave.iterdir())=={'plan.json','session.json'}, 'Original suffix changed')
    old=training_resources.atomic_json
    training_resources.atomic_json=locked_atomic(old,OWNER,OWNER/'project_status.json')
    tight_project.ENTRIES['paired_evaluate_fragment13']=('evaluate',MODULE,[])
    done=threading.Event(); monitor=threading.Thread(target=startup_watchdog,args=(done,value),name='startup-watchdog13')
    monitor.start()
    try:
        tight_project.execute(read(JOB),'evaluate')
    finally:
        done.set(); monitor.join(timeout=5)
        training_resources.atomic_json=old
        require(not monitor.is_alive(),'Owned startup watchdog13 did not join')


def driver():
    """One finite fragment and still-unattempted original post-eval audits."""
    sys.path.insert(0,str(HERE))
    import run_main_recovery05 as original
    import shared_capacity06 as capacity
    import psutil
    from shared_capacity13 import startup_row,AUDIT as watch_audit
    value=validate_binding()
    require(not OUT.exists() and not LOGS.exists(),'Attempted driver13 cannot restart')
    OUT.mkdir(); LOGS.mkdir()
    original.OUT=OUT
    class Driver(original.Driver):
        def gpu_gate(self,name):
            gpu,ram=(11264,16*2**30) if name=='evaluate_fragment13' else (1024,2*2**30)
            while True:
                self.finite()
                try:
                    row=read(watch_audit/'resource_watch_v4_status.json')
                    identity=startup_row(row,psutil.Process(row['pid']),time.time(),allow_gap=True)
                    decision=capacity.capacity_decision(capacity.live_capacity(),gpu_resident_mib=gpu,ram_resident_bytes=ram)
                    require(decision['ready'],'Actual capacity below original allowance plus reserve')
                    fresh=read(watch_audit/'resource_watch_v4_status.json')
                    startup_row(fresh,psutil.Process(fresh['pid']),time.time(),allow_gap=True)
                    self.write(OUT/(name+'_capacity_admission.json'),dict(utc=time.time(),watcher=identity,capacity=decision))
                    return
                except (OSError,KeyError,TypeError,ValueError,psutil.Error,__import__('subprocess').SubprocessError) as error:
                    self.note('waiting_actual_capacity_before_'+name,reason=repr(error)); time.sleep(5)
        def run(self,name,args,gpu):
            validate_binding()
            return super().run(name,args,gpu)
    instance=Driver(); instance.logs=LOGS; instance.source_identity=value['source_sha256']
    write_new(OUT/'registration.json',dict(value,activated_utc=time.time(),activated=True))
    try:
        instance.run('evaluate_fragment13',['-m',MODULE,'project','evaluate','--job',str(JOB)],True)
        for row in value['tail_stages']:
            if not row['skip_closed']:
                instance.run(row['name'],row['args'],True)
        write_new(OUT/'closure.json',dict(utc=time.time(),status='missing_original_suffix_and_unattempted_audits_closed',
                  original12_remains_failed=True,report_required=True))
    except BaseException as error:
        write_new(OUT/'failure.json',dict(utc=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        write_new(OUT/'driver_exit.json',dict(utc=time.time(),pid=os.getpid(),returned_after_owned_helpers=True))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='project':
        return project()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','worker','driver'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':
        print(json.dumps(prepare(),indent=2))
    elif args.mode=='worker':
        require(args.output is not None,'Explicit original wave output required')
        worker(args.output)
    else:
        def interrupted(signum,frame):
            raise KeyboardInterrupt(signum)
        for sig in (signal.SIGTERM,signal.SIGHUP):
            signal.signal(sig,interrupted)
        driver()


if __name__=='__main__':
    main()
