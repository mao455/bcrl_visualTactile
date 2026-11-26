"""INACTIVE finite host-initialization placement aid; no scientific payload I/O.

Activation requires separate exact root registration/review. Physics constructor
initialization can overlap placement; lifecycle13.lock fences the subsequent
Python initialization commit/control. No source, model, noise, rate or outcome
changes. Normal/error paths retain the lock until restoration or exact worker
exit. SIGKILL/host loss cannot be made safe by a Python finally/flock protocol.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import time
import psutil

B = Path('/home/CNS2025331827/docs/bc+rl_RSS')
A = B/'outputs/diagnostics/tight_paired_dagger_20261122_01'
R = B/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
O = R/'owners/test_continuation15'
TEST = R/'evaluation/test'
D = A/'host_init_capacity18_actual01'
REGISTRY = A/'HOST_INIT_CAPACITY18_REGISTRATION.json'
REVIEW = A/'ROOT_HOST_INIT_CAPACITY18_REVIEW.json'
REGISTRY15 = A/'TEST_CONTINUATION15_REGISTRATION.json'
REVIEW15 = A/'ROOT_TEST_CONTINUATION15_REVIEW.json'
ACTIVATED15 = A/'TEST_CONTINUATION15_ACTIVATED.json'
JOB = A/'test_continuation15_job.json'
MODULE = 'outputs.diagnostics.tight_paired_dagger_20261122_01.run_test_continuation15'
SELF_MODULE = 'outputs.diagnostics.tight_paired_dagger_20261122_01.host_init_capacity18'
PYTHON = '/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python'
DEADLINE = 1795684571.2261844
OLD = set(range(8, 24))
CANDIDATES = {1, 2, 4, 5}
ELIGIBLE = list(range(43, 64))
CAP = 40.
MIN_AGE = 60.
MAX_EXTRA = 2
STOP_REQUESTED = False
# Total busy, including any unobserved process, must initially be below15%.
BUSY_LIMIT = 15.
# Conservative global CPU delta of any intersecting external process; one
# quarter core is enough to retract. This may retract unnecessarily because
# affinity intersection is not attribution to a particular borrowed core.
EXTERNAL_LIMIT = .25
PINNED = {
    A/'run_test_continuation15.py': '2b99c3383172ed654c78dafe29074ffe52a9c9bf1768e0f1d69d4194a52aa192',
    A/'shared_capacity15.py': '18615cf1ed3d44e3779eaafd0c475d04816b9d29fc4d300a895651ccb9fb6c2d',
    A/'test_fragment13.py': '79e43e6f2496b0ccfbd40e200f1dcb53ca1d9c2579083ef27632c69997e03647',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(not path.is_symlink(), 'Symlinked lifecycle artifact')
    return json.loads(path.read_bytes())


def digest(path):
    path = Path(path)
    require(not path.is_symlink(), 'Symlinked pinned artifact')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, row):
    """Exclusive durable evidence; never replace an attempted wave."""
    path = Path(path)
    with path.open('x') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def stopped():
    return STOP_REQUESTED or (D/'STOP').exists() or (O/'STOP').exists() or time.time() >= DEADLINE


def source_guard():
    """Small source/lifecycle binding files only; no weight/cache/rollout read."""
    registration = read(REGISTRY)
    require(registration.get('version') == 'host_init_capacity18_v1'
            and registration.get('eligible_indices') == ELIGIBLE
            and registration.get('deadline') == DEADLINE
            and registration.get('audit') == str(D)
            and registration.get('candidate_cpus') == sorted(CANDIDATES)
            and registration.get('maximum_extra_cpus') == MAX_EXTRA
            and registration.get('cap_seconds') == CAP
            and registration.get('minimum_worker_age_seconds') == MIN_AGE
            and registration.get('physics_initialization_overlap_allowed') is True,
            'Exact finite capacity18 registration required')
    paths = registration.get('source_sha256', {})
    required = {str(Path(__file__).resolve()), *(str(p) for p in PINNED)}
    require(set(paths) == required, 'Exact source pin set required')
    for path, expected in paths.items():
        require(digest(path) == expected, 'Capacity18 source pin changed: '+path)
    for path, expected in PINNED.items():
        require(paths[str(path)] == expected, 'Original frozen source identity differs')
    artifacts = registration.get('binding_sha256', {})
    require(set(artifacts) == {str(p) for p in (REGISTRY15, REVIEW15, ACTIVATED15, JOB, TEST/'registration.json')},
            'Exact lifecycle binding pin set required')
    for path, expected in artifacts.items():
        require(digest(path) == expected, 'Capacity18 lifecycle pin changed: '+path)
    owner = registration.get('owner_identity', {})
    require(type(owner.get('pid')) is int and owner['pid'] > 0
            and type(owner.get('created')) in (int, float) and math.isfinite(owner['created'])
            and owner['created'] > 0, 'Exact registered owner18 PID/birth required')
    review = read(REVIEW)
    require(review.get('status') == 'inactive_root_reviewed_for_activation'
            and review.get('registration_sha256') == digest(REGISTRY), 'Independent root18 review required')
    return registration


def same(proc, birth):
    try:
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE and abs(proc.create_time()-birth) < .001
    except psutil.NoSuchProcess:
        return False


def process_row(pid):
    proc = psutil.Process(pid)
    with proc.oneshot():
        return dict(pid=proc.pid, created=proc.create_time(), ppid=proc.ppid(), command=proc.cmdline(), cwd=proc.cwd())


def descends(row, owner):
    seen = set()
    for _ in range(64):
        if row['pid'] in seen:
            return False
        seen.add(row['pid'])
        if row['pid'] == owner['pid']:
            return abs(row['created']-owner['created']) < .001
        try:
            row = process_row(row['ppid'])
        except psutil.NoSuchProcess:
            return False
    return False


def receipt_birth(birth, utc):
    if type(birth) not in (int, float) or type(utc) not in (int, float):
        return False
    if not math.isfinite(birth) or not math.isfinite(utc) or birth <= 0 or utc <= 0:
        return False
    age = utc-birth
    return -.05 <= age <= 120.


def identity(status, proc, birth, wave):
    """Bind independent durable receipts to live birth, ancestry and exact argv."""
    require(type(status.get('stage')) is int and 26 <= status['stage'] <= 46, 'Original43..63 only')
    index = status['stage']+17
    require(wave == TEST/f'wave{index:02d}' and wave.resolve() == wave and not wave.is_symlink(), 'Canonical eligible wave')
    require(type(status.get('pid')) is int and status['pid'] == proc.pid and type(birth) in (int, float)
            and math.isfinite(birth) and birth > 0,
            'Exact worker PID/birth required')
    expected = [PYTHON, '-m', MODULE, 'worker', '--output', str(wave)]
    require(same(proc, birth) and receipt_birth(birth, status['utc']) and status['command'] == expected
            and proc.cmdline() == expected and proc.cwd() == str(B) and proc.nice() == 10, 'Live worker/status identity differs')
    require(O.resolve() == O and not O.is_symlink() and TEST.resolve() == TEST and not TEST.is_symlink(), 'Canonical owner/test')
    binding, review, activated, job = map(read, (REGISTRY15, REVIEW15, ACTIVATED15, JOB))
    reg_sha = digest(REGISTRY15)
    require(review.get('status') == 'source_and_actual_missing47_review_passed_inactive'
            and review.get('registration_sha256') == reg_sha
            and activated.get('registry_sha256') == reg_sha
            and activated.get('root_review_sha256') == digest(REVIEW15), 'Actual owner15 root binding differs')
    require(binding['job_sha256'] == digest(JOB) and binding['original_test_registration_sha256'] == digest(TEST/'registration.json')
            and binding['eligible_indices'] == list(range(17, 64)), 'Original finite owner binding differs')
    started = read(O/'project_job_started.json')
    owner = process_row(started['pid'])
    owner_command = [PYTHON, '-m', MODULE, 'project', 'evaluate', '--job', str(JOB)]
    registered_owner = read(REGISTRY)['owner_identity']
    require(owner['pid'] == registered_owner['pid'] and abs(owner['created']-registered_owner['created']) < .001
            and receipt_birth(owner['created'], started['utc']) and owner['command'] == owner_command
            and owner['cwd'] == str(B) and descends(process_row(proc.pid), owner), 'Live exact owner ancestry differs')
    require(started['job'] == job and job['resource_root'] == str(O) and job['operation'] == 'evaluate'
            and job['stages'][status['stage']] == dict(entry='paired_evaluate_continuation15', args=expected[3:])
            and started['commands'][status['stage']] == expected, 'Owner job/stage command differs')
    launch_path = O/'launches'/f'wave{index:02d}.json'
    launch, fence = read(launch_path), read(O/'dispatches'/f'wave{index:02d}.json')
    require(launch['stage'] == status['stage'] and launch['index'] == index and launch['command'] == expected
            and launch['registration_sha256'] == reg_sha and launch['dispatch_attempt'] == 1 and launch['rerun'] is False
            and type(launch['utc']) in (int, float) and math.isfinite(launch['utc'])
            and -.05 <= birth-launch['utc'] <= 120., 'Exact launch/birth fence differs')
    freeze, plan = read(TEST/'registration.json'), read(wave/'plan.json')
    plan_sha = digest(wave/'plan.json')
    require(plan_sha == freeze['wave_plan_sha256'][str(wave)] and plan['pairs'] == binding['frozen_pairs']
            and plan['resource_root'] == str(TEST), 'Frozen wave plan differs')
    require(fence['pid'] == proc.pid and fence['index'] == index and fence['wave'] == str(wave)
            and fence['dispatch_attempt'] == 1 and fence['rerun'] is False
            and fence['registration_sha256'] == reg_sha
            and fence['original13_registration_sha256'] == binding['original13_registry_sha256']
            and fence['launch_sha256'] == digest(launch_path) and fence['plan_sha256'] == plan_sha
            and receipt_birth(birth, fence['utc']), 'Exclusive birth-bound dispatch fence differs')
    require(read(O/'project_status.json') == status, 'Owner stage/status changed')
    return dict(index=index, owner_pid=owner['pid'], owner_created=owner['created'], plan_sha256=plan_sha,
                launch_sha256=digest(launch_path), dispatch_sha256=digest(O/'dispatches'/f'wave{index:02d}.json'))


def thread_token(tid, proc, birth):
    """Linux Tgid + task starttick, rechecked immediately around each syscall."""
    if not same(proc, birth):
        return None
    try:
        fields = dict(line.split(':', 1) for line in Path(f'/proc/{tid}/status').read_text().splitlines() if ':' in line)
        stat = Path(f'/proc/{tid}/stat').read_text().rsplit(')', 1)[1].split()
        if int(fields['Tgid']) != proc.pid:
            return None
        return (int(tid), int(stat[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def masks(proc, birth):
    result = []
    for thread in proc.threads():
        token = thread_token(thread.id, proc, birth)
        if token is None:
            continue
        try:
            mask = set(os.sched_getaffinity(thread.id))
            nice = os.getpriority(os.PRIO_PROCESS, thread.id)
            if thread_token(thread.id, proc, birth) == token:
                result.append((token, mask, nice))
        except ProcessLookupError:
            pass
    return result


def set_mask(token, proc, birth, cpus, row, *, restore=False):
    if thread_token(token[0], proc, birth) != token:
        return False
    require(os.getpriority(os.PRIO_PROCESS, token[0]) == 10, 'Thread nice changed; never modify priority')
    # Mark before the syscall so partial failure still enters restoration.
    row['mutation_attempted'] = True
    os.sched_setaffinity(token[0], cpus)
    row['affinity_writes'] += 1
    if thread_token(token[0], proc, birth) != token:
        return False
    require(set(os.sched_getaffinity(token[0])) == set(cpus), 'Affinity write verification failed')
    row['restoration_writes' if restore else 'expansion_writes'] += 1
    return True


def ready(wave):
    return (wave/'initialization.json').exists() or read(wave/'environment_initialization.json').get('current_stage') == 'environment_ready'


def constructor_pending(wave):
    return not ready(wave) and read(wave/'environment_initialization.json').get('current_stage') == 'environment_constructor_started'


def cpu_snapshot():
    return psutil.cpu_times(percpu=True)


def busy_between(before, after):
    result = {}
    require(len(before) == len(after) and len(after) > max(CANDIDATES), 'CPU topology changed')
    for cpu in sorted(CANDIDATES):
        fields = before[cpu]._fields
        delta = {key: float(getattr(after[cpu], key))-float(getattr(before[cpu], key)) for key in fields}
        require(all(math.isfinite(v) and v >= 0 for v in delta.values()), 'Invalid CPU counter delta')
        total = sum(v for k, v in delta.items() if k not in ('guest', 'guest_nice'))
        require(total > 0, 'No CPU sample progress')
        idle = delta.get('idle', 0)+delta.get('iowait', 0)
        result[cpu] = 100.*(total-idle)/total
    return result


def spare_samples(interval=1.5, count=2):
    rows = []
    before = cpu_snapshot()
    for _ in range(count):
        if stopped():
            return [], []
        time.sleep(interval)
        after = cpu_snapshot()
        rows.append(busy_between(before, after))
        before = after
    eligible = [cpu for cpu in sorted(CANDIDATES) if all(row[cpu] < BUSY_LIMIT for row in rows)]
    eligible.sort(key=lambda cpu: (max(row[cpu] for row in rows), cpu))
    return eligible[:MAX_EXTRA], rows


def external_snapshot(proc, birth, cpus):
    result = {}
    for other in psutil.process_iter(['pid']):
        try:
            with other.oneshot():
                other_birth = other.create_time()
                if (other.pid == proc.pid and abs(other_birth-birth) < .001) or other.pid == os.getpid():
                    continue
                if not set(other.cpu_affinity()).intersection(cpus):
                    continue
                values = other.cpu_times()
                result[(other.pid, other_birth)] = values.user+values.system
        except psutil.NoSuchProcess:
            continue
    return result


def external_busy(before, after, elapsed):
    require(elapsed > 0, 'External sample clock did not advance')
    # Newly observed intersecting processes retract, including reused PIDs.
    if set(after)-set(before):
        return True
    deltas = [after[key]-before[key] for key in set(before).intersection(after)]
    require(all(math.isfinite(delta) and delta >= 0 for delta in deltas), 'External CPU counter changed')
    return sum(deltas)/elapsed > EXTERNAL_LIMIT


@contextmanager
def lifecycle_lock():
    require(not (O/'lifecycle13.lock').is_symlink(), 'Unsafe lifecycle lock')
    with (O/'lifecycle13.lock').open('a') as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if stopped():
                    raise InterruptedError('Stopped while waiting for lifecycle lock')
                time.sleep(.1)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def restore(proc, birth, wide, row):
    """Also restore threads born after expansion, before the lock is released."""
    for _ in range(20):
        if not same(proc, birth):
            row['worker_exited'] = True
            row['restored'] = None
            return
        observed = masks(proc, birth)
        require(observed, 'Live worker has no verifiable threads')
        for token, mask, nice in observed:
            require(nice == 10 and OLD <= mask <= wide, 'Unexpected thread affinity/priority during restore')
            if mask != OLD:
                set_mask(token, proc, birth, OLD, row, restore=True)
        current = masks(proc, birth)
        if current and all(mask == OLD and nice == 10 for _, mask, nice in current):
            row['restored'] = True
            row['restored_thread_count'] = len(current)
            row['restored_masks'] = [sorted(OLD)]
            return
    raise ValueError('Failed original all-thread restoration')


def wait_failed_restore(proc, birth, row):
    """No evidence/STOP failure may release the lock around a live worker.

    Native owner observes STOP without taking lifecycle13.lock. If filesystem
    writes themselves fail, retry them while retaining the lock. The unchanged
    owner also independently checks its original absolute deadline. This can
    remain waiting and must never be presented as successful helper closure.
    """
    evidence_errors = []
    stop_committed = abort_written = False
    while True:
        try:
            try:
                fd = os.open(O/'STOP', os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                sync_directory(O)
                stop_committed = True
            except BaseException as exc:
                evidence_errors.append(repr(exc))
            if not abort_written:
                try:
                    write(D/(Path(row['wave']).name+'_infrastructure_abort.json'), dict(row, infrastructure_abort=True,
                          queue_completion_claimed=False, owner_stop_committed=stop_committed,
                          lock_retained_until_birth_bound_worker_exit=True, evidence_errors=evidence_errors[-20:]))
                    abort_written = True
                except BaseException as exc:
                    evidence_errors.append(repr(exc))
            if not same(proc, birth):
                break
            time.sleep(.2)
        except BaseException as exc:
            # Catch identity inspection and interruptions across the entire
            # iteration. Uncertainty never authorizes releasing this lock.
            evidence_errors.append(repr(exc))
            try:
                time.sleep(.2)
            except BaseException:
                pass
    # Target is now gone: a final receipt failure can safely propagate.
    write(D/(Path(row['wave']).name+'_abort_worker_exited.json'), dict(utc=time.time(), pid=proc.pid,
          created=birth, owner_stop_committed=stop_committed, evidence_errors=evidence_errors[-20:],
          actual_placement_elapsed_seconds=(time.monotonic()-row['mutation_start_monotonic']
              if 'mutation_start_monotonic' in row else None),
          lock_releasable_after_exact_worker_exit=True, signals_sent=0))


def probe(status):
    """One durable attempt/wave; all process and affinity functions mockable."""
    wave = TEST/f'wave{status["stage"]+17:02d}'
    if wave.name not in {f'wave{i:02d}' for i in ELIGIBLE} or stopped():
        return None
    proc = psutil.Process(status['pid'])
    birth = proc.create_time()
    identity(status, proc, birth, wave)
    if time.time()-status['utc'] < MIN_AGE or ready(wave):
        return None
    if read(wave/'environment_initialization.json').get('current_stage') != 'environment_constructor_started':
        return None
    chosen, pre_samples = spare_samples()
    if not chosen:
        return None
    with lifecycle_lock():
        row = dict(utc=time.time(), pid=proc.pid, created=birth, wave=str(wave), stage=status['stage'],
                   old=sorted(OLD), chosen=[], cap_seconds=CAP, mutation_attempted=False,
                   affinity_writes=0, expansion_writes=0, restoration_writes=0, restored=None,
                   signals_sent=0, external_mutations=0, physics_initialization_overlap_allowed=True,
                   no_python_control_commit_during_extension=True, pre_capacity_samples=pre_samples)
        write(D/(wave.name+'_intent.json'), row)
        error = restore_error = None
        try:
            source_guard()
            row['binding'] = identity(status, proc, birth, wave)
            if stopped() or not constructor_pending(wave):
                row['no_change_reason'] = 'stop_or_readiness_race'
            else:
                fresh, samples = spare_samples(interval=.25, count=1)
                row['locked_capacity_samples'] = samples
                chosen = [cpu for cpu in chosen if cpu in fresh]
                require(len(chosen) <= MAX_EXTRA and all(type(cpu) is int and cpu in CANDIDATES for cpu in chosen),
                        'Invalid borrowed CPU whitelist/budget')
                if not chosen or stopped() or not constructor_pending(wave):
                    row['no_change_reason'] = 'capacity_stop_or_readiness_race'
                else:
                    source_guard()
                    identity(status, proc, birth, wave)
                    observed = masks(proc, birth)
                    require(observed and all(mask == OLD and nice == 10 for _, mask, nice in observed), 'Original all-thread8..23/nice10 required')
                    row['chosen'] = chosen
                    row['original_threads'] = [dict(tid=token[0], starttick=token[1], mask=sorted(mask), nice=nice)
                                               for token, mask, nice in observed]
                    wide = OLD | set(chosen)
                    write(D/(wave.name+'_expansion_intent.json'), row)
                    external = external_snapshot(proc, birth, chosen)
                    previous_sample = time.monotonic()
                    source_guard()
                    identity(status, proc, birth, wave)
                    # Cap includes per-thread enumeration/writes, not just wait.
                    row['mutation_start_utc'] = time.time()
                    started = time.monotonic()
                    row['mutation_start_monotonic'] = started
                    for token, _, _ in observed:
                        if time.monotonic()-started >= CAP or stopped() or not constructor_pending(wave):
                            break
                        set_mask(token, proc, birth, wide, row)
                    while row['mutation_attempted'] and same(proc, birth) and time.monotonic()-started < CAP and not stopped() and constructor_pending(wave):
                        time.sleep(min(.25, max(0., CAP-(time.monotonic()-started))))
                        now = time.monotonic()
                        current = external_snapshot(proc, birth, chosen)
                        if external_busy(external, current, now-previous_sample):
                            row['retracted_for_external_cpu_activity'] = True
                            break
                        external, previous_sample = current, now
        except BaseException as exc:
            error = exc
            row['error'] = repr(exc)
        finally:
            try:
                if row['mutation_attempted']:
                    restore(proc, birth, OLD | set(row['chosen']), row)
                    row['mutation_end_utc'] = time.time()
                else:
                    row['no_affinity_writes'] = row['affinity_writes'] == 0
            except BaseException as exc:
                restore_error = exc
                row['restore_error'] = repr(exc)
                row['restored'] = False
            finally:
                row['ended'] = time.time()
                if row['mutation_attempted']:
                    row['actual_placement_elapsed_seconds'] = time.monotonic()-row['mutation_start_monotonic']
                    row['placement_end_verified'] = row.get('restored') is True or row.get('worker_exited') is True
                row['infrastructure_abort'] = restore_error is not None
                # This receipt is attempted even when restore/verification fails.
                try:
                    write(D/(wave.name+'_exit.json'), row)
                finally:
                    if restore_error is not None:
                        wait_failed_restore(proc, birth, row)
        if restore_error is not None:
            raise RuntimeError('Restore failed; global abort requested and exact worker exited; inspect owner_stop_committed') from restore_error
        if error is not None:
            raise error
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    require(args.apply, 'Inactive source; explicit reviewed --apply required')
    require(Path.cwd().resolve() == B, 'Canonical cwd required')
    source_guard()
    require(time.time() < DEADLINE and not D.exists(), 'Finite unique helper identity required')
    D.mkdir()
    me = psutil.Process()
    write(D/'started.json', dict(pid=os.getpid(), created=me.create_time(), utc=time.time(), deadline=DEADLINE,
          command=me.cmdline(), cwd=me.cwd(), registration_sha256=digest(REGISTRY), indices=ELIGIBLE,
          borrowed_cpus=sorted(CANDIDATES), max_extra=MAX_EXTRA, external_mutations=0, signals_sent=0))
    def interrupt(signum, frame):
        # Cooperative signal stop never asynchronously interrupts restoration.
        global STOP_REQUESTED
        STOP_REQUESTED = True
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    attempted = []
    error = None
    try:
        while not stopped():
            if (A/'driver_test_continuation15/driver_exit.json').exists():
                break
            status = read(O/'project_status.json')
            index = status['stage']+17
            if index > ELIGIBLE[-1]:
                break
            name = f'wave{index:02d}'
            if index in ELIGIBLE and name not in attempted and not (D/(name+'_intent.json')).exists():
                try:
                    row = probe(status)
                except (FileNotFoundError, psutil.NoSuchProcess):
                    # A pre-intent dispatch race can be ignored. Any durable
                    # attempt with an error ends this helper, even if restored.
                    if (D/(name+'_intent.json')).exists():
                        raise
                    row = None
                if row is not None:
                    attempted.append(name)
            time.sleep(.5)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        write(D/'exit.json', dict(pid=os.getpid(), created=me.create_time(), utc=time.time(), attempted=attempted,
              durable_attempts=sorted(p.name for p in D.glob('wave*_intent.json')), error=error,
              signals_sent=0, external_mutations=0, scientific_payloads_read=False,
              original_startup_timeout_receipts_untouched=True))


if __name__ == '__main__':
    main()
