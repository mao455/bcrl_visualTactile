"""Inactive, finite software-exception cleanup aid for original waves59..63.

Import performs no filesystem/process lifecycle I/O. Activation requires a separate
root-written registration binding this source. No scientific results are read.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import time

PROJECT = Path('/home/CNS2025331827/docs/bc+rl_RSS')
AUDIT = PROJECT/'outputs/diagnostics/tight_paired_dagger_20261122_01'
ROOT = PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
OWNER = ROOT/'owners/test_continuation15'
TEST = ROOT/'evaluation/test'
REGISTRY = AUDIT/'TEST_CONTINUATION15_REGISTRATION.json'
REGISTRY_SHA = '9873b3fe5d14ada55c1510752550eef319241d042465654fa5c6d94a61817cee'
MODULE = 'outputs.diagnostics.tight_paired_dagger_20261122_01.run_test_continuation15'
PYTHON = '/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python'
DEADLINE = 1795684571.2261844
ELIGIBLE = tuple(range(59, 64))
MIN_STABLE = 120.
GRACE = 15.
OWNER_ID = (3319405, 1795489340.77)
DRIVER_ID = (3319358, 1795489339.69)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    # Never overwrite an earlier intent, signal record, failure, or exit.
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def complete_known_exception(text):
    """Only an already-emitted, complete terminal RAM-reserve traceback qualifies."""
    tail = text.rsplit('Traceback (most recent call last):', 1)
    return len(tail) == 2 and all(term in tail[1] for term in (
        'visuotactile/queued_rollout.py', 'with lease.block():',
        'visuotactile/state_resources.py', "raise RuntimeError('RAM reserve')")) \
        and tail[1].rstrip().endswith('RuntimeError: RAM reserve')


def fingerprint(index, birth, log_sha, progress_sha):
    return (index, birth, log_sha, progress_sha)


def stable_ready(observation, previous, now, monotonic_now=None):
    """Require both source timestamps and a fresh local120s unchanged observation."""
    monotonic_now = time.monotonic() if monotonic_now is None else monotonic_now
    return previous is not None and observation['fingerprint'] == previous['fingerprint'] \
        and monotonic_now-previous['first_seen_monotonic'] >= MIN_STABLE \
        and now-observation['log_mtime'] >= MIN_STABLE \
        and now-observation['progress_mtime'] >= MIN_STABLE \
        and now-observation['progress_utc'] >= MIN_STABLE


def validate_registration(path):
    row = read(path)
    require(row.get('version') == 'final_error_cleanup20_registration_v1', 'Explicit root registration required')
    require(row.get('source_sha256') == digest(__file__), 'Cleanup source changed')
    require(row.get('registry_sha256') == REGISTRY_SHA and digest(REGISTRY) == REGISTRY_SHA,
            'Original runtime registry changed')
    require(row.get('eligible_indices') == list(ELIGIBLE) and row.get('deadline') == DEADLINE,
            'Only original59..63 under original deadline')
    require(row.get('minimum_stable_seconds') == MIN_STABLE and row.get('term_grace_seconds') == GRACE,
            'Fixed exception-only stability/grace scheme required')
    require(row.get('owner_identity') == dict(pid=OWNER_ID[0], birth=OWNER_ID[1]) and
            row.get('driver_identity') == dict(pid=DRIVER_ID[0], birth=DRIVER_ID[1]),
            'Exact original owner/driver identity required')
    require(row.get('receipts') == str(AUDIT/'final_error_cleanup20_actual01'),
            'Single exclusive receipt directory required; registration cannot authorize a restart')
    require(row.get('activation_authorized') is True, 'Root activation registration required')
    return row


def validate_sources():
    require(digest(REGISTRY) == REGISTRY_SHA, 'Original registry changed')
    frozen = read(REGISTRY)
    require(frozen['deadline'] == DEADLINE, 'Original deadline changed')
    require(digest(AUDIT/'test_continuation15_job.json') == frozen['job_sha256']
            and digest(OWNER/'session.json') == frozen['owner_session_sha256']
            and digest(TEST/'registration.json') == frozen['original_test_registration_sha256'],
            'Frozen original job/session/test registration changed')
    for path, expected in frozen['orchestration_sha256'].items():
        require(digest(path) == expected, 'Frozen runtime source changed: '+path)
    return frozen


def same_process(proc, identity):
    return proc.pid == identity[0] and proc.create_time() == identity[1]


def group_census(proc, processes):
    """Reject every group member whose ancestry cannot be proven to exact worker."""
    import psutil
    pid, birth = proc.pid, proc.create_time()
    require(os.getpgid(pid) == pid and os.getsid(pid) == pid, 'Own worker must lead separate session/group')
    rows = []
    for member in processes:
        try:
            if os.getpgid(member.pid) != pid:
                continue
            require(os.getsid(member.pid) == pid, 'Different-session group member')
            if member.pid != pid:
                require(member.create_time() >= birth and any(same_process(p, (pid, birth)) for p in member.parents()),
                        'Unproven or foreign process group member')
            rows.append(dict(pid=member.pid, birth=member.create_time(), ppid=member.ppid(),
                             status=member.status(), command=[] if member.status() == psutil.STATUS_ZOMBIE else member.cmdline()))
        except (psutil.NoSuchProcess, ProcessLookupError):
            # A disappearing member never authorizes a signal. Fresh census is required.
            raise ValueError('Process group changed while inspecting')
    require(any(r['pid'] == pid and r['birth'] == birth for r in rows), 'Exact worker absent from own group')
    return sorted(rows, key=lambda r: r['pid'])


def observe(now):
    import psutil
    frozen = validate_sources()
    require(now < DEADLINE and not (OWNER/'STOP').exists(), 'Original stop/deadline active')
    owner = psutil.Process(OWNER_ID[0]); driver = psutil.Process(DRIVER_ID[0])
    require(same_process(owner, OWNER_ID) and same_process(driver, DRIVER_ID), 'Original owner/driver identity changed')
    require(owner.ppid() == driver.pid and owner.cwd() == str(PROJECT) and driver.cwd() == str(PROJECT),
            'Original owner ancestry/cwd changed')
    require(driver.cmdline() == [PYTHON, '-m', MODULE, 'driver'], 'Driver command changed')
    require(owner.cmdline() == [PYTHON, '-m', MODULE, 'project', 'evaluate', '--job', str(AUDIT/'test_continuation15_job.json')],
            'Owner command changed')
    state = read(OWNER/'project_status.json')
    stage = state['stage']; index = stage+17
    if index not in ELIGIBLE or (OWNER/'stage_exits'/f'wave{index:02d}.json').exists():
        return None
    wave = TEST/f'wave{index:02d}'
    expected = [PYTHON, '-m', MODULE, 'worker', '--output', str(wave)]
    launch_path = OWNER/'launches'/f'wave{index:02d}.json'
    dispatch_path = OWNER/'dispatches'/f'wave{index:02d}.json'
    if not dispatch_path.exists() or not (wave/'initialization.json').exists() or not (wave/'batch_progress.json').exists():
        return None
    launch, dispatch = read(launch_path), read(dispatch_path)
    require(state['command'] == expected and launch['command'] == expected and launch['stage'] == stage
            and launch['index'] == dispatch['index'] == index, 'Current exact original wave differs')
    require(launch['registration_sha256'] == dispatch['registration_sha256'] == REGISTRY_SHA
            and dispatch['launch_sha256'] == digest(launch_path) and dispatch['wave'] == str(wave)
            and dispatch['plan_sha256'] == digest(wave/'plan.json'), 'Immutable launch/dispatch binding changed')
    require(launch['dispatch_attempt'] == dispatch['dispatch_attempt'] == 1 and
            launch['rerun'] is dispatch['rerun'] is False, 'Consumed-wave retry prohibited')
    require(digest(wave/'session.json') == frozen['original_wave_session_sha256'][str(wave/'session.json')],
            'Original wave session changed')
    require(digest(wave/'plan.json') == read(TEST/'registration.json')['wave_plan_sha256'][str(wave)],
            'Original scientific plan changed')
    proc = psutil.Process(state['pid']); birth = proc.create_time()
    require(dispatch['pid'] == proc.pid and proc.ppid() == owner.pid and proc.cwd() == str(PROJECT)
            and proc.cmdline() == expected, 'Worker identity/ancestry/command differs')
    require(launch['utc']-.1 <= birth <= state['utc']+.1 and birth-.1 <= dispatch['utc'] <= state['utc']+120.,
            'PID birth outside immutable original launch window')
    log_path = OWNER/f'project_stage_{stage:02d}.log'
    text = log_path.read_text(errors='replace')
    if not complete_known_exception(text):
        return None
    progress_path = wave/'batch_progress.json'; progress = read(progress_path)
    require(type(progress['tick']) is int and progress['tick'] >= 0 and
            type(progress['utc']) in (int, float) and math.isfinite(progress['utc']) and
            birth <= progress['utc'] <= log_path.stat().st_mtime+.1 <= now+.2,
            'A prior real-progress event followed by terminal exception is required')
    census = group_census(proc, psutil.process_iter())
    return dict(index=index, stage=stage, pid=proc.pid, birth=birth, fingerprint=fingerprint(index, birth,
                digest(log_path), digest(progress_path)), log_mtime=log_path.stat().st_mtime,
                progress_mtime=progress_path.stat().st_mtime, progress_utc=progress['utc'],
                last_progress=progress, group=census, status_sha256=digest(OWNER/'project_status.json'),
                launch_sha256=digest(launch_path), dispatch_sha256=digest(dispatch_path),
                initialization_sha256=digest(wave/'initialization.json'))


@contextmanager
def schedule_lock():
    # Existing owner uses the same lock before dispatch; never create owner STOPs.
    with (ROOT/'continuation15_schedule.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def fresh_match(expected, previous, now):
    current = observe(now)
    require(current is not None and current['fingerprint'] == expected['fingerprint']
            and current['pid'] == expected['pid'] and current['birth'] == expected['birth']
            and stable_ready(current, previous, now), 'Fresh exception/identity/progress boundary no longer matches')
    return current


def cleanup_guard(out, registration_path, registration_sha256):
    require(not (out/'STOP').exists(), 'Helper-local stop active')
    require(digest(registration_path) == registration_sha256, 'Root aid registration changed')
    validate_registration(registration_path)


def cleanup(expected, previous, out, registration_sha256, registration_path):
    """Only function that signals. Root must separately review/register/activate."""
    import psutil
    index = expected['index']; prefix = out/f'wave{index:02d}'
    with schedule_lock():
        cleanup_guard(out, registration_path, registration_sha256)
        current = fresh_match(expected, previous, time.time())
        write_new(str(prefix)+'_intent.json', dict(utc=time.time(), registration_sha256=registration_sha256,
                    action='TERM exact owned group, fresh verification before KILL after15s', observation=current,
                    deadline=DEADLINE, startup_stop=False, global_stop=False, retry=False))
        signals = []
        try:
            # A second complete census immediately follows durable intent I/O.
            cleanup_guard(out, registration_path, registration_sha256)
            current = fresh_match(expected, previous, time.time())
            write_new(str(prefix)+'_term_intent.json', dict(utc=time.time(), signal='SIGTERM',
                        delivery_state='Intended; syscall not yet observed', observation=current,
                        registration_sha256=registration_sha256))
            cleanup_guard(out, registration_path, registration_sha256)
            current = fresh_match(expected, previous, time.time())
            os.killpg(current['pid'], signal.SIGTERM)
            signals.append(dict(utc=time.time(), signal='SIGTERM', observation=current))
            write_new(str(prefix)+'_term.json', signals[-1])
            until = time.monotonic()+GRACE
            while time.monotonic() < until:
                if time.time() >= DEADLINE or (OWNER/'STOP').exists() or (out/'STOP').exists():
                    break
                try:
                    proc = psutil.Process(expected['pid'])
                    if not same_process(proc, (expected['pid'], expected['birth'])) or proc.status() == psutil.STATUS_ZOMBIE:
                        break
                except psutil.NoSuchProcess:
                    break
                time.sleep(.2)
            # Never infer that another PID or next wave is ours to kill.
            try:
                proc = psutil.Process(expected['pid'])
                live = same_process(proc, (expected['pid'], expected['birth'])) and proc.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                live = False
            if live:
                require(time.monotonic() >= until, 'Grace interrupted: no KILL permitted')
                cleanup_guard(out, registration_path, registration_sha256)
                current = fresh_match(expected, previous, time.time())
                write_new(str(prefix)+'_kill_intent.json', dict(utc=time.time(), signal='SIGKILL',
                            delivery_state='Intended; syscall not yet observed', observation=current,
                            registration_sha256=registration_sha256))
                cleanup_guard(out, registration_path, registration_sha256)
                current = fresh_match(expected, previous, time.time())
                os.killpg(current['pid'], signal.SIGKILL)
                signals.append(dict(utc=time.time(), signal='SIGKILL', observation=current))
                write_new(str(prefix)+'_kill.json', signals[-1])
            write_new(str(prefix)+'_exit.json', dict(utc=time.time(), signals=signals,
                        signal_scope='Only freshly verified exact owned worker session/group',
                        native_owner_responsible_for_reaping=True, child_reaped_by_helper=False,
                        scientific_result_interpreted=False, retries=0, global_or_startup_stop_created=False))
        except BaseException as error:
            write_new(str(prefix)+'_failure.json', dict(utc=time.time(), error=repr(error), signals=signals,
                        note='No fabricated owner exit, startup STOP or global STOP; helper aborts'))
            raise


def registration_template():
    return dict(version='final_error_cleanup20_registration_v1', source_sha256=digest(__file__),
                registry_sha256=REGISTRY_SHA, eligible_indices=list(ELIGIBLE), deadline=DEADLINE,
                minimum_stable_seconds=MIN_STABLE, term_grace_seconds=GRACE,
                owner_identity=dict(pid=OWNER_ID[0], birth=OWNER_ID[1]),
                driver_identity=dict(pid=DRIVER_ID[0], birth=DRIVER_ID[1]),
                receipts=str(AUDIT/'final_error_cleanup20_actual01'), activation_authorized=False)


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--receipts', type=Path)
    parser.add_argument('--registration-template', action='store_true')
    args = parser.parse_args()
    if args.registration_template:
        print(json.dumps(registration_template(), indent=2)); return
    require(args.registration is not None and args.receipts is not None, 'Explicit root registration/receipts required')
    require(args.registration.resolve().parent == AUDIT and args.receipts.resolve().parent == AUDIT,
            'Run-local registration and new receipt directory only')
    validate_registration(args.registration)
    require(str(args.receipts.resolve()) == read(args.registration)['receipts'],
            'Exact single root-registered receipt directory required')
    require(time.time() < DEADLINE, 'Original deadline passed')
    args.receipts.mkdir()  # Exclusive new actual attempt; cannot resume/retry an old aid.
    out = args.receipts; registration_sha256 = digest(args.registration)
    write_new(out/'started.json', dict(utc=time.time(), pid=os.getpid(), registration_sha256=registration_sha256,
                source_sha256=digest(__file__), eligible_indices=list(ELIGIBLE), no_outcomes_read=True))
    previous = None; handled = set(); reason = None; error = None
    try:
        while time.time() < DEADLINE and not (out/'STOP').exists() and not (OWNER/'STOP').exists():
            require(digest(args.registration) == registration_sha256, 'Root aid registration changed')
            validate_registration(args.registration)
            if (OWNER/'project_complete.json').exists() or (OWNER/'project_exit.json').exists():
                reason = 'Original owner closed'; break
            try:
                observation = observe(time.time())
            except psutil.NoSuchProcess:
                # A naturally exited wave cannot authorize cleanup; retain no stale identity.
                observation = None
            if observation is None or observation['index'] in handled:
                previous = None
            else:
                if previous is None or observation['fingerprint'] != previous['fingerprint']:
                    previous = dict(fingerprint=observation['fingerprint'], first_seen=time.time(),
                                    first_seen_monotonic=time.monotonic())
                if stable_ready(observation, previous, time.time()):
                    cleanup(observation, previous, out, registration_sha256, args.registration)
                    handled.add(observation['index']); previous = None
            time.sleep(2.)
        reason = reason or 'Original deadline or stop marker reached'
    except BaseException as exc:
        error = repr(exc); reason = 'Helper error; no original queue mutation'
        raise
    finally:
        write_new(out/'exit.json', dict(utc=time.time(), reason=reason, error=error,
                    handled_indices=sorted(handled), deadline=DEADLINE, original_sources_modified=False,
                    global_or_startup_stop_created=False, retries=0, queue_complete_fabricated=False))


if __name__ == '__main__':
    main()
