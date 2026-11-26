"""Continue only original unstarted sealed64 after the final development startup failed.

Preparation freezes a lifecycle census without interpreting outcomes. Original10
remains failed. No completed collection, training or development wave is rerun.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import run_main_recovery05 as original
import run_backend_recovery10 as prior
import shared_capacity06 as capacity

PROJECT, ROOT, CONFIG = prior.PROJECT, prior.ROOT, prior.CONFIG
OUT = HERE/'driver_eval_recovery12'
LOGS = ROOT/'cycle_logs_recovery12'
REGISTRY = HERE/'EVALUATION_RECOVERY12_REGISTRATION.json'
REVIEW = HERE/'ROOT_EVALUATION_RECOVERY12_REVIEW.json'
MODULE = original.MODULE+'run_eval_recovery12'
WORKER_MODULE = original.MODULE+'eval_startup12'
WATCH_MODULE = original.MODULE+'shared_capacity12'
WATCH = HERE/'watch_shared_capacity12'
STARTUP_SECONDS = 1800


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def stages():
    remaining = list(prior.stages())
    first = next(i for i, (name, _, _) in enumerate(remaining) if name == 'summarize_development')
    for name, args, gpu in remaining[first:]:
        if name == 'evaluate_test':
            args = ['-m', WORKER_MODULE, 'project', 'evaluate', '--job', str(CONFIG/'evaluate_test.json')]
        yield name, args, gpu


def startup_expired(now, started, *, initialized, limit=STARTUP_SECONDS):
    if (type(now) not in (int, float) or type(started) not in (int, float)
            or not all(__import__('math').isfinite(v) for v in (now, started))):
        raise ValueError('Finite native startup timestamps required')
    return not initialized and now-started >= limit


def freeze_lifecycle():
    """No success/outcome values are interpreted here."""
    import psutil
    development = ROOT/'evaluation/development'
    registry = original.read(development/'registration.json')
    job = original.read(registry['job'])
    started = original.read(development/'project_job_started.json')
    failure = original.read(development/'project_failure.json')
    exit_row = original.read(development/'project_exit.json')
    driver_exit = original.read(prior.OUT/'driver_exit.json')
    original.require(exit_row['children_reaped'] is True and driver_exit['returned_after_owned_helpers'] is True,
                     'Original owner/driver helpers not closed')
    original.require(not psutil.pid_exists(started['pid']) and not psutil.pid_exists(driver_exit['pid']),
                     'Original owner/driver still exists')
    original.require((prior.OUT/'failure.json').is_file() and not (prior.OUT/'closure.json').exists(),
                     'Original10 failure must remain truthful')
    original.require(job == started['job'] and len(job['stages']) == registry['n'] == 8,
                     'Exact original8 development population required')
    original.require(original.digest(Path(registry['job'])) == registry['job_sha256'], 'Original job changed')
    rows = []
    for index in range(8):
        wave = development/f'wave{index:02d}'
        original.require(original.digest(wave/'plan.json') == registry['wave_plan_sha256'][str(wave)],
                         'Original development wave changed')
        original.require((development/f'project_stage_{index:02d}.log').is_file()
                         and (wave/'run_started.json').is_file(), 'All8 original waves must be consumed')
        terminal = (wave/'results.json').exists() and (wave/'process_exit_requested.json').exists()
        original.require(terminal == (index < 7), 'Expected closed7 / consumed final startup boundary differs')
        if index == 7:
            original.require(not (wave/'initialization.json').exists() and not list(wave.glob('*/started.json')),
                             'Final consumed wave advanced beyond registered startup failure')
        inventory = {}
        for path in sorted(wave.rglob('*')):
            if not path.is_file():
                continue
            original.require(not path.is_symlink(), 'Ambiguous symlink in consumed wave')
            stat = path.stat()
            row = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if path.suffix in ('.json', '.log') and stat.st_size <= 2**20:
                row['sha256'] = original.digest(path)
            inventory[str(path.relative_to(wave))] = row
        rows.append(dict(index=index, wave=str(wave), consumed=True, eligible_for_rerun=False,
                         terminal_receipts_present=terminal, physical_steps_claim='Unknown beyond durable records',
                         inventory=inventory))
    original.require(not (ROOT/'evaluation/test').exists() and not (LOGS).exists() and not OUT.exists(),
                     'Preserve any attempted test or recovery12 identity')
    original.require(not (ROOT/'cycle_logs_recovery10/prepare_test.log').exists(),
                     'Original10 may already have prepared/launched test')
    return dict(development_population=8, development_consumed=8, development_closed_wave_receipts=7,
                development_reruns=0, test_prepared=False, test_registered_families=64,
                test_launched=0, rows=rows, original_owner_failure_sha256=original.digest(development/'project_failure.json'),
                original_driver_failure_sha256=original.digest(prior.OUT/'failure.json'),
                lifecycle_only=True, outcome_values_interpreted=False)


def prepare():
    original.require(not REGISTRY.exists(), 'Preserve attempted12 registration')
    old = original.read(prior.REGISTRATION)
    from tools.tight_paired_dagger import sources
    original.require(sources() == old['source_sha256'], 'Frozen production295 identity changed')
    original.require(time.time() < old['original_deadline'] == original.read(ROOT/'session.json')['deadline'],
                     'Original finite deadline changed/expired')
    restoration = original.read(HERE/'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json')
    original.require(restoration['source_pid_reaped'] and restoration['conflicts'] == [], 'Watcher10 not restored')
    census = freeze_lifecycle()
    development = ROOT/'evaluation/development'
    freeze = original.read(development/'registration.json')
    pairs = freeze['pairs']
    for pair in pairs.values():
        for key in ('frontend', 'backend'):
            original.require(original.digest(pair[key]) == pair[key+'_sha256'], 'Frozen complete pair changed')
    names = ('run_eval_recovery12.py', 'eval_startup12.py', 'shared_capacity12.py',
             'run_backend_recovery10.py', 'run_main_recovery05.py', 'shared_capacity10.py',
             'shared_capacity06.py', 'resource_watch_v4.py', 'coexistence_watch_recovery05.py',
             'driver_support.py', 'driver_support_v4.py')
    value = dict(version='paired_unstarted_test_recovery12_v1', created_utc=time.time(),
                 original_deadline=old['original_deadline'], root=str(ROOT),
                 source_sha256=old['source_sha256'], frozen_pairs=pairs,
                 cohorts_sha256=original.digest(ROOT/'cohorts.json'), lifecycle=census,
                 original_registration_sha256=original.digest(prior.REGISTRATION),
                 development_registration_sha256=original.digest(development/'registration.json'),
                 restoration_sha256=original.digest(HERE/'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json'),
                 original_startup_failure_sha256=original.digest(HERE/'EVAL_STARTUP12_OWNED_TERMINATION.json'),
                 orchestration_sha256={str(HERE/name): original.digest(HERE/name) for name in names},
                 stages=[dict(name=name, args=args, gpu=gpu) for name, args, gpu in stages()],
                 startup_timeout_seconds=STARTUP_SECONDS,
                 timeout_scope='Before environment initialization only; original native owner STOP ends cohort; no retry or automatic suffix continuation',
                 scientific_changes=0, collection_retries=0, training_updates=0,
                 resource_policy='Actual measured capacity; external process presence never vetoes',
                 diagnostics_only='Existing StartupDiagnostics plus pre-environment SIGUSR2 handler; original8tasking/2physics backend unchanged')
    write_new(REGISTRY, value)
    return value


def validate(*, require_unstarted=False):
    from tools.tight_paired_dagger import sources
    value = original.read(REGISTRY)
    original.require(value['version'] == 'paired_unstarted_test_recovery12_v1', 'Wrong12 registry')
    original.require(sources() == value['source_sha256'], 'Frozen production295 changed')
    original.require(value['original_deadline'] == original.read(ROOT/'session.json')['deadline'] == 1795684571.2261844
                     and time.time() < value['original_deadline'], 'Finite session changed or expired')
    for path, expected in value['orchestration_sha256'].items():
        original.require(original.digest(path) == expected, 'Reviewed12 source changed: '+path)
    original.require(value['cohorts_sha256'] == original.digest(ROOT/'cohorts.json'), 'Original sealed cohort changed')
    original.require(value['development_registration_sha256'] == original.digest(ROOT/'evaluation/development/registration.json'),
                     'Frozen development registration changed')
    original.require(value['restoration_sha256'] == original.digest(HERE/'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json'),
                     'Watcher10 restoration evidence changed')
    original.require(value['stages'] == [dict(name=n, args=a, gpu=g) for n,a,g in stages()], 'Exact remaining stages changed')
    if require_unstarted:
        original.require(not OUT.exists() and not LOGS.exists() and not (ROOT/'evaluation/test').exists(),
                         'Preserve any attempted12/test identity')
    return value


def startup_watchdog(done, *, test_root, receipt_root, now=time.time, limit=STARTUP_SECONDS):
    """Touch only this native cohort STOP on a preregistered startup timeout."""
    while not done.wait(.5):
        status_path = test_root/'project_status.json'
        if not status_path.exists():
            continue
        try:
            status = original.read(status_path)
            index = status['stage']
            if type(index) is not int or not 0 <= index < 64:
                raise ValueError('Unexpected original test stage')
            wave = test_root/f'wave{index:02d}'
            begin = wave/'run_started.json'
            if not begin.exists():
                continue
            started = original.read(begin)['utc']
            if not startup_expired(now(), started, initialized=(wave/'initialization.json').exists(), limit=limit):
                continue
            # Recheck just before the exclusive stop. Native construction may
            # advance internal physics; no zero-physics-step claim is made.
            if (wave/'initialization.json').exists() or original.read(status_path) != status:
                continue
            write_new(receipt_root/f'startup_timeout_wave{index:02d}.json',
                      dict(utc=now(), index=index, worker_pid=status['pid'], wave=str(wave),
                           startup_started=started, startup_seconds=limit, entire_wave_consumed=True,
                           rerun=False, physical_steps_unknown=True))
            with (test_root/'STOP').open('x') as stream:
                stream.write('Preregistered recovery12 pre-control startup timeout; consumed wave, no retry.\n')
            return
        except FileNotFoundError:
            continue
        except BaseException as error:
            # Invalid lifecycle telemetry is terminal, never interpreted as
            # permission to launch another wave or ignore the watchdog.
            if not (receipt_root/'startup_watchdog_error.json').exists():
                write_new(receipt_root/'startup_watchdog_error.json', dict(utc=now(), error=repr(error)))
            if not (test_root/'STOP').exists():
                (test_root/'STOP').touch()
            return


class Driver(original.Driver):
    def gpu_gate(self, name):
        import psutil
        from shared_capacity12 import startup_row
        gpu, ram = (11264, 16*2**30) if name == 'evaluate_test' else (1024, 2*2**30)
        while True:
            self.finite()
            try:
                row = original.read(WATCH/'resource_watch_v4_status.json')
                identity = startup_row(row, psutil.Process(row['pid']), time.time(), allow_gap=True)
                decision = capacity.capacity_decision(capacity.live_capacity(), gpu_resident_mib=gpu, ram_resident_bytes=ram)
                original.require(decision['ready'], 'Actual capacity below original allowance plus reserve')
                fresh = original.read(WATCH/'resource_watch_v4_status.json')
                startup_row(fresh, psutil.Process(fresh['pid']), time.time(), allow_gap=True)
                self.write(OUT/(name+'_capacity_admission.json'), dict(utc=time.time(), watcher=identity, capacity=decision))
                return
            except (OSError, KeyError, TypeError, ValueError, psutil.Error, subprocess.SubprocessError) as error:
                self.note('waiting_actual_capacity_before_'+name, reason=repr(error))
                time.sleep(5)

    def run(self, name, args, gpu):
        validate()
        done = threading.Event()
        monitor = None
        if name == 'evaluate_test':
            monitor = threading.Thread(target=startup_watchdog, kwargs=dict(done=done,
                test_root=ROOT/'evaluation/test', receipt_root=OUT), name='own-startup-watchdog12')
            monitor.start()
        try:
            return super().run(name, args, gpu)
        finally:
            done.set()
            if monitor is not None:
                monitor.join(timeout=3)
                original.require(not monitor.is_alive(), 'Owned watchdog12 not joined')

    def main(self):
        value = validate(require_unstarted=True)
        review = original.read(REVIEW)
        original.require(review['status'] == 'source_and_actual_boundary_review_passed_inactive'
                         and review['registration_sha256'] == original.digest(REGISTRY), 'Root12 review required')
        OUT.mkdir(exist_ok=False)
        LOGS.mkdir(exist_ok=False)
        self.logs = LOGS
        self.source_identity = value['source_sha256']
        self.write(OUT/'registration.json', dict(value, activated=True, activated_utc=time.time()))
        for name, args, gpu in stages():
            self.run(name, args, gpu)
        self.note('remaining_evaluation_complete', original10_remains_failed=True, report_required=True)
        self.write(OUT/'closure.json', dict(utc=time.time(), original10_remains_failed=True,
                  status='original_sealed_test_and_remaining_summaries_complete', report_required=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        value = prepare()
        print(json.dumps(dict(registration=str(REGISTRY), development_consumed=value['lifecycle']['development_consumed'], test_launched=0)))
        return
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONFAULTHANDLER='1')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(signum)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    original.OUT = OUT
    driver = Driver()
    try:
        driver.main()
    except BaseException as error:
        if OUT.exists():
            driver.write(OUT/'failure.json', dict(utc=time.time(), error=repr(error), traceback=traceback.format_exc()))
            driver.note('stopped_with_evidence', error=repr(error))
        raise
    finally:
        if OUT.exists():
            driver.write(OUT/'driver_exit.json', dict(utc=time.time(), pid=os.getpid(), returned_after_owned_helpers=True))


if __name__ == '__main__':
    main()
