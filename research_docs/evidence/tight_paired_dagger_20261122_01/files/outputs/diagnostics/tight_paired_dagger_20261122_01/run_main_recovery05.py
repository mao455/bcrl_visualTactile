"""Unactivated finite main-round driver after truthful 15+81 collection closure.

No process launch or experiment mutation on import. Root owns activation.
DP/cache/offline remain serial; only reviewed rear trainers may overlap.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOT = PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
CONFIG = PROJECT/'configs/experiments/paired_dagger_20261122'
MODULE = 'outputs.diagnostics.tight_paired_dagger_20261122_01.'
WATCH = HERE/'watch_service_recovery05/resource_watch_v4_status.json'
OUT = HERE/'driver_main_recovery05'
OWNER_START_STAGES = frozenset(('cache_own_observations', 'train_four_components_parallel_rears',
    'evaluate_development', 'evaluate_test'))
STANDALONE_CUDA_STAGES = frozenset(('summarize_development', 'summarize_test',
    'audit_collection_cuda_noise_recovery05'))


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def storage_forecast(entries, free):
    lengths = [v['length'] for v in entries.values()]
    require(all(type(v) is int and v > 0 for v in lengths), 'Positive actual complete histories required')
    ticks = sum(lengths)
    features = math.ceil(ticks*1334200*1.06)
    required = features+47*2**30
    return dict(admitted_trajectories=len(entries), complete_history_ticks=ticks,
        estimated_feature_bytes=features, remaining_fixed_headroom_bytes=35*2**30,
        operational_reserve_bytes=12*2**30, required_free_bytes=required,
        actual_free_bytes=free, ready=free >= required, forecast=True)


def stages(root=ROOT, config=CONFIG, audit=HERE):
    """Ordered exact commands; no collection or historical rerun is included."""
    root, config, audit = map(Path, (root, config, audit))
    collection = root/'collection'
    yield 'validate_collection_recovery05', ['-m', MODULE+'collection_recovery05_scope', 'validate', '--collection', str(collection)], False
    yield 'audit_collection_integrity_recovery05', [str(audit/'audit_collection_recovery05.py'), 'both', '--collection', str(collection), '--output', str(audit/'collection_integrity_recovery05_actual01')], False
    yield 'storage_gate', [], False
    yield 'prepare_learning_recovery05', ['-m', MODULE+'collection_recovery05_scope', 'prepare-learning', '--collection', str(collection)], False
    yield 'preflight_learning', ['-m', 'tools.tight_paired_prepare', 'preflight-learning'], False
    yield 'register_learning_jobs', ['-m', 'tools.tight_paired_prepare', 'learning-jobs'], False
    yield 'cache_own_observations', ['-m', 'tools.tight_project', 'prepare', '--job', str(config/'main_cache.json')], True
    yield 'train_four_components_parallel_rears', ['-m', MODULE+'parallel_rear_proposal', 'project', 'train', '--job', str(audit/'proposed_main_train_parallel.json')], True
    yield 'verify_final_actual_training', [str(audit/'presealed_guard.py'), '--mode', 'main', '--output', str(audit/'presealed_guard_main_recovery05.json')], False
    for cohort in ('development', 'test'):
        yield 'prepare_'+cohort, ['-m', 'tools.tight_paired_evaluate', 'prepare', '--cohort', cohort], False
        yield 'evaluate_'+cohort, ['-m', 'tools.tight_paired_evaluate', 'project', 'evaluate', '--job', str(config/('evaluate_'+cohort+'.json'))], True
        yield 'summarize_'+cohort, ['-m', 'tools.tight_paired_evaluate', 'summarize', '--output', str(root/'evaluation'/cohort), '--reconstruct-cuda-noise'], True
    yield 'audit_collection_cuda_noise_recovery05', [str(audit/'audit_collection_recovery05.py'), 'noise', '--collection', str(collection), '--output', str(audit/'collection_noise_recovery05_actual01'), '--reconstruct-cuda-noise'], True


def watcher_ready(row, process, now, *, allow_owner_start=False, require_idle=False):
    age = now-float(row['utc'])
    require(math.isfinite(age) and -2 <= age < 10, 'Live watcher heartbeat is stale')
    require(row.get('errors') == [] and row.get('schedule_conflicts') == [],
            'Watcher has errors/conflicts')
    require(row.get('external_gpu_processes') == []
            and row.get('external_test_python_processes') == []
            and row.get('external_test_python_scan_errors') == [], 'Watcher reports external work or unknown scan')
    require(process.pid == row['pid'] and process.is_running(), 'Live watcher PID differs')
    command = process.cmdline()
    expected = ['-m', MODULE+'coexistence_watch_recovery05', '--audit', str(WATCH.parent), '--apply']
    require(command[1:] == expected and Path(process.cwd()).resolve() == PROJECT,
            'Exact reviewed recovery watcher command/cwd required')
    require(process.create_time() <= row['utc']+1, 'Watcher receipt predates live identity')
    idle = (row.get('active_resource_roots') == [] and row.get('raw_monitor_rows') == []
            and row.get('registered_workers_excluded') == [])
    gap = (row.get('paused') is True and row.get('reason') == 'no_fresh_owned_telemetry' and idle)
    normal = row.get('paused') is False and row.get('reason') == 'no_external_gpu'
    require(normal or ((allow_owner_start or require_idle) and gap),
            'Watcher pause is not the exact clean gap between owners')
    require(not require_idle or idle, 'Standalone CUDA requires no active owner or workers')
    return dict(starting_without_owned_telemetry=gap, watcher_remains_paused=gap,
                active_resource_roots=row['active_resource_roots'])


def standalone_snapshot():
    """Independent read-only queries; no owner telemetry or CUDA context is made."""
    import psutil
    from resource_watch_v4 import external_test_processes
    before = time.time()
    free = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=5).strip().splitlines()
    require(len(free) == 1, 'Exactly one selected GPU capacity result required')
    pids = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=5).strip().splitlines()
    processes = [int(value.strip()) for value in pids if value.strip()]
    scan = external_test_processes()
    return dict(query_started_utc=before, utc=time.time(), gpu_free_mib=float(free[0]),
        ram_available_bytes=psutil.virtual_memory().available,
        disk_free_bytes=shutil.disk_usage(ROOT).free, gpu_compute_pids=processes,
        external_test_scan=scan, source='independent nvidia-smi/proc queries; not owned telemetry')


def standalone_admission(row, session, now):
    """Reserve plus explicit small CUDA auditor residence, including its context."""
    age, duration = now-float(row['utc']), float(row['utc'])-float(row['query_started_utc'])
    require(math.isfinite(age) and -2 <= age < 3 and math.isfinite(duration) and 0 <= duration < 12,
            'Standalone independent capacity sample stale or slow')
    scan = row['external_test_scan']
    require(scan.get('processes') == [] and scan.get('errors') == [], 'Standalone external scan is busy or unknown')
    require(row.get('gpu_compute_pids') == [], 'Standalone CUDA requires no resident compute process')
    for key in ('gpu_free_mib','ram_available_bytes','disk_free_bytes'):
        value = row.get(key)
        require(not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and value >= 0,
                'Standalone capacity must be finite and nonnegative')
    resource = session['resources']
    configured = (resource['min_gpu_free_mib'],resource['min_ram_gib'],session['reserve_bytes'])
    require(all(not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and v > 0
                for v in configured), 'Standalone reserve configuration invalid')
    gpu = max(4096.,float(resource['min_gpu_free_mib']))+1024
    ram = (max(10.,float(resource['min_ram_gib']))+2)*2**30
    disk = max(12*2**30,session['reserve_bytes'])
    require(all(math.isfinite(v) and v > 0 for v in (gpu,ram,disk)), 'Standalone reserve configuration invalid')
    require(row['gpu_free_mib'] >= gpu and row['ram_available_bytes'] >= ram
            and row['disk_free_bytes'] >= disk, 'Standalone reserve plus resident capacity unavailable')
    return dict(gpu_required_free_mib=gpu, ram_required_available_bytes=ram, disk_required_free_bytes=disk,
        gpu_resident_allowance_mib=1024, ram_resident_allowance_bytes=2*2**30,
        owned_telemetry_claimed=False, independent_snapshot=row)


def validate_parallel():
    from parallel_rear_proposal import proposed_job, serial_job
    reg = read(HERE/'parallel_rear_registration.json')
    require(read(HERE/'proposed_main_train_parallel.json') == proposed_job(), 'Reviewed parallel job changed')
    require(read(CONFIG/'main_train.json') == serial_job() == reg['expected_serial_job'], 'Generated serial fallback differs')
    require(reg['wrapper_sha256'] == digest(HERE/'parallel_rear_proposal.py'), 'Parallel wrapper changed')
    for path, value in reg['source_sha256'].items():
        require(digest(path) == value, 'Parallel frozen source differs: '+path)
    require(reg['epochs'] == reg['deploy_epoch'] == 60, 'Reviewed sixty-epoch schedule required')


class Driver:
    def __init__(self):
        from visuotactile.training_resources import atomic_json
        self.write = atomic_json
        self.deadline = read(ROOT/'session.json')['deadline']
        self.stop = ROOT/'STOP_CYCLE'
        self.logs = ROOT/'cycle_logs'
        self.source_identity = None

    def note(self, stage, **extra):
        self.write(OUT/'status.json', dict(stage=stage, pid=os.getpid(), utc=time.time(), **extra))

    def finite(self):
        require(time.time() < self.deadline and not self.stop.exists(), 'Original finite deadline/own STOP reached')

    def storage_gate(self):
        entries = read(ROOT/'collection/correction_entries.json')
        while True:
            self.finite()
            row = dict(utc=time.time(), **storage_forecast(entries, shutil.disk_usage(ROOT).free))
            self.write(OUT/'storage_gate.json', row)
            if row['ready']:
                return
            self.note('waiting_verified_storage_capacity', storage=row)
            time.sleep(30)

    def gpu_gate(self, name):
        import psutil
        from driver_support_v4 import wait_external
        wait_external(self.deadline, self.stop, lambda scan:self.note('waiting_external_before_'+name, external_scan=scan))
        require(name in OWNER_START_STAGES | STANDALONE_CUDA_STAGES, 'Unknown GPU stage admission contract')
        standalone = name in STANDALONE_CUDA_STAGES
        while True:
            self.finite()
            try:
                row = read(WATCH)
                decision = watcher_ready(row, psutil.Process(row['pid']), time.time(),
                    allow_owner_start=not standalone, require_idle=standalone)
                if standalone:
                    capacity = standalone_admission(standalone_snapshot(), read(ROOT/'session.json'), time.time())
                    # Recheck heartbeat/external fields after the independent queries.
                    row = read(WATCH)
                    decision = watcher_ready(row, psutil.Process(row['pid']), time.time(), require_idle=True)
                    self.write(OUT/(name+'_resource_admission.json'), dict(utc=time.time(), watcher=decision,
                        watcher_status=row, capacity=capacity, context='Standalone native RNG audit; no owner telemetry created'))
                else:
                    self.write(OUT/(name+'_startup_admission.json'), dict(utc=time.time(), watcher=decision,
                        watcher_status=row, context='Only owner startup admitted; watcher alone restores its schedules after fresh monitor'))
                return
            except (OSError, KeyError, TypeError, ValueError, psutil.Error, subprocess.SubprocessError) as exc:
                self.note('waiting_reviewed_watcher_before_'+name, reason=repr(exc))
                time.sleep(5)

    def telemetry(self):
        path = ROOT/'learning/shared_resources.json'
        require(path.is_symlink(), 'Only an existing telemetry symlink may be rebound')
        old = os.readlink(path)
        target = ROOT/'owners/main_train/shared_resources.json'
        self.write(OUT/'training_telemetry_transition.json', dict(old=old, new=str(target), utc=time.time()))
        path.unlink()
        path.symlink_to(target)

    def run(self, name, args, gpu):
        from driver_support_v4 import run_owned
        from tools.tight_paired_dagger import sources
        self.finite()
        if self.source_identity is not None:
            require(sources() == self.source_identity, 'Production source changed after main registration')
        if gpu:
            self.gpu_gate(name)
        if name == 'cache_own_observations':
            validate_parallel()
        if name == 'train_four_components_parallel_rears':
            validate_parallel()
            self.telemetry()
            self.write(ROOT/'learning/training_source_identity.json', dict(utc=time.time(), sources=sources()))
        command = [sys.executable, *args]
        # Preserve the proven native evaluation CPU placement. Other stages
        # use the reviewed training partition and two BLAS threads.
        command = ['taskset', '-c', '8-23' if name.startswith('evaluate_') else '8-15', *command]
        self.note(name, command=command)
        with (self.logs/(name+'.log')).open('x') as stream:
            rc = run_owned(command, stream, self.deadline, self.stop)
        self.write(self.logs/(name+'_exit.json'), dict(returncode=rc, utc=time.time(), recovery='main_recovery05'))
        require(rc == 0, name+' failed; no retry or replacement is launched')

    def main(self):
        from tools.tight_paired_dagger import sources
        from collection_recovery05_scope import validated_summary
        require(not OUT.exists() and not (ROOT/'learning').exists(), 'Preserve any previous main-round attempt')
        require(read(HERE/'driver_collection_recovery05/exit.json')['returncode'] == 0,
                'The unassigned collection suffix must be closed first')
        closed = read(ROOT/'owners/collection_recovery05/project_exit.json')
        require(closed.get('children_reaped') is True, 'Collection owner has not reaped its children')
        summary = validated_summary(ROOT/'collection')
        require(summary['observable_pair_checks'] == dict(passed=47, unverified=1, failed=0, all_verified=False),
                'Truthful observed 47/1 collection registration required')
        ages = read(ROOT/'actual_parent_ages.json')['roles']
        for component in ('frontend', 'backend'):
            require(ages['A'][component+'_epochs'] == ages['B'][component+'_epochs'] == 60
                    and ages['A'][component+'_updates'] == ages['B'][component+'_updates'], 'Actual parent training age differs')
        OUT.mkdir(exist_ok=False)
        self.source_identity = sources()
        require(self.source_identity == read(ROOT/'common180_parents/training_source_identity.json')['sources'],
                'Production source differs from the matched actual parent training')
        snapshot = {str(HERE/p):digest(HERE/p) for p in ('run_main_recovery05.py', 'parallel_rear_proposal.py',
            'parallel_rear_registration.json', 'proposed_main_train_parallel.json', 'coexistence_watch_recovery05.py',
            'resource_watch_parallel_proposal.py', 'resource_watch_v4.py', 'driver_support_v4.py', 'driver_support.py',
            'collection_recovery05_scope.py', 'collection_recovery05.py', 'audit_collection_recovery05.py',
            'audit_collection_integrity.py', 'audit_collection_noise.py', 'presealed_guard.py', 'final_report_assembler.py')}
        self.write(OUT/'registration.json', dict(utc=time.time(), deadline=self.deadline, source_sha256=self.source_identity,
            orchestration_sha256=snapshot, stages=list(stages()), collection_summary=summary,
            scope='One own-data DAgger round; original96 opportunities and sealed64 unchanged; only rear training may overlap'))
        for name, args, gpu in stages():
            for path, value in snapshot.items():
                require(digest(path) == value, 'Registered orchestration source changed: '+path)
            if name == 'storage_gate':
                self.storage_gate()
            else:
                self.run(name, args, gpu)
        self.note('main_stages_complete', report_required=True, training_or_evaluation_reruns=0)


def main():
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(signum)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    driver = Driver()
    try:
        driver.main()
    except BaseException as exc:
        if OUT.exists():
            driver.write(OUT/'failure.json', dict(utc=time.time(), error=repr(exc), traceback=traceback.format_exc()))
            driver.note('stopped_with_evidence', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
