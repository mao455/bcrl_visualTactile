"""Run-local cooperative GPU yielding. No process signals or production patches.

Dry-run by default. --apply is required to write schedules. Launch only after the
old watcher has exited/restored its schedules; do not run both writers together.
This module imports no torch, starts no GPU process, and never rewrites telemetry.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import psutil

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
DEFAULT_CONFIG = PROJECT/'configs/experiments/paired_dagger_20261122'
PAUSES = (None, 'A', 'B', 'G00', 'G10', 'G01', 'G11')
MAX_AGE = 8.
ADMISSION_AGE = 3.  # Early cooperative pause; production's hard 8s gate remains.


def read(path):
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object: '+str(path))
    return value


def sha(content):
    return hashlib.sha256(content).hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    temporary = path.with_name(path.name+f'.watch-{os.getpid()}-{time.time_ns()}')
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, allow_nan=False).encode())


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def resolved(value, cwd):
    path = Path(value)
    return (path if path.is_absolute() else Path(cwd)/path).resolve()


def process(pid):
    """Read a live process identity; uncertainty grants no ownership."""
    try:
        proc = psutil.Process(int(pid))
        with proc.oneshot():
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                return None
            return dict(pid=proc.pid, ppid=proc.ppid(), created=proc.create_time(),
                        command=proc.cmdline(), cwd=proc.cwd())
    except (psutil.Error, OSError, ValueError, TypeError):
        return None


def matching_command(actual, expected, cwd):
    """Exact argv, except executable spelling and canonical path option values."""
    if not isinstance(actual, list) or not isinstance(expected, list) or len(actual) != len(expected):
        return False
    if not actual or not Path(actual[0]).name.startswith('python'):
        return False
    if not Path(expected[0]).name.startswith('python'):
        return False
    for i, (left, right) in enumerate(zip(actual[1:], expected[1:]), 1):
        if actual[i-1] in ('--job', '--root', '--output'):
            if resolved(left, cwd) != Path(right).resolve():
                return False
        elif left != right:
            return False
    return True


def born_before_receipt(proc, utc, *, slack=120.):
    try:
        age = float(utc)-float(proc['created'])
        return math.isfinite(age) and -2 <= age <= slack
    except (KeyError, TypeError, ValueError):
        return False


def descends(proc, owner, lookup=process):
    """Live parent chain plus owner create-time, not just a reused PID."""
    seen = set()
    current = proc
    for _ in range(64):
        if current is None or current['pid'] in seen:
            return False
        seen.add(current['pid'])
        if current['pid'] == owner['pid']:
            return abs(current['created']-owner['created']) < .001
        current = lookup(current.get('ppid', 0))
    return False


def specs(root, config):
    """Only these exact run-local monitor/job roots may participate."""
    root, config = Path(root).resolve(), Path(config).resolve()
    rows = []
    for name, module, operation in (
        ('experts', 'tools.tight_project', 'prepare'),
        ('experts_admission01', 'tools.tight_project', 'prepare'),
        ('shared_demo_cache', 'tools.tight_project', 'prepare'),
        ('parent_train', 'tools.tight_paired_learning', 'train'),
        ('parent_train_recovery01', 'tools.tight_paired_learning', 'train'),
        ('main_cache', 'tools.tight_project', 'prepare'),
        ('main_train', 'tools.tight_paired_learning', 'train'),
    ):
        rows.append(dict(name=name, owner=root/'owners'/name,
                         job=config/(name+'.json'), module=module, operation=operation))
    rows.append(dict(name='collection', owner=root/'owners/collection',
                     job=root/'collection_job.json', module='tools.tight_paired_collection', operation='dagger'))
    for cohort in ('development', 'test'):
        rows.append(dict(name=cohort, owner=root/'evaluation'/cohort,
                         job=config/f'evaluate_{cohort}.json', module='tools.tight_paired_evaluate',
                         operation='evaluate'))
    return rows


def active_registration(spec, root, lookup=process):
    """Bind the configured job to its live owner and immutable start receipt."""
    owner = Path(spec['owner'])
    if not inside(owner, root) or owner.is_symlink():
        raise ValueError('Owner root outside registered run')
    receipt = read(owner/'project_job_started.json')
    # Establish live identity before validating historical files. A closed root
    # whose PID was recycled cannot make unrelated current telemetry fail.
    proc = live_owner(spec, receipt, lookup)
    if proc is None:
        return None
    job = read(spec['job'])
    if (job.get('version') != 'tight_project_job_v1' or job.get('operation') != spec['operation']
            or Path(job.get('resource_root', '/')).resolve() != owner.resolve()):
        raise ValueError('Exact job/resource root mismatch')
    if receipt.get('job') != job:
        raise ValueError('Current job differs from started job receipt')
    return dict(spec=spec, job=job, receipt=receipt, owner_process=proc)


def live_owner(spec, receipt, lookup=process):
    proc = lookup(receipt.get('pid'))
    expected = ['python', '-m', spec['module']]
    if spec['module'] != 'tools.tight_project':
        expected.append('project')
    expected.extend([spec['operation'], '--job', str(spec['job'])])
    if not proc or not born_before_receipt(proc, receipt.get('utc')):
        return None
    if not matching_command(proc['command'], expected, proc['cwd']):
        return None
    return proc


def external_test_processes(records=None, *, self_pid=None):
    """Independent read-only detection, including CPU-only external Python tests.

    This is only a pause predicate. It grants no authority to signal anything.
    Synthetic records avoid all process enumeration in unit tests.
    """
    self_pid = os.getpid() if self_pid is None else self_pid
    found, errors = [], []
    if records is None:
        records = []
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.pid == self_pid:
                    continue
                try:
                    if 'python' not in (proc.info.get('name') or '').lower():
                        continue
                    with proc.oneshot():
                        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                            continue
                        records.append(dict(pid=proc.pid, command=proc.cmdline(),
                                            cwd=proc.cwd(), created=proc.create_time()))
                except psutil.NoSuchProcess:
                    continue
                except (psutil.Error, OSError) as exc:
                    errors.append(dict(pid=proc.pid, error=repr(exc)))
        except (psutil.Error, OSError) as exc:
            errors.append(dict(error=repr(exc)))
    for record in records:
        try:
            if record['pid'] == self_pid:
                continue
            argv = record['command']
            if not argv or 'python' not in Path(argv[0]).name.lower():
                continue
            if any('FingerEyeLab' in argument for argument in argv) or 'FingerEyeLab' in record['cwd']:
                found.append(dict(record))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(dict(error=repr(exc)))
    return dict(processes=found, errors=errors)


def monitor_row(registration, now, lookup=process):
    owner = Path(registration['spec']['owner'])
    mon = read(owner/'monitor.json')
    proc = lookup(mon.get('pid'))
    expected = ['python', '-m', 'visuotactile.state_resources', '--root', str(owner)]
    if (not proc or not born_before_receipt(proc, mon.get('started'))
            or not descends(proc, registration['owner_process'], lookup)
            or not matching_command(proc['command'], expected, proc['cwd'])):
        raise ValueError('Monitor is not a live registered-owner descendant')
    row = read(owner/'shared_resources.json')
    age = now-float(row.get('utc_epoch', 0))
    if not math.isfinite(age) or not -2 <= age < MAX_AGE or row.get('pid') != proc['pid']:
        raise ValueError('Missing, stale, future or mismatched monitor telemetry')
    return row


def registered_worker(pid, registrations, root, lookup=process):
    """Exclude only exact current collection/eval workers with bound plan SHA."""
    proc = lookup(pid)
    if not proc:
        return None
    for reg in registrations:
        spec = reg['spec']
        if spec['name'] not in ('collection', 'development', 'test'):
            continue
        try:
            status = read(Path(spec['owner'])/'project_status.json')
            index = status['stage']
            if type(index) is not int or index < 0:
                continue
            stage = reg['job']['stages'][index]
            expected_entry = 'paired_collection' if spec['name'] == 'collection' else 'paired_evaluate'
            prefix = [] if spec['name'] == 'collection' else ['worker']
            args = stage['args']
            if stage.get('entry') != expected_entry or len(args) != len(prefix)+2 or args[:len(prefix)] != prefix:
                continue
            if args[len(prefix)] != '--output':
                continue
            output = Path(args[-1]).resolve()
            if not inside(output, root):
                continue
            expected = ['python', '-m', spec['module'], 'worker', '--output', str(output)]
            if (status.get('pid') != pid or not born_before_receipt(proc, status.get('utc'))
                    or not descends(proc, reg['owner_process'], lookup)
                    or not matching_command(proc['command'], expected, proc['cwd'])
                    or not matching_command(status['command'], expected, proc['cwd'])
                    or not matching_command(reg['receipt']['commands'][index], expected, proc['cwd'])):
                continue
            plan = read(output/'plan.json')
            if spec['name'] == 'collection':
                if output != Path(root).resolve()/'collection':
                    continue
                binding = read(output/'registration.json')
                if (Path(binding.get('root', '/')).resolve() != Path(root).resolve()
                        or Path(binding.get('owner', '/')).resolve() != Path(spec['owner']).resolve()
                        or Path(binding.get('output', '/')).resolve() != output
                        or Path(binding.get('job', '/')).resolve() != Path(spec['job']).resolve()
                        or binding.get('plan_sha256') != sha((output/'plan.json').read_bytes())
                        or Path(plan.get('resource_root', '/')).resolve() != Path(root).resolve()):
                    continue
            else:
                if output.parent != Path(spec['owner']).resolve():
                    continue
                binding = read(Path(spec['owner'])/'registration.json')
                if (Path(binding.get('job', '/')).resolve() != Path(spec['job']).resolve()
                        or binding.get('job_sha256') != sha(Path(spec['job']).read_bytes())
                        or binding.get('wave_plan_sha256', {}).get(str(output)) != sha((output/'plan.json').read_bytes())
                        or Path(plan.get('resource_root', '/')).resolve() != Path(spec['owner']).resolve()):
                    continue
            return dict(pid=pid, created=proc['created'], name=spec['name'],
                        output=str(output), owner_pid=reg['owner_process']['pid'],
                        owner_created=reg['owner_process']['created'])
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
    return None


def inspect(root, config, now=None, lookup=process, external_scan=external_test_processes):
    """Read-only decision with raw telemetry retained unchanged in the result."""
    now = time.time() if now is None else now
    registrations, rows, errors = [], [], []
    for spec in specs(root, config):
        owner = Path(spec['owner'])
        if not (owner/'project_job_started.json').exists():
            continue
        try:
            reg = active_registration(spec, root, lookup)
            if reg is None:
                continue
            registrations.append(reg)
            row = monitor_row(reg, now, lookup)
            rows.append(dict(owner=str(owner), name=spec['name'], raw=row))
            if now-float(row['utc_epoch']) >= ADMISSION_AGE:
                errors.append(dict(name=spec['name'], error='early_monitor_age_guard',
                                   age_s=now-float(row['utc_epoch'])))
            if not isinstance(row.get('process_telemetry_errors'), list) or row['process_telemetry_errors']:
                errors.append(dict(name=spec['name'], error='process_telemetry_errors',
                                   detail=row.get('process_telemetry_errors')))
            if not isinstance(row.get('gpu_processes'), list) or not isinstance(row.get('external_gpu_processes'), list):
                errors.append(dict(name=spec['name'], error='incomplete_process_telemetry'))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Ignore closed historical roots, but never ignore a live claimed owner
            # with missing/invalid registration or telemetry.
            try:
                candidate = live_owner(spec, read(owner/'project_job_started.json'), lookup)
            except (OSError, ValueError, TypeError):
                candidate = None
            if candidate:
                errors.append(dict(name=spec['name'], error=str(exc)))
    excluded, external = {}, {}
    for observed in rows:
        raw = observed['raw']
        if not isinstance(raw.get('gpu_processes'), list) or not isinstance(raw.get('external_gpu_processes'), list):
            continue
        # Union all fresh monitor views; a clean monitor cannot mask another's error.
        listed = list(raw.get('external_gpu_processes') or [])
        for item in raw['gpu_processes']:
            if not isinstance(item, dict):
                errors.append(dict(name=observed['name'], error='Malformed GPU process row'))
            elif not item.get('owned', False):
                listed.append(item)
        for item in listed:
            try:
                if not isinstance(item, dict):
                    raise ValueError('Malformed external GPU process row')
                pid = item['pid']
                if type(pid) is not int or pid <= 0:
                    raise ValueError('Invalid GPU PID')
                binding = registered_worker(pid, registrations, root, lookup)
                if binding:
                    excluded[pid] = binding
                else:
                    external[pid] = item
            except (KeyError, ValueError, TypeError) as exc:
                errors.append(dict(name=observed['name'], error=str(exc)))
    try:
        cpu_external = external_scan()
        if (not isinstance(cpu_external, dict) or not isinstance(cpu_external.get('processes'), list)
                or not isinstance(cpu_external.get('errors'), list)):
            raise ValueError('Malformed external Python scan')
    except Exception as exc:
        cpu_external = dict(processes=[], errors=[repr(exc)])
    if cpu_external['errors']:
        errors.append(dict(error='external_python_scan_errors', detail=cpu_external['errors']))
    if errors:
        reason = 'telemetry_or_registration_error'
    elif cpu_external['processes']:
        reason = 'external_test_python_present'
    elif not rows:
        reason = 'no_fresh_owned_telemetry'
    elif external:
        reason = 'external_gpu_present'
    else:
        reason = 'no_external_gpu'
    return dict(paused=bool(errors or not rows or external or cpu_external['processes']), reason=reason,
                external_gpu_processes=list(external.values()),
                external_test_python_processes=cpu_external['processes'],
                external_test_python_scan_errors=cpu_external['errors'],
                registered_workers_excluded=list(excluded.values()), errors=errors,
                active_resource_roots=[str(r['spec']['owner']) for r in registrations],
                raw_monitor_rows=rows, utc=now)


class Schedules:
    """Restore exact original bytes only while the last written SHA still matches."""
    def __init__(self, root, backup):
        self.root, self.backup = Path(root).resolve(), Path(backup)
        self.backup.mkdir(parents=True, exist_ok=False)
        self.saved, self.changed, self.conflicts = {}, {}, []

    def targets(self):
        for session in self.root.rglob('session.json'):
            if session.is_symlink() or not inside(session, self.root):
                continue
            try:
                value = read(session)
                if value.get('resource_policy') == 'full_gpu_v1':
                    yield session.parent/'concurrency.json'
            except (OSError, ValueError):
                continue

    def content(self, path):
        if path.is_symlink() or not inside(path, self.root):
            raise ValueError('Unsafe schedule path')
        return path.read_bytes() if path.exists() else None

    def pause(self):
        for path in self.targets():
            key = str(path)
            if key in self.conflicts:
                continue
            try:
                current = self.content(path)
                if key not in self.saved:
                    self.saved[key] = current
                    atomic_json(self.backup/(sha(key.encode())+'.json'),
                                dict(path=key, existed=current is not None,
                                     sha256=None if current is None else sha(current),
                                     base64=None if current is None else base64.b64encode(current).decode()))
                expected = self.changed.get(key)
                if expected is not None:
                    if current is None or sha(current) != expected:
                        self.conflicts.append(key)
                        continue
                elif current != self.saved[key]:
                    self.conflicts.append(key)
                    continue
                original = {} if self.saved[key] is None else json.loads(self.saved[key])
                groups = list(original.get('pause_groups', []))
                for group in PAUSES:
                    if group not in groups:
                        groups.append(group)
                value = dict(original, pause_groups=groups, paired_resource_watch_v3_pid=os.getpid())
                encoded = json.dumps(value, indent=2, allow_nan=False).encode()
                if current != encoded:
                    atomic_bytes(path, encoded)
                self.changed[key] = sha(encoded)
            except (OSError, ValueError, TypeError):
                if key not in self.conflicts:
                    self.conflicts.append(key)

    def restore(self):
        restored = []
        for key, expected in list(self.changed.items()):
            path = Path(key)
            try:
                current = self.content(path)
                if current is not None and sha(current) == expected:
                    original = self.saved[key]
                    if original is None:
                        path.unlink()
                    else:
                        atomic_bytes(path, original)
                    restored.append(key)
                elif key not in self.conflicts:
                    self.conflicts.append(key)
            except (OSError, ValueError):
                if key not in self.conflicts:
                    self.conflicts.append(key)
            del self.changed[key]
        return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--audit', type=Path, default=Path(__file__).parent)
    parser.add_argument('--apply', action='store_true', help='Explicit schedule mutations; default is one read-only snapshot')
    args = parser.parse_args()
    if not args.apply:
        print(json.dumps(inspect(args.root, args.config), indent=2))
        return
    root, audit = args.root.resolve(), args.audit.resolve()
    deadline = float(read(root/'session.json')['deadline'])
    if not math.isfinite(deadline) or deadline <= time.time():
        raise ValueError('Original finite session expired')
    # Refuse any previous writer rather than racing its byte-restore logic.
    for status_name, filename in (('resource_watch_status.json', 'resource_watch.py'),
                                  ('resource_watch_v2_status.json', 'resource_watch_v2.py')):
        old = audit/status_name
        if old.exists():
            running = process(read(old).get('pid'))
            if running and any(Path(arg).name == filename for arg in running['command']):
                raise RuntimeError('Old watcher still live; root must close it before launching v3')
    lock = audit/'resource_watch_v3.lock'
    import fcntl
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        schedule = Schedules(root, audit/f'resource_watch_v3_backups_{int(time.time())}_{os.getpid()}')
        def interrupt(signum, frame):
            raise KeyboardInterrupt(signum)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupt)
        log = audit/'resource_watch_v3_decisions.jsonl'
        last = None
        try:
            while time.time() < deadline and not (root/'STOP_RESOURCE_WATCH').exists():
                state = inspect(root, args.config)
                if state['paused']:
                    schedule.pause()
                else:
                    schedule.restore()
                state.update(pid=os.getpid(), modified_schedules=len(schedule.changed),
                             schedule_conflicts=list(schedule.conflicts))
                atomic_json(audit/'resource_watch_v3_status.json', state)
                signature = {k: v for k, v in state.items() if k not in ('utc', 'raw_monitor_rows')}
                if signature != last:
                    with log.open('a') as stream:
                        stream.write(json.dumps(state)+'\n')
                    last = signature
                time.sleep(min(.5, max(0., deadline-time.time())))
        finally:
            restored = schedule.restore()
            atomic_json(audit/'resource_watch_v3_exit.json',
                        dict(utc=time.time(), pid=os.getpid(), restored=restored,
                             conflicts=schedule.conflicts, modified_schedules=len(schedule.changed),
                             no_process_signals=True, exact_original_byte_restoration=True))


if __name__ == '__main__':
    main()
