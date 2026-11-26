"""Missing-backend recovery owner; unchanged actual-capacity sharing policy.

Only this exact new owner is added. Original monitor identity, all-group
cooperative schedules and byte restoration remain delegated to the reviewed
watcher. Importing this module starts no process and writes no schedule.
"""
import math
import time
from pathlib import Path

from outputs.diagnostics.tight_paired_dagger_20261122_01 import shared_capacity06 as base

HERE = Path(__file__).resolve().parent
MODULE = 'outputs.diagnostics.tight_paired_dagger_20261122_01.shared_capacity09'
AUDIT = HERE/'watch_shared_capacity09'
ROOT = base.ROOT
watch = base.watch
live_capacity = base.live_capacity
capacity_decision = base.capacity_decision
inspect = base.inspect


def specs(root, config):
    rows = base.specs(root, config)
    rows.append(dict(name='backend_recovery09', owner=Path(root)/'owners/backend_recovery09',
        module='outputs.diagnostics.tight_paired_dagger_20261122_01.parallel_rear_recovery09',
        operation='train', job=HERE/'proposed_backend_recovery09.json'))
    return rows


def startup_row(row, proc, now, *, allow_gap=False):
    age = now-float(row['utc'])
    if not math.isfinite(age) or not -2 <= age < 10:
        raise ValueError('Watcher heartbeat stale')
    if row.get('errors') != [] or row.get('schedule_conflicts') != []:
        raise ValueError('Watcher errors/conflicts')
    expected = ['-m', MODULE, '--audit', str(AUDIT), '--apply']
    if proc.pid != row['pid'] or not proc.is_running() or proc.cmdline()[1:] != expected:
        raise ValueError('Exact recovery capacity watcher PID/command required')
    if Path(proc.cwd()).resolve() != watch.PROJECT or proc.create_time() > row['utc']+1:
        raise ValueError('Capacity watcher cwd/create time differs')
    idle = row.get('active_resource_roots') == [] and row.get('raw_monitor_rows') == [] and row.get('registered_workers_excluded') == []
    normal = row.get('paused') is False and row.get('reason') == 'shared_capacity_available'
    gap = allow_gap and idle and row.get('paused') is True and row.get('reason') == 'no_fresh_owned_telemetry'
    if not (normal or gap):
        raise ValueError('Capacity watcher not ready')
    return dict(starting_without_owned_telemetry=bool(gap), watcher_remains_paused=bool(gap),
        active_resource_roots=row['active_resource_roots'], external_gpu_processes=row['external_gpu_processes'],
        external_test_python_processes=row['external_test_python_processes'], external_presence_is_not_veto=True)


def main():
    watch.specs = specs
    watch.registered_worker = base.identity.registered_worker
    watch.inspect = inspect
    watch.main()


if __name__ == '__main__':
    main()
