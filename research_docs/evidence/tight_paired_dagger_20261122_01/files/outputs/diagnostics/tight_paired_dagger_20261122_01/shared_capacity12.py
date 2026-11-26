"""Exact diagnostic test identity; existing capacity/restore policy unchanged."""
from pathlib import Path
import math
from outputs.diagnostics.tight_paired_dagger_20261122_01 import shared_capacity10 as prior

HERE = Path(__file__).resolve().parent
MODULE = 'outputs.diagnostics.tight_paired_dagger_20261122_01.shared_capacity12'
WORKER = 'outputs.diagnostics.tight_paired_dagger_20261122_01.eval_startup12'
AUDIT = HERE/'watch_shared_capacity12'
watch = prior.watch
inspect = prior.inspect


def specs(root, config):
    rows = prior.specs(root, config)
    found = 0
    for row in rows:
        if row['name'] == 'test':
            if row['module'] != 'tools.tight_paired_evaluate':
                raise ValueError('Unexpected original test identity')
            row['module'] = WORKER
            found += 1
    if found != 1:
        raise ValueError('Exactly one original test owner required')
    return rows


def startup_row(row, process, now, *, allow_gap=False):
    age = now-float(row['utc'])
    if not math.isfinite(age) or not -2 <= age < 10:
        raise ValueError('Watcher heartbeat stale')
    if row.get('errors') != [] or row.get('schedule_conflicts') != []:
        raise ValueError('Watcher errors/conflicts')
    expected = ['-m', MODULE, '--audit', str(AUDIT), '--apply']
    if (process.pid != row['pid'] or not process.is_running()
            or process.cmdline()[1:] != expected or Path(process.cwd()).resolve() != watch.PROJECT
            or process.create_time() > row['utc']+1):
        raise ValueError('Exact live watcher12 identity required')
    idle = (row.get('active_resource_roots') == [] and row.get('raw_monitor_rows') == []
            and row.get('registered_workers_excluded') == [])
    normal = row.get('paused') is False and row.get('reason') == 'shared_capacity_available'
    gap = (allow_gap and idle and row.get('paused') is True
           and row.get('reason') == 'no_fresh_owned_telemetry')
    if not (normal or gap):
        raise ValueError('Capacity watcher12 not ready')
    return dict(starting_without_owned_telemetry=bool(gap), external_presence_is_not_veto=True,
                active_resource_roots=row['active_resource_roots'])


def main():
    watch.specs = specs
    watch.registered_worker = prior.base.identity.registered_worker
    watch.inspect = inspect
    watch.main()


if __name__ == '__main__':
    main()
