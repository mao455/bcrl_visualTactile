"""Small-metadata-only console snapshot. No experiment writes or model imports."""
import json
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
ROOT = PROJECT / 'outputs/visuotactile/run/tight_paired_dagger_20261122_01'


def read(path):
    if not path.exists():
        return None
    if path.stat().st_size > 8 * 1024 * 1024:
        return {'unread_large_metadata': str(path)}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return {'read_error': repr(error)}


def snapshot():
    now = time.time()
    result = {'utc': now}
    for name, path in (
        ('waiter', AUDIT / 'wait_main_recovery05/status.json'),
        ('main', AUDIT / 'driver_main_recovery05/status.json'),
        ('main_exit', AUDIT / 'driver_main_recovery05/exit.json'),
        ('collection_exit', AUDIT / 'driver_collection_recovery05/exit.json'),
    ):
        row = read(path)
        if row is not None:
            result[name] = row
    watch = read(AUDIT / 'watch_service_recovery05/resource_watch_v4_status.json')
    if watch:
        result['watcher'] = {key: watch.get(key) for key in (
            'paused', 'reason', 'external_gpu_processes', 'errors', 'schedule_conflicts')}
        result['watcher']['heartbeat_age_s'] = round(now - watch.get('utc', now), 1)
    directory = ROOT / 'collection_recovery05'
    assignments = read(directory / 'paired_queue_assignments.json')
    if isinstance(assignments, list):
        paths = list(directory.glob('*/result.json'))
        complete = sum(read(path).get('software_complete') is True for path in paths)
        progress = read(directory / 'batch_progress.json')
        result['collection_suffix'] = dict(
            assigned=len(assignments), registered=81, result_files=len(paths),
            software_complete_result_files=complete, admission_not_inferred=True,
            progress=progress,
            progress_age_s=round(now - progress.get('utc', now), 1) if progress else None)
    training = {}
    for role in 'AB':
        for component, sub in (('DP', 'training'), ('rear', f'backend/{role}/training')):
            directory = ROOT / 'learning' / role / sub
            row = read(directory / 'progress.json')
            if row:
                training[role + '_' + component] = row
            deployment = read(directory / 'deployment.json')
            if deployment:
                training[role + '_' + component + '_deployment'] = deployment
    result['training'] = training
    result['evaluation'] = {}
    for cohort in ('development', 'test'):
        directory = ROOT / 'evaluation' / cohort
        registration = read(directory / 'registration.json')
        if not registration:
            continue
        rows = [read(path) for path in sorted(directory.glob('wave*/results.json'))]
        result['evaluation'][cohort] = dict(
            registered=4 * registration['n'],
            result_rows=sum(len(row) for row in rows if isinstance(row, list)),
            summary_exists=(directory / 'summary/primary_summary.json').exists())
    return result


if __name__ == '__main__':
    print(json.dumps(snapshot(), ensure_ascii=False))
