"""Inactive CPU-only final accounting for truthful failed06 plus recovery07.

Only small JSON receipts are read. Certified owner intervals include failed work,
the new backend owner and evaluation owners once each; missing historical
collection closure stays unknown. Wall duration is never called GPU-active time.
"""
from __future__ import annotations
import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_final07_common_compute', HERE/'final_recovery07_common.py')
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)
BASE_SHA256 = '9bd109d3895856ab58e9aa3c67d952985b546b5701e5f7df0e01d82c152e1aae'
base = common.pinned_module('final_compute_shared06.py', BASE_SHA256, 'compute')
ROOT = base.ROOT
union = base.union


def inventory(root=ROOT, audit=HERE):
    root, audit = Path(root).resolve(), Path(audit).resolve()
    lifecycle = common.closure_gate(root, audit)
    common.require(lifecycle['status'] == 'closed', 'Real recovery07 closure required')
    watcher = common.watcher_closed(audit)
    sources = {}

    def read(path, required=True):
        if not path.is_file() and not required: return None
        value = common.read_small(path, maximum=8*2**20)
        sources[str(path)] = common.sha(path)
        return value

    session = read(root/'session.json'); now = time.time()
    components = []
    for phase, directory in [('parents', 'common180_parents'), ('new', 'learning')]:
        for role in 'AB':
            for component, sub in [('DP', 'training'), ('Transformer', f'backend/{role}/training')]:
                path = root/directory/role/sub
                deployment, finished = read(path/'deployment.json'), read(path/'finished.json')
                completed = (finished.get('complete') is True if component == 'DP' else
                             finished.get('target_reached') is True and finished.get('control_tick_epochs', 0) >= 60)
                updates = finished.get('updates')
                common.require(deployment.get('epoch') == 60 and completed and type(updates) is int
                    and updates > 0 and deployment.get('update') == updates,
                    'Eight real epoch60 components with matching committed updates required: '+str(path))
                expected = (7200 if component == 'DP' else 22763) if phase == 'parents' else (
                    10200 if component == 'DP' else 38400)
                common.require(updates == expected, 'Registered actual update budget differs: '+str(path))
                components.append(dict(phase=phase, role=role, component=component,
                    deployment=deployment, actual_committed_updates=updates,
                    timer_fields={key:value for key,value in finished.items() if key in (
                        'wall_seconds','wall_s','training_wall_s','setup_wall_s','wait_s','work_s')},
                    timer_limit='Host work and resource waits included; recovered timers may reset; not GPU-active hours'))
    owners, intervals = [], []
    required_owners = {root/'owners'/name for name in ('main_cache','main_train','backend_recovery07')}
    required_owners.update(root/'evaluation'/cohort for cohort in ('development','test'))
    paths = sorted(set((root/'owners').glob('*')) | required_owners)
    for directory in paths:
        started = read(directory/'project_job_started.json', False)
        if started is None:
            common.require(directory not in required_owners, 'Current owner start receipt missing: '+str(directory))
            continue
        start = started['utc']; union([[start, start]])
        closed = read(directory/'project_exit.json', False)
        if closed is None:
            common.require(directory == root/'owners/collection', 'Owner exit receipt missing: '+str(directory))
            owners.append(dict(path=str(directory), start_utc=start, end_utc=None, elapsed_s=None,
                failure=read(directory/'project_failure.json', False), children_reaped=None,
                closure_status='historical_exit_missing', wall_time_certified=False,
                start_receipt=str(directory/'project_job_started.json'), exit_receipt=None,
                limitation='Original collection end remains unknown and is excluded from certified interval union.'))
            continue
        common.require(closed.get('children_reaped') is True, 'Unreaped owner '+str(directory))
        end = closed['utc']; union([[start, end]])
        intervals.append([start, end])
        owners.append(dict(path=str(directory), start_utc=start, end_utc=end, elapsed_s=end-start,
            failure=read(directory/'project_failure.json', False), children_reaped=True,
            closure_status='closed', wall_time_certified=True,
            start_receipt=str(directory/'project_job_started.json'), exit_receipt=str(directory/'project_exit.json')))
    standalone = []
    for stage in common.STANDALONE_STAGES:
        admission = read(audit/'driver_backend_recovery07'/(stage+'_shared_capacity06_admission.json'))
        end = read(root/'cycle_logs_recovery07'/(stage+'_exit.json'))
        common.require(end.get('returncode') == 0, 'Standalone audit did not close: '+stage)
        start_utc, end_utc = admission['utc'], end['utc']; union([[start_utc, end_utc]])
        intervals.append([start_utc, end_utc])
        standalone.append(dict(stage=stage, start_admission_utc=start_utc, end_utc=end_utc,
            elapsed_s=end_utc-start_utc,
            scope='Recorded admission-to-exit wall time; includes host/setup/waits, not measured GPU activity'))
    discarded = []
    for scope in ('common180_parents','learning'):
        for path in sorted((root/scope).glob('*/training/recovery.*.json')):
            row = read(path); ids = row.get('known_discarded_update_ids', [])
            common.require(len(ids) == len(set(ids)), 'Duplicate discarded update IDs')
            discarded.append(dict(path=str(path), scope=scope, update_ids=ids, age_and_exposure_credit=0))
    archives = []
    for name in ('leased_archive_capacity01','leased_archive_capacity02','leased_archive_shared06','leased_archive_shared07','leased_archive_shared08','leased_archive_shared09'):
        directory = audit/name; registration = read(directory/'registration.json', False)
        if registration is None: continue
        end = read(directory/'complete.json', False) or read(directory/'failure.json', False)
        common.require(end is not None, 'Cold auxiliary archive not closed: '+name)
        rows = [read(path) for path in sorted(directory.glob('file_*.json'))]
        common.require(end.get('completed_files',end.get('files')) == len(rows)
            and end.get('saved_bytes') == sum(row['saved_bytes'] for row in rows),
            'Auxiliary archive terminal/progress totals differ: '+name)
        archives.append(dict(attempt=name, registration=registration.get('binding'),
            start_utc=registration['utc'], end=end, completed_files=len(rows),
            saved_bytes=sum(row['saved_bytes'] for row in rows),
            scope='CPU-only auxiliary work; originals/failures retained, not added to deployed model ages'))
    collection = read(root/'collection/collection_summary.json')
    final = read(root/'evaluation/test/summary/primary_summary.json')
    return dict(version='paired_final_compute_recovery07_v1', final=True, cutoff_utc=now,
        cutoff_iso=datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat(),
        original_deadline=session['deadline'], session_started_utc=session['started_utc'],
        wall_elapsed_to_accounting_cutoff_s=now-session['started_utc'], components=components,
        committed_parent_updates=sum(row['actual_committed_updates'] for row in components if row['phase']=='parents'),
        committed_main_updates=sum(row['actual_committed_updates'] for row in components if row['phase']=='new'),
        discarded_updates=discarded, discarded_update_count=sum(len(row['update_ids']) for row in discarded),
        discarded_update_limit='Actual discarded work is retained separately and contributes zero selected age/exposure.',
        owners=owners, standalone_audits=standalone, owner_and_standalone_interval_union=union(intervals),
        interval_scope='Certified closed owner union includes failed06 and backend07 plus evaluation/standalone audits; waits/setup/failures included. Unknown historical collection excluded. No GPU-active integration.',
        failed06_and_recovery07_lifecycle=lifecycle, watcher_closure=watcher,
        cold_archives=archives, collection=collection,
        test_summary_path=str(root/'evaluation/test/summary/primary_summary.json'), test_summary_fields=list(final),
        gpu_hours=None, gpu_hours_reason='Per-job GPU-active integration is unavailable; wall/CUDA residence is not GPU-hours.',
        source_small_metadata_sha256=sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--audit', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = inventory(args.root, args.audit)
    common.require(args.output.resolve().is_relative_to(args.audit.resolve()) and not args.output.exists(),
                   'Only a NEW AUDIT output file is allowed')
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__': main()
