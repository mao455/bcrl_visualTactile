"""CPU collection census: JSON-only live snapshots, strict completed physical audit.

No simulator, checkpoint tensors, RGB/DINO payloads or active HDF5 are opened.
Completed mode delegates physical/label checks to the existing collection review.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from visuotactile.compact import sha256
from visuotactile.training_expansion import (configuration_digest, read_registry,
    validate_collection_plan)
from visuotactile.training_resources import atomic_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def source_audit(plan):
    """Only small registered metadata; do not rehash weights or trajectories."""
    validate_collection_plan(plan)
    registry = read_registry(plan['training_expansion_registry'])
    original = read(registry['original_frozen_families']['path'])
    prior = [read_registry(binding) for binding in registry['prior_training_registries']]
    old_rows = original + [row for item in prior for row in item['families'].values()]
    old_ids = {row['family'] for row in old_rows}
    old_seeds = {row['seed'] for row in old_rows}
    old_hashes = {configuration_digest(row['narrow_reset']) for row in old_rows}
    rows = registry['families']; new_ids = set(rows)
    require(not (new_ids & old_ids) and not (new_ids & set(registry['protected_families'])),
        'New family intersects an original/prior/protected family')
    require(not ({r['seed'] for r in rows.values()} & old_seeds)
        and not ({r['configuration_sha256'] for r in rows.values()} & old_hashes),
        'New family seed/configuration duplicates prior registration')
    schedules = Counter(case['expansion_schedule'] for case in plan['cases'])
    return dict(metadata_only=True, source_pairs=plan['pairs'], source_selection=plan.get('source_selection'),
        source_data_sha256=plan['source_data_sha256'], source_commit=plan['source_commit'],
        registry=plan['training_expansion_registry'], new_families=sorted(new_ids),
        new_family_count=len(new_ids), registered_opportunities=len(plan['cases']),
        opportunities_by_schedule=dict(schedules), protected_families=registry['protected_families'],
        validation_families=registry['source_validation_families'],
        original_test_identity_count=len(registry['original_test_identities']),
        prior_registered_family_count=len(old_ids), distinct_family_seed_configuration=True,
        distinctness_scope=registry['initial_configuration_hash_scope'],
        heldout_training_payloads_opened=0, large_payload_sha_recomputed=False)


def held_recheck(case, result, *, boundary=None):
    """Receipt-level classification, distinct from replay/physical audit below."""
    schedule = case['expansion_schedule']
    if schedule not in ('held60', 'held120'):
        return None
    if not result.get('legal'):
        return dict(category='invalid_initialization', receipt=None)
    dagger = result.get('dagger', {}); receipt = dagger.get('trigger_receipt')
    require(isinstance(receipt, dict), 'Held recheck result lacks trigger receipt')
    first = receipt['first_contact_height_window_end_completed_tick']
    earliest = receipt['earliest_allowed_completed_tick']
    rechecked = receipt['first_rechecked_contact_height_window_end_completed_tick']
    completed = receipt['completed_student_transitions_observed']
    delay = int(schedule[4:]); switched = dagger.get('expert_ticks', 0) > 0
    require(receipt['configuration'] == case['dagger']['trigger']
        and receipt['configuration']['min_delay_ticks'] == delay,
        'Held recheck configuration differs from registered schedule')
    if first is None:
        require(earliest is None and rechecked is None and not switched,
            'Takeover cannot precede first held window')
        category = 'no_completed_held_window'
    else:
        require(earliest == first+delay and first <= completed,
            'Held recheck elapsed-time index differs')
        if rechecked is None:
            require(not switched, 'Expert takeover lacks a current held recheck')
            category = ('minimum_delay_not_reached' if completed < earliest
                else 'minimum_elapsed_without_current_held_window')
        else:
            require(earliest <= rechecked == completed
                and receipt['actual_scheduled_takeover_tick'] == rechecked
                and receipt['latest_contact_height_window'] is True,
                'Held takeover must use the latest completed held window')
            category = 'actual_rechecked_takeover' if switched else 'rechecked_without_next_action'
    if switched and boundary is not None:
        require(boundary['completed_student_tick'] == rechecked and boundary['held_window'] is True,
            'Numeric boundary disagrees with current held recheck')
    return dict(category=category, receipt=receipt,
        wait_after_first_held_ticks=None if rechecked is None else rechecked-first,
        additional_wait_after_minimum_ticks=None if rechecked is None else rechecked-earliest,
        expert_recovery_success=bool(dagger.get('expert_takeover_success')),
        native_termination_reasons=result.get('reasons'),
        terminal_authority='expert' if switched else 'student', boundary=boundary,
        interpretation='Missing windows are not drops; nonheld_windows_after_first counts overlapping condition windows, not drop events.')


def snapshot(root):
    """Read atomic small files only; results index is the admission barrier.

    The live writer publishes entries before its results index. Report this
    bounded publication interval explicitly instead of claiming an atomic batch.
    """
    root = Path(root).resolve(); began = time.time(); bindings = {}
    def load(relative, default=None):
        path = root/relative
        if not path.exists():
            return default
        # Hash exactly the bytes decoded, even if a later atomic rename occurs.
        import hashlib
        payload = path.read_bytes()
        bindings[str(relative)] = dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())
        return json.loads(payload)
    owner = load('owner_finished.json')  # completion evidence frozen before census
    plan = load('plan.json'); provenance = source_audit(plan)
    indexed = load('results.json', [])
    entries = load('correction_entries.json', {})
    cases = {case['name']:case for case in plan['cases']}
    require(len(indexed) == len({row['case'] for row in indexed})
        and all(row['case'] in cases for row in indexed) and set(entries) <= set(cases),
        'Unregistered/duplicate completed opportunity or manifest entry')
    by_name = {row['case']:row for row in indexed}; rows = []; accepted = set()
    for name, case in cases.items():
        execution = load(Path(name)/'execution_complete.json')
        capture = load(Path(name)/'capture_complete.json')
        indexed_row = by_name.get(name)
        result = indexed_row['result'] if indexed_row is not None else execution
        verified = capture is not None and capture.get('status') == 'source_verified'
        if verified:
            require(capture.get('dataset_capture', {}).get('verified_source_fields') == 185,
                'Source-verified capture does not preserve all185 fields')
        if indexed_row is not None:
            require(indexed_row['role'] == case['role'] and result.get('software_complete') is True,
                'Indexed role or completion differs')
            status = 'admission_committed'
            if result.get('accepted_for_training'):
                require(name in entries and verified and result.get('legal') is True and result.get('success') is True
                    and result['dataset_capture'] == capture['dataset_capture']
                    and result['dagger']['expert_takeover_success'] and result['dagger']['expert_valid_ticks'] > 0,
                    'Accepted index lacks manifest or verified successful expert capture')
                entry = entries[name]; pair = plan['pairs'][case['role']]
                require(entry.get('episode') == case['spec']['identity'] == name
                    and entry.get('family') == case['spec']['family']
                    and entry.get('source_sha256') == capture['dataset_capture']['sha256']
                    and all(entry.get(key) == pair[key] for key in ('frontend_sha256', 'backend_sha256')),
                    'Accepted manifest entry differs from actual case/capture/policy')
                accepted.add(name)
        elif verified:
            status = 'source_verified_pending_admission'
        elif execution is not None:
            status = 'execution_complete_pending_verification'
        else:
            status = 'started_or_pending' if (root/name).exists() else 'not_started'
        rows.append(dict(case=name, role=case['role'], family=case['spec']['family'],
            schedule=case['expansion_schedule'], status=status, source_verified=verified,
            accepted=name in accepted, legal=None if result is None else result.get('legal'),
            steps=None if result is None else result.get('steps', 0),
            expert_recovery_success=bool((result or {}).get('dagger', {}).get('expert_takeover_success')),
            eligible_expert_ticks=(result or {}).get('dagger', {}).get('expert_valid_ticks', 0),
            held_recheck=None if result is None else held_recheck(case, result)))
    started = load('owner_started.json', {}); progress = load('batch_progress.json')
    return dict(version='training_collection_census_v1', mode='snapshot', complete=False,
        root=str(root), snapshot_started_utc=began, snapshot_finished_utc=time.time(),
        report_code_sha256=sha256(__file__),
        source=provenance, planned_opportunities=len(cases), committed_opportunities=len(indexed),
        counters=dict(status= dict(Counter(row['status'] for row in rows)),
            committed_legal=sum(bool(by_name[name]['result'].get('legal')) for name in by_name),
            source_verified=sum(row['source_verified'] for row in rows),
            committed_accepted=len(accepted), manifest_entries=len(entries),
            manifest_ahead_of_result_index=sorted(set(entries)-set(by_name)),
            committed_accepted_families=len({cases[name]['spec']['family'] for name in accepted})),
        cases=rows, progress=progress, owner_finished_observed=owner,
        owner_started_utc=started.get('started'), bindings=bindings,
        hdf5_files_opened=0, rates=None,
        interpretation='Partial JSON census, not a completed physical audit or throughput conclusion. Source verification alone is not admission; eligible labels are not optimizer exposure.')


def held_boundary_window(directory, completed_tick):
    """Small slices only, after complete() has audited and closed the whole batch."""
    import h5py
    import numpy as np
    from tools.tight_fresh_takeover_pose_audit import window_components
    require(completed_tick >= 48, 'Held boundary needs the original student reference')
    with h5py.File(Path(directory)/'numeric.h5', 'r') as stream:
        t = stream['episode/transition']; after = t['diagnostics_after']
        require((t['dagger/control_source'][:completed_tick] == 0).all(),
            'Held window/reference must contain only actual student transitions')
        height = float(np.median(after['object_pose_w'][24:48, 2]))
        window = slice(completed_tick-12, completed_tick)
        value = window_components(after['object_pose_w'][window], after['palm_body_pose_w'][window],
            t['object_contact_force_w'][window], height, 12)
    value['completed_tick'] = completed_tick
    return value


def complete(root):
    from tools.tight_batch_collection_review import review, summarize as behavior_summary
    from tools.tight_batch_throughput_report import summarize as throughput_summary
    root = Path(root).resolve(); plan = read(root/'plan.json')
    provenance = source_audit(plan)
    require(all((root/name).exists() for name in ('owner_finished.json', 'result.json', 'results.json')),
        'Complete collection markers required before opening numeric data')
    owner, result = read(root/'owner_finished.json'), read(root/'result.json')
    require(owner.get('software_complete') is True and owner.get('returncode') == 0
        and result.get('software_complete') is True and result.get('worker_error') is None
        and len(read(root/'results.json')) == len(plan['cases']),
        'Incomplete collection cannot receive a completed physical audit')
    behavior = review(root)  # rejects unfinished collection before reading numeric
    throughput = throughput_summary(root, batch=True)
    indexed = {row['case']:row['result'] for row in read(root/'results.json')}
    specs = {case['name']:case for case in plan['cases']}
    delayed = []
    for row in behavior['cases']:
        audit = held_recheck(specs[row['case']], indexed[row['case']], boundary=row.get('boundary'))
        if audit is not None:
            audit['boundary_window_components'] = (held_boundary_window(root/row['case'],
                row['boundary']['completed_student_tick']) if row.get('boundary') is not None else None)
            delayed.append(dict(case=row['case'], role=row['role'], schedule=row['schedule'],
                accepted=row['accepted'], student_native_lift=row.get('student_native_lift'),
                student_old_stable_window=row.get('student_stable_window_seen'), **audit))
    extra_timing = {key:sum(result.get('async_capture', {}).get('profile', {}).get(key, 0.)
        for result in indexed.values()) for key in ('verification_slot_wait_s', 'verification_queue_wait_s')}
    return dict(version='training_collection_census_v1', mode='complete', complete=True,
        root=str(root), created_utc=time.time(), report_code_sha256=sha256(__file__), source=provenance, behavior=behavior,
        by_schedule={schedule:behavior_summary([row for row in behavior['cases'] if row['schedule']==schedule])
            for schedule in sorted({row['schedule'] for row in behavior['cases']})},
        held_recheck_cases=delayed,
        held_recheck_categories=dict(Counter(row['category'] for row in delayed)),
        throughput=throughput, additional_async_profile_sums=extra_timing,
        definitions=dict(height_reference='Median object height after completed student transitions25–48; unavailable before48 student transitions, never inferred from later expert motion.',
            held='Latest12 completed transitions: object>=reference+2cm and >=2 fingers >0.1N in each of both120Hz samples; rotation allowed.',
            old_stable='Held plus original palm-relative1cm/0.2rad window; native lift remains its separate task diagnostic.',
            stage_counts='Overlapping actual legal expert action masks, not schedule entrance labels or optimizer exposure. Fixed40/80 later expert actions can include turning/alignment/insertion.',
            async_timing='Attribution only; append, verification, producer waits and simulation overlap. Do not add these sums to wall clock.',
            success='Only original task success. Expert recovery is separate from student autonomous success; no-trigger and native drop/fall are separate observations.'),
        optimizer_updates=0, simulator_starts=0, heldout_training_payloads_opened=0)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=('snapshot', 'complete'))
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier report; choose a new output')
    value = (snapshot if args.mode == 'snapshot' else complete)(args.collection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, value)


if __name__ == '__main__':
    main()
