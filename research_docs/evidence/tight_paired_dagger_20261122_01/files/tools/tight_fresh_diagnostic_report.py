"""Rebuild one registered training diagnostic across preserved worker attempts.

CPU-only analysis of completed records. Censored HDF5/video files are never
opened or repaired; their last saved progress is not a terminal tick.
"""
import argparse
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from tools.tight_fresh_phases import case as phase_case
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


ATTEMPTS = ('round03_training_diagnostic', 'round03_training_diagnostic_recovery',
            'round03_training_diagnostic_remaining')


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(path, root):
    return dict(path=str(path.relative_to(root)), sha256=sha256(path))


def grasp_components(path):
    """Explain the existing 12-tick rule without changing its thresholds."""
    with h5py.File(path) as f:
        transition = f['episode/transition']
        pose = transition['diagnostics_after/object_pose_w'][:]
        palm = transition['diagnostics_after/palm_body_pose_w'][:]
        force = transition['object_contact_force_w'][:]
    height = float(np.median(pose[24:48, 2]))
    two = ((np.linalg.norm(force, axis=-1) > .1).sum(-1) >= 2).all(-1)
    inverse = palm[:, 3:].copy()
    inverse[:, 1:] *= -1
    qv, v = inverse[:, 1:], pose[:, :3] - palm[:, :3]
    cross = 2 * np.cross(qv, v)
    relative = v + inverse[:, :1] * cross + np.cross(qv, cross)
    q = pose[:, 3:]
    relative_q = np.concatenate((inverse[:, :1] * q[:, :1] - (qv * q[:, 1:]).sum(-1, keepdims=True),
        inverse[:, :1] * q[:, 1:] + q[:, :1] * qv + np.cross(qv, q[:, 1:])), axis=-1)
    candidates, contact_windows = [], 0
    for stop in range(12, len(pose) + 1):
        start = stop - 12
        if not two[start:stop].all():
            continue
        contact_windows += 1
        if not (pose[start:stop, 2] >= height + .02).all():
            continue
        drift = float(np.linalg.norm(relative[start:stop] - relative[start], axis=-1).max())
        cosine = np.clip(np.abs((relative_q[start:stop] * relative_q[start]).sum(-1)), 0, 1)
        angle = float((2 * np.arccos(cosine)).max())
        candidates.append(dict(end_completed_tick=stop, relative_drift_m=drift, relative_rotation_rad=angle))
    indices = np.flatnonzero(two)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []
    return dict(two_contact_ticks=int(two.sum()), longest_two_contact_run_ticks=max(map(len, runs), default=0),
        two_contact_windows=contact_windows, elevated_two_contact_windows=len(candidates),
        position_stable_elevated_contact_windows=sum(c['relative_drift_m'] <= .01 for c in candidates),
        orientation_stable_elevated_contact_windows=sum(c['relative_rotation_rad'] <= .2 for c in candidates),
        jointly_stable_windows=sum(c['relative_drift_m'] <= .01 and c['relative_rotation_rad'] <= .2 for c in candidates),
        best_joint_stability_windows=sorted(candidates, key=lambda c:
            max(c['relative_drift_m'] / .01, c['relative_rotation_rad'] / .2))[:3],
        ranking='Ascending max(position drift / 1cm, rotation drift / 0.2rad); descriptive only.')


def report(root):
    root = root.resolve()
    directories = [root / name for name in ATTEMPTS]
    original, recovery, remaining = directories
    plan = read(original / 'plan.json')
    cases = plan['cases']
    names = [c['name'] for c in cases]
    require(len(set(names)) == len(names) and all(Path(n).name == n for n in names), 'Unique local case names required')
    require(plan['kind'] == 'autonomous' and plan.get('test_used') is False, 'Expected autonomous training diagnostic')
    require(set(plan['pairs']) == {'A', 'B'}, 'Complete A/B pairs required')
    for c in cases:
        spec = c['spec']
        require(spec['split'] == 'train' and spec['diagnostic_cohort'] == 'original_training_diagnostic'
                and spec['cohort'] == 'fresh', 'Unexpected diagnostic cohort')
        source = plan['original_sources'][spec['source_identity']]
        require(source['source_family'] == spec['family'] and all(source[k] == spec[k]
                for k in ('seed', 'event_seed', 'narrow_reset')), 'Original training reset identity differs')
    require((original / 'plan.json').read_bytes() == (recovery / 'plan.json').read_bytes(), 'Recovery plan differs')
    expected_remaining = [c for c in cases if not any((d / c['name']).exists() for d in (original, recovery))]
    subset = read(remaining / 'plan.json')
    require(subset == dict(plan, cases=expected_remaining), 'Remaining queue changed more than never-started cases')
    session_bytes = (original / 'session.json').read_bytes()
    require(all((d / 'session.json').read_bytes() == session_bytes for d in directories), 'Session/deadline changed')
    session = json.loads(session_bytes)
    registration = read(remaining / 'remaining_registration.json')
    original_hash, subset_hash = sha256(original / 'plan.json'), sha256(remaining / 'plan.json')
    require(registration['original_plan_sha256'] == original_hash
            and registration['remaining_plan_sha256'] == subset_hash
            and registration['remaining_cases'] == [c['name'] for c in expected_remaining]
            and registration['original_case_count'] == len(cases)
            and registration['source_pairs'] == plan['pairs']
            and registration['deadline'] == session['deadline'], 'Remaining registration differs')
    require([x['name'] for x in registration['excluded_cases']] ==
            [c['name'] for c in cases if c not in expected_remaining], 'Exclusion population differs')
    source = Path(registration['frozen_execution_source'])
    source_files = {str(p.relative_to(source)): p for folder in ('tools', 'visuotactile')
                    for p in (source / folder).rglob('*.py')}
    require(set(source_files) == set(plan['source_sha256']) and all(sha256(source_files[p]) == digest
            for p, digest in plan['source_sha256'].items()), 'Frozen execution source population/hash differs')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
            == plan['source_commit'] == registration['frozen_execution_source_commit'], 'Frozen execution commit differs')
    for pair in plan['pairs'].values():
        for name in ('frontend', 'backend'):
            require(sha256(Path(pair[name])) == pair[name + '_sha256'], 'Selected model differs from registered pair')

    attempts, locations = [], {}
    for directory in directories:
        current_plan = read(directory / 'plan.json')
        started = read(directory / 'run_started.json')
        require(started['plan_sha256'] == sha256(directory / 'plan.json'), 'Worker plan binding differs')
        finish_name = 'recovery_finished.json' if directory == recovery else 'owner_finished.json'
        finish = read(directory / finish_name)
        require(isinstance(finish.get('returncode'), int), 'Worker has no exit receipt')
        own_name = 'recovery_registration.json' if directory == recovery else 'owner_registration.json'
        owner = read(directory / own_name)
        require(owner['pid'] == started['pid'] and owner['deadline'] == session['deadline'], 'Owner/worker binding differs')
        if directory == recovery:
            require(owner['preserve_registered_plan'] and owner['prior_plan_sha256'] == original_hash
                    and owner['plan_sha256'] == original_hash
                    and owner['prior_run_started_sha256'] == sha256(original / 'run_started.json')
                    and owner['frozen_execution_source_commit'] == plan['source_commit'], 'Recovery provenance differs')
        else:
            require(owner['source_commit'] == plan['source_commit'], 'Owner source differs')
        recorded = read(directory / 'results.json') if (directory / 'results.json').exists() else []
        require(len({r['case'] for r in recorded}) == len(recorded), 'Duplicate result index')
        completed_names = []
        present_names = []
        registered_names = {c['name'] for c in current_plan['cases']}
        require(not any(p.is_dir() and (p / 'started.json').exists() and p.name not in registered_names
                        for p in directory.iterdir()), 'Unregistered case directory')
        for c in current_plan['cases']:
            case_dir = directory / c['name']
            if not case_dir.exists():
                continue
            require(c['name'] not in locations, 'Registered case repeated across attempts')
            locations[c['name']] = case_dir
            present_names.append(c['name'])
            if (case_dir / 'result.json').exists():
                completed_names.append(c['name'])
                indexed = [r for r in recorded if r['case'] == c['name']]
                require(len(indexed) == 1 and indexed[0]['role'] == c['role']
                        and indexed[0]['result'] == read(case_dir / 'result.json'), 'Result index differs from case record')
        require({r['case'] for r in recorded} == set(completed_names), 'Result index population differs')
        evidence = ['plan.json', 'session.json', 'run_started.json', finish_name, own_name,
                    'remaining_registration.json', 'recovery_plan_binding.json', 'results.json', 'result.json']
        attempts.append(dict(attempt=directory.name, planned_cases=[c['name'] for c in current_plan['cases']],
            started_case_directories=present_names, complete_cases=completed_names, worker_finish=finish,
            pre_control_startup_failure=finish.get('stop_reason') == 'pre_control_startup_timeout' and not present_names,
            owner_source_commit=owner.get('owner_source_commit', registration['owner_source_commit']
                if directory == remaining else None), frozen_execution_commit=plan['source_commit'],
            evidence=[artifact(directory / n, root) for n in evidence if (directory / n).exists()]))

    rows = []
    for c in cases:
        pair = plan['pairs'][c['role']]
        directory = locations.get(c['name'])
        row = dict(case=c['name'], role=c['role'], family=c['spec']['family'],
            source_identity=c['spec']['source_identity'], frontend_sha256=pair['frontend_sha256'],
            backend_sha256=pair['backend_sha256'], attempt=directory.parent.name if directory else None)
        if directory is None:
            row.update(status='not_started', included_in_completed_denominator=False)
        else:
            started = read(directory / 'started.json')
            require(started['spec'] == c['spec'] and started['checkpoint'] == pair['backend'], 'Case start differs')
            row['started'] = artifact(directory / 'started.json', root)
            if not (directory / 'result.json').exists():
                progress = directory / 'progress.json'
                row.update(status='censored', included_in_completed_denominator=False,
                    last_saved_progress=read(progress) if progress.exists() else None,
                    progress_evidence=artifact(progress, root) if progress.exists() else None,
                    exact_terminal_tick=None, success=None, partial_files=[p.name for p in sorted(directory.iterdir())
                        if p.name in ('numeric.h5', 'rollout.mp4.partial')],
                    interpretation='Worker exited without a completed case result; last saved progress is not an exact terminal tick. Partial HDF5/video were not opened.')
            else:
                result = read(directory / 'result.json')
                require(result.get('software_complete') and result.get('legal')
                        and all(result[k] == value for k, value in c['spec'].items())
                        and result['checkpoint_sha256'] == pair['backend_sha256'], 'Completed result identity differs')
                phases = phase_case(directory)
                require(phases['steps'] == result['steps'] and phases['authority']['expert_ticks'] == 0
                        and phases['authority']['student_ticks'] == result['steps'], 'Nonautonomous or inconsistent complete record')
                require(result['simulation_step_hz'] == 60 and result['physics_hz'] == 120
                        and result['plan_hz'] == 5 and result['residual_ticks'] == result['steps']
                        and result['native_lift'] == phases['native_lift']
                        and result['native_subgoals'] == phases['native_subgoals']
                        and result['native_alignment'] == (phases['native_alignment_first_tick'] is not None),
                        'Result control rates or native diagnostics differ from numeric record')
                components = grasp_components(directory / 'numeric.h5')
                require(components['jointly_stable_windows'] == phases['grasp_progress']['stable_ticks'], 'Stable rule disagrees')
                row.update(status='complete', included_in_completed_denominator=True, success=result['success'],
                    steps=result['steps'], phases=phases, grasp_window_components=components,
                    stages=dict(lift_native=phases['native_lift'], hold_stable=phases['settled_height_stable_grasp'],
                        turn_aligned_native=phases['native_alignment_first_tick'] is not None,
                        near_hole_native=phases['native_near_goal_first_tick'] is not None,
                        insertion_complete_native=bool(result['success'])),
                    evidence=[artifact(directory / n, root) for n in ('result.json', 'numeric.h5', 'trace.npz', 'plans.npz')])
        rows.append(row)
    roles = {}
    for role in sorted(plan['pairs']):
        selected = [r for r in rows if r['role'] == role]
        completed = [r for r in selected if r['status'] == 'complete']
        roles[role] = dict(registered=len(selected), complete=len(completed),
            censored=sum(r['status'] == 'censored' for r in selected),
            not_started=sum(r['status'] == 'not_started' for r in selected),
            successes=sum(r['success'] for r in completed), denominator='Completed legal autonomous records only',
            stage_counts={key: sum(r['stages'][key] for r in completed) for key in
                ('lift_native', 'hold_stable', 'turn_aligned_native', 'near_hole_native', 'insertion_complete_native')})
    repo = Path(__file__).resolve().parent.parent
    return dict(version='registered_training_diagnostic_across_attempts_v1', run_root=str(root),
        report_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        report_source_sha256={name: sha256(repo / name) for name in
            ('tools/tight_fresh_diagnostic_report.py', 'tools/tight_fresh_phases.py', 'visuotactile/takeover_trigger.py')},
        original_plan=artifact(original / 'plan.json', root), remaining_plan=artifact(remaining / 'plan.json', root),
        frozen_execution_commit=plan['source_commit'], verified_execution_source_files=len(source_files),
        selected_models=plan['pairs'], original_training_sources=plan['original_sources'],
        session_deadline=session['deadline'], registered=len(rows), completed=sum(r['status'] == 'complete' for r in rows),
        censored=sum(r['status'] == 'censored' for r in rows), not_started=sum(r['status'] == 'not_started' for r in rows),
        roles=roles, attempts=attempts, cases=rows,
        audits=dict(same_original_and_recovery_plan=True, remaining_only_never_started_original_cases=True,
            no_repeated_case_directories=True, unchanged_session_deadline=True, model_hashes_verified=True,
            frozen_source_hashes_verified=True, completed_result_index_exact=True),
        stage_definitions=dict(lift_native='拿起: native lifted_object diagnostic',
            hold_stable='拿稳: existing 12-tick two-contact, >=2cm settled-height rise, <=1cm/0.2rad palm-relative drift',
            turn_aligned_native='转正: recorded native alignment condition',
            near_hole_native='对孔: recorded native near_goal condition; alone does not establish insertion',
            insertion_complete_native='插入: unchanged native complete task success'),
        limits=['Original training families with fresh native reset; no exact hidden-PhysX replay or held-out generalization claim.',
            'Three worker attempts resolve one original registered cohort, not additional evaluation opportunities.',
            'Censored outcomes are unknown and excluded from completed denominators; last_saved_progress is not a terminal tick.',
            'Native lift and goal-distance minima do not establish stable grasp, orientation, or insertion.',
            'Phase diagnostics do not alter native success; position and angle minima can occur at different ticks.'],
        optimizer_updates=0, model_forward_calls=0, simulator_starts=0, test_used=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, report(args.root))


if __name__ == '__main__':
    main()
