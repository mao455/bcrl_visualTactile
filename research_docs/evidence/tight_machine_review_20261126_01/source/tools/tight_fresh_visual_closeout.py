"""Re-read complete visual-study trajectories with rotation-permitting contact diagnostics."""
import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np

from tools.tight_fresh_contact_height_coverage import causal_contact_height
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def contact_motion(directory, expected):
    numeric = directory / 'numeric.h5'
    if sha256(numeric) != expected['numeric_sha256'] or sha256(directory / 'result.json') != expected['result_sha256']:
        raise ValueError('Reviewed physical trajectory changed')
    with h5py.File(numeric) as stream:
        t = stream['episode/transition']
        pose = t['diagnostics_after/object_pose_w'][:]
        goal = t['diagnostics_before/goal_pose_w'][:]
        force = t['object_contact_force_w'][:]
        keypoint = t['native_keypoints_max_dist'][:]
        tolerance = t['native_success_diagnostic/insertion_tolerance_m'][:]
        subgoals = t['diagnostics_before/successes'][:]
        mask, reference = causal_contact_height(pose, force)
        if len(pose) != expected['steps'] or reference is None:
            raise ValueError('Complete clock and settled reference required')
        if 'dagger/control_source' in t and np.any(t['dagger/control_source'][:] != 0):
            raise ValueError('Expected wholly autonomous execution')
        two = ((np.linalg.norm(force, axis=-1) > .1).sum(-1) >= 2).all(-1)
        high = pose[:, 2] >= reference + .02
        # Only after the reference is fully observed. This diagnostic permits rotation.
        contact_high = two & high & (np.arange(len(pose)) >= 48)
        indices = np.flatnonzero(contact_high)
        runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []
        window_ends = np.flatnonzero(mask)
        candidates = np.flatnonzero(mask & (subgoals == 0))
        best = None
        if len(candidates):
            i = int(candidates[np.argmin(keypoint[candidates])])
            q, g = pose[i, 3:], goal[i, 3:]
            cosine = abs(float(q @ g)) / float(np.linalg.norm(q) * np.linalg.norm(g))
            best = dict(completed_tick=i + 1, active_goal_index=int(subgoals[i]),
                root_angle_rad=float(2 * np.arccos(np.clip(cosine, 0., 1.))),
                root_distance_m=float(np.linalg.norm(pose[i, :3] - goal[i, :3])),
                native_keypoint_distance_m=float(keypoint[i]), tolerance_m=float(tolerance[i]),
                both_substeps_two_finger_contact=bool(two[i]))
        return dict(first_contact_height_window_end_tick=int(window_ends[0]+1) if len(window_ends) else None,
            qualifying_window_ends=int(mask.sum()),
            contact_and_height_ticks_after_reference=int(contact_high.sum()),
            longest_contact_and_height_run_ticks=max(map(len, runs), default=0),
            elevated_without_two_finger_contact_ticks_after_reference=int((high & ~two & (np.arange(len(pose)) >= 48)).sum()),
            contact_and_height_at_final_tick=bool(contact_high[-1]),
            first_goal_best_keypoint_during_contact_height_window=best)


def run(study, complete_report, output):
    if output.exists():
        raise FileExistsError('Preserve previous review evidence')
    full = read(complete_report)
    if full.get('complete') is not True:
        raise ValueError('Full source, budget, model and execution audit required')
    registration = read(study/'registration.json')
    if full['registration_sha256'] != sha256(study/'registration.json'):
        raise ValueError('Report belongs to another study')
    source = Path(registration['source_evidence']['root'])/'development'
    cases = []
    for group, directory, rows in [('source', source, full['autonomous']['source']['cases']),
                                   ('study', study/'development', full['autonomous']['cases'])]:
        for case in rows:
            if not case['legal']:
                raise ValueError('This closeout requires the actual fully legal cohort')
            cases.append(dict(group=group, arm=case['arm'], role=case['role'], case=case['case'],
                steps=case['steps'], success=case['success'], reasons=case['reasons'],
                stable_window=case['settled_height_stable_grasp'], native_lift=case['native_lift'],
                alignment=case['native_alignment_first_tick'] is not None,
                near_goal=case['native_near_goal_first_tick'] is not None,
                subgoals=case['native_subgoals'], numeric_sha256=case['numeric_sha256'],
                result_sha256=case['result_sha256'],
                contact_motion=contact_motion(directory/case['case'], case)))
    totals = {}
    for arm in ('source', 'control', 'auxiliary'):
        totals[arm] = {}
        for role in ('A', 'B'):
            rows = [r for r in cases if r['arm'] == arm and r['role'] == role]
            if len(rows) != 4:
                raise ValueError('Expected exactly four original opportunities per cell')
            totals[arm][role] = dict(opportunities=len(rows),
                contact_height_cases=sum(r['contact_motion']['first_contact_height_window_end_tick'] is not None for r in rows),
                stable_window_cases=sum(r['stable_window'] for r in rows),
                native_lift_cases=sum(r['native_lift'] for r in rows),
                alignment_cases=sum(r['alignment'] for r in rows), near_goal_cases=sum(r['near_goal'] for r in rows),
                first_subgoal_cases=sum(r['subgoals'] > 0 for r in rows), successes=sum(r['success'] for r in rows),
                termination_reasons=dict(Counter(reason for r in rows for reason, happened in r['reasons'].items() if happened)))
    atomic_json(output, dict(version='visual_study_contact_closeout_v1', complete=True,
        full_report_sha256=sha256(complete_report), registration_sha256=sha256(study/'registration.json'),
        source_sha256={name:sha256(Path(name)) for name in
            ('tools/tight_fresh_visual_closeout.py', 'tools/tight_fresh_contact_height_coverage.py')},
        totals=totals, cases=cases, simulator_starts=0, optimizer_updates=0, test_used=False,
        semantics='Contact-height uses the existing causal 12-tick two-finger/two-substep/2cm rule. '
            'Relative palm rotation is unrestricted. Losing this condition or the old stable window does not prove dropping. '
            'The best keypoint snapshot uses the same tick and its active pre-action goal; root quaternion angle ignores shape symmetry. '
            'Source is an earlier cohort, and repeated reset opportunities are not exact hidden-state pairs. '
            'These descriptive diagnostics do not change native success or promote a model.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--complete-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.complete_report, args.output)
