"""Read-only comparison around first detected fingertip/object contact."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_phases import ROOT_POSE_SEMANTICS


def read(path):
    return json.loads(path.read_text())


def rms(value):
    return float(np.sqrt(np.square(value.astype(np.float64)).mean()))


def case(directory, pair):
    result = read(directory / 'result.json')
    metadata = read(directory / 'metadata.json')
    if (not result.get('legal') or not result.get('software_complete')
            or result.get('complete_student_policy', pair) != pair
            or result['checkpoint_sha256'] != pair['backend_sha256']
            or result['residual_version'] != pair['residual_version']
            or result['residual_ticks'] != result['steps']):
        raise ValueError('Expected completed registered full-student autonomy')
    numeric = directory / 'numeric.h5'
    with h5py.File(numeric) as f:
        t = f['episode/transition']
        obj = t['diagnostics_after/object_pose_w'][:]
        palm = t['diagnostics_after/palm_body_pose_w'][:]
        actual = t['diagnostics_after/joint_position_rad'][:]
        command = t['target_after_rad'][:]
        previous = f['episode/obs/actuator_target_rad'][:]
        force = t['object_contact_force_w'][:]
        if (len(obj) != result['steps'] or not np.array_equal(command[:-1], previous[1:])
                or not all(np.isfinite(x).all() for x in (obj, palm, actual, command, force))):
            raise ValueError('Invalid recorded numeric continuity')
        if 'dagger/control_source' in t and (t['dagger/control_source'][:] != 0).any():
            raise ValueError('Expert actions cannot enter autonomous comparison')
        contact = np.linalg.norm(force, axis=-1) > .1
        first = np.flatnonzero(contact.any(axis=(1, 2)))
        output = dict(case=directory.name, steps=len(obj), success=result['success'],
            pose_reference_semantics=ROOT_POSE_SEMANTICS,
            native_subgoals=result['native_subgoals'], numeric_sha256=sha256(numeric),
            result_sha256=sha256(directory / 'result.json'),
            initial_observable_sha256=sha256(directory / 'initial_observable.json'),
            frontend_sha256=pair['frontend_sha256'], backend_sha256=pair['backend_sha256'])
        if not len(first):
            return dict(output, first_detected_contact_tick=None)
        onset = int(first[0])
        origin = t['diagnostics_before/object_pose_w'][onset]
        xy = np.linalg.norm(obj[:, :2] - origin[:2], axis=-1)
        crossed = np.flatnonzero((np.arange(len(obj)) >= onset) & (xy >= .01))
        two = (contact.sum(-1) >= 2).all(-1)
        tracking = command - actual
        rows = slice(onset, min(onset + 24, len(obj)))
        per_joint = np.sqrt(np.square(tracking[rows].astype(np.float64)).mean(axis=0))
        top = np.argsort(per_joint[7:])[-4:][::-1] + 7
        endpoint = min(onset + 23, len(obj) - 1)
        checkpoints = {}
        for count in (12, 24, 48):
            tick = min(onset + count - 1, len(obj) - 1)
            checkpoints[str(count)] = dict(completed_tick=tick + 1,
                object_horizontal_displacement_m=float(xy[tick]),
                object_vertical_displacement_m=float(obj[tick, 2] - origin[2]),
                palm_vertical_displacement_m=float(palm[tick, 2] - palm[onset, 2]),
                palm_object_center_distance_m=float(np.linalg.norm(palm[tick, :3] - obj[tick, :3])),
                palm_body_to_object_root_distance_m=float(np.linalg.norm(palm[tick, :3] - obj[tick, :3])),
                object_contact_fingers_per_physics_sample=contact[tick].sum(-1).tolist(),
                observed_ticks=tick - onset + 1)
        return dict(output, first_detected_contact_tick=onset + 1,
            first_post_contact_10mm_horizontal_displacement_tick=int(crossed[0] + 1) if len(crossed) else None,
            checkpoints=checkpoints,
            initial24=dict(ticks=endpoint - onset + 1,
                two_fingers_both_physics_samples_ticks=int(two[rows].sum()),
                arm_tracking_rmse_rad=rms(tracking[rows, :7]),
                hand_tracking_rmse_rad=rms(tracking[rows, 7:]),
                hand_command_endpoint_change_rms_rad=rms(command[endpoint, 7:] - command[onset, 7:]),
                hand_actual_endpoint_change_rms_rad=rms(actual[endpoint, 7:] - actual[onset, 7:]),
                highest_hand_tracking_errors=[dict(joint=metadata['joint_order'][i],
                    tracking_rmse_rad=float(per_joint[i]),
                    mean_target_minus_actual_rad=float(tracking[rows, i].astype(np.float64).mean())) for i in top]),
            preceding10_hand_tracking_rmse_rad=rms(tracking[max(0, onset - 10):onset, 7:]) if onset else None)


def report(before, after):
    import torch
    plans = {name: read(path / 'plan.json') for name, path in (('before', before), ('after', after))}
    for plan in plans.values():
        for pair in plan['pairs'].values():
            if pair['controller'] != 'ResidualController':
                raise ValueError('Expected complete DP plus backend')
            for name in ('frontend', 'backend'):
                if sha256(Path(pair[name])) != pair[name + '_sha256']:
                    raise ValueError('Registered model identity changed')
            saved = torch.load(pair['backend'], map_location='cpu', weights_only=False)
            if saved['configuration']['base_sha256'] != pair['frontend_sha256']:
                raise ValueError('Backend and registered frontend differ')
    old_cases = {c['name']: c for c in plans['before']['cases']}
    new_cases = {c['name']: c for c in plans['after']['cases']}
    if set(old_cases) != set(new_cases):
        raise ValueError('Registered autonomous cohorts differ')
    cases = []
    for name in old_cases:
        old, new = old_cases[name], new_cases[name]
        if old['role'] != new['role'] or old['spec'] != new['spec']:
            raise ValueError('Registered autonomous reset specifications differ')
        cases.append(dict(case=name, role=old['role'],
            recorded_initial_observable_equal=read(before / name / 'initial_observable.json') == read(after / name / 'initial_observable.json'),
            before=case(before / name, plans['before']['pairs'][old['role']]),
            after=case(after / name, plans['after']['pairs'][new['role']])))
    return dict(version='first_detected_contact_motion_v1', source_sha256=sha256(Path(__file__)),
        before_plan_sha256=sha256(before / 'plan.json'), after_plan_sha256=sha256(after / 'plan.json'), cases=cases,
        tick_index='1-based completed control transitions; checkpoint12 means first detected contact transition through the following11 transitions',
        origin='Object pose immediately before first detected fingertip/object contact; no midair or settled-height reference used.',
        contact='Any of five recorded fingertip/object forces >0.1N in either native physics sample. Other hand-body contacts are not measured by this criterion, so it need not be the first physical collision.',
        interpretation='Recorded paired specifications and observable initial fields do not establish exact hidden simulator replay. Horizontal movement is descriptive, not a task failure threshold. Tracking error compares actual joint configuration to the same-transition executed target and can reflect contact as well as control response. No causal claim from these numerical differences.',
        model_forward_calls=0, optimizer_updates=0, simulator_starts=0, test_used=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, report(args.before, args.after))


if __name__ == '__main__':
    main()
