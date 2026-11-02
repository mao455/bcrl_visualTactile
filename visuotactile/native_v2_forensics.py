"""Read-only first-divergence analysis of all original V2 diagnostics."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .compact import sha256
from .training_resources import atomic_json


def configuration_diff(source, current, prefix=''):
    differences = []
    for key in sorted(set(source) | set(current)):
        before, after = source.get(key), current.get(key)
        name = prefix + key
        if isinstance(before, dict) and isinstance(after, dict):
            differences.extend(configuration_diff(before, after, name + '.'))
        elif before != after:
            row = dict(field=name, source=before, evaluation=after)
            if isinstance(before, str) and isinstance(after, str) and Path(before).is_file() and Path(after).is_file():
                row.update(source_sha256=sha256(before), evaluation_sha256=sha256(after))
            differences.append(row)
    return differences


def first_difference(actual, expected, tolerance=0.):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f'Unaligned comparison: {actual.shape} != {expected.shape}')
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    indices = np.argwhere(difference > tolerance)
    first = None
    if len(indices):
        location = tuple(int(value) for value in indices[0])
        first = dict(tick=location[0], component=list(location[1:]),
                     source=float(expected[location]), evaluation=float(actual[location]),
                     absolute_error=float(difference[location]))
    return dict(first=first, maximum=float(difference.max()), tolerance=tolerance)


def contact_start(values, threshold=.1):
    norms = np.linalg.norm(values, axis=-1)
    hits = np.argwhere(norms > threshold)
    return None if not len(hits) else dict(tick=int(hits[0, 0]), substep=int(hits[0, 1]), finger=int(hits[0, 2]))


def audit(root, output):
    output.mkdir(parents=True, exist_ok=False)
    session = json.loads((root/'session.json').read_text())
    data = json.loads((Path(session['source_root'])/'data.json').read_text())
    environment = json.loads((root/'diagnostics/environment.json').read_text())
    results = json.loads((root/'diagnostics/result.json').read_text())['trials']
    mapping = dict(command=('command_clipped_pre_delay', 3e-6),
                   delayed=('control_trace/delayed_command', 3e-6),
                   delay_index=('control_trace/delay_index', 0.), target=('target_after_rad', 3e-6),
                   joint=('diagnostics_after/joint_position_rad', 1e-3),
                   velocity=('diagnostics_after/joint_velocity_rad_s', 1e-2),
                   object=('diagnostics_after/object_pose_w', 0.),
                   force=('applied_wrench/force_w_N', 1e-5), torque=('applied_wrench/torque_w_Nm', 1e-6),
                   contact=('contact_substeps_after_w_N', .1))
    rows = []
    student_rows = []
    for result in results:
        directory = root/'diagnostics'/result['trial_name']
        restoration = json.loads((directory/'restoration.json').read_text())
        trace = dict(np.load(directory/'trace.npz'))
        if result['kind'] != 'replay':
            comparison = dict(np.load(directory/'expert_comparison.npz'))
            plans = dict(np.load(directory/'plans.npz'))
            valid = np.isfinite(plans['expert_history_shadow']).all(axis=(1, 2))
            delta = plans['student'][valid, :12] - plans['expert_history_shadow'][valid, :12]
            student_rows.append(dict(identity=result['identity'], name=result['trial_name'],
                success=result['success'], reasons=result['reasons'], steps=len(trace['command']),
                restoration=restoration, trace_sha256=sha256(directory/'trace.npz'),
                first_divergence=result['first_divergence'],
                shadow_comparable_anchors=int(valid.sum()),
                shadow_vs_student_arm_rmse=float(np.sqrt(np.mean(delta[..., :7]**2))),
                shadow_vs_student_hand_rmse=float(np.sqrt(np.mean(delta[..., 7:]**2))),
                scope='Shadow changes observed trajectory and history together; not a history-only causal intervention',
                finite_trace=all(np.isfinite(value).all() for value in trace.values()),
                compared_ticks=len(comparison['command_arm_rmse'])))
            continue
        entry = data['entries'][result['identity']]
        with h5py.File(entry['path'], 'r') as stream:
            metadata = json.loads(stream.attrs['metadata_json'])
            group = stream['episodes'][entry['episode']]
            length = len(trace['command'])
            source_origin = np.asarray(metadata['env_origins_w'][0])
            evaluation_origin = np.zeros(3)
            comparisons = {}
            source_contact = np.asarray(group['transition/contact_substeps_after_w_N'][:length])
            for name, (field, tolerance) in mapping.items():
                expected = np.asarray(group['transition/'+field][:length])
                if name == 'object':
                    world = expected.copy()
                    expected = expected.astype(np.float64)
                    expected[:, :3] += evaluation_origin - source_origin
                    reconstructed_world = trace[name].astype(np.float64).copy()
                    reconstructed_world[:, :3] += source_origin - evaluation_origin
                    comparisons['object_source_world'] = first_difference(reconstructed_world, world)
                comparisons[name] = dict(exact=first_difference(trace[name], expected),
                                         threshold=first_difference(trace[name], expected, tolerance))
            row = dict(identity=result['identity'], trial=result['trial_name'], source_lane=metadata['source_env_index'],
                evaluation_lane=0, source_envs=metadata['num_parallel_envs'], evaluation_envs=1,
                source_origin=source_origin.tolist(), evaluation_origin=evaluation_origin.tolist(),
                source_global_tick=metadata.get('initial_global_control_tick'),
                source_length=result['source_length'], steps=result['steps'], success=result['success'],
                outcome='success' if result['success'] else 'actions_exhausted_censored' if result['recorded_actions_exhausted']
                        else '+'.join(key for key, value in result['reasons'].items() if value),
                source_first_fingertip_contact=contact_start(source_contact),
                evaluation_first_fingertip_contact=contact_start(trace['contact']), comparisons=comparisons,
                restoration=restoration, configuration_differences=configuration_diff(metadata['config'], environment['config']),
                source_h5_sha256=sha256(entry['path']), trace_sha256=sha256(directory/'trace.npz'),
                missing=['Historical actual PhysX masses/inertias/material/COM snapshots',
                         'Other three lanes and complete shared random stream', 'Solver/broadphase/contact warm-start',
                         'Expert observation/object-state queues', 'Object-specific historical contact identity'])
            atomic_json(output/(result['trial_name']+'.json'), row)
            rows.append(row)
    atomic_json(output/'summary.json', dict(replays=rows, students=student_rows, read_trials=len(results),
        scope='Zero-based post-control tick and chronological 120Hz substep; aggregate finger contact is not all-body collision'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for row in rows:
        directory = root/'diagnostics'/row['trial']
        trace = dict(np.load(directory/'trace.npz'))
        entry = data['entries'][row['identity']]
        with h5py.File(entry['path'], 'r') as stream:
            group = stream['episodes'][entry['episode']]
            length = len(trace['command'])
            joint = np.asarray(group['transition/diagnostics_after/joint_position_rad'][:length])
            pose = np.asarray(group['transition/diagnostics_after/object_pose_w'][:length], dtype=np.float64)
            pose[:, :3] -= row['source_origin']
            label = row['identity'].split('_', 2)[-1] + ' ' + row['outcome']
            axes[0].semilogy(np.maximum(np.abs(trace['joint']-joint).max(-1), 1e-9), label=label)
            axes[1].semilogy(np.maximum(np.linalg.norm(trace['object'][:, :3]-pose[:, :3], axis=-1), 1e-9))
    axes[0].set_ylabel('Max joint difference (rad)'); axes[1].set_ylabel('Object position difference (m)')
    axes[1].set_xlabel('Native control tick (60 Hz)'); axes[0].legend(fontsize=7)
    figure.tight_layout(); figure.savefig(output/'historical_divergence.png', dpi=150); plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.root, arguments.output)
