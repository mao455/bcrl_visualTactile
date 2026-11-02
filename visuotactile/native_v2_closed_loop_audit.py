"""Frozen expert shadow on actual student observations; no simulated actions."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .compact import sha256
from .native_v2_actor_audit import load_actor, normalized, predictions
from .native_v2_forensics import first_difference
from .training_resources import atomic_json


def error_metrics(actual, expected):
    difference = actual - expected
    return dict(commands=len(actual), arm_rmse=float(np.sqrt(np.mean(difference[:, :7]**2))),
                hand_rmse=float(np.sqrt(np.mean(difference[:, 7:]**2))))


def audit(root, output):
    output.mkdir(parents=True, exist_ok=False)
    session = json.loads((root/'session.json').read_text())
    checkpoint = Path(session['expert_checkpoint'])
    if sha256(checkpoint) != session['expert_sha256']:
        raise ValueError('Frozen expert checkpoint changed')
    network, weights = load_actor(checkpoint)
    reports = {}
    for kind in ('old_state', 'bc_state', 'bc_actor'):
        if not (root/kind/'result.json').exists():
            raise ValueError('Complete native cohort required: '+kind)
        reports[kind] = []
        for seed in session['single_seeds']:
            student_path = root/kind/f'seed_{seed}'/'lane_00.h5'
            expert_path = root/'expert1'/f'seed_{seed}'/'lane_00.h5'
            with h5py.File(student_path, 'r') as student, h5py.File(expert_path, 'r') as expert:
                group, reference = student['episode'], expert['episode']
                actor = torch.from_numpy(group['transition/expert_actor_input_diagnostic'][:])
                shadow = predictions(network, normalized(actor, weights)).numpy()
                command = group['transition/command_clipped_pre_delay'][:]
                common = min(len(command), len(reference['obs/step_id']))
                fields = dict(delay_index='transition/control_trace/delay_index',
                    actor_input='transition/expert_actor_input_diagnostic',
                    command='transition/command_clipped_pre_delay',
                    force='transition/applied_wrench/force_w_N',
                    torque='transition/applied_wrench/torque_w_Nm',
                    joint='transition/diagnostics_after/joint_position_rad',
                    object='transition/diagnostics_after/object_pose_w',
                    rng_before_step='transition/torch_rng_before_step')
                differences = {name: first_difference(group[path][:common], reference[path][:common])
                               for name, path in fields.items()}
                phase = np.zeros(len(command), dtype=np.int64)
                contact = group['obs/contact_substeps_w_N'][:]
                phase[np.linalg.norm(contact, axis=-1).reshape(len(command), -1).max(-1) > .1] = 1
                phase[np.asarray(group['transition/diagnostics_before/lifted_object'], dtype=bool)] = 2
                phase[np.asarray(group['transition/diagnostics_before/successes']) > 0] = 3
                phases = {}
                for index, name in enumerate(('pre_contact', 'contact_pre_lift', 'lift', 'after_subgoal')):
                    mask = phase == index
                    phases[name] = error_metrics(command[mask], shadow[mask]) if mask.any() else dict(commands=0)
                report = dict(seed=seed, actual_student_vs_frozen_expert_shadow=error_metrics(command, shadow),
                    first_command=error_metrics(command[:1], shadow[:1]), phases=phases,
                    same_seed_expert_event_differences=differences,
                    student_sha256=sha256(student_path), expert_sha256=sha256(expert_path))
                reports[kind].append(report)
                np.savez_compressed(output/f'{kind}_{seed}.npz', command=command, expert_shadow=shadow, phase=phase)
    atomic_json(output/'result.json', dict(software_complete=True, models=reports,
        checkpoint_sha256=sha256(checkpoint), optimizer_updates=0, simulator_actions=0,
        scope='Frozen recurrent expert observes actual student noisy/delayed Actor inputs from reset. '
              'Shadow commands never execute; this is conditional command disagreement, not demonstrated rescue. '
              'Equal initial seed does not pair subsequent RNG or state-dependent wrench events.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    audit(arguments.root, arguments.output)
