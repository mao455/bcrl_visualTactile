"""Stage-conditioned BC command errors on immutable expert observations."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .native_v2_bc_compare import ActorBC, features, actor_records
from .native_v2_state import StateBC, StateWindows
from .training_resources import atomic_json


def stage_labels(group):
    length = len(group['obs/step_id'])
    phase = np.zeros(length, dtype=np.int64)
    force = np.asarray(group['obs/contact_substeps_w_N'])
    phase[np.linalg.norm(force, axis=-1).reshape(length, -1).max(-1) > .1] = 1
    phase[np.asarray(group['transition/diagnostics_before/lifted_object'], dtype=bool)] = 2
    phase[np.asarray(group['transition/diagnostics_before/successes']) > 0] = 3
    phase[np.asarray(group['transition/diagnostics_before/retract_phase'], dtype=bool)] = 4
    return phase


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    session = json.loads((root/'session.json').read_text()); source = Path(session['source_root'])
    data = json.loads((source/'data.json').read_text()); reports = {}
    names = ['pre_contact', 'finger_contact_pre_lift', 'lift_before_first_subgoal', 'after_first_subgoal', 'retract']
    for kind in ('state', 'actor'):
        checkpoint = root/'bc_compare'/kind/f"milestone_{session['bc_updates']}.pt"
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = StateBC() if kind == 'state' else ActorBC()
        model.load_state_dict(saved['ema']); model.eval()
        reports[kind] = {}
        for split, identities in [('train4', session['bc_train_ids']), ('validation4', session['bc_validation_ids'])]:
            windows = StateWindows(source, identities)
            values = torch.cat((windows.robot, windows.geometry), -1) if kind == 'state' else features(
                windows, actor_records(source, identities), model.actor_mean, model.actor_std)
            with torch.no_grad():
                predictions = torch.cat([model.predict(values[start:start+32]).clamp(-1, 1)
                                         for start in range(0, len(windows), 32)])
            labels = {}
            for identity in identities:
                entry = data['entries'][identity]
                with h5py.File(entry['path'], 'r') as stream:
                    labels[identity] = stage_labels(stream['episodes'][entry['episode']])
            errors = {name: [] for name in names}
            for index, (identity, tick) in enumerate(windows.samples):
                width = min(12, len(labels[identity])-tick)
                difference = (predictions[index, :width]-windows.command[index, :width]).numpy()
                for offset in range(width):
                    errors[names[labels[identity][tick+offset]]].append(difference[offset])
            reports[kind][split] = {}
            for phase, differences in errors.items():
                if differences:
                    values = np.stack(differences)
                    reports[kind][split][phase] = dict(commands=len(values), arm_rmse=float(np.sqrt(np.mean(values[:, :7]**2))),
                                                       hand_rmse=float(np.sqrt(np.mean(values[:, 7:]**2))))
                else:
                    reports[kind][split][phase] = dict(commands=0, arm_rmse=None, hand_rmse=None)
    atomic_json(output/'result.json', dict(software_complete=True, models=reports,
        scope='Teacher-forced offline command errors by true source phase; not autonomous stage or complete success',
        phase_precedence=names, finger_contact='Aggregate fingertip forces >0.1N; source has no object-filtered contacts'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args(); torch.set_num_threads(2); run(arguments.root, arguments.output)
