"""Prepare fresh shared-command backends, bound to their current full-frame DP.

This prepares a concrete stage. It does not choose the next research experiment.
"""
import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from visuotactile.fresh_contracts import FRESH_DATA_CONTRACT, residual_normalization


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root, role, epochs):
    import torch
    from visuotactile.position_control import REACHABLE_POSITION_EXECUTION
    from visuotactile.execution_learning import SPAN_PROJECTION_FY_LOSS
    from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT
    from visuotactile.residual_tick_window import TICK_WINDOW_CONTRACT
    from visuotactile.models.native_dp import TargetResidual, CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION
    if epochs < 1:
        raise ValueError('Positive shared training exposure required')
    root = root.resolve()
    source = root / role
    destination = source / 'backend'
    if destination.exists():
        raise FileExistsError('Preserve existing backend; use explicit continuation identity')
    data = read(source / 'state_data/data.json')
    command_data = read(root / 'B/state_data/data.json')
    if data['splits'] != command_data['splits']:
        raise ValueError('A/B cohort differs')
    front = source / 'training/best.pt'
    saved = torch.load(front, map_location='cpu', weights_only=False)
    if (saved['updates'] < 1 or saved['contract'].get('visual_stride') != 1
            or saved['contract'].get('supervision') != ('q_reached' if role == 'A' else 'u_expert')):
        raise ValueError('Expected trained fresh role-specific full-frame frontend')
    if saved['configuration']['statistics_sha256'] != sha256(source / 'state_data/statistics.pt'):
        raise ValueError('Selected frontend normalizer differs')
    stats = root / 'B/state_data/statistics.pt'
    v = dict(version=TargetResidual.version, fresh_data_contract=FRESH_DATA_CONTRACT,
        groups=[role], source_root=str(source), bases={role: dict(group='G00', modality='G00',
            checkpoint=str(front), sha256=sha256(front))},
        residual_statistics_path=str(stats), residual_statistics_sha256=sha256(stats),
        initialization_seed=2026111404, fixed_residual_scales=[1.] * 29,
        execution_adapter=REACHABLE_POSITION_EXECUTION, loss_contract=SPAN_PROJECTION_FY_LOSS,
        history_contract=dict(version=CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION),
        output_contract=REACHABLE_CENTERED_OUTPUT, tick_window_contract=TICK_WINDOW_CONTRACT,
        learning_rate=1e-4, weight_decay=1e-4, microbatch=4,
        maximum_control_tick_epochs=epochs, minimum_control_tick_epochs=epochs,
        validation_epoch_interval=1, comparison_control_tick_epochs=[epochs],
        cache_transfer_mode='episode', cache_scheduler_index_mode='host',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p): sha256(p) for folder in ['visuotactile', 'tools']
            for p in sorted(Path(folder).rglob('*.py'))})
    v['residual_normalization_identity'] = residual_normalization(data, v)
    if command_data['normalization_identity'] != v['residual_normalization_identity']:
        raise ValueError('Fresh common command affine required')
    session = copy.deepcopy(read(root / 'session.json'))
    session.update(residual_pair=v, base_contracts={role: saved['contract']}, source_root=str(source),
        cache_deadline=session['deadline'], training_deadline=session['deadline'],
        offline_deadline=session['deadline'], source_head=v['source_commit'])
    protected = {str(p): sha256(p) for p in [front, stats, source / 'state_data/data.json',
        source / 'state_data/statistics.pt', source / 'cache_index.json']}
    destination.mkdir()
    (destination / 'state_data').symlink_to(source / 'state_data', target_is_directory=True)
    (destination / 'shared_resources.json').symlink_to(root / 'shared_resources.json')
    atomic_json(destination / 'session.json', session)
    atomic_json(destination / 'configuration.json', dict(residual_pair=v))
    atomic_json(destination / 'base_identities.json', dict(protected=protected))
    child = destination / role
    child.mkdir()
    (child / 'state_data').symlink_to(source / 'state_data', target_is_directory=True)
    (child / 'nominal').mkdir()
    (child / 'training').mkdir()
    atomic_json(child / 'configuration.json', dict(residual_pair=v))
    atomic_json(child / 'session.json', dict(session, resource_root=str(destination), modality=role))
    atomic_json(destination / 'prepared.json', dict(role=role, source_frontend_sha256=sha256(front),
        command_statistics_sha256=sha256(stats), random_initialization=True,
        inherited_student_parameters=False, train=30, validation=6, epochs=epochs, utc=time.time()))
    print(str(destination), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--role', choices=['A', 'B'], required=True)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    prepare(args.root, args.role, args.epochs)


if __name__ == '__main__':
    main()
