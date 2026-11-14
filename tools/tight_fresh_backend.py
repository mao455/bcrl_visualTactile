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
from visuotactile.fresh_contracts import FRESH_DATA_CONTRACT, FRESH_AGGREGATE_CONTRACT, residual_normalization


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root, role, epochs, *, continue_rear=False, fresh_recovery_weighting=None,
            output_contract=None, continuation_head_reset=None, offline_anchor_batch=None):
    import torch
    from visuotactile.position_control import REACHABLE_POSITION_EXECUTION
    from visuotactile.execution_learning import SPAN_PROJECTION_FY_LOSS
    from visuotactile.residual_output import (REACHABLE_CENTERED_OUTPUT,REACHABLE_PREVIOUS_TARGET_OUTPUT,
        validate_output_contract,validate_head_reset)
    from visuotactile.residual_tick_window import TICK_WINDOW_CONTRACT
    from visuotactile.models.native_dp import TargetResidual, CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION
    from visuotactile.fresh_recovery import fresh_recovery_weighting_config
    fresh_recovery_weighting=fresh_recovery_weighting_config(fresh_recovery_weighting)
    cache_planning=None
    if offline_anchor_batch is not None:
        from visuotactile.offline_planning import profile
        cache_planning=profile(offline_anchor_batch)
    output_contract=validate_output_contract(REACHABLE_CENTERED_OUTPUT if output_contract is None else output_contract)
    continuation_head_reset=validate_head_reset(continuation_head_reset)
    if continuation_head_reset is not None and not continue_rear:
        raise ValueError('Common head reset requires explicit backend continuation')
    if output_contract==REACHABLE_PREVIOUS_TARGET_OUTPUT and (not continue_rear or continuation_head_reset is None):
        raise ValueError('Previous-target comparison requires explicit common head reset and continuation')
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
    identity=FRESH_AGGREGATE_CONTRACT if data.get('aggregation') else FRESH_DATA_CONTRACT
    v = dict(version=TargetResidual.version, fresh_data_contract=identity,
        groups=[role], source_root=str(source), bases={role: dict(group='G00', modality='G00',
            checkpoint=str(front), sha256=sha256(front))},
        residual_statistics_path=str(stats), residual_statistics_sha256=sha256(stats),
        initialization_seed=2026111404, fixed_residual_scales=[1.] * 29,
        execution_adapter=REACHABLE_POSITION_EXECUTION, loss_contract=SPAN_PROJECTION_FY_LOSS,
        history_contract=dict(version=CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION),
        output_contract=output_contract, tick_window_contract=TICK_WINDOW_CONTRACT,
        learning_rate=1e-4, weight_decay=1e-4, microbatch=4, fresh_validation_count_metadata=True,
        maximum_control_tick_epochs=epochs, minimum_control_tick_epochs=epochs,
        validation_epoch_interval=1, comparison_control_tick_epochs=[epochs],
        cache_transfer_mode='episode', cache_scheduler_index_mode='host',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p): sha256(p) for folder in ['visuotactile', 'tools']
            for p in sorted(Path(folder).rglob('*.py'))})
    v['residual_normalization_identity'] = residual_normalization(data, v)
    if cache_planning is not None:v['cache_offline_planning_profile']=cache_planning
    if continuation_head_reset is not None:
        v['continuation_head_reset']=continuation_head_reset
    if fresh_recovery_weighting is not None:
        if not continue_rear or not data.get('aggregation'):
            raise ValueError('Fresh onset comparison requires an explicit aggregate backend continuation')
        v['fresh_recovery_weighting']=fresh_recovery_weighting
    if data.get('aggregation'):
        v['dagger_round']=max(e.get('round_index',0) for e in data['entries'].values())
    if command_data['normalization_identity'] != v['residual_normalization_identity']:
        raise ValueError('Fresh common command affine required')
    previous_rear = None
    if continue_rear:
        from visuotactile.anchor_continue import continuation_source, load_residual_continuation
        if not data.get('aggregation'):
            raise ValueError('Backend continuation requires an actual fresh DAgger aggregate')
        prior = read(source / 'continuation.json')
        previous_rear = Path(prior['previous_backend'])
        v['continuation_from'] = dict(path=str(previous_rear), sha256=prior['previous_backend_sha256'],
            teacher_supervision='u_expert', statistics_sha256=v['residual_statistics_sha256'],
            normalization_identity=v['residual_normalization_identity'],
            round_index=v['dagger_round'], source_group=role)
        prior_saved = continuation_source(v['continuation_from'], 'u_expert', stats, v['residual_normalization_identity'])
        if prior_saved['configuration']['base_sha256'] != prior['sha256']:
            raise ValueError('Prior backend was not paired with the inherited frontend')
        candidate = TargetResidual(torch.load(stats, map_location='cpu', weights_only=False), v['fixed_residual_scales'],
            execution_adapter=v['execution_adapter'], loss_contract=v['loss_contract'],
            history_contract=v['history_contract'], output_contract=v['output_contract'])
        load_residual_continuation(candidate, prior_saved, v['residual_statistics_sha256'],
            head_reset=continuation_head_reset)
        del candidate, prior_saved
    session = copy.deepcopy(read(root / 'session.json'))
    session.update(residual_pair=v, base_contracts={role: saved['contract']}, source_root=str(source),
        cache_deadline=session['deadline'], training_deadline=session['deadline'],
        offline_deadline=session['deadline'], source_head=v['source_commit'])
    protected = {str(p): sha256(p) for p in [front, stats, source / 'state_data/data.json',
        source / 'state_data/statistics.pt', source / 'cache_index.json']}
    if previous_rear is not None: protected[str(previous_rear)] = sha256(previous_rear)
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
        command_statistics_sha256=sha256(stats), random_initialization=not continue_rear,
        inherited_student_parameters=continue_rear, continuation_from=v.get('continuation_from'),
        **({'continuation_head_reset':continuation_head_reset} if continuation_head_reset is not None else {}),
        train=len(data['splits']['train']), validation=6, epochs=epochs, utc=time.time()))
    print(str(destination), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--role', choices=['A', 'B'], required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--continue-rear', action='store_true')
    parser.add_argument('--offline-anchor-batch',type=int,
        help='Explicit offline residual-cache DDIM anchor batch; default retains serial plans')
    args = parser.parse_args()
    prepare(args.root, args.role, args.epochs, continue_rear=args.continue_rear,offline_anchor_batch=args.offline_anchor_batch)


if __name__ == '__main__':
    main()
