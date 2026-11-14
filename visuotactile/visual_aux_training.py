"""Explicit auxiliary-training admission and bookkeeping; never starts a run."""
import json
import math
from pathlib import Path

import torch

from .compact import sha256
from .narrow_data import aggregate_family_partitions
from .position_control import continuous_image_contract, visual_auxiliary_image_contract
from .visual_pose_aux import (ROUTES, VisualPoseHead, load_pose_labels, parameter_partitions,
                             pose_loss, pose_metrics, _tensor_sha)


SPEC_VERSION = 'visual_pose_aux_training_v1'
CALIBRATION_VERSION = 'visual_pose_aux_calibration_v1'
ESTIMATOR = 'target_ratio_times_median_base_gradient_over_median_pose_gradient_v1'


def read(path):
    return json.loads(Path(path).read_text())


def calibration_ids(data):
    """Sorted first six original train families, including their accepted corrections."""
    families = aggregate_family_partitions(data)
    original = data['normalization_train_ids']
    if (len(original) != 30 or len(set(original)) != 30
            or not set(original) <= set(data['splits']['train'])):
        raise ValueError('Original30 training population required for calibration')
    original_families = {data['entries'][i]['family'] for i in original}
    if len(original_families) != 30:
        raise ValueError('Exactly thirty original training families required')
    chosen = set(sorted(original_families)[:6])
    ids = [i for i in data['splits']['train'] if data['entries'][i]['family'] in chosen]
    if any(families[data['entries'][i]['family']] != 'train' for i in ids):
        raise ValueError('Calibration contains a held-out family')
    return ids


def load_auxiliary_spec(root, role, continuation):
    """Missing spec means exact legacy behavior; malformed opt-in never falls back."""
    root = Path(root)
    path = root/role/'visual_auxiliary.json'
    if not path.exists() and not path.is_symlink():
        return None
    spec = read(path)
    expected = {'version', 'route', 'head_seed', 'loss_weight', 'calibration'}
    if (set(spec) != expected or spec['version'] != SPEC_VERSION or spec['route'] not in ROUTES
            or type(spec['head_seed']) is not int or not 0 <= spec['head_seed'] < 2**63
            or type(spec['loss_weight']) not in (float, int) or not math.isfinite(spec['loss_weight'])
            or spec['loss_weight'] <= 0 or not isinstance(spec['calibration'], dict)
            or set(spec['calibration']) != {'path', 'sha256'} or not continuation):
        raise ValueError('Explicit calibrated auxiliary training schema and v6 continuation required')
    calibration_path = Path(spec['calibration']['path'])
    if not calibration_path.is_absolute():
        calibration_path = root/calibration_path
    if sha256(calibration_path) != spec['calibration']['sha256']:
        raise ValueError('Auxiliary calibration receipt hash differs')
    receipt = read(calibration_path)
    if (receipt.get('version') != CALIBRATION_VERSION or receipt.get('passed') is not True
            or receipt.get('roles') != ['A', 'B'] or receipt.get('routes') != list(ROUTES)
            or receipt.get('shared_weight') is not True or receipt.get('head_seed') != spec['head_seed']
            or receipt.get('gradient_ratio_target') != .1 or receipt.get('estimator') != ESTIMATOR
            or receipt.get('loss_weight') != spec['loss_weight']):
        raise ValueError('Exact jointly calibrated A/B auxiliary weight and initialization required')
    for key in ('data_sha256', 'frontend_sha256', 'statistics_sha256'):
        values = receipt.get(key)
        if (not isinstance(values, dict) or set(values) != {'A', 'B'}
                or any(not isinstance(value, str) or len(value) != 64
                       or any(c not in '0123456789abcdef' for c in value) for value in values.values())):
            raise ValueError('Both calibrated A/B source/data/statistics identities are required')
    data = read(root/role/'state_data/data.json')
    if (receipt.get('data_sha256', {}).get(role) != sha256(root/role/'state_data/data.json')
            or receipt.get('frontend_sha256', {}).get(role) != continuation['sha256']
            or receipt.get('statistics_sha256', {}).get(role) != sha256(root/role/'state_data/statistics.pt')
            or set(receipt.get('target_window_contract', {})) != {'A', 'B'}
            or receipt['target_window_contract'][role] != continuation.get('target_window_contract')
            or receipt.get('subset_ids') != calibration_ids(data)
            or sha256(continuation['path']) != continuation['sha256']):
        raise ValueError('Calibration source, data, statistics or train-only family population differs')
    # Bind receipt bytes and all selected spec fields into the training identity.
    return dict(spec, spec_sha256=sha256(path), calibration=dict(
        path=str(calibration_path.resolve()), sha256=spec['calibration']['sha256']))


def attach_auxiliary_head(model, spec):
    """Call only AFTER strict copying the trained v6 policy, before EMA/optimizer."""
    if (hasattr(model, 'visual_pose_head') or model.contract not in
            [continuous_image_contract(s) for s in ('q_reached', 'u_expert')]):
        raise ValueError('Only explicit v6 policy to new auxiliary identity is supported')
    route, seed = spec['route'], spec['head_seed']
    model.visual_pose_head = VisualPoseHead(seed=seed).to(next(model.parameters()).device)
    model.contract = visual_auxiliary_image_contract(model.contract['supervision'], route)


def optimizer_groups(model):
    original, head = parameter_partitions(model, model.visual_pose_head)
    return [dict(params=original), dict(params=head)]


def prepare_pose_contexts(data, records):
    partitions = aggregate_family_partitions(data)
    selected = data['splits']['train']+data['splits']['validation']
    if set(records) != set(selected):
        raise ValueError('Auxiliary contexts must match actual train/validation records exactly')
    contexts, reports = {}, {}
    for split in ('train', 'validation'):
        allowed = {family for family,value in partitions.items() if value == split}
        for identity in data['splits'][split]:
            context, report = load_pose_labels(data['entries'][identity], split=split, allowed_families=allowed)
            if not torch.equal(context.time_s, records[identity]['time'].detach().cpu()):
                raise ValueError('Auxiliary pose and actual training history clocks differ')
            authority = report['authority_tensor_sha256']
            if authority:
                for name, source in (('supervision_mask', 'expert_label_valid'), ('control_segment', 'segment_id')):
                    if name not in records[identity] or _tensor_sha(records[identity][name]) != authority[source]:
                        raise ValueError('Actual action mask/segment differs from recorded expert authority: '+identity)
            contexts[identity], reports[identity] = context, report
    return contexts, reports


def supervised_visual_conditions(model, record):
    """One actual full-prefix encoding, with unchanged legal action-anchor filtering."""
    from .anchor_sampling import planning_anchors
    from .continuous_vision import encode_dense
    condition, anchors, visual = encode_dense(model, record, return_visual=True)
    permitted = planning_anchors(record).to(anchors.device)
    keep = torch.isin(anchors, permitted)
    if not keep.any():
        raise ValueError('No legal expert planning windows; perception labels do not authorize actions')
    if not torch.equal(visual['ticks'], torch.arange(len(record['time']), device=anchors.device)):
        raise ValueError('Every actual control tick must receive one auxiliary readout')
    return condition[keep], anchors[keep], visual


def auxiliary_episode_loss(model, record, visual, context, spec):
    if model.contract != visual_auxiliary_image_contract(model.contract['supervision'], spec['route']):
        raise ValueError('Auxiliary gradient route differs from explicit model identity')
    prediction = model.visual_pose_head(visual['readout'], route=spec['route'])
    parts = pose_loss(prediction, context, time_s=record['time'][visual['ticks']])
    return parts['loss']*spec['loss_weight'], parts


def empty_exposure():
    return dict(episode_updates=0, populations={}, phases={}, per_episode={})


def add_exposure(ledger, identity, report, original_ids):
    """Update only after a completed optimizer update; counts are actual doses."""
    ledger['episode_updates'] += 1
    population = 'original' if identity in original_ids else 'correction'
    row = ledger['populations'].setdefault(population, dict(episode_updates=0, observed_ticks=0, perception_ticks=0))
    row['episode_updates'] += 1
    row['observed_ticks'] += report['ticks']
    row['perception_ticks'] += report['valid_perception_ticks']
    ledger['per_episode'][identity] = ledger['per_episode'].get(identity, 0)+1
    for name, values in report['perception_exposure'].items():
        phase = ledger['phases'].setdefault(name, dict(observed_ticks=0, perception_ticks=0, invalid_perception_ticks=0))
        for key,value in values.items():
            phase[key] += value


def validate_exposure(ledger, reports, train_ids, original_ids, *, completed_epochs, updates):
    """Check complete-epoch resume/checkpoint doses against the admitted population."""
    if (type(completed_epochs) is not int or completed_epochs < 0
            or type(updates) is not int or updates != completed_epochs*len(train_ids)):
        raise ValueError('Auxiliary updates must equal completed full episode traversals')
    expected = empty_exposure()
    for identity in train_ids:
        for _ in range(completed_epochs):
            add_exposure(expected, identity, reports[identity], original_ids)
    if expected != ledger:
        raise ValueError('Auxiliary exposure differs from actual complete-epoch population')


@torch.no_grad()
def auxiliary_metrics(model, records, identities, contexts, spec):
    """Separate sensing diagnostics; never change the original action selection score."""
    model.eval()
    rows = []
    for identity in identities:
        record = records[identity]
        _, _, visual = supervised_visual_conditions(model, record)
        prediction = model.visual_pose_head(visual['readout'], route=spec['route'])
        result = pose_metrics(prediction, contexts[identity], time_s=record['time'][visual['ticks']])
        rows.append(dict(identity=identity, **result))
    valid = sum(row['valid_ticks'] for row in rows)
    orientations = sum(row['valid_orientation_predictions'] for row in rows)
    return dict(episodes=rows, valid_ticks=valid,
        invalid_perception_ticks=sum(row['invalid_perception_ticks'] for row in rows),
        degenerate_orientation_predictions=sum(row['degenerate_orientation_predictions'] for row in rows),
        position_vector_rmse_mm=(math.sqrt(sum(row['position_vector_rmse_mm']**2*row['valid_ticks']
            for row in rows if row['valid_ticks'])/valid) if valid else None),
        orientation6_mse=(sum(row['orientation6_mse']*row['valid_ticks'] for row in rows if row['valid_ticks'])/valid if valid else None),
        orientation_mean_degrees=(sum(row['orientation_mean_degrees']*row['valid_orientation_predictions']
            for row in rows if row['valid_orientation_predictions'])/orientations if orientations else None),
        scope='Current-position/orientation auxiliary diagnostics on actual histories; not autonomy or action supervision.')
