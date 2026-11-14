"""CPU comparison of complete rears on identical, actually learned recovery histories.

Each candidate recomputes its own recurrent history. Frozen current-DP plans and
visual features are shared only after checking the exact model/data identities.
This is a fitting diagnostic, never an autonomous rollout or a held-out test.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import torch

from tools.tight_fresh_onset_report import array_sha
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(path.read_text())


class CPUAnalysis:
    def block(self):
        return nullcontext()


def error_sum(prediction, target, mask):
    if (prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 29
            or mask.shape != prediction.shape[:1] or mask.dtype != torch.bool
            or not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all()):
        raise ValueError('Finite matching absolute-radian predictions and legal tick mask required')
    error = (prediction.double() - target.double())[mask]
    return dict(ticks=len(error), arm_squared_error_rad2=float(error[:, :7].square().sum()),
                hand_squared_error_rad2=float(error[:, 7:].square().sum()))


def aggregate(rows):
    ticks = sum(r['ticks'] for r in rows)
    result = dict(ticks=ticks)
    for part, dofs in [('arm', 7), ('hand', 22)]:
        squared = sum(r[part + '_squared_error_rad2'] for r in rows)
        result[part + '_rmse_rad'] = math.sqrt(squared / (ticks * dofs)) if ticks else None
    return result


def command_motion_sum(prediction, target, previous, mask):
    """Diagnose a hold-target shortcut using the exact pre-action target, in rad."""
    error_sum(prediction, target, mask)
    if previous.shape != target.shape or not torch.isfinite(previous[mask]).all():
        raise ValueError('Finite aligned actual pre-action targets required')
    predicted = (prediction.double()-previous.double())[mask]
    actual = (target.double()-previous.double())[mask]
    parts = {}
    for part, selection in [('arm', slice(0,7)), ('hand', slice(7,29))]:
        p, a = predicted[:,selection], actual[:,selection]
        active = a.abs()>1e-6
        parts[part] = dict(predicted_delta_squared_rad2=float(p.square().sum()),
            target_delta_squared_rad2=float(a.square().sum()),
            effective_gt_components=int(active.sum()),
            same_direction_components=int(((p*a>0)&active).sum()))
    return dict(ticks=len(actual), parts=parts)


def aggregate_command_motion(rows):
    ticks = sum(r['ticks'] for r in rows)
    parts = {}
    for part, dofs in [('arm',7), ('hand',22)]:
        predicted = sum(r['parts'][part]['predicted_delta_squared_rad2'] for r in rows)
        actual = sum(r['parts'][part]['target_delta_squared_rad2'] for r in rows)
        active = sum(r['parts'][part]['effective_gt_components'] for r in rows)
        same = sum(r['parts'][part]['same_direction_components'] for r in rows)
        parts[part] = dict(predicted_command_delta_rms_rad=math.sqrt(predicted/(ticks*dofs)) if ticks else None,
            actual_expert_command_delta_rms_rad=math.sqrt(actual/(ticks*dofs)) if ticks else None,
            hold_previous_target_rmse_rad=math.sqrt(actual/(ticks*dofs)) if ticks else None,
            effective_gt_components=active, same_direction_components=same,
            same_direction_fraction=same/active if active else None)
    return dict(ticks=ticks, parts=parts)


def recovery_bins(record):
    from visuotactile.anchor_sampling import recovery_prefix_rows, supervision_rows
    mask, _ = supervision_rows(record)
    first = recovery_prefix_rows(record, 12).bool()
    first24 = recovery_prefix_rows(record, 24).bool()
    bins = dict(first12=first, next12=first24 & ~first, remaining=mask & ~first24)
    if (not mask.any() or not torch.equal(torch.stack(list(bins.values())).sum(0), mask.long())):
        raise ValueError('Recovery bins must partition legal supervision exactly once')
    return bins


def correction_cohorts(data, study_version):
    """Derive exact fitting populations from the registered training manifest."""
    from visuotactile.narrow_data import aggregate_family_partitions
    families = aggregate_family_partitions(data)
    train = data['splits']['train']
    original = data['normalization_train_ids']
    current = data['aggregation']['new_training_ids']
    if (len(original) != 30 or len(set(original)) != 30 or not set(original) <= set(train)
            or len(set(current)) != len(current)):
        raise ValueError('Original30 or current correction population differs')
    corrections = sorted(set(train) - set(original))
    original_families = {data['entries'][identity]['family'] for identity in original}
    if len(original_families) != 30:
        raise ValueError('Original30 must retain distinct admitted training families')
    if not corrections or not current or not set(current) <= set(corrections):
        raise ValueError('Nonempty actual training corrections and current subset required')
    for identity in corrections:
        entry = data['entries'][identity]
        if (entry.get('trajectory_kind') != 'expert_takeover_v1'
                or families[entry['family']] != 'train' or entry['family'] not in original_families):
            raise ValueError('Fitting requires actual admitted training takeovers')
    if study_version == 'matched_rear_output_anchor_study_v1':
        names = ('all_corrections', 'current_corrections')
    elif study_version is None:
        if len(corrections) != 53 or len(current) != 12:
            raise ValueError('Historical weight study requires its original53/new12 population')
        names = ('all53', 'new12')
    else:
        raise ValueError('Unknown registered study version')
    return {names[0]: corrections, names[1]: sorted(current)}


def run(study, arm, output):
    from visuotactile.anchor_continue import residual_prediction
    from visuotactile.models.native_dp import load_residual
    if any(p.exists() for p in (output, *(output.with_name(output.stem + '_' + role + '.json') for role in ('A', 'B')))):
        raise FileExistsError('Preserve earlier analysis')
    torch.set_num_threads(2)
    began = time.time()
    registration = read(study / 'registration.json')
    root = Path(registration['source_root']) if arm == 'source' else study / arm
    roles = {}
    for role in ('A', 'B'):
        directory = root / role
        source_pair = registration['source_pairs'][role]
        data_path = directory / 'state_data/data.json'
        data = read(data_path)
        numeric_path = directory / 'state_data/numeric.pt'
        numeric = torch.load(numeric_path, map_location='cpu', weights_only=False)
        if numeric['data_sha256'] != sha256(data_path):
            raise ValueError('Numeric population changed')
        index_path = directory / 'backend' / role / 'nominal/index.json'
        index = read(index_path)
        if (index['base_sha256'] != source_pair['frontend_sha256']
                or index['base_sha256'] != sha256(directory / 'training/best.pt')
                or index['data_sha256'] != sha256(data_path)):
            raise ValueError('Shared DP/cache/data identity differs')
        rear = directory / 'backend' / role / 'training'
        # Independent sessions inherit a read-only best/finished pair, not the
        # entire training directory. The registered source checkpoint identifies
        # its original validation-selection receipt without copying or changing it.
        selected_path = (Path(source_pair['backend']).resolve(strict=True).with_name('selected.json')
                         if arm == 'source' else rear / 'selected.json')
        selected = read(selected_path)
        rear_sha = sha256(rear / 'best.pt')
        if not read(rear / 'finished.json')['target_reached'] or selected['checkpoint_sha256'] != rear_sha:
            raise ValueError('Rear is incomplete or not validation-selected')
        saved = torch.load(rear / 'best.pt', map_location='cpu', weights_only=False)
        config = saved['configuration']
        if (config['base_sha256'] != index['base_sha256']
                or config['cache_index_sha256'] != sha256(index_path)
                or config['data_sha256'] != index['data_sha256']):
            raise ValueError('Rear was trained against different cached inputs')
        expected_rear = (source_pair['backend_sha256'] if arm == 'source'
                         else config['continuation_from']['sha256'])
        if expected_rear != source_pair['backend_sha256']:
            raise ValueError('Rear continuation differs from the registered source')
        if arm == 'source' and rear_sha != source_pair['backend_sha256']:
            raise ValueError('Source diagnostic does not use the registered selected rear')
        model = load_residual(saved, 'cpu').eval().requires_grad_(False)
        statistics = torch.load(Path(config['residual_statistics_path']), map_location='cpu', weights_only=False)
        if sha256(config['residual_statistics_path']) != config['residual_statistics_sha256']:
            raise ValueError('Command statistics changed')
        for key in ('robot_mean', 'robot_std', 'action_offset', 'action_scale'):
            if not torch.equal(saved['model'][key], statistics[key]):
                raise ValueError('Normalization buffer differs')
        cohorts = correction_cohorts(data, registration.get('version'))
        corrections, current_ids = cohorts.values()
        current = set(current_ids)
        cases = []
        for identity in corrections:
            path = index_path.parent / (identity + '.pt')
            digest = sha256(path)
            if digest != index['episodes'][identity]['sha256']:
                raise ValueError('Cached episode changed')
            record = torch.load(path, map_location='cpu', weights_only=False)
            actual = numeric['records'][identity]
            entry = data['entries'][identity]
            mask = record['supervision_mask'].bool()
            target = actual.get('u_expert', actual['command'])
            if (record['identity'] != identity
                    or entry.get('trajectory_kind') != 'expert_takeover_v1'
                    or record['cache_identity']['base_checkpoint_sha256'] != index['base_sha256']
                    or record['cache_identity']['data_sha256'] != index['data_sha256']
                    or not torch.equal(record['robot'], actual['robot'])
                    or not torch.equal(record['time'], actual['time'])
                    or not torch.equal(mask, actual['supervision_mask'].bool())
                    or not torch.equal(record['target'][mask], target[mask])
                    or not torch.equal(record['control_segment'], actual['control_segment'])):
                raise ValueError('Cached actual history or legal supervision differs')
            bins = recovery_bins(record)
            prediction = residual_prediction(model, record, CPUAnalysis())
            cases.append(dict(identity=identity, current_correction=identity in current,
                source_group=entry['source_group'], actual_history_ticks=len(record['robot']),
                cache_sha256=digest, legal_mask_sha256=array_sha(mask.numpy()),
                bin_mask_sha256={name: array_sha(selected_mask.numpy()) for name, selected_mask in bins.items()},
                prediction_sha256=array_sha(prediction.numpy()),
                bins={name: error_sum(prediction, record['target'], selected_mask)
                      for name, selected_mask in bins.items()},
                command_motion_bins={name: command_motion_sum(prediction, record['target'],
                    record['robot'][:,58:87], selected_mask) for name,selected_mask in bins.items()}))
        summaries = {}
        for cohort, identities in cohorts.items():
            chosen = [c for c in cases if c['identity'] in identities]
            summaries[cohort] = dict(episodes=len(chosen),
                source_group_counts={group: sum(c['source_group'] == group for c in chosen) for group in ('A', 'B')},
                bins={name: aggregate([c['bins'][name] for c in chosen])
                                                              for name in ('first12', 'next12', 'remaining')},
                command_motion_bins={name: aggregate_command_motion([c['command_motion_bins'][name] for c in chosen])
                    for name in ('first12','next12','remaining')})
        roles[role] = dict(frontend_sha256=index['base_sha256'], backend_sha256=rear_sha,
            data_sha256=sha256(data_path), numeric_sha256=sha256(numeric_path), cache_index_sha256=sha256(index_path),
            selected=selected, selected_receipt_sha256=sha256(selected_path),
            fitting_population=cohorts, summaries=summaries, cases=cases)
        atomic_json(output.with_name(output.stem + '_' + role + '.json'), roles[role])
    if torch.cuda.is_initialized():
        raise ValueError('CPU onset report unexpectedly initialized CUDA')
    atomic_json(output, dict(complete=True, arm=arm, roles=roles, seconds=time.time() - began,
        registration_sha256=sha256(study / 'registration.json'), source_sha256=sha256(__file__),
        error_contract='physical_command_rmse_first_contiguous_legal12_next12_remaining_v1',
        command_motion_contract=dict(version='actual_pre_action_target_delta_v1',
            previous='Same-tick actual pre-action target robot[58:87], rad; never shifted q or future target',
            effective_gt_delta_threshold_rad=1e-6, direction='Positive prediction_delta*expert_delta on effective GT components; zero prediction counts as not same-direction',
            scope='Already learned corrections in each exact fitting_population; hold baseline and command deltas do not establish autonomous motion or validation generalization'),
        dependency_sha256={str(path): sha256(path) for path in (
            Path(__file__).resolve().parents[1] / 'visuotactile' / name for name in
            ('anchor_continue.py', 'anchor_sampling.py', 'native_data.py', 'models/native_dp.py'))},
        device='cpu', cuda_initialized=False, threads=2, optimizer_updates=0, simulator_starts=0, test_used=False,
        scope='Same current-DP caches and actual student/expert histories; each selected rear recomputes complete history. '
              'Unweighted physical command errors on trained corrections, not independent validation or autonomous performance. '
              'All three candidates must be computed on CPU before comparing; do not mix earlier GPU measurements.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--arm', choices=['source', 'control', 'weighted', 'previous'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study.resolve(), args.arm, args.output.resolve())
