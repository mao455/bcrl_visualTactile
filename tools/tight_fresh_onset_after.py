"""Evaluate a new full pair on the exact successful recorded onset subsets.

Reuses validated current-DP caches; never recomputes the old pair or runs Kit.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

import h5py
import numpy as np

from tools.tight_fresh_onset_report import BINS, PARTS, array_sha, finish, read_case
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def validate_record(record, binding, reference, numeric):
    """Require the original complete history and exact legal comparison rows."""
    if record.get('cache_identity') != binding or record.get('identity') != reference['case']:
        raise ValueError('Current cache model/data binding differs')
    if record.get('history_source') != 'strictly_past_actual_executed_targets_including_student_prefix; never regenerated nominal labels':
        raise ValueError('Expected the complete actual student/expert history')
    mask = record['supervision_mask'].cpu().numpy().astype(bool)
    indices = np.flatnonzero(mask)
    if (array_sha(mask.astype(np.uint8)) != reference['legal_mask_sha256_uint8']
            or array_sha(indices.astype('<i8')) != reference['legal_indices_sha256_int64_le']):
        raise ValueError('Cache supervision rows differ from recorded onset report')
    target = record['target'].cpu().numpy().astype(np.float64)
    previous = record['robot'][:, 58:87].cpu().numpy().astype(np.float64)
    segment = record['control_segment'].cpu().numpy()
    if target.shape != (len(mask), 29) or previous.shape != target.shape:
        raise ValueError('Expected complete current cache targets and robot history')
    with h5py.File(numeric) as f:
        dagger = f['episode/transition/dagger']
        original_mask = dagger['expert_label_valid'][:].astype(bool)
        original_target = dagger['u_expert'][:].astype(np.float64)
        original_previous = f['episode/obs/actuator_target_rad'][:].astype(np.float64)
        if not np.array_equal(mask, original_mask):
            raise ValueError('Cache legal supervision mask differs')
        if not np.array_equal(segment, dagger['segment_id'][:]):
            raise ValueError('Cache control segment differs')
        if not np.array_equal(target[mask], original_target[mask]):
            raise ValueError('Cache legal expert labels differ')
        if not np.array_equal(previous, original_previous):
            raise ValueError('Cache actual previous-target history differs')
    if not np.isfinite(target[mask]).all() or not np.isfinite(previous).all():
        raise ValueError('Nonfinite onset targets/history')
    return indices, target, previous


def comparison_bins(prediction, target, previous, indices, old_bins):
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError('Expected a finite full-episode prediction')
    bins = {}
    for name, rows in zip(BINS, (indices[:12], indices[12:24], indices[24:])):
        if len(rows) != old_bins[name]['ticks']:
            raise ValueError('Comparison bin population differs')
        sums = {part: [float(np.square(prediction[rows, section] - target[rows, section]).sum()),
                       float(np.square(previous[rows, section] - target[rows, section]).sum())]
                for part, section in PARTS.items()}
        new = finish(sums, len(rows))
        for part in PARTS:
            if new[part]['previous_target_squared_error_rad2'] != old_bins[name][part]['previous_target_squared_error_rad2']:
                raise ValueError('Compared previous-target baseline differs')
        bins[name] = {'before': old_bins[name], 'after': new}
    return bins


def aggregate(cases):
    result = {}
    for name in BINS:
        result[name] = {}
        for when in ('before', 'after'):
            ticks = sum(case['bins'][name][when]['ticks'] for case in cases)
            sums = {part: [sum(case['bins'][name][when][part][key] for case in cases)
                           for key in ('shadow_squared_error_rad2', 'previous_target_squared_error_rad2')]
                    for part in PARTS}
            result[name][when] = finish(sums, ticks)
        result[name]['improved_cases'] = {
            part: sum(case['bins'][name]['after'][part]['shadow_rmse_rad'] < case['bins'][name]['before'][part]['shadow_rmse_rad']
                      for case in cases if case['bins'][name]['after']['ticks'])
            for part in PARTS}
    return result


def role_inputs(root, role, plan, references):
    import torch
    directory = root / role
    data_path = directory / 'state_data/data.json'
    data = read(data_path)
    identities = [case['name'] for case in plan['cases']
                  if case['role'] == role and references[case['name']]['expert_takeover_success']]
    all_successes = {name for name, row in references.items() if row['expert_takeover_success']}
    if (set(data.get('aggregation', {}).get('new_training_ids', [])) != all_successes
            or not set(identities) <= set(data['splits']['train'])
            or set(identities) & set(data['splits']['validation'] + data['splits'].get('test', []))):
        raise ValueError('Actual successful training-correction population differs')
    continuation = read(directory / 'continuation.json')
    pair = plan['pairs'][role]
    if (continuation['sha256'] != pair['frontend_sha256']
            or continuation['previous_backend_sha256'] != pair['backend_sha256']):
        raise ValueError('New pair did not continue the recorded old full pair')
    front_path = directory / 'training/best.pt'
    rear_path = directory / 'backend' / role / 'training/best.pt'
    index_path = directory / 'backend' / role / 'nominal/index.json'
    finished_path = rear_path.parent / 'finished.json'
    finished = read(finished_path)
    if not finished.get('target_reached') or finished.get('disposable'):
        raise ValueError('Current backend formal training is incomplete')
    index = read(index_path)
    saved = torch.load(rear_path, map_location='cpu', weights_only=False)
    config = saved['configuration']
    statistics_path = directory / 'state_data/statistics.pt'
    residual_statistics = Path(config['residual_statistics_path'])
    binding = dict(base_checkpoint_sha256=sha256(front_path), statistics_sha256=sha256(statistics_path),
        data_sha256=sha256(data_path), residual_statistics_sha256=sha256(residual_statistics))
    keys = {'base_sha256': 'base_checkpoint_sha256', 'statistics_sha256': 'statistics_sha256',
            'data_sha256': 'data_sha256', 'residual_statistics_sha256': 'residual_statistics_sha256'}
    if any(index.get(k) != binding[v] or config.get(k) != binding[v] for k, v in keys.items()):
        raise ValueError('Current selected pair/cache/statistics identity differs')
    if config.get('cache_index_sha256') != sha256(index_path):
        raise ValueError('Backend was trained using a different cache index')
    if saved['contract'].get('visual_stride') != 1 or saved.get('updates', 0) < 1:
        raise ValueError('Expected trained full-frame student backend')
    from visuotactile.fresh_contracts import residual_normalization
    expected_normalization = residual_normalization(data, config)
    if (config.get('residual_normalization_identity') != expected_normalization
            or index.get('residual_normalization_identity') != expected_normalization
            or continuation['statistics_sha256'] != binding['statistics_sha256']):
        raise ValueError('Fixed original-training normalization differs')
    command_statistics = torch.load(residual_statistics, map_location='cpu', weights_only=False)
    front_statistics = torch.load(statistics_path, map_location='cpu', weights_only=False)
    for key in ('robot_mean', 'robot_std', 'action_scale', 'action_offset'):
        if not torch.equal(saved['model'][key], command_statistics[key]):
            raise ValueError('Current backend normalization buffer differs')
    for key in ('robot_mean', 'robot_std'):
        if not torch.equal(front_statistics[key], command_statistics[key]):
            raise ValueError('Current frontend/backend robot normalization differs')
    # The cached plans must use the same public sampling contract as live control.
    old = torch.load(pair['backend'], map_location='cpu', weights_only=False)
    sampler = index['sampler']
    if (sampler.get('seed') != old['contract'].get('sampler_seed', 7093)
            or sampler.get('seed') != saved['contract'].get('sampler_seed', 7093)
            or sampler.get('steps') != 16 or sampler.get('commit') != 12 or sampler.get('horizon') != 24):
        raise ValueError('Old live and new cached diffusion sampling contracts differ')
    del old
    return saved, data, identities, index_path, index, binding, dict(
        frontend_sha256=binding['base_checkpoint_sha256'], backend_sha256=sha256(rear_path),
        statistics_sha256=binding['statistics_sha256'], residual_statistics_sha256=binding['residual_statistics_sha256'],
        data_sha256=binding['data_sha256'], index_sha256=sha256(index_path), finished_sha256=sha256(finished_path),
        sampler=sampler, identity_selection='Only successful own-role recorded roll-ins, exactly matching old shadow predictions')


def run(root, collection, before_path, output, *, preflight_only=False):
    import torch
    from visuotactile.anchor_continue import residual_prediction
    from visuotactile.models.native_dp import load_residual
    from visuotactile.precision import state_fp32
    from visuotactile.state_resources import lease_for
    if output.exists():
        raise FileExistsError('Preserve earlier diagnostic outputs')
    before = read(before_path)
    plan = read(collection / 'plan.json')
    if (before.get('version') != 'recorded_full_student_recovery_onset_v1'
            or before['plan_sha256'] != sha256(collection / 'plan.json')
            or before['results_sha256'] != sha256(collection / 'results.json')
            or before.get('test_used') or plan.get('test_used') or plan.get('kind') != 'takeover'):
        raise ValueError('Expected the unchanged completed training-only onset collection')
    references = {row['case']: row for row in before['cases']}
    if len(references) != len(plan['cases']) or set(references) != {case['name'] for case in plan['cases']}:
        raise ValueError('Onset report and registered collection population differ')
    for pair in plan['pairs'].values():
        for name in ('frontend', 'backend'):
            if sha256(Path(pair[name])) != pair[name + '_sha256']:
                raise ValueError('Recorded old model identity changed')
    for case in plan['cases']:
        if read_case(collection, case, plan['pairs'][case['role']]) != references[case['name']]:
            raise ValueError('Recorded old prediction or comparison rows changed')
    output.mkdir(parents=True)
    sources = {str(p): sha256(p) for folder in ('tools', 'visuotactile') for p in Path(folder).rglob('*.py')}
    atomic_json(output / 'registration.json', dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=sources, root=str(root), collection=str(collection), before_report_sha256=sha256(before_path),
        started=time.time(), preflight_only=preflight_only, optimizer_updates=0, simulator_starts=0, test_used=False))
    lease = None
    if not preflight_only:
        state_fp32()
        lease = lease_for(root, deadline=read(root / 'session.json')['deadline'])
    roles = {}
    for role in ('A', 'B'):
        if any(sha256(p) != value for p, value in sources.items()):
            raise ValueError('Diagnostic source drift')
        saved, data, identities, index_path, index, binding, identity = role_inputs(root, role, plan, references)
        model = None
        if not preflight_only:
            with lease.block():
                model = load_residual(saved, 'cuda:0').eval().requires_grad_(False)
        cases = []
        for name in identities:
            cache_path = index_path.parent / (name + '.pt')
            cache_sha = sha256(cache_path)
            if cache_sha != index['episodes'][name]['sha256']:
                raise ValueError('Current nominal cache file changed')
            entry = data['entries'][name]
            if sha256(Path(entry['path'])) != entry['source_sha256']:
                raise ValueError('Underlying admitted correction data changed')
            record = torch.load(cache_path, map_location='cpu', weights_only=False)
            rows, target, previous = validate_record(record, binding, references[name], collection / name / 'numeric.h5')
            case = dict(case=name, numeric_sha256=references[name]['numeric_sha256'], cache_sha256=cache_sha,
                admitted_source_sha256=entry['source_sha256'], legal_indices_sha256_int64_le=array_sha(rows.astype('<i8')),
                legal_mask_sha256_uint8=references[name]['legal_mask_sha256_uint8'], legal_ticks=len(rows))
            if model is not None:
                prediction = residual_prediction(model, record, lease).cpu().numpy()
                case['bins'] = comparison_bins(prediction, target, previous, rows, references[name]['bins'])
                case['prediction_sha256_float32_le'] = array_sha(prediction.astype('<f4'))
            cases.append(case)
            atomic_json(output / 'progress.json', dict(role=role, identity=name, completed=len(cases), utc=time.time()))
        roles[role] = dict(identity=identity, cases=cases)
        if not preflight_only:
            roles[role]['bins'] = aggregate(cases)
        atomic_json(output / (role + '.json'), roles[role])
        del model, saved
    atomic_json(output / ('preflight.json' if preflight_only else 'result.json'), dict(
        complete=True, preflight_only=preflight_only, roles=roles, before_report_sha256=sha256(before_path),
        plan_sha256=sha256(collection / 'plan.json'),
        scope='Same recorded actual student prefixes and expert suffixes; only legal executed expert labels, first12/next12/after24 matched exactly. Before is the recorded old full-student shadow; after reuses exact current-DP cached plans/features and replays the selected new backend on complete actual histories. Both use the same recorded family and mask; learned histories are recomputed by the new models. These successful corrections entered new training, so this measures fitting, not independent generalization or autonomous success. No claim of bitwise equivalence between live raw-image inference and cached inference.',
        optimizer_updates=0, old_pair_forward_calls=0, simulator_starts=0, test_used=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true', help='CPU identities/cache/row checks only; no model inference')
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        run(args.root.resolve(), args.collection.resolve(), args.before.resolve(), args.output.resolve(), preflight_only=args.preflight_only)
    except BaseException as error:
        if not existed and args.output.exists():
            atomic_json(args.output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=time.time()))
        raise


if __name__ == '__main__':
    main()
