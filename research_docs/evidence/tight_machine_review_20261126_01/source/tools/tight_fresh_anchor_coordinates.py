"""Read-only target-coordinate comparison for two possible rear output anchors.

Both maps can express the same legal box. This measures target magnitudes and
changes, not trained performance, gradients along a rollout, or a selected change.
"""
import argparse
import json
from pathlib import Path

import torch

from visuotactile.compact import sha256
from visuotactile.execution_learning import ideal_execution_bounds
from visuotactile.position_control import REACHABLE_POSITION_EXECUTION
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(path.read_text())


def summarize(rows):
    result = {}
    for name, section in (('arm', slice(0, 7)), ('hand', slice(7, 29))):
        metrics = {}
        for anchor in ('nominal', 'previous'):
            values = torch.cat([row[anchor][:, section] for row in rows])
            radians = torch.cat([row[anchor + '_rad'][:, section] for row in rows])
            changes = [row[anchor + '_change'][:, section] for row in rows]
            changes = torch.cat(changes)
            metrics[anchor] = dict(
                required_raw_rms=float(values.square().mean().sqrt()),
                required_raw_mean_abs=float(values.abs().mean()),
                required_raw_over_half_span_fraction=float((values.abs() > .5).double().mean()),
                zero_raw_command_rmse_rad=float(radians.square().mean().sqrt()),
                adjacent_legal_raw_change_rms=float(changes.square().mean().sqrt()) if len(changes) else None,
                adjacent_legal_pairs=len(changes))
        result[name] = metrics
    return dict(episodes=len(rows), legal_ticks=sum(len(row['nominal']) for row in rows), parts=result)


def run(root, output):
    if output.exists():
        raise FileExistsError('Preserve previous diagnostic')
    torch.set_num_threads(2)
    plan = read(root / 'development/plan.json')
    roles = {}
    for role in ('A', 'B'):
        directory = root / role
        data_path = directory / 'state_data/data.json'
        data = read(data_path)
        numeric_path = directory / 'state_data/numeric.pt'
        numeric = torch.load(numeric_path, map_location='cpu', weights_only=False)
        if numeric['data_sha256'] != sha256(data_path):
            raise ValueError('Numeric source identity differs')
        original = set(data['normalization_train_ids'])
        index_path = directory / 'backend' / role / 'nominal/index.json'
        index = read(index_path)
        front = Path(plan['pairs'][role]['frontend'])
        if (sha256(front) != plan['pairs'][role]['frontend_sha256']
                or index['base_sha256'] != plan['pairs'][role]['frontend_sha256']
                or index['data_sha256'] != sha256(data_path)):
            raise ValueError('Frozen DP/cache/data identity mismatch')
        buckets = {name: [] for name in ('original_train', 'correction_first12',
                   'correction_after12', 'validation')}
        bindings = []
        for identity in data['splits']['train'] + data['splits']['validation']:
            path = index_path.parent / (identity + '.pt')
            digest = sha256(path)
            if digest != index['episodes'][identity]['sha256']:
                raise ValueError('Changed dependent cache: ' + identity)
            record = torch.load(path, map_location='cpu', weights_only=False)
            if (record['identity'] != identity
                    or record['cache_identity']['base_checkpoint_sha256'] != index['base_sha256']
                    or record['cache_identity']['data_sha256'] != index['data_sha256']):
                raise ValueError('Wrong cached input identity')
            previous = record['robot'][:, 58:87]
            bounds = ideal_execution_bounds(previous, record['limits'], REACHABLE_POSITION_EXECUTION)
            lower, upper = bounds[..., 0].double(), bounds[..., 1].double()
            span = upper - lower
            mask = record['supervision_mask'].bool()
            target = record['target'].double()
            actual = numeric['records'][identity]
            actual_target = actual.get('u_expert', actual['command'])
            if (not torch.equal(record['robot'], actual['robot'])
                    or not torch.equal(target[mask], actual_target.double()[mask])
                    or not torch.equal(mask, actual.get('supervision_mask', torch.ones(len(mask), dtype=torch.bool)))):
                raise ValueError('Cache history or legal command labels differ from actual numeric data')
            legal = target[mask]
            canonical = torch.maximum(torch.minimum(target, upper), lower)
            if (span <= 1e-6).any() or not torch.isfinite(legal).all():
                raise ValueError('This diagnostic requires finite nondegenerate active joint intervals')
            if ((canonical - target)[mask].abs() > 1e-6).any():
                raise ValueError('Unreachable legal expert label')
            if ((previous.double() < lower - 1e-6) | (previous.double() > upper + 1e-6)).any():
                raise ValueError('Previous actuator target is not inside its reachable box')
            anchors = dict(nominal=torch.maximum(torch.minimum(record['base'].double(), upper), lower),
                           previous=previous.double())
            ids = torch.nonzero(mask).flatten()
            if identity in original:
                groups = [('original_train', ids)]
            elif identity in data['splits']['validation']:
                groups = [('validation', ids)]
            else:
                # Only the actual first contiguous expert segment is called onset.
                first_end = 1
                while first_end < len(ids) and ids[first_end] == ids[first_end - 1] + 1:
                    first_end += 1
                onset = ids[:min(12, first_end)]
                groups = [('correction_first12', onset), ('correction_after12', ids[len(onset):])]
            segment = record['control_segment']
            for group, ticks in groups:
                if not len(ticks):
                    continue
                row = {}
                contiguous = ((ticks[1:] == ticks[:-1] + 1)
                              & (segment[ticks[1:]] == segment[ticks[:-1]]))
                for name, anchor in anchors.items():
                    error = (canonical - anchor)[ticks]
                    raw = error / span[ticks]
                    row[name] = raw
                    row[name + '_rad'] = error
                    row[name + '_change'] = (raw[1:] - raw[:-1])[contiguous]
                buckets[group].append(row)
            bindings.append(dict(identity=identity, cache_sha256=digest, supervised_ticks=len(ids)))
        roles[role] = dict(frontend_sha256=index['base_sha256'], index_sha256=sha256(index_path),
            data_sha256=sha256(data_path), numeric_sha256=sha256(numeric_path), entries=bindings,
            groups={name: summarize(rows) for name, rows in buckets.items()})
    if torch.cuda.is_initialized():
        raise ValueError('CPU diagnostic unexpectedly initialized CUDA')
    atomic_json(output, dict(version='legal_rear_output_coordinate_diagnostic_v1', roles=roles,
        source_sha256=sha256(Path(__file__)), test_used=False, optimizer_updates=0,
        model_forward_calls=0, simulator_starts=0,
        definition='For the same real pre-action history, compute (legal expert command - anchor)/reachable_span. Nominal anchor is the current clipped DP base; alternative anchor is the actual previous actuator target. Label roundoff <=1e-6 rad is canonicalized exactly as in current loss.',
        interpretation='Both unbounded linear raw maps followed by the same box projection have identical command reachability and raw-to-command interior derivative. Smaller target magnitude or temporal variation does not establish easier neural fitting, closed-loop stability, or autonomous benefit. No candidate model or output change is selected or deployed. A/B nominal labels differ; both rear supervision labels here are the same actual expert commands. Validation6 is reused development, not an untouched test.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
