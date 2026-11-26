"""Fixed CPU family folds separate probe-fit coverage from visual decodability.

The encoders already learned these trajectories. Family exclusion applies only
to the diagnostic readout, never claimed as held-out student evaluation.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from tools.tight_fresh_visual_prefix_probe import (
    admit_population, source_labels, cache_binding, validate_cache, summarize, write_new)
from tools.tight_fresh_position_probe import fit, predict, inputs, read
from visuotactile.compact import sha256


def family_folds(families):
    if len(families) != 30 or len(set(families)) != 30:
        raise ValueError('Exactly30 distinct original training families required')
    ordered = sorted(families)
    return [ordered[k::5] for k in range(5)]


def fold_population(data, population, heldout):
    family = lambda identity: data['entries'][identity]['family']
    known = {family(i) for i in population['original']}
    if len(heldout) != 6 or len(set(heldout)) != 6 or not set(heldout) <= known:
        raise ValueError('Each fold excludes six known complete training families')
    evaluate = [i for i in population['corrections'] if family(i) in heldout]
    original = [i for i in population['original'] if family(i) not in heldout]
    augmented = original + [i for i in population['corrections'] if family(i) not in heldout]
    return evaluate, {'original_only': original, 'with_corrections': augmented}


def run(root, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve prior probe evidence')
    if torch.cuda.is_initialized():
        raise ValueError('Fresh CPU-only process required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    start = time.monotonic()
    root = Path(root).resolve()
    data = read(root/'A/state_data/data.json')
    population = admit_population(data, read(root/'B/state_data/data.json'))
    identities = population['original'] + population['corrections']
    folds = family_folds([data['entries'][i]['family'] for i in population['original']])
    truth, masks, stamps, segments, labels = {}, {}, {}, {}, {}
    for identity in identities:
        truth[identity], masks[identity], stamps[identity], segments[identity], labels[identity] = source_labels(
            data['entries'][identity], 'train', identity in population['corrections'])
    result = dict(version='fixed_five_family_folds_cpu_probe_v1', source_root=str(root),
        data_sha256={role:sha256(root/role/'state_data/data.json') for role in 'AB'},
        folds=folds, labels=labels, roles={}, test_used=False,
        interpretation='Student encoders already trained on these histories. Only linear-probe fitting excludes '
            'each evaluated family. Fixed lambda=.01 and original-only versus original+all actual correction '
            'observations; no parameter selection. This diagnoses representation decodability versus probe-fit '
            'coverage, not student generalization or autonomous behavior.',
        payload_hash_scope='Full current nominal shards verified against the bound index; raw source SHA is '
            'manifest provenance, not rehashed here. Read labels, clocks and authority arrays are hashed.',
        optimizer_updates=0, simulator_starts=0)
    residual_sha = sha256(root/'B/state_data/statistics.pt')
    for role in 'AB':
        index_path, index, binding = cache_binding(root/role, residual_sha)
        records = {}
        for identity in identities:
            path = index_path.parent/(identity+'.pt')
            if sha256(path) != index['episodes'][identity]['sha256']:
                raise ValueError('Current nominal cache changed')
            record = torch.load(path, map_location='cpu', weights_only=False)
            validate_cache(record, identity, stamps[identity], binding, masks[identity], segments[identity])
            records[identity] = {key:record[key] for key in ('visual', 'robot')}
        methods = {}
        for method in ('visual_only', 'robot178_only', 'combined'):
            pooled = {key:{} for key in ('original_only', 'with_corrections')}
            fold_details = []
            for number, heldout in enumerate(folds):
                evaluate, train_groups = fold_population(data, population, heldout)
                details = dict(fold=number, evaluated_ids=evaluate, training_ids=train_groups, fits={})
                for kind, training in train_groups.items():
                    parameters, info = fit(torch.cat([inputs(records[i], method) for i in training]),
                        torch.cat([truth[i] for i in training]))
                    info['feature_standardization'] = 'Exact per-fold training_ids frame mean and population std; constant scale1; float64 solve'
                    details['fits'][kind] = info
                    for identity in evaluate:
                        pooled[kind][identity] = predict(parameters, inputs(records[identity], method))-truth[identity]
                fold_details.append(details)
            methods[method] = dict(folds=fold_details, metrics={kind:summarize(errors, masks,
                population['corrections'], ('student_prefix', 'student_after_contact', 'legal_expert'))
                for kind, errors in pooled.items()})
            print(role, method, {kind:round(value['student_prefix']['frame_weighted_vector_rmse_mm'], 3)
                for kind, value in methods[method]['metrics'].items()}, flush=True)
        result['roles'][role] = dict(binding=binding, cache_index_sha256=sha256(index_path), methods=methods)
    if torch.cuda.is_initialized():
        raise ValueError('CPU diagnostic unexpectedly initialized CUDA')
    result.update(seconds=time.monotonic()-start, cuda_initialized=False, source_sha256=sha256(__file__),
        dependency_sha256={name:sha256(Path(__file__).resolve().parents[1]/name) for name in (
            'tools/tight_fresh_visual_prefix_probe.py', 'tools/tight_fresh_position_probe.py',
            'visuotactile/narrow_data.py', 'visuotactile/native_data.py')})
    write_new(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
