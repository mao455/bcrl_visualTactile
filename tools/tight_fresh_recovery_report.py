"""Bounded before/after prediction on the same actual recovery observations.

No training, simulator, test families or replacement teacher states are used.
"""
import argparse
import json
from pathlib import Path
import time
import subprocess
import traceback
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(path.read_text())


def run(root, output):
    import torch
    from visuotactile.models.native_dp import load_policy, load_residual
    from visuotactile.native_data import visual_records, cache_residual_episode
    from visuotactile.anchor_continue import residual_metrics
    from visuotactile.checkpointing import save
    from visuotactile.precision import state_fp32
    from visuotactile.state_resources import lease_for
    if output.exists():
        raise FileExistsError('Preserve previous diagnostic attempt')
    output.mkdir(parents=True)
    sources = {str(p): sha256(p) for folder in ('tools', 'visuotactile') for p in Path(folder).rglob('*.py')}
    atomic_json(output / 'registration.json', dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=sources, root=str(root), started=time.time(), optimizer_updates=0, simulator_starts=0, test_used=False))
    state_fp32()
    session = read(root / 'session.json')
    lease = lease_for(root, deadline=session['deadline'])
    reports = {}
    for role in ('A', 'B'):
        if any(sha256(p) != value for p, value in sources.items()):
            raise ValueError('Diagnostic source drift')
        directory = root / role
        data_root = directory / 'state_data'
        data = read(data_root / 'data.json')
        identities = data['aggregation']['new_training_ids']
        if not set(identities) <= set(data['splits']['train']):
            raise ValueError('Recovery diagnostic must be confined to actual training corrections')
        continuation = read(directory / 'continuation.json')
        old_front = Path(continuation['path'])
        old_rear = Path(continuation['previous_backend'])
        if sha256(old_front) != continuation['sha256'] or sha256(old_rear) != continuation['previous_backend_sha256']:
            raise ValueError('Original selected pair identity differs')
        new_front = directory / 'training/best.pt'
        new_rear = directory / 'backend' / role / 'training/best.pt'
        index_path = directory / 'backend' / role / 'nominal/index.json'
        index = read(index_path)
        if index['base_sha256'] != sha256(new_front) or index['data_sha256'] != sha256(data_root / 'data.json'):
            raise ValueError('New cache identity differs')
        raw = visual_records(data_root, identities, 'G00')
        visual_index = read(directory / 'cache_index.json')
        reports[role] = {}
        for label, front_path, rear_path in (('before', old_front, old_rear), ('after', new_front, new_rear)):
            started = time.time()
            saved = torch.load(rear_path, map_location='cpu', weights_only=False)
            if saved['configuration']['statistics_sha256'] != sha256(data_root / 'statistics.pt'):
                raise ValueError('Before/after frontend normalization differs')
            if saved['configuration']['base_sha256'] != sha256(front_path):
                raise ValueError('Complete pair frontend/backend mismatch')
            with lease.block():
                rear = load_residual(saved, 'cuda:0').eval().requires_grad_(False)
            binding = dict(base_checkpoint_sha256=sha256(front_path),
                statistics_sha256=sha256(data_root / 'statistics.pt'),
                data_sha256=sha256(data_root / 'data.json'),
                residual_statistics_sha256=saved['configuration']['residual_statistics_sha256'])
            records = {}
            if label == 'before':
                old_directory = output / role / 'before_nominal'
                old_directory.mkdir(parents=True)
                with lease.block():
                    base = load_policy(torch.load(front_path, map_location='cpu', weights_only=False), 'cuda:0').eval().requires_grad_(False)
                for identity in identities:
                    perception = data_root / 'perception' / (identity + '.pt')
                    if sha256(perception) != visual_index['entries'][identity]['sha256']:
                        raise ValueError('Frozen actual-frame feature identity differs')
                    record = cache_residual_episode(base, raw[identity], data['entries'][identity], lease,
                        cache_identity=binding, transfer_mode='episode', scheduler_index_mode='host')
                    save(old_directory / (identity + '.pt'), record)
                    records[identity] = record
                    atomic_json(output / 'progress.json', dict(role=role, stage=label, identity=identity, utc=time.time()))
                del base
            else:
                for identity in identities:
                    path = index_path.parent / (identity + '.pt')
                    if sha256(path) != index['episodes'][identity]['sha256']:
                        raise ValueError('Current cache file identity differs')
                    records[identity] = torch.load(path, map_location='cpu', weights_only=False)
                    if records[identity]['cache_identity'] != binding:
                        raise ValueError('Current cache episode binding differs')
            metrics, predictions = residual_metrics(rear, records, identities, lease)
            per_episode = []
            for identity, prediction in zip(identities, predictions):
                record = records[identity]
                valid = record['supervision_mask'].bool()
                first = torch.zeros_like(valid)
                first[valid.nonzero().flatten()[:24]] = True
                errors = prediction - record['target']
                values = {}
                for name, mask in (('all', valid), ('initial24', first)):
                    error = errors[mask]
                    values[name] = dict(ticks=int(mask.sum()), arm_rmse_rad=float(error[:, :7].square().mean().sqrt()),
                        hand_rmse_rad=float(error[:, 7:].square().mean().sqrt()))
                per_episode.append(dict(identity=identity, metrics=values))
            reports[role][label] = dict(metrics=metrics, episodes=per_episode,
                frontend_sha256=sha256(front_path), backend_sha256=sha256(rear_path),
                cache_identity=binding, seconds=time.time()-started)
            atomic_json(output / (role + '_' + label + '.json'), reports[role][label])
            del rear, saved, records, predictions
    atomic_json(output / 'result.json', dict(complete=True, roles=reports,
        scope='Same actual student-prefix/expert-suffix observations. Own selected DP plans and learned features recomputed for old pair; new exact cache verified. Only legally executed expert suffix labels, initial24 separately. These corrections were unseen by before and used in after training; this is fitting, not generalization or autonomous improvement.',
        test_used=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root.resolve(), args.output.resolve())
    except BaseException as error:
        if args.output.exists() and not (args.output / 'result.json').exists():
            atomic_json(args.output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=time.time()))
        raise


if __name__ == '__main__':
    main()
