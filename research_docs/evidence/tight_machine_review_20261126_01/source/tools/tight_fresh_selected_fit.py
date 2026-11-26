"""Measure selected frontend EMA fit on the latest train-only corrections.

No optimizer, new sampling choice, validation/test evaluation, or checkpoint
selection is performed. The original frozen trainer supplies metrics, model,
visual reader and target-window semantics; every imported research dependency
must come from that same verified source tree.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError('Frozen input SHA256 differs: '+str(path))
    return actual


def latest_training_ids(data):
    identities = data['aggregation']['new_training_ids']
    train = set(data['splits']['train'])
    heldout = set(data['splits'].get('validation', [])) | set(data['splits'].get('test', []))
    if (not identities or len(identities) != len(set(identities))
            or not set(identities) <= train or set(identities) & heldout):
        raise ValueError('Latest correction identities must be unique and exclusively training data')
    heldout_families = {data['entries'][identity]['family'] for identity in heldout}
    for identity in identities:
        entry = data['entries'][identity]
        if (entry['split'] != 'train' or entry['trajectory_kind'] != 'expert_takeover_v1'
                or entry['family'] in heldout_families):
            raise ValueError('Only real latest training-family expert takeovers may be evaluated')
    return list(identities)


def validate_inputs(root, role, frozen_root):
    """CPU-only artifact admission; the model has not been imported or run."""
    root, frozen_root = Path(root).resolve(strict=True), Path(frozen_root).resolve(strict=True)
    role_root = root/role
    config = read(role_root/'training/configuration.json')
    paths = dict(data=role_root/'state_data/data.json', statistics=role_root/'state_data/statistics.pt',
        cache_index=role_root/'cache_index.json', checkpoint=role_root/'training/best.pt')
    hashes = {key:checked_hash(paths[key], config[key+'_sha256']) for key in
        ('data', 'statistics', 'cache_index')}
    hashes['checkpoint'] = checked_hash(paths['checkpoint'], read(role_root/'training/finished.json')['checkpoint_sha256'])
    data, cache = read(paths['data']), read(paths['cache_index'])
    identities = latest_training_ids(data)
    if (not read(root/'gate.json')['passed'] or cache.get('complete') is not True
            or cache.get('data_sha256') != hashes['data'] or cache.get('stride') != 1
            or cache.get('trainable_projection_cached') is not False
            or cache.get('dino_sha256') != config['dino_sha256']):
        raise ValueError('Complete matching frozen every-frame visual cache required')
    continuation = read(role_root/'continuation.json')
    if continuation != config.get('continuation'):
        raise ValueError('Actual target-window continuation differs from trained configuration')
    sources = config['source_sha256']
    if 'tools/tight_fresh_train.py' not in sources:
        raise ValueError('Original trainer source identity is unavailable')
    for relative, expected in sources.items():
        path = (frozen_root/relative).resolve(strict=True)
        if not path.is_relative_to(frozen_root):
            raise ValueError('Frozen source path escapes the source tree')
        checked_hash(path, expected)
    selected_cache = {}
    for identity in identities:
        entry = cache['entries'][identity]
        path = role_root/'state_data/perception'/(identity+'.pt')
        # Episode reads this exact path, rather than the index's suggestion.
        digest = checked_hash(path, entry['sha256'])
        if (entry['source_sha256'] != data['entries'][identity]['source_sha256']
                or entry['frames'] != data['entries'][identity]['length']):
            raise ValueError('Latest visual cache source/frame identity differs')
        selected_cache[identity] = dict(path=str(path.resolve()), sha256=digest, frames=entry['frames'])
    hashes['numeric'] = sha256(role_root/'state_data/numeric.pt')
    return dict(root=str(root), role=role, frozen_source_root=str(frozen_root), configuration=config,
        paths={key:str(value) for key,value in paths.items()}, sha256=hashes, identities=identities,
        selected_visual_cache=selected_cache, source_files_verified=len(sources),
        evaluator_source=str(Path(__file__).resolve()), evaluator_sha256=sha256(__file__),
        frozen_trainer_sha256=sources['tools/tight_fresh_train.py'], target_window=continuation.get('target_window_contract'))


def validate_checkpoint(saved, receipt):
    if (saved['configuration'] != receipt['configuration'] or saved.get('kind') != 'dp'
            or saved.get('group') != 'I-HistoryDP' or saved.get('epoch', 0) < 1
            or saved.get('updates', 0) < 1):
        raise ValueError('Selected checkpoint metadata differs from the completed training')
    identities = receipt['identities']
    if (not set(identities) <= set(saved['train_ids'])
            or set(identities) & set(saved['validation_ids'])
            or any(saved.get('sampling_exposure', {}).get(identity, 0) < 1 for identity in identities)):
        raise ValueError('Selected checkpoint has no valid training exposure for every latest correction')
    expected = 'q_reached' if receipt['role'] == 'A' else 'u_expert'
    if (saved['contract'].get('supervision') != expected
            or saved['contract'].get('visual_stride') != 1):
        raise ValueError('Selected frontend target/visual contract differs')


def import_frozen_trainer(receipt):
    root = Path(receipt['frozen_source_root'])
    for name, module in list(sys.modules.items()):
        if name == 'visuotactile' or name.startswith('visuotactile.'):
            path = getattr(module, '__file__', None)
            if path is not None and not Path(path).resolve().is_relative_to(root):
                raise RuntimeError('Start a fresh process: research dependencies already imported from another tree')
    sys.path.insert(0, str(root))
    name = '_tight_selected_fit_frozen_trainer'
    spec = importlib.util.spec_from_file_location(name, root/'tools/tight_fresh_train.py')
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[name] = trainer
    spec.loader.exec_module(trainer)
    return trainer


def exclusive_json(path, value):
    """New diagnostic evidence never overwrites a prior run."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def run(root, role, frozen_root, output, *, check_only=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    exclusive_json(output/'started.json', dict(root=str(Path(root).resolve()), role=role,
        frozen_source_root=str(Path(frozen_root).resolve()), pid=os.getpid(), utc=time.time(),
        check_only=check_only, optimizer_updates=0, validation_evaluated=False, test_evaluated=False))
    try:
        receipt = validate_inputs(root, role, frozen_root)
        import torch
        saved = torch.load(receipt['paths']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        validate_checkpoint(saved, receipt)
        receipt.update(epoch=saved['epoch'], updates=saved['updates'],
            sampling_exposure={identity:saved['sampling_exposure'][identity] for identity in receipt['identities']},
            ema_optimization_step=saved['ema_optimization_step'], checkpoint_weights='ema')
        exclusive_json(output/'admission.json', receipt)
        if check_only:
            result = dict(complete=True, admission_only=True, cuda_initialized=torch.cuda.is_initialized(),
                checkpoint_sha256=receipt['sha256']['checkpoint'], epoch=receipt['epoch'],
                identities=receipt['identities'], metrics=None, optimizer_updates=0,
                validation_evaluated=False, test_evaluated=False, seconds=time.perf_counter()-began)
            exclusive_json(output/'result.json', result)
            return result
        trainer = import_frozen_trainer(receipt)
        from visuotactile.precision import state_fp32
        from visuotactile.state_resources import lease_for
        from visuotactile.target_windows import with_target_window
        state_fp32()
        root = Path(receipt['root'])
        lease = lease_for(root, deadline=read(root/'session.json')['deadline'])
        with lease.block():
            # Same reader as trainer.inputs, restricted to the preregistered
            # latest train identities. No validation/test visual shard is read.
            records = trainer.visual_records(root/role/'state_data', receipt['identities'], 'G00', device='cuda:0')
            records = with_target_window(records, receipt['target_window'])
            statistics = torch.load(receipt['paths']['statistics'], map_location='cpu', weights_only=False)
            model = trainer.new_model(role, statistics)
            if model.contract != saved['contract']:
                raise ValueError('Frozen model constructor and selected checkpoint contract differ')
            model.load_state_dict(saved['ema'], strict=True)
            model.eval().requires_grad_(False)
            counts = {identity:dict(real_history_frames=len(record['time']),
                legal_expert_ticks=int(record['supervision_mask'].sum())) for identity,record in records.items()}
            torch.cuda.synchronize()
            inference_started = time.perf_counter()
            metrics = trainer.metrics(model, records, receipt['identities'])
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter()-inference_started
        result = dict(complete=True, admission_only=False, role=role,
            checkpoint=receipt['paths']['checkpoint'], checkpoint_sha256=receipt['sha256']['checkpoint'],
            epoch=receipt['epoch'], updates=receipt['updates'], checkpoint_weights='ema',
            sampling_exposure=receipt['sampling_exposure'], identities=receipt['identities'], counts=counts,
            metrics=metrics, frozen_source_root=receipt['frozen_source_root'],
            evaluator_source=receipt['evaluator_source'], evaluator_sha256=receipt['evaluator_sha256'],
            frozen_trainer_sha256=receipt['frozen_trainer_sha256'], source_files_verified=receipt['source_files_verified'],
            data_sha256=receipt['sha256']['data'], statistics_sha256=receipt['sha256']['statistics'],
            cache_index_sha256=receipt['sha256']['cache_index'], target_window=receipt['target_window'],
            inference_seconds=inference_seconds, seconds=time.perf_counter()-began,
            optimizer_updates=0, validation_evaluated=False, test_evaluated=False,
            scope='Exact frozen training metric on latest real training corrections; selected EMA, full real causal prefix, DDIM16 commit12; no autonomy or new checkpoint selection')
        exclusive_json(output/'result.json', result)
        return result
    except BaseException as exc:
        exclusive_json(output/'error.json', dict(type=type(exc).__name__, error=str(exc), utc=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--role', choices=('A', 'B'), default='B')
    parser.add_argument('--frozen-source-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    result = run(args.root, args.role, args.frozen_source_root, args.output, check_only=args.check_only)
    print(json.dumps({key:value for key,value in result.items() if key not in ('counts', 'metrics', 'identities', 'sampling_exposure')}))


if __name__ == '__main__':
    main()
