"""Finite train-only CUDA gate for opt-in offline anchor batching.

The completed best EMA and its frozen trainer are explicit inputs. The candidate
helper alone comes from this tool's tree. No training/validation/test evaluation,
checkpoint selection, rear cache, online controller or ordinary path is changed.
Run in a fresh process under the session's ordinary persistent owner.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import time

from tools.tight_fresh_selected_fit import (read, sha256, checked_hash,
    exclusive_json, import_frozen_trainer)

VERSION = 'offline_contiguous_anchor_cuda_gate_v1'
BATCHES = (1, 8, 32)
TARGET_ATOL = 2e-5


def require(value, message):
    if not value:
        raise ValueError(message)


def select_train_ids(data, requested=None):
    train = data['splits']['train']
    heldout = set(data['splits'].get('validation', [])) | set(data['splits'].get('test', []))
    if requested is None:
        latest = data.get('aggregation', {}).get('new_training_ids', [])
        requested = list(dict.fromkeys([train[0], *latest[:1]]))
    identities = list(requested)
    require(1 <= len(identities) <= 4 and len(identities) == len(set(identities)),
        'One to four unique training episodes required')
    require(set(identities) <= set(train) and not set(identities) & heldout,
        'Gate accepts training episodes only')
    heldout_families = {data['entries'][i]['family'] for i in heldout}
    # Current state_data keeps split membership in data['splits']; original
    # entries need not duplicate it. Reject contradictory optional metadata.
    require(all(data['entries'][i].get('split', 'train') == 'train' and
        data['entries'][i]['family'] not in heldout_families for i in identities),
        'Gate training families must be independent of validation/test')
    return identities


def validate_checkpoint(saved, receipt):
    require(saved.get('kind') == 'dp' and saved.get('group') == 'I-HistoryDP'
        and saved.get('epoch', 0) > 0 and saved.get('updates', 0) > 0
        and saved['configuration'] == receipt['configuration'],
        'Completed best checkpoint metadata differs')
    require(set(receipt['identities']) <= set(saved['train_ids']) and
        not set(receipt['identities']) & set(saved['validation_ids']),
        'Checkpoint does not bind exclusively training identities')
    require(saved['contract'].get('supervision') == ('q_reached' if receipt['role'] == 'A' else 'u_expert')
        and saved['contract'].get('visual_stride') == 1,
        'Expected native-radian every-frame frontend contract')


def admit(root, role, frozen_root, requested=None):
    """Only selected train visual shards; all frozen research source hashes hold."""
    root, frozen_root = Path(root).resolve(strict=True), Path(frozen_root).resolve(strict=True)
    require(role in ('A', 'B'), 'Unknown role')
    rr = root/role
    config = read(rr/'training/configuration.json')
    finished = read(rr/'training/finished.json')
    require(finished.get('complete') is True, 'Training must be complete before this gate')
    paths = dict(checkpoint=rr/'training/best.pt', data=rr/'state_data/data.json',
        statistics=rr/'state_data/statistics.pt', cache_index=rr/'cache_index.json',
        numeric=rr/'state_data/numeric.pt')
    hashes = {key: checked_hash(paths[key], config[key+'_sha256'])
        for key in ('data', 'statistics', 'cache_index')}
    hashes['checkpoint'] = checked_hash(paths['checkpoint'], finished['checkpoint_sha256'])
    hashes['numeric'] = sha256(paths['numeric'])
    data, cache = read(paths['data']), read(paths['cache_index'])
    identities = select_train_ids(data, requested)
    require(read(root/'gate.json')['passed'] and cache.get('complete') is True
        and cache.get('data_sha256') == hashes['data'] and cache.get('stride') == 1
        and cache.get('trainable_projection_cached') is False
        and cache.get('dino_sha256') == config['dino_sha256'],
        'Matching complete frozen every-frame visual cache required')
    continuation_path = rr/'continuation.json'
    continuation = read(continuation_path) if continuation_path.exists() else None
    require(continuation == config.get('continuation'), 'Continuation contract drift')
    sources = config['source_sha256']
    require('tools/tight_fresh_train.py' in sources, 'Frozen trainer identity required')
    for relative, expected in sources.items():
        path = (frozen_root/relative).resolve(strict=True)
        require(path.is_relative_to(frozen_root), 'Frozen source path escapes tree')
        checked_hash(path, expected)
    visual = {}
    for identity in identities:
        entry = cache['entries'][identity]
        path = rr/'state_data/perception'/(identity+'.pt')
        require(entry['source_sha256'] == data['entries'][identity]['source_sha256']
            and entry['frames'] == data['entries'][identity]['length'], 'Visual shard source/frame drift')
        visual[identity] = dict(path=str(path.resolve()), sha256=checked_hash(path, entry['sha256']),
            frames=entry['frames'])
    helper = Path(__file__).resolve().parents[1]/'visuotactile/offline_planning.py'
    return dict(root=str(root), role=role, frozen_source_root=str(frozen_root), configuration=config,
        paths={k:str(v) for k,v in paths.items()}, sha256=hashes, identities=identities,
        selected_visual_cache=visual, source_files_verified=len(sources),
        target_window=(continuation or {}).get('target_window_contract'),
        candidate_helper=str(helper), candidate_helper_sha256=sha256(helper),
        evaluator_source=str(Path(__file__).resolve()), evaluator_sha256=sha256(__file__),
        admission_helper_sha256=sha256(Path(__file__).with_name('tight_fresh_selected_fit.py')))


def train_projection(output, data, numeric, receipt):
    """Frozen visual_records otherwise eagerly loads the mixed numeric container."""
    import torch
    ids = select_train_ids(data, receipt['identities'])
    require(numeric['data_sha256'] == receipt['sha256']['data'], 'Numeric data binding drift')
    size = sum(t.numel()*t.element_size() for i in ids for t in numeric['records'][i].values())
    require(size <= 512*1024**2, 'Train-only numeric projection exceeds 512MiB')
    root = Path(output)/'train_only'
    root.mkdir(); (root/'perception').mkdir()
    projected = copy.deepcopy(data)
    projected['entries'] = {i:projected['entries'][i] for i in ids}
    projected['splits'] = dict(train=ids, validation=[], test=[])
    projected['diagnostic_train_projection'] = VERSION
    exclusive_json(root/'data.json', projected)
    records = {i:{k:v.clone() for k,v in numeric['records'][i].items()} for i in ids}
    torch.save(dict(data_sha256=sha256(root/'data.json'), records=records), root/'numeric.pt')
    for i in ids:
        (root/'perception'/(i+'.pt')).symlink_to(receipt['selected_visual_cache'][i]['path'])
    return dict(root=str(root), data_sha256=sha256(root/'data.json'),
        numeric_sha256=sha256(root/'numeric.pt'), numeric_tensor_bytes=size,
        scope='Original mixed numeric integrity bytes and mmap metadata only; only selected train tensors copied; no held-out raw or visual shard opened')


def error_metrics(prediction, truth, valid, action_scale):
    e = (prediction[:, :12]-truth[:, :12])[valid[:, :12]].double()
    require(len(e) > 0, 'No legal committed target slots')
    n = e*action_scale.double()
    return dict(legal_commit_slots=len(e), arm_rmse_rad=float(e[:, :7].square().mean().sqrt()),
        hand_rmse_rad=float(e[:, 7:].square().mean().sqrt()),
        selection=float(.5*n[:, :7].square().mean()+.5*n[:, 7:].square().mean()))


def compare_plans(reference, candidate, reference_bounded, candidate_bounded, truth, valid, scale):
    """Fixed target tolerance plus metric perturbation bounds, no fitted limits."""
    import torch
    require(reference.shape == candidate.shape == truth.shape and
        torch.isfinite(reference).all() and torch.isfinite(candidate).all(), 'Finite aligned plans required')
    r = error_metrics(reference_bounded, truth, valid, scale)
    c = error_metrics(candidate_bounded, truth, valid, scale)
    raw = float((reference-candidate).abs().max())
    bounded = float((reference_bounded-candidate_bounded).abs().max())
    differences = {key:abs(c[key]-r[key]) for key in ('arm_rmse_rad', 'hand_rmse_rad', 'selection')}
    # |(e+d)^2-e^2| <= 2|e|eps+eps^2 after the fixed normalization.
    e = (reference_bounded[:, :12]-truth[:, :12])[valid[:, :12]].double()
    n = e*scale.double()
    eps = TARGET_ATOL*scale.double().abs()
    bound = 2*n.abs()*eps+eps.square()
    selection_bound = float(.5*bound[:, :7].mean()+.5*bound[:, 7:].mean())
    passed = (raw <= TARGET_ATOL and bounded <= TARGET_ATOL
        and differences['arm_rmse_rad'] <= TARGET_ATOL+1e-12
        and differences['hand_rmse_rad'] <= TARGET_ATOL+1e-12
        and differences['selection'] <= selection_bound+1e-12)
    return dict(passed=bool(passed), raw_target_max_abs_rad=raw, bounded_target_max_abs_rad=bounded,
        fixed_target_tolerance_rad=TARGET_ATOL, reference_metrics=r, candidate_metrics=c,
        metric_absolute_differences=differences, selection_difference_bound=selection_bound)


def load_candidate(receipt):
    path = Path(receipt['candidate_helper'])
    checked_hash(path, receipt['candidate_helper_sha256'])
    spec = importlib.util.spec_from_file_location('_offline_planning_explicit_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(root, role, frozen_root, output, *, resource_root=None, identities=None,
        check_only=False, max_seconds=600):
    require(type(max_seconds) is int and 1 <= max_seconds <= 600, 'Finite gate wall bound 1..600 seconds required')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def check():
        if time.monotonic()-began > max_seconds:
            raise TimeoutError('Finite offline planning gate deadline reached; no automatic retry')
    exclusive_json(output/'started.json', dict(version=VERSION, pid=os.getpid(), utc=time.time(),
        root=str(Path(root).resolve()), role=role, frozen_source_root=str(Path(frozen_root).resolve()),
        resource_root=str(Path(resource_root).resolve()) if resource_root is not None else None,
        requested_identities=identities,
        max_seconds=max_seconds, check_only=check_only, batches=BATCHES, target_atol=TARGET_ATOL,
        optimizer_updates=0, validation_evaluated=False, test_evaluated=False))
    try:
        receipt = admit(root, role, frozen_root, identities)
        import torch
        saved = torch.load(receipt['paths']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        validate_checkpoint(saved, receipt)
        numeric = torch.load(receipt['paths']['numeric'], map_location='cpu', weights_only=False, mmap=True)
        receipt['projection'] = train_projection(output, read(receipt['paths']['data']), numeric, receipt)
        del numeric
        receipt.update(epoch=saved['epoch'], updates=saved['updates'], checkpoint_weights='ema')
        exclusive_json(output/'admission.json', receipt)
        check()
        if check_only:
            result = dict(complete=True, admission_only=True, passed=None, cuda_initialized=torch.cuda.is_initialized(),
                checkpoint_sha256=receipt['sha256']['checkpoint'], epoch=receipt['epoch'], identities=receipt['identities'])
        else:
            require(resource_root is not None, 'Explicit active resource session required for CUDA gate')
            trainer = import_frozen_trainer(receipt)
            candidate = load_candidate(receipt)
            from visuotactile.precision import state_fp32
            from visuotactile.state_resources import lease_for
            from visuotactile.target_windows import with_target_window
            state_fp32()
            resource_root = Path(resource_root).resolve(strict=True)
            lease = lease_for(resource_root, deadline=read(resource_root/'session.json')['deadline'])
            rows = []
            with lease.block(), torch.no_grad():
                check()
                records = trainer.visual_records(receipt['projection']['root'], receipt['identities'], 'G00', device='cuda:0')
                records = with_target_window(records, receipt['target_window'])
                statistics = torch.load(receipt['paths']['statistics'], map_location='cpu', weights_only=False)
                model = trainer.new_model(role, statistics)
                require(model.contract == saved['contract'], 'Frozen constructor/checkpoint contract drift')
                model.load_state_dict(saved['ema'], strict=True)
                model.eval().requires_grad_(False)
                for identity in receipt['identities']:
                    check(); record = records[identity]
                    torch.cuda.synchronize(); start = time.perf_counter()
                    conditions, anchors = trainer.supervised_conditions(model, record)
                    torch.cuda.synchronize(); encode_s = time.perf_counter()-start
                    truth, valid = trainer.labels(record, anchors, 24)
                    def timed(call):
                        check(); lease.check(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter(); value = call(); torch.cuda.synchronize()
                        return value, dict(seconds=time.perf_counter()-started,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    def bounded(value):
                        return trainer.bounded_targets(value.flatten(0, 1), record['joint_limits']).reshape_as(value)
                    reference, reference_time = timed(lambda:trainer.predict_on_native_clock(model, conditions, anchors))
                    reference_bounded = bounded(reference)
                    row = dict(identity=identity, full_history_frames=len(record['time']),
                        legal_expert_ticks=int(record['supervision_mask'].sum()), anchors=anchors.cpu().tolist(),
                        encode_seconds=encode_s, baseline=reference_time, candidates={})
                    for batch in BATCHES:
                        trace = {}
                        predicted, timing = timed(lambda:candidate.predict_contiguous(model, conditions, anchors,
                            configuration=candidate.profile(batch), check=check, receipt=trace))
                        compared = compare_plans(reference, predicted, reference_bounded, bounded(predicted),
                            truth, valid, model.action_scale)
                        row['candidates'][str(batch)] = dict(**compared, timing=timing, scheduling=trace)
                        del predicted
                    rows.append(row)
                    exclusive_json(output/('episode_%02d.json'%len(rows)), row)
                    del conditions, anchors, reference, reference_bounded, truth, valid
            result = dict(complete=True, admission_only=False, passed=all(
                item['passed'] for row in rows for item in row['candidates'].values()), rows=rows,
                checkpoint_sha256=receipt['sha256']['checkpoint'], epoch=receipt['epoch'],
                resource_root=str(resource_root), identity=VERSION, batches=BATCHES, fixed_target_atol=TARGET_ATOL,
                scope='Train-only frontend DDIM inference shape gate; no rear/autonomous success or training-speed claim',
                timing_limitations='Fixed order serial,M1,M8,M32; warmed kernels/cache and concurrent workers can bias timings. Full-history encode is shared and excluded from DDIM timings. Peak includes resident model/conditions/reference; wall check is between calls, not a hung-kernel watchdog.',
                metric_reduction='Float64 reductions of the same bounded, legal commit12 error population; trainer metrics use float32. Fixed max target difference is the primary acceptance criterion.',
                noise_contract='Same seed7091 shared generator; individual (1,24,29) draw per native tick including unsupervised prefix/gaps; DDIM eta0')
            result['passed_by_batch'] = {str(batch):all(row['candidates'][str(batch)]['passed'] for row in rows)
                for batch in BATCHES}
        result.update(seconds=time.monotonic()-began, optimizer_updates=0, validation_evaluated=False, test_evaluated=False)
        exclusive_json(output/'result.json', result)
        if result.get('passed') is False:
            raise ValueError('Offline anchor batch gate failed fixed tolerance; retain serial path')
        return result
    except BaseException as exc:
        exclusive_json(output/'error.json', dict(type=type(exc).__name__, error=str(exc), utc=time.time(),
            seconds=time.monotonic()-began, automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Completed source or selected training round')
    parser.add_argument('--role', choices=('A', 'B'), required=True)
    parser.add_argument('--frozen-source-root', type=Path, required=True)
    parser.add_argument('--resource-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identity', action='append', dest='identities')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=600)
    args = parser.parse_args()
    result = run(args.root, args.role, args.frozen_source_root, args.output,
        resource_root=args.resource_root, identities=args.identities, check_only=args.check_only, max_seconds=args.max_seconds)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
