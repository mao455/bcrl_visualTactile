"""Five disposable singleton updates diagnosing a failed B2 recovery gate.

This never changes the trainer or the failed gate's exact acceptance criterion.
Launch in a fresh process with CUBLAS_WORKSPACE_CONFIG=:4096:8 and an external
600-second process-group owner. The internal work budget is at most 570 seconds.
"""
import argparse
import copy
import gc
import os
from pathlib import Path
import time
import traceback

from tools.tight_offline_planning_gate import (admit, checked_hash, exclusive_json,
    import_frozen_trainer, read, require, sha256)
from tools.tight_trajectory_training_gate import load_candidate, optimizer_spec

VERSION = 'trajectory_recovery_arithmetic_probe_v1'
ARMS = (('default_first', False, True), ('default_same_objects', False, False),
    ('default_new_objects', False, True), ('deterministic_first', True, True),
    ('deterministic_new_objects', True, True))
STATE_FIELDS = {'contract', 'model', 'ema', 'optimizer', 'ema_optimization_step',
    'schedule_rng', 'noise_rng', 'torch_rng', 'cuda_rng'}


def compare(reference, candidate, path='state'):
    """Exact equality remains the verdict; distances only describe a failure."""
    import torch
    if isinstance(reference, torch.Tensor):
        require(isinstance(candidate, torch.Tensor) and reference.shape == candidate.shape
            and reference.dtype == candidate.dtype, 'Tensor structure differs: '+path)
        a, b = reference.detach().cpu(), candidate.detach().cpu()
        row = dict(exact=bool(torch.equal(a, b)), shape=list(a.shape), dtype=str(a.dtype),
            different_elements=int(torch.count_nonzero(a != b)), elements=a.numel())
        if not a.is_complex():
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            row.update(finite=finite, maximum_absolute=None, maximum_relative=None)
            if finite:
                difference = (a.double()-b.double()).abs()
                # Symmetric relative difference, denominator floor is diagnostic only.
                scale = torch.maximum(a.double().abs(), b.double().abs()).clamp_min(1e-30)
                row.update(maximum_absolute=float(difference.max()) if a.numel() else 0.,
                    maximum_relative=float((difference/scale).max()) if a.numel() else 0.)
        return {path: row}
    if isinstance(reference, dict):
        require(isinstance(candidate, dict) and reference.keys() == candidate.keys(),
            'Dictionary structure differs: '+path)
        result = {}
        for key, value in reference.items():
            result.update(compare(value, candidate[key], path+'/'+str(key)))
        return result
    if isinstance(reference, (list, tuple)):
        require(type(reference) is type(candidate) and len(reference) == len(candidate),
            'Sequence structure differs: '+path)
        result = {}
        for index, value in enumerate(reference):
            result.update(compare(value, candidate[index], path+'/'+str(index)))
        return result
    return {path: dict(exact=type(reference) is type(candidate) and reference == candidate)}


def all_exact(rows):
    return all(row['exact'] for row in rows.values())


def configure_arithmetic(deterministic):
    import torch
    from visuotactile.precision import state_fp32, snapshot
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
        'Launcher must set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA starts')
    state_fp32()
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    return dict(name='explicit_deterministic' if deterministic else 'default_nondeterministic',
        **snapshot(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])


def validate_failure(started, result):
    require(started.get('version') == 'trajectory_batch_two_training_recovery_gate_v1'
        and started.get('role') == 'B' and result.get('complete') is True
        and result.get('passed') is False and result.get('role') == 'B'
        and result.get('optimizer_steps_confirmed') == dict(serial_reference=2, batched=2, restored=1),
        'Only the completed failed B singleton recovery gate is admitted')
    for key in ('comparisons', 'boundary_restoration_comparisons'):
        require(result.get(key) and all(row['passed'] for row in result[key].values()),
            'Source serial/B2 or restored boundary did not pass')
    recovery = result.get('recovery_comparisons', {})
    require(recovery and any(not row['passed'] for row in recovery.values()), 'No recovery difference to diagnose')
    require(all(row['passed'] for key, row in recovery.items() if 'rng' in key), 'Source RNG differs')


def admission(source_gate, resource_root, check):
    source_gate, resource_root = Path(source_gate).resolve(strict=True), Path(resource_root).resolve(strict=True)
    require(source_gate.is_relative_to(resource_root), 'Source gate must belong to this original session')
    previous = read(source_gate/'admission.json'); result = read(source_gate/'result.json')
    started = read(source_gate/'started.json'); validate_failure(started, result)
    require(started['original_deadline'] == read(resource_root/'session.json')['deadline'],
        'Original session deadline differs')
    check()
    verified = admit(previous['root'], 'B', previous['frozen_source_root'], previous['identities'])
    for key in ('configuration', 'paths', 'sha256', 'identities', 'selected_visual_cache', 'target_window'):
        require(verified[key] == previous[key], 'Original training source changed: '+key)
    for item in previous['explicit_candidate_helpers'].values(): checked_hash(item['path'], item['sha256'])
    checked_hash(previous['gate_source'], previous['gate_source_sha256'])
    projection = previous['projection']; projected_root = Path(projection['root']).resolve(strict=True)
    require(projected_root == source_gate/'train_only', 'Original train-only projection path differs')
    for name in ('data', 'numeric'):
        checked_hash(projected_root/(name+('.json' if name == 'data' else '.pt')), projection[name+'_sha256'])
    for identity, item in previous['selected_visual_cache'].items():
        require((projected_root/'perception'/(identity+'.pt')).resolve(strict=True) == Path(item['path']),
            'Train-only visual link differs')
    boundary = source_gate/'diagnostic_boundary.pt'
    checked_hash(boundary, result['boundary_checkpoint_sha256']); check()
    singleton = previous['registered_batches'][1]
    require(singleton == [previous['identities'][0]] and len(previous['registered_deck']) == 3,
        'Original singleton/deck differs')
    expected = result['reports']['batched'][1]['visits']
    require(len(expected) == 1 and expected[0]['episode'] == singleton[0] and expected[0]['loss_weight'] == 1.,
        'Original singleton exposure missing')
    return dict(source_gate=str(source_gate), source_files={name:sha256(source_gate/name)
        for name in ('admission.json', 'started.json', 'result.json')},
        source_boundary=str(boundary), source_boundary_sha256=result['boundary_checkpoint_sha256'],
        original=previous, singleton=singleton, expected_visit={key:expected[0][key]
            for key in ('episode', 'anchors', 'target_ticks', 'visual_frames', 'loss_weight')})


def probe_updates(boundary, make_objects, update, configure, write, check, synchronize, progress,
                  expected_visit, memory=lambda reset:None):
    """CPU-testable orchestration; every branch starts from the same full state."""
    import torch
    from visuotactile.checkpointing import restore_training, training_state
    metadata = {key:value for key,value in boundary.items() if key not in STATE_FIELDS}
    require(metadata['updates'] == 1 and metadata['visits_seen'] == 2, 'Exact first-update boundary required')
    objects = None; references = {}; summaries = []
    for name, deterministic, fresh in ARMS:
        check(); progress['active'] = dict(arm=name, optimizer_step_may_be_partial=False)
        arithmetic = configure(deterministic)
        write(name+'_started.json', dict(utc=time.time(), arithmetic=arithmetic, fresh_objects=fresh))
        if fresh:
            if objects is not None: del model, averaged, optimizer, ema, schedule, noise
            objects = None; gc.collect()
            objects = make_objects()
        model, averaged, optimizer, ema, schedule, noise = objects
        # CPU tests also need isolation: optimizer.load_state_dict can alias CPU
        # checkpoint tensors when no device conversion is necessary.
        synchronize(); began = time.perf_counter()
        restore_training(copy.deepcopy(boundary), *objects)
        synchronize(); restore_seconds = time.perf_counter()-began
        began = time.perf_counter()
        before = training_state(*objects, **metadata)
        snapshot_seconds = time.perf_counter()-began
        began = time.perf_counter()
        restored = compare(boundary, before, 'boundary'); del before
        boundary_compare_seconds = time.perf_counter()-began
        write(name+'_boundary.json', dict(arithmetic=arithmetic, exact=all_exact(restored), comparisons=restored))
        require(all_exact(restored), 'Restored boundary differs before '+name)
        gradients = {}; layouts = {}; hook_calls = []; gradient_copy_seconds = []
        def capture_gradient(opt, args, kwargs):
            require(not hook_calls, 'More than one optimizer update attempted in '+name)
            hook_calls.append(True)
            progress['active']['optimizer_step_may_be_partial'] = True
            copy_began = time.perf_counter()
            for key, parameter in model.named_parameters():
                gradient = parameter.grad
                gradients[key] = None if gradient is None else gradient.detach().cpu().clone()
                layouts[key] = dict(parameter_stride=list(parameter.stride()), parameter_device=str(parameter.device),
                    gradient_stride=None if gradient is None else list(gradient.stride()),
                    gradient_dtype=None if gradient is None else str(gradient.dtype))
            gradient_copy_seconds.append(time.perf_counter()-copy_began)
            write(name+'_step_entered.json', dict(utc=time.time(), clipped_gradients_copied=True,
                optimizer_step_may_be_partial=True))
        handle = optimizer.register_step_pre_hook(capture_gradient)
        synchronize(); memory(True); began = time.perf_counter()
        try: report = update(objects)
        finally: handle.remove()
        synchronize(); elapsed = time.perf_counter()-began
        peak_memory = memory(False)
        require(len(hook_calls) == 1, 'One actual AdamW update required')
        progress['confirmed_updates'] += 1; progress['active'] = None
        require(ema.optimization_step == boundary['ema_optimization_step']+1, 'One actual EMA update required')
        require(len(report['visits']) == 1 and all(report['visits'][0][k] == v for k,v in expected_visit.items()),
            'Restored singleton supervision exposure differs')
        began = time.perf_counter()
        after = training_state(*objects, **dict(metadata, updates=2, visits_seen=3))
        post_snapshot_seconds = time.perf_counter()-began
        key = 'deterministic' if deterministic else 'default'
        reference = references.get(key)
        began = time.perf_counter()
        comparisons = None if reference is None else dict(
            states=compare(reference[0], after), clipped_gradients=compare(reference[1], gradients, 'gradient'))
        post_compare_seconds = time.perf_counter()-began
        if reference is None: references[key] = (after, gradients)
        norms = {k:None if g is None else dict(elements=g.numel(), finite=bool(torch.isfinite(g).all()),
            l2=float(g.double().norm()), maximum_absolute=float(g.abs().max())) for k,g in gradients.items()}
        row = dict(arm=name, fresh_objects=fresh, arithmetic=arithmetic, report=report,
            diagnostic_update_seconds=elapsed, boundary_exact=True, gradient_capture='after clipping, before AdamW',
            timings_seconds=dict(restore_with_cpu_copy=restore_seconds, boundary_cpu_snapshot=snapshot_seconds,
                boundary_comparison=boundary_compare_seconds, synchronized_execute_update=elapsed,
                gradient_copy_within_execute_update=sum(gradient_copy_seconds),
                post_update_cpu_snapshot=post_snapshot_seconds, post_update_comparison=post_compare_seconds),
            peak_cuda_memory=peak_memory,
            gradient_summary=norms, tensor_layouts=layouts, comparisons=comparisons,
            reference_arm=None if reference is None else key+'_first',
            state_exact=None if reference is None else all_exact(comparisons['states']),
            gradients_exact=None if reference is None else all_exact(comparisons['clipped_gradients']))
        write(name+'.json', row)
        summaries.append({k:row[k] for k in ('arm','fresh_objects','state_exact','gradients_exact','diagnostic_update_seconds')})
        del after, gradients, comparisons, row
        check()
    cross = dict(states=compare(references['default'][0], references['deterministic'][0]),
        clipped_gradients=compare(references['default'][1], references['deterministic'][1], 'gradient'))
    write('arithmetic_profiles_comparison.json', cross)
    return summaries


def run(source_gate, output, resource_root, *, check_only=False):
    output = Path(output).resolve(); resource_root = Path(resource_root).resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time(); session = read(resource_root/'session.json')
    deadline = min(started+570, float(session['deadline'])-30)
    progress = dict(confirmed_updates=0, active=None)
    write = lambda name, value: exclusive_json(output/name, value)
    def check():
        if time.time() >= deadline or any(p.exists() for p in (output/'STOP', resource_root/'STOP')):
            raise InterruptedError('Original session/finite work deadline or STOP; no retry')
    write('started.json', dict(version=VERSION, pid=os.getpid(), utc=started, deadline=deadline,
        original_deadline=session['deadline'], maximum_optimizer_updates=5, diagnostic_only=True,
        formal_training_candidate=False, check_only=check_only,
        launcher_cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        required_external_owner_seconds=600))
    try:
        check(); receipt = admission(source_gate, resource_root, check)
        receipt.update(probe_source=str(Path(__file__).resolve()), probe_source_sha256=sha256(__file__))
        write('admission.json', receipt); check()
        import torch
        boundary = torch.load(receipt['source_boundary'], map_location='cpu', weights_only=False, mmap=True)
        require(boundary['updates'] == 1 and boundary['visits_seen'] == 2, 'Source boundary budget differs')
        if check_only:
            result = dict(complete=True, check_only=True, cuda_initialized=torch.cuda.is_initialized())
        else:
            require(not torch.cuda.is_initialized(), 'Probe must start in a fresh CUDA process')
            require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Missing launcher CUBLAS workspace configuration')
            original = receipt['original']; trainer = import_frozen_trainer(original)
            from visuotactile.state_resources import lease_for
            from visuotactile.target_windows import with_target_window
            from visuotactile.models.vendor.ema_model import EMAModel
            helpers = {name:load_candidate(item['path'], 'visuotactile.'+name)
                for name,item in original['explicit_candidate_helpers'].items()}
            candidate = helpers['trajectory_training']
            require(boundary['full_trajectory_batch'] == candidate.profile(), 'Original candidate profile differs')
            configure_arithmetic(False); torch.cuda.set_per_process_memory_fraction(.45, device=0)
            lease = lease_for(resource_root, deadline=deadline)
            with lease.block():
                ids = receipt['singleton']; projection = original['projection']['root']
                records = with_target_window(trainer.visual_records(projection, ids, 'G00', device='cuda:0'), original['target_window'])
                objective, geometry, proof = trainer.fresh_objective(Path(original['root']), 'B', read(original['paths']['data']), records)
                statistics = torch.load(original['paths']['statistics'], map_location='cpu', weights_only=False)
                def make():
                    model = trainer.new_model('B', statistics).train()
                    require(model.contract == boundary['contract'], 'Frozen model contract differs')
                    averaged = copy.deepcopy(model).eval().requires_grad_(False)
                    return (model, averaged, torch.optim.AdamW(model.parameters(), **optimizer_spec(original['configuration'])),
                        EMAModel(averaged, power=.75), torch.Generator().manual_seed(11), torch.Generator(device='cuda:0').manual_seed(12))
                def guarded(): check(); lease.check()
                def update(objects):
                    model, _, opt, ema, _, noise = objects
                    episodes = [trainer.Episode(projection, i, 'I-HistoryDP') for i in ids]
                    return candidate.execute_update(model, opt, ema, objective, ids, [records[i] for i in ids], episodes, noise, check=guarded)
                def memory(reset):
                    if reset: torch.cuda.reset_peak_memory_stats(); return None
                    return dict(allocated_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.max_memory_reserved())
                summaries = probe_updates(boundary, make, update, configure_arithmetic, write, guarded,
                    torch.cuda.synchronize, progress, receipt['expected_visit'], memory=memory)
                result = dict(complete=True, check_only=False, arms=summaries, geometry=geometry, geometry_selection_proof=proof)
        result.update(version=VERSION, diagnostic_only=True, formal_training_candidate=False,
            confirmed_optimizer_updates=progress['confirmed_updates'], wall_seconds=time.time()-started,
            original_failed_gate_unchanged=True, adoption=False,
            limitations='New diagnostic process has CUBLAS_WORKSPACE_CONFIG and synchronous CPU gradient copies; '
                'not a bitwise replay of the old gate. Deterministic arithmetic is diagnostic only. '
                'Update timings include the separately measured gradient copy and cannot establish full-epoch throughput. '
                'No gate threshold, trainer, selected checkpoint, validation/test or autonomous behavior changed.')
        check(); write('result.json', result); return result
    except BaseException as error:
        write('failed.json', dict(error=repr(error), traceback=traceback.format_exc(), **progress,
            elapsed_seconds=time.time()-started, diagnostic_only=True, adoption=False, utc=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('source-gate', 'output', 'resource-root'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    run(**vars(parser.parse_args()))


if __name__ == '__main__': main()
