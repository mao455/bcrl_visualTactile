"""Registered real-source auxiliary acceptance; four disposable AdamW updates only."""
import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from tools.tight_fresh_train import fresh_objective, supervised_conditions
from tools.tight_visual_aux_calibrate import (check_label_binding, gradient_norm,
    head_sha, paired_subset, source_binding, tensor_sha)
from visuotactile.anchor_continue import projection_batch
from visuotactile.anchor_sampling import labels
from visuotactile.compact import sha256
from visuotactile.models.native_dp import load_policy
from visuotactile.native_data import visual_records
from visuotactile.training_resources import atomic_json
from visuotactile.visual_aux_training import (CALIBRATION_VERSION, ESTIMATOR,
    attach_auxiliary_head, auxiliary_episode_loss, optimizer_groups,
    supervised_visual_conditions)
from visuotactile.visual_pose_aux import (ROUTES, VisualPoseHead,
    clip_auxiliary_gradients, load_pose_labels, pose_loss)


VERSION = 'visual_pose_aux_acceptance_v1'
# Fixed before any actual acceptance. FP32 forward roundoff has 2e-5 allowance;
# independently clipped gradients permit 2e-6 absolute / 0.02% relative roundoff.
# Updated weights use a smaller relative allowance; update deltas are checked
# separately so large unchanged weights cannot conceal a different Adam step.
TOLERANCES = dict(forward=dict(atol=2e-5, rtol=0.),
    gradient=dict(atol=2e-6, rtol=2e-4),
    updated_parameter=dict(atol=2e-6, rtol=2e-6),
    update_delta=dict(atol=2e-7, rtol=2e-4))


def read(path):
    return json.loads(Path(path).read_text())


def code_hashes(project):
    return {str(p.relative_to(project)): sha256(p)
        for folder in ('tools', 'visuotactile')
        for p in sorted((project/folder).rglob('*.py'))}


def first_original(data):
    """First original in the actual source training order, never a chosen outcome."""
    original = set(data['normalization_train_ids'])
    return next(identity for identity in data['splits']['train'] if identity in original)


def compare_tensors(reference, actual, kind):
    if kind not in TOLERANCES or reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise ValueError('Parity tensor shape/dtype or tolerance identity differs')
    a, b = reference.detach().cpu(), actual.detach().cpu()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Parity tensors must be finite')
    difference = (a.double()-b.double()).abs()
    tolerance = TOLERANCES[kind]
    allowed = tolerance['atol']+tolerance['rtol']*a.double().abs()
    error = float(difference.max()) if difference.numel() else 0.
    if (difference > allowed).any():
        raise ValueError(f'{kind} parity failed: max_abs={error}, fixed tolerance={tolerance}')
    return dict(max_abs=error, reference_max_abs=float(a.abs().max()) if a.numel() else 0.,
        max_tolerance_fraction=float((difference/allowed).max()) if difference.numel() else 0.)


def compare_maps(reference, actual, kind):
    if reference.keys() != actual.keys():
        raise ValueError('Parity parameter identities differ')
    rows = {}
    for name, value in reference.items():
        other = actual[name]
        if (value is None) != (other is None):
            raise ValueError('Parity gradient connectivity differs: '+name)
        if value is not None:
            rows[name] = compare_tensors(value, other, kind)
    return dict(parameters=len(reference), active_parameters=len(rows),
        max_abs=max((r['max_abs'] for r in rows.values()), default=0.),
        max_tolerance_fraction=max((r['max_tolerance_fraction'] for r in rows.values()), default=0.),
        rows=rows)


def same_draws(truth, model):
    generator = torch.Generator(device=truth.device).manual_seed(2026111403)
    target = truth*model.action_scale+model.action_offset
    return (torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator),
        torch.randint(100, (len(target),), device=target.device, generator=generator))


def check_head_rng(seed, device):
    cpu = torch.random.get_rng_state().clone()
    cuda = [x.clone() for x in torch.cuda.get_rng_state_all()] if device.type == 'cuda' else []
    head = VisualPoseHead(seed=seed).to(device)
    if (not torch.equal(cpu, torch.random.get_rng_state())
            or any(not torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all() if cuda else []))):
        raise ValueError('Head construction changed CPU or CUDA RNG')
    return dict(cpu_rng_unchanged=True, cuda_rng_unchanged=True if cuda else None,
        cuda_devices_checked=len(cuda), head_sha256=head_sha(head))


def step_snapshot(model, loss, *, auxiliary, learning_rate):
    """A disposable trainer-identical update; never serialize updated weights."""
    opt = torch.optim.AdamW(optimizer_groups(model) if auxiliary else model.parameters(),
        lr=learning_rate, betas=(.95, .999), weight_decay=1e-6)
    opt.zero_grad(set_to_none=True)
    if loss.ndim or not torch.isfinite(loss):
        raise ValueError('Finite scalar disposable objective required')
    loss.backward()
    if auxiliary:
        norms = clip_auxiliary_gradients(model, model.visual_pose_head)
    else:
        norms = dict(policy_norm=torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    parameters = {name: value for name, value in model.named_parameters()
        if not name.startswith('visual_pose_head.')}
    gradients = {name: value.grad.detach().cpu().clone() if value.grad is not None else None
        for name, value in parameters.items()}
    before = {name: value.detach().cpu().clone() for name, value in parameters.items()}
    opt.step()
    after = {name: value.detach().cpu().clone() for name, value in parameters.items()}
    delta = {name: after[name]-value for name, value in before.items()}
    return dict(gradients=gradients, updated=after, delta=delta,
        norms={name: float(value) for name, value in norms.items()})


def calibration_admission(receipt, data, bindings, source_sha):
    _, ids = paired_subset(data)
    if (receipt.get('version') != CALIBRATION_VERSION or receipt.get('passed') is not True
            or receipt.get('roles') != ['A', 'B'] or receipt.get('routes') != list(ROUTES)
            or receipt.get('shared_weight') is not True or receipt.get('gradient_ratio_target') != .1
            or receipt.get('estimator') != ESTIMATOR or receipt.get('subset_ids') != ids
            or type(receipt.get('head_seed')) is not int or not 0 <= receipt['head_seed'] < 2**63
            or type(receipt.get('loss_weight')) not in (int, float)
            or not math.isfinite(receipt['loss_weight']) or receipt['loss_weight'] <= 0
            or receipt.get('source_sha256') != source_sha
            or receipt.get('optimizer_updates') != 0):
        raise ValueError('Actual matching frozen-code calibration receipt required')
    for field, name in (('data_sha256', 'data'), ('frontend_sha256', 'frontend'), ('statistics_sha256', 'statistics')):
        if receipt.get(field) != {role: bindings[role]['files'][name] for role in ('A', 'B')}:
            raise ValueError('Complete calibration A/B source/data/statistics SHA map differs')
    if receipt.get('target_window_contract') != {role: bindings[role]['target_window_contract'] for role in ('A', 'B')}:
        raise ValueError('Calibration target window differs')
    if any(receipt.get('source_binding', {}).get(role, {}).get('files') != bindings[role]['files']
            for role in ('A', 'B')):
        raise ValueError('Calibration numeric/cache/configuration source bytes differ')


def register(output, registration):
    output = Path(output)
    started = output.with_suffix('.started.json')
    if any(p.exists() or p.is_symlink() for p in (output, started, output.with_suffix('.progress.json'))):
        raise FileExistsError('Preserve every acceptance attempt; use a new output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    with started.open('x') as stream:
        json.dump(registration, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return started


def run(root, calibration, output, session_root=None, *, learning_rate=3e-5):
    from visuotactile.precision import state_fp32
    from visuotactile.state_resources import lease_for
    root, calibration, output = Path(root).resolve(), Path(calibration).resolve(), Path(output).absolute()
    session_root = Path(session_root).resolve() if session_root else root
    project = Path(__file__).resolve().parents[1]
    registration = dict(version=VERSION, source_root=str(root), session_root=str(session_root),
        calibration_path=str(calibration), tolerance=TOLERANCES, optimizer_updates=0,
        formal_updates=0, formal_training_updates=0, device='cuda:0', simulator_starts=0,
        learning_rate=learning_rate, test_used=False, validation_used=False, utc=time.time())
    started = register(output, registration)
    rows = []; updates = 0; began = time.perf_counter()
    try:
        if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('Explicit finite positive common AdamW learning rate required')
        session = read(session_root/'session.json')
        if not math.isfinite(session['deadline']) or time.time() >= session['deadline']:
            raise TimeoutError('Registered session expired')
        calibration_sha = sha256(calibration); receipt = read(calibration)
        data = {role: read(root/role/'state_data/data.json') for role in ('A', 'B')}
        paired_subset(data)
        identity = first_original(data['A'])
        if first_original(data['B']) != identity:
            raise ValueError('Matched first original training episode required')
        bindings = {role: source_binding(root, role, data[role], [identity]) for role in ('A', 'B')}
        source_sha = code_hashes(project)
        calibration_admission(receipt, data, bindings, source_sha)
        registration.update(calibration_sha256=calibration_sha, source_sha256=source_sha,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip(),
            source_binding=bindings, source_metadata_sha256=sha256(root/'metadata.json'),
            session_sha256=sha256(session_root/'session.json'), deadline=session['deadline'],
            frontend_sha256={r: bindings[r]['files']['frontend'] for r in ('A', 'B')},
            data_sha256={r: bindings[r]['files']['data'] for r in ('A', 'B')},
            statistics_sha256={r: bindings[r]['files']['statistics'] for r in ('A', 'B')},
            target_window_contract={r: bindings[r]['target_window_contract'] for r in ('A', 'B')},
            identity=identity, episode_rule='First original in actual source train order, same A/B',
            roles=['A', 'B'], routes=list(ROUTES), head_seed=receipt['head_seed'], loss_weight=receipt['loss_weight'])
        atomic_json(output.with_suffix('.progress.json'), dict(registration, phase='admitted'))
        torch.set_num_threads(2); state_fp32()
        lease = lease_for(session_root, deadline=session['deadline'])
        for role in ('A', 'B'):
            family = data[role]['entries'][identity]['family']
            pose, proof = load_pose_labels(data[role]['entries'][identity], split='train', allowed_families=[family])
            with lease.block():
                records = visual_records(root/role/'state_data', [identity], 'G00', device='cuda:0',
                    target_window_contract=bindings[role]['target_window_contract'])
                record = records[identity]
                check_label_binding(record, proof, bindings[role]['entries'][identity])
                objective, geometry, geometry_proof = fresh_objective(root, role, data[role], records)
            variants = {}; candidate = None; initial_heads = {}
            for route in ('base', 'detached_readout', 'shared_vision'):
                with lease.block():
                    saved = torch.load(root/role/'training/best.pt', map_location='cpu', weights_only=False)
                    source_learning_rate = saved['configuration']['learning_rate']
                    model = load_policy(saved, 'cuda:0').train(); del saved
                    before_versions = {name: value._version for name, value in model.state_dict(keep_vars=True).items()}
                    spec = dict(route=route, head_seed=receipt['head_seed'], loss_weight=receipt['loss_weight'])
                    if route != 'base':
                        rng = check_head_rng(spec['head_seed'], next(model.parameters()).device)
                        attach_auxiliary_head(model, spec)
                        initial_heads[route] = head_sha(model.visual_pose_head)
                        if initial_heads[route] != rng['head_sha256']:
                            raise ValueError('Attached head initialization differs')
                        condition, anchors, visual = supervised_visual_conditions(model, record)
                        if not torch.equal(visual['ticks'], torch.arange(len(record['time']), device=anchors.device)):
                            raise ValueError('Every actual native tick must have a visual readout')
                    else:
                        condition, anchors = supervised_conditions(model, record)
                    if route == 'shared_vision':
                        pred = model.visual_pose_head(visual['readout'], route=route)
                        auxiliary = pose_loss(pred, pose, time_s=record['time'][visual['ticks']])
                        norm = gradient_norm(auxiliary['loss'], tuple(model.vision.parameters()), retain_graph=False)
                        if (auxiliary['valid_ticks'] <= 0 or any(p.grad is not None for p in model.parameters())
                                or before_versions != {name: value._version for name, value in model.state_dict(keep_vars=True).items()
                                    if not name.startswith('visual_pose_head.')}):
                            raise ValueError('Candidate gradient probe mutated parameters or lacks real labels')
                        candidate = dict(shared_vision_gradient_norm=norm, visual_ticks=len(visual['ticks']),
                            valid_ticks=auxiliary['valid_ticks'], loss=float(auxiliary['loss'].detach()),
                            optimizer_updates=0, rng=rng, head_sha256=initial_heads[route])
                        del pred, auxiliary, visual, condition, model
                    else:
                        truth, valid = labels(record, anchors, 24)
                        context = projection_batch(objective, [(identity, int(t)) for t in anchors], 'cuda:0')
                        draws = same_draws(truth, model)
                        loss, parts, prediction = objective.forward(model, condition, truth, valid, None,
                            geometry_context=context, diffusion_inputs=draws)
                        snapshot = dict(condition=condition.detach().cpu(), prediction=prediction.detach().cpu(),
                            parts={name: value.detach().cpu() for name, value in parts.items()},
                            action_loss=loss.detach().cpu(), anchors=tensor_sha(anchors), mask=tensor_sha(valid),
                            labels=tensor_sha(truth), diffusion=[tensor_sha(x) for x in draws],
                            geometry={name: tensor_sha(value) if isinstance(value, torch.Tensor) else value
                                for name, value in (context or {}).items()},
                            legal_anchors=len(anchors), legal_target_slots=int(valid.sum()))
                        if route != 'base':
                            extra, _ = auxiliary_episode_loss(model, record, visual, pose, spec)
                            loss = loss+extra
                            snapshot.update(rng=rng, visual_ticks=len(visual['ticks']))
                        snapshot.update(step_snapshot(model, loss, auxiliary=route != 'base', learning_rate=learning_rate))
                        updates += 1
                        variants[route] = snapshot
                        del snapshot, condition, prediction, loss, parts, draws, context, model
                        if route != 'base':
                            del visual, extra
                gc.collect(); torch.cuda.empty_cache()
                atomic_json(output.with_suffix('.progress.json'), dict(registration, role=role, route=route,
                    optimizer_updates=updates, phase='completed_disposable_variant', utc=time.time()))
            base, control = variants['base'], variants['detached_readout']
            for key in ('anchors', 'mask', 'labels', 'diffusion', 'geometry', 'legal_anchors', 'legal_target_slots'):
                if base[key] != control[key]:
                    raise ValueError('Base/control actual supervision or random sampling differs: '+key)
            if initial_heads['detached_readout'] != initial_heads['shared_vision']:
                raise ValueError('Candidate and control initial heads differ')
            row = dict(role=role, identity=identity, family=family, pose_label_proof=proof,
                hand_geometry=geometry, geometry_supervision=geometry_proof,
                condition=compare_tensors(base['condition'], control['condition'], 'forward'),
                prediction=compare_tensors(base['prediction'], control['prediction'], 'forward'),
                action_loss=compare_tensors(base['action_loss'], control['action_loss'], 'forward'),
                objective_parts=compare_maps(base['parts'], control['parts'], 'forward'),
                gradients=compare_maps(base['gradients'], control['gradients'], 'gradient'),
                updated_parameters=compare_maps(base['updated'], control['updated'], 'updated_parameter'),
                update_deltas=compare_maps(base['delta'], control['delta'], 'update_delta'),
                gradient_norms={key: variants[key]['norms'] for key in variants},
                action_identity={key: base[key] for key in ('anchors', 'mask', 'labels', 'diffusion', 'geometry')},
                legal_anchors=base['legal_anchors'], legal_target_slots=base['legal_target_slots'],
                candidate=candidate, control_rng=control['rng'], optimizer_updates=2,
                learning_rate=learning_rate, source_learning_rate=source_learning_rate)
            rows.append(row)
            del variants, base, control, records, record, objective
            gc.collect(); torch.cuda.empty_cache()
        if updates != 4:
            raise ValueError('Exactly four disposable optimizer updates required')
        lease.check()
        if (code_hashes(project) != source_sha or sha256(calibration) != calibration_sha
                or sha256(root/'metadata.json') != registration['source_metadata_sha256']
                or sha256(session_root/'session.json') != registration['session_sha256']
                or any(source_binding(root, r, data[r], [identity]) != bindings[r] for r in ('A', 'B'))):
            raise ValueError('Frozen code/calibration/source/session changed during acceptance')
        for row in rows:
            _, current_proof = load_pose_labels(data[row['role']]['entries'][identity],
                split='train', allowed_families=[row['family']])
            if current_proof != row['pose_label_proof']:
                raise ValueError('Actual pose/camera/authority source labels changed during acceptance')
        atomic_json(output, dict(registration, passed=True, rows=rows, optimizer_updates=updates,
            formal_updates=0, source_unchanged=True, test_used=False, validation_used=False,
            seconds=time.perf_counter()-began, source_receipt_sha256=sha256(started),
            interpretation='Actual training-path transport and optimizer acceptance only; no fitted model or autonomy result. Disposable updates are discarded.'))
    except BaseException as error:
        atomic_json(output, dict(registration, passed=False, rows=rows, optimizer_updates=updates,
            formal_updates=0, error_type=type(error).__name__, error=str(error),
            seconds=time.perf_counter()-began, source_receipt_sha256=sha256(started)))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session-root', type=Path)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    args = parser.parse_args()
    run(args.root, args.calibration, args.output, args.session_root, learning_rate=args.learning_rate)


if __name__ == '__main__':
    main()
