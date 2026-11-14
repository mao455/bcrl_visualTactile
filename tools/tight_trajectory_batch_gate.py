"""Bounded train-only forward/gradient check of full-trajectory batch two.

No optimizer is created and no checkpoint is written. Compare the candidate
with the mean serial gradients evaluated at exactly the same parameters, NOT
with two successive optimizer updates. Use a fresh process with a 600s external
owner bound and the existing session deadline.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback
import sys

from tools.tight_offline_planning_gate import (admit, validate_checkpoint,
    train_projection, require, read, sha256, checked_hash, exclusive_json,
    import_frozen_trainer)

VERSION = 'trajectory_batch_two_gradient_gate_v1'
TOLERANCE = 2e-5


def difference(reference, candidate):
    import torch
    require(reference.shape == candidate.shape, 'Compared tensor shapes differ')
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    delta = (reference.double() - candidate.double()).abs()
    allowed = TOLERANCE + TOLERANCE * reference.double().abs()
    return dict(passed=finite and bool((delta <= allowed).all()), finite=finite,
        maximum_absolute=float(delta.max()), maximum_tolerance_ratio=float((delta/allowed).max()))


def run(root, role, frozen_root, output, resource_root, *, identities=None, check_only=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    session = read(Path(resource_root)/'session.json')
    deadline = min(started + 570, float(session['deadline']) - 30)
    require(deadline > started, 'Existing session has no remaining gate budget')
    exclusive_json(output/'started.json', dict(version=VERSION, pid=os.getpid(),
        started=started, deadline=deadline, original_deadline=session['deadline'],
        resource_root=str(Path(resource_root).resolve()), role=role, check_only=check_only,
        tolerance=dict(atol=TOLERANCE, rtol=TOLERANCE), optimizer_updates=0,
        comparison='Mean serial gradients at identical parameters, not sequential Adam updates'))
    def check():
        if time.time() >= deadline or any(p.exists() for p in
                (output/'STOP', Path(resource_root)/'STOP')):
            raise InterruptedError('Original deadline, finite gate bound or STOP')
    try:
        receipt = admit(root, role, frozen_root, identities)
        require(len(receipt['identities']) == 2, 'Exactly two different training trajectories required')
        helper = Path(__file__).resolve().parents[1]/'visuotactile/trajectory_batch.py'
        receipt['candidate_helper'] = str(helper)
        receipt['candidate_helper_sha256'] = sha256(helper)
        receipt['gate_source'] = str(Path(__file__).resolve())
        receipt['gate_source_sha256'] = sha256(__file__)
        import torch
        saved = torch.load(receipt['paths']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        validate_checkpoint(saved, receipt)
        numeric = torch.load(receipt['paths']['numeric'], map_location='cpu', weights_only=False, mmap=True)
        receipt['projection'] = train_projection(output, read(receipt['paths']['data']), numeric, receipt)
        del numeric
        exclusive_json(output/'admission.json', receipt)
        check()
        if check_only:
            result = dict(complete=True, passed=None, check_only=True,
                cuda_initialized=torch.cuda.is_initialized())
        else:
            trainer = import_frozen_trainer(receipt)
            from visuotactile.precision import state_fp32
            from visuotactile.state_resources import lease_for
            from visuotactile.target_windows import with_target_window
            from visuotactile.anchor_continue import projection_batch
            # The baseline dependencies remain from their audited frozen tree.
            # Load only this explicitly hashed new helper into that package.
            candidate_name = 'visuotactile._trajectory_batch_explicit_candidate'
            spec = importlib.util.spec_from_file_location(candidate_name, helper)
            candidate = importlib.util.module_from_spec(spec)
            sys.modules[candidate_name] = candidate
            spec.loader.exec_module(candidate)
            state_fp32()
            torch.cuda.set_per_process_memory_fraction(.45, device=0)
            lease = lease_for(Path(resource_root).resolve(), deadline=deadline)
            with lease.block():
                check()
                records = trainer.visual_records(receipt['projection']['root'], receipt['identities'],
                    'G00', device='cuda:0')
                records = with_target_window(records, receipt['target_window'])
                statistics = torch.load(receipt['paths']['statistics'], map_location='cpu', weights_only=False)
                model = trainer.new_model(role, statistics).train()
                require(model.contract == saved['contract'], 'Frozen constructor/checkpoint contract drift')
                model.load_state_dict(saved['ema'], strict=True)
                data = read(receipt['paths']['data'])
                objective, geometry, proof = trainer.fresh_objective(Path(root), role, data, records)
                ids = receipt['identities']
                episodes = [trainer.Episode(receipt['projection']['root'], identity, 'I-HistoryDP') for identity in ids]
                contexts = []
                for identity in ids:
                    from visuotactile.anchor_sampling import planning_anchors
                    contexts.append(projection_batch(objective,
                        [(identity, int(t)) for t in planning_anchors(records[identity])], 'cuda:0'))
                serial_generator = torch.Generator(device='cuda:0').manual_seed(990031)
                serial_rows, serial_losses = [], []
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                began = time.perf_counter()
                for identity, episode, context in zip(ids, episodes, contexts):
                    check()
                    cond, anchors = trainer.supervised_conditions(model, records[identity], episode=episode)
                    truth, valid = trainer.labels(records[identity], anchors, 24)
                    loss, parts, prediction = objective.forward(model, cond, truth, valid,
                        serial_generator, geometry_context=context)
                    (loss/len(ids)).backward()
                    serial_rows.append(dict(condition=cond.detach().cpu(), anchors=anchors.detach().cpu(),
                        truth=truth.detach().cpu(), valid=valid.detach().cpu(), prediction=prediction.detach().cpu(),
                        parts={k:float(v.detach()) for k,v in parts.items()}))
                    serial_losses.append(float(loss.detach()))
                    del cond, anchors, truth, valid, loss, parts, prediction
                torch.cuda.synchronize()
                check()
                serial_seconds = time.perf_counter()-began
                serial_peak = torch.cuda.max_memory_allocated()
                gradients = {name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
                model.zero_grad(set_to_none=True)
                candidate_generator = torch.Generator(device='cuda:0').manual_seed(990031)
                check(); lease.check()
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                began = time.perf_counter()
                batches = candidate.supervised_trajectories(model, [records[i] for i in ids], episodes)
                loss, reports, predictions = candidate.trajectory_objective(model, objective, batches, candidate_generator, contexts)
                loss.backward()
                torch.cuda.synchronize()
                check()
                candidate_seconds = time.perf_counter()-began
                candidate_peak = torch.cuda.max_memory_allocated()
                checks = {}
                offset = 0
                for identity, reference, candidate in zip(ids, serial_rows, batches):
                    end = offset+len(candidate['condition'])
                    for field in ('condition', 'truth'):
                        checks[identity+'/'+field] = difference(reference[field], candidate[field].detach().cpu())
                    for field in ('anchors', 'valid'):
                        checks[identity+'/'+field] = dict(passed=torch.equal(reference[field], candidate[field].detach().cpu()))
                    checks[identity+'/prediction'] = difference(reference['prediction'], predictions[offset:end].detach().cpu())
                    offset = end
                checks['loss'] = difference(torch.tensor(sum(serial_losses)/len(ids)), loss.detach().cpu())
                checks['noise_state'] = dict(passed=torch.equal(serial_generator.get_state(), candidate_generator.get_state()))
                actual_names = {name for name,p in model.named_parameters() if p.grad is not None}
                require(actual_names == set(gradients), 'Gradient parameter population differs')
                gradient_checks = {name:difference(expected, dict(model.named_parameters())[name].grad.detach().cpu())
                    for name,expected in gradients.items()}
                passed = all(row['passed'] for row in (*checks.values(), *gradient_checks.values()))
                checked_hash(helper, receipt['candidate_helper_sha256'])
                checked_hash(receipt['paths']['checkpoint'], receipt['sha256']['checkpoint'])
                check(); lease.check()
                result = dict(complete=True, passed=passed, checks=checks, gradient_checks=gradient_checks,
                    role=role, identities=ids, full_history_frames=[len(records[i]['time']) for i in ids],
                    optimizer_updates=0, checkpoint_sha256=receipt['sha256']['checkpoint'],
                    geometry=geometry, geometry_selection_proof=proof,
                    serial_seconds=serial_seconds, candidate_seconds=candidate_seconds,
                    serial_peak_allocated_bytes=serial_peak, candidate_peak_allocated_bytes=candidate_peak,
                    per_process_allocator_fraction=.45,
                    training_exposure=[dict(identity=i, anchors=r['anchors'], target_ticks=r['target_ticks']) for i,r in zip(ids,reports)],
                    limitations='Two train trajectories, serial then batch warm order and concurrent jobs; no epoch speed, optimizer equivalence, validation or autonomous claim')
        check()
        result.update(wall_seconds=time.time()-started, finished_utc=time.time())
        exclusive_json(output/'result.json', result)
        return result
    except BaseException as error:
        exclusive_json(output/'failed.json', dict(error=repr(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.time()-started, utc=time.time(), optimizer_updates=0))
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('root', 'frozen-root', 'output', 'resource-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--role', choices=['A','B'], required=True)
    parser.add_argument('--identity', action='append', dest='identities')
    parser.add_argument('--check-only', action='store_true')
    args = vars(parser.parse_args())
    result = run(**args)
    print(json.dumps({k:result[k] for k in ('complete','passed','wall_seconds')}, sort_keys=True), flush=True)
    if result['passed'] is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
