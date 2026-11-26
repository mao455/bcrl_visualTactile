"""Finite train-only B2 optimizer/EMA and actual checkpoint recovery diagnostic.

A new optimization method is compared with serial mean-gradient accumulation,
never with two sequential Adam steps. Results are not formal model candidates.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from tools.tight_offline_planning_gate import (admit, validate_checkpoint,
    train_projection, require, read, sha256, checked_hash, exclusive_json,
    import_frozen_trainer)
from tools.tight_trajectory_batch_gate import difference

VERSION = 'trajectory_batch_two_training_recovery_gate_v1'


def optimizer_spec(configuration):
    import math
    result=dict(lr=configuration['learning_rate'],betas=(.95,.999),weight_decay=configuration['weight_decay'])
    require(all(type(result[k]) in (int,float) and math.isfinite(result[k]) for k in ('lr','weight_decay'))
            and result['lr']>0 and result['weight_decay']>=0, 'Invalid frozen optimizer specification')
    return result


def compare_state(reference, candidate, *, exact=False, path='state'):
    """All optimizer/model/RNG fields are checked, including tree structure."""
    import torch
    results = {}
    if isinstance(reference, torch.Tensor):
        require(isinstance(candidate, torch.Tensor) and reference.shape == candidate.shape
                and reference.dtype == candidate.dtype, 'Tensor structure differs: ' + path)
        if exact or not reference.is_floating_point():
            results[path] = dict(passed=torch.equal(reference, candidate))
        else:
            results[path] = difference(reference, candidate)
    elif isinstance(reference, dict):
        require(isinstance(candidate, dict) and reference.keys() == candidate.keys(),
                'Dictionary structure differs: ' + path)
        for key in reference:
            results.update(compare_state(reference[key], candidate[key], exact=exact, path=path+'/'+str(key)))
    elif isinstance(reference, (list, tuple)):
        require(type(reference) is type(candidate) and len(reference) == len(candidate),
                'Sequence structure differs: ' + path)
        for index, value in enumerate(reference):
            results.update(compare_state(value, candidate[index], exact=exact, path=path+'/'+str(index)))
    else:
        results[path] = dict(passed=type(reference) is type(candidate) and reference == candidate)
    return results


def serial_update(trainer, model, optimizer, ema, objective, identities, records, episodes, noise, check):
    import torch
    from visuotactile.anchor_continue import projection_batch
    check(); model.train(); optimizer.zero_grad(set_to_none=True)
    reports = []; losses = []
    for identity, episode in zip(identities, episodes):
        check()
        condition, anchors = trainer.supervised_conditions(model, records[identity], episode=episode)
        truth, valid = trainer.labels(records[identity], anchors, 24)
        context = projection_batch(objective, [(identity, int(t)) for t in anchors], condition.device)
        loss, parts, _ = objective.forward(model, condition, truth, valid, noise, geometry_context=context)
        require(bool(torch.isfinite(loss)), 'Nonfinite serial reference loss')
        (loss/len(identities)).backward()
        losses.append(float(loss.detach()))
        reports.append(dict(episode=identity, anchors=len(anchors), target_ticks=int(valid.sum()),
            visual_frames=len(records[identity]['time']), loss_weight=1/len(identities)))
        del condition, anchors, truth, valid, context, loss, parts
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    require(bool(torch.isfinite(norm)), 'Nonfinite serial reference gradient')
    check(); optimizer.step(); ema.step(model)
    return dict(loss=sum(losses)/len(losses), gradient_norm=float(norm), visits=reports)


def load_candidate(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(root, role, frozen_root, output, resource_root, *, identities=None, check_only=False):
    root = Path(root).resolve(); resource_root = Path(resource_root).resolve()
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.time(); session = read(resource_root/'session.json')
    deadline = min(started+570, float(session['deadline'])-30)
    require(deadline > started, 'Existing session has no remaining gate budget')
    confirmed = dict(serial_reference=0, batched=0, restored=0)
    active = None
    exclusive_json(output/'started.json', dict(version=VERSION, started=started, pid=os.getpid(),
        deadline=deadline, original_deadline=session['deadline'], role=role,
        check_only=check_only, diagnostic_only=True, formal_training_candidate=False,
        maximum_diagnostic_optimizer_steps=5, comparison_atol=2e-5, comparison_rtol=2e-5,
        recovery_comparison='Exact model/EMA/optimizer/all RNG states after one restored singleton update'))
    def check():
        if time.time() >= deadline or any(p.exists() for p in (output/'STOP', resource_root/'STOP')):
            raise InterruptedError('Original deadline, finite diagnostic bound or STOP')
    try:
        receipt = admit(root, role, frozen_root, identities)
        require(len(receipt['identities']) == 2, 'Exactly two distinct train trajectories required')
        base = Path(__file__).resolve().parents[1]
        helpers = {name: base/'visuotactile'/(name+'.py') for name in ('trajectory_batch', 'trajectory_training')}
        receipt['explicit_candidate_helpers'] = {name:dict(path=str(path),sha256=sha256(path)) for name,path in helpers.items()}
        receipt.update(gate_source=str(Path(__file__).resolve()), gate_source_sha256=sha256(__file__))
        import torch
        saved = torch.load(receipt['paths']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        validate_checkpoint(saved, receipt)
        numeric = torch.load(receipt['paths']['numeric'], map_location='cpu', weights_only=False, mmap=True)
        receipt['projection'] = train_projection(output, read(receipt['paths']['data']), numeric, receipt)
        del numeric
        opt_spec=optimizer_spec(receipt['configuration'])
        receipt['optimizer_spec']=dict(type='AdamW',**opt_spec)
        receipt['registered_deck'] = [*receipt['identities'], receipt['identities'][0]]
        receipt['registered_batches'] = [receipt['identities'], [receipt['identities'][0]]]
        exclusive_json(output/'admission.json', receipt); check()
        if check_only:
            result = dict(complete=True, passed=None, check_only=True,
                cuda_initialized=torch.cuda.is_initialized(), optimizer_steps_confirmed=confirmed)
        else:
            trainer = import_frozen_trainer(receipt)
            from visuotactile.precision import state_fp32
            from visuotactile.state_resources import lease_for
            from visuotactile.target_windows import with_target_window
            from visuotactile.checkpointing import training_state, restore_training, save
            from visuotactile.models.vendor.ema_model import EMAModel
            for name, path in helpers.items():
                load_candidate(path, 'visuotactile.'+name)
            candidate = sys.modules['visuotactile.trajectory_training']
            state_fp32(); torch.cuda.set_per_process_memory_fraction(.45, device=0)
            lease = lease_for(resource_root, deadline=deadline)
            with lease.block():
                check()
                records = trainer.visual_records(receipt['projection']['root'], receipt['identities'], 'G00', device='cuda:0')
                records = with_target_window(records, receipt['target_window'])
                data = read(receipt['paths']['data'])
                objective, geometry, proof = trainer.fresh_objective(root, role, data, records)
                statistics = torch.load(receipt['paths']['statistics'], map_location='cpu', weights_only=False)
                model = trainer.new_model(role, statistics).train()
                require(model.contract == saved['contract'], 'Frozen constructor/checkpoint contract drift')
                model.load_state_dict(saved['ema'], strict=True)
                averaged = copy.deepcopy(model).eval().requires_grad_(False)
                ema = EMAModel(averaged, power=.75)
                optimizer = torch.optim.AdamW(model.parameters(), **opt_spec)
                schedule = torch.Generator().manual_seed(2026111501)
                noise = torch.Generator(device='cuda:0').manual_seed(990031)
                initial = training_state(model, averaged, optimizer, ema, schedule, noise)
                batches = receipt['registered_batches']
                reports = {}; snapshots = {}; timings = {}; peaks = {}
                def snapshot(updates, visits):
                    return training_state(model, averaged, optimizer, ema, schedule, noise,
                        updates=updates, visits_seen=visits, full_trajectory_batch=candidate.profile())
                def verify_time():
                    check(); lease.check()
                for arm in ('serial_reference', 'batched'):
                    restore_training(initial, model, averaged, optimizer, ema, schedule, noise)
                    reports[arm] = []; visits = 0
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); began=time.perf_counter()
                    for number, ids in enumerate(batches, start=1):
                        verify_time(); active=dict(arm=arm, update=number, identities=ids)
                        episodes = [trainer.Episode(receipt['projection']['root'], identity, 'I-HistoryDP') for identity in ids]
                        if arm == 'serial_reference':
                            report=serial_update(trainer,model,optimizer,ema,objective,ids,records,episodes,noise,verify_time)
                        else:
                            report=candidate.execute_update(model,optimizer,ema,objective,ids,
                                [records[i] for i in ids],episodes,noise,check=verify_time)
                        confirmed[arm]+=1; visits+=len(ids); active=None; reports[arm].append(report)
                        del episodes
                        require(ema.optimization_step == number, 'EMA steps differ from optimizer steps')
                        if arm == 'batched' and number == 1:
                            save(output/'diagnostic_boundary.pt',snapshot(number,visits))
                        verify_time()
                    torch.cuda.synchronize(); timings[arm]=time.perf_counter()-began
                    peaks[arm]=torch.cuda.max_memory_allocated(); snapshots[arm]=snapshot(2,3)
                checks=compare_state(snapshots['serial_reference'],snapshots['batched'],path='reference_vs_batched')
                del snapshots['serial_reference']
                # New objects plus serialized complete state, not an in-memory optimizer rewind.
                del model, averaged, optimizer, ema, initial
                import gc
                gc.collect(); torch.cuda.empty_cache(); verify_time()
                model=trainer.new_model(role,statistics).train()
                averaged=copy.deepcopy(model).eval().requires_grad_(False); ema=EMAModel(averaged,power=.75)
                optimizer=torch.optim.AdamW(model.parameters(),**opt_spec)
                schedule=torch.Generator().manual_seed(11);noise=torch.Generator(device='cuda:0').manual_seed(12)
                boundary=torch.load(output/'diagnostic_boundary.pt',map_location='cpu',weights_only=False)
                require(boundary['updates']==1 and boundary['visits_seen']==2 and boundary['full_trajectory_batch']==candidate.profile(),
                    'Saved boundary budget/profile differs')
                restore_training(boundary,model,averaged,optimizer,ema,schedule,noise)
                boundary_restored=snapshot(1,2)
                boundary_checks=compare_state(boundary,boundary_restored,exact=True,path='saved_vs_restored_boundary')
                del boundary,boundary_restored
                ids=batches[1];active=dict(arm='restored',update=2,identities=ids)
                episodes=[trainer.Episode(receipt['projection']['root'],i,'I-HistoryDP') for i in ids]
                began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                reports['restored']=[candidate.execute_update(model,optimizer,ema,objective,ids,
                    [records[i] for i in ids],episodes,noise,check=verify_time)]
                confirmed['restored']+=1;active=None;torch.cuda.synchronize()
                timings['restored_singleton']=time.perf_counter()-began
                peaks['restored_singleton']=torch.cuda.max_memory_allocated()
                recovered=snapshot(2,3);verify_time()
                recovery=compare_state(snapshots['batched'],recovered,exact=True,path='uninterrupted_vs_recovered')
                for arm in ('serial_reference','batched'):
                    expected=[(i,1/len(ids)) for ids in batches for i in ids]
                    actual=[(v['episode'],v['loss_weight']) for row in reports[arm] for v in row['visits']]
                    require(actual==expected,'Ordered visit/tail/repeat ledger differs')
                for index in range(2):
                    ref=reports['serial_reference'][index]; cand=reports['batched'][index]
                    checks['update'+str(index+1)+'/loss']=difference(torch.tensor(ref['loss']),torch.tensor(cand['loss']))
                    for a,b in zip(ref['visits'],cand['visits']):
                        require(all(a[k]==b[k] for k in a),'Supervision exposure differs')
                for name,path in helpers.items(): checked_hash(path,receipt['explicit_candidate_helpers'][name]['sha256'])
                checked_hash(receipt['paths']['checkpoint'],receipt['sha256']['checkpoint']);verify_time()
                result=dict(complete=True,passed=all(v['passed'] for v in [*checks.values(),*boundary_checks.values(),*recovery.values()]),
                    diagnostic_only=True,formal_training_candidate=False,role=role,optimizer_spec=receipt['optimizer_spec'],
                    optimizer_steps_confirmed=confirmed,registered_deck=receipt['registered_deck'],
                    optimizer_updates_per_path=2,episode_visits_per_path=3,ema_updates_per_path=2,
                    comparisons=checks,boundary_restoration_comparisons=boundary_checks,recovery_comparisons=recovery,reports=reports,
                    timings_seconds=timings,peak_allocated_bytes=peaks,per_process_allocator_fraction=.45,
                    geometry=geometry,geometry_selection_proof=proof,
                    boundary_checkpoint_sha256=sha256(output/'diagnostic_boundary.pt'),
                    source_checkpoint_sha256=receipt['sha256']['checkpoint'],
                    limitations='Two training trajectories and three visits; serial then batched warm order and checkpoint I/O differ. Fresh EMA uses zero decay in these first two updates; exact pre-update boundary checks separately verify restored EMA. No epoch throughput, model selection, validation/test or autonomy claim.')
        check();result.update(wall_seconds=time.time()-started,finished_utc=time.time())
        exclusive_json(output/'result.json',result)
        return result
    except BaseException as error:
        exclusive_json(output/'failed.json',dict(error=repr(error),traceback=traceback.format_exc(),
            optimizer_steps_confirmed=confirmed,active_update_may_be_partial=active,
            elapsed_seconds=time.time()-started,utc=time.time()))
        raise


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('root','frozen-root','output','resource-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--role',choices=['A','B'],required=True)
    parser.add_argument('--identity',action='append',dest='identities')
    parser.add_argument('--check-only',action='store_true')
    result=run(**vars(parser.parse_args()))
    print(json.dumps({k:result[k] for k in ('complete','passed','wall_seconds')},sort_keys=True),flush=True)
    if result['passed'] is False:raise SystemExit(2)


if __name__=='__main__':main()
