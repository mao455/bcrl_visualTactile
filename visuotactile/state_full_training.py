"""Resumable fixed-split recurrent training; milestone budgets do not reset LR/EMA."""
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import time

import torch

from .compact import sha256
from .models.vendor.ema_model import EMAModel
from .state_history import PersistentState, STATE_CONTRACT, load_records
from .state_history_training import labels, require_foundation, evaluate
from .state_resources import lease_for
from .train_long import cpu_snapshot, save
from .training_resources import atomic_json


def split_ids(session, data, pilot):
    train = ([s['identity'] for s in session['saved_specs'] if s['cohort'] == 'train']
        if pilot else data['splits']['train'])
    validation = ([s['identity'] for s in session['saved_specs'] if s['cohort'] == 'validation']
        if pilot else data['splits']['validation'])
    assert set(train) <= set(data['splits']['train'])
    assert set(validation) <= set(data['splits']['validation'])
    assert not set(train) & set(validation)
    assert not (set(train) | set(validation)) & set(data['splits']['test'])
    if not pilot and (len(train), len(validation), len(data['splits']['test'])) != (320, 40, 40):
        raise ValueError('Original split required')
    return train, validation


def training_state(model, averaged, optimizer, ema, generator, noise, **metadata):
    return cpu_snapshot(dict(contract=STATE_CONTRACT, model=model.state_dict(), ema=averaged.state_dict(),
        optimizer=optimizer.state_dict(), ema_optimization_step=ema.optimization_step,
        schedule_rng=generator.get_state(), noise_rng=noise.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), **metadata))


def restore_training(saved, model, averaged, optimizer, ema, generator, noise):
    if saved['contract'] != STATE_CONTRACT:
        raise ValueError('Checkpoint contract mismatch')
    model.load_state_dict(saved['model']); averaged.load_state_dict(saved['ema'])
    optimizer.load_state_dict(saved['optimizer']); ema.optimization_step = saved['ema_optimization_step']
    generator.set_state(saved['schedule_rng']); noise.set_state(saved['noise_rng'])
    torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])


def train(root, output, kind, target, pilot=False, final_test=False):
    require_foundation(root)
    output.mkdir(parents=True, exist_ok=True)
    session = json.loads((root/'session.json').read_text()); source = Path(session['source_root'])
    data = json.loads((source/'data.json').read_text()); train_ids, val_ids = split_ids(session, data, pilot)
    directory = root/(kind+('_pilot_train' if pilot else '_train')); directory.mkdir(exist_ok=True)
    lease = lease_for(root, session['deadline'] if final_test else session['training_deadline'])
    maximum = session['pilot_updates'] if pilot else session[kind+'_updates']
    if target > maximum:
        raise ValueError('Registered update budget exceeded')
    config = dict(kind=kind, contract=STATE_CONTRACT, train_ids=train_ids, validation_ids=val_ids,
        statistics_sha256=sha256(source/'statistics.pt'), maximum_updates=maximum, segment_ticks=96,
        episodes_per_batch=2, lr=1e-4, betas=[.95, .999], weight_decay=1e-6, warmup=500,
        schedule='one fixed cosine horizon, never restarted at milestones', loss='B', seed=20261102,
        expert_normalizer_sha256=session['expert_sha256'], source_sha256=session['implementation_sha256'])
    if (directory/'configuration.json').exists():
        if json.loads((directory/'configuration.json').read_text()) != config:
            raise ValueError('Training configuration drift')
    else:
        atomic_json(directory/'configuration.json', config)
    ids = train_ids+val_ids
    if final_test:
        selection = json.loads((root/(kind+'_selection.json')).read_text())
        if sha256(directory/'best.pt') != selection['checkpoint_sha256']:
            raise ValueError('Selection must freeze before test')
        ids = data['splits']['test']
    cache = root/('numeric_test_cache.pt' if final_test else 'numeric_trainval_cache.pt')
    if not cache.exists():
        records = load_records(source, ids if final_test else data['splits']['train']+data['splits']['validation'])
        save(cache, dict(data_sha256=session['data_sha256'], records=records))
    cached = torch.load(cache, map_location='cpu', weights_only=False)
    if cached['data_sha256'] != session['data_sha256']:
        raise ValueError('Numeric cache source mismatch')
    records = {i: cached['records'][i] for i in ids}; del cached
    stats = torch.load(source/'statistics.pt', map_location='cpu', weights_only=False)
    expert = torch.load(session['expert_checkpoint'], map_location='cpu', weights_only=False)[0]['model']
    with lease.block():
        torch.manual_seed(20261102)
        model = PersistentState(kind); model.set_statistics(stats, expert); model.cuda()
        averaged = copy.deepcopy(model).eval().requires_grad_(False); ema = EMAModel(averaged, power=.75)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.95, .999), weight_decay=1e-6)
        records = {i: {k: v.cuda() if torch.is_tensor(v) else v for k, v in r.items()} for i, r in records.items()}
    del expert
    if final_test:
        saved = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
        averaged.load_state_dict(saved['ema'])
        metrics = evaluate(averaged, records, lease)
        atomic_json(output/'result.json', dict(software_complete=True, test=metrics,
            checkpoint_sha256=selection['checkpoint_sha256'], selection_frozen=True, updates=saved['updates']))
        return
    schedule = [(i, start) for i in train_ids for start in range(0, len(records[i]['time']), 96)]
    generator = torch.Generator().manual_seed(20261102)
    noise = torch.Generator(device='cuda:0').manual_seed(20261102)
    best = math.inf; cursor = 0; order = []; update = 0; seen = set(); history = []
    if (directory/'last.pt').exists():
        saved = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
        if saved['config_sha256'] != sha256(directory/'configuration.json'):
            raise ValueError('Resume provenance mismatch')
        restore_training(saved, model, averaged, optimizer, ema, generator, noise)
        update = saved['updates']; cursor = saved['schedule_cursor']; order = saved['schedule_order']
        best = saved['best']; history = saved['history']; seen = set(saved['seen_ids'])
        if update > target:
            raise ValueError('Cannot repeat an already trained milestone')
        atomic_json(output/'resume.json', dict(from_updates=update, checkpoint_sha256=sha256(directory/'last.pt'),
            exact_rng_optimizer_ema_schedule=True))
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    began = time.time(); start_update = update; losses = []; history_grads = []; pending_schedule = None
    def snapshot():
        return training_state(model, averaged, optimizer, ema, generator, noise, kind=kind, group='S-NativeDP',
            updates=update, schedule_order=order, schedule_cursor=cursor, best=best, history=history,
            seen_ids=sorted(seen), train_ids=train_ids, validation_ids=val_ids,
            config_sha256=sha256(directory/'configuration.json'))
    try:
        while update < target:
            pending_schedule = (order, cursor, generator.get_state())
            selected = []
            for _ in range(2):
                if cursor == len(order):
                    order = torch.randperm(len(schedule), generator=generator).tolist(); cursor = 0
                selected.append(schedule[order[cursor]]); cursor += 1
            with lease.block():
                model.train(); optimizer.zero_grad(set_to_none=True)
                conditions = []; commands = []; masks = []
                for identity, start in selected:
                    seen.add(identity); record = records[identity]
                    condition, anchors, _ = model.encode(record, start, min(start+96, len(record['time'])))
                    command, valid = labels(record, anchors)
                    conditions.append(condition); commands.append(command); masks.append(valid)
                loss = model.loss(torch.cat(conditions), torch.cat(commands), torch.cat(masks), noise)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                history_grad = float(model.history.weight_hh_l0.grad.norm())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm) or not math.isfinite(history_grad):
                    raise FloatingPointError('Nonfinite gradient')
                step = update+1
                scale = min(1., step/500)*.5*(1+math.cos(math.pi*max(0, step-500)/max(1, maximum-500)))
                for group in optimizer.param_groups:
                    group['lr'] = 1e-4*scale
                optimizer.step(); ema.step(model); update = step; pending_schedule = None
            losses.append(float(loss.detach())); history_grads.append(history_grad)
            with (directory/'train.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(update=update, loss=losses[-1], norm=float(norm), history_grad=history_grad,
                    lr=optimizer.param_groups[0]['lr'], elapsed_s=time.time()-began, utc=time.time(),
                    seen_episodes=len(seen), work_s=lease.work_s, wait_s=lease.wait_s, pacing_s=lease.pacing_s))+'\n')
            if update % 250 == 0 or stop[0]:
                save(directory/'last.pt', snapshot())
                atomic_json(directory/'progress.json', dict(updates=update, target=target, seen_episodes=len(seen),
                    updates_per_s=(update-start_update)/(time.time()-began), utc=time.time()))
            if stop[0]:
                raise InterruptedError('Checkpoint saved at complete optimizer boundary')
        train_metrics = evaluate(averaged, {i: records[i] for i in train_ids[:16]}, lease)
        metrics = dict(update=update, train=train_metrics, train_metric_scope='fixed first16 train (pilot4)',
            validation=evaluate(averaged, {i: records[i] for i in val_ids}, lease))
        history = [row for row in history if row['update'] != update]+[metrics]
        improved = metrics['validation']['selection_mse'] < best
        if improved:
            best = metrics['validation']['selection_mse']
        payload = snapshot(); save(directory/'last.pt', payload); save(directory/f'milestone_{update}.pt', payload)
        if improved:
            save(directory/'best.pt', payload)
        atomic_json(directory/'metrics.json', history)
        result = dict(software_complete=True, updates=update, train_episodes=len(train_ids), validation_episodes=len(val_ids),
            unique_train_episodes_seen=len(seen), history=history, best_sha256=sha256(directory/'best.pt'),
            first25_loss=sum(losses[:25])/len(losses[:25]) if losses else None,
            last25_loss=sum(losses[-25:])/len(losses[-25:]) if losses else None,
            history_gradient_nonzero=any(v > 0 for v in history_grads), elapsed_s=time.time()-began,
            updates_per_s=(update-start_update)/(time.time()-began), test_used=False,
            pilot=pilot, resource_wait_s=lease.wait_s, pacing_s=lease.pacing_s)
        atomic_json(output/'result.json', result); atomic_json(directory/'result.json', result)
    except (InterruptedError, TimeoutError):
        if pending_schedule is not None:
            order, cursor, state = pending_schedule; generator.set_state(state)
        save(directory/'last.pt', snapshot())
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--kind', choices=['bc', 'dp'], required=True)
    p.add_argument('--target', type=int, required=True); p.add_argument('--pilot', action='store_true')
    p.add_argument('--final-test', action='store_true'); a = p.parse_args(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    train(a.root, a.output, a.kind, a.target, a.pilot, a.final_test)
