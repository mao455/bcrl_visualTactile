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
    return cpu_snapshot(dict(contract=getattr(model,'contract',STATE_CONTRACT), model=model.state_dict(), ema=averaged.state_dict(),
        optimizer=optimizer.state_dict(), ema_optimization_step=ema.optimization_step,
        schedule_rng=generator.get_state(), noise_rng=noise.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), **metadata))


def restore_training(saved, model, averaged, optimizer, ema, generator, noise):
    if saved['contract'] != getattr(model,'contract',STATE_CONTRACT):
        raise ValueError('Checkpoint contract mismatch')
    model.load_state_dict(saved['model']); averaged.load_state_dict(saved['ema'])
    optimizer.load_state_dict(saved['optimizer']); ema.optimization_step = saved['ema_optimization_step']
    generator.set_state(saved['schedule_rng']); noise.set_state(saved['noise_rng'])
    torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])


def validate_training_configuration(config, path, approvals=()):
    """Keep checkpoint identity frozen; accept only explicitly reviewed hash pairs."""
    frozen = json.loads(path.read_text())
    if frozen == config:
        return None
    old_sources = frozen.get('source_sha256', {})
    new_sources = config.get('source_sha256', {})
    unchanged_recipe = dict(config, source_sha256=old_sources) == frozen
    if not unchanged_recipe or old_sources.keys() != new_sources.keys():
        raise ValueError('Training configuration drift')
    changes = {p: dict(before=old_sources[p], after=new_sources[p])
        for p in old_sources if old_sources[p] != new_sources[p]}
    runtime_files = {'visuotactile/state_full_run.py', 'visuotactile/state_resources.py',
        'visuotactile/state_full_training.py'}
    if not changes or not set(changes) <= runtime_files:
        raise ValueError('Unreviewed training source drift')
    for approval in approvals:
        if (approval.get('configuration_sha256') == sha256(path)
                and approval.get('source_changes') == changes):
            return approval
    raise ValueError('Training source changes require exact reviewed hash pairs')


def train(root, output, kind, target, pilot=False, final_test=False, profile=None):
    require_foundation(root)
    output.mkdir(parents=True, exist_ok=True)
    session = json.loads((root/'session.json').read_text()); source = Path(session['source_root'])
    if session.get('position_study') and (kind != 'dp' or profile is None):
        raise ValueError('Position study authorizes only explicit DP profiles; no BC updates')
    data = json.loads((source/'data.json').read_text()); train_ids, val_ids = split_ids(session, data, pilot)
    study=session.get('profiles',{}).get(profile) if profile else None
    if profile and (study is None or kind!='dp' or pilot):
        raise ValueError('Unknown DP-only profile')
    model_contract=study['contract'] if study else STATE_CONTRACT
    group=study.get('group','S-NativeDP') if study else 'S-NativeDP'
    image=group!='S-NativeDP'
    if image:
        from .state_full_run import image_success_gate
        gate=json.loads((root/'image_gate.json').read_text())
        cohort_path=root/gate['validation_stage']/'result.json'
        if (not gate['passed'] or gate['validation_sha256']!=sha256(cohort_path)
                or gate['checkpoint_sha256']!=sha256(Path(gate['checkpoint']))
                or gate['foundation_sha256']!=sha256(root/'foundation.json')
                or not image_success_gate(json.loads(cohort_path.read_text()),gate['registered_validation'])['passed']
                or not json.loads((root/'position_audit/result.json').read_text())['visual_alignment_passed']):
            raise ValueError('Image training needs intact State success/control/visual alignment evidence')
    directory = root/((profile or kind)+('_pilot_train' if pilot else '_train')); directory.mkdir(exist_ok=True)
    lease = lease_for(root, session['deadline'] if final_test else session['training_deadline'])
    maximum = study['updates'] if study else session['pilot_updates'] if pilot else session[kind+'_updates']
    optimizer_horizon=study.get('optimizer_horizon',maximum) if study else maximum
    if target > maximum:
        raise ValueError('Registered update budget exceeded')
    config = dict(kind=kind, contract=model_contract, train_ids=train_ids, validation_ids=val_ids,
        statistics_sha256=sha256(source/'statistics.pt'), maximum_updates=maximum, segment_ticks=96,
        episodes_per_batch=2, lr=1e-4, betas=[.95, .999], weight_decay=1e-6, warmup=500,
        schedule='one fixed cosine horizon, never restarted at milestones', loss='B', seed=20261102,
        expert_normalizer_sha256=session['expert_sha256'], source_sha256=session['implementation_sha256'])
    if study:config.update(group=group,optimizer_horizon=optimizer_horizon)
    ids = train_ids+val_ids
    if final_test:
        selection = json.loads((root/((profile or kind)+'_selection.json')).read_text())
        if sha256(directory/'best.pt') != selection['checkpoint_sha256']:
            raise ValueError('Selection must freeze before test')
        ids = data['splits']['test']
    cache = root/((profile+'_' if profile else '')+('numeric_test_cache.pt' if final_test else 'numeric_trainval_cache.pt'))
    if not cache.exists():
        records = load_records(source, ids if final_test else data['splits']['train']+data['splits']['validation'],model_contract)
        save(cache, dict(data_sha256=session['data_sha256'], contract=model_contract,records=records))
    cached = torch.load(cache, map_location='cpu', weights_only=False)
    if cached['data_sha256'] != session['data_sha256']:
        raise ValueError('Numeric cache source mismatch')
    if study and cached.get('contract')!=model_contract:
        raise ValueError('Numeric cache action contract mismatch')
    records = {i: cached['records'][i] for i in ids}; del cached
    stats = torch.load(source/'statistics.pt', map_location='cpu', weights_only=False)
    if model_contract['action']=='absolute_joint_position_target_rad':
        train_robot=torch.cat([records[i]['robot'] for i in train_ids]) if not final_test else None
        stat_path=root/'position_train_statistics.pt'
        if not stat_path.exists():
            if final_test:raise ValueError('Training-only position statistics missing')
            stats['robot_mean'][116:176]=train_robot.mean(0)[116:176]
            stats['robot_std'][116:176]=train_robot.std(0).clamp_min(1e-4)[116:176]
            stats.update(torch.load(root/'position_audit/position_statistics.pt',map_location='cpu',weights_only=False))
            save(stat_path,stats)
        stats=torch.load(stat_path,map_location='cpu',weights_only=False)
        del train_robot
        config.update(position_statistics_sha256=sha256(stat_path),
            target_statistics_sha256=sha256(root/'position_audit/position_statistics.pt'))
    if image:
        records={i:dict(**{k:r[k] for k in ('robot','command','time','joint_limits')},
            identity=i,source_root=str(source)) for i,r in records.items()}
        config['input_allowlist']=['robot178','rgbd','past_position_rad','timestamps','valid']
        config['expert_normalizer_sha256']=None
    if (directory/'configuration.json').exists():
        compatibility = validate_training_configuration(config, directory/'configuration.json',
            session.get('training_runtime_compatibility', []))
        if compatibility is not None:
            atomic_json(output/'runtime_configuration_compatibility.json', compatibility)
    else:
        atomic_json(directory/'configuration.json', config)
    expert = None if image else torch.load(session['expert_checkpoint'], map_location='cpu', weights_only=False)[0]['model']
    with lease.block():
        torch.manual_seed(20261102)
        if image:
            from .state_history import PersistentImage
            model=PersistentImage(group,model_contract)
            for key,value in stats.items():getattr(model,key).copy_(value)
        else:
            model = PersistentState(kind,model_contract); model.set_statistics(stats, expert)
        model.cuda()
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
        return training_state(model, averaged, optimizer, ema, generator, noise, kind=kind, group=group,
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
                    command, valid = labels(record, anchors,model.horizon)
                    conditions.append(condition); commands.append(command); masks.append(valid)
                loss = model.loss(torch.cat(conditions), torch.cat(commands), torch.cat(masks), noise)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                branch=model.native_gru.weight_hh if group=='I-HistoryDP' else model.vision.keypoints.weight if image else model.history.weight_hh_l0
                history_grad = float(branch.grad.norm()) if branch.grad is not None else 0.
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm) or not math.isfinite(history_grad):
                    raise FloatingPointError('Nonfinite gradient')
                step = update+1
                scale = min(1., step/500)*.5*(1+math.cos(math.pi*max(0, step-500)/max(1, optimizer_horizon-500)))
                for parameter_group in optimizer.param_groups:
                    parameter_group['lr'] = 1e-4*scale
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
            gradient_scope='native_gru' if group=='I-HistoryDP' else 'vision.keypoints' if image else 'own_state_gru',
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
    p.add_argument('--final-test', action='store_true'); p.add_argument('--profile'); a = p.parse_args(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    train(a.root, a.output, a.kind, a.target, a.pilot, a.final_test,a.profile)
