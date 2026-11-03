"""Bounded recurrent State BC/DP diagnostics; no ImageDP or test-set access."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch

from .compact import sha256
from .models.vendor.ema_model import EMAModel
from .native_v2_resources import LeaseV2
from .state_history import PersistentState, STATE_CONTRACT, load_records
from .train_long import cpu_snapshot, save
from .training_resources import atomic_json


def labels(record, anchors, horizon=64):
    indices = anchors[:, None]+torch.arange(horizon, device=anchors.device)
    valid = indices < len(record['time'])
    commands = record['command'][indices.clamp_max(len(record['time'])-1)]
    return torch.where(valid[..., None], commands, 0.), valid


def require_foundation(root):
    gate = json.loads((root/'foundation.json').read_text())
    session = json.loads((root/'session.json').read_text())
    if not gate['passed']:
        raise RuntimeError('Current legal expert/control foundation failed')
    for name, digest in gate['evidence_sha256'].items():
        if sha256(root/name) != digest:
            raise ValueError('Foundation evidence changed: '+name)
    for path, digest in session['implementation_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Registered code changed: '+path)
    for path,digest in session.get('dependency_sha256',{}).items():
        if sha256(path)!=digest:raise ValueError('Registered dependency changed: '+path)
    if sha256(Path(session['source_root'])/'data.json') != session['data_sha256']:
        raise ValueError('Data identity changed')


@torch.no_grad()
def evaluate(model, records, lease, sampling=None, motion=False):
    total = torch.zeros(2); hold = torch.zeros(2); count = 0; rows = []; losses=[]
    for identity, record in records.items():
        generator = torch.Generator(device='cuda:0').manual_seed(7091)
        with lease.block():
            condition, anchors, _ = model.encode(record)
            prediction = model.predict(condition, generator, **(sampling or {}))
            if model.contract['action']=='absolute_joint_position_target_rad':
                from .position_control import bounded_targets
                prediction=bounded_targets(prediction.flatten(0,1),record['joint_limits']).reshape_as(prediction)
            else:
                prediction=prediction.clamp(-1,1)
            command, valid = labels(record, anchors,model.horizon)
            loss_generator=torch.Generator(device=condition.device).manual_seed(8091)
            losses.append(float(model.loss(condition,command,valid,loss_generator)))
            error = prediction[:, :12]-command[:, :12]
            errors = torch.stack((error[..., :7].square().mean(-1), error[..., 7:].square().mean(-1)), -1)
            sums = (errors*valid[:, :12, None]).sum((0, 1)).cpu()
            total += sums; count += int(valid[:, :12].sum())
            if model.contract['action']=='absolute_joint_position_target_rad':
                # robot58:87 is the already applied target at the observation.
                baseline=record['robot'][anchors,58:87,None].transpose(1,2)-command[:,:12]
                errors_hold=torch.stack((baseline[...,:7].square().mean(-1),baseline[...,7:].square().mean(-1)),-1)
                hold+=(errors_hold*valid[:,:12,None]).sum((0,1)).cpu()
            rows.append(dict(identity=identity, arm_rmse=float((sums[0]/valid[:, :12].sum().cpu()).sqrt()),
                hand_rmse=float((sums[1]/valid[:, :12].sum().cpu()).sqrt()),
                reset_arm_rmse=float(error[0, 0, :7].square().mean().sqrt()),
                reset_hand_rmse=float(error[0, 0, 7:].square().mean().sqrt())))
            if motion:
                from .state_motion import motion_metrics
                metric=motion_metrics(prediction.cpu().numpy(),command.cpu().numpy(),valid.cpu().numpy(),
                    record['robot'][anchors,58:87].cpu().numpy(),model.action_scale.cpu().numpy(),
                    (prediction*model.action_scale+model.action_offset).cpu().numpy(),record['joint_limits'].cpu().numpy(),
                    __import__('numpy').zeros(len(anchors),dtype=bool))
                rows[-1].update({k:v for k,v in metric.items() if k.startswith('moving')})
    result = dict(arm_rmse=float((total[0]/count).sqrt()), hand_rmse=float((total[1]/count).sqrt()),
        loss=sum(losses)/len(losses), hold_arm_rmse=float((hold[0]/count).sqrt()),
        hold_hand_rmse=float((hold[1]/count).sqrt()),
        selection_mse=float(total.mean()/count), episodes=rows, scope='Teacher-forced past execution, not autonomous success')
    if motion:
        import numpy as np
        for key in ('moving_arm12_rmse_rad','moving12_amplitude_ratio','moving12_direction_cosine'):
            values=[r[key] for r in rows if r.get(key) is not None]
            result[key]=float(np.mean(values)) if values else None
    return result


def train(root, output, kind):
    require_foundation(root)
    output.mkdir(parents=True, exist_ok=False)
    session = json.loads((root/'session.json').read_text())
    source = Path(session['source_root'])
    train_ids = [s['identity'] for s in session['saved_specs'] if s['cohort'] == 'train']
    validation_ids = [s['identity'] for s in session['saved_specs'] if s['cohort'] == 'validation']
    data = json.loads((source/'data.json').read_text())
    if not set(train_ids) <= set(data['splits']['train']) or not set(validation_ids) <= set(data['splits']['validation']):
        raise ValueError('Split violation')
    records = load_records(source, train_ids+validation_ids)
    stats = torch.load(source/'statistics.pt', map_location='cpu', weights_only=False)
    expert = torch.load(session['expert_checkpoint'], map_location='cpu', weights_only=False)[0]['model']
    lease = LeaseV2(root, deadline=session['deadline'])
    maximum = session[kind+'_updates']; milestones = sorted(set((250, maximum)))
    with lease.block():
        torch.manual_seed(20261102)
        model = PersistentState(kind); model.set_statistics(stats, expert); model.cuda()
        averaged = copy.deepcopy(model).eval().requires_grad_(False)
        ema = EMAModel(averaged, power=.75)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.95, .999), weight_decay=1e-6)
        records = {identity: {k: v.cuda() if torch.is_tensor(v) else v for k, v in record.items()}
            for identity, record in records.items()}
    del expert
    atomic_json(output/'configuration.json', dict(kind=kind, contract=STATE_CONTRACT, train_ids=train_ids,
        validation_ids=validation_ids, statistics_sha256=sha256(source/'statistics.pt'),
        maximum_updates=maximum, segment_ticks=96, episodes_per_batch=2, lr=1e-4,
        betas=[.95, .999], weight_decay=1e-6, warmup=500, schedule='cosine', loss='B',
        expert_normalizer_sha256=session['expert_sha256'], copied_expert_hidden=False,
        parameters=sum(p.numel() for p in model.parameters()), seed=20261102))
    schedule = [(identity, start) for identity in train_ids for start in range(0, len(records[identity]['time']), 96)]
    generator = torch.Generator().manual_seed(20261102)
    noise = torch.Generator(device='cuda:0').manual_seed(20261102)
    history = []; best = math.inf; began = time.time(); cursor = 0; order = []
    for update in range(1, maximum+1):
        selected = []
        for _ in range(2):
            if cursor == len(order):
                order = torch.randperm(len(schedule), generator=generator).tolist(); cursor = 0
            selected.append(schedule[order[cursor]]); cursor += 1
        with lease.block():
            model.train(); optimizer.zero_grad(set_to_none=True)
            conditions = []; commands = []; masks = []
            for identity, start in selected:
                record = records[identity]
                condition, anchors, _ = model.encode(record, start, min(start+96, len(record['time'])))
                command, valid = labels(record, anchors)
                conditions.append(condition); commands.append(command); masks.append(valid)
            loss = model.loss(torch.cat(conditions), torch.cat(commands), torch.cat(masks), noise)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            scale = min(1., update/500)*.5*(1+math.cos(math.pi*max(0, update-500)/max(1, maximum-500)))
            for group in optimizer.param_groups:
                group['lr'] = 1e-4*scale
            optimizer.step(); ema.step(model)
        with (output/'train.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(update=update, loss=float(loss.detach()), norm=float(norm),
                lr=optimizer.param_groups[0]['lr'], elapsed_s=time.time()-began))+'\n')
        if update in milestones:
            metrics = dict(update=update, train=evaluate(averaged, {i: records[i] for i in train_ids}, lease),
                validation=evaluate(averaged, {i: records[i] for i in validation_ids}, lease))
            history.append(metrics); atomic_json(output/'metrics.json', history)
            payload = cpu_snapshot(dict(contract=STATE_CONTRACT, kind=kind, group='S-NativeDP', updates=update,
                model=model.state_dict(), ema=averaged.state_dict(), optimizer=optimizer.state_dict(),
                schedule_order=order, schedule_cursor=cursor, schedule_rng=generator.get_state(),
                noise_rng=noise.get_state(), train_ids=train_ids, validation_ids=validation_ids))
            save(output/f'milestone_{update}.pt', payload)
            if metrics['validation']['selection_mse'] < best:
                best = metrics['validation']['selection_mse']; save(output/'best.pt', payload)
    atomic_json(output/'result.json', dict(software_complete=True, updates=maximum, history=history,
        best_sha256=sha256(output/'best.pt'), test_used=False, automatic_expansion=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--kind', choices=['bc', 'dp'], required=True)
    a = p.parse_args(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    train(a.root, a.output, a.kind)
