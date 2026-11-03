"""Measured H24 batch and checkpoint audit without optimizer updates."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from visuotactile.compact import sha256
from visuotactile.dp_objectives import masked_objectives
from visuotactile.position_control import contract
from visuotactile.state_history import PersistentState
from visuotactile.state_history_training import labels
from visuotactile.training_resources import atomic_json


def tensor_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def signature(root, prediction_type='sample'):
    plan=json.loads((root/'run_plan.json').read_text());previous=Path(plan['old_root'])
    config=json.loads((previous/'h24_train/configuration.json').read_text())
    session=json.loads((previous/'session.json').read_text())
    statistics=torch.load(previous/'position_train_statistics.pt',map_location='cpu',weights_only=False)
    expert=torch.load(session['expert_checkpoint'],map_location='cpu',weights_only=False)[0]['model']
    model_contract=contract(24) if prediction_type=='sample' else contract(24,True,prediction_type)
    torch.manual_seed(config['seed']);model=PersistentState('dp',model_contract);model.set_statistics(statistics,expert)
    initial=tensor_hash(model.state_dict())
    generator=torch.Generator().manual_seed(509)
    condition=torch.randn(2,512,generator=generator,requires_grad=True)
    commands=torch.randn(2,24,29,generator=generator)
    valid=torch.ones(2,24,dtype=torch.bool);valid[1,7:]=False
    loss=model.loss(condition,commands,valid,torch.Generator().manual_seed(709))
    loss.backward()
    gradients=tensor_hash({name:parameter.grad for name,parameter in model.named_parameters() if parameter.grad is not None})
    with torch.no_grad():
        predicted=model.predict(condition.detach(),torch.Generator().manual_seed(809),inference_steps=16,clip_sample=False)
    return dict(initial_model_sha256=initial,loss=float(loss.detach()),gradient_sha256=gradients,
        predicted_sha256=tensor_hash({'prediction':predicted}),contract=model.contract,
        noise_prediction_type=model.noise.config.prediction_type,sampler_prediction_type=model.sampler.config.prediction_type,
        parameters=sum(parameter.numel() for parameter in model.parameters()))


def batch_audit(root):
    plan = json.loads((root/'run_plan.json').read_text())
    previous = Path(plan['old_root'])
    config = json.loads((previous/'h24_train/configuration.json').read_text())
    session = json.loads((previous/'session.json').read_text())
    saved = torch.load(previous/'h24_train/milestone_3000.pt', map_location='cpu', weights_only=False)
    if sha256(previous/'h24_train/milestone_3000.pt') != plan['sample_checkpoint_sha256']:
        raise ValueError('Reference checkpoint changed')
    if saved['config_sha256'] != sha256(previous/'h24_train/configuration.json'):
        raise ValueError('Reference checkpoint configuration mismatch')
    records = torch.load(session['profiles']['h24']['numeric_cache'], map_location='cpu', weights_only=False)['records']
    statistics = torch.load(previous/'position_train_statistics.pt', map_location='cpu', weights_only=False)
    expert = torch.load(session['expert_checkpoint'], map_location='cpu', weights_only=False)[0]['model']
    schedule = [(identity, start) for identity in config['train_ids'] for start in range(0, len(records[identity]['time']), 96)]
    generator = torch.Generator().manual_seed(config['seed'])
    order = []; cursor = 0; batches = []; selected_first = None
    for update in range(3000):
        selected = []
        for _ in range(2):
            if cursor == len(order):
                order = torch.randperm(len(schedule), generator=generator).tolist(); cursor = 0
            selected.append(schedule[order[cursor]]); cursor += 1
        if selected_first is None:
            selected_first = selected
        anchors = []
        for identity, start in selected:
            stop = min(start+96, len(records[identity]['time']))
            anchors.append(len(range(start, stop, 12)))
        batches.append(dict(update=update+1, anchors=sum(anchors), independent_ids=len({item[0] for item in selected})))
    if order != saved['schedule_order'] or cursor != saved['schedule_cursor'] or not torch.equal(generator.get_state(), saved['schedule_rng']):
        raise ValueError('Actual sample scheduling differs from reconstruction')
    torch.manual_seed(config['seed']); model = PersistentState('dp', contract(24)); model.set_statistics(statistics, expert)
    initial_hash = tensor_hash(model.state_dict())
    conditions = []; commands = []; masks = []; segment_rows = []
    for identity, start in selected_first:
        record = records[identity]; stop = min(start+96, len(record['time']))
        condition, anchor, _ = model.encode(record, start, stop)
        command, valid = labels(record, anchor, 24)
        conditions.append(condition); commands.append(command); masks.append(valid)
        segment_rows.append(dict(identity=identity, start=start, stop=stop, actor_shape=list(record['actor'][start:stop].shape),
            robot_shape=list(record['robot'][start:stop].shape), geometry_shape=list(record['geometry'][start:stop].shape),
            condition_shape=list(condition.shape), command_shape=list(command.shape), mask_shape=list(valid.shape),
            valid_positions=int(valid.sum()), anchors=anchor.tolist()))
    condition = torch.cat(conditions); condition.retain_grad()
    command = torch.cat(commands); valid = torch.cat(masks)
    captured = {}
    original_add = model.noise.add_noise
    def capture(clean, noise, timestep):
        captured.update(clean=clean.detach(), noise=noise.detach(), timestep=timestep.detach())
        return original_add(clean, noise, timestep)
    model.noise.add_noise = capture
    def prediction_hook(module, inputs, output):
        output.retain_grad(); captured['prediction'] = output
    handle = model.denoiser.register_forward_hook(prediction_hook)
    loss = model.loss(condition, command, valid, torch.Generator().manual_seed(config['seed']))
    manual = masked_objectives(captured['prediction'], captured['clean'], valid)
    if not torch.equal(loss, .5*(manual['arm']+manual['hand'])):
        raise ValueError('Loss B aggregation mismatch')
    loss.backward(); handle.remove()
    split = len(conditions[0])
    gradient_norms = [float(condition.grad[:split].norm()), float(condition.grad[split:].norm())]
    if not all(value > 0 for value in gradient_norms):
        raise ValueError('A sampled segment does not contribute gradients')
    padded_gradient = captured['prediction'].grad[~valid]
    if torch.count_nonzero(padded_gradient):
        raise ValueError('Padding contributes objective gradient')
    result = dict(passed=True, optimizer_updates=0, cuda_initialized=torch.cuda.is_initialized(),
        sample_checkpoint_sha256=plan['sample_checkpoint_sha256'], checkpoint_updates=saved['updates'],
        ema_optimization_step=saved['ema_optimization_step'], optimizer_state_count=len(saved['optimizer']['state']),
        optimizer_lr=saved['optimizer']['param_groups'][0]['lr'], scheduler_horizon=19000,
        schedule_segments=len(schedule), schedule_order_and_cursor_and_rng_match=True, cursor=cursor,
        train_episodes=len(config['train_ids']), validation_episodes=len(config['validation_ids']),
        anchors_per_update=dict(mean=float(np.mean([row['anchors'] for row in batches])),
            counts=dict(Counter(row['anchors'] for row in batches))),
        independent_episode_counts=dict(Counter(row['independent_ids'] for row in batches)),
        adjacent_full_windows_overlap=dict(action_ticks=12, fraction=.5, stride=12, horizon=24),
        first_update_segments=segment_rows, batch_condition_shape=list(condition.shape),
        batch_action_shape=list(command.shape), mask_shape=list(valid.shape), valid_positions=int(valid.sum()),
        diffusion_timesteps=captured['timestep'].tolist(),
        independent_noise_rows=all(not torch.equal(captured['noise'][0], row) for row in captured['noise'][1:]),
        noise_shape=list(captured['noise'].shape), segment_condition_gradient_norms=gradient_norms,
        padded_output_gradient_nonzero=int(torch.count_nonzero(padded_gradient)),
        loss=float(loss.detach()), arm=float(manual['arm'].detach()), hand=float(manual['hand'].detach()),
        aggregation='0.5*(mean(valid squared error over Arm7)+mean(valid squared error over Hand22)); one backward/update; no division by segments/anchors beyond valid-position denominator',
        initial_model_sha256=initial_hash, parameters=sum(parameter.numel() for parameter in model.parameters()),
        padding_scope='Masked output positions do not contribute objective; padded noisy input can influence valid outputs through convolutions, unchanged from Sample')
    tail_identity,tail_start=next((identity,start) for identity,start in schedule
        if len(records[identity]['time'])-start<96 and len(records[identity]['time'])-start>0)
    tail=records[tail_identity];model.zero_grad(set_to_none=True)
    tail_condition,tail_anchors,_=model.encode(tail,tail_start,len(tail['time']))
    tail_command,tail_valid=labels(tail,tail_anchors,24)
    handle=model.denoiser.register_forward_hook(prediction_hook)
    tail_loss=model.loss(tail_condition,tail_command,tail_valid,torch.Generator().manual_seed(config['seed']))
    tail_loss.backward();handle.remove()
    tail_gradient=captured['prediction'].grad
    if not (~tail_valid).any() or torch.count_nonzero(tail_gradient[~tail_valid]) or not tail_gradient[tail_valid].norm()>0:
        raise ValueError('Actual episode-tail padding gradient gate failed')
    result['real_tail_probe']=dict(identity=tail_identity,start=tail_start,anchors=tail_anchors.tolist(),
        action_shape=list(tail_command.shape),valid_mask=tail_valid.tolist(),padded_positions=int((~tail_valid).sum()),
        padded_output_gradient_nonzero=int(torch.count_nonzero(tail_gradient[~tail_valid])),
        valid_output_gradient_norm=float(tail_gradient[tail_valid].norm()))
    logged=[json.loads(line) for line in (previous/'h24_train/train.jsonl').read_text().splitlines()]
    lr_errors=[abs(row['lr']-1e-4*min(1.,row['update']/500)*.5*(1+math.cos(math.pi*max(0,row['update']-500)/18500))) for row in logged]
    if len(logged)!=3000 or max(lr_errors)!=0:raise ValueError('Sample learning-rate trajectory differs')
    result['sample_lr_logged_updates']=len(logged);result['sample_lr_formula_max_error']=max(lr_errors)
    atomic_json(root/'batch_audit.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--signature-output',type=Path)
    parser.add_argument('--prediction-type',choices=('sample','epsilon'),default='sample')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.signature_output:atomic_json(args.signature_output,signature(args.root,args.prediction_type))
    else:batch_audit(args.root)
