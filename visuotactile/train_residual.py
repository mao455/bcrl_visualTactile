"""Full-episode supervised residual training with matched force-masked control."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .compact import sha256
from .history import detach_state
from .models.residual import ResidualCorrection, compose_residual, residual_loss
from .residual_data import BASE_HASH, load_base
from .residual_session import SessionGate
from .training_resources import atomic_json


def read_record(root, episode, index):
    path = root / 'cache' / (episode+'.pt')
    if sha256(path) != index['entries'][episode]['sha256']:
        raise ValueError('Derived data hash mismatch')
    record = torch.load(path, map_location='cpu', weights_only=False)
    if record['base_sha256'] != BASE_HASH or record['cache_version'] != 1:
        raise ValueError('Wrong derived-data/base version')
    return record


def amplitude_statistics(root, ids, index):
    errors, gates = [], []
    for episode in ids:
        record = read_record(root, episode, index)
        errors.append(record['target']-record['base'])
        gates.append(record['gate'])
    error, contact = torch.cat(errors), torch.cat(gates)
    if not contact.any():
        raise ValueError('No observed contacts in train-only amplitude sample')
    report, scales = {}, []
    for name, selected in (('arm', slice(0, 7)), ('hand', slice(7, 29))):
        values = error[contact, selected].abs().flatten()
        chosen = max(.1, min(2., math.ceil(float(torch.quantile(values, .99))*10)/10))
        scales.extend([chosen]*(7 if name == 'arm' else 22))
        report[name] = dict(p50=float(values.median()), p95=float(torch.quantile(values, .95)),
            p99=float(torch.quantile(values, .99)), maximum=float(values.max()), scale=chosen,
            candidates={str(scale): dict(coverage=float((values<=scale).float().mean()),
                contact_rmse_infimum=float((values-scale).clamp_min(0).square().mean().sqrt()),
                all_rmse_infimum=float(torch.where(contact[:, None],
                    (error[:, selected].abs()-scale).clamp_min(0).square(),
                    error[:, selected].square()).mean().sqrt())) for scale in (.1, .5, 1., 1.5, 2.)})
    return scales, dict(episodes=ids, rows=len(error), contact_rows=int(contact.sum()), groups=report,
                        selection='ceil(train16 contact |expert-base| p99 * 10)/10, bounded .1..2; then frozen',
                        labels_clipped=False, units='native clipped pre-delay commands, not radians or Cartesian meters')


def chunk_record(record, start, stop):
    return dict(features=record['features'][None, start:stop],
        tactile={name: value[None, 2*start:2*stop] for name, value in record['tactile'].items()},
        tactile_valid=record['tactile_valid'][None, 2*start:2*stop],
        plans=record['plans'][None, start:stop], plan_valid=record['plan_valid'][None, start:stop],
        motion={name: value[None, start:stop] for name, value in record['motion'].items()})


def metrics_empty():
    return dict(rows=0, contact_rows=0, arm_square=0., hand_square=0., contact_arm_square=0.,
                contact_hand_square=0., base_arm_square=0., base_hand_square=0.,
                base_contact_arm_square=0., base_contact_hand_square=0., saturation=0, bypass_exact=True,
                episodes=[], updates=0, loss_sum=0.)


def accumulate(metrics, action, base, target, contact):
    action, base, target = action.detach(), base.detach(), target.detach()
    metrics['rows'] += len(action)
    metrics['contact_rows'] += int(contact.sum())
    for name, selected in (('arm', slice(0, 7)), ('hand', slice(7, 29))):
        error = (action[:, selected]-target[:, selected]).square().mean(-1)
        base_error = (base[:, selected]-target[:, selected]).square().mean(-1)
        metrics[name+'_square'] += float(error.sum())
        metrics['contact_'+name+'_square'] += float(error[contact].sum())
        metrics['base_'+name+'_square'] += float(base_error.sum())
        metrics['base_contact_'+name+'_square'] += float(base_error[contact].sum())
    metrics['saturation'] += int((action.abs() >= .99999).sum())
    metrics['bypass_exact'] &= torch.equal(action[~contact], base[~contact])


def finish_metrics(metrics):
    result = dict(metrics)
    for name in ('arm', 'hand'):
        for prefix, count in (('', metrics['rows']), ('contact_', metrics['contact_rows']),
                              ('base_', metrics['rows']), ('base_contact_', metrics['contact_rows'])):
            result[prefix+name+'_rmse'] = math.sqrt(metrics[prefix+name+'_square']/count) if count else None
    result['selection_mse'] = .5*(metrics['arm_square']+metrics['hand_square'])/max(1, metrics['rows'])
    result['saturation_fraction'] = metrics['saturation']/max(1, 29*metrics['rows'])
    return result


def run_epoch(models, optimizers, root, ids, index, gate, on_episode=None):
    training = optimizers is not None
    metrics = {name: metrics_empty() for name in models}
    for model in models.values():
        model.train(training)
        model.visual.vision.eval()
    for episode in ids:
        gate.before(reserve_s=30)
        record = read_record(root, episode, index)
        states = {name: None for name in models}
        for start in range(0, record['steps'], 32):
            gate.before()
            stop = min(start+32, record['steps'])
            inputs = chunk_record(record, start, stop)
            base, target, contact = (record[name][start:stop] for name in ('base', 'target', 'gate'))
            for name, model in models.items():
                with torch.set_grad_enabled(training):
                    raw, states[name] = model.forward_cached(**inputs, state=states[name])
                    action, delta = compose_residual(base, raw[0], contact, model.scales)
                    loss = residual_loss(action, target, delta, contact)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite residual loss')
                if training and contact.any():
                    optimizers[name].zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters()
                                                          if parameter.requires_grad], 1.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite residual gradient')
                    optimizers[name].step()
                    metrics[name]['updates'] += 1
                    metrics[name]['loss_sum'] += float(loss.detach())
                states[name] = detach_state(states[name])
                accumulate(metrics[name], action, base, target, contact)
        for result in metrics.values():
            result['episodes'].append(episode)
        if on_episode is not None:
            on_episode(episode, metrics)
    return {name: finish_metrics(result) for name, result in metrics.items()}


def save_checkpoint(path, model, optimizer, config, index, updates, coverage, epoch, validation):
    payload = dict(residual_checkpoint_version=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
        base_sha256=BASE_HASH, base_config=config['base_config'], config=config,
        normalizer=dict(proprio_mean=model.visual.proprio_mean, proprio_std=model.visual.proprio_std,
                        forces='signed log1p N', targets_tracking='radians, unstandardized'),
        action_version='gated_additive_native_command_v1', input_version='residual_frozen_frame_memory2_v1',
        force_masked=model.force_masked, scales=model.scales.tolist(), optimizer_updates=updates,
        full_episode_coverage=coverage, epoch=epoch, validation=validation,
        cache_index_sha256=config['cache_index_sha256'], derived_sources=index['entries'],
        train_episodes=index['splits']['train'], validation_episodes=index['splits']['validation'],
        test_episodes=index['splits']['test'], trained=updates>0)
    temporary = path.with_suffix('.partial.pt')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('pilot', 'full', 'test'), default='pilot')
    args = parser.parse_args()
    torch.set_num_threads(2)
    gate = SessionGate(args.root)
    index = json.loads((args.root / 'cache/index.json').read_text())
    base = load_base()
    frozen_reference = {name:value.clone() for name,value in base.model.state_dict().items()}
    output = args.root / ('training_'+args.stage)
    output.mkdir(exist_ok=False)
    scales_path = args.root / 'amplitude.json'
    if scales_path.exists():
        amplitude = json.loads(scales_path.read_text())
        scales = amplitude['scales']
    else:
        scales, stats = amplitude_statistics(args.root, index['splits']['train'][:16], index)
        amplitude = dict(scales=scales, statistics=stats)
        atomic_json(scales_path, amplitude)
    models, optimizers = {}, {}
    for name in ('b', 'c'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261030)
            models[name] = ResidualCorrection(base.model.encoder, base.model.cfg, scales, force_masked=name=='c')
        optimizers[name] = torch.optim.AdamW([parameter for parameter in models[name].parameters()
                                            if parameter.requires_grad], lr=1e-3, weight_decay=1e-4)
    if not all(torch.equal(value, models['c'].state_dict()[name]) for name, value in models['b'].state_dict().items()):
        raise RuntimeError('Matched independent initialization failed')
    config = dict(stage=args.stage, seed=20261030, started_epoch=time.time(), pid=os.getpid(),
        base_config=asdict(base.model.cfg),
        cache_index_sha256=sha256(args.root / 'cache/index.json'),
        architecture='independent copied frozen B0 frame encoder; train visual/tactile/intent/motion GRUs and raw residual MLP',
        base_frozen=True, visual_frame_frozen=True, batch_episodes=1, tbptt=32, padding='none; exact variable tails',
        loss='0.5 arm mean squared command error + 0.5 hand; only real contact; delta L2 1e-4',
        optimizer='AdamW', lr=1e-3, weight_decay=1e-4, clip_grad_norm=1., ema=False, lr_schedule=None,
        force_control='same real gate/mask; zero both forces before derivatives/contact features; FK/time retained',
        order='fixed original checkpoint split order, matched B/C',
        training_history='expert t-1 clipped command, expert actuator targets/tracking; teacher forcing',
        deployment_history='actual acknowledged student or declared expert-rollin command',
        precision='CPU float32', scales=scales,
        parameter_counts={name:dict(total=sum(parameter.numel() for parameter in model.parameters()),
            trainable=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
            for name, model in models.items()},
        source_sha256={str(path): sha256(path) for path in (Path(__file__), Path('visuotactile/models/residual.py'),
                       Path('visuotactile/residual_data.py'), Path('visuotactile/residual_session.py'))})
    atomic_json(output / 'config.json', config)
    train_ids = index['splits']['train'][:16] if args.stage=='pilot' else index['splits']['train']
    val_ids = index['splits']['validation'][:8] if args.stage=='pilot' else index['splits']['validation']
    updates, coverage, history = {name: 0 for name in models}, {name: [] for name in models}, []
    epochs = 2 if args.stage=='pilot' else 5
    if args.stage == 'full':
        pilot = json.loads((args.root / 'training_pilot/report.json').read_text())
        causal = json.loads((args.root / 'causal_validation.json').read_text())
        if not pilot['learning_signal'] or not causal['passed']:
            raise ValueError('Pilot learning signal or causal replay admission failed')
    if args.stage == 'test':
        for name in models:
            saved = torch.load(args.root / 'training_full' / (name+'_best.pt'), weights_only=False)
            models[name].load_state_dict(saved['model'])
        result = run_epoch(models, None, args.root, index['splits']['test'], index, gate)
        atomic_json(output / 'report.json', dict(test=result, selected_by='validation B; C matched epoch/updates'))
        return
    baseline = run_epoch(models, None, args.root, val_ids, index, gate)
    atomic_json(output / 'zero_residual.json', baseline)
    best, selected_epoch = float('inf'), None
    try:
        for epoch in range(1, epochs+1):
            prior_updates = dict(updates)
            def boundary(episode, totals):
                for name, model in models.items():
                    save_checkpoint(output / (name+'_last.pt'), model, optimizers[name], config, index,
                        prior_updates[name]+totals[name]['updates'], coverage[name]+totals[name]['episodes'],
                        epoch, None)
                atomic_json(output / 'status.json', dict(state='training', epoch=epoch,
                            complete_episodes=len(totals['b']['episodes']), episode=episode))
            train = run_epoch(models, optimizers, args.root, train_ids, index, gate, boundary)
            validation = run_epoch(models, None, args.root, val_ids, index, gate)
            for name in models:
                updates[name] += train[name]['updates']
                coverage[name].extend(train[name]['episodes'])
            if updates['b'] != updates['c']:
                raise RuntimeError('B/C effective updates diverged')
            row = dict(epoch=epoch, train=train, validation=validation, updates=dict(updates))
            history.append(row)
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if validation['b']['selection_mse'] < best:
                best, selected_epoch = validation['b']['selection_mse'], epoch
                for name, model in models.items():
                    save_checkpoint(output / (name+'_best.pt'), model, optimizers[name], config, index,
                                    updates[name], coverage[name], epoch, validation[name])
            print(json.dumps(dict(epoch=epoch, updates=updates, validation={name:result['selection_mse']
                                                                          for name,result in validation.items()})), flush=True)
    except InterruptedError as error:
        atomic_json(output / 'budget_stop.json', dict(reason=str(error), completed_epochs=len(history)))
        for name in models:
            last_path = output / (name+'_last.pt')
            if last_path.exists():
                last = torch.load(last_path, weights_only=False)
                updates[name] = last['optimizer_updates']
                coverage[name] = last['full_episode_coverage']
    except BaseException as error:
        (output / 'error.txt').write_text(traceback.format_exc())
        atomic_json(output / 'status.json', dict(state='failed', reason=repr(error), completed_epochs=len(history)))
        raise
    if any(parameter.grad is not None or parameter.requires_grad for parameter in base.model.parameters()):
        raise RuntimeError('Base was not frozen')
    if any(not torch.equal(value, base.model.state_dict()[name]) for name,value in frozen_reference.items()):
        raise RuntimeError('Frozen base tensors changed')
    report = dict(completed_epochs=len(history), requested_epochs=epochs, selected_epoch=selected_epoch,
        effective_updates=updates, episode_visits={name:len(value) for name,value in coverage.items()},
        unique_training_episodes={name:len(set(value)) for name,value in coverage.items()},
        baseline=baseline, history=history, wait_s=gate.wait_s, base_sha256=BASE_HASH,
        learning_signal=bool(history and best < baseline['b']['selection_mse']), finished_epoch=time.time(),
        source_and_config=config, base_gradients_absent=True, frozen_base_unchanged=True)
    atomic_json(output / 'report.json', report)
    atomic_json(output / 'status.json', dict(state='complete' if len(history)==epochs else 'budget_stopped',
                                           completed_epochs=len(history)))


if __name__ == '__main__':
    main()
