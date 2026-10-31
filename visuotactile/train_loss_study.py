"""Independent paired continuations from a frozen epoch40 EMA checkpoint."""
import argparse
import copy
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import torch

from .compact import sha256
from .long_data import inputs_for, load_record, supervision
from .models.long_horizon import LongHorizonConfig, LongHorizonPolicy
from .study_metrics import denoising_objectives, evaluate, joint_groups
from .study_resources import StudyLease
from .train_long import cpu_snapshot, learning_rate, move, save, update_ema
from .training_resources import atomic_json


def train(args):
    root = args.root
    session = json.loads((root/'session.json').read_text())
    index = json.loads((root/'cache_index.json').read_text())
    splits = index['splits']
    group,objective = args.line.split('-')
    source = Path(session['source_root'])/(group+'-new_full')/'best.pt'
    identities = splits['train'][:4] if args.pilot else splits['train']
    validation_ids = splits['validation'][:4] if args.pilot else splits['validation']
    epochs = 1 if args.pilot else session['dp_epochs']
    output = root/(args.line+('_pilot' if args.pilot else ''))
    output.mkdir(exist_ok=args.resume)
    device = 'cuda:0'
    seed = 20261101 if group == 'G00' else 20261102
    torch.manual_seed(seed)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    lease = StudyLease(root,deadline=session['dp_deadline'])
    saved = torch.load(source,map_location='cpu',weights_only=False)
    if saved['config']['prediction_type'] != 'sample' or saved['epoch'] != 40 or saved['updates'] != 33280:
        raise ValueError('Expected audited sample epoch40 EMA initialization')
    cfg = LongHorizonConfig(**saved['config'])
    metadata,vision = saved['metadata'],saved['vision']
    with lease.block():
        model = LongHorizonPolicy(cfg).to(device)
        model.load_state_dict(saved['model'])
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
    del saved
    gc.collect()
    updates_per_epoch = sum(math.ceil(index['entries'][identity]['native_rows']/96) for identity in identities)
    total = epochs*updates_per_epoch
    configuration = dict(line=args.line,loss=objective,model=asdict(cfg),seed=seed,
        source=str(source),source_sha256=sha256(source),source_epoch=40,source_updates=33280,
        epochs=epochs,updates_per_epoch=updates_per_epoch,total_updates=total,
        optimizer=dict(name='AdamW',lr=1e-4,betas=[.95,.999],weight_decay=1e-6,fresh=True),
        schedule='new cosine',warmup_updates=500,ema='fresh independent copy of epoch40 EMA; inv_gamma1 power.75 max.9999',
        loss_A='masked element mean29',loss_B='0.5 arm mean7 +0.5 hand mean22',hand_weighting='joint_equal',
        joint_groups=joint_groups(metadata['joint_order']),train_episodes=identities,validation_episodes=validation_ids,
        train_diagnostic_episodes=splits['train'][:16],test_episodes=splits['test'],
        cache_index_sha256=sha256(root/'cache_index.json'),native_chunk=96,precision='BF16/FP32 loss',
        selection='EMA validation prefix12 native command .5arm MSE+.5hand MSE; RNG7091',
        source_hashes={str(path):sha256(path) for path in (Path(__file__),Path('visuotactile/study_metrics.py'),
            Path('visuotactile/models/long_horizon.py'),Path('visuotactile/long_data.py'),
            Path('visuotactile/study_resources.py'),Path('visuotactile/long_resources.py'))})
    generator = torch.Generator(device=device).manual_seed(seed)
    updates = epoch_start = episode_start = 0
    history = []
    best = float('inf')
    accumulator = dict(loss_sum=0.,arm_sum=0.,hand_sum=0.,targets=0,gradient_sum=0.,updates=0)
    if args.resume:
        resumed = torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        if resumed['configuration'] != configuration:
            raise ValueError('Continuation configuration/source changed')
        with lease.block():
            model.load_state_dict(resumed['model'])
            ema.load_state_dict(resumed['ema'])
            optimizer.load_state_dict(resumed['optimizer'])
        updates,epoch_start,episode_start = resumed['updates'],resumed['next_epoch'],resumed['next_episode']
        history,best,accumulator = resumed['history'],resumed['best'],resumed['accumulator']
        generator.set_state(resumed['generator'].cpu())
        torch.set_rng_state(resumed['cpu_rng'].cpu())
        torch.cuda.set_rng_state(resumed['cuda_rng'].cpu())
        del resumed
        gc.collect()
    atomic_json(output/'config.json',configuration)
    stopping = [False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))

    def checkpoint(next_epoch,next_episode):
        with lease.block():
            payload = cpu_snapshot(dict(configuration=configuration,model=model.state_dict(),ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),generator=generator.get_state(),cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),updates=updates,next_epoch=next_epoch,next_episode=next_episode,
                history=history,best=best,accumulator=accumulator))
        save(output/'last.pt',payload)
        del payload
        gc.collect()

    def best_checkpoint(epoch,metrics):
        with lease.block():
            payload = cpu_snapshot(dict(version=2,config=asdict(cfg),model=ema.state_dict(),
                updates=33280+updates,epoch=40+epoch,metadata=metadata,vision=vision,
                study_line=args.line,additional_updates=updates,additional_epochs=epoch,
                configuration=configuration,validation=metrics))
        save(output/'best.pt',payload)
        del payload
        gc.collect()

    def measurements(epoch):
        validation = evaluate(ema,root,validation_ids,lease,trajectories=True)
        training = evaluate(ema,root,identities if args.pilot else splits['train'][:16],lease,seed=7091)
        return dict(epoch=epoch,additional_updates=updates,total_updates=33280+updates,
                    validation=validation,train_diagnostic=training,ema=True,
                    source_checkpoint=configuration['source_sha256'])

    began = time.monotonic()
    try:
        if not args.resume:
            baseline = measurements(0)
            best = baseline['validation']['selection_mse']
            history.append(baseline)
            best_checkpoint(0,baseline['validation'])
            atomic_json(output/'metrics.json',history)
            checkpoint(0,0)
        for epoch in range(epoch_start,epochs):
            order = list(identities)
            random.Random(seed+epoch).shuffle(order)
            model.train()
            for ordinal,identity in enumerate(order):
                if epoch == epoch_start and ordinal < episode_start:
                    continue
                record = load_record(root,identity)
                state = None
                for start in range(0,len(record['time']),96):
                    stop = min(start+96,len(record['time']))
                    anchors = torch.arange(start,stop,12)
                    inputs_cpu = inputs_for(record,start,stop,cfg.tactile)
                    labels_cpu = supervision(record,anchors)
                    count = int(labels_cpu['valid'].sum())
                    with lease.block():
                        inputs,labels = move(inputs_cpu,device),move(labels_cpu,device)
                        rate = 1e-4*learning_rate(updates,total,500)
                        for parameter_group in optimizer.param_groups:
                            parameter_group['lr'] = rate
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            condition,state = model.encode(inputs,state)
                            losses = denoising_objectives(model,condition,labels['reference'],labels['valid'],generator)
                        loss = losses[objective]
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite objective')
                        loss.backward()
                        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        if not torch.isfinite(gradient):
                            raise FloatingPointError('Nonfinite gradient')
                        optimizer.step()
                        update_ema(ema,model,updates)
                        state = state.detach()
                        updates += 1
                        values = {key:float(value.detach()) for key,value in losses.items()}
                        gradient_value = float(gradient)
                    accumulator['loss_sum'] += values[objective]*count
                    accumulator['arm_sum'] += values['arm']*count
                    accumulator['hand_sum'] += values['hand']*count
                    accumulator['targets'] += count
                    accumulator['gradient_sum'] += gradient_value
                    accumulator['updates'] += 1
                    if updates%25 == 0:
                        atomic_json(output/'status.json',dict(state='training',pid=os.getpid(),
                            epoch=epoch+1,updates=updates,total_updates=total,lr=rate,loss=values[objective],
                            gradient_norm=gradient_value,elapsed_s=time.monotonic()-began,memory=lease.memory()))
                if (ordinal+1)%64 == 0 or stopping[0]:
                    checkpoint(epoch,ordinal+1)
                if stopping[0]:
                    atomic_json(output/'status.json',dict(state='stopped',updates=updates,recover='last.pt'))
                    return
            metrics = measurements(epoch+1)
            metrics['train'] = dict(loss=accumulator['loss_sum']/max(1,accumulator['targets']),
                arm=accumulator['arm_sum']/max(1,accumulator['targets']),hand=accumulator['hand_sum']/max(1,accumulator['targets']),
                gradient_norm=accumulator['gradient_sum']/max(1,accumulator['updates']),lr=rate,
                valid_future_positions=accumulator['targets'],updates=accumulator['updates'],episodes=len(identities),
                loss_definition=objective,ema=False)
            if metrics['validation']['selection_mse'] < best:
                best = metrics['validation']['selection_mse']
                best_checkpoint(epoch+1,metrics['validation'])
            history.append(metrics)
            atomic_json(output/'metrics.json',history)
            accumulator = {key:0. if key.endswith('sum') else 0 for key in accumulator}
            checkpoint(epoch+1,0)
            print(args.line,'epoch',epoch+1,'updates',updates,'prefix_validation',metrics['validation']['selection_mse'],flush=True)
        final = dict(state='complete',epochs=epochs,updates=updates,best=best,memory=lease.memory(),
                     elapsed_s=time.monotonic()-began,work_s=lease.work_s,wait_s=lease.wait_s,pacing_s=lease.pacing_s)
        atomic_json(output/'report.json',final)
        atomic_json(output/'status.json',final)
    except BaseException as error:
        atomic_json(output/'interruption.json',dict(error=repr(error),executed_updates=updates,
            recovery='last.pt complete episode boundary; later updates are not committed'))
        raise


def test(args):
    root = args.root
    output = root/args.line
    if (output/'test.json').exists():
        raise ValueError('Held-out test already evaluated')
    session = json.loads((root/'session.json').read_text())
    lease = StudyLease(root,deadline=session['deadline'])
    saved = torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    with lease.block():
        model = LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to('cuda:0').eval()
        model.load_state_dict(saved['model'])
    splits = json.loads((root/'splits.json').read_text())
    result = evaluate(model,root,splits['test'],lease,seed=7093)
    atomic_json(output/'test.json',dict(checkpoint_sha256=sha256(output/'best.pt'),epoch=saved['epoch'],
        updates=saved['updates'],additional_updates=saved['additional_updates'],metrics=result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--line',choices=['G00-A','G00-B','G10-A','G10-B'],required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--test',action='store_true')
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    test(arguments) if arguments.test else train(arguments)
