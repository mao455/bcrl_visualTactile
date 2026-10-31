"""Matched real-force and force-masked full-episode residual training."""
import argparse
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
from .models.transformer_residual import HISTORY_PROTOCOL, TransformerResidual, TransformerResidualConfig, combine, residual_loss
from .study_metrics import joint_groups
from .study_resources import StudyLease
from .train_long import cpu_snapshot, learning_rate, move, save
from .training_resources import atomic_json
from .transformer_data import load_residual_record, residual_inputs, residual_labels


class ResidualMetrics:
    def __init__(self, groups):
        self.groups = groups
        self.sums = {name:torch.zeros(29,dtype=torch.float64) for name in ('residual','contact_residual','combined','base','contact_combined',
            'contact_base','noncontact_residual')}
        self.count = self.contact = self.noncontact = self.clipped = self.wrong = self.amplitude = self.saturated = 0

    def add(self, prediction, labels):
        prediction = prediction.detach().float().cpu()
        labels = {name:value.detach().cpu() for name,value in labels.items()}
        valid,gate = labels['valid'],labels['gate']&labels['valid']
        inactive = valid&~gate
        action = combine(labels['base'],prediction,labels['gate'])
        errors = dict(residual=(prediction-labels['delta']).square(),combined=(action-labels['expert']).square(),
                      base=(labels['base']-labels['expert']).square())
        for name,error in errors.items():
            self.sums[name] += torch.where(valid[...,None],error,0.).sum((0,1)).double()
        self.sums['contact_combined'] += torch.where(gate[...,None],errors['combined'],0.).sum((0,1)).double()
        self.sums['contact_residual'] += torch.where(gate[...,None],errors['residual'],0.).sum((0,1)).double()
        self.sums['contact_base'] += torch.where(gate[...,None],errors['base'],0.).sum((0,1)).double()
        self.sums['noncontact_residual'] += torch.where(inactive[...,None],errors['residual'],0.).sum((0,1)).double()
        unclipped = labels['base']+prediction
        clip_mask = (unclipped.abs()>1)&gate[...,None]
        wrong = clip_mask & (prediction*labels['delta']<0)
        self.count += int(valid.sum())
        self.contact += int(gate.sum())
        self.noncontact += int(inactive.sum())
        self.clipped += int(clip_mask.sum())
        self.wrong += int(wrong.sum())
        self.saturated += int(((action.abs()>=.99999)&valid[...,None]).sum())
        self.amplitude += float(torch.where(valid[...,None],prediction.abs(),0.).sum())
        if not torch.equal(action[inactive],labels['base'][inactive]):
            raise RuntimeError('No-contact bypass is not exact')

    def result(self):
        result = dict(valid_steps=self.count,contact_steps=self.contact,noncontact_steps=self.noncontact,
            gate_fraction=self.contact/max(1,self.count),delta_mean_abs=self.amplitude/max(1,29*self.count),
            execution_clip_fraction=self.clipped/max(1,29*self.contact),
            wrong_direction_saturation_fraction=self.wrong/max(1,29*self.contact),
            command_saturation_fraction=self.saturated/max(1,29*self.count),metrics={})
        for name,squared in self.sums.items():
            count = self.contact if name.startswith('contact_') else self.noncontact if name.startswith('noncontact_') else self.count
            result['metrics'][name] = {group:math.sqrt(float(squared[indices].mean())/max(1,count))
                                      for group,indices in self.groups.items()}
        result['selection_mse'] = .5*sum(result['metrics']['contact_combined'][group]**2 for group in ('arm','hand'))
        return result


@torch.no_grad()
def evaluate(model, root, identities, lease, groups):
    model.eval()
    metrics = ResidualMetrics(groups)
    loss_sum = count = tactile_valid = tactile_total = 0
    forward_s = anchors_total = 0
    for identity in identities:
        record = load_residual_record(root,identity)
        state = None
        tactile_valid += int(record['tactile'][...,-1].sum())
        tactile_total += record['tactile'][...,-1].numel()
        for start in range(0,len(record['time']),96):
            stop = min(start+96,len(record['time']))
            anchors = torch.arange(start,stop,6)
            inputs_cpu,labels_cpu = residual_inputs(record,start,stop),residual_labels(record,anchors)
            with lease.block():
                inputs,labels = move(inputs_cpu,'cuda:0'),move(labels_cpu,'cuda:0')
                torch.cuda.synchronize()
                began = time.perf_counter()
                prediction,state = model(inputs,state)
                torch.cuda.synchronize()
                forward_s += time.perf_counter()-began
                anchors_total += len(anchors)
                mask = labels['valid']&labels['gate']
                objective = residual_loss(prediction,labels['delta'],mask,model.cfg.amplitude_weight)
                weight = int(mask.sum())
                loss_sum += float(objective['loss'])*weight
                count += weight
                prediction_cpu = prediction.cpu()
            metrics.add(prediction_cpu,labels_cpu)
    result = metrics.result()
    result.update(loss=loss_sum/max(1,count),episodes=len(identities),ema=False,
        tactile_valid_fraction=tactile_valid/max(1,tactile_total),
        batched_forward_ms_per_anchor=1000*forward_s/max(1,anchors_total),
        history_source='expert actual-command teacher forcing for nominal and residual; deployment uses actual mixed feedback')
    return result


def build_model(root, force_masked, lease):
    index = json.loads((root/'transformer_cache/index.json').read_text())
    checkpoint = Path(index['base'])
    if sha256(checkpoint)!=index['base_sha256']:
        raise ValueError('Frozen base changed')
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg = TransformerResidualConfig(force_masked=force_masked)
    with lease.block():
        model = TransformerResidual(cfg).to('cuda:0')
        for name in ('proprio_mean','proprio_std','target_scale','target_offset'):
            getattr(model,name).copy_(saved['model'][name])
    metadata = saved['metadata']
    del saved
    gc.collect()
    return model,index,metadata


def train(args):
    root = args.root
    pilot = args.pilot
    if not json.loads((root/'transformer_interface_audit.json').read_text())['passed']:
        raise ValueError('Real residual and actual-history interface audit must pass first')
    label = args.variant+('_pilot' if pilot else '')
    output = root/'transformer_training'/label
    output.mkdir(parents=True,exist_ok=args.resume)
    lease = StudyLease(root)
    torch.manual_seed(20261103)
    model,index,metadata = build_model(root,args.variant=='masked',lease)
    groups = joint_groups(metadata['joint_order'])
    splits = index['splits']
    identities = splits['train'][:4] if pilot else splits['train']
    validation = splits['validation'][:4] if pilot else splits['validation']
    for identity in identities+validation:
        if sha256(root/'transformer_cache'/f'{identity}.pt')!=index['entries'][identity]['sha256']:
            raise ValueError('Residual cache mutated after frozen-base generation')
    if not pilot:
        for variant in ('force','masked'):
            result = json.loads((root/'transformer_training'/(variant+'_pilot')/'report.json').read_text())
            if not result['complete'] or result['updates']<1 or result['max_gradient_norm']<=0:
                raise ValueError('Both real pilot gradients must pass before formal training')
        budget_path = root/'transformer_budget.json'
        if not budget_path.exists():
            atomic_json(budget_path,dict(epochs=5,train_episodes=320,validation_episodes=40,
                optimizer='AdamW lr3e-4 betas(.9,.999) wd1e-4; new cosine warmup50',
                loss='contact-valid .5arm MSE+.5hand MSE +1e-4 unweighted delta mean square',
                selection='validation contact combined native .5arm MSE+.5hand MSE',
                history_protocol=HISTORY_PROTOCOL,established_epoch=time.time(),
                base_sha256=index['base_sha256'],scope='matched force/masked budgets, no test-based tuning'))
        epochs = json.loads(budget_path.read_text())['epochs']
    else:
        epochs = 1
    per_epoch = 0
    for identity in identities:
        record = load_residual_record(root,identity)
        per_epoch += sum(bool(record['gate'][start:start+96].any()) for start in range(0,len(record['time']),96))
    total = per_epoch*epochs
    configuration = dict(variant=args.variant,model=asdict(model.cfg),history_protocol=HISTORY_PROTOCOL,
        base_sha256=index['base_sha256'],cache_index_sha256=sha256(root/'transformer_cache/index.json'),
        epochs=epochs,updates_per_epoch=per_epoch,total_updates=total,seed=20261103,train=identities,validation=validation,
        precision='FP32 residual training/validation/deployment; frozen nominal BF16 native-tick replay',
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        source_hashes={str(path):sha256(path) for path in (Path(__file__),Path('visuotactile/transformer_data.py'),
            Path('visuotactile/models/transformer_residual.py'),Path('visuotactile/private_history.py'))})
    with lease.block():
        optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=1e-4)
    updates = epoch_start = episode_start = 0
    best = float('inf')
    history = []
    max_gradient = 0.
    accumulator = dict(loss_sum=0.,contact_count=0,gradient_sum=0.,updates=0)
    if args.resume:
        resumed = torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        if resumed['configuration']!=configuration:
            raise ValueError('Residual resume source/configuration changed')
        with lease.block():
            model.load_state_dict(resumed['model'])
            optimizer.load_state_dict(resumed['optimizer'])
        updates,epoch_start,episode_start = resumed['updates'],resumed['next_epoch'],resumed['next_episode']
        best,history,accumulator = resumed['best'],resumed['history'],resumed['accumulator']
        max_gradient = resumed['max_gradient_norm']
        torch.set_rng_state(resumed['cpu_rng'])
        torch.cuda.set_rng_state(resumed['cuda_rng'])
        del resumed
    atomic_json(output/'config.json',configuration)
    stopping = [False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))

    def checkpoint(next_epoch,next_episode):
        with lease.block():
            payload = cpu_snapshot(dict(configuration=configuration,model=model.state_dict(),optimizer=optimizer.state_dict(),
                updates=updates,next_epoch=next_epoch,next_episode=next_episode,best=best,history=history,
                accumulator=accumulator,max_gradient_norm=max_gradient,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
        save(output/'last.pt',payload)

    def measurements(epoch):
        return dict(epoch=epoch,updates=updates,validation=evaluate(model,root,validation,lease,groups),
            train_diagnostic=evaluate(model,root,identities if pilot else splits['train'][:16],lease,groups),ema=False)

    began = time.monotonic()
    try:
        if not args.resume:
            history.append(measurements(0))
            atomic_json(output/'metrics.json',history)
            checkpoint(0,0)
        for epoch in range(epoch_start,epochs):
            order = list(identities)
            random.Random(20261103+epoch).shuffle(order)
            model.train()
            for ordinal,identity in enumerate(order):
                if epoch==epoch_start and ordinal<episode_start:
                    continue
                record = load_residual_record(root,identity)
                state = None
                for start in range(0,len(record['time']),96):
                    stop = min(start+96,len(record['time']))
                    anchors = torch.arange(start,stop,6)
                    inputs_cpu,labels_cpu = residual_inputs(record,start,stop),residual_labels(record,anchors)
                    with lease.block():
                        inputs,labels = move(inputs_cpu,'cuda:0'),move(labels_cpu,'cuda:0')
                        optimizer.zero_grad(set_to_none=True)
                        prediction,state = model(inputs,state)
                        mask = labels['valid']&labels['gate']
                        loss = residual_loss(prediction,labels['delta'],mask,model.cfg.amplitude_weight)['loss']
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite residual objective')
                        if mask.any():
                            rate = 3e-4*learning_rate(updates,total,50)
                            for parameter_group in optimizer.param_groups:
                                parameter_group['lr'] = rate
                            loss.backward()
                            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                            if not torch.isfinite(gradient):
                                raise FloatingPointError('Nonfinite residual gradient')
                            optimizer.step()
                            updates += 1
                            max_gradient = max(max_gradient,float(gradient))
                            weight = int(mask.sum())
                            accumulator['loss_sum'] += float(loss)*weight
                            accumulator['contact_count'] += weight
                            accumulator['gradient_sum'] += float(gradient)
                            accumulator['updates'] += 1
                        state = state.detach()
                    if updates%25==0:
                        atomic_json(output/'status.json',dict(state='training',pid=os.getpid(),epoch=epoch+1,
                            updates=updates,total_updates=total,elapsed_s=time.monotonic()-began))
                if (ordinal+1)%64==0 or stopping[0]:
                    checkpoint(epoch,ordinal+1)
                if stopping[0]:
                    atomic_json(output/'status.json',dict(state='stopped',updates=updates,recover='last.pt'))
                    return
            metrics = measurements(epoch+1)
            metrics['train'] = dict(loss=accumulator['loss_sum']/max(1,accumulator['contact_count']),
                gradient_norm=accumulator['gradient_sum']/max(1,accumulator['updates']),lr=rate,
                contact_steps=accumulator['contact_count'],updates=accumulator['updates'],episodes=len(identities))
            if metrics['validation']['selection_mse']<best:
                best = metrics['validation']['selection_mse']
                with lease.block():
                    payload = cpu_snapshot(dict(residual_version=1,trained=updates>0,config=asdict(model.cfg),
                        model=model.state_dict(),configuration=configuration,base_sha256=index['base_sha256'],
                        history_protocol=HISTORY_PROTOCOL,epoch=epoch+1,updates=updates,validation=metrics['validation']))
                save(output/'best.pt',payload)
            history.append(metrics)
            atomic_json(output/'metrics.json',history)
            accumulator = dict(loss_sum=0.,contact_count=0,gradient_sum=0.,updates=0)
            checkpoint(epoch+1,0)
            print(args.variant,'epoch',epoch+1,'updates',updates,'validation',metrics['validation']['selection_mse'],flush=True)
        result = dict(complete=updates==total,epochs=epochs,updates=updates,max_gradient_norm=max_gradient,
            best=best,elapsed_s=time.monotonic()-began,memory=lease.memory(),base_sha256=index['base_sha256'])
        atomic_json(output/'report.json',result)
        atomic_json(output/'status.json',dict(state='complete',**result))
    except BaseException as error:
        atomic_json(output/'interruption.json',dict(error=repr(error),updates=updates,recover='last.pt complete-episode boundary'))
        raise


def test(args):
    root = args.root
    directory = root/'transformer_training'/args.variant
    if (directory/'test.json').exists():
        raise ValueError('Residual test already evaluated')
    lease = StudyLease(root)
    model,index,metadata = build_model(root,args.variant=='masked',lease)
    saved = torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    with lease.block():
        model.load_state_dict(saved['model'])
    metrics = evaluate(model,root,index['splits']['test'],lease,joint_groups(metadata['joint_order']))
    atomic_json(directory/'test.json',dict(checkpoint_sha256=sha256(directory/'best.pt'),epoch=saved['epoch'],
        updates=saved['updates'],metrics=metrics,base_sha256=index['base_sha256']))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--variant',choices=('force','masked'),required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--test',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    test(args) if args.test else train(args)
