"""Sequential H64 training, EMA validation, resumable updates and held-out test."""
import argparse
import copy
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import signal
import time

import torch

from .action_reference import reference_to_command
from .compact import sha256
from .long_data import inputs_for, load_record, supervision
from .long_resources import Lease
from .models.long_horizon import LongHorizonConfig, LongHorizonPolicy
from .training_resources import atomic_json


def move(values, device):
    return {name:value.to(device) for name,value in values.items()}


def fit_statistics(root, identities):
    minimum = torch.full((29,),float('inf'))
    maximum = -minimum
    total = torch.zeros(58,dtype=torch.float64)
    squares = torch.zeros_like(total)
    count = 0
    for identity in identities:
        record = load_record(root,identity)
        minimum = torch.minimum(minimum,record['reference'].amin(0))
        maximum = torch.maximum(maximum,record['reference'].amax(0))
        proprio = record['proprio'].double()
        total += proprio.sum(0)
        squares += proprio.square().sum(0)
        count += len(proprio)
    spread = maximum-minimum
    constant = spread < 1e-4
    scale = 2./torch.where(constant,2.,spread)
    offset = torch.where(constant,-minimum,-1.-scale*minimum)
    mean = total/count
    std = (squares/count-mean.square()).clamp_min(1e-6).sqrt()
    return dict(target_scale=scale,target_offset=offset,proprio_mean=mean.float(),proprio_std=std.float())


@torch.no_grad()
def update_ema(ema, model, updates):
    step = max(0,updates-1)
    decay = min(.9999,1-(1+step)**(-.75)) if step > 0 else 0.
    original = list(model.parameters())
    averaged = list(ema.parameters())
    torch._foreach_mul_(averaged,decay)
    torch._foreach_add_(averaged,original,alpha=1-decay)


def learning_rate(updates, total, warmup):
    if updates < warmup:
        return (updates+1)/max(warmup,1)
    return max(0.,.5*(1+math.cos(math.pi*(updates-warmup)/max(1,total-warmup))))


def sampled_metrics(model,root,identities,lease,seed=7091):
    metadata = json.loads((root/'metadata.json').read_text())
    limits = torch.tensor(metadata['joint_limits_rad'][0])
    speed,dt = metadata['config']['action']['dof_speed_scale'],metadata['control_dt']
    generator = torch.Generator(device='cuda:0').manual_seed(seed)
    model.eval()
    totals = dict(reference_arm_square=0.,reference_hand_square=0.,command_arm_square=0.,
        command_hand_square=0.,targets=0,saturated=0,anchors=0,boundaries=0,boundary_square=0.,
        first12_targets=0,first12_command_square=0.,copy_q_arm_square=0.,copy_q_hand_square=0.,
        reference_target_square=0.,reference_motion_square=0.,zero_command_arm_square=0.,
        zero_command_hand_square=0.,copy_q_command_arm_square=0.,copy_q_command_hand_square=0.)
    with torch.no_grad():
        for identity in identities:
            record = load_record(root,identity)
            state = None
            last_command = None
            for start in range(0,len(record['time']),96):
                stop = min(start+96,len(record['time']))
                anchors = torch.arange(start,stop,12)
                inputs_cpu = inputs_for(record,start,stop,model.cfg.tactile)
                labels_cpu = supervision(record,anchors)
                with lease.block():
                    inputs = move(inputs_cpu,'cuda:0')
                    labels = move(labels_cpu,'cuda:0')
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        condition,state = model.encode(inputs,state)
                        prediction = model.predict_from_condition(condition,generator).float()
                    command = reference_to_command(prediction,labels['target_before'],limits,speed_scale=speed,control_dt=dt)
                    copy_reference = record['proprio'][anchors,:29].to('cuda:0')[:,None].expand_as(prediction)
                    copy_command = reference_to_command(copy_reference,labels['target_before'],limits,
                                                        speed_scale=speed,control_dt=dt)
                    valid = labels['valid']
                    count = int(valid.sum())
                    totals['targets'] += count
                    totals['anchors'] += len(anchors)
                    totals['saturated'] += int(((command.abs() >= .99999)&valid[...,None]).sum())
                    for name,selection in (('arm',slice(0,7)),('hand',slice(7,29))):
                        totals['reference_'+name+'_square'] += float(((prediction[...,selection]-labels['reference'][...,selection]).square().mean(-1)*valid).sum())
                        totals['command_'+name+'_square'] += float(((command[...,selection]-labels['command'][...,selection]).square().mean(-1)*valid).sum())
                        totals['zero_command_'+name+'_square'] += float((labels['command'][...,selection].square().mean(-1)*valid).sum())
                        totals['copy_q_command_'+name+'_square'] += float(((copy_command[...,selection]-labels['command'][...,selection]).square().mean(-1)*valid).sum())
                        copy_q = record['proprio'][anchors,:29].to('cuda:0')[:,None,selection]
                        totals['copy_q_'+name+'_square'] += float(((copy_q-labels['reference'][...,selection]).square().mean(-1)*valid).sum())
                    totals['reference_target_square'] += float(((prediction-labels['target_after']).square().mean(-1)*valid).sum())
                    totals['reference_motion_square'] += float(((prediction-labels['joint_after']).square().mean(-1)*valid).sum())
                    first_valid = valid[:,:12]
                    totals['first12_targets'] += int(first_valid.sum())
                    totals['first12_command_square'] += float(((command[:,:12]-labels['command'][:,:12]).square().mean(-1)*first_valid).sum())
                    for plan in command:
                        if last_command is not None:
                            totals['boundary_square'] += float((plan[0]-last_command).square().mean())
                            totals['boundaries'] += 1
                        last_command = plan[11]
    result = {key:value for key,value in totals.items() if not key.endswith('_square')}
    for key,value in totals.items():
        if key.endswith('_square'):
            count = totals['boundaries'] if key == 'boundary_square' else totals['first12_targets'] if key == 'first12_command_square' else totals['targets']
            result[key.replace('_square','_rmse')] = math.sqrt(value/max(1,count))
    result['command_saturation_fraction'] = totals['saturated']/max(1,29*totals['targets'])
    result['selection_mse'] = .5*(totals['command_arm_square']+totals['command_hand_square'])/max(1,totals['targets'])
    result['scope'] = 'teacher-forced recorded targets for native conversion; not simulated future controller targets'
    result['episodes'] = len(identities)
    return result


def save(path,payload):
    temporary = path.with_suffix('.partial.pt')
    torch.save(payload,temporary)
    temporary.replace(path)


def cpu_snapshot(value):
    if isinstance(value,torch.Tensor):
        return value.detach().to('cpu',copy=True)
    if isinstance(value,dict):
        return {name:cpu_snapshot(item) for name,item in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(cpu_snapshot(item) for item in value)
    return value


def train(args):
    root = args.root
    index = json.loads((root/'cache_index.json').read_text())
    splits = index['splits']
    if args.stage == 'full':
        admission = json.loads((root/'pilot_admission.json').read_text())
        if not admission.get('passed') or admission.get('prediction_type') != args.prediction_type:
            raise ValueError('Full-episode fitting admission has not passed')
    identities = splits['train'][:4] if args.stage == 'pilot' else splits['train']
    validation_ids = splits['validation'][:4] if args.stage == 'pilot' else splits['validation']
    output = root/(args.group+'_'+args.stage+('_'+args.tag if args.tag else ''))
    output.mkdir(exist_ok=args.resume)
    cfg = LongHorizonConfig(group=args.group,prediction_type=args.prediction_type)
    lease = Lease(root)
    device = 'cuda:0'
    torch.manual_seed(args.seed)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    statistics = fit_statistics(root,splits['train'])
    with lease.block():
        model = LongHorizonPolicy(cfg).to(device)
        for name,value in statistics.items():
            getattr(model,name).copy_(value.to(device))
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
    updates_per_epoch = sum(math.ceil(index['entries'][identity]['native_rows']/96) for identity in identities)
    total_updates = updates_per_epoch*args.epochs
    warmup = min(500,max(10,total_updates//10)) if args.stage == 'pilot' else 500
    configuration = dict(model=asdict(cfg),seed=args.seed,epochs=args.epochs,updates_per_epoch=updates_per_epoch,
        total_updates=total_updates,learning_rate=1e-4,schedule='cosine',warmup_updates=warmup,
        ema=dict(inv_gamma=1.,power=.75,min_value=0.,max_value=.9999),
        normalizer='DP limits affine train320 min/max, constant channels preserved; no label/sample clipping',
        optimizer=dict(name='AdamW',betas=[.95,.999],weight_decay=1e-6),
        precision='BF16 autocast, FP32 parameters and loss',native_chunk=96,anchors_per_full_chunk=8,
        selection='lowest EMA sampled arm/hand equally weighted native-command MSE on validation; fixed RNG7091',
        cache_index_sha256=sha256(root/'cache_index.json'),data_history='past expert commands explicitly teacher-forced',
        train_episodes=identities,validation_episodes=validation_ids,test_episodes=splits['test'],
        source_hashes={str(path):sha256(path) for path in (Path(__file__),Path('visuotactile/long_data.py'),
            Path('visuotactile/models/long_horizon.py'),Path('visuotactile/action_reference.py'),Path('visuotactile/long_resources.py'))})
    updates = 0
    epoch_start = episode_start = 0
    history = []
    best = float('inf')
    if args.resume:
        saved = torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        if saved['configuration'] != configuration:
            before = dict(saved['configuration'])
            after = dict(configuration)
            old_sources = before.pop('source_hashes')
            new_sources = after.pop('source_hashes')
            changed = {path for path in set(old_sources)|set(new_sources)
                       if old_sources.get(path)!=new_sources.get(path)}
            runtime_files = {str(Path(__file__)),str(Path('visuotactile/long_resources.py'))}
            if not args.resume_runtime_update or before!=after or not changed<=runtime_files:
                raise ValueError('Resume configuration differs; model/data/optimizer changes are never migrated')
            atomic_json(output/f'resume_runtime_{saved["updates"]}.json',dict(
                original_source_hashes=old_sources,current_source_hashes=new_sources,
                restored_updates=saved['updates'],changed=sorted(changed),utc_epoch=time.time(),
                scope='explicitly reviewed runtime-only update; model/data/optimizer/RNG contract unchanged'))
        with lease.block():
            model.load_state_dict(saved['model'])
            ema.load_state_dict(saved['ema'])
            optimizer.load_state_dict(saved['optimizer'])
        updates,epoch_start,episode_start = saved['updates'],saved['next_epoch'],saved['next_episode']
        history,best = saved['history'],saved['best_validation']
        torch.set_rng_state(saved['cpu_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        del saved
    atomic_json(output/'config.json',configuration)
    stop = [False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    def checkpoint(next_epoch,next_episode):
        with lease.block():
            payload = cpu_snapshot(dict(version=2,configuration=configuration,model=model.state_dict(),
                ema=ema.state_dict(),optimizer=optimizer.state_dict(),updates=updates,next_epoch=next_epoch,
                next_episode=next_episode,history=history,best_validation=best,cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),progress_scope='episode boundary; no missing-prefix continuation'))
        save(output/'last.pt',payload)
        del payload
        gc.collect()
    def inference_payload(epoch,validation):
        return dict(version=2,config=asdict(cfg),model=ema.state_dict(),updates=updates,epoch=epoch,
            validation=validation,configuration=configuration,metadata=json.loads((root/'metadata.json').read_text()),
            vision={name:index[name] for name in ('dino_revision','dino_weights_sha256','dino_preprocess')})
    if args.stage == 'pilot' and not args.resume:
        zero = sampled_metrics(ema,root,identities,lease)
        atomic_json(output/'initial_train_sampled.json',zero)
    began = time.monotonic()
    measured_updates = updates
    timings = dict(io_s=0.,compute_s=0.,updates=0,native_rows=0,anchors=0,valid_targets=0)
    try:
        for epoch in range(epoch_start,args.epochs):
            order = list(identities)
            import random
            random.Random(args.seed+epoch).shuffle(order)
            model.train()
            for ordinal,identity in enumerate(order):
                if epoch == epoch_start and ordinal < episode_start:
                    continue
                tick = time.monotonic()
                record = load_record(root,identity)
                timings['io_s'] += time.monotonic()-tick
                state = None
                for start in range(0,len(record['time']),96):
                    stop_at = min(start+96,len(record['time']))
                    anchors = torch.arange(start,stop_at,12)
                    inputs_cpu = inputs_for(record,start,stop_at,cfg.tactile)
                    labels_cpu = supervision(record,anchors)
                    target_count = int(labels_cpu['valid'].sum())
                    with lease.block():
                        tick = time.monotonic()
                        inputs = move(inputs_cpu,device)
                        labels = move(labels_cpu,device)
                        rate = 1e-4*learning_rate(updates,total_updates,warmup)
                        for group in optimizer.param_groups:
                            group['lr'] = rate
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            condition,state = model.encode(inputs,state)
                            loss = model.loss_from_condition(condition,labels['reference'],labels['valid'])
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite loss')
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        if not torch.isfinite(norm):
                            raise FloatingPointError('Nonfinite gradients')
                        optimizer.step()
                        update_ema(ema,model,updates)
                        state = state.detach()
                        updates += 1
                        torch.cuda.synchronize()
                        loss_value = float(loss.detach())
                        timings['compute_s'] += time.monotonic()-tick
                    timings['updates'] += 1
                    timings['native_rows'] += stop_at-start
                    timings['anchors'] += len(anchors)
                    timings['valid_targets'] += target_count
                    if updates%25 == 0:
                        elapsed = time.monotonic()-began
                        throughput = (updates-measured_updates)/max(elapsed,1e-9)
                        status = dict(state='training',epoch=epoch+1,episode=ordinal,
                            updates=updates,scheduled_updates=total_updates,loss=loss_value,lr=rate,
                            seconds_per_update=1/max(throughput,1e-9),estimated_epoch_seconds=updates_per_epoch/max(throughput,1e-9),
                            timings=timings,wait_s=lease.wait_s,pacing_s=lease.pacing_s,
                            memory=lease.memory(),
                            remaining_train_seconds=(total_updates-updates)/max(throughput,1e-9))
                        atomic_json(output/'status.json',status)
                        if updates == 25 and not args.resume:
                            atomic_json(output/'first25.json',status)
                if (ordinal+1)%64 == 0 or stop[0]:
                    checkpoint(epoch,ordinal+1)
                if stop[0]:
                    atomic_json(output/'stopped.json',dict(updates=updates,epoch=epoch,next_episode=ordinal+1))
                    return
            if args.stage == 'full' or (epoch+1)%5 == 0 or epoch+1 == args.epochs:
                checkpoint(epoch,len(order))
            if args.stage == 'full' or (epoch+1)%5 == 0 or epoch+1 == args.epochs:
                validation = sampled_metrics(ema,root,validation_ids,lease)
                history.append(dict(epoch=epoch+1,updates=updates,validation=validation))
                if validation['selection_mse'] < best:
                    best = validation['selection_mse']
                    with lease.block():
                        payload = cpu_snapshot(inference_payload(epoch+1,validation))
                    save(output/'best.pt',payload)
                    del payload
                    gc.collect()
                atomic_json(output/'metrics.json',history)
                print(args.group,args.stage,'epoch',epoch+1,'updates',updates,'validation',validation['selection_mse'],flush=True)
            if args.stage == 'full' or (epoch+1)%5 == 0 or epoch+1 == args.epochs:
                checkpoint(epoch+1,0)
        if args.stage == 'pilot':
            learned = sampled_metrics(ema,root,identities,lease)
            initial = json.loads((output/'initial_train_sampled.json').read_text())
            improved = (learned['reference_arm_rmse']+learned['reference_hand_rmse'] <
                        initial['reference_arm_rmse']+initial['reference_hand_rmse'])
            atomic_json(output/'fit.json',dict(initial=initial,trained=learned,improved=improved,updates=updates))
        atomic_json(output/'report.json',dict(completed_epochs=args.epochs,updates=updates,history=history,
            elapsed_s=time.monotonic()-began,work_s=lease.work_s,wait_s=lease.wait_s,pacing_s=lease.pacing_s,
            memory=lease.memory(),timings=timings))
        atomic_json(output/'status.json',dict(state='complete',completed_epochs=args.epochs,updates=updates,
            scheduled_updates=total_updates,best_validation=best,memory=lease.memory()))
    except BaseException as error:
        atomic_json(output/'interruption.json',dict(error=repr(error),updates_executed=updates,
            recover_from='last.pt; only its completed update count and episode boundary are recoverable'))
        raise


def test(args):
    root = args.root
    output = root/(args.group+'_full')
    if (output/'test.json').exists():
        raise ValueError('The held-out test has already been evaluated')
    saved = torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    lease = Lease(root,deadline=json.loads((root/'session.json').read_text())['deadline'])
    with lease.block():
        model = LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to('cuda:0').eval()
        model.load_state_dict(saved['model'])
    splits = json.loads((root/'splits.json').read_text())
    result = sampled_metrics(model,root,splits['test'],lease,seed=7093)
    atomic_json(output/'test.json',dict(checkpoint_sha256=sha256(output/'best.pt'),epoch=saved['epoch'],
        updates=saved['updates'],metrics=result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--group',choices=['G00-new','G10-new'],required=True)
    parser.add_argument('--stage',choices=['pilot','full','test'],required=True)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--seed',type=int,default=20261031)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--resume-runtime-update',action='store_true')
    parser.add_argument('--tag',default='')
    parser.add_argument('--prediction-type',choices=['epsilon','sample'],default='sample')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'test':
        test(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
