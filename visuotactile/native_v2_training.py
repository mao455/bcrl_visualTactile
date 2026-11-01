"""State-first stages. New optimizer identity; immutable checkpoints and deterministic resume."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import torch

from .compact import sha256
from .native_data import CONTRACT
from .models.native_dp import NativePolicy
from .models.vendor.ema_model import EMAModel
from .native_v2_gates import require_gate
from .native_v2_resources import LeaseV2
from .native_v2_state import BC_CONTRACT, StateBC, StateWindows, condition_windows, load_state_policy
from .train_long import cpu_snapshot, save
from .training_resources import atomic_json


def order_for(size,epoch,seed=20261102):
    return torch.randperm(size,generator=torch.Generator().manual_seed(seed+epoch))


def next_batch(size,epoch,cursor,batch_size):
    if cursor==size: epoch,cursor=epoch+1,0
    indices=order_for(size,epoch)[cursor:cursor+batch_size]
    return indices,epoch,cursor+len(indices)


@torch.no_grad()
def offline(model,windows,lease,seeds=(7091,17091,27091),batch_size=32):
    model.eval();device=next(model.parameters()).device;rows=[];started=time.monotonic()
    for seed in seeds:
        generator=torch.Generator(device=device).manual_seed(seed)
        totals={f'{h}_{p}':0. for h in (12,64) for p in ('arm','hand')};counts={12:0,64:0}
        saturation=elements=0;loss_sum=valid_sum=0.
        for start in range(0,len(windows),batch_size):
            indices=torch.arange(start,min(start+batch_size,len(windows)))
            with lease.block():
                robot,geometry,command,valid=windows.batch(indices,device)
                condition=condition_windows(model,robot,geometry)
                # Validation objective uses its own generator, so sampling stream is stable.
                lg=torch.Generator(device=device).manual_seed(seed+start+100000)
                loss=model.loss(condition,command,valid,lg)
                loss_sum+=float(loss)*int(valid.sum());valid_sum+=int(valid.sum())
                raw=model.predict(condition,generator);prediction=raw.clamp(-1,1)
                saturation+=int(((raw.abs()>1)&valid[...,None]).sum());elements+=int(valid.sum())*29
                for horizon in (12,64):
                    mask=valid[:,:horizon];counts[horizon]+=int(mask.sum())
                    for part,selection in (('arm',slice(0,7)),('hand',slice(7,29))):
                        error=(prediction[:,:horizon,selection]-command[:,:horizon,selection]).square().mean(-1)
                        totals[f'{horizon}_{part}']+=float(torch.where(mask,error,0.).sum())
        metrics={k+'_rmse':math.sqrt(v/counts[int(k.split('_')[0])]) for k,v in totals.items()}
        rows.append(dict(seed=seed,metrics=metrics,selection_mse=.5*(totals['12_arm']+totals['12_hand'])/counts[12],
            validation_objective=loss_sum/valid_sum,preclip_saturation_fraction=saturation/elements))
    scores=[r['selection_mse'] for r in rows]
    return dict(seeds=rows,selection_mse=sum(scores)/len(scores),selection_std=float(torch.tensor(scores).std(unbiased=False)),
        episodes=len(windows.identities),anchors=len(windows),elapsed_s=time.monotonic()-started,
        source='expert history teacher forcing; no autonomous success inference')


def train(root,output,kind,until,resume=False):
    require_gate(root,'G0')
    session=json.loads((root/'session.json').read_text());config=session['config']['state_training']
    source=Path(session['source_root']);data=json.loads((source/'data.json').read_text())
    identities=data['splits']['train'][:4] if kind=='bc' else data['splits']['train']
    total=session['config']['diagnostic_bc']['max_updates'] if kind=='bc' else max(config['milestones'])
    if not 0<until<=total:raise ValueError('Update milestone outside preregistered bound')
    output.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol=2,kind=kind,config=config,total=total,identities=identities,
        data_sha256=sha256(source/'data.json'),statistics_sha256=sha256(source/'statistics.pt'),
        implementation_sha256=session['implementation_sha256'])
    if (output/'configuration.json').exists() and json.loads((output/'configuration.json').read_text())!=identity:
        raise ValueError('Training configuration drift; new root required')
    if (output/'last.pt').exists() and not resume:raise ValueError('Explicit resume required')
    atomic_json(output/'configuration.json',identity)
    windows=StateWindows(source,identities);validation=StateWindows(source,data['splits']['validation'])
    lease=LeaseV2(root,deadline=session['deadline']);device='cuda:0'
    with lease.block():
        torch.manual_seed(20261102);model=(StateBC() if kind=='bc' else NativePolicy('S-NativeDP')).to(device)
        for k,v in windows.stats.items():getattr(model,k).copy_(v.to(device))
        averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],betas=tuple(config['betas']),weight_decay=config['weight_decay'])
    updates=epoch=cursor=anchors_seen=0;best=math.inf;history=[]
    if resume:
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        if saved['configuration']!=identity:raise ValueError('Resume identity mismatch')
        with lease.block():
            model.load_state_dict(saved['model']);averaged.load_state_dict(saved['ema']);optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        updates,epoch,cursor,anchors_seen=(saved[k] for k in ('updates','epoch','cursor','anchors_seen'))
        best,history=saved['best'],saved['history'];ema.optimization_step=saved['ema_step'];del saved
    if updates>until:raise ValueError('Cannot rewind committed updates')
    checkpoint_s=0.;began=time.monotonic()
    def checkpoint(name):
        nonlocal checkpoint_s
        at=time.monotonic()
        # CPU transfer after completed GPU block; no I/O under GPU lock.
        payload=cpu_snapshot(dict(contract=BC_CONTRACT if kind=='bc' else CONTRACT,training_protocol=2,
            group='S-NativeDP',configuration=identity,updates=updates,epoch=epoch,cursor=cursor,anchors_seen=anchors_seen,
            model=model.state_dict(),ema=averaged.state_dict(),optimizer=optimizer.state_dict(),
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),ema_step=ema.optimization_step,
            best=best,history=history))
        save(output/name,payload);checkpoint_s+=time.monotonic()-at
    while updates<until:
        indices,next_epoch,next_cursor=next_batch(len(windows),epoch,cursor,config['batch_anchors'])
        with lease.block():
            model.train();optimizer.zero_grad(set_to_none=True)
            robot,geometry,command,valid=windows.batch(indices,device)
            condition=condition_windows(model,robot,geometry);loss=model.loss(condition,command,valid)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            warmup=config['warmup']
            scale=min(1.,(updates+1)/warmup)*.5*(1+math.cos(math.pi*max(0,updates-warmup)/max(1,total-warmup)))
            for pg in optimizer.param_groups:pg['lr']=config['lr']*scale
            optimizer.step();ema.step(model)
        updates+=1;epoch,cursor=next_epoch,next_cursor;anchors_seen+=len(indices)
        row=dict(updates=updates,completed_traversals=epoch+int(cursor==len(windows)),traversal_fraction=anchors_seen/len(windows),
            episode_traversals_equivalent=anchors_seen/len(windows)*len(identities),effective_batch_anchors=len(indices),
            lr=optimizer.param_groups[0]['lr'],loss=float(loss.detach()),gradient_norm=float(norm),
            work_s=lease.work_s,wait_s=lease.wait_s,pacing_s=lease.pacing_s,load_s=windows.load_s+validation.load_s,
            checkpoint_s=checkpoint_s,elapsed_s=time.monotonic()-began)
        atomic_json(output/'status.json',row)
        with (output/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if updates==1 or updates%100==0 or updates==until:checkpoint('last.pt')
        if updates==until:
            score=offline(averaged,validation,lease)
            improved=score['selection_mse']<best
            if improved:best=score['selection_mse']
            history.append(dict(updates=updates,metrics=score))
            if improved:checkpoint('best_validation_offline.pt')
            checkpoint('last.pt');checkpoint(f'milestone_{updates}.pt')
            atomic_json(output/'validation.json',history)
    atomic_json(output/f'completed_{until}.json',dict(software_complete=True,updates=updates,
        offline=history[-1]['metrics'],closed_loop_success=None,checkpoint_sha256=sha256(output/f'milestone_{updates}.pt')))


def frozen_offline(root,output):
    session=json.loads((root/'session.json').read_text());source=Path(session['source_root'])
    data=json.loads((source/'data.json').read_text());lease=LeaseV2(root,deadline=session['deadline'])
    output.mkdir(parents=True,exist_ok=True)
    reports={}
    for label,entry in session['checkpoints'].items():
        if label=='last' and session['checkpoints']['best']['ema_sha256']==entry['ema_sha256']:
            reports[label]=dict(reused='best',reason='All EMA tensor names and bytes identical',ema_sha256=entry['ema_sha256']);continue
        saved=torch.load(entry['path'],map_location='cpu',weights_only=False)
        if sha256(entry['path'])!=entry['sha256']:raise ValueError('Old checkpoint drift')
        with lease.block():model=load_state_policy(saved,'cuda:0')
        from .native_data import Episode
        from .models.native_dp import encode_segment
        probe=StateWindows(source,session['cohorts']['train'][:1])
        with torch.no_grad(),lease.block():
            expected,_,_=encode_segment(model,Episode(source,probe.identities[0],'S-NativeDP'),0,len(probe.records[probe.identities[0]]['time']))
            robot,geom,_,_=probe.batch(torch.arange(len(probe)),'cuda:0')
            actual=condition_windows(model,robot,geom)
            error=float((actual-expected).abs().max())
        atomic_json(output/(label+'_cache_parity.json'),dict(condition_max_error=error,threshold=1e-5,passed=error<=1e-5))
        if error>1e-5:raise ValueError('State CUDA cache/online condition mismatch')
        reports[label]={}
        for split,ids in [('train8',session['cohorts']['train']),('validation40',data['splits']['validation'])]:
            reports[label][split]=offline(model,StateWindows(source,ids),lease,session['config']['diffusion_seeds'])
            atomic_json(output/'result.json',dict(software_complete=False,checkpoints=reports))
        del saved,model
    atomic_json(output/'result.json',dict(software_complete=True,checkpoints=reports,test_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['bc','dp','frozen-offline'],required=True);p.add_argument('--until',type=int);p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.kind=='frozen-offline':frozen_offline(a.root,a.output)
    else:train(a.root,a.output,a.kind,a.until,a.resume)
