"""Resume final H24 models with diverse anchors; simulation remains independent."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import time

import torch

from .anchor_sampling import AnchorSampler,anchor_batch,mask_totals,weighted_parts,continuation_lr
from .compact import sha256
from .models.vendor.ema_model import EMAModel
from .state_full_training import restore_training,training_state
from .state_image_offline import setup,evaluate,no_kit,read
from .train_long import save
from .training_resources import atomic_json


def load(root,job,output):
    session,data,ids,vals,records,model,objective,lease,metadata=setup(root,job)
    objective.weights=read(root/(job+'_audit')/'calibration.json')['weights']
    directory=root/(job+'_train');directory.mkdir(exist_ok=True)
    config=dict(version='anchor_continue_v1',base=session['checkpoints'][job],
        effective_batch=session['effective_batch'],microbatch=session['microbatch'],
        sampler='episode_balanced_shuffled_episode_and_anchor_decks',
        history='unchanged current-weight complete prefix, original96tick TBPTT boundaries',
        horizon=24,commit=12,loss_weights=objective.weights,loss_id=session['checkpoints'][job]['config']['loss_id'],
        lr_peak=session['lr_peak'],lr_floor=session['lr_floor'],warmup=session['warmup'],
        continuation_schedule_steps=session['targets'][job]['300'],statistics_sha256=sha256(session['position_statistics']))
    path=directory/'configuration.json'
    if path.exists():assert read(path)==config
    else:atomic_json(path,config)
    with lease.block():
        averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
    generator=torch.Generator();noise=torch.Generator(device='cuda')
    source=directory/'last.pt' if (directory/'last.pt').exists() else Path(session['checkpoints'][job]['path'])
    if source.name!='last.pt':assert sha256(source)==session['checkpoints'][job]['sha256']
    saved=torch.load(source,map_location='cpu',weights_only=False)
    with lease.block():restore_training(saved,model,averaged,optimizer,ema,generator,noise)
    sampler=AnchorSampler(session['lengths'],generator,session['effective_batch'])
    if 'anchor_sampler' in saved:
        assert saved['config_sha256']==sha256(path)
        sampler.load_state_dict(saved['anchor_sampler'])
        step=saved['continuation_updates'];history=saved['continuation_history'];best=saved['best']
    else:
        assert saved['updates']==10000
        step=0;history=[];best=saved['history'][-1]['validation']['selection_mse']
    restored=dict(model=all(torch.equal(v.cpu(),saved['model'][k]) for k,v in model.state_dict().items()),
        ema=all(torch.equal(v.cpu(),saved['ema'][k]) for k,v in averaged.state_dict().items()),
        ema_step=ema.optimization_step==saved['ema_optimization_step'],
        sampler_rng=torch.equal(generator.get_state(),saved['schedule_rng']),noise_rng=torch.equal(noise.get_state(),saved['noise_rng']),
        optimizer_steps=all(int(v['step'])==10000+step for v in optimizer.state.values()))
    assert all(restored.values())
    atomic_json(output/'restore.json',dict(passed=True,source=str(source),from_update=10000+step,checks=restored,
        old_lr=saved['optimizer']['param_groups'][0]['lr'],scheduler_replaced=step==0,
        source_sha256=sha256(source),sampler_schema_changed=step==0))
    del saved
    return session,ids,vals,records,model,averaged,optimizer,ema,sampler,noise,objective,lease,directory,config,step,history,best


def contract_check(root,job,output):
    args=load(root,job,output)
    s,ids,vals,records,model,averaged,opt,ema,sampler,noise,obj,lease,directory,config,step,history,best=args
    model.eval();checks=[]
    from .state_history import HistoryBank
    from .native_data import Episode
    # Current final model, nonzero prefix, episode boundary, and a masked tail.
    for identity in (ids[0],ids[1],vals[0]):
        rec=records[identity];ticks=sorted({0,96,(len(rec['time'])-1)//12*12})
        ticks=[t for t in ticks if t<len(rec['time'])]
        with torch.no_grad(),lease.block():
            full,anchors,_=model.encode(rec)
            bank=HistoryBank() if job=='state' else None;memory=None;online={}
            episode=Episode(s['source_root'],identity,'I-HistoryDP') if job=='image' else None
            for t in range(max(ticks)+1):
                if job=='state':
                    c=bank.observe(0,1,t,model,rec['actor'][t],rec['robot'][t],rec['geometry'][t],rec['time'][t])
                else:
                    memory=model.observe(rec['robot'][t],rec['time'][t],t,episode.frame(t,'cuda:0') if t%12==0 else None,memory)
                    c=model.condition(memory,rec['time'][t]) if t%12==0 else None
                if t in ticks:online[t]=c
            for t in ticks:
                c,y,m=anchor_batch(model,records,[(identity,t)])
                err=float((c-full[t//12:t//12+1]).abs().max());online_err=float((c-online[t]).abs().max())
                assert err<2e-5 and online_err<2e-5,(err,online_err)
                assert torch.equal(y[0][m[0]],rec['command'][t:t+24])
                checks.append(dict(identity=identity,tick=t,full_condition_max_abs=err,online_max_abs=online_err,valid_targets=int(m.sum())))
    # The sampler must not leave episode-hidden state on the model between calls.
    with torch.no_grad(),lease.block():
        a=anchor_batch(model,records,[(ids[0],96)])[0]
        anchor_batch(model,records,[(ids[1],0)])
        b=anchor_batch(model,records,[(ids[0],96)])[0]
        cross=float((a-b).abs().max());assert cross==0
        changed=dict(records[ids[0]])
        for name in ('actor','robot','geometry'):
            if name in changed:
                changed[name]=changed[name].clone();changed[name][97:]+=0.125
        c=anchor_batch(model,{ids[0]:changed},[(ids[0],96)])[0]
        future=float((a-c).abs().max());assert future==0
    atomic_json(output/'result.json',dict(passed=True,job=job,checks=checks,cross_episode_contamination_max_abs=cross,
        future_observation_intervention_max_abs=future,optimizer_updates=0,kit_dependency=no_kit(),original_weights_unchanged=True))


def deterministic_loss(model,records,ids,obj,lease):
    values=[]
    with torch.no_grad():
        for identity in ids:
            with lease.block():
                c,a,_=model.encode(records[identity]);a=a.to(records[identity]['command'].device)
                from .state_history_training import labels
                y,m=labels(records[identity],a,24)
                loss,parts,_=obj.forward(model,c,y,m,torch.Generator(device='cuda').manual_seed(8091))
                values.append(dict(loss=float(loss),**{k:float(v) for k,v in parts.items()}))
    return {k:sum(v[k] for v in values)/len(values) for k in values[0]}


def train(root,job,target,output):
    s,ids,vals,records,model,averaged,opt,ema,sampler,noise,obj,lease,directory,config,step,history,best=load(root,job,output)
    assert step<=target<=s['targets'][job]['300']
    started=step;began=time.time();stop=[False];row=None
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    initial=next(model.denoiser.parameters()).detach().clone()
    if step==target and (directory/f'continued_{step}.pt').exists():
        # Complete stage result was lost after checkpoint commit: emit a truthful
        # readback result without optimizing or replacing the immutable milestone.
        milestone=directory/f'continued_{step}.pt'
        prior=torch.load(milestone,map_location='cpu',weights_only=False)
        assert prior['updates']==10000+step and prior['config_sha256']==sha256(directory/'configuration.json')
        assert all(torch.equal(v.cpu(),prior['model'][k]) for k,v in model.state_dict().items())
        metrics=next(x for x in prior['continuation_history'] if x['continuation_update']==step)
        atomic_json(output/'result.json',dict(passed=True,job=job,from_update=10000+step,update=10000+step,
            continuation_updates=step,recovered_completed_stage=True,new_optimizer_updates=0,effective_batch=s['effective_batch'],
            parameter_max_change=0.,cumulative_anchor_epochs=metrics['cumulative_anchor_epochs'],checkpoint_readback=True,
            checkpoint=str(milestone.resolve()),checkpoint_sha256=sha256(milestone),metrics=metrics,coverage=sampler.coverage(),kit_dependency=no_kit()))
        return
    def snapshot():
        return training_state(model,averaged,opt,ema,sampler.generator,noise,kind='dp',group=model.group,
            updates=10000+step,continuation_updates=step,anchor_sampler=sampler.state_dict(),
            continuation_history=history,best=best,history=history,config_sha256=sha256(directory/'configuration.json'),
            loss_id=config['loss_id'],configuration=config,job=job,train_ids=ids,validation_ids=vals,
            base_checkpoint=s['checkpoints'][job],cumulative_sampled_anchors=s['old_exposure'][job]['anchors']+sampler.draws)
    def checkpoint():save(directory/'last.pt',snapshot())
    def assess(full):
        nonlocal best,history
        train_ids=ids[:16] if full else ids[:4];val_ids=vals if full else vals[:4]
        m=dict(update=10000+step,continuation_update=step,
            cumulative_anchor_epochs=(s['old_exposure'][job]['anchors']+sampler.draws)/s['anchor_population'],
            train=evaluate(averaged,records,train_ids,obj,lease),validation=evaluate(averaged,records,val_ids,obj,lease),
            train_loss=deterministic_loss(averaged,records,train_ids,obj,lease),
            validation_loss=deterministic_loss(averaged,records,val_ids,obj,lease),validation_scope='validation40' if full else 'smoke_validation4')
        history=[x for x in history if x['continuation_update']!=step]+[m]
        improved=full and m['validation']['selection_mse']<best
        if improved:best=m['validation']['selection_mse']
        checkpoint()
        if improved:
            tmp=directory/'best.partial.pt';tmp.unlink(missing_ok=True);os.link(directory/'last.pt',tmp);tmp.replace(directory/'best.pt')
            atomic_json(directory/'best.json',dict(update=10000+step,score=best,path=str((directory/'best.pt').resolve()),sha256=sha256(directory/'best.pt')))
        atomic_json(directory/'metrics.json',history)
        return m
    if not (directory/'best.pt').exists():
        # Preserve the original full model/optimizer payload as the baseline best.
        os.link(s['checkpoints'][job]['path'],directory/'best.pt')
        atomic_json(directory/'best.json',dict(update=10000,score=best,path=str((directory/'best.pt').resolve()),sha256=s['checkpoints'][job]['sha256']))
    try:
        while step<target:
            lease.check()
            selected=sampler.draw();totals=mask_totals(s['lengths'],selected)
            model.train();opt.zero_grad(set_to_none=True);loss_sum=0.;parts_sum={}
            tick=time.time()
            for j in range(0,len(selected),s['microbatch']):
                with lease.block():
                    c,y,m=anchor_batch(model,records,selected[j:j+s['microbatch']])
                    _,parts,_=obj.forward(model,c,y,m,noise)
                    loss,scaled=weighted_parts(parts,m,totals,obj.weights)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite continuation loss')
                    loss.backward();loss_sum+=float(loss.detach())
                    for k,v in scaled.items():parts_sum[k]=parts_sum.get(k,0.)+float(v.detach())
                    del c,y,m,loss,parts,scaled
            with lease.block():
                branch=model.history.weight_hh_l0 if job=='state' else model.native_gru.weight_hh
                history_grad=float(branch.grad.norm());norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm) or not math.isfinite(history_grad):raise FloatingPointError('Nonfinite gradient')
                lr=continuation_lr(step+1,config['continuation_schedule_steps'],config['warmup'],config['lr_peak'],config['lr_floor'])
                for g in opt.param_groups:g['lr']=lr
                opt.step();ema.step(model);step+=1
            exposure=sampler.coverage();cumulative=(s['old_exposure'][job]['anchors']+sampler.draws)/s['anchor_population']
            row=dict(update=10000+step,continuation_update=step,loss=loss_sum,components=parts_sum,
                gradient_norm=float(norm),history_grad=history_grad,lr=lr,effective_batch=len(selected),
                distinct_episodes=len(set(i for i,t in selected)),valid_target_rows=totals[0],valid_adjacent_pairs=totals[1],
                cumulative_anchor_epochs=cumulative,old_segment_epochs=s['old_exposure'][job]['segment_equivalent_epochs'],
                coverage=exposure,anchor_pairs=selected,anchors_per_second=len(selected)/(time.time()-tick),
                utc=time.time(),elapsed_s=time.time()-began,work_s=lease.work_s,wait_s=lease.wait_s,pacing_s=lease.pacing_s,**lease.memory())
            with (directory/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            atomic_json(directory/'progress.json',dict(state='running',update=10000+step,continuation_update=step,target=target,
                cumulative_anchor_epochs=cumulative,lr=lr,loss=loss_sum,pid=os.getpid(),utc=time.time(),**no_kit()))
            if step%100==0 or stop[0]:checkpoint()
            if stop[0]:raise InterruptedError('Saved complete optimizer boundary')
            if step%1000==0 and step<target:assess(True)
    except BaseException:
        # Never save an advanced sampler after an incomplete optimizer step.
        # last.pt remains an atomic, complete boundary; every failed log is retained.
        atomic_json(output/'interrupted.json',dict(last_complete_update=10000+step,last_saved=str(directory/'last.pt'),utc=time.time()))
        raise
    metrics=assess(target>50)
    milestone=directory/f'continued_{step}.pt'
    if milestone.exists():raise FileExistsError('Committed checkpoint already exists')
    os.link(directory/'last.pt',milestone)
    saved=torch.load(milestone,map_location='cpu',weights_only=False)
    assert saved['updates']==10000+step and saved['anchor_sampler']['draws']==step*s['effective_batch']
    assert saved['ema_optimization_step']==10000+step
    delta=float((next(model.denoiser.parameters()).detach()-initial).abs().max())
    if step>started:assert delta>0 and row['lr']>0 and row['history_grad']>0
    atomic_json(output/'result.json',dict(passed=True,job=job,from_update=10000+started,update=10000+step,
        continuation_updates=step,effective_batch=s['effective_batch'],parameter_max_change=delta,
        cumulative_anchor_epochs=(s['old_exposure'][job]['anchors']+sampler.draws)/s['anchor_population'],
        finite_loss_gradient=True,checkpoint_readback=True,checkpoint=str(milestone.resolve()),checkpoint_sha256=sha256(milestone),
        metrics=metrics,coverage=sampler.coverage(),kit_dependency=no_kit(),elapsed_s=time.time()-began))
    atomic_json(directory/'progress.json',dict(state='completed_stage',update=10000+step,continuation_update=step,target=target,pid=None,utc=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['contract','train']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job',choices=['state','image'],required=True);p.add_argument('--target',type=int);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.mode=='contract':contract_check(a.root,a.job,a.output)
    else:train(a.root,a.job,a.target,a.output)
