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
from .checkpointing import restore_training,training_state
from .state_image_offline import setup,evaluate,no_kit,read
from .checkpointing import save
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
                from .anchor_sampling import labels
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


def fresh(root, data_root, output, target, smoke=False, smoke_device='cpu'):
    """Same H24 model/objective/anchor algorithm, new dataset and random run.

    A disposable one-update CPU check exercises the same save/resume path. Formal
    use starts only after the independently accepted dataset gate.
    """
    import subprocess
    from contextlib import nullcontext
    from .state_history import PersistentState, HistoryBank
    from .position_control import contract
    from .state_image_loss import GeometryObjective, LOSS_ID
    from .state_resources import lease_for
    from .narrow_data import read as read_data
    import h5py
    data_root=data_root.resolve();root=root.resolve()
    data=read_data(data_root/'data.json');collection=data_root.parent
    if not smoke and not (collection/'COMPLETE.json').exists():raise ValueError('Dataset not ready')
    if target is None or target<1 or (smoke and target!=1):raise ValueError('Explicit positive update budget; smoke exactly one')
    session=read_data(collection/'session.json');cached=torch.load(data_root/'numeric.pt',map_location='cpu',weights_only=False)
    if session.get('version')=='tight_state_foundation_v1' and not smoke:
        from .tight_gate import validate_formal_manifest
        validate_formal_manifest(read_data(collection/'manifest.json'),read_data(collection/'configuration.json'))
        if data['manifest_sha256']!=sha256(collection/'manifest.json'):
            raise ValueError('Frozen manifest changed after cache/statistics construction')
        from .tight_gate import validate_training_gates
        validate_training_gates(collection,data_root)
    assert cached['data_sha256']==sha256(data_root/'data.json') and cached['contract']==contract(24,True)
    ids=data['splits']['train'];vals=data['splits']['validation']
    # Never load test trajectories into the optimizer or validation worker.
    records={i:cached['records'][i] for i in ids+vals};del cached
    stats=torch.load(data_root/'statistics.pt',map_location='cpu',weights_only=False)
    if not ids or (not smoke and not vals):raise ValueError('Nonempty train/validation required')
    runtime=read(root/'session.json') if (root/'session.json').exists() else {}
    continuation=runtime.get('continuation')
    seed=continuation['configuration']['seed'] if continuation else runtime.get('random_seed',20261105)
    device=smoke_device if smoke else 'cuda:0';torch.manual_seed(seed)
    model=PersistentState('dp',contract(24,True))
    expert=torch.load(session['expert_checkpoint'],map_location='cpu',weights_only=False)[0]['model']
    model.set_statistics(stats,expert);del expert
    with h5py.File(data['entries'][ids[0]]['path']) as f:metadata=json.loads(f.attrs['metadata_json'])
    objective=GeometryObjective(metadata);objective.weights=session['loss_weights']
    config=dict(version='fresh_narrow_state_v1',initialization='random_model_fresh_optimizer_ema_schedule_counters',
        seed=seed,data_sha256=sha256(data_root/'data.json'),statistics_sha256=sha256(data_root/'statistics.pt'),
        contract=contract(24,True),loss_id=LOSS_ID,loss_weights=objective.weights,target=target,
        effective_batch=64,microbatch=4,lr_peak=3e-5,lr_floor=3e-6,warmup=min(200,max(1,target//10)),
        sampler='episode_balanced_shuffled_decks_short_tail_no_drop',smoke=smoke,device=device,
        source_sha256={str(p):sha256(p) for p in sorted(Path('visuotactile').rglob('*.py'))})
    if continuation:
        config.update(initialization='complete_checkpoint_continuation_original_schedule_clamped_at_floor',
                      continuation=continuation)
    if session.get('version')=='tight_state_foundation_v1':
        config['input_builder']='canonical_joint_state_v1'
    root.mkdir(parents=True,exist_ok=True)
    cp=root/'configuration.json'
    if cp.exists():assert read(cp)==config,'Resume configuration/source mismatch'
    else:atomic_json(cp,config)
    step=0;best=float('inf');monitor=None;lease=None
    if device!='cpu':
        if not (root/'session.json').exists():atomic_json(root/'session.json',dict(resource_policy='shared_recent_v1',resource_target_percent=40.,
            training_deadline=session['deadline'],external_monitor_root=str(collection)))
        runtime=read(root/'session.json')
        if runtime.get('external_monitor_root'):
            lease=lease_for(Path(runtime['external_monitor_root']),deadline=runtime['training_deadline'])
        else:
            (root/'STOP_MONITOR').unlink(missing_ok=True)
            log=(root/'monitor.log').open('a')
            monitor=subprocess.Popen(['bash','visuotactile/run_cpu.sh','-m','visuotactile.state_resources','--root',str(root)],stdout=log,stderr=subprocess.STDOUT)
            lease=lease_for(root,deadline=runtime['training_deadline'])
    block=lambda: lease.block() if lease else nullcontext()
    def checkpoint():
        save(root/'last.pt',training_state(model,averaged,opt,ema,generator,noise,kind='dp',group=model.group,
            updates=step,anchor_sampler=sampler.state_dict(),best=best,config_sha256=sha256(cp),configuration=config,
            loss_id=LOSS_ID,train_ids=ids,validation_ids=vals))
    try:
        # Admission and monitoring precede the first CUDA allocation, including
        # model/EMA construction and same-run optimizer restoration.
        with block():
            model.to(device);records={i:{k:v.to(device) for k,v in r.items()} for i,r in records.items()}
            averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
            opt=torch.optim.AdamW(model.parameters(),lr=3e-6,betas=(.95,.999),weight_decay=1e-6)
            generator=torch.Generator().manual_seed(seed);noise=torch.Generator(device=device).manual_seed(seed+1)
            sampler=AnchorSampler({i:len(records[i]['time']) for i in ids},generator,64)
            if (root/'last.pt').exists():
                old=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
                assert old['config_sha256']==sha256(cp)
                restore_training(old,model,averaged,opt,ema,generator,noise);sampler.load_state_dict(old['anchor_sampler'])
                step=old['updates'];best=old['best'];del old
            elif continuation:
                source=Path(continuation['path'])
                assert sha256(source)==continuation['sha256'],'Continuation source changed'
                old=torch.load(source,map_location='cpu',weights_only=False)
                for key in ('contract','data_sha256','statistics_sha256','loss_id','loss_weights','effective_batch','microbatch','sampler'):
                    assert old['configuration'][key]==config[key],('Continuation contract mismatch',key)
                restore_training(old,model,averaged,opt,ema,generator,noise);sampler.load_state_dict(old['anchor_sampler'])
                step=old['updates'];best=old['best']
                assert step==ema.optimization_step and all(int(v['step'])==step for v in opt.state.values())
                checks=dict(model=all(torch.equal(v.cpu(),old['model'][k]) for k,v in model.state_dict().items()),
                    ema=all(torch.equal(v.cpu(),old['ema'][k]) for k,v in averaged.state_dict().items()),
                    sampling_rng=torch.equal(generator.get_state(),old['schedule_rng']),
                    noise_rng=torch.equal(noise.get_state(),old['noise_rng']),
                    torch_rng=torch.equal(torch.get_rng_state(),old['torch_rng']),
                    cuda_rng=all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),old['cuda_rng'])),
                    lr=opt.param_groups[0]['lr']==old['optimizer']['param_groups'][0]['lr'])
                current=opt.state_dict()
                checks['optimizer_groups']=current['param_groups']==old['optimizer']['param_groups']
                checks['optimizer_moments']=all(torch.equal(value.cpu(),old['optimizer']['state'][i][key])
                    if torch.is_tensor(value) else value==old['optimizer']['state'][i][key]
                    for i,state in current['state'].items() for key,value in state.items())
                assert all(checks.values())
                atomic_json(output/'restore.json',dict(passed=True,checks=checks,from_update=step,
                    source_sha256=continuation['sha256'],optimizer_and_ema_preserved=True,scheduler_replaced=False))
                checkpoint();del old
        if smoke:
            with torch.no_grad():
                identity=ids[0];rec=records[identity];model.eval();full,anchors,_=model.encode(rec)
                bank=HistoryBank();maximum=0.
                for t in range(len(rec['time'])):
                    c=bank.observe(0,1,t,model,rec['actor'][t],rec['robot'][t],rec['geometry'][t],rec['time'][t])
                    if c is not None:maximum=max(maximum,float((c-full[t//12:t//12+1]).abs().max()))
                assert maximum<2e-5
                # Full-prefix anchor at a padded tail and independent lane/reset.
                tick=int(anchors[-1]);c,y,m=anchor_batch(model,records,[(identity,tick)])
                assert int(m.sum())==len(rec['time'])-tick and torch.equal(y[0][m[0]],rec['command'][tick:])
                changed={k:v.clone() for k,v in rec.items()}
                for key in ('actor','robot','geometry'):changed[key][13:]+=1
                a=anchor_batch(model,records,[(identity,12)])[0]
                b=anchor_batch(model,{identity:changed},[(identity,12)])[0];assert torch.equal(a,b)
            atomic_json(output/'history_check.json',dict(passed=True,online_max_abs=maximum,tail_valid=int(m.sum()),future_leakage=False))
        stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
        training_started=time.time();initial_step=step
        interval=runtime.get('validation_interval',session.get('validation_interval',500));stop_reason='update_budget'
        stage_stop=runtime.get('stage_stop_update',target)
        if not step<=stage_stop<=target:raise ValueError('Invalid finite stage boundary')
        if stage_stop<target:stop_reason='stage_update_budget'
        while step<stage_stop:
            if (root/'STOP').exists():checkpoint();raise InterruptedError('Stopped at optimizer boundary')
            if not smoke and lease and time.time()>=lease.deadline-600:
                stop_reason='fixed_wall_budget';checkpoint()
                if (root/'best.pt').exists():break
                raise TimeoutError('Training deadline without validated checkpoint')
            before_batch=(step,copy.deepcopy(sampler.state_dict()),generator.get_state(),noise.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            selected=sampler.draw();totals=mask_totals(sampler.lengths,selected);model.train();opt.zero_grad(set_to_none=True);loss_sum=0.
            for start in range(0,len(selected),4):
                with block():
                    c,y,m=anchor_batch(model,records,selected[start:start+4]);_,parts,_=objective.forward(model,c,y,m,noise)
                    loss,_=weighted_parts(parts,m,totals,objective.weights)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();loss_sum+=float(loss.detach())
            with block():
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(norm)
                step+=1
                if continuation:
                    from .narrow_decision import continuation_rate
                    lr=continuation_rate(step,continuation['configuration'])
                else:lr=3e-6 if target==1 else continuation_lr(step,target,config['warmup'],3e-5,3e-6)
                for group in opt.param_groups:group['lr']=lr
                opt.step();ema.step(model)
            progress=dict(update=step,target=target,loss=loss_sum,lr=lr,actual_batch=len(selected),
                sampled_anchors=sampler.draws,formal_updates=0 if smoke else step,
                stage_updates=step-initial_step,restored_updates=initial_step,
                updates_per_hour=(step-initial_step)*3600/max(1,time.time()-training_started),utc=time.time(),**no_kit())
            atomic_json(root/'progress.json',progress)
            with (root/'training.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
            bounded_end=step==stage_stop or (lease is not None and time.time()>=lease.deadline-600)
            if step%100==0 or bounded_end or stop[0] or step==1:checkpoint()
            if stop[0]:raise InterruptedError('Saved at optimizer boundary')
            if not smoke and (step%interval==0 or bounded_end):
                metrics=evaluate(averaged,records,vals,objective,lease)
                train_metrics=evaluate(averaged,records,ids[:min(8,len(ids))],objective,lease)
                early=fresh_early_metrics(averaged,records,dict(train=ids[:min(8,len(ids))],validation=vals),objective,lease)
                if metrics['selection_mse']<best:
                    best=metrics['selection_mse'];checkpoint();save(root/'best.pt',torch.load(root/'last.pt',map_location='cpu',weights_only=False))
                report=dict(update=step,metrics=metrics,train8=train_metrics,early=early,utc=time.time())
                if runtime.get('deterministic_validation_loss'):
                    report['validation_loss']=deterministic_loss(averaged,records,vals,objective,lease)
                    report['train8_loss']=deterministic_loss(averaged,records,ids[:min(8,len(ids))],objective,lease)
                atomic_json(root/'validation.json',report)
                with (root/'validation.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
                fresh_curves(root)
                if runtime.get('early_stop_minimum_update'):
                    from .narrow_decision import early_stop
                    rows=[json.loads(line) for line in (root/'validation.jsonl').read_text().splitlines()]
                    if early_stop(rows,runtime['early_stop_minimum_update']):
                        stop_reason='validation_plateau_or_overfit';checkpoint();break
            if bounded_end:
                if step<stage_stop:stop_reason='fixed_wall_budget'
                break
        saved=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        restore_training(saved,model,averaged,opt,ema,generator,noise)
        assert saved['updates']==step==ema.optimization_step
        assert all(int(v['step'])==step for v in opt.state.values())
        assert all(torch.equal(v.cpu(),saved['model'][k]) for k,v in model.state_dict().items())
        assert all(torch.equal(v.cpu(),saved['ema'][k]) for k,v in averaged.state_dict().items())
        atomic_json(output/'result.json',dict(passed=True,updates=step,disposable=smoke,formal_updates=0 if smoke else step,
            complete_restore=True,statistics_train_only=True,old_policy_checkpoint_loaded=bool(continuation),
            stage_updates=step-initial_step,restored_updates=initial_step,**no_kit()))
        if not smoke:atomic_json(root/'finished.json',dict(updates=step,target_updates=target,target_reached=step==target,
            stop_reason=stop_reason,
            best_checkpoint=str(root/'best.pt'),utc=time.time()))
    except (TimeoutError,InterruptedError):
        if 'before_batch' in locals() and step==before_batch[0]:
            sampler.load_state_dict(before_batch[1]);generator.set_state(before_batch[2]);noise.set_state(before_batch[3])
            torch.set_rng_state(before_batch[4]);torch.cuda.set_rng_state_all(before_batch[5])
        if 'opt' in locals():checkpoint()
        raise
    finally:
        if monitor is not None:
            (root/'STOP_MONITOR').touch();monitor.wait(timeout=15);log.close()


@torch.no_grad()
def fresh_early_metrics(model,records,cohorts,objective,lease):
    """First committed chunk: motion accuracy vs holding the current target."""
    from .position_control import bounded_targets
    report={}
    for cohort,ids in cohorts.items():
        rows=[]
        for identity in ids:
            rec=records[identity]
            with lease.block():
                cond,_,_=model.encode(rec,stop=1)
                prediction=bounded_targets(model.predict(cond,torch.Generator(device='cuda').manual_seed(7091))[0,:12],rec['joint_limits'])
                truth=rec['command'][:12];n=len(truth);prediction=prediction[:n];before=rec['robot'][0,58:87]
                pred_delta=prediction[:,:7]-before[:7];true_delta=truth[:,:7]-before[:7]
                moving=true_delta.norm(dim=-1)>1e-6
                p,_=objective.fk(prediction);q,_=objective.fk(truth)
                rows.append(dict(identity=identity,arm_rmse_rad=float((prediction[:,:7]-truth[:,:7]).square().mean().sqrt()),
                    hold_arm_rmse_rad=float(true_delta.square().mean().sqrt()),fk_mm=float((p[:,0]-q[:,0]).square().sum(-1).mean().sqrt()*1000),
                    direction_cos=float(torch.nn.functional.cosine_similarity(pred_delta[moving],true_delta[moving]).mean()) if moving.any() else None,
                    amplitude_ratio=float((pred_delta[moving].norm(dim=-1)/true_delta[moving].norm(dim=-1)).mean()) if moving.any() else None))
        report[cohort]=rows
    return report


def fresh_curves(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(line) for line in (root/'validation.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for key,label in [('train8','train8'),('metrics','validation')]:
        axes[0].plot([r['update'] for r in rows],[r[key]['selection_mse'] for r in rows],label=label)
        axes[1].plot([r['update'] for r in rows],[r[key]['arm_rmse'] for r in rows],label=label)
    for ax,title in zip(axes,['Equal arm/hand MSE','Arm RMSE (rad)']):ax.set_title(title);ax.set_xlabel('Fresh optimizer updates');ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(root/'curves.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['contract','train','fresh-smoke','fresh-train']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--smoke-device',choices=['cpu','cuda:0'],default='cpu')
    p.add_argument('--job',choices=['state','image'],required=True);p.add_argument('--target',type=int);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.mode.startswith('fresh-'):
        if a.job!='state' or a.data is None:p.error('Fresh entry requires --job state and --data')
        fresh(a.root,a.data,a.output,a.target,smoke=a.mode=='fresh-smoke',smoke_device=a.smoke_device)
    elif a.mode=='contract':contract_check(a.root,a.job,a.output)
    else:train(a.root,a.job,a.target,a.output)
