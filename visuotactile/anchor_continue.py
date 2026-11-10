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
                context=projection_batch(obj,[(identity,int(t)) for t in a],c.device)
                loss,parts,_=obj.forward(model,c,y,m,torch.Generator(device='cuda').manual_seed(8091),geometry_context=context)
                if hasattr(obj,'current_loss_weights'):
                    loss=parts['base']+sum(obj.current_loss_weights[k]*parts[k] for k in obj.current_loss_weights)
                values.append(dict(loss=float(loss),**{k:float(v) for k,v in parts.items()}))
    return {k:sum(v[k] for v in values)/len(values) for k in values[0]}


def geometry_selection(objective, selected, lengths, seed):
    """Stratify two valid commit configurations over one complete update.

    Sampling uses no predicted distance/feasibility and does not change the
    anchor or diffusion RNG. Base supervision still consumes every valid row.
    """
    mask=torch.zeros((len(selected),24),dtype=torch.bool)
    candidates=[[],[]]
    for row,(identity,tick) in enumerate(selected):
        context=objective.loss_context[identity]
        for offset in range(min(12,lengths[identity]-tick)):
            contact=bool(context['stage'][tick+offset]==1 and context['contact_mask'][tick+offset].any())
            candidates[int(contact)].append((row,offset))
    generator=torch.Generator().manual_seed(seed)
    chosen=[]
    for pool in candidates:
        if pool:chosen.append(pool[int(torch.randint(len(pool),(1,),generator=generator))])
    remaining=[x for pool in candidates for x in pool if x not in chosen]
    if len(chosen)<2 and remaining:chosen.append(remaining[int(torch.randint(len(remaining),(1,),generator=generator))])
    for row,tick in chosen:mask[row,tick]=True
    return mask,dict(selected_count=len(chosen),eligible_commit_count=sum(map(len,candidates)),
        eligible_contact_count=len(candidates[1]),selected_contact_count=sum(x in candidates[1] for x in chosen),
        sampling='one required-contact and one noncontact if present; otherwise two without replacement',
        normalization='equal selected configuration mean, then full-update accumulation; not population-unbiased')


def projection_batch(objective, selected, device, *, selected_mask=None, update_selected_count=None, active=True):
    """Loss-only future context, never inserted into the model record/condition."""
    if objective.hand_geometry is None:return None
    from .state_image_loss import HandGeometrySoftLoss
    direct=isinstance(objective.hand_geometry,HandGeometrySoftLoss)
    if direct and not active:return dict(active=False)
    if direct and selected_mask is not None and not selected_mask.any():
        return dict(active=True,selected_mask=selected_mask.to(device),update_selected_count=update_selected_count)
    context=objective.loss_context
    result={key:torch.stack([context[i][key][torch.arange(t,t+24).clamp_max(len(context[i][key])-1)]
        for i,t in selected]).to(device) for key in ('object_pose_base','contact_mask','stage')}
    if direct:
        if selected_mask is None:
            selected_mask,report=geometry_selection(objective,selected,
                {i:len(context[i]['stage']) for i,t in selected},8091)
            update_selected_count=report['selected_count']
        result.update(selected_mask=selected_mask.to(device),update_selected_count=update_selected_count,active=active)
    return result


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
    from .state_image_loss import GeometryObjective, LOSS_ID, PALM_LOSS_ID
    from .state_resources import lease_for
    from .narrow_data import read as read_data
    import h5py
    data_root=data_root.resolve();root=root.resolve()
    data=read_data(data_root/'data.json');collection=data_root.parent
    if not smoke and not (collection/'COMPLETE.json').exists():raise ValueError('Dataset not ready')
    if target is None or target<1 or (smoke and target!=1):raise ValueError('Explicit positive update budget; smoke exactly one')
    session=read_data(collection/'session.json');cached=torch.load(data_root/'numeric.pt',map_location='cpu',weights_only=False)
    vision=session.get('vision_pair')
    palm=session.get('palm_training')
    if vision and not smoke:
        gate=read(collection/'vision_acceptance.json')
        assert gate['passed'] and gate['data_sha256']==sha256(data_root/'data.json')
        assert read(collection/'cache_index.json')['complete']
    if session.get('version')=='tight_state_foundation_v1' and not smoke:
        from .tight_gate import validate_formal_manifest
        validate_formal_manifest(read_data(collection/'manifest.json'),read_data(collection/'configuration.json'))
        if data['manifest_sha256']!=sha256(collection/'manifest.json'):
            raise ValueError('Frozen manifest changed after cache/statistics construction')
        if palm:
            check=read(collection/'palm_acceptance.json')
            assert check['passed'] and check['data_sha256']==sha256(data_root/'data.json') and check['statistics_sha256']==sha256(data_root/'statistics.pt')
        else:
            from .tight_gate import validate_training_gates
            validate_training_gates(collection,data_root)
    assert cached['data_sha256']==sha256(data_root/'data.json') and cached['contract']==contract(24,True)
    ids=data['splits']['train'];vals=data['splits']['validation']
    if vision and smoke:
        ids=list(read(collection/'cache_acceptance_index.json')['entries']);vals=[]
        assert len(ids)==2 and set(ids)<=set(data['splits']['train'])
    # Never load test trajectories into the optimizer or validation worker.
    records={i:cached['records'][i] for i in ids+vals};del cached
    stats=torch.load(data_root/'statistics.pt',map_location='cpu',weights_only=False)
    if not ids or (not smoke and not vals):raise ValueError('Nonempty train/validation required')
    runtime=read(root/'session.json') if (root/'session.json').exists() else {}
    continuation=runtime.get('continuation');warm_start=runtime.get('warm_start');warm_start_audit=None
    if continuation and warm_start:raise ValueError('Target adaptation and optimizer continuation are distinct')
    seed=continuation['configuration']['seed'] if continuation else runtime.get('random_seed',20261105)
    device=smoke_device if smoke else 'cuda:0';torch.manual_seed(seed)
    if vision:
        from .state_history import PersistentImage
        from .position_control import image_contract
        from .native_data import visual_records
        modality=runtime['modality']
        model=PersistentImage('I-HistoryDP',image_contract('I-HistoryDP',24,modality,data.get('supervision')))
        if not warm_start:
            initial=torch.load(collection/'common_initial.pt',map_location='cpu',weights_only=False)
            state=model.state_dict()
            for key,value in initial.items():
                if key not in state or state[key].shape!=value.shape:raise ValueError('Shared initialization mismatch')
                state[key].copy_(value)
            assert set(state)-set(initial)==(set() if modality=='G00' else {k for k in state if k.startswith('tactile_')})
            del initial,state
        for k,v in stats.items():getattr(model,k).copy_(v)
        if modality=='G10':
            touch_stats=torch.load(data_root/'tactile_statistics.pt',map_location='cpu',weights_only=False)
            for k,v in touch_stats.items():getattr(model,k).copy_(v)
        if warm_start:
            from .state_history import load_image_warm_start
            source=Path(warm_start['path'])
            if sha256(source)!=warm_start['sha256']:raise ValueError('Warm-start source changed')
            saved=torch.load(source,map_location='cpu',weights_only=False)
            model,warm_start_audit=load_image_warm_start(saved,model.contract,stats)
            warm_start_audit.update(source=str(source.resolve()),source_sha256=warm_start['sha256'])
            del saved
        records=visual_records(data_root,ids+vals,modality)
    else:
        model=PersistentState('dp',contract(24,True))
        expert=torch.load(session['expert_checkpoint'],map_location='cpu',weights_only=False)[0]['model']
        model.set_statistics(stats,expert);del expert
    with h5py.File(data['entries'][ids[0]]['path']) as f:metadata=json.loads(f.attrs['metadata_json'])
    objective=GeometryObjective(metadata,mode=PALM_LOSS_ID if palm else LOSS_ID,commit=model.contract['commit'])
    objective.weights=dict(session['loss_weights'])
    projection=runtime.get('hand_geometry')
    if projection:
        if data.get('supervision')!='q_reached':raise ValueError('Hand geometry requires actual configuration labels')
        acceptance=read(Path(projection['acceptance']))
        if not acceptance['passed']:raise ValueError('Hand geometry gate failed')
        live=read(Path(projection['live']))
        objective.enable_hand_geometry(live['inventory'],acceptance,
            loss_kind=projection.get('loss_kind','projection_v1'),**projection['options'])
        objective.weights['hand_geometry']=acceptance['hand_geometry_weight']
        objective.loss_context=torch.load(data_root/'loss_context.pt',map_location='cpu',weights_only=False)
    config=dict(version='fresh_narrow_state_v1',initialization='random_model_fresh_optimizer_ema_schedule_counters',
        seed=seed,data_sha256=sha256(data_root/'data.json'),statistics_sha256=sha256(data_root/'statistics.pt'),
        contract=model.contract,loss_id=objective.mode,loss_weights=objective.weights,target=target,
        effective_batch=64,microbatch=4,lr_peak=3e-5,lr_floor=3e-6,warmup=min(200,max(1,target//10)),
        sampler='episode_balanced_shuffled_decks_short_tail_no_drop',smoke=smoke,disposable=smoke or runtime.get('disposable_gate',False),device=device,
        source_sha256={str(p):sha256(p) for p in sorted(Path('visuotactile').rglob('*.py'))})
    if continuation:
        config.update(initialization='complete_checkpoint_continuation_original_schedule_clamped_at_floor',
                      continuation=continuation)
    if warm_start:
        config.update(initialization='old_G00_EMA_q_reached_adaptation_fresh_optimizer_ema_schedule',
            warm_start=warm_start_audit)
    if session.get('version')=='tight_state_foundation_v1':
        config['input_builder']='canonical_joint_state_v1'
    if palm:
        if continuation: raise ValueError('This palm experiment requires a fresh model')
        config.update(geometry_warmup=palm['geometry_warmup_updates'],geometry_commit=model.contract['commit'],
            geometry_position_units='m^2',geometry_rotation='0.5 Frobenius matrix squared',
            selection=palm['selection'],diagnostic_cohorts=palm['diagnostic_cohorts'],
            effective_batch=session['training']['effective_batch'],microbatch=session['training']['microbatch'])
    if vision:
        config.update(version='fresh_visual_pair_v1',modality=modality,
            shared_initial_sha256=warm_start['sha256'] if warm_start else sha256(collection/'common_initial.pt'),dino_sha256=session['dino_sha256'],
            input_builder='canonical_joint_state_v1',microbatch=runtime.get('microbatch',config['microbatch']))
        if data.get('supervision') is not None:
            config.update(version='matched_visual_supervision_v1',supervision=data['supervision'],
                normalization_identity=model.contract['normalization_identity'],
                clean_sample_field=model.contract['clean_sample_field'])
        if projection:config.update(hand_geometry=projection,geometry_acceptance_sha256=sha256(projection['acceptance']),
            loss_context_sha256=sha256(data_root/'loss_context.pt'),loss_weights=dict(objective.weights))
    if runtime.get('geometry_probe_gain_sequence'):
        if not config['disposable']:raise ValueError('Geometry timing gain overrides are disposable only')
        config['geometry_probe_gain_sequence']=runtime['geometry_probe_gain_sequence']
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
            loss_id=objective.mode,train_ids=ids,validation_ids=vals))
    try:
        # Admission and monitoring precede the first CUDA allocation, including
        # model/EMA construction and same-run optimizer restoration.
        with block():
            model.to(device);records={i:{k:v.to(device) if torch.is_tensor(v) else v for k,v in r.items()} for i,r in records.items()}
            averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
            opt=torch.optim.AdamW(model.parameters(),lr=3e-6,betas=(.95,.999),weight_decay=1e-6)
            generator=torch.Generator().manual_seed(seed);noise=torch.Generator(device=device).manual_seed(seed+1)
            sampler=AnchorSampler({i:len(records[i]['time']) for i in ids},generator,config['effective_batch'])
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
        if smoke and not vision:
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
        if runtime.get('probe_limit_updates'):
            if not config['disposable']:raise ValueError('Probe stop requires disposable run')
            stage_stop=min(stage_stop,step+int(runtime['probe_limit_updates']))
        if not step<=stage_stop<=target:raise ValueError('Invalid finite stage boundary')
        if stage_stop<target:stop_reason='stage_update_budget'
        while step<stage_stop:
            if stop[0]:checkpoint();raise InterruptedError('SIGTERM at complete optimizer boundary')
            if (root/'STOP').exists():checkpoint();raise InterruptedError('Stopped at optimizer boundary')
            if not smoke and lease and time.time()>=lease.deadline-600:
                stop_reason='fixed_wall_budget';checkpoint()
                if (root/'best.pt').exists():break
                raise TimeoutError('Training deadline without validated checkpoint')
            before_batch=(step,copy.deepcopy(sampler.state_dict()),generator.get_state(),noise.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
            selected=sampler.draw();totals=mask_totals(sampler.lengths,selected,model.contract['commit'] if palm else None)
            model.train();opt.zero_grad(set_to_none=True);loss_sum=0.;parts_sum={};batch_started=time.time()
            objective.gain=min(1.,(step+1)/config['geometry_warmup']) if palm else 1.
            weights={k:v*objective.gain for k,v in objective.weights.items()}
            from .state_image_loss import HandGeometrySoftLoss
            direct_geometry=isinstance(objective.hand_geometry,HandGeometrySoftLoss)
            geometry_gain=0.;geometry_report={};geometry_wall=0.;geometry_gradients=[];base_output_hand_grad_squared=0.
            if direct_geometry:
                geometry_gain=max(0.,min(1.,(step+1-projection.get('warmup_updates',100))/max(1,projection.get('ramp_updates',400))))
                if config.get('geometry_probe_gain_sequence'):
                    geometry_gain=config['geometry_probe_gain_sequence'][step-initial_step]
                weights['hand_geometry']=objective.weights['hand_geometry']*geometry_gain
                selected_geometry,geometry_report=geometry_selection(objective,selected,sampler.lengths,seed+1000003*(step+1))
            objective.current_loss_weights=weights
            for start in range(0,len(selected),config['microbatch']):
                with block():
                    selected_micro=selected[start:start+config['microbatch']]
                    c,y,m=anchor_batch(model,records,selected_micro)
                    context=projection_batch(objective,selected_micro,c.device,
                        selected_mask=selected_geometry[start:start+config['microbatch']] if direct_geometry else None,
                        update_selected_count=geometry_report.get('selected_count'),active=geometry_gain>0 if direct_geometry else True)
                    _,parts,prediction=objective.forward(model,c,y,m,noise,geometry_context=context)
                    if objective.last_geometry_metrics is not None:
                        geometry_wall+=objective.last_geometry_metrics.get('elapsed_s',0.)
                        with (root/'geometry_projection.jsonl').open('a') as f:f.write(json.dumps(dict(update=step+1,
                            selected=selected_micro,**objective.last_geometry_metrics))+'\n')
                    if direct_geometry and config['disposable'] and geometry_gain:
                        base_grad=torch.autograd.grad(parts['base'],prediction,retain_graph=True)[0]*(m.sum()/max(1,totals[0]))
                        base_output_hand_grad_squared+=float(base_grad[...,7:].square().sum())
                        if objective.last_geometry_metrics.get('attempted',0):
                            grad=torch.autograd.grad(parts['hand_geometry'],prediction,retain_graph=True)[0]
                            geometry_gradients.append(dict(arm_norm=float(grad[...,:7].norm()),hand_norm=float(grad[...,7:].norm()),
                                base_hand_norm=float(base_grad[...,7:].norm()),weighted_hand_norm=float(grad[...,7:].norm())*weights['hand_geometry'],
                                scope='selected microbatch contributions to complete-update output gradient; not full model-parameter gradient ratio'))
                    loss,scaled=weighted_parts(parts,m,totals,weights,model.contract['commit'],global_hand_geometry=direct_geometry)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();loss_sum+=float(loss.detach())
                    for k,v in scaled.items():parts_sum[k]=parts_sum.get(k,0.)+float(v.detach())
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
                sampled_anchors=sampler.draws,formal_updates=0 if config.get('disposable',smoke) else step,
                stage_updates=step-initial_step,restored_updates=initial_step,
                updates_per_hour=(step-initial_step)*3600/max(1,time.time()-training_started),utc=time.time(),**no_kit())
            if palm:
                elapsed=time.time()-training_started
                estimated=(target-step)*elapsed/max(1,step-initial_step)
                progress.update(components=parts_sum,geometry_gain=objective.gain,loss_weights=weights,
                    anchor_equivalent_epochs=sampler.coverage()['anchor_equivalent_epochs'],
                    anchor_population=sampler.coverage()['anchor_population'],seconds_per_update=time.time()-batch_started,
                    resource_wait_s=lease.wait_s+lease.pacing_s if lease else 0.,
                    remaining_seconds_estimate=estimated,estimated_completion_utc=time.time()+estimated,
                    hard_stop_utc=lease.deadline if lease else None,eta_includes_observed_validation_and_resource_wait=True,
                    numerical_gradient_norm=float(norm))
            if direct_geometry:
                progress.update(hand_geometry_gain=geometry_gain,hand_geometry_seconds=geometry_wall,
                    geometry_sampling=geometry_report,geometry_probe_gradients=geometry_gradients,
                    base_full_update_output_hand_gradient_norm=math.sqrt(base_output_hand_grad_squared) if config['disposable'] and geometry_gain else None,
                    geometry_full_update_output_hand_gradient_norm=math.sqrt(sum(x['hand_norm']**2 for x in geometry_gradients)) if config['disposable'] and geometry_gain else None,
                    geometry_solver_calls=0)
            atomic_json(root/'progress.json',progress)
            with (root/'training.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
            bounded_end=step==stage_stop or (lease is not None and time.time()>=lease.deadline-600)
            if step%100==0 or bounded_end or stop[0] or step==1:checkpoint()
            if stop[0]:raise InterruptedError('Saved at optimizer boundary')
            if not smoke and (step%interval==0 or bounded_end or (palm and step in runtime.get('comparison_updates',[]))):
                if palm:
                    from .state_image_offline import deployment_evaluate
                    val_ids=palm['diagnostic_cohorts']['validation'][:palm['interim_validation_episodes']]
                    train_ids=palm['diagnostic_cohorts']['train'][:palm['interim_train_episodes']]
                    metrics=deployment_evaluate(averaged,records,val_ids,objective,lease)
                    train_metrics=deployment_evaluate(averaged,records,train_ids,objective,lease)
                    early={};score=metrics['selection_score']
                else:
                    val_ids=vals;train_ids=ids[:min(8,len(ids))]
                    metrics=evaluate(averaged,records,vals,objective,lease)
                    train_metrics=evaluate(averaged,records,train_ids,objective,lease)
                    early=fresh_early_metrics(averaged,records,dict(train=train_ids,validation=vals),objective,lease)
                    score=metrics['selection_mse']
                if score<best:
                    best=score;checkpoint();save(root/'best.pt',torch.load(root/'last.pt',map_location='cpu',weights_only=False))
                report=dict(update=step,metrics=metrics,train8=train_metrics,early=early,utc=time.time())
                if palm: report.update(anchor_equivalent_epochs=sampler.coverage()['anchor_equivalent_epochs'],train_scope='fixed_train2',validation_scope='fixed_validation4',loss_weights=weights,geometry_gain=objective.gain)
                if runtime.get('deterministic_validation_loss'):
                    report['validation_loss']=deterministic_loss(averaged,records,val_ids,objective,lease)
                    report['train8_loss']=deterministic_loss(averaged,records,train_ids,objective,lease)
                atomic_json(root/'validation.json',report)
                with (root/'validation.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
                if step in (runtime.get('comparison_updates') or []):
                    checkpoint();save(root/f'comparison_{step}.pt',torch.load(root/'last.pt',map_location='cpu',weights_only=False))
                fresh_curves(root)
                if runtime.get('early_stop_minimum_update'):
                    rows=[json.loads(line) for line in (root/'validation.jsonl').read_text().splitlines()]
                    if palm:
                        scores=[r['metrics']['selection_score'] for r in rows]
                        plateau=len(rows)>=4 and step>=runtime['early_stop_minimum_update'] and 1-min(scores[-4:])/max(scores[-4],1e-12)<.01
                        severe=len(rows)>=5 and all(v>2*min(scores[:-3]) for v in scores[-3:])
                        should_stop=plateau or severe
                    else:
                        from .narrow_decision import early_stop
                        should_stop=early_stop(rows,runtime['early_stop_minimum_update'])
                    if should_stop:
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
        atomic_json(output/'result.json',dict(passed=True,updates=step,disposable=config.get('disposable',smoke),formal_updates=0 if config.get('disposable',smoke) else step,
            complete_restore=True,statistics_train_only=True,old_policy_checkpoint_loaded=bool(continuation or warm_start),
            warm_start=warm_start_audit,scheduler_total_updates=target,probe_limit_updates=runtime.get('probe_limit_updates'),
            stage_updates=step-initial_step,restored_updates=initial_step,**no_kit()))
        if not smoke:atomic_json(root/'finished.json',dict(updates=step,target_updates=target,target_reached=step==target,
            stop_reason=stop_reason,
            best_checkpoint=str(root/'best.pt'),wall_s=time.time()-training_started,
            wait_s=lease.wait_s+lease.pacing_s if lease else 0.,coverage=sampler.coverage(),utc=time.time()))
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
    if 'selection_score' in rows[0]['metrics']:
        import csv
        flattened=[]
        for row in rows:
            for key in ('train8','metrics'):
                values=row[key]
                flattened.append(dict(update=row['update'],anchor_epochs=row['anchor_equivalent_epochs'],
                    cohort='fixed_train2' if key=='train8' else 'fixed_validation4',
                    **{k:values[k] for k in ('arm_rmse','hand_rmse','ee_distance_rmse_mm','ee_rotation_mean_deg','selection_score')}))
        with (root/'validation_curves.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flattened[0]));writer.writeheader();writer.writerows(flattened)
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        fields=('arm_rmse','hand_rmse','ee_distance_rmse_mm','ee_rotation_mean_deg','selection_score')
        titles=('Arm RMSE (rad)','Hand RMSE (rad)','Predicted palm FK position (mm)','Predicted palm FK angle (deg)','Frozen validation selection score')
        for ax,key,title in zip(axes.flat,fields,titles):
            for cohort in ('fixed_train2','fixed_validation4'):
                rr=[r for r in flattened if r['cohort']==cohort]
                ax.plot([r['anchor_epochs'] for r in rr],[r[key] for r in rr],label=cohort)
            ax.set_title(title);ax.set_xlabel('Anchor-equivalent epochs');ax.legend();ax.grid(alpha=.3)
        losses=[json.loads(line) for line in (root/'training.jsonl').read_text().splitlines()]
        with (root/'training_curves.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['update','anchor_epochs','base','position_m2','rotation','lambda_pos','lambda_rot','geometry_gain',
                'hand_geometry','lambda_hand','hand_geometry_gain','hand_geometry_seconds'])
            writer.writeheader()
            for r in losses:writer.writerow(dict(update=r['update'],anchor_epochs=r['anchor_equivalent_epochs'],base=r['components']['base'],
                position_m2=r['components']['position'],rotation=r['components']['rotation'],lambda_pos=r['loss_weights']['position'],
                lambda_rot=r['loss_weights']['rotation'],geometry_gain=r['geometry_gain'],hand_geometry=r['components'].get('hand_geometry'),
                lambda_hand=r['loss_weights'].get('hand_geometry'),hand_geometry_gain=r.get('hand_geometry_gain'),
                hand_geometry_seconds=r.get('hand_geometry_seconds')))
        ax=axes[1,2]
        for field in ('base','position','rotation')+(('hand_geometry',) if any('hand_geometry' in r['components'] for r in losses) else ()):
            # Plot each validation interval's training mean; raw every-update
            # components and effective warm-up weights remain in CSV/JSONL.
            points=[];previous=0
            for row in rows:
                part=[r for r in losses if previous<r['update']<=row['update']];previous=row['update']
                points.append(sum(r['components'][field]*(1 if field=='base' else r['loss_weights'][field]) for r in part)/max(1,len(part)))
            ax.plot([r['anchor_equivalent_epochs'] for r in rows],points,label=field)
        ax.set_title('Training loss: base and weighted auxiliary terms');ax.set_xlabel('Anchor-equivalent epochs');ax.legend();ax.grid(alpha=.3)
        fig.tight_layout();fig.savefig(root/'curves.png',dpi=150);plt.close(fig)
        return
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for key,label in [('train8','train8'),('metrics','validation')]:
        axes[0].plot([r['update'] for r in rows],[r[key]['selection_mse'] for r in rows],label=label)
        axes[1].plot([r['update'] for r in rows],[r[key]['arm_rmse'] for r in rows],label=label)
    for ax,title in zip(axes,['Equal arm/hand MSE','Arm RMSE (rad)']):ax.set_title(title);ax.set_xlabel('Fresh optimizer updates');ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(root/'curves.png',dpi=150);plt.close(fig)


def residual_setup(root, *, selected=False, smoke=False):
    from .models.native_dp import TargetResidual
    from .state_resources import lease_for
    s=read(root/'session.json');cfg=read(root/'configuration.json');v=cfg['residual_pair']
    if v['version']!=TargetResidual.version:raise ValueError('Residual architecture/configuration identity differs')
    data=read(root/'state_data/data.json');index=read(root/'nominal/index.json')
    if smoke:data=dict(data,splits={k:values[:1] for k,values in data['splits'].items()})
    base=v['bases'][root.name]
    if sha256(base['checkpoint'])!=base['sha256']:raise ValueError('Frozen base changed')
    if index['base_sha256']!=base['sha256'] or index['data_sha256']!=sha256(root/'state_data/data.json'):
        raise ValueError('Nominal cache identity mismatch')
    residual_statistics_path=Path(v['residual_statistics_path'])
    residual_normalization_identity='u_expert_train166_affine_v1'
    if not residual_statistics_path.is_absolute():raise ValueError('Residual statistics path must be absolute')
    residual_data=read(residual_statistics_path.parent/'data.json')
    if residual_data.get('normalization_identity')!=residual_normalization_identity:
        raise ValueError('New residual requires common u_expert training affine')
    expected_train=read(root/'state_data/data.json')['splits']['train']
    if residual_data['splits']['train']!=expected_train:raise ValueError('Residual normalization training population differs')
    residual_statistics_sha256=sha256(residual_statistics_path)
    amplitudes=read(root/'nominal/scales.json')
    for artifact in (index,amplitudes):
        if artifact.get('residual_statistics_sha256')!=residual_statistics_sha256 or artifact.get('residual_normalization_identity')!=residual_normalization_identity:
            raise ValueError('Residual cache/scales normalization identity differs')
    records={}
    for identity in data['splits']['train']+data['splits']['validation']:
        path=root/'nominal'/(identity+'.pt')
        if sha256(path)!=index['episodes'][identity]['sha256']:raise ValueError('Nominal cache drift')
        records[identity]=torch.load(path,map_location='cpu',weights_only=False)
    stats=torch.load(residual_statistics_path,map_location='cpu',weights_only=False)
    base_stats=torch.load(root/'state_data/statistics.pt',map_location='cpu',weights_only=False)
    for key in ('robot_mean','robot_std'):
        if not torch.equal(stats[key],base_stats[key]):raise ValueError('Residual/base robot normalization differs')
    scale=amplitudes['scales']
    torch.manual_seed(v['initialization_seed']);model=TargetResidual(stats,scale)
    configuration=dict(v,group=root.name,base_checkpoint=base['checkpoint'],base_sha256=base['sha256'],
        statistics_sha256=sha256(root/'state_data/statistics.pt'),data_sha256=sha256(root/'state_data/data.json'),
        residual_statistics_sha256=residual_statistics_sha256,residual_normalization_identity=residual_normalization_identity,
        dino_sha256=s['dino_sha256'],cache_index_sha256=sha256(root/'nominal/index.json'),
        loss_id='all_tick_equal_arm_hand_plus_1e-4_residual',ema=False,
        architecture=model.architecture(),residual_version=model.version,
        history='full causal episodes; 32tick detach only; expert past acknowledgements teacher forced offline')
    if selected:
        saved=torch.load(root/('training_smoke' if smoke else 'training')/'best.pt',map_location='cpu',weights_only=False)
        if saved.get('disposable',False) and not smoke:raise ValueError('Disposable residual cannot become a selected formal checkpoint')
        if saved['configuration']!=configuration:raise ValueError('Selected residual configuration differs')
        model.load_state_dict(saved['model'])
    lease=lease_for(Path(s['resource_root']),deadline=s['offline_deadline'] if selected and not smoke else s['training_deadline'])
    with lease.block():model.cuda()
    return s,v,data,records,model,configuration,lease


def residual_aux(records,start,stop,key,device):
    values=[]
    for r in records:
        value=r[key];padded=value.new_zeros(stop-start,*value.shape[1:]);n=max(0,min(stop,len(value))-start)
        if n:padded[:n]=value[start:start+n]
        values.append(padded)
    return torch.stack(values).to(device)


@torch.no_grad()
def residual_prediction(model,record,lease,*,mask_force=False):
    from .native_data import residual_batch
    model.eval();state=None;predictions=[];device=next(model.parameters()).device
    for start in range(0,len(record['robot']),32):
        stop=min(start+32,len(record['robot']))
        with lease.block():
            inputs=residual_batch([record],start,stop,device)
            raw,state=model(inputs,state,mask_force=mask_force)
            action,_=model.compose(record['base'][None,start:stop].to(device),raw,
                record['gate'][None,start:stop].to(device),record['limits'].to(device))
            if not torch.isfinite(action).all():raise FloatingPointError('Nonfinite residual prediction')
            predictions.append(action[0].cpu())
    return torch.cat(predictions)


@torch.no_grad()
def residual_metrics(model,records,ids,lease,*,mask_force=False):
    predictions=[residual_prediction(model,records[i],lease,mask_force=mask_force) for i in ids]
    error=torch.cat([p-records[i]['target'] for i,p in zip(ids,predictions)])
    gate=torch.cat([records[i]['gate'] for i in ids]);normalized=error*model.action_scale.cpu()
    def metrics(mask):
        e=error[mask];z=normalized[mask]
        return dict(ticks=len(e),arm_rmse_rad=float(e[:,:7].square().mean().sqrt()) if len(e) else None,
            hand_rmse_rad=float(e[:,7:].square().mean().sqrt()) if len(e) else None,
            normalized_equal_mse=float(.5*(z[:,:7].square().mean()+z[:,7:].square().mean())) if len(e) else None)
    return dict(all=metrics(torch.ones(len(error),dtype=torch.bool)),contact=metrics(gate),noncontact=metrics(~gate)),predictions


def residual_train(root, *, smoke=False, probe=False):
    """Full native-tick command supervision, complete histories, no base optimizer."""
    from .native_data import residual_batch
    from .checkpointing import cpu_snapshot
    if smoke and probe:raise ValueError('Choose residual smoke or throughput probe, not both')
    setup_began=time.time()
    s,v,data,records,model,configuration,lease=residual_setup(root,smoke=smoke)
    if smoke or probe:v=dict(v,maximum_control_tick_epochs=1)
    if probe:
        if len(data['splits']['train'])<4:raise ValueError('Residual throughput probe requires four complete train episodes')
        v=dict(v,microbatch=4)
        configuration=dict(configuration,disposable=True,run_kind='throughput_probe',
            probe_train_ids=data['splits']['train'][:4],probe_effective_batch=4,probe_epochs=1)
    disposable=smoke or probe
    run_kind='throughput_probe' if probe else 'smoke' if smoke else 'formal'
    out=root/('training_probe' if probe else 'training_smoke' if smoke else 'training');out.mkdir(exist_ok=True)
    if (out/'last.pt').exists():raise ValueError('No implicit second residual training run')
    atomic_json(out/'configuration.json',configuration)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    device=next(model.parameters()).device
    ids=data['splits']['train'][:4] if probe else data['splits']['train']
    vals=data['splits']['validation'][:4];population=sum(len(records[i]['robot']) for i in ids)
    native_population=sum(len(records[i]['robot']) for i in ids)
    if population<1:raise ValueError('No valid native training ticks')
    target_ticks=v['maximum_control_tick_epochs']*population
    generator=torch.Generator().manual_seed(v['initialization_seed'])
    updates=draws=ticks=contact_ticks=0;best=float('inf');history=[];began=time.time();stop='maximum_control_tick_exposure';request=[]
    setup_wall_s=began-setup_began;training_wall_s=0.
    signal.signal(signal.SIGTERM,lambda *_:request.append('SIGTERM'))
    def snapshot(epoch,next_group):
        return cpu_snapshot(dict(residual_version=model.version,group='I-HistoryDP',contract=s['base_contracts'][root.name],
            configuration=configuration,model=model.state_dict(),optimizer=opt.state_dict(),updates=updates,
            formal_updates=0 if disposable else updates,disposable=disposable,run_kind=run_kind,
            contact_anchors=contact_ticks,supervised_ticks=draws,control_ticks=ticks,epoch=epoch,next_episode_group=next_group,episode_order=order,
            schedule_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            best=best,history=history,train_ids=ids,validation_ids=vals,ema=False))
    def validate(epoch):
        nonlocal best
        validation,_=residual_metrics(model,records,vals,lease)
        train,_=residual_metrics(model,records,ids[:2],lease)
        row=dict(epoch=epoch,update=updates,control_tick_epochs=draws/population,control_epoch=ticks/native_population,
            train=train,validation=validation,wall_s=time.time()-began,wait_s=lease.wait_s)
        history.append(row);atomic_json(out/'validation.json',history)
        score=validation['all']['normalized_equal_mse']
        if score is None or not math.isfinite(score):raise FloatingPointError('Invalid validation score')
        if score<best:
            best=score;save(out/'best.pt',snapshot(epoch,0))
            atomic_json(out/'selected.json',dict(epoch=epoch,update=updates,validation_score=score,
                selection='fixed validation4 all-tick normalized equal Arm/Hand MSE',checkpoint_sha256=sha256(out/'best.pt')))
        residual_curves(out)
    for epoch in range(1,v['maximum_control_tick_epochs']+1):
        epoch_training_began=time.time()
        order=[ids[i] for i in torch.randperm(len(ids),generator=generator).tolist()]
        for offset in range(0,len(order),v['microbatch']):
            selected=[records[i] for i in order[offset:offset+v['microbatch']]];state=None
            for start in range(0,max(len(r['robot']) for r in selected),32):
                end=min(start+32,max(len(r['robot']) for r in selected))
                with lease.block():
                    model.train();inputs=residual_batch(selected,start,end,device)
                    raw,state=model(inputs,state)
                    gate=residual_aux(selected,start,end,'gate',device)&inputs['valid']
                    action,delta=model.compose(residual_aux(selected,start,end,'base',device),raw,gate,
                        torch.stack([r['limits'] for r in selected]).to(device)[:,None])
                    loss=model.objective(action,residual_aux(selected,start,end,'target',device),delta,inputs['valid'])
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite residual loss')
                    if inputs['valid'].any():
                        opt.zero_grad(set_to_none=True);loss.backward()
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite residual gradient')
                        opt.step();updates+=1;draws+=int(inputs['valid'].sum());contact_ticks+=int(gate.sum())
                        with (out/'loss.jsonl').open('a') as f:f.write(json.dumps(dict(update=updates,epoch=epoch,
                            loss=float(loss.detach()),contact_ticks=int(gate.sum()),supervised_ticks=int(inputs['valid'].sum()),gradient_norm=float(norm)))+'\n')
                    ticks+=int(inputs['valid'].sum())
                    def detach(x):return {k:detach(z) for k,z in x.items()} if isinstance(x,dict) else x.detach()
                    state=detach(state)
                if updates%32==0:
                    elapsed=time.time()-began;rate=draws/max(elapsed,1e-6)
                    atomic_json(out/'progress.json',dict(update=updates,epoch=epoch,control_tick_epochs=draws/population,
                        sampled_anchors=draws,anchor_population=population,control_ticks=ticks,wall_s=elapsed,
                        wait_s=lease.wait_s,work_s=lease.work_s,remaining_seconds_estimate=(v['maximum_control_tick_epochs']*population-draws)/max(rate,1e-6),
                        utc=time.time()))
            if request or time.time()>s['training_deadline']-180:
                stop='requested_stop' if request else 'fixed_wall_budget';break
        training_wall_s+=time.time()-epoch_training_began
        save(out/'last.pt',snapshot(epoch,offset+len(selected)))
        if epoch==1 or epoch%v['validation_epoch_interval']==0 or stop!='maximum_control_tick_exposure':validate(epoch)
        if not disposable and epoch in v.get('comparison_control_tick_epochs',[]):
            save(out/f'comparison_epoch_{epoch}.pt',snapshot(epoch,offset+len(selected)))
        if stop!='maximum_control_tick_exposure':break
        # Minimum exposure and a fixed validation trend define bounded stopping;
        # maximum exposure is a ceiling, not an admission condition.
        if epoch>=v['minimum_control_tick_epochs'] and len(history)>=4:
            scores=[r['validation']['all']['normalized_equal_mse'] for r in history[-4:]]
            if (scores[0]-min(scores[1:]))/max(scores[0],1e-12)<.01:stop='validation_plateau_after_minimum_exposure';break
    if not history or history[-1]['update']!=updates:validate(epoch)
    save(out/'last.pt',snapshot(epoch,offset+len(selected)))
    atomic_json(out/'finished.json',dict(stop_reason=stop,updates=updates,control_tick_epochs=draws/population,
        formal_updates=0 if disposable else updates,disposable=disposable,run_kind=run_kind,
        target_reached=draws>=target_ticks and ticks>=target_ticks,required_supervised_ticks=target_ticks,
        required_control_tick_epochs=v['maximum_control_tick_epochs'],train_episode_count=len(ids),effective_batch=v['microbatch'],
        minimum_control_tick_epochs=v['minimum_control_tick_epochs'],minimum_exposure_reached=draws>=v['minimum_control_tick_epochs']*population,
        sampled_contact_anchors=contact_ticks,contact_anchor_population=sum(int(records[i]['gate'].sum()) for i in ids),
        control_tick_population=population,supervised_ticks=draws,control_ticks=ticks,control_equivalent_epochs=ticks/native_population,
        wall_s=time.time()-began,supervised_ticks_per_wall_s=draws/max(time.time()-began,1e-6),
        setup_wall_s=setup_wall_s,training_wall_s=training_wall_s,
        supervised_ticks_per_training_wall_s=draws/max(training_wall_s,1e-6),
        timing_scope='wall_s includes training, resource waits, validation, plots and checkpoints after setup; training_wall_s includes training and its resource waits only; setup_wall_s includes model/cache setup',
        wait_s=lease.wait_s,work_s=lease.work_s,selection='fixed validation4 all-tick normalized equal Arm/Hand MSE',
        last=str(out/'last.pt'),best=str(out/'best.pt'),utc=time.time(),base_optimizer_updates=0))


def residual_curves(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read(out/'validation.json');fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for split in ('train','validation'):
        for ax,key in zip(axes,('normalized_equal_mse','arm_rmse_rad','hand_rmse_rad')):
            ax.plot([r['control_tick_epochs'] for r in rows],[r[split]['all'][key] for r in rows],label=split)
            ax.set(xlabel='Control tick equivalent epochs',ylabel=key);ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=140);plt.close(fig)


def residual_offline(root, *, smoke=False):
    import h5py
    import numpy as np
    from .state_image_loss import GeometryObjective
    from .state_image_offline import diagnostic_rows,summarize_diagnostics
    from tools.narrow_repair_audit import stages
    from tools.tight_diagnostics import table,target_plot
    s,v,data,records,model,configuration,lease=residual_setup(root,selected=True,smoke=smoke)
    out=root/('offline_smoke' if smoke else 'offline');out.mkdir(exist_ok=False);rows=[];base_rows=[];sensitivity=[];completed=0
    ids={'train':data['splits']['train'][:8],'validation':data['splits']['validation']}
    with h5py.File(data['entries'][ids['train'][0]]['path']) as f:objective=GeometryObjective(json.loads(f.attrs['metadata_json']))
    @torch.no_grad()
    def measured(prediction,record):
        truth=record['target'];hold=record['robot'][torch.arange(len(truth))//12*12,58:87]
        with lease.block():
            p,r=objective.fk(prediction.cuda());q,t=objective.fk(truth.cuda());distance=(p-q).norm(dim=-1).cpu()*1000
            rot=r[:,0].transpose(-1,-2)@t[:,0]
            sine=torch.stack((rot[:,2,1]-rot[:,1,2],rot[:,0,2]-rot[:,2,0],rot[:,1,0]-rot[:,0,1]),-1).norm(dim=-1)/2
            angle=torch.atan2(sine,(rot.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).cpu()*180/torch.pi
        return dict(prediction=prediction,truth=truth,hold=hold,ee_mm=distance[:,0],finger_mm=distance[:,1:],rotation_deg=angle)
    for split,identities in ids.items():
        for n,identity in enumerate(identities):
            record=records[identity];pred=residual_prediction(model,record,lease);sample=measured(pred,record)
            with h5py.File(data['entries'][identity]['path']) as f:
                g=f['episodes'][data['entries'][identity]['episode']];phase=stages(g)
                contact=np.linalg.norm(g['transition/object_contact_force_w'][:],axis=-1).reshape(len(phase),-1).max(-1)>.1
                lift=g['transition/diagnostics_before/lifted_object'][:].reshape(len(phase),-1).any(-1)
            rows.extend(diagnostic_rows(sample,phase,identity=identity,split=split))
            base_rows.extend(diagnostic_rows(measured(record['base'],record),phase,identity=identity,split=split))
            if n==0:target_plot(out,identity,sample,phase,contact,lift)
            if split=='validation' and n<4:
                masked=residual_prediction(model,record,lease,mask_force=True);gate=record['gate']
                e=pred-record['target'];me=masked-record['target'];change=pred-masked
                sensitivity.append(dict(identity=identity,contact_ticks=int(gate.sum()),
                    force_zero_output_rms_rad=float(change[gate].square().mean().sqrt()),
                    force_zero_output_max_abs_rad=float(change.abs().max()),
                    normal_contact_equal_mse_rad2=float(.5*(e[gate,:7].square().mean()+e[gate,7:].square().mean())),
                    zero_force_contact_equal_mse_rad2=float(.5*(me[gate,:7].square().mean()+me[gate,7:].square().mean())),
                    fixed_nominal=True,fixed_gate=True,test_used=False))
            completed+=1;atomic_json(out/'progress.json',dict(offline_episodes=completed,utc=time.time()))
    metrics=summarize_diagnostics(rows);base=summarize_diagnostics(base_rows)
    table(out/'sampled_ticks.csv',rows);table(out/'base_sampled_ticks.csv',base_rows)
    table(out/'stage_errors.csv',metrics);table(out/'base_stage_errors.csv',base)
    import csv
    import matplotlib.pyplot as plt
    with (out/'stage_errors.csv').open() as f:draw=list(csv.DictReader(f))
    with (out/'base_stage_errors.csv').open() as f:reference=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,part in zip(axes,('arm','hand')):
        rr=[x for x in draw if x['split']=='validation' and x['group']==part and x['motion']=='all' and x['rmse_rad']]
        bb=[x for x in reference if x['split']=='validation' and x['group']==part and x['motion']=='all' and x['rmse_rad']]
        ax.plot(range(len(rr)),[float(x['rmse_rad']) for x in rr],label='60Hz residual')
        ax.plot(range(len(bb)),[float(x['rmse_rad']) for x in bb],label='frozen base')
        ax.set_xticks(range(len(rr)),[x['phase']+' n='+x['ticks'] for x in rr],rotation=65,ha='right',fontsize=7)
        ax.set(ylabel=part+' RMSE (rad)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'stage_errors.png',dpi=140);plt.close(fig)
    atomic_json(out/'force_sensitivity.json',dict(rows=sensitivity,scope='Fixed validation4; only residual force world/local set to zero; gate, FK and nominal held; sensitivity is not causal task benefit'))
    atomic_json(out/'result.json',dict(complete=True,metrics=metrics,base_metrics=base,force_sensitivity=sensitivity,
        checkpoint=str(root/('training_smoke' if smoke else 'training')/'best.pt'),checkpoint_sha256=sha256(root/('training_smoke' if smoke else 'training')/'best.pt'),
        scope='Expert observation/past-command history; complete frozen DDIM16 plans, each native tick residual; no autonomous action-label comparison',test_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['contract','train','fresh-smoke','fresh-train','residual-probe','residual-train','residual-offline']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--smoke-device',choices=['cpu','cuda:0'],default='cpu')
    p.add_argument('--job',choices=['state','image'],required=True);p.add_argument('--target',type=int);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.mode.startswith('residual-'):
        if a.mode=='residual-probe':residual_train(a.root,probe=True)
        else:(residual_train if a.mode=='residual-train' else residual_offline)(a.root)
    elif a.mode.startswith('fresh-'):
        if a.job!='state' or a.data is None:p.error('Fresh entry requires --job state and --data')
        fresh(a.root,a.data,a.output,a.target,smoke=a.mode=='fresh-smoke',smoke_device=a.smoke_device)
    elif a.mode=='contract':contract_check(a.root,a.job,a.output)
    else:train(a.root,a.job,a.target,a.output)
