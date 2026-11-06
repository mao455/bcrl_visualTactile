"""Field-level CPU and real FP32 controller/training equivalence acceptance.

Historical data may be supplied explicitly as diagnostic fixtures. It is never
promoted into the current formal dataset by this audit.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from tools.runtime import read,write,sha
from visuotactile.tight_gate import require


def fixture(root):
    """Current development-only numerical fixture, never a formal dataset gate."""
    from visuotactile.narrow_data import record
    from visuotactile.native_data import limits_affine
    from visuotactile.position_control import contract
    from visuotactile.checkpointing import save
    out=root/'diagnostic_data';out.mkdir(exist_ok=False);records={};entries={}
    for path in sorted((root/'episodes').glob('*/admission_inputs.json')):
        row=read(path)
        if not row.get('numeric_valid'):continue
        key,meta,rec=record(row['path']);identity=row['identity'];records[identity]=rec
        entries[identity]=dict(path=row['path'],episode=key,length=len(rec['time']),family=row['family'],source_sha256=row['source_sha256'])
    require(len(records)>=5,'At least5 real diagnostic trajectories for a full and short microbatch')
    stats={}
    for k in ('robot','geometry'):
        x=torch.cat([r[k][::12] if k=='geometry' else r[k] for r in records.values()])
        stats[k+'_mean']=x.mean(0);stats[k+'_std']=x.std(0).clamp_min(1e-4)
    stats['action_scale'],stats['action_offset']=limits_affine(torch.cat([r['command'] for r in records.values()]))
    write(out/'data.json',dict(contract=contract(24,True),entries=entries,splits=dict(train=list(records),validation=[],test=[]),
        scope='Development-only diagnostic fitting/normalization fixture, no formal training/data admission',formal_dataset=False))
    save(out/'numeric.pt',dict(contract=contract(24,True),data_sha256=sha(out/'data.json'),records=records));save(out/'statistics.pt',stats)
    return out


def cpu(root,data_root):
    from visuotactile.narrow_data import record
    from visuotactile.native_data import limits_affine
    from visuotactile.anchor_sampling import labels,AnchorSampler
    data=read(data_root/'data.json');cache=torch.load(data_root/'numeric.pt',map_location='cpu',weights_only=False)
    require(cache['data_sha256']==sha(data_root/'data.json'),'Cache manifest fingerprint')
    stats=torch.load(data_root/'statistics.pt',map_location='cpu',weights_only=False)
    families={};ticks=0;canonical=None
    for identity,entry in data['entries'].items():
        require(sha(entry['path'])==entry['source_sha256'],'Source HDF5 changed: '+identity)
        key,meta,rec=record(entry['path'])
        require(all(torch.equal(v,cache['records'][identity][k]) for k,v in rec.items()),'HDF5/cache field mismatch')
        order=meta.get('joint_order',meta.get('joint_names'))
        require(order is not None,'Missing canonical joint order')
        if canonical is None:canonical=order
        require(order==canonical,'Inconsistent canonical joint order')
        split=next(k for k,v in data['splits'].items() if identity in v)
        require(meta['dataset_split']==split and meta['initial_family']==entry['family'],'Episode selection provenance')
        require(families.setdefault(entry['family'],split)==split,'Family crosses splits')
        a=torch.arange(0,len(rec['time']),12);y,m=labels(rec,a,24)
        for j,t in enumerate(a.tolist()):
            count=min(24,len(rec['time'])-t)
            require(int(m[j].sum())==count and torch.equal(y[j,:count],rec['command'][t:t+count]) and not torch.count_nonzero(y[j,count:]),'Label/padding mask mismatch')
        ticks+=len(rec['time'])
        write(root/'cpu_progress.json',dict(identity=identity,ticks=ticks,utc=time.time()))
    for key in ('robot','geometry'):
        x=torch.cat([cache['records'][i][key][::12] if key=='geometry' else cache['records'][i][key] for i in data['splits']['train']])
        require(torch.equal(stats[key+'_mean'],x.mean(0)) and torch.equal(stats[key+'_std'],x.std(0).clamp_min(1e-4)),'Train-only statistics '+key)
    scale,offset=limits_affine(torch.cat([cache['records'][i]['command'] for i in data['splits']['train']]))
    require(torch.equal(scale,stats['action_scale']) and torch.equal(offset,stats['action_offset']),'Train-only action statistics')
    sampler=AnchorSampler({i:len(cache['records'][i]['time']) for i in data['splits']['train']},torch.Generator().manual_seed(17),64)
    batches=[sampler.draw() for _ in range(math.ceil(len(sampler.ids)/64))]
    require(len({i for b in batches for i,t in b})==sum(map(len,batches))==len(sampler.ids),'Short tail duplicated/dropped')
    result=dict(passed=True,episodes=len(data['entries']),ticks=ticks,fields=list(next(iter(cache['records'].values()))),
        joint_order=canonical,train_only_statistics=True,family_splits_disjoint=True,batch_sizes=list(map(len,batches)),
        source_data_sha256=sha(data_root/'data.json'),source_data=str(data_root),
        scope='Numerical fixtures only; old RGB-D validity and measured slowing are not upgraded')
    write(root/'cpu_audit.json',result)


def gpu(root,data_root):
    from visuotactile.precision import state_fp32
    from visuotactile.state_resources import lease_for
    from visuotactile.state_history import PersistentState,HistoryController
    from visuotactile.position_control import contract
    from visuotactile.anchor_sampling import AnchorSampler,anchor_batch,mask_totals,weighted_parts,continuation_lr
    from visuotactile.state_image_loss import GeometryObjective
    from visuotactile.models.vendor.ema_model import EMAModel
    from visuotactile.checkpointing import training_state,restore_training,save
    state_fp32();torch.set_num_threads(2);s=read(root/'session.json');lease=lease_for(root,deadline=s['deadline'])
    data=read(data_root/'data.json');cache=torch.load(data_root/'numeric.pt',map_location='cpu',weights_only=False)
    stats=torch.load(data_root/'statistics.pt',map_location='cpu',weights_only=False)
    ids=data['splits']['train'][:7];require(len(ids)>=5,'Full and short microbatch required');records={i:cache['records'][i] for i in ids};del cache
    with h5py.File(data['entries'][ids[0]]['path']) as f:meta=json.loads(f.attrs['metadata_json'])
    objective=GeometryObjective(meta);objective.weights=s['loss_weights']
    torch.manual_seed(s['training']['random_seed'])
    model=PersistentState('dp',contract(24,True))
    expert=torch.load(s['expert_checkpoint'],map_location='cpu',weights_only=False)[0]['model']
    model.set_statistics(stats,expert);del expert
    with lease.block():
        model.cuda();records={i:{k:v.cuda() for k,v in r.items()} for i,r in records.items()}
        averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-6,betas=(.95,.999),weight_decay=1e-6)
        rng=torch.Generator().manual_seed(37);noise=torch.Generator(device='cuda').manual_seed(38)
    sampler=AnchorSampler({i:len(records[i]['time']) for i in ids},rng,64)
    selected=[(i,(len(records[i]['time'])-1)//12*12 if j%2 else 12) for j,i in enumerate(ids)]
    totals=mask_totals(sampler.lengths,selected);captured=[]
    original=training_state(model,averaged,optimizer,ema,rng,noise,kind='dp',updates=0)
    def hook(module,args,kwargs):captured.append((args[0].detach().clone(),args[1].detach().clone()))
    handle=model.denoiser.register_forward_pre_hook(hook,with_kwargs=True)
    model.train();optimizer.zero_grad(set_to_none=True);micro_loss=0.
    for j in range(0,len(selected),4):
        with lease.block():
            c,y,m=anchor_batch(model,records,selected[j:j+4]);_,parts,_=objective.forward(model,c,y,m,noise)
            loss,_=weighted_parts(parts,m,totals,objective.weights);loss.backward();micro_loss+=float(loss)
    handle.remove()
    with lease.block():
        grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();ema.step(model)
        micro=training_state(model,averaged,optimizer,ema,rng,noise,kind='dp',updates=1)
        restore_training(original,model,averaged,optimizer,ema,rng,noise);optimizer.zero_grad(set_to_none=True)
        c,y,m=anchor_batch(model,records,selected)
        pred=model.denoiser(torch.cat([x[0] for x in captured]),torch.cat([x[1] for x in captured]),global_cond=c)
        parts=objective.components(pred,y,m,model.action_scale,model.action_offset)
        loss=parts['base']+sum(objective.weights[k]*parts[k] for k in objective.weights);loss.backward()
        denom=sum(float(g.double().square().sum()) for g in grads.values())
        delta=sum(float((p.grad.detach().cpu().double()-grads[n].double()).square().sum()) for n,p in model.named_parameters() if p.grad is not None)
        relative=math.sqrt(delta/max(denom,1e-30))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();ema.step(model)
        max_param=max(float((v.cpu()-micro['model'][k]).abs().max()) for k,v in model.state_dict().items())
        max_ema=max(float((v.cpu()-micro['ema'][k]).abs().max()) for k,v in averaged.state_dict().items())
    result=dict(gradient_relative_l2=relative,loss_relative_error=abs(micro_loss-float(loss))/abs(micro_loss),
        parameter_update_max_abs=max_param,ema_max_abs=max_ema,actual_batch=len(selected),microbatch_sizes=[4,len(selected)-4],
        valid_targets=totals[0],adjacent_pairs=totals[1],formal_updates=0,diagnostic_optimizer_updates=2,
        source_data_sha256=sha(data_root/'data.json'),statistics_sha256=sha(data_root/'statistics.pt'))
    result['passed']=relative<2e-4 and result['loss_relative_error']<2e-5 and max_param<2e-5 and max_ema<2e-5
    write(root/'gradient_gate.json',result);require(result['passed'],'Microbatch gradient/optimizer/EMA mismatch')
    # Persist and restore every tensor, optimizer moment, RNG and sampler state.
    # Exercise a nonempty shuffled deck and partially consumed next deck; an
    # empty sampler state would not validate resume behavior.
    sampler.draw();sampler.draw();sampler.draw()
    snapshot=training_state(model,averaged,optimizer,ema,rng,noise,kind='dp',group=model.group,updates=1,
        configuration=dict(input_builder='canonical_joint_state_v1'),
        anchor_sampler=sampler.state_dict(),scheduler=dict(step=1,total=1000,warmup=100,lr=continuation_lr(1,1000,100)))
    path=root/'audit_checkpoint.pt';save(path,snapshot);loaded=torch.load(path,map_location='cpu',weights_only=False)
    with lease.block():restore_training(loaded,model,averaged,optimizer,ema,rng,noise)
    def equal(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    restored=training_state(model,averaged,optimizer,ema,rng,noise,kind='dp',group=model.group,updates=1,
        configuration=loaded['configuration'],
        anchor_sampler=sampler.state_dict(),scheduler=loaded['scheduler'])
    require(equal(loaded,restored),'Complete optimizer/EMA/RNG/sampler restore mismatch')
    with lease.block():
        expected_draw=sampler.draw();expected_noise=torch.randn((7,24,29),device='cuda',generator=noise)
        expected_cpu=torch.rand(7);expected_cuda=torch.rand(7,device='cuda')
        restore_training(loaded,model,averaged,optimizer,ema,rng,noise);sampler.load_state_dict(copy.deepcopy(loaded['anchor_sampler']))
        checks=dict(next_anchor_deck=sampler.draw()==expected_draw,
            next_diffusion_noise=torch.equal(torch.randn((7,24,29),device='cuda',generator=noise),expected_noise),
            next_torch_random=torch.equal(torch.rand(7),expected_cpu),next_cuda_random=torch.equal(torch.rand(7,device='cuda'),expected_cuda))
        require(all(checks.values()),'Restored next computation changed')
        restore_training(loaded,model,averaged,optimizer,ema,rng,noise);sampler.load_state_dict(copy.deepcopy(loaded['anchor_sampler']))
    write(root/'restore_gate.json',dict(passed=True,sha256=sha(path),diagnostic_only=True,scheduler_next_lr=continuation_lr(2,1000,100),
        nonempty_sampler_draws=loaded['anchor_sampler']['draws'],next_step_checks=checks))
    del original,micro,snapshot,grads,optimizer,ema,model,averaged
    # A complete real trajectory through the actual deployed constructor. Past
    # expert targets are explicit teacher forcing here, never student predictions.
    identity=ids[0];entry=data['entries'][identity];rec=records[identity]
    with h5py.File(entry['path']) as f:
        meta=json.loads(f.attrs['metadata_json']);g=f['episodes'][entry['episode']]
        with lease.block():controller=HistoryController(loaded,meta,inference=dict(inference_steps=16,clip_sample=False,overlap_steps=0,sampler_seed=7091))
        maximum={k:0. for k in ('actor','robot','geometry','normalized_actor','normalized_robot','normalized_geometry','hidden','condition','action')}
        with torch.no_grad(),lease.block():
            model=controller.model;conditions,anchors,final_hidden=model.encode(rec)
            features,_=model.native_features(rec['actor'],rec['robot'],rec['geometry'])
            states,_=model.history(features[None]);gen=torch.Generator(device='cuda').manual_seed(7091)
        for t in range(len(rec['time'])):
            def tree(group):return {k:tree(v) if isinstance(v,h5py.Group) else torch.from_numpy(np.asarray(v[t]))[None] for k,v in group.items()}
            packet=tree(g['obs']);diag=tree(g['transition/diagnostics_before'])
            with torch.no_grad(),lease.block():
                controller.act(rec['actor'][t:t+1].cpu(),packet,diag)
                for key in ('actor','robot','geometry'):
                    observed=controller.last_inputs[key].cuda();expected=rec[key][t]
                    maximum[key]=max(maximum[key],float((observed-expected).abs().max()))
                    if key=='actor':a,b=model.actor_input(observed),model.actor_input(expected)
                    elif key=='robot':a,b=model.normalized_robot(observed),model.normalized_robot(expected)
                    else:a,b=(observed-model.geometry_mean)/model.geometry_std,(expected-model.geometry_mean)/model.geometry_std
                    maximum['normalized_'+key]=max(maximum['normalized_'+key],float((a-b).abs().max()))
                maximum['hidden']=max(maximum['hidden'],float((controller.bank.lanes[0]['hidden'][0,0]-states[0,t]).abs().max()))
                if t%12==0:
                    maximum['condition']=max(maximum['condition'],float((controller.last_condition-conditions[t//12:t//12+1]).abs().max()))
                    predicted=model.predict(conditions[t//12:t//12+1],gen)
                    maximum['action']=max(maximum['action'],float((predicted.cpu()-controller.raw_plan).abs().max()))
                controller.command=rec['command'][t:t+1].cpu()
                controller.observe_execution(dict(position_applied_rad=rec['command'][t:t+1].cpu()))
            write(root/'gpu_progress.json',dict(tick=t,maximum=maximum,utc=time.time()))
    report=dict(passed=all(v<2e-5 for v in maximum.values()),maximum=maximum,ticks=len(rec['time']),
        source=identity,precision='GPU FP32 matmul/CuDNN TF32 off',threshold=2e-5,
        source_data_sha256=sha(data_root/'data.json'),statistics_sha256=sha(data_root/'statistics.pt'),
        history='Recorded expert requested/applied targets, strictly previous tick; explicit teacher forcing',
        scope='Full trajectory real HistoryController vs offline prefix, fixed sequential noise; no simulator replay-success criterion')
    write(root/'gpu_parity.json',report);require(report['passed'],'Full normalized input/controller parity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu','fixture']);p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    if a.mode=='fixture':fixture(a.root.resolve())
    else:
        if a.data is None:p.error('--data required for cpu/gpu')
        (cpu if a.mode=='cpu' else gpu)(a.root.resolve(),a.data.resolve())
