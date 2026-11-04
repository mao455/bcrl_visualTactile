"""Versioned Kit-independent State -> Image00 training and offline acceptance."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import h5py
import numpy as np
import torch

from .compact import sha256
from .models.vendor.ema_model import EMAModel
from .position_control import bounded_targets, contract, image_contract
from .state_full_training import restore_training, training_state
from .state_history import PersistentState, PersistentImage, HistoryBank
from .state_history_training import labels
from .state_image_loss import GeometryObjective, COMPONENTS, LOSS_ID, calibrated_weights
from .state_resources import lease_for
from .train_long import save
from .training_resources import atomic_json


def read(path): return json.loads(Path(path).read_text())


def no_kit():
    loaded=sorted(k for k in sys.modules if k.split('.')[0] in ('omni','carb','isaacsim','isaaclab'))
    if loaded: raise RuntimeError('Offline process imported simulator: '+str(loaded[:10]))
    return dict(kit_loaded=False, checked_namespaces=['omni','carb','isaacsim','isaaclab'])


def setup(root, job):
    session=read(root/'session.json'); frozen=read(root/'offline_freeze.json')
    for name,digest in frozen.items():
        if sha256(name)!=digest: raise ValueError('Offline source/input drift: '+name)
    source=Path(session['source_root']); data=read(source/'data.json')
    train_ids=data['splits']['train']; val_ids=data['splits']['validation']
    assert (len(train_ids),len(val_ids),len(data['splits']['test']))==(320,40,40)
    assert not set(train_ids)&set(val_ids) and not (set(train_ids)|set(val_ids))&set(data['splits']['test'])
    cached=torch.load(session['numeric_cache'],map_location='cpu',weights_only=False)
    assert cached['data_sha256']==session['data_sha256']
    cached_contract=cached['contract'];records={i:cached['records'][i] for i in train_ids+val_ids};del cached
    expected=contract(24,True)
    assert dict(cached_contract,horizon=24,prediction_type='sample')==expected
    assert records[train_ids[0]]['command'].shape[1]==29
    stats=torch.load(session['position_statistics'],map_location='cpu',weights_only=False)
    group='S-NativeDP' if job=='state' else 'I-HistoryDP'
    action_contract=expected if job=='state' else image_contract(group,24)
    entry=data['entries'][train_ids[0]]
    with h5py.File(entry['path'],'r') as f:metadata=json.loads(f.attrs['metadata_json'])
    objective=GeometryObjective(metadata)
    if job=='image':
        records={i:dict(**{k:r[k] for k in ('robot','command','time','joint_limits')},identity=i,source_root=str(source)) for i,r in records.items()}
    lease=lease_for(root)
    with lease.block():
        torch.manual_seed(20261102)
        if job=='state':
            model=PersistentState('dp',action_contract)
            expert=torch.load(session['expert_checkpoint'],map_location='cpu',weights_only=False)[0]['model']
            model.set_statistics(stats,expert);del expert
        else:
            model=PersistentImage(group,action_contract)
            for k,v in stats.items():getattr(model,k).copy_(v)
        model.cuda()
        records={i:{k:v.cuda() if torch.is_tensor(v) else v for k,v in r.items()} for i,r in records.items()}
    no_kit()
    return session,data,train_ids,val_ids,records,model,objective,lease,metadata


def batch(model,records,selected):
    conditions=[];commands=[];masks=[]
    for identity,start in selected:
        rec=records[identity];condition,anchors,_=model.encode(rec,start,min(start+96,len(rec['time'])))
        # Image encode returns CPU anchors; use one device for labels and mask.
        anchors=anchors.to(rec['command'].device)
        command,valid=labels(rec,anchors,24)
        conditions.append(condition);commands.append(command);masks.append(valid)
    return torch.cat(conditions),torch.cat(commands),torch.cat(masks)


def audit(root,job,output):
    session,data,train_ids,_,records,model,objective,lease,metadata=setup(root,job)
    model.eval();identity=train_ids[0];rec=records[identity]
    with torch.no_grad(),lease.block():
        full,anchors,_=model.encode(rec,0,96)
        left,_,_=model.encode(rec,0,48);right,_,_=model.encode(rec,48,96)
        prefix_error=float((full-torch.cat((left,right))).abs().max())
        assert prefix_error<2e-5,('prefix',prefix_error)
        if job=='state':
            bank=HistoryBank();online=[]
            for t in range(96):
                c=bank.observe(0,1,t,model,rec['actor'][t],rec['robot'][t],rec['geometry'][t],rec['time'][t])
                if c is not None: online.append(c)
            online_error=float((full-torch.cat(online)).abs().max())
            assert online_error<2e-5,('online',online_error)
        else:
            from .native_data import Episode
            episode=Episode(session['source_root'],identity,'I-HistoryDP');memory=None;online=[]
            for t in range(96):
                memory=model.observe(rec['robot'][t],rec['time'][t],t,episode.frame(t,'cuda:0') if t%12==0 else None,memory)
                if t%12==0:online.append(model.condition(memory,rec['time'][t]))
            online_error=float((full-torch.cat(online)).abs().max());assert online_error<2e-5
    vision=None
    if job=='image':
        from .models.native_vision import NativeDinoSpatial
        from .native_data import Episode
        from .long_data import depth_grid
        with lease.block():dino=NativeDinoSpatial().cuda().eval()
        vision=[]
        for identity in train_ids[:2]:
            cached=Episode(session['source_root'],identity,'I-HistoryDP');raw=Episode(session['source_root'],identity,'I-OfficialDP')
            for tick in sorted(set((0,(len(raw)-1)//12*12))):
                with torch.no_grad(),lease.block():
                    f=raw.frame(tick,'cuda:0');c=cached.frame(tick,'cuda:0')
                    diff=float((dino(f['rgb']).float()-c['patches'].float()).abs().max())
                    depth=depth_grid(f['depth'].cpu(),f['depth_valid'].cpu()).to('cuda:0')
                    depth_error=float((depth-c['depth']).abs().max())
                    assert diff<=.005 and depth_error<=1e-6,(diff,depth_error)
                    assert (f['time']==c['time']).all() and (f['time']<=records[identity]['time'][tick]+1e-6).all()
                    vision.append(dict(identity=identity,tick=tick,patch_max_abs=diff,depth_max_abs=depth_error))
        del dino;torch.cuda.empty_cache()
    model.train();rows=[]
    for identity in train_ids[:4]:
        with lease.block():
            condition,command,valid=batch(model,records,[(identity,0)])
            _,parts,prediction=objective.forward(model,condition,command,valid,torch.Generator(device='cuda').manual_seed(6193))
            grads={k:float(torch.autograd.grad(v,prediction,retain_graph=True)[0].norm()) for k,v in parts.items()}
            rows.append(dict(identity=identity,loss={k:float(v.detach()) for k,v in parts.items()},gradient=grads))
        del condition,command,valid,parts,prediction
    objective.weights=calibrated_weights(rows)
    atomic_json(output/'calibration.json',dict(loss_id=LOSS_ID,weights=objective.weights,rows=rows,
        rule='worst first4 train segments; each auxiliary <=5% base loss and x0-output gradient norm; cap1',
        units=dict(position='squared base-frame distance / (0.1m)^2',rotation='sin(theta/2)^2',
            fingertip='mean five link-frame squared distance / (0.05m)^2',temporal='normalized demonstrated adjacent difference; equal Arm/Hand'),
        temporal_limit='within chunk only; no claimed cross-replan consistency or controller smoothing'))
    # Disposable short-prefix overfit. Formal worker is freshly seeded afterwards.
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6);curve=[]
    for step in range(12):
        with lease.block():
            opt.zero_grad(set_to_none=True)
            c,y,m=batch(model,records,[(train_ids[0],0)])
            loss,_,_=objective.forward(model,c,y,m,torch.Generator(device='cuda').manual_seed(6193))
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm) and torch.isfinite(loss)
            curve.append(float(loss.detach()));opt.step()
    assert curve[-1]<curve[0],('short-prefix fit did not improve',curve)
    atomic_json(output/'result.json',dict(passed=True,job=job,kit_dependency=no_kit(),prefix_max_abs=prefix_error,
        online_max_abs=online_error,vision_cache_checks=vision,small_fit_curve=curve,disposable_updates=12,
        formal_updates=0,train_only_calibration=True,calibration_sha256=sha256(output/'calibration.json')))


@torch.no_grad()
def evaluate(model,records,identities,objective,lease):
    model.eval();errors=[];xyz_errors=[];angles=[];fingers=[];motion=[];continuity=[];rows=[]
    from .state_motion import plan_metrics
    for identity in identities:
        rec=records[identity]
        with lease.block():
            cond,anchors,_=model.encode(rec);anchors=anchors.to(rec['command'].device)
            raw=model.predict(cond,torch.Generator(device='cuda').manual_seed(7091))
            pred=bounded_targets(raw.flatten(0,1),rec['joint_limits']).reshape_as(raw)
            truth,valid=labels(rec,anchors,24);mask=valid[:,:12]
            err=(pred[:,:12]-truth[:,:12])[mask];errors.append(err.cpu())
            p,r=objective.fk(pred[:,:12][mask]);q,s=objective.fk(truth[:,:12][mask])
            xyz=(p[:,0]-q[:,0])*1000;xyz_errors.append(xyz.cpu())
            rel=r[:,0].transpose(-1,-2)@s[:,0]
            sine=torch.stack((rel[:,2,1]-rel[:,1,2],rel[:,0,2]-rel[:,2,0],rel[:,1,0]-rel[:,0,1]),-1).norm(dim=-1)/2
            theta=torch.atan2(sine,(rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2)*180/torch.pi;angles.append(theta.cpu())
            fingers.append((p[:,1:]-q[:,1:]).norm(dim=-1).cpu()*1000)
            before=rec['robot'][anchors,58:87];a=pred[:,11,:7]-before[:,:7];b=truth[:,11,:7]-before[:,:7]
            moving=valid[:,:12].all(1)&(b.square().mean(-1).sqrt()>.01)
            if moving.any():
                an=a[moving].norm(dim=-1);bn=b[moving].norm(dim=-1)
                motion.append(torch.stack(((a[moving]*b[moving]).sum(-1)/(an*bn).clamp_min(1e-12),an/bn),-1).cpu())
            continuity.append(plan_metrics(pred.cpu().numpy(),valid.cpu().numpy()))
            rows.append(dict(identity=identity,arm_rmse=float(err[:,:7].square().mean().sqrt()),hand_rmse=float(err[:,7:].square().mean().sqrt())))
    e=torch.cat(errors);xyz=torch.cat(xyz_errors);angle=torch.cat(angles);finger=torch.cat(fingers)
    return dict(scope='teacher-forced observations; executed first12 of each H24 plan; not autonomous success',
        episodes=rows,arm_rmse=float(e[:,:7].square().mean().sqrt()),hand_rmse=float(e[:,7:].square().mean().sqrt()),
        selection_mse=float(.5*(e[:,:7].square().mean()+e[:,7:].square().mean())),
        joint_rmse_rad=e.square().mean(0).sqrt().tolist(),joint_mae_rad=e.abs().mean(0).tolist(),joint_p95_rad=torch.quantile(e.abs(),.95,dim=0).tolist(),
        ee_xyz_rmse_mm=xyz.square().mean(0).sqrt().tolist(),ee_distance_rmse_mm=float(xyz.square().sum(-1).mean().sqrt()),
        ee_rotation_mean_deg=float(angle.mean()),ee_rotation_p95_deg=float(angle.quantile(.95)),
        fingertip_rmse_mm=finger.square().mean(0).sqrt().tolist(),motion_mean_cosine_amplitude=torch.cat(motion).mean(0).tolist() if motion else None,
        continuity={k:float(np.mean([r[k] for r in continuity if r.get(k) is not None])) for k in continuity[0] if any(r.get(k) is not None for r in continuity)})


def train(root,job,target,output):
    gate=root/(job+'_audit');assert read(gate/'result.json')['passed']
    assert read(gate/'result.json')['calibration_sha256']==sha256(gate/'calibration.json')
    session,data,train_ids,val_ids,records,model,objective,lease,metadata=setup(root,job)
    directory=root/(job+'_train');directory.mkdir(exist_ok=True)
    objective.weights=read(gate/'calibration.json')['weights']
    config=dict(identity=session['run_id']+'/'+job,loss_id=LOSS_ID,contract=model.contract,train_ids=train_ids,validation_ids=val_ids,
        seed=20261102,maximum_updates=10000,optimizer_horizon=10000,batch_segments=2,segment_ticks=96,lr=1e-4,warmup=500,
        betas=[.95,.999],weight_decay=1e-6,loss_weights=objective.weights,calibration_sha256=sha256(gate/'calibration.json'),
        statistics_sha256=sha256(session['position_statistics']),offline_freeze_sha256=sha256(root/'offline_freeze.json'))
    config_path=directory/'configuration.json'
    if config_path.exists():assert read(config_path)==config
    else:atomic_json(config_path,config)
    with lease.block():
        averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
    generator=torch.Generator().manual_seed(20261102);noise=torch.Generator(device='cuda').manual_seed(20261102)
    schedule=[(i,s) for i in train_ids for s in range(0,len(records[i]['time']),96)]
    assert len(schedule)==832
    order=[];cursor=update=0;history=[];best=math.inf;seen=set()
    if (directory/'last.pt').exists():
        saved=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
        assert saved['config_sha256']==sha256(config_path) and saved['loss_id']==LOSS_ID
        restore_training(saved,model,averaged,optimizer,ema,generator,noise)
        update=saved['updates'];cursor=saved['schedule_cursor'];order=saved['schedule_order'];history=saved['history'];best=saved['best'];seen=set(saved['seen_ids'])
        assert update<=target
        assert torch.equal(generator.get_state(),saved['schedule_rng']) and torch.equal(noise.get_state(),saved['noise_rng'])
        assert ema.optimization_step==saved['ema_optimization_step']
        assert all(torch.equal(v.detach().cpu(),saved['model'][k]) for k,v in model.state_dict().items())
        atomic_json(output/'resume.json',dict(from_updates=update,optimizer_ema_rng_sampler_restored=True,model_tensors_exact=True,checkpoint_sha256=sha256(directory/'last.pt')))
        del saved
    start_update=update;began=time.time();stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    def snapshot():
        return training_state(model,averaged,optimizer,ema,generator,noise,kind='dp',group=model.group,updates=update,
            schedule_order=order,schedule_cursor=cursor,best=best,history=history,seen_ids=sorted(seen),train_ids=train_ids,validation_ids=val_ids,
            config_sha256=sha256(config_path),loss_id=LOSS_ID,configuration=config,job=job)
    while update<target:
        # Check before advancing the sampler, so interruption has exact committed state.
        with lease.block():
            selected=[]
            for _ in range(2):
                if cursor==len(order):order=torch.randperm(len(schedule),generator=generator).tolist();cursor=0
                selected.append(schedule[order[cursor]]);cursor+=1
            model.train();optimizer.zero_grad(set_to_none=True)
            condition,command,valid=batch(model,records,selected)
            loss,parts,_=objective.forward(model,condition,command,valid,noise)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();branch=model.history.weight_hh_l0 if job=='state' else model.native_gru.weight_hh
            history_grad=float(branch.grad.norm());norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm) or not math.isfinite(history_grad):raise FloatingPointError('Nonfinite gradient')
            step=update+1;lr=1e-4*min(1.,step/500)*.5*(1+math.cos(math.pi*max(0,step-500)/9500))
            for g in optimizer.param_groups:g['lr']=lr
            optimizer.step();ema.step(model);update=step;seen.update(i for i,s in selected)
        row=dict(update=update,loss=float(loss.detach()),gradient_norm=float(norm),history_grad=history_grad,lr=lr,
            components={k:float(v.detach()) for k,v in parts.items()},utc=time.time(),elapsed_s=time.time()-began,
            seen_episodes=len(seen),work_s=lease.work_s,wait_s=lease.wait_s,pacing_s=lease.pacing_s,**lease.memory())
        with (directory/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if update%10==0:atomic_json(directory/'progress.json',dict(state='supervised' if update<=200 else 'running',updates=update,target=target,utc=time.time(),pid=os.getpid(),**no_kit()))
        if update%50==0 or stop[0]:save(directory/'last.pt',snapshot())
        if stop[0]:raise InterruptedError('Saved complete optimizer boundary')
    metrics=dict(update=update,train=evaluate(averaged,records,train_ids[:4 if update<=200 else 16],objective,lease),
        validation=evaluate(averaged,records,val_ids[:4] if update<=200 else val_ids,objective,lease))
    history=[r for r in history if r['update']!=update]+[metrics]
    best=min(best,metrics['validation']['selection_mse'])
    save(directory/'last.pt',snapshot())
    milestone=directory/f'milestone_{update}.pt'
    if milestone.exists():raise FileExistsError('Never replace committed milestone')
    os.link(directory/'last.pt',milestone)
    saved=torch.load(milestone,map_location='cpu',weights_only=False)
    assert saved['updates']==update and saved['ema_optimization_step']==ema.optimization_step
    assert saved['optimizer']['state'] and saved['schedule_cursor']==cursor
    checkpoint_sha=sha256(milestone)
    atomic_json(directory/'metrics.json',history)
    atomic_json(output/'result.json',dict(software_complete=True,job=job,updates=update,from_updates=start_update,metrics=metrics,
        checkpoint=str(milestone.resolve()),checkpoint_sha256=checkpoint_sha,checkpoint_readback=True,
        elapsed_s=time.time()-began,kit_dependency=no_kit(),parameters=sum(p.numel() for p in model.parameters()),test_used=False))
    atomic_json(directory/'progress.json',dict(state='completed_stage',updates=update,target=target,pid=None,utc=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','train']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job',choices=['state','image'],required=True);p.add_argument('--target',type=int);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.mode=='audit':audit(a.root,a.job,a.output)
    else:train(a.root,a.job,a.target,a.output)
