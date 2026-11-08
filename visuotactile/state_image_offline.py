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
from .state_history import PersistentState, PersistentImage, HistoryBank
from .anchor_sampling import labels
from .state_image_loss import GeometryObjective, COMPONENTS, LOSS_ID, calibrated_weights
from .state_resources import lease_for
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
        from .native_data import depth_grid
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


@torch.no_grad()
def deployment_sample(model, record, objective, lease, *, seed=7091, progress=lambda **kw: None, full_horizon=False):
    """Expert-history diagnostic, with the deployment batch=1 RNG call order.

    Consume each native observation exactly once. HistoryBank is also used by
    HistoryController; labels/events never enter it. The older evaluate() keeps
    its episode-batched noise protocol for comparable historical curves.
    """
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    bank = HistoryBank()
    plans, holds, full_plans, full_truth, full_holds, full_ticks, full_anchors = [], [], [], [], [], [], []
    n = len(record['time'])
    for start in range(0, n, 12):
        with lease.block():
            # The previous chunk left history at start-1, just like deployment.
            condition = bank.observe(0, 1, start, model, record['actor'][start],
                record['robot'][start], record['geometry'][start], record['time'][start])
            raw = model.predict(condition, generator, inference_steps=16, clip_sample=False)
            prediction = bounded_targets(raw[0], record['joint_limits'])
            count = min(12, n-start)
            plans.append(prediction[:count].cpu())
            holds.append(record['robot'][start, 58:87].cpu().expand(count, -1))
            if full_horizon:
                extent = min(model.horizon, n-start)
                full_plans.append(prediction[:extent].cpu())
                full_truth.append(record['command'][start:start+extent].cpu())
                full_holds.append(record['robot'][start, 58:87].cpu().expand(extent, -1))
                full_ticks.extend(range(start, start+extent));full_anchors.extend([start]*extent)
            for tick in range(start+1, start+count):
                assert bank.observe(0, 1, tick, model, record['actor'][tick],
                    record['robot'][tick], record['geometry'][tick], record['time'][tick]) is None
        progress(native_ticks=min(start+12, n), sampled_anchors=len(plans))
    prediction, hold = torch.cat(plans), torch.cat(holds)
    truth = record['command'].cpu()
    def measured(prediction, truth, hold):
        with lease.block():
            p, r = objective.fk(prediction.to(device))
            q, s = objective.fk(truth.to(device))
            distance = (p-q).norm(dim=-1).cpu()*1000
            rotation = r[:, 0].transpose(-1, -2)@s[:, 0]
            sine = torch.stack((rotation[:, 2, 1]-rotation[:, 1, 2],
                rotation[:, 0, 2]-rotation[:, 2, 0], rotation[:, 1, 0]-rotation[:, 0, 1]), -1).norm(dim=-1)/2
            angle = torch.atan2(sine, (rotation.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).cpu()*180/torch.pi
        return dict(prediction=prediction, truth=truth, hold=hold,
            ee_mm=distance[:, 0], finger_mm=distance[:, 1:], rotation_deg=angle)
    result=measured(prediction,truth,hold)
    if full_horizon:
        result['full_horizon']=dict(measured(torch.cat(full_plans),torch.cat(full_truth),torch.cat(full_holds)),
            target_ticks=full_ticks,anchor_ticks=full_anchors)
    return result


def diagnostic_rows(sample, phases, *, identity, split):
    """One measured target row per real tick; empty stages remain unmeasured.

    The .01rad movement threshold is the existing evaluate() diagnostic scale,
    applied separately to arm/hand displacement from the anchor's held target.
    This is a reporting bin, never a success label or training input.
    """
    truth, prediction, hold = (sample[k] for k in ('truth', 'prediction', 'hold'))
    if truth.shape != prediction.shape or truth.shape != hold.shape or len(phases) != len(truth):
        raise ValueError('Diagnostic target/event lengths differ')
    error, hold_error = prediction-truth, hold-truth
    rows = []
    for tick in range(len(truth)):
        real_tick=sample.get('target_ticks',range(len(truth)))[tick]
        anchor=sample['anchor_ticks'][tick] if 'anchor_ticks' in sample else tick//12*12
        row = dict(identity=identity, split=split, tick=real_tick, time_s=real_tick/60,
            anchor=anchor, phase=str(phases[tick]), replan=real_tick==anchor,
            ee_squared_mm2=float(sample['ee_mm'][tick]**2),
            fingertip_squared_mm2=float(sample['finger_mm'][tick].square().mean()),
            ee_rotation_deg=float(sample['rotation_deg'][tick]))
        for name, part in (('arm', slice(0, 7)), ('hand', slice(7, 29))):
            row[name+'_mse_rad2'] = float(error[tick, part].square().mean())
            row['hold_'+name+'_mse_rad2'] = float(hold_error[tick, part].square().mean())
            row[name+'_motion'] = 'moving' if row['hold_'+name+'_mse_rad2'] > .01**2 else 'near_static'
        if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise FloatingPointError('Nonfinite complete-sampling metric')
        rows.append(row)
    return rows


def summarize_diagnostics(rows):
    phases = ('all', 'approach_no_object_contact', 'contact_low_object',
        'transport_above_3cm', 'native_alignment', 'second_insertion', 'ordered_complete_retract')
    result = []
    for split in sorted({r['split'] for r in rows}):
        for phase in phases:
            selected = [r for r in rows if r['split'] == split and (phase == 'all' or r['phase'] == phase)]
            for group in ('arm', 'hand'):
                for motion in ('all', 'moving', 'near_static'):
                    part = [r for r in selected if motion == 'all' or r[group+'_motion'] == motion]
                    result.append(dict(split=split, phase=phase, group=group, motion=motion,
                        ticks=len(part), episodes=len({r['identity'] for r in part}),
                        rmse_rad=float(np.sqrt(np.mean([r[group+'_mse_rad2'] for r in part]))) if part else None,
                        hold_rmse_rad=float(np.sqrt(np.mean([r['hold_'+group+'_mse_rad2'] for r in part]))) if part else None,
                        ee_rmse_mm=float(np.sqrt(np.mean([r['ee_squared_mm2'] for r in part]))) if part else None,
                        fingertip_rmse_mm=float(np.sqrt(np.mean([r['fingertip_squared_mm2'] for r in part]))) if part else None,
                        palm_angle_mean_deg=float(np.mean([r['ee_rotation_deg'] for r in part])) if part else None))
    return result


def sampled_metrics(sample):
    error=sample['prediction']-sample['truth'];hold=sample['hold']-sample['truth']
    arm=float(error[:,:7].square().mean().sqrt());hand=float(error[:,7:].square().mean().sqrt())
    palm=float(sample['ee_mm'].square().mean().sqrt());angle=float(sample['rotation_deg'].mean())
    # Frozen physical scales, equal four terms; never fitted on validation.
    score=.25*((arm/.01)**2+(hand/.1)**2+(palm/10.)**2+(angle/5.)**2)
    return dict(arm_rmse=arm,hand_rmse=hand,selection_mse=.5*(arm**2+hand**2),selection_score=score,
        ee_distance_rmse_mm=palm,ee_rotation_mean_deg=angle,ticks=len(error),
        hold_arm_rmse=float(hold[:,:7].square().mean().sqrt()),hold_hand_rmse=float(hold[:,7:].square().mean().sqrt()))


@torch.no_grad()
def deployment_evaluate(model,records,identities,objective,lease):
    samples=[deployment_sample(model,records[i],objective,lease,full_horizon=True) for i in identities]
    def merged(values):
        return {k:torch.cat([v[k] for v in values]) for k in
                ('prediction','truth','hold','ee_mm','finger_mm','rotation_deg')}
    result=sampled_metrics(merged(samples))
    result['full_horizon']=sampled_metrics(merged([s['full_horizon'] for s in samples]))
    result['identities']=list(identities)
    result['protocol']='complete real prefix, deployment batch1 DDIM16 seed7091; committed12 vs all valid24'
    return result
