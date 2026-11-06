"""Measured numerical parity and stage/motion diagnostics for the repair run."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from tools.runtime import read, write, sha


def stats(x):
    from visuotactile.state_motion import distribution
    return distribution(x)


@torch.no_grad()
def gpu(root):
    from visuotactile.precision import snapshot, state_fp32
    from visuotactile.state_resources import lease_for
    from visuotactile.state_history import PersistentState, HistoryBank
    from visuotactile.state_image_loss import GeometryObjective
    from visuotactile.anchor_sampling import anchor_batch
    from visuotactile.position_control import bounded_targets
    s=read(root/'session.json');parent=Path(s['parent'])
    lease=lease_for(root,deadline=s['gate_deadline'])
    data=read(parent/'state_data/data.json')
    cached=torch.load(parent/'state_data/numeric.pt',map_location='cpu',weights_only=False)
    saved=torch.load(parent/'training/best.pt',map_location='cpu',weights_only=False)
    ids=[data['splits']['train'][0],data['splits']['validation'][0]]
    with h5py.File(data['entries'][ids[0]]['path']) as f:meta=json.loads(f.attrs['metadata_json'])
    fk=GeometryObjective(meta).fk
    report=dict(checkpoint_sha256=sha(parent/'training/best.pt'),default_precision=snapshot(),
        thresholds=dict(condition_max_abs=2e-5,target_max_abs_rad=2e-5,fk_max_mm=.02),
        identical_noise_seed=7091,optimizer_updates=0,rows=[],passed=False)
    all_predictions={}
    with lease.block():
        model=PersistentState('dp',saved['contract']).cuda().eval();model.load_state_dict(saved['ema'])
        records={i:{k:v.cuda() for k,v in cached['records'][i].items()} for i in ids}
    for mode in ('legacy_default','unified_fp32'):
        if mode=='legacy_default':
            torch.backends.cuda.matmul.allow_tf32=report['default_precision']['matmul_tf32']
            torch.backends.cudnn.allow_tf32=report['default_precision']['cudnn_tf32']
        else:state_fp32()
        for identity in ids:
            r=records[identity];conditions=[];bank=HistoryBank()
            with lease.block():full,anchors,_=model.encode(r)
            for t in range(len(r['time'])):
                with lease.block():
                    c=bank.observe(0,1,t,model,r['actor'][t],r['robot'][t],r['geometry'][t],r['time'][t])
                    if c is not None:conditions.append(c)
            online=torch.cat(conditions)
            chosen=sorted(set([0,len(anchors)-1,*range(0,len(anchors),4)]))
            row=dict(mode=mode,identity=identity,precision=snapshot(),ticks=len(r['time']),
                condition_max_abs=float((full-online).abs().max()),condition_rms=float((full-online).square().mean().sqrt()),
                sampled_anchors=[int(anchors[i]) for i in chosen],predictions=[])
            all_predictions[mode,identity]=[]
            for index in chosen:
                t=int(anchors[index]);pred=[]
                with lease.block():
                    segmented=anchor_batch(model,records,[(identity,t)])[0]
                    model.train();train_condition=anchor_batch(model,records,[(identity,t)])[0];model.eval()
                    for condition in (full[index:index+1],online[index:index+1],segmented):
                        pred.append(model.predict(condition,torch.Generator(device='cuda').manual_seed(7091)))
                    a,b,c=pred;qa=bounded_targets(a[0,:12],r['joint_limits']);qb=bounded_targets(b[0,:12],r['joint_limits'])
                    xa,_=fk(qa);xb,_=fk(qb)
                    item=dict(tick=t,prefix12_raw_max_rad=float((a[:,:12]-b[:,:12]).abs().max()),
                        target_max_rad=float((qa-qb).abs().max()),fk_max_mm=float((xa[:,0]-xb[:,0]).norm(dim=-1).max()*1000),
                        full_vs_training_condition_max=float((full[index:index+1]-segmented).abs().max()),
                        train_vs_eval_condition_max=float((segmented-train_condition).abs().max()),
                        training_vs_online_target_max_rad=float((c[:,:12]-b[:,:12]).abs().max()))
                    row['predictions'].append(item);all_predictions[mode,identity].append(qb.cpu())
                write(root/'heartbeat.json',dict(stage='gpu_parity',mode=mode,identity=identity,tick=t,utc=time.time()))
            # Independent episode reset must reproduce tick0 without truncating its history.
            with lease.block():
                c=bank.observe(0,2,0,model,r['actor'][0],r['robot'][0],r['geometry'][0],r['time'][0])
                row['episode_reset_error']=float((c-online[:1]).abs().max())
            report['rows'].append(row);write(root/'gpu_parity.json',report)
    report['precision_effect']=[]
    for identity in ids:
        a=torch.cat(all_predictions['legacy_default',identity]);b=torch.cat(all_predictions['unified_fp32',identity])
        xa,_=fk(a);xb,_=fk(b)
        report['precision_effect'].append(dict(identity=identity,target_max_abs_rad=float((a-b).abs().max()),
            target_rms_rad=float((a-b).square().mean().sqrt()),fk_max_mm=float((xa[:,0]-xb[:,0]).norm(dim=-1).max()*1000)))
    good=[r for r in report['rows'] if r['mode']=='unified_fp32']
    report['passed']=all(r['condition_max_abs']<2e-5 and r['episode_reset_error']<2e-5 and all(
        p['prefix12_raw_max_rad']<2e-5 and p['target_max_rad']<2e-5 and p['fk_max_mm']<.02 and
        p['full_vs_training_condition_max']<2e-5 and p['train_vs_eval_condition_max']<2e-5 and
        p['training_vs_online_target_max_rad']<2e-5 for p in r['predictions']) for r in good)
    write(root/'gpu_parity.json',report)
    if not report['passed']:raise RuntimeError('Measured GPU parity gate failed; see all actual values in gpu_parity.json')


def stages(g):
    """Diagnostic proxies, not new success labels or guaranteed grasp state."""
    before=g['transition/diagnostics_before'];n=len(before['joint_position_rad'])
    obj=before['object_pose_w'][:];z=obj[:,2]-obj[0,2]
    force=g['transition/object_contact_force_w'][:]
    contact=np.linalg.norm(force,axis=-1).reshape(n,-1).max(-1)>.1
    label=np.full(n,'approach_no_object_contact',dtype=object)
    label[contact]='contact_low_object';label[z>.03]='transport_above_3cm'
    alignment=np.asarray(g['transition/native_alignment_condition'][:]).reshape(n,-1).any(-1)
    label[alignment]='native_alignment'
    sub=np.asarray(before['successes'][:]).reshape(n,-1).max(-1)
    label[sub>=1]='second_insertion';label[sub>=2]='ordered_complete_retract'
    return label


def motion(g):
    from visuotactile.state_motion import target_metrics
    label=stages(g);n=len(label)
    q=g['transition/diagnostics_before/joint_position_rad'][:]
    qd=g['transition/diagnostics_before/joint_velocity_rad_s'][:]
    target=g['transition/target_after_rad'][:];before=g['transition/target_before_rad'][:]
    delta=(target-before)*60
    result=dict(duration_s=n/60,steps=n,target_continuity=target_metrics(target),phases={})
    for phase in sorted(set(label)):
        mask=label==phase
        result['phases'][phase]=dict(seconds=int(mask.sum())/60,ticks=int(mask.sum()),
            arm_actual_speed_rms=float(np.sqrt(np.mean(qd[mask,:7]**2))),
            hand_actual_speed_rms=float(np.sqrt(np.mean(qd[mask,7:]**2))),
            arm_target_speed_rms=float(np.sqrt(np.mean(delta[mask,:7]**2))),
            hand_target_speed_rms=float(np.sqrt(np.mean(delta[mask,7:]**2))),
            arm_tracking_rmse=float(np.sqrt(np.mean((target[mask,:7]-q[mask,:7])**2))),
            target_hold_fraction=float((np.linalg.norm(delta[mask],axis=-1)<1e-3).mean()))
    result['anchor_phase_counts']=dict(Counter(label[::12]));result['tick_phase_counts']=dict(Counter(label))
    result['h24_fraction_of_episode']=24/n
    return result


def dataset(root):
    from visuotactile.narrow_data import record
    from visuotactile.narrow_decision import curve_summary
    s=read(root/'session.json');parent=Path(s['parent']);manifest=read(parent/'manifest.json')
    data=read(parent/'state_data/data.json');cached=torch.load(parent/'state_data/numeric.pt',map_location='cpu',weights_only=False)
    rows=[]
    for e in manifest['episodes']:
        assert sha(e['path'])==e['sha256']
        key,meta,rec=record(e['path'])
        assert meta['dataset_split']==e['split'] and meta['initial_family']==e['family']
        assert all(torch.equal(v,cached['records'][e['id']][k]) for k,v in rec.items())
        with h5py.File(e['path']) as f:
            row=dict(id=e['id'],family=e['family'],split=e['split'],alpha=e['alpha'],**motion(f['episodes'][key]))
        rows.append(row)
        write(root/'heartbeat.json',dict(stage='original_data_audit',episode=e['id'],completed=len(rows),utc=time.time()))
    st=torch.load(parent/'state_data/statistics.pt',map_location='cpu',weights_only=False)
    for k in ('robot','geometry'):
        x=torch.cat([cached['records'][i][k][::12] if k=='geometry' else cached['records'][i][k] for i in data['splits']['train']])
        assert torch.equal(st[k+'_mean'],x.mean(0)) and torch.equal(st[k+'_std'],x.std(0).clamp_min(1e-4))
    from visuotactile.native_data import limits_affine
    scale,offset=limits_affine(torch.cat([cached['records'][i]['command'] for i in data['splits']['train']]))
    assert torch.equal(scale,st['action_scale']) and torch.equal(offset,st['action_offset'])
    native={r['family']:r for r in rows if r['alpha']==1.}
    for r in rows:
        ref=native.get(r['family'])
        r['duration_ratio_same_family']=r['duration_s']/ref['duration_s'] if ref else None
    val=[json.loads(x) for x in (parent/'training/validation.jsonl').read_text().splitlines()]
    report=dict(passed=True,episodes=len(rows),duration=stats([r['duration_s'] for r in rows]),
        speeds={name:dict(count=len(rr),duration=stats([r['duration_s'] for r in rr])) for name,rr in
            [('native',[r for r in rows if r['alpha']==1]),('alpha_le_065',[r for r in rows if r['alpha']<=.65]),
             ('alpha_gt_065_lt_1',[r for r in rows if .65<r['alpha']<1])]},
        curve=curve_summary(val),rows=rows,train_statistics_exact=True,cache_labels_exact=True,
        phase_scope='Mutually exclusive diagnostic proxies: contact force >0.1N, object rise>3cm, native alignment/subgoals; not stable-grasp truth',
        modality=dict(state_valid=len(rows),multi_view_valid=0,old_blank_fixed_RGB_preserved=True))
    best=torch.load(parent/'training/best.pt',map_location='cpu',weights_only=False)
    last=torch.load(parent/'training/last.pt',map_location='cpu',weights_only=False)
    report['checkpoints']=dict(best_update=best['updates'],last_update=last['updates'],
        ema_equal=all(torch.equal(v,last['ema'][k]) for k,v in best['ema'].items()),
        model_equal=all(torch.equal(v,last['model'][k]) for k,v in best['model'].items()))
    # Episode balancing remains fixed. Quantify its unequal per-anchor exposure.
    train=[r for r in rows if r['split']=='train'];anchors=[int(np.ceil(r['steps']/12)) for r in train]
    report['sampling']=dict(episodes=len(train),anchors=sum(anchors),anchor_counts=stats(anchors),
        per_anchor_probability_ratio_shortest_to_longest=max(anchors)/min(anchors),
        explanation='Each episode equally likely; an anchor in the longest receives fewer selections than one in the shortest. No sampler change.')
    report['original_student_motion']=[]
    for p in sorted((parent/'evaluation').glob('*/result.json')):
        result=read(p)
        if not result['legal']:continue
        with h5py.File(p.parent/'numeric.h5') as f:
            report['original_student_motion'].append(dict(identity=p.parent.name,split=result['split'],**motion(f['episode'])))
    write(root/'data_diagnosis.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('gpu','dataset'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();(gpu if a.mode=='gpu' else dataset)(a.root.resolve())
