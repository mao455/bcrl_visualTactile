"""Measured dependence on real intermediate frames, with robot history fixed.

This is an offline input intervention, not a physical task-success experiment.
"""
import argparse
import copy
import json
from pathlib import Path
import torch
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def backend_probe(root, role, lease):
    """Isolate the rear's direct visual channel; keep its nominal plans fixed."""
    from visuotactile.models.native_dp import load_residual
    from visuotactile.anchor_continue import residual_metrics
    directory=root/role
    data=json.loads((directory/'state_data/data.json').read_text())
    ids=data['splits']['validation']
    rear=directory/'backend'/role
    checkpoint=rear/'training/best.pt'
    index_path=rear/'nominal/index.json'
    index=json.loads(index_path.read_text())
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (index['base_sha256']!=sha256(directory/'training/best.pt') or
            index['base_sha256']!=saved['configuration']['base_sha256'] or
            index['data_sha256']!=sha256(directory/'state_data/data.json')):
        raise ValueError('Direct visual probe requires the current complete pair and cache')
    normal={}
    for identity in ids:
        path=index_path.parent/(identity+'.pt')
        if sha256(path)!=index['episodes'][identity]['sha256']:
            raise ValueError('Rear visual probe cache content differs')
        normal[identity]=torch.load(path,map_location='cpu',weights_only=False)
    changed={}
    for identity,record in normal.items():
        ticks=torch.arange(len(record['time']))
        changed[identity]=dict(record,visual=record['visual'][ticks//12*12])
    with lease.block():model=load_residual(saved,'cuda:0').eval().requires_grad_(False)
    _,predictions=residual_metrics(model,normal,ids,lease)
    _,interventions=residual_metrics(model,changed,ids,lease)
    rows=[]
    for identity,prediction,intervention in zip(ids,predictions,interventions):
        record=normal[identity]
        valid=record.get('supervision_mask',torch.ones(len(record['time']),dtype=torch.bool)).bool()
        def errors(value):
            error=(value-record['target'])[valid]
            return dict(arm_rmse_rad=float(error[:,:7].square().mean().sqrt()),
                hand_rmse_rad=float(error[:,7:].square().mean().sqrt()))
        change=(prediction-intervention)[valid]
        rows.append(dict(identity=identity,normal=errors(prediction),held_visual=errors(intervention),
            command_arm_rms_change_rad=float(change[:,:7].square().mean().sqrt()),
            command_hand_rms_change_rad=float(change[:,7:].square().mean().sqrt())))
    return dict(rows=rows,backend_sha256=sha256(checkpoint),cache_index_sha256=sha256(index_path),
        intervention='Only learned visual features delivered to rear are held at preceding planning tick. Current normal DP plans and all real numeric history are fixed.',
        scope='Direct backend visual-channel sensitivity; distinct from raw-image intervention and not a physical frame-dropping experiment')


def run(root, role, output, *, backend=False):
    from visuotactile.models.native_dp import load_policy
    from visuotactile.native_data import Episode,visual_records
    from visuotactile.continuous_vision import encode_dense
    from visuotactile.position_control import bounded_targets
    from visuotactile.anchor_sampling import labels
    from visuotactile.state_resources import lease_for
    from visuotactile.precision import state_fp32
    if output.exists():raise FileExistsError('Preserve prior intervention evidence')
    state_fp32();session=json.loads((root/'session.json').read_text());lease=lease_for(root,deadline=session['deadline'])
    checkpoint=root/role/'training/best.pt';dr=root/role/'state_data';data=json.loads((dr/'data.json').read_text())
    ids=data['splits']['validation'];rows=[]
    with lease.block():
        model=load_policy(torch.load(checkpoint,map_location='cpu',weights_only=False),'cuda:0').eval()
        records=visual_records(dr,ids,'G00',device='cuda:0')
    @torch.no_grad()
    def predicted(record,episode):
        conditions,anchors=encode_dense(model,record,episode=episode)
        generator=torch.Generator(device='cuda').manual_seed(7091)
        plans=torch.cat([model.predict(c[None],generator,scheduler_index_mode='host') for c in conditions])
        plans=bounded_targets(plans.flatten(0,1),record['joint_limits']).reshape_as(plans)
        truth,valid=labels(record,anchors,24);mask=valid[:,:12]
        pred=plans[:,:12][mask];truth=truth[:,:12][mask];error=pred-truth
        return conditions.cpu(),pred.cpu(),dict(arm_rmse_rad=float(error[:,:7].square().mean().sqrt()),
            hand_rmse_rad=float(error[:,7:].square().mean().sqrt()))
    for identity in ids:
        record=records[identity];episode=Episode(dr,identity,model.group)
        intervention=copy.copy(episode);intervention.perception=dict(episode.perception)
        ticks=torch.arange(len(record['time']));index=ticks//12*12
        for key in ['patches','depth']:intervention.perception[key]=episode.perception[key][index]
        with lease.block():
            normal,pred,error=predicted(record,episode)
            changed,other,other_error=predicted(record,intervention)
        rows.append(dict(identity=identity,normal=error,held_intermediate_images=other_error,
            condition_rms_change=float((normal-changed).square().mean().sqrt()),
            plan_arm_rms_change_rad=float((pred[:,:7]-other[:,:7]).square().mean().sqrt()),
            plan_hand_rms_change_rad=float((pred[:,7:]-other[:,7:]).square().mean().sqrt()),
            intervened_ticks=int((ticks%12!=0).sum())))
    result=dict(checkpoint_sha256=sha256(checkpoint),role=role,rows=rows,test_used=False,
        intervention='RGB-D at nonplanning ticks replaced by preceding planning-frame pixels/features; robot/touch/time unchanged',
        scope='offline causal dependence; no claim of physical improvement or task success')
    if backend:
        del model
        result['direct_backend']=backend_probe(root,role,lease)
    atomic_json(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--role',choices=['A','B'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);run(a.root,a.role,a.output,backend=a.backend)
