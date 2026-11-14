"""Compare complete cached DP commitments with causal observation-hold forecasts."""
import argparse
import json
from pathlib import Path

import torch
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def report(root, output):
    roles={}
    for role in ('A','B'):
        directory=root/role;data_path=directory/'state_data/data.json';data=read(data_path)
        numeric_path=directory/'state_data/numeric.pt'
        numeric=torch.load(numeric_path,map_location='cpu',weights_only=False)['records']
        index_path=directory/'backend'/role/'nominal/index.json';index=read(index_path)
        checkpoint=directory/'training/best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        label='q_reached' if role=='A' else 'u_expert'
        if (saved['contract']['supervision']!=label or saved['contract'].get('visual_stride')!=1
                or index['base_sha256']!=sha256(checkpoint) or index['data_sha256']!=sha256(data_path)
                or index['statistics_sha256']!=sha256(directory/'state_data/statistics.pt')
                or read(directory/'training/finished.json')['checkpoint_sha256']!=sha256(checkpoint)):
            raise ValueError('Require exact selected full-frame frontend and current cache')
        selected_epoch=saved['epoch'];del saved
        errors=[];bindings={}
        for identity in data['splits']['validation']:
            if identity in data['splits']['train'] or identity in data['splits'].get('test',[]):
                raise ValueError('Validation family must stay independent')
            path=index_path.parent/(identity+'.pt');digest=sha256(path)
            if digest!=index['episodes'][identity]['sha256']:raise ValueError('DP cache changed')
            record=torch.load(path,map_location='cpu',weights_only=False);actual=numeric[identity]
            if (record['nominal_supervision']!=label or not torch.equal(record['robot'],actual['robot'])
                    or not record['supervision_mask'].all() or not actual['supervision_mask'].all()):
                raise ValueError('Expected complete aligned validation demonstration')
            target=actual[label];count=len(target);anchors=torch.arange(count)//12*12
            observed=actual['robot'][:,:29] if role=='A' else actual['robot'][:,58:87]
            predictions=torch.stack((record['base'],observed[anchors],observed)).double()
            if not torch.isfinite(predictions).all() or not torch.isfinite(target).all():raise ValueError('Nonfinite prediction')
            errors.append(predictions-target.double());bindings[identity]=dict(cache_sha256=digest,ticks=count)
        error=torch.cat(errors,dim=1)
        metrics={name:dict(ticks=x.shape[0],arm_rmse_rad=float(x[:,:7].square().mean().sqrt()),
            hand_rmse_rad=float(x[:,7:].square().mean().sqrt()))
            for name,x in zip(('selected_DP_commit12','hold_plan_start_observation','hold_current_control_observation'),error)}
        validation=[json.loads(s) for s in (directory/'training/validation.jsonl').read_text().splitlines()]
        selected=next(row for row in validation if row['epoch']==selected_epoch)
        for key in ('arm_rmse_rad','hand_rmse_rad'):
            if abs(metrics['selected_DP_commit12'][key]-selected['validation'][key])>1e-7:
                raise ValueError('Cached commitment error differs from actual selected DP validation')
        roles[role]=dict(label=label,metrics=metrics,validation_ids=data['splits']['validation'],cache_bindings=bindings,
            checkpoint_sha256=sha256(checkpoint),index_sha256=sha256(index_path),data_sha256=sha256(data_path),
            numeric_sha256=sha256(numeric_path))
    atomic_json(output,dict(roles=roles,test_used=False,model_forward_calls=0,simulator_starts=0,
        source_sha256=sha256(Path(__file__)),scope='Same six validation demonstrations and actual full causal histories. A forecasts q[t+1], B expert absolute target. Hold-plan-start updates only at the 5Hz planning ticks, then holds the current observed q (A) or pre-transition actuator target (B) for twelve controls. Hold-current updates that observation every60Hz and is a stronger informational baseline. These are forecast errors, not executed policies; neither baseline establishes autonomous success. DP arm/hand label spaces differ between A/B.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.root,a.output)
