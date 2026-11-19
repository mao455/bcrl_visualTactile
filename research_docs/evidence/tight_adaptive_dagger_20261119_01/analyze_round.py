"""Read-only CPU analysis of completed models and recorded autonomous prefixes."""
import argparse, collections, csv, json
from pathlib import Path
import numpy as np
import torch
from tools.tight_adaptive_dagger import ROOT, BASE, PUBLIC, read, write
from tools.tight_frozen_comparison import predictions, table
from tools.tight_representation_probe import source
from visuotactile.compact import sha256

@torch.no_grad()
def offline(root, output, limit=None):
    from visuotactile.models.native_dp import load_residual
    from visuotactile.geometry_sampling import STAGES
    torch.set_num_threads(2)
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for role in 'AB':
        data=read(root/role/'state_data/data.json')
        original=set(data['normalization_train_ids'])
        latest=set(data.get('aggregation',{}).get('new_training_ids',[]))
        validation=set(data['splits']['validation'])
        checkpoint=root/role/f'backend/{role}/training/best.pt'
        model=load_residual(torch.load(checkpoint,map_location='cpu',weights_only=False)).eval()
        cache=root/role/f'backend/{role}/nominal';index=read(cache/'index.json')
        ids=data['splits']['train']+data['splits']['validation']
        for ordinal,identity in enumerate(ids if limit is None else ids[:limit]):
            path=cache/(identity+'.pt')
            assert sha256(path)==index['episodes'][identity]['sha256']
            record=torch.load(path,map_location='cpu',weights_only=False)
            raw=source(data['entries'][identity]);legal=raw['legal']
            truth=torch.from_numpy(raw['command']);reached=torch.from_numpy(raw['q_reached'])
            assert np.array_equal(record['supervision_mask'].numpy(),legal)
            assert torch.equal(record['target'][legal],truth[legal])
            out=predictions(model,record)
            masks=dict(legal=legal,first24=raw['onset'])
            masks.update({name:legal&(raw['phase']==k) for k,name in enumerate(STAGES)})
            cohort='validation6' if identity in validation else 'original30' if identity in original else 'this_round' if identity in latest else 'historical_correction'
            for stage,mask in masks.items():
                if not mask.any():continue
                for component in ('dp_own_target','dp_command','anchor','correction','final'):
                    pred=out['dp'] if component.startswith('dp_') else out[component]
                    target=reached if component=='dp_own_target' and role=='A' else truth-out['anchor'] if component=='correction' else truth
                    error=(pred-target)[mask];span=out['span'][mask]
                    rows.append(dict(role=role,identity=identity,family=data['entries'][identity]['family'],cohort=cohort,stage=stage,component=component,ticks=len(error),
                        arm_squared_sum=float(error[:,:7].square().sum()),hand_squared_sum=float(error[:,7:].square().sum()),
                        arm_span_squared_sum=float((error[:,:7]/span[:,:7]).square().sum()),hand_span_squared_sum=float((error[:,7:]/span[:,7:]).square().sum())))
            write(output/'progress.json',dict(role=role,episodes=ordinal+1,total=len(ids),cuda_initialized=torch.cuda.is_initialized()))
            del record,raw,out
        del model
    table(output/'per_episode.csv',rows)
    total=collections.defaultdict(collections.Counter)
    for row in rows:
        total[row['role'],row['cohort'],row['stage'],row['component']].update({k:row[k] for k in ('ticks','arm_squared_sum','hand_squared_sum','arm_span_squared_sum','hand_span_squared_sum')})
    summary=[]
    for (role,cohort,stage,component),v in total.items():
        summary.append(dict(role=role,cohort=cohort,stage=stage,component=component,ticks=v['ticks'],
            arm_rmse_rad=(v['arm_squared_sum']/v['ticks']/7)**.5,hand_rmse_rad=(v['hand_squared_sum']/v['ticks']/22)**.5,
            arm_span_rmse=(v['arm_span_squared_sum']/v['ticks']/7)**.5,hand_span_rmse=(v['hand_span_squared_sum']/v['ticks']/22)**.5))
    table(output/'summary.csv',summary)
    write(output/'scope.json',dict(root=str(root),complete=limit is None,limit=limit,cuda_initialized=torch.cuda.is_initialized(),
        protocol='Own matching DP nominal and selected backend, full recorded history, legal expert masks only; training pool errors are fitting diagnostics; DP A and B targets differ; no counterfactual expert labels on autonomous prefix',rows=len(rows)))
    print(json.dumps(summary[:5]),flush=True)

def forks(round_index):
    import h5py
    destination=PUBLIC/f'round{round_index:02d}'/'evaluation'
    changes=read(destination/'paired_changes.json');rows=[]
    for change in changes:
        if not (change['lost'] or change['gained']):continue
        root=ROOT/f'round{round_index:02d}'/'evaluation'
        for wave_name in read(root/'registration.json')['waves']:
            wave=Path(wave_name);plan=read(wave/'plan.json')
            cases=[c for c in plan['cases'] if c['spec']['seed']==change['seed'] and c['role'] in (change['role']+'_parent',change['role']+'_new')]
            if len(cases)!=2:continue
            cases=sorted(cases,key=lambda c:c['role'].endswith('_new'))
            with h5py.File(wave/cases[0]['name']/'numeric.h5') as a,h5py.File(wave/cases[1]['name']/'numeric.h5') as b:
                x=a['episode/transition'];y=b['episode/transition'];n=min(len(x['target_after_rad']),len(y['target_after_rad']))
                command=np.abs(x['target_after_rad'][:n]-y['target_after_rad'][:n]).max(-1)
                xp=x['diagnostics_after/object_pose_w'][:n,:3]-x['diagnostics_after/base_pose_w'][:n,:3]
                yp=y['diagnostics_after/object_pose_w'][:n,:3]-y['diagnostics_after/base_pose_w'][:n,:3]
                distance=np.linalg.norm(xp-yp,axis=-1)
                def first(mask):
                    indices=np.flatnonzero(mask);return int(indices[0]) if len(indices) else None
                rows.append(dict(**change,first_command_difference_tick=first(command>1e-6),first_object_difference_1mm_tick=first(distance>.001),common_ticks=n,source_wave=str(wave)))
    write(destination/'forks.json',dict(rows=rows,ticks='zero-based completed control transitions',limitation='Observable paired starts, no exact hidden state or counterfactual expert causality; all outcomes retained'))
    if rows:table(destination/'forks.csv',rows)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['offline','forks']);parser.add_argument('--round',type=int);parser.add_argument('--root',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.mode=='offline':offline(args.root or ROOT/f'round{args.round:02d}'/'learning',args.output or PUBLIC/f'round{args.round:02d}'/'offline',args.limit)
    else:forks(args.round)
