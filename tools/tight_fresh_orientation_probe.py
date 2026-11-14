"""Fixed CPU rotation-column readouts; probes, never privileged policy inputs."""
import argparse
import json
from pathlib import Path
import time

import h5py
import torch

from tools.tight_fresh_visual_family_probe import family_folds, fold_population
from tools.tight_fresh_visual_prefix_probe import (admit_population, source_labels,
    cache_binding, validate_cache, write_new)
from tools.tight_fresh_position_probe import fit, predict, inputs, read, tensor_sha
from visuotactile.native_data import rotate
from visuotactile.compact import sha256


def relative_columns(palm, obj):
    if palm.shape != obj.shape or palm.ndim != 2 or palm.shape[1] != 7:
        raise ValueError('Aligned current palm/object poses required')
    quats = [palm[:,3:], obj[:,3:]]
    if any(not torch.isfinite(v).all() or (v.norm(dim=-1)-1).abs().max()>1e-3 for v in quats):
        raise ValueError('Finite scalar-first unit quaternions required')
    p,o = [v/v.norm(dim=-1,keepdim=True) for v in quats]
    inv=p.clone();inv[:,1:]*=-1
    columns=[]
    for axis in range(2):
        basis=torch.zeros_like(obj[:,:3]);basis[:,axis]=1
        columns.append(rotate(inv,rotate(o,basis)))
    return torch.cat(columns,dim=-1)


def decode_rotation(columns, epsilon=1e-8):
    """Gram-Schmidt with explicit invalid flags; never substitute identity."""
    if columns.ndim!=2 or columns.shape[1]!=6 or not torch.isfinite(columns).all():
        raise ValueError('Finite consecutive first/second xyz columns required')
    a,b=columns[:,:3],columns[:,3:]
    n=a.norm(dim=-1,keepdim=True);x=a/n.clamp_min(epsilon)
    orthogonal=b-(x*b).sum(-1,keepdim=True)*x
    m=orthogonal.norm(dim=-1,keepdim=True);y=orthogonal/m.clamp_min(epsilon)
    valid=(n[:,0]>epsilon)&(m[:,0]>epsilon)
    return torch.stack((x,y,torch.linalg.cross(x,y,dim=-1)),dim=-1),valid


def errors(prediction, truth):
    p,valid=decode_rotation(prediction);t,target_valid=decode_rotation(truth)
    if not target_valid.all():raise ValueError('Malformed observed orientation labels')
    cosine=((p*t).sum((-2,-1))-1)/2
    angle=torch.acos(cosine.clamp(-1,1))
    return dict(column_mse=(prediction-truth).square().mean(-1),
        angle_rad=angle,valid=valid)


def summarize(rows,masks,identities,phases):
    result={}
    for phase in phases:
        chosen=[(rows[i],masks[i][phase]) for i in identities if masks[i][phase].any()]
        cols=torch.cat([r['column_mse'][m] for r,m in chosen]) if chosen else torch.empty(0)
        angles=[r['angle_rad'][m&r['valid']] for r,m in chosen]
        merged=torch.cat(angles) if angles else torch.empty(0)
        result[phase]=dict(frames=len(cols),episodes=len(chosen),valid_rotation_frames=len(merged),
            invalid_rotation_frames=len(cols)-len(merged),column_mse=float(cols.mean()) if len(cols) else None,
            valid_geodesic_mean_rad=float(merged.mean()) if len(merged) else None,
            valid_geodesic_rms_rad=float(merged.square().mean().sqrt()) if len(merged) else None,
            valid_geodesic_p90_rad=float(torch.quantile(merged,.9)) if len(merged) else None,
            episode_equal_valid_geodesic_mean_rad=(sum(float(v.mean()) for v in angles if len(v))/
                sum(bool(len(v)) for v in angles)) if len(merged) else None)
    return result


def feature(record,method):
    return torch.zeros(len(record['robot']),1) if method=='constant' else inputs(record,method)


def run(root,output):
    if output.exists():raise FileExistsError('Preserve prior probe')
    if torch.cuda.is_initialized():raise ValueError('Fresh CPU process required')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    began=time.monotonic();root=root.resolve()
    data=read(root/'A/state_data/data.json')
    pop=admit_population(data,read(root/'B/state_data/data.json'))
    all_train=pop['original']+pop['corrections'];identities=all_train+pop['validation']
    folds=family_folds([data['entries'][i]['family'] for i in pop['original']])
    truth,masks,stamps,segments,labels={},{},{},{},{}
    for identity in identities:
        entry=data['entries'][identity];split='validation' if identity in pop['validation'] else 'train'
        _,masks[identity],stamps[identity],segments[identity],labels[identity]=source_labels(entry,split,identity in pop['corrections'])
        with h5py.File(entry['path']) as stream:
            group=stream['episodes'][entry['episode']]
            if not torch.equal(torch.from_numpy(group['transition/time_before_s'][:]),stamps[identity]):
                raise ValueError('Orientation label time differs from observed features')
            pose=[torch.from_numpy(group['transition/diagnostics_before/'+k][:]).double()
                for k in ('palm_body_pose_w','object_pose_w')]
            truth[identity]=relative_columns(*pose)
        labels[identity]['orientation_columns_sha256']=tensor_sha(truth[identity])
    roles={};residual_sha=sha256(root/'B/state_data/statistics.pt')
    for role in 'AB':
        index_path,index,binding=cache_binding(root/role,residual_sha);records={}
        for identity in identities:
            path=index_path.parent/(identity+'.pt')
            if sha256(path)!=index['episodes'][identity]['sha256']:raise ValueError('Current cache changed')
            record=torch.load(path,map_location='cpu',weights_only=False)
            validate_cache(record,identity,stamps[identity],binding,masks[identity],segments[identity])
            records[identity]={k:record[k] for k in ('visual','robot')}
        methods={}
        for method in ('constant','visual_only','robot178_only','combined'):
            pooled={kind:{} for kind in ('original_only','with_corrections')};details=[]
            for fold,heldout in enumerate(folds):
                evaluate,groups=fold_population(data,pop,heldout);info={}
                for kind,training in groups.items():
                    params,info[kind]=fit(torch.cat([feature(records[i],method) for i in training]),
                        torch.cat([truth[i] for i in training]))
                    info[kind]['feature_standardization']='Exact per-fold fitting population; never evaluated families'
                    for i in evaluate:pooled[kind][i]=errors(predict(params,feature(records[i],method)),truth[i])
                details.append(dict(fold=fold,excluded_families=heldout,training_ids=groups,evaluated_ids=evaluate,fits=info))
            validation={}
            for kind,training in [('original_only',pop['original']),('with_corrections',all_train)]:
                params,info=fit(torch.cat([feature(records[i],method) for i in training]),torch.cat([truth[i] for i in training]))
                info['feature_standardization']='Exact registered training_ids frame mean/std; validation excluded'
                rows={i:errors(predict(params,feature(records[i],method)),truth[i]) for i in pop['validation']}
                validation[kind]=dict(training_ids=training,fit=info,metrics=summarize(rows,masks,pop['validation'],('all','before_contact','after_contact')))
            metrics={kind:summarize(rows,masks,pop['corrections'],('student_prefix','student_after_contact','legal_expert')) for kind,rows in pooled.items()}
            methods[method]=dict(folds=details,correction_family_folds=metrics,fixed_validation6=validation)
            print(role,method,{kind:round(v['student_after_contact']['valid_geodesic_mean_rad'],4) for kind,v in metrics.items()},flush=True)
        roles[role]=dict(binding=binding,cache_index_sha256=sha256(index_path),methods=methods)
    if torch.cuda.is_initialized():raise ValueError('Unexpected CUDA initialization')
    project=Path(__file__).resolve().parents[1]
    write_new(output,dict(complete=True,version='relative_rotation_columns_probe_v1',source_root=str(root),
        roles=roles,labels=labels,population=pop,folds=folds,seconds=time.monotonic()-began,
        label='Current pre-action object orientation in current palm body frame; wxyz to R_palm^T R_object, consecutive first-column xyz then second-column xyz',
        target_scope='Diagnostic pose labels only; never a policy input or action label. No shape-symmetry quotient or native-success substitution.',
        fit='Fixed ridge lambda=.01 with unpenalized intercept; no hyperparameter or model selection',
        interpretation='Student encoders already learned these correction families; only probe fitting excludes them. Fixed validation6 is already used for student selection. This tests decoding and probe coverage, not causal policy use, student held-out performance or task success.',
        degeneracy='Predicted first-column or orthogonalized second-column norm <=1e-8 is invalid; report denominator and all-frame column MSE, never silently replace with identity',
        payload_hash_scope='Current nominal shards actually rehashed. Raw source SHA inherited from manifest; read label/authority/time arrays hashed.',
        source_sha256=sha256(__file__),dependency_sha256={p:sha256(project/p) for p in (
            'tools/tight_fresh_visual_family_probe.py','tools/tight_fresh_visual_prefix_probe.py',
            'tools/tight_fresh_position_probe.py','visuotactile/native_data.py')},
        optimizer_updates=0,simulator_starts=0,model_forward_calls=0,cuda_initialized=False,test_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.output)
