"""Fixed CPU ridge probe of frozen visual512 and robot178; diagnostic only."""
import argparse
import hashlib
import json
import resource
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import torch

from visuotactile.compact import sha256
from visuotactile.native_data import rotate
from visuotactile.training_resources import atomic_json


RIDGE = .01
TARGETS = ('geometric_aabb_center_palm', 'object_root_origin_palm')
PHASES = ('all', 'before_first_observed_contact', 'from_first_observed_contact')


def read(path):
    return json.loads(path.read_text())


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def center_definition(path):
    """Current task spawns this URDF at unit physical scale."""
    tree=ET.parse(path); lower=[]; upper=[]
    links=tree.findall('link')
    if len(links)!=1: raise ValueError('Probe center requires the audited single-link object')
    for collision in links[0].findall('collision'):
        box=collision.find('geometry/box'); origin=collision.find('origin')
        if box is None: raise ValueError('Expected audited box colliders')
        xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        if np.any(rpy): raise ValueError('Rotated collider requires a different center calculation')
        half=np.fromstring(box.get('size'),sep=' ')/2
        lower.append(xyz-half); upper.append(xyz+half)
    minimum=np.min(lower,axis=0); maximum=np.max(upper,axis=0)
    center=(minimum+maximum)/2
    if not np.allclose(center,[0,.025,0],rtol=0,atol=1e-12):
        raise ValueError('Audited L-shape center changed')
    return torch.from_numpy(center),dict(urdf_sha256=sha256(path),
        local_aabb_min_m=minimum.tolist(),local_aabb_max_m=maximum.tolist(),
        local_geometric_aabb_center_m=center.tolist(),
        definition='Midpoint of local collision-geometry AABB; neither mass center nor object root origin.',
        physical_scale='Unit physical spawn scale in precise_assembly.scene_utils._pih_rigid_object_cfg; config object_scale is an expert descriptor, not an extra URDF geometry scale.')


def labels(entry):
    with h5py.File(entry['path']) as f:
        group=f['episodes'][entry['episode']]
        metadata=json.loads(f.attrs['metadata_json'])
        get=lambda name:torch.from_numpy(group[name][:])
        stamp=get('obs/time_s'); q=get('transition/diagnostics_before/palm_body_pose_w').double()
        obj=get('transition/diagnostics_before/object_pose_w').double()
        if (not torch.equal(stamp,get('transition/time_before_s')) or len(q)!=entry['length']
                or not bool(group.attrs['complete'])):
            raise ValueError('Pre-action labels are not aligned with actual observations')
        center,definition=center_definition(Path(metadata['config']['assets']['object_urdf']))
        palm_quat=q[:,3:]; object_quat=obj[:,3:]
        for value in (palm_quat,object_quat):
            if not torch.isfinite(value).all() or (value.norm(dim=-1)-1).abs().max()>1e-3:
                raise ValueError('Expected finite unit scalar-first pose quaternions')
        palm_quat=palm_quat/palm_quat.norm(dim=-1,keepdim=True)
        object_quat=object_quat/object_quat.norm(dim=-1,keepdim=True)
        inverse=palm_quat.clone(); inverse[:,1:]*=-1
        center_world=obj[:,:3]+rotate(object_quat,center.expand(len(obj),3))
        target=dict(geometric_aabb_center_palm=rotate(inverse,center_world-q[:,:3]),
            object_root_origin_palm=rotate(inverse,obj[:,:3]-q[:,:3]))
        force=get('obs/contact_substeps_w_N').double(); valid=get('obs/contact_valid').bool()
        observed_force_time=get('obs/contact_time_s')
        if ((observed_force_time>stamp[:,None]+1e-6)&valid[:,None]).any():
            raise ValueError('Future tactile sample in observation-side contact split')
        if not torch.equal(force[1:].float(),get('transition/contact_substeps_after_w_N')[:-1]):
            raise ValueError('Observation contact must match previously completed native substeps')
        contact=valid&(force.norm(dim=-1)>.1).any(dim=(1,2))
        indices=torch.nonzero(contact).flatten()
        onset=int(indices[0]) if len(indices) else None
        after=contact.long().cumsum(0)>0
        masks=dict(all=torch.ones(len(stamp),dtype=torch.bool),
            before_first_observed_contact=~after,from_first_observed_contact=after)
        return target,masks,stamp,dict(first_observed_contact_step_id=onset,
            current_contact_observations=int(contact.sum()),valid_contact_observations=int(valid.sum()),
            invalid_contact_observations=int((~valid).sum()),
            center_definition=definition,quaternion_order='wxyz',
            label_sha256={name:tensor_sha(value) for name,value in target.items()})


def fit(train_x,train_y):
    """Mean over frames of squared Euclidean error plus fixed ridge penalty."""
    x=train_x.double(); y=train_y.double()
    mean=x.mean(0); std=x.std(0,correction=0)
    constant=std<1e-12; scale=torch.where(constant,1.,std)
    z=(x-mean)/scale
    ymean=y.mean(0)
    gram=z.T@z/len(z)+RIDGE*torch.eye(z.shape[1],dtype=torch.float64)
    rhs=z.T@(y-ymean)/len(z)
    weights=torch.linalg.solve(gram,rhs)
    return (mean,scale,ymean,weights),dict(feature_dimensions=x.shape[1],constant_dimensions=int(constant.sum()),
        training_frames=len(x),feature_standardization='Original train30 frame mean and population std; constant scale set to1; float64 solve.',
        weights_sha256=tensor_sha(weights),feature_mean_sha256=tensor_sha(mean),feature_scale_sha256=tensor_sha(scale))


def predict(parameters,x):
    mean,scale,ymean,weights=parameters
    return (x.double()-mean)/scale@weights+ymean


def metrics(errors,masks):
    result={}
    for phase in PHASES:
        rows=[error[mask[phase]] for error,mask in zip(errors,masks) if mask[phase].any()]
        if not rows:
            result[phase]=dict(frames=0,episodes=0);continue
        merged=torch.cat(rows)
        individual_xyz=torch.stack([value.square().mean(0).sqrt() for value in rows])*1000
        individual_vector=torch.tensor([value.square().sum(-1).mean().sqrt() for value in rows])*1000
        result[phase]=dict(frames=len(merged),episodes=len(rows),
            frame_weighted_xyz_rmse_mm=(merged.square().mean(0).sqrt()*1000).tolist(),
            frame_weighted_vector_rmse_mm=float(merged.square().sum(-1).mean().sqrt()*1000),
            episode_equal_mean_xyz_rmse_mm=individual_xyz.mean(0).tolist(),
            episode_equal_mean_vector_rmse_mm=float(individual_vector.mean()),
            episode_equal_xyz_root_mean_mse_mm=(torch.stack([value.square().mean(0) for value in rows]).mean(0).sqrt()*1000).tolist())
    return result


def inputs(record,kind):
    if kind=='visual_only':return record['visual']
    if kind=='robot178_only':return record['robot']
    if kind=='combined':return torch.cat((record['visual'],record['robot']),-1)
    raise ValueError(kind)


def run(root,output):
    if output.exists(): raise FileExistsError('Preserve earlier diagnostic probe')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    began=time.monotonic();plan=read(root/'development/plan.json')
    reference=read(root/'A/state_data/data.json')
    train=reference['normalization_train_ids'];validation=reference['splits']['validation']
    if len(train)!=30 or len(validation)!=6 or set(train)&set(validation):
        raise ValueError('Expected disjoint original30/validation6')
    if {reference['entries'][i]['family'] for i in train}&{reference['entries'][i]['family'] for i in validation}:
        raise ValueError('Training and validation initial-state families overlap')
    targets={};masks={};stamps={};label_reports={}
    for name in train+validation:
        targets[name],masks[name],stamps[name],label_reports[name]=labels(reference['entries'][name])
    roles={};reference_robot={}
    for role in ('A','B'):
        directory=root/role;data_path=directory/'state_data/data.json';data=read(data_path)
        if data['normalization_train_ids']!=train or data['splits']['validation']!=validation:
            raise ValueError('A/B probe populations differ')
        index_path=directory/'backend'/role/'nominal/index.json';index=read(index_path)
        checkpoint=Path(plan['pairs'][role]['frontend']);base_sha=sha256(checkpoint)
        if base_sha!=plan['pairs'][role]['frontend_sha256'] or index['base_sha256']!=base_sha:
            raise ValueError('Cached visual features must bind current selected round03 frontend')
        if index['data_sha256']!=sha256(data_path):raise ValueError('Cache dataset binding differs')
        records={};cache_reports=[]
        for name in train+validation:
            path=index_path.parent/(name+'.pt');digest=sha256(path)
            if digest!=index['episodes'][name]['sha256']:raise ValueError('Current feature cache hash differs')
            record=torch.load(path,weights_only=False,map_location='cpu')
            if (record['cache_identity']['base_checkpoint_sha256']!=base_sha
                    or record['cache_identity']['data_sha256']!=index['data_sha256']
                    or record['identity']!=name or not torch.equal(record['time'],stamps[name])
                    or record['visual'].shape!=(len(stamps[name]),512)
                    or record['robot'].shape!=(len(stamps[name]),178)):
                raise ValueError('Feature identity, dimensions or pre-action clock differ')
            if not torch.isfinite(record['visual']).all() or not torch.isfinite(record['robot']).all():
                raise ValueError('Nonfinite features')
            records[name]={k:record[k].clone() for k in ('visual','robot')}
            if role=='A':reference_robot[name]=tensor_sha(record['robot'])
            elif tensor_sha(record['robot'])!=reference_robot[name]:raise ValueError('A/B robot-only input differs')
            cache_reports.append(dict(identity=name,sha256=digest,ticks=len(stamps[name])))
            del record
        comparisons={}
        for target_name in TARGETS:
            y=torch.cat([targets[name][target_name] for name in train]);constant=y.mean(0)
            methods={}
            for kind in ('train_mean','visual_only','robot178_only','combined'):
                parameters=None;fitted=dict(training_frames=len(y),feature_dimensions=0)
                if kind!='train_mean':parameters,fitted=fit(torch.cat([inputs(records[name],kind) for name in train]),y)
                split_results={}
                for split,ids in (('train',train),('validation',validation)):
                    errors=[];episode_results=[]
                    for name in ids:
                        prediction=constant.expand_as(targets[name][target_name]) if kind=='train_mean' else predict(parameters,inputs(records[name],kind))
                        error=prediction-targets[name][target_name];errors.append(error)
                        episode_results.append(dict(identity=name,**metrics([error],[masks[name]])))
                    split_results[split]=dict(aggregate=metrics(errors,[masks[name] for name in ids]))
                    if split=='validation':split_results[split]['episodes']=episode_results
                methods[kind]=dict(fit=fitted,**split_results)
            comparisons[target_name]=methods
        roles[role]=dict(frontend_sha256=base_sha,data_sha256=sha256(data_path),
            cache_index_sha256=sha256(index_path),cache_episodes=cache_reports,targets=comparisons)
        print(json.dumps(dict(role=role,completed=True,wall_seconds=time.monotonic()-began)),flush=True)
        del records
    if torch.cuda.is_initialized():raise ValueError('CPU diagnostic unexpectedly initialized CUDA')
    sources=[Path(__file__),Path('visuotactile/native_data.py'),Path('visuotactile/models/native_dp.py'),
        Path('visuotactile/sim/environment.py'),Path('isaacsimenvs/tasks/precise_assembly/scene_utils.py')]
    atomic_json(output,dict(version='frozen_visual_position_ridge_probe_v1',roles=roles,
        registration=dict(train_ids=train,validation_ids=validation,regularization_lambda=RIDGE,
            objective='mean_over_frames(sum_xyz((prediction-target)^2)) + 0.01*sum(weights^2); unpenalized intercept; same fixed lambda for every method/role/label.',
            input_dimensions=dict(visual_only=512,robot178_only=178,combined=690,train_mean=0),
            train_stat_scope='original train30 only; excludes all41 corrections and independent validation6',
            parameter_selection=False,probe_fit_changes_student=False),
        representation=dict(field='nominal_cache.visual',
            computation='native_data.residual_visual -> model.vision.read over last3 actual RGB-D spatial tokens plus frame age/valid; updated every control observation.',
            excludes=['robot178','native_gru','visual_gru persistent state','condition_projection','tactile','plans','diagnostic object/palm poses'],
            labels_only='Pre-action diagnostic object and palm poses enter ridge targets only. No policy, controller, model normalization, sampling, cache or current round04 run is changed.'),
        coordinates=dict(frame='palm body local xyz',quaternion_order='wxyz',
            equation='R(palm_quaternion)^T * (object_root_position + R(object_quaternion)*local_center - palm_position)',
            phases='First observation with contact_valid and any five-finger observed world force >0.1N in either already completed physics substep; subsequent observations remain postcontact even if contact is later lost. Force includes any contacting body, not privileged object-filtered force.',
            observation_alignment='All features and labels use the same actual pre-action time; recorded observed forces equal previous transition substeps. Initial invalid force is not counted as contact.'),
        labels=label_reports,sources={str(path):sha256(path) for path in sources},
        wall_seconds=time.monotonic()-began,process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        cpu_threads=2,cuda_initialized=False,student_optimizer_updates=0,simulator_starts=0,test_used=False,
        interpretation='Fixed frozen-representation decodability on30 successful training families and6 independent successful validation families. A low probe error does not establish that the current policy uses position, that a position estimator is deployed, or any closed-loop grasp/insertion benefit. Narrow scripted-motion correlations can make robot-only position decoding strong. Probe sees the last three real visual frames, not the persistent visual GRU. Validation6 has been reused for student model selection and development; it is not an untouched final test. No intervention or causal vision claim.'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--round-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.round_root,args.output)


if __name__=='__main__':main()
