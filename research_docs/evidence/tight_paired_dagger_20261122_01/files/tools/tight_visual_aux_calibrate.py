"""One registered, train-only gradient scale measurement; zero optimizer updates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import torch

from visuotactile.anchor_sampling import labels,planning_anchors
from visuotactile.compact import sha256
from visuotactile.continuous_vision import encode_dense
from visuotactile.dp_objectives import masked_objectives
from visuotactile.models.native_dp import load_policy
from visuotactile.native_data import CAMERAS, DINO_REVISION, visual_records
from visuotactile.position_control import continuous_image_contract
from visuotactile.training_resources import atomic_json
from visuotactile.visual_pose_aux import VisualPoseHead,load_pose_labels,pose_loss
from visuotactile.visual_aux_training import calibration_ids
from visuotactile.target_windows import validate_target_window


def read(path):return json.loads(Path(path).read_text())


def subset(data):
    """Prespecified six sorted original families, including their real corrections."""
    selected=calibration_ids(data)
    original=data['normalization_train_ids']
    families=sorted({data['entries'][i]['family'] for i in original})
    if data['splits'].get('test'):
        raise ValueError('Test payloads must remain in the separate held-out manifest')
    for identity in data['splits']['train']:
        entry=data['entries'][identity]
        if (Path(identity).name!=identity or identity in ('.','..')
                or entry['family'] not in families or entry.get('split','train')!='train'
                or (identity not in original and entry.get('trajectory_kind')!='expert_takeover_v1')):
            raise ValueError('Only original training families and admitted actual corrections are allowed')
    # Family selection is sorted; episode order is the trainer's actual train order.
    return families[:6],selected


def paired_subset(data):
    for key in ('splits','entries','normalization_train_ids'):
        if data['A'][key]!=data['B'][key]:raise ValueError('Matched A/B source population required')
    result=subset(data['A'])
    if subset(data['B'])!=result:raise ValueError('Matched A/B calibration population required')
    return result


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def head_sha(head):
    return {name:tensor_sha(value) for name,value in head.state_dict().items()}


def source_binding(root,role,data,ids):
    """CPU admission before any calibrated forward, with actual shard byte hashes."""
    directory=root/role;state=directory/'state_data';training=directory/'training'
    files={name:sha256(path) for name,path in dict(frontend=training/'best.pt',
        data=state/'data.json',statistics=state/'statistics.pt',numeric=state/'numeric.pt',
        cache_index=directory/'cache_index.json',configuration=training/'configuration.json',
        finished=training/'finished.json').items()}
    saved=torch.load(training/'best.pt',map_location='cpu',weights_only=False)
    config=saved['configuration'];finished=read(training/'finished.json')
    contract=continuous_image_contract('q_reached' if role=='A' else 'u_expert')
    if (saved['contract']!=contract or saved['updates']<1 or saved.get('kind')!='dp'
            or saved.get('group')!='I-HistoryDP' or config!=read(training/'configuration.json')
            or not finished.get('complete') or finished.get('checkpoint_sha256')!=files['frontend']
            or config['data_sha256']!=files['data'] or config['statistics_sha256']!=files['statistics']
            or config['cache_index_sha256']!=files['cache_index']
            or saved['train_ids']!=data['splits']['train']
            or saved['validation_ids']!=data['splits']['validation']):
        raise ValueError('Completed selected v6 source/configuration/population binding differs')
    stats=torch.load(state/'statistics.pt',map_location='cpu',weights_only=False)
    if (set(stats)!={'robot_mean','robot_std','action_scale','action_offset'}
            or any(not torch.equal(saved['ema'][k],v) for k,v in stats.items())):
        raise ValueError('Selected EMA normalization differs from fixed training statistics')
    window=validate_target_window((config.get('continuation') or {}).get('target_window_contract'))
    index=read(directory/'cache_index.json')
    expected=dict(data_sha256=files['data'],dino_revision=DINO_REVISION,dino_sha256=config['dino_sha256'],
        stride=1,cameras=CAMERAS,resolution=[448,252],patches=[384,18,32],depth=[2,18,32],
        random_augmentation=False,trainable_projection_cached=False)
    if any(index.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen every-tick visual cache contract differs')
    numeric=torch.load(state/'numeric.pt',map_location='cpu',weights_only=False)
    if numeric['data_sha256']!=files['data']:raise ValueError('Numeric cache source binding differs')
    entries={}
    for identity in ids:
        entry=data['entries'][identity];cached=index['entries'][identity]
        path=state/'perception'/(identity+'.pt');digest=sha256(path)
        if (digest!=cached['sha256'] or cached['source_sha256']!=entry['source_sha256']
                or cached['frames']!=entry['length']):
            raise ValueError('Selected perception shard bytes/source/length differ')
        shard=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        stamp=numeric['records'][identity]['time'];ct=shard['camera_time'];cv=shard['camera_valid'];n=entry['length']
        if (stamp.shape!=(n,) or not torch.isfinite(stamp).all()
                or not torch.allclose(stamp[1:]-stamp[:-1],torch.full_like(stamp[1:],1/60),atol=1e-6,rtol=0)
                or ct.shape!=(n,3) or tensor_sha(ct)!=cached['camera_time_sha256']
                or not torch.isfinite(ct).all() or not torch.allclose(ct,stamp[:,None].expand_as(ct),atol=1e-6,rtol=0)
                or cv.shape!=(n,3) or cv.dtype!=torch.bool or not cv.all()
                or shard['patches'].shape!=(n,3,384,18,32) or shard['depth'].shape!=(n,3,2,18,32)):
            raise ValueError('Every actual native observation requires aligned three-view cached frames')
        entries[identity]=dict(sha256=digest,source_manifest_sha256=entry['source_sha256'],
            frames=n,camera_time_sha256=tensor_sha(ct),observation_time_sha256=tensor_sha(stamp))
    return dict(files=files,target_window_contract=window,entries=entries,selected_updates=saved['updates'])


def check_label_binding(record,proof,cache_proof):
    if (tensor_sha(record['time'])!=proof['label_sha256']['time_s']
            or cache_proof['camera_time_sha256']!=proof['input_tensor_sha256']['camera_time_s']):
        raise ValueError('Actual cached observations and diagnostic pose clocks differ')
    authority=proof['authority_tensor_sha256']
    if authority:
        for name,source in (('supervision_mask','expert_label_valid'),('control_segment','segment_id')):
            if name not in record or tensor_sha(record[name])!=authority[source]:
                raise ValueError('Actual cached action mask/segment differs from recorded expert authority')


def weight_from_rows(rows,ratio):
    if ratio!=.1 or {r['role'] for r in rows}!={'A','B'}:
        raise ValueError('Fixed common 0.1 ratio and both roles required')
    ids={role:sorted(r['identity'] for r in rows if r['role']==role) for role in ('A','B')}
    if not ids['A'] or ids['A']!=ids['B'] or len(set(ids['A']))!=len(ids['A']):
        raise ValueError('Matching unique calibration episodes required')
    for row in rows:
        if any(not math.isfinite(row[k]) or row[k]<=0 for k in ('base_gradient_norm','pose_gradient_norm')):
            raise ValueError('Nonzero finite observed shared-vision gradients required')
    base=statistics.median(r['base_gradient_norm'] for r in rows)
    pose=statistics.median(r['pose_gradient_norm'] for r in rows)
    return ratio*base/pose,dict(base_gradient_median=base,pose_gradient_median=pose)


def base_loss(model,condition,truth,valid,generator):
    """Exactly the fresh trainer's x0 equal Arm/Hand objective and draw order."""
    if model.noise.config.prediction_type!='sample' or model.horizon!=24:
        raise ValueError('Calibration requires the fresh H24 sample objective')
    target=torch.where(valid[...,None],truth*model.action_scale+model.action_offset,0.)
    noise=torch.randn(target.shape,device=target.device,generator=generator)
    ticks=torch.randint(100,(len(target),),device=target.device,generator=generator)
    prediction=model.denoiser(model.noise.add_noise(target,noise,ticks),ticks,global_cond=condition)
    return masked_objectives(prediction,target,valid)['B']


def gradient_norm(loss,parameters,*,retain_graph):
    if (loss.ndim or not torch.isfinite(loss) or not parameters
            or len({id(p) for p in parameters})!=len(parameters)):
        raise ValueError('Finite scalar loss and unique shared parameters required')
    gradient=torch.autograd.grad(loss,parameters,retain_graph=retain_graph,allow_unused=True)
    result=math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradient if g is not None))
    if not math.isfinite(result) or result<=0:raise ValueError('Nonzero finite shared-vision gradient required')
    return result


def session_binding(root,session_root=None):
    """Bind an existing active owner while preserving the source's old deadline."""
    root=Path(root).resolve(strict=True)
    session_root=Path(session_root).resolve(strict=True) if session_root else root
    session=read(session_root/'session.json');source_session=read(root/'session.json')
    if (type(session.get('deadline')) not in (int,float) or not math.isfinite(session['deadline'])
            or time.time()>=session['deadline']):raise TimeoutError('Registered session expired')
    if session.get('domain')!=source_session.get('domain'):
        raise ValueError('Source and active session domains differ')
    resource=(session_root/'shared_resources.json').resolve(strict=True)
    owner=Path(session.get('resource_root',resource.parent)).resolve(strict=True)
    if (owner/'shared_resources.json').resolve(strict=True)!=resource:
        raise ValueError('Active session and telemetry owner differ')
    binding=dict(session_root=str(session_root),session_sha256=sha256(session_root/'session.json'),
        source_session_sha256=sha256(root/'session.json'),resource_root=str(owner),
        shared_resources_target=str(resource),deadline=session['deadline'])
    return session,binding


def run(root,output,head_seed,session_root=None):
    from visuotactile.precision import state_fp32
    from visuotactile.state_resources import lease_for
    root=root.resolve();output=output.absolute()
    started=output.with_suffix('.started.json')
    if any(p.exists() or p.is_symlink() for p in (output,started,output.with_suffix('.progress.json'))):
        raise FileExistsError('Preserve each calibration attempt')
    if type(head_seed) is not int or not 0<=head_seed<2**63:raise ValueError('Explicit nonnegative int63 head seed required')
    session,active_session=session_binding(root,session_root)
    data={r:read(root/r/'state_data/data.json') for r in ('A','B')}
    families,ids=paired_subset(data)
    bindings={r:source_binding(root,r,data[r],ids) for r in ('A','B')}
    frontend={r:sha256(root/r/'training/best.pt') for r in ('A','B')}
    data_sha={r:sha256(root/r/'state_data/data.json') for r in ('A','B')}
    project=Path(__file__).resolve().parents[1]
    registration=dict(version='visual_pose_aux_calibration_v1',source_root=str(root),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
        source_sha256={str(p.relative_to(project)):sha256(p) for folder in ('tools','visuotactile') for p in sorted((project/folder).rglob('*.py'))},
        data_sha256=data_sha,frontend_sha256=frontend,
        statistics_sha256={r:bindings[r]['files']['statistics'] for r in ('A','B')},
        target_window_contract={r:bindings[r]['target_window_contract'] for r in ('A','B')},
        source_binding=bindings,subset_ids=ids,families=families,
        subset_rule='sorted first six original training families and all their admitted training corrections',
        head_seed=head_seed,gradient_ratio_target=.1,**active_session,utc=time.time(),
        diffusion_draws='Fresh device generator seed2026111403 for each episode and role; randn then randint100, matching the base objective draw order. Calibration draws are fixed probes, not a continuation of the training RNG.')
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(started,registration)
    torch.set_num_threads(2);state_fp32();lease=lease_for(Path(active_session['session_root']),deadline=session['deadline'])
    rows=[];began=time.perf_counter();heads={};gradient_parameters={}
    for role in ('A','B'):
        directory=root/role/'state_data'
        with lease.block():
            saved=torch.load(root/role/'training/best.pt',map_location='cpu',weights_only=False)
            if (saved['contract']!=continuous_image_contract('q_reached' if role=='A' else 'u_expert')
                    or saved['updates']<1):raise ValueError('Selected trained v6 source required')
            model=load_policy(saved,'cuda:0').train();del saved
            head=VisualPoseHead(seed=head_seed).cuda()
            heads[role]=head_sha(head)
            if role=='B' and heads['A']!=heads['B']:raise ValueError('Matched random auxiliary heads differ')
            records=visual_records(directory,ids,'G00',device='cuda:0',
                target_window_contract=bindings[role]['target_window_contract'])
            parameters=tuple(model.vision.parameters())
            gradient_parameters[role]={name:list(value.shape) for name,value in model.vision.named_parameters()}
            versions={name:value._version for name,value in model.state_dict(keep_vars=True).items()}
        for identity in ids:
            entry=data[role]['entries'][identity]
            pose,proof=load_pose_labels(entry,split='train',allowed_families=families)
            with lease.block():
                record=records[identity]
                check_label_binding(record,proof,bindings[role]['entries'][identity])
                condition,anchors,visual=encode_dense(model,record,return_visual=True)
                if not torch.equal(visual['ticks'],torch.arange(len(record['time']),device=anchors.device)):
                    raise ValueError('Calibration must read every actual visual observation')
                permitted=planning_anchors(record).to(anchors.device);keep=torch.isin(anchors,permitted)
                if not keep.any():raise ValueError('No legal calibration actions')
                truth,valid=labels(record,anchors[keep],24)
                base=base_loss(model,condition[keep],truth,valid,
                    torch.Generator(device='cuda').manual_seed(2026111403))
                pred=head(visual['readout'],route='shared_vision')
                auxiliary=pose_loss(pred,pose,time_s=record['time'][visual['ticks']])
                if auxiliary['valid_ticks']==0:raise ValueError('No valid calibration perception')
                gn=gradient_norm(base,parameters,retain_graph=True)
                pn=gradient_norm(auxiliary['loss'],parameters,retain_graph=False)
                rows.append(dict(role=role,identity=identity,family=entry['family'],
                    base_gradient_norm=gn,pose_gradient_norm=pn,base_loss=float(base.detach()),
                    pose_loss=float(auxiliary['loss'].detach()),pose_label_proof=proof,
                    anchors=int(keep.sum()),legal_target_slots=int(valid.sum()),perception_ticks=auxiliary['valid_ticks']))
                del condition,visual,base,pred,auxiliary
            atomic_json(output.with_suffix('.progress.json'),dict(completed=len(rows),total=2*len(ids),utc=time.time()))
        if (versions!={name:value._version for name,value in model.state_dict(keep_vars=True).items()}
                or any(p.grad is not None for p in (*model.parameters(),*head.parameters()))
                or heads[role]!=head_sha(head)):
            raise ValueError('Gradient calibration mutated model/head or accumulated optimizer gradients')
        del model,head,records,parameters
    weight,medians=weight_from_rows(rows,.1)
    for row in rows:row['weighted_pose_to_base_gradient_ratio']=weight*row['pose_gradient_norm']/row['base_gradient_norm']
    lease.check()
    if (session_binding(root,Path(active_session['session_root']))[1]!=active_session
            or any(source_binding(root,r,data[r],ids)!=bindings[r] for r in ('A','B'))):
        raise ValueError('Frozen source changed during calibration')
    atomic_json(output,dict(registration,passed=True,loss_weight=weight,roles=['A','B'],
        routes=['detached_readout','shared_vision'],shared_weight=True,rows=rows,head_initialization_sha256=heads,
        gradient_parameter_shapes=gradient_parameters,**medians,
        estimator='target_ratio_times_median_base_gradient_over_median_pose_gradient_v1',
        optimizer_updates=0,simulator_starts=0,test_used=False,validation_used=False,
        seconds=time.perf_counter()-began,source_receipt_sha256=sha256(started),
        gradient_scope='Shared HistoryVision parameters only. Calibration reference is masked DP base loss, excluding additional A/B geometry terms. Head gradients are clipped independently during training.',
        interpretation='Train-only scale choice, not a fitted policy, selected candidate or autonomy result'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--head-seed',type=int,default=2026111405)
    parser.add_argument('--session-root',type=Path,
        help='Existing active session/telemetry owner; source models and original deadline remain unchanged')
    args=parser.parse_args();run(args.root,args.output,args.head_seed,args.session_root)


if __name__=='__main__':main()
