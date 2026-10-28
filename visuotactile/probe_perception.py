"""Audit real v2 recordings, render stage sheets and check trainable I/O.

No optimizer, correction loss, policy training or success-rate tuning. Simulation
truth is used ONLY for offline alignment/visibility diagnostics in this tool.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import default_collate

from .contracts import assert_policy_compatible
from .correction import CorrectionContext
from .dataset import EpisodeDataset, CorrectionEpisodeDataset
from .evaluate import audit_trajectory
from .models.nominal import NominalDiffusionPolicy, PolicyConfig
from .models.correction import CorrectionPolicy
from .observations import nominal_keys


def check_tree(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a: check_tree(a[k],b[k])
    else:
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def stage_sheet(group,metadata,path):
    images=group['obs/image'][:]
    mask=group['obs/depth_valid'][:]
    n=len(images)
    successes=group['transition/diagnostics_after/successes'][:]
    contact=np.linalg.norm(group['obs/contact_w_N'][:],axis=-1).max(-1)>.1
    first_contact=int(np.flatnonzero(contact)[0]) if contact.any() else 0
    pre=int(np.flatnonzero(successes>=1)[0]) if (successes>=1).any() else n-1
    inserted=int(np.flatnonzero(successes>=2)[0]) if (successes>=2).any() else n-1
    retract=np.flatnonzero(group['transition/diagnostics_after/retract_phase'][:])
    # A done at the same tick as entering retract leaves no returned phase image.
    retract_obs=int(retract[0])+1 if len(retract) else n
    last_label='Retract' if retract_obs<n else 'Last pre-terminal; no separate retract image'
    rows=[('Approach',0),('Contact / grasp',first_contact),('Transport (review visually)',(first_contact+pre)//2),
          ('Pre-insert (before first goal)',max(0,pre-1)),('Insert (before second goal)',max(0,inserted-1)),(last_label,min(retract_obs,n-1))]
    h,w=images.shape[-2:]
    sheet=Image.new('RGB',(3*len(metadata['cameras'])*w,len(rows)*(h+28)),(22,22,22));draw=ImageDraw.Draw(sheet)
    env_id=int(group.name.split('/env')[1].split('_')[0])
    metrics=[]
    # Observation t corresponds to transition t-1. No terminal/reset image relabeling.
    for row,(label,t) in enumerate(rows):
        per_view=[]
        for v,cam in enumerate(metadata['cameras']):
            im=images[t,v]
            rgb=(im[:3].transpose(1,2,0).clip(0,1)*255).astype('uint8')
            # Fixed 0..2 m scale, invalid is magenta; keep masks visible separately.
            z=(im[3].clip(0,2)/2*255).astype('uint8')
            depth=np.repeat(z[...,None],3,-1);depth[~mask[t,v,0]]=(255,0,255)
            valid=(mask[t,v,0]*255).astype('uint8')
            for offset,array in enumerate((rgb,depth,valid)):
                sheet.paste(Image.fromarray(array).convert('RGB'),((v*3+offset)*w,row*(h+28)+28))
            record=dict(camera=cam['id'],valid_depth_fraction=float(mask[t,v].mean()),rgb_std=float(im[:3].std()))
            if t>0:
                origin=np.array(metadata['env_origins_w'][env_id])
                p=np.array(cam['pos_env'][env_id])+origin
                quat=np.array(cam['quat_ros_wxyz'][env_id])
                if 'camera_pose_w' in group['obs']:
                    pose=group['obs/camera_pose_w'][t,v];p=pose[:3];quat=pose[3:]
                r=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                xyz=group['transition/diagnostics_after/finger_position_w'][t-1]
                optical=(xyz-p)@r;uv=optical@np.array(cam['intrinsics'][env_id]).T
                uv=uv[:,:2]/uv[:,2:]
                inside=(optical[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
                record['finger_origin_in_frame_count']=int(inside.sum())
                record['finger_origin_uv']=uv.tolist()
            per_view.append(record)
        draw.text((4,row*(h+28)+4),f'{label}; observation t={t} | each declared camera: RGB / depth 0..2m / valid',fill='white')
        metrics.append(dict(label=label,step=t,views=per_view))
    sheet.save(path)
    return metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(11)
    report=dict(kind='real-sensor engineering probe; no training or executed network actions',trajectory=audit_trajectory(args.data))
    with h5py.File(args.data) as f:
        if int(f.attrs['schema_version'])!=2: raise ValueError('This probe requires v2 data')
        metadata=json.loads(f.attrs['metadata_json'])
        assert_policy_compatible(metadata,metadata)
        checks=[]
        for name,g in f['episodes'].items():
            n=len(g['obs/step_id']);s=g['obs/time_s'][:];valid=g['obs/contact_valid'][:]
            np.testing.assert_allclose(np.diff(s),metadata['control_dt'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(g['obs/camera_time_s'][:],np.repeat(s[:,None],len(metadata["camera_ids"]),axis=1),rtol=0,atol=0)
            ts=g['obs/contact_time_s'][:]
            np.testing.assert_allclose(ts[:,1],s,rtol=0,atol=0)
            np.testing.assert_allclose(ts[:,1]-ts[:,0],metadata['physics_dt'],rtol=0,atol=1e-12)
            fw=g['obs/contact_substeps_w_N'][:];fl=g['obs/contact_substeps_local_N'][:]
            rot=g['obs/finger_rotation_base'][:];pos=g['obs/finger_position_base_m'][:]
            np.testing.assert_allclose(fl,np.einsum('...ji,...j->...i',rot,fw),atol=1e-5,rtol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(fl,axis=-1),np.linalg.norm(fw,axis=-1),atol=2e-5,rtol=1e-6)
            truth_q=g['transition/diagnostics_after/finger_quat_wxyz'][:-1]
            truth_rot=Rotation.from_quat(truth_q.reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(n-1,5,3,3)
            rot_error=float(np.abs(rot[1:,-1]-truth_rot).max()) if n>1 else 0.
            assert rot_error<1e-4
            assert np.isfinite(pos).all() and np.isfinite(rot).all()
            masks=g['obs/depth_valid'][:];images=g['obs/image'][:]
            np.testing.assert_array_equal(images[:,:,3:4],g['obs/depth_m'][:])
            assert np.all(images[:,:,3:4][~masks]==0)
            checks.append(dict(episode=name,steps=n,position_error_m_max=float(g['transition/diagnostics_after/fk_position_error_m'][:].max()),
                               rotation_matrix_error_max=rot_error,force_max_N_by_finger=np.linalg.norm(fw,axis=-1).max((0,1)).tolist(),
                               depth_valid_fraction_by_view=masks.mean((0,2,3,4)).tolist()))
        report['sensor_checks']=checks
        selected=next(g for g in f['episodes'].values() if g.attrs['complete'] and g['transition/diagnostics_after/retract_succeeded'][-1])
        report['visibility_episode']=selected.name
        report['visibility']=stage_sheet(selected,metadata,args.output/'views.png')
    data=EpisodeDataset(args.data)
    correction_data=CorrectionEpisodeDataset(args.data,allow_untrained_probe=True)
    cfg=PolicyConfig(memory_version=1,views=len(metadata["camera_ids"]))
    model=NominalDiffusionPolicy(cfg)
    correction=CorrectionPolicy(cfg)
    chosen=next(i for i,(key,t) in enumerate(correction_data.index) if t>=100)
    batch=default_collate([data[chosen]])
    start=time.perf_counter()
    pred=model.predict(batch['obs'],generator=torch.Generator().manual_seed(10))
    report['untrained_nominal_prediction_ms']=1000*(time.perf_counter()-start)
    # Real recorded expert-action denoising graph: backward only, NO optimizer.
    loss=model.loss(batch['obs'],batch['action_target'],batch['action_valid'])
    loss.backward()
    inputs=default_collate([correction_data[chosen]])['inputs']
    output=correction(**inputs);output.sum().backward()
    gradients={}
    modules={'nominal_rgb':model.encoder.vision.rgb,'nominal_depth':model.encoder.vision.depth,
             'nominal_context':model.encoder.context,'denoiser':model.denoiser,
             'correction_rgb':correction.visual.vision.rgb,'correction_depth':correction.visual.vision.depth,
             'finger_tactile':correction.tactile,'motion':correction.motion,'intent':correction.intent,'correction':correction.policy}
    for name,module in modules.items():
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        gradients[name]=float(sum(g.abs().sum() for g in grads));assert gradients[name]>0
    report.update(gradient_l1=gradients,nominal_output_shape=list(pred.shape),correction_output_shape=list(output.shape),
                  parameter_updates=0,trained_correction=False)
    checkpoint=dict(checkpoint_version=2,config=asdict(cfg),step=0,model=model.state_dict(),correction=correction.state_dict(),metadata=metadata)
    torch.save(checkpoint,args.output/'untrained_roundtrip.pt')
    restored=torch.load(args.output/'untrained_roundtrip.pt',weights_only=False)
    other=NominalDiffusionPolicy(PolicyConfig.from_dict(restored['config']));other.load_state_dict(restored['model'])
    torch.testing.assert_close(pred,other.predict(batch['obs'],generator=torch.Generator().manual_seed(10)),rtol=0,atol=0)
    other_c=CorrectionPolicy(cfg);other_c.load_state_dict(restored['correction'])
    torch.testing.assert_close(output,other_c(**inputs),rtol=0,atol=0)
    report['checkpoint_roundtrip_exact']=True
    # Compare saved online fields to both offline loaders with causal command t-1.
    key=correction_data.index[chosen][0]
    context=CorrectionContext(1,cfg)
    with h5py.File(args.data) as f:
        g=f['episodes'][key]
        keys=set(nominal_keys(2))|set(context.feedback.keys)|{'episode_id','step_id'}
        keys-={'previous_command','command_valid'}
        arrays={k:g['obs/'+k][:] for k in keys}
        offsets={t:i for i,(k,t) in enumerate(correction_data.index) if k==key}
        for t in range(len(g['obs/step_id'])):
            packet={k:torch.from_numpy(v[t:t+1]) for k,v in arrays.items()}
            online=context.observe(packet,torch.from_numpy(g['transition/nominal_predicted_chunk'][t:t+1]),
                                   torch.from_numpy(g['transition/nominal_chunk_cursor'][t:t+1]))
            if t in (0,1,9,11,12,100,len(g['obs/step_id'])-1):
                offline=default_collate([correction_data[offsets[t]]])['inputs']
                check_tree(online,offline)
            context.commit(torch.from_numpy(g['transition/executed_action'][t:t+1]))
    report['offline_online_inputs_exact']=True
    report['nominal_input_keys']=list(nominal_keys(2))
    report['actual_resolution']=metadata['cameras'][0]['resolution']
    report['limitations']=['Randomly initialized networks; no competence claim',
        'Finger origin projection is not contact-patch visibility or segmentation',
        'Aggregate normal force cannot provide pressure, tangential force or measured slip',
        'Task ID present; goal pose absent unless externally commanded; no hidden goal fallback']
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('nominal_output_shape','correction_output_shape','parameter_updates','checkpoint_roundtrip_exact','offline_online_inputs_exact')}))


if __name__=='__main__': main()
