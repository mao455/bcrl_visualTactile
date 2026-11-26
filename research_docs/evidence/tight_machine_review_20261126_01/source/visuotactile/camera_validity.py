"""Actual RGB-D content acceptance, independent of sensor presence flags.

Thresholds reject blank views, not task failures. Visibility projections are
diagnostics only; neither object poses nor these statistics enter State/Image inputs.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import torch


def content_valid(rgb, depth, valid):
    """Per-camera batches N,C,H,W; no variation across channels as fake texture."""
    # Use the same full-pixel criterion online and in offline acceptance.
    spatial=rgb.flatten(-2).std(-1,correction=0).amax(-1)
    coverage=(valid & torch.isfinite(depth) & (depth>0)).float().flatten(1).mean(-1)
    return (spatial>1/255) & (coverage>.01) & torch.isfinite(rgb).flatten(1).all(-1)


def frame_metrics(rgb, depth, valid):
    # V,C,H,W. Source RGB is lossless uint8; source depth is metric float32.
    spatial=rgb.astype(np.float32).reshape(len(rgb),3,-1).std(-1).max(-1)
    mask=np.asarray(valid,dtype=bool)
    expected=np.isfinite(depth)&(depth>0)
    fraction=expected.reshape(len(rgb),-1).mean(-1)
    good=(spatial>1)&(fraction>.01)
    return dict(rgb_spatial_std_u8=spatial.tolist(),depth_valid_fraction=fraction.tolist(),
        mask_exact=bool(np.array_equal(mask,expected)),content_valid=good.tolist())


def inspect_capture(path, *, preview=None):
    """All frames/all cameras; no constant-image acceptance from camera_valid=True."""
    from tools.runtime import write
    from scipy.spatial.transform import Rotation
    rows=[];projection=[]
    with h5py.File(path) as f:
        meta=json.loads(f.attrs['metadata_json']);g=next(iter(f['episodes'].values()))
        n=int(g.attrs['length']);poses=g['obs/camera_pose_w'][:]
        stamp=g['obs/time_s'][:];camstamp=g['obs/camera_time_s'][:]
        frame=g['obs/camera_frame'][:]
        if not np.allclose(camstamp,stamp[:,None],atol=1e-8,rtol=0):raise ValueError('Camera acquisition clock mismatch')
        if n>1 and not np.all(np.diff(frame,axis=0)>0):raise ValueError('Stale/nonmonotonic camera frame counter')
        constant_pose=float(np.abs(poses[:,:2]-poses[0:1,:2]).max())
        force=g['transition/object_contact_force_w'][:]
        contact=np.linalg.norm(force,axis=-1).reshape(n,-1).max(-1)>.1
        contact_ids=np.flatnonzero(contact)
        middle=int(contact_ids[len(contact_ids)//2]) if len(contact_ids) else n//2
        sampled=sorted(set([0,middle,n-1]));pictures=[]
        for t in range(n):
            rgb=g['frames/rgb_u8'][t];depth=g['frames/depth_m'][t];mask=g['frames/depth_valid'][t]
            row=frame_metrics(rgb,depth,mask);rows.append(row)
            if t in sampled:
                pictures.append((t,rgb.transpose(0,2,3,1)))
                points=np.stack([g['transition/diagnostics_before/'+key][t,:3] for key in ('object_pose_w','hole_pose_w')])
                projected=[]
                for k,camera in enumerate(meta['cameras']):
                    pose=poses[t,k];rot=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
                    local=(points-pose[:3])@rot
                    k_mat=np.asarray(camera['intrinsics'][0]);pixels=local@k_mat.T;uv=pixels[:,:2]/pixels[:,2:]
                    projected.append(dict(view=camera['id'],uv=uv.tolist(),z=local[:,2].tolist(),
                        in_frame=((local[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<448)&(uv[:,1]>=0)&(uv[:,1]<252)).tolist()))
                projection.append(dict(tick=t,views=projected))
        flags=g['obs/camera_valid'][:]
        invalid=np.array([r['content_valid'] for r in rows])==False
        change=np.max([np.abs(rgb.astype(float)-pictures[0][1].astype(float)).reshape(3,-1).mean(-1)
            for _,rgb in pictures[1:]],axis=0) if len(pictures)>1 else np.zeros(3)
        report=dict(passed=bool(not invalid.any() and all(r['mask_exact'] for r in rows) and flags.all() and constant_pose<2e-5 and (change>1e-3).all()),
            frames=n,views=meta['camera_ids'],invalid_frames_per_view=invalid.sum(0).tolist(),
            minimum_rgb_spatial_std_u8=np.min([r['rgb_spatial_std_u8'] for r in rows],axis=0).tolist(),
            minimum_depth_valid_fraction=np.min([r['depth_valid_fraction'] for r in rows],axis=0).tolist(),
            depth_mask_exact=all(r['mask_exact'] for r in rows),camera_flags_all_true=bool(flags.all()),
            fixed_pose_max_drift=constant_pose,first_fixed_pose=poses[0,:2].tolist(),sampled_target_projection=projection,
            sampled_rgb_change_mean_abs_u8=change.tolist(),
            synchronized_clock=True,frames_strictly_increase=True,
            scope='All RGB/depth pixels, all pre-action frames. Projection checks field of view, not occlusion; preview requires visual review.')
        if preview:
            from PIL import Image,ImageDraw
            canvas=Image.new('RGB',(3*448,len(pictures)*(252+28)),(24,24,24));d=ImageDraw.Draw(canvas)
            for j,(t,images) in enumerate(pictures):
                for k,rgb in enumerate(images):
                    d.text((k*448+8,j*280+7),f'{meta["camera_ids"][k]} | tick {t} / {n}',fill='white')
                    canvas.paste(Image.fromarray(rgb),(k*448,j*280+28))
            canvas.save(preview)
        write(Path(path).parent/'visual_acceptance.json',report)
        return report
