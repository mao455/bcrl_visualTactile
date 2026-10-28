"""Offline camera evidence from committed raw frames; no simulation or training.

Projection is diagnostic, never proof of visibility. Full videos and regular
contact sheets complement the raw RGB/depth/validity exports for human review.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from .compact import EpisodeFile, write_video
from .trial_utils import write_json


def data_path(directory):
    directory = Path(directory)
    return directory / ('trajectories.h5' if (directory/'trajectories.h5').exists() else 'data.h5')


def compare(reference, candidate):
    """Exact comparison of all non-image numerical fields, including teacher/RNG."""
    with EpisodeFile(data_path(reference)) as a, EpisodeFile(data_path(candidate)) as b:
        ma, mb = [json.loads(f.attrs['metadata_json']) for f in (a,b)]
        ga, gb = [next(iter(f['episodes'].items()))[1] for f in (a,b)]
        fields = []
        def visit(name, obj):
            if hasattr(obj, 'dtype') and not any(part in name.split('/') for part in
                    ('image', 'depth_m', 'depth_valid', 'camera_pose_w', 'camera_parent_pose_w',
                     'camera_frame', 'camera_valid', 'camera_time_s')):
                fields.append(name)
        ga.visititems(visit)
        results = {}
        for name in fields:
            x, y = ga[name][:], gb[name][:]
            same_shape = x.shape == y.shape
            results[name] = dict(equal=same_shape and np.array_equal(x,y), shape_a=list(x.shape),shape_b=list(y.shape),
                max_abs_difference=float(np.max(np.abs(x.astype(float)-y.astype(float)),initial=0)) if same_shape else None)
        physical_sections = ('action','assets','domain_randomization','obs','reset','reward','termination','precise_assembly')
        metadata_equal = all(ma['config'][k]==mb['config'][k] for k in physical_sections)
        for key in ('seed','expert_model_sha256','initial_torch_rng_sha256','physics_dt','control_dt','wrench_frame_mode'):
            metadata_equal &= ma.get(key)==mb.get(key)
        return dict(reference=str(reference),candidate=str(candidate),fields=results,
            physical_protocol_equal=bool(metadata_equal),exact=bool(metadata_equal and all(r['equal'] for r in results.values())),
            interpretation='No image comparison; equality applies only to these recorded arrays in this bounded run.')


def review(directory):
    directory=Path(directory);out=directory/'review';out.mkdir(exist_ok=False)
    with EpisodeFile(data_path(directory)) as f:
        meta=json.loads(f.attrs['metadata_json']);g=next(iter(f['episodes'].items()))[1]
        n=int(g.attrs['length']);ids=meta['camera_ids'];h,w=meta['cameras'][0]['resolution'];v=len(ids)
        indices=sorted(set([0,*range(12,n,12),n-1]))
        sheet=Image.new('RGB',(2*w*v,(2*h+22)*len(indices)),'white');draw=ImageDraw.Draw(sheet)
        red_counts=[];depth_fractions=[]
        def frames():
            for t in range(n):
                im=g['obs/image'][t];rgb=np.rint(im[:,:3]*255).astype('u1').transpose(0,2,3,1)
                red=(rgb[...,0].astype(float)>rgb[...,1]+20.)&(rgb[...,0].astype(float)>rgb[...,2]+20.)
                red_counts.append(red.sum((1,2)).tolist())
                depth_fractions.append(g['obs/depth_valid'][t].mean((1,2,3)).tolist())
                if t in indices:
                    r=indices.index(t)
                    for j,identity in enumerate(ids):
                        base=out/f'{t:04d}_{identity}'
                        Image.fromarray(rgb[j]).save(str(base)+'_rgb.png')
                        depth=im[j,3];valid=g['obs/depth_valid'][t,j,0]
                        Image.fromarray((np.clip(depth/1.5,0,1)*255).astype('u1')).save(str(base)+'_depth_preview.png')
                        Image.fromarray((valid*255).astype('u1')).save(str(base)+'_valid.png')
                        np.save(str(base)+'_depth_m.npy',depth)
                        sheet.paste(Image.fromarray(rgb[j]).resize((2*w,2*h),resample=Image.Resampling.NEAREST),(2*w*j,(2*h+22)*r))
                        draw.text((2*w*j,(2*h+22)*r+2*h),f'{identity} step {t}  {t/60:.3f}s',fill='black')
                canvas=Image.new('RGB',(w*v,h+24),'white');d=ImageDraw.Draw(canvas)
                for j,identity in enumerate(ids):
                    canvas.paste(Image.fromarray(rgb[j]),(w*j,0));d.text((w*j,h),f'{identity} {t/60:.3f}s',fill='black')
                yield np.asarray(canvas)
        write_video(out/'synchronized.mp4',frames(),fps=60)
        sheet.save(out/'all_stages.png')
        # Colour counts describe this red asset, not universal semantic segmentation.
        counts=np.asarray(red_counts)
        result=dict(steps=n,ids=ids,raw_resolution=[h,w],sampled_steps=indices,
            frame_clock='stored pre-action observation; native 60 Hz, video 1x; terminal image missing',
            red_asset_pixel_counts_by_step=red_counts,
            red_pixel_quantiles_by_view=np.quantile(counts,[0,.25,.5,.75,1],axis=0).tolist(),
            depth_valid_fraction_by_step=depth_fractions,
            red_count_limit='Red-dominance pixels of current asset; occlusion and lighting affect count; not learned segmentation or hole visibility.')
        write_json(out/'index.json',result)
    return result


def clearance(directory):
    """Conservative mesh-AABB distance along the actually executed trajectory.

    Positive distance guarantees optical point lies outside each enclosing box.
    Zero is inconclusive. Does not certify a real housing/bracket or all motions.
    """
    import torch
    import trimesh
    from .kinematics import EncoderKinematics, rpy_matrix
    from .initial_geometry import boxes
    with EpisodeFile(data_path(directory)) as f:
        m=json.loads(f.attrs['metadata_json']);g=next(iter(f['episodes'].items()))[1]
        if 'wrist' not in m.get('camera_rig',{}): raise ValueError('Not a wrist rig')
        urdf=Path(m['config']['assets']['robot_urdf']);root=ET.parse(urdf).getroot()
        entries=[]
        for link in root.findall('link'):
            for visual in link.findall('visual')+link.findall('collision'):
                mesh=visual.find('geometry/mesh')
                if mesh is None: continue
                filename=mesh.get('filename');file=urdf.parent/filename
                if filename.startswith('package://'): raise ValueError('Resolve package paths explicitly')
                vertices=np.asarray(trimesh.load(file,force='mesh',process=False).vertices)
                vertices*=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                origin=visual.find('origin')
                xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
                rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
                vertices=vertices@rpy_matrix(rpy).numpy().T+xyz
                entries.append((link.get('name'),vertices.min(0),vertices.max(0)))
        links=sorted({x[0] for x in entries});fk=EncoderKinematics(urdf,m['joint_order'],links=links)
        q=torch.from_numpy(g['transition/diagnostics_before/joint_position_rad'][:])
        pos,rot=fk(q);pos,rot=pos.numpy(),rot.numpy()
        base=Rotation.from_quat(np.array(m['base_orientation_wxyz'])[[1,2,3,0]])
        centre=g['obs/camera_pose_w'][:,2,:3]
        local=base.inv().apply(centre-np.array(m['base_position_env']))
        distances={}
        for name,lo,hi in entries:
            j=links.index(name);point=np.einsum('nji,nj->ni',rot[:,j],local-pos[:,j])
            distance=np.linalg.norm(np.maximum(np.maximum(lo-point,point-hi),0),axis=-1)
            distances[name]=min(distances.get(name,float('inf')),float(distance.min()))
        box_distances={}
        for name,asset in [('object','object_urdf'),('table','table_urdf'),('hole','receptive_urdf')]:
            ds=[]
            for t,p in enumerate(g['transition/diagnostics_before/'+name+'_pose_w'][:]):
                for c,r,half in boxes(m['config']['assets'][asset],p):
                    ds.append(np.linalg.norm(np.maximum(np.abs(r.T@(centre[t]-c))-half,0)))
            box_distances[name]=float(min(ds))
        return dict(robot_visual_and_collision_AABB_min_distance_m=distances,
            object_table_hole_box_min_distance_m=box_distances,
            min_robot_m=min(distances.values()),frames=len(q),
            limitation='Optical centre only, conservative mesh AABBs, actual sampled motion only; no CAD bracket/housing/cable or full joint envelope certification.')



def aperture_diagnostics(directory):
    """Depth-tested interior of this task's URDF hole opening; diagnostic truth.

    A clear-depth pixel is not proof that edges are visually distinguishable.
    The result must be read beside RGB, especially during peg occlusion.
    """
    with EpisodeFile(data_path(directory)) as f:
        m=json.loads(f.attrs['metadata_json']);g=next(iter(f['episodes'].items()))[1]
        if 'camera_pose_w' not in g['obs']: raise ValueError('Requires per-frame calibrated rig3')
        root=ET.parse(m['config']['assets']['receptive_urdf']).getroot()
        blocks=[]
        for c in root.findall('.//collision'):
            size=np.fromstring(c.find('geometry/box').get('size'),sep=' ')
            centre=np.fromstring(c.find('origin').get('xyz'),sep=' ')
            blocks.append((centre,size))
        # Explicit audit for the released hole_tol0p5mm geometry; reject drift.
        assert len(blocks)==5 and abs(max(c[2]+s[2]/2 for c,s in blocks)-.06)<1e-7
        x,y=np.meshgrid(np.linspace(-.0095,.0095,32),np.linspace(-.0145,.0145,40))
        patch=np.c_[x.ravel(),y.ravel(),np.full(x.size,.06)]
        h,w=m['cameras'][0]['resolution'];rows=[]
        for t,hole in enumerate(g['transition/diagnostics_before/hole_pose_w'][:]):
            world=Rotation.from_quat(hole[[4,5,6,3]]).apply(patch)+hole[:3]
            views=[]
            for v,camera in enumerate(m['cameras']):
                pose=g['obs/camera_pose_w'][t,v];local=Rotation.from_quat(pose[[4,5,6,3]]).inv().apply(world-pose[:3])
                k=np.array(camera['intrinsics'][0]);uv=local@k.T;uv=uv[:,:2]/uv[:,2:]
                pixel=np.rint(uv).astype(int);inside=(local[:,2]>0)&(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
                pix,indices=np.unique(pixel[inside],axis=0,return_index=True);z=local[inside,2][indices]
                depth=g['obs/depth_m'][t,v,0];valid=g['obs/depth_valid'][t,v,0]
                measured=depth[pix[:,1],pix[:,0]];good=valid[pix[:,1],pix[:,0]]
                views.append(dict(projected_extent_px=np.ptp(uv,axis=0).tolist(),in_frame_pixels=len(pix),
                    clear_depth_pixels=int(np.sum(good&(measured>z+.003))),
                    foreground_occluded_pixels=int(np.sum(good&(measured<z-.005))),
                    invalid_pixels=int(np.sum(~good))))
            rows.append(dict(step=t,views=views))
        return dict(rows=rows,opening_m=[.021,.031],sampled_interior_m=[.019,.029],
            limitation='Interior projection deduplicated at pixel centres, depth test 3mm behind/5mm ahead. Rim aliasing, inserted peg and denoising require RGB inspection; not visibility ground truth. Never policy input.')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['review','compare','clearance','aperture']);p.add_argument('paths',nargs='+',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    result=(compare(*a.paths) if a.mode=='compare' else aperture_diagnostics(a.paths[0]) if a.mode=='aperture' else clearance(a.paths[0]) if a.mode=='clearance' else review(a.paths[0]))
    if a.output: write_json(a.output,result)
    else: print(json.dumps(result,indent=2))


if __name__=='__main__': main()
