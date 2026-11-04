"""Shared read-only paired trajectory, native event and FK diagnostics."""
from pathlib import Path
import numpy as np
import torch
from tools.runtime import read
from tools.h24_diagnosis_common import THRESHOLDS,events,contact_force
from visuotactile.kinematics import EncoderKinematics

def validated_fk(metadata):
    evidence=read(Path('research_docs/evidence/state_motion_20261103/fk_readonly/fk_validation.json'))
    fk=EncoderKinematics(metadata['config']['assets']['robot_urdf'],metadata['joint_order'],links=('iiwa14_link_7',))
    if not evidence['passed'] or fk.sha256!=evidence['urdf_sha256'] or fk.root!=evidence['root'] or metadata['joint_order']!=evidence['joint_order']:
        raise ValueError('FK asset/order/base validation mismatch')
    return fk
def quat_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def local_xyz(p, base):
    return np.einsum('...ji,...j->...i', quat_matrix(base[..., 3:]), p-base[..., :3])


def fk_pose(fk, q):
    shape = q.shape[:-1]
    with torch.no_grad():
        p, r = fk(torch.as_tensor(np.array(q, copy=True)).double().reshape(-1, 29))
    return p[:, 0].numpy().reshape(shape+(3,)), r[:, 0].numpy().reshape(shape+(3, 3))
def arrays(path):
    with np.load(path) as stream:
        return {k:stream[k] for k in stream.files}


def rms(values):
    values = np.asarray(values)
    return float(np.sqrt(np.mean(values**2))) if values.size else None


def distance_rms(a, b):
    return rms(np.linalg.norm(a-b,axis=-1))*1000


def pose_q(trace, fk):
    return fk_pose(fk, trace['joint'])[0], fk_pose(fk,trace['target'])[0]


def recorded_raw(path, n):
    file = path/'plans.npz'
    if not file.exists():
        return None
    p = arrays(file)
    raw = p.get('raw_plans', p['plans'])
    return raw[:,:12].reshape(-1,29)[:n]


def describe(path, reference=None):
    result = read(path/'result.json')
    trace = arrays(path/'trace.npz')
    metadata = read(path/'metadata.json')
    fk = validated_fk(metadata)
    actual, target = pose_q(trace,fk)
    n = len(actual)
    e = events(trace,result)
    mask = trace.get('command_is_position', np.full(n,result.get('action_control_mode')=='absolute_joint_position_target_rad'))
    tracking = trace['joint']-trace['target']
    clip = recorded_raw(path,n)
    clip_error = np.abs(clip-trace['command']) if clip is not None else None
    summary = dict(path=str(path),steps=n,result=result,events=e,
        target_actual_arm_rmse_rad=rms(tracking[:,:7]),target_actual_hand_rmse_rad=rms(tracking[:,7:]),
        target_actual_fk_mm=distance_rms(actual,target),
        actuator_target_max_error_rad=float(np.abs(trace['actuator_targets']-trace['target'][:,None,:]).max()),
        position_request_applied_max_error_rad=float(np.abs(trace['command'][mask]-trace['target'][mask]).max()) if mask.any() else None,
        clipping_max_rad=float(clip_error[mask].max()) if clip_error is not None and mask.any() else None,
        clipped_joint_ticks=int((clip_error[mask]>THRESHOLDS['clipping_tolerance_rad']).sum()) if clip_error is not None and mask.any() else None,
        command_units='position radians only where command_is_position; expert native command dimensionless',
        contact_points_available=False,tangential_slip_force_available=False)
    from visuotactile.state_motion import target_metrics
    summary['continuity'] = target_metrics(trace['target'])
    if clip_error is not None and mask.any():
        location=np.unravel_index(np.argmax(np.where(mask[:,None],clip_error,-1)),clip_error.shape)
        summary['clipping_detail']=dict(maximum_tick=int(location[0]),joint=metadata['joint_order'][location[1]],
            arm_max_rad=float(clip_error[mask,:7].max()),hand_max_rad=float(clip_error[mask,7:].max()),
            per_joint_clipped_tick_counts=(clip_error[mask]>THRESHOLDS['clipping_tolerance_rad']).sum(0).tolist(),
            definition='raw denormalized sample vs actual submitted bounded position; physical joint limits remain enabled even though DDIM clip_sample=False')
    seen_contact = np.maximum.accumulate(contact_force(trace)>.1)
    for name, selected in [('precontact',~seen_contact),('postcontact',seen_contact),('native_lift',trace['lifted'].astype(bool))]:
        summary[name] = dict(ticks=int(selected.sum()),arm_tracking_rmse_rad=rms(tracking[selected,:7]),
            hand_tracking_rmse_rad=rms(tracking[selected,7:]),
            target_actual_fk_mm=distance_rms(actual[selected],target[selected]) if selected.any() else None)
    jump = np.zeros((n,2))
    for i,sl in enumerate((slice(0,7),slice(7,29))):
        jump[1:,i] = np.sqrt(np.mean(np.diff(trace['target'][:,sl],axis=0)**2,axis=-1))
    series = dict(tick=np.arange(n), actual_xyz=actual,target_xyz=target,
        target_actual_fk_mm=np.linalg.norm(actual-target,axis=-1)*1000,
        arm_tracking_rad=np.sqrt(np.mean(tracking[:,:7]**2,axis=-1)),
        hand_tracking_rad=np.sqrt(np.mean(tracking[:,7:]**2,axis=-1)),
        arm_jump_rad=jump[:,0],hand_jump_rad=jump[:,1],object_contact_N=contact_force(trace),
        object_height_m=trace['object'][:,2],native_lift=trace['lifted'],subgoals=trace['subgoals'])
    if reference is not None:
        ref = arrays(reference/'trace.npz')
        r_actual,r_target = pose_q(ref,fk)
        count = min(n,len(r_actual))
        summary['reference'] = dict(path=str(reference),alignment='same control tick, common prefix only; no time warping',
            ticks=count,actual_expert_fk_mm=distance_rms(actual[:count],r_actual[:count]),
            target_expert_target_fk_mm=distance_rms(target[:count],r_target[:count]),
            target_expert_arm_rmse_rad=rms(trace['target'][:count,:7]-ref['target'][:count,:7]),
            target_expert_hand_rmse_rad=rms(trace['target'][:count,7:]-ref['target'][:count,7:]),
            actual_expert_arm_rmse_rad=rms(trace['joint'][:count,:7]-ref['joint'][:count,:7]))
        before = np.cumsum(contact_force(ref)[:count]>.1)==0
        summary['reference']['expert_precontact_ticks'] = int(before.sum())
        summary['reference']['actual_expert_precontact_fk_mm'] = distance_rms(actual[:count][before],r_actual[:count][before]) if before.any() else None
        for key,values in [('actual_expert_fk_mm',np.linalg.norm(actual[:count]-r_actual[:count],axis=-1)*1000),
                ('target_expert_fk_mm',np.linalg.norm(target[:count]-r_target[:count],axis=-1)*1000)]:
            series[key] = np.r_[values,np.full(n-count,np.nan)]
    # Contact configuration is observable; fingertip location is not a contact point.
    if 'palm' in trace:
        relative = local_xyz(trace['object'][:,:3],trace['palm'])
        series['object_palm_motion_mm'] = np.r_[0,np.linalg.norm(np.diff(relative,axis=0),axis=-1)*1000]
        summary['object_palm_motion_p95_mm_per_tick'] = float(np.quantile(series['object_palm_motion_mm'],.95))
    if clip_error is not None:
        series['raw_request_arm_rad'] = np.sqrt(np.mean(clip_error[:,:7]**2,axis=-1))
        series['raw_request_hand_rad'] = np.sqrt(np.mean(clip_error[:,7:]**2,axis=-1))
    return summary, series, trace


def compare_initial(a,b):
    initial_a,initial_b = read(a/'initial_observable.json'),read(b/'initial_observable.json')
    differences = {k:float(np.abs(np.asarray(initial_a[k],dtype=float)-np.asarray(initial_b[k],dtype=float)).max())
        for k in initial_a if k in initial_b}
    records = []
    def compare(x,y,name):
        if isinstance(x,dict) and isinstance(y,dict):
            for key in sorted(set(x)|set(y)):
                if key in x and key in y:compare(x[key],y[key],name+'/'+str(key))
                else:records.append(dict(path=name+'/'+str(key),equal=False,reason='missing'))
        elif isinstance(x,(list,tuple)) and isinstance(y,(list,tuple)):
            if len(x)!=len(y):records.append(dict(path=name,equal=False,reason='length'))
            else:
                for i,(u,v) in enumerate(zip(x,y)):compare(u,v,name+'/'+str(i))
        elif torch.is_tensor(x) and torch.is_tensor(y):
            equal=x.shape==y.shape and x.dtype==y.dtype and torch.equal(x,y)
            row=dict(path=name,equal=equal,shape=list(x.shape))
            if not equal and x.shape==y.shape and x.numel():row['max_absolute_difference']=float((x.double()-y.double()).abs().max())
            records.append(row)
        elif isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
            records.append(dict(path=name,equal=np.array_equal(x,y)))
        else:
            records.append(dict(path=name,equal=x==y))
    for file in ['initial_export.pt','environment_rng_initial.pt']:
        compare(torch.load(a/file,map_location='cpu',weights_only=False),torch.load(b/file,map_location='cpu',weights_only=False),file)
    return dict(observable_max_errors=differences,observable_equal=all(v==0 for v in differences.values()),
        exported_fields_checked=len(records),exported_differences=[r for r in records if not r['equal']],
        environment_rng_equal=all(r['equal'] for r in records if r['path'].startswith('environment_rng_initial.pt')),
        missing='PhysX solver/contact warm-start, broadphase and other unsaved internal state; matching observations/seeds do not prove identical simulator state')
def tick_metrics(trace,raw=None):
    n=len(trace['target']);delta=np.diff(trace['target'],axis=0,prepend=trace['target'][:1])
    return dict(arm_jump=np.sqrt(np.mean(delta[:,:7]**2,axis=-1)),
        hand_jump=np.sqrt(np.mean(delta[:,7:]**2,axis=-1)),
        arm_tracking=np.sqrt(np.mean((trace['target'][:,:7]-trace['joint'][:,:7])**2,axis=-1)),
        hand_tracking=np.sqrt(np.mean((trace['target'][:,7:]-trace['joint'][:,7:])**2,axis=-1)),
        contact=contact_force(trace),object_z=trace['object'][:,2],
        clipping=np.max(np.abs(raw-trace['target']),axis=-1) if raw is not None else np.full(n,np.nan))


def window_summary(series,n):
    ticks=np.arange(n);boundary=(ticks>0)&(ticks%12==0);internal=(ticks>0)&~boundary
    out={}
    for name in ('arm_jump','hand_jump'):
        for label,mask in [('boundary',boundary),('internal',internal)]:
            out[name+'_'+label+'_rms_rad']=rms(series[name][:n][mask])
    for name in ('arm_tracking','hand_tracking'):
        out[name+'_rms_rad']=rms(series[name][:n])
    return out
