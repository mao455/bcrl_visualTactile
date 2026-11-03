"""CPU-only evidence analysis and real MP4 delivery for the bounded H24 run."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from tools.h24_diagnosis_common import OLD, ROOT, TRAIN, THRESHOLDS, read, sha, events, first_sustained, contact_force
from tools.h24_comparison_report import validated_fk
from tools.state_fk_audit import fk_pose, local_xyz
from visuotactile.training_resources import atomic_json

PUBLIC = Path('research_docs/evidence/h24_diagnosis_20261104')
REPORT = Path('research_docs/H24_CLOSED_LOOP_DIAGNOSIS.md')


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


def pair_analysis(a,b):
    x,y = arrays(a/'trace.npz'),arrays(b/'trace.npz')
    n=min(len(x['joint']),len(y['joint']))
    fk=validated_fk(read(a/'metadata.json'))
    ee1,_=pose_q(x,fk);ee2,_=pose_q(y,fk)
    angle=np.abs(np.sum(x['object'][:n,3:]*y['object'][:n,3:],axis=-1)).clip(0,1)
    metrics=dict(ee_mm=np.linalg.norm(ee1[:n]-ee2[:n],axis=-1)*1000,
        arm_rad=np.sqrt(np.mean((x['joint'][:n,:7]-y['joint'][:n,:7])**2,axis=-1)),
        hand_rad=np.sqrt(np.mean((x['joint'][:n,7:]-y['joint'][:n,7:])**2,axis=-1)),
        object_mm=np.linalg.norm(x['object'][:n,:3]-y['object'][:n,:3],axis=-1)*1000,
        object_rotation_deg=np.rad2deg(2*np.arccos(angle)))
    thresholds={'ee_mm':5.,'arm_rad':.01,'hand_rad':.05,'object_mm':5.,'object_rotation_deg':5.}
    first={key:first_sustained(value,thresholds[key]) for key,value in metrics.items()}
    raw1,raw2=recorded_raw(a,n),recorded_raw(b,n)
    return dict(initial=compare_initial(a,b),compared_ticks=n,first_sustained_divergence=first,
        thresholds=thresholds,consecutive_ticks=3,
        first_command_difference_tick=first_sustained(np.max(np.abs(x['command'][:n]-y['command'][:n]),axis=-1),1e-6,1),
        first_raw_target_difference_tick=first_sustained(np.max(np.abs(raw1-raw2),axis=-1),1e-6,1),
        initial_raw_target_arm_difference_rad=rms(raw1[0,:7]-raw2[0,:7]),
        initial_raw_target_hand_difference_rad=rms(raw1[0,7:]-raw2[0,7:]))


def offline_metrics(root,identity):
    session=read(OLD/'session.json')
    data=read(Path(session['source_root'])/'data.json')
    entry=data['entries'][identity]
    with h5py.File(entry['path']) as stream:
        meta=json.loads(stream.attrs['metadata_json'])
        g=stream['episodes'][entry['episode']]
        source_joint=g['transition/diagnostics_after/joint_position_rad'][:]
        source_target=g['transition/target_after_rad'][:]
    fk=validated_fk(meta)
    rows={}
    for updates in (8000,10000):
        file=OLD/f'sample_{updates}_offline/numeric_plans/{identity}.npz'
        if not file.exists():continue
        p=arrays(file)
        name=next(k for k in p if k not in ('anchors','truth','valid','before','contact'))
        predicted=p[name][:,:12];truth=p['truth'][:,:12];valid=p['valid'][:,:12]
        pred_xyz=fk_pose(fk,predicted)[0];gt_xyz=fk_pose(fk,truth)[0]
        rows[str(updates)]=dict(source=str(file),sha256=sha(file),configuration=name,
            valid_positions=int(valid.sum()),arm_rmse_rad=rms((predicted-truth)[valid][:,:7]),
            hand_rmse_rad=rms((predicted-truth)[valid][:,7:]),
            target_fk_rmse_mm=distance_rms(pred_xyz[valid],gt_xyz[valid]),
            conditions='original recorded expert history; existing batch offline prediction, seed7091; no new inference',
            source_demonstration_actual_target_tracking_fk_mm=distance_rms(fk_pose(fk,source_joint)[0],fk_pose(fk,source_target)[0]))
        recorded_metrics=read(OLD/f'sample_{updates}_offline/result.json')['rows']
        rows[str(updates)]['existing_motion_metrics']=[{k:v for k,v in row.items()
            if any(w in k for w in ('cosine','amplitude','moving12','velocity'))}
            for row in recorded_metrics if row['identity']==identity]
        # Compare autonomous execution to exactly the same original expert recording.
        student=arrays(OLD/f'sample_{updates}_loop'/identity/'trace.npz')
        n=min(len(source_joint),len(student['joint']))
        rows[str(updates)]['same_source_autonomous']=dict(ticks=n,
            target_expert_target_fk_mm=distance_rms(fk_pose(fk,student['target'][:n])[0],fk_pose(fk,source_target[:n])[0]),
            actual_expert_actual_fk_mm=distance_rms(fk_pose(fk,student['joint'][:n])[0],fk_pose(fk,source_joint[:n])[0]),
            target_actual_fk_mm=distance_rms(fk_pose(fk,student['target'][:n])[0],fk_pose(fk,student['joint'][:n])[0]),
            limitation='Common episode identity and tick, not exact historical four-environment hidden state; no subtraction of aggregate RMSEs')
    return rows


def draw_cohort(name, entries, summaries, series):
    fig=plt.figure(figsize=(13,11))
    ax=fig.add_subplot(321,projection='3d')
    other=[fig.add_subplot(3,2,i) for i in range(2,7)]
    for label,key in entries:
        s=series[key];time=(s['tick']+1)/60
        p=s['actual_xyz']*1000
        ax.plot(p[:,0],p[:,1],p[:,2],label=label)
        for axis,field in zip(other,['actual_expert_fk_mm','target_actual_fk_mm','arm_jump_rad','hand_jump_rad','object_contact_N']):
            if field in s:axis.plot(time,s[field],label=label,linewidth=1)
    ax.set(title='Actual link7 FK in robot base (mm)',xlabel='x',ylabel='y',zlabel='z');ax.legend(fontsize=8)
    for axis,title,unit in zip(other,['Actual vs online expert: same tick','Applied target vs reached actual','Arm target jumps','Hand target jumps','Object-filtered contact'],['mm','mm','rad RMS','rad RMS','N']):
        axis.set(title=title,xlabel='simulation seconds after control tick',ylabel=unit)
        axis.legend(fontsize=7);axis.grid(alpha=.2)
    for axis in other[2:4]:
        for t in np.arange(0, max(len(series[k]['tick']) for _,k in entries)/60,.2):axis.axvline(t,color='gray',alpha=.07)
    fig.tight_layout();fig.savefig(PUBLIC/f'{name}.png',dpi=145);plt.close(fig)


def control_plot(cohort,series):
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for column,updates in enumerate((8000,10000)):
        s=series[f'historical_{cohort}_{updates}'];t=(s['tick']+1)/60
        for key,label in [('raw_request_arm_rad','Arm'),('raw_request_hand_rad','Hand')]:
            if key in s:axes[0,column].plot(t,s[key],label=label)
        for key,label in [('arm_tracking_rad','Arm'),('hand_tracking_rad','Hand')]:axes[1,column].plot(t,s[key],label=label)
        for key,label in [('target_expert_fk_mm','predicted target vs expert target'),('actual_expert_fk_mm','reached vs expert actual')]:
            if key in s:axes[2,column].plot(t,s[key],label=label)
        axes[0,column].set_title(f'Sample {updates}: raw target vs submitted position')
        axes[1,column].set_title('Applied actuator target vs reached joint position')
        axes[2,column].set_title('Same-tick online expert comparison')
        for row in range(3):
            axes[row,column].grid(alpha=.2);axes[row,column].legend(fontsize=8)
            axes[row,column].set_ylabel('rad RMS' if row<2 else 'mm')
        axes[2,column].set_xlabel('simulation seconds after control tick')
    fig.tight_layout();fig.savefig(PUBLIC/f'control_chain_{cohort}.png',dpi=145);plt.close(fig)


def input_shift(root,results,identity):
    from visuotactile.state_history import load_records
    from visuotactile.position_control import contract
    source=Path(read(OLD/'session.json')['source_root'])
    record=load_records(source,[identity],contract(24,True))[identity]
    stats=torch.load(OLD/'position_train_statistics.pt',map_location='cpu',weights_only=False)
    rows={}
    for name in ('V4','V5'):
        result=results.get(name,{})
        if result.get('state')!='completed':continue
        file=Path(result['path'])/'policy_inputs.h5'
        if not file.exists():continue
        with h5py.File(file) as stream:
            group=stream['episode'];n=min(len(group['robot']),len(record['robot']))
            row={}
            for field in ('robot','geometry'):
                difference=group[field][:n]-record[field][:n].numpy()
                standard=stats[field+'_std'].numpy()
                normalized=difference/np.maximum(standard,1e-6)
                row[field+'_standardized_rms']=rms(normalized)
                row[field+'_tick0_standardized_rms']=rms(normalized[0])
            row.update(compared_ticks=n,reference='original expert recording; same episode identity/tick',
                interpretation='Observed feature-history difference, not independent proof of a distribution-shift causal mechanism')
            rows[name]=row
    return rows


def validate_video(path):
    import av
    media=read(Path(read(OLD/'session.json')['source_root'])/'video_tools.json')['ffprobe']
    if sha(media['path']) != media['sha256']:
        raise ValueError('Registered ffprobe binary changed')
    info=json.loads(subprocess.check_output([media['path'],'-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size','-of','json',str(path)],text=True))
    frames=0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frames==0:assert frame.to_ndarray(format='rgb24').size>0
            frames+=1
    assert frames>0 and info['streams'][0]['codec_name']=='h264'
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path),
        duration_s=float(info['format']['duration']),decoded_frames=frames,stream=info['streams'][0])


def decorate(raw,path,name,record,series,events_):
    import av
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    result=record['result'];stage='success' if result['success'] else next((k for k,v in result['reasons'].items() if v),'censored')
    desired={0,max(0,(events_.get('contact_tick') or 0)//2),max(0,(events_.get('lift_tick') or result['steps']-1)//2)}
    count=0
    with av.open(str(raw)) as source,av.open(str(path),'w') as output:
        stream=output.add_stream('libx264',rate=30);stream.width=1280;stream.height=720
        stream.pix_fmt='yuv420p';stream.codec_context.thread_count=2;stream.options={'crf':'18','preset':'fast'}
        for frame in source.decode(video=0):
            image=frame.to_image();draw=ImageDraw.Draw(image)
            tick=count*2;previous=max(0,min(tick-1,len(series['tick'])-1))
            lines=[f'{name}: '+('Expert' if record['updates'] is None else f'Sample {record["updates"]}'),
                f'FINAL OUTCOME: {stage}; {result["steps"]} ticks',
                f'pre-action image tick {tick} | {tick/60:.3f} s',
                f'seed {result["seed"]}; policy seed7091',
                f'checkpoint {result["checkpoint_sha256"][:12]}',
                result['identity'],
                f'prefix {result.get("expert_prefix_ticks",0)} ticks; no state injection',
                f'previous target/actual FK: {series["target_actual_fk_mm"][previous]:.1f} mm',
                f'previous object contact: {series["object_contact_N"][previous]:.2f} N',
                'Views: top left global; top right side; lower left wrist',
                '1x simulation time; no post-terminal/reset image']
            for i,line in enumerate(lines):draw.text((650,372+i*29),line,font=font if i<3 else small,fill='white')
            if count in desired:
                image.save(PUBLIC/f'{name}_tick{tick:04d}.jpg',quality=88)
            out=av.VideoFrame.from_image(image);out.pts=count
            for packet in stream.encode(out):output.mux(packet)
            count+=1
        for packet in stream.encode():output.mux(packet)


def reconstruction(root,historical):
    """Numerical 3D visualization, explicitly not a camera or physical replay."""
    import av
    key='historical_train_8000'
    summary,series,trace=historical[key]
    path=root/'videos/historical_8000_success_NUMERIC_RECONSTRUCTION.mp4'
    fig=plt.figure(figsize=(12.8,7.2),dpi=100)
    actual=series['actual_xyz'];target=series['target_xyz']
    initial=read(Path(summary['path'])/'initial_observable.json')
    obj=local_xyz(trace['object'][:,:3],np.broadcast_to(np.asarray(initial['base_pose_w']).reshape(7),(len(actual),7)))
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libx264',rate=30);stream.width=1280;stream.height=720;stream.pix_fmt='yuv420p'
        stream.codec_context.thread_count=2;stream.options={'crf':'20','preset':'fast'}
        for i,tick in enumerate(range(0,len(actual),2)):
            fig.clear();ax=fig.add_subplot(121,projection='3d');text=fig.add_subplot(122);text.axis('off')
            for xyz,label,color in [(actual,'actual link7','blue'),(target,'target FK','orange'),(obj,'object center','green')]:
                ax.plot(*xyz[:tick+1].T,label=label,color=color);ax.scatter(*xyz[tick],color=color)
            combined=np.concatenate([actual,target,obj]);lo=combined.min(0)-.03;hi=combined.max(0)+.03
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='base x (m)',ylabel='base y (m)',zlabel='base z (m)');ax.legend()
            text.text(0,.9,'NUMERICAL RECONSTRUCTION\nNOT an original success camera video\nNOT a physical simulator replay',fontsize=17,color='darkred')
            text.text(0,.6,f'Historical Sample8000 | seed982403\n{TRAIN}\nRecorded native success:342ticks\nShown tick:{tick} / time:{tick/60:.3f}s\nLift:{bool(trace["lifted"][tick])}; subgoals:{int(trace["subgoals"][tick])}',fontsize=14)
            text.text(0,.18,'Only recorded FK/object centers are shown.\nNo reconstructed contact geometry, solver state,\nRGB scene or unrecorded physical motion.',fontsize=12)
            fig.tight_layout();fig.canvas.draw()
            frame=av.VideoFrame.from_ndarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy(),format='rgb24');frame.pts=i
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    plt.close(fig)
    return dict(kind='numeric_reconstruction_not_camera_or_physics',**validate_video(path))


def run(root):
    PUBLIC.mkdir(parents=True,exist_ok=True)
    plan=read(root/'plan.json');results=read(root/'results.json') if (root/'results.json').exists() else {}
    old_session=read(OLD/'session.json');expert_root=Path(old_session['previous_root'])/'expert_learning'
    summary=dict(thresholds=plan['thresholds'],registration_sha256=sha(root/'plan.json'),new_optimizer_updates=0,
        source_commit=read(root/'frozen.json')['source_commit'],historical={},new={},pairs={},offline={},videos={},missing_videos=[])
    all_data={};summaries={};series={}
    for cohort,spec in [('train',plan['train']),('validation',plan['validation'])]:
        identity=spec['identity'];expert=expert_root/identity
        for label,path in [('expert',expert),('8000',OLD/'sample_8000_loop'/identity),('10000',OLD/'sample_10000_loop'/identity)]:
            key=f'historical_{cohort}_{label}'
            value=describe(path,expert)
            all_data[key]=value;summaries[key]=value[0];series[key]=value[1];summary['historical'][key]=value[0]
        summary['pairs'][cohort]=pair_analysis(OLD/'sample_8000_loop'/identity,OLD/'sample_10000_loop'/identity)
        draw_cohort(f'historical_{cohort}',[(label,f'historical_{cohort}_{label}') for label in ('expert','8000','10000')],summaries,series)
        control_plot(cohort,series)
    summary['offline']=offline_metrics(root,plan['validation']['identity'])
    for name,row in results.items():
        if row['state']!='completed' or not row['result'].get('legal'):
            summary['new'][name]=row;continue
        reference_name='V3' if row['cohort']=='train' else 'V6'
        reference=results.get(reference_name,{})
        reference=Path(reference['path']) if reference.get('state')=='completed' and reference['result'].get('legal') else None
        value=describe(Path(row['path']),reference)
        all_data[name]=value;summaries[name]=value[0];series[name]=value[1];summary['new'][name]=value[0]
        if name in ('V1','V1_attempt2','V2','V4','V5'):
            historical=OLD/f'sample_{row["updates"]}_loop'/row['result']['identity']
            old_trace=arrays(historical/'trace.npz');current=value[2]
            value[0]['historical_reproduction']={k:bool(old_trace[k].shape==current[k].shape and np.array_equal(old_trace[k],current[k]))
                for k in ('command','target','joint','velocity','object','contact','object_contact','lifted','subgoals','alignment')}
        raw=Path(row['path'])/'clean.mp4'
        if raw.exists():
            clean=root/'videos'/(name+'_clean.mp4');annotated=root/'videos'/(name+'_diagnostic.mp4')
            if not clean.exists():shutil.copy2(raw,clean)
            decorate(clean,annotated,name,row,value[1],value[0]['events'])
            summary['videos'][name]=dict(record=row,result=value[0]['result'],events=value[0]['events'],
                clean=validate_video(clean),diagnostic=validate_video(annotated))
    for cohort,names in [('train',['V3','V1','V2']),('validation',['V6','V4','V5'])]:
        ready=[(name,name) for name in names if name in series]
        if ready:draw_cohort('new_'+cohort,ready,summaries,series)
    for name in ('V1','V2','V3','V4','V5','V6'):
        if name not in summary['videos']:summary['missing_videos'].append(name)
    success_videos=[name for name,row in summary['videos'].items() if name.startswith('V1') and row['result']['success']]
    summary['reproduced_success_videos']=success_videos
    if not success_videos:
        summary['historical_success_reconstruction']=reconstruction(root,all_data)
    # Public10Hz curves retain exact native-rate values privately in numeric traces.
    with (PUBLIC/'curves_10hz.csv').open('w') as stream:
        fields=['run','tick','time_s','actual_expert_fk_mm','target_actual_fk_mm','arm_tracking_rad','hand_tracking_rad',
            'arm_jump_rad','hand_jump_rad','object_contact_N','object_height_m','native_lift','subgoals']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name,s in series.items():
            for tick in range(0,len(s['tick']),6):
                row=dict(run=name,tick=tick,time_s=(tick+1)/60)
                for k in fields[3:]:
                    if k in s and np.isfinite(s[k][tick]):row[k]=float(s[k][tick])
                writer.writerow(row)
    summary['takeover_expansion']=read(root/'takeover_expansion.json') if (root/'takeover_expansion.json').exists() else None
    summary['history_input_shift']=input_shift(root,results,plan['validation']['identity'])
    summary['resource_summary']=read(root/'monitor_exit.json') if (root/'monitor_exit.json').exists() else None
    atomic_json(root/'analysis/summary.json',summary)
    atomic_json(PUBLIC/'summary.json',summary)
    atomic_json(root/'videos/manifest.json',dict(videos=summary['videos'],reconstruction=summary.get('historical_success_reconstruction'),missing=summary['missing_videos']))
    report(summary,root)


def fmt(value):
    return '缺失' if value is None else f'{value:.3f}'


def report(s,root):
    pair=s['pairs']['train'];first=pair['first_sustained_divergence']
    lines=['# H24 闭环失败诊断与视频', '',
        '本轮只读已有权重，新增录像和有限专家前缀诊断；无训练、策略/物理/控制参数修改、test40选择。',
        f'运行目录：`{root}`；源码冻结提交 `{s["source_commit"]}`。原训练比较和本轮录像分开计数。', '',
        '## 录像交付与实际结果','',
        ('8000训练初态成功录像已实际生成：'+', '.join(s['reproduced_success_videos'])) if s['reproduced_success_videos'] else
        '本轮没有可靠的8000原生成功录像。历史342tick成功证据仍保留；提供明确标注的数值重建，不能当作原始成功录像或物理重放。', '',
        '|录像|模型|初态/seed|实际结果|ticks|诊断MP4|','|---|---|---|---|---:|---|']
    for name,row in s['videos'].items():
        result=row['result'];outcome='success' if result['success'] else next((k for k,v in result['reasons'].items() if v),'censored')
        lines.append(f'|{name}|{row["record"]["updates"] or "Expert"}|{result["cohort"]}/{result["seed"]}|{outcome}|{result["steps"]}|`{row["diagnostic"]["path"]}`|')
    lines+=['',f'缺失指定视频：{s["missing_videos"] or "无"}。每次尝试均保留，未按成功筛除失败。',
        '每段clean/diagnostic视频均全帧解码并由ffprobe核验；精确时长、大小、SHA及初态/checkpoint在本地 `videos/manifest.json` 和公开summary。',
        '三路既有448×252传感器拼成1280×720，CPU缩放；30fps取每两个60Hz控制tick的动作前图像。终止后reset帧不录入。',
        '带注释版本标明最终结果；瞬时接触/追踪读数来自上一已完成tick。三视图布局：左上全局、右上侧面、左下腕部；文字只占右下空区。', '',
        '## 8000成功与10000失败：历史证据','',
        f'完整成功原记录342ticks，10000同可观测初态208ticks fall。初态可观测字段完全相同：{pair["initial"]["observable_equal"]}；环境随机流相同：{pair["initial"]["environment_rng_equal"]}。',
        f'实际导出状态比较了{pair["initial"]["exported_fields_checked"]}个字段；不一致字段见summary，不把未保存PhysX内部状态当作已一致。',
        '分析前固定：末端5mm、Arm RMS0.01rad、Hand RMS0.05rad、物体5mm/5deg，连续3tick超阈值才报显著分歧。',
        f'第一次原始目标差异tick={pair["first_raw_target_difference_tick"]}，第一次实际提交输入差异tick={pair["first_command_difference_tick"]}；',
        f'实际末端首次持续分歧tick={first["ee_mm"]}，Arm={first["arm_rad"]}，Hand={first["hand_rad"]}，物体位置={first["object_mm"]}，物体姿态={first["object_rotation_deg"]}。tick从0开始。',
        f'首个请求目标Arm/Hand差异为{fmt(pair["initial_raw_target_arm_difference_rad"])} / {fmt(pair["initial_raw_target_hand_difference_rad"])} rad RMS。',
        '两个checkpoint在状态明显偏离前已产生不同目标，随后闭环状态和历史共同改变。首差是时间顺序证据，并不能单独排除隐藏物理状态或证明接触根因。', '',
        '![历史训练轨迹](evidence/h24_diagnosis_20261104/historical_train.png)', '',
        '## 离线预测、控制输入与实际运动','',
        'H24直接提交post-delay绝对目标(rad)，沿原执行器及120/60Hz物理/控制执行；没有第二次原生命令delay/EMA。专家输入是dimensionless native command。',
        '本轮直接记录DDIM最终归一化样本、反归一化raw plan、真实提交输入、transition实际应用target和两个物理子步执行器target。不存在的“学生native command”不通过逆映射伪造。',
        '离线17.33mm是全val40专家历史下的接触前目标FK误差；约46mm是共同7个validation初态、专家接触前同tick的实际FK误差。',
        '两者的预测目标/实际运动、历史、采样批次、数据子集和时间条件不同，不能相减得到控制误差。下表使用本次固定validation ID的原数据参照：', '',
        '|checkpoint|离线目标Arm/Hand rad|离线目标FK mm|自主目标对原专家目标FK mm|自主实际对原专家实际FK mm|自主target对actual FK mm|',
        '|---|---|---:|---:|---:|---:|']
    for label,row in s['offline'].items():
        b=row['same_source_autonomous'];lines.append(f'|{label}|{fmt(row["arm_rmse_rad"])}/{fmt(row["hand_rmse_rad"])}|{fmt(row["target_fk_rmse_mm"])}|{fmt(b["target_expert_target_fk_mm"])}|{fmt(b["actual_expert_actual_fk_mm"])}|{fmt(b["target_actual_fk_mm"])}|')
    lines+=['','上表离线使用已有valid prefix12预测，不重跑推理；自主三列使用同原expert记录及公共tick长度。它们仍不是因果分解或可相减误差。',
        '同tick比较没有时间伸缩；接触后进度不一致可能扩大误差。FK端点为原审计使用的iiwa14_link_7框架，不是接触点。', '',
        '|历史运行|实际目标到达Arm/Hand rad|target/actual FK mm|最大裁剪 rad|执行器目标误差 rad|',
        '|---|---|---:|---:|---:|']
    for name,row in s['historical'].items():
        lines.append(f'|{name}|{fmt(row["target_actual_arm_rmse_rad"])}/{fmt(row["target_actual_hand_rmse_rad"])}|{fmt(row["target_actual_fk_mm"])}|{fmt(row["clipping_max_rad"])}|{fmt(row["actuator_target_max_error_rad"])}|')
    lines+=['','分接触前/后跟踪误差、Arm/Hand重规划边界跳变和各事件tick见summary与10Hz CSV，图使用全60Hz数据。',
        '![历史验证轨迹](evidence/h24_diagnosis_20261104/historical_validation.png)', '',
        '![控制链训练初态](evidence/h24_diagnosis_20261104/control_chain_train.png)', '',
        '![控制链验证初态](evidence/h24_diagnosis_20261104/control_chain_validation.png)', '',
        f'实际记录的本体/几何历史与原专家同tick输入标准化差异：`{json.dumps(s["history_input_shift"],ensure_ascii=False)}`。初始差异也明确列出，不能将全部差异归因于后来学生控制。', '',
        '## Expert prefix / Student takeover','',
        '接管前每tick都用实际专家前缀更新学生自己的GRU；过去位置历史接收原专家处理后的真实target，未复制专家RNN或瞬移物理状态。',
        '切换点对齐原12tick规划边界，DDIM seed7091持续前进，不在切换时重置。专家shadow与学生前缀计划均标为未执行预测。',
        '切换前逐字段对照该初态新录专家的command/target/q/qd/object/contact；超1e−6会在学生控制前停止。即使这些字段一致，也不声称全部PhysX内部状态相同。','',
        '|诊断|专家前缀ticks|前缀核验|原生结果|lift/subgoals|','|---|---:|---|---|---|']
    for name,row in s['new'].items():
        if not name.startswith('T_'):continue
        if 'result' not in row:
            lines.append(f'|{name}|缺失|软件失败|未计任务结果|—|');continue
        r=row['result'];lines.append(f'|{name}|{r["expert_prefix_ticks"]}|{r.get("prefix_validation",{}).get("passed")}|{r["reasons"]}|{r["native_lift"]}/{r["native_subgoals"]}|')
    lines+=['',f'扩展判据实测：`{json.dumps(s["takeover_expansion"],ensure_ascii=False)}`。未达到判据则不扩展，少量接管不能证明唯一因果根因。', '',
        '## 接触后现象与证据限度','',
        '记录object-filtered normal force、五指位置/姿态、物体位姿速度、掌部相对位移、native lift/subgoal/alignment。',
        '物体与掌部相对运动是可观测现象；没有逐接触点几何或可靠切向摩擦/滑移状态，不能仅凭录像称为摩擦不足、真实滑移或某个手指接触点错误。',
        '已接触但无native lift支持接触后抓取/抬升进展不足；多指受力或stable_grasp代理均不等于原生任务成功。', '',
        '## 原因排序与后续建议','',
        '1. 优先考虑自主状态/历史偏移与目标预测误差共同累积：首个预测已不同，teacher-forced误差没有约束整个自主闭环。支持/反对证据应结合上表与接管结果，未建立单一因果归因。',
        '2. 目标变化和物理跟踪/接触构型：直接位置目标可跳变；target/actual差值是同tick真实量，但受惯性与碰撞影响，不能仅据此指认控制器代码错误。',
        '3. 接触后抓取与抬升阶段：接触不能保证构型、承载或后续装配正确。力与轨迹支持阶段定位，缺少接触点和摩擦证据限制具体物理解释。',
        '4. 隐藏模拟器状态仍是未排除因素；本轮独立Kit与可观测初态相同不构成精确历史4lane重放。',
        '不建议仅凭离线误差下降继续无诊断加训或立即加FK/Temporal loss。本轮没有修改控制，后续是否修改目标连续性/数据分布，应先结合真实请求/执行器一致性及接管证据提出独立对照。', '',
        '## 查看、取回与完整性','',
        f'本地视频目录：`{root}/videos`。在连接该主机的VS Code中打开同目录README和MP4；或在该主机运行：',
        '```bash',f'python3 -m http.server 8765 --bind 127.0.0.1 --directory {root}/videos','```',
        '在VS Code Ports中转发8765，然后在本机浏览器打开 `http://127.0.0.1:8765/`。',
        '命令仅按需运行；本轮没有自动开放网络服务。亦可在本机终端将已配置SSH主机别名替换为实际连接名：',
        '```bash',f'scp -r <你的SSH主机别名>:{root}/videos ./h24_diagnosis_videos','```',
        '所有MP4、原始HDF5/轨迹、权重和大日志只留本机。公开清单含每段准确绝对路径、时长、字节数、SHA及核验结果。',
        'owner在报告生成后重新只读校验全部7个受保护checkpoint和冻结源码，结果写入本地final_integrity.json；尚无该文件时不声称最终校验完成。',
        '资源监测总结见summary.resource_summary；不把整卡利用率当作本进程独占量。','']
    REPORT.write_text('\n'.join(lines))
    index=['# H24诊断录像索引','','所有录像均来自本轮真实运行；NUMERIC_RECONSTRUCTION单独标注为历史数值重建。','']
    for name,row in s['videos'].items():
        index.append(f'## {name}: {row["result"]["identity"]} / seed{row["result"]["seed"]}')
        index.append('')
        for kind in ('clean','diagnostic'):
            v=row[kind];index.append(f'- [{kind}]({Path(v["path"]).name}): {v["duration_s"]:.3f}s, {v["bytes"]}bytes, SHA256 `{v["sha256"]}`')
        index.extend(['',f'结果：{row["result"]["reasons"]}；关键控制tick：{row["events"]}；视频时间=tick/60秒。',''])
    if s.get('historical_success_reconstruction'):
        v=s['historical_success_reconstruction'];index+=['## 历史成功的数值重建（不是原始录像）','',f'[{Path(v["path"]).name}]({Path(v["path"]).name})','']
    index+=['## 浏览器获取','','在主机运行：','```bash',f'python3 -m http.server 8765 --bind 127.0.0.1 --directory {root}/videos','```',
        'VS Code Ports转发8765，在本机访问 http://127.0.0.1:8765/ 。完整身份与SHA见manifest.json。','']
    (root/'videos/README.md').write_text('\n'.join(index))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();torch.set_num_threads(2);run(args.root.resolve())
