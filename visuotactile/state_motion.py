"""Paired StateDP motion/sampling diagnostics on a frozen split and checkpoint.

This is an analysis worker for the existing state_full_run owner, not a trainer.
All positions are canonical radians; normalized errors have explicit names.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .compact import sha256
from .kinematics import EncoderKinematics
from .position_control import bounded_targets, overlap_transition
from .state_history import HistoryBank, PersistentState, load_records
from .state_history_training import labels, require_foundation
from .state_resources import lease_for
from .training_resources import atomic_json

SAMPLERS = {f'A{i}': dict(inference_steps=steps, clip_sample=clip)
            for i, (steps, clip) in enumerate((s, c) for s in (16, 32, 100) for c in (False, True))}
SEEDS = (7091, 7092, 7093)
MOVEMENT_RAD = .01  # RMS displacement from observed target, full valid prefix only.


def distribution(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    return dict(count=int(a.size), mean=float(a.mean()) if a.size else None,
                **{k: float(np.quantile(a, q)) if a.size else None for k, q in
                   [('p05', .05), ('p50', .5), ('p95', .95), ('max', 1.)]})


def target_metrics(targets):
    x = np.asarray(targets)
    result = {}
    for name, sl in [('arm', slice(0, 7)), ('hand', slice(7, 29))]:
        delta = np.diff(x[:, sl], axis=0)
        boundary = np.arange(1, len(x)) % 12 == 0
        for label, mask in [('boundary', boundary), ('internal', ~boundary)]:
            result[f'{name}_{label}_jump_rms_rad'] = float(np.sqrt(np.mean(delta[mask]**2))) if mask.any() else None
        if len(delta)>1:
            acceleration = np.diff(delta, axis=0)*60**2
            result[f'{name}_acceleration_rms_rad_s2'] = float(np.sqrt(np.mean(acceleration**2)))
            for label, mask in [('boundary', (np.arange(2,len(x))%12<=1)),
                                ('internal', (np.arange(2,len(x))%12>1))]:
                result[f'{name}_{label}_acceleration_rms_rad_s2'] = float(np.sqrt(np.mean(acceleration[mask]**2))) if mask.any() else None
        result[f'{name}_speed_rms_rad_s'] = float(np.sqrt(np.mean(delta**2))*60) if len(delta) else None
    return result


def plan_metrics(plans, valid=None):
    plans = np.asarray(plans)
    if len(plans)<2:
        return {}
    overlap = plans[:-1, 12:] - plans[1:, :-12]
    if valid is None:valid=np.ones(plans.shape[:2],dtype=bool)
    aligned_valid=valid[:-1,12:] & valid[1:,:-12]
    result = target_metrics(plans[:, :12][valid[:,:12]])
    for name, sl in [('arm', slice(0, 7)), ('hand', slice(7, 29))]:
        result[name+'_overlap_rms_rad'] = float(np.sqrt(np.mean(overlap[..., sl][aligned_valid]**2))) if aligned_valid.any() else None
        result[name+'_aligned_first_rms_rad'] = float(np.sqrt(np.mean(overlap[:, 0, sl][aligned_valid[:,0]]**2))) if aligned_valid[:,0].any() else None
    return result


def motion_metrics(pred, gt, valid, before, scale, raw, limits, contact_anchor, fk=None):
    """No padding or static anchors enter moving-only displacement estimates."""
    p, y, mask, base = (np.asarray(v) for v in (pred, gt, valid, before))
    contact_times=np.asarray(contact_anchor)
    if contact_times.ndim==1:contact_times=np.broadcast_to(contact_times[:,None],mask.shape)
    result = {}
    for width in (12, p.shape[1]):
        tag = str(width)
        good = mask[:, :width].all(1)
        expert_delta = y[:, width-1, :7]-base[:, :7]
        student_delta = p[:, width-1, :7]-base[:, :7]
        magnitude = np.sqrt(np.mean(expert_delta**2, axis=-1))
        moving = good & (magnitude > MOVEMENT_RAD)
        result['moving'+tag+'_anchors'] = int(moving.sum())
        for phase, rows in [('', np.ones(len(p), dtype=bool)), ('moving_', moving), ('precontact_', ~contact_times[:,0])]:
            weights = mask[:, :width] & rows[:, None]
            if phase=='precontact_':weights &= ~contact_times[:,:width]
            for name, sl in [('arm', slice(0, 7)), ('hand', slice(7, 29)), ('all29', slice(0, 29))]:
                error = (p[:, :width, sl]-y[:, :width, sl])[weights]
                hold = (base[:, None, sl]-y[:, :width, sl])[weights]
                key = f'{phase}{name}{tag}'
                result[key+'_rmse_rad'] = float(np.sqrt(np.mean(error**2))) if error.size else None
                result[key+'_mae_rad'] = float(np.mean(np.abs(error))) if error.size else None
                result[key+'_hold_rmse_rad'] = float(np.sqrt(np.mean(hold**2))) if hold.size else None
                result[key+'_rmse_normalized'] = float(np.sqrt(np.mean((error*scale[sl])**2))) if error.size else None
            adjacent = mask[:, 1:width] & mask[:, :width-1] & rows[:, None]
            if phase=='precontact_':adjacent &= ~contact_times[:,1:width]
            velocity_error = (np.diff(p[:, :width, :7], axis=1)-np.diff(y[:, :width, :7], axis=1))*60
            result[f'{phase}arm{tag}_velocity_rmse_rad_s'] = float(np.sqrt(np.mean(velocity_error[adjacent]**2))) if adjacent.any() else None
        for joint in range(7):
            errors = (p[:, :width, joint]-y[:, :width, joint])[mask[:, :width]]
            result[f'joint{joint+1}_{tag}_rmse_rad'] = float(np.sqrt(np.mean(errors**2)))
        if moving.any():
            a, b = student_delta[moving], expert_delta[moving]
            an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
            result.update({f'moving{tag}_expert_displacement_rad': float(np.mean(bn)),
                f'moving{tag}_student_displacement_rad': float(np.mean(an)),
                f'moving{tag}_amplitude_ratio': float(np.mean(an/bn)),
                f'moving{tag}_direction_cosine': float(np.mean(np.sum(a*b, axis=-1)/np.maximum(an*bn, 1e-12))),
                f'moving{tag}_endpoint_rmse_rad': float(np.sqrt(np.mean((a-b)**2)))})
        if fk is not None:
            xyz, _ = fk(torch.from_numpy(p[:, :width].reshape(-1, 29)).float())
            truth, _ = fk(torch.from_numpy(y[:, :width].reshape(-1, 29)).float())
            squared = ((xyz-truth)**2).sum(-1).reshape(len(p), width).numpy()
            for phase, rows in [('', np.ones(len(p),bool)), ('moving_',moving), ('precontact_',~contact_times[:,0])]:
                weights=mask[:,:width]&rows[:,None]
                if phase=='precontact_':weights &= ~contact_times[:,:width]
                result[f'{phase}ee{tag}_target_fk_rmse_m']=float(np.sqrt(squared[weights].mean())) if weights.any() else None
    for name, sl in [('arm',slice(0,7)),('hand',slice(7,29)),('all29',slice(0,29))]:
        result[name+'_normalized_outside_fraction'] = float((np.abs(raw[...,sl])[mask]>1+1e-6).mean())
        result[name+'_joint_limit_fraction'] = float(((p[...,sl]<=limits[sl,0]+1e-6)|(p[...,sl]>=limits[sl,1]-1e-6))[mask].mean())
    return result


def metadata_and_contact(entry):
    with h5py.File(entry['path'], 'r') as stream:
        g=stream['episodes'][entry['episode']]
        meta=json.loads(stream.attrs['metadata_json'])
        force=g['obs/contact_substeps_w_N'][:]
        contact=np.linalg.norm(force,axis=-1).reshape(len(force),-1).max(-1)>.1
        # Source400 lacks object-filtered contact. Never call this object contact.
        return meta, np.maximum.accumulate(contact)


def rows_summary(rows):
    keys=sorted({k for r in rows for k,v in r.items() if isinstance(v,(float,int)) and k!='seed'})
    return {k:distribution([r[k] for r in rows if r.get(k) is not None]) for k in keys}


def paired_differences(rows, reference='A0'):
    lookup={(r['configuration'],r['identity'],r['seed']):r for r in rows}
    result={}
    for name in sorted({r['configuration'] for r in rows}):
        pairs=[]
        for r in rows:
            if r['configuration']!=name:continue
            base=lookup.get((reference,r['identity'],r['seed']))
            if base is None:continue
            pairs.append({k:v-base[k] for k,v in r.items() if k!='seed' and isinstance(v,(int,float)) and isinstance(base.get(k),(int,float))})
        result[name]=rows_summary(pairs)
    return result


@torch.no_grad()
def deployment_timing(model, record, sampler, seed, lease, count=3):
    """Sequential B=1 native memory, H2D, DDIM and D2H; excludes admission waits.

    No full-prefix recomputation at plan time. Cycle includes intervening 11 native
    updates. Live controller additionally measures packet/FK/geometry construction.
    """
    bank=HistoryBank(); generator=torch.Generator(device='cuda:0').manual_seed(seed)
    times=[]; cycle=0.; plan=[]
    for tick in range(min(len(record['time']), 12*count+1)):
        with lease.block():
            torch.cuda.synchronize(); began=time.perf_counter()
            condition=bank.observe(0,0,tick,model,record['actor'][tick].cuda(),record['robot'][tick].cuda(),
                record['geometry'][tick].cuda(),record['time'][tick].cuda())
            if condition is not None:
                prediction=model.predict(condition,generator,**sampler).cpu()
                bounded_targets(prediction.flatten(0,1),record['joint_limits'])
            torch.cuda.synchronize(); elapsed=time.perf_counter()-began
        cycle+=elapsed
        if condition is not None:
            times.append(elapsed)
            if tick:plan.append(cycle)
            cycle=0.
    return times,plan


@torch.no_grad()
def run(root, output, job_path):
    require_foundation(root)
    session=json.loads((root/'session.json').read_text()); job=json.loads(job_path.read_text())
    checkpoint=Path(job['checkpoint'])
    if sha256(checkpoint)!=job['checkpoint_sha256']:raise ValueError('Checkpoint hash drift')
    source=Path(session['source_root']); data=json.loads((source/'data.json').read_text())
    identities=job['identities']; split=job.get('split','validation')
    if not set(identities)<=set(data['splits'][split]) or len(identities)!=len(set(identities)):
        raise ValueError('Motion diagnostic split mismatch')
    if split=='test' and not (root/'motion_selection.json').exists():raise ValueError('Test requires frozen selection')
    if split=='test':
        selection=json.loads((root/'motion_selection.json').read_text())
        if selection['checkpoint_sha256']!=job['checkpoint_sha256'] or selection['sampler']!=job['samplers']['selected']:
            raise ValueError('Test configuration differs from frozen selection')
        if selection['inference']['overlap_steps']!=job.get('overlap_steps',0):raise ValueError('Test transition differs from frozen selection')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'configuration.json',dict(job=job,source_sha256=session['implementation_sha256'],
        movement_threshold_rms_rad=MOVEMENT_RAD,scope='teacher_forced; prefix0 label is action issued at current observed tick',
        contact='first aggregate fingertip force >0.1N; source has no object-filtered contacts',
        timing_scope='B1 every-native-tick history + inference + transfers; live packet construction measured separately'))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['kind']!='dp' or saved['contract']['action']!='absolute_joint_position_target_rad':
        raise ValueError('Trained position StateDP required')
    model=PersistentState('dp',saved['contract']);model.load_state_dict(saved['ema']);model.eval();del saved
    stats=torch.load(root/'position_train_statistics.pt',map_location='cpu',weights_only=False)
    if not all(torch.equal(getattr(model,k).cpu(),v) for k,v in stats.items()):
        raise ValueError('Checkpoint normalization differs from frozen train320 statistics')
    lease=lease_for(root,session['deadline'])
    with lease.block():model.cuda()
    # Read immutable numeric data; no image duplication and no cache generated.
    records=load_records(source,identities,model.contract)
    meta,_=metadata_and_contact(data['entries'][identities[0]])
    fk=EncoderKinematics(meta['config']['assets']['robot_urdf'],meta['joint_order'],links=('iiwa14_link_7',))
    rows=[]; artifacts={}; timing={name:[] for name in job['samplers']};cycles={name:[] for name in job['samplers']}
    for identity, cpu in records.items():
        _,contact=metadata_and_contact(data['entries'][identity]);per_config={}
        with lease.block():
            record={k:v.cuda() if torch.is_tensor(v) else v for k,v in cpu.items()}
            condition,anchors,_=model.encode(record)
            truth,valid=labels(record,anchors,model.horizon)
        anchors_cpu=anchors.cpu().numpy();truth_cpu=truth.cpu().numpy();valid_cpu=valid.cpu().numpy()
        before=cpu['robot'][anchors_cpu,58:87].numpy()
        scale=model.action_scale.cpu().numpy();offset=model.action_offset.cpu().numpy()
        for name,sampler in job['samplers'].items():
            predictions=[]
            for seed in job['seeds']:
                generator=torch.Generator(device='cuda:0').manual_seed(seed)
                with lease.block():
                    pred=model.predict(condition,generator,**sampler)
                    if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite diffusion sample')
                    raw=(pred*model.action_scale+model.action_offset).cpu().numpy()
                    if job.get('overlap_steps'):
                        previous=None;converted=[]
                        for plan in pred:
                            previous=overlap_transition(previous,plan[None],12,job['overlap_steps'])
                            converted.append(previous[0])
                        pred=torch.stack(converted)
                    prediction=bounded_targets(pred.flatten(0,1),record['joint_limits']).reshape_as(pred).cpu().numpy()
                predictions.append(prediction)
                row=dict(configuration=name,identity=identity,seed=seed,
                    **motion_metrics(prediction,truth_cpu,valid_cpu,before,scale,raw,cpu['joint_limits'].numpy(),
                        contact[(anchors_cpu[:,None]+np.arange(model.horizon)).clip(max=len(contact)-1)],fk),
                    **plan_metrics(prediction,valid_cpu))
                rows.append(row)
                if job.get('timing',True):
                    times,cycle=deployment_timing(model,cpu,sampler,seed,lease)
                    timing[name]+=times;cycles[name]+=cycle
                if seed==job['seeds'][0]:
                    per_config[name]=prediction
            stack=np.stack(predictions)
            for row in rows[-len(job['seeds']):]:
                row['arm_seed_variance_rad2']=float(stack[...,:7].var(0)[valid_cpu].mean())
                row['hand_seed_variance_rad2']=float(stack[...,7:].var(0)[valid_cpu].mean())
        # One compact first-seed plan per configuration; all paired scalar rows public.
        if identity==identities[0]:
            np.savez_compressed(output/'representative_plans.npz',anchors=anchors_cpu,truth=truth_cpu,valid=valid_cpu,
                before=before,contact=contact[anchors_cpu],**per_config)
        atomic_json(output/'progress.json',dict(episodes_complete=len({r['identity'] for r in rows}),episodes_total=len(identities),rows=rows))
    summary={name:rows_summary([r for r in rows if r['configuration']==name]) for name in job['samplers']}
    result=dict(software_complete=True,split=split,episodes=len(identities),seeds=job['seeds'],updates=0,
        checkpoint_sha256=job['checkpoint_sha256'],summary=summary,paired_difference=paired_differences(rows),
        planning_latency_s={n:distribution(v) for n,v in timing.items()},
        planning_cycle_latency_s={n:distribution(v) for n,v in cycles.items()},
        rows=rows,parameters=sum(p.numel() for p in model.parameters()),sampler_config=dict(model.sampler.config),
        resource_wait_s=lease.wait_s,pacing_s=lease.pacing_s,work_s=lease.work_s)
    fields=sorted({k for r in rows for k in r})
    with (output/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    plot_plans(output)
    atomic_json(output/'result.json',result)


def plot_plans(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=np.load(output/'representative_plans.npz');fig,axs=plt.subplots(3,2,figsize=(13,10))
    names=[n for n in p.files if n not in ('anchors','truth','valid','before','contact')]
    x=p['anchors']/60
    for joint in range(3):
        ax=axs[joint,0]; ax.plot(x,p['truth'][:,0,joint],color='black',label='expert target')
        for name in names:ax.plot(x,p[name][:,0,joint],label=name,alpha=.7)
        ax.set(xlabel='expert observation time (s)',ylabel=f'joint {joint+1} target (rad)')
        contact=np.flatnonzero(p['contact'])
        if len(contact):ax.axvline(x[contact[0]],color='gray',ls='--',label='first fingertip contact proxy')
    axs[0,0].legend(fontsize=7)
    full_prefix=p['valid'][:,:12].all(1)
    for name in names:
        prediction=p[name]; axs[0,1].plot(x,np.where(full_prefix,np.linalg.norm(prediction[:,11,:7]-p['before'][:,:7],axis=-1),np.nan),label=name)
        error=((prediction[:,:12,:7]-p['truth'][:,:12,:7])**2*p['valid'][:,:12,None]).sum((1,2))
        axs[1,1].plot(x,np.sqrt(error/(p['valid'][:,:12].sum(1)*7)),label=name)
        axs[2,1].plot(x[1:],np.sqrt(np.mean((prediction[:-1,12,:7]-prediction[1:,0,:7])**2,axis=-1)),label=name)
    axs[0,1].plot(x,np.where(full_prefix,np.linalg.norm(p['truth'][:,11,:7]-p['before'][:,:7],axis=-1),np.nan),'k',label='expert')
    for ax,label in zip(axs[:,1],['prefix12 arm displacement norm (rad)','prefix12 arm error (rad)','aligned replan disagreement (rad)']):
        ax.set(xlabel='expert observation time (s)',ylabel=label);ax.legend(fontsize=7)
    fig.suptitle('Teacher-forced source trajectory; target FK / contact proxy are diagnostics')
    fig.tight_layout();fig.savefig(output/'motion_curves.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(13,10))
    for column,(group,sl) in enumerate([('Arm7',slice(0,7)),('Hand22',slice(7,29))]):
        for name,values in [('expert',p['truth'])]+[(n,p[n]) for n in names]:
            style='--' if name=='expert' else '-'
            axs[0,column].plot(x,values[:,0,sl],style,alpha=.35)
            pairs=p['valid'][:,1:12]&p['valid'][:,:11]
            square=(np.diff(values[:,:12,sl],axis=1)*60)**2
            velocity=np.sqrt((square*pairs[...,None]).sum((1,2))/np.maximum(pairs.sum(1)*square.shape[-1],1))
            axs[1,column].plot(x,velocity,style,label=name)
            error=(values[:,:12,sl]-p['truth'][:,:12,sl])**2
            denom=p['valid'][:,:12].sum(1)*error.shape[-1]
            rmse=np.sqrt((error*p['valid'][:,:12,None]).sum((1,2))/denom)
            axs[2,column].plot(x,rmse,style,label=name)
        for ax,y in zip(axs[:,column],['target joint positions (rad)','prefix12 target speed RMS (rad/s)','prefix12 error RMSE (rad)']):
            ax.set(xlabel='expert observation time (s)',ylabel=group+' '+y)
            contact=np.flatnonzero(p['contact'])
            if len(contact):ax.axvline(x[contact[0]],color='gray',ls=':')
        axs[1,column].legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'joint_curves.png',dpi=140);plt.close(fig)


def report(root, output):
    """Publish small measured evidence only; never start work or push Git."""
    import shutil
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    session=json.loads((root/'session.json').read_text());output.mkdir(parents=True,exist_ok=True)
    public=Path('research_docs/evidence/state_motion_20261103');public.mkdir(parents=True,exist_ok=True)
    status=json.loads((root/'status.json').read_text())
    summary=dict(status=status,run_root=str(root),source_commit=session['source_commit'],
        source_verified=all(sha256(p)==h for p,h in session['implementation_sha256'].items()),
        protected_verified=all(sha256(p)==h for p,h in json.loads((root/'preserved.json').read_text()).items()),
        experiments={},offline={},pairing={},commands={},videos=[],training={},cleanup=[],
        internal_state_limitation='Observable reset + isolated CPU environment RNG; PhysX/contact solver internal state not fully captured',
        primary_motion_threshold_rms_rad=MOVEMENT_RAD)
    stages={'Original':'C0','Sampling adjusted':'C1','Overlap transition':'C2','H24':'h24_selected','Temporal-loss':'temporal_h32_selected'}
    baseline={}
    for label,stage in stages.items():
        path=root/stage/'result.json'
        if not path.exists():
            progress=root/stage/'progress.json'
            if not progress.exists():
                summary['experiments'][label]=dict(status='incomplete' if (root/stage).exists() else 'not run');continue
            result=dict(json.loads(progress.read_text()),software_complete=False)
        else:result=json.loads(path.read_text())
        trials=result['trials'];cohorts={}
        for cohort in ('train','validation'):
            rows=[r for r in trials if r['cohort']==cohort]
            cohorts[cohort]=dict(registered=sum(s['cohort']==cohort for s in session['motion_specs']),
                completed=len(rows),legal=sum(r.get('legal',False) for r in rows),successes=sum(r.get('success',False) for r in rows),
                **{key:sum(bool(r.get(key,False)) for r in rows) for key in ('native_lift','native_subgoals','native_alignment','approach_proxy')},
                object_contact=sum(r.get('contact',{}).get('object_contact',False) for r in rows),
                stable_grasp_proxy=sum(r.get('contact',{}).get('stable_grasp_diagnostic',False) for r in rows),
                falls=sum(r.get('reasons',{}).get('fall',False) for r in rows),
                timeouts=sum(r.get('reasons',{}).get('time_out',r.get('reasons',{}).get('timeout',False)) for r in rows))
        summary['experiments'][label]=dict(status='complete' if result['software_complete'] else 'incomplete',cohorts=cohorts,
            trials=trials,checkpoint_sha256=trials[0].get('checkpoint_sha256') if trials else None)
        pairs=[]
        for r in trials:
            identity=r['identity'];folder=root/stage/identity
            if not (folder/'trace.npz').exists():continue
            trace=np.load(folder/'trace.npz');old=Path(session['previous_root'])/'expert_learning'/identity
            if not (old/'trace.npz').exists():continue
            expert=np.load(old/'trace.npz');initial=json.loads((folder/'initial_observable.json').read_text())
            oldinitial=json.loads((old/'initial_observable.json').read_text())
            force=np.linalg.norm(expert['object_contact'],axis=-1).reshape(len(expert['joint']),-1).max(-1)
            contact=np.flatnonzero(force>.1);stop=min(int(contact[0]) if len(contact) else len(force),len(trace['joint']))
            if stop:
                e=trace['joint'][:stop]-expert['joint'][:stop]
                meta=json.loads((folder/'metadata.json').read_text());fk=EncoderKinematics(meta['config']['assets']['robot_urdf'],meta['joint_order'],links=('iiwa14_link_7',))
                student_xyz,_=fk(torch.from_numpy(trace['joint'][:stop]));expert_xyz,_=fk(torch.from_numpy(expert['joint'][:stop]))
                pair=dict(identity=identity,observable_initial_equal=initial==oldinitial,
                    reset_actor_equal=np.array_equal(trace['actor'][0],expert['actor'][0]),precontact_ticks=stop,
                    precontact_arm_joint_rmse_rad=float(np.sqrt(np.mean(e[:,:7]**2))),
                    precontact_hand_joint_rmse_rad=float(np.sqrt(np.mean(e[:,7:]**2))),
                    precontact_ee_tracking_rmse_m=float(((student_xyz-expert_xyz).square().sum(-1).mean()).sqrt()),
                    **target_metrics(trace['target']))
                pairs.append(pair)
                if identity==session['motion_specs'][0]['identity']:
                    fig,axes=plt.subplots(2,2,figsize=(12,8));t=np.arange(stop)/60
                    for j in range(7):
                        axes[0,0].plot(t,expert['joint'][:stop,j],ls='--',alpha=.6)
                        axes[0,0].plot(t,trace['joint'][:stop,j],alpha=.6)
                    axes[0,0].set(xlabel='time (s)',ylabel='precontact Arm7 q (rad)',title='expert dashed / student solid')
                    a=student_xyz[:,0].numpy();b=expert_xyz[:,0].numpy()
                    axes[0,1].plot(b[:,0],b[:,2],label='expert');axes[0,1].plot(a[:,0],a[:,2],label=label)
                    axes[0,1].set(xlabel='link7 base x (m)',ylabel='link7 base z (m)');axes[0,1].legend()
                    target=trace['target'];delta=np.diff(target[:,:7],axis=0)*60
                    axes[1,0].plot(np.arange(1,len(target))/60,np.sqrt(np.mean(delta**2,axis=-1)))
                    axes[1,0].set(xlabel='time (s)',ylabel='Arm7 target RMS speed (rad/s)')
                    plans=np.load(folder/'plans.npz')['plans']
                    axes[1,1].plot(np.arange(1,len(plans))*12/60,np.sqrt(np.mean((plans[:-1,12,:7]-plans[1:,0,:7])**2,axis=-1)))
                    axes[1,1].set(xlabel='time (s)',ylabel='aligned plan discrepancy (rad)')
                    fig.tight_layout();fig.savefig(public/(stage+'_closed_loop.png'),dpi=140);plt.close(fig)
            movie=folder/'rollout.mp4'
            if movie.exists():
                import av
                count=0
                try:
                    with av.open(str(movie)) as container:
                        for frame in container.decode(video=0):frame.to_ndarray(format='rgb24');count+=1
                    decoded=count>0
                except Exception:decoded=False
                summary['videos'].append(dict(path=str(movie),sha256=sha256(movie),frames=count,decoded=decoded))
        summary['pairing'][label]=pairs
    for stage in ('sampling16','sampling40','overlap_offline','h24_offline','temporal_h32_offline','final_test'):
        path=root/stage/'result.json'
        if not path.exists():continue
        result=json.loads(path.read_text());result.pop('rows',None)
        summary['offline'][stage]=result
        for name in ('metrics.csv','motion_curves.png','joint_curves.png'):
            src=root/stage/name
            if src.exists():shutil.copy2(src,public/(stage+'_'+name))
    for profile in ('single_fit','h24','temporal_h32'):
        path=root/(profile+'_train')/'metrics.json'
        if path.exists():
            rows=json.loads(path.read_text());summary['training'][profile]=[
                {**r,'train':{k:v for k,v in r['train'].items() if k!='episodes'},
                 'validation':{k:v for k,v in r['validation'].items() if k!='episodes'}} for r in rows]
    historical=Path(session['previous_root'])/'position_h32_train/metrics.json'
    if historical.exists():
        summary['historical_h32_same_updates']=[{**r,'train':{k:v for k,v in r['train'].items() if k!='episodes'},
            'validation':{k:v for k,v in r['validation'].items() if k!='episodes'}} for r in json.loads(historical.read_text())]
    if summary['training']:
        fig,axs=plt.subplots(1,3,figsize=(14,4))
        for profile,rows in summary['training'].items():
            for scope,style in [('train','--'),('validation','-')]:
                for ax,key in zip(axs,['loss','arm_rmse','hand_rmse']):
                    ax.plot([r['update'] for r in rows],[r[scope][key] for r in rows],style,label=profile+' '+scope)
                    ax.set(xlabel='updates',ylabel=key+(' (normalized)' if key=='loss' else ' (rad)'))
        for ax in axs:ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(public/'training_curves.png',dpi=140);plt.close(fig)
    for path in root.glob('*_launch.json'):
        row=json.loads(path.read_text());summary['commands'][path.name]=row['command']
    for name in ('sampling_selection','overlap_decision','single_fit_gate','training_withheld','motion_selection','repeated_success','sampling_audit','blocked','monitor_exit'):
        path=root/(name+'.json')
        if path.exists():summary[name]=json.loads(path.read_text())
    resource=root/'resources/resources.json'
    if resource.exists():summary['resources']=json.loads(resource.read_text())
    telemetry=root/'resources/shared.jsonl'
    if telemetry.exists():
        rows=[json.loads(line) for line in telemetry.read_text().splitlines()]
        if rows:
            fig,ax=plt.subplots(figsize=(10,3));start=rows[0]['utc_epoch']
            ax.plot([(r['utc_epoch']-start)/60 for r in rows],[r['gpu_util_percent'] for r in rows],alpha=.5,label='whole GPU')
            ax.plot([(r['utc_epoch']-start)/60 for r in rows],[r.get('rolling_60s_percent') for r in rows],label='complete60s')
            ax.axhline(50,color='gray',ls='--');ax.set(xlabel='elapsed minutes',ylabel='utilization (%)');ax.legend()
            fig.tight_layout();fig.savefig(public/'resources.png',dpi=140);plt.close(fig)
    atomic_json(output/'summary.json',summary);atomic_json(public/'summary.json',summary)
    # Replace only this run's bounded section; preserve every prior experiment.
    document=Path('research_docs/STATE_TRAINING_RESULTS.md');text=document.read_text()
    begin='<!-- state-motion-20261103:start -->';end='<!-- state-motion-20261103:end -->'
    if begin in text:text=text[:text.index(begin)]+text[text.index(end)+len(end):]
    lines=[begin,'## StateDP运动根因对照：自动汇总，待人工审查','',
        '实际状态：`'+status['state']+'`。结果按原分母保留；not run不补造结果。',
        '[小型证据、实际命令、曲线与checkpoint身份](evidence/state_motion_20261103/summary.json)。','',
        '|模型|H/A|完成状态|train成功|validation成功|','|---|---|---|---|---|']
    for label,row in summary['experiments'].items():
        counts=row.get('cohorts',{})
        def rate(scope):
            c=counts.get(scope)
            return f"{c['successes']}/{c['registered']}（完成{c['completed']}）" if c else 'not run'
        lines.append(f"|{label}|{'24/12' if label=='H24' else '32/12'}|{row['status']}|{rate('train')}|{rate('validation')}|")
    lines+=['','原400无物体专属接触：离线接触前用指尖力代理；闭环用物体过滤力。',
        '相同可观测初态和环境随机流不代表完整PhysX内部状态逐位一致。离线误差不等于任务成功。',end,'']
    first=text.find('\n');document.write_text(text[:first+1]+'\n'+'\n'.join(lines)+text[first+1:])
    atomic_json(root/'postprocess_complete.json',dict(utc=time.time(),git_published=False,output=str(public)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--job',type=Path);p.add_argument('--report',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.report:report(a.root,a.output)
    else:run(a.root,a.output,a.job)
