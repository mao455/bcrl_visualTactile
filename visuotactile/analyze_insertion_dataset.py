"""Descriptive signed-offset results, storage, force exposure and <=4 previews."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .analyze_rl_trials import load_trial, wilson
from .compact import write_video
from .manifest import ManifestReader
from .initial_geometry import inspect
from .run_insertion_dataset import tree_bytes
from .trial_utils import write_json


def wrench_events(path, episode):
    with h5py.File(path) as f:
        g=f['episodes'][episode]
        forces=g['transition/applied_wrench/force_w_N'][:].reshape(-1,3)
        torques=g['transition/applied_wrench/torque_w_Nm'][:].reshape(-1,3)
        time=g['transition/applied_wrench/time_s'][:].reshape(-1)
        before=g['transition/diagnostics_before']
        stage=np.repeat(before['successes'][:],2)
        retract=np.repeat(before['retract_phase'][:],2)
        active=(np.linalg.norm(forces,axis=-1)>0)|(np.linalg.norm(torques,axis=-1)>0)
        return [dict(time_s=float(time[i]),control_step=int(i//2),substep=int(i%2),
            force_w_N=forces[i].tolist(),torque_w_Nm=torques[i].tolist(),
            stage_goals_done=int(stage[i]),retract=bool(retract[i])) for i in np.flatnonzero(active)]


def preview(directory, destination, label, outcome):
    with h5py.File(directory/'data.h5') as f:
        name=next(iter(f['episodes']));g=f['episodes'][name]
        rgb=g['frames/rgb_u8'];n=int(g.attrs['length']);terminal=bool(g.attrs['complete']);frames_count=len(rgb)
        # 2x integer enlargement for viewing only. Formal RGB is untouched.
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
        def frames():
            for i,x in enumerate(rgb):
                pair=np.concatenate([v.transpose(1,2,0) for v in x],axis=1)
                pair=Image.fromarray(pair).resize((896,252),Image.Resampling.NEAREST)
                canvas=Image.new('RGB',(896,312),'black');canvas.paste(pair,(0,60));draw=ImageDraw.Draw(canvas)
                draw.text((8,3),f'{label} | outcome: {outcome} | 1x real simulation time',font=font,fill='white')
                draw.text((8,22),f'frame {i}/{len(rgb)-1} | t={i/60:.3f}s | left: global_oblique; right: local_opposite',font=font,fill='white')
                note=('Terminal image missing; final shown frame is pre-action.' if terminal else
                      'Budget-censored: extra final valid post-frame retained.')
                draw.text((8,41),note,font=font,fill='yellow')
                yield np.asarray(canvas)
        write_video(destination,frames())
    return dict(path=str(destination),episode=label,outcome=outcome,frames=frames_count,
                playback_speed=1,terminal_image_missing=terminal)


def analyze(root, *, make_previews=False):
    plan=json.loads((root/'run_status/plan.json').read_text())
    reader=ManifestReader(root/'manifest.json',verify=True)
    output=root/'analysis';output.mkdir(exist_ok=True)
    loaded={key:load_trial(root/row['directory']) for key,row in reader.entries.items()}
    for key,t in loaded.items():
        if t['summary']['audit_errors']:raise RuntimeError('Committed record audit failed: '+key)
    rows=[]
    for entry in plan['scan']:
        status_path=root/'run_status'/(entry['id']+'.json')
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        t=loaded.get(entry['id'])
        row=dict(**entry,attempted=bool(status.get('attempt')),started=False,completed=False,outcome='not_started',software_error=False)
        if t:
            s=t['summary'];a=t['arrays'];m=t['metadata']
            geometry=inspect(m,a['initial'])
            row.update(started=True,completed=s['outcome'] in ('success','failure'),outcome=s['outcome'],
                length=s['control_steps'],duration_s=s['simulated_seconds'],native_reasons=s['native_reasons'],
                geometry=geometry,raw_pose_w=m['initial_offset']['raw_pose_w'][0],
                final_pose_w=m['initial_offset']['final_pose_w'][0],
                object_minus_hole_m=m['initial_offset']['object_minus_hole_m'][0],
                actual_force=s['applied_force'],actual_torque=s['applied_torque'],
                force_contact_steps=s['force_steps_with_finger_contact_after'],
                force_world_error_N=s['requested_vs_applied_peak_vector_error_N'],
                bytes=sum(x['bytes'] for x in reader.entries[entry['id']]['files'].values()),
                process_wall_s=status['wall_seconds'],loop_wall_s=status['result']['wall_seconds'],
                launch_to_commit_wall_s=(datetime.fromisoformat(status['finish_utc'])-datetime.fromisoformat(status['launch_utc'])).total_seconds())
            events=wrench_events(root/reader.entries[entry['id']]['directory']/'data.h5',s['episode'])
            write_json(output/(entry['id']+'_wrench.json'),events)
            row['wrench_events']=len(events)
        elif status.get('initialization_rejected'):
            row.update(outcome='invalid_initialization',geometry=status['geometry'])
        elif status.get('state')=='skipped_invalid_initial_group':
            row.update(outcome='skipped_invalid_initial_group')
        elif status:
            row.update(software_error=True,outcome='software_error',error=status.get('error'),returncode=status.get('returncode'))
        rows.append(row)
    conditions={}
    for condition in dict.fromkeys(r['condition'] for r in plan['scan']):
        selected=[r for r in rows if r['condition']==condition]
        counts=Counter(r['outcome'] for r in selected)
        started=sum(r['started'] for r in selected);completed=sum(r['completed'] for r in selected)
        success=counts['success']
        conditions[condition]=dict(planned=len(selected),attempted=sum(r['attempted'] for r in selected),
            started=started,completed=completed,success=success,failure=counts['failure'],
            censored=counts['censored'],software_errors=counts['software_error'],not_started=counts['not_started'],
            invalid_initialization=counts['invalid_initialization'],skipped_initial_group=counts['skipped_invalid_initial_group'],
            success_rate_started=success/started if started else None,wilson95_started=wilson(success,started),
            completed_rate=success/completed if completed else None,
            n_unique_starts=started,note='Exploratory; <=4 shared initial states per condition, not independent across conditions')
    paired=[]
    for row in rows:
        if not row['started'] or row['condition']=='baseline':continue
        baseline=loaded.get(f"s{row['seed']}_baseline")
        if baseline is None:continue
        other=loaded[row['id']];a,b=baseline['arrays'],other['arrays'];n=min(len(a['time']),len(b['time']))
        paired.append(dict(id=row['id'],seed=row['seed'],condition=row['condition'],
            baseline=baseline['summary']['outcome'],offset=other['summary']['outcome'],
            duration_difference_s=other['summary']['simulated_seconds']-baseline['summary']['simulated_seconds'],
            common_prefix_steps=n,object_max_distance_m=float(np.linalg.norm(a['object'][:n,:3]-b['object'][:n,:3],axis=-1).max()),
            joint_max_abs_difference_rad=float(np.abs(a['joint'][:n]-b['joint'][:n]).max()),
            raw_start_exact=baseline['metadata']['initial_offset']['raw_pose_w']==other['metadata']['initial_offset']['raw_pose_w']))
    actual=[r for r in rows if r['started']]
    counts=Counter(r['outcome'] for r in actual)
    estimate=None
    if actual:
        rate=counts['success']/len(actual)
        mean_bytes=np.mean([r['bytes'] for r in actual]);mean_wall=np.mean([r['launch_to_commit_wall_s'] for r in actual])
        estimate=dict(observed_success_fraction=rate,observed_attempts=len(actual),
            unique_initial_states=len({r['seed'] for r in actual}),mean_bytes_per_attempt=float(mean_bytes),
            mean_launch_to_commit_wall_seconds=float(mean_wall),
            attempts_for_400_point_estimate=int(np.ceil(400/rate)) if rate else None,
            disk_GiB_point_estimate=float(np.ceil(400/rate)*mean_bytes/1024**3) if rate else None,
            collector_and_audit_hours_point_estimate=float(np.ceil(400/rate)*mean_wall/3600) if rate else None,
            scenarios=[dict(assumed_success_fraction=p,attempts=int(np.ceil(400/p)),
                disk_GiB=float(np.ceil(400/p)*mean_bytes/1024**3),collector_and_audit_hours=float(np.ceil(400/p)*mean_wall/3600)) for p in (.25,.5,.75)],
            caveat='Planning scenarios only for legal starts; include launch/collection/encoding/audit, exclude invalid initialization/crashes and final analysis. Few shared starts cannot estimate production generalization. Failed trajectories can be longer. Do not pool acceptance or old75%.')
    # Signed direction/magnitude with n at every point.
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for axis,ax in enumerate(axs):
        values=[]
        for cm in (-5,-3,-1,0,1,3,5):
            key='baseline' if cm==0 else f"{'x' if axis==0 else 'y'}{'plus' if cm>0 else 'minus'}{abs(cm):02d}cm"
            c=conditions[key];values.append((cm,c))
            if c['started']:
                rate=c['success_rate_started'];lo,hi=c['wilson95_started']
                ax.errorbar(cm,rate,yerr=[[rate-lo],[hi-rate]],fmt='o',capsize=3)
                ax.text(cm,rate+.07,f"{c['success']}/{c['started']}",ha='center')
            else:ax.text(cm,.03,'n=0',ha='center',color='gray')
        ax.set(xlabel=f'Extra {"x" if axis==0 else "y"} offset (cm)',ylabel='Success / started',ylim=(-.05,1.2),xticks=[-5,-3,-1,0,1,3,5])
        ax.grid(alpha=.2)
    fig.suptitle(f'Exploratory only: {len({r["seed"] for r in actual})}/4 planned starts valid; Wilson95 per cell')
    fig.tight_layout();fig.savefig(output/'offset_results.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for r in actual:
        point=np.array(r['final_pose_w'][:2]);raw=np.array(r['raw_pose_w'][:2]);color='tab:green' if r['outcome']=='success' else 'tab:red'
        axs[0].scatter(*point,c=color,s=24);axs[0].plot([raw[0],point[0]],[raw[1],point[1]],color='gray',alpha=.3)
        axs[1].scatter(*r['object_minus_hole_m'][:2],c=color,s=24)
    axs[0].set(title=f'Actual object XY; n={len(actual)}',xlabel='x m',ylabel='y m')
    axs[1].set(title='Total object minus hole XY (not extra offset)',xlabel='dx m',ylabel='dy m')
    for ax in axs:ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'initial_coverage.png',dpi=160);plt.close(fig)
    acceptance=[]
    for key,t in loaded.items():
        if not key.startswith('accept_'): continue
        s=t['summary']
        events=wrench_events(root/reader.entries[key]['directory']/'data.h5',s['episode'])
        write_json(output/(key+'_wrench.json'),events)
        acceptance.append(dict(id=key,outcome=s['outcome'],length=s['control_steps'],
            applied_force=s['applied_force'],applied_torque=s['applied_torque'],wrench_events=len(events)))
    previous_report=output/'report.json'
    prior_previews=json.loads(previous_report.read_text()).get('previews',[]) if previous_report.exists() else []
    report=dict(parameter_updates=0,conditions=conditions,rows=rows,paired=paired,
        totals=dict(planned=52,attempted=sum(r['attempted'] for r in rows),started=len(actual),
                    **dict(Counter(r['outcome'] for r in rows))),
        native_force=dict(exposed_episodes=sum(r['actual_force']['nonzero_steps']>0 for r in actual),
            zero_exposure_episodes=sum(r['actual_force']['nonzero_steps']==0 for r in actual),
            duration_s=sum(r['actual_force']['duration_s'] for r in actual),
            peak_N=max((r['actual_force']['peak'] for r in actual),default=0),
            maximum_world_error_N=max((r['force_world_error_N'] for r in actual),default=0)),
        native_torque=dict(exposed_episodes=sum(r['actual_torque']['nonzero_steps']>0 for r in actual),
            zero_exposure_episodes=sum(r['actual_torque']['nonzero_steps']==0 for r in actual),
            duration_s=sum(r['actual_torque']['duration_s'] for r in actual),
            peak_Nm=max((r['actual_torque']['peak'] for r in actual),default=0)),
        acceptance=acceptance,
        estimate_400=estimate,output_bytes=tree_bytes(root),previews=[] if make_previews else prior_previews)
    if make_previews:
        preview_dir=root/'previews';preview_dir.mkdir(exist_ok=True)
        selected=[]
        candidates=[r for r in actual if r['outcome']=='success' and r['condition']=='baseline']
        if candidates:selected.append(candidates[0]['id'])
        else:selected.append('accept_baseline')
        successes=sorted((r for r in actual if r['outcome']=='success'),key=lambda r:-np.linalg.norm(r['offset_xy']))
        if successes and successes[0]['id'] not in selected:selected.append(successes[0]['id'])
        elif 'accept_offset' not in selected:selected.append('accept_offset')
        failures=[r for r in actual if r['outcome']=='failure']
        for r in failures:
            if len(selected)>=4:break
            if r['id'] not in selected:selected.append(r['id'])
        if not failures:
            # Legacy terminal failure has an invalid initial box overlap. Keep
            # it as a diagnostic illustration, never a legal-start failure.
            old=load_trial(root/'conversion_checks/failure')
            geometry=inspect(old['metadata'],old['arrays']['initial'])
            label=('OLD seed910101 original | INVALID initial overlap' if not geometry['valid_box_initialization']
                   else 'OLD cohort_v3 seed910101 original')
            report['previews'].append(preview(root/'conversion_checks/failure',preview_dir/'old_failure.mp4',
                label,'native failure (fall)'))
            report['previews'][-1]['initial_geometry']=geometry
            report['previews'][-1]['included_in_new_statistics']=False
        for key in selected[:4-len(report['previews'])]:
            row=reader.entries[key]
            report['previews'].append(preview(root/row['directory'],preview_dir/(key+'.mp4'),key,row['outcome']))
    write_json(output/'report.json',report)
    lines=['# Bounded insertion dataset', '', 'Exploratory only; four starts planned, actual valid unique starts reported in JSON. No parameter training.', '',
           '|condition|planned|started|success|failure|censored|software error|Wilson95|','|---|---:|---:|---:|---:|---:|---:|---|']
    for k,c in conditions.items():lines.append(f"|{k}|{c['planned']}|{c['started']}|{c['success']}|{c['failure']}|{c['censored']}|{c['software_errors']}|{c['wilson95_started']}|")
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ('totals','native_force','estimate_400','previews')},indent=2))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--previews',action='store_true')
    a=p.parse_args();analyze(a.root,make_previews=a.previews)


if __name__=='__main__':main()
