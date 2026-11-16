"""Read-only native outcomes and video-linked, physically timed failure evidence."""
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.tight_fulltask_20261116.reliable_eval import ROOT, OLD, TREE, read
from scripts.tight_fulltask_20261116.report import behavior
from visuotactile.execution_learning import ideal_execution_bounds
from visuotactile.position_control import REACHABLE_POSITION_EXECUTION, project_processed_position_targets

OUT = TREE/'research_docs/evidence/tight_diagnosis_20261116/reliable_eval'


def first(mask, start=48):
    ids = np.flatnonzero(mask & (np.arange(len(mask)) >= start))
    return int(ids[0]+1) if len(ids) else None


def intervals(mask):
    ends = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a+1), int(b)) for a, b in zip(np.flatnonzero(ends == 1), np.flatnonzero(ends == -1))]


def trajectory(directory):
    with np.load(directory/'trace.npz') as f:
        a = {k: f[k] for k in f.files}
    with h5py.File(directory/'numeric.h5') as f:
        metadata = json.loads(f.attrs['metadata_json'])
        keypoints = f['episode/transition/native_keypoints_max_dist'][:]
        issued = f['episode/transition/target_after_rad'][:].reshape(-1,29)
    counts = (np.linalg.norm(a['object_contact'], axis=-1) > .1).sum(-1)
    z = a['object'][:,2] - np.median(a['object'][24:48,2])
    n = len(z)
    previous = torch.from_numpy(a['execution_previous_target_rad'])
    base = torch.from_numpy(a['nominal'])
    limits = torch.tensor(metadata['joint_limits_rad']).reshape(29,2)
    bounds = ideal_execution_bounds(previous, limits, REACHABLE_POSITION_EXECUTION)
    anchor = base.clamp(min=bounds[...,0], max=bounds[...,1])
    pre = base + torch.from_numpy(a['residual_proposed_rad'])
    learned = (pre-anchor).numpy()
    recenter = (anchor-base).numpy()
    replay = project_processed_position_targets(pre.clamp(min=limits[:,0],max=limits[:,1]), previous, limits, REACHABLE_POSITION_EXECUTION)
    result = read(directory/'result.json')
    mixed = 'dagger' in result
    # Expert suffixes execute the expert, not the stored diagnostic student output.
    student_n = result['dagger']['student_ticks'] if mixed else n
    error = float(np.max(np.abs(replay.numpy()[:student_n]-a['target'][:student_n])))
    assert error < 2e-6
    assert float(np.abs(issued-a['target']).max()) < 2e-6
    obj = Rotation.from_quat(a['object'][:,[4,5,6,3]])
    palm = Rotation.from_quat(a['palm'][:,[4,5,6,3]])
    rel = palm.inv()*obj
    relpos = palm.inv().apply(a['object'][:,:3]-a['palm'][:,:3])
    events = dict(first_contact=first(counts.min(-1)>0), first_two_fingers=first(counts.min(-1)>=2),
        elevated_2cm=first(z>=.02), native_lift=first(a['lifted']>0), alignment=first(a['alignment']>0),
        first_subgoal=first(a['subgoals']>0), native_retract=first(a['retract']>0))
    summary = dict(case=directory.name, source='student_then_expert' if mixed else 'autonomous_student',
        controls=n, success=result['success'], events_completed_tick=events,
        student_controls=student_n, legal_expert_controls=result.get('dagger',{}).get('expert_valid_ticks',0),
        two_finger_elevated_runs=[v for v in intervals((counts.min(-1)>=2)&(z>=.02)&(np.arange(n)>=48)) if v[1]-v[0]>=4],
        all_contact_absent_runs=[v for v in intervals((counts.max(-1)==0)&(np.arange(n)>=48)) if v[1]-v[0]>=8],
        minimum_native_keypoint_error_after48_m=float(keypoints[48:].min()),
        final_native_keypoint_error_m=float(keypoints[-1]),
        final_height_change_m=float(z[-1]), actual_target_trace_max_error_rad=float(np.abs(issued-a['target']).max()),
        student_composition_reconstruction_max_error_rad=error)
    return a, counts, z, keypoints, learned, recenter, rel, relpos, summary


def videos():
    summaries = []
    for p in sorted(OLD.glob('baseline_evaluation/*/autonomous.mp4')) + sorted(OLD.glob('collection/*/student_then_expert.mp4')):
        a, counts, z, kp, learned, recenter, rel, relpos, row = trajectory(p.parent)
        if p.parent.name == 'B_fulltask_development_28500101':
            i = 443  # completed tick444: zero contact at BOTH physics samples
            sl = slice(i-12,i+1)
            row['loss_event'] = dict(completed_tick=i+1, action_index=i, planning_phase=i%12,
                seconds=(i+1)/60, last_replan_completed_tick=433, next_replan_completed_tick=445,
                index_PIP_target_change_last12_rad=float(a['target'][i,14]-a['target'][i-12,14]),
                index_MCP_target_change_last12_rad=float(a['target'][i,12]-a['target'][i-12,12]),
                relative_object_rotation_last12_deg=float((rel[i-12].inv()*rel[i]).magnitude()*180/np.pi),
                relative_object_translation_last12_m=float(np.linalg.norm(relpos[i]-relpos[i-12])),
                learned_hand_rms_last12_rad=float(np.sqrt(np.mean(learned[sl,7:]**2))),
                deterministic_recenter_hand_rms_last12_rad=float(np.sqrt(np.mean(recenter[sl,7:]**2))),
                projection_hand_rms_last12_rad=float(np.sqrt(np.mean(a['execution_projection_rad'][sl,7:]**2))))
            ticks = np.arange(len(z))+1
            fig, axes = plt.subplots(4,1,figsize=(10,10),sharex=True)
            axes[0].plot(ticks,z*100,label='Object height relative to settled median (cm)')
            axes[0].plot(ticks,kp*100,label='Native keypoint max distance (cm)')
            axes[1].step(ticks,counts.min(-1),label='Min contacting fingers across two physics steps')
            axes[1].step(ticks,counts.max(-1),alpha=.5,label='Max contacting fingers')
            for field,label in [('nominal','DP target'),('execution_preprojection_target_rad','Combined pre-projection'),('target','Actually issued'),('joint','Actual joint')]:
                axes[2].plot(ticks,a[field][:,14],label=label)
            axes[2].set_ylabel('Index PIP (rad)')
            axes[3].plot(ticks,np.sqrt((learned[:,7:]**2).mean(-1)),label='Learned rear displacement')
            axes[3].plot(ticks,np.sqrt((recenter[:,7:]**2).mean(-1)),label='Deterministic recentering')
            axes[3].plot(ticks,np.sqrt((a['execution_projection_rad'][:,7:]**2).mean(-1)),label='Final projection')
            axes[3].set_ylabel('Hand RMS (rad)')
            for ax in axes:
                for tick in range(313,469,12):ax.axvline(tick,color='gray',alpha=.12)
                ax.axvline(444,color='red',ls='--',label='All contact lost at444')
                ax.legend(fontsize=7);ax.grid(alpha=.15)
            axes[-1].set_xlim(310,510);axes[-1].set_xlabel('Completed control tick (60 Hz); gray = planning refresh')
            fig.suptitle('Existing B video28500101: failed alignment followed by contact loss; no native subgoal')
            fig.tight_layout();fig.savefig(OUT/'b_video_loss.png',dpi=140);plt.close(fig)
        summaries.append(row)
    (OUT/'video_metrics.json').write_text(json.dumps(dict(rows=summaries,
        timing='Completed control ticks; seconds=tick/60. Video pre-action index t shows state after t completed controls.',
        contact='Force norm>0.1N; distinguish min(two physics samples) from zero contact at both samples.',
        inspection='All nine full sequences visually inspected at20-control-step spacing, with denser selected event frames. Not a claim of manual inspection of every frame.'),indent=2))


def results():
    rows=[];jobs=[]
    for job in read(ROOT/'registration.json')['jobs']:
        directory=Path(job['path'])
        if (directory/'owner_accepted.json').exists():
            result=behavior(directory);rows.extend(result['rows'])
            jobs.append(dict(seed=job['seed'], initial_pairing=result['paired_initial'],
                worker=read(directory/'worker_status.json'),result=read(directory/'result.json')))
    counts=[]
    for role in ['A_long','A11','B_short','B15']:
        legal=[r for r in rows if r['role']==role and r['legal']]
        full=[r for r in legal if r['full_task']]
        counts.append(dict(role=role,full_n=len(full),success=sum(r['success'] for r in full),
            illegal=sum(not r['legal'] for r in rows if r['role']==role),
            **{k:sum(bool(r[k]) for r in legal) for k in ['contact','two_finger','native_lift','alignment','subgoal']},
            **{'hold'+str(w):sum(r['hold'][str(w)]['sustained'] for r in legal) for w in [12,60,120,180]}))
    result=dict(utc=time.time(),complete=(ROOT/'complete.json').exists(),rows=rows,counts=counts,jobs=jobs,
        identities=read(ROOT/'model_identities.json'),registered=16,software_unresolved=16-len(rows),
        protocol='4 existing development families x4 full pairs, same actual observable starts, full native termination; no final test use')
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(complete=result['complete'],counts=counts)))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    videos();results()
