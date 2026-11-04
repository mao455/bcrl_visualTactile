"""Read-only expert/control and paired failure audit; never starts a simulator."""
import argparse
import ast
import csv
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from tools.state_image_queue import CODE, read, write, sha
from tools.h24_diagnosis_report import arrays, describe, compare_initial, recorded_raw, rms
from tools.h24_diagnosis_common import first_sustained, contact_force
from tools.h24_comparison_report import validated_fk
from tools.state_fk_audit import fk_pose

BASE = CODE/'outputs/visuotactile/native_dp_v2'
FINAL = BASE/'final_dp_20261104_01'
POSITION = BASE/'position_20261103_01'
PUBLIC = CODE/'research_docs/evidence/grasp_diagnosis_20261104_01'


def function_tree(text, name):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


def replay_evidence(root):
    session = read(POSITION/'session.json')
    filename = 'visuotactile/position_control.py'
    # The session's source_commit predates its documented runtime amendment.
    # Match the amendment's actual frozen file SHA, not that stale initial label.
    replay_revision = 'c7fcbb0'
    old = subprocess.check_output(['git','show',replay_revision+':'+filename], cwd=CODE, text=True)
    assert __import__('hashlib').sha256(old.encode()).hexdigest()==session['implementation_sha256'][filename]
    current = (CODE/filename).read_text()
    functions = {k:function_tree(old,k)==function_tree(current,k) for k in ('bounded_targets','apply_position_targets')}
    core = {k:sha(CODE/k)==session['implementation_sha256'][k] for k in (
        'visuotactile/sim/environment.py','isaacsimenvs/tasks/play/play_env.py',
        'isaacsimenvs/tasks/play/utils/action_utils.py')}
    rows = []
    for i, identity in enumerate(('fresh_982501','fresh_982502')):
        expert = POSITION/f'expert_single_{i:02d}'/identity
        replay = POSITION/f'position_replay_single_{i:02d}'/identity
        if not expert.exists(): expert = POSITION/'expert'/identity
        a,b = arrays(expert/'trace.npz'),arrays(replay/'trace.npz')
        result = read(replay/'result.json')
        fields = {k:np.array_equal(a[k],b[k]) and a[k].dtype==b[k].dtype for k in (
            'actor','target','joint','actuator_targets','velocity','object','contact','object_contact',
            'force','torque','lifted','subgoals','retract','alignment')}
        row = dict(identity=identity,steps=result['steps'],success=result['success'],
            mode=result['action_control_mode'],bytewise_fields=fields,
            submitted_target_max_rad=float(np.abs(b['command']-a['target']).max()),
            actuator_target_max_rad=float(np.abs(b['actuator_targets']-a['target'][:,None]).max()),
            tracking_rms_rad=rms(b['joint']-b['target']),
            reference=str(expert),replay=str(replay),reference_trace_sha256=sha(expert/'trace.npz'),
            replay_trace_sha256=sha(replay/'trace.npz'))
        row['passed'] = (row['success'] and row['mode']=='absolute_joint_position_target_rad'
            and all(fields.values()) and row['submitted_target_max_rad']==row['actuator_target_max_rad']==0.)
        rows.append(row)
    result = dict(passed=all(functions.values()) and all(core.values()) and all(x['passed'] for x in rows),
        reused=True,new_kit_starts=0,original_source=replay_revision,
        unchanged_execution_function_AST=functions,unchanged_core_file_SHA=core,rows=rows,
        scope='Recorded post-delay/EMA target radians replayed through current DP absolute interface; not native-command replay. Only saved physical fields claim bytewise equality.')
    write(root/'replay_evidence.json',result)
    return result


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


def event_windows(series):
    hit=series['contact']>.1
    losses=[t for t in range(3,len(hit)-2) if hit[t-3:t].all() and not hit[t:t+3].any()]
    rows=[]
    for t in losses:
        sl=slice(max(0,t-3),min(len(hit),t+4))
        rows.append(dict(tick=t,window='[-3,+3] native ticks',replan_in_window=any(i%12==0 for i in range(sl.start,sl.stop)),
            max_hand_jump_rad=float(series['hand_jump'][sl].max()),
            max_hand_tracking_rad=float(series['hand_tracking'][sl].max()),
            max_clipping_rad=float(np.nanmax(series['clipping'][sl])) if np.isfinite(series['clipping'][sl]).any() else None))
    return rows


def analyze(root):
    root.mkdir(parents=True,exist_ok=True); PUBLIC.mkdir(parents=True,exist_ok=True)
    replay=replay_evidence(root)
    attempts=[x for x in read(FINAL/'attempts.json') if x.get('valid')]
    assert len(attempts)==24
    rows=[];series_all={}
    for record in attempts:
        ordinal=record['ordinal'];identity=record['trial']['identity']
        student=Path(record['output'])/identity
        expert=POSITION/f'expert_learning_single_{ordinal:02d}'/identity
        s,ss,a=describe(student);e,es,b=describe(expert)
        assert e['result']['success'] and e['result']['environment_sha256']==record['trial']['environment_sha256']
        initial=compare_initial(student,expert)
        # Full matching exports include expert recurrent state at reset; no hidden solver-state claim.
        fk=validated_fk(read(student/'metadata.json'));ax=fk_pose(fk,a['joint'])[0];bx=fk_pose(fk,b['joint'])[0]
        n=min(len(a['joint']),len(b['joint']));raw=recorded_raw(student,len(a['joint']))
        ts,te=tick_metrics(a,raw),tick_metrics(b)
        angle=np.abs(np.sum(a['object'][:n,3:]*b['object'][:n,3:],axis=-1)).clip(0,1)
        differences=dict(actual_ee_mm=np.linalg.norm(ax[:n]-bx[:n],axis=-1)*1000,
            actual_arm_rad=np.sqrt(np.mean((a['joint'][:n,:7]-b['joint'][:n,:7])**2,axis=-1)),
            actual_hand_rad=np.sqrt(np.mean((a['joint'][:n,7:]-b['joint'][:n,7:])**2,axis=-1)),
            target_arm_rad=np.sqrt(np.mean((a['target'][:n,:7]-b['target'][:n,:7])**2,axis=-1)),
            target_hand_rad=np.sqrt(np.mean((a['target'][:n,7:]-b['target'][:n,7:])**2,axis=-1)),
            object_mm=np.linalg.norm(a['object'][:n,:3]-b['object'][:n,:3],axis=-1)*1000,
            object_rotation_deg=np.rad2deg(2*np.arccos(angle)))
        thresholds=dict(actual_ee_mm=5.,actual_arm_rad=.01,actual_hand_rad=.05,target_arm_rad=.01,
            target_hand_rad=.05,object_mm=5.,object_rotation_deg=5.)
        first={k:first_sustained(v,thresholds[k],3) for k,v in differences.items()}
        ec=e['events']['contact_tick'];el=e['events']['lift_tick']
        def phase(t):
            if t is None:return 'not_observed'
            if ec is None or t<ec:return 'before_expert_sustained_contact'
            if el is None or t<el:return 'after_expert_contact_before_native_lift'
            return 'after_expert_native_lift'
        row=dict(model=record['model'],ordinal=ordinal,identity=identity,cohort=record['trial']['cohort'],
            success=False,termination=record['trial']['reasons'],student=str(student),expert=str(expert),
            expert_success=True,initial=initial,common_ticks=n,expert_steps=len(b['joint']),student_steps=len(a['joint']),
            expert_events=e['events'],student_events=s['events'],first_sustained_divergence=first,
            first_divergence_phase={k:phase(t) for k,t in first.items()},thresholds=thresholds,consecutive_ticks=3,
            first_target_difference=first_sustained(np.max(np.abs(a['target'][:n]-b['target'][:n]),axis=-1),1e-6,1),
            same_prefix_student=window_summary(ts,n),same_prefix_expert=window_summary(te,n),
            full_student=window_summary(ts,len(a['joint'])),full_expert=window_summary(te,len(b['joint'])),
            clip=s.get('clipping_detail'),clipped_joint_ticks=s['clipped_joint_ticks'],
            expert_post_delay_target_at_limits_joint_ticks=int(((b['target']<=np.array(read(expert/'metadata.json')['joint_limits_rad'])[0,:,0]+1e-6)|
                (b['target']>=np.array(read(expert/'metadata.json')['joint_limits_rad'])[0,:,1]-1e-6)).sum()),
            expert_post_limit_target_clipping_zero_by_definition=True,
            student_contact_loss_windows=event_windows(ts),expert_contact_loss_windows=event_windows(te),
            actual_expert_precontact_ee_mean_mm=float(differences['actual_ee_mm'][:min(n,ec or n)].mean()),
            actuator_target_error_rad=s['actuator_target_max_error_rad'],
            source_hashes={str(p):sha(p) for p in (student/'trace.npz',student/'plans.npz',expert/'trace.npz')})
        rows.append(row);series_all[(record['model'],ordinal)]=(ts,te,differences)
        # Full tick data stays local; retain correlations and event neighborhoods for review.
        out=root/'per_tick'/f"{record['model']}_{ordinal:02d}.csv";out.parent.mkdir(exist_ok=True)
        columns={'tick':np.arange(n),**{'student_'+k:v[:n] for k,v in ts.items()},
            **{'expert_'+k:v[:n] for k,v in te.items()},**differences}
        with out.open('w') as f:
            w=csv.writer(f);w.writerow(columns);w.writerows(zip(*columns.values()))
        write(root/'analysis_progress.json',dict(completed=len(rows),planned=24))
    result=dict(replay=replay,rows=rows,passed_interfaces=all(x['actuator_target_error_rad']==0 for x in rows),
        identical_recorded_initials=all(x['initial']['observable_equal'] and x['initial']['environment_rng_equal'] and
            not x['initial']['exported_differences'] for x in rows),
        limitations=['Same control tick, common prefix; no time warping or hidden PhysX identity claim.',
            'Expert post-limit target has zero additional position-interface clipping by construction; cannot infer upstream native clamp magnitude.',
            'Contact event uses three consecutive ticks above frozen 0.1N; cohort contact counts use any tick.',
            'Event coincidence is descriptive, not a causal test.'])
    write(root/'analysis.json',result);write(PUBLIC/'analysis.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for ordinal in (0,4):
        fig,axes=plt.subplots(4,2,figsize=(13,11),sharex='col')
        for col,model in enumerate(('StateDP','ImageDP')):
            ts,te,d=series_all[(model,ordinal)];n=len(d['actual_ee_mm']);t=np.arange(n)/60
            axes[0,col].plot(t,d['actual_ee_mm'],label='Actual link7 vs expert (mm)');axes[0,col].axhline(5,color='gray',ls=':')
            for ax,key in zip(axes[1:,col],('hand_jump','hand_tracking','contact')):
                ax.plot(t,ts[key][:n],label=model);ax.plot(t,te[key][:n],label='Expert',alpha=.8)
                ax.set_ylabel({'hand_jump':'Target jump RMS (rad)','hand_tracking':'Target/reached RMS (rad)','contact':'Object contact (N)'}[key])
            for ax in axes[:,col]:ax.legend(fontsize=8);ax.grid(alpha=.2)
            axes[0,col].set_title(f'{model} vs successful expert; ordinal {ordinal}')
            axes[-1,col].set_xlabel('Simulation time (s); same-tick common prefix')
        fig.tight_layout();fig.savefig(PUBLIC/f'paired_{ordinal:02d}.png',dpi=140);plt.close(fig)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    analyze(a.root)
