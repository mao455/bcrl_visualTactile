"""Read-only analysis of completed approximate-initial-state diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .compact import sha256
from .long_data import MANIFEST
from .training_resources import atomic_json


def first_difference(actual, expected, tolerance):
    error=np.abs(np.asarray(actual,dtype=float)-np.asarray(expected,dtype=float))
    if error.ndim>1:
        error=error.reshape(len(error),-1).max(1)
    positions=np.flatnonzero(error>tolerance)
    return int(positions[0]) if len(positions) else None


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def audit(root, output):
    status=json.loads((root/'status.json').read_text())
    if status['state']!='complete' or not (root/'worker_exit.json').exists():
        raise ValueError('Analyze only the completed native cohort')
    session=json.loads((root/'session.json').read_text())
    source=Path(session['source'])
    entries={row['id']:row for row in json.loads(MANIFEST.read_text())['episodes']}
    trials=json.loads((root/'trials.json').read_text())
    if len(trials)!=12 or len({row['attempt'] for row in trials})!=12:
        raise ValueError('Expected exactly twelve preregistered attempts')
    rows,timeline,artifacts=[],[],{}
    frame_figure,frame_axes=plt.subplots(4,3,figsize=(13,9))
    figure,axes=plt.subplots(4,3,figsize=(14,11))
    for ordinal,identity in enumerate(session['selected_train']):
        entry=entries[identity]
        with h5py.File(MANIFEST.parent/entry['directory']/'data.h5','r') as stream:
            group=stream['episodes'][entry['episode']]
            initial_z=float(group['transition/diagnostics_before/object_initial_z_m'][0])
            source_snapshot={key:value[0] for key,value in group['transition/diagnostics_before'].items()}
            source_snapshot.update(target=group['obs/actuator_target_rad'][0],
                cpu_rng_before_step=group['transition/torch_rng_before_step'][0])
            snapshot=root/(identity+'_source_initial_snapshot.npz')
            np.savez_compressed(snapshot,**source_snapshot)
            artifacts[snapshot.name]=sha256(snapshot)
            source_height=group['transition/diagnostics_after/object_pose_w'][:,2]-initial_z
            axes[ordinal,0].plot(np.arange(len(source_height))/60.,source_height,label='source expert',color='black')
            rgb=group['frames/rgb_u8'][0]
            for camera,axis in enumerate(frame_axes[ordinal]):
                axis.imshow(rgb[camera].transpose(1,2,0))
                axis.set_title(identity.rsplit('_',1)[-1]+' / '+('fixed high','fixed side','wrist')[camera],fontsize=9)
                axis.axis('off')
            for trial in (row for row in trials if row['identity']==identity):
                path=root/(trial['attempt']+'_trace.npz')
                artifacts[path.name]=sha256(path)
                with np.load(path) as trace, h5py.File(root/(trial['attempt']+'_inputs.h5'),'r') as inputs:
                    count=min(len(trace['step']),entry['length'])
                    command_error=np.max(np.abs(trace['command'][:count]-trace['expert'][:count]),axis=1)
                    joint_error=np.max(np.abs(trace['joint_after'][:count]-trace['expert_joint_after'][:count]),axis=1)
                    delayed=group['transition/control_trace/delayed_command'][:count]
                    target=group['transition/target_after_rad'][:count]
                    source_delay=group['transition/control_trace/delay_index'][:count]
                    first_input=inputs['0']
                    image_error=np.abs(first_input['rgb_u8'][:].astype(float)-rgb.astype(float))
                    depth_valid=first_input['depth_valid'][:].astype(bool)&group['frames/depth_valid'][0].astype(bool)
                    depth_error=np.abs(first_input['depth_m'][:]-group['frames/depth_m'][0])
                    before=trace['target_before']
                    expected_reference=trace['command'].copy()
                    limits=np.asarray(json.loads((source/'metadata.json').read_text())['joint_limits_rad'][0])
                    reconstructed=[]
                    for step in range(len(trace['step'])):
                        if trial['mode']=='T0_recorded':
                            break
                        reference=inputs[str(step//12*12)]['plan'][step%12]
                        expected_reference[step,:7]=(reference[:7]-before[step,:7])/.025
                        expected_reference[step,7:]=2*(reference[7:]-limits[7:,0])/(limits[7:,1]-limits[7:,0])-1
                        reconstructed.append(np.clip(expected_reference[step],-1,1))
                    conversion_error=float(np.max(np.abs(np.asarray(reconstructed)-trace['command']))) if reconstructed else None
                    if conversion_error is not None and conversion_error>1e-5:
                        raise ValueError('Saved plan/current target/native command alignment failed')
                    row={key:trial[key] for key in ('identity','mode','steps','success','contact_steps','native_lift',
                        'height_peak_m','subgoals','recorded_actions_exhausted','first_command_max_error',
                        'first_joint_after_max_error_rad','first_q_deviation_over_1mrad_step','checkpoint_sha256')}
                    row.update(first_delay_index_difference=first_difference(trace['delay_index'][:count],source_delay,0),
                        first_delayed_command_difference=first_difference(trace['delayed'][:count],delayed,1e-6),
                        first_target_difference=first_difference(trace['target_after'][:count],target,1e-6),
                        initial_rgb_mean_absolute_u8=float(image_error.mean()),initial_rgb_changed_fraction=float((image_error>0).mean()),
                        initial_depth_mean_absolute_m=float(depth_error[depth_valid].mean()),
                        plan_command_conversion_max_error=conversion_error,
                        initial_observable_max_error=max(trial['restoration']['saved_observable_errors'].values()))
                    rows.append(row)
                    native_time=trace['step']/60.
                    height=trace['object_after'][:,2]-initial_z
                    axes[ordinal,0].plot(native_time,height,label=trial['mode'],linewidth=1)
                    axes[ordinal,1].plot(native_time[:count],command_error,label=trial['mode'],linewidth=1)
                    axes[ordinal,2].plot(native_time[:count],joint_error,label=trial['mode'],linewidth=1)
                    for step in sorted(set(range(0,len(native_time),6))|set(range(min(6,len(native_time))))|{len(native_time)-1}):
                        timeline.append(dict(identity=identity,mode=trial['mode'],step=step,time_s=float(native_time[step]),
                            height_m=float(height[step]),gate=bool(trace['gate'][step]),
                            maximum_finger_force_N=float(np.linalg.norm(trace['force'][step],axis=-1).max()),
                            command_max_error=float(command_error[step]) if step<count else None,
                            joint_max_error_rad=float(joint_error[step]) if step<count else None))
            for column,axis in enumerate(axes[ordinal]):
                axis.set_title(identity.rsplit('_',1)[-1]+' / '+('height above reset','command deviation','joint deviation')[column])
                axis.set_ylabel(('m','native command','rad')[column]);axis.set_xlabel('Native time (s)')
                axis.legend(fontsize=7);axis.grid(alpha=.2)
    figure.suptitle('Approximate training-state diagnostics:4 preregistered demonstrations\nT0 ends at recorded-action exhaustion; students run to native termination; G00-B / G10-B epoch52 EMA',fontsize=12)
    figure.tight_layout(rect=(0,0,1,.94));figure.savefig(output/'training_initial_diagnostics.png',dpi=150);plt.close(figure)
    frame_figure.suptitle('Selected train demonstrations:actual first RGB frames; same task/geometry/camera calibration\nSource images only; these do not establish student-state or exact simulator equivalence',fontsize=12)
    frame_figure.tight_layout(rect=(0,0,1,.94));frame_figure.savefig(output/'training_initial_frames.png',dpi=120);plt.close(frame_figure)
    write_csv(output/'native_outcomes.csv',rows)
    write_csv(output/'native_timeline.csv',timeline)
    existing=json.loads((source/'dp_closed_loop/trials.json').read_text())
    id_trials=[row for row in existing if row['line'] in ('G00-B','G10-B')]
    for row in id_trials:
        row['failure_stage']='no native lift or first subgoal; transient height is not a successful grasp'
    resources=json.loads((root/'resources/resources.json').read_text())
    protected=['visuotactile/sim/environment.py']+subprocess.check_output(['git','ls-files','isaacsimenvs'],text=True).splitlines()
    protected_changes=subprocess.check_output(['git','diff','--name-only','--',*protected],text=True).splitlines()
    if protected_changes:
        raise ValueError('Protected source changed')
    owned=json.loads((root/'owner.json').read_text())
    alive={key:Path('/proc',str(owned[key])).exists() for key in ('pid','monitor_pid','worker_pid')}
    if any(alive.values()):
        raise ValueError('Owned diagnostic process still alive')
    result=dict(run_id=root.name,scope='approximate observable-state restoration; not exact training-environment replay',
        session=session,trials=trials,first_deviation_analysis=rows,source_manifest_sha256=sha256(MANIFEST),
        local_artifact_sha256=artifacts,resources=resources,owned_process_alive=alive,
        native_source_unchanged=True,protected_source_sha256={path:sha256(Path(path)) for path in protected},
        id_existing_trials=id_trials,id_interval=dict(method='two-sided exact Clopper-Pearson95%',
            successes=0,trials_per_model=3,lower=0.,upper=1-.025**(1/3),scope='separate current G00-B/G10-B; no pooling'),
        limitations=trials[0]['restoration']['missing']+[
            'source initial snapshots extracted post-hoc; actual observable snapshots are first trace/input rows',
            'sampling seed20261031+ordinal and saved plans available; individual Gaussian draws/RNG tensor not captured',
            'every-tick q/qd/commands/delays saved; images and policy condition only at12tick replan anchors',
            'T0 recorded actions exhausted without terminal reason: censored replay, not native timeout',
            'new T1 approximate diagnostics and old T2 are distinct; no new20trial success-rate campaign'])
    atomic_json(output/'native_audit.json',result)
    atomic_json(root/'analysis_artifacts.json',dict(source_snapshots='post-hoc source extraction, not actual simulator dump',
        sampling=dict(seed_formula='20261031+zero_based_demo_ordinal',device='cuda:0',
            initial_noise_shape=[1,64,29],one_randn_per_replan=True,ddim_eta=0.,exact_draws_saved=False),artifacts=artifacts))
    print(json.dumps(dict(trials=len(rows),resources=resources,owned_process_alive=alive)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.root,args.output)
