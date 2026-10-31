"""Publish bounded replay errors and autonomous outcomes without raw trajectories."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .compact import sha256
from .long_data import MANIFEST
from .replay_events import bitwise_equal, pose_error
from .training_resources import atomic_json


def fresh_precision(root):
    result={}
    with h5py.File(root/'fresh/expert_numeric.h5','r') as stream:
        group=stream['episode/transition']
        for phase in ('fresh','fresh_exact'):
            arrays=np.load(root/phase/'replay/trace.npz')
            fields={}
            for name,source in (('command','command_clipped_pre_delay'),('delayed','control_trace/delayed_command'),
                ('delay_index','control_trace/delay_index'),('target','target_after_rad'),('subgoals','diagnostics_after/successes'),
                ('lifted','diagnostics_after/lifted_object'),('retract','diagnostics_after/retract_succeeded'),
                ('joint','diagnostics_after/joint_position_rad'),
                ('velocity','diagnostics_after/joint_velocity_rad_s'),('object','diagnostics_after/object_pose_w'),
                ('substep_joint','physics_samples/joint_position_rad'),
                ('substep_velocity','physics_samples/joint_velocity_rad_s'),('substep_object','physics_samples/object_pose_w'),
                ('force','applied_wrench/force_w_N'),('torque','applied_wrench/torque_w_Nm')):
                expected=group[source][:len(arrays[name])]
                difference=np.abs(arrays[name].astype(np.float64)-expected.astype(np.float64))
                locations=np.argwhere(difference!=0)
                fields[name]=dict(bitwise_equal=bitwise_equal(arrays[name],expected),maximum=float(difference.max()),
                    first_nonzero_step=int(locations[0,0]) if len(locations) else None)
            result[phase]=dict(fields=fields,compared_steps=len(arrays['joint']),source_steps=len(group['command_clipped_pre_delay']),
                full_length_match=len(arrays['joint'])==len(group['command_clipped_pre_delay']))
        forces=group['applied_wrench/force_w_N'][:]
        nonzero=np.flatnonzero(np.abs(forces).reshape(len(forces),-1).max(-1)>0)
        result['first_nonzero_force']=None
        if len(nonzero):
            step=int(nonzero[0])
            result['first_nonzero_force']={name:group[path][step].tolist() for name,path in
                (('request','external_force_after_dr_w_N'),('source_quaternion','applied_wrench/object_quat_wxyz'),
                 ('body_buffer','applied_wrench/force_b_N'),('actual_world','applied_wrench/force_w_N'))}
            result['first_nonzero_force']['step']=step
    return result


def verify_dp_inputs(root, trials):
    results=[]
    for row in trials:
        path=root/'dp'/row['trial']
        arrays=np.load(path/'trace.npz')
        history_error=reference_error=0.
        plan_count=0
        with h5py.File(path/'inputs.h5','r') as stream:
            if len(stream)!=row['steps']:
                raise ValueError('Missing recorded native input steps')
            for step in range(row['steps']):
                group=stream[str(step)]
                if int(group['step_id'][()])!=step:
                    raise ValueError('Native input step identity mismatch')
                if step%12==0:
                    plan=group['plan'][:]
                    if plan.shape!=(64,29) or not np.isfinite(plan).all():
                        raise ValueError('Invalid recorded H64 plan')
                    plan_count+=1
                reference_error=max(reference_error,float(np.abs(group['policy/reference'][0]-plan[step%12]).max()))
                history=group['policy/action_history'][:]
                if step:
                    history_error=max(history_error,float(np.abs(history[:29]-arrays['command'][step-1]).max()),
                        float(np.abs(history[29:58]-arrays['delayed'][step-1]).max()))
                    if history[-1]!=1:
                        raise ValueError('Missing execution acknowledgment')
                elif np.any(history):
                    raise ValueError('First input must mask unavailable history')
                if not np.array_equal(group['command'][0],arrays['command'][step]):
                    raise ValueError('Recorded command mismatch')
        finite=all(np.isfinite(value).all() for value in arrays.values())
        legal=bool(np.max(np.abs(arrays['command']))<=1)
        if not finite or not legal or history_error or reference_error:
            raise ValueError('Recorded DP input/commit contract failed')
        results.append(dict(trial=row['trial'],steps=row['steps'],plans=plan_count,commit=12,
            reference_max_error=reference_error,previous_actual_command_and_delayed_max_error=history_error,
            finite=bool(finite),legal_commands=legal))
    return results


def analyze(root, output):
    output.mkdir(parents=True,exist_ok=True)
    session=json.loads((root/'session.json').read_text())
    manifest=json.loads(MANIFEST.read_text())
    entries={row['id']:row for row in manifest['episodes']}
    phases={}
    for phase in ('t0','source_origin','fresh','fresh_exact','dp'):
        path=root/phase/'report.json'
        if path.exists():
            phases[phase]=json.loads(path.read_text())
            for row in phases[phase].get('trials',[]):
                row.setdefault('wrench_reconstruction','actual_world')
    localization=[]
    for phase in ('t0','source_origin'):
        for row in phases.get(phase,{}).get('trials',[]):
            entry=entries[row['identity']]
            arrays=np.load(root/phase/row['trial']/'trace.npz')
            with h5py.File(MANIFEST.parent/entry['directory']/'data.h5','r') as stream:
                group=stream['episodes'][entry['episode']]
                count=min(len(arrays['joint']),entry['length'])
                contact=np.linalg.norm(group['transition/contact_substeps_after_w_N'][:count],axis=-1).max(axis=(1,2))
                touching=np.flatnonzero(contact>.1)
                first_contact=int(touching[0]) if len(touching) else None
                q_error=np.abs(arrays['joint'][:count]-group['transition/diagnostics_after/joint_position_rad'][:count]).max(-1)
                source_substeps=group['transition/kinematics_after/finger_joint_position_rad'][:count]
                substep_error=np.abs(arrays['substep_joint'][:count]-source_substeps)
                locations=np.argwhere(substep_error>1e-6)
                first=None
                if len(locations):
                    step,substep,joint=[int(value) for value in locations[0]]
                    first=dict(step=step,physics_substep=substep,canonical_joint=joint,
                        source_rad=float(source_substeps[step,substep,joint]),
                        replay_rad=float(arrays['substep_joint'][step,substep,joint]),
                        absolute_error_rad=float(substep_error[step,substep,joint]),
                        source_contact_max_N=float(contact[step]))
                metadata=json.loads(stream.attrs['metadata_json'])
                expected_object=group['transition/diagnostics_after/object_pose_w'][:count].copy()
                if phase=='t0':
                    expected_object[:,:3]-=np.asarray(metadata['env_origins_w'][0],dtype=np.float32)
                object_delta=np.abs(arrays['object'][:count]-expected_object)
                object_locations=np.argwhere(object_delta>1e-8)
                first_object=None
                if len(object_locations):
                    step,component=[int(value) for value in object_locations[0]]
                    first_object=dict(step=step,pose_component=component,source=float(expected_object[step,component]),
                        replay=float(arrays['object'][step,component]),absolute_error=float(object_delta[step,component]))
                localization.append(dict(phase=phase,identity=row['identity'],mode=row['mode'],
                    control_bytewise_equal={name:bitwise_equal(arrays[name][:count],group['transition/'+source][:count])
                        for name,source in (('command','command_clipped_pre_delay'),('delayed','control_trace/delayed_command'),
                                            ('delay_index','control_trace/delay_index'),('target','target_after_rad'))},
                    first_source_fingertip_contact_step=first_contact,
                    precontact_q_max_rad=float(q_error[:first_contact].max()) if first_contact else None,
                    precontact_q_bytewise_equal=bitwise_equal(arrays['joint'][:first_contact],
                        group['transition/diagnostics_after/joint_position_rad'][:first_contact]) if first_contact else None,
                    first_substep_joint_error_over_1urad=first,
                    first_object_component_error_over_1e_8=first_object,
                    physical_error_scope='Source only records per-substep joint position, not per-substep qd or object pose'))
    grouped={}
    for row in phases.get('dp',{}).get('trials',[]):
        key=row['line']+'/'+row['mode']
        group=grouped.setdefault(key,dict(trials=0,successes=0,lifts=0,approach_proxies=0,max_subgoals=0))
        group['trials']+=1
        group['successes']+=int(row['success'])
        group['lifts']+=int(row['native_lift'])
        group['approach_proxies']+=int(row['approach_proxy'])
        group['max_subgoals']=max(group['max_subgoals'],row['subgoals'])
    resources_path=root/'monitor_exit.json'
    resources=(json.loads(resources_path.read_text())['summary'] if resources_path.exists() else
               json.loads((root/'resources/resources.json').read_text()))
    resources={key:value for key,value in resources.items() if key not in ('pid','latest')}
    report=dict(session=session,
        phases=phases,first_physical_divergences=localization,dp_groups=grouped,
        launch_identities={phase:json.loads((root/phase/'launch.json').read_text()) for phase in phases},
        fresh_precision=fresh_precision(root),dp_recording_contracts=verify_dp_inputs(root,phases.get('dp',{}).get('trials',[])),
        dp_control_audit=json.loads((root/'dp_control_audit.json').read_text()),resources=resources,
        monitor_exited=resources_path.exists(),original_manifest_unchanged=sha256(MANIFEST)==session['manifest_sha256'],
        limitations=['Historical initial state is observable-only; source world origin is a separate ablation',
            'Matched DP events repeat after source end and bypass original state-dependent wrench gating only in that explicit test mode',
            'Approach is a geometric proxy; aggregate fingertip force cannot establish object grasp',
            'Training demonstration initial states are not independent or OOD evaluation'])
    atomic_json(output/'results.json',report)
    plot(root,output,session,entries,phases)
    print(json.dumps(dict(phases={name:value.get('complete') for name,value in phases.items()},dp_groups=grouped)))


def plot(root,output,session,entries,phases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(4,2,figsize=(12,12),constrained_layout=True)
    for ordinal,identity in enumerate(session['selected']):
        entry=entries[identity]
        with h5py.File(MANIFEST.parent/entry['directory']/'data.h5','r') as stream:
            group=stream['episodes'][entry['episode']]
            metadata=json.loads(stream.attrs['metadata_json'])
            for phase in ('t0','source_origin'):
                for row in phases.get(phase,{}).get('trials',[]):
                    if row['identity']!=identity:
                        continue
                    arrays=np.load(root/phase/row['trial']/'trace.npz')
                    count=min(len(arrays['joint']),entry['length'])
                    q_error=np.abs(arrays['joint'][:count]-group['transition/diagnostics_after/joint_position_rad'][:count]).max(-1)
                    expected=group['transition/diagnostics_after/object_pose_w'][:count].astype(np.float64)
                    if phase=='t0':
                        expected[:,:3]-=np.asarray(metadata['env_origins_w'][0])
                    position,_=pose_error(arrays['object'][:count],expected)
                    label='source world coordinates' if phase=='source_origin' else row['mode']
                    axes[ordinal,0].semilogy(np.maximum(q_error,1e-9),label=label)
                    axes[ordinal,1].semilogy(np.maximum(position,1e-9),label=label)
        for column,title in enumerate(('maximum joint error [rad]','object position error [m]')):
            axes[ordinal,column].axhline(1e-3,color='black',linestyle='--',linewidth=.7,label='tolerance')
            axes[ordinal,column].set_title(identity[-11:]+' / '+title)
            axes[ordinal,column].set_xlabel('native control step')
            axes[ordinal,column].grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    figure.savefig(output/'historical_replay_errors.png',dpi=150)
    plt.close(figure)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.root,args.output)
