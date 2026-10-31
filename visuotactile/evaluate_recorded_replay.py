"""Bounded real Kit replay and frozen DP evaluation with explicit event modes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import h5py
import numpy as np
import torch

from .compact import sha256
from .diagnostic_recording import snapshot_before_policy, write_policy_step, write_tensors
from .evaluate_training_initials import restore_observable_state
from .long_data import MANIFEST
from .replay_events import MODES, TOLERANCES, RecordedEvents, error_summary, pose_error
from .study_resources import StudyLease
from .training_resources import atomic_json


SOURCE = Path('outputs/visuotactile/run/dp_loss_trt_20261031_01')
SUFFIXES = ('e00_p000029','e00_p000184','e00_p000146','e03_p000151')
SOURCES = ('visuotactile/evaluate_recorded_replay.py','visuotactile/replay_events.py',
    'visuotactile/sim/replay_environment.py','visuotactile/evaluate_training_initials.py',
    'visuotactile/sim/environment.py','isaacsimenvs/tasks/play/utils/action_utils.py',
    'visuotactile/action_reference.py','visuotactile/long_controller.py','visuotactile/diagnostic_recording.py',
    'visuotactile/replay_fresh.py','visuotactile/audit_replay_control.py','visuotactile/long_data.py',
    'visuotactile/models/long_horizon.py','visuotactile/action_history.py','visuotactile/study_resources.py')


def initialize(root):
    root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(MANIFEST.read_text())
    selected=[next(row['id'] for row in manifest['episodes'] if row['id'].endswith(suffix)) for suffix in SUFFIXES]
    started=time.time()
    atomic_json(root/'session.json',dict(started_epoch=started,deadline=started+4*3600,training_deadline=started+4*3600,
        selected=selected,manifest_sha256=sha256(MANIFEST),checkpoint_sha256={line:sha256(SOURCE/line/'best.pt') for line in ('G00-B','G10-B')},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={name:sha256(name) for name in SOURCES},tolerances=TOLERANCES,
        planned_t0=12,planned_dp=16,max_additional_diagnostic_starts=7,parameter_updates=0,
        allocation='Four historical starts x three T0 modes; four starts x G00/G10 x recorded/native; bounded fallback only if T0 fails',
        event_extension='DP recorded mode repeats the source event block after its end; explicitly controlled intervention, not historical continuation',
        stage_metrics='Native lift/subgoals/retract; approach proxy min fingertip-object distance<=0.08m; grasp unverified without contact object identity'))
    atomic_json(root/'gpu_schedule.json',dict(target_percent=38,admission_percent=42,min_gpu_free_mib=6144,min_ram_gib=16))
    atomic_json(root/'status.json',dict(state='registered',utc_epoch=started))


def summarize(arrays, group, metadata, origin):
    length=min(len(arrays['command']),len(group['transition/command_clipped_pre_delay']))
    source=lambda name: np.asarray(group['transition/'+name][:length])
    sources=dict(command=source('command_clipped_pre_delay'),delayed=source('control_trace/delayed_command'),
        delay_index=source('control_trace/delay_index'),target=source('target_after_rad'),
        joint=source('diagnostics_after/joint_position_rad'),velocity=source('diagnostics_after/joint_velocity_rad_s'),
        force=source('applied_wrench/force_w_N'),torque=source('applied_wrench/torque_w_Nm'),
        substep_joint=source('kinematics_after/finger_joint_position_rad'))
    tolerances=dict(command=TOLERANCES['command'],delayed=TOLERANCES['command'],delay_index=0.,
        target=TOLERANCES['target_rad'],joint=TOLERANCES['joint_rad'],velocity=TOLERANCES['joint_velocity_rad_s'],
        force=TOLERANCES['force_N'],torque=TOLERANCES['torque_Nm'],substep_joint=TOLERANCES['joint_rad'])
    errors={name:error_summary(arrays[name][:length],value,tolerances[name]) for name,value in sources.items()}
    expected=source('diagnostics_after/object_pose_w').astype(np.float64)
    expected[:,:3]+=np.asarray(origin)-np.asarray(metadata['env_origins_w'][0])
    position,angle=pose_error(arrays['object'][:length].astype(np.float64),expected)
    errors['object_position']=error_summary(position,np.zeros_like(position),TOLERANCES['object_position_m'])
    errors['object_angle']=error_summary(angle,np.zeros_like(angle),TOLERANCES['object_angle_rad'])
    errors['stage']=error_summary(arrays['subgoals'][:length],source('diagnostics_after/successes'),0.)
    first=errors['joint']['first_exceeded']
    force_norm=np.linalg.norm(arrays['contact'],axis=-1).reshape(len(arrays['contact']),-1).max(-1)
    contact=np.flatnonzero(force_norm>.1)
    first_contact=int(contact[0]) if len(contact) else None
    return dict(compared_steps=length,errors=errors,first_observed_fingertip_contact=first_contact,
        first_joint_divergence_after_observed_contact=None if first is None or first_contact is None else first>=first_contact,
        contact_scope='Aggregate fingertip contact includes robot/table, not object-specific grasp',
        trajectory_within_tolerance=all(errors[name]['first_exceeded'] is None for name in
            ('command','delayed','delay_index','target','joint','velocity','force','torque','object_position','object_angle','stage')))


def run_trial(env, lease, output, group, metadata, *, mode, line, controller, seed, source_origin=False, exported_state=None,
              wrench_reconstruction='native_request'):
    output.mkdir(exist_ok=False)
    atomic_json(output/'started.json',dict(pid=os.getpid(),seed=seed,mode=mode,line=line,source_origin=source_origin,
        source_lane=metadata.get('source_env_index',0),source_envs=metadata.get('num_parallel_envs',1),
        wrench_reconstruction=wrench_reconstruction,utc_epoch=time.time()))
    with lease.block():
        env.clear_recorded_events()
        env.reset(seed=seed)
        if source_origin:
            env.scene.env_origins[0]=torch.tensor(metadata['env_origins_w'][0])
            env.robot.write_root_pose_to_sim(torch.from_numpy(group['transition/diagnostics_before/base_pose_w'][0:1]))
        restoration=restore_observable_state(env,group,metadata)
        if exported_state is not None:
            restored=[]
            for name,value in exported_state['env_tensors'].items():
                current=getattr(env,name,None)
                if torch.is_tensor(current) and current.shape==value.shape and not name.startswith('research_'):
                    current.copy_(value.to(current.device))
                    restored.append(name)
            restoration['exported_tensor_fields_restored']=restored
            properties={}
            for body_name,values in exported_state['physics'].items():
                view=getattr(env,body_name).root_physx_view
                for label,method in (('masses','get_masses'),('inertias','get_inertias'),('materials','get_material_properties'),
                                     ('dof_stiffness','get_dof_stiffnesses'),('dof_damping','get_dof_dampings')):
                    if label in values:
                        properties[body_name+'/'+label]=float((getattr(view,method)().cpu()-values[label].cpu()).abs().max())
            restoration['exported_physics_property_errors']=properties
            restoration['missing']=['PhysX contact/solver/broadphase internal state','sensor implementation internal histories']
            restoration['rng']='Source initial RNG restored; all native draws retained; historical delay/wrench events explicitly selected'
            if any(value!=0 for value in properties.values()):
                raise ValueError('Fresh replay physics properties changed')
            torch.set_rng_state(torch.from_numpy(group['transition/torch_rng_before_step'][0]))
    atomic_json(output/'restoration.json',restoration)
    if max(restoration['saved_observable_errors'].values())>1e-4:
        raise ValueError('Initial observable restoration outside tolerance')
    events=RecordedEvents(group,mode,extend=controller is not None,wrench_reconstruction=wrench_reconstruction)
    if controller is not None:
        controller.generator.manual_seed(seed)
    length=len(group['transition/command_clipped_pre_delay'])
    bound=length if controller is None else (int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
    rows=[]
    started=time.monotonic()
    with h5py.File(output/'inputs.h5','w') as inputs:
        inputs.attrs['metadata_json']=json.dumps(metadata)
        for step in range(bound):
            packet=env.research_packet
            with lease.block():
                env.set_recorded_step(events,step)
                before=snapshot_before_policy(env,controller)
                began=time.perf_counter()
                command=(controller.act(packet) if controller is not None else
                    torch.from_numpy(group['transition/command_clipped_pre_delay'][step:step+1]))
                inference_s=time.perf_counter()-began
                saved=write_policy_step(inputs,step,packet,before,command,controller)
                _,_,terminated,truncated,_=env.step(command)
                transition=env.last_transition
                write_tensors(saved.create_group('transition'),transition)
                if controller is not None:
                    controller.observe_execution(transition)
            after=transition['diagnostics_after']
            take=lambda value:value[0].detach().cpu().numpy().copy()
            rows.append(dict(command=take(command),delayed=take(transition['control_trace']['delayed_command']),
                delay_index=take(transition['control_trace']['delay_index']),target=take(transition['target_after_rad']),
                target_before=take(transition['target_before_rad']),joint=take(after['joint_position_rad']),
                velocity=take(after['joint_velocity_rad_s']),object=take(after['object_pose_w']),
                force=take(transition['applied_wrench']['force_w_N']),torque=take(transition['applied_wrench']['torque_w_Nm']),
                substep_joint=take(transition['physics_samples']['joint_position_rad']),
                substep_velocity=take(transition['physics_samples']['joint_velocity_rad_s']),
                substep_object=take(transition['physics_samples']['object_pose_w']),
                contact=take(transition['contact_substeps_after_w_N']),
                lifted=take(after['lifted_object']),subgoals=take(after['successes']),
                retract=take(after['retract_succeeded']),inference_s=inference_s,
                fingertip_distance=float((after['finger_position_w'][0]-after['object_pose_w'][0,:3]).norm(dim=-1).min()),
                event_source_step=env.recorded_event['source_step'],event_cycle=env.recorded_event['cycle']))
            if step%100==0:
                atomic_json(output/'progress.json',dict(step=step,mode=mode,line=line,utc_epoch=time.time()))
            if bool(terminated[0] or truncated[0]):
                break
    arrays={key:np.stack([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(output/'trace.npz',**arrays)
    comparison=summarize(arrays,group,metadata,env.scene.env_origins[0].cpu().tolist())
    reasons={name:bool(value[0]) for name,value in transition['termination_reasons'].items()}
    elapsed=time.monotonic()-started
    result=dict(mode=mode,line=line,wrench_reconstruction=wrench_reconstruction,steps=len(rows),success=reasons['max_successes'],reasons=reasons,
        source_length=length,recorded_actions_exhausted=controller is None and len(rows)==length and not any(reasons.values()),
        source_origin=source_origin,comparison=comparison,native_lift=bool(arrays['lifted'].any()),
        subgoals=int(arrays['subgoals'].max()),retract=bool(arrays['retract'].any()),
        approach_proxy=bool((arrays['fingertip_distance']<=.08).any()),grasp_verified=None,
        height_peak_m=float(arrays['object'][:,2].max()-group['transition/diagnostics_before/object_initial_z_m'][0]),
        wall_s=elapsed,wall_control_hz=len(rows)/elapsed,inference_mean_s=float(arrays['inference_s'].mean()),
        inference_max_s=float(arrays['inference_s'].max()),physics_dt=env.physics_dt,control_dt=env.step_dt,
        matched_event_extension='repeat source block' if controller is not None and mode!='original' else None,
        checkpoint_sha256=sha256(SOURCE/line/'best.pt') if controller is not None else None)
    atomic_json(output/'result.json',result)
    print(json.dumps(dict(trial=output.name,success=result['success'],steps=len(rows),lift=result['native_lift'],
        target_error=comparison['errors']['target']['maximum'],q_error=comparison['errors']['joint']['maximum'])),flush=True)
    return result


def evaluate(root, phase, wrench_reconstruction='native_request'):
    session=json.loads((root/'session.json').read_text())
    if phase=='dp':
        if any(sha256(SOURCE/line/'best.pt')!=expected for line,expected in session['checkpoint_sha256'].items()):
            raise ValueError('Frozen DP checkpoint identity changed')
        audit=json.loads((root/'dp_control_audit.json').read_text())
        replay=json.loads((root/'t0/report.json').read_text())
        fixed=[row for row in replay['trials'] if row['mode']=='delay_wrench_fixed']
        if (not audit['passed'] or not replay['complete'] or len(fixed)!=4 or
                any(row['comparison']['errors'][name]['first_exceeded'] is not None
                    for row in fixed for name in ('command','delayed','delay_index','target','force','torque'))):
            raise ValueError('Recorded control-chain and real-data DP interface admission required')
    lease=StudyLease(root,deadline=session['deadline'])
    output=root/phase
    output.mkdir(exist_ok=False)
    atomic_json(output/'launch.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={name:sha256(name) for name in SOURCES},command=list(__import__('sys').argv),
        wrench_reconstruction=wrench_reconstruction,utc_epoch=time.time()))
    with lease.block(cuda=False):
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=True,device='cpu',kit_args=(
            '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
            '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    env=None
    results=[]
    try:
        import isaacsimenvs
        from evaluation.eval_isaacsim import _load_env_cfg
        from .sim.replay_environment import RecordedReplayEnv
        from .observations import TaskCommand
        torch.set_num_threads(2)
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        cfg=_load_env_cfg('Isaacsimenvs-PreciseAssembly-Direct-v0')
        cfg.seed=20261031
        cfg.scene.num_envs,cfg.scene.env_spacing=1,8.
        cfg.sim.device='cpu'
        cfg.precise_assembly.problem='tight_insertion'
        cfg.student_obs.image_modality,cfg.student_obs.camera_backend='rgbd','standard'
        cfg.student_obs.camera_update_period_s=0.
        cfg.student_obs.image_width=cfg.student_obs.image_input_width=448
        cfg.student_obs.image_height=cfg.student_obs.image_input_height=252
        cfg.sim.render.rendering_mode,cfg.sim.render.antialiasing_mode='quality','Off'
        with lease.block():
            env=RecordedReplayEnv(cfg,perception_version=2,task_command=TaskCommand('tight_insertion'),
                wrench_frame_mode='world_hold',camera_rig='wrist_bracket_close')
        if phase=='fresh':
            from .replay_fresh import fresh_expert_and_replay
            fresh_expert_and_replay(env,lease,output,wrench_reconstruction=wrench_reconstruction)
            return
        if phase=='fresh_exact':
            source=root/'fresh'
            with h5py.File(source/'expert_numeric.h5','r') as stream:
                metadata=json.loads(stream.attrs['metadata_json'])
                result=run_trial(env,lease,output/'replay',stream['episode'],metadata,mode='delay_wrench_fixed',
                    line='T0_fresh_exact',controller=None,seed=17,
                    exported_state=torch.load(source/'initial_export.pt',map_location='cpu',weights_only=False),
                    wrench_reconstruction=wrench_reconstruction)
            atomic_json(output/'report.json',dict(replay=result,complete=True,
                scope='Same saved fresh expert in a new Kit process; explicit '+wrench_reconstruction+' reconstruction'))
            return
        controllers={}
        if phase=='dp':
            from .long_controller import LongController
            from .models.long_horizon import FrozenDinoSpatial
            with lease.block():
                dino=FrozenDinoSpatial().to('cuda:0').eval()
                controllers={line:LongController(torch.load(SOURCE/line/'best.pt',map_location='cpu',weights_only=False),dino)
                    for line in ('G00-B','G10-B')}
        entries={row['id']:row for row in json.loads(MANIFEST.read_text())['episodes']}
        for ordinal,identity in enumerate(session['selected']):
            entry=entries[identity]
            with h5py.File(MANIFEST.parent/entry['directory']/'data.h5','r') as stream:
                metadata=json.loads(stream.attrs['metadata_json'])
                group=stream['episodes'][entry['episode']]
                trials=([(mode,'T0') for mode in MODES] if phase=='t0' else
                    [('delay_wrench_fixed','T0')] if phase=='source_origin' else
                    [(mode,line) for mode in ('delay_wrench_fixed','original') for line in ('G00-B','G10-B')])
                for mode,line in trials:
                    name=f'{ordinal+1:02d}_{identity}_{line}_{mode}'
                    atomic_json(root/'status.json',dict(state='running',phase=phase,trial=name,completed=len(results),pid=os.getpid()))
                    result=run_trial(env,lease,output/name,group,metadata,mode=mode,line=line,
                        controller=controllers.get(line),seed=20261031+ordinal,source_origin=phase=='source_origin',
                        wrench_reconstruction=wrench_reconstruction)
                    result.update(identity=identity,trial=name)
                    results.append(result)
                    if not (output/'environment.json').exists():
                        environment=env.research_metadata()
                        environment.update(config=env.cfg.to_dict(),env_origins_w=env.scene.env_origins.cpu().tolist())
                        atomic_json(output/'environment.json',environment)
                    atomic_json(output/'report.json',dict(trials=results,complete=False))
        atomic_json(output/'report.json',dict(trials=results,complete=True))
        atomic_json(root/'status.json',dict(state='phase_complete',phase=phase,trials=len(results),utc_epoch=time.time()))
    finally:
        if env is not None:
            env.close()
        app.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--initialize',action='store_true')
    parser.add_argument('--phase',choices=('t0','dp','source_origin','fresh','fresh_exact'),default='t0')
    parser.add_argument('--wrench-reconstruction',choices=('native_request','actual_world'),default='native_request')
    args=parser.parse_args()
    def interrupted(signum,frame):
        raise InterruptedError('Owned replay worker stopped; preserve partial diagnostic artifacts')
    signal.signal(signal.SIGTERM,interrupted)
    if args.initialize:
        initialize(args.root)
    else:
        evaluate(args.root,args.phase,args.wrench_reconstruction)


if __name__=='__main__':
    main()
