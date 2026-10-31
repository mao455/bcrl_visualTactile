"""Bounded native diagnostics from explicitly approximate saved demonstration states."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .compact import sha256
from .diagnostic_recording import snapshot_before_policy, write_policy_step, write_tensors
from .training_resources import atomic_json


def initialize(root, source, evidence):
    if root.exists():
        raise FileExistsError('Fresh diagnostic directory required')
    root.mkdir(parents=True)
    started=time.time()
    distribution=json.loads((evidence/'dataset_distribution.json').read_text())
    atomic_json(root/'session.json',dict(started_epoch=started,deadline=started+10800,training_deadline=started+10800,
        source=str(source),evidence=str(evidence),selected_train=distribution['selected_train'],
        max_native_starts=12,parameter_updates=0,initialization='saved observable state restoration, not exact simulator restoration',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        diagnostic_input_version=2,
        source_sha256={str(path):sha256(path) for path in (Path(__file__),Path('visuotactile/diagnostic_recording.py'),
            Path('visuotactile/long_controller.py'),Path('visuotactile/long_data.py'),
            Path('visuotactile/models/long_horizon.py'),Path('visuotactile/sim/environment.py'),
            Path('isaacsimenvs/tasks/play/utils/action_utils.py'))}))
    atomic_json(root/'gpu_schedule.json',dict(target_percent=38,admission_percent=42,min_gpu_free_mib=6144,min_ram_gib=16))
    atomic_json(root/'status.json',dict(state='registered',utc_epoch=started))


def restore_observable_state(env, group, metadata):
    import numpy as np
    import torch
    initial={name:torch.from_numpy(np.asarray(value[0]).copy()).reshape(1,*value.shape[1:])
             for name,value in group['transition/diagnostics_before'].items()}
    original_origin=torch.tensor(metadata['env_origins_w'][0])
    target_origin=env.scene.env_origins[0]
    for name,key,velocity in (('object','object_pose_w','object_velocity_w_m_s_rad_s'),
        ('table','table_pose_w',None),('hole','hole_pose_w',None),('goal_viz','goal_pose_w','goal_velocity_w_m_s_rad_s')):
        pose=initial[key].clone()
        pose[:,:3]+=target_origin-original_origin
        body=getattr(env,name)
        body.write_root_pose_to_sim(pose)
        if name=='object':
            body.write_root_velocity_to_sim(initial[velocity])
        elif velocity and bool(initial[velocity].abs().max()>0):
            raise ValueError('Nonzero recorded kinematic velocity cannot be restored through a dynamic-body API')
        if name in ('table','hole'):
            env._kinematic_reset_poses[name]=pose[:,:3].clone()
    permutation=env._perm_canon_to_lab
    env.robot.write_joint_state_to_sim(initial['joint_position_rad'][:,permutation],initial['joint_velocity_rad_s'][:,permutation])
    target=torch.from_numpy(group['obs/actuator_target_rad'][0:1])[:,permutation]
    env._prev_targets[:]=target
    env._cur_targets[:]=target
    env.robot.set_joint_position_target(target)
    env._table_z_per_env[:]=initial['table_pose_w'][:,2]-original_origin[2]
    env.hole_pos[:]=initial['hole_pose_w'][:,:3]-original_origin
    env.hole_quat_wxyz[:]=initial['hole_pose_w'][:,3:]
    for destination,key in (('_successes','successes'),('retract_phase','retract_phase'),
        ('retract_succeeded','retract_succeeded'),('env_max_goals','max_goals'),('_lifted_object','lifted_object'),
        ('_object_init_z','object_initial_z_m'),('is_random_goal_env','is_random_goal'),('episode_length_buf','native_stage_step')):
        getattr(env,destination)[:]=initial[key].reshape_as(getattr(env,destination))
    for destination,key in (('_random_force_prob','external_force_probability'),('_random_torque_prob','external_torque_probability')):
        getattr(env,destination)[:]=torch.from_numpy(group['transition/'+key][0:1]).reshape_as(getattr(env,destination))
    if env._num_prelude_goals!=0:
        raise ValueError('Saved prelude hidden state unavailable; no approximate reconstruction implemented')
    env._fk_samples=[]
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.)
    env.configure_camera_views((-.7,-.35,1.15),(0.,-.03,.68))
    for camera in env.research_cameras:
        camera.reset()
    env.sim.render()
    env._get_observations()
    torch.set_rng_state(torch.from_numpy(group['transition/torch_rng_before_step'][0]))
    current=env._diagnostics()
    errors={}
    for name in ('joint_position_rad','joint_velocity_rad_s','object_pose_w','object_velocity_w_m_s_rad_s',
                 'table_pose_w','hole_pose_w','goal_pose_w'):
        expected=initial[name].clone()
        if name.endswith('pose_w'):
            expected[:,:3]+=target_origin-original_origin
        errors[name]=float((current[name]-expected).abs().max())
    errors['target']=float((env.research_packet['actuator_target_rad']-torch.from_numpy(group['obs/actuator_target_rad'][0:1])).abs().max())
    return dict(saved_observable_errors=errors,exact_simulator_restore=False,
        missing=['PhysX solver/contact warm-start','sensor histories','other persistent environment states',
                 'native multi-environment RNG consumption sequence','expert observation-delay queues'],
        queue='native reset clears action queue; first native action fills it; source step0 verified',
        rng='single initial CPU RNG restored; subsequent native randomization untouched; multi-environment draw schedule differs')


def evaluate(root):
    import numpy as np
    import torch
    import h5py
    from .study_resources import StudyLease
    session=json.loads((root/'session.json').read_text())
    source=Path(session['source'])
    lease=StudyLease(root,deadline=session['deadline'])
    with lease.block(cuda=False):
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=True,device='cpu',kit_args=(
            '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
            '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    env=None
    outcomes=[]
    try:
        import isaacsimenvs
        from evaluation.eval_isaacsim import _load_env_cfg
        from .long_controller import LongController
        from .long_data import MANIFEST
        from .models.long_horizon import FrozenDinoSpatial
        from .models.residual import observed_contact
        from .observations import TaskCommand
        from .sim.environment import InstrumentedAssemblyEnv
        torch.set_num_threads(2)
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        manifest=json.loads(MANIFEST.read_text())
        entries={row['id']:row for row in manifest['episodes']}
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
            env=InstrumentedAssemblyEnv(cfg,perception_version=2,task_command=TaskCommand('tight_insertion'),
                wrench_frame_mode='world_hold',camera_rig='wrist_bracket_close')
            dino=FrozenDinoSpatial().to('cuda:0').eval()
            controllers={line:LongController(torch.load(source/line/'best.pt',map_location='cpu',weights_only=False),dino)
                         for line in ('G00-B','G10-B')}
        for ordinal,identity in enumerate(session['selected_train']):
            entry=entries[identity]
            with h5py.File(MANIFEST.parent/entry['directory']/'data.h5','r') as stream:
                metadata=json.loads(stream.attrs['metadata_json'])
                group=stream['episodes'][entry['episode']]
                if int(group['obs/step_id'][0])!=0:
                    raise ValueError('Demonstration lacks step0')
                for mode in ('T0_recorded','G00-B','G10-B'):
                    attempt=f'{ordinal+1:02d}_{identity}_{mode}'
                    if (root/(attempt+'_started.json')).exists():
                        raise RuntimeError('Attempt already exists; no replay retry')
                    atomic_json(root/(attempt+'_started.json'),dict(identity=identity,mode=mode,utc_epoch=time.time()))
                    atomic_json(root/'status.json',dict(state='running',pid=os.getpid(),attempt=attempt,completed=len(outcomes)))
                    with lease.block():
                        env.reset(seed=20261031+ordinal)
                        restoration=restore_observable_state(env,group,metadata)
                        if not (root/'environment.json').exists():
                            environment=env.research_metadata()
                            environment.update(config=env.cfg.to_dict(),problem='tight_insertion',camera_mode='oblique')
                            atomic_json(root/'environment.json',environment)
                    if max(restoration['saved_observable_errors'].values())>1e-4:
                        raise RuntimeError('Observable initial state restoration outside tolerance')
                    controller=controllers.get(mode)
                    if controller:
                        controller.generator.manual_seed(20261031+ordinal)
                    trace=[]
                    start=time.monotonic()
                    source_length=entry['length']
                    bound=source_length if controller is None else (int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
                    with h5py.File(root/(attempt+'_inputs.h5'),'w') as inputs_file:
                        for step in range(bound):
                            packet=env.research_packet
                            before={key:value[0].cpu().numpy().copy() for key,value in env._diagnostics().items()}
                            with lease.block():
                                snapshot=snapshot_before_policy(env,controller)
                                if controller:
                                    command=controller.act(packet)
                                else:
                                    command=torch.from_numpy(group['transition/command_clipped_pre_delay'][step:step+1])
                                saved_input=write_policy_step(inputs_file,step,packet,snapshot,command,controller)
                                _,_,terminated,truncated,_=env.step(command)
                                transition=env.last_transition
                                write_tensors(saved_input.create_group('native'),dict(
                                    control_trace=transition['control_trace'],
                                    cpu_rng_after_preprocess=transition['torch_rng_after_preprocess'],
                                    cpu_rng_after_step=torch.get_rng_state(),target_after_rad=transition['target_after_rad'],
                                    diagnostics_after=transition['diagnostics_after']))
                                if controller:
                                    controller.observe_execution(transition)
                            after={key:value[0].cpu().numpy().copy() for key,value in transition['diagnostics_after'].items()}
                            expert=group['transition/command_clipped_pre_delay'][step] if step<source_length else np.full(29,np.nan)
                            expert_q=group['transition/diagnostics_after/joint_position_rad'][step] if step<source_length else np.full(29,np.nan)
                            expert_object=group['transition/diagnostics_after/object_pose_w'][step].copy() if step<source_length else np.full(7,np.nan)
                            expert_object[:3]-=np.asarray(metadata['env_origins_w'][0])
                            trace.append(dict(step=step,command=command[0].numpy(),expert=expert,expert_joint_after=expert_q,
                                expert_object_after=expert_object,joint_before=before['joint_position_rad'],joint_after=after['joint_position_rad'],
                                velocity_before=before['joint_velocity_rad_s'],object_after=after['object_pose_w'],
                                target_before=transition['target_before_rad'][0].numpy(),target_after=transition['target_after_rad'][0].numpy(),
                                delayed=transition['control_trace']['delayed_command'][0].numpy(),
                                delay_index=int(transition['control_trace']['delay_index'][0]),
                                force=packet['contact_w_N'][0].numpy(),gate=bool(observed_contact(packet['contact_w_N'],packet['contact_valid'])[0]),
                                lifted=bool(after['lifted_object']),subgoals=int(after['successes']),
                                applied_force=transition['applied_wrench']['force_w_N'][0].numpy(),
                                applied_torque=transition['applied_wrench']['torque_w_Nm'][0].numpy(),
                                reasons={key:bool(value[0]) for key,value in transition['termination_reasons'].items()}))
                            if bool(terminated[0] or truncated[0]):
                                break
                    arrays={key:np.stack([row[key] for row in trace]) for key in trace[0] if key!='reasons'}
                    np.savez_compressed(root/(attempt+'_trace.npz'),**arrays)
                    count=len(trace)
                    prefix=min(count,source_length)
                    height=arrays['object_after'][:,2]-float(group['transition/diagnostics_before/object_initial_z_m'][0])
                    command_error=np.max(np.abs(arrays['command'][:prefix]-arrays['expert'][:prefix]),axis=1)
                    q_error=np.max(np.abs(arrays['joint_after'][:prefix]-arrays['expert_joint_after'][:prefix]),axis=1)
                    deviation=np.flatnonzero(q_error>1e-3)
                    outcome=dict(identity=identity,mode=mode,attempt=attempt,steps=count,success=trace[-1]['reasons']['max_successes'],
                        reasons=trace[-1]['reasons'],recorded_actions_exhausted=controller is None and count==source_length and not any(trace[-1]['reasons'].values()),
                        restoration=restoration,contact_steps=int(arrays['gate'].sum()),native_lift=bool(arrays['lifted'].any()),
                        height_peak_m=float(height.max()),subgoals=int(arrays['subgoals'].max()),
                        first_q_deviation_over_1mrad_step=int(deviation[0]) if len(deviation) else None,
                        command_max_error=float(command_error.max()),first_command_max_error=float(command_error[0]),
                        first_joint_after_max_error_rad=float(q_error[0]),wall_s=time.monotonic()-start,
                        checkpoint_sha256=sha256(source/mode/'best.pt') if controller else None,
                        scope='training demonstration observable initial state; approximate simulator/RNG/sensor reconstruction; not exact replay')
                    outcomes.append(outcome)
                    atomic_json(root/'trials.json',outcomes)
                    print(json.dumps({key:outcome[key] for key in ('identity','mode','steps','success','height_peak_m','first_q_deviation_over_1mrad_step')}),flush=True)
        atomic_json(root/'status.json',dict(state='complete',trials=len(outcomes),utc_epoch=time.time()))
    finally:
        if env is not None:
            env.close()
        app.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--evidence',type=Path)
    parser.add_argument('--initialize',action='store_true')
    parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    if args.initialize:
        initialize(args.root,args.source,args.evidence)
    elif args.supervise:
        with (args.root/'monitor.log').open('w') as monitor_log, (args.root/'native.log').open('w') as native_log:
            monitor=subprocess.Popen([sys.executable,'-m','visuotactile.study_resources','--root',str(args.root)],stdout=monitor_log,stderr=subprocess.STDOUT)
            try:
                while not (args.root/'resources/resources.json').exists():
                    if monitor.poll() is not None:
                        raise RuntimeError('Monitor exited before readiness')
                    time.sleep(.5)
                worker=subprocess.Popen([sys.executable,'-m','visuotactile.evaluate_training_initials','--root',str(args.root)],stdout=native_log,stderr=subprocess.STDOUT)
                atomic_json(args.root/'owner.json',dict(pid=os.getpid(),monitor_pid=monitor.pid,worker_pid=worker.pid,utc_epoch=time.time()))
                code=worker.wait()
                atomic_json(args.root/'worker_exit.json',dict(returncode=code,utc_epoch=time.time()))
                if code:
                    atomic_json(args.root/'status.json',dict(state='software_interruption',returncode=code,utc_epoch=time.time()))
            finally:
                atomic_json(args.root/'STOP_MONITOR',dict(utc_epoch=time.time()))
                monitor.wait(timeout=30)
            if code:
                raise SystemExit(code)
    else:
        def interrupted(signum,frame):
            raise InterruptedError('Bounded diagnostic interrupted; preserve attempt')
        signal.signal(signal.SIGTERM,interrupted)
        evaluate(args.root)


if __name__=='__main__':
    main()
