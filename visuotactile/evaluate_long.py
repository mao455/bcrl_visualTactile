"""Six preregistered autonomous candidate diagnostics, unchanged native simulator."""
import argparse
import json
import os
from pathlib import Path
import time
import subprocess

from .compact import sha256
from .long_resources import Lease
from .training_resources import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    root = args.root
    output = root/'closed_loop'
    output.mkdir(exist_ok=False)
    session = json.loads((root/'session.json').read_text())
    deadline = session['deadline']
    lease = Lease(root)
    def block(cuda=True):
        return lease.block(deadline=deadline,cuda=cuda)
    groups = session['groups']
    sources = {group:root/(group+'_full')/'best.pt' for group in groups}
    atomic_json(output/'status.json',dict(state='initializing',pid=os.getpid(),started_epoch=time.time()))
    for group,path in sources.items():
        tested = json.loads((path.parent/'test.json').read_text())
        if tested['checkpoint_sha256'] != sha256(path):
            raise ValueError('Only the once-tested validation-selected checkpoint may be deployed')
    atomic_json(output/'plan.json',dict(seeds=session['diagnostic_seeds'],groups=groups,
        autonomous=True,formal_denominators_unchanged=True,horizon=64,replan=12,
        checkpoint_hashes={group:sha256(path) for group,path in sources.items()},
        invalid_policy='record before action; no replacement seed or outcome selection',
        video='first preregistered seed per group only',
        native_bound='(max_goals+2)*(native max_episode_length+1); external cutoff labeled censored',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={str(p):sha256(p) for p in (Path(__file__),Path('visuotactile/long_controller.py'),
            Path('visuotactile/models/long_horizon.py'),Path('visuotactile/long_resources.py'),
            Path('visuotactile/long_data.py'),Path('visuotactile/sim/environment.py'))}))
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    with block(cuda=False):
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True,enable_cameras=True,device='cpu',kit_args=(
            '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
            '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    env = video = None
    trace = []
    attempt = None
    trials = []
    try:
        import numpy as np
        import torch
        import isaacsimenvs
        from evaluation.eval_isaacsim import _load_env_cfg
        from .sim.environment import InstrumentedAssemblyEnv
        from .observations import TaskCommand
        from .contracts import assert_policy_compatible, assert_execution_compatible
        from .initial_geometry import inspect
        from .models.long_horizon import FrozenDinoSpatial
        from .long_controller import LongController
        from .evaluate_baselines import RolloutVideo
        torch.set_num_threads(2)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        cfg = _load_env_cfg('Isaacsimenvs-PreciseAssembly-Direct-v0')
        cfg.seed = session['diagnostic_seeds'][0]
        cfg.scene.num_envs,cfg.scene.env_spacing = 1,8.
        cfg.sim.device = 'cpu'
        cfg.precise_assembly.problem = 'tight_insertion'
        cfg.student_obs.image_modality,cfg.student_obs.camera_backend = 'rgbd','standard'
        cfg.student_obs.camera_update_period_s = 0.
        cfg.student_obs.image_width = cfg.student_obs.image_input_width = 448
        cfg.student_obs.image_height = cfg.student_obs.image_input_height = 252
        cfg.sim.render.rendering_mode,cfg.sim.render.antialiasing_mode = 'quality','Off'
        with block():
            env = InstrumentedAssemblyEnv(cfg,perception_version=2,task_command=TaskCommand('tight_insertion'),
                wrench_frame_mode='world_hold',camera_rig='wrist_bracket_close')
            env.configure_camera_views((-.7,-.35,1.15),(0.,-.03,.68))
            dino = FrozenDinoSpatial().to('cuda:0').eval()
            controllers = {group:LongController(torch.load(path,map_location='cpu',weights_only=False),dino)
                           for group,path in sources.items()}
            dino_hash = sha256(Path(torch.hub.get_dir())/'checkpoints/dinov2_vits14_pretrain.pth')
            for group,controller in controllers.items():
                if controller.model.cfg.group != group:
                    raise ValueError('Checkpoint group differs from allocated group')
                if controller.vision['dino_weights_sha256'] != dino_hash:
                    raise ValueError('Frozen DINO weight identity differs')
        for seed in session['diagnostic_seeds']:
            reference_initial = None
            for group,controller in controllers.items():
                attempt = f'{group}_s{seed}'
                trace = []
                plans = []
                atomic_json(output/'status.json',dict(state='running',pid=os.getpid(),attempt=attempt,
                    completed=len(trials),started_epoch=time.time()))
                with block():
                    env.reset(seed=seed)
                metadata = env.research_metadata()
                metadata.update(problem='tight_insertion',camera_mode='oblique',config=cfg.to_dict())
                assert_policy_compatible(controller.metadata,metadata)
                assert_execution_compatible(controller.metadata,metadata)
                if not (output/'environment.json').exists():
                    atomic_json(output/'environment.json',metadata)
                initial = {name:value[0].cpu().numpy().copy() for name,value in env._diagnostics().items()}
                initial['actuator_target_rad'] = env.research_packet['actuator_target_rad'][0].numpy().copy()
                initial.update(force_probability=env._random_force_prob[0].numpy().copy(),
                    torque_probability=env._random_torque_prob[0].numpy().copy(),
                    object_mass=env._object_mass[0].numpy().copy())
                np.savez_compressed(output/f'{attempt}_initial.npz',**initial)
                geometry = inspect(metadata,initial)
                if not geometry['valid_box_initialization']:
                    trials.append(dict(seed=seed,group=group,outcome='invalid_before_action',geometry=geometry))
                    atomic_json(output/'trials.json',trials)
                    continue
                if reference_initial is None:
                    reference_initial = initial
                error = max(float(np.max(np.abs(value.astype(float)-reference_initial[name].astype(float))))
                            for name,value in initial.items())
                if error > 1e-5:
                    raise RuntimeError('Paired native initial states differ')
                controller.generator.manual_seed(seed+1000000)
                if seed == session['diagnostic_seeds'][0]:
                    video = RolloutVideo(output/f'{group}_s{seed}.mp4',group,seed,controller.checkpoint_epoch,metadata['camera_ids'])
                trace = []
                done = False
                started = time.monotonic()
                natural_bound = (int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
                for step in range(natural_bound):
                    packet = env.research_packet
                    rng = torch.get_rng_state().clone()
                    with block():
                        command = controller.act(packet)
                        if step%12 == 0:
                            plans.append(controller.plan[0].numpy().copy())
                        if not torch.equal(rng,torch.get_rng_state()):
                            raise RuntimeError('Policy changed native CPU RNG')
                        if video is not None:
                            video.append(packet,step,status=f'{group} | new trained candidate | H64 / 5 Hz | seed {seed}')
                        _,_,terminated,truncated,_ = env.step(command)
                    transition = env.last_transition
                    controller.observe_execution(transition)
                    diagnostic = transition['diagnostics_after']
                    row = dict(command=command,reference=controller.reference,
                        submitted=transition['submitted_action'],
                        clipped=transition['command_clipped_pre_delay'],
                        time_before=transition['time_before_s'],time_after=transition['time_after_s'],
                        target_before=transition['target_before_rad'],target_after=transition['target_after_rad'],
                        joint_before=transition['diagnostics_before']['joint_position_rad'],
                        joint_after=diagnostic['joint_position_rad'],object_pose=diagnostic['object_pose_w'],
                        force=transition['contact_substeps_after_w_N'],delayed=transition['control_trace']['delayed_command'],
                        successes=diagnostic['successes'],lifted=diagnostic['lifted_object'],
                        finger_distance=(diagnostic['finger_position_w']-diagnostic['object_pose_w'][:,None,:3]).norm(dim=-1).mean(-1))
                    trace.append({name:value[0].detach().cpu().numpy().copy() for name,value in row.items()})
                    if step%60 == 0:
                        atomic_json(output/'status.json',dict(state='running',pid=os.getpid(),attempt=attempt,
                            completed=len(trials),step=step,wall_s=time.monotonic()-started))
                    done = bool((terminated|truncated)[0])
                    if done:
                        break
                arrays = {name:np.stack([row[name] for row in trace]) for name in trace[0]}
                file = output/f'{group}_s{seed}.npz'
                np.savez_compressed(file,**arrays,plans=np.stack(plans),
                    generation_steps=np.arange(len(plans))*12)
                video_info = video.close(complete=done) if video is not None else None
                video = None
                force = np.linalg.norm(arrays['force'],axis=-1)
                contact = (force>.1).any(axis=(1,2))
                jumps = np.diff(arrays['command'],axis=0)
                boundary = np.arange(1,len(trace))%12 == 0
                success = bool(done and diagnostic['successes'][0]>=diagnostic['max_goals'][0]
                               and diagnostic['retract_succeeded'][0])
                result = dict(group=group,seed=seed,steps=len(trace),success=success,
                    outcome='success' if success else 'failure' if done else 'censored',
                    reasons=[name for name,value in transition['termination_reasons'].items() if bool(value[0])],
                    initial_match_max_error=error,checkpoint_sha256=sha256(sources[group]),
                    epoch=controller.checkpoint_epoch,updates=controller.updates,trace=file.name,video=video_info,
                    max_subgoals=int(arrays['successes'].max()),native_lift=bool(arrays['lifted'].any()),
                    max_object_height_change_m=float((arrays['object_pose'][:,2]-initial['object_initial_z_m']).max()),
                    contact_steps=int(contact.sum()),peak_force_N=float(force.max()),
                    first_contact=int(np.flatnonzero(contact)[0]) if contact.any() else None,
                    initial_finger_distance_m=float(arrays['finger_distance'][0]),
                    minimum_finger_distance_m=float(arrays['finger_distance'].min()),
                    final_finger_distance_m=float(arrays['finger_distance'][-1]),
                    approach_fraction=float((np.diff(arrays['finger_distance']) < -1e-4).mean()),
                    retreat_fraction=float((np.diff(arrays['finger_distance']) > 1e-4).mean()),
                    plan_boundary_command_rmse=float(np.sqrt(np.mean(jumps[boundary]**2))),
                    saturation_fraction=float((np.abs(arrays['command'])>=.99999).mean()),
                    reference_controller_rmse_rad=float(np.sqrt(np.mean((arrays['reference']-arrays['target_after'])**2))),
                    reference_motion_rmse_rad=float(np.sqrt(np.mean((arrays['reference']-arrays['joint_after'])**2))),
                    wall_s=time.monotonic()-started,grasp='aggregate finger force has no partner identity; sustained grasp not established by force alone')
                trials.append(result)
                atomic_json(output/'trials.json',trials)
                print('TRIAL',json.dumps(result),flush=True)
        atomic_json(output/'report.json',dict(trials=trials,complete=len(trials)==6,
            resources=json.loads((root/'resources/resources.json').read_text())))
        atomic_json(output/'status.json',dict(state='complete',trials=len(trials),pid=os.getpid()))
    except BaseException as error:
        if trace and attempt:
            np.savez_compressed(output/f'{attempt}_interrupted.npz',
                **{name:np.stack([row[name] for row in trace]) for name in trace[0]})
        atomic_json(output/'interruption.json',dict(error=repr(error),attempt=attempt,
            recorded_steps=len(trace),completed_trials=len(trials)))
        atomic_json(output/'status.json',dict(state='interrupted',error=repr(error),attempt=attempt,pid=os.getpid()))
        raise
    finally:
        if video is not None:
            video.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
