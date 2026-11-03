"""Frozen, pre-action legal State benchmark and shared native command runner."""
import argparse
import json
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
import torch

from .compact import sha256
from .environment_rng import IsolatedEnvironment, RandomStream
from .evaluate_baselines import RolloutVideo
from .evaluate_training_initials import restore_observable_state
from .initial_geometry import inspect, boxes as object_boxes
from .native_data import CAMERAS
from .native_evaluation import disable_omniclient_watches, make_env, kit_process_snapshot, exit_completed_kit_worker
from .native_v2_expert import create_expert, environment_identity, export_initial
from .state_resources import lease_for
from .native_v2_state import StateController
from .replay_fresh import append_numeric
from .state_history import HistoryController
from .training_resources import atomic_json


def collision_evidence(env):
    """Query real cooked PhysX shapes before any action; no integration probe."""
    import omni.usd
    import carb
    from omni.physx import get_physx_scene_query_interface
    from pxr import PhysicsSchemaTools, Sdf, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    root = env.object.root_physx_view.prim_paths[0]
    if not env.cfg.sim.enable_scene_query_support:
        raise RuntimeError('Headless scene queries disabled: empty hits cannot prove legality')
    shapes = [str(prim.GetPath()) for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(root+'/') and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
    usd_hits = []
    for shape in shapes:
        def report(hit):
            usd_hits.append(dict(shape=shape, collider=str(hit.collision), rigid_body=str(hit.rigid_body)))
            return True
        encoded = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(shape))
        get_physx_scene_query_interface().overlap_shape(*encoded, report, False)
    if not shapes:
        raise RuntimeError('No cooked object shapes found: '+root)
    # A large box around this lane must hit the actual object. This detects a
    # disabled/stale query backend without moving anything or integrating time.
    positive = []
    def positive_hit(hit):
        positive.append(str(hit.rigid_body)); return True
    position = env.object.data.root_pos_w[0].tolist()
    get_physx_scene_query_interface().overlap_box(carb.Float3(1., 1., 1.), carb.Float3(*position),
        carb.Float4(0., 0., 0., 1.), positive_hit, False)
    if root not in positive:
        raise RuntimeError('PhysX query positive control did not see the actual object')
    from pxr import UsdGeom
    initial = {key: value[0].cpu().numpy() for key, value in env._diagnostics().items()}
    boxes = inspect(dict(config=dict(assets=env.cfg.assets.to_dict())), initial)
    live_boxes = object_boxes(env.cfg.assets.object_urdf, initial['object_pose_w'])
    if len(live_boxes) != len(shapes):
        raise RuntimeError('Original object collider census changed')
    live_hits = []; transforms = []
    body = UsdGeom.Xformable(stage.GetPrimAtPath(root)).ComputeLocalToWorldTransform(0)
    q = initial['object_pose_w'][3:]
    for shape, (center, rotation, half) in zip(shapes, live_boxes):
        prim = stage.GetPrimAtPath(shape)
        if not prim.IsA(UsdGeom.Cube):
            raise RuntimeError('Live pose query requires original box GPrim')
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        local = world*body.GetInverse()
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        usd_half = np.array([np.linalg.norm(np.array(local)[i, :3]) for i in range(3)])*size/2
        if not np.allclose(usd_half, half, rtol=1e-5, atol=1e-7):
            raise RuntimeError('USD/URDF object collider dimensions differ')
        expected_center = np.asarray(initial['object_pose_w'][:3])+rotation@np.asarray(local.ExtractTranslation())
        if not np.allclose(center, expected_center, rtol=0, atol=1e-6):
            raise RuntimeError('USD/URDF object collider local positions differ')
        transforms.append(dict(shape=shape, usd_world_center=list(world.ExtractTranslation()),
            live_world_center=center.tolist(), half_extents=half.tolist()))
        def live_report(hit):
            live_hits.append(dict(shape=shape, collider=str(hit.collision), rigid_body=str(hit.rigid_body)))
            return True
        get_physx_scene_query_interface().overlap_box(carb.Float3(*half.tolist()), carb.Float3(*center.tolist()),
            carb.Float4(float(q[1]), float(q[2]), float(q[3]), float(q[0])), live_report, False)
        if not any(row['shape'] == shape and row['rigid_body'] == root for row in live_hits):
            raise RuntimeError('Live object box query failed self-hit positive control')
    external = [row for row in live_hits if row['rigid_body'] != root and not row['rigid_body'].startswith(root+'/')]
    return dict(object_body=root, shapes=shapes, query_positive_control=positive,
        usd_query_hits=usd_hits, collider_transforms=transforms, query_hits=live_hits, external_hits=external, boxes=boxes,
        legal=boxes['valid_box_initialization'] and not external,
        definition='Original verified object boxes at live PhysX pose vs all scene shapes; OBB check also passes',
        limitation='Overlap query is conservative: touching may be rejected; not a signed penetration-depth oracle')


def prepare(env, isolated, spec, source, output):
    """One reset and optional observable initialization, never trajectory forcing."""
    env.clear_recorded_events()
    teacher, _ = isolated.reset(spec['seed'])
    restoration = dict(kind='fresh_native_reset')
    if spec['cohort'] != 'fresh':
        entry = source['entries'][spec['identity']]
        last = {}; original = env._get_observations
        def observed():
            result = original(); last['teacher'] = result
            return result
        env._get_observations = observed
        try:
            with h5py.File(entry['path'], 'r') as stream, isolated.stream.scope():
                restoration = restore_observable_state(env, stream['episodes'][entry['episode']],
                    json.loads(stream.attrs['metadata_json']))
            teacher = last['teacher']
        finally:
            env._get_observations = original
        # restore_observable_state historically restores a source CPU RNG. That
        # source included RL inference draws; this new contract deliberately
        # starts a separately registered native-event stream after initialization.
        isolated.stream = RandomStream(spec['seed']+100000)
        restoration['event_stream_seed'] = spec['seed']+100000
        restoration['historical_physics_exact'] = False
    legal = collision_evidence(env)
    atomic_json(output/'initial_collision.json', legal)
    atomic_json(output/'restoration.json', restoration)
    atomic_json(output/'initial_observable.json', {k: v.cpu().tolist() for k, v in env._diagnostics().items()})
    return teacher, legal


@torch.no_grad()
def trial(env, lease, output, spec, source, player, checkpoint, *, saved=None, replay=None, video=False,position=False,
          inference=None, takeover_steps=None):
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'started.json', dict(spec=spec, utc=time.time(), checkpoint=checkpoint))
    isolated = IsolatedEnvironment(env, spec['seed'])
    with lease.block():
        env.action_control_mode = 'native_command_clipped_pre_delay'
        teacher, legal = prepare(env, isolated, spec, source, output)
        env.action_control_mode = 'absolute_joint_position_target_rad' if position else 'native_command_clipped_pre_delay'
        if not legal['legal']:
            result = dict(**spec, software_complete=True, legal=False, success=False,
                not_started=True, reason='pre-action collision rejection; no replacement by outcome')
            atomic_json(output/'result.json', result)
            return result
        metadata = env.research_metadata(); metadata['config'] = env.cfg.to_dict()
        config, config_hash = environment_identity(env)
        atomic_json(output/'metadata.json', json.loads(json.dumps(metadata, default=str)))
        player.reset()
        torch.save(export_initial(env, player), output/'initial_export.pt')
        torch.save(isolated.stream.state, output/'environment_rng_initial.pt')
        if saved is not None and saved.get('group','').startswith('I-'):
            from .native_controller import NativeController
            dino=None
            if saved['group']=='I-HistoryDP':
                from .models.native_vision import NativeDinoSpatial
                dino=NativeDinoSpatial().to('cuda:0').eval()
            controller=NativeController(saved,metadata,dino=dino)
        else:
            controller = None if saved is None else (HistoryController(saved, metadata, inference=inference) if 'kind' in saved else StateController(saved, metadata))
    rows = []; plans = []; raw_plans = []; timings = []; numeric = h5py.File(output/'numeric.h5', 'w')
    numeric.attrs['metadata_json'] = json.dumps(metadata, default=str)
    group = numeric.create_group('episode')
    movie = RolloutVideo(output/'rollout.mp4', 'expert' if saved is None else saved.get('kind', 'oldState'),
        spec['seed'], 0, CAMERAS) if video else None
    actor_parity = 0.
    try:
        bound = (int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
        if replay is not None:
            bound = len(replay)
        for tick in range(bound):
            packet = env.research_packet
            if movie:
                movie.append(packet, tick)
            with lease.block():
                actor = torch.cat((teacher['policy'], player.intr_reward_coef_embd), -1)
                # Expert shadow follows the actual observed student trajectory;
                # it never executes and its RNG is outside the environment stream.
                expert_command = player.get_action(actor, is_deterministic=True).clamp(-1, 1)
                if replay is not None:
                    command = torch.from_numpy(replay[tick:tick+1])
                    # Deliberately perturb policy RNG in this second runner.
                    torch.rand(37+tick % 17)
                elif controller is None:
                    command = expert_command
                elif takeover_steps is not None and tick >= takeover_steps:
                    env.action_control_mode = 'native_command_clipped_pre_delay'
                    command = expert_command
                elif isinstance(controller, HistoryController):
                    if inference is not None: torch.cuda.synchronize()
                    planning_started = time.perf_counter()
                    command = controller.act(actor, packet, env._diagnostics())
                    if inference is not None:
                        torch.cuda.synchronize()
                        timings.append(dict(tick=tick,wall_s=time.perf_counter()-planning_started,planning=tick%12==0))
                    expected = player.model.norm_obs(actor)[:, :140]
                    actual = controller.model.actor_input(actor.to(controller.device))[:, :140].cpu()
                    actor_parity = max(actor_parity, float((expected-actual).abs().max()))
                else:
                    command = controller.act(packet) if saved.get('group','').startswith('I-') else controller.act(packet, env._diagnostics())
                if controller is not None and tick % 12 == 0 and (takeover_steps is None or tick < takeover_steps):
                    plans.append(controller.plan[0].cpu().numpy())
                    if isinstance(controller, HistoryController): raw_plans.append(controller.raw_plan[0].numpy())
                if takeover_steps is not None and tick >= takeover_steps:
                    # Diagnostic hand-back to the real expert, with its recurrent
                    # shadow continuously updated above. Native queue stays native.
                    env.action_control_mode = 'native_command_clipped_pre_delay'
                    command = expert_command
                if not torch.isfinite(command).all():
                    raise FloatingPointError('Nonfinite policy output')
                before_rng = isolated.stream.state[2].clone()
                teacher, _, terminated, truncated, _ = isolated.step(command)
                transition = env.last_transition
                transition.update(expert_actor_input_diagnostic=actor.clone(),
                    torch_rng_before_step=before_rng[None], expert_shadow_command=expert_command.clone())
                if controller is not None and (takeover_steps is None or tick < takeover_steps):
                    controller.observe_execution(transition)
                before_packet = {k: v for k, v in packet.items() if torch.is_tensor(v)
                    and k not in ('image', 'depth_m', 'depth_valid')}
                append_numeric(group, dict(obs=before_packet, transition=transition), tick)
                after = transition['diagnostics_after']
                def take(value):
                    return value[0].cpu().numpy().copy()
                rows.append(dict(actor=take(actor), command=take(command), expert_shadow=take(expert_command),
                    delayed=take(transition['control_trace'].get('delayed_command',transition.get('position_applied_rad'))), delay_index=take(transition['control_trace']['delay_index']),
                    target=take(transition['target_after_rad']), joint=take(after['joint_position_rad']),
                    actuator_targets=take(transition['actuator_substep_target_rad']),
                    velocity=take(after['joint_velocity_rad_s']), object=take(after['object_pose_w']),
                    contact=take(transition['contact_substeps_after_w_N']), object_contact=take(transition['object_contact_force_w']),
                    force=take(transition['applied_wrench']['force_w_N']), torque=take(transition['applied_wrench']['torque_w_Nm']),
                    lifted=take(after['lifted_object']), subgoals=take(after['successes']),
                    retract=take(after['retract_succeeded']), alignment=take(transition['native_alignment_condition'])))
                if inference is not None:
                    rows[-1]['palm']=take(after['palm_body_pose_w'])
                    rows[-1]['fingertip_distance']=np.linalg.norm(take(after['finger_position_w'])-
                        take(after['object_pose_w'])[:3],axis=-1).min()
            if tick % 100 == 0:
                atomic_json(output/'progress.json', dict(tick=tick, utc=time.time()))
            if bool((terminated | truncated)[0]):
                break
        arrays = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
        np.savez_compressed(output/'trace.npz', **arrays)
        if plans:
            np.savez_compressed(output/'plans.npz', plans=np.stack(plans),
                **({'raw_plans':np.stack(raw_plans)} if raw_plans else {}))
        reasons = {k: bool(v[0]) for k, v in transition['termination_reasons'].items()}
        result = dict(**spec, software_complete=True, legal=True, success=reasons['max_successes'],
            reasons=reasons, steps=len(rows), native_lift=bool(arrays['lifted'].any()),
            native_subgoals=int(arrays['subgoals'].max()), native_alignment=bool(arrays['alignment'].any()),
            actor_normalization_max_error=actor_parity, environment_sha256=config_hash,
            checkpoint_sha256=sha256(checkpoint), action_exhausted=not any(reasons.values()),
            simulation_step_hz=60, physics_hz=120, plan_hz=5 if saved is not None else 60)
        result['action_control_mode']=env.action_control_mode
        result['actuator_target_max_error']=float(np.abs(arrays['actuator_targets']-arrays['target'][:,None,:]).max())
        result['tracking_rmse_rad']=float(np.sqrt(np.mean((arrays['joint']-arrays['target'])**2)))
        if position:
            result['submitted_target_max_error']=float(np.abs(arrays['command']-arrays['target']).max())
        if inference is not None:
            from .state_motion import plan_metrics, distribution
            from .native_evaluation import stable_grasp
            initial=json.loads((output/'initial_observable.json').read_text())
            result['contact']=stable_grasp(arrays,float(initial['object_pose_w'][0][2]))
            result['approach_proxy']=bool((arrays['fingertip_distance']<=.08).any())
            result['inference'] = inference
            result['plan_consistency'] = plan_metrics(np.stack(plans)) if len(plans)>1 else {}
            result['planning_latency_s'] = distribution([r['wall_s'] for r in timings if r['planning']])
            result['native_history_latency_s'] = distribution([r['wall_s'] for r in timings if not r['planning']])
            atomic_json(output/'inference_timing.json',dict(rows=timings,scope='synchronized wall including observation history, inference and host transfer; excludes resource waits and simulation'))
        if takeover_steps is not None:
            result['takeover'] = dict(student_ticks=takeover_steps,expert_recovery=True,
                limitation='Direct position to original native expert mode; original delay queue resumes unchanged. Not autonomous success.')
        if movie:
            movie.close(complete=True)
        atomic_json(output/'result.json', result)
        return result
    finally:
        numeric.close()
        if movie:
            movie.close()


def run(root, output, mode, job=None):
    output.mkdir(parents=True, exist_ok=False)
    session = json.loads((root/'session.json').read_text())
    source = json.loads((Path(session['source_root'])/'data.json').read_text())
    lease = lease_for(root, deadline=session['deadline'])
    atomic_json(output/'runtime_controls.json', disable_omniclient_watches())
    atomic_json(output/'runtime_before.json', kit_process_snapshot())
    atomic_json(output/'lifecycle.json',dict(state='launching',utc=time.time()))
    with lease.block(cuda=False):
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device='cpu', fast_shutdown=False, kit_args=(
            '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
            '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    env = None
    try:
        from .native_evaluation import omniclient_watch_state
        atomic_json(output/'app_ready.json',dict(utc=time.time(),watch_state=omniclient_watch_state()))
        atomic_json(output/'runtime_ready.json', kit_process_snapshot())
        if mode=='startup':
            for _ in range(10):app.update()
            atomic_json(output/'result.json',dict(software_complete=True,mode=mode,watch_state=omniclient_watch_state(),trials=[]))
            atomic_json(output/'lifecycle.json',dict(state='results_committed',utc=time.time()))
            return
        import isaacsimenvs
        env = make_env(True, lease, scene_queries=True)
        with lease.block():
            player, config = create_expert(env, session['expert_checkpoint'])
        atomic_json(output/'actor_configuration.json', config)
        env_config, digest = environment_identity(env)
        atomic_json(output/'environment.json', dict(config=env_config, sha256=digest))
        atomic_json(output/'lifecycle.json',dict(state='environment_ready',utc=time.time()))
        if mode == 'smoke':
            with lease.block():
                isolated = IsolatedEnvironment(env, 982501)
                teacher, _ = isolated.reset(982501)
                actor = torch.cat((teacher['policy'], player.intr_reward_coef_embd), -1)
                action = player.get_action(actor, is_deterministic=True).clamp(-1,1)
                isolated.step(action)
                if not torch.isfinite(env.last_transition['target_after_rad']).all():
                    raise FloatingPointError('Nonfinite smoke actuator target')
            atomic_json(output/'result.json',dict(software_complete=True,mode=mode,control_steps=1,trials=[]))
            return
        trials = []; saved = None; checkpoint = session['expert_checkpoint']
        if mode == 'freeze':
            specs = []
            # Initial-state-only screen. Every rejection is retained before any
            # expert/student outcome is observed. Never search by task success.
            for seed in session['fresh_candidates']:
                spec = dict(identity=f'fresh_{seed}', seed=seed, cohort='fresh')
                directory = output/spec['identity']; directory.mkdir()
                with lease.block():
                    _, legal = prepare(env, IsolatedEnvironment(env, seed), spec, source, directory)
                if legal['legal']:
                    specs.append(spec)
                if len(specs) == session['fresh_count']:
                    break
            for spec in session['saved_specs']:
                directory = output/spec['identity']; directory.mkdir()
                with lease.block():
                    _, legal = prepare(env, IsolatedEnvironment(env, spec['seed']), spec, source, directory)
                if not legal['legal']:
                    raise RuntimeError('Registered saved initialization illegal: '+spec['identity'])
                specs.append(spec)
            for spec in session.get('validation_candidates',[]):
                directory=output/spec['identity'];directory.mkdir()
                with lease.block():
                    _,legal=prepare(env,IsolatedEnvironment(env,spec['seed']),spec,source,directory)
                if legal['legal']:specs.append(spec)
                if sum(s['cohort']=='validation' for s in specs)==session['validation_count']:break
            if len(specs) != session['fresh_count']+len(session['saved_specs'])+session.get('validation_count',0):
                raise RuntimeError('Predeclared candidate pool lacks enough legal starts')
            atomic_json(output/'benchmark.json', dict(specs=specs, configuration_sha256=digest,
                outcome_selected=False, frozen_before_expert=True))
        else:
            benchmark = json.loads((root/'freeze/benchmark.json').read_text())
            if benchmark['configuration_sha256'] != digest:
                raise ValueError('Frozen environment configuration changed')
            specs = benchmark['specs']
            job_data = json.loads(job.read_text()) if job else None
            if mode in ('bc', 'dp', 'old'):
                specs = [s for s in specs if s['cohort'] != 'fresh']
                checkpoint = (job_data['checkpoint'] if job_data else
                    session['old_state_checkpoint'] if mode == 'old' else str(root/(mode+'_train')/'best.pt'))
                if job_data and sha256(checkpoint)!=job_data['checkpoint_sha256']:
                    raise ValueError('Checkpoint hash drift')
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if job_data:
                specs = job_data.get('specs', specs)
            if mode in ('replay','position_replay'):
                specs = specs[:2]
            for i, spec in enumerate(specs):
                replay = None
                if mode in ('replay','position_replay'):
                    replay = np.load(root/'expert'/spec['identity']/'trace.npz')['target' if mode=='position_replay' else 'command']
                position=mode=='position_replay' or (saved is not None and saved.get('contract',{}).get('action')=='absolute_joint_position_target_rad')
                result = trial(env, lease, output/spec['identity'], spec, source, player, checkpoint,
                    saved=saved, replay=replay,position=position, video=(job_data.get('video', i == 0) if job_data else i == 0) and mode not in ('replay','position_replay'),
                    inference=job_data.get('inference') if job_data else None,
                    takeover_steps=job_data.get('takeover_steps') if job_data else None)
                if mode in ('replay','position_replay') and result.get('legal'):
                    with np.load(root/'expert'/spec['identity']/'trace.npz') as original, np.load(output/spec['identity']/'trace.npz') as current:
                        result['bytewise_fields'] = {key: original[key].shape == current[key].shape and
                            original[key].dtype == current[key].dtype and original[key].tobytes() == current[key].tobytes()
                            for key in original.files if key not in (('expert_shadow','command','delayed','delay_index') if position else ('expert_shadow',))}
                trials.append(result)
                atomic_json(output/'progress.json', dict(completed=len(trials), trials=trials))
        atomic_json(output/'result.json', dict(software_complete=True, mode=mode, trials=trials))
        atomic_json(output/'lifecycle.json',dict(state='results_committed',utc=time.time()))
    except BaseException as error:
        atomic_json(output/'worker_error.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        atomic_json(output/'lifecycle.json',dict(state='closing',utc=time.time()))
        atomic_json(output/'runtime_closing.json', kit_process_snapshot())
        if session.get('kit_gate',{}).get('exit_on_error') and (output/'worker_error.json').exists():
            import os
            import sys
            atomic_json(output/'failed_exit_requested.json', dict(pid=os.getpid(), utc=time.time(),
                reason='Durable worker error; process isolation replaces known hanging SDK cleanup'))
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
        if session.get('kit_gate',{}).get('shutdown') == 'isolated_process_exit' and (output/'result.json').exists() and not (output/'worker_error.json').exists():
            exit_completed_kit_worker(output)
        if env is not None:
            env.close()
        app.close()
        atomic_json(output/'closed.json',dict(utc=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['freeze', 'expert', 'replay', 'position_replay','startup', 'smoke', 'old', 'bc', 'dp'], required=True)
    p.add_argument('--job', type=Path)
    a = p.parse_args(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    run(a.root, a.output, a.mode, a.job)
