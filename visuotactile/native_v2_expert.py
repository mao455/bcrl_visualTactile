"""Registered fresh expert, four-lane comparison and frozen State diagnostics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
import torch

from .compact import sha256
from .evaluate_baselines import RolloutVideo
from .evaluate_recorded_replay import run_trial
from .native_data import CAMERAS
from .native_evaluation import disable_omniclient_watches, make_env
from .native_v2_resources import LeaseV2
from .native_v2_state import StateController
from .replay_fresh import append_numeric
from .training_resources import atomic_json


def create_expert(env, checkpoint):
    from evaluation.eval_isaacsim import _configure_agent
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner, _load_checkpoint_weights
    config = _configure_agent('Isaacsimenvs-PreciseAssembly-Direct-v0', 'rl_games_sapg_cfg_entry_point',
        rl_device='cpu', num_envs=env.num_envs, deterministic=True, games=1, extra_overrides={})
    register_rlgames_env(env, rl_device='cpu', clip_obs=config['params']['env']['clip_observations'], clip_actions=1.)
    runner = Runner(); runner.load(config); runner.reset()
    player = runner.create_player(); player.set_weights(_load_checkpoint_weights(player, str(checkpoint)))
    player.has_batch_dimension = True; player.reset()
    if player.intr_reward_coef_embd is not None:
        player.intr_reward_coef_embd.fill_(50.)
    return player, config


def export_initial(env, player):
    physics = {}
    for name in ('robot', 'object', 'table', 'hole', 'goal_viz'):
        body = getattr(env, name); view = body.root_physx_view
        physics[name] = dict(root_state_w=body.data.root_state_w.clone())
        for label, method in (('masses', 'get_masses'), ('inertias', 'get_inertias'),
                              ('materials', 'get_material_properties'), ('dof_stiffness', 'get_dof_stiffnesses'),
                              ('dof_damping', 'get_dof_dampings'), ('coms', 'get_coms')):
            if hasattr(view, method):
                physics[name][label] = getattr(view, method)().clone()
    return dict(env_tensors={name: value.detach().cpu().clone() for name, value in vars(env).items()
                            if torch.is_tensor(value)}, physics=physics, actor_states=copy.deepcopy(player.states),
                cpu_rng=torch.get_rng_state(), missing=['PhysX contact/solver/broadphase internal state'])


def lane_values(value, lane):
    if torch.is_tensor(value):
        return value[lane:lane+1]
    if isinstance(value, dict):
        return {key: lane_values(item, lane) for key, item in value.items()}
    return value


def environment_identity(env):
    config = json.loads(json.dumps(env.cfg.to_dict(), default=str))
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return config, digest


@torch.no_grad()
def fresh_trial(env, lease, output, seed, player, expert_checkpoint, *, saved=None, student_checkpoint=None,
                video_enabled=False):
    output.mkdir(parents=True, exist_ok=False)
    expert = saved is None
    policy = 'released_online_expert' if expert else saved.get('contract', {}).get('diagnostic_model', 'old_S-NativeDP')
    checkpoint = expert_checkpoint if expert else student_checkpoint
    config, environment_hash = environment_identity(env)
    atomic_json(output/'started.json', dict(seed=seed, policy=policy, utc=time.time(), lanes=env.num_envs,
        checkpoint_sha256=sha256(checkpoint), environment_sha256=environment_hash))
    with lease.block():
        env.clear_recorded_events(); player.reset()
        teacher, _ = env.reset(seed=seed)
        env.configure_camera_views((-.7, -.35, 1.15), (0., -.03, .68))
        metadata = env.research_metadata()
        metadata.update(config=config, env_origins_w=env.scene.env_origins.cpu().tolist(),
                        num_parallel_envs=env.num_envs, seed=seed, expert_checkpoint_sha256=sha256(expert_checkpoint),
                        purpose='Separate diagnostic; never appended to original400')
        exported = export_initial(env, player)
        torch.save(exported, output/'initial_export.pt')
        initial = {key: value.cpu().numpy().tolist() for key, value in env._diagnostics().items()}
        atomic_json(output/'initial_observable.json', initial)
        if not expert:
            if env.num_envs != 1:
                raise ValueError('State deployment currently single lane only')
            from .native_v2_bc_compare import ACTOR_CONTRACT, ActorController
            actor_student = saved.get('contract') == ACTOR_CONTRACT
            controller = ActorController(saved) if actor_student else StateController(saved, metadata)
            if not actor_student:
                controller.generator.manual_seed(7091)
    atomic_json(output/'metadata.json', metadata)
    video = RolloutVideo(output/'rollout.mp4', policy, seed, 0, CAMERAS) if video_enabled else None
    streams = []; groups = []; rows = [[] for _ in range(env.num_envs)]; outcomes = [None]*env.num_envs
    try:
        for lane in range(env.num_envs):
            stream = h5py.File(output/f'lane_{lane:02d}.h5', 'w')
            local = copy.deepcopy(metadata)
            local['env_origins_w'] = [metadata['env_origins_w'][lane]]
            local['joint_limits_rad'] = [metadata['joint_limits_rad'][lane]]
            local['source_env_index'] = lane
            stream.attrs['metadata_json'] = json.dumps(local)
            streams.append(stream); groups.append(stream.create_group('episode'))
        bound = (int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
        for tick in range(bound):
            packet = env.research_packet
            if video:
                video.append(packet, tick)
            with lease.block():
                before_rng = torch.get_rng_state().clone()
                actor = player.obs_to_torch(teacher['policy'])
                if player.intr_reward_coef_embd is not None:
                    actor = torch.cat((actor, player.intr_reward_coef_embd), dim=1)
                command = (player.get_action(actor, is_deterministic=True) if expert
                           else controller.act(actor, packet) if actor_student
                           else controller.act(packet, env._diagnostics()))
                before_step = torch.get_rng_state().clone()
                teacher, _, terminated, truncated, _ = env.step(command)
                transition = env.last_transition
                transition.update(torch_rng_before_step=before_step[None].expand(env.num_envs, -1),
                    torch_rng_before_inference=before_rng[None].expand(env.num_envs, -1),
                    expert_actor_input_diagnostic=actor.clone())
                if not expert:
                    controller.observe_execution(transition)
                numeric_packet = {name: value for name, value in packet.items() if torch.is_tensor(value)
                                  and name not in ('image', 'depth_m', 'depth_valid')}
                done = terminated | truncated
                for lane in range(env.num_envs):
                    if outcomes[lane] is not None:
                        continue
                    append_numeric(groups[lane], lane_values(dict(obs=numeric_packet, transition=transition), lane), tick)
                    after = transition['diagnostics_after']
                    def take(value):
                        return value[lane].cpu().numpy().copy()
                    rows[lane].append(dict(command=take(transition['command_clipped_pre_delay']),
                        delayed=take(transition['control_trace']['delayed_command']),
                        delay_index=take(transition['control_trace']['delay_index']),
                        target=take(transition['target_after_rad']), joint=take(after['joint_position_rad']),
                        velocity=take(after['joint_velocity_rad_s']), object=take(after['object_pose_w']),
                        palm=take(after['palm_body_pose_w']), contact=take(transition['contact_substeps_after_w_N']),
                        object_contact=take(transition['object_contact_force_w']),
                        force=take(transition['applied_wrench']['force_w_N']), torque=take(transition['applied_wrench']['torque_w_Nm']),
                        lifted=take(after['lifted_object']), subgoals=take(after['successes']),
                        alignment=take(transition['native_alignment_condition']), actor=take(actor)))
                    if bool(done[lane]):
                        reasons = {name: bool(value[lane]) for name, value in transition['termination_reasons'].items()}
                        arrays = {name: np.stack([row[name] for row in rows[lane]]) for name in rows[lane][0]}
                        np.savez_compressed(output/f'lane_{lane:02d}_trace.npz', **arrays)
                        outcomes[lane] = dict(identity=f's{seed}_lane{lane}', policy=policy, seed=seed, lane=lane,
                            success=reasons['max_successes'], reasons=reasons, steps=tick+1,
                            native_lift=bool(arrays['lifted'].any()), native_subgoals=int(arrays['subgoals'].max()),
                            native_alignment=bool(arrays['alignment'].any()), software_complete=True,
                            native_termination=bool(any(reasons.values())), fresh_native_reset=True,
                            checkpoint_sha256=sha256(checkpoint), environment_sha256=environment_hash,
                            numeric_path=str((output/f'lane_{lane:02d}.h5').resolve()),
                            initial_state_sha256=sha256(output/'initial_observable.json'))
                if player.is_rnn and done.any():
                    for state in player.states:
                        state[:, done, :] = 0
            if tick % 100 == 0:
                atomic_json(output/'progress.json', dict(tick=tick, completed=sum(row is not None for row in outcomes)))
            if all(row is not None for row in outcomes):
                break
        if any(row is None for row in outcomes):
            raise RuntimeError('Native termination absent at registered safety bound')
        if video:
            video.close(complete=True)
        atomic_json(output/'result.json', dict(software_complete=True, trials=outcomes))
        return outcomes
    finally:
        if video:
            video.close()
        for stream in streams:
            stream.close()


def run(root, output, mode):
    output.mkdir(parents=True, exist_ok=False)
    session = json.loads((root/'session.json').read_text())
    lease = LeaseV2(root, deadline=session['deadline'])
    checkpoint = Path(session['expert_checkpoint'])
    if sha256(checkpoint) != session['expert_sha256']:
        raise ValueError('Expert checkpoint changed')
    atomic_json(output/'runtime_controls.json', disable_omniclient_watches())
    with lease.block(cuda=False):
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device='cpu', kit_args=(
            '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
            '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    env = None
    try:
        import isaacsimenvs
        env = make_env(True, lease, num_envs=4 if mode == 'expert4' else 1)
        with lease.block():
            player, actor_config = create_expert(env, checkpoint)
        atomic_json(output/'actor_config.json', actor_config)
        config, digest = environment_identity(env)
        atomic_json(output/'environment.json', dict(config=config, sha256=digest))
        trials = []
        if mode in ('expert1', 'expert4', 'old_state', 'bc_state', 'bc_actor'):
            saved = None
            student_path = Path(session['old_state_checkpoint'])
            if mode.startswith('bc_'):
                student_path = root/'bc_compare'/mode.removeprefix('bc_')/f"milestone_{session['bc_updates']}.pt"
            if mode == 'old_state' or mode.startswith('bc_'):
                saved = torch.load(student_path, map_location='cpu', weights_only=False)
            seeds = session['single_seeds'] if mode != 'expert4' else session['four_lane_seeds']
            for ordinal, seed in enumerate(seeds):
                trials.extend(fresh_trial(env, lease, output/f'seed_{seed}', seed, player, checkpoint,
                    saved=saved, student_checkpoint=student_path,
                    video_enabled=ordinal == 0 and mode != 'expert4'))
                atomic_json(output/'progress.json', dict(completed=len(trials), trials=trials))
        elif mode == 'fresh_replay':
            for seed in session['single_seeds'][:2]:
                source = root/'expert1'/f'seed_{seed}'
                with h5py.File(source/'lane_00.h5', 'r') as stream:
                    result = run_trial(env, lease, output/f'seed_{seed}', stream['episode'],
                        json.loads(stream.attrs['metadata_json']), mode='delay_wrench_fixed',
                        line='fresh_single_same_config_new_Kit', controller=None, seed=seed,
                        exported_state=torch.load(source/'initial_export.pt', map_location='cpu', weights_only=False))
                    trials.append(dict(result, identity=f's{seed}_lane0'))
        elif mode == 'historical_world':
            data = json.loads((Path(session['source_root'])/'data.json').read_text())
            for ordinal, identity in enumerate(session['historical_world_ids']):
                entry = data['entries'][identity]
                with h5py.File(entry['path'], 'r') as stream:
                    result = run_trial(env, lease, output/f'trial_{ordinal}', stream['episodes'][entry['episode']],
                        json.loads(stream.attrs['metadata_json']), mode='delay_wrench_fixed',
                        line='historical_source_world', controller=None, seed=20261102+ordinal, source_origin=True)
                    trials.append(dict(result, identity=identity))
        atomic_json(output/'result.json', dict(software_complete=True, trials=trials, mode=mode))
    except BaseException as error:
        atomic_json(output/'worker_error.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['expert1', 'expert4', 'old_state', 'fresh_replay', 'historical_world',
                                         'bc_state', 'bc_actor'], required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(arguments.root, arguments.output, arguments.mode)
