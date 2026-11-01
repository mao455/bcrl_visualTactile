"""True Kit oracle and prespecified autonomous cohorts with in-trial videos."""
import argparse
import ctypes
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .compact import sha256
from .evaluate_baselines import RolloutVideo
from .evaluate_training_initials import restore_observable_state
from .native_data import CAMERAS, GROUPS, MANIFEST, multiply_quaternion, rotate
from .native_resources import NativeLease
from .training_resources import atomic_json


EXPERT_SOURCE = Path('outputs/visuotactile/run/t0_control_replay_20261031_01/fresh')


def disable_omniclient_watches():
    import isaacsim
    library = Path(isaacsim.__file__).parent/'kit'/'extscore'/'omni.client.lib'/'bin'/'libomniclient.so'
    if not library.is_file():
        raise RuntimeError('OmniClient library not found: '+str(library))
    client = ctypes.CDLL(str(library))
    setter = client.testSetWatchesEnabled
    setter.argtypes = [ctypes.c_bool]
    setter.restype = None
    setter(False)
    return dict(library=str(library.resolve()), symbol='testSetWatchesEnabled', enabled=False)


def stable_grasp(arrays, initial_height):
    force = np.asarray(arrays['object_contact'])
    contact = np.linalg.norm(force, axis=-1) > .1
    two_fingers = contact.sum(-1) >= 2
    pose = torch.from_numpy(np.asarray(arrays['object']))
    palm = torch.from_numpy(np.asarray(arrays['palm']))
    inverse = palm[:, 3:].clone()
    inverse[:, 1:] *= -1
    relative_position = rotate(inverse, pose[:, :3]-palm[:, :3])
    relative_quaternion = multiply_quaternion(inverse, pose[:, 3:])
    stable = False
    longest = current = 0
    for active in two_fingers.reshape(-1):
        current = current+1 if active else 0
        longest = max(longest, current)
    for stop in range(12, len(pose)+1):
        start = stop-12
        if not two_fingers[start:stop].all() or not (pose[start:stop, 2] >= initial_height+.02).all():
            continue
        position_drift = (relative_position[start:stop]-relative_position[start]).norm(dim=-1).max()
        cosine = (relative_quaternion[start:stop]*relative_quaternion[start]).sum(-1).abs().clamp(0, 1)
        rotation_drift = (2*torch.acos(cosine)).max()
        stable |= bool(position_drift <= .01 and rotation_drift <= .2)
    return dict(object_contact=bool(contact.any()), max_fingers=int(contact.sum(-1).max()),
        two_finger_continuous_s=longest/120, stable_grasp_diagnostic=stable,
        object_contact_threshold_N=.1, stable_window_s=.2, scope='filtered normal forces; not native success')


def make_env(instrumented, lease):
    from evaluation.eval_isaacsim import _load_env_cfg
    from .observations import TaskCommand
    from .sim.replay_environment import RecordedReplayEnv
    from .sim.native_diagnostics import NativeDiagnosticEnv
    cfg = _load_env_cfg('Isaacsimenvs-PreciseAssembly-Direct-v0')
    cfg.seed = 20261031
    cfg.scene.num_envs, cfg.scene.env_spacing = 1, 8.
    cfg.sim.device = 'cpu'
    cfg.precise_assembly.problem = 'tight_insertion'
    cfg.student_obs.image_modality, cfg.student_obs.camera_backend = 'rgbd', 'standard'
    cfg.student_obs.camera_update_period_s = 0.
    cfg.student_obs.image_width = cfg.student_obs.image_input_width = 448
    cfg.student_obs.image_height = cfg.student_obs.image_input_height = 252
    cfg.sim.render.rendering_mode, cfg.sim.render.antialiasing_mode = 'quality', 'Off'
    with lease.block():
        return (NativeDiagnosticEnv if instrumented else RecordedReplayEnv)(cfg, perception_version=2,
            task_command=TaskCommand('tight_insertion'), wrench_frame_mode='world_hold', camera_rig='wrist_bracket_close')


def oracle(root, output, env, lease, instrumented):
    from .evaluate_recorded_replay import run_trial
    video = None
    if instrumented:
        video = RolloutVideo(output/'01_expert_success.mp4', 'expert reference', 17, 0, CAMERAS)
        video.title = '220-step expert reference | seed17 | different start from students | 1x physical time'
        original = env.step
        def recorded_step(command):
            video.append(env.research_packet, int(env.research_packet['step_id'][0]))
            return original(command)
        env.step = recorded_step
    try:
        with h5py.File(EXPERT_SOURCE/'expert_numeric.h5', 'r') as stream:
            result = run_trial(env, lease, output/'replay', stream['episode'],
                json.loads(stream.attrs['metadata_json']), mode='delay_wrench_fixed',
                line='native_command_oracle', controller=None, seed=17,
                exported_state=torch.load(EXPERT_SOURCE/'initial_export.pt', map_location='cpu', weights_only=False),
                wrench_reconstruction='native_request')
        if not result['success'] or result['steps'] != 220:
            raise RuntimeError('220-step native-request oracle did not reproduce native success')
        if any(value['maximum'] != 0 for value in result['comparison']['errors'].values()):
            raise RuntimeError('Native-request oracle fields are not exact')
        if instrumented:
            baseline = Path(json.loads((root/'oracle_baseline.json').read_text())['trace'])
            with np.load(baseline) as before, np.load(output/'replay/trace.npz') as after:
                fields = [name for name in before.files if name not in ('inference_s',)]
                differences = [name for name in fields if not np.array_equal(before[name], after[name])]
            if differences:
                raise RuntimeError('Diagnostic/video behavior-neutrality failed: '+','.join(differences))
            video.close(complete=True)
        else:
            atomic_json(root/'oracle_baseline.json', dict(trace=str((output/'replay/trace.npz').resolve())))
        atomic_json(output/'result.json', dict(passed=True, native_success=True, steps=220,
            instrumentation=instrumented, compared_fields=result['comparison']['errors']))
    finally:
        if video:
            video.close()


def autonomous(root, output, env, lease, group, checkpoint_path, cohort):
    from .models.long_horizon import FrozenDinoSpatial
    from .native_controller import NativeController
    data = json.loads((root/'data.json').read_text())
    identities = (sorted(data['splits']['test'])[:20] if cohort == 'formal' else
                  data['splits']['train'][:4] if cohort == 'train' else data['splits']['train'][:1])
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if saved['group'] != group:
        raise ValueError('Evaluation checkpoint group differs')
    with lease.block():
        dino = FrozenDinoSpatial().to('cuda:0') if group == 'I-HistoryDP' else None
    results = []
    for ordinal, identity in enumerate(identities):
        entry = data['entries'][identity]
        trial = output/identity
        trial.mkdir(exist_ok=False)
        atomic_json(trial/'started.json', dict(identity=identity, cohort=cohort, checkpoint_sha256=sha256(checkpoint_path), utc=time.time()))
        video = None
        with h5py.File(entry['path'], 'r') as stream:
            source = stream['episodes'][entry['episode']]
            metadata = json.loads(stream.attrs['metadata_json'])
            with lease.block():
                env.clear_recorded_events()
                env.reset(seed=20261101+ordinal)
                restoration = restore_observable_state(env, source, metadata)
                if max(restoration['saved_observable_errors'].values()) > 1e-4:
                    raise ValueError('Prespecified observable restoration failed; no replacement')
                controller = NativeController(saved, metadata, dino)
                controller.generator.manual_seed(7093+ordinal)
            initial_height = float(source['transition/diagnostics_before/object_pose_w'][0, 2])
        atomic_json(trial/'restoration.json', restoration)
        if cohort == 'formal' and ordinal == 0:
            video = RolloutVideo(trial/'student.mp4', group, 20261101, saved['epoch'], CAMERAS)
            video.title = group+' | first registered test start | autonomous formal trial | 1x physical time'
        rows, planning, boundary, plans, previous = [], [], [], [], None
        keyframe_events = set()
        bound = (int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
        try:
            for tick in range(bound):
                with lease.block():
                    packet = env.research_packet
                    if video:
                        video.append(packet, tick)
                    before = env._diagnostics() if group == 'S-NativeDP' else None
                    began = time.perf_counter()
                    command = controller.act(packet, before)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-began
                    if tick % 12 == 0:
                        planning.append(elapsed)
                        plans.append(controller.plan[0].numpy().copy())
                        if previous is not None:
                            boundary.append(float((command-previous).square().mean().sqrt()))
                    _, _, terminated, truncated, _ = env.step(command)
                    transition = env.last_transition
                    controller.observe_execution(transition)
                    previous = command.clone()
                    after = transition['diagnostics_after']
                    def take(value):
                        return value[0].detach().cpu().numpy().copy()
                    rows.append(dict(command=take(command), target=take(transition['target_after_rad']),
                        target_before=take(transition['target_before_rad']), joint=take(after['joint_position_rad']),
                        delayed=take(transition['control_trace']['delayed_command']), object=take(after['object_pose_w']),
                        palm=take(after['palm_body_pose_w']), object_contact=take(transition['object_contact_force_w']),
                        native_keypoints=take(transition['native_keypoints_max_dist']), lifted=take(after['lifted_object']),
                        native_alignment=take(transition['native_alignment_condition']),
                        subgoals=take(after['successes']), inference_s=elapsed,
                        fingertip_distance=float((after['finger_position_w'][0]-after['object_pose_w'][0, :3]).norm(dim=-1).min())))
                    events = {'initial'} if tick == 0 else set()
                    if bool(transition['object_contact_force_w'].norm(dim=-1).gt(.1).any()):
                        events.add('object_contact')
                    if bool(after['lifted_object'][0]):
                        events.add('native_lift')
                    if bool(after['successes'][0]):
                        events.add('native_subgoal')
                    if bool(terminated[0] or truncated[0]):
                        events.add('termination')
                    for event in sorted(events-keyframe_events):
                        from PIL import Image
                        rgb = np.rint(packet['image'][0, :, :3].cpu().numpy().clip(0, 1)*255).astype('uint8')
                        Image.fromarray(np.concatenate(rgb.transpose(0, 2, 3, 1), axis=1)).save(trial/(event+'_pre_action.png'))
                        atomic_json(trial/(event+'_frame.json'), dict(tick=tick, observation='before triggering action',
                            terminal_post_action_frame_available=False))
                    keyframe_events |= events
                if bool(terminated[0] or truncated[0]):
                    break
            reasons = {name: bool(value[0]) for name, value in transition['termination_reasons'].items()}
            if not any(reasons.values()):
                raise RuntimeError('Native termination absent at diagnostic safety bound')
            arrays = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
            np.savez_compressed(trial/'trace.npz', **arrays)
            np.save(trial/'plans_h64.npy', np.stack(plans))
            contact = stable_grasp(arrays, initial_height)
            result = dict(identity=identity, group=group, cohort=cohort, steps=len(rows), success=reasons['max_successes'],
                reasons=reasons, native_lift=bool(arrays['lifted'].any()), native_subgoals=int(arrays['subgoals'].max()),
                approach_proxy=bool((arrays['fingertip_distance'] <= .08).any()),
                native_keypoint_min_m=float(arrays['native_keypoints'].min()),
                native_alignment=bool(arrays['native_alignment'].any()),
                planning_p50_p95_max_s=[float(np.percentile(planning, 50)), float(np.percentile(planning, 95)), max(planning)],
                planning_over_200ms=int((np.asarray(planning) > .2).sum()),
                chunk_boundary_rmse_mean=float(np.mean(boundary)) if boundary else None,
                contact=contact, exact_historical_physics_restoration=False, checkpoint_sha256=sha256(checkpoint_path))
            if video:
                video.close(complete=True)
                result['video'] = str((trial/'student.mp4').resolve())
            atomic_json(trial/'result.json', result)
            results.append(result)
            atomic_json(output/'progress.json', dict(completed=len(results), required=len(identities)))
        finally:
            if video:
                video.close()
    atomic_json(output/'result.json', dict(passed=True, cohort=cohort, group=group, trials=results,
        completed=len(results), required=len(identities), successes=sum(row['success'] for row in results)))


def preflight(root, output, env, lease):
    with lease.block():
        reset = env.reset(seed=20261101)
        packet = env.research_packet
        command = torch.zeros((1, 29), dtype=torch.float32, device=env.device)
        _, _, terminated, truncated, _ = env.step(command)
        transition = env.last_transition
    image = packet['image']
    if image.shape[-2:] != (252, 448) or not torch.isfinite(image).all():
        raise ValueError('Kit preflight returned invalid image tensor')
    if (terminated | truncated).shape != (1,):
        raise ValueError('Kit preflight returned invalid termination shape')
    atomic_json(output/'result.json', dict(passed=True, mode='kit-preflight',
        reset_type=type(reset).__name__, image_shape=list(image.shape),
        step_id=int(transition['step_id'][0]), finite_image=True,
        zero_command_step=True, termination=bool((terminated | truncated)[0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('oracle-base', 'oracle-instrumented', 'autonomous', 'preflight'), required=True)
    parser.add_argument('--group', choices=GROUPS)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--cohort', choices=('fit', 'train', 'formal'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    session = json.loads((args.root/'session.json').read_text())
    gate = args.mode.startswith('oracle') or args.mode == 'preflight' or args.cohort == 'fit'
    recovery = session.get('late_gate_recovery', {})
    lease_deadline = session['deadline'] if gate and recovery.get('authorized') else session['gate_deadline'] if gate else session['deadline']
    lease = NativeLease(args.root, deadline=lease_deadline)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime_controls = disable_omniclient_watches()
    atomic_json(args.output/'runtime_controls.json', runtime_controls)
    with lease.block(cuda=False):
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device='cpu', kit_args=(
            '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
            '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    env = None
    try:
        import isaacsimenvs
        env = make_env(args.mode != 'oracle-base', lease)
        if args.mode.startswith('oracle'):
            oracle(args.root, args.output, env, lease, args.mode == 'oracle-instrumented')
        elif args.mode == 'preflight':
            preflight(args.root, args.output, env, lease)
        else:
            autonomous(args.root, args.output, env, lease, args.group, args.checkpoint, args.cohort)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
