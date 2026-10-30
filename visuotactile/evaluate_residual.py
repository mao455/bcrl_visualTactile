"""Twelve prespecified diagnostic trials, autonomous and real expert roll-in."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from .residual_session import SessionGate
from .training_resources import atomic_json


def main():
    process_started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    output = args.root / 'closed_loop'
    output.mkdir(exist_ok=False)
    atomic_json(output / 'status.json', dict(state='initializing', started_epoch=process_started, pid=os.getpid()))
    gate = SessionGate(args.root)
    plan = json.loads((args.root / 'session.json').read_text())
    gate.before(reserve_s=600)
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device='cpu', kit_args=(
        '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
        '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    env, rows = None, []
    try:
        import numpy as np
        import torch
        import isaacsimenvs
        from evaluation.eval_isaacsim import _load_env_cfg
        from .compact import sha256
        from .contracts import assert_policy_compatible, assert_execution_compatible
        from .expert_reference import ExpertReference
        from .initial_geometry import inspect
        from .observations import TaskCommand
        from .residual_controller import ResidualController
        from .sim.environment import InstrumentedAssemblyEnv
        torch.set_num_threads(2)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        gate.before()
        config = _load_env_cfg('Isaacsimenvs-PreciseAssembly-Direct-v0')
        config.seed = plan['diagnostic_seeds'][0]
        config.scene.num_envs = 1
        config.scene.env_spacing = 8.
        config.sim.device = 'cpu'
        config.precise_assembly.problem = 'tight_insertion'
        config.student_obs.image_modality = 'rgbd'
        config.student_obs.camera_backend = 'standard'
        config.student_obs.camera_update_period_s = 0.
        config.student_obs.image_width = config.student_obs.image_input_width = 448
        config.student_obs.image_height = config.student_obs.image_input_height = 252
        config.sim.render.rendering_mode = 'quality'
        config.sim.render.antialiasing_mode = 'Off'
        env = InstrumentedAssemblyEnv(config, perception_version=2, task_command=TaskCommand('tight_insertion'),
                                      wrench_frame_mode='world_hold', camera_rig='wrist_bracket_close')
        env.configure_camera_views((-.7, -.35, 1.15), (0., -.03, .68))
        checkpoint = args.root / 'training_full/b_best.pt'
        policies = {name: ResidualController(checkpoint if name!='c' else args.root/'training_full/c_best.pt',
                                            bypass=name=='a') for name in ('a', 'b', 'c')}
        expert_path = Path('pretrained_assembly/tight_insertion/model.pth')
        expert = ExpertReference(env, expert_path, policies['a'].base.metadata)
        if sha256(expert_path) != policies['a'].base.metadata['expert_model_sha256']:
            raise ValueError('Released expert identity mismatch')
        metadata = env.research_metadata()
        metadata.update(problem='tight_insertion', camera_mode='oblique', config=config.to_dict())
        for policy in policies.values():
            assert_policy_compatible(policy.base.metadata, metadata)
            assert_execution_compatible(policy.base.metadata, metadata)
        atomic_json(output / 'environment.json', metadata)
        atomic_json(output / 'launch.json', dict(started_epoch=process_started, pid=os.getpid(), plan=plan,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            checkpoints={name:policy.checkpoint_sha256
            for name,policy in policies.items()}, source_sha256={str(path):sha256(path) for path in (
                Path(__file__), Path('visuotactile/residual_controller.py'), Path('visuotactile/models/residual.py'),
                Path('visuotactile/rollout.py'), Path('visuotactile/sim/environment.py'))}))
        references, switches = {}, {}
        for ordinal, trial in enumerate(plan['diagnostics']):
            gate.before(reserve_s=240)
            seed, variant, mode = trial['seed'], trial['variant'], trial['mode']
            attempt = f'{ordinal+1:02d}_{mode}_{variant}_s{seed}'
            atomic_json(output / 'status.json', dict(state='running', attempt=attempt, completed=len(rows), pid=os.getpid()))
            teacher_observation, _ = env.reset(seed=seed)
            initial = {name:value[0].detach().cpu().numpy().copy() for name,value in env._diagnostics().items()}
            initial.update(actuator_target_rad=env.research_packet['actuator_target_rad'][0].numpy().copy(),
                           force_probability=env._random_force_prob[0].numpy().copy(),
                           torque_probability=env._random_torque_prob[0].numpy().copy(),
                           object_mass=env._object_mass[0].numpy().copy())
            geometry = inspect(metadata, initial)
            if not geometry['valid_box_initialization']:
                rows.append(dict(**trial, attempt=attempt, outcome='invalid_start', geometry=geometry))
                atomic_json(output / 'trials.json', rows)
                continue
            pair = (mode, seed)
            if pair not in references:
                references[pair] = initial
            initial_error = max(float(np.max(np.abs(initial[name].astype(float)-reference.astype(float))))
                                for name, reference in references[pair].items())
            if initial_error > 1e-5:
                raise RuntimeError('Paired initial states differ')
            policy = policies[variant]
            policy.base.generator.manual_seed(seed+1000000)
            switched = mode=='autonomous'
            switch_step, switch_error = (0 if switched else None), None
            contact_streak, prefix_steps = 0, 0
            prefix = hashlib.sha256()
            trace = {name:[] for name in ('base', 'command', 'delta', 'gate', 'cursor', 'generation_step',
                'previous_command', 'command_valid', 'memory_updates', 'submitted', 'clipped', 'delayed',
                'target', 'tracking_error', 'object', 'goal', 'successes', 'lifted', 'force', 'student',
                'finger_object_distance', 'palm_object_distance')}
            events, done, rng_unchanged = [], False, True
            started = time.monotonic()
            natural_bound = (int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
            for step in range(natural_bound):
                gate.before()
                tick = time.monotonic()
                packet = env.research_packet
                rng = torch.get_rng_state().clone()
                command = policy.act(packet)
                rng_unchanged &= torch.equal(rng, torch.get_rng_state())
                if not rng_unchanged:
                    raise RuntimeError('Student altered environment RNG')
                contact_streak = contact_streak+1 if bool(policy.last['contact'][0]) else 0
                if not switched and contact_streak >= 3 and step <= plan['rollin_limit_steps']:
                    switched, switch_step = True, step
                    state = {name:value[0].detach().cpu().numpy().copy() for name,value in env._diagnostics().items()}
                    state.update(target=packet['actuator_target_rad'][0].numpy().copy(),
                                 contact=packet['contact_w_N'][0].numpy().copy())
                    if pair not in switches:
                        switches[pair] = (step, state, prefix.hexdigest())
                    ref_step, reference, prefix_digest = switches[pair]
                    switch_error = max(float(np.max(np.abs(state[name].astype(float)-value.astype(float))))
                                       for name,value in reference.items())
                    if step != ref_step or switch_error>1e-5 or prefix.hexdigest()!=prefix_digest:
                        atomic_json(output / (attempt+'_switch_error.json'), dict(step=step, reference_step=ref_step,
                            max_state_error=switch_error, prefix_equal=prefix.hexdigest()==prefix_digest))
                        raise RuntimeError('Expert roll-in prefix/switch state did not match')
                    np.savez_compressed(output / (attempt+'_switch.npz'), **state)
                if mode=='expert_rollin' and not switched:
                    expert.teacher_observation = teacher_observation
                    command = expert.act(packet)
                    prefix.update(command.numpy().tobytes())
                    prefix_steps += 1
                source = 'student' if switched else 'released_expert_rollin'
                teacher_observation, _, terminated, truncated, _ = env.step(command)
                transition = env.last_transition
                policy.acknowledge(transition, command, source=source)
                diagnostic = transition['diagnostics_after']
                values = dict(base=policy.last['base'], command=command, delta=policy.last['delta'], gate=policy.last['contact'],
                    cursor=policy.last['cursor'], generation_step=policy.last['generation_step'],
                    previous_command=policy.last['previous_command'], command_valid=policy.last['command_valid'],
                    memory_updates=policy.last['memory_updates'], submitted=transition['submitted_action'],
                    clipped=transition['command_clipped_pre_delay'], delayed=transition['control_trace']['delayed_command'],
                    target=transition['target_after_rad'], tracking_error=transition['target_after_rad']-diagnostic['joint_position_rad'],
                    object=diagnostic['object_pose_w'], goal=diagnostic['goal_pose_w'], successes=diagnostic['successes'],
                    lifted=diagnostic['lifted_object'], force=packet['contact_w_N'], student=torch.tensor([switched]),
                    finger_object_distance=(diagnostic['finger_position_w']-
                        diagnostic['object_pose_w'][:, None, :3]).norm(dim=-1).mean(-1),
                    palm_object_distance=(diagnostic['palm_body_pose_w'][:, :3]-
                        diagnostic['object_pose_w'][:, :3]).norm(dim=-1))
                for name, value in values.items():
                    trace[name].append(value[0].detach().cpu().numpy().copy())
                if step in (0, 1, 5, 6, 7) or step==switch_step or (bool(policy.last['contact'][0]) and not any(event['contact'] for event in events)):
                    events.append(dict(step=step, cursor=int(policy.last['cursor'][0]),
                        generated_step=int(policy.last['generation_step'][0]), memory_updates=int(policy.last['memory_updates'][0]),
                        contact=bool(policy.last['contact'][0]), source=source,
                        command=command[0].tolist(), base=policy.last['base'][0].tolist(),
                        previous_command=policy.last['previous_command'][0].tolist(),
                        native_ack=bool(torch.equal(command.clamp(-1, 1), transition['command_clipped_pre_delay'])),
                        obs_time_s=float(packet['time_s'][0]), after_time_s=float(transition['time_after_s'][0])))
                done = bool((terminated|truncated)[0])
                gate.pace(tick)
                if done:
                    break
            arrays = {name:np.stack(value) for name,value in trace.items()}
            np.savez_compressed(output / (attempt+'.npz'), **arrays)
            student = arrays['student'].astype(bool)
            command = arrays['command'][student]
            jumps = np.diff(arrays['command'], axis=0)
            boundaries = (arrays['cursor'][1:]==0) & student[1:]
            no_contact = student & ~arrays['gate'].astype(bool)
            success = bool(done and diagnostic['successes'][0]>=diagnostic['max_goals'][0] and diagnostic['retract_succeeded'][0])
            row = dict(**trial, attempt=attempt, steps=step+1, student_steps=int(student.sum()),
                success=success if switched else None, outcome=('success' if success else 'failure') if done and switched else
                    'no_contact_switch' if not switched else 'censored',
                native_reasons=[name for name,value in transition['termination_reasons'].items() if bool(value[0])],
                prefix_steps=prefix_steps, prefix_sha256=prefix.hexdigest(), switch_step=switch_step, switch_max_state_error=switch_error,
                initial_max_state_error=initial_error, max_subgoals=int(arrays['successes'].max()),
                native_lifted=bool(arrays['lifted'].any()),
                max_object_z_rise_from_reset_m=float(arrays['object'][:, 2].max()-initial['object_pose_w'][2]),
                min_goal_position_distance_m=float(np.linalg.norm(arrays['object'][:, :3]-arrays['goal'][:, :3], axis=-1).min()),
                min_student_finger_object_distance_m=float(arrays['finger_object_distance'][student].min()) if student.any() else None,
                first_student_finger_object_distance_m=float(arrays['finger_object_distance'][student][0]) if student.any() else None,
                final_finger_object_distance_m=float(arrays['finger_object_distance'][-1]),
                min_student_palm_object_distance_m=float(arrays['palm_object_distance'][student].min()) if student.any() else None,
                student_gate_fraction=float(arrays['gate'][student].mean()) if student.any() else None,
                student_saturation_fraction=float((np.abs(command)>=.99999).mean()) if len(command) else None,
                bypass_exact=bool(np.array_equal(arrays['command'][no_contact], arrays['base'][no_contact])),
                tracking_rmse_rad={name:float(np.sqrt(np.mean(arrays['tracking_error'][student, selected]**2))) if student.any() else None
                                   for name,selected in (('arm', slice(0, 7)), ('hand', slice(7, 29)))},
                boundary_jump_rmse={name:float(np.sqrt(np.mean(jumps[boundaries, selected]**2))) if boundaries.any() else None
                                    for name,selected in (('arm', slice(0, 7)), ('hand', slice(7, 29)))},
                within_plan_jump_rmse={name:float(np.sqrt(np.mean(jumps[~boundaries & student[1:], selected]**2)))
                    if (~boundaries & student[1:]).any() else None for name,selected in (('arm', slice(0, 7)), ('hand', slice(7, 29)))},
                effective_correction_rmse={name:float(np.sqrt(np.mean((arrays['command'][student, selected]-arrays['base'][student, selected])**2)))
                    if student.any() else None for name,selected in (('arm', slice(0, 7)), ('hand', slice(7, 29)))},
                contact_partner='unknown; five aggregate normal forces do not identify the object',
                rng_unchanged=rng_unchanged, native_ack_exact=bool(np.array_equal(arrays['command'].clip(-1, 1), arrays['clipped'])),
                actual_feedback_exact=bool(np.array_equal(arrays['previous_command'][1:], arrays['clipped'][:-1])),
                elapsed_s=time.monotonic()-started, trace_sha256=sha256(output / (attempt+'.npz')),
                checkpoint_sha256=policy.checkpoint_sha256, causal_events=events)
            rows.append(row)
            atomic_json(output / 'trials.json', rows)
            print(json.dumps({name:value for name,value in row.items() if name!='causal_events'}), flush=True)
        atomic_json(output / 'report.json', dict(trials=rows, complete=len(rows)==12, wait_s=gate.wait_s,
                    started_epoch=process_started, finished_epoch=time.time(),
                    purpose='separate diagnostic cohort; never appended to formal 0/50', plan=plan))
        atomic_json(output / 'status.json', dict(state='complete', completed=len(rows)))
    except BaseException as error:
        (output / 'error.txt').write_text(traceback.format_exc())
        atomic_json(output / 'status.json', dict(state='failed', error=repr(error), completed=len(rows)))
        raise
    finally:
        gate.save_waits()
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
