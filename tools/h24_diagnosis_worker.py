"""Opt-in recording and causal expert-prefix diagnostics; frozen policy is reused."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from tools.h24_diagnosis_common import prefix_comparison, prefix_feedback, sha
from tools.state_motion_recovery import append_mode_numeric
from visuotactile import state_benchmark as base
from visuotactile.state_history import HistoryController
from visuotactile.training_resources import atomic_json

JOB = {}

def append_policy_inputs(group, values, step):
    """Flat policy fields live directly in the group, without an empty parent."""
    for name, value in values.items():
        array = value.detach().cpu().numpy()[0]
        if step == 0:
            group.create_dataset(name, shape=(0,*array.shape), maxshape=(None,*array.shape),
                dtype=array.dtype, chunks=(1,*array.shape), compression='lzf')
        dataset = group[name]
        if len(dataset) != step:
            raise ValueError('Policy history recorder skipped/repeated a tick')
        dataset.resize(step+1, axis=0)
        dataset[step] = array

class CleanVideo:
    """Three unchanged sensor views, 30 fps; CPU upscale, no renderer change."""
    def __init__(self, path):
        import av
        self.path = path
        self.partial = path.with_suffix('.mp4.partial')
        self.container = av.open(str(self.partial), 'w', format='mp4')
        self.stream = self.container.add_stream('libx264', rate=30)
        self.stream.width, self.stream.height = 1280, 720
        self.stream.pix_fmt = 'yuv420p'
        self.stream.codec_context.thread_count = 2
        self.stream.options = {'crf': '18', 'preset': 'fast'}
        self.frames = 0

    def append(self, packet, tick):
        if tick % 2:
            return
        import av
        from PIL import Image
        rgb = np.rint(packet['image'][0, :, :3].cpu().numpy().clip(0, 1)*255).astype('uint8').transpose(0, 2, 3, 1)
        assert rgb.shape == (3, 252, 448, 3)
        canvas = Image.new('RGB', (896, 504))
        for image, xy in zip(rgb, ((0, 0), (448, 0), (0, 252))):
            canvas.paste(Image.fromarray(image), xy)
        canvas = canvas.resize((1280, 720), Image.Resampling.LANCZOS)
        frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')
        frame.pts = self.frames
        for encoded in self.stream.encode(frame):
            self.container.mux(encoded)
        self.frames += 1

    def close(self, complete=False):
        if self.container is None:
            return
        for encoded in self.stream.encode():
            self.container.mux(encoded)
        self.container.close()
        self.container = None
        if complete:
            self.partial.replace(self.path)


class CapturedController(HistoryController):
    @torch.no_grad()
    def act(self, actor, packet, diagnostics):
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        original = DDIMScheduler.step
        self.normalized_plan = None
        self.rng_before = self.generator.get_state().clone()
        state = self.bank.lanes.get(0)
        self.hidden_before = None if state is None else state['hidden'].detach().cpu().clone()
        def captured(scheduler, *args, **kwargs):
            result = original(scheduler, *args, **kwargs)
            self.normalized_plan = result.prev_sample.detach().cpu().clone()
            return result
        DDIMScheduler.step = captured
        try:
            result = super().act(actor, packet, diagnostics)
        finally:
            DDIMScheduler.step = original
        self.rng_after = self.generator.get_state().clone()
        return result


@torch.no_grad()
def trial(env, lease, output, spec, source, player, checkpoint, *, saved=None, replay=None,
          video=False, position=False, inference=None, takeover_steps=None):
    assert replay is None and takeover_steps is None
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'started.json', dict(spec=spec, utc=time.time(), checkpoint=checkpoint, job=JOB))
    isolated = base.IsolatedEnvironment(env, spec['seed'])
    with lease.block():
        env.action_control_mode = 'native_command_clipped_pre_delay'
        teacher, legal = base.prepare(env, isolated, spec, source, output)
        if not legal['legal']:
            result = dict(**spec, software_complete=True, legal=False, success=False,
                not_started=True, reason='pre-action collision rejection')
            atomic_json(output/'result.json', result)
            return result
        env.action_control_mode = ('absolute_joint_position_target_rad'
            if saved is not None and not JOB.get('expert_prefix_ticks', 0)
            else 'native_command_clipped_pre_delay')
        metadata = env.research_metadata()
        metadata['config'] = env.cfg.to_dict()
        _, config_hash = base.environment_identity(env)
        atomic_json(output/'metadata.json', json.loads(json.dumps(metadata, default=str)))
        player.reset()
        torch.save(base.export_initial(env, player), output/'initial_export.pt')
        torch.save(isolated.stream.state, output/'environment_rng_initial.pt')
        controller = None if saved is None else CapturedController(saved, metadata, inference=inference)
    prefix = int(JOB.get('expert_prefix_ticks', 0))
    assert prefix >= 0 and prefix % 12 == 0
    reference = np.load(Path(JOB['prefix_reference'])/'trace.npz') if prefix else None
    if prefix and len(reference['joint']) <= prefix:
        raise ValueError('Expert reference does not reach takeover tick')
    rows, plan_rows, normalized_rows = [], [], []
    movie = CleanVideo(output/'clean.mp4') if video else None
    numeric = h5py.File(output/'numeric.h5', 'w')
    numeric.attrs['metadata_json'] = json.dumps(metadata, default=str)
    group = numeric.create_group('episode')
    inputs = h5py.File(output/'policy_inputs.h5', 'w') if controller else None
    input_group = inputs.create_group('episode') if inputs else None
    prefix_check = None
    actor_parity = 0.
    try:
        bound = (int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
        for tick in range(bound):
            packet = env.research_packet
            if movie:
                movie.append(packet, tick)
            with lease.block():
                actor = torch.cat((teacher['policy'], player.intr_reward_coef_embd), -1)
                expert_command = player.get_action(actor, is_deterministic=True).clamp(-1, 1)
                student_action = controller.act(actor, packet, env._diagnostics()) if controller else None
                if controller:
                    expected = player.model.norm_obs(actor)[:, :140]
                    actual = controller.model.actor_input(actor.to(controller.device))[:, :140].cpu()
                    actor_parity = max(actor_parity, float((expected-actual).abs().max()))
                    hidden = controller.bank.lanes[0]['hidden'].detach().cpu().clone()
                    values = {k:v[None] for k,v in controller.last_inputs.items()}
                    values.update(hidden_before=(torch.zeros_like(hidden) if controller.hidden_before is None else controller.hidden_before),
                        hidden_after=hidden, condition=controller.last_condition.cpu(),
                        sampler_rng_before=controller.rng_before[None], sampler_rng_after=controller.rng_after[None],
                        planning=torch.tensor([[tick % 12 == 0]]),
                        candidate_requested_position=student_action.cpu(),
                        raw_position=controller.raw_plan[:, tick % 12])
                    append_policy_inputs(input_group, values, tick)
                    if tick % 12 == 0:
                        assert controller.normalized_plan is not None
                        plan_rows.append(controller.raw_plan[0].numpy().copy())
                        normalized_rows.append(controller.normalized_plan[0].numpy().copy())
                if prefix and tick == prefix:
                    prefix_check = prefix_comparison(rows, reference, prefix)
                    atomic_json(output/'prefix_validation.json', prefix_check)
                    if not prefix_check['passed']:
                        raise ValueError('Expert prefix parity failed before any student action')
                student_control = controller is not None and tick >= prefix
                env.action_control_mode = ('absolute_joint_position_target_rad' if student_control
                    else 'native_command_clipped_pre_delay')
                command = student_action if student_control else expert_command
                if not torch.isfinite(command).all():
                    raise FloatingPointError('Nonfinite command')
                before_rng = isolated.stream.state[2].clone()
                teacher, _, terminated, truncated, _ = isolated.step(command)
                transition = env.last_transition
                transition.update(expert_actor_input_diagnostic=actor.clone(),
                    torch_rng_before_step=before_rng[None], expert_shadow_command=expert_command.clone())
                if controller:
                    if student_control:
                        controller.observe_execution(transition)
                    else:
                        prefix_feedback(controller, transition)
                before_packet = {k:v for k,v in packet.items() if torch.is_tensor(v)
                    and k not in ('image', 'depth_m', 'depth_valid')}
                append_mode_numeric(group, dict(obs=before_packet, transition=transition), tick)
                after = transition['diagnostics_after']
                take = lambda value: value[0].cpu().numpy().copy()
                rows.append(dict(actor=take(actor), command=take(command), expert_shadow=take(expert_command),
                    command_is_position=np.array(student_control),
                    delayed=take(transition['control_trace'].get('delayed_command', transition.get('position_applied_rad'))),
                    delay_index=take(transition['control_trace']['delay_index']),
                    target=take(transition['target_after_rad']), joint=take(after['joint_position_rad']),
                    actuator_targets=take(transition['actuator_substep_target_rad']),
                    velocity=take(after['joint_velocity_rad_s']), object=take(after['object_pose_w']),
                    contact=take(transition['contact_substeps_after_w_N']), object_contact=take(transition['object_contact_force_w']),
                    force=take(transition['applied_wrench']['force_w_N']), torque=take(transition['applied_wrench']['torque_w_Nm']),
                    lifted=take(after['lifted_object']), subgoals=take(after['successes']),
                    retract=take(after['retract_succeeded']), alignment=take(transition['native_alignment_condition']),
                    palm=take(after['palm_body_pose_w']), fingers=take(after['finger_position_w']),
                    object_velocity=take(after['object_velocity_w_m_s_rad_s'])))
            if tick % 60 == 0:
                numeric.flush()
                if inputs: inputs.flush()
                atomic_json(output/'progress.json', dict(tick=tick, utc=time.time(), student_control=student_control))
            if bool((terminated | truncated)[0]):
                break
        arrays = {k:np.stack([r[k] for r in rows]) for k in rows[0]}
        np.savez_compressed(output/'trace.npz', **arrays)
        if plan_rows:
            np.savez_compressed(output/'plans.npz', plans=np.stack(plan_rows), raw_plans=np.stack(plan_rows),
                normalized_plans=np.stack(normalized_rows), anchors=np.arange(len(plan_rows))*12)
        reasons = {k:bool(v[0]) for k,v in transition['termination_reasons'].items()}
        from visuotactile.native_evaluation import stable_grasp
        initial = base.json.loads((output/'initial_observable.json').read_text())
        mask = arrays['command_is_position']
        result = dict(**spec, software_complete=True, legal=True, success=reasons['max_successes'],
            reasons=reasons, steps=len(rows), native_lift=bool(arrays['lifted'].any()),
            native_subgoals=int(arrays['subgoals'].max()), native_alignment=bool(arrays['alignment'].any()),
            checkpoint_sha256=sha(checkpoint), environment_sha256=config_hash,
            actor_normalization_max_error=actor_parity, simulation_step_hz=60, physics_hz=120,
            plan_hz=5 if controller else 60, inference=inference, expert_prefix_ticks=prefix,
            student_ticks=int(mask.sum()), prefix_validation=prefix_check,
            autonomous=prefix == 0 and controller is not None,
            action_control_mode='position_rad_after_native_expert_prefix' if prefix else env.action_control_mode,
            actuator_target_max_error=float(np.abs(arrays['actuator_targets']-arrays['target'][:,None,:]).max()),
            submitted_target_max_error=float(np.abs(arrays['command'][mask]-arrays['target'][mask]).max()) if mask.any() else None,
            contact=stable_grasp(arrays, float(initial['object_pose_w'][0][2])),
            video=dict(fps=30, frames=movie.frames, source='unaltered 448x252 sensors, CPU mosaic upscale; pre-action even ticks; terminal reset frame excluded') if movie else None,
            limitation='Observable reset only; no PhysX internal-state identity claim. Fingertip frames and normal forces do not identify contact points or tangential slip.')
        if movie: movie.close(complete=True)
        atomic_json(output/'result.json', result)
        return result
    finally:
        numeric.close()
        if inputs: inputs.close()
        if reference: reference.close()
        if movie: movie.close()


def main():
    global JOB
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['expert', 'dp'], required=True)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    JOB = json.loads(args.job.read_text())
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base.trial = trial
    base.run(args.root, args.output, args.mode, args.job)

if __name__ == '__main__':
    main()
