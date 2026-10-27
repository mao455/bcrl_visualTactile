"""Bounded CPU collection with original teacher/control; see research_docs/RUNBOOK."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--max-steps", type=int,
                        help="External control-step budget; does not change the native timeout or success rules")
    parser.add_argument("--episodes", type=int,
                        help="Stop after this many naturally completed episodes; omitted keeps the legacy fixed-step mode")
    parser.add_argument("--condition", choices=("original", "force_1p5"), default="original",
                        help="Paired experiment label. original keeps the configured wrench scale; force_1p5 multiplies only force_scale")
    parser.add_argument("--pair-id", help="Stable paired-seed identifier stored in metadata/results")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--problem", default="tight_insertion")
    parser.add_argument("--checkpoint", default="pretrained_assembly/tight_insertion/model.pth")
    parser.add_argument("--teacher-group", type=int, choices=range(6), default=0,
                        help="Released SAPG parameter group, 0..5 maps to identifiers 50..0")
    parser.add_argument("--camera", choices=["original", "oblique"], default="oblique")
    parser.add_argument("--audit-default-camera", action="store_true")
    parser.add_argument("--perception-version", type=int, choices=(1,2), default=2)
    parser.add_argument("--local-eye", type=float, nargs=3, default=(-.7,-.35,1.15))
    parser.add_argument("--local-target", type=float, nargs=3, default=(0.,-.03,.68))
    parser.add_argument("--goal-command", type=Path, help="Operator-provided base-frame pose JSON; no simulator truth fallback")
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--correction-shadow", action="store_true",
                        help="Log untrained correction I/O on real histories; NEVER execute its output")
    parser.add_argument("--perception-shadow", action="store_true",
                        help="Teacher executes; random-initialized v2 nominal AND correction run as input/output probes only")
    parser.add_argument("--contact-threshold-N", type=float, default=.1)
    args = parser.parse_args()
    if args.correction_shadow and not args.policy_checkpoint:
        parser.error("Correction shadow requires a nominal checkpoint; expert future actions cannot substitute for nominal plans")
    if args.perception_shadow and (args.policy_checkpoint or args.perception_version != 2):
        parser.error("Perception shadow is an untrained v2 probe under teacher execution")
    if args.perception_version == 2 and args.camera != "oblique":
        parser.error("Perception v2 requires the calibrated dual-view mount; use v1 for original-camera diagnostics")
    if args.perception_version == 1 and args.goal_command:
        parser.error("Goal command conditioning is a v2 interface")
    if not 1 <= args.num_envs <= 2 or not 1 <= args.steps <= 10000:
        parser.error("Protected-workload collection requires 1..2 environments and 1..10000 steps")
    if args.max_steps is not None:
        args.steps = args.max_steps
    if not 1 <= args.steps <= 10000:
        parser.error("--max-steps/--steps must be between 1 and 10000")
    if args.episodes is not None and not 1 <= args.episodes <= 100:
        parser.error("--episodes must be between 1 and 100")
    if args.episodes is not None and args.num_envs != 1:
        parser.error("Episode-target trials require a single environment; fixed-step mode still supports two")
    force_multiplier = 1.5 if args.condition == "force_1p5" else 1.0
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device="cpu",
                      kit_args="--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 --/persistent/physics/numThreads=2").app
    import numpy as np
    import torch
    from PIL import Image
    import isaacsimenvs  # registry
    from evaluation.eval_isaacsim import _load_env_cfg, _configure_agent
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner, _load_checkpoint_weights
    from .sim.environment import InstrumentedAssemblyEnv
    from .recording import EpisodeWriter
    torch.set_num_threads(2)
    task = "Isaacsimenvs-PreciseAssembly-Direct-v0"
    cfg = _load_env_cfg(task)
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cpu"
    cfg.precise_assembly.problem = args.problem
    configured_force_scale = float(cfg.domain_randomization.force_scale)
    cfg.domain_randomization.force_scale = configured_force_scale * force_multiplier
    cfg.student_obs.image_modality = "rgbd"
    cfg.student_obs.camera_backend = "standard"
    cfg.student_obs.camera_update_period_s = 0.0
    if args.perception_version == 2:
        cfg.student_obs.image_width = cfg.student_obs.image_input_width = 224
        cfg.student_obs.image_height = cfg.student_obs.image_input_height = 126
    env = None
    writer = None
    source = "nominal_bc" if args.policy_checkpoint else "released_rl_checkpoint"
    try:
        from .observations import TaskCommand
        command = (TaskCommand(args.problem, json.loads(args.goal_command.read_text()) if args.goal_command else None)
                   if args.perception_version == 2 else None)
        env = InstrumentedAssemblyEnv(cfg, perception_version=args.perception_version, task_command=command)
        player = None
        if not args.policy_checkpoint:
            agent_cfg = _configure_agent(task, "rl_games_sapg_cfg_entry_point", rl_device="cpu",
                                         num_envs=args.num_envs, deterministic=True, games=1, extra_overrides={})
            wrapped = register_rlgames_env(env, rl_device="cpu", clip_obs=agent_cfg["params"]["env"]["clip_observations"], clip_actions=1.0)
            runner = Runner()
            runner.load(agent_cfg)
            runner.reset()
            player = runner.create_player()
            player.set_weights(_load_checkpoint_weights(player, str(args.checkpoint)))
            player.has_batch_dimension = True
            player.reset()
            if player.intr_reward_coef_embd is not None:
                # Avoid interpolated identifiers that match none of six trained groups.
                player.intr_reward_coef_embd.fill_(50.0-10.0*args.teacher_group)

        def save_frame(name):
            if args.perception_version == 2:
                results = {}
                for v, identity in enumerate(env.research_metadata()["camera_ids"]):
                    im = env.research_packet["image"][0, v].cpu().numpy()
                    valid = env.research_packet["depth_valid"][0, v, 0].cpu().numpy()
                    Image.fromarray((im[:3].transpose(1,2,0).clip(0,1)*255).astype(np.uint8)).save(args.output/f"{name}_{identity}_rgb.png")
                    Image.fromarray((im[3].clip(0,2)/2*255).astype(np.uint8)).save(args.output/f"{name}_{identity}_depth.png")
                    Image.fromarray((valid*255).astype(np.uint8)).save(args.output/f"{name}_{identity}_valid.png")
                    results[identity] = dict(rgb_std=float(im[:3].std()), valid_fraction=float(valid.mean()))
                return results
            im = env.research_packet["image"][0, 0].cpu().numpy()
            Image.fromarray((im[:3].transpose(1, 2, 0).clip(0, 1)*255).astype(np.uint8)).save(args.output / f"{name}_rgb.png")
            Image.fromarray((im[3].clip(0, 1)*255).astype(np.uint8)).save(args.output / f"{name}_depth.png")
            raw = env.student_camera.data.output["distance_to_image_plane"][0].cpu().numpy()
            np.save(args.output / f"{name}_depth_m.npy", raw)
            return {"rgb_std": float(im[:3].std()), "raw_depth_finite_fraction": float(np.isfinite(raw).mean()),
                    "normalized_depth_range": [float(im[3].min()), float(im[3].max())]}

        camera_audit = {}
        if args.audit_default_camera:
            env.reset(seed=args.seed)
            camera_audit["original"] = save_frame("original")
        if args.camera == "oblique":
            # Declared static mounting calibration, independent of object/goal state.
            eye = torch.tensor([[0.6, -0.8, 1.1]], device=env.device) + env.scene.env_origins
            target = torch.tensor([[0.0, 0.15, 0.65]], device=env.device) + env.scene.env_origins
            env.student_camera.set_world_poses_from_view(eye, target)
            if args.perception_version == 2:
                env.configure_camera_views(args.local_eye, args.local_target)
        teacher_obs, _ = env.reset(seed=args.seed)
        camera_audit[args.camera] = save_frame("initial")
        metadata = env.research_metadata()
        from .trial_utils import code_identity
        metadata["code_identity"] = code_identity()
        metadata.update(problem=args.problem, seed=args.seed, action_source=source,
                        checkpoint=args.checkpoint if player is not None else None,
                        teacher_group=args.teacher_group if player is not None else None,
                        nominal_checkpoint=str(args.policy_checkpoint) if args.policy_checkpoint else None,
                        config=cfg.to_dict(), camera_mode=args.camera,
                        condition=args.condition, pair_id=args.pair_id,
                        force_scale_multiplier=force_multiplier,
                        configured_force_scale=configured_force_scale,
                        applied_force_scale=float(cfg.domain_randomization.force_scale),
                        torque_scale_unchanged=float(cfg.domain_randomization.torque_scale),
                        external_wrench_formula="force = randn(3) * object_mass_kg * force_scale; sampled once per native control step, then probability/lift/retract gated",
                        camera_pos_w=env.student_camera.data.pos_w.tolist(),
                        camera_pos_env=(env.student_camera.data.pos_w-env.scene.env_origins).tolist(),
                        env_origins_w=env.scene.env_origins.tolist(),
                        camera_quat_ros_wxyz=env.student_camera.data.quat_w_ros.tolist(),
                        camera_intrinsics=env.student_camera.data.intrinsic_matrices.tolist(),
                        successful_expert_demonstrations_verified=False)
        metadata.update(expert_inference_mode="deterministic" if player else None,
                        teacher_group_embedding=50.0-10.0*args.teacher_group if player else None,
                        native_max_episode_length=int(env.max_episode_length),
                        external_step_budget=args.steps,
                        initial_torch_rng_sha256=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
                        nominal_rate_hz=None if player else "see policy configuration",
                        expert_rate_hz=1/env.step_dt if player else None)
        writer = EpisodeWriter(args.output / "trajectories.h5", metadata)
        policy = None
        shadow = None
        if args.policy_checkpoint:
            from .rollout import NominalController
            from .contracts import assert_policy_compatible
            policy = NominalController.load(args.policy_checkpoint, num_envs=args.num_envs)
            assert_policy_compatible(policy.metadata, metadata)
            if args.correction_shadow:
                from .correction import CorrectionShadow
                shadow = CorrectionShadow(policy,args.num_envs,args.contact_threshold_N)
        probe = None
        if args.perception_shadow:
            from .rollout import NominalController
            from .models.nominal import NominalDiffusionPolicy, PolicyConfig
            from .correction import CorrectionShadow
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(123)
                probe = NominalController(NominalDiffusionPolicy(PolicyConfig()),args.num_envs)
            shadow = CorrectionShadow(probe,args.num_envs,args.contact_threshold_N)
        intent = policy or probe
        metadata.update(correction_mode="untrained_shadow_never_executed" if shadow else "disabled",
                        contact_gate_threshold_N=args.contact_threshold_N if shadow else None,
                        nominal_action_history=("v1: stored only, not neural input" if args.perception_version == 1 else
                            "private causal model predictions" if intent else "absent, masked; never teacher history"),
                        nominal_plan_source="untrained_perception_probe" if probe else "trained_checkpoint" if policy else "absent")
        if intent is not None:
            from .recording import model_fingerprint
            metadata.update(nominal_model_sha256=model_fingerprint(intent.model),nominal_model_config=asdict(intent.model.cfg),
                            memory_version=intent.model.cfg.memory_version)
        if shadow is not None:
            metadata.update(correction_model_sha256=model_fingerprint(shadow.model),
                            correction_model_config=asdict(intent.model.cfg),correction_initialization_seed=123)
        metadata.update(command_source_codes={"1":"expert","2":"nominal"},
                        nominal_prediction_outcome_attribution="Only selected commands cause transitions; shadow plans were not executed")
        if player is not None:
            with open(args.checkpoint,"rb") as checkpoint_stream:
                metadata["expert_model_sha256"] = hashlib.file_digest(checkpoint_stream,"sha256").hexdigest()
        writer.file.attrs["metadata_json"] = json.dumps(metadata,default=str)
        (args.output / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2))
        norms, wrench_norms, lift_gate, terminated_count, success_count, inference_ms = [], [], [], 0, 0, []
        episode_records = []
        episode_step_start = [0]*args.num_envs
        all_goals_count = 0
        completed_target = args.episodes
        start = time.perf_counter()
        for step in range(args.steps):
            packet = env.research_packet
            rng_before_inference = torch.random.get_rng_state().clone()
            t = time.perf_counter()
            with torch.no_grad():
                if probe is not None:
                    probe.act(packet)  # discarded action; the expert remains sole controller
                if policy is None:
                    actor_obs = player.obs_to_torch(teacher_obs["policy"])
                    if player.intr_reward_coef_embd is not None:
                        actor_obs = torch.cat([actor_obs, player.intr_reward_coef_embd], dim=1)
                    action = player.get_action(actor_obs, is_deterministic=True)
                    actor_obs = actor_obs.clone()  # freeze before env.step can reuse observation buffers
                else:
                    action = policy.act(packet)
                shadow_log = shadow.observe(packet,intent) if shadow else None
                if intent is not None:
                    packet = intent.last_packet
            inference_ms.append(1000*(time.perf_counter()-t))
            rng_before_step = torch.random.get_rng_state().clone()
            teacher_obs, reward, terminated, truncated, info = env.step(action)
            env.last_transition["torch_rng_before_inference"] = rng_before_inference[None].expand(args.num_envs,-1)
            env.last_transition["torch_rng_before_step"] = rng_before_step[None].expand(args.num_envs,-1)
            if shadow is not None:
                shadow.context.commit(action.clamp(-1,1))
                env.last_transition["correction_shadow"] = shadow_log
            env.last_transition["provenance"] = dict(
                expert_computed=torch.full((args.num_envs,),player is not None,dtype=torch.bool),
                expert_command=action.clone() if player is not None else torch.zeros_like(action),
                nominal_computed=torch.full((args.num_envs,),intent is not None,dtype=torch.bool),
                correction_computed=torch.full((args.num_envs,),shadow is not None,dtype=torch.bool),
                correction_executed=torch.zeros(args.num_envs,dtype=torch.bool),
                selected_source=torch.full((args.num_envs,),1 if player is not None else 2,dtype=torch.int8))
            if player is not None:
                # Exact input given to get_action, before its internal normalization;
                # includes privileged teacher fields and SAPG identity. Never student input.
                env.last_transition["expert_actor_input_diagnostic"] = actor_obs.clone()
            if intent is not None:
                env.last_transition["plan_trace"] = dict(
                    generated_now=intent.replanned.clone(),generation_step=intent.plan_generation_step.clone(),
                    generation_time_s=intent.plan_generation_time_s.clone(),
                    coverage_start_step=intent.plan_generation_step.clone(),
                    coverage_end_step_exclusive=intent.plan_generation_step+intent.model.cfg.horizon,
                    selected_for_execution=torch.full((args.num_envs,),policy is not None,dtype=torch.bool),
                    memory_updates=intent.memory_updates.clone())
                env.last_transition["nominal_action"] = intent.previous_action.clone()
                env.last_transition["nominal_predicted_chunk"] = intent.plan.clone()
                env.last_transition["nominal_chunk_cursor"] = (intent.cursor-1).clone()
            from .recording import post_observation
            env.last_transition["observation_after"] = post_observation(env.research_packet,terminated|truncated)
            writer.append(packet, env.last_transition, terminated, truncated, source)
            force = env.last_transition["contact_substeps_after_w_N"].norm(dim=-1)
            norms.append(force.cpu().numpy())
            wrench_norms.append(env.last_transition["external_force_after_dr_w_N"].norm(dim=-1).squeeze(-1).cpu().numpy())
            lift_gate.append(env.last_transition["external_wrench_lift_gate"].cpu().numpy())
            done = terminated | truncated
            terminated_count += int(done.sum())
            diag = env.last_transition["diagnostics_after"]
            task_success = (diag["successes"] >= diag["max_goals"])
            all_goals_count += int((task_success & done).sum())
            if cfg.precise_assembly.enable_retract:
                task_success &= diag["retract_succeeded"] | diag["is_random_goal"]
            success_count += int((task_success & done).sum())
            for env_index in done.nonzero().flatten().tolist():
                # The transition is already committed before the automatic
                # reset.  Summaries use only the completed prefix and the
                # native termination flags; no post-reset observation is used.
                stop = step + 1
                prefix_force = np.asarray(wrench_norms[episode_step_start[env_index]:stop])[:, env_index]
                reasons = env.last_transition["termination_reasons"]
                reason_names = [name for name, value in reasons.items() if bool(value[env_index].item())]
                episode_records.append(dict(
                    env_index=env_index, episode_id=int(env.last_transition["episode_id"][env_index].item()),
                    control_steps=stop - episode_step_start[env_index],
                    terminated=bool(terminated[env_index].item()),
                    truncated=bool(truncated[env_index].item()),
                    success=bool(task_success[env_index].item()),
                    termination_reasons=reason_names,
                    external_force_nonzero_steps=int((prefix_force > 0).sum()),
                    external_force_peak_N=float(prefix_force.max(initial=0.0)),
                    external_force_sum_N_step=float(prefix_force.sum()),
                    external_force_mean_N=float(prefix_force.mean()) if prefix_force.size else 0.0,
                ))
                episode_step_start[env_index] = stop
            if completed_target is not None and len(episode_records) >= completed_target:
                break
            if player is not None and player.is_rnn and done.any():
                for state in player.states:
                    state[:, done, :] = 0
            if step in (0, args.steps//2, args.steps-1) and not bool(done[0]):
                save_frame(f"step{step+1:05d}")
                print(f"RESEARCH_STEP {step+1}/{args.steps} max_normal_force_N={float(force.max()):.5g}", flush=True)
        forces = np.stack(norms)
        wrench_array = np.stack(wrench_norms)
        lift_array = np.stack(lift_gate)
        censored_records = [dict(env_index=i,control_steps=len(norms)-start_step,
                                episode_id=int(env.research_episode[i]))
                            for i,start_step in enumerate(episode_step_start) if start_step < len(norms)]
        externally_censored = bool(censored_records)
        result = dict(steps=len(norms), step_budget=args.steps, num_envs=args.num_envs, wall_seconds=time.perf_counter()-start,
                      planned_episodes=args.episodes, started_episodes=len(episode_records)+len(censored_records),
                      completed_episodes=len(episode_records), terminated_count=terminated_count,
                      truncated_count=sum(int(row["truncated"]) for row in episode_records),
                      successful_episodes=success_count,
                      externally_censored=externally_censored,
                      censored_episodes=len(censored_records), censored_records=censored_records,
                      trial_status="external_budget_censored" if externally_censored else "completed",
                      episode_records=episode_records,
                      all_goals_successes=all_goals_count,
                      max_normal_force_N_by_finger=forces.max(axis=(0,1,2)).tolist(),
                      contact_fraction_gt_0p01N=float((forces > .01).mean()),
                      external_force_nonzero_control_steps=int((wrench_array > 0).sum()),
                      external_force_peak_N=float(wrench_array.max(initial=0.0)),
                      external_force_total_N_step=float(wrench_array.sum()),
                      external_wrench_lift_gate_steps=int(lift_array.sum()),
                      inference_ms_median=float(np.median(inference_ms)), inference_ms_max=float(max(inference_ms)),
                      camera_audit=camera_audit, teacher_obs_shape=list(teacher_obs["policy"].shape),
                      nominal_proprio_shape=list(env.research_packet["proprio"].shape))
        if policy is not None:
            result.update(nominal_replans=len(policy.inference_ms),
                          replan_ms_median=float(np.median(policy.inference_ms)),
                          replan_ms_max=float(max(policy.inference_ms)),
                          checkpoint_smoke_only=policy.smoke_only)
        if intent is not None:
            result.update(memory_version=intent.model.cfg.memory_version,
                          memory_updates_last_episode=intent.memory_updates.tolist(),
                          memory_ms_median=float(np.median(intent.memory_ms)) if intent.memory_ms else None)
        if shadow is not None:
            result.update(correction_mode="untrained_shadow_never_executed",
                          shadow_inference_ms_median=float(np.median(shadow.inference_ms)),
                          shadow_inference_ms_max=float(max(shadow.inference_ms)))
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        (args.output / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2))
        print("RESEARCH_RESULT " + json.dumps(result), flush=True)
    except BaseException:
        import traceback
        failure = traceback.format_exc()
        print(failure, flush=True)
        (args.output / "FAILED.txt").write_text(failure)
        raise
    finally:
        if writer is not None:
            writer.close(source)
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
