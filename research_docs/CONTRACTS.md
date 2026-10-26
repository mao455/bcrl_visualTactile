# Audited contracts

Baseline: upstream commit ff2fc62873f6c6cd93164123c8358e9ce2d65c9b. Local Isaac Lab VERSION=2.3.2, Isaac Sim=5.1.0.0, torch=2.7.0+cu128. Upstream installation pins are not identical to this local fork.

## Control (read action_utils.py and reset_utils.py)
`env.step(a)` accepts canonical-order `(N,29)` actions. `JOINT_NAMES_CANONICAL` in play/utils/scene_utils.py is authoritative. `_perm_canon_to_lab` indexes canonical arrays into Lab order; `_perm_lab_to_canon` does the inverse. First 7 Lab joints must be arm joints: existing action code slices `:7` before writing via joint IDs. Validate that assumption at startup.

The original RL wrapper clips raw actions to [-1,1]. New execution must do the same without bypassing `env.step`. Original action delay samples from a queue of length `max(1, action_delay_max)` (actual lags 0..length-1).

With dt=decimation*sim.dt=1/60 s and previous target p (Lab order):
- Arm: r=clip(p+1.5*dt*a, limits); target=clip(0.1*r+0.9*p, limits). Maximum unsaturated target change is 0.0025 rad per control step for |a|=1.
- Hand: r=lower+0.5*(a+1)*(upper-lower); target=clip(0.1*r+0.9*p, limits).
- `_apply_action` sets identical joint position targets twice per step; implicit actuators retain original gains, armature and URDF properties.
- `_prev_targets` is already overwritten with current targets during preprocessing; record pre-action and post-processed targets separately. Raw commands and realized joint positions are distinct. Position tracking errors can encode preloading; no current BC target nominalization exists.

## Observation provenance (obs_utils.py)
Original actor: 140 dims; privileged critic: 162 dims. Joint position is encoder-like sim joint position normalized by limits; joint velocity is rad/s; previous targets are radians (not normalized). Canonical order, 29 each. Student proprio is 87 dims and uses clean joint velocity; it does NOT inherit actor velocity noise unless explicitly configured in a separate documented student estimator.

Palm/body poses and fingertip positions are simulator rigid-body states. A deployable alternative would require encoder forward kinematics; do not automatically feed them as measured values. Palm center offset is (-0.0,-0.02,0.16) in palm-body coordinates; fingertip pad offset=(0.02,0.002,0). `*_rel_palm` is world-axis subtraction, NOT rotation into palm coordinates. Positions subtract environment origins. Internal quaternions=wxyz; original actor palm/object quaternion fields=xyzw.

The source's PALM_BODY_NAME is actually `iiwa14_link_7` (fixed hand mounts are merged), not an independently observed hand sensor. Apply its configured offset in that body's orientation before world-axis subtraction; do not reinterpret it as measured palm-frame tactile coordinates. Runtime actuator gains are sourced from scene_utils dictionaries: arm stiffness [600,600,500,400,200,200,200]; hand stiffness/damping/armature remain per-joint, not one common gain.

Object pose/velocity, geometry scales and goal keypoints originate from simulation state. Actor may delay object state, add position/rotation noise, apply goal yaw/position noise, and delay complete actor observation. Noise does not make simulator object state a real estimator. These remain teacher/diagnostic data for the student. Critic geometry is clean. Reward, success counters, lifted flag, closest distances and task progress are privileged diagnostics.

`get_student_obs()` provides image + proprio, optionally camera delay and bundled student delay. Repeated calls advance queues/add noise: sample once per control observation and cache. PreciseAssembly switches its RL `policy` group to flattened student data when `student_obs.enabled`; new integration must preserve teacher routing and avoid calling both paths inadvertently. RGB-D dimension handling in its original constructor uses 3 channels for any non-depth modality, although image reader returns 4: do not rely on that flat student contract.

## Existing camera (initial image verified; rollout coverage pending)
One per environment by default, opt-in disabled. World-mounted `(0,-1,1)`, ROS convention, quaternion identity wxyz, 160x90 depth. ROS optical +Z with identity points world +Z, which is suspect for looking down at the table; verify actual images. Default depth window [0.45,1.25] m; processed depth window-normalized, raw no-hit becomes max. GoalViz hidden when enabled. Camera has no legal task command field. RGB/depth/RGB-D and tiled/standard backends exist; raycaster and FoundationStereo reference additional infrastructure and cannot be presumed usable.

Probe01/02 actually rendered default camera: near-uniform background, depth no-hit. Fixed oblique eye=(0.6,-0.8,1.1), look-at=(0,0.15,0.65), relative to environment origin, shows geometry. This is an opt-in research mounting change, not a rewrite of original defaults. CPU standard-camera logs warn aperture offsets ignored; use measured intrinsics stored by collector, not requested calibration. More settled frames/occlusion analysis needed.

New nominal schema uses only first 58 encoder fields, not the full 87-D tensor: previous executed target can carry tactile correction. Sensor packet `(N,V,4,90,160)`, proprio `(N,58)`, contact `(N,5,3)`, two chronological 120-Hz force samples per 60-Hz control interval `(N,2,5,3)`.

## Step / reset ordering
DirectRLEnv: preprocess action → 2×(set target, physics, scene.update) → counters → dones (may advance subgoal) → rewards → auto-reset done envs → observations. Returned observations for done envs belong to NEXT episode. Capture executed targets before reset; terminal diagnostics before `_reset_idx`; never join histories across reset. `episode_length_buf` can be reset at subgoal success to extend episode; use separate monotonically tracked episode/control IDs. Phase/success changes are labels, not deployable inputs.

## Contact sensor (installed version is authority)
Installed ContactSensor uses PhysX ContactReportAPI; `net_forces_w`=(N,B,3) is aggregate NORMAL contact force in world axes, not total friction force, pressure or tactile image. `history_length` newest first; update_period=0 and nonzero history force evaluation each physics update. Contact filtering is one sensor body to multiple partners; separate sensors required for multiple finger filters. Optional local version supports contact positions/friction but validity must be measured before use. Fixed URDF fingertip/elastomer links may merge into distal phalanges during conversion: discover actual rigid bodies, do not attach to nonexistent fingertip prims.

## Additional validated boundaries and limits

The adapter requires the exact audited original proprio field order before projecting the first 58 fields. Student sensor delay/augmentation/pose-randomization settings currently fail explicitly, rather than being silently disabled. Original actor/action/object DR stays enabled. Model construction is scoped to CPU-only forked RNG, and diffusion uses a dedicated generator. The original deterministic teacher still consumes global RNG for discarded distribution samples; a seed is not a matched exogenous disturbance schedule across policies.

The installed CPU no-render path loses kinematic reset transforms on the first step. A rendered-original reference matches the instrumented environment bitwise for the documented 32-step probe, including PhysX poses, rewards and RNG. All new camera rollouts retain rendering and assert table/hole positions each step. Read AUDIT.md before interpreting old CPU compatibility numbers. The underlying engine has not been patched or certified globally.

Correction receives chronological force history, encoder/motion history, actual past-issued commands and a masked current/future nominal prediction. It is independently parameterized. The online shadow path never changes submitted actions; full trajectory evaluation checks exact equality. Nominal still receives ONLY image/proprio/valid. Its plan/history is recorded for execution but own action history is not yet a neural input. Neither branch receives privileged stage/goal/object state as a task observation.
