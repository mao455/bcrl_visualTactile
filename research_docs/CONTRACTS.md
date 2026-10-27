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

## Legacy v1 camera (historical; see v2 extension below)
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

Correction receives chronological force history, encoder/motion history, actual past-issued commands and a masked current/future nominal prediction. It is independently parameterized. The online shadow path never changes submitted actions; full trajectory evaluation checks exact equality. In legacy v1 only, nominal receives image/proprio/valid and stored plan history is not a neural input. Neither branch receives privileged stage/goal/object state as a task observation.

## Perception v2 extension (current)

See `PERCEPTION.md` and schema v2 in `DATA_SCHEMA.md` for complete shapes. Current images are (N,2,4,126,224), RGB in [0,1] and **metric axial depth**, with separate depth validity. v1 normalized depth cannot be reinterpreted. Cameras are static, independently calibrated per environment and ordered by ID. Frames are stamped at the same rendered control observation; force/FK samples are chronological 120Hz samples ending at that observation. Terminal/reset image limitations persist.

Nominal's exact allowlist additionally includes task ID, external goal pose+mask, camera validity/times, previous **own predicted** native action+mask and the previous plan's remaining future predictions+mask. These private fields are built before current inference; environment-supplied action fields cannot override them. Corrected commands remain excluded; the default encoder also excludes teacher commands. Separately labeled past expert history is available to future explicit teacher-forcing variants, never as fake model predictions. Invalid goals/plans do not affect encoding. Task ID currently registers tight_insertion only for v2; adding tasks requires an explicit vocabulary/config change. v1 retains its prior problem support.

Correction additionally receives controller actuator targets and target-minus-encoder tracking errors (29+29), actual past-issued commands, and FK link position/rotation from encoder measurements. FK uses the original URDF including fixed joint transforms; current base orientation identity is checked. World-to-link force uses R_base_link transpose, not translation or COM rotation. Body truth appears only in diagnostics to validate this transform. Current sensors do not identify contact pairs or measure tangential force.

Every new capture declares perception_version; recorder schema and checkpoint configuration/header must agree. Compatibility includes actual per-view calibrations, view order, depth units, task vocabulary and FK asset hash. A v2 untrained perception probe is labeled explicitly and is rejected by ordinary train mode. Student sensor delay/noise/pose randomization remains unsupported and explicitly rejected; original teacher/action DR is unchanged.

## Persistent memory / execution trace extension (current)

Read [HISTORY_AND_TRACE.md](HISTORY_AND_TRACE.md) for the complete current contract. `memory_version=2` updates Nominal/Correction state every60Hz control observation and finger state on both120Hz samples; replanning remains every6steps. Each env resets only on independent episode ID change. Ordered episode chunks carry state; detach truncates gradients, not values. Missing memory version means1 even in perception2 checkpoints. Future training CLI has not been run in this milestone; memory2 validation loss and exact mid-episode recovery remain unimplemented.

`trace_version=2` records actual selected delayed command/index/source-step/effective-age using a read-only hook after the original RNG draw. Initial queue slots can all reference step0; index and effective age need not match. Original formulas/gains/limits/delay/RNG are preserved. `raw_action` aliases `submitted_action`; `executed_action` is the legacy name for `command_clipped_pre_delay`, not actual delayed execution or joint motion. Player clip-before-mean remains unobserved. Correction may read true past issued commands/targets/tracking errors; Nominal's default context remains private predictions.

Task ID enters `nn.Embedding`, with categorical meaning only. Cross-task shared structure is an open research question. Full raw trajectories include120Hz canonical joint radians for force/FK alignment; base orientation identity is explicitly required. Local force delta is inN, position derivative inbase m/s with actualdt. No pressure/slip claim. Terminal physical state/contact are pre-reset; terminal image unavailable, post-observation placeholders have available=false.
