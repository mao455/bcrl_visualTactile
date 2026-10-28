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

## 本机外力执行观察补充（2026-10-28）

原helper在60Hz生成/门控world-request wrench；当前Lab2.3.2 fork的deprecated setter经permanent composer转成body-frame，再在120Hz `write_data_to_sim`提交。composer姿态缓存与请求world方向可能产生差异，不能仅用`_object_forces`宣称测得最终世界系施力。新wrench_trace3记录每个子步实际local buffer、同时刻object quaternion、世界系换算与提交时间；不修改原helper、Lab或动力学。详见[RL_EXPERT_TRIALS.md](RL_EXPERT_TRIALS.md)。这些字段全是特权诊断，模型输入allowlist不变。

## 2026-10-28 可选初始化与施力兼容模式

`InstrumentedAssemblyEnv(initial_offset_xy=(0,0), wrench_frame_mode="legacy")`默认保留旧路径。额外XY各不超过.05m（仅本轮轴向条件实测），要求桌yaw0；在原Play reset之后、assembly目标构造之前平移物体，无新增RNG，记录raw/final pose和总object−hole。带experiment-condition的采集在第一专家动作前检查实际URDF box重叠，非法即保留现场停止。不是修改reset分布后悄悄重采样。

可选world_hold只重新表达**原生DR世界请求**：每个physics write用当前object root wxyz的逆旋转变成body力/力矩，调用permanent composer公开setter(is_global=False)，随后原write。没有重写随机力公式/概率/门控/scale，没有改变共享Lab；instantaneous composer活跃拒绝。wrench_trace3记录实际每子步body提交及当前姿态对应world值，120Hz/60Hz不变。legacy与旧seed17的220步16数组逐位一致；world_hold非零力方向误差<1e−6N；非零力矩实测尚缺，详见INSERTION_DATASET。两模式数据分批，不能混算。

CPU必须render，保持table/hole变换审计。格式压缩不改变pre/post时间：terminal图明确缺失；预算尾post另存一帧。公开仓库不含轨迹/视频/权重。本轮已停止，不自动再采样。

## 2026-10-28 三视角补充（新授权覆盖旧批次的“不自动扩量”）

新rig3选择/配对证据见CAMERA_VALIDATION。旧双路默认保持，新增固定A/B和link7固定外参腕部。渲染与校准来自同render product，CPU下不能信任移动camera.data的USD缓存。相机数量、顺序、分辨率、mount、实际K、render配置均纳入新视觉checkpoint契约；原RL专家输入仍140D，不受新图像尺寸影响。Schema2/trace2/math/dt不变，storage2支持V路+动态pose，禁止双视角权重静默解释成三视角。

用户已明确授权通过验收后采400成功，旧60尝试/10GiB停止规则只属于旧实验。新protocol `triple400_native_seed_rejection_v1`按原reset seed序列记录/拒绝非法起点，不改物理，不重试同seed；95GiB/1600尝试/单条2400步900s/40GiB空闲底线，软件或审计错误停。所有数据留本地，0参数训练。

## 2026-10-29 persistent production extension

The optional production adapter preserves schema2/trace2/storage2 logical arrays. `stream_writer_version=1` writes RGB uint8, exact float32 metric depth and masks directly with lossless LZF. `source_field_digests_json` records dtype, per-row shape and SHA256 over chronological contiguous source bytes for **every** logical field; an independent CPU worker reconstructs and verifies all fields before commit. Terminal image remains unavailable; graceful nonterminal censor preserves its actual final post frame. `source_env_index`, `source_episode_id`, `persistent_run_id` and `env_origins_w` identify each independent episode. Physical/control dt, released expert and success criteria remain unchanged.

Per-environment metadata stores its own matrices/limits/origin; camera mount compatibility canonicalizes only the `/env_N/` instance component, preserving actual mounting parent and offset. The 8m parallel grid isolates cameras; camera local calibration is still v4. One seed belongs to an entire persistent run, never to an independently reseeded episode; historical metadata `initial_torch_rng_sha256` denotes the run's first reset. Actual per-step CPU RNG is recorded in transitions. Invalid native starts are resampled for that env only, logged before any expert action; this conditions the dataset on box legality and eventual success, not a native success-rate estimate. See NIGHT_COLLECTION for actual measured changes.
