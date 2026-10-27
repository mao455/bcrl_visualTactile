# Dataset schema v2 (current) and v1 compatibility

HDF5 `schema_version=2`, `metadata.perception_version=2`. Original pre-action / post-action / reset contract remains unchanged. All fields below have a leading episode length T when recorded; batch/history axes are added only by loaders. Complete and incomplete episodes are retained separately. See the v1 tables below for unchanged action/diagnostic fields.

| v2 field per observation | Shape / type | Source / permitted use |
| --- | --- | --- |
| image | 2,4,126,224 float32 | RGB [0,1] + axial depth **meters**, both policies |
| depth_m / depth_valid | 2,1,126,224 float32 / bool | Invalid/no-return depth=0 with false mask; duplicate metric field permits explicit audits |
| camera_valid / camera_time_s / camera_frame | 2 bool / 2 float64 / 2 int | Real acquired frame validity, global simulation acquisition time, backend frame counter (may restart at reset) |
| proprio | 58 | Original canonical normalized joint positions and rad/s encoder velocity |
| time_s | scalar float64 | Simulated global control clock; never wall time |
| task_id | scalar int64 | Operator-selected category/embedding, not magnitude; currently tight_insertion=0 |
| goal_command / goal_valid | 9 float32 / bool | Optional operator xyz + first two rotation columns, robot_base frame; absent=zero/false |
| previous_nominal_action / previous_nominal_valid | 29 / bool | Prediction at t−1 from nominal's private context, never teacher/corrected action |
| prior_plan / prior_plan_valid | H=12,29 / H bool | Old nominal plan's remaining predictions starting at t, constructed BEFORE predicting at t; missing tails masked |
| contact_substeps_w_N / contact_substeps_local_N | 2,5,3 | Aggregate normal force at two chronological physics samples; correction only |
| finger_position_base_m / finger_rotation_base | 2,5,3 / 2,5,3,3 | Encoder-only URDF FK, link origin and R_base_link; not body truth/contact point |
| contact_time_s | 2 float64 | t−1/120, t, both invalid when contact_valid=false on reset |
| actuator_target_rad / tracking_error_rad | 29 / 29 | Original controller target and target−encoder; correction only |
| diagnostics_after/finger_position_w / finger_quat_wxyz | 5,3 / 5,4 | Simulator body truth, offline FK check only |
| diagnostics_after/fk_position_error_m | 5 | Difference between encoder FK and simulated body origin, diagnostic only |

Metadata `cameras` stores ordered IDs, resolution, per-env pos_env, ROS wxyz quaternion and actual intrinsic matrices; `env_origins_w` converts to world. Camera pose is fixed calibration, not inferred from hidden object poses. `task_command` declares vocabulary/source/frame/optional commanded pose. `fk_urdf_sha256` identifies the exact chain. `nominal_plan_source` distinguishes absent, trained_checkpoint and untrained_perception_probe.

Nominal Dataset returns only `observations.NOMINAL_KEYS_V2` + valid, and separate action_target/action_valid. CorrectionEpisodeDataset returns `inputs` (visual_history, tactile_history, tactile_valid, nominal_chunk/valid, motion_history) and **separate** `supervision.expert_command`; no correction loss is defined. It rejects absent or untrained predictions by default. Probe override is an engineering-only opt-in. No future state, expert current/future command or success field enters either policy.

memory2 defaults: persistent Nominal visual/proprio/task/own-plan state updates on every control observation; Correction carries visual/motion/intent/finger state, consuming one new visual/motion row and two physical samples each step. Only episode reset clears state. memory1 retains four-frame/stride3 visual and twelve-step motion windows. Padding/reset masks and old-plan cursors remain compatible. Packet observations are saved at60Hz, even if nominal replans at10Hz simulated time. Frame times, sensor times and actual command t−1 are recorded independently. There is no supported asynchronous resampling/delay yet.

Legacy data lack these fields and are never zero-padded into v2. `PolicyConfig.from_dict` maps missing version to1. New checkpoint headers carry version2; loader cross-checks header/model/sensor metadata. Changing resolution/calibration/depth units/IDs causes compatibility rejection. Old v1 checkpoint use is explicit via collector `--perception-version 1`.

## Trace version 2: complete native step records

Independent root `trace_version=2`; older files without it are trace1. Schema/perception2 specifies sensor tensors; model `config.memory_version=2` specifies persistent recurrence. Missing model memory_version means1, independently of camera schema. Metadata stores nominal parameter SHA256/config/memory version; expert checkpoint SHA256; newest captures also store Correction parameter SHA256/config/initialization seed. These identify actually computed predictions, not proof of learned ability.

Each episode group `length` is the **committed row count**, `complete` indicates a true simulator done. The writer appends every native control step and flushes instead of buffering the episode inRAM; readers/datasets ignore any uncommitted tail. This is single-writer HDF5, not a power-loss transaction. Shapes below omit leadingT; per-env groups prevent cross-env mixing.

| New field | Per-step shape/type | Meaning, units, availability |
| --- | --- | --- |
| `obs/finger_joint_position_rad` | 2,29 float32 | Actual canonical encoder radians at the two physics samples used for FK; invalid interval at reset follows contact_valid |
| `transition/submitted_action` | 29 float32 | Selected controller output before adapter clipping; legacy raw_action aliases this, **not** unclipped expert network mean |
| `transition/command_clipped_pre_delay` | 29 float32 | Submitted normalized command clipped[-1,1], before random delay; legacy executed_action aliases this |
| `transition/time_before_s,time_after_s` | scalar float64 | Global simulated time spanning the native step |
| `transition/diagnostics_before/*` | same as diagnostics_after | Pre-action physical truth; diagnostic only, never blanket model input |
| `transition/control_trace/valid` | bool | Native processing actually observed; unsupported replay-target bypass rejects capture |
| `.../delayed_command` | 29 float32 | Canonical command actually chosen by original delay RNG |
| `.../delay_index,selected_command_step,effective_lag_steps,queue_length` | scalar int64 | Actual slot, source native step, age and original queue size; initial slots may all hold step0 |
| `.../selected_command_time_s` | scalar float64 | Source submission time from tracked source step; not hardware exposure time |
| `.../delay_enabled` | bool | Whether original action-delay branch ran |
| `transition/provenance/expert_computed,nominal_computed,correction_computed,correction_executed` | bool | Explicit actual computation/execution flags; all Correction candidates are untrained Shadow |
| `.../expert_command,selected_source` | 29 / int8 | Command only valid when expert_computed; source1=expert,2=nominal |
| `transition/expert_actor_input_diagnostic` | actor input D | Exact tensor passed to get_action before its internal normalization, including privileged/SAPG context; absent without expert; no student access |
| `transition/plan_trace/generated_now,generation_step,generation_time_s` | bool / int64 / float64 | Actual model prediction event, repeated across plan execution ticks; fields absent without Nominal |
| `.../coverage_start_step,coverage_end_step_exclusive` | int64 | Predicted horizon interval; not a promise that all predictions executed |
| `.../selected_for_execution,memory_updates` | bool / int64 | Current item chosen for submission vs Shadow; episode-local observation update count |
| `transition/kinematics_after/*` | same as sample FK fields | Actual post-action/pre-reset q, FK, local force |
| `transition/contact_time_after_s` | 2 float64 | Timestamps of post-action physical samples |
| `transition/observation_after/available` | bool | False at done because returned packet already belongs to next episode |
| `transition/observation_after/{image,depth_valid,camera_valid,camera_time_s,camera_frame,proprio,time_s,episode_id,step_id}` | same as obs | Nonterminal post-observation equals next pre-observation; terminal entries zero placeholders with available=false |

Existing `target_before/after_rad`, post physical diagnostics, two world-force samples, reward and done flags remain. `metadata.joint_limits_rad` records actual runtime per-env canonical limits; `base_orientation_wxyz`, Lab/canonical joint names and URDF hash establish FK source/order. `missing_fields` declares terminal images and expert network mean before player clip unavailable. No fake exposure, terminal frame or delayed action is reconstructed from future state.

`EpisodeSequenceDataset.chunks` returns ordered `obs` `(1,L,...)`, `action_target` `(L,H,29)`, `action_valid` `(L,H)`, and **separate** `expert_history` `(L,29)+valid+source`. Past actual expert commands are legal labeled history; current/future targets are not inputs. Default Nominal encoder never consumes expert_history. `CorrectionSequenceDataset` yields ordered one-step inputs with2physical samples and separate current expert supervision; actual model predictions/config/memory/hash are required. Success filtering selects episodes offline, never injects a future success flag into policies.

`EpisodeReader.read(episode,index)` and `.steps(episode)` return all raw nested arrays. `.export_pickle` preserves metadata/group attrs, schema/trace/export versions and all arrays with the native-step leading axis; atomic no-overwrite publication after fsync. It loads a whole episode, so HDF5 remains canonical for random slices/large RGB-D. PKL preference and alternatives are discussed in [HISTORY_AND_TRACE.md](HISTORY_AND_TRACE.md); only load trusted exports. HDF5/PKL/raw data are ignored by Git.

## Historical schema v1

# Trajectory and policy contracts — schema 1

HDF5 root attributes: schema_version, metadata_json (full resolved task config, seed, source checkpoint, camera actual intrinsics/pose, contact paths/order, joint order, control/physics dt). Groups `episodes/envNNN_epNNNNNN` never cross a full reset. A group has complete, length, action_source, terminal_image_available=false. Incomplete captures are retained and excluded from ordinary training.

Each dataset has leading length T. Row t contains **pre-action observation**, action executed during [t,t+1], and **post-action/pre-auto-reset** diagnostics. Returned done-environment images belong to a new episode; they are never used as terminal images. Terminal joint state and contact are available. End-of-capture truncation is the group's complete=false; simulator timeout is the row's truncated flag.

| Field (per step) | Shape | Source / use |
| --- | --- | --- |
| obs/image | V,4,90,160 | Existing camera reader: RGB [0,1], window-normalized noisy depth [0,1]; nominal and correction |
| obs/proprio | 58 | Original canonical normalized encoder position 29 + encoder velocity rad/s 29; both policies |
| obs/full_proprio_diagnostic | 87 | Original student tensor, including previous target radians; diagnostic only |
| obs/contact_w_N | 5,3 | Five PhysX ContactSensors, aggregate normal world-axis force; correction only |
| obs/contact_substeps_w_N | 2,5,3 | Chronological 120-Hz samples before action; correction history/diagnostic |
| obs/contact_valid | scalar bool | False immediately after reset (no interval measured); true after native control step |
| obs/episode_id, step_id | scalar int64 | Independent full-reset and monotonic control counters, not task episode_length_buf |
| obs/time_s | scalar | Global simulated control clock; not wall clock |
| transition/raw_action | 29 | Caller output before adapter clipping; released player may already clip its network mean |
| transition/executed_action | 29 | Canonical native command after [-1,1] clip, BEFORE original action delay/smoothing |
| transition/target_before_rad, target_after_rad | 29 | Canonical actuator position targets immediately before/after original action pipeline |
| transition/contact_substeps_after_w_N | 2,5,3 | Same interval as action, saved before auto-reset |
| transition/diagnostics_after/joint_position_rad, joint_velocity_rad_s | 29 | True terminal encoder proxies, not reset state |
| transition/diagnostics_after/object_pose_w, goal_pose_w, hole_pose_w, table_pose_w | 7 | Simulator position + wxyz quaternion; privileged diagnostics/labels only; hole/table added in teacher03 and later |
| transition/diagnostics_after/successes, max_goals, retract_phase, retract_succeeded | scalar | Original state machine, diagnostic/evaluation only |
| transition/reward | scalar | Original reward, never inference input |
| terminated, truncated | scalar bool | Original native env.step flags kept distinct |
| transition/nominal_action, nominal_predicted_chunk, nominal_chunk_cursor | 29 / H,29 / scalar | Nominal rollout only; predicted plan and executed cursor, not expert future |
| transition/correction_shadow/untrained_candidate_diagnostic | 29 | Untrained forward output for integration checks; never submitted to env or used as training target |
| transition/correction_shadow/nominal_remaining_chunk, nominal_remaining_valid | H,29 / H bool | Causal model plan from current cursor; repeated end padding masked |
| transition/correction_shadow/previous_issued_command, previous_command_valid | 29 / bool | Actual command t−1, not current expert label; invalid at reset |
| transition/correction_shadow/contact_mask | bool | Valid observed distal force above exploratory threshold; not pressure/slip/safety |

The initial view count is 1. Native physical samples are saved for every control step, not only every network replan. Original action delay is preserved (default sampled lag 0..2 control ticks). In historical trace1 the applied lag was not instrumented; before/after physical targets are authoritative and must not be reverse-labelled as an exact unobserved delayed command.

## Histories and splits

`history_indices(t,L,stride)` returns oldest→current indices. Left padding repeats the episode's first sample and supplies a false validity mask. `OnlineHistory` maintains a separate deque per environment and resets only when independent episode ID changes; duplicate/missing control observations raise an error. Subgoal changes do not clear history. The same index/mask convention is used by HDF5 datasets. In historical memory1 Nominal GRU state is recomputed within each window with masked updates. Current memory2 carries state across steps within an episode; neither version persists across episode boundaries.

Training outputs are `obs={image,proprio,valid}`, `action_target=(H,29)`, `action_valid=(H,)`. Future native action targets are supervision only. End padding repeats the last action and is excluded from the epsilon loss. Normalization uses training episodes only; validation is whole-episode held out within one capture seed (not cross-seed generalization). Default dataset accepts completed all-goal AND retract-success teacher episodes; `--smoke-only` explicitly allows unsuccessful/incomplete real trajectories for software tests. This criterion targets insertion with retract enabled and random_goal_fraction=0, not all original task variants. It never generates labels.

New metadata adds actual camera positions relative to environment origins, origins themselves, correction mode and contact threshold. Input compatibility checks problem, mounting, actual orientation/intrinsics, joint order, physics/control dt, modality/preprocessing and sensor settings. Legacy schema-1 oblique captures lack explicit origins; their fixed declared mount (.6,−.8,1.1) is used with recorded orientation/K, based on the audited collector code. Simulator object/goal pose never supplies camera calibration.

Correction receives 12 consecutive 60-Hz motion/previous-command samples and 24 chronological force samples at 120 Hz, plus its own visual window. Nominal histories remain image/proprio only. Predicted chunks are logged to distinguish inference inputs from expert supervision. This historical v1 milestone had no correction dataset/loss. Current perception2 has window and sequential input datasets; no Correction loss has been selected.

## Limitations and provenance gaps

- Current adapter explicitly rejects student camera/bundle delay, camera pose randomization and depth augmentation until timestamps/RNG/mount contracts are validated. They are not silently changed. Original teacher/action/object delays and disturbances stay on.
- Original camera read performs an extra camera.update; camera's internal timestamp is not a trustworthy physical exposure clock here. Store the control availability time and config. Camera skip interval other than zero is rejected by the adapter because the original RGB-D cache returns only depth on skipped reads.
- In historical v1 the saved RGB-D is the normalized policy input; raw metric-depth example frames are separate .npy artifacts. Current v2 continuously records metric depth plus validity. Continuous raw RGB/depth calibration-grade streams and per-sample exposure timestamps are not yet recorded.
- Standard ContactSensor gives whole distal-body contact, including merged elastomer/fingertip collision shapes; not just the pad, no partner identity, tangential force, distributed pressure or contact moment. All five body channels respond in episodes02; calibration and saturation tests remain open.
- Aggregate normal force is not a slip measurement. Force peak statistics must not be called pressure or grasp-stability proof.

## Wrench trace 3：2026-10-28专家扰动实验

感知schema2、trace2和模型memory版本不变；新增**仅诊断**的metadata `wrench_trace_version=3`。此前验收数据没有该key，按1理解；中间诊断2有lift/task gate、RNG与补充状态，但无物理子步composer记录。旧文件缺字段保持缺失，不能补零声称测过。

| 新transition字段（省略T） | 形状 | 来源/时间/单位 |
| --- | --- | --- |
| external_force_after_dr_w_N / external_torque_after_dr_w_Nm | 1,3 | 原helper门控后的**请求世界向量**，控制步开始t，非最终PhysX世界向量保证 |
| external_force_nonzero / external_torque_nonzero | bool | 上述请求范数是否非零 |
| external_wrench_lift_gate / external_wrench_task_gate | bool | 本次preprocess所见latched lifted / 非retract；原生配置是否启用lift限制另见config |
| external_force_probability / external_torque_probability | scalar | 本episode原reset采样概率，无重新抽样 |
| external_object_mass_kg / external_force_scale / external_torque_scale | scalar | 原native扰动公式使用的质量及尺度 |
| applied_wrench/force_b_N,torque_b_Nm | 2,1,3 | 每次object.write_data_to_sim前实际composer之和，刚体坐标，position_data=None |
| applied_wrench/object_quat_wxyz | 2,1,4 | 同时刻对象root方向，转换到世界系的诊断真值 |
| applied_wrench/force_w_N,torque_w_Nm | 2,1,3 | 上述local buffer按同时刻quaternion旋转，实际向PhysX提交量的世界表示 |
| applied_wrench/time_s | 2 float64 | t、t+physics_dt；与contact_after的t+physics_dt、t+control_dt区分 |
| torch_rng_before_inference / before_step / after_preprocess | RNG state byte vector | 实际CPU torch RNG；第一项完整字段名`torch_rng_before_inference`，后三段分开保存；跨环境是共同全局state副本，不是独立env RNG |
| termination_reasons/{fall,max_successes,hand_far,dropped,timeout} | bool | 原任务done判断，奖励后/自动reset前克隆 |
| diagnostics_before,after/object_velocity_w_m_s_rad_s | 6 | 世界系线速度m/s及角速度rad/s |
| diagnostics_before,after/goal_velocity_w_m_s_rad_s | 6 | 目标刚体原state速度字段，非部署可见信号 |
| diagnostics_before,after/base_pose_w,palm_body_pose_w | 7 | xyz+wxyz；palm_body为原`iiwa14_link_7`，不是接触点 |
| diagnostics_before,after/lifted_object,object_initial_z_m,native_stage_step,is_random_goal | scalar | 原lift/初始高度/可被子目标清零的原进度/任务类型诊断 |

metadata新增condition、pair_id、原/应用force_scale、teacher推理模式/group embedding、checkpoint SHA256、code_identity（base commit、working tree、源码hash）、原timeout、外部预算、初始RNG hash。新采集器`result.steps`是实际控制tick，`step_budget`另存；完成/删失episode分别列出。独立trial的启动失败由外层`.status.json`保留，未产生HDF5不等于机器人失败。

详细原生公式、world→body composer限制、统计与缺失说明见[RL_EXPERT_TRIALS.md](RL_EXPERT_TRIALS.md)。所有新增真值/RNG/扰动仅供数据审计，不进入Nominal或Correction模型输入。
