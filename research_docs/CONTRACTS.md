# Audited contracts

## 2026-10-31配对损失研究

本轮A/B均固定sample/x0、同一旧epoch40 EMA/normalizer，独立新AdamW/cosine及RNG，额外12遍/9984更新。
A按29维均权，B先7Darm/22Dhand各自平均再各50%；尾部按实际有效目标数，无时间权重/额外动作loss。
validation以EMA前12步arm/hand等权native MSE选模，保留H64/每指/每位置指标；test各线仅最终一次。
新checkpoint仍用candidate2执行接口，另记录study_line/additional_updates/source_sha256，原checkpoint不覆盖。
细节见DP_LOSS_AND_TRANSFORMER_RESIDUAL_20261031。

## 2026-10-31 G01 Transformer residual1

独立2层/128维/4heads Transformer、128维native GRU，29维当前base独立token；剩余H64绝对参考
按当前cursor屏蔽，未来controller target不进入input。视觉来自选定G00冻结投影，q/qd、当前target/tracking、
严格t−1真实issued/delayed/时间、五指world/local合力与encoder FK/时间/valid输入独立残差。
近期13个60Hz观察保留26个120Hz样本及五指身份；GRU每60Hz更新，完整episode持久，TBPTT仅detach。
每6tick生成6个delta，执行只读取未执行下标；原名义plan不被改写。原生执行仍60Hz，base重规划5Hz。
训练标签同tick expert_clipped_pre_delay−base，允许[-2,2]；contact-valid raw MSE的arm/hand各50%，
另1e−4 delta均方幅度正则；clip只在执行组合，不阻断监督梯度。无接触逐tick精确返回base。
force_masked在任何force派生特征/记忆前清零连续world/local力，保留真实gate/valid/FK，不能称无触觉。
用户明确授权新G01/G11名义读取过去实际混合命令，使用`past_actual_mixed_commands_v1`。
名义和残差均只读取严格t−1真实issued/delayed/时间；新G01名义通过该反馈间接受到残差/触觉影响，
不能称严格触觉隔离。缓存在专家观察上逐native tick运行冻结DP，双方实际历史均明确teacher forcing；
部署使用学生实际混合命令ack。当前/未来专家命令仍仅为监督，禁止进入因果输入。
无接触bypass仅保证当前输出与当前base相等；先前残差经actual history可能改变后续名义预测。
原G00/G10 actual-history DP比较不变，旧权重不切换history flag；新残差checkpoint显式绑定base hash与此协议。
训练前必须通过真实数据缓存/在线数值验收；测试只在最终validation选模后一次；旧GRU是不同基座历史参照。

## 2026-10-30 candidate2：H64绝对参考和明确历史

G00-new/G10-new与B0/B1/residual旧checkpoint隔离，固定H64/replan12；原120/60Hz不变。
输入allowlist、三相机/DINO pin、11视觉/121控制/242物理样本显式窗口、native60Hz持续记忆及mask见
[新候选报告](LONG_HORIZON_20261030.md)。G00无触觉旁路；两组无correction、无privileged输入。
训练过去专家执行反馈严格t−1，当前未来仅监督；部署从实际学生native ack构造。subgoal/重规划/TBPTT不reset。
输出absolute_joint_reference，arm未提前限位；每tick由当前controller target逆转回命令，经原env.step执行。
controller target不进神经条件。candidate1因泄漏/划分错误禁止部署或计入有效训练。
prediction_type显式进入新checkpoint；旧缺省epsilon，新sample pilot独立初始化，禁止切换旧权重含义。
恢复默认要求配置及source hash一致；显式runtime-only恢复仅允许trainer/resource文件差异，
模型/数据/优化器/schedule/seed等配置仍必须完全一致，保存旧/新源码hash与恢复更新数。

## 2026-10-30残差观看副本

`residual_session --viewing`预分配seed981002的A/B/C自主与roll-in共6次，独立2小时含等待预算。
`evaluate_residual --viewing-checkpoints-root`显式指定既有训练root；session类型必须一致，拒绝接续或替换。
流式三视图RGB字幕记录真实学生/专家来源、观测gate、cursor与模拟时间；视频旁路不改policy图像。
原生自然终止才发布MP4，异常保留partial，不写自动reset图。switch不一致保存证据并标非配对观看。
观看结果不新增正式或旧诊断分母，原始策略/控制/判据和所有旧checkpoint语义不变。

## 2026-10-30独立Residual v1

当前授权包含真实独立Correction训练和12条诊断，详见[CORRECTION_TRAINING.md](CORRECTION_TRAINING.md)。
新增checkpoint residual1/input `residual_frozen_frame_memory2_v1`/action `gated_additive_native_command_v1`。
base必须为指定hash的正式B0 command v2、memory2/own_plan=none/action_history=none，全部冻结。
Correction独立视觉逐帧副本冻结、视觉GRU及触觉/intent/motion/MLP可训练；独立参数不共享梯度。
训练plan来自B0因果推理，监督仅当前专家clipped命令；过去专家执行明确teacher forcing。
部署每step native ack后仅Correction读取真实clipped命令、target/tracking；B0永不读mixed或触觉。
无接触exact-base；接触仅观测合力>.1N且valid，不代表工件接触。C先计算真实gate/loss mask再屏蔽网络力及派生特征。
旧Correction完整candidate语义、v3和旧checkpoint均不被重新解释；无训练有效更新明确拒绝执行。
原生控制、仿真wrapper、原始数据与正式统计保持只读。

## 2026-10-30独立审查：执行历史与原生配置准入

本轮授权版本化行动历史与有界诊断；旧B0/B1版本2与正式0/50保持不变。
新增`action_history_mode=issued`仅适用于perception2/memory2、B0/B1无correction、own_plan_mode=none。
command checkpoint版本3/history版本1明确保存训练来源`past_executed_expert_teacher_forcing`、
部署来源`past_executed_student_no_correction`；缺字段旧模型默认none，不自动升级或切换权重。
训练仅取transition[t−1]的clipped命令、实际delayed命令及source时间；当前未来专家命令只作监督。
部署每次env.step后ack，核对实际命令、episode/step/time；缺失、重复、覆盖/mixed命令拒绝。
t0置零/false；跨chunk、重规划和任务子目标不清历史。B0仍无触觉，B1保留因果力/FK与融合状态。
新增字段、单位、7类权限表及teacher-forcing分布偏差见[独立审查](INDEPENDENT_AUDIT_20261030.md)。

`assert_execution_compatible`增加action/assets/DR/obs/reset/reward/termination/task、dt/decimation、
物理配置、timeout、world_hold、offset、关节顺序与实际limits比较；seed和num_envs允许不同。
这不是资产文件/源码身份或全栈确定性证明。诊断sampler/plans/expert仅development可用，
expert最多2起点并校验示范专家SHA，原正式评估不加入expert。原native源码与研究wrapper不变。

## 2026-10-30资源与观看副本补充（当前）

用户恢复GPU时间加权整卡批次均值与完整60秒滚动均值<=50%。ResourceGuard默认cap50、
准入均值45，未测得时间段不填0；只等本任务，不更改仿真dt/动作或外部任务。内存/显存余量保留。
10月29日不限GPU授权只描述历史实验，不能用于本次或后续启动。

`evaluate_baselines --record-video`在每个控制步读取已有动作前RGB，流式CPU H264，三相机ID顺序
取实际metadata；字幕在新增64像素顶部，不进入模型。601个观察=601帧，60fps对应10.0167秒。
最后一帧是终止动作前观察；自动reset之后的帧不属于本episode，不录入。没有声称捕获终止后RGB。
完成才把.mp4.partial重命名.mp4，异常保留partial；trial视频字段保存帧数、fps、相机IDs及缺失说明。
`purpose=viewing_rerun`与正式统计分开，不将重复种子计入原50组，不假定跨运行渲染/策略逐位一致。
原数据/模型/物理/时钟/成功判据不变；MP4为有损观看副本，Depth仍作为模型输入但不写入RGB视频。

## 2026-10-29 active command-baseline extension

The 400 production episodes use frozen camera layout4 (three 448×252 views).
B0/B1 formal training is now authorized; the no-training statements below
describe earlier milestones. See BASELINE_TRAINING.md for actual evidence.
Both supervise canonical `command_clipped_pre_delay[t:t+12]` from obs[t],
never joint position or a shifted action. B0 excludes all tactile; B1 adds
causal observed five-finger force/FK and a separate fusion GRU. Both disable
own-plan conditioning online and offline. Original 120/60Hz/control is unchanged.

Command checkpoint version2 stores variant, perception/camera metadata,
memory2/no-own-plan contract, optimizer, CPU/CUDA RNG, AMP and fixed split.
B=2/T=32 batches hold independent episode slots; only group boundaries can
be saved without serialized live memory. Resume starts the next unconsumed
group; prefetched chunks never advance the committed cursor. Epoch validation
now replays complete ordered histories; old statements that it is absent are
superseded for train_dp.py, not for unrelated legacy train.py.

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
## 2026-10-29 B0/B1评估协议2补充

训练标签、checkpoint格式、模型结构与数据schema不变。正式评估准入由
`evaluate_baselines.checkpoint_identity` 核对训练完成report的best路径、variant、epoch及manifest，
并在加载前后检查SHA256；禁止last替代、诊断权重进入正式结果。`plan.json`记录实际源码hash。
网络、图像与循环状态/扩散迁至CUDA FP32，返回CPU canonical29D命令；120/60Hz、H12/replan6与
checkpoint中DDIM步数不变。CPU/CUDA随机生成器可能产生不同轨迹，协议2结果不与旧CPU探针混合。
memory2在线不保留未使用的原图滑窗时，仍按episode/step验证连续性且每控制观察更新持续状态。
正式仍50配对。只有完整cohort生成report.json；候选耗尽生成incomplete_report.json，错误单列。
原生阶段上限实测601步，子目标会重置计数；外部2400不裁短2子目标＋撤手的保守1803步允许量。
关键执行NPZ为诊断记录，不伪装成完整训练轨迹；未保存的原始图像不能从特征宣称可还原。
实测与限制见BASELINE_TRAINING末尾，训练未因本次修改重新开始。
