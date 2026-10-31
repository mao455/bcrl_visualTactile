# Architecture and staged acceptance gates

## 2026-10-31完整系统审计

[SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT](SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md)
集中说明真实场景、RL actor、两代DP/残差、动态历史及六条原生执行链。
逐文件定义/调用和历史接口身份见PROJECT_FILE_MAP与EXPERIMENT_VALIDITY_MATRIX。
下文旧阶段的“current”仅代表其日期；新G01允许实际mixed反馈，不是严格触觉隔离nominal。
未来正式H64训练已共用`dp_objectives.py`，默认Loss B（arm7/hand22各50%），A仅历史复现；
pilot/full/resume记录并核对loss，不改变网络参数形状、已有checkpoint推理或原生动作接口。
`diagnostic_recording.py`为未来近似初态诊断逐tick记录传感/记忆/RNG/queue；CPU行为中性测试通过，未启动新Kit验收。

## 2026-10-31 actual-checkpoint architecture audit

[NETWORK_ARCHITECTURE_EXACT](NETWORK_ARCHITECTURE_EXACT.md) is the current measured specification:
594 observed/inactive module rows,743 named parameters, exact G00/G10 and Transformer training/deployment diagrams.
Residual1 uses204 tokens, two pre-LN128/4-head layers and a separate persistent761→128 GRU;
padding mask only, causal sensor windows, six29D unbounded raw outputs,10Hz refresh and60Hz execution.
Continuous force is zeroed before both memory and tokens in Masked; true contact gate remains.
G10's force-only history does not contain the residual FK features or historical encoder derivatives.
No architecture or trained weights changed in this audit; training-state replay limitations are explicit.

## 2026-10-30 G00-new/G10-new candidate2

Independent of old checkpoints: frozen pinned DINOv2-S/14 patch features plus separate depth/valid processing,
11 visual /121 native /242 physical-sample explicit histories, and a separate native60Hz persistent GRU.
512D fusion conditions the official pinned U-Net [256,512,1024], kernel5, FiLM, H64; replan every12native ticks.
G00 strictly excludes tactile; G10 adds real five-finger force history, neither has an external residual.
Absolute pre-delay joint references are converted to original native commands on every tick; unchanged env.step.
See [complete candidate specification and measured status](LONG_HORIZON_20261030.md). Initial candidate1 was
invalidated for history leakage/wrong split; only version2 can pass deployment admission.

## 2026-10-30：已实现独立加性Residual v1

原Correction GRU+MLP full-candidate/blend保留为旧研究接口。新增`models/residual.py`与
`residual_data/train_residual/residual_controller/evaluate_residual`：冻结正式B0，复制已训练frame encoder并冻结，
独立训练visual128、tactile64、intent64、motion64及MLP256/128/29，输出头零初始化。
当前观测contact gate控制加性native command residual，无接触exact-base；训练标签不经过残差[-1,1]裁剪。
新版本有真实完整episode训练、checkpoint、native ack执行链；属于监督命令纠正，不是去预加载、联合训练或RL。
当前实测与尚未完成项见[CORRECTION_TRAINING.md](CORRECTION_TRAINING.md)，方法区别见
[POLICY_COMPARISON_20261030.md](POLICY_COMPARISON_20261030.md)。历史“无Correction训练”不覆盖新授权。

> **最新2026-10-30独立审查：** 用户要求补齐真实行动历史并授权有界验证，覆盖下面观看阶段范围。
> B0/B1新增独立的61→32命令历史编码，经原context进入每60Hz持续GRU；B1仍有五指力/FK与融合GRU。
> 训练为过去已执行专家命令teacher forcing，部署为真实学生执行反馈；新版本3，不升级旧权重。
> 私有剩余计划仍关闭；没有把当前/未来专家监督伪装成学生计划。目标/混合命令不新增到nominal。
> [独立审查](INDEPENDENT_AUDIT_20261030.md)包含权限表、官方对照和实测；没有证明历史已改善成功率。

> **最新2026-10-30：** 两组50epoch和正式50配对完成，B0/B1均0/50、第一阶段601步超时。
> 本轮仅各2条完整视频与文档更新，不新增模型或训练；GPU均值恢复<=50%，内存保护保留。
> 观看入口复用evaluate_baselines与rollout，新增流式RGB视频旁路，不改变模型输入/控制链。
> 真实结果及当前原则见BASELINE_TRAINING顶部、USER_RESEARCH_DIRECTION最新追加。

> **Current 2026-10-29:** camera layout4 production is complete (400 episodes).
> Independent command B0/B1 training uses the existing spatial encoder and
> persistent memory, without changing the simulator. The batch/AMP/save/online
> call graph and actual checks are in [BASELINE_TRAINING.md](BASELINE_TRAINING.md).
> B1 is a tactile-conditioned full-command comparison, not the Correction
> policy. Historical camera-candidate and no-training statements below remain evidence.

> **2026-10-28 最新纠正：** v4近高位三视角已完成seed17/920101两条真实验收（220/214步），与各自v3参考128个非图像字段逐位相同；143字段无损转换和三路模型接口检查通过，零训练。此前“零v4成功/资源保护已完整”的文字为当时快照：上一提交的容量统计仅单root、v4仍有两处版本漏判，本轮已修正。实际背景批次看最新SESSION_HANDOFF/status；不要把下面历史状态当实时。源码通俗说明见[PROJECT_GUIDE_ZH.md](PROJECT_GUIDE_ZH.md)。


## Current camera profile (2026-10-28)

The active collection candidate is **three RGB-D views at 448×252**:
`fixed_high`, `fixed_side`, and the `iiwa14_link_7`-mounted `wrist_rgbd`.
The already committed production episodes use the earlier version-3
`wrist_bracket` calibration. The candidate version-4 `wrist_bracket_close`
moves only `fixed_high` to `(0.288,-0.405,1.463)`; until two complete
successes pass the final audit, it is a separate dataset and must not be mixed
with version 3. Older dual-view descriptions below are historical baselines.

## Frozen boundaries
Original environment/control/physics/RL remain authoritative. New package `visuotactile` is independent of Isaac Lab for model, data and history code; `visuotactile.sim` loads only after AppLauncher. Opt-in subclasses reuse task `_setup_scene`, `_pre_physics_step`, rewards and reset, attaching sensors/capturing records without replacing task math. No change to existing Gym IDs or teacher RL algorithms.

Policy-visible nominal inputs are explicitly selected camera history, encoder/command history, externally supplied task ID/goal command and their validity/time metadata. Correction sees its OWN visual/history encoder plus tactile history and frozen nominal predicted action chunk. It does not write features into the nominal network. Previous corrected commands must not become nominal action-history inputs; nominal action history is its own plan history. Encoder measurements can naturally reflect contact, which is unavoidable proprioception, not an extra tactile channel.

## Current implementation: perception v2

The current component and tensor contracts are in `PERCEPTION.md` and schema v2 in `DATA_SCHEMA.md`. `observations.py` owns the nominal allowlist, task vocabulary and operator command validation; `kinematics.py` supplies encoder-only FK; `models/vision.py` and `models/tactile.py` provide replaceable encoders. The adapter attaches three cameras/contact sensors and a read-only post-scene-update FK observer. Original task/RL/assets and control mathematics remain unchanged; the only original-source edit is an optional read-only action trace hook after delay selection.

`NominalController.act` constructs private previous-nominal/remaining-plan fields **before** adding the current observation to history and predicting a new chunk. Collector records these exact inputs. `CorrectionContext` separately constructs prior-issued-command and target/error feedback. `CorrectionEpisodeDataset` uses the same causal windows and returns expert command supervision separately. `probe_perception` compares offline/online tensors on real episodes, checks gradients without weight updates, and generates visibility evidence. Goal commands are optional external inputs, not extracted from task success state.

v1 encoder/checkpoint paths remain explicitly available. The following earlier choices and milestone text are the **historical v1 baseline**, superseded by v2 where noted. No formal model training was performed in this perception milestone.

## Provisional initial implementation choices
- RGB-D per-view CNN + GRU over finite history; tensor view axis allows adding calibrated cameras without changing data schema. Start with one verified view, inspect occlusion before deciding on two.
- Conditional temporal denoiser and DDPM training / DDIM inference for action sequences, following Diffusion Policy. Preserve 29-D native normalized action space. Bounded action chunk execution at 60 Hz; 6-step replanning gives nominal 10 Hz. Independent correction period configured in control steps; no assumed high-frequency requirement.
- HDF5 episode files, complete step streams, metadata/provenance; dataset windows use same boundary padding/masks as online history. Full episodes separated from truncated captures. Train/validation split by episode, never random frame split.
- Initial legal task context: externally configured task identity (which assembly to perform), with fixture visible in image. No simulator goal pose or automatic success-driven phase enters policy. Fixed task ID may be insufficient; explicit commanded goal pose or vision-derived fixture estimator can be added with provenance, not privileged fallback.
- Contact encoder: per-finger vector MLP + temporal GRU; correction policy separately fuses visual/proprio history and nominal future chunk, predicts a complete candidate native action. Hard no-contact nominal bypass; configured blend on contact, clipping and original actuator path. This is a replaceable baseline, not a frozen research conclusion or guaranteed safety controller.
- Correction initially disabled unless a trained checkpoint exists. Candidate training: causal frozen nominal predictions on expert/DAgger states, imitate expert action in contact and regularize deviations outside contact. Expert action is target only, no future state inputs. Later contact-outcome prediction/risk learning and on-policy data required to evaluate improvements. No claim that fitting demonstration action difference proves tactile utility.

## Phases and gates
1. Audit + persistent contracts: trace action/observation/reset and primary code; document uncertainty.
2. Sensors + collection: real image inspection, actual distal-body force readout, timestamp/boundary/action records; original paths still runnable.
3. Dataset + nominal: causal window/reset/leak tests, real recorded data denoising training, checkpoint save/load and independent policy rollout. Small data smoke test has no performance claim.
4. Correction: separate encoder and policy, real input/output/combination integration; disabled controller exact equivalence. Only train if real paired data and meaningful objective available.
5. Evaluation: same seeds/tasks/success rule, log success + force proxies + latency + provenance; camera and tactile ablations; significant data collection/GPU runs deferred while protected background workload occupies GPU.

## Open research questions
Nominalization of preload; tactile force calibration/slip observability; legal target pose estimator and phase logic; camera visibility/3D ambiguity; history length; denoiser capacity/step count; correction objective and output representation; contact threshold/blending; frequency vs inference latency; CPU/GPU physics differences. Track evidence rather than silently freezing choices.

## Implemented call graph (2026-10-26)

`collect.main` boots AppLauncher → loads the original task config → `InstrumentedAssemblyEnv` → original `PreciseAssemblyEnv` → `PlayEnv` → installed `DirectRLEnv`. Native `env.step` remains the only execution entry. The subclass adds camera/contact reporting, captures pre-reset diagnostics and asserts kinematic table/hole positions remain at their reset values. It never implements reward/action math.

| Files | Responsibility and boundary |
| --- | --- |
| `sim/environment.py` | Optional scene instrumentation and cached sensor packet; no new collisions or gains |
| `collect.py`, `recording.py` | Released teacher or independent nominal rollout; per-environment HDF5 episodes, real action provenance |
| `history.py`, `dataset.py` | Shared causal indexing and carried state; expert future actions are supervision; past expert commands are separate labeled history |
| `models/nominal.py`, `models/vendor/` | RGB-D/proprio history encoder and attributed MIT diffusion denoiser; exact input allowlist |
| `train.py`, `rollout.py` | Real-data loss, checkpoint/optimizer/RNG restore; native 60-Hz chunk cursor and 10-Hz replanning |
| `models/correction.py` | Independent tactile encoder, visual encoder, motion/issued-action history GRU and predicted-chunk GRU → full 29-D candidate; bounded composition helper |
| `correction.py` | Causal feedback window and shadow integration; no trained checkpoint loader and no live candidate execution |
| `probe_correction.py` | CPU-only replay of real contact episodes with frozen nominal predictions; validates past-command/reset alignment, never trains or executes |
| `contracts.py`, `evaluate.py` | Input/calibration compatibility, experiment fingerprints, full-trajectory alignment checks and comparable metrics |
| `regression.py`, `tests/test_visuotactile.py` | Separate-process rendered reference and local information/reset/control tests |
| `run_cpu.sh` | Limits only the newly launched process; leaves other workloads untouched |

### Historical v1 histories (superseded by memory2 below)

Nominal sees four RGB-D/encoder observations at stride 3, spanning 0.15 seconds. Its internal GRU starts from zero for each finite window, with padded frames masked. Own predicted action chunks/cursors are stored for analysis and execution; they are **not yet neural inputs**. This choice avoids feeding demonstrated or corrected commands into nominal. Causal self-plan conditioning is a later ablation requiring a consistent training rollout construction, not an excuse to substitute expert history.

Correction has its own matching visual history plus twelve consecutive encoder/issued-command samples at 60 Hz and 24 chronological force samples at 120 Hz. A command stored at observation t is the command actually issued at t−1. At a full reset its validity is false. Current/future nominal predictions start at the active chunk cursor; padded unpredicted tail is masked. No phase/reset diagnostic is fed as task information. Episode/step IDs only manage lifecycle. Subgoal changes do not reset any policy history.

The shadow path computes the candidate on real measurements, then records it under `transition/correction_shadow/untrained_candidate_diagnostic`. The environment always receives nominal. This establishes tensor/lifecycle integration, not a learned controller. A trained correction loader, justified objective, data coverage and matched rollout ablations remain acceptance gates before enabling correction.

### Historical roadmap (not authorization to train this milestone)

1. Improve/validate camera coverage and lawful goal command information under occlusion; retain fixed external task identity meanwhile.
2. Gather more successful episodes with scene health and provenance checks; train a competent nominal baseline and evaluate held-out reset seeds.
3. Generate causal frozen-nominal plans on expert and on-policy states; optionally query the expert for labels. Separate these from model inputs.
4. Compare proposed full-command supervised correction and contact-outcome objectives; train only from real transitions. Require no-contact equality, tactile-zero/shuffle and same-budget nominal-only ablations.
5. Only then enable trained correction and consider constrained joint fine-tuning. Add self-plan history, additional views or more expressive temporal models based on measured failure modes.

## Current persistent memory and trace architecture (2026-10-27)

`NominalController.act` → private own-plan fields → current frame encoder → carried GRU state at **every native control step** → diffusion only when replan is due. `CorrectionShadow.observe` → most recent visual/motion row and two new physical samples → independent carried visual/motion/intent/finger states → diagnostic candidate. Feedback windows remain available for memory1/inspection, and are not repeatedly consumed by memory2. Episode ID changes clear only the relevant batch rows; subgoals do not clear any branch.

`EpisodeSequenceDataset.chunks` → `encode_episode_chunks` → per-step condition/isolated future supervision. `CorrectionSequenceDataset` → `replay_correction_chunks` → isolated actual model plan/context and expert labels. Chunking, detach and reset have distinct meanings. Default future trainer uses sequential TBPTT; no optimizer was run this milestone. Historical training roadmap above is not active authorization.

`env.step` → original delay selection → optional `_record_action_trace` → unchanged target math → physics/FK/contact → pre-reset diagnostics → reset/observation → `EpisodeWriter.append`. `EpisodeReader.steps/read` reconstruct raw nested rows; PKL export is optional. Trace2 retains command provenance, plan parameter hash/generation/coverage/selection, targets and outcomes. Actual terminal image is missing and masked. See [HISTORY_AND_TRACE.md](HISTORY_AND_TRACE.md) and [DATA_SCHEMA.md](DATA_SCHEMA.md) for exact rates, units and version boundaries.

Task ID remains categorical. Expert history is useful but separate from private predicted plans; ordinary imitation targets still contain potential preload. Current encoders are untrained; soft spatial moments are not semantic keypoints. Fixed state compresses history without lossless or task-utility guarantees. User requirements and open research decisions live in [USER_RESEARCH_DIRECTION.md](USER_RESEARCH_DIRECTION.md).

## 独立原专家实验入口（2026-10-28）

`run_rl_trials`负责不可覆盖的配对计划/逐attempt状态、顺序启动、资源边界与验收；复用`collect → InstrumentedAssemblyEnv → EpisodeWriter`。`analyze_rl_trials`负责纯专家轨迹审计、真实终止分类、缺失值与配对共同前缀分析。它们不调用optimizer，不需要Nominal/Correction计划来冒充专家数据。wrench_trace3新增字段都是特权诊断，不修改感知编码、memory或Dataset模型输入。具体入口、字段和本机Lab施力限制见[RL_EXPERT_TRIALS.md](RL_EXPERT_TRIALS.md)。

## 2026-10-28 完整专家数据接入层

新增`run_insertion_dataset`调用原`collect`，通过`run_rl_trials.validate_completion`共用新/恢复完成门槛；`initial_geometry`仅诊断初始box合法性。`compact.convert_episode`把已审核逐步记录转换成三主文件并精确往返验证；`manifest`统一定位和筛选，`EpisodeFile`让原Reader/Dataset保持相同模型输入；`ManifestSequenceDataset → encode_episode_chunks → Nominal.encoder`已用两条真实新episode共411步零更新验证。没有模型结构改动。`analyze_insertion_dataset`读相同reader审计、统计、最多4预览。

接入读取不等于正式BC训练完成。现有train.py只支持单H5，现能识别storage1但显式拒绝manifest正式训练；纯专家perception2缺自身计划时也拒绝正式训练。未来需确定无计划baseline或因果模型回放，接入按pair_id分组、训练统计、memory2验证loss和checkpoint绑定；不得用专家未来动作填plan。Correction纯专家训练仍不成立。旧checkpoint/rollout原接口保留，本轮没有训练或声称完整新训练闭环。

## 三视角观测扩展（2026-10-28）

`camera_rigs.rig_contract`给研究wrapper提供固定外参/ID。wrapper在原场景之上添加纯Camera prim，运行时解析真实link7刚体；CameraParams沿每个render product返回动态世界位姿，原student+第二相机缓存扩成V路。`collect`→`EpisodeWriter`→`compact.convert_episode`→`ManifestReader`/`EpisodeSequenceDataset`→原可变views编码器/持续记忆，结构未重写。Nominal与Correction视觉支路均实测V=3；world calibration用于诊断而不新增特权输入。

`camera_review`提供完整视频、原图/深度/有效性、非图像配对、光心mesh-AABB距离和孔口深度诊断；`camera_diagnostics`只在固定物理状态做render对照；`check_camera_interfaces`只做真实数据因果计划/Shadow/Jacobian/恢复，没有optimizer。`collect_success_batch`复用现有collector、统一completion gate、无损转换和manifest，管理新400成功协议，和旧offset实验runner分开。没有改原RL、传感编码网络、动作、资产、动力学或成功判据。

## 2026-10-29持久专家采集

本轮生产入口和实测见[NIGHT_COLLECTION.md](NIGHT_COLLECTION.md)。一个CPU Isaac实例/4独立env、原released循环专家批量推理；标准三Camera、原120/60Hz。Recorder线程有界队列直接写无损storage2，两spawn审核进程完成143字段来源hash＋完整原生/传感审核，单manifest拥有者发布成功。原资产/原task/RL未重构，Nominal/Correction不参与。本轮8m环境原点间距用于隔离视场，不改变每个环境局部几何。
