# Engineering change and verification log

## 2026-10-26 — Phase 1 started
- Added root `AGENTS.md`: persistent instructions protecting original physics/RL and information boundaries.
- Added `research_docs/{README,CONTRACTS,ARCHITECTURE,CHANGELOG}.md`: research objective, actual audited runtime semantics and staged gates. No original source modified.
- Read PlayEnv hook wiring; native action pipeline; canonical mapping/reset queues; actor/critic/student builders; student camera setup/reader; PreciseAssembly student routing/state; installed DirectRLEnv step/reset and ContactSensor implementation.
- Findings: arm delta vs hand absolute; student 87-D proprio not actor-noise equivalent; actor noisy object state still privileged; default camera optical axis questionable; phase resets episode_length_buf; auto-reset happens before returned observations; original RGB-D flat student dimension mismatch.
- Verification: source audit only at this point. New camera/contact/model behavior remains unverified. Prior compatibility outputs are not evidence for these components.

## 2026-10-26 — Phase 2 initial integration and failed probe
- Added `visuotactile/{__init__,recording,collect}.py` and `sim/{__init__,environment}.py`. Previously no research data path existed. New opt-in subclass preserves teacher 140-D policy space, original hooks and RL code; adds reporting APIs on existing distal rigid bodies and separate cached RGB-D/encoder/contact packet. Original source files unchanged.
- `collect` loads original YAML, preserves DR/physics/rewards, changes only batch/device/sensor configuration, calls released SAPG teacher and native `env.step`. `recording` separates per-episode sensor observations from transitions/privileged diagnostics in HDF5. Terminal joint state/contact captured after reward and before auto-reset. Terminal image deliberately marked unavailable.
- Nominal input correction: use 58-D normalized joint positions + rad/s velocities; original 87-D student tensor retained as diagnostic only. Previous executed targets can contain correction output, so cannot be a nominal input once correction is enabled.
- Probe01/02: camera and five distal sensor initialization succeeded; original RGB image near uniform and depth no-hit, manually inspected PNGs. Fixed oblique view shows robot/table; camera aperture offset warning means intended original calibration is not honored by this simulator. Actual intrinsics are recorded. No completed rollout from these runs.
- Probe02 exposed missing SAPG coefficient identifier when bypassing player.env_step; collector now appends `player.intr_reward_coef_embd` exactly as original BasePlayer does. No vendored code changed.
- Kit shutdown can mask exceptions with exit 0; failures are now written as `FAILED.txt`, successful runs require `result.json`.
- Initial checks: compileall passed; runtime action loop pending retry. Own process sampled at 598 MiB GPU; protected job untouched, still present.

## 2026-10-26 — Sensor/episode vertical slice verified; model integration started
- Probe03 failed inside Kit/omniclient startup (C++ system_error/abort), before simulation data; subsequent launch succeeded. Probe04 ran 120 native control steps, 1 environment, original DR enabled. Camera audit: original RGB std=0 and finite depth fraction=0; oblique finite depth fraction=.6904. Three force channels positive; all tensors finite. Arm target maximum change=.00250006 rad, matching original smoothing/dt contract.
- episodes01 hit a genuine async reset and exposed mutation of teacher RNN inference tensors outside inference_mode. Replaced collection inference_mode with no_grad (original RL source untouched). Failed file retained, not used for results.
- episodes02 ran 660 control steps × 2 envs in 25.55 s after startup: 7 completed episodes, 6 all-goal successes, 2 incomplete tails. All five distal sensors positive; maxima [38.61,26.62,27.32,91.28,25.11] N. This is training-config collection, not the earlier eval_offline protocol (which disables DR and requires consecutive near-goal steps). Do not compare their rates directly. Recorded joint/action/image histories and resets checked. Own 2-camera process sampled at 2251 MiB GPU; protected workload untouched.
- Added `history.py` (causal indexing + per-env deques), `dataset.py` (strict nominal allowlist, episode splits and target masks), `models/nominal.py` (visual/proprio GRU + action diffusion), MIT `models/vendor` modules with license/hash provenance, `models/correction.py` (independent tactile encoder/candidate policy and execution guard), `train.py` (masked real-action denoising, optimizer/checkpoint/RNG restore), `rollout.py` (6-tick replanning, one native action per tick).
- Added research_docs RESEARCH and DATA_SCHEMA with sources, hypotheses and actual tensor provenance. Changed packaging-only `pyproject.toml` / `setup.py` to include independent package; optional dependency versions match inspected environment. No task/physics/RL files changed.
- Added `tests/test_visuotactile.py`: 5 passed in 2.53 s. Tests cover async reset/history alignment, duplicate-read rejection, actual original arm/hand mapping, nominal illegal-key rejection, independent correction parameters, disabled/noncontact exact nominal equality and untrained rejection. Synthetic sentinels are unit-test fixtures, never demonstration data.
- Collector now allows teacher-free nominal checkpoint execution, records nominal plans/cursors, validates task/camera identity. Added explicit six-way SAPG teacher group selector to avoid interpolated identifiers with no trained group match. episodes02 predates selector: two original player IDs were 50 and 0, both valid. Default future runs select group 0/ID50 for every env.
- Nominal short training, checkpoint roundtrip and live rollout are in progress; no learning-performance claim yet. Correction training is intentionally not implemented.

## 2026-10-26/27 local — Regression diagnosis, completed learning/rollout checks

This entry supersedes the preceding “in progress” status; earlier entries remain as history. Experiments started October 26, and documentation continued across the Asia/Shanghai date boundary.

- **Regression root behavior:** extended visuotactile/regression.py from joint/target/reward-only traces to initial state, object/goal/hole/table states, raw PhysX transforms and torch RNG. Added a rendered-original mode: original class plus sim.render in observations, without contact/camera sensors. Original CPU no-render kinematic transforms jump to spawn origin on step one; rendered original and instrumented match bitwise for all 15 trace fields over 32 steps. Evidence: regression_v2_original, regression_v2_rendered, regression_v2_instrumented, regression_v3_rendered and regression_rendered_comparison.json. The earlier stochastic-DR explanation was wrong and has been explicitly withdrawn. Original simulation/task/RL files remain unchanged. No claim about the untested GPU physics path or entire local fork.
- **Sensor adapter hardening:** visuotactile/sim/environment.py now validates the audited proprio field order and hides GoalViz, rejects unvalidated student sensor delays/augmentation/random mounting and RGB-D frame skipping, records table/hole poses, and checks their reset positions after every control interval. Original teacher/action/object DR remains active. Inspection found camera pose randomization does run despite the student enabled routing flag; its baseline config conflicts with a manually remounted oblique camera, so that untested combination is rejected rather than silently miscalibrated.
- **Checkpoint RNG:** visuotactile/rollout.py now constructs loaded models within CPU-only fork_rng so loading after reset does not change simulation randomness. Sampling uses a dedicated generator. train.py also avoids CUDA RNG initialization and checks smoke-mode consistency on resume. Original teacher code still samples an unused Normal action before choosing its mean; same-seed cross-policy trials are not matched disturbance histories.
- **Data eligibility:** dataset.py originally accepted all-goal completion; now it also requires retract_succeeded for the tested insertion protocol. All six prior eligible episodes satisfy this. Expert future native actions remain supervision only; neither encoder observes them. No demonstration nominalization was added.
- **Independent correction history:** models/correction.py now has a separate motion/previous-issued-command GRU, a masked predicted-intent GRU and the independent tactile/visual encoders. Inputs: visual (B,4,1,4,90,160), proprio (B,4,58); 12 native-step motion/previous-command samples; 24 force samples (B,24,5,3); a masked (B,12,29) current/future predicted chunk. Output remains a complete bounded (B,29) candidate. No expert current/future actions or state-machine labels are inputs.
- **Actual integration, without fake control:** added correction.py with per-environment causal context/commit/reset and CorrectionShadow. collect.py gains --correction-shadow and logs candidate/context under a diagnostic group while ALWAYS submitting nominal. Before this phase correction had only unit-test tensors; now it runs on real online observations, and probe_correction.py replays real nonzero-contact episodes using causal nominal predictions. No correction loss, trained loader or live corrected execution is claimed.
- **Compatibility/evaluation:** new contracts.py compares checkpoint sensor/action/calibration contracts; collect.py records camera poses relative to environment origins, correction mode and group provenance. Added evaluate.py: audits every episode's row alignment, terminal/reset boundaries, finite/bounded commands, next-observation vs prior terminal velocity/contact, and shadow's exact submitted-action equality with nominal. It fingerprints task/physics/sensor/seed/env count and checks equal rollout budget before allowing rate comparison. This does not turn two trials into a statistically meaningful result.
- **Resource execution and discoverability:** added run_cpu.sh (new process only: nice19, affinity22/23, two threads), bounded collection to at most two envs. README.md now links the independent research directory. Updated AGENTS.md, README/ARCHITECTURE/CONTRACTS/DATA_SCHEMA/RESEARCH/RUNBOOK and added AUDIT.md with original call chains, native semantics, camera evidence and CPU limitations. Packaging-only pyproject.toml/setup.py changes include the new package and optional pinned dependencies; no shared environment upgrade or original-source edits.

### Real verification results

- Unit suite: 8 passed in 2.37 s (unit_tests_v2.log). Added load-preserves-RNG, nominal tactile/privileged-input invariance through replans and resets, and correction current-cursor/tail-mask/asynchronous-reset/past-command checks. Existing tests cover the actual original action module's different arm/hand rules. Synthetic test sentinels are not used for learning.
- episodes02 full data audit: 7 complete episodes, 6 all-goal+retract successes, 2 incomplete tails; velocity/contact transition alignment exactly matches. Its two SAPG IDs were 50 and 0; future runs explicitly select group0/ID50 per environment.
- train01: 100 updates, five training episodes/939 frames and one validation episode/208 frames; epsilon loss 1.15147→1.00413, validation .999816; exact checkpoint roundtrip. resume01 restored to 102 updates with another exact roundtrip.
- train02: resumed train01 for 900 more updates, batch8, total1000; first continuation loss .9890, last .79790, validation .73599; exact checkpoint roundtrip; 142.89 s on constrained CPU. Data are real, but very small and same-capture held out; not cross-seed generalization or competence evidence.
- teacher03: unchanged original teacher with explicit group0, 660 steps ×2 envs, 6 complete episodes/5 all-goal+retract successes, 2 incomplete tails, 23.44 s excluding startup. Five force peaks [33.34,21.09,43.65,45.56,24.74] N. Every reset/step passed table/hole health checks.
- nominal_shadow02: train01 checkpoint, same seed/config/budget, 2 complete episodes/0 successes, zero contact, 111 batch replan events. Replan median28.58/max37.75 ms. Candidate outputs were recorded and never executed.
- nominal_shadow03: train02 checkpoint, same budget, 2 complete episodes/0 successes, zero contact. Replan median28.44/max35.73 ms; whole per-step policy+shadow maximum171.48 ms, shadow maximum158.80 ms. No hard real-time guarantee. The improved denoising loss did NOT establish useful task behavior.
- correction_replay01.json: two completed teacher episodes, 225/230 steps, 184/134 contact frames and 28.71/44.92 N force peaks. Real-force, motion, nominal-intent and action-history forward passes stayed finite; command t−1 and episode-reset masks checked. This is an untrained replay integration probe, not a tactile controller result.
- Existing original warnings about writing velocity to kinematic bodies remain; startup and renderer warnings are logged. A requests dependency warning remains. We did not upgrade shared dependencies, signal/renice any background program, or change driver/global GPU settings. GPU cameras still consume resources; low-priority CPU work is not a proof of zero contention.

### Open gates, deliberately not called complete

Camera geometry/detail under occlusion and lawful goal commands; additional demonstrations and a competent nominal baseline; consistent optional self-plan input history; correction dataset/objective/trained checkpoint/execution; no-contact and tactile-zero/shuffle ablations with meaningful performance; calibrated contact/slip signals; continuous raw/terminal images and per-exposure timestamps; student delay/noise support; real inference delay/deadline behavior; multi-task/GPU/real-robot validation. The next stages and information boundaries are in ARCHITECTURE.md and RESEARCH.md.

### Final verification — 2026-10-27 Asia/Shanghai

- After the final cursor/threshold/resource guards, unit_tests_final.log reports 8 passed in 2.60 s. compileall, bash syntax check and git diff --check passed. No original task/RL/assets/evaluation changes appear in git diff. New schema/context code remains isolated.
- comparison_final.json audits teacher03, nominal_shadow02 and nominal_shadow03 completely; all alignment/finite-number/boundary checks pass, including exact shadow-vs-nominal executed commands and previous-command/reset alignment. The three runs share the recorded comparison protocol and budget; success totals are respectively 5/6, 0/2 and 0/2.
- pyproject.toml explicitly includes the upstream vendor LICENSE and PROVENANCE.md as package data; no wheel-distribution test is claimed. Added provenance fields to evaluate.py output. Corrected stale README/RUNBOOK status and preserved earlier changelog entries.
- Copied the completed simulation/regression process logs into their output directories. verification_manifest.json records the baseline commit, source/document hashes and result checks. No ongoing research simulation or training job is intentionally left running. No commit was made.

### 2026-10-27 — 首次公开同步

- Verified the staged public tree, local-only `user.name/user.email`, fork/upstream remotes and final test result, then created commit `159599f0575d8f61fbf21b615e2b02d0fd4168ee` (`Add visual tactile imitation research system`).
- Pushed only to `origin/main` at https://github.com/mao455/bcrl_visualTactile . `upstream` remains the original Play2Perfect remote. The follow-up documentation status update is intentionally tracked in the next local commit.


### 2026-10-27 — 双视角/五指感知 v2（无训练）

用户本轮授权建设真实可获得的感知、可替换编码器和完整网络接口，并按Git流程提交/同步。开发在`feature/multimodal-perception-v2`上进行，原分支干净、起点177a080。未改动任何原task/RL/资产/控制/物理源文件，未启动BC/Correction/RL训练，未操作后台程序。

#### 文件与前后调用关系

| 路径 | 原行为 → 本轮行为 / 输入输出 |
| --- | --- |
| `visuotactile/observations.py`（新增） | 集中v2 nominal allowlist、固定camera/task身份、外部TaskCommand验证和缺失plan mask。绝不从teacher目标/动作填值。 |
| `visuotactile/kinematics.py`（新增） | 读取原URDF和编码器q → 五个末节base位置/旋转，world法向力转link；不读取sim body pose。 |
| `visuotactile/sim/environment.py` | 单相机+world force → 兼容v1，v2增加第二固定相机、raw metric depth+mask、per-view frame/time、两物理子步的encoder FK/local force、外部task命令、target/error。只在原scene.update后读取编码器；body真值仅用于diagnostics。 |
| `visuotactile/models/vision.py`（新增） | 两视图分别经共享RGB ResNet18到layer2、独立2-channel depth+mask stem、空间keypoint/grid；按相机顺序融合、mask时序，128D输出。无预训练权重下载。 |
| `visuotactile/models/nominal.py` | 保留原CNN作为LegacyVisualHistoryEncoder；PolicyConfig显式version，缺失版本旧checkpoint→v1；v2接新视觉/目标/自身历史。扩散动作仍H×29。 |
| `visuotactile/models/tactile.py`（新增） | 原扁平15D力 → 五指身份、world/local力、FK pose/运动、接触事件与时间，逐指共享MLP/GRU后融合64D；未训练。 |
| `visuotactile/models/correction.py` | v1兼容；v2使用独立视觉和新触觉编码器，motion额外接target/error共58D，仍单独policy head→29D候选，禁止未训练执行。 |
| `visuotactile/rollout.py` | 只有image/proprio → 版本化历史；v2在当前预测前由私有旧plan构造上一nominal命令/剩余plan。真实加载cross-check版本，step0拒绝执行。 |
| `visuotactile/correction.py` | 新视觉allowlist；触觉12control×2physics的force/FK/time；motion12步并含原控制目标。shadow直接使用nominal实际输入packet，不反向污染nominal。 |
| `visuotactile/dataset.py` | schema1专用 → 严格识别1/2及camera轴/metadata；新增CorrectionEpisodeDataset，inputs与expert supervision分离，只接受有来源的因果nominal预测。untrained probe需显式测试开关。 |
| `visuotactile/recording.py`, `visuotactile/__init__.py` | 记录声明的schema；当前版本2，legacy缺版本明确1。沿用每env独立episode与pre-action / pre-reset transition组织。 |
| `visuotactile/collect.py` | 默认v2两相机224×126；新增局部mount、外部goal JSON及teacher执行的perception-shadow模式。旧checkpoint需显式v1。准确记录预测源与动作来源。 |
| `visuotactile/train.py` | 按真实数据schema建模/保存/恢复；普通train拒绝untrained perception probe数据。仅维护入口，本轮没有运行训练。 |
| `visuotactile/contracts.py` | 单相机校准 → 按版本比较ID、分辨率、各env实际K/pose、depth语义、goal词表/frame与FK hash；不同env mount未建模时拒绝。 |
| `visuotactile/evaluate.py` | 支持v2；精确校验过去控制目标/跟踪误差、旧nominal动作/计划因果关系。teacher旁路不再错误要求专家动作等于nominal候选；真正nominal rollout仍要求精确相等。 |
| `visuotactile/probe_correction.py` | 保留旧真实数据replay，按version补齐v2所需观测读取。 |
| `visuotactile/probe_perception.py`（新增） | 真实v2数据→时钟/坐标/FK检查、阶段RGB/depth/mask展示、前向/梯度/重载、offline-online完整输入逐项比较；没有optimizer。 |
| `tests/test_visuotactile.py`, `tests/test_perception_v2.py` | 原8项标记legacy，新增v2多模态梯度/独立权重/私有plan/信息边界/goal合法源/无效mask/异步reset/版本/重载测试。合成量仅为边界sentinel，不作训练标签。 |
| `pyproject.toml` | research extra显式记录已验证torchvision0.22系列（torch2.7配套）；没有安装/升级共享环境。 |
| `research_docs/PERCEPTION.md`（新增） | 固定来源commit/许可与实际阅读文件、选型、相机候选、空间/时间/数据边界、视野和未解决问题。 |
| `research_docs/media/perception_v2_views.png`, `research_docs/evidence/perception_v2.json`（新增） | 审计后的少量公开研究图和数值汇总，含local数据hash；不包含原轨迹/模型/视频。 |
| `README.md`, `AGENTS.md`, `research_docs/{README,ARCHITECTURE,CONTRACTS,DATA_SCHEMA,RESEARCH,AUDIT,RUNBOOK,CHANGELOG}.md` | 当前v2状态/规范/调用关系/字段、真实结果、可复现无训练命令；旧v1证据保留且明确标识。 |
| `research_docs/GIT_WORKFLOW.md` | 修正原来无效的多remote fetch示例为`git fetch --multiple --prune upstream origin`，补充完成分支后的授权fast-forward同步说明。 |

#### 实际检查与未完成项

1. 公开源码DP/LeRobot/OpenViTac/3DTacDex读到固定commit，许可/环境差异记录PERCEPTION。未复制新增第三方源文件或引入TacEx/GelSight图像模型。
2. 相机候选真实短轨迹：camera01=1×300、camera02=2×260、camera03=2×240ticks，第一/第二候选分别有上沿/下沿裁切；最终eye(−.7,−.35,1.15),look-at(0,−.03,.68)。最终采集final01=2×240ticks，双224×126，480transition。全局/局部光轴夹角84.98°，阶段图手工查看；原点in-frame检查不冒充接触区可见率。
3. checks02全轨迹时钟/force旋转/FK/sensor有限性/先前command-target-plan对齐通过；FK误差≤6.60e−7m，旋转矩阵≤9.82e−7。真实数据Nominal(1,12,29)/Correction(1,29)各分支梯度非零且有限，参数更新0；checkpoint精确重载，offline-online完整输入相等。
4. 原渲染环境vsinstrumented 32步15字段逐位一致；camera03/final01含不同时刻reset的480个专家transition动作/目标/奖励/关节/物体/孔座/success/done逐位一致。CPU无渲染路径仍禁止作有效场景数据。
5. v1真实1000step checkpoint与schema1数据仍可前向；v1/v2混用被拒绝，旧文件未改写。requests版本warning保留，没有升级共享依赖。
6. 最终完整旁路median206.94ms、Correction199.16ms（2env，两核CPU）；未满足真实10/60Hz。相机RTX仍使用GPU，未做GPU训练/修改后台任务。所有自建仿真进程完成后退出。
7. 两个完整成功episode在同tick进入retract并终止，没有独立retract阶段/terminal图片；图的最后行明确标为终止前可得帧。默认合法task ID已有，goal pose缺失则mask=false；目标估计、异步传感器、触觉预训练/Correction目标、模型能力、预加载nominalization仍未解决。

本轮公共提交只包含源代码、测试、文档及一张小型合成场景证据图/汇总JSON；outputs、checkpoint、HDF5、日志、参考源码下载、虚拟环境均忽略。原MIT和已有vendor版权声明保留。提交/同步结果在完成时向用户报告。

最终单元测试：15 passed（包含v2两环境异步reset检查）；详细本地日志`outputs/visuotactile/perception_unit_tests.log`。

### 2026-10-27 — 持续因果历史与完整动作执行追溯（无训练）

从干净`200ea3490530c4bd64a6351a5e43a7c9d76c4da9`建立`feature/causal-memory-trace`。重新审计发现原GRU只重放有限窗口，无跨forward状态；旧`executed_action`只是延迟前裁剪命令；writer原先整episode存RAM。按用户本轮要求修复这些机制，未改变机器人资产、动作数学、动力学、原RL或成功判据。唯一原源码变化是action helper的7行可选只读hook；没有重新抽delay/RNG。

#### 文件、调用关系与前后行为

| 路径 | 原行为 → 现在；输入输出与职责 |
| --- | --- |
| `visuotactile/history.py` | 只有有限窗口索引 → 增加B-first递归state reset/detach、显式scan；块边界不清记忆。 |
| `visuotactile/models/nominal.py` | 隐含每窗零状态 → 独立memory_version，缺字段按1；encoder可带state/返回逐步conditions，diffusion分离predict/loss_from_condition；memory2拒绝隐式随机零状态window loss。29D/horizon不变。 |
| `visuotactile/models/vision.py` | 每forward重置GRU → `encode_frames`帧局部特征与跨块recurrence分离；双视图、depth/valid、任务embedding、合法自身计划仍严格allowlist；输出128D。 |
| `visuotactile/rollout.py` | 每6步窗口推理 → 每60Hz控制观察更新记忆、每6步重规划；每env独立reset、拒绝缺episode前缀；计划生成步/时间、计数和memory耗时。 |
| `visuotactile/models/tactile.py` | 每窗口重新逐指编码 → 持续5×32状态及上一个物理样本/时间/valid；跨块delta/onset/release连续，上一样本只作差分、不重复循环更新；融合64D不变。 |
| `visuotactile/models/correction.py` | 所有分支只窗口历史 → visual128/motion64/tactile/intent64持续state，另增跨控制步intent_memory；当前未来plan内部GRU仍逐plan总结。独立policy29D候选、未训练执行禁止不变。 |
| `visuotactile/correction.py` | Shadow重放重叠窗口 → memory2仅消费最新视觉/运动行及2个新物理样本，保存分支state并按env reset；memory1保留。 |
| `visuotactile/dataset.py` | 随机窗口、旧动作名 → 读committed length/显式clip-pre-delay；新增EpisodeSequenceDataset与encode_episode_chunks、CorrectionSequenceDataset与replay_correction_chunks，完整前缀/顺序校验；expert_history独立于Nominal inputs，future action标签分离。 |
| `visuotactile/train.py` | 默认随机window → 默认memory2、单episode batch、chunk8/TBPTT、每步condition监督；memory1显式保留。恢复从episode边界，不是exact mid-chunk resume；memory2 validation loss=null。本轮只维护接口，**没有运行optimizer**。 |
| `isaacsimenvs/tasks/play/utils/action_utils.py` | 无可读实际delay结果 → 原抽样后可选`_record_action_trace`，无hook时原路径保持；clone用于只读telemetry。 |
| `visuotactile/sim/environment.py` | 只raw/executed和targets → submitted/clipped/delayed分开、原delay slot/source step/effective lag/time、前后物理diag与时间；120Hz实际canonical q/FK/local force、pre-reset结果；metadata实际限位/基座方向/缺失声明。 |
| `visuotactile/collect.py` | 动作来源粗粒度 → expert/nominal/correction实际计算/执行标志、计划生成/覆盖/选择、Nominal参数hash/config、专家checkpoint hash及实际actor输入诊断、最新Correction hash/config；非terminal post观测保存、terminal mask；专家输入先于env.step冻结。 |
| `visuotactile/recording.py` | 整episode缓存后写盘 → 每native step append/flush、最后发布length；新增model_fingerprint、post_observation、EpisodeReader.read/steps与无覆盖PKL导出，保留原始arrays/attrs/version/metadata。非断电事务。 |
| `visuotactile/probe_history.py`（新增） | 真实trace审计、FK/目标公式/时钟/来源、225步full/chunk/online和Shadow复现、真实接触前缀梯度/权重+state重载、PKL字段往返；无optimizer/仿真启动。 |
| `visuotactile/probe_perception.py` | 明确memory1，保留之前视觉构图/有限窗口诊断，不混充memory2测试。 |
| `tests/test_causal_memory.py`（新增） | 8个测试补充远历史影响/梯度、逐控制步记忆/重规划/异步reset、触觉chunk边界、Shadow/direct、mask梯度、原helper动作/RNG、完整读写/顺序loss反传、版本/保存恢复。测试fixture不是研究训练数据。 |
| `.gitignore` | 增加PKL/pickle/partial，本轮轨迹/导出/模型/日志均排除公开提交。 |
| `research_docs/USER_RESEARCH_DIRECTION.md`（新增） | 独立保存用户方向/理由、开放问题、助手建议、Codex暂定决策、事实与追加变更记录；明确与cuRobo小物体项目无关。 |
| `research_docs/HISTORY_AND_TRACE.md`（新增） | 记忆/训练分块/信息边界、动作来源、频率公式、FK与特征语义、固定来源实现/许可、HDF5/PKL/Zarr/Parquet比较及实际验证。 |
| `research_docs/evidence/causal_history.json`（新增） | 经审计数值报告与原报告hash，无原始图像/轨迹/权重/私人文件。 |
| `AGENTS.md`, `README.md`, `research_docs/{README,ARCHITECTURE,CONTRACTS,DATA_SCHEMA,PERCEPTION,RUNBOOK,AUDIT,RESEARCH,CHANGELOG}.md` | 新方向入口和当前状态、独立memory/trace版本、字段/时序/单位/来源、命令与测试，旧window/训练证据明确历史；纠正“一概禁止专家过去动作”和“原源码全未改”表述。 |

#### 实测与限制

- 固定源码参考Robomimic的显式RNN state/step接口和HDF5，**未采用**其周期rnn_horizon reset；DP Zarr+episode_ends与finite-window sampler；LeRobot Parquet/video。具体commit/路径/许可见HISTORY_AND_TRACE，无新增vendored第三方代码/依赖。
- 真实渲染`causal_trace01/02`各2×240，03为2×32，04为2×8；场景健康断言始终开启。前两次迭代补齐时间/字段后保留原记录，不篡改历史文件。2的480行15专家字段与原有效capture精确相等；03/04也通过对应前缀比较。最终04专门验证最新模型hash与actor快照，不作为接触证据。
- `causal_checks01`：225步完整/分块/在线Nominal最大差4.66e−7，Correction分块/逐步0、与原在线4.47e−8；3个terminal图像明确缺失，FK/local force重算0；113字段PKL精确。03/04的114字段PKL精确；03延迟时间float64差5.55e−17s，04包含实际限位/目标公式检查。
- `causal_gradients01`：真实接触t37之前逐步burn-in；RGB/Depth两视图、Nominal temporal、Correction tactile/motion/intent/current-plan/head均有限非零梯度，输入force梯度非零；所有参数hash前后相同，权重+state重载输出精确。没有训练loss伪目标或参数更新。
- `causal_regression01`：原rendered与instrumented独立32步initial13/trace15字段逐位一致，包含RNG。PhysX已有kinematic setVelocity告警在原路径也出现；CPU无渲染场景仍不可用。**23 passed，4.52s**；requests既有依赖warning保留。
- 双核2env完整旁路median99.49ms，memory44.11ms、Correction48.75ms。原生60Hz执行+10Hz重规划不变，未机械缩放动作；这些是同步模拟时钟安排，墙钟速度不满足60Hz。
- 只对新进程用nice19/CPU22,23/两线程；未向后台任务发送信号、调其优先级或改GPU设置。渲染仍有GPU资源占用，不承诺零竞争。所有本轮仿真自然结束。
- 未解决：示范预加载去除、Correction目标/训练/控制权限、长期记忆的学习效用、实机异步时钟、终止前RGB-D、player裁剪前均值、实时期限。memory2训练只验证接口/反传，完整优化与validation loss尚未验证/接入；不得把旧checkpoint称作新结构能力证据。

本轮用户已授权提交与同步。公共提交只包含源码、测试、Markdown及小型数值JSON，保留MIT和已有vendor声明，使用origin/main且不强推。具体commit与推送结果在完成时报告。
