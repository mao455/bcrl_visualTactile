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

### 2026-10-28 — 原论文专家配对扰动采集（无参数训练）

从本机`6fc04a8`继续，建立`feature/paired-expert-trials`，保留先前迭代输出。用户本轮授权原tight_insertion expert的3对验收及20对pilot，未授权训练；Nominal/Correction均不运行。只修改研究采集/诊断与独立分析工具，原任务、资产、动作helper、RL及本机Lab源码本轮均未改。

| 文件 | 前后行为、职责、调用关系 |
| --- | --- |
| `visuotactile/collect.py` | 原固定tick采集 → 保留原模式，新增单env episode目标/外部预算、两个严格条件、pair ID；只将force_scale20→30。记录deterministic/group、源码hash/RNG、原timeout；实际step/完成/删失分别统计，success需全目标＋retract，保留all-goals计数原语义。metadata先于循环落盘；终止后不把reset帧保存成该episode示例图。 |
| `visuotactile/sim/environment.py` | 原obs/action/target/接触trace → 增加native请求force/torque、概率/质量/门控、原生终止原因、速度/base/palm/阶段与RNG诊断；增加object.write_data_to_sim只读observer，逐物理步记录实际local composer、object quaternion及world换算。request(1,3)与applied(2,1,3)分开；wrench_trace3独立于schema2/trace2，无学生输入维度改变。 |
| `visuotactile/trial_utils.py`（新增） | 记录base commit/branch/工作区及Python/YAML/TOML/shell源文件hash；原子更新独立实验计划/状态JSON，不读取凭据或其他项目。 |
| `visuotactile/run_rl_trials.py`（新增） | 先保存验收/pilot全部seed与顺序；复用run_cpu.sh/collect启动独立单env首episode；每条runtime/status/error/audit持久化。资源边界、墙钟限额只约束新child。允许检查后在新attempt目录同seed重试一次“无HDF5的启动SIGABRT”；不挑选任务成败结果。 |
| `visuotactile/analyze_rl_trials.py`（新增） | 从HDF5 committed rows审计真实字段、目标公式/延迟/因果边界、全RGB-D分块数值、local force/实际施力；支持缺字段旧尝试与纯专家无Nominal计划。episode分母、删失/运行错误分离、Wilson、计划上下界、初始配对、共同前缀位置/旋转/命令差、接触/tracking/外力统计与每对图；不硬编码结论。 |
| `tests/test_rl_trials.py`（新增） | 原helper下固定RNG验证尺度、torque不变、lift/retract门控、零decay及实际setter值；单位/统计/条件交替测试；启动重试不得覆盖部分HDF5测试。合成fixture仅测试边界，不是研究采样。 |
| `research_docs/RL_EXPERT_TRIALS.md`（新增） | 用户要求、工程选择、实际checkpoint/config、native成功/timeout、施力frame兼容问题、调用链、完整结果/限制与旧错误记录。 |
| `AGENTS.md`, `research_docs/{README,ARCHITECTURE,CONTRACTS,DATA_SCHEMA,RUNBOOK,USER_RESEARCH_DIRECTION,CHANGELOG}.md` | 指向本次实验合约，追加用户新阶段授权，说明新诊断维度/来源/时序、可复现采集/恢复/分析/PKL命令；不改写此前研究目标。 |

本机Lab外力composer缓存姿态造成request-world与实际world方向不同，实际220步诊断中峰值向量差5.60690N、范数一致。此行为本轮保持，仅新增观测；不能把这里的实验称为论文原安装栈的严格复现。详情见实验报告。旧一次分析曾因缺字段失败，旧“六条均未达到lift门控”解释被真实门控记录纠正为35/0/53个eligible步；旧数据/错误保留，分析不能以文档推测补齐字段。

最终cohort统计、所有实际验证与公开摘要见本条后续结果记录及`RL_EXPERT_TRIALS.md`；所有raw HDF5/PKL/weights/日志仍被忽略。

#### 最终结果与验证（本轮实际执行）

- `cohort_v3`完成验收6条＋pilot40条，7932控制步；全自然结束，34成功/12fall/0预算删失/0审计错误。验收两条件各2/3，pilot各15/20（Wilson95%53.1%–88.8%）；配对pilot15双方成功、5双方失败、0discordant。pilot保留2个启动SIGABRT错误，均无HDF5、同seed新attempt一次重试通过。
- 每条件4/20实际外力曝光，其余16保留分母。original13控制步、最大2.3642N；增强14步、最大3.5464N。只有seed910214在指接触时受力，两侧成功、206→215步，共同前缀最大位置差4.60cm；最大差异seed910217为65.62cm但两侧均fall、无指接触。不能推断抗扰动等价或触觉/记忆收益。
- 当前27项单元测试通过；32步原渲染回归initial13/trace15字段精确；220步observer前后128非图像字段精确，含RNG/奖励/接触。46条完整记录审计通过，所有图像/深度有限性与mask逐块检查；258步FK从q重算误差0；139字段真实v3 PKL往返及metadata精确。原生timeout分类单测，真实短smoke8步censored；本次没采到自然timeout。
- 代表性真实RGB已手工查看：固定场景正常，图像有RTX噪点，pre-action末帧没有冒充terminal。只有两张审计后的静态统计图加入`research_docs/media/rl_expert_pair_910214.png`及`rl_expert_pair_910217.png`；`research_docs/evidence/rl_expert_trials.json`保存数值/哈希，不含原RGB-D/权重/日志。
- 主HDF5共4,281,244,905bytes；pilot3,809,235,306bytes。local `cohort_v3/hdf5_manifest.json`含46条完整hash；采集/控制源码快照保存在本机供追溯，metadata有当时工作区hash。最终代码只进一步澄清注释和约束resume预算，不改变已采集物理路径。
- pilot控制循环墙钟总609.70s（不含Kit启动），每trial有效Hz中位11.75，专家前向中位4.13ms。保留60Hz模拟时间控制契约，不能称为60Hz真实时间性能；后台程序未被停止、发信号或修改优先级。
- 先前失败/旧尝试另列：6次预创建输出目录FileExistsError、1次诊断Kit启动SIGABRT、旧6episode验收、2条seed17诊断、1条8步censored。全都保留，重复seed不混入pilot；另一次旧分析KeyError保留修复说明。实验全部结果/限制见`RL_EXPERT_TRIALS.md`。
- 未改原MIT及vendor版权，无新增第三方代码/依赖。公开候选只有研究源码/测试/Markdown、约145KB结果JSON和两张静态图；HDF5/PKL/checkpoint/缓存/日志/虚拟环境均忽略。

## 2026-10-28 — 有界插入数据集、精确紧凑存储与零更新回放

用户授权仅tight_insertion公开expert，≤60尝试/10GiB，先验收再配对XY扫描，完整数据/少量视频/文档，不训练。起点7aed4d9，功能分支feature/bounded-insertion-dataset。完整报告[INSERTION_DATASET.md](INSERTION_DATASET.md)，状态[SESSION_HANDOFF.md](SESSION_HANDOFF.md)，可公开小证据`evidence/insertion_dataset.json`与`insertion_offsets.png`/`insertion_coverage.png`。

### 文件与前后行为

- `visuotactile/sim/environment.py`：原默认legacy保留；新增initial_offset_xy和world_hold。原reset后/目标前平移，仅XY、无额外RNG；可选每physics当前quat逆旋转原世界请求，公开composer body setter，未改共享Lab。记录offset原/最终/总偏差、模式。没有改动作/关节/碰撞/资产/判据。
- `visuotactile/collect.py`：新增offset/mode/experiment-condition CLI及来源字段；带实验条件时，在第一动作前调用新`initial_geometry.py`的URDF box OBB诊断；非法留metadata/FAILED不发动作。原采集路径不默认启用新实验协议。
- `visuotactile/run_rl_trials.py`：此前恢复可能仅凭退出状态复用；现fresh/resume共用validate_completion检查result/FAILED/length/审计/身份/结果/删失；bounded新记录要求wrench3完整。分析失败阻止后续stage。
- 新`visuotactile/run_insertion_dataset.py`：冻结4验收＋52位置条件计划，串行低优先级bounded运行，启动/完成状态、容量/RAM预算、配对检查、转换/审计/manifest发布/新临时源清理。明确非法初始组跳过，其他错误停止；没有重试补种子。
- 新`visuotactile/compact.py`：storage0→1保留原schema2。正式uint8 RGB(F,2,3,126,224)可逆恢复float32/255，float32 depth与bool mask(F,2,1,126,224)，post引用(T,)含−1缺失；预算保留T+1。三主文件H5＋双派生MP4，逐字段往返校验后发布；失败partial保留，旧源不改。
- `visuotactile/recording.py`、`dataset.py`、`analyze_rl_trials.py`改用EpisodeFile统一旧/新H5，原模型输入与监督形状不变；没有把观看MP4当正式RGB。
- 新`visuotactile/manifest.py`：三文件存在/bytes/SHA/版本/路径检查；success+scan默认Dataset、逐完整episode chunks，禁止静默混legacy/world_hold；pair_id分割断言、自身计划训练拒绝。失败仍在manifest，筛选不影响统计分母。
- 新`visuotactile/check_dataset.py`：独立CPU零更新检查，没有optimizer；old/new模型输入标签比较、完整顺序chunk7/step1/reset、有限diffusion前向、参数和buffer hash。新`storage_audit.py`测实际HDF5分配，混合RGB-D压缩chunk诚实合计。
- 新`visuotactile/analyze_insertion_dataset.py`：真实逐行审计、条件计数/CI、配对差异、实际wrench方向时刻阶段、覆盖/方向图、资源情景、最多4个拼接预览。非法初始/软件启动错误独立，当前只生成3例。
- `visuotactile/train.py`：单H5读取支持storage facade，正式manifest路径与纯专家缺self-plan情形增加预检拒绝；没有实现/运行正式manifest训练，不把旧smoke optimizer说成零更新。模型结构/损失未改。
- `visuotactile/contracts.py`：评估合约hash加入wrench_frame_mode/额外XY，防止相同resolved原config掩盖wrapper配置变化；旧metadata缺字段按legacy/零偏移解释。
- 新`tests/test_compact.py`、`test_insertion_dataset.py`：真实需要的格式/时序/损坏拒绝、共同完成门槛、OBB旋转接触/穿入、manifest分组/完整性/筛选/混配置拒绝等8项。与原27项合计35通过。
- 文档：新INSERTION_DATASET/SESSION_HANDOFF；更新根README、AGENTS、研究README、ARCHITECTURE、CONTRACTS、DATA_SCHEMA、RUNBOOK、RL_EXPERT_TRIALS、USER_RESEARCH_DIRECTION和本日志。记录用户新预算/停止条件/三文件偏好，修正旧数据质量解释而不覆盖历史。新增小汇总证据与图，未复制新第三方代码/修改许可。

### 真实结果与边界

- 20新尝试，17完整成功记录=4验收+13扫描，2原生非法初始化动作前停止，1libomniclient SIGSEGV(-11)未产HDF5，未重试。扫描仅1独立合法起点全部13条件，0任务失败/0删失、0原生外力/力矩曝光。另24非法组变体跳过、12因软件错误不启动。不能用13/13推断方向鲁棒或生产率。
- 旧46文件manifest SHA检查通过；pilot40H5实测3,809,235,306 bytes/7060步。新增box审计旧pilot20/40初始重叠（含全部10失败），旧75%不能直接解释为合法场景质量。保留全部旧源；失败转换例只验证存储，不验证物理场景合法。
- 旧成功258步/失败38步各139字段、预算8步112字段逐字段dtype/shape/value相同；预算9帧正确。FFV1真实像素无损对比已测，选择H5正式RGB+双观看MP4易随机切片但空间更大，明确记录代价。新17条均139字段精确往返，完整审计＋manifest SHA通过；总三主文件577,176,219 bytes，新目录约0.684GiB，已记录临时峰值0.750GiB。
- seed17 legacy220步16组数组与旧记录逐位相同。world_hold在首力前174步一致，力世界向量误差5.6069N→9.54e−7N；14实际力样本/7burst/0.1167s/峰值3.2243N，原阶段及方向记录完整。无非零力矩样本，不声称其动态回归充分。
- 旧258步零更新chunk7/step1最大差5.36e−7；新manifest两episode411步最大差5.07e−7，reset精确、参数buffer前后hash完全相同；有限预测(1,12,29)，parameter_updates=0。纯专家Correction/self-plan训练拒绝。训练CLI预检exit2且无目录/optimizer；RUNBOOK实际读取例通过。
- 三拼接视频解码/首中末检查完成，保留原颗粒/遮挡，末图缺失明示。旧失败例特别标注非法初始化，不混入新统计。当前没有合法失败例可补，不重跑凑样本。
- `bash visuotactile/run_cpu.sh -m pytest -q tests`：35 passed，既有requests依赖warning未动；`git diff --check`通过。原仿真/资产/RL未改，本轮全程CPU physics+render，未使用已知无效无渲染场景。后台任务未发信号/改优先级，共享环境未修改。

未完成：多起点统计、合法初始生产协议、Kit稳定性、正式manifest trainer/group normalization/memory2 validation、真实self-plan回放或无计划基线、Correction目标、去预加载。完整采集已按问题停止，不自动扩量或训练。

## 2026-10-28：三相机、动态外参、真实视觉验收与新400协议

用户明确允许通过相机/原专家回归/数据审计后启动新的400有效成功批次，覆盖旧批次的no400限制；原批次保持历史单列。本轮不训练，原task/physics/assets/RL没有改动。实际设计和逐项限制见[CAMERA_VALIDATION](CAMERA_VALIDATION.md)，小汇总[evidence/camera_validation.json](evidence/camera_validation.json)，ignored原始证据在camera_v3_20261028。

### 文件、调用与语义变化

- 新`camera_rigs.py`声明固定A/B及少量腕部候选；最终`wrist_bracket`固定局部pos(.12,.16,.07)，固定朝向(0,-.14,.24)，不动态追踪真值。`sim/environment.py`在现有USD中解析link7刚体，增加纯Camera child，不改变固定关节合并/碰撞；由原2路stack扩成按实际ID数量V路。
- 新动态`camera_pose_w (N,V,7)`来自每个相机/环境render product CameraParams，`camera_parent_pose_w (N,7)`用于诊断。之前CPU camera.data返回静态USD缓存，最初两候选被审计拒绝；现在逐帧检验位置/旋转/时间对齐。固定/局部安装/初始外参作用、K、裁剪、时钟、ID存metadata；Nominal allowlist不增加这些真值。
- `collect.py`新增rig/image-size/render-mode/render-aa/static-diagnostics可选参数，旧双路默认不变；shadow模型views由实际相机数配置。原专家权重、140Dteacher、动作队列/数学/随机化不动。`camera_diagnostics.py`记录真实USD/运行时render设置，做24帧静止原图/采样对照，0额外物理步。
- `recording.py`新增有效post动态标定，终止mask保持；`compact.py`增加storage2、按V生成视频和严格ID数量/位姿衔接，全字段无损验证；`run_rl_trials.py`completion gate接受storage2；`run_insertion_dataset.py`删除重复storage_version keyword，保留旧实验逻辑。`manifest.py`按声明视角检查文件完整，Dataset拒绝混布局/分辨率。
- `contracts.py`将rig3局部mount、实际K、V/ID/resolution/render配置纳入视觉checkpoint兼容检查，旧双权重拒绝；原RL checkpoint不受视觉契约影响。`check_dataset.py`按实际views初始化零更新模型。旧`probe_perception.py`分镜宽度/clock/FK投影改为实际V并用动态pose；它仍是旧memory1/probe数据入口，不作为新完整验证结果。
- 新`camera_review.py`：实际raw/compact统一读取、完整60Hz同步视频、每12步原RGB/深度/valid导出、128组非图像数组配对、FK+mesh-AABB光心距离、孔口投影＋深度诊断。后两项用真值作离线诊断，不提供给策略。新`check_camera_interfaces.py`：16步真实因果回放生成真正未训练Nominal计划，独立Correction保持Shadow，Jacobian与保存恢复检查，无optimizer。
- 新`collect_success_batch.py`复用现有collector/completion/convert，不重写物理或记录管线；新的冻结配置/seed序列、400成功计数、1600尝试/95GiB/40GiB空闲/2400步900s界限、完整attempt记录、错误停止、source hash冻结。初始化重叠仅动作前拒绝，不篡改高度/随机分布，不重试同seed。每条成功完整审计/转换/manifest持久化后才清理本条冗余temporary。
- 新`tests/test_camera_rigs.py`验证独立非单位矩阵row/OpenGL→ROS和局部光轴；`test_success_batch.py`拒绝把未知错误/已开始轨迹当初始重叠。扩展`test_perception_v2.py`三路两环境前向/每路梯度/恢复，`test_compact.py`2/3路终止和预算尾帧往返，`test_insertion_dataset.py`文件数契约断言。全tests最终41通过；此前一次旧错误文字断言失败已修正并重测，未掩盖失败。
- 文档：新CAMERA_VALIDATION；更新AGENTS、根/研究README、USER_RESEARCH_DIRECTION、SESSION_HANDOFF、DATA_SCHEMA、CONTRACTS、ARCHITECTURE、PERCEPTION、RUNBOOK和本日志。来源网页/CAD索引仅本地，没有复制新第三方源码或改变许可。

### 已测事实

- 本轮13完整专家验收轨迹（多次同seed方案/分辨率比较，不是13独立起点）：最终三种起点920302/17/920101共602步全部自然成功并通过审计；不作为成功率估计/不计400。另1个引入新接口时缺numpy import的reset前异常已修复；1个Kit启动SIGSEGV发生在旧双路baseline，明确记录并在独立目录重做该配对验收，未删除现场。前两个腕部候选的旧动态标定错误保留证据，不准进入正式数据。
- seed920302旧dual repeat和dual→三448的128数组逐位一致；seed17旧world_hold→新三448亦一致，包含实际非零力。重复基线差为0；physics120/control60和所有事件不变，墙钟较慢不解释为新控制频率。
- 143字段三路转换精确；168步顺序Dataset chunk7/step1最大4.917e−7、reset一致、参数buffer hash不变。16步三路真实Jacobian/各模态梯度非零有限，(1,29)Correction候选未执行，保存恢复精确。所有检查0参数更新，旧视觉checkpoint拒绝，不宣称学会感知/插入。
- quality三448主文件119–151MB/episode，双448同168步67.69MB、三448119.01MB（+75.8%）；单进程GPU快照2932→3001MiB。原始静止RGB已存在颗粒，quality显著降低；sampledLighting关闭请求未生效，16spp在performance未解决，不能把根因唯一归到某项。无RGB额外加噪、未降任务DR、未做后处理模糊。
- 光心到已测机器人mesh包围体>=.127m；并非实体支架/全动作包络认证。腕部真实补充部分搬运/对孔信息，但末段仍遮挡。400运行状态独立追加在交接文档，不能由gate已写推断已运行。

旧失败清理：核验旧cohort46文件SHA和完整结果后，仅删除12条已确认native failure/非法box起点的原H5，共337,127,269 bytes；保留完整统计/原因/metadata/小数组、原manifest与逐文件hash，新增旧root tombstones记录。旧成功和未知partial不动，预览保留。该清理有本轮用户明确授权，不把未知打不开文件当失败删掉。

### 同晚启动故障修复（第二个独立提交）

400第一次启动seed930000在Kit/libomniclient FileThread抛system_error Invalid argument，未创建metadata/HDF5。日志同时出现inotify ENOSPC，实查524205/524288 watch、磁盘201GiB可用；未改系统/IDE/他人进程。`collect.py`增加显式进程级`--disable-extension-watch`并记录实际Kit值；两次独立168步完整probe通过、无watch错误、128数组与原版及repeat一致。`collect_success_batch.py`增加bootstrap gate、检查watcher=false，支持pre-episode blocked前批作为可审计predecessor；前失败保留，从930001继续，不重新尝试930000，尝试总数含前失败，target仍400自然有效成功。unknown错误继续停止。修复状态和过程见CAMERA_VALIDATION；focused3tests通过。没有把这个选项当作已证明修复所有native崩溃。

### 正式批次启动后的真实首条检查

新watchoff driver PID1996734已经启动；原930000错误作为predecessor保留，首条s930001自然成功160步，四文件约107.65MB、143字段往返/完整审计/逐文件SHA通过。真实manifest顺序Dataset读出(1,2,3,4,252,448)图像与(2,12,29)监督，0更新。1/400时正在继续930002，状态并非最终完成。仅更新交接/公开小证据，未更改运行中的源码或推送数据；live入口见SESSION_HANDOFF。

### 首条之后的最终状态纠正（21:18 CST）

运行继续提交s930002/3后，s930004非法初始化被正确拒绝，s930005在Kit启动报LLVM out of memory/exit−6，无场景/H5，driver按错误停止。因此上文“正在运行1/400”为较早快照，**最新3/400且无运行进程**。三条438步/292,101,725bytes全部重验SHA。累计6尝试含2软件启动错误、1非法初始化；0合法任务失败。记录资源快照/cgroup无OOM事件，未下结论是GPU内存耗尽；不盲目重试，不删已有成功。更新交接、相机报告、小证据/AGENTS/RUNBOOK当前状态，没有更改冻结研究代码。
# 2026-10-28 — version-4 close-camera audit and bounded collection guard

## Changed

- `visuotactile/camera_rigs.py`: added independent `wrist_bracket_close`
  layout version 4; only the first fixed camera moves to
  `(0.288,-0.405,1.463)`. Fixed a dispatch bug so the close name still creates
  the link7 wrist camera. `environment.py`, `contracts.py` and the analysis
  path accept layout versions >=3 without changing physics or control.
- `visuotactile/compact.py` and `manifest.py`: MP4s are explicit optional
  viewing derivatives; HDF5 remains lossless source-of-truth. Existing
  video-bearing manifests and tests remain compatible. `tests/test_compact.py`
  adds a no-video HDF5 roundtrip check.
- `visuotactile/collect_success_batch.py`: added 50% conservative whole-card
  GPU telemetry, rolling/batch means, six-hour wall guard, decimal 60 GB guard,
  inter-episode throttling and nonempty-predecessor preservation. Production
  conversion defaults to no MP4 copies.
- Documentation: updated current three-view entry in `ARCHITECTURE.md`,
  `DATA_SCHEMA.md`, `HISTORY_AND_TRACE.md`, `PERCEPTION.md`, `RUNBOOK.md` and
  `CAMERA_VALIDATION.md`; added `INITIALIZATION_AND_PERTURBATIONS.md` and
  `OUTPUT_INVENTORY_20261028.md`; AGENTS now points at the new resource and
  startup constraints.
- Review artifacts: a Chinese-labeled three-column demo, camera overview,
  URDF fixed-chain schematic and source/pose audit are under
  `outputs/visuotactile/camera_v3_20261028/review_artifacts/`.

## Actual verification

- `bash visuotactile/run_cpu.sh -m pytest -q tests/test_compact.py
  tests/test_insertion_dataset.py tests/test_perception_v2.py
  tests/test_visuotactile.py`: **26 passed** (one existing requests warning);
  the full `tests` suite after the optional-video and calibration checks is
  **42 passed**.
- Existing `bracket_17`, `bracket_920101` and `bracket_448` synchronized videos
  were decoded and matched to HDF5 view order; fixed-high pose delta was 0,
  wrist pose delta was 0.180–0.323 m, frames and timestamps were aligned.
- Version-4 probes: 930006/008/009 invalid before action; 930007 completed but
  had only two cameras and is rejected; 930010 hit LLVM OOM at startup. **No
  version-4 triple-view success and no 400 batch** are claimed.
- All-output inventory after authorized candidate cleanup is
  10,074,898,432 bytes; 3,037,673,433 bytes of redundant candidate payloads
  were deleted with a hash ledger. No training, optimizer update, background
  process signal or shared simulator change occurred.
