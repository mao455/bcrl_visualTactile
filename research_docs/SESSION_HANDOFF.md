# 当前会话交接：有限插入数据集（2026-10-28）

## 用户授权与停止条件

本轮只运行公开 tight_insertion RL expert，零参数更新。最多60次新episode尝试（验收≤8、位置扫描≤52），新产物和临时转换合计≤10 GiB。保留旧数据，原生120Hz物理/60Hz控制；不改资产、碰撞、增益、摩擦、成功判据和共享依赖。保护后台任务，只用run_cpu.sh约束自己的进程。关键数据/同步/容量错误停止后续受影响采集；每次≤2400控制步/480s，不无限重试。完成本轮即停止，不自动采400/1200或训练。

## 起点（已核实）

- 工作区 `/home/CNS2025331827/docs/bc+rl_RSS`，干净main，HEAD `7aed4d9e3cc9e2af0e747a486da6b865a4907cc5`；新分支 `feature/bounded-insertion-dataset`。
- origin为mao455/bcrl_visualTactile；upstream为kushal2000/play2perfect。只向origin同步已审计源码/文档，不公开数据。
- 旧主数据：`outputs/visuotactile/rl_experiment_20261028/cohort_v3/`；manifest与真实存储正在重新核验。旧结果不充当本轮结果。
- checkpoint：`pretrained_assembly/tight_insertion/model.pth`；预期SHA256 `7e61015dad660430ab07bab4f715c90c6f76b84cc358da6be81319a6aee09990`，本轮将重算。
- 本轮输出根：`outputs/visuotactile/insertion_dataset_20261028/`。启动时新episode=0；预算与不可变计划将在run_status记录。
- 安装栈沿用项目overlay、Lab2.3.2本地fork、Sim5.1；CPU必须渲染。启动时4090D已有约68%利用率/1932MiB显存，19GiB可用RAM、248GiB磁盘；不操作后台PID。

## 实施顺序与尚未验证

1. 审计原训练/评估/当前采集配置、原有HDF5字段实际压缩占用和checkpoint。
2. 用旧成功/失败/预算尾段验证无损紧凑记录、双视频、末post额外帧与统一reader；建立完成验收及预算门槛。
3. 仅在局部可证明时加入opt-in施力frame兼容修复；否则停止方向力试验，位置实验不受影响。
4. 冻结13条件×4新seed配对初始偏移；少量验收后有限采集，保留每次结果/错误。
5. 逐字段往返、零更新网络检查、方向/覆盖统计、最多4个双视角预览，文档与Git同步。

本文件开始记录目标，不代表这些阶段已经完成。结束时追加实际执行、证据、预算消耗与停止位置。

## 中途核验（同日，最终统计待本文件末尾）

- checkpoint SHA256已重算且与上值一致，217,700,467 bytes；没有假定存在pretrained config.yaml，配置来自原YAML和实际resolved metadata。
- 旧cohort_v3的46份manifest源文件校验通过。40份pilot HDF5实际3,809,235,306 bytes / 7060步（3.548 GiB），30成功/10失败；原文件未改。
- 新storage_version1：一episode三主文件（data.h5正式无损RGB/float32 depth/数值；双MP4观看派生）。旧成功258步、失败38步、预算中止8步逐字段精确转换；预算样本保留第9张有效末post图。已比较FFV1逐像素无损方案，最终选HDF5随机读取简化方案，承认观看RGB副本额外空间。
- 4次验收全部成功：seed17 legacy/world_hold各220步；seed920101 baseline214步/+y5cm219步。legacy与旧同seed16组数组逐位相同；world_hold实际世界施力误差从5.6069N降至9.54e-7N。仅研究wrapper可选修复，不改共享Lab。
- 258步旧/新Dataset输入与标签完全相同；固定参数分块7/逐步1最大差5.36e-7，reset重复完全相同，参数和buffer hash不变，parameter_updates=0。纯专家记录缺Nominal计划，Correction训练入口拒绝。
- 扫描固定52项/4个起点；seed920301零偏移原生初始化即物体盒穿桌，动作前拒绝，保留temporary/元数据/FAILED；该组剩余12项跳过，不换seed。按明确规则只继续其余原定组，其他错误停止整批。不能把此项作为机器人任务失败或已执行轨迹。
- 证据统一在 `outputs/visuotactile/insertion_dataset_20261028/` 的storage_audit、conversion_checks、video_comparison_measured、zero_update_conversion和run_status/acceptance_gate JSON。扫描仍在运行，不能用中途样本推断最终成功率。

## 最终交接：本轮已停止，勿自动续跑

- 20次新尝试：4验收全部成功；扫描16次尝试产生13成功记录，2个非法初始动作前拒绝，1个Kit/libomniclient SIGSEGV(-11)启动失败。新任务失败0、预算尾0；17条记录3462控制步。扫描只完成seed920302全部13条件，其他组没有有效轨迹。seed920303穿孔座，seed920304没有metadata/H5；没有重试或换seed。扫描24项因非法组跳过、12因崩溃不启动。
- run_status/final_stop.json记录停止/预算；manifest.json提交17条三文件记录及SHA256。episodes主文件577,176,219 bytes，整个新root约0.684GiB，已记录转换前峰值805,171,515 bytes，远低10GiB；不是连续监控的精确peak。旧源保留。
- analysis/report.json及两个PNG是最终统计；扫描13/13成功但仅1起点/每格n=1，全部零外力/力矩曝光，不能据此断言方向鲁棒或生产成功率。验收world_hold14力样本单列，未与扫描混算。
- **新发现**：旧pilot40记录20条初始box重叠（10旧成功＋全部10旧失败）；旧acceptance6条2重叠。old_initial_geometry_audit.json留证，旧75%不能直接用于合法起点质量/扩量预测。未改旧源与旧任务标记。
- previews只3个双相机1×60fps完整视频：s920302_baseline、s920302_yminus05cm、old_failure。最后一个明确旧seed910101非法穿桌/原fall诊断，不算本轮任务失败。已看首中末画面：场景与动作可见，粒状RGB/手遮挡仍明显，没有宣称精细视觉能力。
- 真实新411步manifest回放、单H5读取一致，跨块/逐步最大差5.066e−7，跨episode reset精确；旧258步转换输入标签精确。参数+buffer hash不变，0更新；正式train预检拒绝manifest/纯专家self-plan误用，无输出目录/optimizer。35项测试通过，旧requests warning未修改环境。
- 原控制/资产/RL文件未动；施力改动只研究wrapper opt-in，默认legacy220步16组数组逐位回归。源码base commit+逐文件hash写入每条metadata，采集时辅助工具/文档仍在完善，最终Git commit见本轮提交历史。期间无共享依赖变更，未向背景任务发信号或修改优先级。
- 新增/修改完整清单、I/O、证据与源码链在CHANGELOG、DATA_SCHEMA、INSERTION_DATASET。公开仅源码、文档、汇总证据/统计图；数据、视频、日志、checkpoint在ignored outputs/pretrained。

### 下一步需要后续决定，当前不执行

先查看3视频，决定是否另立“合法初始化成功示范”协议；随机原生姿态有穿入问题，不能擅自修改初始高度/姿态/物理并混算。Kit启动SIGSEGV需要独立稳定性排查，不能用无上限重试掩盖。未来400成功预算按25/50/75%情景约1600/800/534合法尝试、49.54/24.77/16.53GiB、20.01/10.00/6.68小时（不含非法/崩溃/最后分析）；目前单一起点不足以选真实成功率。

正式训练前还要选择无自身计划baseline或因果模型回放、按pair_id划分、多episode训练统计/绑定、memory2验证loss。Correction目标与去预加载未解决；不运行任何参数训练，也不自动继续400/1200采集。

## 新阶段开始：三视角验证→400成功（2026-10-28）

当前用户明确更新授权：完成相机/专家回归/数据审计/资源检查后可直接启动新的400成功批次；旧停止规则仅限旧批次。起点f39c968，原工作树干净，新分支feature/triple-camera-validation；本项目无采集进程，外部GPU生产任务在运行，绝不操作。新证据目录outputs/visuotactile/camera_v3_20261028。候选补丁未发现，不假定已应用。实际完成结果将在后续追加；当前只启动一个腕部候选完整专家probe，不是400批次。

## 三视角验收完成，准备启动新400协议

读CAMERA_VALIDATION及evidence/camera_validation.json。本轮13条完整专家probe：其中最终448三视角三个合法起点920302/17/920101，共602步、3成功；其余是同seed相机/渲染/分辨率比较，不能算独立成功率或400生产数据。初期2条动态校准错误的数据仅诊断；1次缺numpy import异常已修复，1次Kit startup SIGSEGV仍留现场（配对验收在新目录重做）。41 tests通过；128数组配对完全相同；143字段无损转换；三视角真实顺序/梯度/恢复0更新。

冻结生产方案wrist_bracket、448×252、quality/AA Off、原公开expert/SAPG group0、world_hold、单env、120Hz physics/60Hz control。gate路径 `outputs/visuotactile/camera_v3_20261028/camera_gate.json`；计划新root `outputs/visuotactile/insertion_triple400_20261028`，driver `collect_success_batch`。目标400有效自然成功；seed930000起，原生初始化+零额外偏移，逐条拒绝非法起点并保留原因；最多1600尝试/95GiB，空闲40GiB底线+12GiB尝试预留。失败/删失完整保留；任意软件/审计异常停止、不自动重试，原批次不续跑。下文将追加真正启动状态，本文段本身不表示已经启动。

旧失败清理已按用户授权执行：cohort_v3旧manifest中12条已完整读取、SHA核验、确认native failure且初始化box重叠的H5已删除，共337,127,269 bytes。每条metadata、失败原因、全部统计/小状态数组及原hash保存在新诊断root/old_failure_archive；旧cohort根新增hdf5_cleanup_tombstones_20261028.json。旧manifest作为历史原样保留，不再可用于断言46文件仍全在；全部成功源、未知损坏/未完成/软件错误现场及旧失败预览保留。没有清理其他项目或后台任务文件。

### 首次400启动失败与已验证的后续启动配置

原生产root `insertion_triple400_20261028`：PID1990175/child1990190均已退出，seed930000在Kit初始化失败，0成功/0HDF5，manifest保留1条software_or_audit_error。libomniclient FileThread Invalid argument伴inotify ENOSPC，当前用户524205/524288 watch，磁盘201GiB可用。未动系统额度/IDE/后台。新增仅本采集进程关闭Kit扩展热重载，两次完整168步通过、无watch错误、128数组配对/repeat一致。

准备采用新gate `camera_v3_20261028/camera_gate_watchoff.json`、新root `insertion_triple400_watchoff_20261028`，将旧blocked root作为predecessor；保留前错误，从930001继续，不重试930000。必须区分下面新root的真实运行状态，旧root仍blocked。15条完整验收probe不计400。

## 当前真实交接状态：新400正在运行（2026-10-28 21:14 CST快照）

- 新root：`/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/insertion_triple400_watchoff_20261028`。
- driver PID **1996734**，低优先级nice19/核22,23；launcher日志 `outputs/visuotactile/camera_v3_20261028/production_watchoff_launcher.log`；live状态 `<新root>/run_status/status.json`；计划 `<新root>/run_status/plan.json`；manifest `<新root>/manifest.json`。不要重复启动driver，先核对PID/status。
- **首条正式记录s930001已经完整发布**：160控制步，原生自然成功，非零原生world_hold力4控制步/8物理样本、峰值约2.60N，全部数据审计通过。143字段精确往返，三份视频各160帧，H5+视频107,645,500 bytes（约102.66MiB）；data.h5=106,934,389 bytes。目录 `<新root>/episodes/s930001`，第一个SHA为975fdf8ab2ded51848667d4c147a9a67e100fd3523d426a9473706a8ba2f0d42。
- 已用ManifestReader(verify=True)重新验证四文件SHA，并经ManifestSequenceDataset(stage='production')读出 `(1,2,3,4,252,448)`真实图像chunk，动作监督 `(2,12,29)`。未训练。证据 `camera_v3_20261028/first_production_check.json`。
- 此快照为1/400有效成功；尝试计数含前批930000的启动错误，930002正在运行。**这不是最终400完成报告**；driver会继续，达到400、容量/尝试上限或软件/审计错误时停止。后续读取最新status/manifest，不能把本静态数字当实时状态。
- camera/接口提交`37fb6e1`，bootstrap修复`6dfc20f330331970275de55d630c4075c5acb928`均已成功推送origin/main。源码hash在运行plan和每条metadata；文档/提交状态快照可继续更新，不动冻结源码。数据/视频/权重/日志都在ignored目录，未公开。
- 后续核心未决：实体KUKA左手相机支架/外壳/线缆；更广初态下可见性/合法初始化覆盖；Kit长期稳定性；正式BC自预测历史与manifest训练；Correction学习目标与示范去预加载。不要用这次接口/渲染验收当作已经学到任务。

## 最新状态覆盖前述运行快照：3/400，已停止（2026-10-28 21:18 CST）

**目前没有本项目采集driver/child继续运行。** 新root `insertion_triple400_watchoff_20261028` 的status已为blocked，PID1996734已退出。6个seed尝试（含predecessor）：930000原Kit启动失败；930001/2/3正式成功；930004原生box初始化非法、动作前拒绝；930005新启动错误。合法任务失败0，不能把启动错误计作任务失败，也不能把3条说成完成400。

新阻断的确切日志：`logs/s930005.log`只有启动信息后`LLVM ERROR: out of memory`，returncode−6，没有metadata/HDF5/场景，也没有之前的change-watch ENOSPC。热重载修复只解决该类watch错误，未证明Kit所有启动可靠。失败后只读资源：RAM available约18.4GiB、swap几乎满、VRAM free约22.7GiB；当前cgroup memory.max=max且oom/oom_kill=0。因此不能把LLVM错误直接解释为GPU显存满或内核OOM kill，也不能把它继续归因于inotify。未调整系统、IDE、后台进程或共享Lab，未盲目重试/补seed。

三条正式成功都再次通过ManifestReader verify=True的四文件SHA检查：s930001=160步/107,645,500 bytes，s930002=163步/107,838,360 bytes，s930003=115步/76,617,865 bytes；合计438步、292,101,725 bytes，3/400。原始成功/失败现场完整保留。`run_status/blocker_llvm.json`记录错误、尝试分母和资源快照；launcher/status/manifest入口沿用上文。已有至少一条正式三视角episode可阅读，但没有运行中的批次可以等待到400。

下一步先定位本机LLVM/Kit启动时分配失败并做有依据的启动对照，再设计**保留这3条和全部attempt的恢复**。当前predecessor恢复只支持0episode，不可删除正式数据绕过guard。用户400授权仍有效；技术阻断不等于需要重新询问是否允许采400。不要以调expert、改物理或无限启动重试掩盖问题。
