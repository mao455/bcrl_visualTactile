# 当前会话交接：有限插入数据集（2026-10-28）

> **2026-10-29夜间最新入口：** [NIGHT_COLLECTION.md](NIGHT_COLLECTION.md)。用户覆盖旧预算/截止/停止规则；持久四环境＋8m间距正在向400采集。原1.2m间距试采因邻居桌子入镜排除。实时数量与PID用新状态命令读取，下面旧快照不可当作当前状态。

> **最终覆盖：2026-10-29 00:05北京时间**：新v4生产已在2/400停止，PID2070793已退出。s930014/016已完整提交；s930017启动-6、libomniclient FileThread Invalid argument。额外一次受限strace直接看到inotify_add_watch ENOSPC，当前524205/524288 watch。关闭扩展watch仍不足，勿自动重启或把下文运行快照当当前。0训练，未改系统配额/其他进程。

> **2026-10-28 最新纠正：** v4近高位三视角已完成seed17/920101两条真实验收（220/214步），与各自v3参考128个非图像字段逐位相同；143字段无损转换和三路模型接口检查通过，零训练。此前“零v4成功/资源保护已完整”的文字为当时快照：上一提交的容量统计仅单root、v4仍有两处版本漏判，本轮已修正。实际背景批次看最新SESSION_HANDOFF/status；不要把下面历史状态当实时。源码通俗说明见[PROJECT_GUIDE_ZH.md](PROJECT_GUIDE_ZH.md)。


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

## Version-4 close-camera follow-up (2026-10-28 late evening)

The only intended geometric change for the next production profile is the
version-4 `wrist_bracket_close` calibration: `fixed_high` eye
`(0.288,-0.405,1.463)` with the old target; `fixed_side`, link7 wrist mount,
448×252 and rendering settings are unchanged. Existing v3 episodes remain
separate and were not relabelled.

Probe accounting is explicit: s930006, s930008 and s930009 were native
pre-action box-overlap rejections; s930007 completed but had only two views
because the new rig name was initially absent from the wrist dispatch (the bug
was fixed, and the payload is not eligible); s930010 stopped at startup with
`LLVM ERROR: out of memory` before a scene/metadata/HDF5. Therefore there are
zero version-4 triple-view successes and no new production driver. The small
ledger is `outputs/visuotactile/camera_v4_close_20261028/version4_probe_audit.json`.

The collection runner now has a decimal 60 GB root guard, six-hour wall budget,
whole-card conservative GPU samples with rolling/batch means and
inter-episode throttling. No further automatic launch is justified until the
native startup blocker is independently bounded and a fresh v4 gate is
created. The old three successes and all error classifications remain intact.

## 当前真实状态：v4两条验收通过，预算内生产已后台启动（2026-10-28续）

本段覆盖旧“零v4成功/没有driver”快照。实现commit **e546bca**已推送origin/main。新driver **PID2070793**，通过start_new_session及日志文件启动，stdin为DEVNULL；不会依赖交互PTY继续存在。输出`outputs/visuotactile/insertion_close400_20261028`；launcher日志`outputs/visuotactile/camera_v4_close_20261028/production_launcher.log`；实时进程/资源/成功数量分别看root下run_status/status.json、resources.json、manifest.json。截止20:51:19Z（北京时间次日04:51:19），从第一轮14:51:19Z计6小时，换目录不延期。

验收seed17/920101为2条v4成功，独立于目标400生产数；旧v3三条保留且重hash/audit，不合并标定。930006–010所有旧错误分类导入新manifest。新seed930011–013原生非法、动作前拒绝后继续；**首条正式s930014已提交**，244步自然成功，HDF5163,059,257bytes、143字段精确往返。独立ManifestReader verify=True、ManifestSequenceDataset `(1,1,3,4,252,448)`、完整validate_completion均通过，audit_errors=[]。这是实际后台新增episode，不是仅检查PID。其后仍运行，请读取实时文件，勿重复启动。

两条验收从启动到转换审计完成实际墙钟205.25/223.20秒，包含中间开发等待；执行阶段合计136.45/135.29秒。GPU监测只覆盖各执行阶段，不能把空档伪造为全周期测量；生产采用单Supervisor贯穿全部阶段。首条生产完整周期123.29秒；前3个非法尝试约19–21秒。400在当前独立进程+精确转换实现下无法保证6小时，甚至忽略全部失败也明显不足；剩余预算尽量采，不暗延长24h。进程复用可省启动，但不消除写盘/精确转换，且需要新的reset/RNG/相机缓存验证，未擅自接入。

全项目清理/真实数字见OUTPUT_INVENTORY；旧普通MP4已显式变为可选，data.h5不变。两个v4完整原始字段分别在`paired17_compact/data.h5`和`paired920101_compact/data.h5`，duplicate raw已经验证后清理。旧bracket17等同理；旧路径metadata与demo保留，读取原始字段改用对应compact。

仍未解决：LLVM根因（arena2两次成功不构成修复证明）、全部硬件安装、广初态下可见性、去预加载、Correction监督与正式manifest训练。未知新软件/审计错误停止并保留现场；普通失败/非法继续。该轮0训练，无其他任务信号/驱动/共享Lab改动。

## 最终停止与诊断证据（2026-10-29 00:05）

新生产7次尝试：s930011/012/013/015非法起点；s930014=244步自然成功、163,059,257bytes；s930016=226步自然成功、145,407,980bytes；s930017启动软件错误(-6)、未产生metadata/HDF5。2/400，缺398；两条相机验收和旧v3三条均单列。driver2070793退出，当前无本项目采集后台可等待；launcher日志/status/manifest均保留。两条再次hash验证、完整审计、Dataset三路读取通过，证据run_status/final_readback.json。

批次349.49秒连续监测整卡均值9.48%，最大已满60秒滚动21.87%；最大采样间隔1.83秒，无缺测；整卡显存峰值5261MiB，项目outputs峰值9,495,693,267bytes。C+G pmon部分SM值缺失，不能给可靠独占RTX百分比。停止时outputs约8.866GB，外部明确归属的Kit/Lab日志约.021GB另计，仍远低60GB。此前验收监测峰值11.692GB，最初盘点约13.10GB；不声称覆盖未监测间隔的精确峰值。

本次错误与LLVM不同：std::system_error Invalid argument，native栈在libomniclient FileThread。只读/proc统计watch524205/524288，余83；instances79/4096。进行**唯一一次**同seed、1步上限的strace诊断，捕获真实inotify_add_watch ENOSPC（见startup_inotify.trace/startup_trace_audit.json），即使`--disable-extension-watch`仍有底层监听失败；该次44.50秒、整卡均值13.22%，仅censored诊断，不计生产成功。未修改limit、IDE、驱动、共享库；没有继续无上限试错。先前LLVM原因仍未唯一确定，arena2不构成稳定性修复。恢复需进程内经验证方案或外部watch压力变化，不能假称400仍在跑。

容量/时间预测：两条验收活跃周期约135–136秒；新首条生产123秒，另有19–21秒非法尝试开销。忽略失败的400条就约13.7–15.2小时，因此当前实现不满足6小时完成400；进程复用即使消除启动仍不足，且需reset/RNG/cache对照，不在此偷偷引入。2条生产平均154.23MB，机械外推400约61.69GB，还未加保留证据，超过60GB；样本小不能作为稳定最终容量，但足以说明不能承诺40GB。停止现状没有可信400完成ETA。

## 2026-10-29 02:02 持久生产交接快照

最新实现与命令见NIGHT_COLLECTION。当前已核验的正式数量约87条（之后继续增长）；supervisor PID2117854，Isaac采集 PID2117864，session r931005_1793209376。不要按快照认定仍存活，先运行status。生产目录data/close_v4、manifest.json；日志run/close400/r931005_1793209376.log；progress/resources/statistics同目录。

一个持久CPU实例、四环境、标准三路相机、8m环境间距、4CPU仿真线程＋两个独立CPU审核进程。旧s930014/016计入，原1.2m的8条有邻居入镜已排除。样本均通过143字段精确重建和完整动作/时间/状态审核。四环境试采约2.47条/分钟，长批已约4.26/min；当前GPU批次均值约30.34%，最大完整60秒仍低50%，仅自身进程组按均值节流过12秒。

不能用此快照声称已经400；真正完成以manifest400条＋status complete＋对应文件审核为准。相机保留原v4，lower-close仅诊断不混入。输出只剩data/demos/run/tmp，路径迁移索引run/path_relocations.json。常见本地问题已修复，无正式训练，无其他项目进程或共享设置操作。

### 02:08真实交付快照（覆盖上段数量）

`80e703f9e6180459bec668ce83321beb858f930a`已推送origin/main，工作树干净；逐项对比持久进程记录的源码hash，当前10个相关运行文件均与此公开提交一致。实际112/400，剩288；文件存在/大小、每条143字段审核记录均核对。新长批106成功、88非法起点、3自然失败、4进行中、2等待审核；另外计入旧2和本轮4条8m验收，不计旧v3或1.2m试采。

supervisor2117854和collector2117864均实查存活，长批含启动吞吐4.18/min，循环4.23/min，预计剩余约68分钟，非完成承诺。GPU全批约30.8%，最大完整60秒46.22%，无缺失采样；磁盘余量约166GB。实时状态命令与日志入口见NIGHT_COLLECTION，当前采集继续后台，不因本轮答复退出。详细公开小统计见evidence/night_collection_20261029.json；所有数据/权重/日志未上传。

## 最终交付：close_v4 400条（2026-10-29）

实时核对后，`outputs/visuotactile/data/close_v4/manifest.json` 已有400/400条 `stage=production` 成功；`progress.json`/`status.json` 为 `complete`，supervisor和Isaac采集器均已退出。完成时间由状态记录为新加坡时间2026-10-29 03:26:58，早于原08:00目标。不要重新启动采集。

最终验收：400个episode ID和HDF5文件唯一；全量ManifestReader(verify=True) SHA256/大小通过；400条均为三路相机、143字段、无audit error/缺失telemetry；400条content_digest已补齐且唯一。`ManifestSequenceDataset(..., stage="production")`抽查5条不同环境、长度和外力曝光样本通过。400条 `length` 合计81,894个episode控制步；progress中的21,271是四环境批处理tick。`tmp/`清空。生产HDF5合计70,186,972,525 bytes，项目输出结束快照76,732,681,445 bytes。

当前统计和入口：`close_v4/initialization_report/{report.json,initial_states.csv}`为400条生产统计，早期相机验收保存在同目录`preproduction_*`；详细分类、控制步、外力/扭矩和GPU统计见`research_docs/NIGHT_COLLECTION.md`及`run/close400/statistics/summary.json`。正式数据入口仍是`close_v4/manifest.json`，读取必须显式`stage="production"`。1.2m邻居入镜8条、失败/删失/错误均不计生产清单。

资源记录为全卡均值29.41%、最后完整60秒34.66%、GPU显存峰值12,356MiB、缺测0，结束时记录磁盘可用108.74GB；当前系统`df`约76GB。全程保持四环境/8m/v4三相机、CPU专家与物理、GPU渲染、120/60Hz，零训练且未操作其他后台任务。
# 2026-10-29 — B0/B1 training handoff

Formal training was prepared from the completed `close_v4` production
manifest.  The implementation is `visuotactile/train_dp.py`; B0 is
`NominalDiffusionPolicy` and B1 is the independent
`TactileCommandDiffusionPolicy`.  Both use the fixed 320/40/40 episode split,
memory2 ordered replay, H=12 and 6-step replanning semantics, and
`transition/command_clipped_pre_delay` as the only supervision label.  B1
consumes exactly two 120 Hz contact samples per 60 Hz control step.  Expert
commands are never conditions; no nominalization or Correction training is
performed.

Preflight on the real HDF5 data passed: manifest split `[320,40,40]`, one
production chunk per variant, finite forward/backward and one CUDA optimizer
step.  The repository regression suite remains 50 passed.  A B0 chunk of 32
steps took about 3.9 s including first-pass startup and 6.8 GB peak allocated
on the 24 GB RTX 4090 D; subsequent chunks were about 0.04 s.  B1 had the
same 6.8 GB peak and about 3.8 s for the first 148-step smoke epoch.  These
are throughput probes, not training results.

The user requested both jobs to run in the background.  Launch only after
checking `nvidia-smi` and the protected workload; write B0/B1 logs and
checkpoints under separate ignored directories:

* `outputs/visuotactile/run/dp_b0_20261028/`
* `outputs/visuotactile/run/dp_b1_20261028/`

Record the actual PIDs, start time, epoch progress and ETA here after launch.
Do not upload checkpoints, logs or data.  B2/B3 remain design-only and no
closed-loop evaluation is authorized until the reports are inspected.
