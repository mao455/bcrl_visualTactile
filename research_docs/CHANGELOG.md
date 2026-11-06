# 2026-11-06：StateDP机制诊断完成复核

- 真实A5/B2/C8全部完成，tiny800拟合通过但闭环0/2；C0/C4各0/4。补充初态精确可观测匹配、首块对齐及后续误差放大证据；保留隐藏状态不确定性。
- tools/state_mechanism.py仅增加学生/冻结专家独立来源hash字段；历史checkpoint_sha256为expert/shadow，未覆盖原结果或更改控制。无新GPU/Kit/optimizer更新。
- STATE_DP_RESULTS、SESSION_HANDOFF、AGENTS与同一轻量evidence/state_mechanism.json原位更新；保护hash、tiny模型/AdamW/EMA800读回及所有worker退出通过。整卡27.47%均值/52.11%最大完整60s，不能声称全程≤50%。既有17测试/真实梯度门控保持；本次来源字段修复做编译检查。
- 运行源码3e4d679，自动结果6d123e7；原数据、正式模型、tiny best/last和8视频全部保留。下一建议仅闭环误差放大定位，不启动新研究。

# 维护变更

<!-- STATE_MECHANISM_START -->
## 2026-11-06：StateDP基础能力、采样与动作块诊断

实测 `complete` / `complete`；独立root `state_mechanism_20261106_01`，固定4h含等待，最多15次Kit轨迹启动、1条训练示范的独立随机模型≤4000更新/90min。正式数据/权重保持，A/B/C不改变H24/12/29D/损失/历史/物理。
只读审查275条/64,663tick：全体均值3.919s、中位3.433s、p10–p90 2.640–5.517s，范围2.117–12.900s。新增75均值4.676s、中位4.4s；尚不符合多数5–7s的设计预期，没有补等待或改时间戳。State-valid275、Multi-view-valid75、旧固定视角无效200；触觉数值有限及有效标记通过，不等于硬件校准。所有成功标签只在末transition出现，275条原始数值与缓存完全相等，q_after[t]=q_before[t+1]。

极小集离线拟合通过但对应初态仍未成功；精准目标回放接口检查已通过，优先研究执行后状态偏离teacher-forced训练轨迹这一机制。下一轮只做有限闭环分布偏移定位；本轮不自动DAgger或改方法。噪声/块跳变的因果归属仍受小样本限制。

实际A/B/C、采样曝光及最新状态见[StateDP统一结果](STATE_DP_RESULTS.md)和[机器统计](evidence/state_mechanism.json)。
<!-- STATE_MECHANISM_END -->

## 2026-11-06 有界机制诊断实现

新增tools/state_mechanism及audit/train/delivery普通后台入口，固定4h/15starts；只读275条全tick/自然成功/标签/缓存/split/hash审查和实际sampler状态曝光。B采用隔离的多窗口采样，正式AnchorSampler不改。state_benchmark仅增加显式replay_horizon=24的诊断Oracle入口，普通调用默认不变；目标分块不改物理/映射。A映射和真实梯度等价失败会阻止后续；C正式模型权重不变，overlap0/4独立记录。17项CPU测试通过，GPU与Kit实证按下方实时生成块报告，准备不等于已成功。整理四个原有主MD，移除过时启动指令，历史证据与5604249来源保持。

## 2026-11-06 收尾复核：完成不等于成功

`tools/narrow_repair_pipeline.py`修正完成状态遗漏正式更新数的问题：原status一直保留启动时0，现从trainer finished读取；本轮依据真实receipt更正为5000，旧status另存。`tools/narrow_repair_delivery.py`将实际训练终点、validation最佳步、同一验证集误差和最终审查结论写入原有主MD及轻量JSON；不改算法、模型或实验结果。8项相关CPU回归通过，真实best4500/last5000和保护hash读回通过，无新GPU运行。

275条模型与200条模型最终均train0/4、unseen0/10有效（各2初态拒绝），离线小幅改善未转化成功率。更新AGENTS/交接为已完成并停止；原有视频索引加入最终6条已录制三视角失败片段，无新增仿真。仅清理2个已验收的1步smoke checkpoint，约2.148GiB；保留正式数据/权重/失败证据。整卡最大完整60s为52.19%，三次monitor交接缺测2518.41s，如实记录，不能称完全合规或无缺测。

<!-- NARROW_REPAIR_START -->
## 2026-11-06：视觉与GPU历史修复、单次慢速数据对照

实测状态 `complete` / `complete`；独立run `narrow_repair_20261106_01`，10h固定总窗，旧run预算和失败保持。代码和排队不代表训练完成。
白屏已定位：direct fresh入口遗漏固定相机配置，两路停在默认相同位姿，RGB恒定/有效深度0；同初态配置前后物理读数完全相同，配置后三视角有真实内容。首条199帧专家成功，起/中/尾画面目视通过，全帧RGB-D、时钟、固定位姿和mask检查通过；单lane持久Kit，不声明多lane已验证。
GPU差异来自未统一的CuDNN TF32：旧默认condition最大2.52545e-4；关闭TF32后最大1.07288e-6、前12目标1.13249e-6rad、FK0.000229mm。原2e-5门限未放宽；完整序列/逐tick/训练prefix/reset均检查，无优化更新。

12:35首次生产停止：第50条原速专家真实掉落，末帧物体离开侧视野；81帧RGB-D和数值合同通过。成功示范的目标可见性检查错误地先于任务失败分类，导致少记1次尝试/掉落。复核49条成功文件哈希后，保留原错误和计数快照，补记为50次/49成功/1任务失败，失败轨迹不入训练。修复仅分离自然任务失败与传感器错误，成功数据门槛不变；8项CPU测试通过。本次复核后的接续入口只运行原冻结清单剩余26条，不重跑已执行条目或门控，不重置计数、17:12采集/21:12总截止；独立阶段reviewed_collection_completion仍只允许一次新训练。实际接续进度以上方机器状态和当前计数为准。

已完成一次扩充：75条新成功、32个新家族，合计275条State-valid；独立随机模型完成5000更新，因validation_plateau_or_overfit结束，validation选第4500步。最终完整成功：state200_fp32: train0/4，unseen0/10有效、2次初态拒绝；state_expanded_fp32: train0/4，unseen0/10有效、2次初态拒绝。这轮有限对照已结束，不自动再训练/扩采或启动Image/触觉训练。

实际扩采/训练/评估结果见[StateDP统一记录](STATE_DP_RESULTS.md)和[机器统计](evidence/narrow_repair.json)，不把准备写成成功。
<!-- NARROW_REPAIR_END -->

## 2026-11-06 — 失败视频交付及相机有效性发现

ade4f1f实际完成4次合法观看复跑，601/824/601/601帧、全超时，仅train004有首子目标。原初态观测一致但首plan与物理轨迹不同，结果不并入正式分母。单独`narrow_video_export`在GPU owner完成后生成腕部观看裁剪；4原视频+4裁剪全帧验证，保留原图像/帧数/60fps，不改仿真。源报告、视频/代码SHA、资源统计和预览写入轻量evidence，原数据manifest、best/last、预算和正式结果hash不变；服务正常退出、全部自有PID消失。

目视验收发现两固定视角空白，进一步只读检查全部200条新示范首帧，两固定RGB各200/200为238常量，腕部非恒定；首尾两条各3帧复核一致。原先数值/存储门控未检出此视觉问题；未修相机或更改数据，未检查全帧/深度/定位原因。State输入不含RGB，ImageDP前需修复验收。更新SLOW_EXPERT_RESULTS/STATE_DP_RESULTS/README/SESSION_HANDOFF/AGENTS，并公开原200训练12验证点、58个100更新loss分箱及原曲线。没有新优化更新、扩采或重复正式测试。


## 2026-11-06 — 有界State200失败视频与发布

新增`tools.narrow_videos`固定4个正式失败案例的独立观看复跑：单worker、1h固定预算、完整资源监控/磁盘停止、源码和原证据hash保护、无训练/采集/自动重试。输入是原评估spec与best5735；输出为本地三视角MP4、完整numeric轨迹、逐帧解码和原/新对比。`state_benchmark.trial`只增加可选视频标题，修正观看用update标注，不改输入、推理、控制或终止。更新README/交接，明确数据生成代码已经发布、扩采尚未执行。启动前语法检查及16项既有窄域数据/条件流程CPU测试通过；真实运行验收待下方完成记录，不能将队列实现当成功。


## 2026-11-06 — 如实记录后续GPU审查阻断

原闭环train0/4、unseen0/10有效（另2次准入拒绝），14次合法运行均超时。后续第一次audit因HistoryBank GPU一致性断言停止，无扩采/追加更新。保持原门限、源码和预算，未自动重试。本次只读CPU FP32/FP64复核相同两条轨迹并保存轻量证据，不能替代GPU验收；更新当前交接、统一结果和采集状态，未修改模型/训练/原控制。


## 2026-11-06 — 隔离的有限条件化StateDP后续

- 新`tools.narrow_followup`在原owner正常完成、进程退出和锁释放后才启动；原活跃工作树/运行目录/截止只读。独立8h预算，最多一次完整状态续训、一次260–296总量扩采、一次随机新StateDP；最终独立16不进入决策，无自动重试或第二轮扩增。
- `visuotactile.narrow_decision`保留验证趋势、Arm/FK、早期方向/幅度/hold和泛化差距的决策证据；合同/控制审查先于扩采。32新家族按训练初态几何覆盖选择并冻结24/4/4，所有数据真实重采，合并cache/train统计重新验收。
- `anchor_continue fresh-*`仅在新runtime显式配置时完整恢复旧checkpoint且保留原LR日程/floor，或以独立seed随机新训练；增加确定性train8/validation loss及平台提前停。既有网络、损失、H24/commit12和控制不改。
- `tools.narrow_delivery`生成主MD更新、一个轻量JSON/曲线；所有GPU退出后校验HEAD/人工正文/工作树，普通非force push到origin同feature。没有上传原始数据、权重或日志。
- 51项决策/家族隔离/停止/恢复日程/原数据保护及既有checkpoint、历史等CPU回归通过。真实200条best/last在5735更新完整读回，原源码SHA未变；已记录学生tick0/12/95的读取/本体恢复误差2.38419e-7。未额外启动GPU/Kit。实际新worker启动/持续状态另在交接和机器文件核验。授权与单测不代表真实闭环成功。

## 2026-11-05 — 直接窄域与自动新StateDP流水线

- 复现SDK scene-query在rollout后reset误报；`live_collision`使用实际碰撞网格与实时link pose的保守OBB，未更改SDK/physics。实际碰撞几何离线核对34网格最小间隔108.62mm，live新入口已完成60家族准入与4条完整成功轨迹的重复reset验收；生产已恢复，启动快照7/7成功，data_ready=false/正式0更新。
- `narrow_data`直接采样相对位姿、20°旋转、较窄native关节/速度/桌高；`narrow_collection`替代旧宽分布拒绝采样，固定family/split和普通后台data→fresh State→闭环阶段转换。旧root不可进入v2。
- `anchor_continue fresh-*`保留模型/loss/历史，增加GPU一次性验收、明确训练截止、完整同run恢复、每500更新验证与首12tick指标/曲线。一次性GPU优化/完整恢复通过、正式训练仍0，不能把授权当完成。
- 按4条实测墙钟与共享负载，在不改原16h绝对截止前提下将生产阶段预算分配为6h采集/最多9h训练/末1h评估；达到最低数据及覆盖才能提前结束采集。增加真实调用耗时分类和预算结束后保存已验证模型再评估。
- 46项CPU回归通过；保留初始化探针的3次软件诊断错误及4条成功native轨迹，不混入新训练数据。
- 清理两份已验收的一次性CPU/GPU checkpoint实际释放2306101248bytes；未改原IDE工作树文档。


## 2026-11-05 — 窄分布采集准备（尚未声称live通过）

- 独立feature/narrow-state-data从84cbb98实施，不覆盖IDE两处既有文档改动或旧暂停任务。
- `narrow_data`读取原400低维，固定相对位姿/旋转/关节条件域与15个旧家族；补9个native初态只按合法性与覆盖。
- `tools.narrow_collection`复用一个Kit和现有trial/记录器/资源工具，明确尝试、固定预算、原子manifest、手动边界停止与无自动训练。
- 修复capture遗漏post-observation/terminal mask；旧错误保留。`anchor_sampling`支持动态数量及短尾批。
- `anchor_continue`增加同一模型/目标的fresh入口，旧checkpoint续训路径保持；新统计仅新train，正式训练未运行。
- 已通过30项相关CPU回归；新增history/目标回归与真实采集验收另记，不把准备当数据就绪。
- 清理只读核对既有依赖和占用；前次126.673GiB清理不重复计入，本轮释放0bytes。


完整旧阶段记录保存在[48a3516 CHANGELOG](https://github.com/mao455/bcrl_visualTactile/blob/48a3516/research_docs/CHANGELOG.md)。此处只保留当前变更，不复制历史巨型日志。

## 2026-11-05 — StateDP150评估与整理

- 原150 epoch trainer继续，不新建训练系统；既有评估入口增加milestone/cohort参数及证据审查hold，固定validation4+train4独立统计。
- 统一端点评估10000/16584/20918，train320/validation40，前12/24/precontact增量、hold基线、FK方向/幅值和累计误差；原始模型与控制保持。
- 从历史runner提取完整checkpoint恢复、进程I/O、视频、轨迹/FK统计；保留函数源码AST等价证据。Native环境、controller、数据时序和网络参数不重设计。
- 删除依赖图不可达旧实验实现与其专属测试/启动服务；将历史阶段文档合并为当前合同、结果、清理和精简历史索引。
- 清理按显式保护清单、文件身份/SHA和进程引用验证；硬链接按最后一条链接的实际分配块计数。完整ledger本地，摘要公开。400原始数据、当前缓存、统计、checkpoint和必要专家/模型视频保留。
- 本轮具体删除数量、释放空间、已运行测试与后续端点评估实测必须见[CLEANUP_AUDIT](CLEANUP_AUDIT.md)与[STATE_DP_RESULTS](STATE_DP_RESULTS.md)，准备或授权不算完成。

## 2026-11-05 — 整理后在线初始化修复

150阶段20918更新、train4/validation4与统一train320/validation40端点比较已经完成。原8次均有效，0软件缺失，未混入额外验收。

整理后的额外验收在模型首次动作前因`native_data.object_dimensions`缺少ElementTree导入失败，原日志及blocked状态保留。补回原导入；新增真实资产/缓存尺寸及在线Controller初始化回归。网络、损失、checkpoint、原生控制与已完成结果不变。只分配一次修复后的独立验收，不修改原失败计数或原评估期限；修复后78测试通过；唯一补充Kit启动在场景前FileThread SIGABRT（−6），保留且停止自动重试。实际回归与限制写统一报告。

## 2026-11-05 — 原生成功审核与随机闭环队列

新增只读success_audit，按物理tick重算两子目标累计计数和撤手条件；原train4与400专家均与原生事件一致，未发现成功漏判，未放宽阈值或改变原RL任务。state_benchmark明确保存首次成功/终止tick，并拒绝完整成功与terminated矛盾后的继续动作；NativeDiagnosticEnv仅增加读数（阈值/计数/实际effort）。现有anchor_interim_eval增加固定种子历史100+全新100和非法初始化专用预注册备用清单、Wilson区间、逐次验证和少量视频保留。独立Kit新缓存、失败系统调用诊断、单owner；旧失败根保留。30项针对成功重构、抽样、CI、位置接口和队列的CPU回归通过；真实Kit验收和200次统计以实时状态为准，尚未宣称通过。

## 2026-11-05 — 终点诊断与实际rollout末尾提交修复

新独立Kit已跑完601tick，未复现FileThread启动崩溃；末尾因旧整理遗漏state_motion.distribution而软件提交失败。原始轨迹/视频/worker错误完整保留，未计入正式性能。恢复原函数及其原返回schema，改为Kit前导入；修复新owner退出时重复pid关键字导致的状态写入错误。增加终点几何与撤手诊断（独立于顺序子目标/原生成功）；用真实专家记录核对最终目标坐标，误差<6.4e-8m。33项CPU回归通过，失败轨迹601tick的原生终止重构通过、与清理前第一plan最大差7.28e-7rad。已退出全部worker后才迁移冻结源码；原失败及截止时间保留。新增一个预注册fresh smoke确保新reset分布在正式200前得到实际验收。

运行前读取已落盘的601tick又确认：旧/新可观察初态全部一致，724个导出字段一致，事件RNG一致；配置差异仅两个URDF的worktree绝对路径。现在仅在文件内容SHA相同时承认这种路径迁移，其他初始化/动力学差异仍阻断。对应回归后34项测试通过。

## 2026-11-05 — OmniClient初始化的明确替代版本

state_random_20261105_01第二启动为场景前SIGSEGV（−11），栈在OmniClient TickThread/FileThread/string操作；无ENOMEM或ENOSPC，256watch探针全通过，约33GiB RAM可用。旧root两个软件失败及第一条601tick轨迹全部保留。资源充足后，新版本仅停止预先ctypes加载OmniClient和调用未公开testSetWatchesEnabled，交还Kit自身生命周期初始化；不改SDK/驱动/物理。此项是有具体生命周期依据的受控假设，尚不宣称根因已证实。35项针对性CPU测试通过。替代root沿用原抽样清单、checkpoint、绝对截止时间和合计245启动上限，旧2次计入总软件尝试；不重置历史预算。

## 2026-11-05 — 真实Kit验收通过，正式随机评估与最终交付

替代版本100d71e已完成4条受监督完整smoke，首plan对清理前差7.28e-7rad，原生专家成功/终止均145tick、无成功后动作，进程资源回收通过；正式历史100→全新100已运行。旧2次软件失败、原始截止和全局启动上限保留，不重新训练或改变原生顺序判据。交接已从旧root/临时worktree更新为实际root02，历史未通过记录仍保留。

新增独立CPU tools.evaluation_delivery：输入owner最终summary及锁定发布请求，输出既有报告的失败阶段/训练建议和轻量JSON；核查真实200分母，终点诊断不提升为原生成功，固定feature HEAD/人工正文/空暂存区后才提交并推送。运行中模型和Kit源码不改。4项针对汇总分母、Image建议、终点诊断隔离及正文保护的测试通过；原训练/批次/模型测试不重复。

用户明确回复保留原生顺序主指标、终点满足另列诊断。仅记录确认并更新交接/报告，已执行结果和运行源码不变；独立CPU发布服务在更新审核HEAD后续接，Kit评估服务保持运行。

## 2026-11-05 — 冻结专家速度研究（隔离源码）

在单一新工作树扩展原anchor_interim_eval/worker与state_benchmark；新expert_speed纯数值函数和显式子类实现后处理目标推进，既有原生RL文件/SDK不改。加入原目标时间重参数化、真实RGB-D小样本采集、官方权重/源码/资产审核、10初态配对及条件20随机扩展的持久owner。原State200不停止、不修改、不争抢Kit；新进程先等待其200完成。33项CPU回归已通过，真正Kit/慢速成功仍待实测；不自动混入训练。正式配置、门槛、未知项见SLOW_EXPERT_RESULTS。

## 2026-11-05 — 分离旧评估样本缺额与独立实验资源依赖

只读检查发现State正式新初态前64次已有32次动作前碰撞拒绝，超过预注册20备用。原固定120名单不可能达到100有效；没有修改其源码/名单/状态/结果。tools/expert_speed_eval新增终止证据审核：仅正常200完成或历史100完成且全部120新初态已持久提交、唯一错误为合法初态备用耗尽时，在进程/监控/owner锁释放后允许独立专家研究。其余软件错误、验收失败、未提交结果、身份不符及资源残留仍阻止新Kit。dependency_release.json和汇总明确保留父评估blocked/缺额，不把非法初态变成任务失败或成功。停止的只有0次Kit的新CPU等待owner；保留旧冻结清单/状态/源版本，固定截止不重置。12项针对性队列测试通过；未重复动作/训练/批次验收，慢速物理结果仍为0。

## 2026-11-05 — 用户优先级切换、边界暂停与小样本专家筛选

按用户授权停止State200后续调度，不打断当前Kit写入：预留下一attempt空目录，原owner完成143有效/175尝试后退出；保留全部结果、原始边界异常、checkpoint和random_fresh_075恢复点，验证无worker/写句柄/GPU/锁残留。tools/evaluation_priority做只读耗时审核和明确pause收据；原随机评估代码不改。新owner接受用户paused并核验文件SHA，从首次实际执行计24h、不计旧等待且恢复不重置；改为低/中/高3初态自适应筛选、最多2候选扩到10，2次无进展连续失败和不可达可靠性门槛均提前停止。正式数据门槛不放宽。tools/rollout_timing只增加主机计时，分别测Kit、场景、RL、物理/渲染、数值写入及视频；无新增随机抽样/物理步。16项队列测试通过；真实物理验收/减速结果以实测报告为准。

当前实测native/A080/A067/A050各3/3成功，0.67中位4.25s、0.5中位9.4s；只新增0.60区间候选3例并优先扩展一组，避免无差别补齐所有10例。B100原目标回放1/3，停止更大拉伸组合；18字段identity逐值一致。全新20扩展预注册最多40seed，仅非法初态补位、合法失败不换seed，18项队列回归覆盖。旧worker全部提交/回收后切换唯一源码版本，24h执行截止保持；所有原始失败与版本证据保留。

A060追加三例实测2/3原生成功（6.400、6.083s）及一次首子目标后掉落，明确尚不可靠。扩展选择优先满足原5–7s时间目标的合格候选，再比较成功数与距6s，不以3/3对2/3宣布成功率优势；选A060进入10例，仍保持8/10和17/20成功门槛。原生基线已运行的补充样本保留，版本只在完整rollout边界切换，既有结果不重复执行，执行截止不变。

恢复23条已有结果时发现初筛错误地计入7条累计native，而非固定3条，触发了错误的“baseline不可靠”停止；该次恢复新增物理rollout为0，错误gate/退出/发布记录完整归档。修正为按冻结初筛identity过滤，正式10/20仍保留全部失败；新增扩展后恢复和committed阶段禁止再次启动GPU回归，共21项通过。原23条不重跑，执行预算不重置。

## 2026-11-05 — 首条真实采集与合法性准入补齐

持久Kit完成360次条件reset抽样，13次进入几何范围、4次碰撞拒绝、9个补充家族接受；
24候选家族已冻结。首train原速171tick完整成功，154源字段hash重建及真实目标/传感器时序通过。
第二个旧家族动作前与拇指网格重叠，0控制步，独立记录为非法初态；owner安全停止，非Kit崩溃。
原400旧检查只有object/table/hole boxes，不能称所有家族已过hand mesh准入。

全部进程退出后补齐所有候选的同一live collision检查，仅按动作前合法性排除，不按结果筛选、不补数。
保留第一worker与冻结版本，使用原4h截止、剩余一次worker额度；旧原速成功不重做。
实际新数据已通过原PersistentState的一次CPU随机初始化前反向/EMA/AdamW/保存恢复；
formal_updates=0、Kit未加载，完整流与在线历史condition最大差1.90735e-6，尾部3个有效目标，未来干预不影响过去。
原400文件按manifest相对路径逐个存在/大小核验通过，原manifest SHA未变；本轮删除0bytes。

## 2026-11-05 — 持久reset准入矛盾与有界停止

第二worker补齐24候选live预检，暂准入23；两条原速171/210和一条alpha.58 366tick成功，
另3次0动作准入拒绝。8组可观察初态逐值一致却查询命中不同，推翻“准入拒绝必为真实非法初态”解释；
全部新数据隔离，未放行生产，原4h截止与2次worker额度不延长。原因未解决，不增加积分步或修改判据。
所有自有进程已退出。新增实际证据回归及矛盾即停/隔离数据禁止ready保护，5项相关测试通过；
这些最后保护没有新增live验证。future fresh-train的monitor/准入移到CUDA分配之前，不改变计算图/参数。
StateDP一次CPU前反向和完整恢复已通过，formal_updates=0；当前数据门槛拒绝正式训练。
清理计划仅一次性smoke checkpoint，工具因4个同UID进程句柄无法完整枚举而拒绝，删除0文件/0bytes。
原400存在/尺寸与manifest SHA核验通过；先前工作树修改保持原样。
