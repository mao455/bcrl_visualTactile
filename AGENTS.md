## 2026-11-18 工程统一：唯一当前入口为原 bc+rl_RSS

当前权威代码目录 `/home/CNS2025331827/docs/bc+rl_RSS`，分支 `feature/tight-state-foundation`。
已从最新研究主线451f591接续，不能再从continuous_vision或其他平行目录开发。
**新实验只增加配置和outputs中的独立run，不在docs下复制完整项目或创建平行工作树。**
130旧工作树已通过Git move移入 `outputs/project_archive/20261118/worktrees`，锁定并保留旧路径兼容软链接。
归档内受跟踪文件为只读，其中内容相同的大文件共享硬链接；不得原地chmod/编辑归档文件。
独有提交、脏内容、未知文件和失败证据保留；需要取用时复制到当前工作区并审阅，不能批量合并旧方法。

统一入口 `python -m tools.tight_project {status,prepare,train,dagger,evaluate}`，执行必须提供显式job及有效有限session。
本轮0训练更新、0DAgger；原400/当前130数据、72阶段完整A/B和候选均保护。两次61tick真实Kit验收不是成功率评估。
完整目录清单、恢复检查、推理一致性、未解决的历史缺失链接及命令见
[工程统一报告](research_docs/PROJECT_UNIFICATION_20261118.md)。以下均为历史阶段，不授权自动重启。

## 2026-11-18 状态退步接管＋130共同池周期已完成：保留72完整A/B

root `tight_recovery_20261118_01`，09:03:00–13:58:52(+08)，4h55m52s；CPU收尾13:59:22。8机会pilot＋32完整正式机会，共25专家救回/4255合法tick，共同池130仍30family。实际pilot8、正式两波16lane；32lane RAM中断保留。新入口23近孔/2抓持，首24共600tick、屏幕到合法动作4tick；旧29均在当前退步屏幕命中前已接管。不是首次因果错误真值。

A DP实际/选2280/2280，B2090/1330；后端均5811，选A966/B4839。各自72父pair、原30每deck3次、其余纠偏1次，4GiB冻结缓存，按新DP刷新后端。A旧72完整命令误差+4.28%，B−5.19%；原早期均改善仍未保证闭环保持。

64实际配对：训练8 A2→1/B0→2，开发8 A1→0/B1→0。A保留1丢2，B新增2训练成功丢1开发成功；B开发hold60为7→1。首次16lane成功后一次RAM中断，剩余6波8lane完成；一次启动依赖错误保留，截止不变。七物理波实际初态检查通过，非隐藏状态精确重放/独立终测。98聚焦检查、279源码/16保护hash通过，所有自有GPU/CPU工作退出。

不晋升、不追加训练/采集；72完整基准、130合法池与候选保护。下一项仅建议冻结72 DP的匹配后端有界学习对照；首子目标后8个未触发失败是次要覆盖限制、恢复性未证明。报告和`evidence/tight_recovery_20261118/{final_decision,closure}.json`为最新终态，下面均为历史。

## 2026-11-18 首个32机会闭环周期已完成：保留72完整A/B

root `tight_cycle_20261118_01`，01:59:40–05:17:12(+08)，3h17m32s。32能力诊断＋32真实并发采集(29接管全救回/3未触发，5946合法tick)＋105共同池A/B完整训练＋64同波配对自主全部结束。30独立family不变；新29多为早接管/正常后缀，未证明覆盖关键晚期偏离。训练/开发能力均不可靠，不支持主要缺初态。

同波训练8：A72 1→0、B72 1→1；开发8：A0→2、B2→1。A新增2丢1、B新增2丢3，原成功无保留，持物60总数下降；小样本不宣称显著。DP各40遍2120更新，选A15/795、B20/1060；后端A20/3261选4/657、B23/3745选20/3261。4GiB缓存显式启用，按新DP刷新后端；不能把实际全部更新算作选中模型年龄。

不晋升、不启动第二批、不自动加epoch/约束/采集。当前正式来源仍`tight_long_20261117_01/round05/learning`完整A/B；保留105/76/所有必要候选。下一项建议调整真实偏离初现时的接管覆盖和首合法标签阶段，非重复容易正常后缀。278源码及保护身份、32聚焦检查通过，所有owned训练/Kit/监控退出。集中报告`TIGHT_TRAINING_DIAGNOSIS_20261116.md`首节和`evidence/tight_cycle_20261118/{final_decision,closure}.json`为终态；以下运行条目是历史。

## 2026-11-18 A普通采样对照完成：两种续训均未恢复完整能力

接续49b8bf1，A普通v2实际DP494/后端369更新，父72完整pair、固定76、统计、初始RNG、优化器与学习率匹配强化终点；强化历史验证选模偏差保留，属于条件预算对照。原示范390 visits/119189早期slots逐一相同；纠偏首24tick系数17.36%→11.10%。4GiB hotset显式启用，后端输入按新DP刷新，无旧策略标签/蒸馏。

四波6lane共24条完整开发结果：72 A1/8、强化0/8、普通0/8，hold120为3/5/3，首子目标初态2/0/0；实际初态检查通过，非新盲测/非严格漏斗。基准002成功260tick，两候选失败；历史006成功在新6lane连基准也未复现，不混合旧2/8。普通在相同真实历史上也没有普遍更接近基准，去除强化不足以解释/修复退化。

00:04:01–00:51:41(+08)闭合，含一次无学生动作的rl_games导入失败及Kit挂起；原失败保留，同截止修正vendored PYTHONPATH恢复评估，无重训。训练/重放/全部owned Kit与monitor退出，279源码/保护hash和完整pair通过，18项相关检查与脚本编译通过。无新DAgger/B训练/损失/网络改动/删除/晋升，保留72完整A/B及所有候选。下一项仅建议普通采样下DP降学习率的单因素保守更新，尚未执行，不自动重启。详见集中报告首节与ordinary76_A/{results,model_match,exposure,same_history,closure}.json。

## 2026-11-17 首次策略分叉诊断已完成（接续c7eac52）

0训练/0新DAgger/0网络控制改动；保留72完整A/B和76合法数据。原三个丢失成功配对tick0策略先变，物体>1mm分叉70/64/64，72处无队列/历史断裂。一次8lane×121tick补录+完整前缀同输入交叉推理，自重放命令最大误差<1e-6rad；参数改变早期输出已证明，正常闭环反馈放大已测量，未证明某个采样因素单独致因。固定76原早期监督系数不变，纠偏入口系数约11%→17.4%；不能说旧监督被slot计数挤掉。A002候选反而更早抬升，三例不是同一种抓取失败。

新root tight_fork_diagnosis_20261117_01_recovery01已关闭；首root在Kit物理前CPU线程准入失败，保留证据。同一截止下仅一次真实捕获；replay初次cuDNN精度不匹配已保存，replay_fp32修正无部署修改。所有owned worker/monitor退出。详见集中报告首节与fork_diagnosis/{original_chain,same_input,exposure_objective,closure}.json。下一轮只建议有界早期输出保留正则的匹配训练对照，尚未执行；不要自动新增训练/采集或把截断诊断计入成功率。

## 2026-11-17 首次策略分叉诊断（进行中）

接续c7eac52，禁止本轮新增DAgger、延长训练或改网络。原三丢失成功配对先按0-based控制tick分析DP/残差/投影/真实执行与历史连续性。原numeric/plans/video不含真实逐帧深度或冻结视觉特征，不能直接重算完整输入历史；登记一次8lane、121tick配对补录（seed28824002/006各四完整策略），截断只作诊断，不加入成功率。新root tight_fork_diagnosis_20261117_01，2小时总界、单次Kit启动最多20分钟、输入载荷最多4GiB，保留12GiB磁盘/10GiB RAM/4GiB VRAM余量。完整pair及76不改；同输入从tick0保留实际外部历史与原规划相位，每模型重算自身潜变量，先核对self replay后解释跨模型差异。

## 2026-11-17 fixed76已闭合：不晋升，无新DAgger

实际19:21:13–20:58:09(+08)，96分56秒，训练/离线/四波32次完整开发均结束；全部自有worker/monitor退出。A72成功2/8→候选0/8，B72 1/8→0/8；hold120各保持4/8、6/8，首子目标A3→0/B3→2，实际可观测初态配对通过。不是统计显著结论。DP实际A874/B722、选494/342；匹配后端实际2170/2411、选369/2048。4GiB hotset正式显式启用；磁盘中止和一次完整状态恢复证据保留，原截止未变。

入口监督曝光增加与训练池拟合改善尚未转成完整自主收益。三个丢失成功配对约70–80tick开始几何偏离，命令/关节更早偏离；不能归为单纯缺数据，未证明新的可恢复覆盖缺口。本轮不加epoch、不采新DAgger、不替换72完整A/B。原76合法数据、所有科学候选及72归一化/配置/来源保留。最新结论见TIGHT_TRAINING_DIAGNOSIS_20261116.md首节与balanced76/completed.json、closure_audit.json；下方准备/运行状态均为历史，禁止自动重启旧服务。

## 2026-11-17 fixed76：磁盘守卫停止与显式恢复

两DP已完成A874/B722更新（分别选择494/342），A后端已完成2170更新。
20:19原attempt因磁盘低于12GiB在B后端中止；原failure/日志/检查点保留。
依赖检查后回收5个I/O profiler临时last权重，释放分配空间5,823,455,232字节；
保留profiling日志、首步梯度、结果及SHA清单，不再能直接从这些已删权重复算最终参数差。
清理不涉及72基准、76数据或任何科学候选。
新attempt recovery02从B durable1170更新/epoch10/group52恢复model、Adam、采样和CPU/CUDA RNG，
恢复校验通过；4条未提交更新日志另存，不算独立有效新更新。六小时总截止和四小时训练截止保持。
后续离线及8初态四策略开发已排队；没有新闭环结果、DAgger或晋升。
证据：balanced76/profile_weight_cleanup.json；本机recovery02/recovery.json与training/restore.json。

## 2026-11-17 fixed76 bounded onset-exposure candidate — preparation

Current authorization supersedes the completed profiling-only scope. Preserve selected72
full A/B pairs, all valid76 and original400. Diagnose existing numeric evidence first;
then one candidate with explicit4GiB frozen-feature hotset, unchanged native physics,
control, architecture, targets and complete histories. Only correction planning-start
allocation changes: half in first24 legal expert ticks, half remainder; original grasp
anchors and geometry RNG preserved. Same episode weights, losses, optimizers and phase
stopping policies. Each DP cap40 traversals/1520 updates; refresh corresponding backend
inputs, then paired complete-policy development on28824000–007. Opened final20 are
analysis-only, never new training/development. Conditional DAgger requires later evidence.

Actual root tight_balance76_20261117_01_recovery01; initial preparation failed on optional
source_data.json before any optimizer/Kit work. Original fixed six-hour deadline retained.
New outputs budget6GiB,12GiB disk reserve, original data/features reused without copies.
No claim of training or autonomous improvement yet. See concentrated report and
research_docs/evidence/tight_long_20261117/balanced76.14 focused tests pass.

## Current override — 2026-11-17 bounded DP I/O diagnosis

Latest explicit instruction permits only same-state short DP profiling and one
bounded immutable-feature I/O optimization, with real validation and semantic
parity checks. No full long training, DAgger, backend training or autonomous tests.
Use an independent diagnostic root; preserve selected72 pairs, valid76, original400.
Read the top of TIGHT_TRAINING_DIAGNOSIS_20261116.md and dp_io_profile evidence.
Completed baseline224/hotset224 and preregistered reverse56 each; no own trainer
remains. Earlier interrupted70 updates are separate evidence. Stage/read gains
are measured but storage-service variability prevents guaranteed long-run scaling.
Exact input/sampling/RNG parity; tiny floating residuals, NOT bitwise equivalence.
Hotset stays explicit diagnostic opt-in; formal default and selected models unchanged.
No automatic further training/DAgger/backend/evaluation from this completed check.
Historical publication-only restrictions below are superseded only for this scope.

## Current override — 2026-11-17 publication audit only

Pause expansion of training and DAgger. Completed experiment outputs are preserved.
This invocation exports lightweight evidence and reviews/pushes documentation only;
no new optimizer updates, collection or simulation. Read the top of
research_docs/TIGHT_TRAINING_DIAGNOSIS_20261116.md and the publication evidence index.
The completed run spans 13h38m58s; shared76 data and selected72 full pairs remain.
Original400, all valid76, experts, visual weights and necessary dependencies are protected.
Historical running/authorization snapshots below do not authorize automatic new runs.

## 2026-11-17 完成：固定56长训、三次DAgger、等更新对照与新20对测试

读SESSION_HANDOFF/集中诊断最新结论。01:50:52–15:29:50(+08)全部实际闭合，所有自有worker/25monitor退出。新test A3/20 B3/20，40合法、0软件中断、实际配对通过；前段持物120均12/20，首子目标均6/20，仍不稳定。最终按预登记八对开发选择共同72条完整A/B，76条候选退化但数据/模型保留。原400/当前76/专家/DINO/必要回退完整，源依赖1445存在；本session未删除，剩余28.49GiB。测试打开后无训练/选模，不自动重启。

固定56新增DP A4648/B4788、配套后端各6011更新；同波长训A0→1/8、B0→0/8，仅有限收益。等更新旧56/新65 A2→0/8、B0→2/8，不支持一致数据收益。三次新DAgger32机会20专家成功3679合法tick，学生前缀与过渡mask0；专家成功不计自主。最终20test已使用，后续仅回归。监控17.67秒间隔、RAM最低10.75GiB等限制如实保留，不称全程无缺口/旧50%资源保证。当前实验无需继续，后续工作需按用户新目标登记。

## 2026-11-17 当前：第三轮76条全流程闭合，最终配对开发运行中

round06 A DP29/1102、选19/722；B22/836、选18/684；两后端各18/2170，选A10/1207、B1/123。完整4对均0/4，持物120各2/4、0原生首子目标，较72条前段退化迹象；视频实际近孔失去支撑，两组各1fall/3timeout，0软件中断。原所有数据与模型保留。三次新DAgger累计20成功纠偏/3679合法专家tick，绝非20学生自主成功。

冻结运行树long03=235943b，final-development服务正串行执行训练闭合前登记的72/76八对同波比较，选整套共同数据快照，不混选数据量。之后执行20对新test，test打开后不再训练/选模。训练已结束，不另启轮次；仍须完成真实评估、科学审阅及推送，不能在后台启动处结束。原训练15:50:52/总17:50:52(+08)截止不变。

## 历史状态：round05已完成，第三轮后段接管待实测

最新真实状态以集中诊断报告及SESSION_HANDOFF开头为准。共同72条：A/B DP各18/648更新；后端A18/2071、B24/2760；完整4对均0/4，持物120两组4/4，原生首子目标A3/4 B2/4，0软件中断。两视频全解码/查看，近孔后失去支撑，尚无完整自主收益结论。原所有数据/模型保留。

下一轮采集专用post_subgoal已通过短检查、55项测试及既有numeric可达检查；尚未宣称专家能救回。拟用原family26–29、early/near/post_subgoal/post_subgoal八机会，当前完整72模型为源，先小规模实采审核后训练。原session训练15:50:52/总17:50:52(+08)截止不变，最终20新测试仍未打开；不得仅后台启动即宣布完成。

## 2026-11-17 — 固定56长训练闭合，首批开发A2/4、B0/4；正在扩展配对比较

两DP实际各192遍/5376更新：A从26/728完整恢复、新增4648、7039.59s，B从21/588恢复、新增4788、7142.87s；原始数据、mask、划分、归一化与采样不变。独立验证96–192窗口选择A97/2716（.005390522）、B99/2772（.005603639）；末尾验证A.005462025/B.005747975，后半程未持续改善，不能把实际完成预算当选中checkpoint更新数。两后端各刷新对应DP输入，实际64遍/6011更新，A1453.61s/B1462.59s；验证选A11/1036（.000295287）、B1/93（.000401766），warm-start旧匹配后端、fresh Adam已披露。

首批完整学生4对开发28820000–003全部合法、0软件中断，实际初态配对通过：A2/4完整插入（000为273tick、003为308tick），B0/4。A持物120tick4/4、原生首子目标3/4；B持物120tick2/4、首子目标1/4，两个跌落。A002首子目标后原生时钟重置，815tick超时，未延长或改成功判据。首对真实视频全部解码A273/B601帧并查看三视角关键帧：A完成抓取、搬运、转正、下插及释放；B搬近孔后姿态偏斜，约240–272tick失去支撑、物体落桌，后未恢复。不是专家救回结果。

这4对与历史基线不是同一执行波，不能单凭A1/4→2/4宣称因果收益。正在使用冻结dd82fc1将新A/B与原完整A/B基线放入同一8lane波、每波2个实际配对开发初态，目标8对后决策。新20对test仍未打开。midpoint owner/monitor正常结束；原冻结25bcc4a运行树未热改。evidence/tight_long_20261117/midpoint_development.json与midpoint_learning保存实际统计/曲线，视频在本地midpoint/development。尚未完成新DAgger、等更新诊断或最终测试。

## 2026-11-17 — 固定56条长训练：已登记，尚未产生新结果

新授权覆盖前轮停止条件；现场HEAD6031c12，无活动Tight任务，约77GiB磁盘/31GiB可用RAM/23GiB空闲VRAM。旧400、现56、原完整基线和所有负结果保留。先固定数据/划分/归一化/架构/采样/损失，后依据完整闭环再继续最多3轮分阶段DAgger；本session16h总界/14h训练界，不沿用旧截止。预留28823000–019共20对新测试，冻结最终模型前不打开，旧两批8test仅回归。

真实前端谱系（每行只计该模型选中祖先，不累加丢弃分支）：A初训选77/1155更新→37条选80/1520→49条选9/225→56条选23/644；B选80/1200→42/798→10/250→12/336。每次旧DAgger是EMA权重warm-start、fresh Adam/EMA，不是一个连续优化器。最近完整last A26/728、B21/588，均lr1.5e-5。原日志/metadata抽取在baseline_frontend_budget.json，含合法窗口目标曝光与实际末尾预算。epoch是完整轨迹遍历，B2每56条28次optimizer/EMA更新，非扩散噪声步/控制tick/DAgger轮次。

本次从上述last恢复raw model、Adam moments、EMA、global update、完整deck/采样曝光与RNG，仅显式重设有限学习率/平台计划：第一节点绝对192epoch（A新增4648/B4788更新）；若证据支持，第二节点360（相对原last新增9352/9492）。差140次来自原last不同，均相同终点5376/10080阶段update，不强称新增计算完全相等。第一节点lr3e-5、96epoch后24遍无实质改善降一次半，192前不以旧短平台停止。中节点只在96–192选模型，长节点264–360，避免把旧低预算best误称新候选。原best独立保留。

复用现有learn/train/bounded_extension入口：新root只读链接原numeric/视觉，复制可追加日志，模型用原子替换安全硬链接；恢复门限定数据/统计/采样完全相同，新增预算和源码身份显式注册。每候选DP之后重建后端计划，匹配旧完整后端权重warm-start、fresh Adam，64完整监督tick遍，不冒称跨输入变化的exact resume。16项聚焦测试通过（含长预算数据变更拒绝、checkpoint身份拒绝及不提前平台停止）。实际训练/自主收益尚未完成，后续持续补充。

## 2026-11-16 23:00 — 两轮分阶段DAgger和新独立测试完成，尚无自主收益

本轮已真实完成审计→4lane多阶段先导→正式共同纠偏→两轮DP/后端训练→每轮4对完整自主开发→冻结第二轮模型→新8对独立测试。运行20:48至22:58，约2h9m21s；没有沿用旧15:30截止、改变10cm布局/网络/控制/物理/成功判据。全部自有worker/monitor退出；保留两个准备阶段故障服务及原回执，不把failed服务误报为活动训练。新测试已打开，此后不按其结果追加训练、选模型或重复试验。

新增24合法采集机会中19次真实接管、19次专家完整救回，5次未达条件保留为未接管机会；2873有效专家tick。两轮共同train37→49→56，始终validation6，旧数据累计。最终26条混合轨迹/4019合法专家tick（含原7条）在A/B当前缓存中全量审计：完整time/robot/control_segment/mask与原始numeric一致，后端target只取合法专家最终命令；DP窗口mask排除所有无标签prefix。原始控制源、真实q_after和u_expert时序的逐轨准入检查通过。学生前缀和2tick控制队列过渡mask0，但完整历史继续作为专家后缀输入；没有学生自监督。phase_audit.py新增--all-corrections仅扩展只读审计范围。

round02：A DP18遍450更新/选9，B20遍500更新/选10；后端各24遍2129更新，A选24/B21。round03：A DP26遍728更新/选23，B21遍588更新/选12；后端A21遍1973更新平台停止/选11，B24遍2254更新/选21。两组同数据/划分/40DP与24后端上限/平台规则；实际更新量不同，不强称等计算量。每轮更新各自DP后均重建后端输入。round03最佳独立验证A DP .00550676/rear .000289578，B .00580506/.000365917，仅各标签空间内解释。

自主开发：原37条A1/4 B0/4；round02 A0/4 B1/4；round03 A1/4 B0/4（A28820003,272tick完整成功）。round03两组近孔筛选均2/4、对准筛选均1/4，持物120tick均1/4；A较round02近孔4/4、持物120tick4/4退步。真实表现不稳定，不能称失败整体后移。首对开发视频各601帧全解码及关键时刻浏览：A未稳定抓起；B近孔转正后约300tick失去支撑。保留前段纠偏数据并未保证前段能力不退化。

最终新测试28821000–007：A0/8、B0/8，16合法、0非法、0rollout软件中断，两波实际可观测初态配对通过，隐藏PhysX/随机流不称精确相同。A/B各1跌落7超时；首子目标A0/8 B1/8。B28821001首子目标后原生时钟重置，897总tick超时，未修改超时。诊断近孔筛选A/B均5/8，对准筛选A1/8 B3/8，持物60tick均5/8、120tick A3/8 B5/8；筛选不是原生成功。首对新测试视频各601帧全解码、查看真实三视角：A抓起搬近孔、转正后未插入，约360–420tick倾斜失去支撑；B未稳抓、物体留桌，后续手部运动没有恢复。近孔误差、姿态和接触丢失时刻在final_part0/1逐轨记录。

两次非物理故障均有界恢复：formal02误用默认2CPU导致16线程执行后端准备拒绝，改用原已验收RESEARCH_CPUS=8-23；round03评估准入漏认phase平台停止配置，补充严格识别后在eval_recovery01只评估原模型，0新增优化/0重复物理起点，10项准入正反测试通过。最终四模型SHA重新核验通过。原400manifest SHA不变且全部文件存在/大小匹配，现有1425个manifest源依赖无缺失；剩余77.20GiB。本轮没有再删派生产物；此前126.03GiB清理证据保留。八个实际GPU阶段监控missing0、最大采样间隔1.94s，采样最低可用RAM20.44GiB、峰值GPU已用12519MiB；这是采样证据而非连续硬保证。

结论：分阶段触发确实取得后段真实可救状态，专家19/19能救回；离线误差下降未转化为独立学生自主插入收益。旧8test仅历史回归，本轮使用另8个预登记未参与采集/模型选择的test；两个0/8群体不同，不构成严格配对前后效果估计。10cm布局减少物体—基座初始几何干扰、专家仍可完成全任务的前轮结论保留；没有新增布局因果证据。当前结果不支持继续盲目增加同类示范或延长训练，应保留本轮负结果与前段退化证据。

证据：research_docs/evidence/tight_phase_20261116/{summary,collection_summary,round03_supervision,final_supervision,round03_development,final_test,final_part0,final_part1,final_audit}.json；两轮learning.json与validation.png在对应round02_learning/round03_learning。完整本地视频位于outputs/visuotactile/run/tight_phase_20261116_01/round03/eval_recovery01/development及final_test/part0。代码/MD/小报告推送origin/feature/tight-state-foundation，大模型/原轨/视频不入仓库。

## 2026-11-16 22:47 — 第二轮模型训练闭合，修复评估准入漏项后恢复

round03 DP A26遍728更新/B21遍588更新，后端A21遍1973更新平台停止、B24遍2254更新达到上限；没有新自主结果。新增7纠偏1172有效专家tick，两组numeric/后端完整time/robot/mask/segment及最终专家target一致，窗口mask排除prefix；旧12条已单独审计。总24合法机会、19真实接管全救回、2873有效tick；collection_summary.json给出分阶段分母和初态family覆盖。

原学习owner于22:43:27回收，monitor0；evaluation_prepare在任何仿真前拒绝A，原因completed_pair仅认识旧预算而遗漏已注册phase_dagger_budget_v1的合法平台停止。tools/tight_fresh_rollouts.py补充严格该后端profile，不降低minimum/stop/hash门；tests/test_fresh_rollouts.py十项通过。learn.py增加唯一evaluation_only_recovery：原failure/owner与未创建development检查、原模型只读链接及hash核对、新监控root、原截止不变、0新增优化/0重复物理机会。原训练失败回执保留，不覆盖。将使用独立eval_recovery01完成原4对开发后再冻结新8对测试。

## 2026-11-16 21:52 — 第二批新增7条纠偏，共同56条继续训练

formal02_recovery01八合法机会、七真实接管全专家救回、1172有效tick；A支撑变弱条件未到达，完整601timeout，未入池。对准接管A211/B286tick，专家分别再执行234/72tick完成原生任务；近孔139/155、支撑变弱B153、早期66/64。控制相位7/10/7/11/9/6/4，真实控制/标签与prefix mask审计通过。八lane循环/准入总335.45s，7成功纠偏约75.12/h，仅本批测量，不泛化为持续吞吐。

采集owner/monitor21:50:23正常退出，round03继承完整49旧轨迹、合入7新轨迹，共同train56/validation6/test0；A/B各自当前完整模型继续，更新DP→重新生成各自后端输入→Transformer→4对自主。新阶段累计24合法机会、19专家救回/2873有效tick，五未触发不算救回失败；所有prefix仅作实际历史。当前缓存运行，第二次学习未完成，新盲测未打开。

## 2026-11-16 21:46 — 首次新阶段全闭环完成，继续第二次迭代

round02共同train49/validation6完成。A DP18遍450更新、验证选9；B20遍500更新、选10，均平台停；两后端24遍2129更新，A选24/B21。独立验证最佳A DP .00574271、rear .000301408；B DP .00581108、rear .000406490，各标签空间内部解释。两组缓存逐time/robot/mask/segment/最终命令检查通过，新增12纠偏1701有效专家tick全部保留真实prefix，不监督学生动作。

8次自主开发全部合法、0软件中断，实际配对通过：A0/4、B1/4（28820003,282tick）。A3fall/1timeout，B其余3timeout；首子目标各1/4。A持物120tick4/4、近孔触发4/4、对准筛选3/4，B2/4、2/4、2/4。上轮A1/4、B0/4，因此不是一致收益，更不能声称失败整体后移。A/B首对三视角视频全解码480/601帧并查看关键时刻：A抓起、搬近孔并转正，持续调整后约360tick失去接触，后跌落；B早期抓持未稳定，物体倾倒在基座/桌上，手继续动作但未恢复。B一次真实原生完整成功仅属开发结果。

round02 owner/monitor于21:42:29正常退出。根据真实失败选择训练family12–15×两组，四条件alignment/near/weakening/early，各一family，formal02_recovery01实际启动；成功后缀至少2条才聚合累计全部49旧数据并进入round03同预算完整学习/开发。第二次学习之后冻结并一次打开独立28821000–007，不依据新test再调参。当前没有新自主结果。

首次formal02准备误用run_cpu默认2CPU affinity，而原验收Kit16线程需要16CPU可见，准备阶段拒绝、0仿真机会；随后服务因prepared缺失退出。保留原formal02及日志，停止仅等待该失败root的自有round03服务。唯一恢复用RESEARCH_CPUS=8-23、torch/BLAS仍2线程，与原验收设置一致，新root/服务，不改变物理、seed、截止或重复执行。冻结phase03 d66755b，无活动热改。

## 2026-11-16 21:05 — 新阶段首轮共同49条训练已接续

formal01八合法机会闭合，A3/B2成功专家纠偏、827有效tick；另外三次未触发接管而原生超时，不作为专家救回失败。两次对准接管A182/B195，支撑变弱A132，早期A61/B59；没有固定规划边界取整。与两先导合计16合法、12真实接管全部救回、1701有效专家tick，四次未达条件未入池。已审核控制权/mask/真实q_after及target标签，学生prefix和2tick过渡保持mask0。

formal01于21:03:49 owner/monitor正常退出，round02随后从冻结phase02树启动共同train49/validation6/test0；原37条全部累计，未改变归一化原train30、采样权重和标签语义。当前缓存阶段，不提前宣称训练完成或自主收益。后续同规则最多40DP/24后端遍，先更新两DP、刷新各自后端输入，再训两Transformer与同4对开发。

本批453.51s含准入/启动后执行，实际循环421.07s/3269有效lane-tick，7.76tick/s、lane利用67.99%；环境含渲染163.82s、控制器107.16s、完成审计105.22s、提交22.66s。八lane本批不构成严格加速证据；保持已验收共享视觉、同帧render复用、512MiB有界writer与2验证进程，不采用未过门的历史裁剪。报告evidence/tight_phase_20261116/formal01.json保存逐起点、监督、触发和计时。

readout.py增加显式--root只读查看新轮回执，禁止与--save组合覆盖旧进度；实际只读调用和py_compile通过。活动冻结源码未热改，旧holdout仅回归，新28821000–007仍未打开。

## 2026-11-16 20:57 — 四类小规模触发验收与新盲测隔离

pilot02四合法全部专家完整救回：搬运A99/B98、近孔A159/B152tick；557合法专家tick。两个先导合计8合法、7成功纠偏/874有效tick，0软件中断。搬运仍接近原早期窗口，不将其称为后段新覆盖；新增价值主要在近孔、对准及近孔支撑变弱。正式formal01四family×A/B共8机会已串行开始，分配对准2family、支撑变弱1family、早期接触1family，保持共同数据和一致执行约束。

普通服务只在formal01完全闭合且owner回收后聚合pilot01/pilot02/formal01审核通过的后缀，继承全部37旧合法轨迹，再串行A/B DP→刷新各自后端→Transformer→配对开发；最多40/24epoch、至少16/12后按验证平台停，两组相同规则。不把开始或专家成功写成学生收益。

final_test准备入口新增显式预登记文件参数：本轮28821000–007与原测试/开发/训练/验证隔离，登记时间必须早于最终训练完成；冻结前不打开，默认旧实验路径保持复现。final_report允许登记的实际资源roots，避免错误读取旧实验监控。当前仍未运行新测试。

## 2026-11-16 20:54 — 首批四环境后段接管验证已闭合

pilot01四机会全合法、0软件中断：对准组B在200tick接管并救回，A未达到注册对准条件而601超时、没有强行固定tick接管；支撑变弱组A183/B133tick接管均成功。共3条成功专家后缀/317有效tick，phase5/3/8，全部学生prefix和队列过渡屏蔽，原始控制/标签与因果触发回放审计通过。两服务串行交接，pilot01 monitor0/owned退出后pilot02才启动，后者当前检验搬运与近孔，尚无完整结果。

实际pilot01循环194.92s/1440有效lane-tick（7.39/s），controller69.08s、env74.04s、finish audit29.60s；四lane尾部利用率59.90%。原八lane队列244.93s/2612（10.66/s），不同轨迹故非严格速度对照。后续优先同源既有8lane批次减少启动/低利用尾部，保留512MiB有界writer和两验证进程。检查旧exact-tail门发现历史完整记忆数值验收失败，未采用该优化。冻结视觉只复用相同输入的基础编码；各轮重新计算可训练历史与后端计划。

新增phase_report.py记录每类student-prefix可达、真实接管/救回、合法mask、有效tick、相位、孔口误差、失去接触及计时；phase_pool.py只合并审核闭合的同一pair采集manifest并链接原轨，不重跑、不覆盖失败。两者尚未用于新训练完成声明。

## 2026-11-16 新阶段DAgger：正在审计与小规模接管验证

最新授权继续10cm布局完整A/B共享DAgger，覆盖旧停止条款；不改变网络、物理、控制或原生成功。新session tight_phase_20261116_01独立6小时总界/5小时训练界，最多两轮完整学习、8个先导及最多32个额外纠偏机会，软件故障每阶段最多一次新身份恢复。首批4lane实际并行验收，之后按吞吐与资源决定；不热改运行源码。旧8test仅回归，新28821000–007保留到最终模型冻结后一次使用。

已真实审计原37训练：7纠偏/1146合法专家tick，两组numeric与原始控制/标签一致，学生前缀与队列过渡mask0；两组后端缓存完整time/robot/mask/segment与原轨相等，命令目标一致，DP窗口mask排除所有无标签tick。完整历史不截断、不将prefix冒充监督。新多阶段触发只读已完成实际transition的特权几何/接触，安排下一动作接管、不按mod12取整；尚未实测救回，不宣称有效。主要按搬运平移、近孔root、孔口keypoint误差和近孔支撑变弱筛选，保留已有早期纠偏能力。原生mask/队列/执行器不改。现有源监督审计与回放证据在evidence/tight_phase_20261116。

## 2026-11-16 20:14 — 新布局完整实验结束：干扰减轻，专家36/36，独立学生测试均0/8

实际清理、原位/5cm/10cm专家试摆、新示范、A/B随机初始化训练、完整共享DAgger一轮及最终独立测试全部完成；三个学习/纠偏/测试服务inactive/MainPID0，无本轮worker遗留。最终测试于20:10:59闭合，20:11:00 owner回收子进程、monitor以0退出。最终测试已打开，不自动续训、添轮、调参或重复测试；旧15:30截止未沿用。以下旧条目中“正在运行/测试未打开”是历史快照。

选定仅插销起点XY平移10cm，位移(-93.401,+35.725)mm，孔位/资产/物理/120-60-5Hz/完整成功判据不改。原位3/3成功；5cm两合法均成功、一保守碰撞门拒绝；10cm三合法均成功。自然沉降后原位两例物体—基座表面间距约0.009/0.044mm且高度受基座影响，10cm三例15.45–18.23mm、无明显竖向位移；新36条沉降间距14.87–24.05mm。支持减轻真实物体—基座几何接触干扰，不等于证明手指碰撞力或视觉遮挡的独立因果改善。专家新train30/validation6全部合法、36/36原生完整成功，没有替换seed；两次物理启动前软件错误与pilot退出卡顿另存。

两学生从随机DP/Transformer开始，共同train30/validation6；初训各前端80遍、后端40遍，独立开发均0/4。一轮完整学生纠偏8合法机会，A3/B4共7条真实成功专家后缀、1146有效tick合并同一冻结train37池，学生前缀和2tick过渡mask0；各自DP更新后刷新后端输入再训Transformer。两组相同40–80遍/平台停止规则，A前端80遍、B51遍停，后端均40遍，实际更新量不强称相等。开发A1/4、B0/4；此后冻结模型，预留8对test各跑一次。

最终独立原生完整插入A0/8、B0/8，16合法、0非法、0软件中断；两波实际初态配对均通过，隐藏物理状态不称精确重放。A一跌落/七超时，B两跌落/六超时；首子目标A1/8、B2/8不算完整成功。B seed28819001首子目标后原生时钟重置，806tick超时，不是延长原超时。首对真实三视角视频各601帧全部解码并浏览关键时刻：A抓起转正并搬近孔，随后倾斜、撤手后在基座/桌面旁失去支撑；B近孔持续调整但未完成对准，后倾倒到桌面，均601tick超时。开发的一次成功没有形成独立测试改善证据；没有同训练剂量旧布局对照，不能将变化单独归因于拉开距离。

清理净释放135318011904bytes（126.03GiB）：256份闭合失败RGB-D，保留全部数值/控制/结果及36份完整代表失败，未将学生失败叫数据损坏。最终审计400保护数据文件存在且大小匹配、manifest SHA与10月31日保护一致；现有1406个manifest依赖均存在，最终四个模型重新验SHA通过。没有再删活动文件/有效源/专家/预训练权重。当前剩余约103.76GiB。最终测试采样最低可用RAM20.85GiB，峰值GPU已用12271MiB；本轮按已授权全GPU、6GiB/12GiB准入门执行。初训最大采样间隔9.70s，后续最大1.58s；missing计数0不代表连续硬保证，保留全部原始监控。

只读final_report.py在生成统计前核对闭合/无worker错误、每个test seed恰好A/B各一次、实际配对通过、最终模型hash；真实报告执行及py_compile通过。最终证据：evidence/tight_layout_20261116/final_test.json、final_audit.json，前期清理/试摆/采集/学习/纠偏证据同目录。代码与小报告推送origin/feature/tight-state-foundation；大数据/模型/视频留本地。

## 2026-11-16 20:01 — 首轮共享DAgger完整闭合，A开发1/4，最终测试已冻结启动

真实采集→共同37train/6validation→A/B DP→各自新DP后端输入→Transformer→8次完整自主→owner/monitor退出全部完成，0软件中断。A前端80epoch/1520更新，选80；B51epoch/969更新、验证平台停，选42/798。两组相同40–80/平台规则，不是强行等epoch；A46末/B43末LR减半。两后端40遍/3200更新，A选39/3122，B选40/3200；A34末LR减半。原验证空间不变，前端最佳误差A .00695559→.00582831、B .00701769→.00586132，后端 .000404844→.000324505、.000585439→.000475277。仅各组内部比较，不以跨标签loss排名。

同四对开发初态全部合法，A1/4原生完整插入（seed28820001，331tick），B0/4；初训均0/4。A对准/首子目标2/4，B1/4；持物120tick A4/4、B2/4只作诊断。A000完成首子目标后滑落、820tick原生超时，B001为819tick原生超时；任务子目标会重置原生时钟，不是改变601超时。其余A一601超时/一399跌落，B两601超时/一411跌落。首对真实三视角视频已浏览，A转正并到孔后撤手但未完成，B早期抓取支撑丢失；不把A首子目标当完整成功。可观测配对通过，隐藏状态不声称精确相同。

source=dagger01/learning，本轮到此冻结；最终8个预登记独立test seed28819000–007各A/B一次，分两波8lane，tight_layout10_20261116_final_test01，冻结99cfaf7/service tight-layout10-final-test01-20261116。测试正在运行，尚无结果；此后不据test调参/选择/续训。A的1/4是小样本开发改善，不证明布局单独的因果效应，也不证明稳定泛化。旧布局没有匹配新数据/训练剂量的同轮对照。

## 2026-11-16 18:58 — 首轮新布局纠偏采集闭合，共同37条已开始更新

8机会全部合法、0软件中断；A3/4专家救回，B4/4。A seed28817002未形成contact-height窗口，完整自主601步超时，未执行fixed fallback、未入池。七次真实接管在96–128tick，共1146有效专家动作；每条前缀mask0，2tick队列过渡mask0。实际numeric逐数组验证q_reached等于同transition真实q_after，u_expert等于实际执行target；没有把学生动作当专家标签。成功恢复不是自主插入。

共同冻结train37/validation6/test0，仍原train30统计、每条权重1。采集monitor0退出后学习owner启动，A DP已完成首epoch19更新；B后续相同规则。原24h/22h截止、完整学生与原生成功不改。新只读correction_report.py与evidence/tight_layout_20261116/dagger01_collection.json保留全部分母、触发与标签审计；运行worker冻结99cfaf7未热改。

## 2026-11-16 18:51 — 新布局初训与完整自主完成，共享DAgger已接续

共同train30/validation6，学生随机初始化；A/B前端各80epoch/1200更新，独立验证选A77/B80；后端各40遍/3080更新，选A40/B39。均达到登记预算上限，不声称完全收敛。前端监督空间不同，不能比较A/B loss大小判断优劣。真实曲线与逐初态行为见evidence/tight_layout_20261116/initial_learning。

四对独立开发初态全部合法，A0/4、B0/4原生完整插入，0软件中断、0对准/首子目标；各1次原生跌落，其余各3次601步超时。持物60/120tick诊断A2/4、B3/4，不替代完整任务结果。首对实际三视角视频已查看：两组抓起、搬近基座并转正，插入端未完成对准后滑落；其他初态还存在抓取失败。暂未证明新布局提高学生自主插入。

四对首动作前q/qdot/目标完全一致，位姿/速度在1e-5容差内；不是隐藏状态精确重放。learning01于18:51:33完成，18:51:34 owner/monitor退出，monitor0；此前评估监控提前退出未在这8次复现。

已基于失败阶段选择一次接触抬升后的纠偏：tight_layout10_20261116_dagger01，冻结99cfaf7，服务tight-layout10-dagger01-20261116；原train前4family×完整A/B共8机会、contact_height触发，无fixed fallback。现在仅采集运行；成功实际专家后缀合并同一冻结池、前缀mask0，DP更新→重新生成后端输入→两Transformer→同4对完整开发评估。保持原24h总界/22h训练界，统计仍原train30，采样权重1，两组预算同规则。尚无新纠偏/训练/自主结果，最终独立8对test未打开。

## 2026-11-16 新布局36条示范完成，随机初始化训练已接续

10cm布局独立train30/validation6已全部闭合：36机会、36合法、36原生完整成功，0任务失败、0替换seed。此前两个启动前软件问题另存原root，不并入物理机会；原pilot关闭卡顿也保留。36份实际初始可观测状态哈希互异。新示范抓前24/36/48tick的碰撞盒基座表面净空14.87–24.05mm；这是实测物体—基座几何，不是手指—基座力归因。专家仍可完成抓取、搬运、对准与插入。

采集owner与monitor已退出，退出码0；数据字段与训练标签审计通过。冻结8bb8e23、隔离树layout_pipeline02已自动接续共同train30/validation6，A优先，DP与Transformer从随机权重开始；尚无新学生自主结果。维持预登记40–80遍前端/20–40遍后端独立验证停滞规则、原总截止、相同A/B预算。没有重启旧15:30任务或混入旧布局数据。

`scripts/tight_layout_20261116/collection_report.py`重算闭合numeric的实际几何并绑定已审计数据manifest。首次汇总误读capture内不存在的sha256字段，改为读取已审计manifest的source_sha256并核对真实路径；不改原始数据或运行worker。报告[collection_complete.json](research_docs/evidence/tight_layout_20261116/collection_complete.json)记录全部36机会。

## 2026-11-16 当前：仅平移插销起点的布局假设验证

当前授权覆盖此前禁止新采集/训练与已结束15:30截止：先实际清理，原位/远离基座5cm/10cm小规模专家全任务比较；可用后新版本约30独立成功示范，A/B共同划分、DP与Transformer随机初始化，再完整DAgger共同冻结池。原孔位/资产/物理/120-60-5Hz/任务成功不改，不大改模型/采样/控制。A=q[t+1]+原几何，B=u_expert，后端均最终专家命令；学生前缀只是真实历史。

现场在continuous_vision_20261114，HEAD e6d48db与origin/feature/tight-state-foundation一致；narrow_followup同名本地分支落后且脏，保留。16:46无活动研究worker；旧可靠评估16次均合法、四组各0/4原生插入已结束。

新root tight_layout_20261116_01。已先登记并执行256份闭合失败RGB-D退休，保留数值/控制/结果/已有视频和36份完整代表失败，所有成功及1363个manifest源保护；净释放135318011904bytes。不是损坏数据删除，不重算旧清理。清单/逐文件SHA在root/cleanup。

新pilot登记3初态×3布局，共9机会、原有慢速专家/真实观察/原生完整成功；独立2h只界定pilot。位移只改native fixed_start_pose XY，孔位按原相同seed生成后冻结；超旧object采样框是明确新初始化，不裁剪。每初态保持z/旋转/关节/孔/事件seed，隐藏状态不称精确配对。真实碰撞盒表面距离与合法性、首组3视频后审查；偏移布局至少2/3原生全任务成功且合法、可见、支撑通过才进入重采，允许有依据的一次小调整。尚未启动/无新结果，不把准备写成有效。

## 2026-11-16 当前完成：可靠的既有完整策略→完整任务评估

最新用户要求接续8b946ca，只完成最小配对/协调/监控修复及既有四模型完整评估，不扩大采集、模型组或重构，不继承已结束15:30截止。任务已于16:05完成：tight_reliable_eval_20261116_01，4开发初态×A长/B短/A11/B15，16合法完整执行、0软件中断，四组原生成功均0/4。真实初态通过配对，四阶段交接和完成后零worker续接通过，所有拥有进程已退出。保留A11/B15及A14+rear22/B23+rear6，原2成功纠偏/全部断点/缓存不删；本轮未续训独立验证退化的候选。阅读集中诊断报告开头和最新handoff，不自动重启旧15:30任务、DAgger或新训练。

## 2026-11-16 当前任务：延长训练后的持物稳定性诊断

接续3bcccc0；保留A长训候选、B长DP+短后端及A11/B15安全回退。
不启动20轮DAgger，不改采样配比。先复用既有轨迹定位A真实失稳与正常任务
变化，比较相同真实输入下短/长后端；再登记少量新开发family作完整学生验证。
可有证据地有限延长一组训练，不按best出现较晚直接认定未收敛或采用候选。
新诊断身份tight_stability_20261116_01，有界3小时，原budget任务已完成且不重启。
所有原记录/模型保留，输入捕获为只读诊断，绝不成为额外控制或训练标签。

## 2026-11-16 前次任务：固定分布延长训练诊断（已完成）

最新用户要求暂停新增DAgger与采样配比修改。复用19末尾完整DP状态，固定348train/6validation、A11/B15初始化、latest_repeat12、B2和既有监督/执行语义，最多24epochs；独立验证连续4次不足0.5%改善时减半LR，减后仍停滞且至少16epochs则停止。DP后刷新对应后端输入，再以相同A11/B15后端初始化和种子做8至24的同序列对照；保留best及8轮best，不把只加载权重称严格续训。用户授权有界真实训练、短闭环、必要清理与原分支推送；不启动DAgger循环。具体登记在scripts/tight_budget_20261116及集中诊断报告。

## 2026-11-16 夜间最新要求：先诊断，两个真实闭环后才允许脱离监督

用户最新消息要求先完成已入库26条训练，追踪真实标签/曝光/接管时机，用固定训练家族诊断学习瓶颈。不得无限堆数据或把专家恢复当自主成功。主Agent必须监督两个连续完整采集→准入→DP→刷新输入→Transformer→自主评估小循环与交接，才能释放无人值守屏障。

原04的02:05:20截止来自自行登记的六小时临时时段（scripts/tight_fresh_20261115_04/launch_config.json hours=6），不是外部机器配额。用户本次明确允许延长到次日上午；原04/13现场和截止不改，待其阶段闭合后另建截至11月16日09:00 +08的新有限session。当前只准备，不能把两循环或后台可靠性写成已通过。

## 2026-11-15 22:25 采集吞吐优先，旧训练保留

用户最新要求已将正式采集吞吐列为工程优先级。04训练保持冻结；未启动的05边界launcher已停止并保留priority_superseded.json，不启动旧的大量完整评估安排。新增GPU观察/历史/控制合批、批量CPU快照、向量专家队列映射、独立lane补任务、空闲标准相机停更、固定外参缓存及抓持阶段评估入口。实际队列/相机验收与正式加速尚待；当前不是成功或提速结论。

11项短检查通过，正式FP32下真实A11权重/记录输入配冻结patch fixture的最大目标差5.18e-7rad；初次CUDA检查遗漏正式precision设置的失败已说明，没有放宽最终2e-5rad检查。旧路径4机会真实早期纠偏对照已启动于04/throughput_baseline15_01，来源A11/B09；新队列将用同负载比较。全部保留04固定02:05:20截止。详情见scripts/tight_fresh_throughput_20261115/README.md与对应证据目录。

## 2026-11-15 21:20 最新优先级：先练稳接近、抓起与持续持物

用户要求在真实早期偏差时接管，优先重访早失败训练family；不能主要等待held入口或用fixed1补充示范代替纠偏。来源按抓持证据选择A11/B09，两个网络分别记录新早期后缀曝光与拟合；仍DP→刷新计划/特征→独立后端→完整自主。90%以上持物目标需要固定及额外隔离开发初态支持，原test22保持隔离；本轮未达标，不自动转后段。第十四轮冻结训练继续，第十五批新源码在旧owner正常退出后衔接，继承04固定02:05截止，不能重置。详见scripts/tight_fresh_20261115_05与最新handoff。

# Tight-Insertion：2026-11-14 当前授权与工作合同

本文件替代此前叠加的任务说明。历史授权及证据保留在 Git 506d185 和研究结果文档，旧run的截止/预算不延续到新任务，也不重启那些run。

## 2026-11-15 新增执行要求

最新用户要求：减少繁琐重复检查。每次实现变化采用统一短检查入口，复用未变化的已有证据，通过后推进真实采集、训练和自主执行，不扩展无关检查矩阵。

前端训练采用显式新版本：从所有合法动作时刻随机抽规划起点，完整真实历史每visit重算一次；部署与后端计划仍commit12/5Hz。A的手物几何监督需要轮换、明显增加样本并记录不同构型覆盖；先少量真实更新测耗时和梯度，再决定半采样或较小轮换批次。保持平均损失、实际optimizer/EMA更新次数和独立验证，不把增batch后减少更新算作等量训练。

持续扩大窄域有效成功纠偏并完成真实采集→前端→缓存→后端→完整自主。约5分钟50条是待争取的工程目标，不是已实现或放宽质量的理由。必要短检查后推进，复用未变化证据。用户允许主Agent观察完整一轮及下一轮交接正常之后，由有界普通脚本在后台继续；必须先验证交接与清晰阶段/失败记录，提交实际代码、脚本和小报告。旧session截止与现场不改，新session独立登记。

## 本轮目标

2026-11-14接续补充授权：朝100–300条有效新增轨迹推进，可增加新的训练初态、完整专家和真实纠偏，验证family保持独立。按证据修改网络、监督、动作表示和采样，允许简单完整学生诊断。必须把可扩展多环境、批量视觉/网络与必要I/O优化接入真实采集→训练→自主评估；8/16/32/64/128及更高由实测有效吞吐和资源决定，没有四环境上限。允许显式版本化的TiledCamera/GPU物理/线程后端验收，保留资产、120/60/5Hz、29D目标及任务成功语义；不同后端不要求逐位轨迹相等。当前域只有毫米级初态变化、未启用质量随机化、摩擦倍数固定，不能宣称广泛物理泛化。

从约30条完整、自然成功、不同初态且动作顺序一致的示范重新训练两个完整学生：A预测实际q_reached，配几何与后端；B预测专家绝对目标命令，配同类后端。DP和后端随机初始化，通用视觉预训练可复用。后续真实DAgger纠偏不受30条限制。采集/训练/损失/采样/网络/历史/接管/性能优化按证据决定，用户已授权直接实施及推送。

## 不变的实验语义

- 保留已确认资产、原生物理/PD/任务成功标准，120Hz物理、60Hz控制、5Hz DP规划，最终29维绝对关节目标rad。不得通过改变判据、重置或物理来改善结果。
- A的q_reached[t]明确是transition t结束的q[t+1]（1/60秒后），不是最终稳定构型。B与后端监督u_expert[t]是该transition实际专家执行的目标，不是后态位置。当前未来标签不进学生条件。
- 每个控制tick的真实三视角RGB-D必须进入学生视觉处理及持续历史，后端读取本tick更新后的视觉；不能以每12tick采样或重复旧特征冒充60Hz输入。DP去噪仍只每12tick。
- 学生前缀留作真实历史，不伪造专家标签。接管后重新观察真实演化，不拼接旧失败后段。队列过渡/标签mask与真实执行对齐。前端更新后刷新后端所依赖计划和可训练特征；仅严格同源的冻结视觉特征可复用。
- 每个环境独立episode/时钟/记忆/reset；子目标不是episode reset。可见初态相等或seed相等不意味着PhysX隐藏状态精确重放。
- 训练统计仅用本轮训练示范；整初态family划分，独立验证/开发，旧test22保持隔离。报告各阶段拿起、拿稳、转正、对孔、插入及完整自主结果，专家救回不算学生成功。

## 自主执行与资源

- 整张GPU可用于研究，不沿用旧50%利用率上限。保留内存/显存/磁盘安全余量，保护无关进程，不做全局驱动/系统更改。
- 主Agent持续分析阶段结果、故障和下一步；普通脚本负责持久作业/事件记录。启动后返回不等于完成监督，禁止用固定脚本假称科研判断。
- 已登记run的截止和现场保持原样。完整阶段审查后，主Agent可按本轮持续研究授权建立独立的新session，记录来源pair、数据、旧截止和新阶段预算；无需为已授权的常规衔接重新询问。新session不是旧run续期，未完成机会不得静默重跑。
- 旧学生权重及失效衍生缓存可按依赖清理；成功专家、原始示范、通用视觉预训练与必要原结果不可删除。删除有清单与真实释放字节。
- 已运行代码固定source commit/源码hash，不热改活动worker。失败现场保留，修复采用新attempt身份，禁止静默覆盖结果或重复计数。
- 公共修改审查并测试后，只推origin/feature/tight-state-foundation，禁止force/upstream。大数据/权重/视频留本机。每次实质变更同步CHANGELOG、合同及实际结果。

## 当前入口

先读research_docs/USER_RESEARCH_DIRECTION.md与research_docs/README.md，再读CONTRACTS、PERCEPTION、DAGGER_METHOD和SESSION_HANDOFF最新条目。本轮隔离树bc+rl_RSS_continuous_vision_20261114从远端506d185建立；开始时GPU无研究worker，旧工作树脏文件保留。未完成验收、训练与真实闭环前，不能宣称路线有效或问题已全部解决。
