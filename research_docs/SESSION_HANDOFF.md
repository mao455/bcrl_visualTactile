# 当前会话交接：有限插入数据集（2026-10-28）

## 最终交接 — 2026-10-30录像与完整参数核算完成

本轮新增6条A/B/C自主/专家roll-in观看视频，seed981002，每条601帧/60fps/1344×316。
全部原生timeout，无子目标/抬升；自主gate0，局部step88交接、513学生步、gate6/10/7。
局部expert prefix/switch保存字段精确匹配，但视觉/nominal计划非逐位重放；不宣称触觉收益。
6新＋4旧视频完整解码/PTS/hash通过，索引`outputs/visuotactile/demos/residual_20261030_01/README.md`。
完整报告`EXPERIMENT_GROUPS_AND_PARAMETERS_20261030.md`；两份小型去敏JSON在evidence，视频/权重/原日志未发布。

实际构造计数：B0 2,003,085，B1 2,166,613，B0＋Residual3,347,682（训练489,125）；
DP hybrid89,317,578、real328,915,202，RDP含AT198,177,884，ImplicitRDP95,006,541。
57个导入官方源码SHA全部匹配固定revision；外部未加载作者权重/运行任务，不把计数当成功率证据。

监测09:02:23–09:27:59 UTC、25.61分钟，均值27.9960%、完整60秒最大60.3794%，四段窗口超50%。
**本轮没有全程满足双50%约束**；自身准入等待250.644秒、主动节流763.655秒，未动外部作业或原控制。
缺测0/max gap1.122秒；最低空闲VRAM11,869MiB、RAM26.081GiB。单列recorder工作窗口均值27.8784%。
全部自有录像/monitor/计数/测试进程已退出，无新训练，不自动追加录像或重新训练。
全套85项＋新增窗口测试的12项针对性通过；原100正式/旧11残差trace、manifest/best/report SHA均未变，
native及sim wrapper相对fe9a634无diff。原生日志kinematic错误/过时API告警保留，根因未由本轮证明。
录像launch提交503f7a8，参数核算提交1a24bf2；后续只读统计与文档不热改正在执行的策略。

Canonical 组名已补齐：G00=B0/A、G10=B1、G01=Residual B；G11 尚无合法训练或录像。
C 只用于保留真实触觉 gate 的连续力屏蔽诊断，不能重命名为 G11，也没有被加入成功率分母。

## 当前任务 — 2026-10-30实验录像与参数统计

参数核算已完成：四个固定公开示例的完整构造/参数及57个源码SHA核验见
`EXPERIMENT_GROUPS_AND_PARAMETERS_20261030.md`与`evidence/parameter_census_20261030.json`。
本机录像前3条自主已自然601步超时、gate0，正在专家roll-in；当前结果不是最终6条完成。
连续资源曾出现完整60秒整卡均值60.38%，本任务按45%门控等待；不能宣称本轮全程<=50%。

用户新授权代表录像与完整参数对照报告；沿用已完成权重，不重新训练。
预分配seed981002 × A/B/C × 自主/专家roll-in共6次观看启动，原12诊断与正式0/50不扩分母。
新目录`outputs/visuotactile/demos/residual_20261030_01`，2小时含等待；旧B0/B1四视频复用索引。
接手fc590341，origin feature/trained-tactile-residual一致、工作树干净；本项目无活跃实验进程。
当前为实施记录，视频/统计结果待完成后追加；不能把旧完成记录当本轮录像完成。

## 最终交接 — 2026-10-30 07:15:56 UTC，残差v1本轮结束

全部自有缓存/训练/诊断/monitor进程已退出，不自动重启或增加样本。完整证据见
[CORRECTION_TRAINING.md](CORRECTION_TRAINING.md)、[比较报告](POLICY_COMPARISON_20261030.md)、
`evidence/residual_20261030.json`；原400/best/正式100及0/50保持不变，旧100 SHA再次全部匹配。

- B0 v2完全冻结；Residual B/C各5完整epoch、7,875更新、320独立episode/1,600完整遍历。
  B验证选epoch5、C同轮；val .339453/.340667，test40仅一次 .336392/.337817（A .659997）。
- 12次预分配启动已用完：6自主均601步timeout、无gate；5个完成的roll-in亦601步timeout；
  均无首子目标/抬升。第8条B981001在step120、学生接管前switch核验中止，不算任务失败。
- 只接续了9–12，未重跑8。C981001与旧A prefix/state不匹配，明确非配对；981002 A/B/C在step88
  交接，prefix及所有保存switch字段精确相同，但三者仍失败。只有这一个完整严格匹配的局部三联。
- 两次运行位于同root下`closed_loop`与`closed_loop_remaining_01`，原失败证据保留；第8条旧异常未存
  prefix数组/逐字段state，不能补造。新runner已保存initial/switch/partial与物理单位解释，不改原生wrapper。
- 连续5,606.67秒：整卡均值22.0602%、完整60秒最大37.1971%，缺测0、最大间隔1.26857秒；
  最低VRAM空闲11,875MiB、RAM26.227GiB。离线首GPU阶段完成上界53.62分钟，总93.45分钟，预算未重置。
  早期wait分拆缺口如实标未知，不能声称硬quota/进程归因。外部生产PID未触碰。
- 全套82项通过；400缓存/11完整诊断hash和memory/cursor/ack/旁路核验通过，原控制器/研究wrapper无diff。
  当前是监督残差，未实现critic/replay/SAC，不声称触觉收益、去预加载或任务成功。

最终checkpoint B `0ce3ce5c…36ada8b`，C `8750ebb9…e3899be`；完整hash/config/source/normalizer见报告与权重。
首次闭环launch `000b069`，未启动分配接续launch `86424fd`；后续只增加只读统计和文档，不热改执行模型。
公开在`origin/feature/trained-tactile-residual`，不合并main、不上传outputs/权重/轨迹/原日志。

## 当前恢复边界 — 2026-10-30 06:56 UTC第8条中止

闭环原进程已退出：6自主+roll-in A自然终止，第8条B在step120接管前switch核验失败。
prefix相同、总state差4.085026；旧异常未存逐字段state，来源未定，不能当策略失败或篡改为有效配对。
本轮只接续原9–12未启动分配；第8条占预算且不重跑，仍最多12次启动。
新目录`closed_loop_remaining_01`显式记录每字段差异，不匹配也只标非配对诊断，不改变阈值/物理。
连续monitor仍运行，原失败现场与7条完整轨迹保持不变；最终需区分11条自然终止与1条中止（若余下顺利）。
不得因旧Kit退出码0认为12条已完成。读取新目录status和真实进程判断后续状态。

## 当前进展 — 2026-10-30 06:41 UTC，离线完成/闭环执行中

`residual_20261030_01`的5轮完整训练与一次test40均完成。B/C各7,875更新、1,600完整episode访问，
B验证选择epoch5、C同轮；val MSE .339453/.340667，test .336392/.337817。
全套80项通过，base无梯度/张量不变，原100正式trace哈希再次全部匹配；旧0/50不改。
固定12条诊断已开始，首条自主A为601步timeout、无gate/抬升/首子目标，不能提前下最终结论。
连续monitor仍运行；查root下`closed_loop/status.json`、`trials.json`、`admission.json`判断真实进度。
不重启、不重置90分钟/6小时预算，不追加trial。待本轮12条结束才停止monitor并生成最终报告。
当前代码提交`df09419`已推origin功能分支；闭环launch使用`000b069`，之后只改报告汇总，不热改运行策略。

## 当前进展 — 独立残差v1，2026-10-30 05:55 UTC

本轮在`feature/trained-tactile-residual`从fe9a634接续。新增独立residual1训练/执行/缓存/连续资源监测链，
正式B0 v2冻结；视觉frame encoder复制并冻结，Correction视觉/触觉/intent/motion GRU独立训练。
24条pilot完整缓存，2轮B/C各160更新；val8加权MSE .727845→.623025/.620782。
真实13步首步/replan6/12/teacher execution ack检查通过，34项关联测试通过。
当前全train320/val40缓存正在生成，完整训练和闭环尚未执行完；不得用本快照冒充最终结果。
本轮root=`outputs/visuotactile/run/residual_20261030_01`；只读`session.json/resources/resources.json/cache/index.json`判断实时进度。
固定12条：seed981001/981002×A/B/C×自主/真实expert roll-in；不并入正式100，不改变原0/50。
详见[CORRECTION_TRAINING.md](CORRECTION_TRAINING.md)。最终状态将追加于本条之上。

## 当前授权 — 2026-10-30独立残差Correction

用户已授权独立Correction训练/执行与最多12条诊断，覆盖下方不要进入下一阶段限制。
正式B0 v2冻结、独立Correction真实反馈，原生及sim wrapper只读；GPU双均值<=50%。
先完整train16/val8，首GPU阶段90分钟、总窗口6小时，随后最多5+5epoch按验证/预算决定。
原400/best/100正式trace保留；本轮代码/MD/小统计已获origin功能分支推送授权。
接手HEAD fe9a634，工作树干净，原项目训练/评估进程未运行；其他项目进程不触动。
本节当前为启动记录，后续追加实测状态；未把旧0/50改为新结果。

## 最新交接 — 2026-10-30独立审查完成

用户本轮要求独立核查B0/B1失败、实现真实自身行动历史、有界验证并推送用户fork功能分支；
这是当前授权，覆盖下面仅视频/旧不限GPU/永久none的历史快照。完整依据见
[INDEPENDENT_AUDIT_20261030.md](INDEPENDENT_AUDIT_20261030.md)与小聚合
`evidence/independent_audit_20261030.json`。全任务结束，不自动继续训练、采集或评估。

- 原400条manifest、best权重与100条正式NPZ的SHA再次核对未变；B0/B1仍各0/50、全部601步timeout。
  训练10个源码hash对应76bb58e，评估13个对应dc77a6b；无动作目标公式或chunk索引错误证据。
- 新command版本3/history1显式`action_history_mode=issued`：过去clipped/delayed命令及时间进入持续状态。
  训练源为专家过去已执行teacher forcing；部署只读学生真实env.step后ack。B0无触觉，B1保留力/FK。
  旧模型默认none/version2，不可改flag直接升级；未来B2混合/correction信息仍需独立权限。
- 冻结4完整前缀DDIM8/20/50/100；两条24步前缀各240更新。拟合误差下降，但mask历史几乎无差别。
  同seed981001共7条诊断：旧8、旧100、新历史各B0/B1均601步timeout；同入口原专家217步成功。
  新历史600个有效反馈的命令/延迟/时间与实际执行精确一致，但接近更差、无抬升，不能称性能改善。
- `audit_rollout_history_20261030_01`在scene/动作前发生libomniclient FileThread SIGABRT，保留完整现场；
  唯一有界恢复`_02`成功结束。未改共享库/系统，不把软件启动错误计为任务失败或宣称根因已修。
- GPU已监测区间加权20.225%、完整60秒最大35.288%；样本缺测0、最大间隔12.303s。
  任务间隙与崩溃尾部未监测，未算0；无硬配额/进程归因承诺。最小空闲VRAM17577MiB、RAM31.814GiB。
  当前50%整卡双均值、4GiB/10GiB余量继续有效，禁止沿用10月29日不限GPU授权。
- 原生控制器及研究sim wrapper本轮均未修改；历史原生差异仅7行只读trace hook，world_hold历史施力
  兼容行为单独说明。全套73项、后续相关22项通过；无完整重训/重采/第二轮正式100条。

本地诊断目录位于`outputs/visuotactile/run/`：`audit_20261030_01`、`history_probe_20261030_01`、
`audit_rollout_old8_20261030_01`、`audit_rollout_old100_20261030_01`、`audit_rollout_history_20261030_01/_02`。
后者表示两个独立目录后缀，不是覆盖重跑。复现见RUNBOOK，每次必须新目录。
公开分支`feature/baseline-audit-action-history`，用户已授权本轮push；仅代码/MD/小统计，
不上传原始输出，不合并main。下一阶段先完整演示拟合/历史消融（总90分钟上限），通过才讨论
5epoch×2、固定320/40/40、6小时阶段训练；本轮未启动这些后续实验。

## 最新交接 — 2026-10-30：实验完成，视频与GPU限额更新

10:22核查真实报告、100个NPZ、best SHA和systemd：两模型50epoch完成、best均epoch49；
50个合法配对共100条，全部601步timeout、首个子目标命中0，各0/50，无删失或软件错误。
39非法初始化前置排除。训练及离线指标06:29:50/06:46:10完成，正式评估和service于07:39:36结束。
旧runner.log traceback属于昨日17:23失败启动；已核对mtime，不是此次最终运行错误。
完整结果见BASELINE_TRAINING顶部和evidence/baseline_final_20261030.json。不要重复启动训练/采集。

用户最新要求：B0/B1各2视频，恢复平均GPU<=50%。本次选最前两个合法正式seed981001/981002、
实际best，单持久场景串行四条录制，原质量/控制/物理不变，非新正式评估分母。
目录 `outputs/visuotactile/demos/baselines_20261030/`，日志 `outputs/visuotactile/run/b0b1_videos_20261030.log`。
本任务unit `p2p-b0b1-videos-20261030.service`；PID需查实时，不引用旧训练PID。
ResourceGuard已默认50%均值上限、45%准入余量；4GiB VRAM/10GiB RAM保留，不操作其他任务。
新增record-video流式编码不积攒整条RGB，不写reset帧。仅新录像与Markdown更新，无新训练。

**10:36完成补记：** 四条601帧MP4已交付，全部原生timeout；完整解码/时间戳检查通过，
三视图0/2/6/10秒快照已查看。service于10:35:17正常退出，当前无录制PID。
GPU均值29.498%、最大完整60秒38.154%、缺测0、视频总8.89MB。观看入口是该demo目录README.md。
数值对照显示B1第二条后段不同于原正式trace（最大命令差1.527、关节差0.0923rad），
其他三条更接近；同权重hash/合法初态不代表逐位执行一致。详见video_checks.json，原50组结果不变。
68项测试通过；本轮只本地commit，不上传视频、原数据、权重或日志。

以下10月29日训练中快照和不限GPU授权均已被本节覆盖。


## 最新有效交接：2026-10-29 B0/B1后台训练

**18:54新加坡时间覆盖以下快照：** 两组均已完成epoch5/50、正在第6轮，保持17:34以来同一
runner2429427、B0 2429429、B1 2429455；本轮未重启训练。每轮64,825训练步、756,780个29D
监督目标、1,221更新。第5轮验证loss B0=0.353511、B1=0.350644。源码10个launch hash匹配
76bb58e，该提交已按本轮授权推送origin/main；未来评估使用后续commit协议2，不反写训练身份。

近期完整epoch约15.6/16.1分钟，50轮估10月30日06:31/06:58，离线最终指标另留10分钟；
对外估两组训练及离线指标07:00–07:30完成，正式50配对约10:00–16:30完成，均新加坡时间。
评估时长依赖原生601步阶段超时和不同任务进度，首两组正式结果再校准；不是硬截止。
旧15小时来自100条都跑满2400步的CPU网络外推，不是测得平均episode9分钟。

本轮评估改为CUDA FP32网络（含三路图像、memory2、DDIM8），CPU物理/RTX渲染不变。
仍一个持久场景，两个模型加载一次，无多环境配对新实现；保留全部50组、质量和原超时。
去除未使用的原图滑窗复制但每步记忆不丢。新单组96+96步真实CUDA接线完成（主动删失），
初态差0、命令/submitted/clipped逐位一致、原延迟0/1/2、时钟误差<7e-12s；不支持成功率结论。
先前reset前取metadata的尝试失败已修复，保留原错误目录；第二次成功。65项测试通过。

正式自动入口仍由当前runner在两训练report完成后启动，无须重启。校验真正best路径/epoch/hash；
不满配对数不写report.json，防Kit退出码掩盖异常。正式前两组计入50并更新ETA。只留小型NPZ与
首配对稀疏PNG；没有100条全RGB-D或重复MP4。默认CUDA的设备/源码身份写plan和trials。
详细表和关键测试对应见BASELINE_TRAINING末尾，本次公开小统计为evidence/baseline_review_20261029.json。
资源限制仍只有内存/显存；GPU50%不再恢复。本轮短仿真退出，训练继续；不存在运行中的正式评估。

下列17:46及更早段落为历史，不再表示最新代码发布权限、进度或ETA。

**17:46覆盖下面17:41快照：** 两组已完成epoch1/50，进入epoch2；每组首轮1,221更新、
64,825有效步，验证8,670步。best.pt已实际加载通过，val loss0.634335/0.618740。
完整首轮训练+验证约20.3分钟，剩49轮约16.6小时（另离线最终test数分钟），暂估10月30日10:30
新加坡时间训练完成；50组仿真评估另按开发速度估至多预算对应约15小时，非墙钟保证。
运行训练源码与launch记录SHA全部一致；之后只增加status读取命令和尚未启动评估的资源检查，
无需再打断有效训练。

先读 `BASELINE_TRAINING.md` 与 `RUNBOOK.md` 顶部。400条已完成、不重新采集。
最新用户允许全部GPU算力及清理确认无用进程；内存/显存要留余量，外部有效生产任务仍保留。
本轮B0/B1各50epoch、320/40/40、命令监督、no-own-plan，之后自动50组配对评估；不训练Correction。

17:41新加坡时间快照：两组均epoch1，925次更新、49,103/64,825训练步；尚无完整epoch。
service `p2p-dp-command-20261029.service` 活跃，runner2429427、B0 2429429、B1 2429455。
运行根 `outputs/visuotactile/run/dp_command_20261029`；实际读取status，不把此快照当实时。
首次并行任务2425777/2425780/2425801已退出；17:34在51组/372次更新保存后恢复，已保存工作未丢失。
此前stop循环反复TERM使B0清理退出码-15，但checkpoint已完整保存并真实成功恢复；现改成只发一次TERM。

RAM高/硬阈值12/16GiB、组内禁swap、allocator每模型36%、块前4GiB显存/10GiB RAM余量。
恢复后整卡采样均值约55–61%、最大完整60秒约77%、显存约4.2GB、资源等待0；最新授权允许>50%。
累计速率外推训练+离线test尚约19小时（约10月30日12:45；未用完整epoch校准），正式仿真评估另算，
开发运行推算预算上界约15小时。后续metrics会记录实际完整epoch速度，不承诺当前ETA。

61项回归通过；真实119/509步共628步batch/状态一致性、真实分支梯度和8次诊断更新、在线checkpoint
回放与2组真实仿真通过。4条开发轨迹都600步预算删失，不是正式成功率。真实last.pt均可加载。
单组恢复测试中下一更新参数/RNG精确匹配；没有声称恢复任意episode中途隐藏状态。

本轮代码在 `feature/command-training-pipeline` 上整理、本地提交；未获本次新推送授权，暂不push。
原始数据/checkpoint/日志/本地IDE设置不提交。外部小物体抓取PID2180305未触动。
前会话已暂停异常C++索引子进程2423801，并以本地设置禁用C++索引；不能把它当训练PID。

下方各段均为历史快照，尤其旧13:16 UTC记录与本机实际时区不一致，不能用于当前ETA。

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

## Background launch (2026-10-29 13:16 UTC)

Both jobs were then relaunched with `setsid`, independent stdin, and
`--throttle-ms 1200`:

| variant | PID | output | observed state |
|---|---:|---|---|
| B0 | 2347815 | `outputs/visuotactile/run/dp_b0_20261028/` | epoch 1, 100 chunks logged |
| B1 | 2347938 | `outputs/visuotactile/run/dp_b1_20261028/` | epoch 1, 100 chunks logged |

The first 100 chunks took 179.6 s (B0) and 181.3 s (B1), which gives a rough
75–80 min/epoch and 62–67 h for 50 epochs before validation overhead.  This
is a running estimate, not a completion promise.  A 60 s sample window after
launch was about 14% GPU utilization (peak sample 54%) and 19.5 GB of 24 GB
VRAM; the protected workload was not signalled or changed.  Check live PIDs
and logs before assuming completion; a `report.json` is the only completion
marker.

## 2026-10-29 — B0/B1 batch validation and guarded restart

The previous parallel B0/B1 attempt exited in epoch 1 without `last.pt`,
`best.pt`, or `report.json`; it is not resumable. Its last logged losses were
B0 0.710049 and B1 0.708066 after about 2,100 chunks. No traceback or training
checkpoint was found.

The current uncommitted implementation adds independent episode packing with
masked tails, BF16/FP16 support, atomic epoch checkpoints, and a process-local
resource guard. The guard waits only the trainer when whole-card free GPU
memory is below 6 GiB or host `MemAvailable` is below 8 GiB, and records its
samples. It does not signal or modify the protected external program.

Real `close_v4` data validation used a 119-step and a 539-step production
episode. B0 and B1 B=2 BF16 forward/backward/optimizer checks were finite, with
condition shape `(2,32,128)`, losses 1.134418 and 1.130959, and nonzero
parameter gradients. Fixed-weight singleton versus packed replay matched
valid conditions within 4.03e-7 (B0) and 2.38e-7 (B1). The full repository
suite is 50 passed with one existing requests warning.

A formal 50-epoch run has not yet completed. New runs must use separate output
folders, `--batch-size 2 --amp bf16`, and are complete only when `report.json`
exists. No closed-loop or success-rate claim follows from these checks.
