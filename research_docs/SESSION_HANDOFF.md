# 当前会话交接：视觉—触觉研究

## SDK关闭超时后的局部关闭候选（尚待真实验收）

第2个start的expert完整成功181ticks，17个共同控制/物理字段与原V3逐字节相同，
但SDK关闭超过60s，原PID最终exit−9。未记为策略失败，也未算生命周期通过。
本机实际导入的Lab SimulationContext STOP回调含“停止后持续render直到再次play”的循环；
新增独立worker仅在全部episode和writer结束后，提前解除该回调，再调用原env.close/app.close。
未修改共享SDK、活跃期物理/控制/随机流。新增10秒周期关闭堆栈、env/app关闭阶段记录。
这只是针对已知关闭风险的候选修复，不能先宣布唯一根因或已修复。
4项生命周期CPU测试通过。前两次记录/source/plan均存档，原deadline和5starts不变；
剩余3次必须全部连续完成，任何再失败都会使本预算下的门控无法通过，不额外补预算。

## Kit监控器权限缺口修复（首个start仍计入预算）

首个expert在场景就绪前被owner终止：监控读取Kit辅助进程/proc/PID/fd返回PermissionError，
不是已证实的FileThread崩溃。原日志、exit−15、launching状态和owner traceback保留。
只修新监控：不可读FD/inotify计数记null及permission缺失，仍跟踪PID/starttime；
退出时再检查整个专用systemd cgroup，不能遗漏被重父化的子进程。
3项针对性测试通过。显式恢复登记只允许这个已识别的pre-scene监控错误，旧SDK/模型/物理不改。
原start计数1/5和原session deadline不变；后续expert/student/expert使用全新attempt目录，
不自动无限恢复，也不把已失败首次start覆盖成成功。

## 2026-11-04 approved resource fix, new Kit gate

Parentdocs exclusion approved and applied. preflight03:256/256,34756watch references,
resource admitted. New state_image_kit_gate module/service requires realSDKclose and
three consecutive complete independent expert/student episodes,<=5starts. No training.
Read actualroot/kit_gate/status.json and attempts.json; new2CPU acceptance tests pass.
Old blocked/resource and pending-permission paragraphs below are historical snapshots.

## 2026-11-04 StateDP/ImageDP request: blocked at resource preflight

Worktree bc+rl_RSS_native_v2, branch feature/native-dp-training-v2, starting2dc8b58.
Read STATE_IMAGE_DP_IMPROVEMENT. Newroot outputs/visuotactile/native_dp_v2/state_image_20261104_01
contains preflight01/02, protected_checkpoints, editor backups, queue_intent and status.
Actual two private256-watch probes allocate0: ENOSPC. Readable watch reference count
524190/524288, mainly anotherVSCode whole-docs workspace; do not call it unique kernel count.
Only own-repository editor exclusions changed; other-project deliverables exclusion in
parentdocs is a concrete pending scope question. No process killed/global changes.
0/5Kitstarts,0trainingupdates,0/200supervised; no service/owner. CPUtests5passed and
three old required checkpointSHAs verified. Queue intent is not implementation or readiness.
Resume by resolving watch pressure then new no-overwrite preflight, <=5Kitstarts with
>=3consecutive full expert/student lifecycle successes. Only then implement/test/calibrate
newlosses and sequentialS thenI training; oldH24 already hasGRU256. Do not restart oldruns.

## 2026-11-04 H24诊断后台结束

本轮后台已退出，状态`software_missing`；8个attempt软件完成，软件缺失/故障记录1条（单列，不计任务失败）。
已核验录像组：T_train_10000_precontact, T_train_8000_precontact, V1, V2, V3, V4, V5, V6；8000真实原生成功录像：V1；指定六段缺失：无。
原始七个受保护checkpoint最终SHA全部匹配；无参数更新。原deadline/启动预算未重置。
本轮首动作前记录器错误及恢复保留，恢复源码`fe99624195d9f9fc9c8521c1617ccf473fe471f1`；详细结果与视频索引见H24_CLOSED_LOOP_DIAGNOSIS和evidence/h24_diagnosis_20261104/completion.json。
不自动重启已完成诊断或追加训练/录像。

## 2026-11-04：恢复已实测，V1原生成功录像已生成

recovery service active，owner2534924，NRestarts0；V1_recording_recovery实际342ticks成功，
subgoals2/alignment，完整171帧MP4已存在；V2在启动检查时运行中。此PID仅是本次核验快照。
初次软件失败完整保留并计预算；未改checkpoint或控制语义。
持久owner继续六视频和条件takeover/报告；独立publisher在最终完整性/媒体/白名单检查后发布轻量成果。
先读root/status.json、results.json、publication.json，避免再启动同一批。

## 2026-11-04：H24新诊断记录器有界修复

原V1首次Kit成功进入场景；新增flat policy HDF5字段用了nested writer，首动作前empty-group错误。
原失败/partial/初态保留，未计任务失败；旧7checkpoint和其余冻结源码仍通过完整性核验。
只修新worker写入，5项CPU测试通过；tools/h24_diagnosis_recovery与独立recovery service
在原4h截止/15start预算继续。原失败计V1第1次，恢复是第2次，不另加V1_attempt2。
原owner结果、报告及源码归档于recorder_recovery_01，未删除旧失败；运行状态查看新root/status。

## 2026-11-04：H24诊断与录像新任务

新增tools/h24_diagnosis_{common,worker,run,report}和独立service；原训练/控制/模型源码未改。
先六段train成功初态/固定validation的8000、10000、Expert视频，再有界真实expert-prefix接管。
四项新增CPU测试通过；已有数学和batch测试不重跑。初步历史只读分析见H24_CLOSED_LOOP_DIAGNOSIS。
新root h24_diagnosis_20261104_01、总15starts/4h/两次连续同类启动故障停止；
本段为实现准备记录，实际启动和结果必须读status/worker/视频文件，不能推测已完成。
后续由持久owner自动生成图、视频清单和完整性校验；不运行旧owner或继续参数训练。

## 2026-11-03：旧失败产物清理已完成

按用户最新要求删除41个已分类旧文件，释放7,157,032,011 bytes（6.6655 GiB）：
6个完整归档内容与保留best逐记录SHA相同的last副本，以及35条已确认失败的旧MP4。
本轮H24及原Sample h24_train完全保留；全部数值轨迹、输入HDF5、结果/崩溃日志、
原始数据、缓存和其余权重保留。17,983个保留文件metadata核验无变化。
旧录像链接可能已标记删除，不能按历史解码表声称文件仍在。
详见CLEANUP_AUDIT和evidence/pre_h24_cleanup_20261103.json；各删除目录有说明。
本地执行脚本/压缩基线在outputs/visuotactile/tmp/pre_h24_cleanup_20261103。
本次没有训练、Kit启动或修改实验结果；不要自动重建删除文件或重启已完成实验。

## 2026-11-03 H24对照最终：Sample到10000，train一次成功，validation全无成功

21:51:21UTC+8 owner/monitor退出，service inactive/exit0，最终software_missing（训练complete）。
Epsilon3000、暂定Sample连续5k/8k/10k全部完成，公平性与恢复状态通过；不再追加训练/重试。
Sample3k0/1完成+11缺失；Epsilon3k0/11+1缺失；Sample5k0/10+2缺失；8k1/11+1缺失；10k0/11+1缺失。
唯一成功来自8k train seed982403，342tick原生max_successes=true/subgoals2/alignment；validation8k/10k均0/8。
10k同可观测train初态208tick fall，不能称稳定成功/泛化。60注册=44完成+7启动故障+9未启动，无failed重试。
Sample valArm/Hand3k .028269/.129117→10k .020862/.109865rad；目标FK21.06→17.33mm，
但同7validation实际接触前FK5k/8k/10k为46.86/45.89/46.23mm，闭环进展未明显改善。
最终只读核验179冻结源码、61stagehash、6checkpoint/optimizer/EMA、连续LR日志、原Sample3000保护hash。
所有自有进程退出，完成service已disable自启；原数据/权重/失败证据全部保留，无删除/新Kit/训练。
资源采样均值29.2521%、最大完整60s64.8483%，累计监控间隔4606.47s，不称全程连续/硬50%合规。
完整结论、曲线、缺失分母、SHA见H24_SAMPLE_EPSILON_COMPARISON与evidence/h24_parameterization_20261103/completed_review.json。
后续旧running/PID/授权段落仅为历史，不自动重启本轮或扩展实验。

## 2026-11-03 20:32:28 UTC+8：恢复服务已实际接续

`h24-comparison-20261103-01-recovery.service`已核验active/running、NRestarts0，
owner2342844、worker2343565，阶段`epsilon_3000_loop_worker_00`；Epsilon完整离线评估已完成。
Sample cohort按原预算完成记账：1实际完成（0成功）+2启动软件故障+9未启动，失败条目没有重试。
新monitor在核验时0.04秒内有采样，旧179项冻结源码和训练状态不改；恢复launcher提交`884dc6d`。
这只是恢复启动核验，不代表Epsilon闭环、选型或5k/8k/10k完成。后台继续既定阶段，
后续读取机器status和结果；新统计不计软件故障为任务失败。公开证据：
`evidence/h24_parameterization_20261103/recovery_startup_review.json`。

## 2026-11-03 20:27 UTC+8复核：两组3000已完成，Kit阻塞恢复

Epsilon3000已完成并覆盖train320。两组实际updates/EMA、采样顺序/游标、train/val IDs、
noise/schedule RNG和3000步LR轨迹全通过公平性检查；不比较两种diffusion loss绝对值。

|val40 / clean action / prefix12|Sample3000|Epsilon3000|
|---|---:|---:|
|Arm RMSE(rad)|0.028269|0.135959|
|Hand RMSE(rad)|0.129117|0.465200|
|Motion cosine|0.556923|0.106870|
|Motion amplitude ratio|1.325756|4.482187|

Sample的这些离线指标更好；完整闭环优势未建立。Sample仅完成train首条0/1，
接触/抬升后364tick fall，无首子目标/alignment。随后两条均在scene/action前SIGABRT。
第2次缺少std::system_error文本但有FileThread/omniClientFreeContent/abort栈，旧分类器未识别，
owner于19:52:19停止，之后没有训练；截至本次复核Epsilon闭环/选型/5000均未开始。
这不是模型数值发散，亦不证明inotify或SDK根因已修复。

修复限于独立`tools/h24_runtime_recovery.py`：保留179项冻结源码、配置、plan/session/deadline；
只在exit−6、父进程已回收、lifecycle=launching、无scene/trajectory及明确FileThread栈时
识别软件故障。不重试旧两条、不扩大预算，按原两次连续失败规则把Sample其余9条记未启动，
合计1完成+2启动故障+9未启动。旧部分汇总（曾漏列首个软件故障）及失败文件原样归档，
重新生成完整分母。接着执行原本未开始的Epsilon离线/闭环及后续暂定选型和里程碑。

恢复服务`h24-comparison-20261103-01-recovery.service`，同一owner锁/单GPU/原截止时间；
原失败服务停用自动启动防止重复。注册记录launcher/service/失败证据/checkpoint SHA。
已验收3000公平性复用其冻结checkpoint和结果，后续Sample日志超过3000不再误判为公平性失败。
4项新的故障分类/禁止重试/分母恢复CPU回归通过，原模型/数据/batch测试不重复。
本节恢复安排不代表恢复启动已通过，实际以startup记录及status为准。

## 2026-11-03 19:47:08 UTC+8：H24 Epsilon 真正开始训练

持久服务`h24-comparison-20261103-01.service` active/running，owner2311842、worker2311883，
NRestarts=0，用户linger=yes；运行源码commit`9b1edd8`已推送。
一次启动核验观察到durable250 checkpoint、实际日志500updates、320轨迹已访问，loss有限。
原Sample3000保持，新增Epsilon实际configuration除prediction_type/source/10000上限外与Sample一致，
optimizer_horizon仍19000；运行源码/plan/session/foundation/dependency SHA检查通过。
整卡可用显存18644MiB、RAM35.593GB，无内存等待；这不是完整会话资源合规结论。
39项原数值检查沿用，恢复/幂等/统计12项测试通过，本次无重复Kit Gate/模型检查/batch审计。
后台负责Epsilon3000→同12初态配对→暂定选型→累计5k/8k/10k及同口径评估；不依赖Codex连接。
目前不声称3000对照/选型/后续训练完成；读status而非历史PID。证据：
`evidence/h24_parameterization_20261103/startup_review.json`，报告`H24_SAMPLE_EPSILON_COMPARISON.md`。
当前GitHub仅启动时审核快照，未来后台产物仍需复核后推送；不自动提交未经审核结果。

后台随后已提交 Epsilon1000 checkpoint（SHA `fb5f832483857b23c471e3cffe32347d91b1d74899dd0e01fcb355afc18b502b`），val40 clean-action Arm/Hand RMSE为 0.579001/0.804972 rad；这是首里程碑观测，尚不是3000公平比较或选型结论。


## 2026-11-03：H24 Sample/Epsilon 恢复准备

工作在bc+rl_RSS_native_v2 / feature/native-dp-training-v2，恢复前HEAD8ef2a16。
用户本轮明确授权接续，覆盖下方旧run不自动新训练的限制；原root和旧实验不重启。
h24_parameterization_20261103_01现场gate_completed、4/4专家成功、复用轨迹不一致；
无本轮存活进程、无Epsilon训练目录/setup_complete。上次修改仍在未提交工作树。
Sample3000 CPU读取验证optimizer/EMA/RNG/sampler完整，SHA70783fcf…29b25c0。
沿用39项数值测试和batch审计，仅新增owner断点/幂等/日志恢复回归，不重跑Kit Gate。
新流水线采用独立Kit、固定12初态配对、暂定选型后累计5k/8k/10k；未来结果读本轮status，
不能把本节准备快照当启动或完成。报告入口H24_SAMPLE_EPSILON_COMPARISON。

## 2026-11-03 17:29 UTC+8复核：H24到3000，服务已停止

recovery2于14:39:53退出。H24完成1000/3000updates、覆盖320训练轨迹，val40 Arm
.043312→.028269rad、Hand .160361→.129117rad。1k闭环0/4全部601timeout；3k
第一条0/1、lift后364tick fall；第二条scene前FileThread SIGABRT(exit−6)，另外两条未启动。
不把缺失/未启动计作失败；2次启动重试此前耗尽，此次不增加重试或重置原deadline。
H24 6k、temporal、FK训练、最终选型和新test均未开始。当前没有本轮自有训练/Kit/monitor。
只读核验171冻结源码、37committed结果、4H24checkpoint、42历史自有PID均已退出。
新5条已有trace末端误差：1k四条接触前RMSE36.75mm，3k一条37.85mm；不能跨cohort认定改善。
已更新现有State报告及h24_partial_review/FK曲线；原数据、权重、失败文件和运行源码未改。
34测试为上次实现验证，本次报告复核不重跑。读取本节，不再用下方运行快照推断存活。

## 2026-11-03 记录器与异常退出恢复

原single_takeover_12在位置→native字段切换tick12报KeyError，随后Kit closing挂住。
用户要求继续后，仅释放已失败benchmark；原owner/monitor退出，未中断训练。
失败保留并记missing，不重跑；tools/state_motion_recovery包装原owner/benchmark，
171冻结源码/session/训练配置/checkpoint不改，registration在原root/runtime_recovery_01。
首个recovery service于14:16UTC+8退出：24tick首次启动在场景前system_error/Invalid argument
导致exit−6，没有动作；原两次启动重试已被C0和single_fit_loop消耗，不追加重试。
runtime_recovery_02保留原launcher、两条失败29文件hash；新service
state-motion-20261103-01-recovery2继续独立未开始H24，再按原门槛/预算执行temporal。
两条接管保持missing，阻止最终选型/test，不伪造成功；新仿真故障仍阻塞。
34项CPU测试通过；实际startup_review与status优先，不把启动当完成。期限原样保留。
14:23UTC+8有界核验：owner2093886/worker2094890，h24_1000真实进度250updates，
281训练episode已访问；monitor正常/1自有GPU worker。详见recovery_startup_review.json。
后续由普通service推进，不前台持续轮询。运行中产物的自动汇总仍需下次审查。
原single-fit1000通过拟合但自主0/1（230tick fall、lift/首子目标），C0/C2各0/12。
源码和结果详见STATE_TRAINING_RESULTS新节；不新增FK训练，不把失败partial当成功证据。

## 2026-11-03：追加只读FK、数据和原生延迟审计

没有中断当前state-motion服务或改动171项冻结运行源码/配置；CPU诊断见
[State结果首节](STATE_TRAINING_RESULTS.md)及其fk_readonly证据目录。
all400/81894控制目标重建完全一致；实际q FK163788姿态最大.001719mm。
train320/val40完整H32窗口没有静止主导证据；val40目标FK接近18.06mm（Hold22.09mm）。
已有C0自主0/12、专家12/12，实际接触前偏差39.07mm；全部C0达到接近代理和过滤接触，
不能声称已证明接近失败或唯一学习原因。C2完成0/12；中途读到single_fit_1000，
最终保存快照为single_takeover_12_single_00运行，
之后仍由普通owner推进，读取实际status，禁止根据此快照重启/改deadline。
FK只读train3梯度探针完成，参数未更新；没有运行时FK loss或FK训练。
当前源码冻结结束后再评审单因素H32+FK，优先替代仍未开始的弱辅助组，不能无条件追加预算。
已有Native/Position专家注入各2/2精确回放复用，不新增Kit；两合法控制合同不可重复加delay。
新的公开内容仅tools报告/测试、文档和小统计/PNG/aggregate CSV，原数据/权重/trace留本机。

## 2026-11-03 09:12 UTC+8：StateDP运动根因实验已实际启动

工作树`bc+rl_RSS_native_v2`、分支`feature/native-dp-training-v2`；实现`14ecedf`。
唯一新root`outputs/visuotactile/native_dp_v2/state_motion_20261103_01`，普通systemd owner
`state-motion-20261103-01.service`，一次启动核验active/running，当前sampling16。
真实Kit环境1tick smoke及父进程PID/GPU回收通过；原best/统计精确一致，109项CPU回归通过。
此快照没有新完整闭环结果或优化器更新，不能把后台启动当根因定位或学习成功。
六组采样×3seed→validation40筛选→原12初态C0/C1/C2→必要单轨迹拟合→有条件H24/temporal阶段，
均由同一owner按预注册门槛推进。无BC/Image/残差。原400和所有已完成run不改、不重启。
固定Gate11月3日12:11:57、训练11月4日06:11:57、总11月4日09:11:57（UTC+8）。
实际读status/blocked、各stage result、source/config/checkpoint hashes；不要根据历史PID推断存活。
STOP文件只停止本轮；不需要前台模型轮询。后台收尾输出state_motion_20261103小证据和更新State结果文档，
随后需审查结果、进程退出和公开文件，再提交推送；脚本本身不自动commit/push、不声称能唤醒Codex。


## 2026-11-03 H32位置：服务结束，最终学生0/12

Worktree `bc+rl_RSS_native_v2` / `feature/native-dp-training-v2`。
`position_20261103_01`于04:45:10UTC+8结束，服务inactive，全部自有owner/worker/monitor退出。
NativeH32短对照1000、PositionH32 StateDP19000完成train320/val40，分别选1000/19000，
冻结后offline test40各一次。两者最终闭环均train0/4、validation0/8，没有最终缺失条目。
位置结果10timeout+2fall，2lift、1train首子目标及alignment；仍无完整学生任务成功。
Image未启动（validation0/8不足预注册2/8成功Gate），H24未执行，没有新BC训练。
位置validation arm/hand=.020641/.110750rad，test=.022406/.112744；hold基线arm更好。
位置best SHA5ecc930ffd52af166e00904b0515159193d8527d9f5726908bf57967c445a416。

原400/81894labels核验，fresh专家2/2、位置重放2/2/14fields exact，学习专家12/12。
12条selected学生resetActor均同专家，提交/真实执行器target完全相同，但第0步即偏离专家；
全部手部误差在对应专家首次接触前超过.1rad。不能把执行接口通过当作策略成功。
Kit关闭隔离4/4probe通过；启动仍不稳定：6k validation单条pre-scene SIGABRT，
0/7已完成+1缺失，后续独立条目继续；这不是最终selected12的缺失或原生任务失败。

本次只读复核170源码hash、119stagehash、2best和15已解码视频hash；原400manifest未变。
未新增训练/仿真，源码未改；先前89回归测试通过，不重复跑测试。
整卡均值37.8656%、最大60秒57.2718%，1376.63秒监测间隔，不声称全覆盖/零干扰。
保留全部best/milestone和原400；先前清理2.512GB逻辑重复last/可重建cache不扩大删除。
只提交复核报告/去重小JSON/曲线，原权重、数据、NPZ、MP4不上传。
阅读STATE_TRAINING_RESULTS、KIT_RELIABILITY和evidence/position_20261103/completed_review.json。
本轮已结束，不自动追加更新/重试/H24/新轨迹/Image。旧blocked.json为已解决修订快照。
下文旧root或运行中快照不能用作当前状态。


## 2026-11-03 00:19UTC+8 — training finished, evaluation incomplete, service stopped

BC10000 andDP19000 complete train320/val40, all320 sampled; selectedEMA10000/19000.
BCselected0/12 alltimeouts/no lift; DP19k milestone0/4 with lift/fall254, selectedDP0/5
(fresh0/4timeouts, train0/1lift/fall273). Sixth startupFileThread SIGABRT before scene;
six later selectedDP starts never launched. Prior10k fourth startup gap also remains.
State training_complete_evaluation_incomplete; all owned processes exitedNov2 23:21:53.
Old12/test40/formal20 cohorts not started. No automatic recovery expansion or new updates.
Read-only review169sources/88stage hashes/2best checkpoints/14decodedvideo hashes passed.
See STATE_TRAINING_RESULTS and training_complete_review.json. No student fullsuccess.
Below active PIDs/stages are historical. Missing results cannot be counted as failures.

22:58:28UTC+8 startup verified: serviceactive,owner1297238/worker1297395;
dp_train_19000 with real resume.json from10000, exactoptimizer/EMA/RNG/sampler.
Committed progress still10000 at that snapshot; no memory pause. Runtimefae5a04
pushed and remoteSHA verified. DP10k missing fourth evaluation remainsdeferred.
No further model polling; ordinary service owns remaining stages/deadlines.

22:52UTC+8 follow-up: runtime amendment44e9424 was correctly deferred, but DP19000
configuration guard rejected the changed owner source hash before any update.
runtime_recovery_04 preserves both empty attempts and consumed automatic retry.
New validation permits only reviewed exact config SHA/source-hash pairs while keeping
all training recipe/model fields and original checkpoint/config bytes unchanged.
Numerical trainer suffix unchanged;10tests pass; actual last checkpoint10000 verified.
Read independent startup evidence/current status; service active alone is insufficient.

## 2026-11-02 22:43UTC+8 — BC10000 complete, DP10000, startup SIGSEGV

BC fulltrain320/val40 complete10000; selected diagnostic0/12 all601timeout/no lift.
DP fulltrain320 reached10000; milestone1k/3k/6k each0/4;10k0/3 with one lift then
fall227, fourth pre-scene FileThread/omniClientFreeContent SIGSEGV(-11). No finalDP/test.
Nine focused tests pass after narrow startup classification and resumed-stage status fix.
Retry budget3/session remains exhausted; preserve missing evaluation and defer it,
continue remaining authorizedDP budget without repeated updates/trajectories. Formal
test withheld while any deferred loop remains. Runtime-only amendment preserves all
model/control sources, root and Nov3 14:12/17:12UTC+8 deadlines. Read current status
for actual resume; runtime_recovery_03 archives the stopped owner and prior session.
Report/curves contain fullBC and partialDP evidence. No student task success yet.

## 2026-11-02 20:42UTC+8 — foundation passes, BC300, second startup interruption

expert12/12 legal native complete, replay2/2 exact16fields; foundation passed.
BC pilot300 train4/val4, loss.881→.562, recurrent gradients nonzero. First closed
loop0/1:601timeout/no lift/subgoal. Next pre-scene startupexit-6 has truncated
system_error/Invalid argument and no FileThread stack. Preserve evidence in
runtime_recovery_02; expand only bounded classification, not1/stage or3/session limits.
Both numerical pilots now first; classified pre-scene loop failures may defer while
fulltrain320 proceeds. Missing outcomes are not success; formaltest withheld until
diagnostics resolved. OriginaldeadlineNov3 17:12:42UTC+8 unchanged. Check status for
actual resume/DP/fullupdates; never infer progress from this snapshot.

## 2026-11-02 19:40UTC+8 — stopped startup discovered, bounded recovery

state_full_20261102_01 stopped17:19 on expert_single_07 FileThread SIGABRT(-6),
before environment or action. Prior7 experts all legal/full success; BC/DP0updates.
No resource pause. Runtime-only recovery retains root/deadline and skips all7 completed.
Known FileThread pre-scene abort retry is bounded1/stage,3/session; no started trial retry.
Missing ffmpeg also broke finish; fixed via installed PyAV,168frame video decode passed.
Read runtime_recovery_01 and current status for actual restart, never assume completion.

启动核验（2026-11-02 17:16 UTC+8）：`state-full-20261102-01.service` active，
root `state_full_20261102_01`已推进到独立专家第4条；新学生训练尚未开始。
代码 `c2e9895` 已push并核对origin同SHA。训练截止11月3日14:12:42，绝对截止17:12:42（UTC+8）。
自有进程识别通过；外部worker当时约8.33GiB显存，整卡free14.27GiB、availableRAM35.25GiB，
没有资源暂停。没有外部吞吐指标，不作零干扰承诺。后续阶段由普通脚本自动推进。

## 2026-11-02 — shared GPU/full recurrent State authorization

Latest attachment147b678c supersedes small-only and45/40/50 hardGPU gates. Read
STATE_TRAINING_RESULTS. Old state_history root02 ended blocked, zero updates;
new live-pose query identifies false ground hits from stale USD origin. Preflight
persistent expert9 executed/9success,3query rejections; freshKit982506 succeeds161steps.
Fresh Kit per controlled trial now isolates prior-scene query state. Original native
physics/reset/controller unchanged. New rootstate_full_20261102_01 owns new24h window,
pilots300each, train320 BC<=10k/DP<=19k with validation earlystop, test only after freeze.
Single GPU worker,6GiB/12GiB memory admission, recent1/5min pacing; no external throughput
claim.11 focused tests pass. Read machine status and result.json for actual updates;
never infer learning/task success from service startup. Preserve every old attempt.


## 2026-11-02 — State persistent history repair in progress

Latest explicit attachment authorizes new Actor parity, own persistent GRU, RNG-isolated
legal benchmark, smallBC and smallStateDP plus actual closed-loop delivery. Worktree
bc+rl_RSS_native_v2; branchfeature/native-dp-training-v2, baseec84bd3.
Read the five requested EXPERT_STATE_INPUT_PARITY / STATE_PERSISTENT_HISTORY /
ENVIRONMENT_REPRODUCIBILITY_CONTRACT / STATE_TRAINING_RESULTS / GIT_DELIVERY_MANIFEST.
Current ordinary owner: state-history-20261102-02.service; rootstate_history_20261102_02.
It inherits root01 deadline (no reset), first hashes/parity passed, currently legal freeze.
Root01 query-disabled admission INVALID; one expert success168steps and one interrupted
start preserved, no training. New query support+positive control fixes that exact defect.
92 tests and400Actor numerical parity are infrastructure evidence only; actual new students
not yet trained. Resource pause protects unrelated work; recorded max60s exceeds50% and is a violation.
Original endpoint2026-11-02 13:08:22UTC; automaticfinish service updates local measured tables
afterowner exits, but never automatically commits/pushes unseen results.
Check machine-readable status/results before claiming future progress; never restart old roots.

## 2026-11-02 12:01（UTC+8）— 环境/学习归因修复实际完成

当前分支`feature/native-dp-training-v2`，独立worktree`bc+rl_RSS_native_v2`；原worktree保持干净。
阅读[NATIVE_DP_ENVIRONMENT_LEARNING_REPAIR](NATIVE_DP_ENVIRONMENT_LEARNING_REPAIR.md)及
[公开证据索引](evidence/native_dp_repair_20261102/README.md)。G0-A/B/C已拆分，历史3/8不再否决400。
新single专家6/8、four8/8，fresh replay2/2（15字段逐字节一致）；历史source-world1/2。
新fresh reset含single6/8、four3/8盒重叠，全保留，不称合法分布可靠性验收；无重叠single2条expert2/2、学生均0/2。
旧State fresh0/8；StateBC1000和ActorBC1000 fresh各0/8；StateBC1000 saved train/val各0/4。
一次有据同结构预算延长至各4000updates，State train arm/hand=.02365/.03965但val=.98176/.71937；
StateBC4000 saved train/val仍各0/4、全601timeout、无lift/子目标。没有学生完整任务成功。

原52条root`repair_20261102_01`、补充8条`repair_fit_20261102_01`均diagnostics_complete；
共同原始截止未改，全程1h17m55s。全部owner/Kit/trainer/monitor退出，三个systemd unit inactive/dead。
补充阶段第一次初始化FileThread SIGSEGV发生在任何轨迹之前，原失败保留，一次审查恢复仅继续原8条，不重训。
两BC各4000更新，无新DP/视觉/test；不要自动延长或重启这些cohort。
全卡有测均值28.37849%、max完整60s54.62724%；恢复缺测438.47688s，启动到首采样13.015s；
不能称全程资源合规。最低free VRAM11.50977GiB、available RAM24.32213GiB；其他任务未改动。
84回归、836保护hash、8视频全解码通过。视频本地索引`outputs/visuotactile/native_dp_v2/repair_20261102_01/README.md`。
公开只包含代码/文档/小统计/曲线，无原始HDF5、权重、NPZ或MP4。原400/旧正式各0/20不变，删除0。
剩余科学缺口：历史4lane内部积分首差未唯一定位；State不含原Actor随机队列与persistentLSTM历史；
BC训练平均误差掩盖reset残差、闭环仍失败。无证据支持直接重跑19000步DP。

## 2026-11-02 — State-first V2 audit and isolated implementation

启动核验补充：服务native-dp-v2-state-first-20261102-01.service active/running，
owner269512，状态waiting_predecessor；源码9431b42与SHA登记一致。
best/last EMA字节相同，实际首批28starts；新updates0，未启动Kit。
新session绝对截止2026-11-03 02:43:45 UTC+8，含旧run等待；不重置旧run截止。
该核验后不驻守轮询。实际未来进度以root状态/结果为准。


User's attached task supersedes automatic three-way full training for the new experiment.
Original run02 continues unchanged; all three old test40 start markers now exist, so
future reporting must disclose prior held-out usage. No new test calls in V2 debugging.

New worktree `bc+rl_RSS_native_v2`, branch `feature/native-dp-training-v2` from3e11d62.
Read the four NATIVE_DP_*_V2/reference/failure reports. All400 original hashes and
81,894 steps audited; command/target/robot/geometry/history and six train-only
statistics reproduce with max error0. Actual native action function on all400
also returns delayed/target max error0. Initial observable and command duplicates0.
These do not establish historical physical replay or student success.

V2 has fail-closed distinct Gate fields, state-only numeric caching, diagnostic
StateBC, repeatable sampler/checkpoints/milestone hooks, single persistent Kit
frozen evaluation, shadow teacher-history comparison and a lock that excludes
resource/pacing waits. Final CPU tests/evidence: `evidence/native_dp_v2/verification.json`.

Root: `outputs/visuotactile/native_dp_v2/state_first_20261102_01`; status command:
`bash visuotactile/run_cpu.sh -m visuotactile.native_v2_pipeline status --root outputs/visuotactile/native_dp_v2/state_first_20261102_01`.
Owner waits for all old workers to exit, then runs8 expert replays, old State
train8/val8 and fixed seed/last checks. Failed G0 prevents all new training.
Visual performance/4env parity/new formal testing are not claimed complete.
Source/control/400/weights preserved; cleanup deletes nothing. Later results must
come from result/trace/Gate files, not this startup-era handoff.

## 2026-11-02 01:30（UTC+8）中期结果统计

新增 `research_docs/NATIVE_DP_CURRENT_RESULTS_20261102.md`。当前可直接复核的六条诊断
闭环为 `0/6` 原生最终成功。六条均有工件接触、稳定抓持诊断和 native lift，但均 fall，首个子目标和 native
alignment 均为0。正式 test40、60条闭环和学生视频尚未执行，不能写正式成功率为0/60。

正式训练状态：S、Official 已各完成19000 updates；History 正在约16923/19000。该统计
只读现有产物，未停止或修改后台服务。

## 2026-11-01 20:05（UTC+8）执行审计更新

用户无需再次确认：同一 `native_20261101_02` 已真实进入正式训练，当前不应停止或重启。
`S-NativeDP` 完成 `19000/19000` updates；`I-OfficialDP_formal` PID `4163572` 在
`3130/19000` 附近持续增长。服务为
`native-dp-20261101-02-dino-recovery.service`，原训练/总截止保持不变。

只读审计记录在 `research_docs/NATIVE_DP_EXECUTION_AUDIT_20261101.md`。确认的问题是
Episode 逐段重复加载、History 每段从 tick0 重算前缀、GPU lock 覆盖等待/pacing，以及
test marker 早于 checkpoint 读取；本次冻结 run 不做运行中 source 替换。新增状态入口：
`python tools/native_dp_status.py --root outputs/visuotactile/native_dp_3way/native_20261101_02`。
该工具与测试不进入 `visuotactile/` source inventory，不改变当前 run 的 source hash。
GPU 当前快照为整卡均值 `33.10%`、最大完整60秒 `46.41%`，Official worker
work/wait/pacing 为 `2051.0/79.3/138.5 s`；历史监测空档照实保留。

## 最新状态 — History真实拟合更新已验收（2026-11-01 18:17 UTC+8）

run02独立user systemd服务`native-dp-20261101-02-dino-recovery.service` active/running，
main4118213、monitor4118231、History fit1 worker4119308；运行源码commit`902b4b1`。
400份缓存首帧复现误差0，202份奇数尾帧单独overlay，原cache未改；History network attempt2
完整148tick原图在线/缓存离线condition、memory、action误差均0。16个先前完成stage复用。
单次启动检查100.49秒结束，真实History fit1 updates1、EMA1、optimizer1 checkpoint读回及RNG通过。
这是拟合Gate，正式训练/test40/60正式闭环仍未开始；脚本会在剩余Gate通过后自动继续。
总截止11月2日10:38:19 UTC+8、训练截止07:38:19均保持；无新预算，无run03恢复。
查询和真实产物见`CODE_WALKTHROUGH_AND_HANDOFF_ZH.md`、`evidence/native_dp_dino_launch_20261101.json`。
不持续轮询；下次用户唤起时一次读取实际结果。历史停机缺测23814.78秒及历史显存低余量必须保留报告。

## 最新核验 — DINO缓存数值修复（2026-11-01 18:11 UTC+8）

唯一有效root仍为`outputs/visuotactile/native_dp_3way/native_20261101_02`，原总截止
2026-11-02 10:38:19 UTC+8、训练截止07:38:19不变。run03不重启。
Official剩余Gate及late-gate真实环境preflight均完成，已有16个完整stage产物hash通过复核。
新阻断为History network attempt1的DINO cache reproduction，非Kit；正式训练尚无更新。
原缓存6图TF32-enabled卷积与新3图FP32计算不一致；精确原配置复现误差0。
Native专用因果6图前端与首train的12个完整双帧来源anchor逐位一致，奇数尾帧需单独overlay。
保留原400、旧cache、失败日志；新增400份缓存核验和尾帧修复阶段后重跑History网络Gate。
Kit自动重试计数与初始化边界已修正；ffprobe恢复强制验收。详见NATIVE_DP_THREE_WAY_RESULTS。
未在此快照宣称正式训练启动；后台启动验收另记，之后不驻守等待训练。

## 历史状态 — late-gate recovery 已启动（2026-11-01 08:48 UTC）

紧急授权已写入 `outputs/visuotactile/native_dp_3way/native_20261101_02/late_gate_recovery.json`。
原始 Gate/训练/总截止保持不变；第一次恢复因错误的 Bash 视频路径在 `video_preflight`
阻断，未启动 GPU，随后改用 bundled `imageio-ffmpeg` 和 PyAV 后端。

当前同一 run 已通过 `setsid + nohup` 脱离会话启动，`late_gate_preflight` worker
和资源监控仍存活，`status.json` 为 `waiting_resources`。preflight 尚未完成，不能
把补验标为通过，也没有新的训练/评估结果。后续只读取 `completion.json`/`blocked.json`
和真实 stage 产物；不创建新 run、不重置 deadline。

## 2026-11-01 — Native DP真实进度：Gate阻断，当前无worker

权威快照为 `evidence/native_dp_status_20261101.json`，详细结果为
`NATIVE_DP_THREE_WAY_RESULTS.md`。run02 的数据400/400、两个220步Oracle、S组
network/fit1/fit4/pilot及两条自主诊断、Official network/fit1均通过对应阶段验收。
Official fit1实际400次更新；S fit1/fit4为400/500次。它们是Gate权重，正式训练0更新。
S两条自主诊断均有目标接触、稳定抓持诊断和native lift，最终fall，成功0/2、子目标0。
History GPU Gate、正式训练、test40和60条正式闭环均未开始；已有专家视频1个（220帧，ffprobe及全片解码通过）。

run02 在2026-11-01 11:02（北京时间）的 `I-OfficialDP_fit1_diagnostic/attempt_1`
Kit初始化期间出现 `std::system_error: Invalid argument`；只有runtime_controls，
没有该阶段轨迹。旧交接把它直接称为FileThread根因，现存worker日志未给出该栈，
所指Kit日志已不可读，不能补造诊断。systemd存在inotify ENOSPC警告，但该服务随后
成功完成多个Kit阶段，不能据此断言换成setsid就能修复。进程内OmniClient开关对
两个Oracle有效，尚未证明能稳定完成所有后续启动。

**截止纠正：** run02首GPU时间为11月1日10:38:19，Gate截止13:38:19；
总截止是 **11月2日10:38:19**，训练截止11月2日07:38:19。15:51恢复的attempt2
因Gate截止退出，不是本run的第二次原生崩溃，也不是24小时总截止已过。
本轮曾按错误摘要启动run03数据复核，现已在GPU前显式停止，保留blocked记录；
其source commit实际为5032af3，started_epoch仍为null，未使用第二个24小时预算。
run02原deadline未改，run01历史失败未改，当前三个root均无自有worker运行。

`native_pipeline`现在在创建attempt/复制checkpoint/启动worker前检查具体截止，
将Gate截止、总截止和monitor退出分别报错，保留失败attempt退出码；预启动拒绝
不会重写旧attempt原因。21项Native CPU测试通过。代码变化未迁移到历史run。
若继续实验，需先有新的Kit故障证据及验证，并明确解决已耗尽的Gate窗口；不能
自行延长、跳Gate或新建run重计时。用户未要求Codex驻守，本轮不等待训练。

原活动区间整卡均值27.8459%、完整60秒最高46.4103%、采样缺失0；恢复后的汇总
额外记录停机期间17311.88秒未采样，不能称整段连续无缺测。原数据/权重和崩溃证据保留。

## 2026-10-31 — T0真实事件重放与冻结DP验收完成

最新入口`T0_REPLAY_AND_DP_CONTROL_FIX.md`，小证据`evidence/t0_replay_20261031`。
唯一root `outputs/visuotactile/run/t0_control_replay_20261031_01`，13:58:18–14:34:57UTC，原4h截止未延长。
12历史T0+4源world坐标+新expert/v1replay/v2replay3次+16DP=35starts完成；所有worker/monitor退出并查/proc。
原方法0/4；fixed delay1/4、delay+wrench v1 1/4，全部fixed命令/index/delayed/target逐字节相同。
源world2/4成功但物理超容差；四条首个object z差step1为1.19209e−7m，源四lane内部积分/contact状态仍不完整。
新seed17专家220步成功；首版实际world重投影在首次force tick174造成误差。native-request v2消除额外坐标往返，
新Kit相同样本220步成功，control/q/qd/子步q/qd/object/wrench/stage全逐字节相同；不扩展成历史全通过。

DP真实1,005步reference/native往返最大target误差4.76837e−7rad，普通adapter/history/model无需修改，无第二层delay、无训练。
G00-B/G10-B固定v1事件和正常随机各0/4，共16条601timeout，无native lift/子目标；approach proxy各3/4，grasp/align未独立证实。
新9,616逐tick输入、816个H64计划/cursor/执行反馈核验通过；stage和原始报告全部独立保存，不与正式0/50混合。
本次DP和历史组启动代码是actual_world v1，只有fresh_exact跑native_request v2；launch源码hash保留，不能回填最终代码身份。

原生仅新增显式历史index7行，原随机draw/queue/math保持；研究sim wrapper无diff。
外围修复kinematic速度API与depth_m记录别名；31项测试、12个Python AST、五条源request/body重建通过。
整卡均值20.6873%、最大完整60秒26.8527%、missing/事件0，最低free VRAM16593MiB/RAM27.0068GiB，磁盘约74.98GB。
worker累计wait125.34s/pacing193.99s可能重叠；monitor wait0不代表worker无等待。旧会话违规不撤销。
原manifest/B0/B1/正式报告/epoch40/G00-B/G10-B重新hash保持；原400图像全hash复用既有证据。
本轮交付代码/文档/小统计已获当前origin feature提交推送授权；不得上传raw outputs/权重/日志或自动增加试验。

## 2026-10-31 — 完整系统审计与局部修复完成

从a07d0aa干净feature开始。本轮主入口SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT、PROJECT_FILE_MAP、
EXPERIMENT_VALIDITY_MATRIX；audit_system本轮重读400/81894数值链、147既有trace队列、1005源RNG draw。
全部通过；原生唯一diff仍既有只读hook。T0单lane与原多lane随机流不同，不能否定原400真实成功。
文档首版60223302682af6fc66b67f9d9b404588c712e088已推送origin feature并核对远端。
随后正式H64入口/模型loss默认B，共享dp_objectives，checkpoint记loss；缺字段仍为历史A，pilot/full/resume不得静默混用。
diagnostic_recording与evaluate_training_initials补未来逐tick数值packet、memory、采样/native RNG和queue记录；
首次restore后保存环境身份。26项定向CPU测试通过，原生4lane/8tick动作/queue/RNG与固定上游相同。
没有新训练/仿真进程、新参数更新或GPU窗口（新轨迹0/12）；记录器无新增Kit验收，完整物理精确重放仍未解决。
受保护manifest、B0/B1 best、formal report、G00/G10 epoch40 EMA重新hash保持；原400图像全hash复用旧证据。
详细完成/未完成边界见主报告第12节与evidence/system_audit_20261031/verification.json。
原400、split、best、正式0/50保持；不得把旧状态段落当实时进度。

## 最终交接 — 2026-10-31网络架构、训练验收与近似训练初态诊断

新run `network_audit_20261031_01`已完成预登记4条train×T0/G00-B/G10-B共12次诊断，
2026-10-31 17:27（Asia/Shanghai）worker、owner、monitor正常退出并核对/proc；无新训练、无重试。
T0四条精确提交原记录命令，但全部命令用尽前未成功，属于删失重放，不是native timeout。
保存的observable初态恢复误差0，仍缺PhysX/contact/sensor/多环境RNG历史；延迟命令于3/1/2/1步偏离源。
T1 G00-B/G10-B各0/4，全部601步timeout，无lift/子目标，第0步动作已偏离；不能据此断言精确训练态失败或泛化差。
原T2各0/3与旧正式0/50分别保留；未新增20条T2/OOD。

实际checkpoint6实例CPU forward/backward无optimizer更新，594层记录、743参数记录；
Force/Masked均828189参数、optimizer全3730step，五遍仍小幅学习而不证明收敛。
同权重置零/过去力倒序：Force当前query arm/hand变化.01654/.00402，Masked精确0；
选定validation4条置零误差没有变差，不证明连续力收益。history协议仍past_actual_mixed_commands_v1。
400数据固定320/40/40，400不同初态/command/hash，同一任务/资产/相机/DR；没有几何OOD证据。

主报告`NETWORK_ARCHITECTURE_EXACT.md`含两页内中文核心结论、五张SVG/PNG网络图、公式、统计、论文对照；
`evidence/network_layers.csv`为真实leaf表，`evidence/network_audit_20261031`为新小统计与图。
原曲线索引`evidence/dp_loss_trt_20261031/README.md`已嵌入全部PNG并补逐图CSV来源。
保存的源初态快照为事后提取；新轨迹只在replan时保存完整输入，未保存逐次Gaussian噪声张量，限制明确。
新GPU监控997.50s、整卡均值30.3900%、完整60秒最大38.1214%、missing0，最低空闲VRAM9472MiB、RAM26.9552GiB；
无资源事件，磁盘结束审核82.45GB。旧会话资源违规不撤销，不自动重启建议阶段。
13项定向测试通过；原native/sim wrapper无diff，原400/所有权重/正式报告保持。

## 最终交接 — 2026-10-31 DP损失对照与Transformer已完成

主报告[DP_LOSS_AND_TRANSFORMER_RESIDUAL_20261031](DP_LOSS_AND_TRANSFORMER_RESIDUAL_20261031.md)
含最终12问；[图表/CSV](evidence/dp_loss_trt_20261031/README.md)、
[去敏统计与验收](evidence/dp_loss_trt_20261031.json)。唯一root为
`outputs/visuotactile/run/dp_loss_trt_20261031_01`，本地README索引两条真实诊断视频。
固定窗口03:01:47–09:01:47 UTC；实际流程和monitor于05:01:00 UTC（13:01:00 UTC+8）结束，
从固定起点用时1h59m13s。pipeline=complete，自有GPU进程全部退出；旧PID3287451/3287455仅为历史。

四线DP各额外12完整train320遍历/9984更新，每线独立pilot11更新，选定后各test40一次。
G00-A选额外epoch8，其余选epoch12；B相对A的test前12步arm误差改善3.26%/3.57%，
hand改善2.74%/0.57%，但H64 arm仍差于零命令。四线自主各0/3、全601步timeout，零子目标/原生lift。
G10-B seed981004最高+7.6863cm，高于5cm最长0.30s后掉落；没有可靠保持或插入。

冻结G00-B的400条逐tick因果缓存完成，train320共64825步/37224接触步。
828,189参数FP32 Transformer的force/masked各pilot11更新后，独立正式训练5遍/3730更新，
均验证选epoch5、各test40一次。相同缓存test接触arm/hand RMSE：base0.947673/0.587045，
force0.711644/0.577887，masked0.710323/0.576170。arm改善约25%，masked略好，未证明连续力输入收益。
两组自主各0/3、全601步timeout，无子目标/原生lift。force实际gate为6/10/191步，确有控制机会。
所有18条新轨迹actual外力/力矩均0；聚合接触不含对象身份，不能据此确认抓握工件。

**配对限制：** 保存物理初态相同，但base/force/masked在首次接触前命令已不同，全部
strict_precontact_match=false。原paired字段仅表示初态，不证明严格轨迹配对或残差因果收益。
缺少当时完整输入快照，具体来源尚未定位；不能武断归咎于渲染，也不能新增试验突破本轮预算。
用户授权的past_actual_mixed_commands_v1已落地：新G01名义可间接受残差/触觉影响，
缓存专家历史明确teacher forcing，部署双方使用真实混合命令ack。原DP A/B和旧契约不变。

选定G00-B145步缓存/在线command/reference/history最大差0；同真实输入与种子的训练后13步
封装复验command/plan/condition差0。最终残差145步CPU递推输出/记忆误差均不超过5.96046e−7。
18条trace、两条601帧视频全解码、checkpoint/保护文件hash和真实时钟/ack审计通过。
早期BF16合批缓存失败保留，残差改逐tick后才放行；现有DP合批训练与逐tick推理仍非逐位等价。

本轮整卡均值33.1988%、最大完整60秒39.4259%，缺测/资源事件/超限窗口均0；
最低空闲VRAM10.6592GiB、RAM28.8657GiB，结束磁盘约119.0GB。监控覆盖全部GPU阶段6650.96s；
软件等待有重叠，不当作GPU逐进程归因。旧阶段资源越界记录保持，不宣称所有历史阶段合规。
原400/HDF5身份、B0/B1 best、正式0/50报告和原epoch40权重保留；native与sim wrapper无diff。
未运行G11、20Hz、新roll-in或额外诊断，不自动追加训练、试验或重试。下方均为历史快照。

## 最终接续 — 2026-10-31长预测候选已结束

03:41:43 Asia/Singapore流程完成，当前所有自有owner/trainer/仿真/monitor已退出，无待运行阶段。
G00-new/G10-new各40遍train320、33,280更新；验证均选epoch40 EMA，test40各一次，3对自主初态全部完成。
六条结果各0/3、全无子目标：5条601步timeout；G10 seed981004为600步fall，曾native lift、最高+0.107093m。
指端聚合力不带碰撞对象身份，不能宣称稳定抓握。test arm命令误差1.145378/1.154378高于零命令0.955774，
绝对参考误差下降不能代替控制质量。详细表、视频路径及统计见LONG_HORIZON_20261030与
evidence/long_horizon_final_20261031.json；旧正式各0/50分母不变。

6条trace有限性、H64/replan12/逐tick执行/60Hz、3对保存初态、best/test/部署hash均复验通过；
两个601帧MP4已在流程结束时完整CPU解码，本次复核hash。原manifest和B0划分来源best保持原SHA。
自适应训练段整卡均值37.8641%；替代monitor累计16.7201%、完整60秒最大41.3339%、缺测0，
但早期RAM9.825729GiB和初版监测缺口仍成立，不标全程合规。截止预算均满足。
主任务代码运行提交e65c306，当前仅结果文档归档；不重跑test、视频、诊断或新增epoch。
下方运行PID与待完成描述均为历史快照。

## 最新接续 — 2026-10-31 02:26 UTC+8，GPU目标40%

用户要求提高平均GPU到约40%，仍保留双50%上限与资源余量。sample pilot已各完成3300更新并通过门槛：
训练四条完整示范的arm命令RMSE G00/G10为0.401126/0.419831，hand0.025333/0.026953；不等于任务成功。
两组全训练已开始，预算仍各40遍/33280更新。为应用共享自适应节流，旧owner3130112定向退出，
旧trainers3136131/3136132在2845/2852更新完整episode边界保存后退出；monitor3063789持续记录未重启。
新owner3142767，trainer3142771/3142772从原断点恢复；这些仅是此时快照PID，查pipeline.json及真实/proc。
模型/data/action/trainer源码未改，resource源码变更有resume_runtime记录；optimizer/RNG/epoch游标不清零。
新目标40%、45%准入、6GiB空闲显存/14GiB可用RAM准入；真实指标见gpu_pacing_state/resources/work。
10:00训练截止、12:00最终截止不延长；test与六条诊断尚未完成，后台仍按原计划接续。
02:26:29观测G00/G10为4950/4975更新、均第6遍；末180秒整卡均值38.4496%，最新60秒38.4583%。
恢复334.787秒均值36.5095%、完整60秒最大39.2677%、超50%窗口0/缺测0；最低VRAM18,889MiB、
RAM32.3581GiB。此前RAM9.825729GiB事件不撤销。详细统计`evidence/gpu_pacing_20261031.json`。

## 当前接续 — 2026-10-30长预测DP新训练授权

详见[长预测候选报告](LONG_HORIZON_20261030.md)。本轮授权实际新模型训练/有限闭环/失败产物清理/push。
初步root `long_horizon_20261030` 因当前命令历史泄漏/错误划分等失效；G00初步822更新不计合法训练，
进程3061303已定向SIGTERM退出，禁止执行其权重。原400/best/0/50未改。
正确替代root `outputs/visuotactile/run/long_horizon_20261030_v2`，版本2；连续monitor+单GPU锁，
直接复用原320/40/40；60Hz持续状态、5Hz全尺寸DINO、11帧/121控制/242物理样本窗口。
缓存400条与真实因果/原控制器公式验收已完成。epsilon强pilot各3300更新完成，但arm命令未优于零命令，
未放行main；新的独立sample3300 pilot正在运行。实际进程和指标须查root中的status/report，勿照旧PID推断。
完整接续审查、准入规则和后台run_long流程见LONG_HORIZON报告；full/test/六条自主诊断仍未完成。
01:48 UTC+8快照：后台owner3130112等待pilot门槛；sample恢复进程3129225/3129240，monitor3063789。
这些是快照PID，须检查真实/proc与pipeline.json。内存快照回收修复后从1115/1122更新恢复，原optimizer/RNG不变。
145真实连续步跨两秒窗口重放通过，400缓存SHA复验通过。提交前sample约1650/3300，主预算保持各33280。
资源订正：01:45:08一个RAM采样点9.825729GiB低于10GiB，不能称替代阶段全部资源合规。
快照回收已恢复低RSS；新worker准入6GiB VRAM/14GiB RAM，旧运行pilot仍按启动hash使用4/10门槛。
最终deadline2026-10-31 12:00 Asia/Singapore，训练截止10:00预留两小时；本轮新授权覆盖下方历史限制。

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

本次恢复新增12项owner/选型/报告回归通过（CPU，0新Kit）；原39项数值验证沿用。
