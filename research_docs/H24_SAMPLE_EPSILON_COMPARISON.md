# H24 Sample / Epsilon 对照

## 最终复核：2026-11-03，训练结束，闭环存在软件缺失

后台于 **21:51:21（UTC+8）** 退出。Epsilon累计3000完成；暂定选择Sample后，按原状态连续完成5000、8000、10000。
最终机器状态为 `training_state=completed`、`evaluation_state=software_missing`。没有继续训练、补种子或重跑缺失条目。
出现 **1次完整原生任务成功：Sample8000的训练初态**；所有已完成的validation初态均无完整成功。
这证明该checkpoint在一个已见初态上曾完成任务，不能据此宣称稳定控制、泛化成功或Sample统计显著优于Epsilon。

### 3000对照与暂定选型

两组真实3000更新的EMA、采样顺序/游标、train/val IDs、noise/schedule RNG及每步LR全部匹配。
同一val40、seed7091，clean-action原单位prefix12 RMSE：Sample Arm/Hand **0.028269/0.129117 rad**；
Epsilon **0.135959/0.465200 rad**。目标接触前FK为 **21.06 vs108.61mm**；
Arm重规划边界跳变 **0.02620 vs0.19668rad**；Motion cosine **0.5569 vs0.1069**，幅值比 **1.3258 vs4.4822**。
多个离线指标一致支持暂定Sample；两组diffusion loss绝对值没有比较。
3000时共同完成的validation初态为0，仅一条共同train初态完成且两组均失败，因此选型不是闭环胜负证明。

### Sample连续训练与任务结果

|模型/累计updates|val Arm(rad)|val Hand(rad)|目标接触前FK(mm)|train成功/完成|validation成功/完成|启动故障|未启动|
|---|---:|---:|---:|---:|---:|---:|---:|
|sample_3000|0.028269|0.129117|21.06|0/1|0/0|2|9|
|epsilon_3000|0.135959|0.465200|108.61|0/4|0/7|1|0|
|sample_5000|0.024556|0.120267|18.89|0/3|0/7|2|0|
|sample_8000|0.021733|0.114476|17.39|1/3|0/8|1|0|
|sample_10000|0.020862|0.109865|17.33|0/3|0/8|1|0|

每组固定注册train4+validation8；软件缺失不计入任务失败，0/0表示无完成样本。
共60个注册条目：44个实际完成、7个pre-scene软件故障、9个因原连续故障上限未启动。这个总计不是独立成功率样本。
Sample8000的成功初态 `r931005_1793209376_e02_p000059`，seed982403，342ticks；
原始HDF5末步 `max_successes=true`、trace末步subgoals=2/alignment=true均已核验。
同一可观测初态在Sample10000时208ticks fall，未抬升；不声称全部隐藏模拟器状态相同。
8000的train结果为1/3、validation0/8；10000为train0/3、validation0/8。
没有因5000零成功停训，也没有以8000一次成功跳过10000。两组3000和所有里程碑权重完整保留。

![Sample里程碑曲线](evidence/h24_parameterization_20261103/sample_continuation.png)

完整Sample/Epsilon训练诊断与验证曲线：[training_validation.png](evidence/h24_parameterization_20261103/training_validation.png)。
训练诊断固定train16，验证val40；不是按不同监督目标的diffusion loss比较。

### 空间误差、连续性与失败分析

Sample验证Arm/Hand RMSE继续下降，目标FK从21.06mm降到17.33mm；
Arm边界跳变从0.02620降到0.02278rad，Hand从0.14229降到0.13614rad，
motion cosine从0.5569升至0.6728，幅值比从1.3258靠近1.1474。
但共同 **7个validation初态、488个专家接触前对齐ticks** 的实际关节FK末端RMSE为：
**5000:46.86mm → 8000:45.89mm → 10000:46.23mm**，没有相应的明显下降。
这些是实际闭环运动误差；上表目标FK来自teacher-forced离线预测，两者不可混称。
10000的validation8条均601tick timeout，其中7条检测到物体接触，但均无native lift/首子目标/alignment。
其train3条均接触、2条发生过lift，最终2fall+1timeout。Epsilon3000的11条无lift/首子目标，7timeout+4fall。
失败主要观察为接触后缺少稳定抬升/后续进展，部分掉落；这不构成单一因果根因证明。
训练/验证误差差距扩大，但验证误差仍下降，不能仅据此认定过拟合已经解释了闭环失败。
实际轨迹曲线、逐条接触/阶段和缺失分类见[轻量summary](evidence/h24_parameterization_20261103/summary.json)。
所有测试集40轨迹未用于本轮选择或新增推理。

### Kit、退出与证据审查

专家Gate既有4/4成功，但正式评估51次独立Kit启动中7次pre-scene故障；SDK根因未解决。
没有使用未通过一致性检查的进程复用。独立恢复仅补齐已有FileThread崩溃栈的分类，未重试旧失败、未改物理/训练。
自有owner/monitor/worker均已退出，GPU无本轮进程；已停用完成服务的自动启动，未修改其他进程。
最终CPU复核 **179冻结源码、61提交阶段hash、6个里程碑checkpoint hash/optimizer/EMA计数**，
Sample10000/Epsilon3000日志连续、有限且LR公式精确一致；3000→5000→8000→10000恢复记录完整。
原Sample3000 SHA及数据身份保持。原39项数值检查、12项恢复检查、4项分类检查作为既有证据，不重跑。
整卡已采样时间加权均值29.2521%，最大完整60秒64.8483%，采样缺失条数0，
但各段停机之间有4606.47秒监控间隔；不宣称全程连续覆盖或硬50%配额合规。
本轮延续约50%调速目标而非硬GPU门槛；最低available RAM22.368GB，显存峰值12442MiB。

复核证据：[completed_review.json](evidence/h24_parameterization_20261103/completed_review.json)。
完整自动分析7.1MB留本机 `final_review/full_report_summary.json`；公开summary约1.1MB，
阶段误差按样本数汇总，实际均值曲线提供10Hz CSV及原完整分辨率PNG，不删原始证据。
未删除任何数据/checkpoint/失败记录；仅公开代码、Markdown、小统计和曲线。

### 关键源码与权重

- Epsilon/评估运行源码：`9b1edd8`，独立恢复入口：`884dc6d`；179文件SHA清单在summary。
- Sample所有续训使用原冻结工作树 `8ef2a16`；optimizer_horizon始终19000，累计上限10000。
- Sample3000：`70783fcf67344f91a68ff9dec66e42bb0125b87e8dca5192c827c49dd29b25c0`。
- Epsilon3000：`48236f836b23e74ccbc4f078f8806089f8f08bd71fc70c2184ec4f9cb704e44e`。
- Sample8000：`00783c7db01b8b269746718a3322238beb9906a88eac1fa8e9adc58bfc5b4399`。
- Sample10000：`9a7cc8c6123563f93b77256104e61e6b357bb38303ede4231fe19fddeb0e17ca`。
- 其余里程碑、配置SHA与本地路径详见summary.checkpoints。权重不上传。

以下为按时间保留的过程快照；其中“running/blocked/尚未完成”不代表最终状态。

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


## 真实恢复现场（2026-11-03，北京时间）

本轮恢复已有 `h24_parameterization_20261103_01`，没有新建替代实验。
工作树 `bc+rl_RSS_native_v2`，分支 `feature/native-dp-training-v2`，恢复前 HEAD `8ef2a16`。
此前连接中断时，Epsilon 代码、审计及 owner/report 脚本已保存但未提交，
`status.json` 为 `gate_completed`；无本轮存活训练/Kit/monitor，无 Epsilon checkpoint，
也没有 `setup_complete.json`。因此尚未开始 Epsilon 更新，不能称为训练中断后重训。

Sample 的原始 `h24_train/milestone_3000.pt` 为 1,153,275,969 字节，SHA256
`70783fcf67344f91a68ff9dec66e42bb0125b87e8dca5192c827c49dd29b25c0`。
CPU 读取确认 updates/AdamW steps/EMA 均为 3000，832 项 schedule 的 cursor 为 176；
optimizer、model、EMA、CPU/CUDA RNG、schedule/noise RNG、history 均在。
当前 LR 为 `9.55614245194068e-05`，沿原 19000-update cosine，不重置。
原文件不写入；续训副本使用干净的冻结工作树 `bc+rl_RSS_h24_sample_pin`，
commit `8ef2a1646c54c47d695b0ca92ec7830f0b4abd03`。

上次 39 项数值/回归测试、Sample 新旧逐值一致性及真实尾段 28 个 padding 梯度为零
作为既有验证沿用，本次不重跑；原 batch 审计不重做。恢复新增测试只覆盖 owner
断点、结果幂等性、日志回退及已有选择/报告接口。实际启动另用状态和已提交更新核验。

## 固定训练合同与预算

- train320 / validation40 / test40，test 不参与本轮选择；832 个至多 96-tick 片段，
  每次更新采样 2 段，平均 13.3267 个 anchor，两个片段均贡献梯度；batch 不变。
- H24 / commit12，原 29D 绝对位置 target_after 弧度动作、State 观察与持续 GRU，
  原生 120 Hz 物理 / 60 Hz 控制。统计仅来自 train320。
- Arm7/Hand22 归一化空间组内 MSE 各 50%，有效 mask；AdamW LR1e-4，warmup500，
  cosine19000，原初始化 seed20261102、EMA、beta schedule / 100 timesteps。
- DDIM16、eta0、clip=False，policy seed7091；唯一学习差异为 Sample/x0 与 Epsilon/noise
  监督及对应 DDIM 重建公式。无 Temporal/FK 辅助损失、无架构或采样改动。
- Epsilon 先 1000 checkpoint 再累计 3000；公平性核对配置、每步 LR、采样顺序/游标、
  EMA count、noise/schedule RNG。两组 3000 各评估后仅选一组到累计 5000、8000、10000。
  每个里程碑保存 optimizer/EMA/RNG/sampler 和指标、执行同口径闭环；0 成功本身不停训。
- 沿用既有固定端点：2026-11-03 21:02:12 setup Gate、11-04 05:02:12 训练端点、
  11-04 07:02:12 总端点（UTC+8），没有因本次恢复延长预算。

## 配对评估和选择

离线统一比较重建 clean action 的原单位 Arm/Hand RMSE，valid positions 加权；
同时报告目标 FK 末端 mm、motion cosine/amplitude、关节变化与重规划边界跳变。
训练诊断为固定 train16、验证为 val40；原 diffusion loss 可追溯，但两类 loss 的绝对值不比较。

使用上次登记的同一 train4 + validation8 初态/seed，各条独立 Kit 启动；每组每里程碑至多12条。
统计完整任务 success、接触、lift、首子目标、alignment、fall、timeout 和实际接触前末端偏差。
观察初态逐字段核验，未证明全部 PhysX 隐状态完全相同。软件故障/未启动不计任务失败，
只在共同完成的合法 validation IDs 上配对选择；train4 作为诊断单列。

沿用 `run_plan.json` 预登记保守排序：完整 validation 成功与退化保护、子目标/lift、
接触前目标 FK、balanced prefix12 MSE、边界跳变、规划耗时。另报告方向/幅值与全部曲线。
单种策略 seed、小样本、冲突或缺失证据下的选择始终标为暂定，不宣称统计显著优势；
一次成功或单项略好不足以得出模型明显更好。两组 3000 权重均保留。
若两次连续已分类的 pre-scene 启动故障，则该 cohort 剩余项记软件缺失，
后续训练/里程碑继续原预算；不反复重试同一条、不改种子、不启用进程复用。

## Kit 与恢复机制

已有4个完整专家 episode，4/4成功；原始 inotify ENOSPC、失败启动和 strace 全保留。
成功启动不证明 FileThread 全由 inotify 引起，也不证明 SDK 根因修复。
进程内 reset 轨迹一致性失败；正式评估强制 `isolated_singleton`。
脚本路径展开修正已落盘；新的固定 shell 入口避免嵌套 shell 字符串中的变量展开错误。
已存结果后隔离退出/父进程回收，durable worker error 后保存证据再退出，防止 SDK cleanup 挂住。

普通 systemd 服务 `h24-comparison-20261103-01.service` 管理 owner、monitor 和单 GPU worker。
每个阶段有独立目录、PID/launch/exit、日志和 SHA 完成标记；已完成结果直接接续。
真实训练中断最多恢复一次，保留失败目录及完整日志，从完整 checkpoint 恢复，
仅将超出 checkpoint 的未提交日志行归档后回退；数值发散、配置冲突和未分类异常阻断。
状态明确区分 running/completed/blocked/software_missing；报告错误另记 `report_error.json`。
共享 GPU 延续既有近期负载调速、约50%目标、6GiB VRAM/12GiB RAM/12GB disk余量，
不更改其他任务、系统 inotify 限额、原生控制和冻结运行源码。

## 结果、曲线与产物

自动结果区块由后台更新，代表实测产物，最终研究结论仍需复核。
[轻量机器报告](evidence/h24_parameterization_20261103/summary.json)保存
配置/源码 SHA、checkpoint SHA、原始指标、选择依据及软件缺失；
[训练/验证曲线](evidence/h24_parameterization_20261103/training_validation.png)按累计 updates 绘制。
尚未完成的 3000 比较、选型及 5000/8000/10000 结果不预填。
失败阶段只能由实际接触/抬升/子目标与终止原因描述；不从离线 MSE 下降推断闭环成功或因果根因。

本地产物根：`outputs/visuotactile/native_dp_v2/h24_parameterization_20261103_01`。
Epsilon：`h24_epsilon_train/`；Sample 续训：`sample_resume/h24_train/`。
GitHub 仅代码、Markdown、小 JSON/CSV/PNG；数据、视频、权重、大日志及失败 strace 均留本机。
本次恢复不删除任何已有实验产物。后台只写报告，不自动提交/推送未经复核的未来结果。

<!-- h24-results:start -->
当前后台状态：`software_missing`。此自动区块待结果复核。

|组/累计updates|val40 Arm(rad)|val40 Hand(rad)|已完成闭环成功|软件缺失|
|---|---:|---:|---|---:|
|epsilon_3000|0.135959|0.465200|0/11完成（注册12）|1|
|sample_10000|0.020862|0.109865|0/11完成（注册12）|1|
|sample_3000|0.028269|0.129117|0/1完成（注册12）|11|
|sample_5000|0.024556|0.120267|0/10完成（注册12）|2|
|sample_8000|0.021733|0.114476|1/11完成（注册12）|1|

以下为同一 val40/seed7091 的 episode 均值；FK 为目标关节的末端误差，实际闭环 FK 另存。

|组|接触前目标FK(mm)|Motion cosine|Amplitude ratio|Arm边界跳变(rad)|Hand边界跳变(rad)|
|---|---:|---:|---:|---:|---:|
|epsilon_3000|108.61277|0.10687|4.48219|0.19668|0.53627|
|sample_10000|17.32545|0.67284|1.14742|0.02278|0.13614|
|sample_3000|21.05591|0.55692|1.32576|0.02620|0.14229|
|sample_5000|18.88530|0.61326|1.22707|0.02400|0.14126|
|sample_8000|17.38549|0.65701|1.16340|0.02307|0.13842|

|组/初态|完成/注册|完整成功|接触|抬升|首子目标|alignment|fall|timeout|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|epsilon_3000/train|4/4|0|2|0|0|0|1|3|
|epsilon_3000/validation|7/8|0|5|0|0|0|3|4|
|sample_10000/train|3/4|0|3|2|0|0|2|1|
|sample_10000/validation|8/8|0|7|0|0|0|0|8|
|sample_3000/train|1/4|0|1|1|0|0|1|0|
|sample_3000/validation|0/8|0|0|0|0|0|0|0|
|sample_5000/train|3/4|0|2|0|0|0|1|2|
|sample_5000/validation|7/8|0|6|0|0|0|0|7|
|sample_8000/train|3/4|1|2|1|1|1|0|2|
|sample_8000/validation|8/8|0|6|0|0|0|0|8|

暂定续训模型：`sample`；配对 validation 数量 0。未建立统计显著优势。
选择依据及各指标原值保存于 summary.selection；test 未参与选择。

实际接触/lift/首子目标/alignment/fall/timeout及逐条结果见[summary](evidence/h24_parameterization_20261103/summary.json)。
<!-- h24-results:end -->

本次恢复新增12项owner/选型/报告回归通过（CPU，0新Kit）；原39项数值验证沿用。
