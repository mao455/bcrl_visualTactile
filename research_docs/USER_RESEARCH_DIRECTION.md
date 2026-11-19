## 2026-11-19 A/B新增逐tick至少50曝光：当前授权与准备

最新任务授权四组件从各自72父权重继续训练，同130池，每轨≥50有效访问、每合法expert tick≥50不同optimizer step新增曝光；祖先/重叠slot不计。新显式覆盖采样、正权mask逐步去重ledger和选模门槛已实现，67项聚焦测试通过；A/B各27508合法ticks预检覆盖，尚未完成训练。预登记60完整deck、仅达标checkpoint可选，DP先完成后刷新各自后端缓存，无encoder/DAgger/RL。详见TIGHT_EXPOSURE50_TRAINING_20261119.md和configs/experiments/exposure50。旧轮“停止/不再训练”不限制本次明确授权；原实验身份/截止不得重启。

## 2026-11-19 B冻结DP对照完成：不晋升

仅后端130训练完成4839更新，验证选中epoch13/update4200；DP72及归一化冻结。48例完整三方自主评估完成，训练/开发成功72B=1/8、0/8，joint130=2/8、2/8，frozen130=1/8、1/8；冻结候选丢失基准唯一成功训练seed28817016。开发hold60=7/8、5/8、4/8，不支持固定DP即可保留能力。新58离线arm/hand改善34.6%/24.2%，没有稳定闭环晋升证据；优先后端保持/反馈，encoder仅未来受门槛probe候选。本轮19保护SHA保持，所有自有GPU进程退出；资源中断额外尝试均保留，165旧视觉缓存无损归档释放6.3303GiB，历史复现先按清单恢复；当前136缓存未归档。无新DAgger/encoder/初态/控制改动，不自动重启或扩展。完整证据见[本轮报告](TIGHT_FROZEN_DP_B_20261119.md)。

## 2026-11-19 选中曝光与感知表征审计完成

唯一工作区仍为bc+rl_RSS，采用72阶段完整A/B；130候选未晋升。本轮CPU诊断、0策略更新/0新DAgger/0新Kit。30个祖先checkpoint和真实日志重建：130候选A/B DP新轨最低24/14访问，选中后端3/15；并非每条轨迹或每合法tick都≥50次。A/B DP尚有2/10个合法tick零slot，不能用重叠窗口或未选中更新补年龄。

216个family隔离probe回执、四模型真实完整历史推理及有界输入干预；多模态改善厘米级几何读出，未证明精确插入能力或encoder是唯一瓶颈。视觉/力确实影响预测；训练family自主亦不可靠，组合拟合与闭环保持优先。原72/130模型、数据和归一化保护身份不变。完整指标、曲线、逐tick采样回执、历史梯度不可恢复的限制及复现命令见[本轮审计](TIGHT_REPRESENTATION_AUDIT_20261119.md)。不自动执行提出的训练或encoder对照。

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
[工程统一报告](PROJECT_UNIFICATION_20261118.md)。以下均为历史阶段，不授权自动重启。

## 2026-11-17 fixed76已闭合：不晋升，无新DAgger

实际19:21:13–20:58:09(+08)，96分56秒，训练/离线/四波32次完整开发均结束；全部自有worker/monitor退出。A72成功2/8→候选0/8，B72 1/8→0/8；hold120各保持4/8、6/8，首子目标A3→0/B3→2，实际可观测初态配对通过。不是统计显著结论。DP实际A874/B722、选494/342；匹配后端实际2170/2411、选369/2048。4GiB hotset正式显式启用；磁盘中止和一次完整状态恢复证据保留，原截止未变。

入口监督曝光增加与训练池拟合改善尚未转成完整自主收益。三个丢失成功配对约70–80tick开始几何偏离，命令/关节更早偏离；不能归为单纯缺数据，未证明新的可恢复覆盖缺口。本轮不加epoch、不采新DAgger、不替换72完整A/B。原76合法数据、所有科学候选及72归一化/配置/来源保留。最新结论见TIGHT_TRAINING_DIAGNOSIS_20261116.md首节与balanced76/completed.json、closure_audit.json；下方准备/运行状态均为历史，禁止自动重启旧服务。

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

## 2026-11-17 最新：仅DP性能短程检查

本轮只允许同状态短程训练profiling与一个有界冻结特征缓存，禁止自动接续长训、DAgger、后端或自主测试。最新实测状态与下一步限制见[主报告顶部](TIGHT_TRAINING_DIAGNOSIS_20261116.md)。原72条选定完整模型、76条数据及原400条保持；下列“没有新参数更新”指此前发布轮，不是此次短程诊断。

## 2026-11-17 最新：实验已结束，证据补齐，暂停新训练/DAgger

当前只完成核查与轻量证据发布，没有新参数更新或仿真。共同池56→65→72→76，最终选中72完整A/B，新20对测试各3/20；不是独立DAgger。完整叙述以[集中主报告](TIGHT_TRAINING_DIAGNOSIS_20261116.md)为准，[证据入口](evidence/tight_long_20261117/publication/README.md)包含四版manifest、模型继承、184次自主执行、64次历轮采集机会、95项时间回执、原始曲线及实际视频。正式窗口13h38m58s；不拼成20h。以下旧“正在运行”条目均为历史。

## 2026-11-16 夜间最新要求：先诊断，两个真实闭环后才允许脱离监督

用户最新消息要求先完成已入库26条训练，追踪真实标签/曝光/接管时机，用固定训练家族诊断学习瓶颈。不得无限堆数据或把专家恢复当自主成功。主Agent必须监督两个连续完整采集→准入→DP→刷新输入→Transformer→自主评估小循环与交接，才能释放无人值守屏障。

原04的02:05:20截止来自自行登记的六小时临时时段（scripts/tight_fresh_20261115_04/launch_config.json hours=6），不是外部机器配额。用户本次明确允许延长到次日上午；原04/13现场和截止不改，待其阶段闭合后另建截至11月16日09:00 +08的新有限session。当前只准备，不能把两循环或后台可靠性写成已通过。

## 2026-11-15 21:20 最新阶段目标

先让完整学生可靠接近、抓起、持续持物，约90%以上并经固定及额外隔离开发初态支持后，再逐步增加转正/对孔/插入纠偏。重点纠正学生已真实走到的早期接触偏差，重访早失败训练family；fixed1仅作补充示范。A11/B09为抓持来源参考，候选不能仅因新或全局loss更低被采用；DP与后端都要有明确新纠偏曝光和误差。原任务、历史、频率、两阶段训练顺序不变。

# 当前用户研究方向

当前有效任务是2026-11-14–15用户要求，合同集中于根[AGENTS](../AGENTS.md)。历史授权已移入[归档](archive/USER_RESEARCH_DIRECTION_through_20261113.md)，原始文本可从Git 4224dba复核；不得以旧禁止训练、旧GPU50%或启动后返回要求限制当前持续研究。

用户授权检查、修改、采集、训练、评估、自主选择下一实验及推送源码/精简结果；整张GPU可用，保护其他项目和实际内存/显存/磁盘余量。保留原资产、成功标准、原生120/60Hz、5Hz DP与最终29维绝对关节目标语义。每控制tick三视角RGB-D真正入网，所有模块使用对齐的真实历史，不能丢弃中间帧冒充加速。

初始约30条不同初态成功示范，学生DP和后端从零训练，后续合法DAgger不受30条限制。新增授权明确允许100–300条有效新增轨迹、新训练初态和完整专家示范；按收益决定500/1000，不把轨迹数当独立family数。采集和自主执行完整学生，专家从学生实际偏差状态接手，保存真实前缀和新观察下的专家后段；当前/未来专家标签不能进入学生条件。DP更新后刷新后端计划和可训练特征。以“拿起、持物、转正、对孔、插入”的完整自主结果判断收益，报告真实学习曝光及可比训练/验证误差。

速度改进必须进入正式采集、训练及自主评估的一整轮。显式多环境没有四lane上限，8/16/32/64/128及更高按实测有效轨迹/小时和合法专家步/小时选择。允许TiledCamera、GPU物理、线程数、批处理、预取及有界异步写入；保留实际三视角60Hz传感与独立lane历史/接管/终止，并用真实任务验收后端。当前固定质量/摩擦和窄初态域，不是广泛质量/摩擦泛化；q_reached是一控制步之后的实际关节位置，目标误差包含动态跟踪和接触，不能全部解释为压力。

实际已完成五轮纠偏和最近后端输出起点对照，仍无稳定完整自主成功；后端候选未采用。视觉辅助同预算对照已完成，没有证明自主行为收益；保留第五轮完整模型，下一步实际采集允许掌中转动的持续接触抬升状态纠偏。真值位姿仍只作训练标签或诊断，不成为学生输入。最新运行、真实分母和下一步集中于[SESSION_HANDOFF](SESSION_HANDOFF.md)，避免此页重复累积运行快照。

研究机CNS202533182702；本轮工作树continuous_vision_20261114从远端506d185建立，原脏工作区未改。推送仅origin/feature/tight-state-foundation；旧run截止和原证据保持，阶段结束后可依当前持续授权另建独立session，必须记录来源与实际代码/模型/数据身份。不能只启动后台就称全程监督，不能把同seed或可见初态相等称隐藏PhysX精确重放。

最新执行偏好：用一套短检查覆盖实际变动，复用不变的检查证据；观察一整轮采集、训练、自主和下一轮交接后，允许有界普通脚本后台继续。随机合法时间点规划、轮换扩大几何覆盖与真实整批吞吐为当前实施重点。
