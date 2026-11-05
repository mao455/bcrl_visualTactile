# 维护变更

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
