<!-- EXECUTION_LEARNING_STATUS_START -->
## 2026-11-13 执行学习修复：机器状态

状态 **PREPARED / CPU_complete_GPU_not_started**；新 run `tight_execution_learning_20261113_01`，绝对截止 **2026-11-14T00:05:15.106664+08:00**。
旧 matched run 的冻结源码及原截止不变；本轮只有一个FY候选，先配对200更新，双组门槛通过后才做共同预算后端重训与每组10次完整学生开发验证。
短探针固定第一轮；后端正式适配复用旧run最新完整配套前端/缓存与共享数据快照，A仍学习q_reached、B仍学习u_expert。test22隔离；等待/启动不是训练或闭环完成。
[数学、执行链与边界](DAGGER_METHOD.md#execution-learning-repair) · [实际状态](evidence/tight_execution_learning_20261113/status.json) · [诊断数值](evidence/tight_execution_learning_20261113/execution_summary.json) · [梯度图](evidence/tight_execution_learning_20261113/loss_and_parameter_gradients.png)。
<!-- EXECUTION_LEARNING_STATUS_END -->

# Tight-Insertion 研究入口

**2026-11-12最新工作**：[DAgger性能实测与有限队列修订](DAGGER_PERFORMANCE.md)记录已测局部加速、未测完整轮收益及待安全边界接入状态。控制匹配run已实际采集，A第一轮前端完成；新配套后端和自主评估尚未完成。完整算法、控制与监督规则见 [DAGGER_METHOD](DAGGER_METHOD.md)，实际进程/阶段见 [交接](SESSION_HANDOFF.md)；专家接管成功不等于学生自主成功。

以下保留此前条件追加的来源和历史范围；最新合同覆盖见[CONTRACTS](CONTRACTS.md)。

当前范围是两个完整学生策略：**A：q_reached配置前端＋既定几何软惩罚＋Transformer；B：u_expert命令前端＋同容量Transformer**。两组前端同为RGB-D＋robot178的G00模态，触觉均由后端读取；裸配置前端保留训练/离线验证，不承担独立DAgger或任务考核。G11等旧变体保留证据并退出活动队列。

原 `tight_dagger_pool_20261111_01` 的第2轮按原配置完成A/B训练与有限自主评估。新授权仅当两组本轮新candidate的实际合法自主评估完整完成且都0成功，才用新配套DP＋Transformer追加一次第3轮，各最多96次真实接管，再分别更新DP（最多800）、刷新缓存和更新后端（最多12有效tick遍）。原Core与全部合格纠偏持续累积；数据不足只是待检验假说。

触发后最多延长6小时：绝对截止2026-11-12 **13:24:31**，计算13:14:31停止；原起点不变，无递归、无新run预算。任一组成功不追加，未完成/软件错误不作零成功。机会按完整训练预留裁减，原4→10自主协议和test22隔离保持。隔离准备、触发判定、实际进程启动与采集验收分别记载，见[交接](SESSION_HANDOFF.md)及实际机器凭据。

安全停止前QG01首轮804/12212次前后端更新、自主0/10成功和2次lift；G11前端754次更新及已有80条纠偏均保留。当前修订的实现/验收与运行状态分开记录，见[交接](SESSION_HANDOFF.md)及[实际结果](STATE_DP_RESULTS.md)。A/B旧残差幅度和历史训练曝光不同，此次不是严格单变量或同预算实验；test22不使用。详细标签、预算和证据边界见[合同](CONTRACTS.md)。

|入口|内容|
|---|---|
|[性能实测](DAGGER_PERFORMANCE.md)|组件与调度对照、未采用候选、实际接入边界|
|[研究方向](USER_RESEARCH_DIRECTION.md)|最新用户授权与比较范围|
|[合同](CONTRACTS.md)|标签/历史/缓存/预算、模块职责与官方结构对照|
|[交接](SESSION_HANDOFF.md)|原运行与后继队列的实际边界、接手核验|
|[实际结果](STATE_DP_RESULTS.md)|训练曝光、自主分母、错误和本地视频索引|
|[感知](PERCEPTION.md)|三路RGB-D、触觉与实际执行历史|
|[变更记录](CHANGELOG.md)|已实施修改与测试；历史条目不重新授权旧阶段|
|[清理记录](CLEANUP_AUDIT.md)|按依赖检查的实际删除和释放空间|
|[配置](../configs/tight_state.json)|`dagger` 与原实验的分别冻结配置|

必要历史对照保留如下。它们属于不同模型和阶段，不直接合并为 DAgger 的前后变化。

|历史阶段|训练/模型|开发原生完整成功|
|---|---|---|
|2026-11-09 StateDP掌部损失|19,865更新，234.374 anchor等效遍历|0/10|
|2026-11-09 G00/G10视觉基础|10,132/6,929更新；119.541/81.753 anchor等效遍历|各0/10|
|2026-11-10旧G01/G11 GRU残差|各98,610更新，200接触tick等效遍历|0/10、1/10|
|已完成 `tight_transformer_20261110_02`|复用视觉前端、Q配置适配及Transformer后端|由该轮实际结果单独报告|

旧 G11 唯一成功为 seed27150000，279tick/4.65s，`native_lift=false`；不能反推原生抬升或稳定改善。旧GRU与新Transformer同时改变监督范围和失触门控，不能把差异全归于网络结构。历史日志/资源问题、FAILED收尾和成功分母见结果页；不因后继阶段改写。

当前CPU数据验收覆盖真实196条 train/validation：train166共55,392tick、4,690有效原生规划anchor，原统计字节一致，test源未读；新增接管 mask/标签/历史与缓存刷新测试通过。这些不证明真实专家能救回，也不证明学生自主性能已改善。

代码和小型结果公开到 `origin/feature/tight-state-foundation`；大 checkpoint、数据与视频留本机。项目基础和上游安装说明见[根 README](../README.md)。
