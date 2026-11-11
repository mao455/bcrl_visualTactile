# Tight-Insertion 研究入口

共享纠偏池DAgger已在 `bc+rl_RSS_qg_report` 启动，最多3轮、总18h，QG01优先、G11随后。2026-11-11 17:01阶段快照：QG01首轮前后端训练与10例自主评估完成，0/10成功、2次原生lift、全部超时；G11前端训练中。本轮59条新成功纠偏加旧21条，共80条（73训练/7开发），另保留Core166；每轮使用共同不可变快照。实际曝光、失败过程、曲线与仍未完成事项见[结果页](STATE_DP_RESULTS.md)，实时状态以owner凭据为准。

每轮现有U-Net前端微调 → 冻结选定前端并刷新全部名义缓存 → Transformer后端微调 → 固定4例自主开发、满足登记条件才扩至10 → 完整配对选择/回退。Core/历史/本轮纠偏按.5/.2/.3采样；实际update0参与旧val/纠偏dev/初24有效tick联合选模。原统计、监督类型和物理不变，test22不使用。详细预算、恢复命令、实际证据边界见 [CONTRACTS](CONTRACTS.md) 与 [SESSION_HANDOFF](SESSION_HANDOFF.md)。

|入口|内容|
|---|---|
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
