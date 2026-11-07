# Tight-Insertion StateDP

当前任务分A/B：A用原275条State-valid混速数据保留checkpoint做修复输入的受控闭环复评；B只读既有轨迹解释阶段减速，判断采集机制。两者独立，不重开已通过基础审计。主文档只有[当前合同](CONTRACTS.md)和[当前结果](STATE_DP_RESULTS.md)；历史事实见[精简历史表](RESEARCH_HISTORY.md)，历史待办不是当前授权。

本轮审查基点为feature/narrow-state-data的642f5a2。IDE原目录HEAD为3e11d62且有既存文档改动；实际修改在`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`，基于干净642f5a2新建本地分支`feature/tight-state-foundation`，未重写共享分支。

唯一配置：[configs/tight_state.json](../configs/tight_state.json)。最新已结束续行root：`outputs/visuotactile/run/tight_state_20261107_01/continuation_02`，父root保留上一段结果。只读状态：

```bash
bash visuotactile/run_cpu.sh -m tools.tight_state status --root outputs/visuotactile/run/tight_state_20261107_01/continuation_02
```

入口拒绝自动覆盖已有物理attempt；`prepare`只允许新root，不能用换root重置已经登记的失败/预算。开发、正式数据、训练与学生成功是四个不同结果，均以本root对应实测文件为准。旧narrow_collection/narrow_repair/旧session不是当前正式生产入口。

当前配置不依赖旧run/session/calibration或学生权重；原400/275路径仅用于保护、明确标记的独立审计fixture和历史索引。旧数据审计通过不等于被准入本轮慢速多模态集。原生工程入口、专家、assets与共享SDK保持。

运行代码cf275d5ec8029782942f2d36a58eb06f669e4b02以e67bfc6完整修复为基础。固定Arm0.50/Hand1.00验证已按预登记规则结束：慢速成功6/6、准入2/6，剩余6条即使全过也达不到9/12。所有自有进程退出；正式数据/更新均0，学生闭环未执行。旧工程验收不重做，不将父root状态或历史运行当成本次进展。

A当前root：`outputs/visuotactile/run/tight_state_20261107_01/baseline_03`，使用同一status入口读取；登记24次启动，实际结果未完成前不报告成功率。B分析也写入同一root，原始轨迹只读。
