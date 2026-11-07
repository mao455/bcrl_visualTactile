# Tight-Insertion StateDP

当前任务分A/B：A用原275条State-valid混速数据保留checkpoint做修复输入的受控闭环复评；B只读既有轨迹解释阶段减速，判断采集机制。两者独立，不重开已通过基础审计。主文档只有[当前合同](CONTRACTS.md)和[当前结果](STATE_DP_RESULTS.md)；历史事实见[精简历史表](RESEARCH_HISTORY.md)，历史待办不是当前授权。

本轮起点为362c6308a6519b57309487583ec401c5acd6f4a7；此前工程审查基点为feature/narrow-state-data的642f5a2。IDE原目录HEAD为3e11d62且有既存文档改动；实际修改在`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`，基于干净642f5a2新建本地分支`feature/tight-state-foundation`，未重写共享分支。

唯一配置：[configs/tight_state.json](../configs/tight_state.json)。最新A受控复评root：`outputs/visuotactile/run/tight_state_20261107_01/baseline_03`，父root保留上一段结果。只读状态：

```bash
bash visuotactile/run_cpu.sh -m tools.tight_state status --root outputs/visuotactile/run/tight_state_20261107_01/baseline_03
```

入口拒绝自动覆盖已有物理attempt；`prepare`只允许新root，不能用换root重置已经登记的失败/预算。开发、正式数据、训练与学生成功是四个不同结果，均以本root对应实测文件为准。旧narrow_collection/narrow_repair/旧session不是当前正式生产入口。

正式慢速生产配置不依赖旧run/session/calibration或学生权重。A受控复评则按最新授权显式引用冻结275数据版本和best4500，独立于生产门槛；旧数据审计通过不等于被准入慢速多模态集。原生工程入口、专家、assets与共享SDK保持。

A实际运行代码991db6a1c02af140ef12fb1f3a35f159c7ba2193。24次尝试完成：旧/canonical训练家族均0/4，未见初态旧0/7、canonical1/7，各另1次初态拒绝；同权重输入修复观察到1个成功改善配对，未证明稳定改善。没有新增训练，原275混速数据不授予新慢速资格。

B分析33条既有记录，不新采样；固定方案尚不可靠，正式慢速集0条。速度上限机制仅有采集侧提案，未实施。全部自有进程退出；最终[当前结果](STATE_DP_RESULTS.md)分别报告A和B。当前生产配置仍是tight_state.json，历史数据复评配置显式冻结在baseline_03/baseline_plan.json，不隐式修改源checkpoint。
