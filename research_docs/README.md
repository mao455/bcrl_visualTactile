# Tight-Insertion StateDP

当前结果分开报告：A固定Hand cap已通过有限可靠性验证；B原275混速State数据的canonical best4500独立闭环未显示稳定新初态泛化。主文档为[当前合同](CONTRACTS.md)与[当前结果](STATE_DP_RESULTS.md)，历史事实见[精简历史表](RESEARCH_HISTORY.md)。

本轮基点`1b70d3b6fce8052300a54f5361f4092da27cba5f`，实际物理运行提交`f8beeef6d408ac20f520f6a26625c31bc3cdb5ee`，干净启动。实际工作树`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`、分支`feature/tight-state-foundation`；IDE原树既存改动保持。

唯一当前配置：[configs/tight_state.json](../configs/tight_state.json)。已冻结Arm.50、Hand1、Hand逐关节目标cap2rad/s及实际速度/路径门槛(.90/1.25)。正式入口仍tools.tight_finish；旧标量alpha开发入口会拒绝当前配置，不能冒用旧gate。工程证据冻结，未重开基础审计；新增限速与配置门控的37项回归通过。

当前生产接续root：`outputs/visuotactile/run/tight_state_20261107_01/formal_05`。只读状态：

```bash
bash visuotactile/run_cpu.sh -m tools.tight_state status --root outputs/visuotactile/run/tight_state_20261107_01/formal_05
```

A pilot5/6成功、4/6准入；独立12/12成功、10/12准入，4家族覆盖3/3/2/2。`qualified_development.json`保留14条7家族train-only合格开发轨迹；尚非正式集。B同一best4500 canonical：训练家族1/4成功、新初态0/10合法成功，另2次初态拒绝；无新训练，无上轮7seed重跑。

2026-11-07 20:05实测可用1018.29GiB，旧容量阻断已解除。正式continuation沿用通过gate和当前配置，固定55新家族/275次启动上限，不重跑开发或扩大范围。保留原采集截止11月8日09:08、整体截止11月9日03:08（Asia/Shanghai）。登记和启动不代表数据或训练已完成；实际状态以formal_05/status.json为准。前次7.510GiB清理及保护记录见[CLEANUP_AUDIT](CLEANUP_AUDIT.md)。

生产配置不读取旧学生checkpoint或旧calibration作为参数来源；已有gate/轨迹仅是明确的准入证据。原275历史baseline显式只读引用其数据/统计/best4500，永不冒称严格慢速多模态集。原400、旧275、专家/资产及失败分母继续保护；无数据/checkpoint上传。
