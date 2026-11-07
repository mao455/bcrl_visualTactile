# Tight-Insertion StateDP

当前结果分开报告：A固定Hand cap已通过有限可靠性验证；B原275混速State数据的canonical best4500独立闭环未显示稳定新初态泛化。主文档为[当前合同](CONTRACTS.md)与[当前结果](STATE_DP_RESULTS.md)，历史事实见[精简历史表](RESEARCH_HISTORY.md)。

本轮基点`1b70d3b6fce8052300a54f5361f4092da27cba5f`，实际物理运行提交`f8beeef6d408ac20f520f6a26625c31bc3cdb5ee`，干净启动。实际工作树`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`、分支`feature/tight-state-foundation`；IDE原树既存改动保持。

唯一当前配置：[configs/tight_state.json](../configs/tight_state.json)。已冻结Arm.50、Hand1、Hand逐关节目标cap2rad/s及实际速度/路径门槛(.90/1.25)。正式入口仍tools.tight_finish；旧标量alpha开发入口会拒绝当前配置，不能冒用旧gate。工程证据冻结，未重开基础审计；新增限速与配置门控的37项回归通过。

当前root：`outputs/visuotactile/run/tight_state_20261107_01/hand_cap_04`。只读状态：

```bash
bash visuotactile/run_cpu.sh -m tools.tight_state status --root outputs/visuotactile/run/tight_state_20261107_01/hand_cap_04
```

A pilot5/6成功、4/6准入；独立12/12成功、10/12准入，4家族覆盖3/3/2/2。`qualified_development.json`保留14条7家族train-only合格开发轨迹；尚非正式集。B同一best4500 canonical：训练家族1/4成功、新初态0/10合法成功，另2次初态拒绝；无新训练，无上轮7seed重跑。

当前明确阻断为磁盘：保留60GiB后约8GiB，复用全部合格开发数据后按最小实测payload仍需约31GiB新增空间。正式采集前容量STOP已退出全部自有进程；正式attempt0、manifest/统计/cache未冻结、训练未执行。不要直接重启已完成队列或删除STOP绕过容量。空间到位后从通过的gate准备正式生产continuation，沿用冻结参数与剩余预算，不重采开发案例。

生产配置不读取旧学生checkpoint或旧calibration作为参数来源；已有gate/轨迹仅是明确的准入证据。原275历史baseline显式只读引用其数据/统计/best4500，永不冒称严格慢速多模态集。原400、旧275、专家/资产及失败分母继续保护；无数据/checkpoint上传。
