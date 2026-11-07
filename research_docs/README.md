# Tight-Insertion StateDP

当前正式生产已按预登记门槛停止：55原速参照54成功、1初态拒绝；20慢速全部任务成功，但仅10条准入，50%低于65%。空间问题已解除。新正式数据未冻结，随机初始化训练与新模型闭环未运行；不能把启动或局部合格写成完成。

[当前合同](CONTRACTS.md) · [当前结果与逐条表](STATE_DP_RESULTS.md) · [清理](CLEANUP_AUDIT.md) · [精简历史](RESEARCH_HISTORY.md)

实际运行HEAD `5b3bd48ef720f71d860b2c76a32b39b4e27d8dc1`，干净启动；工作树`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`，分支`feature/tight-state-foundation`。原IDE工作树不变。唯一配置为[configs/tight_state.json](../configs/tight_state.json)，Arm.50/Hand1/Hand cap2rad/s，实际速度/路径门槛.90/1.25；正式入口tools.tight_finish。相关源码和工程证据未变。

当前root：`outputs/visuotactile/run/tight_state_20261107_01/formal_05`。只读状态：

```bash
bash visuotactile/run_cpu.sh -m tools.tight_state status --root outputs/visuotactile/run/tight_state_20261107_01/formal_05
```

所有自有进程已退出；不能通过重启或换seed绕过production gate。10条本批合格轨迹和14条合格开发记录分开保留，未混成正式训练集。下一机制仅提案：采集侧路径进度控制，需独立有界验证；不继续固定alpha/cap网格，不启ImageDP或触觉训练。

旧275混速State baseline独立有效：canonical best4500训练家族1/4、新初态0/10合法，另2初态拒绝；未新增训练。原400/275、专家/资产、成功模型和失败分母继续保护；不上传数据/checkpoint，不重写共享历史。
