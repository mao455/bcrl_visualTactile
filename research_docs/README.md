# Tight-Insertion StateDP

当前按用户新版Core/Auxiliary合同接续，已通过61项回归并恢复生产，尚未声称数据冻结、训练或闭环完成。Core目标约200，Auxiliary最多100且仅允许边界外10%；首轮训练仅Core。旧20条10/20严格准入保留，10完整Core可复用，2条数值Aux缺完整RGB-D不可复用。

[当前合同](CONTRACTS.md) · [当前结果](STATE_DP_RESULTS.md) · [交接](SESSION_HANDOFF.md) · [清理](CLEANUP_AUDIT.md)

使用修复工作树`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`、feature/tight-state-foundation；[唯一配置](../configs/tight_state.json)和tools.tight_finish原入口，Arm.50/Hand1/cap2rad/s。模型/控制/模态实现冻结；当前持续GRU不截断64tick，未来H24/commit12不变。

当前root：`outputs/visuotactile/run/tight_state_20261107_01/composition_06`。运行源码4d43a3975c9bd153ff90d7c13e03ca9a0e6aa917，22:30启动；只读状态：

```bash
bash visuotactile/run_cpu.sh -m tools.tight_state status --root outputs/visuotactile/run/tight_state_20261107_01/composition_06
```

原400/275、专家/资产/成功模型和失败分母保护；不上传数据或权重。旧275混速State结果独立为train1/4、新初态0/10合法+2拒绝，不作为新Core模型结果。
