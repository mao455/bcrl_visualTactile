# Tight-Insertion StateDP

当前显式任务：核实并复用已有robot/geometry直达condition，补齐DP3风格StateDP动作/行为诊断；不声称视觉DP3复现。
等价路径已存在，旧采样曲线触发平台/过拟合停止，本轮按条件分支**不重训/续训**。
独立[诊断配置](../configs/tight_dp3_diagnostics.json)、[当前字段与时序合同](CONTRACTS.md)、[结果](STATE_DP_RESULTS.md)。
run：`outputs/visuotactile/run/tight_dp3_20261108_01`；最多1h、固定train8/validation30、新Kit0。
一次查看：`bash visuotactile/run_cpu.sh -m tools.tight_diagnostics status --root outputs/visuotactile/run/tight_dp3_20261108_01`。
后台正常结束会原位更新结果MD；启动与完成分开报告。下文保留原Core任务结案。


此前Core任务已完成：208Core/50家族＋15Auxiliary/11家族冻结，Core-only随机初始化训练8500步/100.21 anchor等效epoch，闭环train0/10、validation0/7、新初态0/10。有限训练停止条件通过，但学生没有完整成功；没有扩展到100、凑300或自动增加模型实验。

[当前合同](CONTRACTS.md) · [当前结果与证据](STATE_DP_RESULTS.md) · [交接](SESSION_HANDOFF.md) · [清理](CLEANUP_AUDIT.md)

使用修复工作树`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`、feature/tight-state-foundation；[唯一配置](../configs/tight_state.json)和tools.tight_finish原入口。固定Arm.50/Hand1/cap2rad/s；command-based target_after_rad、持续GRU、H24/commit12、5/60/120Hz不变，不是64tick截断历史。

已完成root：`outputs/visuotactile/run/tight_state_20261107_01/composition_06`。采集源码4d43a3975c9bd153ff90d7c13e03ca9a0e6aa917，训练/评估fdb7ea94a1457de57ff2a9fc21d1a9486742d5ab。只读状态：

```bash
bash visuotactile/run_cpu.sh -m tools.tight_state status --root outputs/visuotactile/run/tight_state_20261107_01/composition_06
```

Core训练视图为root/manifest.json和state_data；完整组成在composition_manifest.json。当前完整模型为training/best.pt，last.pt为同inode恢复别名。所有自有服务已退出，不重复启动已完成队列；未来恢复必须保留完整状态和数据指纹，不能把新实验混入本次分母。

清理136文件释放36.191GiB，原400/275、专家/资产、成功历史和失败分母保护。整卡60秒峰值59.632%、最低空闲VRAM4154MiB，本轮不是全程资源合规。仅发布代码/文档/小统计，不上传数据或权重。
