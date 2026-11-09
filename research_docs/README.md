# Tight-Insertion StateDP

当前训练和评估均已完成：**19,865更新 / 234.374 anchor等效epoch；新初态完整成功0/10，3个视频已验收**。
不自动继续训练或追加评估。数据为218 Core（166/30/22），Aux15未加入。

[当前合同](CONTRACTS.md) · [结果、误差图与解释](STATE_DP_RESULTS.md) · [清理清单与恢复](CLEANUP_AUDIT.md) · [配置](../configs/tight_state.json)

![完整部署采样误差曲线](evidence/tight_palm_20261109/sampling_curves.png)

[训练/验证loss](evidence/tight_palm_20261109/loss_curves.png) · [分阶段误差与hold基线](evidence/tight_palm_20261109/stage_errors.png) · [运动/近静止误差](evidence/tight_palm_20261109/motion_errors.png)

模型复用已有robot/geometry直达condition与持续GRU，仅新增掌部目标FK位置/旋转训练损失。
固定专家历史validation30的Arm RMSE为0.005645rad、Hand 0.056618rad、掌部4.9823mm；离线改善未转化成完整任务成功。
训练中的固定train2/validation4曲线与最终train8/validation30诊断分别报告；同tick自主动作不与不同进度专家标签硬配对。
保持29D绝对target_after_rad、H24/commit12、5Hz规划/60Hz控制/120Hz物理，未训练视觉/触觉，不声称完整视觉DP3复现。

源码、MD、CSV/JSON和6张图公开在当前origin feature分支；完整checkpoint、源轨迹和视频仍在本地。
[机器结果](evidence/tight_palm_20261109/summary.json)、[10例闭环](evidence/tight_palm_20261109/closed_loop.csv)、[数据身份索引](evidence/tight_palm_20261109/dataset_index.csv)、[全部文件SHA](evidence/tight_palm_20261109/artifacts.json)。

本地训练run：`outputs/visuotactile/run/tight_palm_20261108_01`。
本地新模型视频：`outputs/visuotactile/demos/tight_palm_eval_20261109_01/remaining_01/README.md`。
IDE中的 `demos/tight_failures_20261108_01/README.md` 是旧Core8500观看队列，不能当作新模型结果。

本地仅重新导出指标/绘图（CPU，不启动训练或Kit）：

```bash
bash visuotactile/run_cpu.sh -m tools.tight_diagnostics publish \
  --root outputs/visuotactile/run/tight_palm_20261108_01 \
  --evaluation outputs/visuotactile/demos/tight_palm_eval_20261109_01/remaining_01 \
  --output research_docs/evidence/tight_palm_20261109
```

上一轮 `tight_state_20261107_01` 已清理一个重复cache、无损归档13个日志，释放289.15MiB。
当前208条继承轨迹仍引用该目录，旧best/统计、Aux、原速参照和全部成功/失败分母保留，未整体删除旧实验。
旧Core8500的27例闭环与本轮10例保持独立；历史原400/275、专家和资产未动。
训练最高完整60秒整卡均值72.127%超过50%约束，不能宣称全程资源合规；本轮10例评估最高35.456%。
