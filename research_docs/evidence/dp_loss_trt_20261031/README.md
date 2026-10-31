# DP损失与Transformer残差图表索引

主报告：[DP_LOSS_AND_TRANSFORMER_RESIDUAL_20261031](../../DP_LOSS_AND_TRANSFORMER_RESIDUAL_20261031.md)。
本目录只保存小型统计和研究曲线，不含checkpoint、原始轨迹、视频或相机图像。
当前DP阶段完整；残差阶段结果尚待生成。数值来源为独立run `dp_loss_trt_20261031_01`。

## DP离线对照

- [Loss A/B各自定义的训练/验证曲线](dp_losses.png)：train320 evolving raw、固定train16 EMA与val40 EMA分开。
- [前12/H64原生命令误差](dp_command_rmse.png)、[绝对参考rad误差](dp_reference_rmse.png)、[归一化参考误差](dp_normalized_rmse.png)。
- [五指误差](dp_fingers.png)、[未来位置误差](dp_horizon.png)、[训练/验证差距](dp_generalization.png)。
- [学习率/梯度/饱和](dp_optimizer.png)、[原40遍已保存的validation曲线](original40_validation.png)。
- 固定验证示范动作对照：[G00-A](G00-A_validation_trace.png)、[G00-B](G00-B_validation_trace.png)、[G10-A](G10-A_validation_trace.png)、[G10-B](G10-B_validation_trace.png)。
- 小统计：[评估CSV](dp_metrics.csv)、[训练CSV](dp_training.csv)、[来源与更新数](dp_figure_metadata.json)。

离线原生命令指标使用专家实际controller target还原，未来target只参与评价，不输入模型。
它们不等于自主控制误差。所有四线额外12遍/9984更新；验证选模，test40每线一次。

## DP自主轨迹

- [三个共同初态的高度/距离/观测接触](dp_closed_loop_progress.png)、[结果CSV](dp_closed_loop_outcomes.csv)。
- [G10-B seed981004短暂抬高和掉落事件](dp_G10_lift_event.png)、[事件统计CSV](dp_G10_lift_event.csv)。

四线各0/3，全部601步timeout，无子目标/原生lift；G10-B有+7.69cm短暂高度提升，不能称稳定抓持。
触觉slot按metadata为index/middle/ring/thumb/pinky；动作hand维度以thumb先行。
聚合接触无对象身份；事件图中的时间关联不确定掉落原因，实际施加外力/力矩均为0。
