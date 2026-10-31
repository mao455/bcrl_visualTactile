# DP损失与Transformer残差图表索引

主报告：[DP_LOSS_AND_TRANSFORMER_RESIDUAL_20261031](../../DP_LOSS_AND_TRANSFORMER_RESIDUAL_20261031.md)。
本目录只保存小型统计和研究曲线，不含checkpoint、原始轨迹、视频或相机图像。
DP与残差训练、最终test和18条自主诊断均已完成，所有自有GPU进程已退出。
数值来源为独立run `dp_loss_trt_20261031_01`；[最终去敏统计与产物验收](../dp_loss_trt_20261031.json)。

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

## Transformer离线对照

- [训练/验证loss与接触组合命令误差](transformer_loss_and_command.png)：实际train320、固定train16与val40分开。
- [arm与五指raw残差误差](transformer_fingers.png)、[gate/有效率/幅度/执行裁剪/饱和/批处理前向延迟](transformer_diagnostics.png)。
- [评估CSV](transformer_metrics.csv)共24行；[训练CSV](transformer_training.csv)共10行，包含更新数、学习率、梯度与冻结base身份。

force/masked各5完整遍历/3730更新，验证选epoch5，test40各一次。两组同基座、缓存、初始化与预算，
屏蔽组仅屏蔽连续力特征，仍保留真实gate/valid/FK，不能称完全无触觉。
force的test接触arm误差相对缓存base下降24.91%、hand下降1.56%；masked略好，未证明连续力特征收益。
这属于专家轨迹teacher forcing下的离线结果，不等于自主策略误差或任务成功。

## Transformer自主轨迹

- [seed981001](transformer_closed_loop_s981001.png)、[seed981002](transformer_closed_loop_s981002.png)、[seed981004](transformer_closed_loop_s981004.png)：高度、观测力、实际修正及实际施加外力。

两组各0/3，六条均601步timeout、无子目标/原生lift；force实际gate分别6/10/191步，获得过控制机会。
**保存物理初态匹配，但接触前命令不匹配：** 所有base/force/masked比较均为strict_precontact_match=false。
这些图展示观测到的轨迹，不能将其差异解释为严格配对下的残差/触觉因果效应。
新G01按用户授权使用过去实际混合命令，名义DP会间接受残差影响；无接触时精确返回当前base。
同输入封装及训练后残差递推核验通过，但不能替代真实轨迹配对；DP BF16合批/逐tick数值差异仍须披露。

本地视频索引为`outputs/visuotactile/run/dp_loss_trt_20261031_01/README.md`，视频不上传。
本轮整卡均值33.20%、最大完整60秒39.43%，缺测/资源越界0；旧会话的资源违规记录不变。

## 直接阅读关键曲线

以下均为原训练记录，除明确标注的固定权重重算图。checkpoint来源和各线更新数见
`dp_figure_metadata.json`；DP评估为EMA、残差为raw权重。不同定义loss不能直接排名。

### DP训练目标与共同外部指标

`dp_losses.png` ← `dp_training.csv`、`dp_metrics.csv`：各loss各自下降，不以A/B目标数值大小判胜负。

![DP训练验证损失](dp_losses.png)

`dp_command_rmse.png` ← `dp_metrics.csv`：B前12步命令误差较低，但完整H64 arm仍较大。

![前12步与H64命令误差](dp_command_rmse.png)

`dp_fingers.png` ← `dp_metrics.csv`：五指单独记录，改善幅度不一致。

![五指误差](dp_fingers.png)

`dp_horizon.png` ← `dp_horizon.csv`（原各线selected_validation.horizon）：远期误差不能由前缀平均代替。

![未来位置误差](dp_horizon.png)

`dp_generalization.png` ← `dp_metrics.csv`：固定train16与val40存在差距，不是训练初态自主成功证据。

![训练验证差距](dp_generalization.png)

`dp_reference_rmse.png`与`dp_normalized_rmse.png` ← `dp_metrics.csv`：分别为rad与归一化参考单位。

![弧度参考误差](dp_reference_rmse.png)
![归一化参考误差](dp_normalized_rmse.png)

`dp_optimizer.png` ← `dp_training.csv`、`dp_metrics.csv`：新学习率计划实际执行，末期衰减到接近0。

![DP优化器和饱和](dp_optimizer.png)

### Transformer训练与输出

`transformer_loss_and_command.png` ← `transformer_training.csv`、`transformer_metrics.csv`：arm改善明显，hand较小；masked略好。

![Transformer训练验证和组合命令](transformer_loss_and_command.png)

`transformer_fingers.png` ← `transformer_metrics.csv`：各分支raw残差误差变化，不是抓握成功率。

![Transformer分支误差](transformer_fingers.png)

`transformer_diagnostics.png` ← `transformer_metrics.csv`：真实gate固定，执行裁剪率较高；此处延迟是批处理前向。

![Transformer诊断曲线](transformer_diagnostics.png)

### 固定真实轨迹：新计算点

以下由真实训练示范和最终epoch5/3730更新权重在CPU FP32重新计算，历史为teacher forcing，
不是原训练逐epoch曲线或自主rollout。CSV含checkpoint身份、样本与时间。

`force_fixed_trajectory.png` ← `../network_audit_20261031/force_fixed_trajectory.csv`：同专家观察上的expert/base/composed命令。

![Force固定真实轨迹](../network_audit_20261031/force_fixed_trajectory.png)

`masked_fixed_trajectory.png` ← `../network_audit_20261031/masked_fixed_trajectory.csv`：独立masked权重的同样比较。

![Masked固定真实轨迹](../network_audit_20261031/masked_fixed_trajectory.png)

### 自主接触和高度事件

`dp_G10_lift_event.png` ← `dp_G10_lift_event.csv`：短暂抬高后接触/动作变化，不能识别唯一掉落原因。

![DP抬高事件](dp_G10_lift_event.png)

`dp_closed_loop_progress.png` ← `dp_closed_loop_curves.csv`及`dp_closed_loop_outcomes.csv`；各线0/3，无稳定抬升。

![DP自主进展](dp_closed_loop_progress.png)

`transformer_closed_loop_s981004.png` ← `transformer_closed_loop_curves.csv`筛选seed981004：残差确有控制机会，未改善抬升。

![残差接触与高度](transformer_closed_loop_s981004.png)

保存物理初态相同但接触前命令不同；该图不能证明严格因果差异。

### 旧40遍、固定验证样本与其余自主轨迹

以下补充曲线的CSV从原保存指标/trace导出，不运行新模型或重算历史指标；来源SHA与权重见`curve_sources.json`。

`original40_validation.png` ← `original40_validation.csv`：原40遍只保存这些val40 EMA指标，缺失train/finger历史不补造。

![原40遍验证](original40_validation.png)

`G00-A_validation_trace.png` ← `dp_validation_examples.csv`筛选G00-A：固定专家观察上的预测仍有较大偏差。

![G00A固定验证示范](G00-A_validation_trace.png)

`G00-B_validation_trace.png` ← 同CSV筛选G00-B：改变损失后命令仍不是专家逐位复制。

![G00B固定验证示范](G00-B_validation_trace.png)

`G10-A_validation_trace.png` ← 同CSV筛选G10-A：触觉输入不保证固定轨迹每个动作准确。

![G10A固定验证示范](G10-A_validation_trace.png)

`G10-B_validation_trace.png` ← 同CSV筛选G10-B：这是teacher-forced未来target换算，非自主轨迹。

![G10B固定验证示范](G10-B_validation_trace.png)

`transformer_closed_loop_s981001.png` ← `transformer_closed_loop_curves.csv`筛选981001：少量接触修正未形成lift。

![残差981001自主轨迹](transformer_closed_loop_s981001.png)

`transformer_closed_loop_s981002.png` ← 同CSV筛选981002：全部native timeout，不能因物理初态相同声称严格因果配对。

![残差981002自主轨迹](transformer_closed_loop_s981002.png)

本轮[完整网络核验与近似训练初态测试](../../NETWORK_ARCHITECTURE_EXACT.md)将离线拟合、T0重放、T1学生闭环、
T2 ID和未运行T3分开；原诊断/正式0/50分母保持不变。
