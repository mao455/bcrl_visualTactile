<!-- METHOD_MATRIX_START -->

## 2026-11-10 复用 U-Net 基础与实际配置适配实验

机器状态 **COMPLETE** / `finite_queue_resolved`；源码 `e9385bcb97243ca4be4e1d495a2d6721ee6021fe`；运行 `/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_transformer_20261110_02`。
G00/G10直接复用既有 best EMA 与开发记录，本轮基础更新和基础评估均为0。Q00/QG00保留U-Net，共用旧G00 EMA热启动，使用q_reached新统计与新优化器；G/Q残差各自独立依赖。
固定train166/validation30与validation4选模。Q本轮有界适配最多100新q规划anchor遍历、50遍后可按验证平台停止；旧u训练曝光不相加。残差最多200/至少50完整控制tick遍历后平台停止，旧200接触tick遍历不能直接等同。每新后端10开发spec；test22不自动运行。
Transformer残差约75.7万参数，2层/128维/4头/FF256，当前有效H24计划与游标、独立上一名义输出、严格过去执行反馈、当前已采集触觉；持续GRU加32tick显式历史。失触不强制归零，全部valid tick监督。
未完成指标为未测；旧G11首次成功及原0/10、1/10独立保留。本轮无滑移/受阻恢复新监督，不声称已具备恢复能力。

几何软表面损失验收：True；几何阻塞只影响QG分支。内力模型BLOCKED，未启用；不声称完整接触力学等价。
[实现差异与完整合同](CONTRACTS.md) · [几何验收与全部工程尝试](evidence/tight_refactor_20261110/geometry_acceptance_summary.json) · [清理清单](CLEANUP_AUDIT.md)

|阶段（本轮开发）|组|本轮更新 / 历史更新|停止原因|原生完整成功/有效|
|---|---|---:|---|---:|
|command|G00|0 / 10132|existing_selected_EMA_reused|0/10|
|command|G10|0 / 6929|existing_selected_EMA_reused|0/10|
|configuration|Q00|7000 / —|validation_plateau_or_overfit|0/10|
|configuration|QG00|7000 / —|validation_plateau_or_overfit|0/10|
|command_residual|G01|27092 / —|validation_plateau_after_minimum_exposure|0/10|
|command_residual|G11|32500 / —|validation_plateau_after_minimum_exposure|0/10|
|configuration_residual|Q01|75823 / —|validation_plateau_after_minimum_exposure|0/10|
|configuration_residual|QG01|97441 / —|validation_plateau_after_minimum_exposure|0/10|

实际预算与原始指标：`/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_transformer_20261110_02/budget.json`、`result.json`及各组training/offline/evaluation；没有把启动或验收当成训练完成。
<!-- METHOD_MATRIX_END -->


<!-- QG_DETAILS_START -->
## 2026-11-11 QG 专项补充：损失、几何与逐次失败过程

本节只读已完成的 `tight_transformer_20261110_02`，补齐此前汇总表背后的 QG 证据；不是新增训练或重跑。
随后 `tight_dagger_20261111_01` 的前后端更新和自主评估由本页独立 DAgger 小节报告，不混用模型或分母。
**QG00、QG01 均为 0/10 完整成功；20 次都接触到物体，但都没有原生 lift、alignment 或插入子目标。**
这支持“接触后抓持／抬升未完成”的阶段判断，尚不能确定失败根因。

[完整目标、距离符号、权重与梯度范围](CONTRACTS.md#qg-loss-details) ·
[前端损失实现](../visuotactile/state_image_loss.py) · [完整更新与渐入调度](../visuotactile/anchor_continue.py) ·
[后端命令损失](../visuotactile/models/native_dp.py) ·
[本次导出机器结果及来源 SHA](evidence/tight_transformer_20261110_02/qg_details/summary.json)。

QG00 的 U-Net 以 `q_reached` 为 clean sample，Arm/Hand 归一化误差等权；再加 commit12 掌部 FK 和手部几何软惩罚。
几何容差仍为穿透 1mm、要求接触距离 2mm，未启用内力项。几何权重为 `2.355370821894758e-6`，来自训练集完整 update 的手部输出梯度比 0.2 校准；这不是以后每次梯度比的保证。
前 100 update 真正跳过几何计算，随后 400 update 渐入；每完整 update 最多两个配置，有条件时接触／非接触各一。
QG01 冻结选定 QG00 后训练 Transformer，以最终 `u_expert` 命令为监督；后端没有新增几何 loss。

|旧离线训练项目|QG00 前端|QG01 后端|
|---|---:|---:|
|正式更新|7,000|97,441|
|曝光|82.588913 规划 anchor 等效遍|180 完整控制 tick 等效遍|
|选定更新|7,000|81,232（epoch150）|
|训练阶段墙钟，含验证/保存/等待|29,577.308s（8.216h）|4,283.153s（1.190h）|
|训练循环计时，后端独立口径|—|4,262.549s|
|停止|validation 平台／过拟合规则|最低曝光后的 validation 平台|

Q00 使用相同起点与曝光，墙钟 28,141.375s；QG00/Q00 实测阶段墙钟比约 1.051。
它包含并行资源竞争、等待及验证，不能全部归因于几何计算。
QG00 有 6,900 个 update 启用几何、4,069 个 update 的几何 loss 非零；实际计算 13,800 个配置，其中 4,112 个记录有效手部几何梯度，SLSQP 调用为 0。
距离计算累计 183.868s，约为记录的 update 累计耗时 29,230.382s 的 0.629%；不包含所有采样或同步开销。
因此某次 progress 中 `hand_geometry=0` 只表示那次采样的 hinge 为零。

|QG 专项图表与原始数值|前端 QG00|后端 QG01|
|---|---|---|
|训练 loss 分量与固定验证曲线|[图](evidence/tight_transformer_20261110_02/qg_details/QG00/training/curves.png)|[图](evidence/tight_transformer_20261110_02/qg_details/QG01/training/curves.png)|
|完整 validation，包括掌部旋转|[CSV](evidence/tight_transformer_20261110_02/qg_details/QG00/training/validation.csv)|[CSV](evidence/tight_transformer_20261110_02/qg_details/QG01/training/validation.csv)|
|10 次自主接触／高度过程|[图](evidence/tight_transformer_20261110_02/qg_details/QG00/rollouts.png)|[图](evidence/tight_transformer_20261110_02/qg_details/QG01/rollouts.png)|
|逐 tick 自主数值，每例独立压缩 CSV|[目录](evidence/tight_transformer_20261110_02/qg_details/QG00/rollouts/)|[目录](evidence/tight_transformer_20261110_02/qg_details/QG01/rollouts/)|
|专家历史离线误差与代表轨迹|[目录](evidence/tight_transformer_20261110_02/qg_details/QG00/offline/)|[目录](evidence/tight_transformer_20261110_02/qg_details/QG01/offline/)|

![QG00 几何损失、有效梯度与距离](evidence/tight_transformer_20261110_02/qg_details/QG00/training/geometry_curves.png)

几何[逐配置原始数值](evidence/tight_transformer_20261110_02/qg_details/QG00/training/geometry_projection.csv.gz)与[逐 update 聚合](evidence/tight_transformer_20261110_02/qg_details/QG00/training/geometry_by_update.csv.gz)保留源文件命名；`geometry_projection` 是历史文件名，本轮实际计算是直接软惩罚。
全部 7,000 次更新的各 loss／实际权重也在[原始 CSV](evidence/tight_transformer_20261110_02/configuration_QG00_updates.csv.gz)，不会只展示最后一次更新。

几何验证并不理想：validation 868 个抽样 tick 中，预测配置有 492 个超过 1mm 穿透，最大 16.795mm；要求接触保持为 455/1,186（38.364%）。
同批记录的实际到达配置为 49/868 超容差、最大 2.568mm、保持 1,143/1,186（96.374%）。
这是**专家观测与记录物体场景下的预测配置 FK 几何**，不是自主 rollout 中实测的物理穿透；也不能单凭该差距断言它造成了闭环失败。

|自主开发结果|QG00|QG01|
|---|---:|---:|
|合法且软件完整 / 尝试|10 / 10|10 / 10|
|完整成功 / 原生 lift / 首子目标 / 第二子目标|0 / 0 / 0 / 0|0 / 0 / 0 / 0|
|发生物体接触|10|10|
|稳定抓持诊断|0（从原始日志重算）|0（重算与原结果一致）|
|timeout / fall / dropped|10 / 0 / 0|9 / 1 / 0|

QG01 `palm_fresh_006` 在完成第 569 个控制转换后记录 fall；其余 19 次均在第 601 个转换后 timeout。
首次接触阈值是物体过滤五指力模长 >0.1N，任一 120Hz 子步满足；tick 为 **1-based 已完成转换编号**。

|案例尾号|QG00 首接触 / 首双指 tick|QG01 首接触 / 首双指 tick|
|---|---:|---:|
|000|101 / 104|72 / 无|
|001|72 / 91|92 / 94|
|002|82 / 83|68 / 109|
|003|76 / 79|66 / 66|
|004|85 / 95|74 / 136|
|005|90 / 102|71 / 95|
|006|93 / 103|72 / 219|
|007|75 / 89|62 / 68|
|008|82 / 86|76 / 77|
|009|64 / 102|70 / 74|

完整事件、连续接触时长、终止原因和来源 SHA 见 [events.csv](evidence/tight_transformer_20261110_02/qg_details/events.csv)。
“有双指接触”不等于稳定抓持。QG00 原 result 缺 contact 汇总，本次按既有 `stable_grasp()` 规则读取 numeric.h5 的掌部位姿重算，明确保留计算来源，未将缺失字段当成零。
自主曲线显示实际关节与执行器目标跟踪，不按同 tick 对齐专家命令。

三段已录制视频（QG00 一段、QG01 两段，均601帧/60fps）及解码凭据见[路径、SHA 与帧数索引](evidence/tight_transformer_20261110_02/qg_details/video_index.csv)。视频仍留本机，GitHub 上是索引而非 MP4 本体。
本地前缀为 `outputs/visuotactile/run/tight_transformer_20261110_02/`：
`configuration/QG00/evaluation/palm_fresh_000/rollout.mp4`；
`configuration_residual/QG01/evaluation/palm_fresh_000/rollout.mp4`、`palm_fresh_001/rollout.mp4`。

此处的阶段归类未使用缺失 `approach_proxy` 时默认返回 approach 的历史简化函数；实际 trace 已证明接触发生。
数据、几何、跟踪与残差幅度可用于进一步定位，但本次没有进行干预实验，不能把 Transformer、几何、触觉或控制器中的某一项写成已证实根因。
每组十次是有限开发评估，test22 未使用；本次未新增 Kit、训练、推理、评估或清理。

复现 CPU 导出：

```bash
bash visuotactile/run_cpu.sh -m tools.tight_diagnostics qg_publish \
  --root outputs/visuotactile/run/tight_transformer_20261110_02 \
  --output research_docs/evidence/tight_transformer_20261110_02/qg_details
```
<!-- QG_DETAILS_END -->

# Tight-Insertion 当前结果

<!-- RESIDUAL_PAIR_START -->
## 2026-11-10 G01/G11：完成训练与20次开发评估

**G01完整成功0/10；G11完整成功1/10。** 两组各10次合法、无拒绝/中断，test22未使用。
G11成功是`palm_fresh_000`（seed27150000），279个控制tick/4.65s、原生两子目标与`max_successes=true`，成功后0额外动作。
该例`native_lift=false`，不把完整成功或稳定抓持诊断反推成原生抬升；G11另有一例记录原生lift。
这是有限开发结果，不是正式test成功率，也不能据1例证明稳定改善或触觉因果收益。

源码`c3386b69113595806dc080149f8c2104ded509e4`；run `outputs/visuotactile/run/tight_residual_20261110_01`。
2026-11-10 10:14:46启动，12:01:02（北京时间）结束，总墙钟1小时46分16秒；status=COMPLETE、exit=0，自有owner/worker/monitor已退出，无下一轮。
[公开机器结果](evidence/tight_residual_20261110/summary.json) · [20例逐次结果](evidence/tight_residual_20261110/closed_loop.csv) · [冻结配置](evidence/tight_residual_20261110/configuration.json) · [验收](evidence/tight_residual_20261110/acceptance.json)。

|开发指标|G00冻结基础|G01残差|G10冻结基础|G11残差|
|---|---:|---:|---:|---:|
|原生完整成功/有效|0/10|0/10|0/10|1/10|
|稳定抓持诊断|4|1|3|3|
|原生lift|1|0|3|1|
|首子目标 / 第二插入子目标|3 / 0|0 / 0|1 / 0|1 / 1|
|原生alignment|3|0|1|1|
|timeout / fall / dropped|9 / 1 / 0|10 / 0 / 0|7 / 3 / 0|8 / 1 / 0|

基础闭环复用既有记录，按相同预登记初态spec/seed比较，没有重跑基础；不宣称隐藏模拟器状态逐位相同。
G01的阶段到达没有改善；G11出现一次完整成功，但原生lift数量少于基础，不能只选有利指标解释。

两组都完成98,610次残差更新、7,620,800个接触tick（人口38,104）、200个接触tick/控制tick等效遍历；
基础策略更新数均为0。G01/G11训练墙钟4069.57/4062.56s，等待24.23/29.66s。
达到预设>=200遍及最近4次validation平台条件后停止，未耗尽9小时预算；这不是DP重规划anchor epoch或已学会任务的证明。
固定validation4接触归一化Arm/Hand等权MSE选定G01 epoch110/update54264、G11 epoch180/update88771；最终train8/validation30各38条诊断，不使用闭环挑选模型。
并行接触anchor吞吐4025.92/s、串行2548.15/s，因此后台选择并行。last/best均留在各组`training/`，权重SHA见公开结果。

|validation30专家历史上的离线指标|G00 nominal|G01组合|G10 nominal|G11组合|
|---|---:|---:|---:|---:|
|Arm RMSE (rad)|0.00542309|0.00542475|0.00564516|0.00564609|
|Hand RMSE (rad)|0.05523516|0.05461565|0.05464898|0.05357456|
|目标掌部FK RMSE (mm)|4.930307|4.929388|5.130683|5.130097|
|目标掌部FK平均旋转误差 (deg)|0.706364|0.706228|0.733293|0.734046|

每列相同validation30/10,238个tick，nominal为各自冻结EMA完整DDIM16真实预测；组合动作逐60Hz tick更新。
这些是专家观测和过去专家执行反馈上的误差，掌部是命令FK，不是实际物理轨迹偏离。Arm几乎不变，Hand有小幅降低；自主轨迹不按同tick对齐专家动作。

![冻结基础与残差的离线误差](evidence/tight_residual_20261110/comparison.png)

[G01训练/验证曲线](evidence/tight_residual_20261110/G01/training_curves.png) · [G11训练/验证曲线](evidence/tight_residual_20261110/G11/training_curves.png) · [G01阶段误差](evidence/tight_residual_20261110/G01/stage_errors.png) · [G11阶段误差](evidence/tight_residual_20261110/G11/stage_errors.png)。
原始指标包含每次更新loss、完整逐tick误差、阶段/运动样本数、hold参照和代表轨迹目标/变化量CSV；压缩文件只含数值记录，不含源轨迹或权重。

固定validation4把残差world/local触觉力置零，保持nominal/接触门/FK不变，输出确有变化：G01接触输出RMS差0.000855–0.001157rad，G11为0.001219–0.001446rad。
误差收益很小且G11方向不一致；这证明对力输入敏感，不证明触觉带来闭环成功。详见[原始敏感性指标](evidence/tight_residual_20261110/force_sensitivity.csv)。

4段视频均完整CPU解码，帧数G01为601/601，G11为279/601，60fps。视频本体留本地；[路径、SHA与解码凭据](evidence/tight_residual_20261110/summary.json)。
本地路径前缀为`outputs/visuotactile/run/tight_residual_20261110_01/`：

- **成功视频**：`G11/evaluation/palm_fresh_000/rollout.mp4`。
- G11失败：`G11/evaluation/palm_fresh_001/rollout.mp4`。
- G01失败：`G01/evaluation/palm_fresh_000/rollout.mp4`、`palm_fresh_001/rollout.mp4`。

正式启动至结束资源记录6,374条，整卡时间加权均值79.17%，符合本轮充分使用空闲GPU授权；最大采样间隔1.1382s，无>8s内部缺口。
最小空闲显存19,541MiB、可用RAM25.696GiB；开头0.059s/尾部1.055s未被共享样本覆盖。
继承的monitor汇总包含前置验收及其监控重启间隔，仍记录807.409s缺口；不能把它改写成全会话无缺口，也不能把整卡占用当成每进程利用率。

评估日志末尾2026-11-10T04:00:56Z仍有4条PhysX非kinematic刚体速度设置错误，完整摘录已公开；其影响未专项审计。
20例独立完成凭据、原生成功字段、视频及exit0如实保留，不能仅凭exit0声称日志无错误或错误无影响。原G00/G10的FAILED收尾记录也保持不变。

本次发布只读取完成产物、核对SHA/分母/视频并导出CSV/JSON/9张图；没有新训练、推理、Kit、补录或清理。
同一CPU导出命令：

```bash
bash visuotactile/run_cpu.sh -m tools.tight_diagnostics publish \
  --root outputs/visuotactile/run/tight_residual_20261110_01 \
  --evaluation outputs/visuotactile/run/tight_residual_20261110_01 \
  --output research_docs/evidence/tight_residual_20261110
```
<!-- RESIDUAL_PAIR_END -->






<!-- VISUAL_PAIR_START -->
## 2026-11-10 G00/G10：测量完成，程序收尾失败

两组训练均按固定墙钟预算停止，各完成最终train8/validation30离线诊断与10次合法开发闭环。
**G00完整成功0/10、G10完整成功0/10；两组拒绝、中断、未启动均为0。**
两段预登记视频完整解码通过。2026-11-10 01:51:00 +08:00写入评估完成凭据，01:51:06自有worker/monitor全部退出。

原run和接续run的机器状态均保留`FAILED`，不改写成正常完成。
原失败发生在首次控制前：评估器漏传当前实测q/qd，0控制transition/0完整trial，修复后在新目录接续。
接续的20例均已有独立`software_complete=true`结果；最后`exit_completed_kit_worker`读取root/result.json，
要求顶层`software_complete=true`，而本队列写的是`evaluation_complete.json`及普通报告result.json，导致退出检查报错。
因此是**测量完成、收尾接口失败**。本次仅发布既有结果及CPU导出代码，不修写历史状态、不重跑Kit。

|项目|G00：视觉+机器人|G10：另加观测触觉|
|---|---:|---:|
|实际更新 / 选定EMA更新|10132 / 10132|6929 / 6929|
|采样anchor|560646|383422|
|anchor等效epoch（人口4690）|119.540725|81.753092|
|停止原因|fixed_wall_budget|fixed_wall_budget|
|完整成功 / 有效次数|0 / 10|0 / 10|
|超时 / 掉落|9 / 1|7 / 3|
|原生lift / 首子目标 / alignment|1 / 3 / 3|3 / 1 / 1|

训练阶段约8h50m36s，9h预算中保留了最后收尾窗口；尚未达到原至少200等效epoch的平台判定起点，不称收敛。
G00/G10的资源等待计数分别127.10s/198.39s，仅代表trainer已记录的等待，不等于全部竞争或GPU空闲时间。
两组曝光不同，观察到的差别不能直接归因于触觉。20例使用相同预登记reset spec和采样seed，不声称隐藏仿真状态逐位配对。
完整成功为原生任务判据；lift、首子目标、alignment是分别统计的阶段事件，不冒充任务成功或可靠抓持。

|相同专家历史validation30、commit12|StateDP Palm19865|G00|G10|保持当前执行器目标|
|---|---:|---:|---:|---:|
|Arm RMSE (rad)|0.005645|0.005423|0.005645|0.007764|
|Hand RMSE (rad)|0.056618|0.055235|0.054649|0.116285|
|掌部命令FK RMSE (mm)|4.9823|4.9303|5.1307|未测|
|掌部命令FK平均角差 (deg)|0.7034|0.7064|0.7333|未测|
|指尖命令FK RMSE (mm)|见旧原始指标|8.7343|8.8921|未测|

完整真实专家前缀、持续GRU、DDIM16/seed7091、反归一化后截取commit12；validation30有10238个有效tick。
图中的训练曲线为固定train2/validation4，最后汇总为train8/validation30（2762/10238 tick），两种样本量不混称。
H24的有效位置含重叠，不当作独立物理tick。自主轨迹与专家进度不同，不计算同tick专家标签误差。
全部20例提交目标/执行器接口最大误差为0，实际关节tracking误差非零，逐例值见闭环CSV。
两组整体关节误差均低于hold，但这没有转化为完整任务成功；末期validation改善有限，训练/验证误差仍有差距。
分阶段与静止/运动结果及样本数全部发布；validation alignment仅30 tick、完成后retract仅19 tick，不能据此作稳定阶段能力结论。

![G00/G10完整采样误差与不等曝光](evidence/tight_vision_20261110/comparison.png)

|可复查图表|G00|G10|
|---|---|---|
|commit12与有效H24完整采样|[图](evidence/tight_vision_20261110/G00/sampling_curves.png)|[图](evidence/tight_vision_20261110/G10/sampling_curves.png)|
|训练分项与固定验证loss|[图](evidence/tight_vision_20261110/G00/loss_curves.png)|[图](evidence/tight_vision_20261110/G10/loss_curves.png)|
|原阶段误差、StateDP/hold及n|[图](evidence/tight_vision_20261110/G00/stage_errors.png)|[图](evidence/tight_vision_20261110/G10/stage_errors.png)|
|运动/近静止误差|[图](evidence/tight_vision_20261110/G00/motion_errors.png)|[图](evidence/tight_vision_20261110/G10/motion_errors.png)|
|验证代表轨迹目标/变化量|[图](evidence/tight_vision_20261110/G00/formal_002_slow_0_targets.png)|[图](evidence/tight_vision_20261110/G10/formal_002_slow_0_targets.png)|
|训练代表轨迹目标/变化量|[图](evidence/tight_vision_20261110/G00/formal_003_slow_0_targets.png)|[图](evidence/tight_vision_20261110/G10/formal_003_slow_0_targets.png)|
|10例闭环原始指标|[CSV](evidence/tight_vision_20261110/G00/closed_loop.csv)|[CSV](evidence/tight_vision_20261110/G10/closed_loop.csv)|

数据仍为218 Core（166/30/22），Aux15未加入；本轮没有新增数据，test22未用于选模或评估。
训练源码`98dfafe77bec63c7d7e69d65c01037f4373618c7`；修复后的评估源码`a82c33f3c1079a7e1d5db1bdd1ea9410e2143646`。
选定G00 best SHA256：`4356acbcaabd95e97b1f236f853d724db5f3ad5d9effad2301fe0e1eca4ba452`；
G10 best：`5f3460ab07bca11d0fa33fc5976a0d3e4fcc77c6823413ab208ad04b4c4efd7b`。
原训练root为`outputs/visuotactile/run/tight_vision_20261109_01`，接续root为其`evaluation_resume_01`。
完整checkpoint、源轨迹、缓存和视频保留本地；公开指标中的路径和SHA用于核查，不等于已上传原载荷。

本地视频位于接续root的`G00/evaluation/palm_fresh_000/rollout.mp4`（523帧、60fps）
和`G10/evaluation/palm_fresh_000/rollout.mp4`（399帧、60fps）。解码凭据与视频SHA见summary.json，不计额外评估次数。

本轮完整GPU授权下，原run监控整卡均值92.169%、完整60s最大96.652%；存在537.81s遥测缺口，最大采样间隔318.63s，不能声称全程资源已证实合规。
接续评估监控均值24.250%、完整60s最大33.830%、遥测缺口0；实测可用RAM最低27.075GiB，最高整卡显存使用4555MiB。
原monitor摘要内的RAM阈值为12GiB，实际FullLease运行准入依据session中的14GiB，不能把摘要阈值当作实际准入配置。
monitor进程RSS不代表trainer/Kit总RSS。监控原始摘要与冻结资源约束一并发布，6GiB显存/14GiB RAM/60GiB磁盘储备未放宽。

[机器汇总与原错误](evidence/tight_vision_20261110/summary.json) · [跨组原始曲线](evidence/tight_vision_20261110/sampling_curves.csv) · [全部分阶段指标](evidence/tight_vision_20261110/stage_metrics.csv) · [数据身份](evidence/tight_vision_20261110/dataset_index.csv) · [冻结配置](evidence/tight_vision_20261110/frozen_configuration.json) · [原产物SHA](evidence/tight_vision_20261110/source_artifacts.json) · [公开文件SHA](evidence/tight_vision_20261110/artifacts.json)

不自动继续训练、扩大评估、启动G01/G11或采集。当前证据未证明触觉带来因果收益或基础策略可靠完成任务。
<!-- VISUAL_PAIR_END -->

## 2026-11-09 结案与可复查图表

掌部几何 StateDP 已完成 **19,865 更新、1,099,214 个采样 anchor、234.374 anchor 等效 epoch**。
最后四次固定 validation4 选模分数改善 **0.7646%**，触发原平台停止条件；未达到300等效epoch目标，不称全局收敛。
最终新初态队列 **0/10 完整成功**：10次合法并自然结束，8 timeout、2 fall，拒绝/中断/未启动均0。
2次达到原生lift、2次达到首子目标，两者不是同一组案例；前3个预登记身份的视频全部验收。
没有自动增训、补seed或扩为100次。原启动在Kit前资源终止，物理0次，单独保留，不并入这10次。

训练执行源码为 `35485d77830b109d61491d3f905f74c865f44c03`，评估接续登记版本为 `a1c16e5023a066026ce884b98aa283945ef7c8bc`；
选定 best19865 EMA SHA256 为 `6558f411eeffe7b55f5012de6b103ac4b8ace94a13e17cc1a22e9ff0da6be6bc`。
本次发布只增加 CPU 指标导出/绘图、文档和小型指标文件，没有修改训练、模型或仿真行为。
checkpoint、RGB-D/HDF5轨迹、视频保留本地，发布的数据身份索引不是训练轨迹本体。

|相同专家历史上的commit12指标|旧Core8500|新best19865|保持当前执行器目标|
|---|---:|---:|---:|
|train8 Arm RMSE (rad)|0.001428|0.000632|0.007797|
|train8 Hand RMSE (rad)|0.009568|0.004542|0.117869|
|validation30 Arm RMSE (rad)|0.006517|0.005645|0.007764|
|validation30 Hand RMSE (rad)|0.057276|0.056618|0.116285|
|validation30 掌部目标FK RMSE (mm)|5.2635|4.9823|未测|
|validation30 掌部目标FK平均角差 (deg)|未测|0.7034|未测|

新数据为218 Core（train166/validation30/test22），Aux15未加入；208条沿用上一轮，18次补充物理尝试产生10条新增Core，17次专家完整成功、1次时间中断、7条Core准入拒绝。家族split与train-only统计保持。
新旧最终38条诊断身份相同，均使用完整真实专家前缀、持续GRU、batch1 DDIM16/seed7091、反归一化后commit12。
训练中的曲线固定train2/validation4；最后诊断是train8/validation30，两种样本量不混称。H24曲线统计每个plan所有有效位置，存在重叠，不能把数量叫独立物理tick。
数据扩充、fresh初始化、掌部损失和曝光量同时变化（旧100.21 vs 新234.37等效epoch），以上差异不是单项损失或条件分支的因果收益。

![完整部署采样误差曲线：commit12与有效H24](evidence/tight_palm_20261109/sampling_curves.png)

![训练分项与固定样本去噪loss](evidence/tight_palm_20261109/loss_curves.png)

训练误差下降，但验证去噪loss没有同步下降，完整采样的验证动作误差趋于平台；不能只凭训练loss宣称动作已经学好。
图中没有新增平滑或补造未测的旧掌部角度指标。

![分阶段误差、hold基线及有效样本数](evidence/tight_palm_20261109/stage_errors.png)

![运动与近静止目标误差](evidence/tight_palm_20261109/motion_errors.png)

沿用原阶段与运动划分，n为有效样本tick。validation的native_alignment仅30、完成后retract仅19tick，阶段均值应谨慎解释。
validation近静止Hand仅40tick，新模型RMSE 0.01713rad，高于hold的0.00829rad；运动Hand有10198tick，新模型0.05672rad、hold 0.11651rad。
因此总体胜过hold并不表示静止阶段也更稳。

![固定validation轨迹目标与变化量叠图](evidence/tight_palm_20261109/formal_002_slow_0_targets.png)

另一条固定train代表轨迹：[目标/变化量图](evidence/tight_palm_20261109/formal_003_slow_0_targets.png)。两图保留接触、抬升、重规划标记，全部29关节数值见对应CSV。
这些是专家历史上的目标预测误差。10次自主闭环的提交/执行器接口最大误差均0，但实际关节tracking误差非零；自主与专家进度不同，不作同tick标签误差。
首次自主行为偏离的严格配对定位仍未测。3个本地视频见 `outputs/visuotactile/demos/tight_palm_eval_20261109_01/remaining_01/README.md`。

[机器结果/资源](evidence/tight_palm_20261109/summary.json) · [10次闭环逐例指标](evidence/tight_palm_20261109/closed_loop.csv) · [数据身份](evidence/tight_palm_20261109/dataset_index.csv) · [冻结配置](evidence/tight_palm_20261109/frozen_configuration.json) · [文件SHA与完整产物索引](evidence/tight_palm_20261109/artifacts.json)

图由同目录CSV/JSON产生；`sampled_ticks.csv.gz`、`full_horizon_ticks.csv.gz`和`every_update_loss.csv.gz`为无损压缩的原始指标。
本地重建命令（只读完成的run，不加载模型或Kit）：

```bash
bash visuotactile/run_cpu.sh -m tools.tight_diagnostics publish \
  --root outputs/visuotactile/run/tight_palm_20261108_01 \
  --evaluation outputs/visuotactile/demos/tight_palm_eval_20261109_01/remaining_01 \
  --output research_docs/evidence/tight_palm_20261109
```

训练整卡均值34.281%、最高完整60秒72.127%，遥测缺失0，**训练不属于全程资源合规**；本次10例评估对应21.990%/35.456%，缺失0。
整卡读数不能归因给单个进程，未用显存限制冒充硬GPU利用率配额。

已按用户指定清理上一轮 `tight_state_20261107_01`：移除1个可重建重复数值缓存，13个大日志无损压缩后删除原文件，实际回收303,198,208 bytes（289.15MiB）。
当前208条继承轨迹仍位于旧目录，连同Aux、原速参照、旧best/normalizer和失败分母保留；没有整体删除旧root。
逐文件删除/归档清单见[CSV](evidence/tight_palm_20261109/cleanup_files.csv)、[校验记录](evidence/tight_palm_20261109/cleanup.json)与[恢复说明](CLEANUP_AUDIT.md)。下文保留历史证据。

<!-- tight-palm-training:start -->
## 掌部几何损失与有限训练数据扩充（当前任务）

状态 **COMPLETE**，阶段 `trained_and_offline_evaluated`。
基础208 Core/train156；新增10，物理尝试18。validation30/test22及固定诊断身份不变，Aux不加入。
全新训练更新19865，anchor等效epoch 234.374；目标300，平台停止最早200，训练墙钟9h优先。
loss = 原完整29D扩散Loss B + 0.3342396122 × 平均掌部位置平方距离(m²) + 0.02994639475 × 平均0.5旋转矩阵Frobenius平方；仅有效commit12，500更新渐入。
实际掌部/关节/决策前执行器目标和严格过去已施加目标已有60Hz通路，复用178/85维，不重复追加；GRU256不变。
随后按用户追加授权完成10例闭环，完整成功0/10；详情见上方结案，不再自动追加。
run：`/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_palm_20261108_01`；`status.json`、`training/progress.json`、`training/validation.jsonl`、`training/curves.png`、`offline/stage_metrics.json`。
已有旧诊断38条/1103次完整采样/13000tick仅为离线参照，不是任务成功数。

|实际字段/切片|动作前时间语义与来源|每tick网络路径|重置与反传|
|---|---|---|---|
|robot[0:58]：q29/qd29，rad/rad·s⁻¹|diagnostics_before实测canonical状态，时刻t|60Hz native_encoder→GRU；当前/过去13帧另直达condition|真实episode重置；完整0…t前向，当前权重重算前缀|
|geometry[28:35]：掌部xyz3+wxyz4|iiwa14_link_7实测刚体，robot base坐标，m/单位四元数；非控制目标FK|60Hz geometry_encoder→GRU；t−24/t−12/t另直达condition|同上，四元数w≥0；train-only mean/std|
|robot[58:116]：目标29/目标−q29|target_before_rad＝动作前已生效执行器目标，rad|同robot两条路径|不读取本tick target_after标签|
|robot[116:174]：过去目标29×2|严格t−1实际施加target_after；保留既有两个兼容槽，两者相等；不是专家滤波前request|每60Hz tick反馈，commit内部仍更新GRU|t=0置零，valid=0；重规划不清空|
|robot[174:178]：age2/valid/elapsed|前次目标距当前时间，s/bool/s|同robot路径；valid不标准化|真实episode重置，缺失窗口全零且valid=false|
|Actor141与geometry其余78维|既有Actor噪声/延迟、物体/目标/孔/桌/指尖几何；不新引入传感器|Actor/robot/geometry每tick拼384→GRU256；条件仍896→512|anchor t的反传区间[96⌊t/96⌋,t]，≤96tick（当前12tick anchors为1–85）；更早历史no_grad重算，不缓存旧权重hidden|

可丢弃验收：checkpoint/EMA/normalizer完全匹配，109tick实际在线controller与完整prefix输入完全相同，condition max1.90735e−6；FK真实位姿与梯度验收见palm_source_check.json，最终新增数据统计下再验收。

|固定专家历史诊断|Arm/hold (rad)|Hand/hold (rad)|掌部FK (mm/deg)|tick|
|---|---:|---:|---:|---:|
|train/all|0.000632/0.007797|0.004542/0.117869|0.5419/0.0845|2762|
|train/approach_no_object_contact|0.000587/0.007792|0.004082/0.098436|0.4912/0.0772|776|
|train/contact_low_object|0.000627/0.007968|0.004316/0.143243|0.5528/0.0847|543|
|train/transport_above_3cm|0.000649/0.007743|0.004902/0.111795|0.5580/0.0878|1032|
|train/native_alignment|0.000630/0.007429|0.004681/0.098119|0.5571/0.0998|8|
|train/second_insertion|0.000674/0.007718|0.004715/0.129291|0.5771/0.0893|403|
|train/ordered_complete_retract|未测|未测|未测|0|
|validation/all|0.005645/0.007764|0.056618/0.116285|4.9823/0.7034|10238|
|validation/approach_no_object_contact|0.004322/0.007705|0.046959/0.097322|4.4282/0.5423|2742|
|validation/contact_low_object|0.005492/0.008156|0.058060/0.144008|5.4379/0.7520|1978|
|validation/transport_above_3cm|0.005715/0.007735|0.059125/0.109322|4.8838/0.7285|3995|
|validation/native_alignment|0.007172/0.007255|0.058593/0.087002|4.5751/0.8916|30|
|validation/second_insertion|0.007466/0.007379|0.063849/0.124859|5.5401/0.8643|1474|
|validation/ordered_complete_retract|0.007939/0.010099|0.037956/0.161878|6.1796/0.8479|19|
<!-- tight-palm-training:end -->

<!-- tight-dp3-diagnostics:start -->
## 2026-11-08 DP3风格StateDP条件与诊断

状态 **COMPLETE**；阶段 `diagnostics_complete_no_training`；已完成离线轨迹 38/38。
原因：Finite diagnosis complete; equivalent path and existing plateau do not justify retraining。正式新增更新0；新Kit/闭环启动0。
等价当前robot/geometry直达condition已存在，保留原模型/GRU/U-Net与旧checkpoint。
旧最近四次验证改善 0.9386%，验证/train8采样MSE比 35.52；依据train/validation冻结为诊断后停止，不自动续训。

本轮root：`/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_dp3_20261108_01`；状态`status.json`，日志`service.log`，验收`acceptance.json`，
冻结身份`session.json`/`inputs.json`/`source_freeze.json`。
图均由本run CSV生成：`historical_curves.csv/png`为原17点完整DDIM曲线（旧episode-batch噪声）；
`sampled_ticks.csv`/`stage_metrics.json`为当前同部署batch1顺序、固定seed7091、真实完整专家前缀、前12步目标误差；
`*_targets.csv/png`含29关节目标/变化量原始数值及固定joint3/7示例、接触/抬升/重规划标记。
`closed_loop_reused.csv`/`closed_loop_summary.json`与`*_physical.csv/png`复用旧27次自主轨迹。
未生成文件代表该阶段未完成，不代表零误差或零成功。离线目标FK误差、命令接口误差和物理tracking各自独立；
自主与专家不同进度下第一次物理偏离未测，不按同tick作动作误差。阶段事件只作诊断，且含transition接触/对齐信息，绝不输入网络。
资源上限：整卡批次/完整60s均值50%，45%准入、35%目标、6GiB显存/14GiB RAM/60GiB磁盘储备；最多1h含等待。
实测资源见`resources/resources.json`与`resource_summary.json`，启动不等于完成或全程资源合规。

|专家历史离线集/阶段|Arm RMSE / hold (rad)|Hand RMSE / hold (rad)|有效tick|
|---|---:|---:|---:|
|train / all|0.001428 / 0.007797|0.009568 / 0.117869|2762|
|train / approach_no_object_contact|0.001334 / 0.007792|0.009049 / 0.098436|776|
|train / contact_low_object|0.001381 / 0.007968|0.009470 / 0.143243|543|
|train / transport_above_3cm|0.001460 / 0.007743|0.009919 / 0.111795|1032|
|train / native_alignment|0.001306 / 0.007429|0.009169 / 0.098119|8|
|train / second_insertion|0.001573 / 0.007718|0.009760 / 0.129291|403|
|train / ordered_complete_retract|未测|未测|0|
|validation / all|0.006517 / 0.007764|0.057276 / 0.116285|10238|
|validation / approach_no_object_contact|0.005032 / 0.007705|0.047946 / 0.097322|2742|
|validation / contact_low_object|0.006486 / 0.008156|0.057774 / 0.144008|1978|
|validation / transport_above_3cm|0.006448 / 0.007735|0.059106 / 0.109322|3995|
|validation / native_alignment|0.009234 / 0.007255|0.059569 / 0.087002|30|
|validation / second_insertion|0.008713 / 0.007379|0.066832 / 0.124859|1474|
|validation / ordered_complete_retract|0.009092 / 0.010099|0.042326 / 0.161878|19|


旧闭环复用：new 0/10成功，拒绝0；train 0/10成功，拒绝0；validation 0/7成功，拒绝0。不加入新实验分母。

<!-- tight-dp3-diagnostics:end -->


## 上轮Core任务结论：数据与有限训练完成，学生完整任务成功仍为零

新版合同下已冻结208条Core＋15条Auxiliary，固定Arm=.50、Hand=1、Hand cap=2rad/s。随机初始化Core-only StateDP完成8500更新、100.2068 anchor等效epoch；真实闭环train **0/10**、validation **0/7**、新初态 **0/10**，27次均合法。新初态首10例零成功，按预登记规则不扩至100；没有补seed或重复旧7例。

生产不再被少量时长越界整体阻断，也没有为凑300继续采样。此方案能通过逐条筛选产出合格数据，不能保证每次尝试都满足倍率/路径要求。**本轮未证明新Core学生具有完整闭环成功能力；不继续增加实验、换网络/损失或开启ImageDP/触觉训练。** 数据通过、有限训练充分性通过与学生失败分别报告。

工作树`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`，branch `feature/tight-state-foundation`；起点干净`e26e3d92e7daa39bc695a9bef3bce64b49ca7538`。采集源码`4d43a3975c9bd153ff90d7c13e03ca9a0e6aa917`，训练/评估源码`fdb7ea94a1457de57ff2a9fc21d1a9486742d5ab`；后者仅修按实际Core成员选择闭环家族。采集、模型、训练与控制函数未变，无旧学生恢复。没有退回IDE原工作树。

root：`outputs/visuotactile/run/tight_state_20261107_01/composition_06`。原服务2026-11-07 22:30启动，11月8日13:21:54全部阶段结束，owner/collector/trainer/Kit/monitor与附加资源guard均实际退出，服务exit0。没有重置原采集/整体deadline。

[当前合同](CONTRACTS.md) · [433次数据尝试表](evidence/tight_composition_attempts_20261108.csv) · [数据、训练、27次闭环及清理机器结果](evidence/tight_composition_results_20261108.json) · [不可变启动前登记](evidence/tight_composition_20261107.json)

## 最终数据与真实生产效率

|冻结组成|条数 / 家族|train / validation / test条数|时长比范围 / 中位数|
|---|---:|---:|---:|
|Core|208 / 50|156 / 30 / 22|1.5067–2.9871 / 2.0000|
|Auxiliary|15 / 11|10 / 1 / 4|1.4124–3.2297 / 1.4394|

Core家族split为36/7/7。55个登记家族中，formal_004/008/009/010各7次慢速均未产Core；formal_033原速初态拒绝后未启动依赖慢速。没有删除这些家族的分母、改变初态范围或替换seed。

总账433次＝55原速参照（54成功、1动作前拒绝）＋378慢速（359完整成功、11任务失败、8动作前拒绝）。合法专家成功359/370=97.03%；Core产出208/378=55.03%，不能把逐条入选全部合格称作尝试100%成功。11次任务失败为7次抬升运输timeout、2次插入timeout、2次抬升运输fall；无采集软件/传感器故障。判定器对动作前拒绝给出的missing/invalid占位字段不是采集后坏帧。

第7个完整家族轮次达到目标即停止。62个计划身份未启动：8个缺成功参照、54个因软目标达到。新规则数值分类208Core、19Aux、151拒绝；19Aux中2条旧记录缺完整RGB-D不可复用，17条完整候选按配额选15、保留2未选。Aux在新采集前冻结为[1.35,1.5)或(3,3.30]，13条落下侧、2条落上侧；按家族轮次选，每家族≤2且≤本家族Core，不按速度区间填数。

旧20条严格10/20保持原样；新规则数值10Core＋2Aux，实际仅复用10完整Core。旧2Aux没有补图。开发14Core仍独立；原400/275、原速参照均不进入新Core训练。

全部223入选轨迹逐条完整成功，实际运动/路径、State、三路RGB-D真实逐帧内容和触觉时序/有效位通过；post_success_actions=0，目标公式最大误差0。没有等待、绕路、复制帧、插值或改时间戳来凑倍率。Core-only冻结208条69992tick，train-only统计只用156条；家族split互斥、字段/时间/关节顺序/缓存一致性通过。Aux未进入本次训练、统计或缓存。

Core manifest SHA：`2c47eb38b27ad728fe7161a1f803407945a295b45ff62d4554174a10a9a28fbc`。完整组成manifest、统计和cache SHA见机器结果；CPU验收原scope文案沿用旧模板，实际输入明确为本批208条，模态与减速由独立逐条准入覆盖。

![组成、真实插入速度及Core家族覆盖](evidence/tight_composition_20261108.png)

## 不同阶段为何时长倍率不同

Core全程实际均速相对原速的中位数：Arm .506、Hand .548、末端 .515、物体 .543。Hand全程实际均速范围.567–.748rad/s，按每关节绝对运动平均；不是仅检查目标限幅。

|Core阶段|时长比中位数|Arm路径/均速比|Hand路径/均速比|末端路径/均速比|物体路径/均速比|
|---|---:|---:|---:|---:|---:|
|接近|2.136|1.046/.508|1.355/.636|1.181/.556|1.122/.565|
|抓取/抬升|2.356|1.174/.498|1.200/.504|1.084/.459|1.228/.532|
|插入|1.375|.716/.515|.840/.599|.720/.534|.821/.573|

插入速度下降而路径常缩短，因此阶段耗时不统一放大1.5–3倍。各项中位数并非同一轨迹，不能直接相乘除；逐条仍满足既定插入路径≤1.25及Hand/末端/物体均速≤.90等全部门槛。完整T=L/v分解在机器结果和CSV。

配对是同姿态家族、同原生配置，事件/关节状态抽样不同，不是完全相同隐藏状态/DR暴露的因果对照。当前证据支持筛选后生产；不支持单凭alpha/cap数值保证每次完整轨迹倍率，不开发新的路径进度控制器。

## 两项实现检查与训练

参考官方固定revision `5ba07ac6661db573af695b419a7947ecb704690f` 的[数据适配器](https://raw.githubusercontent.com/real-stanford/diffusion_policy/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/robomimic_replay_lowdim_dataset.py)与[低维策略](https://raw.githubusercontent.com/real-stanford/diffusion_policy/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_lowdim_policy.py)完成针对性检查：我们的标签是Arm7＋Hand22、rad单位的`target_after_rad`绝对目标，仅仿射归一化；没有套用Robomimic末端abs_action的旋转/坐标变换。已知29D目标通过实际数据—loss归一化—predict反归一化—实际目标缓冲往返测试。

实际模型保持完整episode持续GRU，**不是64tick截断历史**；短窗口为13个robot观察和3个geometry观察，96tick仅TBPTT梯度边界。obs[a]在a/60秒动作前，标签y[a…a+23]覆盖未来.4秒；执行前12tick/.2秒，a+12观察后从新plan索引0开始。过去动作严格y[a−1]；首尾padding/mask及episode reset、63/64/65与重规划边界人工tick测试通过。没有新动作/时间错位，也没有因此使旧checkpoint失效。

61项定向动作/时序/准入/复用回归＋1项闭环家族成员回归通过。168项/510tick GPU历史、microbatch/EMA/恢复/执行器等已接受证据按未变源码复用，未重开全仓审计。模型、损失/权重、H24/commit12、5/60/120Hz、DDIM和overlap0不变。

06:24:57全新初始化model/optimizer/EMA/scheduler/RNG/计数，12:46:40结束8500更新。实际442012个anchor采样＝100.2068等效epoch；4411/4411 anchors、620/620原始分段均访问，每anchor57–150次，每训练episode2833–2834次。这是anchor曝光，不是100次独立数据或100次完整轨迹梯度遍历。

预登记最低6787更新约80等效epoch，初始10180约120、上限16966约200。最后四次验证改善0.9386%＜1%，且验证/train8 MSE差距35.52倍，触发平台/过拟合停止；**有限充分性条件通过，但不声称全局收敛**，没有把时间/资源耗尽叫收敛。没有使用最终test选模。

validation选择8500 EMA：MSE .00166178，Arm RMSE .0065166rad，Hand .0572808rad；train8 MSE .00004678。验证训练目标loss在早期低点后上升，泛化差距明显。完整训练恢复passed，optimizer和EMA均8500，旧学生加载false。唯一闭环权重SHA：`6940b066cf8296d8e90203c0aa714f2ad780d5440c6128b89ddc5698bbbaa4da`。

![实际训练与验证曲线](evidence/tight_composition_training_20261108.png)

## 真实闭环与剩余问题

|队列|尝试 / 合法|完整成功|终止|既有诊断阶段分类|
|---|---:|---:|---|---|
|实际Core训练家族|10 / 10|0/10|7 timeout、3 fall|8抓取/抬升未完成、1抬升后对齐/丢失、1第二次插入|
|validation家族|7 / 7|0/7|7 timeout|5抓取/抬升未完成、2抬升后对齐/丢失|
|冻结后新初态|10 / 10|0/10|10 timeout|9抓取/抬升未完成、1抬升后对齐/丢失|

没有初态拒绝、软件故障或补跑。formal_017完成1个原生子目标后fall，不能说所有案例毫无任务进展；没有案例完成全部任务。所有27条实际执行器/提交目标误差和canonical actor归一化误差均0；学生未再次施加采集alpha/delay/EMA/cap。新初态剩余90个身份未启动，不能报告0/100。

基础数据与执行一致性已通过，当前可观察的剩余问题是**离线拟合未转化为自主抓取、抬升和连续插入的稳定闭环行为**，且存在训练/validation误差差距。闭环分布偏移、模型/监督方式的限制仍是解释假设，不能由这27例唯一归因，更不能外推为某个旧诊断入口造成旧模型失败。新数据/随机初始化与旧275不同，不是matched-data修复因果对照；Aux训练对照本轮未运行。

## 清理、恢复与资源实况

复用既有分类清理入口，全部owned进程退出后删除135份新未准入RGB-D载荷＋1份重复checkpoint，共136文件、38,859,898,880分配bytes＝**36.191101GiB**。135份numeric、原始SHA、模态验收、初态/seed/终止和全部433尝试分母保留；27份学生失败轨迹与结果也保留。全部223入选＋2未选完整Aux候选保留。旧7.510GiB与2.705GiB不重复计数。

best与last完整状态逐字段相同，保留best原字节，last用硬链接恢复别名，原last SHA和映射留账；唯一被评估权重无需额外best开发闭环副本。清理后211执行源码/计划SHA、6保护SHA、225合格载荷SHA、135拒绝numeric SHA、数据manifest/统计/cache指纹与3个CLI通过；last CPU读回8500计数、442012draws、完整状态路径有效。未上传数据/checkpoint，未改专家、资产、原400/275或共享Git历史；清理后可用约884.57GiB。

整卡均值28.389%，最高完整60秒59.632%，1507个采样窗口＞50%，遥测缺口0；最低空闲VRAM4154MiB低于6GiB储备，最低可用RAM16.85GiB。**本轮资源不是全程合规。** 共享卡读数不作单进程占用归因。运行时v1 SIGSTOP把暂停算入work并触发重复比例等待；10:13改为原gpu.lock协作等待，未重启trainer或改变梯度/优化器/RNG/预算，v1原日志保留。v2活体验证17445个计算段最大约1.009秒，等待记入lease wait；两版guard均正常退出。

## 保留的历史结果（不覆盖本轮合同）

|历史队列|真实结果与保留范围|
|---|---|
|旧alpha机制|33capture包含21慢速：旧15为6合格＋6成功未准入＋3失败，追加6全部成功但仅2合格；B00 Arm.50/Hand1为2/2、B01/B02合计0/4。来源/阶段/DR与真实分母见[原阶段CSV](evidence/tight_slow_phase_comparison_20261107.csv)，没有重跑。|
|Hand cap开发|pilot5/6任务成功、4/6严格准入；独立12/12成功、10/12准入；14合格/7开发家族保留，不混独立val/test。30新增capture三模态通过。旧阶段图与[原机器证据](evidence/tight_state_20261107.json)保留。|
|formal_05旧严格合同|55原速54成功＋1拒绝；20慢速20成功/10严格准入，当时按65%产出门槛停止。原10/20与[75次尝试表](evidence/tight_production_attempts_20261107.csv)、[旧结果](evidence/tight_production_results_20261107.json)不改；新版数值2Aux因旧RGB-D已删除不可复用。|
|旧275混速State|best4500曝光54.86等效epoch，last5000约60.95；同权重旧/canonical配对0/7→1/7，独立canonical训练家族1/4、新初态0/10合法＋2初态拒绝。旧完整状态、成功证据及分母保留；本轮没重复这7seed或重训275，不称新慢速matched-data结果。|


<!-- tight-failure-videos:start -->
## 2026-11-08 用户请求的三个失败案例视频

状态 **COMPLETE**，完成 2/2；Finite viewing queue finished; no additional starts。
观看索引：`/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/demos/tight_failures_20261108_01/remaining_01/README.md`；阶段/计数见同目录`status.json`。
同Core8500 EMA与原控制/物理，三个按历史结果挑选的独立观看复跑；不改原27例分母，不训练、不补seed。
首次队列因整卡60s均值51.29%停止，保留592帧已解码中断视频；本次仅接续未启动两例，原截止时间不变。
<!-- tight-failure-videos:end -->


<!-- tight-palm-evaluation:start -->
## 2026-11-09 有条件成功率测试与视频

状态 **COMPLETE**；成功0/合法10，拒绝0，中断0，未启动0。无合法完成时成功率未测。
Finite evaluation/video queue finished; no additional starts；索引：`/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/demos/tight_palm_eval_20261109_01/remaining_01/README.md`。
仅原200epoch/last4改善<1%平台达成后使用最终validation选定模型，最多10次新初态/1h、前3身份视频；不扩跑。
<!-- tight-palm-evaluation:end -->

<!-- tight-dagger-results:start -->
## 2026-11-11 DAgger：一轮前端与后端增量续训

本轮 owner 状态为 **COMPLETE**；产物核验完整。以下按已有凭据整理，缺失项不记为零成功。

|组|采集已启动 / 机会|接管成功 / 已启动|准入纠偏 episode / 有效专家 tick / DP anchor|新 DP 更新 / anchor 遍|新后端更新 / tick 遍|自主开发成功 / 有效 / 计划|
|---|---:|---:|---:|---:|---:|---:|
|G11|12 / 12|11 / 12|11 / 3265 / 266|1680 / 20.000|9043 / 15.000|0 / 10 / 10|
|QG01|12 / 12|10 / 12|10 / 2893 / 236|1680 / 20.008|11666 / 20.000|0 / 10 / 10|

采集成功包含专家接管，不等于学生自主成功；只接纳成功专家纠偏会产生选择偏差。学生前缀保留作真实历史，不监督；前端沿各组既有监督语义续训，后端监督仍为 `u_expert`。
自主列为预登记十次开发机会；非法、软件中断、模型身份未核验和未启动均单列保留，不替换样本，也不作为正式 test22 成功率。
整卡时间加权利用率：28.416%；资源样本 19843 条，>8s 内部缺口 0s。这不是每进程利用率；头尾未覆盖时间及原 monitor 汇总见机器结果。

[原始机器结果](evidence/tight_dagger_20261111_01/summary.json) · [采集逐次记录](evidence/tight_dagger_20261111_01/collection.csv) · [自主逐次记录](evidence/tight_dagger_20261111_01/closed_loop.csv) · [两阶段曝光](evidence/tight_dagger_20261111_01/training_stages.csv) · [权重/视频本地索引](evidence/tight_dagger_20261111_01/local_artifacts.json)。

![DAgger 训练与固定验证曲线](evidence/tight_dagger_20261111_01/training_curves.png)

仅 CPU 导出已有数值、日志与 SHA；未新增训练、推理、Kit、评估、清理或模型选择。初始化身份、失败原因、视频解码凭据和全部缺失项见公开结果。
<!-- tight-dagger-results:end -->
