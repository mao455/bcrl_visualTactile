# Tight-Insertion 当前结果

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
