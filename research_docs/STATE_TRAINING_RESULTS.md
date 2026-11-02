# State observation/history repair — actual results

## 2026-11-03：H32位置训练和最终评估结束，尚无完整自主成功

`position_20261103_01`于 **04:45:10（UTC+8）**结束，服务inactive/dead，owner、worker和monitor均已退出。
机器状态为`position_study_finished`、`research_success=false`。位置StateDP完成19000 updates，
最终选定模型完整评估 **0/12：train 0/4、validation 0/8**，没有缺失的最终结果。
Image两组均未启动：预注册8条validation中的完整成功为0，未达到至少2个不同起点成功的门槛。
本轮不再自动增加更新、重试、H24或新轨迹；没有新增StateBC训练。

### 模型、训练与闭环

保留Actor141+robot178+geometry85+独立持续GRU256、U-Net、sample/Loss B、DDIM16；
72,047,005个参数，H32/执行12，规划5Hz、控制60Hz、物理120Hz。位置动作是直接记录的
29D `target_after_rad`，部署按原执行器限位提交绝对弧度，合同`position_rad_v1`。
全部train320参与训练，validation40选模，冻结后两模型各做一次offline test40。
训练集曲线仅固定前16条的指标，不冒充全train320指标。H24未执行。

|模型|实际updates|选定模型train/validation完整成功|原生终止|局部阶段|
|---|---:|---|---|---|
|H32 Native-command短对照|1000|0/4、0/8|11 timeout、1 fall|无lift/首子目标|
|H32 Position StateDP|19000|0/4、0/8|10 timeout、2 fall|2条lift、1条首子目标及alignment|

位置模型的训练初态`r931005_1793209376_e02_p000059`在第115 tick达到首子目标，
最终717 tick timeout；没有完整成功，该条自身native lift标志为false。两条lift均最终fall。
上述局部进展不能称为验证集成功。里程碑和最终模型重复起点的结果分列，不能合并为独立样本。
1000/3000/6000/10000/19000位置validation分别0/8、0/8、0/7完成+1启动缺失、0/8、0/8。

位置validation前12步arm/hand RMSE从1000时0.044552/0.176333降至19000时
**0.020641/0.110750 rad**；冻结后的test40为 **0.022406/0.112744 rad**。
但hold-current-target基线的arm误差更低（validation 0.015538、test 0.015746 rad），
hand基线较差（0.196482/0.197362 rad）。低离线误差仍未转化为完整任务成功。
19000由validation first12 balanced MSE选择；diffusion loss在10000时0.008629，
19000反升至0.012547，不能声称所有验证指标持续改善。
原生命令对照validation arm/hand为0.767768/0.582914、test为0.747553/0.564498，
单位与位置弧度不同，不能直接比较数值倍数。1000与19000预算不同，也不能把差异唯一归因于动作语义。

最佳位置checkpoint：
`outputs/visuotactile/native_dp_v2/position_20261103_01/position_h32_train/best.pt`，
SHA256 `5ecc930ffd52af166e00904b0515159193d8527d9f5726908bf57967c445a416`。
原生命令best SHA256 `efb9a3f4b560f4024f1382a3ef9531827ba5d4a76fadb23a861625e8596c5ddf`。
[实际模型配置](evidence/position_20261103/model_configuration.json)、
[完整训练指标](evidence/position_20261103/results/position_h32_train.json)、
[最终逐条结果](evidence/position_20261103/results/position_h32_selected.json)、
[离线test40](evidence/position_20261103/results/position_h32_offline_test.json)。

### 已有故障归因证据

原400/81,894直接位置标签核验通过；新鲜专家2/2及位置重放2/2完整成功（168/153 tick），
14项控制/物理字段逐字节一致，提交和真实执行器target误差均0。
学习cohort在线专家 **12/12完整成功**。真实train/val各一条共310 tick通过在线输入路径，
robot178最大差3.58e-7，Actor/geometry/target差0。这些证明所测条件下接口有效，不证明学生能力。

选定位置学生12条的reset Actor均与对应在线专家完全相同；预测目标从第0步开始不同，
第0个动作后的q/target与专家差异均已超过1mrad。12条的hand target RMSE均在对应专家
首次物体接触前超过0.1rad（逐条tick见下方summary）。首步arm/hand RMSE跨轨迹均值
0.017588/0.051204 rad；重规划边界跳变量RMS均值0.070083 rad，chunk内部0.023435 rad。
完整学生trace的提交目标、记录目标和两个物理子步执行器目标逐字节一致；
实际q跟踪target的arm/hand RMSE跨轨迹均值0.013351/0.058156 rad。
因此已完成的失败不能归因于目标没有送达执行器；错误决策在接触前已经出现。
训练的教师强制过去目标与部署的学生历史，以及重规划连续性，仍是待干预验证的学习问题，
现有证据不能将任何一个因素确定为唯一根因，也不支持废弃原400条成功数据。

Kit关闭故障经显式独立worker退出得到缓解，4/4隔离probe通过父进程PID/GPU释放核验。
启动仍有故障：正式学生阶段两次有界恢复成功；6000 validation第3条在场景建立前SIGABRT，
未进入预期异常分类，保留为缺失并继续后续独立评价。该故障不影响最终选定模型12条全部完成。
详见[KIT_RELIABILITY](KIT_RELIABILITY.md)，不能声称SDK根因已经修复。

### 核验、曲线与观看证据

本次只读收尾核对170项运行源码hash、119项stage结果hash、两份best权重hash、
全部12条有限值trace及15条已完整解码视频的当前hash。之前89项相关回归测试通过；
本次未改源码，未重复训练、仿真或测试。
整卡采样时间加权均值37.8656%、最大完整60秒均值57.2718%；missing_samples为0，
但记录有1376.63秒监测间隔、最大间隔911.35秒（包含早期停机修订），不声称全时段覆盖。
当前资源协议没有固定利用率上限；没有其他项目吞吐指标，不声称零干扰。

[收尾核验](evidence/position_20261103/completed_review.json) /
[逐轨迹首次分歧与分阶段误差](evidence/position_20261103/results/summary.json) /
[学习、跟踪与阶段曲线](evidence/position_20261103/results/position_curves.png) /
[资源曲线](evidence/position_20261103/results/resource_curves.png) /
[15条视频解码清单](evidence/position_20261103/results/video_decode.json)。
曲线中不同profile的action RMSE单位分别为原生命令、rad；各cohort不合并，缺失不计任务失败。

本地代表视频（不上传）：
[对应专家成功145帧](../outputs/visuotactile/native_dp_v2/position_20261103_01/expert_learning_single_00/r931005_1793209376_e00_p000000/rollout.mp4) /
[位置学生lift后fall339帧](../outputs/visuotactile/native_dp_v2/position_20261103_01/position_h32_selected_single_00/r931005_1793209376_e00_p000000/rollout.mp4)。
达到首子目标的另一条只有完整numeric trace，没有录像，不能用这条fall视频冒充。

## 2026-11-03 00:19 UTC+8：训练完成，评估不完整，尚无学生完整成功

旧state_full服务inactive，owner/全部自有worker/monitor于11月2日23:21:53退出。
机器状态为`training_complete_evaluation_incomplete`，不是整个实验complete。
StateBC完成10000updates，StateDP完成19000updates；两者均train320/validation40，
覆盖全部320训练身份，按验证集选择10000/19000的EMA，未使用test选择。

|模型/评估|完整成功/已完成|其他实测结果|缺失项|
|---|---:|---|---|
|BC所选10000|0/12|fresh0/4、train0/4、val0/4；全601timeout，无lift/子目标|无|
|DP19000里程碑|0/4|train1条lift后254步fall；另3条601timeout；无子目标|无|
|DP所选19000|0/5|fresh4条601timeout；train1条lift后273步fall；无子目标|第6条启动失败，后6条未启动|

DP里程碑与所选模型诊断分开计数，不合并为9个独立起点。
第6条`dp_single_05`在建立场景前发生FileThread/std::system_error SIGABRT(-6)，
只有runtime_controls.json，无动作或轨迹；该条未知任务结果，不能记为任务失败或成功。
另有DP10000里程碑第4条启动SIGSEGV缺失。三次自动Kit恢复已消耗，未追加重试。
因为两项deferred未解决，旧State12条对照、新test40离线和各组正式20条均未开始。
因此当前不能给出完整正式成功率或新旧State在同一新cohort上的最终改善幅度。

最终validation前12步arm/hand RMSE：BC0.617104/0.491214，DP0.715060/0.500374。
DP从10000步0.798099/0.559021继续下降，仍未形成完整成功。
训练指标仅固定前16条：BC0.540785/0.468841，DP0.612155/0.436233；
优化器实际使用全部train320，不能将16条指标当成全部训练分布指标。

所选DP已完成5条的reset Actor与对应专家均逐字节相同，第0步arm/hand命令已偏离；
reset动作RMSE跨轨迹平均0.619776/0.275060。chunk边界/内部跳变量RMS平均
0.455739/0.188447，提示重规划不连续；尚无因果干预证明其为唯一原因。
在线专家12/12成功与fresh replay2/2 exact16仍支持可运行场景下的控制基础设施。
软件启动不稳定解释缺失评估，不能解释已经实际完成的学生闭环失败。

只读收尾核验169源码哈希、88已提交stage结果、两份best权重SHA；
复用14条视频完整解码结果并重新核对文件哈希，全部通过，未新增训练或仿真。
资源报告含13279.81秒监测间隔（包括停机），整卡采样均值36.599%、最大完整60秒69.208%；
当前协议无固定利用率上限，且无另一项目吞吐指标，不能据此声称零干扰或全时段覆盖。
证据：[训练完成核验](evidence/state_shared_20261102/training_complete_review.json)、
[学习曲线](evidence/state_shared_20261102/results/learning_curves.png)、
[逐轨迹诊断](evidence/state_shared_20261102/results/summary.json)。
旧运行的逐次恢复快照和自动表已从当前报告去重；完整历史仍可由基准提交f1953c7中的本文件及上述小证据恢复。


## 本轮固定预算与保留边界（服务已结束）

同一root的deadline不重置：Gate为11月3日03:55:10，训练为21:55:10，绝对终点11月4日00:55:10（UTC+8）。
Stage前后核验源码/数据哈希，位置normalizer只用train320，第一阶段1000后的arm/hand误差阈值均0.25rad。
学习初态专家控制要求完整12条合法结果、至少10条成功且validation至少6/8；原生成功定义不变。
State较长训练在验证连续两次无1%改善时停止，低误差不能替代完整任务成功。
两组Image原计划各按300/1000/3000有界预算顺序执行；本轮State成功Gate未通过，均未开始。
Image只读三路RGB-D、机器人本体和严格过去位置历史，不读Actor/特权几何。

原400、旧best及所有milestone保留。已删除4份与best内部逐成员SHA一致的last副本及旧可重建numeric cache，
释放2,512,265,377字节；清单见[evidence/position_20261103/cleanup.json](evidence/position_20261103/cleanup.json)。
本报告删去重复的运行中快照，不删历史Git或重要旧失败结果。普通owner已完成自动整理；
本次人工复核后仅提交核心报告、去重小JSON和曲线，不上传权重、原数据、MP4或NPZ。



## 收尾实测表（已与原始结果复核）

此表只读取普通owner结束后的产物，不新增训练或rollout。旧/前置失效样本不混入新分母。
“已完成”仅统计有原生结果的轨迹；6000 validation另有1次启动失败，已尝试8、完成7，结果未知。

|组|cohort|已完成|合法|完整成功|native lift|至少1子目标|
|---|---|---:|---:|---:|---:|---:|
|expert|fresh|2|2|2|0|2|
|expert_learning|train|4|4|4|2|4|
|expert_learning|validation|8|8|8|4|8|
|native_h32_loop_1000|train|2|2|0|0|0|
|native_h32_loop_1000|validation|2|2|0|0|0|
|native_h32_selected|train|4|4|0|0|0|
|native_h32_selected|validation|8|8|0|0|0|
|position_h32_loop_1000|train|2|2|0|1|0|
|position_h32_loop_1000|validation|2|2|0|0|0|
|position_h32_loop_10000|train|2|2|0|0|0|
|position_h32_loop_10000|validation|2|2|0|0|0|
|position_h32_loop_19000|train|2|2|0|2|0|
|position_h32_loop_19000|validation|2|2|0|0|0|
|position_h32_loop_3000|train|2|2|0|2|0|
|position_h32_loop_3000|validation|2|2|0|1|0|
|position_h32_loop_6000|train|2|2|0|1|0|
|position_h32_loop_6000|validation|2|2|0|0|0|
|position_h32_selected|train|4|4|0|2|1|
|position_h32_selected|validation|8|8|0|0|0|
|position_h32_validation_1000|validation|8|8|0|1|0|
|position_h32_validation_10000|validation|8|8|0|0|0|
|position_h32_validation_19000|validation|8|8|0|0|0|
|position_h32_validation_3000|validation|8|8|0|1|0|
|position_h32_validation_6000|validation|7|7|0|0|0|
|position_replay|fresh|2|2|2|0|2|

|组|updates|固定train16 arm/hand RMSE|validation40 arm/hand RMSE|
|---|---:|---|---|
|native_h32|1000|0.735955/0.560575|0.767768/0.582914|
|position_h32|1000|0.039490/0.166288|0.044552/0.176333|
|position_h32|3000|0.024006/0.126084|0.030114/0.146135|
|position_h32|6000|0.018490/0.105303|0.024476/0.130733|
|position_h32|10000|0.014195/0.077889|0.021932/0.119499|
|position_h32|19000|0.009947/0.041417|0.020641/0.110750|

最终owner状态：`{"state": "position_study_finished", "utc": 1793652309.8322816, "research_success": false, "image": {"state": "not_started", "reason": "Fewer than two complete native successes on a complete preregistered eight-ID validation cohort"}}`。

逐条reset误差、接触前后首差及阶段误差见公开`summary.json/trace_diagnostics`。
这些观测不唯一证明网络失败机制；同输入/同命令验收与完整成功分开报告。
