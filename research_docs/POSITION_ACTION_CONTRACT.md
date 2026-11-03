# 29D绝对关节位置动作合同

2026-11-03追加只读审计：两个合法接口明确分列，不在线改动当前冻结训练。

|合同|标签与历史|当前tick执行链|
|---|---|---|
|A Post-delay PositionDP|target_after_rad；严格过去请求/应用target|观测t→预测处理后target[t]→限位→原执行器，不再次delay或EMA|
|B Pre-delay Native-command DP|command_clipped_pre_delay；严格过去issued/delayed command|观测t→command[t]→原delay queue→原Arm增量/Hand绝对映射→EMA/限位→原执行器|

obs.time_s=time_before_s；obs actuator target=target_before[t]=target_after[t−1]。
当前target_after[t]在[t,t+1]两个物理子步施加，label第0项就是观察t之后即将发出的目标；
延迟选择的command来自max(0,t−index)，其有效age在启动填队列时可小于index。
原player已clip网络mu，raw_action是submitted别名，旧executed_action也是pre-delay别名；
只有control_trace.delayed_command明确表示真实队列输出。不要把raw误写为未裁剪logit。

400/81894步逐值重建2,374,926个最终目标全部一致，index/effective lag/timestamps验收通过。
lag0/1/2计数27324/27558/27012，均值16.603ms、最长33.333ms。
两个合同都有现存fresh专家注入2/2成功、168/153ticks精确物理字段证据；详见
[只读FK/控制研究结果](STATE_TRAINING_RESULTS.md)和旧foundation_review。
这不保证学习后的目标分布相同：原生Arm目标步幅最大.002500057rad，
C0有6662/6663步至少一关节超过.002501rad，最大.204428rad。
不得直接给A再套原延迟；新的限速/平滑也会改变合同，必须另立对照。
Native1000和Position19000训练预算不同，不能作为动作表示优劣结论。

本轮在现有V2实现，不新增一套训练框架。默认H32（未来0.5333秒），可选H24，
每12个60Hz控制步重新规划（5Hz），物理120Hz。预测长度、执行长度与观察历史分开定义。
State保留Actor141、本体178、几何85、自己的持续GRU256、U-Net和DDIM16；未再训练BC。

400条原HDF5的81,894步均通过直接标签审计，证据为本地
`outputs/visuotactile/native_dp_v2/position_20261103_01/position_audit/result.json`。
标签是`transition/target_after_rad`：研究wrapper在原生action pipeline完成后直接复制
`_cur_targets`，原PlayEnv在两个物理子步将其提交给`set_joint_position_target`。
canonical顺序29维，单位弧度，已经包含原专家动作的延迟、平滑和限位结果。
`obs/actuator_target_rad`等于`target_before_rad`，不是当前标签。
逐轨迹核验before/obs相等、after[t]=before[t+1]、时钟/索引、真实限位和三相机帧对齐。
所有400条保留，无反算标签，无重采。train320-only action统计SHA：
`d402a9f9b412c0928eab6b67e55e30e0372376945a1b4cdcf2ffeeaa54366d9d`。

`position_control.apply_position_targets`将预测弧度按真实限位裁剪，canonical→Lab，
写入现有目标缓冲，仍由原`_apply_action`提交给原执行器。没有速度反算、额外动作延迟、
移动平均、速率限制或实际q/物体轨迹写入。每tick保留一个**不参与控制**的原delay随机抽样，
以保持其后观测噪声和外力随机流的调用对齐，单独记录`unused_delay_rng_draw`。
新模式逐120Hz子步读取真正的`robot.data.joint_pos_target`，验证实际执行器目标。
原RL和旧NativeDP保留原动作路径、任务、物理参数和成功判据。

robot178中的116:145为上一tick请求位置，145:174为上一tick实际应用位置，174:176为age，
176为valid；初始全部零。训练历史来自严格过去的专家target_after，部署来自自己的执行反馈。
current/future目标只作监督。目标反归一化在采样后完成，随后只做同一真实限位裁剪。
checkpoint明确`action_contract=position_rad_v1`、H32及位置历史；旧H64权重/EMA/优化器不兼容。
合同version3是已有checkpoint模式的版本号，不是新工程V3。

CPU测试通过不等于接口Gate通过。必须先完成新鲜专家在线成功，以及相同初态/外生随机流下
position replay完整成功、逐tick目标/执行器零误差和物理字段对照。当前实测见
[KIT_RELIABILITY](KIT_RELIABILITY.md)和[State结果](STATE_TRAINING_RESULTS.md)。

## 实际位置回放验收

fresh_982501/982502在线专家均成功，position replay均成功（168/153步）。
actor、target、实际关节位置/速度、两个物理子步执行器目标、物体位姿、接触、过滤物体接触、
外力/力矩、lift、subgoal、retract、alignment共14字段逐字节一致。提交目标和执行器目标最大误差均0。
新接口没有第二次delay或平滑；实际跟踪RMSE为0.085416/0.106316rad，和原专家一致。
证据为[foundation_review.json](evidence/position_20261103/foundation_review.json)。
这是控制接口验收，不是新StateDP自主成功。

## 2026-11-03 可选推理消融

`position_rad_v1`、监督字段、原执行器与120/60Hz均不变。新实验job单独记录DDIM步数、
归一化x0裁剪、策略seed和`overlap_steps`；旧checkpoint不修改、默认16/false/K0逐数值回归。
K4在绝对位置计划上把上一条尚未执行的`[12:16]`与当前`[0:4]`按新计划权重
1/4、1/2、3/4、1混合；cursor仍从0执行，没有额外等待、时间平移或重复执行旧命令。
本轮是可关闭的过渡消融，不称为原版DP标准要求。raw和执行plan分别保存，实际请求/应用位置
仍反馈进严格过去的历史；新future计划不作为已执行历史。数值测试不等于闭环收益。
