# Expert / State input parity

本轮基于`ec84bd3`，仅新增State v2身份；旧State、ImageDP输入和权重不重解释。
实现：[state_history.py](../visuotactile/state_history.py)。源观测来自
`play/utils/obs_utils.py:build_observations`，实际任务还经过PreciseAssembly的goal构造。

## Actor完整字段与时间语义

以下index为送入Actor的141D数组，左闭右开。每个140D字段最终共同经过原生whole-observation queue、
clip±10、checkpoint running mean/variance（FP32运算，epsilon1e−5）、clip±5。
Student直接读取同一`teacher['policy']`，不另建随机观测生成器；训练读取原400真实保存的
`transition/expert_actor_input_diagnostic[t]`。当前命令`command[t]`只作为标签。

| RL字段 / index | 原始tensor、坐标与预处理 | 时间戳 | State对应 / 一致性 |
|---|---|---|---|
| joint_pos 0:29 | `robot.data.joint_pos[:, _perm_lab_to_canon]`，关节上下限映射[-1,1] | whole queue选中tick | Actor分支原值；机器人分支另保留当前rad |
| joint_vel 29:58 | 同canonical顺序，qd加σ=.1rad/s高斯噪声 | 同上 | 实际带噪声值完整保留；机器人分支另保留干净qd |
| prev_action_targets 58:87 | `_prev_targets[:,perm]`，rad，前一次原生处理完成的目标 | 同上 | 实际Actor值；robot58:87另保留当前target |
| palm_pos 87:90 | palm body位置+旋转后的局部(0,−.02,.16)m，再减env origin | 同上 | Actor局部世界轴位置完整保留 |
| palm_rot 90:94 | 当前palm wxyz转xyzw，不做State几何的半球规范化 | 同上 | 完整保留xyzw |
| object_rot 94:98 | object13D queue延迟0–9tick，随机轴/±5°噪声，转xyzw | object queue再whole queue | 完整保留实际采样结果 |
| fingertip_pos_rel_palm 98:113 | 5个tip body加旋转后的(.02,.002,0)m垫中心偏移，再减palm center | whole queue选中tick | 完整保留；不是85D几何中未经pad偏移的body位置 |
| keypoints_rel_palm 113:125 | 4角点，原offset×object_scale_multiplier；带延迟噪声物体姿态生成，减palm center | 混合object queue和该时刻palm，再whole queue | 完整保留 |
| keypoints_rel_goal 125:137 | 同noisy object角点减goal角点；包含任务goal观测扰动 | 同上 | 完整保留，不能用干净object−goal替代 |
| object_scales 137:140 | `_object_scale_per_env * _object_scale_multiplier` | whole queue选中tick | 完整保留；不等同URDF bbox |
| group 140:141 | rollout固定50，checkpoint六组映射50,40,…,0；不经RMS | 当前任务常量 | 保存原ID，使用冻结checkpoint的32D静态embedding |

Actor没有独立object position字段，位置通过keypoint相对关系体现。palm/object velocity、progress、
successes、reward等虽然在critic字典中，**不在Actor obs_list**；不把它们偷偷加入学生。
172D有效Actor输入=140D原标准化结果+32D静态group embedding。没有使用专家LSTM hidden/cell、
专家当前动作、奖励、未来目标切换事件或离线成功标签。

## 队列生命周期与额外特权分支

原生`_sample_delay`先在真实episode起点填满当前值，再roll1、写slot0、randint选择slot。
object queue长度10，whole Actor queue长度3；whole延迟0–2tick可能叠加object延迟。
`_episode_start=(episode_length_buf==0)&(_successes==0)`，普通子目标step归零不flush。
物体位置σ1cm；姿态轴先正态采样归一化，角度uniform±5°；qd噪声在whole queue之前加入。
新State不再合成噪声，不增加控制延迟。旧400未保存每次obs queue索引，所以不声称能确定每个
旧混合观测字段的唯一物理采样tick；其实际Actor输入逐步完整保存，读取时间就是obs[t]、执行u[t]之前。

178D机器人分支沿用：q29、qd29、target29、target−q29、严格t−1 issued29/delayed29、两种age、valid、
episode elapsed。target为当时真正控制目标；学生运行时使用实际执行反馈，非未来计划。
85D几何沿用原字段/坐标：base-frame五pose35、object/goal速度12、五指body位置15/四元数20、URDF bbox3。
四元数wxyz且scalar>=0；几何是当前干净特权状态，没有强制伪造Actor噪声。新GRU每tick读取85D，
保留旧3个规划anchor的详细几何窗口；原train320统计不改。增加采样频率与Actor分支均由新合同明确标记。

## 实测

[数值证据](evidence/state_history_20261102/input_history_parity.json)：400条81,894行Actor标准化max error **0**，
group embedding逐值完全相同。8条train/validation的真实controller输入，Actor/geometry误差0，
机器人q从归一化encoder逆变换带来约3.58e−7rad量级误差，逐项最大值见JSON；不称全分支bitwise。
8条共1,955tick：整段/逐tickcondition最大2.26498e−6，当前权重前缀分段最大1.66893e−6，
实际controller condition最大1.12057e−5（包含q逆变换），均在各自登记FP32容差内。
这些是输入和历史数值证据；在线配对、任务结果见[STATE_TRAINING_RESULTS](STATE_TRAINING_RESULTS.md)。
