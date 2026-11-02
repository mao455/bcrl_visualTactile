# Environment reproducibility contract — State core v1

2026-11-02 correction: `overlap_shape` evaluates USD geometry at its USD transform,
which remained at spawn while PhysX/Fabric moved the actual body. The benchmark now
verifies original USD/URDF box dimensions and offsets and uses explicit overlap_box
at the live PhysX body pose. All external hits still reject; each box must self-hit.
Post-rollout persistent reset query state was inconsistent with independently observed
poses; every new controlled comparison runs in fresh Kit. No physics/reset change.
The latest shared-card resource contract replaces the historical50% rules below;
6GiB freeVRAM/12GiB availableRAM plus recent1/5min pacing. See STATE_TRAINING_RESULTS.


本合同用于新State诊断，历史400及旧run保持原身份。CPU PhysX单lane、spacing8m、scene seed20261031；
原4lane采集origin±4m、共享随机流及不可导出内部状态不被假设为single等价。
修正后的core v1启用`sim.enable_scene_query_support=True`，只是查询支持，不改积分参数。
每次以包围真实object的box query作阳性对照，查不到object即失败；root01关闭query的错误证据不计入新基准。
版本化实现为`state_benchmark.py`和`environment_rng.py`，原生任务/控制器/研究物理wrapper不改。
详细冻结配置、runtime/asset hash随实验导出；配置hash漂移即拒绝后续阶段。

## 固定内容

| 项目 | 生效约束 |
|---|---|
| runtime | 当前Isaac Sim5.1 / 安装Lab与PhysX扩展；实际包版本与二进制hash见交付manifest，不升级共享环境 |
| 时间 | YAML physics_dt=.00833333333s、decimation2；控制约1/60s，DP每12step=5Hz |
| 原生命令 | canonical29；clip±1；canonical→Lab；3槽随机延迟一次；arm1.5×dt累积，hand绝对限位映射；arm/hand moving-average .1；限位；两物理substep |
| 成功 | `PreciseAssemblyEnv._get_dones`；preInsertAndFinal两目标，insertion tolerance .01×keypoint_scale1.5m；原success_steps10；retract object误差≤.005×1.5m且mean fingertip距离>.1m |
| 失败 | 原fall object local z<.1m、hand_far条件、native timeout及任务额外终止；不以lift、接触或RMSE代成功 |
| 求解器 | YAML TGS solver1；position iteration bound1..192、velocity0..1；bounce .02、friction offset .01、correlation .0005；资产actuator/solver具体覆盖以实际cfg/property snapshot为准 |
| 重力/摩擦 | (0,0,−9.81)；robot/object/table .5、tip1.5；不改碰撞mesh、质量、COM、限位、gains |
| obs DR | object queue10、xyzσ.01m、旋转±5°；qdσ.1；whole queue3；clip±10；goal噪声按原任务cfg |
| action DR | 原3槽选择index0..2，不再额外插入任何延迟 |
| force DR | 原force20/torque2，prob log-uniform[.001,.1]，decay0，仅lifted时、retract禁用；保留world_hold已验证兼容转换 |
| 渲染 | 原三RGB-D、448×252、quality/AA Off、wrist_bracket_close；State不读取图像，视频保持观看证据 |

## 随机流

`IsolatedEnvironment`在每次reset/step进入环境拥有的Python/NumPy/Torch CPU状态，退出后恢复策略侧状态。
所有原生random调用次序和分布保留；只隔离其状态归属。RL确定性输出仍会采样distribution消耗CPU RNG，
现不再改变下一步环境noise/delay/wrench。DP初始噪声使用独立CUDA Generator7091。
本合同仅允许CPU physics；不声称CUDA-physics或每lane独立流已验收。

fresh reset的环境seed按登记表；保存初态诊断先原生reset，再一次恢复observable状态、刷新观察，
从独立`seed+100000`环境流开始控制。旧restore函数写入历史RNG的行为被明确替换，历史推理随机消耗不再隐含控制新环境。
这不是历史PhysX内部状态完全恢复：solver/contact/sensor内部状态缺口仍保留。
策略不同会引起不同状态条件分支（如lift/retract），因此同seed不保证跨不同轨迹的每类后续事件逐tick相同。
用**相同预录命令+故意不同策略随机消耗**的两runner实测控制/物理/Actor字段，才能验收隔离效果。

## 合法初态与冻结顺序

预先登记fresh candidates982501..982532，依顺序选择前4个通过者；另登记原split的train4/validation4。
全部screen在任何新expert/student结果出现前完成。源定义见[registration.json](evidence/state_history_20261102/registration.json)。
每个初态同时做原object/table/hole OBB检查和实际cooked PhysX object-shape overlap查询，记录shape、hit rigid body。
保守地拒绝所有外部shape overlap（可能包括恰好接触）；不称signed penetration depth测量。
所有被拒候选保留`initial_collision.json`，不进入控制、不按成功补seed；登记saved初态不合法则阶段停止。
每条运行前重复检查同一条件；后续若不合法仍保留为协议失败，不事后从表中删掉。

冻结core12条专家需≥10完整成功且saved8需≥6，才支持这一小cohort的基础设施门槛；不声称总体成功概率已可靠估计。
同命令replay2必须实测字段逐字节一致；历史4lane开环3/8不作为本Gate否决项。
若当前专家不过关，先定位环境初态/状态恢复问题，不能降低成功定义放行训练。
后续随机鲁棒性需要另行seed/版本/分母；不和core能力混合。

资源：整卡time-weighted与完整60s原则≤50%，准入45%、pacing目标40%，准入6GiB/14GiB，硬保留4GiB/10GiB、disk12GB。
记录全部超限与缺测；等待只改变墙钟，不改变sim.dt或command执行次数。持续monitor到所有worker退出。
