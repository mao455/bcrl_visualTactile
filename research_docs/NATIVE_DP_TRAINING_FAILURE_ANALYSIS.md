# Native DP 失败分析：确认、假设与阻塞（2026-11-02）

## 新证据与纠正

旧三路正式闭环已全部完成，各0/20；旧V2共28条已完成，历史replay3/8、旧State train0/8、val0/8及第二seed0/4。
旧G0把四lane历史开环8/8作为所有训练的绝对条件不合理；新用户授权的G0-A/B/C修正见
[环境/学习归因报告](NATIVE_DP_ENVIRONMENT_LEARNING_REPAIR.md)，旧结果保持不变。

8条回放的命令/index/delayed/target均一致；q/qd实际首次非零误差在tick0，早于指端接触。
4条失败是源动作耗尽删失、1条fall。不要把它们写成5条timeout或数据无效。
冻结released expert实际含LSTM1024 hidden+cell，Actor140D经checkpoint running statistics，再将group50映射32D。
原Actor不直接读critic的stage/success/reward。8条历史Actor序列完整递归复算，arm RMSE约4.81e-7–6.86e-7；
只保留最近13tick输入的相同权重，anchor arm RMSE为.281–.619，最近25tick为.073–.315。
这是记忆依赖的具体干预证据；仍不能证明其他无长记忆的最优策略不存在，更不能单独解释全部闭环失败。
360条train/validation共6286anchors，实际StateController特征对缓存最大归一化差5.20e-5，来自proprio限位逆变换精度；
此前只比较encode_segment不足以覆盖该边界。未发现时序错位，原始量与模型影响继续单列，不能先称根因已修复。

本轮single fresh expert6/8、four8/8、fresh记录→新Kit2/2且15字段逐字节一致；历史原点处理能影响结果，
但不是tick1 object z差的充分解释。旧State fresh0/8；tick0相同Actor观测上已与原expert动作明显不同。
原生reset本身有盒初始重叠（single6/8、four3/8），合法起点总体可靠性仍需独立验收；
无该重叠的两个fresh初态expert2/2而旧State/两BC均0/2，只作小样本描述。
BC1000 train拟合不充分，预算对照至4000使State train arm/hand降至.02365/.03965，
val反为.98176/.71937；不把小数据过拟合解释为成功或扩散模型必然失败。

以下是旧实验启动时的历史假设与限制；新在线对照真实结果优先。

目前不能宣布 State 的19000次更新无效。旧run仍在完成原定评估；新run先审计与冻结诊断，
不抢先把fit权重的0/2当正式权重表现。参考旧快照 `NATIVE_DP_CURRENT_RESULTS_20261102.md`。

## 已确认事实

1. **旧Gate缺乏闭环验收。** `native_evaluation.autonomous`的汇总 `passed=True` 不依赖success，
   `native_pipeline.Pipeline.stage/run`据该字段继续所有三组formal。三组fit诊断0/6仍全部进入训练。
   这是实验调度错误，不能据此推导网络损坏。V2 Gate将软件完成和原生完整成功分开，G4在State证据不足时关闭。
2. **Oracle覆盖不足。** 旧base/instrumented均使用同一seed17新220步单lane样本，独立轨迹数为1。
   历史四lane样本的物理复现此前只有结果级1/4或2/4，未通过轨迹一致性。
   新任务固定8条不同历史示范，明确记录每条原生结果和控制/物理差；不以两种模式重复计数。
3. **数据没有发现已知接口错误。** 本轮400原文件完整SHA重读，81,894数值步；400条末尾原生success、retract。
   robot/geometry/过去issued-delayed/命令/target-before全部复算误差0，train-only统计误差0。
   初始可观测特征与整条command内容没有精确重复；这不证明同一并行采集run中隐藏状态统计独立。
4. **CPU加载重复和GPU锁范围过宽。** `native_training.train`为有效target计数再创建Episode，随后每段重新创建；
   Official每个frame重新打开HDF5；History每次载完整patch cache并从episode0无梯度预热。
   `long_resources.Lease.block`在等待和pacing时持有锁。V2独立State缓存和新lease修此开销，不改原运行。
5. **旧checkpoint选择不稳定性的证据尚不足。** 旧validation固定seed7091，只有一个样本路径。
   V2先用7091/17091/27091报告均值与离散程度；固定train8/val8再对各首两初态重复seed17091。
   相同EMA的best/last只检查身份后复用结果，避免把相同模型重复执行当新证据。
6. **旧中间权重缺失。** 旧formal仅last/best，没有1000、3000等完整里程碑；不能凭曲线伪造恢复。
   V2保留milestone、last、best_validation_offline及指向不可变checkpoint的best_validation_rollout索引。
7. **不能安全启用4env评估。** `NativeDiagnosticEnv._sample_kinematics`接触矩阵明确要求(1,1,1,3)，
   `restore_observable_state`为single lane且缺其他lane、solver/contact、sensor与共享RNG状态。
   本轮保持单env持久实例；不通过扩大env维度掩盖这些假设。

## 尚待实验区分

| 假设 | 现有证据 | 可区分它的实验/新记录 | 当前能否下结论 |
|---|---|---|---|
| 欠拟合 | 正式State验证arm/hand约.6695/.4612；fit1更小 | 正式best/last闭环；新采样设置里程碑；train/val多seed误差 | 不能仅由RMSE判定全部训练无效 |
| 随机采样/多模态 | DP随机初始噪声，单seed选模 | 固定条件3seed离线；同初态第二seed自主；饱和/方差 | 未测前不声称主要根因 |
| teacher-forcing分布偏移 | 输入含过去专家命令，部署含实际学生执行 | 同tick、同DDIM噪声的expert-history shadow与student-history plan | shadow不控制机器人，不称它自主成功；长于专家记录处明确缺失 |
| 控制时序错误 | 现有label/target/history复算为0 | 原native函数逐400轨迹delay/target回归；真实trace首处偏离 | 已测数字链支持语义正确，不代表完整物理复现 |
| State信息不足 |85D几何并不包含全部expert actor/simulator隐藏状态 | StateBC相同2601D因果窗→64×29；fit与自主分开 | 不因名为State-based就断言信息必然充分 |
| 12步chunk后期太长 |5Hz重规划，原生60Hz执行，物理120Hz | 规划边界RMSE、first divergence、alignment阶段曲线 | 首轮保持12，不同时改频率/学习率/损失 |
| 历史物理恢复不足 | 旧接触前已有object z末位差，接触后放大 | 本轮8历史replay逐条command/target/q/object误差 | 保存可见状态相同≠精确模拟状态；replay失败不作废原成功400 |

## 失败时的明确处理

G0若不满足8/8独立专家完整成功与记录控制一致性，**所有新optimizer更新停止**。
冻结旧State仍完成已分配诊断，用来区分已有策略行为与物理恢复问题；这些只能称“保存可观测初态诊断”，
不能称严格训练世界复现，也不能作为已解决公共基础设施的证明。

G0通过后才允许最多1000步StateBC诊断；若BC仍无训练初态完整成功，不自动启动19000步新DP。
若BC能成功而旧DP不能，再以相同模型/optimizer/objective、仅改变State batch与缓存的独立新训练作可归因对照。
三个里程碑仍无完整成功且阶段指标不再改善时停止；早期一个零成功不直接判死。

未通过G4不启动视觉大预算、触觉残差或RL。测试集的旧使用登记保留，V2调试不调用test。
失败的replay不通过强制每步状态、原生target bypass、改变wrench/timeout/任务判据来“修复”。
历史缺失内部状态可能无法从现存数据重建；若本轮仍阻塞，将交付首次分歧字段/时间与恢复字段缺口，
而不是静默降低门槛或新采集替换400。

## 证据入口

- `evidence/native_dp_v2/data_audit_summary.json`、`control_audit_summary.json`、`reference_sources.json`。
- 本机完整逐轨迹审计 `outputs/visuotactile/native_dp_v2/audit_20261102/`。
- 新诊断 `outputs/visuotactile/native_dp_v2/state_first_20261102_01/diagnostics/`，仅实际生成后可作为结论。
- 每trial的 `restoration.json`、`trace.npz`、`expert_comparison.npz`、`plans.npz`、`result.json`。
  源命令差异阈值RMSE .1、joint/target .01rad、物体1mm用于定位；不是新的成功标准。
