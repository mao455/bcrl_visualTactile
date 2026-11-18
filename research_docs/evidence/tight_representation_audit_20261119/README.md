# 曝光与表征审计证据

集中解释：[主报告](../../TIGHT_REPRESENTATION_AUDIT_20261119.md)。源基线fe1dd74；只有CPU诊断，无新策略训练/仿真。

| 文件 | 内容 |
|---|---|
| exposure.json / stage_summary.csv | 30个选中祖先节点、父路径/SHA、实际/选中更新、epoch访问、分阶段权重 |
| episode_exposure.csv / tick_exposure.csv.gz | 130每轨与27508每合法tick的8模型/组件计数、未达50的明确ID和tick；slot与不同访问分开 |
| source_receipts.json / input_contracts.json | 配置、optimizer真实LR/step、EMA、来源日志hash、当前源码差异、真实张量合同 |
| training_curves.csv.gz / candidate_training_curves.png | 全祖先实际update日志，当前130训练曲线和选点 |
| validation_curves.csv / validation_curves.png | 历史train2/validation6曲线，非全训练池统计 |
| offline_errors.csv / offline_summary.csv | 全合法记录、阶段、Arm/Hand、DP自身目标/专家命令与最终命令的误差；rad和可达区间尺度 |
| offline_common42_corrections.csv | 相同42条旧纠偏上的72/130比较，避免人口变化混淆 |
| probe_metrics.csv / probe_folds.json / probe_scope.json | 9种表示、216family隔离拟合回执、位置/姿态/运动/接触/未来无接触代理及范围 |
| feature_extraction.json | 实际缓存绑定、模型SHA、全历史递推与采样帧数 |
| nearest_observation.json / privileged_readouts.json | 近邻歧义检查及有限精确状态诊断，均不声称不可辨识证明/闭环上界 |
| history_shift.csv / history_shift_scope.json | 真学生前缀与专家历史，同阶段、跨family距离 |
| interventions.csv / intervention_summary.csv / dp_interventions.csv | 固定真实历史、相同DP计划/随机噪声下数值依赖，不是闭环因果收益 |
| current_phase_gradients.csv / current_gradient_scope.json | 76当前参数片段的固定128 FY目标梯度，不能还原历史梯度 |
| normalization.csv / verification.json | 输入尺度、保护hash、测试及数据路径/分母检查 |
| exposure_cdf.png / phase_underexposure.png / representation_readouts.png | 曝光与表征比较图 |

大特征、原始HDF5、checkpoint和先期代码修正回执保留本机outputs/visuotactile/run/tight_representation_audit_20261119_01，不推送。复现命令和所有限制见主报告。
