from pathlib import Path
import json
P=Path.cwd();D=P/'outputs/diagnostics/tight_full_curve_20261127_01';B=P/'research_docs/evidence/tight_full_curve_20261127_01'
assert (D/'audits_complete.json').exists();B.mkdir(parents=True,exist_ok=True)
(B/'README.md').write_text('''# 正确布局全量训练曲线：证据索引

[短报告](../../TIGHT_FULL_CURVE_20261127.md)给出实际自主结果与限制；[完整方法](METHODS.md)保留阶段分子、训练与恢复细节。原机运行根为
`outputs/visuotactile/run/tight_full_curve_20261127_01`，诊断根同名。不是新运行授权。

| 内容 | 文件 |
| --- | --- |
| 自主成功率0/30/60曲线、阶段全体分母 | [learning_curve.csv](audit/learning_curve.csv)、[图](audit/autonomous_learning_curve.png) |
| 每初态完整结果、模型/条件/时间来源 | [per_initial_state.csv](audit/per_initial_state.csv) |
| 首次技术失败、重试与未派发 | [all_attempts.csv](audit/all_attempts.csv)、runs内owner回执 |
| 绝对成功区间、A−B、新数据−等更新原池、保留/新增/丢失 | [primary_summary.json](audit/primary_summary.json) |
| 正确24完整专家实测布局与入池 | [data_contract_audit.json](audit/data_contract_audit.json)、[逐条验收](audit/corrected_full_layout_admissions.csv) |
| 原始/旧纠偏/旧未位移24/新正确24/本组后缀采样 | [pool_sampling.csv](audit/pool_sampling.csv)、runs内data/deck |
| 所选模型实际年龄、DP/后端绑定和SHA | [model_identities.csv](audit/model_identities.csv)、[selected_training_audit.json](audit/selected_training_audit.json) |
| 每合法专家tick实际成功步曝光 | [actual_selected_tick_exposure.csv.gz](audit/actual_selected_tick_exposure.csv.gz) |
| 阶段/家族训练与验证误差 | [curves](audit/fitting_phase_family_curves.csv.gz)、[trends](audit/fitting_phase_family_trends.csv)、[口径](audit/fitting_audit.json) |
| 每次reset的可观测配对与实际CUDA采样器消费 | [evaluation_contract_audit.json](audit/evaluation_contract_audit.json) |
| 原池B33轮恢复的模型、EMA、Adam及所有RNG | [恢复校验](audit/frontend_B_restored_state_verified.json)、[弃用尾部](audit/runtime_recoveries.json) |
| 评估容量中断、未完成重试与启动看门狗修正 | [23已完成保留/9重试](audit/evaluation_retry12.json)、[资源等待](audit/evaluation_recovery13.json)、[启动修正](audit/watchdog_fix14.json) |
| 后端并行资源绑定中断及实际状态恢复 | [中断](audit/interruption09.json)、[恢复核验](audit/backend_restores09.json) |
| 原cell弃用/临时更新/packed调度精度与短计时 | [recurrent_actual_probe.json](audit/recurrent_actual_probe.json)、[recurrent_full_probe.json](audit/recurrent_full_probe.json) |
| 无收益预取pilot、聚焦检查 | [prefetch_pilot_summary.json](audit/prefetch_pilot_summary.json)、[targeted_checks.json](audit/targeted_checks.json) |
| 实际时间、进程退出、资源采样与唯一inode占用 | [runtime_accounting.json](audit/runtime_accounting.json) |
| 全部逐tick核心数值的字段与原始SHA | [core_numeric_index.json](audit/core_numeric_index.json) |
| 发布训练日志简化口径及本机详细日志SHA | [local_detailed_training_logs.json](audit/local_detailed_training_logs.json) |

`config/`为实际运行配置，`source/diagnostic_scripts/`为本轮包装/统计/核验脚本；生产消费者位于仓库canonical路径，前端292文件冻结身份见audit/execution_amendment_sources.json；后端准入修正见audit/backend_source_freeze07.json与audit/normalization_gate_fix07.json。
`runs/`含真实专家成功/碰撞拒绝、自主成功/失败、模型配套、固定末轮选择和退出回执。
每条numeric导出完整核心数值为npz，排除纯自主expert_shadow占位和RNG字节数组，后者有逐例单独回执。
train_evaluation仅为训练入口固定的前2条轨迹，不代表全池；train_updates来自实际优化前预测，DP扩散训练误差与DDIM验证误差口径不同。各阶段/family样本数随表公开，不按A/B不同nominal目标空间排名。
详细训练日志去除重复逐接触几何数组；原日志SHA和本机路径完整保留，不把压缩摘要冒充逐接触原载荷。
所有原始RGB-D、特征、模型/优化器载荷与完整详细日志留本机，原400、当前合法回放池、历史模型及负结果均保留。
未读原64终测，不把专家恢复算作自主成功；独立16在所有末段模型冻结后打开，不据结果续训或选模。
80项末轮相关聚焦检查、此前50项原检查、42项后端准入校验检查与8套实际池/轮次预检通过，temporary actual-data探针8步不计正式训练年龄；实际B恢复尾156步另存不计谱系。

`MANIFEST.json`记录本包每个文件大小、SHA及来源。离线仅需Python标准库：

```bash
python source/diagnostic_scripts/verify_public_bundle.py
```

在证据包根目录执行。脚本逐项核验SHA、224个唯一机会、整批阶段分子、Wilson区间与配对差算术；不替代留在原机的RGB-D、权重和CUDA噪声审计。报告链接只引用真实已有文件。
发布只写mao455/bcrl_visualTactile既有feature/tight-state-foundation；第二个后来新建仓库持续冻结，本轮文件/分支/标签/PR/设置零写入。
本轮没有新增退役路径，历史f069a576的30个删除不重复计；没有重复计上轮磁盘释放。
''')
print('Evidence index prepared from closed round')
