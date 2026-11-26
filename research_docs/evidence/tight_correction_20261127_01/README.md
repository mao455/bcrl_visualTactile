# 正确布局修正轮：证据索引

[短报告](../../TIGHT_CORRECTION_20261127.md)给出实际闭环结论。
运行根为`outputs/visuotactile/run/tight_correction_20261127_01`，
审计根为`outputs/diagnostics/tight_correction_20261127_01`。
原终测64家族没有用于本轮训练、选点或选模，不重启历史session。

| 内容 | 文件 |
| --- | --- |
| 实际分子/分母、区间和保留/新增/丢失 | [primary_results.json](audit/primary_results.json) |
| 同family修前/修后/原数据对照 | [per_family_before_after.csv](audit/per_family_before_after.csv) |
| 64次已完成自主开发逐例 | [all_completed_development_cases.csv](audit/all_completed_development_cases.csv) |
| 首次技术中断与被调用worker的派发情况 | [all_attempts.csv](audit/all_attempts.csv) |
| 跨阶段实际可观测初态差异 | [cross_phase_initial_observables.json](audit/cross_phase_initial_observables.json) |
| 首次输出/执行/物体分叉 | [first_divergence.csv](audit/first_divergence.csv)、[阈值](audit/first_divergence.json) |
| 64例实际噪声消费 | [sampler_consumption_audit.json](audit/sampler_consumption_audit.json) |
| 抓持链和专家正控 | [grasp_chain_summary.json](audit/grasp_chain_summary.json)、[逐tick](audit/grasp_chain_ticks.csv.gz)、[图](audit/grasp_comparison.png) |
| 规则可能误拦成功轨迹 | [contact_screen_audit.json](audit/contact_screen_audit.json) |
| 当前学生真正交权/恢复/合法标签 | [current_capture_cases.csv](audit/current_capture_cases.csv)、[summary](audit/current_capture_summary.json) |
| 四套实际更新与逐tick/入口曝光 | [short_optimization_audit.json](audit/short_optimization_audit.json)、[逐轨迹](audit/short_exposure_by_episode.csv) |
| 两个既有B72样本的真实训练/曝光 | [b72_optimization_audit.json](audit/b72_optimization_audit.json) |
| 既有B72拟合前后，208合法tick | [同历史结果](runs/b72_fit_history/inference/summary.json) |
| 本组当前后缀，前/纠偏后/旧数据等更新 | [A](runs/own_history_A/inference/summary.json)、[B](runs/own_history_B/inference/summary.json) |
| 最终模型身份与原始四组保持 | [final_model_identities.json](audit/final_model_identities.json) |
| 原始数值路径、SHA与公开字段 | [core_numeric_index.json](audit/core_numeric_index.json) |
| 98项相关检查 | [targeted_checks.json](audit/targeted_checks.json) |
| 实际运行时间和owner退出 | [runtime_accounting.json](audit/runtime_accounting.json) |
| 大仓真正删除的30路径 | [retired_paths.csv](audit/retired_paths.csv) |

`runs/`保留真实reset布局读回回执、配置、结果、专家标签审计、开始/退出、模型/源码身份、采样器收据及训练日志。
每个`core_numeric.npz`保留同名原numeric.h5中的核心数值字段及原键名，排除RNG字节大阵列和纯自主占位expert_shadow；
真实专家命令和合法标签字段保留，RNG起止身份另存。`trace.npz`/`plans.npz`保留实际命令链。
RGB-D原始载荷、冻结特征、原HDF5、Adam/EMA模型大载荷仍在注册的本机路径，不上传。
原400、既有回放池、成功与失败、历史正式基线全部保留。

`MANIFEST.json`逐文件记录SHA、字节数与原机相对路径；公开包不是独立训练集或新盲测。
实验包装源码放在`source/diagnostic_scripts/`，保留当时相对路径/模块约定；复核原运行时可按其名称恢复到审计根，
它们调用现有训练/推理入口，没有另造模型实现。正式主代码只保留原统一入口和可配置选项。

离线检查（不启动Kit、不读取私有权重）：

```bash
python scripts/tight_correction_20261127/verify_published.py
```

25份重复历史源码快照已退役，科学数值与原SHA保留，两个历史验证器通过显式Gitblob回退复核；
5个一次性工具也已退役，精确身份见删除清单。其他历史证据不因文件名含failed而删除。
只发布大仓`mao455/bcrl_visualTactile`既有`feature/tight-state-foundation`；第二仓持续冻结、本轮零写入。
