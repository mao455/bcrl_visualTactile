# 实际网络核验与近似训练初态诊断证据

[中文核心结论和完整报告](../../NETWORK_ARCHITECTURE_EXACT.md)；
[已有训练曲线直接阅读](../dp_loss_trt_20261031/README.md)。

| 文件 | 来源、范围与限制 |
|---|---|
| `../network_layers.csv` | 6个真实checkpoint实例的594行leaf/MHA hook；未调用的函数式访问明确标注 |
| `network_parameters.csv` | 743个named parameter，含冻结/训练、optimizer membership/step、固定权重探针梯度与重建初始参数差 |
| `network_audit.json` | 参数总数、训练覆盖、权重SHA、force同权重干预，分train/validation和当前query；没有optimizer更新 |
| `architecture_*.png/.svg` | DP、Transformer、历史GRU、训练和部署五图；From列补齐跨节点关系 |
| `force_interventions.png` | `network_audit.json`固定权重干预；4条train+4条val，连续力敏感性不等于收益 |
| `force_fixed_trajectory.csv/.png`、`masked_fixed_trajectory.csv/.png` | 真实train示范7个代表关节命令与gate；epoch5/3730更新CPU重算，非自主轨迹 |
| `dataset_distribution.json`、`dataset_episodes.csv/.png` | 原400条逐条元数据/状态/命令统计；固定320/40/40，配置只保留公开研究字段 |
| `native_audit.json`、`native_outcomes.csv` | 12条预登记T0/T1结果、首偏离、旧T2引用、保护源码SHA和资源；T0未通过精确重放 |
| `native_timeline.csv`、`training_initial_diagnostics.png` | 图为本地完整trace的标量比较，CSV每6tick及首6tick/末tick抽取；不是全量原始轨迹 |
| `training_initial_frames.png` | 用户允许的4条训练示范首帧三视角缩略拼图；非原始图像序列 |
| `verification.json` | 13项定向测试、32张PNG解码、5张SVG解析、表格总数、嵌图/链接和公开路径检查 |

本地源：`outputs/visuotactile/run/dp_loss_trt_20261031_01`、`residual_20261030_01`和
`network_audit_20261031_01`。原始HDF5/NPZ、checkpoint、视频、完整日志与私有配置不在公开提交内。
新脚本没有导入第三方源码副本；原MIT及vendored DP许可/固定pin不变。

T0原记录命令准确提交，但环境内部状态未完全恢复，4条均命令用尽而未成功；T1两组各0/4原生timeout。
所有自有GPU进程退出，原400/历史结果保持，不自动重新启动任何阶段。
