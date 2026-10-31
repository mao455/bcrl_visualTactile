# T0重放与DP控制验收证据

完整解释与复现命令：[主报告](../../T0_REPLAY_AND_DP_CONTROL_FIX.md)。

- [results.json](results.json)：12条历史T0、4条源world诊断、新expert及两版replay、16条DP、首差、版本来源、输入合同和最终资源。
- [verification.json](verification.json)：31项回归、AST、9,616个native输入/816个H64计划、request/body核验和退出状态。
- [protected_artifacts.json](protected_artifacts.json)：原manifest/旧权重/正式报告及本轮冻结权重的重新hash。
- [历史误差图](historical_replay_errors.png)：四条源轨迹的关节与物体位置误差，黑虚线为预登记容差。

`fresh_precision`与`control_bytewise_equal`使用shape、dtype和原始bytes比较；运行时旧摘要的array_equal标志不替代此复核。
历史fixed控制链一致不等于物理轨迹一致；新单lane220步精确观测重现不证明历史四lane全部恢复。
DP每模式每模型独立0/4，均无lift/子目标；approach是几何proxy，grasp/align不补造标签。
发布物仅小统计和图，源HDF5/NPZ、输入、日志和checkpoint保留本机ignored outputs。
