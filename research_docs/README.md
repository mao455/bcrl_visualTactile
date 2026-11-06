# Tight-Insertion StateDP

当前任务：command-based StateDP基础链路整改、真实慢速数据准入、随机初始化训练、闭环评估与分类清理。主文档只有[当前合同](CONTRACTS.md)和[当前结果](STATE_DP_RESULTS.md)；历史事实见[精简历史表](RESEARCH_HISTORY.md)，历史待办不是当前授权。

本轮审查基点为feature/narrow-state-data的642f5a2。IDE原目录HEAD为3e11d62且有既存文档改动；实际修改在`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`，基于干净642f5a2新建本地分支`feature/tight-state-foundation`，未重写共享分支。

唯一配置：[configs/tight_state.json](../configs/tight_state.json)。当前独立root：`outputs/visuotactile/run/tight_state_20261107_01`。只读状态：

```bash
bash visuotactile/run_cpu.sh -m tools.tight_state status --root outputs/visuotactile/run/tight_state_20261107_01
```

入口拒绝自动覆盖已有物理attempt；`prepare`只允许新root，不能用换root重置已经登记的失败/预算。开发、正式数据、训练与学生成功是四个不同结果，均以本root对应实测文件为准。旧narrow_collection/narrow_repair/旧session不是当前正式生产入口。

当前配置不依赖旧run/session/calibration或学生权重；原400/275路径仅用于保护、明确标记的独立审计fixture和历史索引。旧数据审计通过不等于被准入本轮慢速多模态集。原生工程入口、专家、assets与共享SDK保持。
