# 提前推进影响面与隔离接管小试：证据索引

主报告为 [TIGHT_PREMATURE_SCOPE_20261122.md](../../TIGHT_PREMATURE_SCOPE_20261122.md)；本目录的 `REPORT.md` 是最终同内容副本。完整终测、R2 同历史误差诊断与原开发案例仍见 [上一轮证据](../tight_entry_final24_20261122_01/README.md)。本轮不重跑原主评测，不重训。

- `transitions_by_seed.csv`、`transition_counts.csv`、`summary.json`：A/B 各 24 个 seed 的 3×3 原生阶段转移；另有 A 原 5 个成功的去向和 entry 另 5 次后段失败的来源。行列均按同一 seed 配对。
- `rules_registration.json`、`rules_addendum.json`：在终测轨迹分类前冻结的规则与原生张手命令缺失的明确限制。所有阈值、累计确认口径和 unknown 处理均保留。
- `train_dev/`：A/B 来源去重的 180 条训练原轨与 96 次既有自主开发；逐轨 JSON/CSV、111468 行逐 tick 压缩 CSV、源 SHA、完整性核验及全部 83 条确认前离门轨迹的事件顺序。
- `train_dev/supplemental_command_withdrawal/`：仅训练/开发的手指命令 FK 几何代理，独立补充登记；它不是原生开指标签，未应用到终测。
- `train_dev/never_gate_stratification/`：仅训练/开发的稳定持物、近目标、既有对准停滞屏幕回放分层，不按结果调整屏幕或小试初态。
- `terminal/`：96 次已完成终测、54751 个 tick 的冻结规则回顾；不用于调阈值、选择接管规则或小试初态，不改写原生结局。
- `audit_inputs/`：372 条训练/开发/终测轨迹中，冻结分析实际读取的原始数值数组及配置、原始 dtype/shape/字节 SHA。按来源分片压缩，提供标准库解码/核验，便于离线重建数值输入；不含视觉载荷。
- `pilot/`：独立 job/session 的 4 个原训练 family × A/B × 自主/接管臂，共 16 个完整机会。包括全部结果、触发状态、逐 tick 数值、初态配对、命令链/专家标签来源审计、资源与进程回执、所有日志和技术失败记录。专家救回与学生自主成功分列；所有机会均禁止入训练池。
- `prior_execution_logs/`：前轮终测和同历史诊断的全部 `.log`/`.jsonl`，用可逆 gzip 保持原字节；未修改前轮冻结交付物。
- `source/`、`checks.json`、`closure.json`、`published_sha256.json`：本轮源码快照、相关检查、闭合回执和所有公开文件的 SHA。

按用户确认沿用项目发布规则：上传全部可复核轻量证据，包括完整逐 tick 数值、CSV/JSON、源码、日志和哈希。原始 HDF5、RGB/深度、权重、特征缓存与视频留在登记的本机路径；本目录没有大权重。上一轮 `.pt` 文件只是很小的推理生成器状态。

仅 Python 标准库即可离线核验哈希、转移矩阵、统计分母、166219 行训练/开发/终测逐 tick 记录及隔离小试的完整性：

```bash
python scripts/tight_premature_scope_20261122/verify_scope.py
python scripts/tight_entry_final24_20261122/verify_published.py
```

这些命令不运行推理、仿真或训练。重新计算 URDF/FK 和补充屏幕需要项目环境及本机原始数据；源码与源身份全部保存，不应重启已完成的实验。

已确认重复的源码快照已退役；原文件身份、唯一当前路径与历史 Git blob 见 [退役源码映射](retired_source_paths.json)。旧科学数据及原 SHA 清单保持，核验入口按映射读取原 Git blob。
