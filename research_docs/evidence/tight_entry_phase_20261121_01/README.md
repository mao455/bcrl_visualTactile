# 入口相位修复实验：公开数据与复核

本目录公开2026-11-21至22日的源码身份、流程、训练曝光及96次完整自主开发结果；[阶段报告](../../TIGHT_ENTRY_PHASE_20261122.md) 区分已完成与尚缺的验收。实际执行源码为 `90b0a7ccebd350de06d2f7c9c83bf2a652c80f21`，此次发布提交仅整理后续证据和此前授权清理，不改变已执行模型。

| 路径 / 格式 | 保存的内容 |
| --- | --- |
| `audit/*.json`、`*.csv` | 50条原始纠偏的mask/相位核验、LR、视觉与原R2复用审计、两组334入口槽曝光、96逐例与配对汇总 |
| `audit/*_planned_sampling.jsonl.gz` | 开跑前实际60个deck的旧采样与入口采样计划；这是计划，不算训练曝光 |
| `run/{A,B}/training/learning.receipts.jsonl.gz` | 实际7200成功更新的anchor采样、入口与绝对tick贡献、loss及梯度范数；从原日志提取字段，未把弃用更新并入 |
| `run/{A,B}/backend/{A,B}/training/loss.receipts.jsonl.gz` | 配套后端各22763次实际更新的loss和有效tick贡献 |
| `A_selected_checkpoint_receipt.json`、`B_selected_checkpoint_receipt.json` | 从选中best checkpoint实际提取的epoch/update、sampler seed和完整入口ledger；不含张量权重 |
| `run/**/*.json`、`*.jsonl`、`*.csv`、`*.png` | 数据train180/validation6、父权重/模型/缓存绑定、运行合同、逐epoch验证、学习率、选模与通用离线曲线 |
| `development/wave00..15/` | 六套完整策略的实际plan、模型SHA、初态配对检查、96条逐例结果、退出与阶段计时 |
| `recovery/`、`run/A/training/learning.abandoned.*.jsonl` | 原资源中断、独立恢复owner、9条弃用尾更新；它们不计有效曝光 |
| `audit/*.py` | 原独立核验和流程登记源码副本；保留原研究机路径，执行前须登记新的输出身份 |
| `cleanup/` | 232个退休脚本/配置的依赖审计、37段已删仓库视频及全部269个Git删除路径/原blob身份 |
| `export_manifest.json` | 每份导出与原始文件的路径、大小、SHA256；字段提取的日志明确标识变换 |
| `local_artifacts_manifest.json` | 本轮873个本地大文件的路径、大小和SHA256；这些载荷未放入Git |

公开副本中的原始JSON和日志保留实际绝对路径，表示历史来源；路径不是下载链接。原始HDF5、逐帧RGB-D、视觉/nominal缓存、完整checkpoint、NPZ和录像按项目约定保留研究机。仅克隆仓库可以复核公开计数、模型身份及结论；重新训练或执行模型需要这些载荷、Isaac环境和新的有限session。没有用哈希清单冒充已公开原始载荷。

在仓库根目录执行以下命令，仅需Python标准库，不需要GPU或本机数据：

```bash
python research_docs/evidence/tight_entry_phase_20261121_01/verify_published.py
```

复核器逐文件检查导出SHA；从实际成功更新回执重算选中update6000时的全150条/956槽入口计数和180轨/35863绝对tick覆盖，再核对新增50条/334槽CSV与选中checkpoint ledger；复算switch108、first_legal110、slot2 mask；从16波结果重新核对96条完整策略身份、成功seed和保留/丢失/新增分母。它不重跑物理、不验证未公开张量的内容，也不证明误差下降或因果恢复。

生产路径回归使用安装了项目依赖的Python：

```bash
python -m pytest -q tests/test_native_entry_sampling.py tests/test_visual_arithmetic_contract.py tests/test_run_facts.py tests/test_effective_exposure.py
```

本次发布前该命令实际12项通过，原35项生产验收见 `audit/focused_tests_retry01.log`。旧R2原采样器对照在等价审计后复用，视觉只做672帧重编码抽检；anchor数量变化会改变训练noise消费，限制见主报告。旧独立审计位于 [相邻目录](../tight_adaptive_audit_20261121_01/REPORT.md)。

数据与实验状态截至2026-11-22：入口确实进入选中模型训练；开发成功增加但旧成功保持不足；入口同历史分层误差、新短诊断自重放和冻结后终测仍未完成。预留28851000..007终测尚未打开，不能将本目录当作最终终测报告。

要重新从研究机导出，可以执行本目录的 `export_evidence.py --include-local-artifact-hashes`；它只读现有实验并写公开证据，不启动训练或仿真。该操作须在审阅新证据、另行保存已有发布版本后进行，不能把新一轮文件静默覆盖为原身份。完整实际作业参数保存在 `run/project_job_started.json`、`recovery/project_job_started.json` 和 `development/project_job_started.json`；已完成的原root禁止重启。
