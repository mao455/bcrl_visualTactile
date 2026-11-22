# 冻结入口修复终测与同历史诊断

主报告为 [`../../TIGHT_ENTRY_FINAL24_20261122.md`](../../TIGHT_ENTRY_FINAL24_20261122.md)，本目录的 `REPORT.md` 保存收尾时的相同副本。

- `registration.json`、`job.json`、`session.json`：一次冻结的两批初态、四套完整模型 SHA、评测协议及有限 session。新批初态未按结果筛选。
- `per_initial_state.csv`：48 行 A/B 逐初态配对；`rollouts.csv`：96 次自主执行。`paired_counts.csv`、`subgoal_funnels.csv`、`primary_summary.json` 为汇总。
- `wave00/` 至 `wave23/` 及初态、协议、RNG 审计：轻量原始回执与检查，各模型逐例记录位于波次子目录。原始 HDF5、视觉、权重、视频保留在本机输出目录。
- `entry_history/`：原 R2 与入口修复对新增 50 条纠偏的同历史、同噪声诊断。50 个首合法动作和 284 个其余已提交合法动作互不重叠；按手臂 7 维、手指 22 维分别计算每 slot RMSE（rad）。该诊断针对已用于训练的纠偏，不能当作独立泛化或自主救回证据。
- `development_cases/`：仅从历史开发记录选取的正反案例、完整轨迹 CSV、阈值定义、图及可用视频关键帧；没有使用终测轨迹来选择诊断案例或调整策略。
- `source/`：本轮新增入口和脚本快照；历史基点及冻结生产源码哈希另见登记。`closure.json` 保存结束时间、零更新、资源采样与进程收尾回执。

只用 Python 标准库即可离线复核轻量交付物：

```bash
python research_docs/evidence/tight_entry_final24_20261122_01/verify_published.py
```

此命令校验 `published_sha256.json` 中全部文件哈希、96 次结果与配对计数、50 条历史／334 个合法槽及零训练／DAgger。不会运行策略或仿真。实际执行命令和环境保存在 job、owner 日志与源码快照中；已完成的主评测无需重启。

本轮固定推理噪声 seed7091，起止生成器状态核验完整；可观测初态配对不等于隐藏物理状态或未来环境随机流相同。资源信息为采样证据。所有主评测均软件完成，原生超时和任务失败保留为失败。
