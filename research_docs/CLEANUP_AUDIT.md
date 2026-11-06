# 清理与回归验收

## 2026-11-06 最终收尾

`narrow_repair_20261106_01`于15:19:38结束后，CPU读回正式best4500/last5000、验证部署hash及原400+200保护清单。新增75条成功、1条真实失败、全部正式模型/缓存/视频均保留；6条最终三视角视频有全帧解码记录，当前文件SHA已记录。无新增CUDA初始化或GPU任务。

仅删除本轮`startup_gpu/last.pt`与`disposable_gpu/last.pt`：均为1步一次性smoke、完整恢复验收通过、正式更新0，已确认正式模型不依赖它们且无打开句柄。实际释放2,306,195,456分配bytes（约2.148GiB），与下方此前1.074GiB分开记录。每个文件的inode/大小/SHA、删除前审查和完成状态保留在现有cleanup_ledger；未知或失败证据不删除。

## 2026-11-06 本轮修复清理

核查引用、原run完成状态、一次性GPU恢复验收receipt、checkpoint更新数及同UID打开文件后，仅删除原narrow_state_20261105_02/disposable_gpu/last.pt（1次可丢弃更新，正式更新0）。原best/last、全部400+200、有效缓存、专家/资产、初态和失败/partial证据保留。本次释放1,153,085,440bytes（约1.074GiB）。精确字节与SHA见新run cleanup.json/cleanup_ledger.json；不按日期/通配符删除目录，不将以前清理重复计数。后续正式模型和新数据仍保护。


## 2026-11-06 后续准备

新代码仅在隔离worktree修改，原训练/采集源码、200条和原400数据、best/last与原失败证据只读。新扩充manifest引用原200路径，不复制18GB数据；需要合并训练统计的numeric cache属于有效依赖。新正式checkpoint与可能续训的完整状态均保留，尚无可确认废弃的新模型可删，本次释放0bytes。原启动阶段2.148GiB及更早126.673GiB不重复计入。

## 2026-11-05当前清理

已删除旧窄域一次性CPU smoke及本轮已验收GPU smoke checkpoint共2份，实际释放2306101248分配字节（2.148GiB）。旧ledger保存路径、inode、大小、SHA和完成的smoke引用。重新核实同UID句柄，当前不可读者均为已识别的systemd/sd-pam/ssh-agent session daemon；没有跳过全部清理，也没有强删未知占用文件。原400、专家、隔离轨迹、仍引用的checkpoint/cache及未知文件保留。

当前46项相关CPU回归通过。新物理/训练验收结果另见SLOW_EXPERT_RESULTS，不能由CPU测试推出。

## 此前清理（不重复计入本轮）


实际释放 **126.673 GiB**；删除20812文件、143模型checkpoint路径（含别名）、1922其他Torch缓存/初始状态文件。完整移除42个已淘汰实验/数据根目录，累计移除5886个空目录。硬链接只在最后一条链接移除时计算物理空间，不用逻辑大小冒充释放量。

当前分支删除160个研究源码文件、26个旧方案专属测试文件、48份重复/过时Markdown。Git历史保留旧代码；不是抹去失败结论。

公共checkpoint/EMA/AdamW/RNG逻辑集中至`visuotactile.checkpointing`；进程与durable JSON至`tools.runtime`；视频至`visuotactile.video`；FK/轨迹统计至`tools.trajectory_metrics`。训练由anchor_continue按job配置，独立Kit worker统一为anchor_continue_worker，报告由anchor_interim_analysis维护。

实际验证：78项保留测试全部通过；真实State16584/Image10000 CPU checkpoint、EMA/AdamW/采样状态一致，20个真实条件/预测/图像张量字段逐值一致；整理后当前State和Image的实际CUDA恢复与历史合同检查通过（0优化更新）；原控制/环境源码保护，视频真实编码/完整解码，Markdown本地链接检查通过。
整理后额外闭环验收：尚未通过。额外2次启动均未取得有效episode：首次为整理引入的ElementTree漏导入（首次模型动作前失败，已修复并通过真实资产初始化测试）；修复后的唯一补充启动在场景创建前FileThread SIGABRT（exit−6）。全部进程/GPU资源已回收，停止自动重试。两次原始日志与状态保留。 这些是代码整理验收，单独记录，不加入150 epoch的train4/validation4分母；原8次均有效、软件缺失0。

保留400成功数据、当前320/40/40索引、现用numeric/DINO缓存及train统计；当前/最佳/150里程碑完整恢复状态、Image10000与本轮三阶段比较的精简数据。旧目录中的仍用缓存不能整目录清理。来源不明的data/checkpoints、小型upstream refs、未知符号链接和代表性崩溃证据保留。
大视频、HDF5和权重留本地，不上传。必要视频与实测结果见[统一结果](STATE_DP_RESULTS.md)。原Kit关闭异常、Image live strict pairing未解决和历史资源违规均保持真实记录。

源码审查commit `0af9272fd444e30a1f7149bf3a5ae7b7c8e4c089`；切入commit `d5ea4f40c5efba354c6c6065aaa0c0641289b9c9`。完整路径/大小/引用/删除理由、逐文件日志、before/after冻结SHA、CPU/CUDA结果在 `/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01/maintenance`。

[精简机器摘要](evidence/cleanup_summary.json) · [当前入口](README.md) · [历史结论](RESEARCH_HISTORY.md)

[实际回归与两次验收失败证据](evidence/cleanup_regression.json)。State10000/16584仍作为本轮同口径对照的可复核权重保留；State150 latest/best/里程碑为同一完整状态的硬链接，未重复占用模型空间。当前Image10000是未续训的唯一有效Image基线。临时源码审查worktree已移除。
