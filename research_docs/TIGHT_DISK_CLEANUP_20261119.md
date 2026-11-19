# Tight-Insertion 磁盘调查与安全清理 — 2026-11-19

**已实际删除201份可重建、非当前使用的冻结视觉特征归档，以及1份本次验证产生的重复缓存。归因释放85.988GiB，其中旧归档85.705GiB；未删除任何checkpoint、原始HDF5、当前视觉缓存或其他项目文件。**

逐文件清单：[cleanup_manifest.json](evidence/tight_disk_cleanup_20261119/cleanup_manifest.json)，实际结果：[cleanup_records.csv](evidence/tight_disk_cleanup_20261119/cleanup_records.csv)，带时间的先写意图/后写结果：[cleanup_operations.jsonl](evidence/tight_disk_cleanup_20261119/cleanup_operations.jsonl)。本报告不代表四组件训练完成。

## 1. 全盘与主要目录

根盘为`/dev/nvme0n1p2`（ext4），根文件系统容量1875.676GiB；项目、其他docs项目和`/tmp`均在同一设备，移动到它们不能释放空间。EFI为独立小分区；`/dev/shm`、`/run`是RAM支持的tmpfs，不是可持久存放训练资产的额外磁盘。未发现另一块可用数据盘。

inode使用约4%，不是inode耗尽。ext4另有95.349GiB空闲块对普通用户不可用（`f_bfree-f_bavail`）；这是文件系统保留空间，不是项目文件，也未修改其设置。根目录`du -x`可访问内容约1751.767GiB；权限受限目录只能报告下界。`/proc`可访问范围内已删但仍打开的根盘文件共32个，仅0.027GiB；memfd和共享内存未算作磁盘，不通过干预其他进程回收它们。

| 目录 | 所属 | 清理前占用GiB | 处理 |
|---|---|---|---|
| ~/docs/bc+rl_RSS | current_project | 869.103 | audited_cleanup |
| ~/docs/grasp_small_obj | other_project_or_unknown | 453.928 | read_only |
| ~/docs/IsaacLab_DexAssemble-sharpa | other_project_or_unknown | 136.586 | read_only |
| ~/docs/IsaacLab_DexAssemble | other_project_or_unknown | 12.117 | read_only |
| ~/docs/IsaacLab_furniture0 | other_project_or_unknown | 3.212 | read_only |
| ~/docs/lamp-formal-delivery-20261025 | other_project_or_unknown | 1.689 | read_only |
| ~/docs/IsaacLab_Dex | other_project_or_unknown | 1.253 | read_only |
| ~/docs/storage_audit_20261107 | other_project_or_unknown | 0.653 | read_only |

另行只读调查：

| 目录 | 占用GiB | 说明 |
|---|---|---|
| ~/miniconda3 | 58.499 | 共享Python/仿真软件环境，未操作 |
| ~/.local/share/Trash | 47.897 | 回收站；来源记录包括其他grasp项目，未操作 |
| /tmp | 46.083 | 临时文件及目录，归属未全部确认，未操作 |
| /var | 10.052 | 系统日志和软件，未操作 |
| ~/.codex | 9.116 | 工具会话与插件，未操作 |
| /usr | 7.307 | 系统软件，未操作 |
| ~/.cache | 2.202 | 包含受保护DINO权重和其他应用缓存，未操作 |

分级证据：[docs_du.tsv](evidence/tight_disk_cleanup_20261119/docs_du.tsv)、[home_du.tsv](evidence/tight_disk_cleanup_20261119/home_du.tsv)、[other_projects_du.tsv](evidence/tight_disk_cleanup_20261119/other_projects_du.tsv)、[df_before.txt](evidence/tight_disk_cleanup_20261119/df_before.txt)、[inodes_before.txt](evidence/tight_disk_cleanup_20261119/inodes_before.txt)。这些采样并非全盘原子快照，其他任务仍在写入。

## 2. 本项目内部详细占用

`du`清理前项目约869.103GiB，其中outputs约866.93GiB。下面按唯一inode累计普通文件的实际分配块，目录项另占少量空间；软链接不跟随，硬链接不重复计费，因此与分别对多个子目录运行du的归属可能略不同。

| 文件组 | 独立文件数 | 实际分配GiB |
|---|---|---|
| raw_hdf5 | 6342 | 552.129 |
| tensor_or_checkpoint | 11910 | 140.604 |
| visual_features.zst | 201 | 85.705 |
| visual_features.pt | 136 | 55.464 |
| other | 321702 | 14.192 |
| partial_evidence | 103 | 12.269 |
| logs_or_archives | 34855 | 6.329 |
| video_image | 14016 | 2.109 |

主要run（清理前，内含原数据、模型及派生产物，不能整目录删除）：

| 目录 | GiB |
|---|---|
| outputs/visuotactile/run/tight_dagger_pool_20261111_01 | 109.795 |
| outputs/visuotactile/run/tight_state_20261107_01 | 70.167 |
| outputs/visuotactile/run/tight_dagger_matched_20261112_01 | 62.125 |
| outputs/visuotactile/run/tight_long_20261117_01 | 41.283 |
| outputs/visuotactile/run/tight_fresh_20261115_01 | 38.558 |
| outputs/visuotactile/run/tight_recovery_20261118_01 | 38.389 |
| outputs/visuotactile/run/tight_fresh_20261114_06 | 35.450 |
| outputs/visuotactile/run/tight_fresh_20261114_01 | 34.966 |
| outputs/visuotactile/run/tight_fresh_20261115_02 | 30.664 |
| outputs/visuotactile/run/tight_fresh_20261115_03 | 30.383 |
| outputs/visuotactile/run/tight_cycle_20261118_01 | 27.772 |
| outputs/visuotactile/run/tight_phase_20261116_01 | 24.373 |
| outputs/visuotactile/run/tight_fresh_20261115_04 | 22.282 |
| outputs/visuotactile/run/narrow_repair_20261106_01 | 19.841 |
| outputs/visuotactile/run/narrow_state_20261105_02 | 19.106 |

原始400数据集中在`outputs/visuotactile/data/close_v4`；当前130＋6验证的HDF5来源分布于layout10、phase、long、cycle、recovery等run，由manifest解析，不根据目录名删减。136份当前视觉特征约55.464GiB，均保留。原始及历史HDF5约552.129GiB，均保留；103份partial约12.268GiB视为未确认的失败证据，均保留。全部视频/图像仅约2.109GiB，未删逐例结果、代表性视频或图像证据。

314个识别出的checkpoint/模型快照约107.609GiB，逐文件用途、所属实验、metadata路径、保护分类在[checkpoint_candidates.csv](evidence/tight_disk_cleanup_20261119/checkpoint_candidates.csv)。其他tensor还包括numeric、loss_context、nominal缓存和闭环输入记录，不能全部计作模型。重复筛查先按大小和分段hash排除，再对候选完整SHA；发现17组字节相同的独立文件，理论重复空间1.206GiB，主要为旧numeric/tactile数据。没有为此改变现有历史路径、权限或建立新的共享写风险，全部保留。

本项目发现25,734个软链接，其中13个指向外部环境等资源；候选清理文件均是当前项目内的真实普通文件、nlink=1、同根盘，未跟随这些软链接。归档工作树及其Git历史均未改动。完整本地清单在`outputs/disk_audit_20261119/project_files.jsonl.gz`和`symlinks.json`，不将它们作为大型日志加入Git。

## 3. 候选分类和删除依据

| 候选 | GiB | 类别 | 依赖与结论 |
|---|---|---|---|
| 201个旧perception/*.pt.zst | 85.705 | 1，已删除 | 无当前136依赖；仅冻结特征；201个原HDF5及DINO/生成合同全SHA验证；可重建 |
| 本次完整缓存重建副本 | 0.283 | 1，已删除 | 验证完成后的重复派生产物；结果、SHA、生成代码保留 |
| 历史checkpoint/恢复/比较模型 | 99.777 | 2，保留 | 包含旧best/last和比较点，历史动态引用未排除；不是全部可释放空间 |
| partial失败证据 | 12.268 | 2，保留 | 不能证明没有研究价值 |
| 17组完全相同独立文件的重复份 | 1.206 | 2，保留 | 虽内容相同，但路径/权限/写入语义须单独处理；容量已足够 |
| 原始400、当前130＋6及其他合法数据 | 见详细表 | 3，保留 | 原始数据不删除 |
| A-DP51/52、各自72父完整pair、当前统计与缓存 | 见保护清单 | 3，保留 | 恢复及正式研究依赖 |

删除旧特征的依据不是名字或年龄。实际解析每个cache_index、data manifest、source_sha256、DINO版本/权重SHA、camera顺序、分辨率、stride和缓存结构；它们不含可训练投影、随机增强或优化器状态。当前136个真实缓存路径逐一排除，201个原HDF5逐一全SHA验证。每个文件删除前再次核对archive SHA、inode、mtime、大小、nlink和活动句柄；先写持久化意图，再逐文件unlink，最后写结果。无目录递归删除、无广泛通配符删除。

完整重建测试预先选择最短的228帧旧轨迹以限定计算量，实际执行原生六图DINO核、原depth_grid和同序列化路径，生成304165321字节文件；SHA `70d83e0ff4293285c0f49b5fc8499e9e9f8ceed6b715fa7c382eee2788aa4174` 与原缓存完全一致，耗时6.777s，0optimizer更新、0仿真。**这是1个完整缓存实测重建，不宣称201个缓存均重新计算过**；其余逐项验证相同生成合同及完整输入。测试中的首次缺少直接session路径错误在未加载GPU模型前退出，改为解析真实祖先session；随后完整测试通过。清理脚本首次日志字段冲突在任何unlink前退出（0字节操作日志），修复后执行，所有失败日志保留在本地audit目录。

这是明确退休可重建缓存，**不再是保留压缩档的无损归档**。之前6.3303GiB及随后缓存/日志压缩节省量不能再累计到本次85.988GiB里。原`.archive.json`回执、raw HDF5、cache index和原SHA均保留，新增`.pt.retired.json`说明状态。此前对这些201份文件执行旧Zstd restore命令将找不到归档；历史复现先按下面的新重建入口恢复。未删除24份历史训练日志压缩档。

## 4. 实际清理记录与空间变化

| 原文件组目录 | 文件数 | 归因释放GiB |
|---|---|---|
| outputs/visuotactile/run/tight_fresh_20261115_01/round_11/A/state_data/perception | 45 | 19.128 |
| outputs/visuotactile/run/tight_fresh_20261115_02/round_12/A/state_data/perception | 36 | 15.406 |
| outputs/visuotactile/run/tight_fresh_20261115_03/round_13/A/state_data/perception | 32 | 14.925 |
| outputs/visuotactile/run/tight_fresh_20261115_04/round_14/A/state_data/perception | 23 | 9.391 |
| outputs/visuotactile/run/tight_fresh_20261115_09/round_15/A/state_data/perception | 18 | 8.546 |
| outputs/visuotactile/run/tight_fresh_20261114_01/round_01_v3/A/state_data/perception | 11 | 4.686 |
| outputs/visuotactile/run/tight_fresh_20261114_05/round_06/A/state_data/perception | 9 | 3.462 |
| outputs/visuotactile/run/tight_fresh_20261115_12/round_15/A/state_data/perception | 8 | 2.385 |
| outputs/visuotactile/run/tight_fresh_20261116_02/round_17/A/state_data/perception | 4 | 1.763 |
| outputs/visuotactile/run/tight_fresh_20261116_02/round_18/A/state_data/perception | 4 | 1.614 |
| outputs/visuotactile/run/tight_fresh_20261116_01/round_16/A/state_data/perception | 4 | 1.452 |
| outputs/visuotactile/run/tight_fresh_20261116_04/round_19/A/state_data/perception | 4 | 1.421 |
| outputs/visuotactile/run/tight_fulltask_20261116_01/update/A/state_data/perception | 2 | 0.945 |
| outputs/visuotactile/run/tight_fresh_20261114_02/round_05/A/state_data/perception | 1 | 0.583 |
| outputs/disk_audit_20261119/rebuild_probe | 1 | 0.283 |

审计开始可用空间2.555GiB；删除紧前2.468GiB；删除结束88.316GiB。文件分配块归因释放85.988GiB，实际可用空间净增85.848GiB；两者差异为并发写入、审计产物及元数据，不能说全部磁盘变化来自本清理。

32GiB恢复阈值已满足。按实际DP快照约1.085GiB、后端快照约8.8MiB、两角色nominal合计约0.563GiB及当前日志增速，保守为剩余训练、原子保存临时文件、日志和64例评估/视频预留约20GiB，另保留原12GiB运行保护线。删除结束约88GiB空闲提供额外约56GiB余量；这是估计而非硬上界，其他项目持续写盘仍可消耗空间。详细分项与实测样本见[remaining_storage_budget.json](evidence/tight_disk_cleanup_20261119/remaining_storage_budget.json)。

这里报告的释放均为**磁盘**，不是RAM或显存。验证期间GPU只用于6.8秒的冻结缓存重建；模型不更新，未修改资源阈值。

## 5. 保护验收与恢复状态

清理前888文件共228.367GiB完整SHA全部通过，包含400原始数据、130训练＋6验证HDF5、136当前视觉缓存、A-DP51/52及72父模型、统计/manifest和201个重建来源。原始数据和模型不仅存在，内容与登记SHA一致。清理后完整复核：888/888再次通过。详见[protected_and_rebuild_hashes.json](evidence/tight_disk_cleanup_20261119/protected_and_rebuild_hashes.json)。

恢复状态：`training`。原监督服务active/running，已启动统一入口A-DP worker（PID356879）；启动观察时最新完整轮仍为52，不将进程启动当作新epoch完成。清理前仅对已确认waiting_disk、无子进程的原恢复监督PID316280施加SIGSTOP，保持原服务、训练根、配置、截止和checkpoint；空间跨32GiB时它没有抢先启动。完成清理和核验后只SIGCONT同一进程，不创建第二训练进程。恢复服务仍使用已登记`train_recovery.json`，从A52/4940的完整模型、Adam、EMA、随机数和曝光计数接续到预登记60轮；不重新累计前52轮。不改变训练方案、学习率、采样、数据或架构。训练是否实际推进以恢复回执为准，磁盘任务不冒充训练完成。

收尾核验已观察到A-DP正在执行第54轮（batch18/95）、累计5053次更新；第54轮尚未完成，原服务active/running且当前attempt无失败回执；证明恢复后已实际推进，仍不代表四组件完成。回执见evidence/tight_disk_cleanup_20261119/recovery_final_check.json。

## 6. 历史缓存重建与后续决策

`cold_cache_dependencies.json`每项给出原路径、所有已发现历史index引用和重建来源；`cleanup_manifest.json`的recipe_row用于下面的命令。单个缓存生成到新目录，原HDF5和权重均先检查SHA，禁止覆盖原文件：

```bash
# ROW取cleanup_manifest.json的recipe_row；NAME取对应original缓存文件名。
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH="$PWD" \
  /home/CNS2025331827/miniconda3/envs/isaaclab/bin/python \
  research_docs/evidence/tight_disk_cleanup_20261119/rebuild_visual_cache.py \
  --row ROW --destination outputs/cache_rebuild/NAME.pt
```

重建工具和原生成模块SHA保存在清理manifest。逐项重新生成可能需要显著GPU时间，不在本次自动执行；必须为此登记资源窗口。新文件SHA与保留的original_payload_sha256完全一致后才可恢复旧路径；若版本/硬件导致不同，不得伪造旧hash，应先核验数值并登记新身份。保留数据和代码使重建可行，不保证未来不同环境下自动获得完全相同序列化。

容量目标已达到，**现在不需要批准进一步删除**。约99.8GiB历史checkpoint、partial与重复独立文件只列为待确认；若以后决定退休它们，需要单独确认哪些历史复现/恢复能力可以放弃。其他项目、其Git包、回收站、conda和系统文件本次全部只读，未删除或移动。
