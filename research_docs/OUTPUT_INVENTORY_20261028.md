# 项目产物盘点与清理（2026-10-28）

快照UTC 2026-10-28T16:01:01.388009+00:00。本表修正5d88060中的统计错误：当时把du分配块字节当成文件字节、对重叠目录求和导致visuotactile少计。本表按lstat文件逻辑字节，不跟随链接，十进制GB；后台写入会使数字变化，实时值看生产`run_status/resources.json`。

| 当前outputs类别 | 字节 |
|---|---:|
| `compat_check` | 406,133,811 |
| `visuotactile` | 8,445,657,829 |
| `source_review_177a080_200ea34_6fc04a8` | 13,636,062 |
| `research_audit` | 463,230 |
| **outputs合计（含在写raw/转换暂存）** | **8,865,890,932** |

从本项目日志明确引用路径识别出的149个外部Kit/Lab日志另计 **20,892,648 bytes**，没有遍历/读取其他项目日志。合计约 **8.887 GB**。源码/原资产/依赖/released权重是必须保留的基础设施，不混入训练产物预算。没有发现outputs之外的runs/train_dir/wandb等项目产物目录。每步及监督器覆盖所列项目输出roots；共享安装目录的Kit日志不由它删除，以上单独盘点；额外512MB边界余量不能被当作已测GPU/磁盘峰值。

本轮累计确认释放 **6,479,709,460 bytes**：

- 废弃相机候选、重复PNG/NPZ/MP4/raw载荷：3,037,673,433 bytes，ledger `camera_v3_20261028/candidate_cleanup_ledger.json`。
- 非法初始化/误两相机v4载荷：281,004,927 bytes，ledger `camera_v4_close_20261028/version4_probe_audit.json`；metadata/results保留，未把启动失败填成空episode。
- 已证明143字段精确转换的重复raw，以及普通生产MP4：3,161,031,100 bytes，ledger `camera_v4_close_20261028/duplicate_cleanup_ledger.json`。删除raw前重新匹配源/compact hash；保留的compact可恢复所有原始字段。普通MP4先明确修改manifest的video_files/files并验证reader，再删除；HDF5 hash没有变。

保留旧v3三条合格生产HDF5、全部必要manifest/标定/seed/小统计、旧三段交付demo及紧凑原始数据、新两条v4验收及少量中文展示。不能把旧pilot有非法初态的成功记录笼统称为全合格训练集。未知损坏/未完成现场不因打不开而删除；不动其他项目、checkpoint、资产、共享依赖。生产尝试后的追加清理另记`insertion_close400_20261028/audits/*_cleanup.json`，不含在上述历史累计数中。

400容量不是已验证承诺：旧三条平均约97MB/条，而新两条较长验收约151MB/条；仅400个成功文件约39–60GB，加当前保留证据会超出部分情景下的60GB硬限。不能以两条平均强行保证目标40GB；新采集按全局预留停机并报告缺口，不降深度精度/采样频率。

最终停止补充：生产2条总HDF5=308,467,237bytes；停止后outputs约8.866GB。额外一次1步启动诊断核验为censored后清理3,638,253bytes，仅保留metadata/result/strace和小统计；累计上述清理约6.483GB，后台各非法尝试的额外小图清理单列。未删除共享Kit中的其他项目或未知minidump。最新本次明确归属的minidump约3.62MB也保留用于启动诊断，不复制大备份。
