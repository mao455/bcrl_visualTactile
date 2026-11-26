# 原机复核清理追加账本

round2 发布基线 303.655537 GiB；本轮快照 306.666183 GiB。本轮删除 912 个路径，按最后硬链接实际释放 10.468266 GiB；不重复计上次 407.722 GiB。

inventory.sqlite 的 cleanup_machine_review_20261126 表为追加记录；逐文件 birth（不可得为 unknown）、mtime、SHA、inode、理由和实际释放见 ../../outputs/diagnostics/tight_machine_review_20261126_01/retirement_deletions.jsonl。完整差异与剩余大户见同目录 disk_final_accounting.json、file_changes_since_round2.csv.gz 和 disk_after.sqlite。

400 原始专家、261 唯一实际回放输入与特征、正式完整模型、成功/失败核心数值均保留。诊断新救回 2 条及特征、中断但有 39 合法专家标签的分支保留；无标签的独立传感副本已退役。
