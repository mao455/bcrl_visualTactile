# 本轮实际清理与清理后检查

当前run `tight_state_20261107_01`。以下只计本轮，历史126.673GiB及其他清理不重复累计。全部删除前检查已完成run、依赖、原始保护目录、同UID打开句柄、inode/mtime/size与必要SHA；不建巨型备份、不删除未知partial。

|ledger|实际文件数|释放分配bytes|内容|
|---|---:|---:|---|
|cleanup_runtime_ledger.json|5409|9,021,136,896|两个已结束旧run的私有Kit临时文件；未知symlink保留|
|cleanup_unused_ledger.json|491|9,631,641,600|400个退休DINO特征cache＋91个已失败观看视频；不是400个模型|
|cleanup_checkpoint_ledger.json|3|3,459,284,992|tiny失败模型best/last及旧275非选中last；best4500保留|
|cleanup_diagnostic_ledger.json|2|2,306,113,536|本轮一次性GPU验收checkpoint；完整恢复receipt和失败诊断保留|
|合计|5905|24,418,177,024|22.741199GiB，按最后硬链接的allocated blocks计数|

账本和每文件删除journal保存在当前root。91条失败视频对应的精简结果、seed、真实分母、终止原因、numeric trace和模型SHA保留于retained_failure_table.json与原结果JSON。旧checkpoint配置/SHA/计数在retired_checkpoint_metadata.json；有成功历史模型与复现配置保留。当前没有正式模型，因而没有可合并的本轮last/best；一次性诊断不冒称正式权重。

源码删除tools/narrow_video_export.py、tools/narrow_videos.py；重复结果文档SLOW_EXPERT_RESULTS.md、GRASP_FAILURE_DIAGNOSIS.md合并入当前结果/历史表。旧全文仍在Git642f5a2，不重写共享历史。其他schema/环境/初始化附录改为当前解释，清除旧阶段待办和冲突授权。

清理后检查：原400全部payload SHA及现275全部payload SHA通过；专家和123个原生Python源码SHA未变；独立入口、坏输入拒绝、自动阶段分支、SIGTERM清理、Markdown链接及依赖检查见机器证据。历史best4500真实model/EMA/optimizer/schedule RNG的CPU读回通过，0优化更新。新随机诊断的完整GPU恢复及下一随机计算通过后，才删除其一次性权重。正式训练/学生闭环没有运行，不宣称这些阶段完成。

首次清理因不可读renderer zombie句柄目录被拒绝，尚未删除文件；仅增加Linux确认Z状态的已退出进程豁免后执行。活进程不豁免，真实zombie/live回归已覆盖。没有对外部进程发信号。

[当前结果](STATE_DP_RESULTS.md) · [精简机器结果](evidence/tight_state_20261107.json)
