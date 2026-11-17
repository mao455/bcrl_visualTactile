# DP I/O：同状态短程诊断

主结论及解释见[现有主报告](../../../TIGHT_TRAINING_DIAGNOSIS_20261116.md)。这不是新DAgger或自主成功评估，也没有更新正式选定模型。

- `summary.json`：历史117.33分钟拆分、四个完整短程运行、224主比较与56反序检查、有效曝光、内存和数值残差。
- 四个运行目录：实际配置/源模型与数据哈希，原始逐更新/逐epoch/遥测，完成回执；大文本用gzip无损压缩。`process_io`字段实际计量训练阶段边界内I/O，排除setup和结束后的哈希/缓存审计，不能理解为整个Python进程终身计数。
- `supervision_reconstruction.json`：根据实际采样回执与冻结numeric重新计算的标签/mask摘要；不是另一套运行时捕获。
- `hotset_reverse/cached_tensor_audit.json`：测量结束后，对10轨×4数组进行逐字节回读核验。
- `order_check_registration.json`：主候选完成前登记的反序56更新预算；没有随结果继续扩容。
- `historical.json`：历史166epoch/4648更新训练读取；原始历史日志已在[既有publication](../publication/README.md)上传。
- `interrupted/`：第一次70更新中断的原始回执，不混入224配对结果。
- `disk_probe.log.gz`、`disk_pressure_probe.json`与`reverse_*`：磁盘等待证据；iostat设备级统计不等于本进程独占归因。
- `provenance.json`：本轮代码/输出身份和工程中断说明。运行目录只在研究机共享outputs，训练checkpoint与首步完整梯度未上传。

复核入口为[诊断脚本](../../../../scripts/tight_layout_20261116/dp_io_profile.py)的`main`/`summarize`，优化仅由`--variant hotset`显式启用；正式训练默认路径未改。所有UTC epoch秒可按Asia/Shanghai（UTC+08）换算。CPU-only汇总不会启动训练，但真实训练命令需要原数据及GPU，不应因阅读报告自动执行。
