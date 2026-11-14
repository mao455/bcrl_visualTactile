# 第九轮一次只读I/O采样与统一特征检查

report.json来自2026-11-15 02:11:10至02:12:10的实际/proc采样，仅支持该窗口的读盘等待。sample.py为原执行字节，目标PID属于当时两个trainer；不会重放历史运行，若复制后用于新采样，必须明确更换目标并在新的私有目录运行。它写入自身目录，因此不要在本证据目录直接执行。原始process_metadata和samples留研究机，公开摘要不含完整其他进程参数。

feature_check02.json来自统一 tools.tight_fresh_training_check 的CPU模式；865帧四种tensor字节/layout相同，0GPU/0模型/0更新。实际argv和冻结源码写在session06/unified_feature_io_check02_launch.json；第一次exec权限拒绝没有启动worker，第二次显式bash完成。两读取的冷暖状态不同，不将单次加载与lazy mmap open比较为加速比。
