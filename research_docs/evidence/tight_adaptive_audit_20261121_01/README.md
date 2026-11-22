# 两轮adaptive DAgger独立审计原始证据

此目录导出研究机 `outputs/diagnostics/tight_adaptive_audit_20261121_01` 的审计、逐值数据、绘图和复核源码，原始阶段为只读检查、0 optimizer更新、0新物理步。[审计报告](REPORT.md)确认入口相位缺口、历史DINO microbatch差异和噪声seed说明错误；这些发现不能单独证明旧成功丢失的原因。

报告和原始JSON中的绝对路径是实际历史来源，原始大数据和checkpoint留研究机；诊断脚本保留其当时路径与源码身份，不能直接在任意clone里运行。后续源码修复、train180重训及96例开发结果见[2026-11-22阶段报告](../../TIGHT_ENTRY_PHASE_20261122.md)及[入口证据](../tight_entry_phase_20261121_01/README.md)。

导出文件的来源、SHA和显式变换统一列于相邻入口证据的 `export_manifest.json`。新短诊断和新终测尚未完成；旧审计中的“未执行修复”是当时状态，不是本次发布时的当前状态。
