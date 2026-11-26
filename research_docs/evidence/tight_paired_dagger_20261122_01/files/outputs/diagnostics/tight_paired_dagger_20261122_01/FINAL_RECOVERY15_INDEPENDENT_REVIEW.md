# Final15 独立源码复核

通过，未发现剩余源码阻断。此签核仅针对以下六份 inactive15 证据工具及其配方，不认证实际运行闭合、成功率、计算开销或最终导出。runtime15 已冻结；本复核未修改它或冻结13/14源码，未读实际生命周期／评估结局／权重／缓存载荷，未执行实际绑定、提取、组装、导出、GPU工作、进程查询或信号。

| 源码 | SHA256 |
| --- | --- |
| final_recovery15_common.py | `2d706ea709fde903f58efdf9c801ad83bd9d9553f7b28d096dd2f8073933ace3` |
| extract_selected_ledgers_recovery15.py | `5963851279881f6f30fcc6dd98361aea1f2d66f1e212b8455fb7404659721505` |
| final_report_assembler_recovery15.py | `fba539cf6cf28503e4baa0ee43899297cf02cac589b434a7503f3b40f2a6fb67` |
| final_compute_recovery15.py | `0eea5f07351e56a6b0cab295c7f1eb019e4bc5e2379244b1f2bf813697f9d8a4` |
| export_final_evidence_recovery15.py | `ffc258ab91ca33facbf94a2e59aa183af73c6788c5e58f8deab4be1d20b614b1` |
| test_final_evidence_recovery15.py | `15c012ee18c971f19d24aaa73a978a31c9dab036af589927ee96460c01b3d486` |

六份源码实际SHA及AST解析均通过，与 `FINAL_RECOVERY15_SOURCE_READY_FOR_REVIEW.json` 一致；36份继承依赖保持登记SHA。配方 `FINAL_RECOVERY15_RECIPE.md` SHA为 `a0e7a8f3910f158202e6e5852fd8ed530fc3a797884175dae953082180db60bd`。独立算力记账审查 `INDEPENDENT_COMPUTE15_REVIEW.md` 结论与所审源码一致。

原阻断已修复：failed13的driver错误现在要求包装层的 `evaluate_fragment13 failed; no retry or replacement is launched`；子进程原因 `Original suffix changed` 从独立SHA绑定、保留在本地的helper日志读取，不再混入driver字段。缺失／错误child cause及错误wrapper有合成拒绝例。builder与reviewer都采用root13/14/15三份boundary中的原始日志，加上必须的failed13日志这一精确并集；额外无绑定日志被拒绝，原始日志仍不进入公开小文件集合。

生命周期审查保持原64家族／256策略机会分母，已消耗0..16及失败12/16不补跑；continuation15只执行17..63。launch与stage-exit逐波绑定；必须先回收再推进，非零退出不能改成软件完成。队列完成、整波terminal与合法已完成lane分开记录；后续worker失败不会抹掉已闭合lane，完全未初始化失败仍保留各策略分母。局部1800s启动超时具有intent／decision／stop及就绪竞争证据；全局STOP、watchdog或基础设施失败不得伪装队列完成。

current15 watcher要求所有实际动态backup恢复到原字节／不存在，历史watcher使用其不可变证书；各代合作pause仅允许精确绑定字节。selected extractor继续对8个实际选中checkpoint做独立SHA／CPU mmap／计数验证；assembler及exporter只复用8份稳定表，新增载荷读取0。仍只有两份精确cache阶段JSON例外，保持16MiB单文件、256MiB总量、不可覆盖导出。成本将真实owner、failed13/14及current15 driver、辅助阶段区间去重合并，保留未知历史collection结束；不伪造GPU小时。

验证复用作者已记录的85项通过（80作者＋5独立，49.04s），未重复扩大测试。此次单独重跑 `test_independent_final15_review.py`：5项全部通过，3.05s；先前复现失败的有效wrapper＋child-log案例现通过，重复launch、未回收、提前推进和伪造软件成功四项仍拒绝。测试只使用临时文件／合成记录，没有真实进程生成或实际评估输入读取。

下一步由root完成已授权的实际收尾：等待真实47机会队列及原辅助汇总退出；核验全部owned actor回收，停止own watcher15并逐backup确认恢复，写真实closure；然后按配方创建并审查实际绑定，执行一次8-payload详细曝光提取、成本记账、草稿组装、干跑审查和有界导出。任何pending仍为pending。上述实际步骤完成前，本签核不能作为实验完成或自主结论的证据。
