# 当前清理与回归验收

本轮用户授权删除已淘汰的代码、权重与大型原始产物；旧保留全部失败文件的限制已被覆盖。历史数值结论仍见[精简历史](RESEARCH_HISTORY.md)。

已完成两批清理：20,808个文件，139个旧模型checkpoint路径（含别名），1,922个其他Torch缓存/初态文件，实际释放122.3768 GiB分配块。删除前记录路径、大小、引用和理由，核验文件身份、SHA及当前进程引用；硬链接只计算最后一条链接释放的块。未建立大型备份。

保留原400成功轨迹、320/40/40索引、当前numeric/DINO缓存与train统计、State/Image10000和State100/150未完成对照所需权重、当前best/latest完整恢复状态、当前评估/专家证据和必要视频。未知data/checkpoints、符号链接、小型来源证据和代表性崩溃日志保留，不能按日期整目录删。

源码整理在隔离worktree进行，正在运行的trainer和评估继续使用其冻结源码。原环境/执行源码21项逐字节保护；56个公共符号AST等价；真实State16584/Image10000的EMA/optimizer/采样状态和20个条件/预测/图像字段CPU对照逐值一致。整理后的77项测试通过；新增收尾保护及报告逻辑13项聚焦检查通过。CUDA恢复和整理后Kit验收仍待当前训练及8次评估完成，不能提前标成通过。

完整删除ledger、历史精简metadata、源码依赖图、迁移等价性与CPU对照在评估root `outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01/maintenance`。最终切入后更新此文为实际回归、源码删除数量、最终保留权重SHA和发布结果。三端点结果见[STATE_DP_RESULTS](STATE_DP_RESULTS.md)。
