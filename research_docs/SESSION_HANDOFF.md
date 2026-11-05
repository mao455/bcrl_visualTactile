# 当前交接

StateDP150训练与正式评估完成；源码/产物清理已完成。完整结果见[统一报告](STATE_DP_RESULTS.md)，清理与未通过项见[清理报告](CLEANUP_AUDIT.md)。

- 当前分支`feature/native-dp-training-v2`，源码目录`bc+rl_RSS_native_v2`；共享outputs真实目录属于原bc+rl_RSS项目，原IDE工作树的用户修改未覆盖。
- StateDP停在20918updates／150.0101anchor等效epoch；最新=最佳checkpoint，完整EMA/AdamW/RNG/sampler已验证。训练owner、monitor及所有本次Kit worker均退出。禁止直接恢复200–300或启动Image。
- 冻结validation4：0/4成功、4接触、0抬升/子目标、4超时。train4拟合诊断：0/4成功、4接触、2抬升后fall、2超时。8启动8有效、软件缺失0，已完成视频完整解码，不重跑、不混入test。
- train320/validation40同口径10000/16584/20918端点分析完成。100→150训练loss−54.5%，验证loss+9.0%，验证动作重建MSE改善1.008%；停止盲目扩展预算。Image10000保持未续训，strict live pairing仍未解决。
- 清理删除160研究源码、26旧测试、48旧MD、143旧模型checkpoint路径，实际释放126.673GiB。400原轨迹、索引/缓存/train统计、当前完整状态及精简比较证据保留；未知项有明确保留原因。临时源码worktree已移除。
- 清理后78CPU测试、实际State/Image CUDA恢复、视频检查通过。额外Kit验收首次因漏ElementTree导入失败（cf684dc修复）；修复后的唯一补充启动在场景创建前FileThread SIGABRT（−6）。额外2启动0有效，均保留；不改变正式8次结果，不宣称完整Kit回归通过，不自动重试。
- 当前run根：`outputs/visuotactile/native_dp_v2/anchor_continue_20261104_01`；结果根：同级`anchor_assessment_20261105_01`。最后状态看结果根`maintenance/finalization_recovery.json`；原`finalization.json`保留初次blocked状态，不覆盖证据。
- 新结果不支持立即改网络/损失或重训。下一轮建议先定位首12tick在线/专家输入差异与小幅动作精度；这只是建议，不是已运行的消融。
