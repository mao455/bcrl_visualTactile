# 当前交接

## 新授权：成功判定审核与历史100/全新100

原生判定独立重构：4条train与400专家全部与原记录逐tick一致；但train982402在tick168–186连续19tick满足最终7.5mm几何与撤手条件，前置预插入仅5/10tick导致原生顺序状态机不终止。严格原生仍0/4，终点诊断1/4，绝不能混计。报告含时序图；用户现已明确确认：保留原生顺序为主指标，终点满足另列诊断。运行记录USER_SUCCESS_CRITERIA_CONFIRMATION.json保存原回复；与已执行队列一致，无需重跑或重分类。

当前正式队列：`outputs/visuotactile/native_dp_v2/state_random_20261105_02`，服务`state-random-20261105-02.service`，冻结评估源码`100d71e`，checkpoint SHA `339fc3ddb8f0835f34d9857a4653dfa0ee706d3ec4901bb21b9069312399a15e`。4条受监督完整smoke通过：State验证601tick超时、State训练200tick任务失败、原生专家145tick成功并同tick终止、全新初态601tick超时。全部数值有限、视频完整解码、状态机重构一致、资源回收；首plan对清理前最大差7.28e-7rad。原生完整成功后额外动作0。`SUPERVISION_ACCEPTED.json`记录人工核查的四条文件SHA，正式历史100→全新100已放行，当前分母必须读summary.json，不能把本次启动当作200次完成。

固定seed20261105，历史100=91train+9validation，fresh100独立，另各20仅作动作前非法初态备用；不使用test40。全部单环境独立进程，无并行或复用等价声称。主指标保留原生顺序完整成功，终点诊断另列。网络、H24、执行12、29D控制、原生阈值和checkpoint未变。

前身`state_random_20261105_01`已blocked且全部资源回收，两个软件失败原证据保留：第一条完成601tick但末尾distribution漏函数导致结果提交失败；第二条场景前OmniClient线程SIGSEGV。原截止时间、抽样和合计245启动上限不重置。新版本停止预加载OmniClient及未公开test setter，使用Kit自身初始化；四条完整smoke及后续正式运行成功，不声称已证明全部历史FileThread故障的根因。真实256watch探针通过，无ENOMEM/ENOSPC证据。进程退出为结果fsync后的隔离回收，不冒充SDK优雅关闭。新正式软件错误将保存具体位置并停止盲目重启。

临时开发worktree `outputs/visuotactile/tmp/state_success_review`已在切换前移除；冻结的运行源码不得修改。独立CPU交付入口`tools.evaluation_delivery`只在owner最终退出后汇总既有结果、失败阶段和Image建议，按锁定HEAD及报告正文校验推送轻量产物；不启动Kit或训练。其独立服务及实际进度看root的delivery_request.json/delivery_status.json，不凭本文推断已启动或已发布。若分支HEAD、人工报告正文或暂存区发生额外改变，发布停止等待审查，不提交其他人的工作。

## 此前阶段（历史结果）

StateDP150训练与正式评估完成；源码/产物清理已完成。完整结果见[统一报告](STATE_DP_RESULTS.md)，清理与未通过项见[清理报告](CLEANUP_AUDIT.md)。

- 当前分支`feature/native-dp-training-v2`，源码目录`bc+rl_RSS_native_v2`；共享outputs真实目录属于原bc+rl_RSS项目，原IDE工作树的用户修改未覆盖。
- StateDP停在20918updates／150.0101anchor等效epoch；最新=最佳checkpoint，完整EMA/AdamW/RNG/sampler已验证。训练owner、monitor及所有本次Kit worker均退出。禁止直接恢复200–300或启动Image。
- 冻结validation4：0/4成功、4接触、0抬升/子目标、4超时。train4拟合诊断：0/4成功、4接触、2抬升后fall、2超时。8启动8有效、软件缺失0，已完成视频完整解码，不重跑、不混入test。
- train320/validation40同口径10000/16584/20918端点分析完成。100→150训练loss−54.5%，验证loss+9.0%，验证动作重建MSE改善1.008%；停止盲目扩展预算。Image10000保持未续训，strict live pairing仍未解决。
- 清理删除160研究源码、26旧测试、48旧MD、143旧模型checkpoint路径，实际释放126.673GiB。400原轨迹、索引/缓存/train统计、当前完整状态及精简比较证据保留；未知项有明确保留原因。临时源码worktree已移除。
- 清理后78CPU测试、实际State/Image CUDA恢复、视频检查通过。额外Kit验收首次因漏ElementTree导入失败（cf684dc修复）；修复后的唯一补充启动在场景创建前FileThread SIGABRT（−6）。额外2启动0有效，均保留；不改变正式8次结果，不宣称完整Kit回归通过，不自动重试。
- 当前run根：`outputs/visuotactile/native_dp_v2/anchor_continue_20261104_01`；结果根：同级`anchor_assessment_20261105_01`。最后状态看结果根`maintenance/finalization_recovery.json`；原`finalization.json`保留初次blocked状态，不覆盖证据。
- 新结果不支持立即改网络/损失或重训。下一轮建议先定位首12tick在线/专家输入差异与小幅动作精度；这只是建议，不是已运行的消融。
