# 2026-11-09 最新授权：平台后小队列闭环与视频

用户明确授权“接近收敛可以尝试成功率实验并录制视频”，仅覆盖此前无新学生闭环限制。
原训练运行于35485d7工作树，保持其代码、预算、数据、优化器和评估完全不动。
本分支/独立工作树复用tools.tight_videos与configs/tight_failure_videos.json的conditional_evaluation；
普通CPU程序等待原训练及最终38条离线诊断正常结束、全部owned进程退出，再检查原≥200 anchor等效epoch/最后4次validation分数改善<1%、无严重退化条件。
未满足写SKIPPED，不称预算耗尽为收敛；失败则终止，不能放宽阈值来凑视频。
满足才使用最终validation4选定EMA，冻结10个新初态seed27150000–27150009，前3身份录制；拒绝不补seed/补录。
最多1h GPU评估，单Kit串行；零成功不扩100，有成功也不扩本次10例。保持29D/H24/commit12/5/60/120Hz与原生完整成功判据。
沿用50%整卡批次/60s上限、35%目标、45%准入、6GiB VRAM/14GiB RAM/60GiB磁盘；实测越界停止自己的录制worker，保留中断分母。
等待截止为原整体deadline+120s，评估绝对截止再加1h；不重置原训练截止、不重试、不新训。
root outputs/visuotactile/demos/tight_palm_eval_20261109_01。机器状态/README为准；Codex启动短查后退出，无LLM守候。

# 当前 Tight-Insertion：掌部几何损失与有限扩充的新训练

最新显式任务授权一个全新StateDP：只原位改已有文件，补齐输入审查，commit12掌部位置/完整旋转辅助loss，
原36训练家族有限新增目标50/最多60Core，最多120物理尝试/2h；原val30/test22、Aux不混入。
一次fresh训练目标300 anchor等效epoch，通常≥200才平台早停，训练墙钟9h优先，最后固定train8/validation30离线诊断≤1h；无自动学生闭环或下一轮。
读CONTRACTS与STATE_DP_RESULTS，配置configs/tight_state.json，入口tools.tight_finish。
root `outputs/visuotactile/run/tight_palm_20261108_01`；读取实际status判断是否启动/结束，不从本段推断训练计数。
采用严格Lease整卡50%上限/35%目标/45%准入，6GiB VRAM/14GiB RAM/60GiB磁盘储备；全流程≤12.5h。
冻结源码后不改运行代码，普通后台完成转换/训练/报告，Codex不驻守；结束后不自动追加。

## 已完成 Core/Auxiliary 基线

使用`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`、`feature/tight-state-foundation`；读research_docs/CONTRACTS.md、STATE_DP_RESULTS.md和SESSION_HANDOFF.md。root为outputs/visuotactile/run/tight_state_20261107_01/composition_06，2026-11-08 13:21:54 complete/evaluated，所有owned服务/worker/monitor已退出，不重启旧队列。

采集源码4d43a3975c9bd153ff90d7c13e03ca9a0e6aa917；训练/评估fdb7ea94a1457de57ff2a9fc21d1a9486742d5ab。208Core/50家族＋15Aux/11家族冻结，Core-only fresh8500更新/100.21 anchor等效epoch，完整恢复及有限充分性条件通过；闭环train0/10、validation0/7、新初态0/10，学生仍无完整成功。没有自动追加实验、采样、Aux模型或100例评估。

保持command-based target_after_rad、Arm7+Hand22 rad绝对目标、持续GRU、原网络/损失/权重、H24/commit12、5/60/120Hz、DDIM/overlap0与原生任务/PD/物理/DR/专家。当前不是64tick截断历史。源码未变的168/510tick等基础证据继续有效，不重开全仓审计；本轮61项定向/准入＋1成员回归通过。

旧20严格10/20与旧275混速结果独立保留。原400/275、专家/资产、开发/成功证据及失败分母保护。135拒绝RGB-D+1重复checkpoint已清理36.191GiB，不重复删除；best原字节保留，last完整恢复硬链接。当前数据/统计/cache路径与225合格载荷SHA清理后通过。

资源实况存在整卡60s59.632%及空闲VRAM4154MiB越界，不能称全程合规；遥测/失败记录保留。v1 SIGSTOP造成比例等待重复计时，已改v2原gpu.lock计算段外等待，trainer未重启、算法未变，guard已退出。不得修改无关进程。

不上传数据/checkpoint、不重写共享Git历史、不新增重复MD。后续显式任务覆盖旧阶段授权限制；本文件记录完成边界，不把数据/训练通过冒称学生成功。
