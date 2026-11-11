# 2026-11-11：完整策略 A/B，原run安全边界接续

工作树`/home/CNS2025331827/docs/bc+rl_RSS_qg_report`。最新范围`complete_strategy_ab_v1`覆盖先前QG01/G11齐步三轮安排；本次不是取消DP训练。原run `outputs/visuotactile/run/tight_dagger_pool_20261111_01` 于2026-11-11 13:24:31启动，仍在2026-11-12 07:24:31绝对结束（07:14:31停止计算）。保留原run/source历史、全部已完成训练和数据，禁止用新run重置18小时。

修改前已请求本任务STOP并核验owner退出：`status.json`为STOPPED、`owned_workers_exited=true`，G11前端在754正式更新保存边界退出，loss=0.00111453、selected=201；没有把未完成后端与自主评估算完成。QG01首轮完整pair保持前端804/后端12212更新，自主0/10成功、2lift、0子目标、全部601tick超时。已有80条纠偏（73train/7dev）与Core166保留。上述为安全边界事实；本次重新注册/启动后的状态以root新机器凭据为准，文档准备不等于新GPU启动。

## 两个完整策略及身份

|角色|前端|后端|标签与模态|
|---|---|---|---|
|A（来源QG01）|本run round_01/QG01/dp/training/best.pt，`df46795c…`|同round backend/QG01/training/best.pt，`5695d85a…`|G00模态，q_reached＋既定几何；后端最终u_expert|
|B（来源G01）|tight_vision_20261109_01/G00/training/best.pt，`4356acbc…`|tight_transformer_20261110_02/command_residual/G01/training/best.pt，`0e9ec397…`|同G00模态，u_expert；后端最终u_expert|

真实CPU读取确认B是10132前端更新和27092后端更新的有效pair；后端非disposable。A/B前端均RGB-D＋robot178、无直接触觉；后端均756581参数、2层d128/4头/FF256、过去32tick＋持续GRU。H24/commit12、DDIM16/seed7091与5/60/120Hz一致，后端u统计逐tensor相同。A残差幅度Arm0.2/Hand0.9，B全0.1，保留已训练范围；历史曝光及本轮预算不等，不能包装成单变量或同预算比较。G11退出活动矩阵，已有数据与754更新恢复状态保留。

## 后续有限队列

|阶段|登记安排|
|---|---|
|第2轮|A最多96/B最多48次完整学生接管采集，冻结共享快照，分别更新DP→刷新缓存→更新Transformer→自主评估|
|第3轮|更新后A最多96次新学生状态采集，再更新两部分并评估；B保留第2轮pair，不等待齐步|
|采集截止|240只是上限；按100秒/机会与7800秒/完整pair预留裁减，并预留后续A至少32机会；绝对截止优先|
|DP增量|每组最多800更新，400后按验证趋势可停，每200验证，保留400/800共同曝光；实际anchor遍数另外报告|
|后端|最多12/最少6有效tick遍，3遍验证，.75h；仅后端训练阶段冻结新DP|
|数据|持续Core＋全部合格旧/新纠偏，保持.5/.2/.3来源层；纠偏恢复初48合法tick前端anchor抽样目标.5、后端loss权重3|
|评估|完整pair固定4例，登记进展条件才同队列扩10；test22不读，接管成功不计自主成功|

每次采集实际验证前后端SHA与nominal/residual/issued/applied链，不能仅检查backend配置；缺少/加载失败/不配对须报错。已有正确路径应写“未发现裸DP采集问题”。独立接管起点、阶段/偏离描述、初段有效监督与来源曝光嵌入采集训练，不追加长诊断。裸Q/QG只做训练与离线验证。

已有实测A采集93.33秒/机会，旧G11参考83.54秒/机会（B需实测）；DP约5.115秒/update；239训练episode时每组预测缓存与后端12遍各约14分钟。池扩大后缓存/后端时间增长，不能将旧常数外推为保证；实际queue按原deadline裁减机会，报告未完成项。

## 恢复、发布与保护

普通owner负责有限阶段、退出、CPU报告与授权推送，启动短监督后不靠模型轮询。`bash visuotactile/run_cpu.sh -m tools.tight_finish dagger_resume --root <root>`按本次重新封存配置与原绝对截止恢复；已完成阶段跳过，中断同阶段保留optimizer/EMA/RNG/decks，未持久化loss另存uncommitted，不重复记曝光。不要从旧G11半轮拼装新pair。

发布器须遍历历史round_01实际目录（含退休G11），新轮按实际活动组及job汇总，未调度B第3轮明确not_scheduled/未测，不算失败，也不阻塞本次scope完成。只有真实完成凭据能证明训练/评估完成；历史停止不能被改成COMPLETE。大数据/权重/视频留本机，公开源码、精简指标/曲线与准确SHA索引到origin指定feature分支。

保护原400、Core、专家/资产、有效聚合数据、A/B初始化及当前/恢复依赖、历史成功/失败证据。本次范围修订本身未授权删除仍有恢复/对照价值的G11产物。实际启动验收与测试结果继续由root凭据和[变更记录](CHANGELOG.md)更新；历史数值见[结果页](STATE_DP_RESULTS.md)。
