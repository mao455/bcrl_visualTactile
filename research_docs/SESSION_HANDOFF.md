<!-- EXECUTION_LEARNING_STATUS_START -->
## 2026-11-13 自主执行学习修复：实际状态

状态 **PREPARED / CLI_recovery_not_started**；原绝对截止 **2026-11-14 00:05:15 UTC+8** 不变。
首个旧/FY各200更新已完成，FY的A/B dev执行误差比1.139/1.181，未通过原门槛。固定DP输出域下界占A归一化错误33.46%，B为0；据此比较保持共同执行器的线性尾部提案。
配对延长、必要优化调整、完整混合数据后端训练、完整DP+Transformer开发评估分层记录；若闭环失败且剩余预算允许，最多两轮、每轮12机会/组的定向DAgger同时更新DP、缓存和后端。初始门槛失败不结束整个修复，也不改写旧结果。
GPU工作与Git发布分别调度；未完成阶段不能称为成功。test22隔离、源数据和旧checkpoint保留。
[方法与限制](DAGGER_METHOD.md#execution-learning-repair) · [阶段状态](evidence/tight_execution_learning_20261113/adaptive_recovery_01/follow_status.json) · [配对计划](evidence/tight_execution_learning_20261113/adaptive_recovery_01/adaptive_probe_plan.json) · [表达下界](evidence/tight_execution_learning_20261113/adaptive_recovery_01/capacity.json)。
<!-- EXECUTION_LEARNING_STATUS_END -->

# 2026-11-12 16:35：性能实测完成，等待同run安全边界接入

性能准备树 `/home/CNS2025331827/docs/bc+rl_RSS_dagger_perf`；详见 [DAGGER_PERFORMANCE](DAGGER_PERFORMANCE.md) 和 `evidence/tight_performance_20261112`。单DP1.228×、预测缓存1.489×、双DP计算1.935×均为局部真实输入探针；完整采集、完整轮和自主评估加速未测。只计划第2轮DP并行，缓存/rear/eval顺序；不启额外线程、多env、AMP、TF32或compile，后端约2%候选已撤回。

同一 `tight_dagger_matched_20261112_01` 的A第一轮前端已于16:24完成1,355更新、6.00263 anchor遍、86,720anchors/2,015,081监督target ticks，目前进入新DP缓存。B第一轮DP及两组新后端、固定10次自主评估尚未完成。A/B已48次采集机会各准入42/41专家恢复，不能写作学生自主成功；旧0/4不属于本次控制匹配结果。完成/验证数值见 [时间快照](evidence/tight_performance_20261112/state_snapshot.json)，不由末次train loss推断收敛。

性能迁移指定在`r01_QG01_predictions`完成边界，仍须核验原owner/自有worker退出，保留所有已完成凭据、原起点和2026-11-13 11:22:40绝对截止。第一轮B继续A所用micro4；第二轮两组共同micro16/effective64/rawcache并行，每worker2CPU/2线程、nice19不变。第一轮A已prepared缓存保留，B准备新缓存时可用逐位等价新路径；后续采集共同LZF。`performance_request/revision/status`与实际启动凭据区分准备、迁移和运行，当前文档不代替它们。16份临时CPU基准HDF5共447.3MB及一份1.1649GB重复disposable训练checkpoint已删除，逻辑合计1.6122GB；已分配块另记，所有正式源文件SHA不变。集成176项定向测试通过（17.77s），不能代替生产接入凭据。

以下为之前控制修订的准备时快照和历史方案，不是当前运行状态。

# 2026-11-12：控制一致性审计与配对 continuation

隔离树 `/home/CNS2025331827/docs/bc+rl_RSS_control_audit`。最新附件授权修复后有限继续；旧run不停止：第3轮A已完成800/24112更新、自主0/4，B在10:06读取时725/800前端更新，后续由原owner推进，13:24:31截止不改。此为带时间快照，不能替代最新机器状态。

CPU审计：327条数值轨迹11项provenance检查通过；564非test轨迹169,853合法专家tick投影改变0、原映射重算误差0。旧学生目标可超专家同历史可达范围是实际问题，尚未证明是零成功的唯一原因。原标签已经减速/delay，不能二次处理。[Method与算法](DAGGER_METHOD.md) · [逐阶段数值](evidence/control_consistency_20261112/summary.json)。

新root `outputs/visuotactile/run/tight_dagger_matched_20261112_01`，普通service `tight-dagger-matched-20261112-01.service`。实际状态读registration、continuation_request、successor_service、status、integration、phase0/finished、round*/matched_exposure、public_git；准备文档不等于启动。等待旧owner/worker退出与发布，核验控制验收和源码SHA，整合原报告推送后admission计24h。无新增仿真来重复旧诊断。

新共同安排：最近共同完成旧轮的candidate与同快照，先双组rear6tick遍适配（DP更新0，仅此phase0不是完整DAgger），复用SHA完全匹配的预测缓存；然后最多2轮，每组48配对机会，DP6anchor遍→刷新缓存→rear12tick遍→固定10纯学生开发。共同.2/.9幅度、新执行投影、endpoint与采样种子；历史初始化仍不同。每轮两组真实曝光一致且完整完成才继续。原合法数据/权重保留，无伪q或全部旧池隔离。

准备：`bash visuotactile/run_cpu.sh -m tools.tight_finish dagger_matched_prepare --root <newroot>`；普通后继 `dagger_follow --root <newroot>`；admit后恢复 `dagger_resume --root <newroot>`。首次正常采集首2例验收新执行器，失败显式停止。新结果使用独立managed block，旧结果与大文件留原处。下面是此前已登记的条件追加历史，不能改写为新公平协议。

# 2026-11-12：完成当前评估后，零成功条件下追加一次

本次准备在`/home/CNS2025331827/docs/bc+rl_RSS_dagger_continue`隔离树进行；原`bc+rl_RSS_qg_report`的当前B训练/评估继续按冻结配置完成。本页新段是授权与实现边界，不声称条件已触发、迁移或真实新采集已经启动；实际状态以同一run的`conditional_extension_request.json`、`conditional_extension.json`、`conditional_extension_status.json`及service凭据为准。

用户授权仅当第2轮A/B的新完整candidate各自完成已分配合法纯学生评估且双方0完整成功，追加第3轮。使用这两套新DP＋Transformer继续产生学生状态、同状态专家接管、成功纠偏加入原Core与历轮池，然后分别更新DP→刷新缓存→更新后端→配套自主评估。不能把前轮开发回退选中的旧pair作为本次追加学生；旧选择证据保留。任一组成功不追加，未完成/非法/软件错误不按0成功处理。

第3轮A/B各最多96机会，DP最多800更新、后端最多12有效tick遍；机会按实测时间及完整训练预留裁减。最多一轮，不递归：仅触发后从原07:24:31截止延长最多6小时，到2026-11-12 **13:24:31**，计算**13:14:31**停止，原2026-11-11 13:24:31起点不变。4→10自主评估协议、test22隔离、两类标签、恢复初段mask/权重与几何计算范围不变。

普通后继须先核验真实评估与原owner/worker退出，再迁移同root/source冻结并恢复；禁止在旧worker运行时替换代码。公开请求、判定、实际进程启动和真实采集验收各自凭据；数据不足只是待验证原因，不预先写成结论。若无触发或证据不足，应明确退出原因和未执行事项。

以下为11月11日已完成的范围修订及当时计划记录，其中“第3轮仅A/不延长”由上面的条件授权覆盖，历史训练、数据和来源不改写。

# 2026-11-11：完整策略 A/B，原run安全边界接续

工作树`/home/CNS2025331827/docs/bc+rl_RSS_qg_report`。最新范围`complete_strategy_ab_v1`覆盖先前QG01/G11齐步三轮安排；本次不是取消DP训练。原run `outputs/visuotactile/run/tight_dagger_pool_20261111_01` 于2026-11-11 13:24:31启动，仍在2026-11-12 07:24:31绝对结束（07:14:31停止计算）。保留原run/source历史、全部已完成训练和数据，禁止用新run重置18小时。

修改前已请求本任务STOP并核验owner退出：`status.json`为STOPPED、`owned_workers_exited=true`，G11前端在754正式更新保存边界退出，loss=0.00111453、selected=201；没有把未完成后端与自主评估算完成。QG01首轮完整pair保持前端804/后端12212更新，自主0/10成功、2lift、0子目标、全部601tick超时。已有80条纠偏（73train/7dev）与Core166保留。上述为安全边界事实；本次重新注册/启动后的状态以root新机器凭据为准，文档准备不等于新GPU启动。

**真实接续核验（北京时间18:26:55）**：同一service已以源码`c80829c`恢复，状态RUNNING / round_02_collection。新A前两次实际采集均通过完整DP＋Transformer身份与数值链读回，264个冻结源码/配置文件未变；没有额外诊断rollout。B的真实采集验收将由同一owner在其首2个登记机会内执行，目前未测。本轮新训练与自主评估尚未完成。[启动验收](evidence/tight_dagger_pool_20261111_01/scope_startup_acceptance.json) · [64次既有采集审计](evidence/tight_dagger_pool_20261111_01/complete_policy_acceptance.json) · [原run修订/完整pair身份](evidence/tight_dagger_pool_20261111_01/scope_revision.json) · [定向测试](evidence/tight_dagger_pool_20261111_01/scope_tests.json)。

安全边界停止至重新监控之间存在资源采样缺口；后续报告保留这一缺口，不声称全会话连续覆盖。各活跃文件按各自写入时间读取，启动验收中两例已完成的读回与progress字段可能相差一次落盘。

## 两个完整策略及身份

|角色|前端|后端|标签与模态|
|---|---|---|---|
|A（来源QG01）|本run round_01/QG01/dp/training/best.pt，`df46795c…`|同round backend/QG01/training/best.pt，`5695d85a…`|G00模态，q_reached＋既定几何；后端最终u_expert|
|B（来源G01）|tight_vision_20261109_01/G00/training/best.pt，`4356acbc…`|tight_transformer_20261110_02/command_residual/G01/training/best.pt，`0e9ec397…`|同G00模态，u_expert；后端最终u_expert|

真实CPU读取确认B是10132前端更新和27092后端更新的有效pair；后端非disposable。A/B前端均RGB-D＋robot178、无直接触觉；后端均756581参数、2层d128/4头/FF256、过去32tick＋持续GRU。H24/commit12、DDIM16/seed7091与5/60/120Hz一致，后端u统计逐tensor相同。A残差幅度Arm0.2/Hand0.9，B全0.1，保留已训练范围；历史曝光及本轮预算不等，不能包装成单变量或同预算比较。G11退出活动矩阵，已有数据与754更新恢复状态保留。

## 11月11日登记的历史后续计划

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

发布器须遍历历史round_01实际目录（含退休G11），新轮按实际活动组、条件判定及job汇总；条件未决、明确未调度和真实已启动分别呈现，不把未调度组记失败。只有真实完成凭据能证明训练/评估完成；历史停止不能被改成COMPLETE。大数据/权重/视频留本机，公开源码、精简指标/曲线与准确SHA索引到origin指定feature分支。

保护原400、Core、专家/资产、有效聚合数据、A/B初始化及当前/恢复依赖、历史成功/失败证据。本次范围修订本身未授权删除仍有恢复/对照价值的G11产物。实际启动验收与测试结果继续由root凭据和[变更记录](CHANGELOG.md)更新；历史数值见[结果页](STATE_DP_RESULTS.md)。
