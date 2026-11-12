# Play2Perfect / Tight-Insertion：现行工作约定

## 最新覆盖 — 2026-11-12 裁剪梯度快检（候选未采用）

读STATE_DP_RESULTS中的梯度诊断与DAGGER_METHOD 4.1。隔离gradient_diag树真实固定6轨迹/1873合法tick及A/B各旧/新200更新已完成。lambda_pre=1恢复u_pre梯度但dev/Core执行误差退化，不能接入正式队列、自动调参或扩展训练；新损失/当前权重prefix历史均为显式opt-in，缺字段旧行为。matched run仍按perf树源码380a20b正常继续第二轮，原2026-11-13 11:22:40截止不变。后端旧hidden近似如实报告，合法数据/旧结果不作废。无可靠同状态恢复闭环，禁止把离线probe说成自主成功。只推小代码/证据到指定feature分支，权重及源数据保留本机。


## 最新覆盖 — 2026-11-12 吞吐优化与同run安全接续

读 [DAGGER_PERFORMANCE](research_docs/DAGGER_PERFORMANCE.md)。性能改动在 `bc+rl_RSS_dagger_perf` 已做真实输入探针，尚待`r01_QG01_predictions`完成边界、原owner及自有worker退出后接入同一matched run；不热改旧树、不重设2026-11-13 11:22:40绝对截止。仅第2轮两组DP并行；缓存、后端、Kit采集与评估继续顺序。第一轮A/B保持共同旧DP数值配置，下一共同轮才启micro16/effective64和不可变特征GPU缓存，扩散RNG保留原micro4顺序；每worker2CPU/2线程、nice19不变。第一轮A已prepared缓存不动，B新准备可用逐位等价缓存路径。未来采集A/B共同使用已验收lossless LZF及全部源字段校验，磁盘成本约增加30%的实测范围见报告。Transformer搬运候选已撤回；不启4线程、多env、AMP、TF32或compile。保持两类监督、真实历史、几何范围、配对曝光/endpoint、test22隔离与全部保护数据；短探针不是正式曝光或自主成功证据。

## 最新覆盖 — 2026-11-12 控制一致性与配对持续实验

用户最新附件授权证据驱动修复和有限继续，覆盖下文旧run结束后不得继续的限制。读 [DAGGER_METHOD](research_docs/DAGGER_METHOD.md) 和最新交接。合法旧run `tight_dagger_pool_20261111_01` 按原冻结源码和13:24:31截止完成；新树 `bc+rl_RSS_control_audit` 准备 `tight_dagger_matched_20261112_01`，旧owner/worker退出并发布后才接续，不热改。

已确认旧学生可超专家同历史目标可达集合。新显式 `expert_reachable_postprocessed_v1` 共同用于训练和部署；post-speed专家u标签不重复减速/delay，A标签必须来自实际专家执行后态。保留native物理/PD/资产、旧合法数据/normalizer/负结果，不能把新控制约束称为已证实成功收益。

新有限安排：先双组后端共同控制/幅度适配6有效tick遍（phase0不算DAgger轮），再最多2个完整配对轮次；各48配对机会、共享快照、DP6有效anchor遍→新DP缓存→rear12有效tick遍、固定10开发机会。共同Arm.2/Hand.9幅度、共同曝光endpoint配套；历史初始化仍不同，validation-best单独保存。两组未完整完成或实际曝光不匹配，不得单组进入更多轮。test22隔离。

旧owner退出及验收后，新admission开始24h绝对计时，CPU准备、等待和末10min发布均计入，不重设旧deadline或递归追加。普通service顺序推进；首2次正常采集内做实际新执行链验收。注册等待、CPU验证、真实GPU启动与闭环完成分别报告。只推origin/feature/tight-state-foundation，大数据/视频/权重留本地。

先读 [研究方向](research_docs/USER_RESEARCH_DIRECTION.md)、[合同](research_docs/CONTRACTS.md)、[交接](research_docs/SESSION_HANDOFF.md) 和 [实际结果](research_docs/STATE_DP_RESULTS.md)。当前授权与实现配置位于 `configs/tight_state.json` 的 `dagger`；历史结果不构成重启旧队列的指令。

## 当前覆盖 — 2026-11-12 零成功时仅追加一轮

用户明确授权：当前第2轮A/B配套新DP＋Transformer完成既定纯学生评估后，仅当两组合法、实际、完整的评估均为0完整成功，才用各组第2轮新candidate完整pair追加第3轮DAgger。不能因旧回退规则改用旧pair采集；缺后端、不配对、未完成或软件错误不能作为零成功触发。任一组有完整成功则不追加。

继续同一`tight_dagger_pool_20261111_01`，起点不变；触发后至多将原2026-11-12 07:24:31截止延长6小时到13:24:31，计算13:14:31停止。最多这一轮，无递归、无新run的18小时。第3轮A/B各最多96真实接管机会，按剩余时间与两组完整训练预留裁减；分别DP最多800更新、新DP缓存刷新、后端最多12有效tick遍，再按原4→10有限协议评估后退出。保留Core与全部合格历轮纠偏，原语义/mask/几何/固定组件/test22隔离不变。

本次只在隔离`bc+rl_RSS_dagger_continue`准备，当前B按原配置完成；普通后继在核验评估与自有worker退出后安全迁移，不热改`bc+rl_RSS_qg_report`运行源码。公开`conditional_extension_request/decision/extension/status/service`实际机器记录，区分授权请求、完整评估判定、进程启动与真实采集验收。计划或测试通过不等于追加已发生。零成功可能与恢复覆盖不足有关，仅为追加探索假说，尚未证明数据量是失败原因。

下列11月11日的仅A第3轮和原截止说明为被本次条件规则覆盖的历史方案，保留其当时证据。

## 历史覆盖 — 2026-11-11 完整策略 A/B 与恢复覆盖

本次 `complete_strategy_ab_v1` 在原 `tight_dagger_pool_20261111_01` 安全停止边界调整后续队列。沿用2026-11-11 13:24:31起点与2026-11-12 07:24:31绝对截止（07:14:31停止计算），不新建同义run、不重新获得18小时。旧第一轮数据、QG01完整pair及G11已持久化754次前端更新全部保留；G11退出活动矩阵。

只运行两个完整学生：A为G00模态的q_reached U-Net前端＋既定手部软几何＋Transformer；B为同G00模态的u_expert U-Net前端＋同容量Transformer，复用已训练G00/G01完整pair。两者均RGB-D＋robot178前端，触觉进入后端，H24/commit12与5/60/120Hz不变。内部目录QG01/G01仅为来源标识，不代表新增变体。裸Q/QG前端只做训练与离线验证，不独立采集或任务评估；不启动第三条完整队列或辅助G10新预算。

完整部署策略采集→成功专家有效后缀进持续池→不可变快照→DP增量训练→新DP全量依赖缓存刷新→Transformer增量训练→配套自主评估/回退。前端每次最多800、最少400更新后可按验证停止，每200更新验证，保留400/800共同曝光checkpoint；后端仍最多12/最少6有效tick遍。学生真实历史不监督，命令与配置标签不互换；后端监督组合后的最终u_expert。

第2轮A最多96、B最多48接管机会，随后共享同一快照分别训练两部分；第3轮仅更新后A最多96次新交互及两阶段训练，B保留其完整pair，不要求齐步。240为后续机会总上限，采集需扣除剩余完整A训练/必要阶段预留而缩减，不能保证全完成。恢复初48有效专家tick增采：前端纠偏anchor层目标一半来自初段，后端初段loss权重3；保留全部合法监督、原Core/历史/本轮.5/.2/.3层级，报告真实人口、曝光、重复与覆盖。不得用不合法标签填初段，也不得退化为只训练后端。

采集入口短验收必须记录实际前后端SHA、nominal/residual及issued/applied，并验证组合关系；后端缺失/不配对/加载失败显式报错。已有完整策略采集若核验正确，写“未发现裸DP采集问题”；不要求每tick残差非零。发布独立接管起点、实际阶段/偏离描述与恢复初段合法人口。test22隔离，固定4例自主评估按登记进展条件才扩10。

A保留原残差归一化幅度Arm0.2/Hand0.9，B保留全0.1；历史训练和本轮机会/曝光不同，不能将结果称为严格单变量或同预算对照。U-Net、DINO冻结范围、Transformer容量/历史、物理/成功标准不变，不新增BOL/PPO/可微物理损失。保留原400/Core/有效纠偏、初始化/当前/恢复模型和历史证据；只按依赖清理。普通owner在实测启动通过后独立运行，Codex不持续轮询。

以下各节是上一轮授权与完成前快照；冲突时以上述覆盖、当前CONTRACTS和最新用户指令为准，原始证据保留。

## 当前授权与隔离

用户已授权当前 `tight_transformer_20261110_02` 正常结束后，自动进行一轮成功筛选的专家接管式 DAgger：G11、QG01 各最多12次训练分布采集，聚合后先更新现有 U-Net 前端，再冻结新前端、刷新预测缓存并更新 Transformer 后端，最后各最多10次固定开发自主评估。最新更正覆盖附件最初“只更新后端、冻结前端”的要求。

旧实验继续使用 `bc+rl_RSS_palm_eval` 的冻结源码、数据、预算和模型；准备工作只在隔离 `bc+rl_RSS_dagger`。不得热改旧运行树、关闭源码哈希检查或抢先启动新 GPU/Kit。普通后继程序核验旧 owner/自有 worker 实际退出、相关选定 checkpoint 有效，再接续；单纯报告收尾失败不能冒充权重损坏。G11 复用既有 G10，QG01 复用上一轮 QG00；模型来源和 SHA 必须明确。

一轮新阶段上限12小时，包含其采集、探针、训练、缓存、评估与等待；使用实际新起点，不继承旧绝对截止。两组共享接管规则和机会，不能追加尝试凑成功。普通脚本负责后台推进；注册后短查实际状态，Codex 不轮询守候。脚本存在不等于已注册，注册不等于已启动 GPU。

## 数据、训练与评估

- 学生从训练初态自主执行，专家 shadow 随真实观察连续推进；接管不重置、不拼接 episode。原生专家动作经过已有映射与明确限速后送同一29D位置执行接口。
- 只有原生完整成功的接管 episode 可进纠正池。学生前缀与切换队列未清空段保留真实历史，监督为零；专家有效后缀才有标签。提前学生成功单独统计。
- 前端标签按语义区分 `u_expert` 与 `q_reached`；后者只允许实际专家执行后的配置，不能把学生 q_after 当反事实标签。H24 不跨无效段、控制权切换或 reset；尾部显式 mask。
- 保持 actual issued/applied、q/qd、触觉与时钟；`previous_nominal` 不替代执行反馈。持续记忆从完整真实前缀推进，监督 mask 不清空记忆。
- 原 Core train166 加新成功纠正；原 validation30 固定选模，test22 不用于本轮采集、调参或评估。原仿射统计与后端幅度界限冻结，不在聚合后重拟合。
- 前端最多20有效 anchor 等效遍历、10后允许验证停止；后端最多30有效监督 tick 遍历、10后允许验证停止。探针决定有限墙钟，完整恢复与正式更新分开计数；硬截止优先，不称低曝光为收敛。
- 新前端选定后，全部 train/validation 的名义预测与持续历史重新生成并绑定 DP、数据和统计 SHA；只有冻结 DINO 特征可逐 SHA 复用。
- 最终学生自主评估无专家动作替换，采用原生完整成功；接管成功不能计入自主成功，G11/QG01 两组比较不能隔离证明几何因果收益。

## 工程与保护

沿用 `native_data/narrow_data/state_history/anchor_sampling/anchor_continue/state_benchmark`；`tools.tight_finish` 是公共入口，`tools.tight_dagger` 只协调后继与阶段，不另造模型、采集器或训练器。H24/commit12、29D Arm7+Hand22绝对 rad、5/60/120Hz、原生物理/资产/PD/任务成功不变；特权几何只进入专家或明确的 loss context。

可充分使用空闲 GPU，保留6GiB显存、14GiB RAM、60GiB磁盘；不操作无关进程。只对自有进程实施退出与恢复。每次实质变更更新现有 CHANGELOG/合同/状态，不新增重复 MD，不把 CPU/mock 验收写成真实接管能力。

用户已授权按实际依赖清理旧代码、权重、缓存与报告。保护原400与仍用 Core、专家/资产、当前和已排队依赖、初始/选定/可恢复模型及必要比较分母；确认无引用后才删除。清理记录文件数、实际释放空间和依赖核验，不搬进归档冒充删除。

审查后只推 `origin/feature/tight-state-foundation`，不 force、不改共享历史、不推 upstream；旧 owner 先完成其发布，再整合新工作树。大轨迹、权重、视频留本地。必要历史已收敛到 STATE_DP_RESULTS 与既有小型结果记录，不沿用过时的“无自动下一阶段”限制。
