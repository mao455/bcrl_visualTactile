## 2026-11-13 18:45 B复验完成；仅补齐当前A固定20机会

`success_replication_01/finished.json`实际完成：B contact **0/20**、B previous **0/20**，两者均20合法/无软件失败；native lift分别8/6、第一子目标0/1。先前B1/10仍保留独立分母，不能声称稳定提升，也不能将额外B样本用于未匹配的A/B排名。旧owner已退出，不重跑已完成40次。

新登记`ab_replication_completion_01`只运行冻结contact A完整DP+Transformer的20机会，逐字复用B contact两份计划（seed27160000–19）；B previous仍仅作B版本诊断，不另给A previous预算。不训练、不采集、不改变模型/cache/执行器/原生成功标准、不按结果扩展或替换。A计划是在B结果之后冻结，明确是回顾性补齐，不称前瞻随机平衡执行顺序或全历史公平；原10例cohort不混入。

准备/完成两种contact训练公平审计均通过：每组DP514更新/765303 target槽，rear1362更新/193581合法tick。评估运行源码逐文件与已完成B相同；A/B实际checkpoint均绑定正确前端。29项CPU测试及真实prepare入口通过。新独立树与最长1h服务，`status/launch/finished/failure.json`交既有事件监督器；初始记录为PREPARED，实际启动以startup_acceptance为准。原计算23:55:15/次日00:05:15绝对截止不变。发布器保有两条live文件的独占写权，其他Git事务用共享锁。

## 2026-11-13 18:29 自动发布服务实际验收

独立CPU服务 `tight-live-publish-20261113-01.service` 已active，首份自动报告29afde1已核实到达远端；[自动进度页](TIGHT_LIVE_STATUS.md)与JSON由冻结普通脚本持续维护。每60秒检查、有变化通常5分钟内推送，阶段结束/失败优先；无变化不提交，断网保留待推内容，截止北京时间11月14日00:05:15。

当前B复验已完成31/40机会（旧B20、新B11），目前观察到0新成功，最后9例未完成；不能计算完整复验结论或改变原A0/10、B1/10。后续数值只看自动页时间戳。15项测试通过，300份活动worker源码无漂移；没有停止/重启训练或评估。下次科研Agent提示已登记实际服务、完整A/B检查与共享Git锁要求。

## 2026-11-13 自动进度发布与 A/B 增量公平检查

用户授权有新信息自动推 GitHub；今后 A/B 同数据、实际曝光、公共修复、优化器处理、选模与评估机会。新独立 CPU 发布器只维护 [进度页](TIGHT_LIVE_STATUS.md) 和小型 JSON，按实际 receipt/plan/result hash 报告，有变化通常五分钟内发布，完成/故障优先；普通脚本不做研究推理。原绝对截止 11月14日 00:05:15 UTC+8 不变。代码已实现/测试，真实服务启动验收另记。

[公平政策](AB_COMPARISON_POLICY.md) 与只读 gate 对当前 contact round 验证通过：DP 各514更新/765303目标槽，后端各1362更新/193581合法tick，coverage账本及相同10机会相符，A0/10、B1/10。两前端统计依法不同；真正后端 command affine 同为53ecc092…，cache绑定及完整pair匹配。历史权重/曝光不同，当前只能称本轮增量匹配，不能称全历史严格等预算；gate未独立重放全部采样顺序/优化器张量。当前 B 新旧版本20机会/版本复验单列，不与 A 排名，不合并原分母。

15项 CPU 测试通过（真实本地 bare Git 推送、并发远端前进、断网保留提交与恢复、无变化不提交、过期禁止 Git；监督语义/曝光/缓存/种子/缺失证据失败）。首次幂等测试发现snapshot被原地加时间戳，改为副本；补充中断写入恢复，并保留Git porcelain开头空格，防止误读路径。修复后全部通过。300份冻结活动源码 hash 未变；无训练/仿真启动或中断。现有Agent下次调用提示已原子追加公平要求和共享Git锁；不热改监督器。

## 2026-11-13 17:53 冻结复验真实启动验收

`success_replication_01`独立systemd服务active，owner PID2908611/start_ticks162322839；首个全新case已实际推进至tick400，不是仅注册/Kit启动。源码冻结73b2e97，hash无漂移；24项测试通过，GPU/RAM资源监测正常。子截止19:45:35 UTC+8，原计算23:55:15/次日00:05:15绝对截止未改。普通owner执行固定40机会后退出；既有监督器根据本轮decision继续监控，真实Agent再解释结果。当前尚无新复验成功率结论，先前B1/10保留原分母。

## 2026-11-13 17:48 contact周期已完成，登记冻结B复验

实际`adaptive_contact_recovery_01/finished.json`：A0/10、B1/10，全部合法/软件完成；B seed27150001原生tick274成功，非专家接管。两组前端514更新、后端1362更新，新cache完整。旧owner正常退出，不重跑已完成20例。loss/phase/历史覆盖根审计均已读，无需再宽审计。

下一阶段`success_replication_01`冻结比较B当前与上一完整pair，各20新开发seed27160000–19，40固定机会、两批交叉顺序，无训练/新纠偏/替换seed/扩展。单例成功不足宣布稳定收益，暂不为A0/10盲增训练。隔离树基于b431c3b，原监督器不改。24测试/真实prepare CLI通过；后续看`{launch,status,startup_acceptance,finished,failure}.json`与监督器decision，不以准备代替启动。最长2h且不晚于原23:55:15计算/次日00:05:15绝对截止。

## 2026-11-13 控制步与复杂度自查完成（活动训练未改）

复核当前A前端update300的真实sampler计数：193520合法tick全部可监督，各phase实际独立覆盖65.15%–65.38%，没有phase1–11整体丢失。旧21条各漏10tick的修复已实际启用，210中150已训练，另60仍可抽样；不是21整条轨迹遗漏。当前新后端未运行，不混入旧后端1191更新证据。

上一轮rear每组171133曝光仅112172独立tick；历史333条只见118条，按权重的一遍不是全数据遍历。当前496条纠偏接管phase分布0:21/10:475，恢复开始时机偏差仍存在。无新mask/时钟bug，不热改采样或增加实验。专家6条/3504tick真实状态、命令、q_reached误差0，接管不重置预算；matched恢复A42/47/B41/47，不能当学生成功。缓存2418tick/suffix批复用无重置相位证据，无新DDIM推理对照。

新增只读CPU phase审计，账本与checkpoint/完成凭据相符，297活动源码hash保持；test22、GPU、仿真和参数均未新增。复杂度主要在历史编排，待安全版本边界整理，不重写当前控制；当前contact-recovery继续至既定子截止19:09:53/绝对次日00:05:15。详细表格、保留的100更新快照和300更新证据见[DAGGER_METHOD](DAGGER_METHOD.md)页首及system_selfcheck证据目录。

## 2026-11-13 13:13 接触恢复已真实启动

`adaptive_contact_recovery_01`独立systemd服务已active，owner PID2764734/start_ticks160645694，冻结源码c9f8e5e、SHA无漂移。A已完成2个登记机会：首个原生初态碰撞拒绝（不补样），第二个完整专家接管成功、已通过合法纠偏准入；这不是学生自主成功。后续两模块训练/新cache/完整自主开发尚未完成。44项测试与24个实际CLI采集配置解析通过，资源满足原6/14/60GiB余量。

独立子截止19:09:53 UTC+8，比原计算23:55:15更早；原绝对11月14日00:05:15不变，正常完成或故障后由既有真实Agent监督器再决策。现场、启动验收与最后状态位于原run同名目录；未改监督器和任何旧worker。后续下面的“仅准备”段落为启动前记录。

## 2026-11-13 13:12 完整tick-window结果与接触恢复新登记

`adaptive_tick_window_01`已完成：A/B各1191真实更新、171133合法监督tick；完整DP+Transformer开发各10次，成功均0，lift分别3/4、首子目标0/0、软件失败0。训练wall分别365.41/358.82秒，相比centered的566.30/559.26秒减少约36%；评估另计，不能称自主收益。选模分数相对各自update0为0.397905/0.596738。

20条真实自主trace首次contact为53–80tick、首次contact-loss为58–127tick；这是可见事件，不证明失败根因。继续重复后端曝光信息量低，故真实Agent登记`adaptive_contact_recovery_01`：用本次实际完整pair，每组12训练家族机会，在58/70/82tick接管，维持原全局commit12/清洁后缀相位。只收原生完整成功的真实专家后缀；2个预先指定机会/组保留correction_dev，不补seed。

新隔离树从已发布58ed6e1建立，下一前端显式`recovery_onset_commit12_v1`；共同快照上DP2 anchor等效遍→各新DP完整依赖缓存→centered/FY reg0/LR1e-4/current-weight prefix/fixed128tick rear1遍→各10完整开发。现有全池530条合法纠偏全部保留，比上一训练快照更多，数据/前后端/窗口均变化，不是单变量因果对照。两模块与新cache不可省略，原A=q_reached＋既定几何、B=u_expert、rear最终u_expert不变。

新阶段已经CPU准备、实际CLI解析和44项测试通过，但本条写入时尚未启动GPU；后续以本地`adaptive_contact_recovery_01/{launch,status,startup_acceptance,finished,failure}.json`为准。子预算最长6h且受原计算11月13日23:55:15、绝对次日00:05:15约束；独立systemd cgroup、无新监督器、无自动下一轮。原adaptive_recovery有意交接InterruptedError与更早CLI故障仍保留，不代表本次评估失败或研究终止。test22与无关任务不触碰。

## 2026-11-13 接管相位优先诊断完成，当前worker继续

409 train纠偏中388条（含当前全部76）合法起点已对齐DP；21条旧phase0接管各漏10个早期合法DP target，共210。它们仍进入历史/后端；实际checkpoint采样与真实EMA/batch逐tensor验证。隔离实现 `recovery_onset_commit12_v1`，补原anchor的合法后缀、保留前导mask和原12tick时钟，53项CPU通过。尚无新参数更新，不声称成功率提升。

下一次新的视觉DP前端continuation在frontend配置登记 `target_window_contract: recovery_onset_commit12_v1`，经prepare_frontend传至training/session和checkpoint配置；只允许fresh continuation，旧同run恢复不得切换。更新DP后必须重建其后端缓存再训练。当前tick-window完整闭环继续，其292份冻结源码未变。监督器prompt/findings已登记本诊断，原截止保持。

两组当前后端各1191更新/171133合法tick；本次1有效曝光实际见到194/409纠偏（118历史＋76当前），不是整池逐episode一遍。缓存抽查2418tick、原HDF5及真实suffix batch无相位/历史错配；全DDIM重算未做。详情与样本表见DAGGER_METHOD页首，证据 `takeover_phase_review`；当前实验结果仍读真实status/finished，不把本次窗口修复当已训练或已部署。

## 2026-11-13 12:24 Agent002后续：tick-window试验准备

**已完成启动核验**：owner PID2735439/start_ticks160362062，独立service/cgroup、RuntimeMax约1h58m，子阶段均早于原计算截止。A实际日志116更新，已持久化111更新/15723合法tick；新root `launch.json/status.json/active_stage.json/startup_acceptance.json`是真实凭据，未来以`finished.json`或`failure.json`交回现有supervisor。agent_002/decision.json登记`tick_window_01_full_policy`和实际owner身份，不沿用centered已完成事件。99项CPU回归通过，292个冻结源码hash无漂移；未改活动源码或supervisor，尚不报告B完成/新成功率。

已读centered的两组finished/evaluation_complete及12:16独立findings：合法自主均0/10，lift3/5、第一子目标0/0，无软件失败；不重复旧cohort。选择独立`bc+rl_RSS_tick_window_01`/原run下`adaptive_tick_window_01`，从同原round01初始化、同DP/cache进行1有效tick遍的固定128tick梯度窗口对照，随后离线与各10完整学生开发。FY、centered、LR、数据不变；优化器更新数和loss归一化共同变化。复用review commit的padding/诊断修复，不热改supervisor或旧树。启动前须真实CLI/回归与资源检查；launch/progress才证明GPU进展，普通脚本只执行这项登记、结束交还Agent，不自动DAgger。截止保持11月13日23:55:15/次日00:05:15 UTC+8。

## 2026-11-13 12:16 损失与效率复查：耗时归因及两处修复

本次沿真实`TargetResidual→compose_chain→objective_components→residual_train`复核，没有发现新的确定性反向梯度错误。FY的理想box梯度为2(P(x)−y)，错误裁剪区保持方向，合法端点等价提案不要求裁剪前数值相等；实际执行器的1e-6rad容差与理想box不是严格逐点等价。centered线性输出有有限逆解，但action_scale²、共享参数和32tick截断仍影响学习，不保证闭环成功。无当前/未来专家标签作为输入；合法mask、完整历史、source/cache配对及同轮optimizer/RNG恢复链核对合理。

最新centered两组各2349更新/171133监督tick，实际训练阶段A566.30秒、B559.26秒，合计18.76分钟；两组离线约1.22分钟，20次完整仿真约25分钟。资源等待各约1.5秒，梯度裁剪发生率均0。上一方案两组各7017更新，训练合计54.41分钟；原03:04 CLI故障到09:57恢复相隔6.879小时，不能解释为模型需要训练这么久，也不能说期间整张GPU空闲（旧matched任务仍在运行一部分时间）。真实Agent事件接续现已调用两次，原截止不变。

训练仍有实测效率与目标权重问题：实际一遍291014分配tick中214711有效，另重放2237985有效历史tick（约为单次历史遍历10.42倍）；这是保持当前权重历史的一项高成本选择，state_only已经跳过decoder，不能直接缓存旧隐状态。2349个更新的监督数为1–128，41个不超过8；per-chunk权重和归一化使50/20/30物理tick曝光不等于梯度贡献。恢复初48占物理tick8.16%，若全局统一加权为21.05%，实际每更新归一化的平均系数为11.41%；这些是loss系数而非Adam参数贡献。仅1个全初段chunk完全抵消3倍系数，不夸大为主因。下一科研决策优先考虑固定有效监督预算的累积与减少重复重放，同时显式登记优化时间尺度；不自动增加epoch、不把稀疏chunk问题说成既有合同实现错误。

已在隔离树修复：`native_data.residual_previous_targets`只给padding槽合法关节占位，防止补零触发执行边界错误；真实非法previous仍拒绝。`state_benchmark.residual_episode_metrics`在linear/linear-tail输出下把旧|delta|/scales归为历史幅度比例，不再误报tanh饱和。原trace和旧报告保留，不改变控制、loss或已有权重；当前冻结worker未热改。前者为可复现潜在批处理bug，当前已完成训练未触发，不能当作这次零成功的解释。

修复后28项定向测试通过（2.37秒）；另复核55项已有loss/history/training测试通过（8.58秒，两批有重叠，不相加）。独立只读Agent完成公式与最小反例交叉审查，其pytest因只读临时目录限制未完成；实际通过数来自主Agent。旧286/新290份冻结源码SHA全保持。无新增GPU/Kit实验。

最新完整策略结果A0/10、B0/10，lift3/5，均无第一子目标，软件失败0；离线改善未变为任务成功。真实Agent002已在分析该边界及本次发现，下一实验由它结合偏离证据决定。更新后的四例Core选择集不能代替完整30例离线结果，更不能代替闭环。证据见[训练与效率复查](evidence/tight_execution_learning_20261113/training_efficiency_review/findings.json)。

## 2026-11-13 11:25 真实研究交接（覆盖下方旧owner自动FAILED汇总）

旧恢复实验已合法完成：A/B各7017次更新、各10完整自主开发，成功0/10与0/10，lift2/1、alignment0/0、软件失败0。两组训练endpoint均未获采用。监督在最后评估子进程退出后有意终止协调owner；`InterruptedError`不构成新训练/评估失败。完整机器回执在原run的`agent_supervisor/initial_boundary.json`，不重复评估。

真实Agent选择唯一有界`adaptive_centered_01`：投影名义锚点＋可达区间宽度缩放的线性提案，FY保留错误饱和区梯度；同原round01权重/cache，1有效tick遍后各10完整学生开发，不自动DAgger。方法/可证伪限制见DAGGER_METHOD 4.2。11:29启动验收已确认A真实256更新/18057监督tick，独立service的owner为2699731、start_ticks160026200，训练源码`d12852d`；104项测试通过，旧286/新290份源码均无漂移。新完整策略结果尚未产生。实际状态以新root的`launch.json/status.json`和训练`progress.json`为准；任务结束再次交由现有supervisor调用真实Agent。原deadline、资源储备、test22隔离均未变。旧FAILED字段只保留历史现场，不是当前科研终局。[启动凭据](evidence/tight_execution_learning_20261113/adaptive_centered_01/startup_acceptance.json)。

<!-- EXECUTION_LEARNING_STATUS_START -->
## 2026-11-13 自主执行学习修复：实际状态

状态 **FAILED / technical_or_budget_stop**；原绝对截止 **2026-11-14 00:05:15 UTC+8** 不变。
首个旧/FY各200更新已完成，FY的A/B dev执行误差比1.139/1.181，未通过原门槛。固定DP输出域下界占A归一化错误33.46%，B为0；据此比较保持共同执行器的线性尾部提案。
配对延长、必要优化调整、完整混合数据后端训练、完整DP+Transformer开发评估分层记录；若闭环失败且剩余预算允许，最多两轮、每轮12机会/组的定向DAgger同时更新DP、缓存和后端。初始门槛失败不结束整个修复，也不改写旧结果。
GPU工作与Git发布分别调度；未完成阶段不能称为成功。test22隔离、源数据和旧checkpoint保留。
[方法与限制](DAGGER_METHOD.md#execution-learning-repair) · [阶段状态](evidence/tight_execution_learning_20261113/adaptive_recovery_01/follow_status.json) · [配对计划](evidence/tight_execution_learning_20261113/adaptive_recovery_01/adaptive_probe_plan.json) · [表达下界](evidence/tight_execution_learning_20261113/adaptive_recovery_01/capacity.json)。
<!-- EXECUTION_LEARNING_STATUS_END -->

<!-- AUTONOMOUS_RESEARCH_SUPERVISION_START -->
**2026-11-13 10:44 最新接续：独立研究事件监督已实际启动。** `tight-research-agent-supervisor-20261113-01.service` 为active，真实Codex工具调用与service运行环境均已验证，286份旧worker源码未改。当前A已完成7017更新但验证退化；B仍正常更新，最新启动快照为4512次/328737监督tick。先完成本次两组完整学生评估，再触发真实Agent做下一研究决策。下面旧owner的自动汇总只描述它自身；有意交接的InterruptedError不代表评估软件失败，也不意味着研究结束。15min无实际进展或异常退出同样触发诊断；普通脚本不宣称具备科研推理能力。原计算23:55:15/绝对次日00:05:15 UTC+8不延长。

[实际启动验收](evidence/tight_execution_learning_20261113/agent_supervisor/startup_acceptance.json) · 本地实时 `outputs/visuotactile/run/tight_execution_learning_20261113_01/agent_supervisor/{status,latest_decision}.json`。后者仅在真实决策发生后存在。
<!-- AUTONOMOUS_RESEARCH_SUPERVISION_END -->

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
