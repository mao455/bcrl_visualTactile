## 2026-11-11 — 完整策略 A/B 范围、恢复覆盖与原run接续

- 真实执行：源码`c80829c`已推送，原run安全修订并于18:23恢复同一service；18:26:55新A首2次实际采集通过完整策略数值链验收，264个冻结文件和原起止时间核对通过。B实际采集门控、新两阶段训练/自主结果仍待后台执行；本次没有新增run、额外诊断、旧数据删除。启动/64次历史采集审计/测试/修订身份已在同一公开目录补齐，监控重启缺口如实保留。
- 最新用户明确仅两个完整策略，DP更新继续保留。A复用本run首轮QG01完整pair，以q_reached和既定几何学习；B复用已训练G00/G01完整pair，以u_expert学习；两者前端均G00模态，触觉在同容量Transformer后端读取。保留A Arm0.2/Hand0.9与B全0.1已训练幅度和不等历史曝光，不能声称严格单变量对照。裸Q/QG只训练/离线验证，其他变体不再获得新采集/闭环预算。
- 原owner已按STOP安全退出：G11持久化754前端更新、选中201更新；旧QG首轮完整结果和80条纠偏原样保留。修订沿用同一run、原13:24:31起点、次日07:24:31绝对截止及07:14:31计算截止，不重跑首轮、不重获18小时。第2轮A96/B48、第3轮仅A96均是机会天花板，按实测吞吐和后续完整pair预留缩减；DP增量每次最多800/400后可停、每200验证，后端维持12/6tick遍。
- `tools/tight_diagnostics.py`原位适配scope/role与每轮实际活动组，保留历史round_01全部实际job（包括退休G11），不伪造未调度B第3轮的评估义务。范围完成判定只检查新scope已调度的必需完整训练/评估凭据，仍拒绝缺失新阶段证据；历史STOP/partial保持原状，不伪装为完成。已有`complete_policy_acceptance`/`recovery_coverage`小型JSON/CSV按root与job导出，发布器不反序列化模型或新增推理。
- 原位更新AGENTS、合同、交接、研究入口/方向，明确完整部署采集验收、学生/过渡mask、两个标签语义、新DP依赖缓存刷新、配套恢复、原Core/历轮纠偏累积及初段恢复48tick采样/加权统计。几何公式、容差、既定权重与手部计算范围保持。阶段/偏离覆盖是描述统计，不是根因证明。
- 最终联合定向验收114项通过（13.83s），覆盖数据语义、连续合法恢复初段、分层抽样与恢复、两阶段队列、实际部署数值审计和发布；G01新导入/旧池资格兼容追加复核16项通过。仅既有requests依赖warning。`narrow_data`显式支持新B命令消费者，旧manifest缺G01键时只继承已有G11资格及拒绝理由，不改历史条目；修订迁移测试确认起点/截止不变、旧元数据归档、源冻结更新且重复迁移拒绝。
- 本次发布适配定向测试：`tests/test_dagger_publication.py` 6项通过，含退休754更新保留、未调度B不算失败、恢复覆盖导出，以及缺失新scope评估凭据仍不能宣称完成。实际collector/训练器修改、它们的联合验收和新启动凭据由本次统一交付记录；此处不把文档或CPU测试称作新真实采集/训练已完成。本条发布适配与文档编辑未新增GPU/Kit工作或删除历史产物。

修订前首轮启动验收（历史）：113 项 DAgger/数据/控制/恢复/发布测试通过；实际旧21条双组聚合 CPU 读回、真实已选权重前向/配对验收通过。无接触硬门控未发现。启动前已实际删除8个已完成 disposable probe 权重，释放4,697,235,456字节，当前4个选定权重SHA未变。

## 2026-11-11 17:01 — 发布运行中的DAgger阶段结果

- 用户要求先提交当前情况。复用 `tools.tight_diagnostics dagger_publish` 做CPU导出，更新 `STATE_DP_RESULTS`、`SESSION_HANDOFF`、研究入口和方向，保留13:27启动快照与旧实验全部分母；公开目录仍为 `evidence/tight_dagger_pool_20261111_01`，没有复制训练管线或改动冻结运行源码/配置。
- QG01第一轮前端804更新/8.02871有效anchor遍、后端12212更新/12.00040有效tick遍；同新前端SHA的后端缓存与完整pair已选定。自主10例0成功、2lift、0子目标，全部601tick超时；采集QG27/32、G11 32/32专家接管成功，新增59条，池合计73train/7dev，另Core166。G11仍前端训练；快照109次更新，后端及本轮自主结果未测。
- 发布逐更新loss、完整固定验证数值、训练/误差曲线、实际分层曝光、几何逐配置数值、接触阶段离线误差、10例终止与接触诊断、权重/视频本地SHA索引；不把接管成功或lift当作完整成功，也不把训练预测配置几何当作自主物理穿透。运行中读取按文件保留时间，资源汇总截至root状态16:50:02，G11进度截至17:01:29。
- 本次检查数值/分母/选定SHA、公开文件清单与SHA、报告链接和264个冻结文件；无新增模型前向、Kit、训练、评估或清理。既有后台继续推进，原截止不变；中期commit推送后更新owner的公开报告HEAD记录，使最终自动发布能够接续。

## 2026-11-11 — 多轮共享纠偏池与分层增量训练

- 最新授权覆盖旧单轮12h/12机会限制：最多3轮、总18h（含wait、最后10min发布），QG01优先、G11随后，从已完成DAgger的完整pair继续；每轮默认每组32机会、4个预登记纠偏dev，最多192总机会，固定自主4例有阶段进展才扩10。
- `narrow_data/native_data`在现有loader增加持续去重池/共同snapshot/整episode dev与来源字段，实际旧21条共享为历史train；不重拟合原统计、不打开test22。原21条各10个干净tick早于首合法DP anchor（总210），保留旧结果；新采集switch=12k−2以对齐原delay清空后的首合法anchor。
- `anchor_sampling/anchor_continue`新增Core/历史/本轮.5/.2/.3分层抽样、实际覆盖/重复/有效曝光；DP最多10/最低6遍、后端最多12/最低6遍，保持正式LR日程的短探针。每阶段真实评估update0，并将旧val/纠偏dev/首24有效tick各自相对update0后按.5/.25/.25选模；后端绑定选定前端SHA，完整pair回退由owner负责。
- `state_image_offline`诊断遵循与训练一致的supervision/segment mask，完整观察前缀仍推进；残差offline新增纠偏dev/最初24个独立有效tick表，并复用已有预测按precontact/initialcontact/sustained/lostcontact导出修正、已有触觉置零差异和有效专家命令误差；不增加forward，raw饱和未知明确留空。现有Transformer无失触立即清零，未引入新门控、网络、幅度调参或物理变更。
- 同round恢复保留optimizer/EMA/RNG/采样decks/cursor；STOP保存可恢复状态而不写完成receipt，完成阶段不重训。崩溃未提交loss移入`*.uncommitted.jsonl`保留，正式计数按last。`tools.tight_finish dagger_resume --root <root>`复用原deadline，普通owner负责有限多轮推进，模型不轮询。
- 42项训练/采样/数据/离线CPU定向测试通过，覆盖稀少纠偏抽样、实际小型optimizer、update0胜出、dev隔离、分层完整组中断恢复逐tensor/Adam/RNG等价及正式LR探针；仍有既有requests版本warning。该验收不代表新GPU启动、真实恢复能力或自主效果，实际新run状态将由机器凭据另报。
- 原位更新AGENTS/CONTRACTS/SESSION_HANDOFF/研究及根README；AGENTS保留历史覆盖记录。STATE_DP_RESULTS旧结果由独立只读发布整理，本文不改写旧结果。以上为代码准备阶段；实际队列启动另按机器凭据记录。

## 2026-11-11 — 补齐已完成 QG 训练与逐次失败证据

- 用户指出 QG 发布不够完整；原位扩展 `tools.tight_diagnostics qg_publish`，只在隔离报告工作树读取已完成 `_02` 产物。保留旧总表，增加 QG00/QG01 专用训练/完整验证曲线、几何距离与有效梯度、离线误差及代表轨迹、20例事件和逐tick压缩数值、视频路径/SHA。源轨迹/模型/视频本体仍留本机。
- `CONTRACTS` 补精确 x0/掌部/几何软惩罚/后端命令目标与选择分数，说明 m/rad 与归一化、1mm/2mm容差、全update两配置、100+400渐入、0.2单次输出梯度校准和手输出梯度范围；保留旧投影与当前直接软惩罚的区别。`STATE_DP_RESULTS` 新增 QG 专项索引和失败过程，DAgger另外报告。
- 旧 QG00/QG01 均0/10，20例全部实际接触但无原生lift/子目标/alignment；QG00全部601tick超时，QG01为9次超时与006在569tick fall。按原规则从日志重算 QG00缺失contact汇总，稳定抓持0/10；QG01重算与原结果一致。不引用缺approach_proxy时的错误阶段概括，不把阶段观察称为已证实根因。
- 几何真实计算13800配置、4112有效手部梯度，SLSQP调用0，几何计时183.868s；validation预测配置492/868超1mm、最大16.795mm、要求接触保持455/1186。它们是专家历史下配置几何，并非自主物理穿透；不隐藏不利指标。
- 实际验证：60个公开文件/16.58MiB，39个CSV共288,710行；SHA/大小、全部数值有限性、20例事件/11,988连续tick、result/summary/trace/HDF交叉检查通过。3段视频601帧/60fps索引保留；完成状态与自有worker退出拒绝门、无torch/Kit导入、语法检查通过，38个文档相对链接有效。
- 本次只作CPU数值导出和一致性检查，没有新GPU/Kit/推理/训练/评估/清理。运行中的DAgger树、冻结源码、HEAD及远端在其收尾前保持原状态；补充报告待既有owner发布退出后以普通CPU进程合并并推送同一指定分支，保留两个阶段各自来源。

## 2026-11-11 — 成功接管聚合与前后端增量训练准备

- 最新用户更正覆盖附件最初后端限定：G11/G10命令前端与QG01/QG00配置几何前端分别先学习聚合数据，再冻结新前端、刷新全部train/validation预测并更新原Transformer后端。原 `_02` 继续用冻结源码运行，准备只在 `bc+rl_RSS_dagger`。
- `narrow_data/native_data/anchor_sampling/state_history` 原位加入共享聚合、u/q语义、监督mask与控制segment、有效anchor人口、同语义EMA续训和DINO复用。真实student prefix继续推进历史，loss为0；H24遇无效/切换/尾部截断；actual issued/applied与previous_nominal保持独立。
- `state_benchmark/position_control` 增加opt-in同episode专家接管与shadow来源记录，复用现有trial、recording/compact及原生映射；不新增任务、物理或成功标准。`anchor_continue` 复用前后端训练与完整恢复、固定normalizer/幅度界限，分开probe和正式曝光。
- `tools.tight_finish`保留公共入口，新增职责单一的`tools.tight_dagger`协调后继与有限阶段；配置在原`tight_state.json.dagger`。首轮各12采集/10自主开发，新起点总12h；前端最多20有效anchor遍、后端30有效tick遍，10后可验证停止，硬墙钟优先；无仅后端fallback、补seed或test22。
- 实际CPU：完整196条train/validation聚合读回，train166共55,392tick、4,690anchor；统计文件字节一致，test源读取0。新增7项数据测试通过，覆盖实际u/q、H24/历史mask、真实旧G10 EMA续训、无GPU DINO复用、重预测前缀与四SHA；相关旧残差/anchor/历史/归一化回归通过。最终联合88项通过（40.00s）；新增后继等待保护随9项定向回归通过（0.81s），去重共89项。保留已有requests版本告警。
- 对照官方Diffusion Policy workspace、robomimic SequenceDataset和imitation collector/trainer职责，保留单一模型/数据/优化实现，仅增加共享接管协调。当前合同内有源码链接与模块/语义表；AGENTS、研究入口、方向、感知、交接与根README去除过时的State-only/Shadow-only/前端零更新/无自动下一阶段声明。
- 普通后继 `tight-dagger-20261111-01.service` 已于01:54:40+08:00登记，PID1021762，启动核验active/WAITING；原队列实际结束并发布后自动接续。已核验旧运行树253个冻结文件全无变化。真实接管、DAgger正式更新、后测仍未开始，不把排队称作GPU成果。实际旧产物删除另按CLEANUP_AUDIT与清理凭据统计，本条不把文件改写当作空间回收。

以下是按日期保留的历史变更，不重新授权其旧实验或恢复过期预算；当前规则以CONTRACTS和最新用户指令为准。

## 2026-11-10 — 更正前端复用与独立配置适配

- 保留native_dp U-Net及已有G00/G10；state_history新增显式旧G00 EMA→q_reached热启动，四统计buffer覆盖且新optimizer/schedule，不改变旧checkpoint身份。
- state_image_loss直接signed-surface soft hinge，复用HandCollisionGeometry距离/Jacobian，1mm/2mm不变，手梯度/接触阶段mask，内力仍未启用；anchor_continue/anchor_sampling完整update分层2配置、独立warmup/ramp、真实关闭计算和固定LR日程探针。
- tight_finish原位改独立依赖队列与旧checkpoint/data/DINO引用；不再重训G前端或因QG异常全局阻断，取消旧200曝光硬准入和60/144小时上限；有限预算将按实际测量封存。
- 真实GPU两Q各8个可丢弃更新及完整恢复通过；同配置几何active更新耗时比1.01034，几何自身0.023625s/update。独立一次完整64anchor校准按20%手输出梯度冻结λ=2.355370821894758e-06；不是参数梯度或未来所有批次上界。正式训练/新闭环须看机器进度，不能把探针算入正式曝光。
- 实删两无依赖随机初始化副本和残留link，释放555.359375MiB；195tensor精确重建与关键保留SHA检查通过。旧400/Core/selected模型、失败及成功证据保留。
- 101项相关CPU测试通过（仅已有requests版本warning），覆盖热启动、loss/梯度/完整update、残差停止、独立DAG和发布；异常Kit锁先清理再解锁，owner按PID/start_ticks核实自有worker退出。
- 后台独立有限依赖、单Kit互斥、实际并行/串行吞吐择优、完整退出和CPU自动小结果提交；绝对停止2026-11-11 18:54:40+08:00，无test22/自动下一轮。

## 2026-11-10 原位匹配方法重构（训练就绪以机器门控为准）

- native_dp/native_controller：旧489125参数GRU→756581参数Transformer，H24计划/游标/mask＋独立previous_nominal＋严格过去执行反馈；persistent GRU+past32；旧成功checkpoint仍只通过明确Legacy loader。U-Net未替换。新残差独立绑定共同u_expert train166统计，Q前端保持q统计；原先直接继承Q affine会使某手关节后端loss权重约减半，此混杂在训练前修正。
- narrow_data/native_data/position_control/state_history/anchor_sampling：直接after-q标签及terminal/time检查，v5 u/q监督与各自train166统计；clean sample参与加噪，u_expert仍用于执行反馈和残差命令监督。196条/65630tick通过，test源读取0。
- kinematics/state_image_loss：真实网格FCL距离/解析Jacobian、局部stop-grad手形投影与独立loss context；固定6train/1mm-2mm-.15rad-30iter，不可行原因记录、手臂额外梯度0。offline通过不等live cooked通过；当前内力不启用。
- precise_assembly_env：仅删除两处kinematic body恒零速度写；reset_query_probe零step定向API验证、全日志错误计数。原162错误及工程失败不覆盖。
- anchor_continue/tight_finish：沿用主训练/缓存/控制与普通owner，增加配置矩阵/全tick残差/固定匹配曝光/真实吞吐预算/冻结后test22；删去旧GRU专用重复MD报告器，统一机器报告。冻结源码后不改运行代码。
- state_benchmark：只读规划/DDIM/残差CPU-CUDA耗时、幅度/饱和/裁剪/失触区间；无新GPU同步等待，不把再接触当恢复能力。
- 实际清理392旧nominal预测cache＋6重复图，释放863805440字节；所有权重/400/Core/旧分母/首次成功视频保留；详见CLEANUP_AUDIT及机器明细。
- 当前102项定向CPU测试通过，覆盖模型因果/历史/完整标签/零step reset/正式plan冻结/owner失败停止/几何梯度；仅requests依赖版本告警。新增同训练器residual-probe保证正式残差之前测吞吐封存预算。实际GPU、训练量和闭环结果随后由机器报告补入，不把实现或授权算完成。

## 2026-11-10 — 发布G01/G11完整结果与数值证据

- c3386b6队列12:01正常结束；两组98610更新/200接触tick等效遍历、validation平台停止，选epoch110/180。开发G01 0/10、G11 1/10，均无拒绝/中断，test22未用；完整成功仅原生max_successes，不以阶段替代。
- G11成功seed27150000、279tick/4.65s，已在预登记首段视频完整记录；native_lift=false如实保留。4视频全部CPU解码核对，权重、源轨迹和MP4本体不上传。
- 原位扩展tools.tight_diagnostics publish适配残差schema；验证两组各38离线身份、20例预登记/分母/选定SHA/完成凭据/帧数，公开逐更新loss和逐tick误差压缩数值、阶段与hold参照、敏感性、9图、配置、源码与产物hash。只有CPU文件读取/绘图/解码，不加载policy、启动Kit或改原产物。
- 原位更新当前结果/README/合同/AGENTS，保留全部G00/G10和StateDP历史。Hand小幅改善、Arm几乎不变，G01阶段退步和G11有限成功均披露；force置零响应不宣称闭环因果收益。
- 正式窗口6374采样、最大间隔1.1382s；整卡均值79.17%、min VRAM19541MiB/RAM25.696GiB。包含验收重启间隙的继承汇总仍有807.409s缺口；退出前4条PhysX错误未专项审计，不把exit0当无错误证明。
- 验证实际导出、压缩数值行数/哈希/公开链接及无大文件；无新训练、评估、补录、清理或下一阶段。

## 2026-11-10 — G01/G11冻结基础60Hz触觉残差

- 核对4164f83实际G00/G10 checkpoint、合同与既有20例闭环/2视频；两组都是视觉I-HistoryDP，G10额外触觉条件；G01/G11是尚未训练的独立残差，不重复基础训练。训练source98dfafe、原闭环sourcea82c33f和原FAILED记录保留。
- 原位扩展models/native_dp、native_data、native_controller、anchor_continue、state_benchmark、tight_finish、state_resources和configs/tight_state.json；复用3e11d62的60Hz GRU触觉残差/历史loss/AdamW，只适配当前冻结512D视觉读出、H24绝对rad目标与原归一化，不采用10Hz Transformer。不加新算法、FK损失、恢复机制或物理变更。
- 完整冻结EMA真实DDIM预测为nominal，训练/部署共用剩余计划与输入构造；过去专家反馈明确teacher forcing，闭环base反馈实际混合命令。数据166/30/22与原DINO缓存复用；不使用test，不采集，不删除原证据。
- 两组真实109tick部署前缀：nominal及robot完全一致，组合最大差G01 2.98e-8/G11 3.73e-9rad；分段连续历史差1.19e-7rad。两次可丢弃更新使所有残差分支梯度非零，完整基础参数/缓存/权重SHA不变，零残差精确恒等，保存/加载预测精确一致。另实际trainer每组10个更新/1条train与1条validation，均有限；这些不计正式训练。
- 实际trainer/最终离线入口均通过两条小样本端到端验收，原阶段CSV、FK、目标图与触觉置零诊断可落盘；16项现有动作/路由回归通过。旧基础稳定抓持按原诊断补测为G00 4/10、G10 3/10，不是完整成功。
- 沿用普通owner和锁、源码冻结、独立日志/状态、连续资源记录、并行/串行吞吐比较。9h训练上限/300接触tick等效遍历、200后平台、固定validation4选模；train8/val30最终离线及最多10×2开发闭环、2视频/组。原50%节流由当前充分用GPU授权覆盖，6GiB/14GiB/60GiB储备保持。不停止无关进程。
- 修复只作用新残差评估的Kit退出完成凭据；原历史20例和FAILED不改。后台自动原位更新当前结果，依赖失败停止、零成功不扩样、不自动下一轮；当前不声称正式训练完成或成功率改善。

## 2026-11-10 — 发布G00/G10测量、误差图与完整失败证据

- 用户要求提交GitHub实验资料，沿用先前大文件本地范围。G00/G10各38条最终离线、各10条合法自主开发闭环，均0/10完整成功；G00 9timeout/1fall，G10 7timeout/3fall，原生lift1/3、首子目标3/1，2视频523/399帧完整验收。
- training源98dfafe，evaluation源a82c33f。原q/qd调用错误在首次控制前失败；接续20例均完整落盘，但exit_completed_kit_worker要求root/result.json顶层software_complete，队列却使用evaluation_complete.json及另一报告schema，故退出1。原FAILED、traceback、20例成功/失败分母及旧证据不改写，未重跑或静默修正收尾合同。
- 原位扩展tools.tight_diagnostics publish复用已有CPU曲线导出，视觉模式核对每例完成凭据、预登记身份、选定权重SHA、最终诊断身份和解码帧数。发布13图、CSV/JSON、逐tick/逐update及代表目标压缩数据、218数据身份、配置、源码/原产物/公开文件SHA。仅已记录指标，不加载policy/Kit；旧StateDP导出继续要求其既有COMPLETE准入。
- STATE_DP_RESULTS/README/CONTRACTS/AGENTS更新为真实完成边界；明确测量完成与收尾失败、不等曝光119.541/81.753、专家历史误差与自主行为、稀少阶段样本，以及训练537.81s遥测缺口。没有触觉因果收益/收敛/可靠抓持或G01/G11就绪声明。
- 本次仅CPU导出和产物一致性验收，不修改模型、训练、采样、控制、物理、原run状态或数据，不新增清理；所有checkpoint/HDF5/MP4仍被Git忽略。公开目标仅origin feature/tight-state-foundation。

## 2026-11-10 — 修复G00/G10闭环实测关节传递，授权一次评估接续

- 98dfafe正式G00/G10训练10132/6929更新，固定墙钟停止；各train8/validation30离线完成。原评估01:09:17 +08:00失败：state_benchmark旧I-*分支省略diagnostics，新canonical输入在首次act读取q/qd时收到None；无控制transition/完整trial，不得报0/10。
- state_benchmark原位增加实际调用路由，只给新G00/G10传当前joint_position_rad/joint_velocity_rad_s；不引入其他诊断真值，旧无modality图像/State调用不变。模型、训练、EMA、normalizer、采样器、历史和物理全未改。
- tools.tight_finish复用原owner/监控/清理/报告增加evaluation_only入口，原锁+新锁、旧产物SHA校验和源码冻结；独立evaluation_resume_01只读链接原training/offline/state_data，保留原失败目录及partial视频。相同10×2身份/各首1视频，原02:08:51截止不重置，不重训、不重新离线、不自动再试。
- tests/test_state_image_kit_gate通过真实trial调用点的CPU替身检查G00/G10精确q/qd及拒绝额外字段，旧图像/State路由兼容；验证接续只启动monitor/evaluate并拒绝过期/重置期限。连同既有动作时序、损失测试共36通过。首次测试中两条期限断言写错异常类型，按既有require抛ValueError修正，阈值不变。尚待真实Kit接续验收。
- 当前结果修正“已有完整离线但缺闭环”的措辞，自动报告训练/评估分别的源码身份及完整解码视频路径；启动后仅短监督，由普通程序自然收尾。

## 2026-11-09 — G00/G10合法视觉输入与有限后台训练队列

- 原位扩展native_dp/PersistentImage/native_data/native_controller/position_control：新的显式modality合同，原模式不变；G00只读robot178和既有RGB-D，G10加原242子步signed-log触觉/GRU64/条件融合，Actor141/Geometry85拒绝进入。公共随机参数从同一初始化复制，全部非DINO模块训练，绝不加载StateDP。
- 复用固定DINOv2-S/14与448×252/18×32空间token、Depth、13tick robot、三帧视觉、256持续GRU和96tick反传。一个writer仅缓存冻结输出，mmap共享；新触觉统计只用166训练轨迹，原robot/action统计逐tensor保持。218 Core身份/split保持，无采集。
- anchor_continue与state_image_offline/tight_diagnostics原位接入视觉完整prefix/完整DDIM采样和原掌部loss、权重、选模。沿用effective64/micro4、25428更新≈300epoch、原≥200平台规则。tight_finish现有owner串行缓存/诊断/单Kit，两个独立trainer并行实测对比串行anchors/s，可降并发；无新脚本/模型文件/配置树/调度框架。
- 最新完整GPU授权覆盖旧利用率限额；state_resources增加明确full_gpu_v1，仅取消利用率节流，保留6GiB VRAM/14GiB RAM/60GiB磁盘和持续记录。严格核实UID/cmd/cwd/output/父子关系后正常停止用户fingereye数据生成service，完成数据保留；其他工作树/用户修改和原数据不动。
- 两条真实缓存轨迹、G00/G10各一次可丢弃更新/EMA/恢复/完整H24采样通过；全部共同模块和触觉模块实际更新。G00拒绝触觉/特权输入；G10触觉条件干预非零。tick108训练与部署条件差0，live/cache DINO差0。27项既有动作与几何loss测试通过。正式训练/离线/开发成功率仍未测，启动后由普通程序更新当前结果，不声称已学好。
- 同一validation30对照StateDP/hold/原阶段，后续各最多10匹配spec开发trial、首1视频，总缓存1.5h/训练9h/离线1h/评估1h、总12.5h。失败不自动重试或扩100；G01/G11仅记录后续冻结基础checkpoint关系。

## 2026-11-09 — 完成结果发布、误差图与指定旧Core清理

- 实际训练19865更新/1099214 anchors/234.374等效epoch，原平台条件停止；后续10个新初态全部合法自然终止，0成功、8timeout/2fall、2native lift/2首子目标，前3视频验收。原Kit前资源失败0次单独保留，不追加实验。
- 原位扩展tools.tight_diagnostics的CPU-only publish入口，读取完成run的CSV/JSON生成6图和可追溯指标：完整DDIM commit12/valid24曲线、训练分项/固定验证去噪loss、分阶段old/new/hold及n、运动/近静止、两条目标/变化量叠图。导出逐tick与逐update指标gzip、218条身份表、10例闭环、冻结配置/源SHA/选定模型SHA与资源实况；不加载policy或Kit、不修改原训练源或产物。
- STATE_DP_RESULTS/README/CONTRACTS/AGENTS更新为真实完成状态，保留历史证据。明确train2/val4选模与最终train8/val30、命令FK/接口/物理tracking区别；近静止Hand仅40tick且不及hold，validation平台/泛化差距与0/10闭环均披露。新增数据和曝光同时变化，不能归因给单一loss。
- 用户明确大文件留本地，仅代码/文档/图表/原始小指标推origin feature。现有outputs忽略规则保持，未添加checkpoint、HDF5或MP4。训练60s整卡峰值72.127%超限，评估35.456%，未抹除资源违规证据。
- 按用户指定tight_state_20261107_01及逐项清单，仅删除208条records与当前cache逐字段一致的旧numeric.pt；13个大日志无损gzip并逐一解压SHA验收后删除原文件。释放303198208 allocated bytes/289.15MiB；208继承轨迹、Aux/原速参照、唯一旧best/normalizer、失败分母与开发cache保留。9002文件identity检查及218载荷size/存在检查通过，清单与恢复路径原位写入CLEANUP_AUDIT。
- 验证完成产物的数值、SHA、图像、文档链接与忽略范围；无GPU、训练或物理回归重跑，未改另一工作树的用户文件。

## 2026-11-09 — 明确授权接续未启动的成功率/视频队列

- 训练已于04:48结束19865更新/234.374 anchor等效epoch，last4分数改善0.7646%，04:54完成38离线轨迹；同validation30 Arm .0065168→.0056455rad、Hand .0572763→.0566184rad、掌部5.2635→4.9823mm，非闭环成功证据。
- 旧视频队列在Kit前约4秒整卡累计均值59.76%触发资源终止，0物理尝试，10身份均未启动；所有worker/monitor已退出。用户明确要求继续。
- 原位复用已有VideoLease等待模式，仅接续run配置切换exceeded_mean_action=wait_for_original_admission，保留50%整卡上限、45%准入、35%目标、6GiB显存/14GiB RAM/60GiB磁盘。等完整60s窗口并等待负载回落，不因短采样偏高立即失败。
- 新run为原目录remaining_01，一次新授权≤1h墙钟含等待。原plan逐项相同、最终EMA/配置/normalizer/旧失败证据SHA校验，原10个seed不补换、前3视频、不重训、不改任何运行源码；普通后台结束后更新原有结果和索引。启动不等于录制完成。

## 2026-11-09 — 原平台条件满足后的有限闭环与视频

- 用户追加授权成功率实验及视频；35485d7活跃训练工作树源码不改，独立feature/tight-palm-video-evaluation工作树原位扩展已有tools/tight_videos.py与configs/tight_failure_videos.json，不新增policy/仿真/脚本/配置文件。
- prepare-after-training冻结10个新身份、首3视频和原平台条件。普通CPU依赖等待仅在训练及最终38离线正常完成且owner/worker/monitor确实退出后继续；未达到≥200等效epoch且last4改善<1%则SKIPPED，预算耗尽不冒充收敛，原严重退化规则也拒绝。
- 只读取最终validation选择best，以离线SHA、EMA计数、配置与统计SHA验证；无strict=False/参数更新。复用同一个trial与录像器，success/合法、拒绝、中断、未启动分别统计；不按结果挑视频，不补seed，最多10次/1h，不扩100。
- 原GPU/内存/磁盘约束沿用，超50%整卡均值终止自有worker；单owner锁、源码冻结、无stdin/重试，完成后普通程序生成README、summary和原位结果段。等待时没有GPU monitor或Kit占用。
- 10项定向CPU测试通过：平台门槛、非有限/退化拒绝、前驱失败阻断、真实进程退出、最终offline选定SHA固定、成功率真实分母；实际prepare验证训练/部署核心源码一致、10个seed无旧队列重叠。实际GPU录制仅以新run状态为准，排队不等于已经评估/录好视频。

# 2026-11-08 Core/Auxiliary完整生产、训练、闭环与清理结果

## 2026-11-08 — 输入审查、掌部几何损失及一次有限新训练

- HEAD起点55ca805，保留自动更新的两例视频完成结果；旧IDE工作树与原400/275/Core/Aux、唯一旧best及normalizer不动。不新增源码/配置/脚本，原位扩展tight_state/tight_finish/tight_audit/anchor_continue/tight_diagnostics。
- 定点追踪确认actual palm=iiwa14_link_7，geometry[28:35]实际base位姿；robot q/qd、target_before、target−q、两组t−1已施加位置目标都每60Hz进入现有GRU/直达condition，无缺失字段，故没有复制输入或改网络。当前两历史槽语义相同，不是专家滤波前request。每tick真实在线controller与完整prefix109tick输入差0、condition差1.90735e−6。
- GeometryObjective增加显式palm_command_sample_commit12模式：原29D H24 Loss B不变，sample/x0反归一化，掌部FP32可微FK，commit有效mask下m²位置与0.5矩阵Frobenius²旋转；空mask安全。新模式不叠旧finger/temporal辅助项，旧loss模式保留。mask_totals/weighted_parts分别按24/12有效计数保留microbatch梯度语义。
- train4固定两批一次尺度/梯度校准：lambda_pos=.3342396121814294m⁻²、lambda_rot=.029946394754032265，各项值/输出梯度≤base5%，500更新渐入；新统计下只验收≤20%保护、不重调。四轨迹真实掌部FK最大位置4.25e−7m、旋转矩阵1.05e−6；模型梯度有限非零，一次CPU更新、保存恢复EMA/normalizer和完整1×24×29采样通过。
- 同一训练划分无剩余未用Core。预排原36训练家族120个未用身份，仅目标50/最多60通过原Core全模态/真实运动门槛的成功轨迹，2h含等待；截止后实际数冻结，val30/test22固定、Aux不加入。复用验收过的基线numeric缓存，新的训练统计来自扩充train；只新载荷重做全准入。
- 原trainer新模型随机初始化，effective64/micro4/短尾、AdamW、GRU/U-Net/扩散/频率/控制保持；按最终anchors冻结300等效epoch目标和新cosine日程，≥200才平台早停，严重退化/数值异常例外。固定validation4动作/掌部四项等权尺度分数选best，保留8500更新与100等效epoch比较点；墙钟9h优先，预算不足不称收敛。
- 原完整采样入口增加可选全部有效H24指标，仍逐tick真实前缀、batch1/seed7091；前12与24分开CSV、阶段/hold/掌部角度，最后固定train8/val30，复用旧38诊断，零自动学生闭环。原绘图入口补loss、Arm/Hand、位置/角度曲线和阶段比较，结果MD普通程序原位更新。
- 严格Lease支持显式bounded_mean_v1并检查配置磁盘/固定deadline，50/45/35%、6GiB/14GiB/60GiB不放宽，外部突发不声称硬配额。CPU checkpoint路径不为保存CUDA RNG意外初始化CUDA，GPU已有RNG恢复保持。仅保留last/best，清理本轮可丢弃验收权重/临时numeric镜像，不删除未确认无引用的旧缓存。
- 原生物理/控制/资产/专家/任务判据未修改；正式进度与失败原因以run/status及stage日志为准。检查通过和后台启动均不等于训练完成或任务成功。

## 2026-11-08 — 用户追加三个失败案例观看视频

- 用户再次要求直接交付视频后：首次worker因整卡60s均值51.285%停止，未自然终止；592帧/9.87秒完整解码通过，以明确interrupted名称硬链接导出可播放MP4，原partial/FAILED/资源证据保留。原批次均值34.916%，不能宣称全程合规。
- `tight_videos prepare --remaining-from`仅接续原计划中从未启动的formal015/017，拒绝重跑第一例或补seed；新子目录继承原绝对deadline。仅该显式接续配置将越界动作设为既有Lease等待至整卡批次/60s/近期均值≤45%，50%上限、内存/磁盘储备和35%目标不改。预算耗尽仍停止，普通程序更新原观看索引，无自动新一轮。
- 原27个Core闭环只留数值，无可直接导出的RGB。`configs/tight_failure_videos.json`冻结formal000/015/017：历史未抬升超时、抬升后fall、首子目标后fall；按结果挑选，仅供观看。相同seed不声称隐藏物理状态/逐帧重现，复跑的真实结果单独落盘，不补seed、不改原分母。
- `tools/tight_videos.py`小型普通串行入口复用tight_state环境、state_benchmark.trial和RolloutVideo：单Kit、3次、总1h，三视角1344×316/60fps，逐帧解码核对自然终止步数，原子状态/观看索引/当前结果更新。锁、源码/输入SHA与有界父进程等待保护；失败即停、不自动重试/训练。
- `tight_state.environment`仅新增可注入既有Lease，默认行为不变；本队列采用35%目标/45%准入/50%整卡均值停止、6GiB VRAM/14GiB RAM/60GiB磁盘储备，保留连续遥测，外部负载不能被称作硬GPU配额。
- `state_benchmark.trial`仅新增可选expert_checkpoint身份参数，旧调用不变。本次明确分开实际学生与shadow专家SHA；旧27份结果checkpoint字段误指专家的原证据保留，实际学生身份由旧evaluation_summary确认。
- CPU编码/完整解码12帧检查、16项既有动作/历史测试通过（合成临时测试片已删除，非任务视频）；真实Kit录制完成度与资源实况只以run文件为准。模型、GRU、采样、动作、相机、任务、物理、数据均未改。
- 前轮DP3诊断现已完成38/38：正式训练0、新Kit0；原位结果表同步实际指标，详细CSV/JSON/曲线留原run。

## 2026-11-08 — 既有StateDP的DP3条件通路与有界部署采样诊断

- 起点0caa908，实际feature/tight-state-foundation工作树干净；IDE旧分支3e11d62和两份未提交历史MD原样保留。
- 数据/观测/forward/部署定点追踪：Actor141含相对关键点，robot178和geometry85的当前帧经短窗口MLP直接进入896→512 condition。
  `state_history.py`及模型/数据/控制实现原样保留，未创建重复64D分支、未改GRU/U-Net/损失/DDIM/频率或旧checkpoint。
- `state_image_offline.py`补`deployment_sample/diagnostic_rows/summarize_diagnostics`：真实完整前缀、逐tick HistoryBank、batch1相同部署噪声顺序、H24前12、hold-target和阶段/运动分箱、FK单位。
  原`evaluate`保持旧episode-batch噪声与历史可比性；新结果明确独立协议。
- `tools/tight_diagnostics.py`是独立有限普通串行入口：只做旧曲线/闭环复用与固定38离线样本，CSV/JSON生成PNG，原子状态及原位结果MD、锁、deadline、现有资源Lease/monitor。
  依赖/NaN失败即停，无自动重试/模型/采集；未建设训练调度框架。
- 新配置`configs/tight_dp3_diagnostics.json`：旧8500更新最近四次采样验证仅改善.9386%、泛化差距35.52倍；按用户“等价则不重训”条件，新增正式训练0、评估最多1h，Aux/test不用于选择，Kit0。
- CPU实际checkpoint验收：366tick，31×512 condition，训练/部署max2.14577e-6；固定Actor/GRU，当前robot/geometry改变分别产生condition差.0038633/.0130241，梯度L1 .06103/.08842。
  一次可丢弃CPU更新、完整optimizer/EMA保存恢复、train-only normalizer匹配、1×24×29完整预测通过；临时smoke checkpoint已删除，旧权重未改。
- 29项定向测试通过（3新诊断测试＋既有动作时间/因果历史回归）；25真实tick/3完整采样plan的CPU指标与CSV→PNG集成检查通过，旧17点曲线和27闭环/2物理图导出通过，复用未变模型先前验收；原400/275/Core/Aux、专家、资产、正式分母保护。
  启动后状态与结果以run文件为准；不把诊断完成、启动成功写作模型性能改善。


- 固定Arm.50/Hand1/cap2，不变任务/物理/DR；55原速54成功+1拒绝、378慢速359成功+11失败+8拒绝。第7轮达到软目标停止，冻结208Core/50家族＋15Aux/11家族；Core产出55.03%，旧20严格10/20保持，旧2缺图Aux不复用。逐条真实运动/路径/State/RGB-D/触觉通过，Core-only69992tick/train-only统计/cache冻结。
- 全新模型/optimizer/EMA/scheduler/RNG训练8500更新、442012draws/100.2068 anchor等效epoch，4411anchor全覆盖。最后4验证改善.9386%触发预登记平台/过拟合；有限充分性及完整恢复通过，不称全局收敛。best8500；测试不选模，Aux未训练。没有重做基础审计，沿用61项定向/准入＋1成员回归。
- 实际闭环27合法：train0/10、validation0/7、新初态0/10；24timeout/3fall，1例完成首子目标后fall。执行器/提交目标及canonical actor误差全0。首10新初态零成功，不扩大100、不换seed，不宣称学生成功或matched-data因果改善。
- 复用既有文档，当前结果收敛为数据/检查/训练/闭环/清理与紧凑历史表；新增本批433行CSV、机器结果和数据/训练图。原失败分母与开发/旧混速证据保留，不新增MD或算法。
- 运行时v1 SIGSTOP将暂停计入work，造成重复比例等待；v2改原gpu.lock计算段外等待，无trainer重启/梯度或随机状态变化，无deadline重置，旧记录保留。全部服务正常exit0。整卡均值28.389%、最高60s59.632%、缺口0、最低VRAM4154MiB，不能宣称全程资源合规。
- 清理135新拒绝RGB-D＋1全状态重复checkpoint，共136文件/36.191101GiB；best原SHA保留，last硬链接恢复别名，旧SHA映射留账。清理后211源码/计划、6保护SHA、225合格payload/135拒绝numeric SHA、manifest/stat/cache、3CLI及8500计数/442012draws恢复路径通过。旧清理不重计，原400/275、专家、资产、失败轨迹不动。

# 2026-11-07 Core/Auxiliary合同与定向动作/时序检查

- 起点e26e3d92e7daa39bc695a9bef3bce64b49ca7538，干净修复工作树。tight_gate增加新版分类/确定性家族配额；tight_finish复用旧参照与已完成身份，新合同只解除旧65%严格产出率停机，实际成功/路径/模态/原生条件不改。
- configs/tight_state.json预先固定Aux1.35–3.30、Core目标200、Aux≤100/≤Core一半/每家族≤2；最多8轮440慢速计划身份含旧20，继承截止时间。Core-only manifest/统计/cache/训练与完整组成manifest分开。旧10/20及原root不改。
- 新增确定性测试覆盖已知29D HDF→loss归一化→predict反归一化→实际目标缓冲、tick编号anchor/首尾/padding/重规划及episode reset；模型/数据适配/执行器实现未变，无新模型实验。当前持续GRU并非64tick截断。新准入回归拒绝物理漂移、绕路、无效传感器/等待与严重时长偏差，并验证不重复旧案例、不因轻微时长越界停止。
- Core-only闭环分组补充：原入口按预设55家族取train/validation，未保证其有实际准入记录；现按最终Core manifest成员、字典序和既定上限取样。1项新增确定性测试通过，排除无Core家族/最终test/重复家族，并保留不足上限的真实分母。采集、训练及所有模型/控制函数不变；当前采集worker继续使用4d43a39，后续评估使用本次显式源码修订receipt，不重启运行或重置deadline。
- 本条为实现/登记，不是生产或训练完成；实际数量和回归结果以本轮receipt及STATE_DP_RESULTS为准。

# 2026-11-07容量恢复与正式生产接续登记

## 2026-11-07 formal_05正式生产完成首检查批次并按yield停止

- 实际源码5b3bd48ef720f71d860b2c76a32b39b4e27d8dc1，干净启动；容量恢复后复用既有入口/配置/accepted gates，继承原截止时间。未改模型/采集/控制源码、参数、输入输出或物理，不重做基础审计。
- 55原速=54成功+1动作前拒绝；20慢速=20成功/10准入。累计任务100%通过85%、准入50%低于65%触发预登记停止，剩余200次未跑。74实际采集三模态全过；无任务/软件/传感器失败。10条候选train8/validation2/test0，正式数据/新训练/新闭环未完成。
- 新增小型evidence/tight_production_results_20261107.json、75行attempts CSV及路径/速度图；更新既有STATE_DP_RESULTS、CONTRACTS、README、SESSION_HANDOFF、AGENTS和清理记录，清除当前容量阻断误述，旧结果保留历史标签。下一机制仅采集侧参考路径进度提案，未实现。
- 验证：207源码/计划、6保护文件、64新HDF5和10拒绝numeric SHA；原400/275全部存在且大小匹配；75预登记/75完成、无重复身份、真实退出日志。未运行新的工程测试，因为实现未改。
- 本次清理10份拒绝RGB-D载荷2.705GiB，保留全部数值/失败分母；此前7.510GiB不重计。所有自有进程退出；整卡均值23.231%、最高完整60s44.909%、遥测缺口0。旧混速State结果train1/4、新初态0/10+2拒绝不受此采集门控影响。


- 起点f94872b8af99a1bb0b08249d357e877ce3ce883f，干净工作区；实测可用约1TiB。只新增接续receipt与状态记录，当前采集/训练/执行/模型源码未改，不重复基础审计或开发验证。
- formal_05继承hand_cap_04通过gate、原绝对截止时间及当前唯一配置；55新生产家族41/7/7split，55原速＋220慢速上限。已有14开发合格轨迹保留，正式集不混入开发筛选案例。全部新生产门控通过后自动进入新数据审查、随机初始化训练和分cohort闭环。尚未声称完成。

# 2026-11-07追加分类清理与当前入口收敛

- 用户明确授权清除旧无用checkpoint/记录及GitHub重复代码；从b85738e执行，非前次清理重复计数。1867产物释放7.510GiB，含旧私有缓存、未训练/一次性权重、9未准入旧慢速RGB-D及144失败诊断重型载荷。原400/275、14合格开发/30本轮capture、实际对照、成功模型和全部失败分母保护。
- State200 best/last完整状态逐字段一致，保留best原字节并让last硬链接同inode；原last序列化SHA与映射留账。当前best4500/算法/配置/已接受工程receipt不变。
- tools/narrow_collection、narrow_followup、narrow_repair_pipeline、narrow_delivery删除重复队列/发布代码，保留现有对照和拒绝回归函数；删除evaluation_priority、narrow_repair_delivery、tight_final_calibration、state_mechanism_delivery。state_mechanism仅将过期自动发布尾调用改为本地pending_review。当前tight_finish生产/训练/评估入口不变。8份字节相同历史报告仅留canonical副本及映射，未删实验对照。
- 54项受影响测试通过；5225文件identity、保护manifest/best/statistics SHA、14合格payload SHA、4个CLI与依赖通过。未重复168/510tick物理审计。清理后仍至少缺19.41GiB，正式采集和新训练均未运行；剩余budget不重置。

# 2026-11-07 Hand cap与独立State实验实际结果、容量停止

- 物理source f8beeef6d408ac20f520f6a26625c31bc3cdb5ee：Hand cap2rad/s、Arm.50/Hand1，pilot5/6任务成功4/6准入；独立12/12成功10/12准入，四家族3/3/2/2覆盖通过。新增30采集全部State/RGB-D/触觉通过，真实执行器目标误差0。14条7家族合格开发记录显式train-only，非正式manifest。
- State best4500 canonical独立16尝试：train1/4，unseen0/10合法、另2初态拒绝；无新参数训练或旧seed复跑。family005176tick完整成功，10个合法新初态均零子目标601timeout。无稳定泛化改进证据，不把不同seed归因canonical修复。
- 完成路径/速度分解及真实DR暴露核查。B00旧时长主要伴随物体路径增长；cap确实降低实际Hand RMS到.454–.594，但仍保留fall、.966和3.118倍率拒绝、插入路径过长等4条拒绝。逐条结果和图/CSV在既有结果文档/evidence，未新增MD。
- 可靠性通过后因实测容量停止：8GiB储备以上空间，最大复用后仍需最乐观约31GiB。formal物理attempt0，3实验worker exit0、owner明确STOP exit1，进程全退出；没有新训练/正式数据冻结。
- 运行结束后configs/tight_state.json固化唯一cap/实际路径速度门槛；tight_finish正式collect改从当前配置读取，tight_state旧scalar开发入口拒绝cap配置；tight_gate拒绝错scheme/缺实际物理路径证据，避免静默退回旧方案。37项受影响回归通过；18条已记录运动判定与运行时一致，cap/原生包装/学生源码未再修改。不重开168/510tick基础审计。
- 整卡均值15.224%、最大完整60秒25.820%、缺口0；209运行源码/计划SHA及保护清单通过。仅删除本轮重复分析JSON和临时pilot图2文件876544bytes；原400/275、checkpoint及全部必要分母保留。

# 2026-11-07 Hand cap有限实验实现与登记

- expert_speed.py / sim/expert_speed_environment.py：可关闭的Hand逐关节目标速度上限，native原处理后应用一次，反馈真实target；None逐位保持旧行为，已处理学生目标绕过。无PD/物理/专家变动。
- narrow_repair_audit.py：实际Arm/Hand/palm/object路径及平均速度、平移/旋转和对称时长分解；不把RMS速度与路径混用，不跨非连续阶段拼接跳变。
- tight_finish.py：复用旧参照的6次单参数pilot，附加物理速度/路径准入，条件独立可靠性队列，单canonical独立16次State评估；A数据否决不取消B。原正式数据门槛保持；限速后的原速伪参照被拒绝。
- 33项受影响回归通过，包括关闭恒等、每关节速率/限位、原生仅一次、学生绕过、绕路拒绝、路径/速度分解。既有168/510tick等不重审。登记root hand_cap_04，物理结果尚未执行；不声称完成。

# 2026-11-07 A/B最终结果与只读分析复现

A以991db6a1c02af140ef12fb1f3a35f159c7ba2193实际完成24次登记尝试：旧/canonical输入train均0/4，未见初态0/7与1/7，各另1动作前拒绝。unseen_05 canonical178tick原生完整成功、legacy601timeout；实际执行器目标误差0。源best4500/data/statistics、241运行源码指纹不变，进程全退出；没有新训练或checkpoint。正向单例不夸大为稳定改进。

B只读分析33capture/21慢速配对，按接近、抓取/抬升、插入/原生结束给出时间、Arm/Hand速度/实际路径和范围、DR实际两子步暴露。运行结束后仅把原分析的实际wrench计算纳入现有narrow_repair_audit.py，33条重算逐值一致；未改或重跑物理。受影响入口24项回归通过，旧168/510tick工程验收继续有效。

现有结果文档、CSV/阶段图和机器证据分开报告A/B，保留全部失败/拒绝分母。B无可靠统一方案；采集侧Hand因果目标速度上限仅提案未实施，不生产正式慢速集。本轮清理0，原400/275和必要成功/开发证据保持。

# 2026-11-07 A/B独立诊断入口

- tools/tight_finish.py：沿用原监督/环境/trial，仅增加历史275显式baseline分支和同权重输入构造对照；正式慢速数据门槛不放宽。旧checkpoint只读、24次有限队列、相同seed及交替顺序、实际执行器核对。
- tools/narrow_repair_audit.py：对33条已完成capture计算逐阶段Arm/Hand速度、真实运动幅度、耗时、阶段切换及DR暴露；不改原准入标签、不新采样。
- tests/test_tight_gate.py：拒绝错H/未训练/未知输入版本，证明显式输入副本不改变源权重/配置。24项受影响测试通过；其他基础工程实现不变。
- 现有合同/结果/入口/交接记录A/B独立授权和预注册队列；未创建新MD或checkpoint。实际结果待落盘，不把登记当完成。

# 2026-11-07 固定方案验证最终结果

运行代码cf275d5ec8029782942f2d36a58eb06f669e4b02，启动工作区干净，结束241个源码/配置SHA不变。3家族各2个新event seed，原速6/6、慢速6/6成功；慢速2/6准入，4条总时长比低于1.5，其中3条插入运动指标也未过。2+剩余6<要求9触发预登记早停；不是任务失败或要求零失败。12条三模态均通过，6条实际两物理子步目标误差0，无等待凑比例。

实际worker exit0、owner因门控拒绝exit1，全部自有进程退出；旧143仍不归因Kit/OOM。29项受影响回归通过，原工程验收未重复执行。正式数据0、更新0、三个学生cohort均未执行；本次清理0文件/0bytes，旧6条和新2条合格开发证据保留。更新同一结果/合同/入口/交接及机器证据，不增加重复MD、不改运行代码。

# 2026-11-07 慢速生产方案有界续行

从干净e67bfc6fa1b66359b55e019ae9899240625d5295继续。保留168项及实际GPU/执行器历史验收，不重开审计。最新用户授权先解释15条再验证固定方案。

- `environment_rng.py`/`state_benchmark.py`：仅显式event_seed增加reset后独立事件流；原分布、reset和无该字段的旧路径不变。4个事件seed/家族，真实新案例，不重采旧轨迹。
- `tight_state.py`/`tight_finish.py`：沿用原capture/pipeline，支持既有Arm/Hand独立参数、单方案12对验证和数学不可达早停；生产按固定scheme验收，所有attempt保留。systemd承载owner，未知143不归因Kit/OOM。
- `tight_gate.py`：分开个体准入、任务可靠性、产出和家族覆盖；绝对1.5–3和模态标准不放松。已接受工程验收凭完整baseline、receipt SHA和未变实现SHA复用，新正式数据仍全量数值/统计验收。
- `configs/tight_state.json`：固定Arm.50/Hand1，初态域不再缩或扩；12slow+12reference、成功11/12/准入9/12、每家族3成功2准入，原5连续失败和新增每20例累计生产停止规则均在启动前冻结。
- 受影响测试29项通过；相机/网络/损失/统计构造/microbatch/EMA/目标执行数学未改。真实新采集/训练是否通过以当前结果为准。15条完整紧凑表及两项修复的实际历史影响已写入同一结果文档，没有新增重复MD或重复清理。

# 当前变更记录

## 2026-11-07 Tight-Insertion StateDP基础链路整改

基点642f5a2；IDE原工作树为3e11d62且两份原文档有既存修改，未切换或覆盖。当前修改在bc+rl_RSS_narrow_followup本地feature/tight-state-foundation。完整旧变更由Git642f5a2追溯，科研失败分母集中在RESEARCH_HISTORY和机器证据。

- `configs/tight_state.json`、`tools/tight_state.py`、`tools/tight_finish.py`、`visuotactile/tight_gate.py`：提取独立当前配置，真实开发校准、逐条慢速/三模态准入、有限训练与评估阶段；实际运行是否通过见STATE_DP_RESULTS，不能把代码存在写成训练完成。
- `visuotactile/narrow_data.py`：direct_configuration读取XY/相对孔位/角度/台高/高度/关节lerp/速度配置并拒绝坏范围。正式缓存拒绝旧manifest与partial准入；原400/275不改。
- `tools/narrow_collection.py`、`tools/narrow_repair_pipeline.py`：旧时长比无条件passed和1.35中位数放行关闭；正式相机必须显式内容检查，旧混速入口停止生产。
- `visuotactile/state_history.py`：定位并修复新入口的normalized-proprio反算舍入误差；新checkpoint显式canonical_joint_state_v1，旧checkpoint保持legacy_packet_inverse_v1。维度、网络、标签、损失与统计规则不变。
- `visuotactile/sim/expert_speed_environment.py`：处理后绝对目标旁路额外alpha；原native专家减速路径不变。真实240tick H24/commit12请求/执行器误差均0；开环未成功记删失，不作为数据否决。
- `tools/tight_audit.py`、`tools/tight_check.py`：全部275条数值审计、实际GPU FP32完整510tick controller对照、真实4+3 microbatch梯度/EMA/optimizer与完整保存恢复。第一次normalized robot8.07e-5失败保留，修后输入差0、condition1.19e-6、动作2.53e-6通过原2e-5门限。
- `visuotactile/recording.py`、`visuotactile/state_benchmark.py`：新入口可选gzip4无损存储；旧LZF默认不变，RGB/depth/valid逐值和原始field digest核对，不复制帧或改时间。
- `visuotactile/anchor_continue.py`：正式训练必须验证新manifest与GPU gates，支持同一完整随机训练run有限阶段边界和原schedule连续恢复；未把预算停止称收敛。实际正式更新0；完整训练阶段未实跑。CPU/GPU准入与正式data/statistics SHA绑定，拒绝陈旧fixture通过标记。
- `tools/project_maintenance.py`：Linux已退出zombie不再误判为仍可持有文件的未知进程；活进程仍不得豁免，按inode/打开句柄/SHA分类删除并计实际分配块。已删除2个废弃观看脚本，旧大视频/cache/checkpoint删除有完整ledger。
- `pytest.ini`：默认只收集当前tests。第一次无目录pytest误扫outputs里的旧源码快照与Kit SDK产生35个收集错误，未隐藏为通过；维护测试的真实结果另记录。新增测试拒绝坏比例/坏阶段运动/假passed/空白相机/错采样范围/错触觉时钟等输入。
- `tools/runtime.py`：末轮owner退出143后复核完成记录、回收遗留监控；新SIGTERM异常化让finally处理owned进程，有真实本进程信号回归，未新增Kit复跑。
- 文档合并：当前README/CONTRACTS/STATE_DP_RESULTS为主；旧SLOW_EXPERT_RESULTS与GRASP_FAILURE_DIAGNOSIS重复结果文档删除，历史表保留事实；AGENTS/交接/方向文件只引用当前授权，清除相互矛盾的旧待办。

本轮物理尝试、真实门控、资源、删除字节、正式数据/训练/闭环状态见STATE_DP_RESULTS及evidence/tight_state_20261107.json；未上传原始数据/checkpoint，未改共享Git历史。


2026-11-11 13:24:31（北京时间）：源码593f109启动普通后台 `tight-dagger-pool-20261111-01.service`，owner PID1344446，root `tight_dagger_pool_20261111_01`，总截止11-12 07:24:31。合并CPU139项通过；启动只代表队列开始，首轮QG01真实采集凭据与后续两阶段更新/自主结果由该root记录。
