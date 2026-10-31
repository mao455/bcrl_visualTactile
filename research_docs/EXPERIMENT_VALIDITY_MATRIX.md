# 历史实验有效性矩阵

审计基线a07d0aa，2026-10-31。本表评价**具体结论**，不按“某模型所有内容一律有效/无效”处理。
Verified=在明确范围内关键接口实测；Qualified=结果真实但解释有限；Invalid / Must rerun=已有实质错误使特定声明不成立；
Unverified=证据不足。重跑只对未来确需使用的无效声明，不代表自动重做所有旧实验。
原400、固定320/40/40、全部权重与旧正式各0/50不修改。

## 后续证据 — 专用T0事件重放

下方各历史分母保持，最新详见[T0_REPLAY_AND_DP_CONTROL_FIX](T0_REPLAY_AND_DP_CONTROL_FIX.md)。
新12条T0：original0/4、delay_fixed1/4、delay_wrench_fixed v1 1/4；fixed控制链误差0是**Verified**，
历史完整物理轨迹仍超容差，精确重放声明仍**Unverified**。源世界坐标额外2/4成功也是**Qualified**结果重现。
新seed17专家与native_request v2在新Kit中均220步成功，所有已比较控制和物理字段逐位相同；
此单lane观测轨迹重现**Verified**，不等于历史4lane/完整PhysX状态验收。
源400有效性不受影响；DP1,005步reference/native控制数学通过，无双重延迟，无普通controller修改或重训。
新增DP16条独立报告，不并入旧T1/T2或正式0/50；record2获得真实Kit整轨迹验收，替代下方当时“尚无新Kit”的状态。

## 1. 身份和共同接口

原manifest为`outputs/visuotactile/data/close_v4/manifest.json`，SHA256
`f4562c18c51a7f7e40bc0a587b7e58aae4889dc5a06dfce4521a1e9e0f003b28`。
HDF5 schema2/trace2/storage2/layout4；train64825、val8670、test8399个native ticks；320/40/40唯一ID无交集。
各方法normalizer只来自train；旧command本身[-1,1]，新reference使用train limits affine，q/qd使用train mean/std。
以下“同源环境”指tight_insertion/Lpeg_matchedmass/.0578kg、layout4三RGB-D、world_hold、120/60Hz及原native DR。
采集398条4env+2条1env，评估1env；这种随机流差异不能隐去。

所有主要入口共同native控制：canonical29→Lab→每tick随机3槽index0..2→arm积分/hand绝对映射→0.1平滑/limits→两物理step。
新绝对reference先按当前target逆映射；Residual在pre-delay command空间组合；T0原样提交源clipped_pre_delay。
本轮81,894个源step检查中target数学、queue来源、前后q/target连续及obs-action时间误差0；command-reference往返≤2.38419e−6。
147个历史numeric trace的delayed均属三槽候选，误差0。这些是接口证据，不能自动证明学习成功。

### 实际源文件身份

| 阶段 | 训练/采集来源 | 评估来源 | 精确补充记录 |
|---|---|---|---|
| 400 | metadata基commit e546bca(2条)/dfcedf4(398条)，含工作树修改；不是以最终HEAD解释 | 每条source SHA、result、143字段审核 | system_audit/control_data_audit中的code_identities/rows |
| B0/B1 | 运行启动76bb58e；command v2 | dc77a6b；evaluate_baselines SHA b2083c19aba846eb68660ac15d8194746b496132868d8c808957db11416b3825 | dp_command_20261029/evaluation/plan.json；baseline_review_20261029 |
| history3 | INDEPENDENT_AUDIT/probe_action_history，各诊断run身份 | 独立development入口，未替换正式eval | INDEPENDENT_AUDIT_20261030.md内逐诊断实际artifact路径 |
| GRU B/C | base commit b88d685；trainer e52600fc4dbc04d03ba4ca1274f2394f892009d95077a7eae797432d7e4cbe63 | 原closed_loop及remaining_01各launch，不合并成一个版本 | residual_20261030_01/training_full/config.json、各launch.json |
| candidate1 | 初步未提交工作树；91fc738首次公开错误归档 | 没有合法闭环 | LONG_HORIZON无效尝试记录 |
| candidate2 | trainer 2ed60c836f6c046fa62bf0c40f02a5506951ab1414853bd49ffbd30e23020065；model 439336c6f66d140a8782b15a90315d13b5e5d70a78ca06dc7c4bedb6a2e6c527 | e65c306；controller 562091b44abb35d9f66059b918b2a517df2d8d56a8b37e445583bfeb5d38301c | full/config.json与closed_loop/plan.json |
| Loss A/B | trainer c887ef319aca82589f1ce00499f529780fb3f662c0999c50af01726851ec7705；study_metrics b4286315aec92d36c5ef88c0286c351e4130bbbff99817f0819ecefbb16034d2 | dp_closed_loop/plan实际hash | dp_loss_trt_20261031_01各config、plan |
| Transformer | transformer_training/force、masked各config保存model/data/controller hash；eef7a9d引入已授权actual反馈 | d11f953；evaluate_study 1fc5114e9cb7470b67f6a29208a55da108c7e2ef2794070abf67f3bd49bd3848 | historical_sources完整保留，两组独立权重 |
| T0/T1 | 无训练；session base3ce0c18；实施首公开a07d0aa | evaluate_training_initials近似恢复实现 | network_audit_20261031_01/session、native_audit |

完整64位hash、实际model/base/源配置和缺失项见[historical_sources.json](evidence/system_audit_20261031/historical_sources.json)。
未提交运行文件以保存SHA识别，不能用base commit代替；缺失历史完整配置不臆造。

## 2. 逐实验有效性

| 实验/声明 | checkpoint/输入/数据 | 实际观察与验证 | 限制/问题 | 状态与行动 |
|---|---|---|---|---|
| released RL专家 | pretrained_assembly/tight_insertion/model.pth，actor140+SAPG32/LSTM；非学生传感权限 | 真正env.step成功采集；本轮实读权重层shape；原native源比对 | 不能拿其特权成功率当RGB-D学生可直接达到的性能；不复现作者训练过程 | **Qualified**：作为固定专家/数据来源保留 |
| 400自然成功 | 固定manifest、layout4；当前/未来clipped命令仅监督 | 本轮400全部retract且goals完成、81894步动作/目标/时间链通过；已存全hash/143字段证据 | 成功筛选、合法性rejection、多env共享随机流；不是400个任务，也不是无条件success rate | **Verified**：自然成功记录与已测数值接口；**Qualified**：数据分布/泛化声明 |
| 早期CPU no-render测试 | 历史原生CPU不render | 首步kinematic table/hole移回spawn；原报告已更正 | 场景实质不正确 | **Invalid / Must rerun**仅针对使用该配置的性能声明；当前render结果不受此项一概牵连 |
| B0训练 | command2，memory2/no-own-plan/no-issued；best49 SHA fd1bd263de38ea6c6e0082d2ba55121adf3908b31c78b55d1cf39477ad6ff9d5 | 50完整epoch/61050更新；best59829；test epsilon .177786 | epsilon loss不等于命令精度；非去预加载；视觉随机初始化/短H为设计差异 | **Qualified**：合法命令BC与离线loss保留；不重训 |
| B1训练 | 同split，额外force/FK；best49 SHA bb23694f94294b20a2bb21b0c260b3ebbd30f2e3ed6a6e22b2465a501b1c5cc2 | 同预算；test epsilon .148961；触觉分支训练/历史有检查 | B1直接读触觉，不能作为视觉nominal或独立Correction | **Qualified**：同预算命令对照保留 |
| 正式50 paired legal starts | 原B0/B1 best，protocol2 CUDA FP32；同源环境1env | 各0/50；100条601step timeout，无首子目标；本轮100条queue匹配 | 同seed/合法初态不是所有随机扰动路径严格相同；0/50不是永远零能力 | **Verified**：该协议实际0/50；**Qualified**：性能外推。不并入后续诊断或录像 |
| B0/B1四录像 | 原best、981001/2 | 四601帧视频完成；记录GPU与解码 | 额外观看重跑；一条late divergence，非bitwise replay | **Qualified**：观看证据保留，不新增50分母 |
| issued history3短拟合 | 新command3；t−1专家反馈teacher forcing；非旧best | 单示范24步前缀拟合，七条独立诊断含专家成功一次/学生六失败 | 非完整episode任务学习；一条startup abort不算任务失败 | **Qualified**：输入接口/有限拟合；不能宣称行动历史提高任务能力 |
| GRU残差B/C离线 | frozen B0；B best0ce3ce5c7df77abfb79c557528b2869cbef38321ad9d1d6047046d73036ada8b；C8750ebb9f2595ee0af96abcbc44a414a73035a3e06f24e99093c777cce3899be | 各5遍7875更新，test40；组合MSE A .659997→B .336392/C .337817 | 独立checkpoint、expert观察/过去命令teacher forcing、gate保留；不是RL | **Qualified**：离线映射改善真实，触觉因果收益未证实 |
| GRU自主/roll-in | 12预分配；原B0/Residual1 | 11native timeout+1switch中止；自主A/B/C各0/2；981002精确switch triad全失败 | roll-in局部不混自主；一组C switch不匹配；中止不补seed | **Qualified**：各结果保留；**Invalid**：把unmatched当严格配对的声明；不重复整批 |
| A/B/C六录像 | 同旧权重，独立viewing cohort | 六条601timeout，无lift/首子目标 | GPU四个60s窗超50%；精确switch不等于所有图像/plan bitwise | **Qualified**：观看；“全程资源合规”**Invalid** |
| candidate1 preliminary | long_horizon_20261030；一遍822更新 | 当前命令泄漏、错误split、持续状态5Hz、DINO未pin/resize | 多处训练语义错误，不能用推理补丁救旧权重 | **Invalid / Must rerun**：已由新candidate2干净训练替代；旧结果不复活 |
| candidate2 epsilon pilots | v2完整episode小集合 | 大幅训练loss降低不保证native arm command改善 | sample/epsilon目标不同，不可改checkpoint flag | **Qualified**：软件/小拟合诊断，不能称正式成功 |
| G00/G10 sample H64 | 40epoch/33280，best EMA40；G00 SHA6410514f9031a891f2df21873c5bcc7e4b82c9aea23b695b06bdda738075ce3d；G10 7976a0ddb33bb33f62ab7728118272ec085863878d4a0e1eb3a8af69d55bef19 | train320/val40/test40；reference native往返通过；三条/组0/3 | G10一条lift .107093m后fall，其余timeout；全无subgoal；arm command误差超过zero基线；与B0多因素不同 | **Qualified**：合法新候选及结果；不能归因容量/DINO或宣称泛化 |
| Loss A/B续训 | 同epoch40EMA，独立新optimizer/cosine，各12遍9984更新；A与B只改组权重 | test prefix12 G00 A .952267/.523374→B .921222/.509035；G10 A .967824/.504618→B .933257/.501724(arm/hand RMSE) | 不同loss绝对值不可直接排序；一次训练seed；离线未来target teacher forcing | **Verified**：该预算离线指标；**Qualified**：总体优越性；默认选B不再重复双线 |
| A/B自主12条 | G00-A/B、G10-A/B各三seed；B best epoch52/累计43264 | 四组各0/3，全部601timeout、无native lift/subgoal；G10-B瞬时+7.6863cm | 3样本不足泛化；短上升不等于native lift/成功 | **Qualified**：失败结果保留，与其他cohort分开 |
| Transformer Force/Masked训练 | frozen G00-B SHA926b2628bb5f5c1eed4869ac257926dcdefdcc1533456553a51e9d0e035cdb92；各828189参数；actual mixed history1 | 各5遍3730更新，optimizer51张量全3730；best5；base未进optimizer | 五遍未证明收敛；Masked仍有真实gate/FK | **Verified**：确实更新/冻结；**Qualified**：学习效果/触觉贡献 |
| Transformer自主六条 | Force SHA114106cd3b0f42e59a12b8c3c5c62e4e4f971a0695e0a01b235f2ae6e8055053；Masked95be645fcb4580ac89e6639420a1713afe84023d727f07ddc704288c9d4b9b5f | 各0/3，全601timeout；physical initial一致；本轮queue匹配 | base/force/masked precontact命令不同；未严格固定神经数值/输入/采样全过程 | **Qualified**：各组实际表现；**Invalid / Must rerun**：将其用作严格残差因果配对，若未来要该声明才重跑 |
| Force同权重干预 | 固定checkpoint/输入/base/gate；4train+4val；96tick入块memory固定 | Force零力输出变化arm .016538/hand .004018；Masked0；val置零误差未恶化 | 非全episode记忆反事实，不是闭环；不证明力带来收益 | **Verified**：输入敏感性；**Qualified**：任务价值 |
| T0四条 | 原源clipped命令、近似初态；本轮1005/1005原RNG delay重建 | submitted误差0，delayed在3/1/2/1tick分歧；全部命令用尽删失 | 多lane→single、仅一次RNG、缺完整模拟器/传感状态 | **Qualified**：近似回放观察；**Invalid**：“证明原示范不成功”；精确replay仍**Unverified** |
| T1近似train初态 | G00-B/G10-B各4；observable误差0 | 各0/4，601timeout，无lift/subgoal，首动作已偏离；adapter误差约1.79e−7 | T0未建立完整重放，不能叫精确训练态；逐tick输入/noise记录不足 | **Qualified**：近似初始化表现，**Unverified**：完全相同训练环境拟合能力 |
| T2 / T3 | 旧三seed自主是T2；没有新20条；没有几何OOD T3 | T2各0/3保留，T3无结果 | 未见seed不是OOD；不与50trial合并 | **Qualified**：T2有限结果；T3 **Unverified** |
| G11/RL/去预加载 | 无合法已训练G11或critic/replay/RL更新 | 未实施 | 用户实际反馈授权不等于已经训练G11；BC标签仍含隐式压力 | **Unverified**，不补造结果、不启动新研究路线 |

## 3. 需要修复什么，为什么没有重新训练所有模型

确认的旧candidate1训练错误已经隔离并由candidate2替代。本轮未发现新的candidate2标签/时间错误。
原生delay、积分/平滑不是失败后可以擅自关掉的bug；T0不同随机流是回放协议不完整。
已完成普通局部修复：未来formal H64默认Loss B，pilot/full/resume核对loss；未来诊断record2逐tick输入/记忆/采样状态记录。
26项定向CPU测试通过，包括原生上游hook行为/RNG一致；修改后的记录器尚无新增Kit整轨迹验收。
旧缺字段不能事后冒充当时保存；完整物理重放、严格残差因果闭环仍未验证，没有新成功率或微调结果。
原正式0/50无需因此全部重跑，原400也不因T0未复现而作废。

新增闭环只有在确认**影响某结论的评估接口错误**后才有必要；保持原权重、独立目录、预登记≤12次/≤4h统一监控。
想使用严格触觉因果声明，需重新设计同输入/同采样/同precision的配对验证；当前六条不能提供该声明。
完整物理重放需单独状态协议，不能将关闭DR、直接写target或每步替换原随机draw包装成原生精确重放。
本轮实际新仿真0/12、参数更新0、GPU窗口未启动；因此本表旧样本分母和研究结论均不因代码测试而升级。

## 4. 研究准入判断

接口/数据基本可审计，足以继续有界工程与拟合研究；**还不能称已具备可靠自主任务和RSS性能结论的基础**。
具体缺口是基础DP自主approach/lift、完整输入/噪声重建、精确重放协议或明确近似边界、严格残差配对、连续力收益、
足够独立测试和几何OOD。它们是独立门槛，不能用训练loss下降或单元测试数量代替。
