## 2026-11-14 第三轮真实纠偏完成，完整学习循环接续

冻结b9b6dcb的24次早期/中期接管全部自然结束：A11/12、B9/12由专家完成原任务，20条通过完整capture/相机/实际执行审计；7969真实历史tick、4689合法专家监督tick。原始学生前缀与2tick队列过渡只提供历史，未伪装专家监督。4次失败均601tick timeout、0子目标；其中3次专家后段出现短稳定窗但最终未完成，另1次未拿稳，均保留。不同接管时刻绑定不同family，不能由救回率声称时间因果优势。

B/formal_007在258tick首次学生拿稳，260tick仍稳定时接管，348tick专家完成；B/formal_053在172tick拿稳但300tick接管前已失稳，538tick专家完成。补入了真实学生先拿稳后的自身偏差，并不算自主成功。A本轮学生前缀没有稳定窗。完整数据/阶段/性能报告绑定source/model/plan/results/numeric SHA，见takeover_03、takeover03_phases/runtime。

依据当前抓取失败及验证各接触分段仍下降，继续已登记共同10遍DP、15遍后端接续、原4开发机会每组。round_03从当前完整pair出发，保留原30+历史21+本轮20=71训练episode，独立6验证与train30统计不变；新DP后全部计划和可训练特征重算。训练已启动，尚无本轮曝光或自主结果；原13:59:40截止保持。活动源码树不热改。

## 2026-11-14 固定验证集的接触分段误差

新增tools/tight_fresh_contact_report.py，从实际完整后端offline CSV聚合同6验证family、同分段tick，按平方误差加权，不平均RMSE。初始/首轮/二轮六个完整模型的身份与分段人口一致。第二轮相对首轮，A/B持续接触手指命令RMSE从0.024483/0.024026降至0.018856/0.020687rad；接触前、初次接触和失去接触也都下降。证据contact_validation_round0_2.json绑定每个模型与原CSV SHA。

这是沿实际示范历史的预测误差，接触分箱不是拿稳/插入成功阶段。自主评估仍未可靠改善，所以继续验证当前学生偏离状态的早期专家恢复覆盖，而非据此宣称学会抓取。真实六实例报告执行通过，未新增训练/改变当前worker。

## 2026-11-14 四环境完整评估完成与第三轮采集启动

冻结628fd92四环境神经批量工程队列全部自然结束：A2404tick/119.096s=20.185tick/s，B2206tick/118.556s=18.607tick/s。当前串行同模型实际12.10/12.17tick/s；对应总吞吐约1.67/1.53倍，批量无视频而串行首例含视频，且不同初态/原生批量RNG，不能解释为严格单因素加速比。启动27.83秒单列；active lane每tick视觉/历史/后端调用与完整numeric长度逐一核验，最终目标误差0。每组环境81.69/82.48秒、控制器32.54/31.40秒，仿真/渲染/传感器仍主要开销。

工程队列A2/4、B3/4出现稳定窗，各0/4成功、0原生对孔/子目标；最终均失稳。它与原4开发分母分开，不把不同初态抓持数增加叫训练改善。统一阶段报告绑定plan/model/numeric SHA：evidence/tight_fresh_20261114/round02_batch4_runtime.json。共享phase工具允许显式lane numeric/result输入，9项现有测试通过，实际8lane记录/身份/逐帧计数全部验收。

batch worker退出后，冻结b9b6dcb的takeover_03在10:24启动，11.44秒初始化；按已登记另12训练family×A/B的真实早期接管执行。前两轮21成功纠偏继续保留。此时仅启动采集，未宣称第三轮训练或自主收益；原13:59:40截止不变。

## 2026-11-14 实测视觉影响/完整批量验收与多family早期纠偏准备

公开当前A/B raw-intermediate-image与独立direct-backend视觉干预，模型输出确实变化；不作物理成功收益声明。完整四lane DINO/DDIM/Transformer与serial真实输入最大命令差A7.28e-7/B7.97e-7rad、独立reset通过，各25真实帧。新增tight_fresh_dagger --family-offset，显式从原30选择未在前两轮接管用过的family，保留默认0与原分割；下一轮计划offset12、12机会/组、80–300tick抓取阶段纠偏。此处仅登记研究决定，采集训练未开始。

## 2026-11-14 第二轮自主结果分析与发布

实际pre-control recovery完成8例，无新训练、同pair/case/源代码，11.29秒启动。发布round02_phases/runtime绑定真实数值轨迹和模型：两组各1/4短暂拿稳、0/4成功，全601timeout、0对准/子目标；A12.10/B12.17tick/s。首对视频显示抓取推偏后继续空手运动，命令幅度没有整体衰减。不将离线改善称自主收益，不把新抓持初态换位称稳定改进。原失败启动保留，全部8例自然结束。

## 2026-11-14 第二轮后端结果与有界启动恢复

公开round02_learning/increment_baseline/learning_curves，实际两组后端10遍1131更新，选第10遍；验证手指A0.0181267/B0.0195016rad。自主Kit启动超300秒、0实际case，保留日志/中断凭据，核验后仅停止本worker。新增tools.tight_fresh_eval_recovery：只能恢复尚未初始化/开始case且旧worker已退出的自主队列，原cases/pair完全一致，独立输出、原deadline、startup180秒、faulthandler90秒。2项CPU测试通过；不重复训练、不预告自主改善。原活动a4db8e4源码未修改。

## 2026-11-14 抓持持续性与同一时刻对孔诊断

扩展tools/tight_fresh_phases：在原成功/稳定窗规则不变的前提下，报告稳定窗连续长度、末次稳定和首次拿稳后的第一目标进展；最近中心和最小keypoint误差附同一tick角度/接触，避免把不同时刻最小值拼成对准。16项阶段/接管测试通过。真实round01 A稳定23tick、B累计63/最长54tick，最终均失稳。B中心最近341tick仅0.91mm，但角度1.363rad且没有双指接触；最小keypoint差279tick仍高于容差18.84mm。接近孔中心不能称对准或插入。公开round01_grasp_progress绑定原numeric/result/plan SHA，无新增仿真。

## 2026-11-14 批量DP/后端推理准备与CPU验证

visuotactile/native_controller拆分观察更新、计划读取与后端输入/组成，serial act保留同一运算路径；models/native_dp.predict支持每lane独立generator列表。batched_controller新增可选neural_batch和嵌套state合并，tools/tight_fresh_batch_gate/eval可登记启用。此前仅DINO批量，DP/后端逐lane；现在可真正合并DDIM及Transformer，训练/活动round02源码未修改。76项定向CPU测试通过；尚未实测GPU一致性和吞吐。各lane计时重叠明确标注。真实效果待后续冻结源码验收，不预告速度或自主改善。

## 2026-11-14 第二轮A/B前端恢复训练完成

冻结a4db8e4的round_02_v2完成A/B各10遍、510更新，两组均验证选第10遍。每组实际13640anchor/315580合法监督槽：原30各10次（300episode更新/103980视觉帧/204360槽），历史11纠偏各10次（110更新/41130帧/59700槽），本轮10纠偏各10次（100更新/60840帧/51520槽）。所有真实学生前缀只更新历史，未冒充专家标签。

各自同6验证family的归一化指标A 0.01096077→0.01000797（-8.69%），B 0.01131391→0.01057726（-6.51%）。A构型hand RMSE 0.069230→0.066458rad，B命令hand 0.078790→0.076374rad；标签定义不同不横向比误差。新10纠偏训练后的指标A 0.00677697/B 0.00665101，仅为训练过数据的拟合，不能叫自主改善。准确初始数值见绑定report。

真实57条numeric全部字段、原30 affine和冻结DINO内容在失败attempt与恢复attempt之间完全一致；失败48更新未继承。公开round02_frontend_learning和round02_recovery_population_audit。新DP后的全部后端计划/可训练视觉缓存正在重算，后端与自主复测尚未完成。本条只记录已完成前端，不把待运行后端或拿稳效果提前计入。

## 2026-11-14 已有首轮A的中间视觉干预结果

补充公开首轮A已完成的visual_intervention实测报告：固定6验证family的真实本体历史，仅把非规划tick图像/深度改为上一规划帧，DP手指计划RMS变化平均0.001182rad；6例中4例手指误差略变差、2例略变好。证明当前网络确实依赖中间帧，但不足以证明这些帧改善误差或真实抓持。源码/权重SHA与原本地报告一致，0新GPU/训练/仿真。第二轮将在完整pair结束后检查前端和后端各自的视觉敏感性；不因该离线小变化丢弃任何真实控制帧。

## 2026-11-14 逐帧视觉利用诊断与可比曲线

扩展tight_fresh_visual_probe的显式--backend诊断：前端仍把非规划tick的真实图像/深度替换为上一规划帧；后端另外只保持直接输入的可训练视觉特征，固定当前正常DP计划和真实本体/触觉/命令历史。两种干预含义不同，分别报告实际预测变化与合法专家命令误差；只用6验证family并绑定完整pair/缓存SHA，不能将敏感性等同真实拿稳/插入收益。实现已语法检查，实际GPU干预等待当前完整训练结束。

learning_plot补充固定train2诊断曲线，后端横轴改为各所示阶段内epoch，避免把本轮接续训练误标从随机初始化。旧真实round01报告的离线绘图执行通过，不重写历史证据。当前round02_v2源码仍冻结a4db8e4，以上只用于后续独立诊断。

## 2026-11-14 四环境真实视觉验收入口

扩展tight_fresh_batch_gate支持A/B及2/4 lane，保留lane0在第13帧独立reset、其他lane历史持续的完整DP+后端命令比较。每lane使用真实RGB-D并核对逐帧输入；仅离线控制器验收，不改变训练、不替代真实吞吐或自主结果。现有两lane真实门槛和物理结果保留，四lane实际运行待当前训练完成后执行。

## 2026-11-14 第二轮训练中断、真实边界修复与恢复

09:27实际复查发现round_02在04:40已失败，并非仍在训练；主Agent未及时接续，监督存在约4小时47分空档。A完成48次更新后，在第49条r02_B_formal_050_slow_3_stable_d0的几何抽样触发合法性检查；没有epoch checkpoint，B/后端/开发均未开始。失败现场保留，48次未保存更新不进入恢复模型，不能算第二轮学习收益。

根因是fresh trainer只绑定loss_context，遗漏已有geometry_selection所需supervision_records，导致恢复起点窗口抽到绝对tick361的队列过渡步。不是新数据物理构型无效，也未放宽判据。现在同一records同时限定base/palm/hand几何合法目标，优化前遍历检查所有episode。真实57条（51train+6val）检查通过，18125个合法commit槽；旧选择器只在上述一条抽到非法位置。旧首轮选择均合法。35项恢复起点/几何/聚合/阶段测试通过。

选择新attempt round_02_v2，从上一轮有效A/B EMA和配套后端继续，仍10遍DP+10遍后端、各4开发机会、同51条训练和6验证、同原13:59:40截止。仅SHA/源轨迹/相机/stride完全一致的冻结DINO复用round_02缓存；可训练视觉/全部计划在新DP后重新计算。阶段status现在明确写failed/complete，避免只残留frontend_A造成误判。此处登记修复与恢复决策，尚无新训练/自主改善结果。

## 2026-11-14 第二轮24机会完成与真实后半段纠偏

冻结b0917f1采集全部24机会自然结束、无软件错误，A专家救回4/12、B6/12；10条通过实际控制/标签/图像审计，6084历史tick、2620合法专家tick。A/B各只有1次学生拿稳后触发，两次均专家完成；另22次是在未出现拿稳窗的360tick fallback。14次失败全部601tick timeout、无子目标，且其中几次专家到569–598tick才拿稳；这不能证明状态永远不可救，只证明在原任务剩余时间内没有完成。本轮未改变超时/任务标准，也不将专家救回计为学生成功。

所有trigger凭真实学生前缀复算通过，新增authority分段阶段报告，显式区别学生拿稳/对孔与专家后段进展。15项边界/阶段测试通过。公开takeover_02、takeover02_phases/runtime绑定真实plan/results/numeric/model SHA；raw证据全部保留。完整采集A/B约7.16/7.20tick/s，必要lossless append合计349.04秒、最终重构审计415.34秒，普通自主评估不含这些成本。

下一轮51条训练episode=原30+首轮11+本轮10，采用已选10遍DP/10遍后端接续，全部新DP后端计划/特征重算，仍6验证和原4开发机会/组。源码完成审查后在新冻结工作树运行；本条不将启动或待学习数据冒充已训练收益。新report在真实首轮产物重新运行，A/B原始/纠偏曝光与先前报告完全一致。

## 2026-11-14 第二轮纠偏曝光分类修复与接续决策

真实B/formal_051在完成第185步拿稳，继续72步、257动作处接管，专家437步完成；接管前最后学生时刻已失去稳定窗且无对孔。A/formal_046在259步拿稳，再走48步、307动作处接管后由专家完成。两条是学生自身后半段偏差的真实恢复，不能算自主成功。第二轮24机会尚未全部完成，不提前宣称最终救回率。

复核聚合实现保留第一轮11条及其真实mask/历史，只追加新成功后段，train30 affine与独立6验证不变。发现learning_report仅用new_training_ids区分来源，第二轮会把历史纠偏错误算作原始示范；现改以normalization_train_ids识别原30，另外报告原示范/历史纠偏/本轮纠偏实际曝光。指标cohort仍对应trainer实际本轮纠偏指标。28项接续、边界、报告与聚合测试通过，不重写已发布第一轮证据。

基于首轮后端5遍后train/validation仍下降、手指误差尚高于上一真实目标基线，选择第二轮A/B共同10遍DP与10遍后端接续；沿当前已训练参数、新optimizer，在新DP全缓存重算后训练后端。当前24机会全部自然结束并审计后才启动；原4开发机会复测，test不读。此处是研究决策，不预告改善或声称已有真实曝光。原session13:59:40截止保持。

## 2026-11-14 完整自主运行的实际分项耗时

新增tight_fresh_profile_report从已结束队列逐case绑定plan/results SHA，核对DINO、每帧视觉历史和后端调用次数均等于真实控制tick。round01自主A2404tick/196.34s=12.24tick/s，B2192/177.10s=12.38tick/s；每组包括首例视频，Kit启动不计入这些case时间。A/B环境step总114.91/104.77s，其中render61.59/56.11s、physics3.74/3.46s、scene更新8.01/7.34s。这些timer存在嵌套，不能相加为额外成本。普通自主没有专家前向、没有lossless RGB-D训练录制，保留数值证据；报告绑定逐帧调用计数而非只声明保存图像。

当前takeover_02首A实际601tick/95.20s，环境28.74s、实时lossless append26.80s、最终完整重构校验10.28s、专家1.82s，说明采集比自主慢主要还包括必要图像入库/校验。DINO wall与depth-transfer存在异步同步归属，不能把DINO单独2.63ms当其全部设备开销。没有移除中间帧或完整标签审计。第二轮采集仍进行，不能提前发布全队列平均或救回率。

## 2026-11-14 同纠偏历史完整pair复算与后端接续准备

旧/新完整pair在同11条真实纠偏重新计算各自DP计划和可训练视觉特征完成，3045合法专家tick，初段264tick。A手指命令RMSE全后缀0.026079→0.024201rad、初24步0.024941→0.022129；B0.025957→0.022419、初段0.024563→0.019699。手臂也下降。旧模型未见这些纠偏，新模型已训练，故这是同状态纠偏拟合，不能称独立泛化或自主收益。冻结ab79b87诊断0更新/0仿真，结果绑定模型/缓存/数据，公开round01_recovery_comparison.json。

首轮后端从随机参数只训练5遍，末轮train/val仍下降。本轮准备显式continue-rear：在新DP全部缓存刷新之后，沿当前已选后端参数、新optimizer继续，保留同一命令affine/执行器/输出/损失/历史合同。准备时验证旧完整pair身份与全部buffer；运行时再次验证新DP cache/新数据/合法mask。初始从零训练及已完成首轮不改。13项缓存/continuation/阶段测试通过，真实A/B旧后端在CPU通过参数/统计buffer/当前DP合同加载预检（0更新、0GPU）。首次预检fixture在A之前未建立B路径，保留失败并用v2修复；非模型或训练故障。另新增保存trigger回执/实际边界篡改拒绝测试；这是下一轮可选择的实现，尚未启动第二轮训练。新阶段接管已在冻结b0917f1运行，未热改。

阶段采集报告新增从真实numeric学生前缀重算trigger的审计，实际switch必须等于因果schedule；聚合训练前也强制执行。结果按拿稳后接管、无拿稳fallback、延迟前fallback及提前终止分类，专家成功只计真实专家后段，不混学生自主成功。

## 2026-11-14 首轮完整自主结果与按拿稳触发的下一轮接管

round_01_v3全部8次自主已自然结束，A/B仍各0/4完整成功。按同一12tick双子步双指接触、升高2cm、掌中相对位姿稳定诊断，两组各1/4拿稳（初始各0/4）；只发生在development_28140000，不能称稳定可复现成功。A该例230步首次拿稳后超时；B204步拿稳后389步掉落。其余6例未出现稳定窗；全部无原生对孔/对齐/子目标。A四例均601超时，B一掉落三超时。数据见round01_phases.json；seed/reset配置相同不代表精确物理轨迹配对。

下一步选择真实后半段覆盖，而非直接启动已准备的额外rearfit。新增可选stable_grasp_then_delay_v1：只读取已完成的学生transition诊断，完成ticks25–48的物体高度中位数冻结后，按与阶段报告同一稳定窗规则触发；稳定窗结束后再让学生走0/24/48/72/96/120tick，最晚360tick接管。阈值不进入学生网络、不修改任务成功。没有稳定窗单列fallback，提前自然终止保留、不补换。12个训练family各A/B一次，延迟列表重复两遍，共24个机会；开发/验证/test均不进入采集。

调度只允许在已完成学生动作后、下一次专家队列提交前一次修改未来接管tick；真实控制source决定实际边界，专家队列冲洗mask仍按实际切换计。训练entry现在写真实switch_tick，另存登记fallback和触发凭据。每步学生视觉/历史仍实际更新，专家后段重新真实观察。42项触发/队列/完整录制/真实历史监督测试通过；本条写入时新动态接管还未实际运行，不作可救回或学习收益声明。原deadline与所有失败证据保留。

## 2026-11-14 首轮纠偏两组完整训练完成，开始自主复测

round_01_v3固定66be2ac：A/B各完成10遍DP、410更新、11380anchor/264060监督槽，选中第10遍。原30条各10次（103980视觉帧、204360槽）；11纠偏各10次（41130真实历史帧、59700合法槽）。实际前缀进入视觉/历史，mask不把前缀变成监督。

各自新DP后的47条名义计划/可训练视觉缓存已全部重算。A/B后端各随机同seed完成5遍、486更新、67215合法监督tick；真实checkpoint逐episode计数为原30各5次/51990tick，纠偏11各5次/15225合法tick及20565历史tick。统计绑定原30，不用test。独立2062tick命令误差A arm/hand 0.001054/0.023295rad，B0.001076/0.022583；手指仍高于保持上一真实目标0.019415基线约20.0%/16.3%。训练与验证均继续下降，尚未证明拟合充分。

公开round01_learning、round01_increment_baseline及可复盘PNG绑定实际选模、数据、回执和cache。上方DP不同标签空间只作组内前后比较，后端共用命令定义。完整自主4机会/组已进入真实Kit，启动35.46秒；此时未完成，不能把本离线变化写成拿稳/插入改善。

资源快照显存空闲约22GiB、RAM约32GiB、磁盘约321GiB，旧GPU50%限制不适用。现有monitor只在启动时读取owned roots，因此把新嵌套root worker误列为external；整卡采样和full_gpu储备判定仍有效，但不能用这个分类宣称外部进程占用。实际自有worker由阶段命令/源码/PID核实，未重启monitor制造新缺口。

## 2026-11-14 后端拟合延长的隔离准备

初始5遍后端的训练与验证手指误差均持续下降，末轮仍高于实际上一目标基线；本轮A新后端同样5遍0.029215→0.023295rad，已完成486更新/67215合法监督tick。尚不能把当前失败归为纯泛化或训练过拟合。先完成既定B后端和完整自主，再根据结果选择下一步。

新增tight_fresh_rearfit供主Agent明确选择一次匹配的后端继续拟合：固定当前已选DP、数据、mask、实际逐帧缓存与共同affine，新的完整pair接续后端参数/新optimizer，然后实际完整学生开发。原deadline不变、旧证据保留、缓存错DP/错数据/错mask/错affine拒绝。没有自动启动该实验，也不把准备当成训练进展。12项准备/缓存绑定/合法监督/阶段拒绝测试通过。当前66be2ac活动源码不变。

## 2026-11-14 后端单步误差的实际上一目标基线

新增只读tight_fresh_increment_report，直接读取同一离线评估tick的真实obs/actuator_target_rad与transition/target_after_rad，检查跨tick连续性，再和已保存的后端误差比较。初始独立验证2062tick：保持上一实际目标的手指RMSE为0.019415rad，而A/B完整后端在专家历史上的RMSE为0.024247/0.023381，即高24.9%/20.4%；接触低位阶段高12.1%/10.8%。手臂保持基线0.001178rad，A/B约0.001098/0.001120。此前hold是规划时刻实际q，其物理含义不同；原报告不改。

该结果不证明保持不动能完成任务，也不是后端导致自主失败的因果证据，但说明仅凭很小的单步rad误差不足以认定学会正确手指调整。后续用当前轮更新后相同验证tick复算，并结合完整自主拿稳结果决定是否需要更多后端拟合或真实偏差覆盖。

另准备tight_fresh_recovery_report：对同11条真实纠偏、旧/新各自完整pair重算计划和可训练视觉特征，比较合法专家后段及最初24个合法tick；旧模型不能借用新DP特征。只计算训练集纠偏拟合，不能称独立泛化；本条写入时尚未执行，当前GPU仍用于既定round_01_v3缓存/训练。A/B前端已各完成410更新，独立验证selection分别下降15.30%/12.13%，完整自主结果待评估。

## 2026-11-14 实际学习曝光与选择模型可复盘报告

新增只读tools.tight_fresh_learning_report，逐条检查连续optimizer回执、训练身份、累计anchor/监督槽、完成回执与selected模型SHA；分别统计原示范和纠偏的episode/视觉帧/anchor/监督槽实际曝光。验证指标绑定所选epoch，末轮纠偏指标明确标注是否等于selected，避免把最后一轮误作部署模型。后端附选模曲线、共同命令空间离线误差和当前DP cache SHA；原日志validation4文字与真实6family区分。

已对真实初始A/B完整产物运行：各600次episode更新、207960视觉帧曝光、17560anchor、408720目标槽，原30条各20次；后端各51990监督tick/406更新。初始统计不含任何纠偏训练。报告initial_learning.json保留源回执/数据/统计/模型摘要，不打开test。

当前round_01_v3的A已完成10遍/410更新/11380anchor/264060槽；独立验证q_next arm/hand 0.012119/0.075698→0.011236/0.069230rad，11条纠偏0.019523/0.103581→0.008089/0.058051rad。B与后端/完整自主结果仍待完成，不能据此宣称拿稳改善。活动66be2ac源码未改；本报告工具仅在审查树运行。

## 2026-11-14 拿稳与后半段失败的可复盘分类

新增只读tools.tight_fresh_phases，以真实object/palm/两物理子步object-contact和原生goal事件复算阶段。原稳定窗阈值保留，另明确报告settled高度参考（完成ticks25–48中位数），因为reset起始物体在空中。该诊断不改变原生成功，也不以单一lift flag代替拿稳。八个初始自主轨迹在两种参考下均未拿稳、原生子目标0。

12个接管都在专家执行后出现稳定抓持；所有首次稳定窗均晚于接管tick。故46–180的时间覆盖不能宣称已经覆盖“学生自己拿稳后的后半段”，后续要依据新完整学生实际推进再安排。唯一失败B/formal_047/t70并非一直拿不起：117步出现稳定窗，338步达第一原生子目标，530步掉落；是专家接管后的后半段失败，不等于该接管状态完全不可救。其完整轨迹仍保留且未纳入成功监督。不同tick绑定不同family，不能从本12机会推断接管时刻因果优劣。

两个真实队列的阶段JSON由同一工具生成，绑定numeric/结果/plan SHA；首次原生对孔条件、子目标、拿稳窗和跟踪误差可用于后续统一比较。首次纠偏训练已实际开始；A首遍41更新、1138anchor/26406监督槽，初始纠偏q标签误差arm0.019523/hand0.103581rad，高于原validation0.012119/0.075698，更新后及完整自主效果待检验。

## 2026-11-14 聚合实物审计与缓存worker资源入口修复

round_01_v2的CPU聚合成功：A/B各41训练episode、14511真实历史tick、13443监督tick，原30 affine SHA不变，原验证6不变、test未打开，两组实际输入与mask逐张量相同。恢复起点窗口合同实际提供1138 anchor/26406目标槽（其中纠偏260/5970）；采集收据的254/5838是旧默认窗口计数，不能误作训练实际曝光。

随后视觉缓存GPU准入前失败：每role缓存worker需要自身shared_resources.json，聚合只给顶层建立了链接。0训练更新，失败现场全部保留。现为A/B子root显式链接原活跃monitor入口，并新增准备到缓存入口的集成测试；不复制过期monitor快照、不放宽资源门槛。使用round_01_v3重新准备，同一12机会/11成功后段/原截止，之前attempt曝光均0。原30统计与实际mask审计保存在round_01_v2/population_audit.json。

## 2026-11-14 纠偏准备故障与显式无触觉前端修复

round_01在CPU聚合准备阶段失败，0参数更新、0新仿真。旧聚合入口强制读取独立tactile.pt，但本轮G00视觉/本体前端本来不生成或读取该文件；真实后端触觉从原HDF5按控制时刻读取。不是缺失真实传感器，也不是训练失败结果。

保留round_01及round_01_stages全部现场。聚合新增显式include_frontend_tactile=False，仅用于本轮无触觉前端；不伪造零触觉、不引入旧normalizer，后端真实触觉路径与历史G10默认不变。新attempt使用round_01_v2，同一11条纠偏、同一原session deadline与10/5遍预算，首attempt没有可计入的学习曝光。主Agent继续修复后推进。

## 2026-11-14 真实首轮接管全部完成，纠偏学习启动

冻结01b8805的12机会全部合法完成：A专家救回6/6，B5/6；B/formal_047/t70接管后530步掉落，其余11条均真实完成原任务。11条通过控制/label/传感审计，合计4113实际历史tick、3045合法专家tick、254规划anchor、5838监督窗口槽；未训练前只是源人口。学生前缀与队列过渡均mask0，实际本次队列过渡两tick；失败数据保留但不进入成功监督。这些是专家救回，不能算学生自主成功。

基于实际救回覆盖接触前/接触后/120与180tick，选择一轮共同10遍DP+5遍后端、原4开发reset机会复测。源563d2c2已推origin，执行树fresh_round01_20261114冻结；round_01_stages记录每阶段命令与结果SHA。先聚合共享11条纠偏，保留train30统计/validation6/test22隔离，再新DP→全后端计划和特征重算→新后端→完整自主执行。主Agent持续检查实际结果，当前没有纠偏后自主收益证据。

公开小证据evidence/tight_fresh_20261114/takeover_01.json由tools.tight_fresh_report生成并绑定采集plan/results/entries摘要。

## 2026-11-14 首轮真实DAgger训练链准备

真实接管使用冻结01b8805，12个登记机会正在自然结束。已经观察到早期和120/180步接管可救回，也有70步B接管后掉落；最终分母以takeover_01/result.json为准。成功只代表专家救回，不是学生自主成功，失败原始记录保留。

训练链新增tools.tight_fresh_cycle：一次明确选择的10遍DP/5遍后端/各4开发机会，逐阶段保存源码、命令、完成SHA、失败及原session deadline。脚本不决定下一研究路线；主Agent继续分析实际学习曝光、可比验证和拿稳/转正/对孔/插入结果。DP沿当前EMA继续，后端随机同种子重拟合，故本轮不是仅前端变化的因果实验。原验证6及test22隔离，原30统计不重拟合。

修复随机新后端聚合时错误要求归一化人口等于全聚合人口的问题；现在明确识别fresh聚合合同，保留原30统计，全面验证新DP/数据/normalizer/cache与真实label mask。36项fresh/affine/sampler测试及28项阶段失败/后端训练合同测试通过（两批有一项重合）。新source尚未用于本轮训练，活动采集未热改。

实际A/B模型三真实condition、四组相同noise的DDIM legacy/host索引搬运比较八对，计划最大误差0。常见0.10–0.11→0.088–0.100秒，存在并发/预热噪声，不作闭环吞吐改善声明。未来新自主计划显式登记host模式；旧采集plan仍legacy。公开采样证据sampler_probe.json。

## 2026-11-14 首轮完整学生结果与真实接管登记

A/B各4个新开发初态全部合法完成，完整成功均0/4；A有2fall/2timeout，B有4timeout，均无稳定抓持、原生对孔或子目标。首次接触约51–71步，与成功train30的55–89步接近；主要偏离在接触后的抓持。native_lift只覆盖11/30成功专家，不能单凭false说未离桌。按settled高度和原相对位姿/双指稳定窗复核，八学生仍均未拿稳。A000和B000的短暂升高分别无接触和仅一指，不算抓稳。

真实双环境A/B各2lane自然结束，2358有效控制步与逐帧视觉计数相等、提交/实际target最大误差0。吞吐A16.21/B16.39 tick/s；无视频串行六例约12.7tick/s。约28%提升是不同轨迹的工程吞吐比较，非严格配对。DP/后端仍逐lane，RTX/传感器占主要时间；未改分辨率/控制时钟。真实录制的完整控制器串行与批量25tick、单lane重置边界命令逐位相同。

60项fresh合同、真实历史mask/梯度、接管/聚合/采集测试通过；compileall及diff检查通过。

下一首轮接管固定六训练family、A/B相同机会与交错次序，46/58/70/84/120/180tick；完整学生真实前缀持续更新，专家接手后真实重新观察并执行。只纳入成功、传感/控制/label审计合格的真实专家后段；学生前缀和队列冲刷mask0。所有失败保留。成功纠偏共同供给A/B，原30统计不重拟合，验证6/test22隔离。

实现新fresh聚合合同、当前DP EMA显式继续训练、新optimizer/EMA、完整真实前缀、恢复起点H24合法窗口；新DP后全部后端计划/可训练特征重建，后端同seed重新拟合，不能继承stale计划。当前只是登记，真实救回、纠偏训练和自主改善尚待产物。入口tools.tight_fresh_dagger；本轮继续监督、不把启动当完成。

## 2026-11-14 真实batch离线gate首轮拒绝说明

首轮在执行任何策略tick前被环境合同拒绝：原专家数据metadata是native command模式，而离线比较器构造绝对目标控制器。没有真实仿真开始，不是任务失败。新attempt显式声明仅离线控制器的绝对目标模式，保留原source_control_mode，原action参数与dt照常严格检查；实际在线环境校验未放宽。首轮log保留，新结果另用batch_controller_gate_v2.json。

## 2026-11-14 初始DP20遍完成、视觉依赖与分类清理

A/B均从随机初始化完成20遍、600更新、17560规划点和408720监督目标槽；初始train30有10398真实tick，36条训练/验证合计12460tick。两组独立验证继续下降，A20验证arm/hand为0.012119/0.075698rad；不同标签空间的loss不用于A/B排名。A后端已完成5遍51990监督tick/406更新、30条各使用5次，原6family全部用于验证（旧固定字符串误写validation4，实际saved validation_ids和代码为6；保留原收据说明）。B已开始新缓存和相同后端预算。

A离线中间帧干预确实改变condition与计划，但变化很小：arm计划RMS约0.00017–0.00020rad、hand约0.00091–0.00102rad，误差变化方向不一致。只能说这些帧进入并影响模型，不能说已证明抓稳/转正/插入收益。后续需真实偏离状态与完整闭环检验，而不是把逐帧接口通过等同于有效视觉利用。

按明确新授权清理旧20261113根18个last checkpoint和1492个名义衍生cache，删除逻辑6,611,324,067字节；唯一链接块约4,284,522,496字节，观察空闲空间增加4,282,011,648字节（同期研究写盘影响差值）。旧selected best、原始轨迹、初态导出、成功专家、DINO与当前run完整保留并校验；本地删除清单逐文件含SHA和inode，小统计在evidence/tight_fresh_20261114/cleanup.json。

多lane当前完整策略新增真实图像/真实专家历史的串行对照入口，包含仅lane0重置而lane1继续的边界；尚待运行验收。所有初始完整学生自主结果与DAgger仍未产生，不能宣称路线成功。

## 2026-11-14 多环境执行实现与待验收边界

新增BatchedResidualController共享冻结权重、按当前时刻批量DINO并保持独立lane历史/随机规划器/执行回执；DP与Transformer暂仍逐lane计算。原单环境接口不变。新的lane reset分派逐lane调用原生reset参数、无额外物理状态强写/全局重播，碰撞准入读取各lane实态。批量原生RNG顺序与串行不同，不许称精确配对。

新增有限batch评估入口保存各lane自然结束/真实命令/实际关节/接触与阶段，终止lane不重复计数；普通自主执行没有专家推理。另有训练后中间视觉帧干预入口：固定真实本体历史，把非规划tick的图像改成前一规划帧，比较预测和误差。该干预只能证明离线依赖，不能替代物理抓持/插入效果。

39项lane路由、消耗帧保护、reset参数隔离、DINO、窄域与位置执行/准入测试通过。真实多环境Kit验收、速度与视觉依赖结果均待运行，当前仍不宣称加速或机制收益。活动后端使用冻结2f9c62a；本次改动只影响后续独立新worker。

## 2026-11-14 完整学生评估入口与控制链复核

新入口tools.tight_fresh_rollouts只接收训练完成、非test离线证据与best哈希一致的完整DP＋Transformer；固定新开发初态，A/B交错顺序，不替换非法初态。每条保存源码/两模型哈希、原生结束原因、拿起与拿稳诊断、逐帧推理和仿真分项计时。相同seed只表示同注册机会，不声称隐藏物理状态精确配对。真实完整学生运行仍待新后端完成。

控制链核查：A的q_reached[t]是该transition结束的q[t+1]；专家目标是同transition实际postprocessed目标，均非最终稳态。当前/未来监督不进学生输入。后端以实际上一执行器目标计算专家同速可达区间；先限制名义锚点，再加可达跨度乘线性提案，最后投影一次。原始DP计划仍作为后端输入，但受限锚点不能解释成未经修改的几何意图。绝对目标提交没有再次叠加延迟/平滑；保留原延迟随机抽样，原生expert队列通过显式bridge处理。

成熟实现对应：vendored DP U-Net与EMA来自已固定MIT源5ba07ac；本轮sample/x0训练、DDPM100、DDIM16、H24/commit12，加入本项目逐帧RGB-D与持续GRU，不能宣称官方DP复现。Play2Perfect原action_utils的arm delta/clamp/EMA与hand absolute-affine/EMA保持；本轮显式slow expert及学生共同可达约束不是原专家无约束等价。观测恢复和命令精确不证明PhysX完整重放。已有真实采集/延迟/历史单测支持这些接口，但新学生在抓持、转动和插入的收益尚无闭环证据。

首次A20遍已完成600更新/17560规划点；独立6family验证q_next的arm/hand RMSE为0.012119/0.075698rad。B同预算接续，不能用A/B不同标签空间的loss排名。后台使用冻结d45f11b；本次评估入口与后端将在独立新冻结源码运行。

## 2026-11-14 首次共同5遍完成与逐帧评估计时

A/B各完成5遍、150 optimizer更新、4390规划点、102180有效目标槽，随机初始化与实际曝光一致。两组固定验证误差持续下降；基于此继续共同20遍，原5遍模型与收据独立保留。首次5遍仅离线证据，不是拿起、拿稳、转正、对孔或插入成功。详见evidence/tight_fresh_20261114/frontend_epoch5.json。

普通完整学生评估关闭未执行专家shadow；DAgger仍保留接管所需真实专家队列。在线异步计时细分robot历史输入、RGB传输、DINO、depth及传输、可训练视觉/历史、DP、后端输入及后端推理，不新增CUDA同步。NativeDinoSpatial增加多lane当前帧入口，仍按相同六图kernel配对，无跨lane/未来图像混合；仅此接口不代表多环境评估已实现。

51项评估/性能/后端/逐帧回归测试及17项DINO lane/因果视觉/评估测试通过。真实逐帧吞吐与训练后视觉利用仍待完整学生运行；旧70.88秒轨迹采用稀疏视觉，不作为本轮速度基线。活动DP训练树保持d45f11b冻结。

## 2026-11-14 新后端与共同train30统计

新后端准备入口`tools.tight_fresh_backend`只接受明确fresh30/every-native-frame身份，保持原历史166条入口。A/B后端随机同种子初始化，共用B的train30命令normalizer，绑定各自当前DP SHA、全部逐帧缓存、数据与统计哈希；验证覆盖6个独立family。现有Transformer/每步实际反馈/当前权重前缀重算继续复用。

本轮选用已实现的可达区间中心化线性提案和span FY损失，以避免硬裁剪外零梯度及关节间极不均衡的尺度；这是工程选择，尚未证明任务收益。DP预测的完整计划仍进入后端；提案锚点限制到真实上一控制目标对应的专家可达区间，最终绝对目标只投影一次，不重复延迟或平滑。不得把锚点称为最终稳态构型。

86项归一化、输入合同、残差、span投影、中心化及tick-window测试通过。A已完成首次5遍/150更新；B同预算在运行，尚未生成新后端或完整学生闭环结论。活动DP使用独立冻结d45f11b源码，本次改动不热改worker。

## 2026-11-14 实帧GPU验收与批量视觉编码

新增continuous_vision.encode_dense只批量独立帧的卷积/注意力读取，每个原生tick仍按顺序更新GRU。真实25tick三视角输入的批量/逐步condition最大差9.53674e-7，0/1/11/12/24帧原图重算DINO及depth均精确匹配缓存。一次局部编码计时0.699→0.162秒；含首次kernel开销且缓存worker同时运行，不能当完整评估加速证据。63项聚焦接口测试通过。新完整episode训练器使用随机同种子初始化、train30统计、每epoch每episode/每规划anchor各一次，实际前缀按当前权重重算，验证用固定6family并单独报告train2。

下一阶段首次共同5epoch只是研究检查点，不是收敛或成功承诺。A保留已接受的手部几何物理参数与loss系数；该系数来自历史训练校准，和本轮重新计算train30归一化统计明确区分。B与A均保留共同掌部位姿项。先通过真实GPU输入gate，后按冻结源码启动；未完成训练时不宣称闭环改善。

## 2026-11-14 当前重启：逐帧视觉与train30

本轮授权替代旧截止、旧初始化和只启动后返回要求；当前指令集中在根AGENTS.md。研究机CNS202533182702；远端feature/tight-state-foundation核验为506d185，旧主工作区3e11d62及其未提交文档未改。新隔离树continuous_vision_20261114。

确认旧视觉每12tick一次，后端重复旧visual。新独立version6/visual_stride1合同把每60Hz真实RGB-D送入训练、在线视觉GRU和后端缓存；DP仍5Hz。旧合同保留。q_reached为transition后的q(t+1)、间隔1/60秒，不是稳定姿态；u_expert是当步实际执行目标。新模型尚未训练，不宣称任务改善。

原Core有36训练family；一家族一条中位时长完整成功，再按实际初态标准化最远点选择30条，原validation独立family取6条，test22未打开。36条共12,460tick，原文件SHA和q_after/q_before时序逐条通过；统计只用train30，未加载旧学生权重。原始数据与专家保留。详见evidence/tight_fresh_20261114/selection.json；标准化混合坐标距离只用于选取与去重，不作物理精确重放声明。

实现当前处于接口验收/全帧缓存准备。初始完整A/B训练、后端、真实多环境吞吐及DAgger研究循环尚未完成，不能把准备记为科研成功。新本地root：outputs/visuotactile/run/tight_fresh_20261114_01。

---

以下为历史记录，当前状态以上文和机器收据为准。

## 2026-11-13 23:35 剂量实验终局与授权边界

- 完整40机会全部合法，四cell各0/10成功、0首子目标；不采用端点，不追加训练/评估。
- 明确continued各2620新rear更新/387040tick，reference零新更新及四cell零DP更新；冻结前端终局公平性审计均通过。
- 保存继承收据解释、泛化/物理复现限制与primary交接；不改活动源码或bot独占文件，原截止不变。

## 2026-11-13 — 真实运行计时与numeric批写

- `visuotactile/recording.py`新增BufferedNumericWriter：同schema/dtype/value的32tick批写及显式flush；旧append_numeric保留。
- `visuotactile/state_benchmark.py`默认使用bounded writer，成功统计读取前和finally刷尾；自主评估也输出wall_profile/setup/总耗时。控制、标签、60Hz/12tick、compact训练记录不变。
- `tools/tight_runtime_probe.py`为一次性有限计划诊断入口，复用Kit并测量真实完整策略/命令回放，无新服务；保留首轮退出助手异常并修复root完成回执/错误退出。
- 23定向测试；274tick真实132字段逐数组相等，4.80→0.42秒。8次真实诊断完成；A601tick numeric13.05→1.43秒，总81.58→70.88秒，物理轨迹非精确配对。原B成功没有复现，所有失败保留。
- 更新DAGGER_METHOD/SESSION_HANDOFF及小证据runtime_review.json；原始数据、模型与旧结果不变，现有正式worker未热修改。基线诊断进程退出残留的短暂资源竞争明确排除速度结论。

## 2026-11-13 22:23 自动页拒绝机会与阶段保留修复

采样对照四cell各0/10已完成；后续span metric对照四cell各10个机会中9合法、1初态拒绝、0成功/0软件失败，不能当成10个合法失败。当前`rear_span_dose_02`按原登记A/B共同追加两遍后端训练，再各自与冻结参考做同10新机会；本轮尚无新闭环结论。前端/cache不更新；313冻结源码文件SHA无漂移。A/B准备配置及评估spec一致，最终实际曝光仍待完成审计。

修复独立CPU发布器：初态拒绝无result checkpoint字段，必须核验receipt与frozen plan/checkpoint绑定及not_started，单列拒绝分母；合法试验仍要求模型哈希。完成阶段保留，继承训练收据标记非本阶段新增，实时训练进度仅读本阶段时间戳。9项测试通过，真实20份summary全部验收，原漏报4份均恢复9合法/1拒绝。原失败证据和bot历史提交保留。

新源码在隔离树审查，发布器在共享Git锁的空闲边界切换独立新runtime；活动训练/监督器不改。原计算23:55:15/绝对次日00:05:15不变，真实替换启动看publication_fairness_01内v2收据。

## 2026-11-13 21:21 — Actual complete-policy recovery progress

- Independent bounded service owner3037113/start_ticks163579189; first reference-A case reaches actual native tick300.
- Verified311 frozen source hashes and full_gpu_v1 memory/disk reserves;63 focused tests pass.
- Publish startup receipts from a separate worktree; retain original40 opportunities and all deadlines. No completed-cohort success claim.

## 2026-11-13 21:18 — Preserve checkpoint-bound offline evidence across evaluation recovery

- Diagnose missing offline admission evidence before any simulator launch; retain the failure and completed A/B1314-update endpoints.
- Reuse complete non-test offline results only with selected-checkpoint and actual data/cache identity checks; never bypass the group-denominator guard.
- Add unique second recovery identity, preparation-only boundary, explicit evaluation entrypoint and partial-start replay refusal; retain all original opportunities/deadlines.
- Both prepared fairness audits and actual four-cell CPU entrypoint preconditions pass; no GPU inference or autonomous result is claimed by these checks.

## 2026-11-13 21:01 — Safe evaluation-only recovery

- Fix cross-metric audit to recognize derived loss_id; no hot edit or interruption of healthy trainers.
- Register bounded evaluation-only successor with original deadline, frozen trained endpoints and unchanged40 opportunities; reject already-evaluated or unrecognized-failure sources.
- 74 focused tests pass, including3 recovery guard tests; real training verified separately from evaluation.

## 2026-11-13 20:53 — Replay boundary and reachable-span metric

- Recorded four valid replay cells, all0/10; passing A/B audits and actual population coverage662/662.
- Added opt-in span-normalized FY, preserving executor/statistics/default losses and direction-correct nonflat gradients.
- Registered matched A/B candidate versus frozen population references: reuse reference training, fixed40 new full-policy opportunities, no DAgger/automatic expansion.
- Real preparation, two prepared fairness checks and71 focused tests pass; startup/completion remain separate receipts.

## 2026-11-13 19:23 重放实验真实启动与独立发布

源commit74c8006已发布，独立限时service owner2967509/start_ticks162865736；实际288更新/41271监督tick，308受检源码无漂移。只在独立publication树增加启动收据与交接，没有热改worker、监督器或两bot专属文件。原截止与固定40机会未变，下一判断仍交真实Agent。

## 2026-11-13 19:20 冻结前端的后端重放分布实验

A/B补齐复验各0/20已确认，保留原分母和负结果。新增有界`tools.tight_replay_balance`：原50/20/30与人口比例后端续训对照，两个arm都训练A/B，固定40完整自主机会；无新DP/采集，不称DAgger。沿用实际训练/执行入口，不改loss或控制器。新增fairness frozen-frontend模式及测试，区分声明/真实完成与历史DP曝光。28测试、真实CLI prepare、两arm准备审计通过；人口比例真实记录预采样662/662独立episode，仅作为设计证据。启动状态另行记录。独立源树与原绝对截止不变。

## 2026-11-13 18:48 A补齐真实启动验收

独立限时service owner2945216/start_ticks162656356，首例已实际tick400；冻结源码无漂移、资源储备通过、29项相关测试通过。source c8d3dd3已发布；本次仅另树发布小型启动证据，未热改worker。下一事件交既有真实Agent监督器。

## 2026-11-13 18:45 B复验收尾与固定A补齐

- B contact/previous各完成20合法自主机会，均0成功，保留此前B1/10独立分母；不称收益已复现。
- 新`tools.tight_ab_replication`登记contact A对应20机会，不重复B，不训练；hash绑定完整pair/原计划/B结果，前后公平审计、真实机会核对、拒绝不补及独立截止服务。
- 29项相关CPU测试通过，真实prepare入口及实际A/B checkpoint前端绑定通过；当前记录只到登记，真实推进另留startup_acceptance。公共评估源码与B冻结版相同，监督器/发布器不动。
- 历史初始化/曝光不同及A事后补齐顺序局限明确披露；原截止与test22隔离不变。

## 2026-11-13 18:29 自动发布服务实际验收

独立CPU服务 `tight-live-publish-20261113-01.service` 已active，首份自动报告29afde1已核实到达远端；[自动进度页](TIGHT_LIVE_STATUS.md)与JSON由冻结普通脚本持续维护。每60秒检查、有变化通常5分钟内推送，阶段结束/失败优先；无变化不提交，断网保留待推内容，截止北京时间11月14日00:05:15。

当前B复验已完成31/40机会（旧B20、新B11），目前观察到0新成功，最后9例未完成；不能计算完整复验结论或改变原A0/10、B1/10。后续数值只看自动页时间戳。15项测试通过，300份活动worker源码无漂移；没有停止/重启训练或评估。下次科研Agent提示已登记实际服务、完整A/B检查与共享Git锁要求。

## 2026-11-13 自动进度发布与 A/B 增量公平检查

用户授权有新信息自动推 GitHub；今后 A/B 同数据、实际曝光、公共修复、优化器处理、选模与评估机会。新独立 CPU 发布器只维护 [进度页](TIGHT_LIVE_STATUS.md) 和小型 JSON，按实际 receipt/plan/result hash 报告，有变化通常五分钟内发布，完成/故障优先；普通脚本不做研究推理。原绝对截止 11月14日 00:05:15 UTC+8 不变。代码已实现/测试，真实服务启动验收另记。

[公平政策](AB_COMPARISON_POLICY.md) 与只读 gate 对当前 contact round 验证通过：DP 各514更新/765303目标槽，后端各1362更新/193581合法tick，coverage账本及相同10机会相符，A0/10、B1/10。两前端统计依法不同；真正后端 command affine 同为53ecc092…，cache绑定及完整pair匹配。历史权重/曝光不同，当前只能称本轮增量匹配，不能称全历史严格等预算；gate未独立重放全部采样顺序/优化器张量。当前 B 新旧版本20机会/版本复验单列，不与 A 排名，不合并原分母。

15项 CPU 测试通过（真实本地 bare Git 推送、并发远端前进、断网保留提交与恢复、无变化不提交、过期禁止 Git；监督语义/曝光/缓存/种子/缺失证据失败）。首次幂等测试发现snapshot被原地加时间戳，改为副本；补充中断写入恢复，并保留Git porcelain开头空格，防止误读路径。修复后全部通过。300份冻结活动源码 hash 未变；无训练/仿真启动或中断。现有Agent下次调用提示已原子追加公平要求和共享Git锁；不热改监督器。

## 2026-11-13 17:53 冻结复验启动验收

- 独立systemd owner PID2908611/start_ticks162322839；新case实际tick400，源码73b2e97 hash不变，未新增训练或改监督器。
- 子截止19:45:35 UTC+8，所有原截止保留；新复验结果未完成，不提前解释收益。

## 2026-11-13 17:48 contact结果与冻结复验

- 完成contact周期：A0/10、B1/10原生完整学生成功，B tick274；保存独立handoff与成功trace/pair/hash核验，不混入专家接管。
- 新`tools.tight_success_replication`只复用fixed10入口，冻结当前/上一B完整pair并登记20新seed/组，两批交叉顺序，不训练/补机会/修改活动监督器。
- 24项测试及真实prepare入口通过；所有原截止、资源储备、test22隔离保持。启动与进度另记，不用CPU测试代替GPU验收。

## 2026-11-13 控制步学习系统自查

- 新增独立tools/tight_control_phase_audit.py：CPU读取train-only原记录、已提交sampler/episode计数，分别统计phase0–11的合法人口、独立覆盖、重复H24目标、commit12目标、条件输入与历史访问；实际运行并交叉核验checkpoint/完成凭据计数通过，无参数更新。
- 更新DAGGER_METHOD、SESSION_HANDOFF及小型system_selfcheck证据：确认58ed6e1已进入当前新训练；update300旧遗漏210目标已见150，当前每phase约65%独立覆盖；历史采样不足、接管相位集中及编排复杂度分别披露。
- 复用已完成53项窗口测试、6真实EMA/batch和2418缓存tick证据；本轮专家额外查6轨迹3504tick，所有数值链误差0。297份冻结活动源码无漂移，未中断/重启实验，没有新的GPU/仿真/训练或test22使用。
- 没有把覆盖验收或专家恢复当成学生自主收益；最近完整学生A/B仍各0/10。当前子截止和原绝对截止不变；代码整理与采样方案不在运行中切换。

## 2026-11-13 13:13 接触恢复已真实启动

`adaptive_contact_recovery_01`独立systemd服务已active，owner PID2764734/start_ticks160645694，冻结源码c9f8e5e、SHA无漂移。A已完成2个登记机会：首个原生初态碰撞拒绝（不补样），第二个完整专家接管成功、已通过合法纠偏准入；这不是学生自主成功。后续两模块训练/新cache/完整自主开发尚未完成。44项测试与24个实际CLI采集配置解析通过，资源满足原6/14/60GiB余量。

独立子截止19:09:53 UTC+8，比原计算23:55:15更早；原绝对11月14日00:05:15不变，正常完成或故障后由既有真实Agent监督器再决策。现场、启动验收与最后状态位于原run同名目录；未改监督器和任何旧worker。后续下面的“仅准备”段落为启动前记录。

## 2026-11-13 13:12 完整tick-window结果与接触恢复新登记

新增隔离有限contact-recovery编排入口与4项入口/合同测试，整组44测试通过。复用现有采集器、DP/rear训练、缓存与完整评估，不改模型执行器；集成已发布58ed6e1。记录tick-window实际A/B0/10、训练wall365.41/358.82秒及20条contact事件，独立登记小规模当前策略恢复实验。此时仅准备，实际启动以startup_acceptance为准。

## 2026-11-13 真实接管监督覆盖与原时钟onset窗口

- `target_windows.py`/`anchor_sampling.py`：显式opt-in原12tick anchor内前导mask，首个合法连续段监督；补21旧anchor/210独立tick，恢复抽样及target/pair/commit几何分母按实际mask。没有修改在线时钟/执行器/原数据。
- `native_data.py`/`state_history.py`/`anchor_continue.py`/`tools/tight_dagger.py`：训练loss元数据、预算、配置/checkpoint贯通，同run恢复拒绝版本切换，缺字段兼容；新前端才启用。
- `tools/tight_takeover_phase_audit.py`读取真实train mask、已训练checkpoint采样计数、HDF5与6个实际EMA编码器/batch；后端缓存另查2418tick。所有当前76纠偏已对齐；旧21条各缺10tick。后端本次只见194/409条是加权采样事实，不是mask遗漏。
- 53项CPU测试通过17.09秒；首次新测试fixture漏空current stratum导致1失败，修正fixture后全组通过；既有requests警告保持。292活动源码hash无漂移，无新增GPU/仿真/训练，test22隔离。候选尚未训练，不作任务成功声明；证据见DAGGER_METHOD页首与takeover_phase_review。

## 2026-11-13 12:24 显式合法tick优化器窗口

- 启动验收：实际日志116更新、完整group checkpoint111更新/15723合法tick；独立service已运行，292份冻结源码保持。99项定向CPU测试通过；源码/登记已安全合并远端review并推`origin/feature/tight-state-foundation`，无force。新闭环待后台完成，机器decision把下一边界交给既有真实Agent监督，不新增监督服务。

- 新增`residual_tick_window.py`显式协议；固定128未加权合法tick分母，完整chunk累积、group尾部flush，日志区分窗口、物理监督量、实际更新与权重质量；旧字段缺失时逐chunk旧行为不变。
- 参数未变时沿用当前权重detached状态，参数更新后完整真实prefix重放；仅在无待提交梯度的episode-group边界保存。定向CPU测试覆盖梯度分子等价、3倍权重、尾部固定分母、真实trainer和参数更新后的历史刷新；不把CPU测试当GPU训练或任务成功。
- 完整centered闭环均0/10且离线改善。下一登记保留FY/centered/LR/原数据与pair/cache，改变优化器度量/频率，不追加数据、不改最终执行器、不宣称等更新预算或因果成功。新root原run下`adaptive_tick_window_01`，原截止和test22隔离保持。

## 2026-11-13 12:16 训练正确性/效率复查与诊断修复

- 复核真实FY、centered、mask/history、AdamW和断点恢复；没有新方向梯度错误，1e-6rad理想化边界仍明确。训练pair18.76分钟；旧故障接续间隔6.879小时另计。记录实际2237985前缀重放ticks、26.22%padding和稀疏chunk/恢复系数，未声称这些已被优化。
- `visuotactile/native_data.py`、`anchor_continue.py`：无效padding的previous用合法占位，防止正下限关节误拒绝；真实观测/合法监督不变。`state_benchmark.py`：版本化线性输出的旧幅度比例不再称tanh饱和；`models/native_dp.py`纠正delta始终有界的过时注释。
- 不等长轨迹/正关节下限、真实非法previous、两个线性输出的诊断兼容回归在`test_execution_centered.py`、`test_target_residual.py`；修复后28项通过2.37秒，已有55项通过8.58秒且有重叠。仅既有requests依赖版本警告。旧286/新290活动源码无漂移，未启动额外GPU。
- 最新闭环A/B均0/10，lift3/5不计成功。真实后台Agent002已接续并收到效率复查；原截止/数据/test22保持。详见DAGGER_METHOD页首与training_efficiency_review证据。

## 2026-11-13 自主研究接续，完成当前闭环后再决策

- A后端真实7017更新/512402合法监督tick，3.00425有效遍；旧validation/纠偏dev/初24执行误差比update0为1.19757/1.22103/1.26783，诊断endpoint没有替换原模型。B持续训练，尚无新闭环结果。六轨迹probe改善不能替代全量泛化；FY overshoot与共享参数冲突是待验证假设。
- 新独立`tools/tight_research_supervisor.py`不修改已冻结worker。监视真实学习/评估进度，资源遥测不算进展；完成边界、异常退出或15min无进展触发真实`codex exec`。PID+start_ticks限定所有信号。最后完整评估子进程运行时仅park协调owner，待子进程退出再有序终止owner，保留整个队列证据，由Agent决定下一阶段；意外交接失败也保留事件。
- 本机Codex0.145.0真实调用成功，第二次read-only科研调用119.5秒完成并读回实际7017更新/1.2209988验证分数，未修改文件/GPU。普通脚本只监控与交接，不宣称自动推理；识别到模型连接临时错误仅重试一次，研究恢复由真实Agent检查完整checkpoint后执行。首次监督启动和后续科研调用状态分别记录。
- 针对PID复用、资源心跳误判、事件重复、输出越界和末次评估交接和无launch时的真实owner退出的6项测试通过。硬截止不变，不加新固定实验矩阵；用户解除的旧轮次/门槛不再作为研究停止条件。

- 10:44独立service实际启动为active/watching_actual_experiment；固定绝对Codex路径并在service内验证0.145.0，286份活动源码SHA未变。B实际4512更新，具体监督tick和PID见startup_acceptance；未来决策尚未触发，不写成已完成。

## 2026-11-13 09:55 接续CLI错误与同截止恢复

- 真实状态核实：短训03:04完成；正式A训练因`anchor_continue`必需`--job/--output`遗漏在argparse阶段退出，两组全量更新均0，未发生新闭环。不是训练继续运行；旧失败状态和日志保留。旧matched第二轮已结束，两组各0/10（A lift1/fall1，B lift2/fall2），旧发布器05:43完成；修复旧失败报告也已推送。
- 调整候选各1600更新/45924监督tick：A dev比.988568、Core比.424799，通过原门槛；B dev比1.165445、Core比.943496，未过。不是等曝光单变量FY因果证据；完整数据混合验证仍必要。
- `tight_execution_follow.residual_command`统一正式/定向训练与离线CLI，真实`anchor_continue.argument_parser`解析回归覆盖四种调用，模型和loss未改。恢复使用`adaptive_recovery_01`，固定同round01起点/cache，复用所有短训；余下时长重新决定共同6/3/1曝光预算，预留6h，不重置原23:55:15计算/次日00:05:15绝对截止。
- 修复阶段状态落盘与独立恢复证据导出，发布包含真实首次更新边界，防止旧短训RUNNING被误读为全量活跃。当前仅准备恢复，实际更新须启动验证。

## 2026-11-13 追加自主修复：真实输出域、扩展配对与普通脚本接续

- `residual_output.py`新增显式真实线性尾部；`native_dp/anchor_continue`分别绑定输出/loss/history身份，实际forward/objective/保存恢复接入。中心区旧行为，最终执行器/物理/A-B标签不变。`tight_execution_capacity`用原缓存计算实际v1含容差下界：A33.4600%、B0，最大.02618rad；不是旧raw数值匹配判据。
- 初始旧/FY各200更新、各5745监督tick已实际完成；FY A/B dev比1.13893/1.18148，原门槛未过。`tight_execution_adaptive`实际开始old800/tail800配对，old800两组已各800更新/22954tick，尾部候选在训；条件下一候选reg0/LR1e-4/1600明确混合因素，失败不自动终止整个修复。
- `tight_execution_follow`在隔离树接续全量后端、每组10完整策略验证及最多两轮12机会/组定向DAgger，后者更新DP/缓存/Transformer。启动仅等待本地短训，Git发布等待旧发布完成；原24h不重置。阶段实际曝光/未完成/技术失败保留；原模型不被候选覆盖。
- 97项模型/输出/loss/训练/控制定向回归通过，另有follow编排测试。新文件不热改活动短训源码；旧WAITING服务已退出并归档原登记。当前未获得新完整学生自主结果，未推送源码；旧发布器仍活动。以下原“GPU尚未开始、唯一候选、双门槛硬停止”快照已过时。

- 02:44启动核验：`tight-execution-follow-20261113-01.service` active、PID2482909，源码`14d5b05`，等待本地adaptive依赖；短训服务PID2473172实际tail800继续B组。按其启动提交`1872db0`核验284源码0漂移；第一次误用首200阶段freeze所得3差异单独留档，全部是新短训启动前的已提交修订。A新尾部800更新/22954tick：Core误差.0731783（旧.1512955）、dev.3542762（旧.3461596），约−51.63%/+2.34%，未通过原dev门槛。后续普通脚本继续，无新完整学生成功结论。

## 2026-11-13 执行等价后端监督候选与有界后台队列

- `visuotactile/execution_learning.py`新增稳定两倍FY、理想nested-box检查及实际v1容差原像/开放tanh可表达性；`models/native_dp.py`和`anchor_continue.py`实际训练调用接入独立loss身份，旧缺字段行为不变。输入为真实pre-action target、共同命令normalizer、原标签与合法mask；输出为主项/真实exec误差/正则和舍入量。没有改native动作、delay、减速、PD、A前端标签或output head。
- `tools/tight_execution_learning.py`复用上一固定清单/NPZ与已有prefix实现，记录三层梯度和共享参数单步。新增普通脚本负责有限配对probe、门槛后最新共同快照的后端适配/开发及安全发布；新24h包含准备等待，旧run不变。测试覆盖真实trainer、边界/退化/容差/tanh/不可表达、版本与阶段防重跑。
- 实测固定六轨迹：FY错误坐标u_pre零梯度/错误方向均0；A实际不可表达Arm23/Hand467，B0，旧A Hand473中6项其实执行可表达。A Hand有41个加权logit梯度仍≤1e-10，未声称所有梯度恢复。共享参数pre项约旧54–85倍，FY约1.36–1.72倍；单步结果和范围见DAGGER_METHOD。GPU短训、正式后端与自主闭环尚未开始，后台需等待旧owner及报告退出。
- CPU 100项定向测试通过（5.72秒，已有requests依赖警告）。原数据/模型SHA绑定；runtime实际274源码哈希匹配。初查误用历史session.cwd的结果本机保留并以活跃PID/cwd重查纠正。没有继续旧日期清理或推送大数据。

- 启动核验：`tight-execution-learning-20261113-01.service` active，PID2406625，冻结源码`bf7d9ee8de097e421092a2e171d2ac90f8b47198`的279文件哈希通过；状态WAITING/parent_workers_and_terminal_publication，gpu_started=false。新计算截止11月13日23:55:15、绝对11月14日00:05:15（UTC+8）。后台无自动restart，等待旧发布完成才推新源码；当前尚未推送，不将等待称作已训练。

## 2026-11-12 — 真实裁剪梯度快检、最小监督候选与否定结果

- 隔离`bc+rl_RSS_gradient_diag`，实际新A/B第一轮均0/10、lift3/1；正式第二轮继续原配置。固定6episode/1873合法tick真实模型检查发现大量有误差但主拟合梯度为零的分量；旧网络并非全部无梯度，执行误差优于hold。原始逐阶段数值集中`evidence/tight_gradient_20261112`。
- `native_dp.TargetResidual.compose_chain/objective_components`显式拆开u_pre、关节clip、可达投影，新增版本化裁剪前监督；`anchor_continue`真实训练、checkpoint续训/恢复核对loss身份，保留旧配置行为。A/B、边界、限位、mask、恢复权重及旧数学逐位一致定向测试通过。
- 独立发现正式rear跨更新沿用旧hidden近似；可选current_weight_prefix_v1从真实t0按当前权重重算，仅immutable输入可缓存。state_only实际4prefix/13state tensor逐位相等。新模块/模型容量/前端定义/物理执行未改变，当前正式队列没有切换。
- 唯一候选lambda_pre=1，各A/B旧新损失200更新、同初值/次序/优化器/当前权重prefix。新项train pre MSE下降85.49%/91.18%，但dev执行误差为旧损失1.8403/1.8292倍，Core2.1393/1.6070倍：拒绝正式采用，无追加调参。全部800更新均disposable、不计正式DAgger。
- 定向CPU集成验收83项通过（6.02s），compile/diff检查通过，仅既有requests依赖警告；真实模型与小样本训练凭据另列。
- `tools.tight_gradient_diagnostics`集中diag/probe/audit/publish，不重采数据/全量缓存，不启新Kit。同状态闭环恢复未完成，不能用离线学动或专家成功代替学生恢复成功。大权重/NPZ/HDF5留本地；未清理当前模型或有效数据。

## 2026-11-12 — DAgger吞吐实测与有限性能修订（待接入）

- 隔离perf树实测真实数据。`anchor_continue/state_image_loss/state_history`支持FP32 effective64、micro16、原micro4扩散RNG顺序与两episode不可变原始特征GPU缓存；可训练历史仍按当前权重从真实前缀重算。单组完整update8.4565→6.8877s（1.228×），梯度/参数差保持在记录的浮点重排范围，监督/几何不减量。
- `native_data/models/native_dp`将固定输入按episode搬运并避免DDIM timestep主机同步，两个完整真实episode共1,297tick、31字段逐位相等，18.9380→12.7174s（1.489×）。Transformer搬运热身后只约2%收益，候选已撤回；4线程不启用。
- `recording.verify_stream_fields`共享RGB/depth解码，完整412tick仍校验184源字段，均值10.3229→7.5134s（1.374×）；13项针对性测试通过，坏图像/索引/数值仍拒绝。既有lossless LZF在64tick交错测量中写入更快、文件增加30.08%；`tight_dagger.collect`增加显式编码配置，未来A/B共同采用，不改图像内容。
- `tight_dagger/tight_finish`准备同run安全边界迁移与第2轮双DP并行，保留deadline/已完成阶段/配对曝光。真实双learner计算探针29.7539→15.3796s（1.935×），不代表完整DAgger轮加速；缓存、后端和评估继续顺序。没有多env、AMP、TF32或compile。
- 新增唯一 [DAGGER_PERFORMANCE](DAGGER_PERFORMANCE.md)，原位更新Method/合同/入口/交接/AGENTS。小JSON公开于`evidence/tight_performance_20261112`；实际队列接入和闭环效果仍未完成，不将探针计正式曝光。集成176项定向测试通过（17.77s，含迁移9/并行6），保留既有requests兼容warning。删除16份已验证自有benchmark HDF5及一份重复disposable checkpoint，逻辑大小合计1.6122GB；已分配块口径和保留副本详见性能报告。正式源数据、模型、有效缓存和历史结果保留。

## 2026-11-12 — 控制一致性审计与两轮匹配DAgger接续

- 在独立工作树准备，当前旧run及冻结源码未热改。327条round2/3数值轨迹的11项来源/标签检查通过，未发现裸DP回退、学生后态冒充专家q_reached、真实issued/applied被新预测覆盖或重复减速。564条非test轨迹的169,853个合法专家tick经CPU执行链读回，映射误差0、专家标签改变0。旧学生最终目标仅受关节限位，存在超出相同真实pre-action target下专家可达范围的跃变；手部学生前缀超2rad/s目标变化分量约11%–16%。这是控制条件差异的证据，未证明为零成功唯一原因。
- `position_control`、`expert_speed_environment`复用一次post-native速度映射；新增有身份的专家可达区间投影。`TargetResidual`、`anchor_continue`、`native_controller`在最终命令loss/预测/执行共用适配器，以robot178的58:87真实动作前目标为参考；不再对已处理标签加delay或减速。`state_benchmark`记录投影前目标、投影量、参考目标并验收首2次真实完整策略采集；旧checkpoint默认行为保持，原native控制/PD/资产/任务未改。
- 真实已训练A/B后端在Core和恢复初段的CPU前向/反向均有限且有非零数据拟合梯度，分别为0.001636/0.001742/0.015742/0.001816。46%–57%关节分量受到新投影，clip零梯度区如实记录。这些检查没有optimizer更新或新Kit启动，实际物理效果待新运行验证。
- `tight_dagger`/`tight_finish`/配置原位增加有限匹配接续：旧owner及workers退出、报告发布后才接续；新admission起24h。先双后端共同控制/幅度适配6有效tick遍，再最多2轮，每轮每组48配对机会、DP6有效anchor遍、刷新新DP缓存、后端12有效tick遍、固定10次自主开发评估。共同池快照/采样规则/随机种子/endpoint选模，实际曝光强检，失败或不足不让单组多轮。A保留q_reached和既定几何，B保留u_expert，前端模态及Transformer容量相同；历史初始化/曝光差异保留披露。旧13:24截止不变、test22未使用。
- Phase0仅允许显式登记的残差幅度适配，不静默更换旧权重buffer后部署；从各自训练权重继续，normalizer与训练参数身份核验。CPU实物准备已逐SHA验证旧r02两组各384条前端预测缓存直接复用，无重算、无优化。后续每轮前端更新仍失效重建其全部参数相关后端缓存。
- `tight_diagnostics`增加CPU控制审计及新协议逐轮表、phase0曲线与实际曝光导出；[DAGGER_METHOD](DAGGER_METHOD.md)集中记录数学目标、流程图、伪代码、历史引入提交、来源/恢复规则、预算依据、局限与复现入口。7张图及原始CSV/JSON公开，HDF5/权重/视频留本地。既有合格纠偏、源数据与历史证据保留；本次没有删除缓存或checkpoint，临时CPU准备输出标记为disposable并保持索引，未以不明依赖清理凑交付。
- 联合149项定向CPU测试通过（15.40s），仅既有requests依赖warning；覆盖执行一致性、真实目标参考、合法标签、残差适配、队列/配对曝光/恢复、采集验收及发布。当前实现与CPU验收已完成，新服务启动/正式训练/闭环分母以接续root的实际凭据为准。代码、审计与Method由后继在旧owner发布退出后自动合并并推送指定origin分支，最终新实验结果同样自动发布；未将待执行写成成功完成。

## 2026-11-12 — 双方零成功时的一次条件DAgger追加

- 最新授权将后续第3轮改为条件追加：当前第2轮A/B新完整candidate的合法纯学生评估完整结束且双方0成功，才使用两套新pair继续；任一成功不追加，证据不完整/非法/软件错误不能作0成功。最多一轮，A/B各最多96机会，DP与后端均继续更新，Core与全部纠偏累积。原4→10评估/test22隔离和标签/历史/几何语义保持。
- 只有触发时允许原2026-11-12 07:24:31截止延长最多21600秒至13:24:31（计算13:14:31），原起点保留；无递归、新run或重置18小时。隔离树准备，现有B先正常完成，在自有owner/worker退出的安全边界迁移。
- `tools/tight_diagnostics.py`原位发布条件request/decision/extension/status/service，区分未决、未触发、已触发、进程启动与实际采集验收，不再硬写“第3轮B未调度/截止不变”。条件未决不能核验整个新范围完成；未触发不虚构未来评估分母，历史证据保留。发布器仍只读CPU，不选择模型或触发队列。
- 原位更新AGENTS、CONTRACTS、SESSION_HANDOFF、研究方向和研究入口。增加恢复覆盖是探索假说，零成功不证明数据不足；本条为准备与定向验收记录，实际触发/迁移/采集由机器凭据另报。未新增GPU/Kit工作，未删除数据或权重。
- 队列、条件接续、发布、owner、评估和采集共69项定向CPU测试通过（9.03s）；真实A新candidate权重与全部4例评估凭据通过CPU核对（0成功、1 lift）。覆盖条件未决不制造0成功、扩展评估不能遗漏后6例成功、一次延期、历史选择保留、进程启动与真实采集验收分离、未触发不虚构未来评估分母；完整后继分支以替身校验部署→迁移→条件启动→凭据发布顺序。代码准备不等于追加已经运行。

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
# 2026-11-13 11:25 — 完整边界后真实Agent选择中心化提案

- 核实恢复实验A/B各7017更新、完整自主各0/10、lift2/1、无软件失败；有意owner交接与训练失败分开。
- 新opt-in可达名义中心化线性提案移除大绝对nominal抵消；真实导数/FY/旧执行器不变，无STE。原checkpoint与缺字段行为保留。
- 隔离源码与独立有界service入口，复用原round01权重/cache，1遍＋固定完整学生开发；不自动DAgger、不重置截止。
- 新测试覆盖逆像有界、错误饱和区方向、实际执行等价、checkpoint往返、旧行为、CLI与原deadline。
- 11:29实际启动验收：104项测试通过，真实A256更新/18057监督tick；既有六轨迹新参数化逆logit绝对值均≤1、最大重构误差1.49e-8rad，仅表示条件表达能力，不表示泛化或自主成功。旧286/新290源码hash无漂移，独立service与真实Agent事件交接登记完成。
