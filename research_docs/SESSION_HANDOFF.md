## 2026-11-14 第二轮完整训练完成，仿真启动卡住后单独恢复评估

round02_v2两组后端各完成10遍、1131更新、160630合法监督tick（原30 103980、历史11 30450、本轮10 26200）；全部57条新DP计划/特征已刷新，两组均选第10遍。验证手指命令A0.0181267/B0.0195016rad，先前0.0232948/0.0225830；保持上一真实目标基线0.019415，A低于基线、B略高。完整离线/曝光报告与曲线已生成，不等于自主改善。

随后同冻结a4db8e4自主队列在Kit初始化卡住：324.7秒仍无initialization、0case/0学生执行；Kit日志从启动14.4秒停在材质初始化，两个carb.tasking线程持续运行。ptrace取栈被环境拒绝，不能确定原生死锁根因。核验自有PID后SIGTERM无效，再SIGKILL退出；cycle如实留下development失败，训练成果完整保留。新增单次pre-control recovery入口，同原4case/组与准确完整pair逐项相等才启动，原截止不变；180秒启动界限及90秒Python栈记录避免无限等待。2项拒绝已开始评估/自有进程回收测试通过。下次实际恢复结果另记，不把准备写成成功。

## 2026-11-14 复查首次拿稳后的实际失败阶段

round01 A/development_28140000只有23个连续稳定窗tick（230–252），随后失稳；离第一孔最近334tick为26.43mm、角度3.057rad且没有双指接触。B同seed累计63稳定tick、最长连续54；341tick中心曾到0.91mm，但当时角度1.363rad且没有双指接触，不能算对准。B最小keypoint误差279tick仍比容差高18.84mm，当时双指接触但稳定窗不成立。新增报告把距离/角度/抓持绑定同一时刻，不把分离最小值拼成成功。原各1/4拿稳、各0/4成功不改。新round02训练仍由冻结a4db8e4继续；此为旧轨迹复查。

## 2026-11-14 第二轮A/B前端恢复训练完成

冻结a4db8e4的round_02_v2完成A/B各10遍、510更新，两组均验证选第10遍。每组实际13640anchor/315580合法监督槽：原30各10次（300episode更新/103980视觉帧/204360槽），历史11纠偏各10次（110更新/41130帧/59700槽），本轮10纠偏各10次（100更新/60840帧/51520槽）。所有真实学生前缀只更新历史，未冒充专家标签。

各自同6验证family的归一化指标A 0.01096077→0.01000797（-8.69%），B 0.01131391→0.01057726（-6.51%）。A构型hand RMSE 0.069230→0.066458rad，B命令hand 0.078790→0.076374rad；标签定义不同不横向比误差。新10纠偏训练后的指标A 0.00677697/B 0.00665101，仅为训练过数据的拟合，不能叫自主改善。准确初始数值见绑定report。

真实57条numeric全部字段、原30 affine和冻结DINO内容在失败attempt与恢复attempt之间完全一致；失败48更新未继承。公开round02_frontend_learning和round02_recovery_population_audit。新DP后的全部后端计划/可训练视觉缓存正在重算，后端与自主复测尚未完成。本条只记录已完成前端，不把待运行后端或拿稳效果提前计入。

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

## 2026-11-13 23:35 后端增量剂量完整终局：四cell各0/10，不追加计算

`rear_span_dose_02`已于23:33完成，owner3070479已退出；不是训练故障或有意交接InterruptedError。reference A/B与continued A/B均完成同一固定10机会：每cell **10合法、0拒绝、0软件失败、0完整成功、0首子目标**。lift依次为3/4/3/5，不能代替完整成功。

continued每组新增2620后端更新、387040监督tick、2.0人口遍，训练墙钟A865.11秒/B853.26秒。reference为原span1冻结端点，本次新增训练为0；继承收据的1314更新/193520tick不能计入本次。四cell前端更新0、cache不刷新，不是新DAgger轮。端点仅诊断、不采用；没有观察到追加剂量带来的成功收益，也不能据此宣称所有欠训练假说已被排除。

终局重新运行`tools.tight_ab_fairness --frozen-frontend`，reference与continued各自A/B均通过，四cell共享相同plan摘要。审计仅覆盖登记合同、缓存绑定和实际曝光/覆盖收据，不证明历史初始化相等、逐步optimizer/采样顺序或隐藏物理状态逐位相同；reference/continued的训练剂量与optimizer重启有意不同。初次终局CLI遗漏该flag、误把继承DP514显示为增量的本地输出保留但已由frozen审计替代；reference继承rear曝光也明确不计本次新增。不混入B-only复现或上一轮9合法/1拒绝分母。

复用主Agent已发布8次诊断，不重复：原B274tick成功保留但未稳定复现，503命令回放命令/target精确而物理轨迹不精确。当前证据不足以归因到某个损失/晚期动作，也不足以宣布A/B优劣。按22:39用户要求，完成此边界后不新开训练、大评估、service或矩阵；剩余窗口交回主Agent。此次监督器结论为授权边界`external_block`，不是科研成功或任务不可能。原23:55:15计算/次日00:05:15绝对截止不变，保护全部证据与test22隔离。publisher与supervisor不改动。

小证据：[终局交接](evidence/tight_execution_learning_20261113/rear_span_dose_02/final_handoff.json)。

## 2026-11-13 23:00 实际运行检查完成，停止追加大矩阵

已完成8次诊断、零新训练，详见DAGGER_METHOD页首和runtime_review.json。原B274步成功证据保护，但同seed重现失败；完整503tick纠偏记录/当前batch正确，命令与target精确回放、物理轨迹不精确。A/B从该纠偏分支初态自主均超时，不声称已学会。完整A601tick写盘优化13.05→1.43秒，总81.58→70.88秒，非严格轨迹配对；23测试通过。探针退出故障已记录并清理，优化探针正常退出，两owned GPU进程均不存在。

健康rear_span_dose_02继续原40机会，不中断、不热改；本次已向现有Agent提示追加“完成现阶段后不再开新训练/大评估”。原23:55:15计算、次日00:05:15绝对截止不变。后续只总结既有结果、核对公平性和资源/清理，不能把建议当作新训练授权。自动publisher仍独占两live文件，Git保持共享publication锁。默认32tick缓冲numeric只影响未来新worker；原compact数据记录不变，硬崩溃最后31行新numeric可能未刷盘，普通异常会flush。

## 2026-11-13 22:23 自动页拒绝机会与阶段保留修复

采样对照四cell各0/10已完成；后续span metric对照四cell各10个机会中9合法、1初态拒绝、0成功/0软件失败，不能当成10个合法失败。当前`rear_span_dose_02`按原登记A/B共同追加两遍后端训练，再各自与冻结参考做同10新机会；本轮尚无新闭环结论。前端/cache不更新；313冻结源码文件SHA无漂移。A/B准备配置及评估spec一致，最终实际曝光仍待完成审计。

修复独立CPU发布器：初态拒绝无result checkpoint字段，必须核验receipt与frozen plan/checkpoint绑定及not_started，单列拒绝分母；合法试验仍要求模型哈希。完成阶段保留，继承训练收据标记非本阶段新增，实时训练进度仅读本阶段时间戳。9项测试通过，真实20份summary全部验收，原漏报4份均恢复9合法/1拒绝。原失败证据和bot历史提交保留。

新源码在隔离树审查，发布器在共享Git锁的空闲边界切换独立新runtime；活动训练/监督器不改。原计算23:55:15/绝对次日00:05:15不变，真实替换启动看publication_fairness_01内v2收据。

## 2026-11-13 22:14 共同曝光接续已真实启动

独立`tight-span-dose-20261113-02.service`已active；owner3070479/start_ticks163893784。`rear_span_dose_02`首组A已实际224更新/32976合法监督tick，313冻结源码SHA无漂移，6GiB VRAM/14GiB RAM/60GiB磁盘储备检查通过。启动源码`aadbad1`已正常推送，活动工作树只读；本条在独立publication树更新。36测试与准备审计不是自主成功，新40机会尚待完成；参考不重训，候选A/B共同两遍。

真实Agent下一watch为`rear_span_dose_02/{status,launch,finished,failure}.json`。阶段23:47:57、原计算23:55:15/绝对次日00:05:15不变；常规owner只执行已登记步骤，不自行判断科学路线。CPU准备01错误和startup收据临时JSON序列化错误均未影响活动训练，原现场保留，不重启健康worker。publisher专属live文件和监督器未改。

## 2026-11-13 22:12 Agent009：完整metric终局与有限曝光接续

`rear_span_metric_eval_recovery_02`40个固定机会已完成；reference/span各A/B均9合法、1初态拒绝、0软件失败、0完整成功、0第一子目标，不能写成每组10个合法失败。两arm完成fairness通过；拒绝机会不补种子。原FY metric与span metric之间，完整离线arm误差小幅下降，hand未见共同收益；这不是闭环成功改善或A/B排名证据。

下一单一问题为一遍span训练是否曝光不足，而非再换损失/网络。独立`bc+rl_RSS_span_dose_01`登记`rear_span_dose_02`：各组从自己的span一遍endpoint共同追加2人口比例遍，fresh AdamW处理一致，保持loss/输出/执行器/真实历史/DP及cache。参考直接复用一遍权重，不重复训练；两arm×A/B各10全新seed27200000–09，共40完整策略机会。跨arm有意改变总曝光和optimizer重启，不能称纯曝光单变量；arm内A/B增量匹配但历史不同。不称DAgger、不开新前端训练、不读test22、不自动采用或追加。

36项测试、真实训练/离线CLI解析、参考完整评估入口CPU前置检查及两准备fairness已通过；candidate必须训练和完整offline后才准入。CPU准备01错读base_checkpoint所在配置层，0更新/0仿真，保留现场后在02修复；错误解释不冒充学习失败。阶段截止23:47:57、计算23:55:15/绝对次日00:05:15不变，100min含准备；实测一遍训练约7.2min/组、40仿真约42min。当前记录为准备完成，启动和实际更新另见`launch/startup_acceptance/status`，不把准备写成训练成功。监督器/发布器及bot专属文件不修改。

## 2026-11-13 21:21 恢复评估已真实推进

`tight-span-metric-eval-recovery-20261113-02.service`独立cgroup已active；owner3037113/start_ticks163579189。首个reference-A完整DP+Transformer评估实际推进到tick300，311冻结源码文件无漂移，GPU/RAM/磁盘储备均通过。源码`2e5d85d`及63项测试已发布；这不是仅PREPARED/加载，也不是完整cohort成功结果。

监督器下一watch应为`rear_span_metric_eval_recovery_02/{status,launch,finished,failure}.json`，不再监控已失败的01。原40机会、22:50:03阶段截止与23:55:15计算/次日00:05:15绝对截止保持；完成后由真实Agent判断。启动证据在独立publication树提交，活动worker和bot专属live路径未改。[启动验收](evidence/tight_execution_learning_20261113/rear_span_metric_eval_recovery_02/startup_acceptance.json)。

## 2026-11-13 21:18 Agent008诊断并恢复同一固定评估

事件为`rear_span_metric_eval_recovery_01/failure.json`。真正异常是offline准入文件未继承；组分母guard在任何仿真前拒绝，不是训练新失败。A/B span各1314更新/193520tick已完成，四份对应checkpoint的完整离线结果都存在。无新训练/推理，按SHA及实际data/cache身份复用原结果；新root `rear_span_metric_eval_recovery_02`，隔离源码`bc+rl_RSS_span_eval_fix_02`。原失败现场保留，监督器/发布器不修改。

新身份保持原40固定机会和22:50:03阶段截止，不重复已开始样本。两准备fairness和四cell实际评估入口CPU前置检查通过；模拟dispatcher不产生结果，不作为自主成功。实际运行以新root `launch/status/startup_acceptance/finished/failure.json`为准，不以PREPARED冒充GPU启动。后续真实Agent应先完成该cohort，不按工程故障机械否定loss候选。原计算23:55:15/绝对次日00:05:15、资源储备和历史A/B不等同限制保持。

## 2026-11-13 21:01 实际训练与安全评估接续

`rear_span_metric_01`真实训练已验收：G01达到1088更新/159498合法tick，owner3022906/start_ticks163431464，310个冻结源码无漂移，资源储备通过。74项定向测试通过；新完整自主结果尚未产生，不把启动说成成功。

发现一个纯编排guard问题：真实trainer会同时记录新的loss_contract和派生loss_id，原跨metric白名单遗漏loss_id，预计会在两组训练/离线完成后误报。没有停止健康训练或热改冻结源码。仅在独立publication树修白名单，并登记`rear_span_metric_eval_recovery_01`评估接续：等原owner实际退出、核验两组保存完整及明确guard异常、确认原eval从未执行，才复用全部已训练endpoint/原计划运行原40机会，不重复训练或已完成试验。原stage约22:50:03及原23:55:15/次日00:05:15截止全部不变；任何其他故障交真实Agent，不静默越过。新CLI准备和3项恢复防重/截止测试已通过，接续owner启动另记机器回执。

## 2026-11-13 20:53 Agent007：完整重放终局与单一metric实验登记

`rear_replay_balance_01`四cell均合法完成10例且0成功，无软件失败，两arm完成fairness均通过。人口比例真实覆盖全部662训练episode、每层恰好1遍，不能把此候选失败说成未采历史纠偏，也不能据单次试验排除所有覆盖问题。旧50/20/30的A/B lift4/3、首目标0/2；人口比例A/B lift3/5、首目标0/1。原单例成功、B-only复验和全部失败独立保留，不宣称稳定A/B优劣。

复用三份既有CPU审计、公平政策及发布器收据。全193520合法训练tick的centered头FY系数(action_scale×reachable_span)^2，其逐joint中位数跨1551.14倍；这是loss条件化证据，不是已证明成功瓶颈。登记原run下`rear_span_metric_01`，隔离树`bc+rl_RSS_span_metric_01`，保留FY错误饱和区真实非零方向梯度，只换可达比例metric。A/B各从同一contact后端起点、fresh AdamW、相同人口比例全1遍，固定endpoint；参考直接复用已完成population endpoint与训练收据，不重训。两arm×A/B各10个全新seed27190000–09，共40完整DP+Transformer机会，无替换/扩展/自动采用；不是DAgger，前端和配套cache冻结。历史A/B初始化不同仍披露。

真实prepare、71项测试、两arm准备fairness通过，本条不代表GPU启动。实际看新root的startup_acceptance/status/launch/finished/failure；独立service登记2h，仍受原23:55:15计算、次日00:05:15绝对截止限制。活动监督器/发布器及其专属文件未改；所有Git事务持共享tight-publication.lock。

## 2026-11-13 19:23 后端重放实验已真实启动

独立`tight-replay-balance-20261113-01.service`已active，owner2967509/start_ticks162865736；首个control-A实际288更新、41271合法监督tick，不是仅准备或模型加载。308个冻结源码文件无漂移，资源储备通过，两准备fairness通过；后续真实结果尚未产生。源码74c8006已推送指定分支，活动研究树只读；本条在另一publication树更新。下一watch为`rear_replay_balance_01/{status,launch,finished,failure}.json`，由既有真实Agent监督器接续；阶段21:48:09、原23:55:15/次日00:05:15不变。

## 2026-11-13 19:20 Agent006完成复验判读，登记单一采样问题

已读事件、三份现有审计、公平政策和发布器收据；不再重复宽审计。A补齐owner正常退出，实际新20例A/B均0/20；没有继续B单组调参或混合旧分母。下一阶段`rear_replay_balance_01`，隔离树`bc+rl_RSS_replay_balance_01`：冻结contact完整pair前端/cache，以原50/20/30为继续训练对照，人口比例为候选，各组同初始tensor/新AdamW、共同FY和1遍，固定最终endpoint，各10新完整策略开发机会。不是DAgger、无新前端训练或采集。真实CPU预采样人口比例覆盖全部662训练episode，对照319；实际训练尚待验收。

新增frozen-frontend fairness准备/完成模式，旧DP曝光不会冒充本次更新。28测试、真实prepare与两arm准备审计通过。机器root内`registration/status/launch/finished/failure`为准；下一监督watch使用真实owner身份，不能监控旧已完成文件。阶段21:48:09、原计算23:55:15/绝对次日00:05:15均不变。发布器两专属路径和监督器源码未改。

## 2026-11-13 18:48 A补齐评估已真实推进

`tight-ab-complement-20261113-01.service`独立cgroup已运行；owner2945216/start_ticks162656356，首个冻结A完整学生case实际推进到tick400，非仅注册或Kit启动。297个以上受检源码文件无漂移，具体数量见startup_acceptance；源码commit c8d3dd3已推送指定分支。GPU空闲约19GiB、RAM可用约29GiB，监测无暂停原因。固定子截止19:43:57 UTC+8，原计算23:55:15/绝对次日00:05:15不变。

既有真实Agent监督器继续监测新root的status/launch/finished/failure，无第二监督器。普通owner只执行固定20机会及公平收据检查。发布启动验收在另一个隔离发布树完成，不修改运行源码或bot所有的live文件；A新cohort未完成，不提前给成功率结论。

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
