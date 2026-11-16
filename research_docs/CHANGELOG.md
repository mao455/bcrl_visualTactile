## 2026-11-16 20:57 — 四类小规模触发验收与新盲测隔离

pilot02四合法全部专家完整救回：搬运A99/B98、近孔A159/B152tick；557合法专家tick。两个先导合计8合法、7成功纠偏/874有效tick，0软件中断。搬运仍接近原早期窗口，不将其称为后段新覆盖；新增价值主要在近孔、对准及近孔支撑变弱。正式formal01四family×A/B共8机会已串行开始，分配对准2family、支撑变弱1family、早期接触1family，保持共同数据和一致执行约束。

普通服务只在formal01完全闭合且owner回收后聚合pilot01/pilot02/formal01审核通过的后缀，继承全部37旧合法轨迹，再串行A/B DP→刷新各自后端→Transformer→配对开发；最多40/24epoch、至少16/12后按验证平台停，两组相同规则。不把开始或专家成功写成学生收益。

final_test准备入口新增显式预登记文件参数：本轮28821000–007与原测试/开发/训练/验证隔离，登记时间必须早于最终训练完成；冻结前不打开，默认旧实验路径保持复现。final_report允许登记的实际资源roots，避免错误读取旧实验监控。当前仍未运行新测试。

## 2026-11-16 20:54 — 首批四环境后段接管验证已闭合

pilot01四机会全合法、0软件中断：对准组B在200tick接管并救回，A未达到注册对准条件而601超时、没有强行固定tick接管；支撑变弱组A183/B133tick接管均成功。共3条成功专家后缀/317有效tick，phase5/3/8，全部学生prefix和队列过渡屏蔽，原始控制/标签与因果触发回放审计通过。两服务串行交接，pilot01 monitor0/owned退出后pilot02才启动，后者当前检验搬运与近孔，尚无完整结果。

实际pilot01循环194.92s/1440有效lane-tick（7.39/s），controller69.08s、env74.04s、finish audit29.60s；四lane尾部利用率59.90%。原八lane队列244.93s/2612（10.66/s），不同轨迹故非严格速度对照。后续优先同源既有8lane批次减少启动/低利用尾部，保留512MiB有界writer和两验证进程。检查旧exact-tail门发现历史完整记忆数值验收失败，未采用该优化。冻结视觉只复用相同输入的基础编码；各轮重新计算可训练历史与后端计划。

新增phase_report.py记录每类student-prefix可达、真实接管/救回、合法mask、有效tick、相位、孔口误差、失去接触及计时；phase_pool.py只合并审核闭合的同一pair采集manifest并链接原轨，不重跑、不覆盖失败。两者尚未用于新训练完成声明。

## 2026-11-16 新阶段DAgger：正在审计与小规模接管验证

最新授权继续10cm布局完整A/B共享DAgger，覆盖旧停止条款；不改变网络、物理、控制或原生成功。新session tight_phase_20261116_01独立6小时总界/5小时训练界，最多两轮完整学习、8个先导及最多32个额外纠偏机会，软件故障每阶段最多一次新身份恢复。首批4lane实际并行验收，之后按吞吐与资源决定；不热改运行源码。旧8test仅回归，新28821000–007保留到最终模型冻结后一次使用。

已真实审计原37训练：7纠偏/1146合法专家tick，两组numeric与原始控制/标签一致，学生前缀与队列过渡mask0；两组后端缓存完整time/robot/mask/segment与原轨相等，命令目标一致，DP窗口mask排除所有无标签tick。完整历史不截断、不将prefix冒充监督。新多阶段触发只读已完成实际transition的特权几何/接触，安排下一动作接管、不按mod12取整；尚未实测救回，不宣称有效。主要按搬运平移、近孔root、孔口keypoint误差和近孔支撑变弱筛选，保留已有早期纠偏能力。原生mask/队列/执行器不改。现有源监督审计与回放证据在evidence/tight_phase_20261116。

## 2026-11-16 20:14 — 新布局完整实验结束：干扰减轻，专家36/36，独立学生测试均0/8

实际清理、原位/5cm/10cm专家试摆、新示范、A/B随机初始化训练、完整共享DAgger一轮及最终独立测试全部完成；三个学习/纠偏/测试服务inactive/MainPID0，无本轮worker遗留。最终测试于20:10:59闭合，20:11:00 owner回收子进程、monitor以0退出。最终测试已打开，不自动续训、添轮、调参或重复测试；旧15:30截止未沿用。以下旧条目中“正在运行/测试未打开”是历史快照。

选定仅插销起点XY平移10cm，位移(-93.401,+35.725)mm，孔位/资产/物理/120-60-5Hz/完整成功判据不改。原位3/3成功；5cm两合法均成功、一保守碰撞门拒绝；10cm三合法均成功。自然沉降后原位两例物体—基座表面间距约0.009/0.044mm且高度受基座影响，10cm三例15.45–18.23mm、无明显竖向位移；新36条沉降间距14.87–24.05mm。支持减轻真实物体—基座几何接触干扰，不等于证明手指碰撞力或视觉遮挡的独立因果改善。专家新train30/validation6全部合法、36/36原生完整成功，没有替换seed；两次物理启动前软件错误与pilot退出卡顿另存。

两学生从随机DP/Transformer开始，共同train30/validation6；初训各前端80遍、后端40遍，独立开发均0/4。一轮完整学生纠偏8合法机会，A3/B4共7条真实成功专家后缀、1146有效tick合并同一冻结train37池，学生前缀和2tick过渡mask0；各自DP更新后刷新后端输入再训Transformer。两组相同40–80遍/平台停止规则，A前端80遍、B51遍停，后端均40遍，实际更新量不强称相等。开发A1/4、B0/4；此后冻结模型，预留8对test各跑一次。

最终独立原生完整插入A0/8、B0/8，16合法、0非法、0软件中断；两波实际初态配对均通过，隐藏物理状态不称精确重放。A一跌落/七超时，B两跌落/六超时；首子目标A1/8、B2/8不算完整成功。B seed28819001首子目标后原生时钟重置，806tick超时，不是延长原超时。首对真实三视角视频各601帧全部解码并浏览关键时刻：A抓起转正并搬近孔，随后倾斜、撤手后在基座/桌面旁失去支撑；B近孔持续调整但未完成对准，后倾倒到桌面，均601tick超时。开发的一次成功没有形成独立测试改善证据；没有同训练剂量旧布局对照，不能将变化单独归因于拉开距离。

清理净释放135318011904bytes（126.03GiB）：256份闭合失败RGB-D，保留全部数值/控制/结果及36份完整代表失败，未将学生失败叫数据损坏。最终审计400保护数据文件存在且大小匹配、manifest SHA与10月31日保护一致；现有1406个manifest依赖均存在，最终四个模型重新验SHA通过。没有再删活动文件/有效源/专家/预训练权重。当前剩余约103.76GiB。最终测试采样最低可用RAM20.85GiB，峰值GPU已用12271MiB；本轮按已授权全GPU、6GiB/12GiB准入门执行。初训最大采样间隔9.70s，后续最大1.58s；missing计数0不代表连续硬保证，保留全部原始监控。

只读final_report.py在生成统计前核对闭合/无worker错误、每个test seed恰好A/B各一次、实际配对通过、最终模型hash；真实报告执行及py_compile通过。最终证据：evidence/tight_layout_20261116/final_test.json、final_audit.json，前期清理/试摆/采集/学习/纠偏证据同目录。代码与小报告推送origin/feature/tight-state-foundation；大数据/模型/视频留本地。

## 2026-11-16 20:01 — 首轮共享DAgger完整闭合，A开发1/4，最终测试已冻结启动

真实采集→共同37train/6validation→A/B DP→各自新DP后端输入→Transformer→8次完整自主→owner/monitor退出全部完成，0软件中断。A前端80epoch/1520更新，选80；B51epoch/969更新、验证平台停，选42/798。两组相同40–80/平台规则，不是强行等epoch；A46末/B43末LR减半。两后端40遍/3200更新，A选39/3122，B选40/3200；A34末LR减半。原验证空间不变，前端最佳误差A .00695559→.00582831、B .00701769→.00586132，后端 .000404844→.000324505、.000585439→.000475277。仅各组内部比较，不以跨标签loss排名。

同四对开发初态全部合法，A1/4原生完整插入（seed28820001，331tick），B0/4；初训均0/4。A对准/首子目标2/4，B1/4；持物120tick A4/4、B2/4只作诊断。A000完成首子目标后滑落、820tick原生超时，B001为819tick原生超时；任务子目标会重置原生时钟，不是改变601超时。其余A一601超时/一399跌落，B两601超时/一411跌落。首对真实三视角视频已浏览，A转正并到孔后撤手但未完成，B早期抓取支撑丢失；不把A首子目标当完整成功。可观测配对通过，隐藏状态不声称精确相同。

source=dagger01/learning，本轮到此冻结；最终8个预登记独立test seed28819000–007各A/B一次，分两波8lane，tight_layout10_20261116_final_test01，冻结99cfaf7/service tight-layout10-final-test01-20261116。测试正在运行，尚无结果；此后不据test调参/选择/续训。A的1/4是小样本开发改善，不证明布局单独的因果效应，也不证明稳定泛化。旧布局没有匹配新数据/训练剂量的同轮对照。

## 2026-11-16 18:58 — 首轮新布局纠偏采集闭合，共同37条已开始更新

8机会全部合法、0软件中断；A3/4专家救回，B4/4。A seed28817002未形成contact-height窗口，完整自主601步超时，未执行fixed fallback、未入池。七次真实接管在96–128tick，共1146有效专家动作；每条前缀mask0，2tick队列过渡mask0。实际numeric逐数组验证q_reached等于同transition真实q_after，u_expert等于实际执行target；没有把学生动作当专家标签。成功恢复不是自主插入。

共同冻结train37/validation6/test0，仍原train30统计、每条权重1。采集monitor0退出后学习owner启动，A DP已完成首epoch19更新；B后续相同规则。原24h/22h截止、完整学生与原生成功不改。新只读correction_report.py与evidence/tight_layout_20261116/dagger01_collection.json保留全部分母、触发与标签审计；运行worker冻结99cfaf7未热改。

## 2026-11-16 18:51 — 新布局初训与完整自主完成，共享DAgger已接续

共同train30/validation6，学生随机初始化；A/B前端各80epoch/1200更新，独立验证选A77/B80；后端各40遍/3080更新，选A40/B39。均达到登记预算上限，不声称完全收敛。前端监督空间不同，不能比较A/B loss大小判断优劣。真实曲线与逐初态行为见evidence/tight_layout_20261116/initial_learning。

四对独立开发初态全部合法，A0/4、B0/4原生完整插入，0软件中断、0对准/首子目标；各1次原生跌落，其余各3次601步超时。持物60/120tick诊断A2/4、B3/4，不替代完整任务结果。首对实际三视角视频已查看：两组抓起、搬近基座并转正，插入端未完成对准后滑落；其他初态还存在抓取失败。暂未证明新布局提高学生自主插入。

四对首动作前q/qdot/目标完全一致，位姿/速度在1e-5容差内；不是隐藏状态精确重放。learning01于18:51:33完成，18:51:34 owner/monitor退出，monitor0；此前评估监控提前退出未在这8次复现。

已基于失败阶段选择一次接触抬升后的纠偏：tight_layout10_20261116_dagger01，冻结99cfaf7，服务tight-layout10-dagger01-20261116；原train前4family×完整A/B共8机会、contact_height触发，无fixed fallback。现在仅采集运行；成功实际专家后缀合并同一冻结池、前缀mask0，DP更新→重新生成后端输入→两Transformer→同4对完整开发评估。保持原24h总界/22h训练界，统计仍原train30，采样权重1，两组预算同规则。尚无新纠偏/训练/自主结果，最终独立8对test未打开。

## 2026-11-16 原登记独立测试入口准备

新增`final_test.py`：只在训练/开发完整结束后固定选中A/B，对原28819000–007八个测试初态各执行一次，两批各8lane，复用真实reset配对、完整原生任务及首例视频。测试plan明确test_used=true，普通开发runner默认仍拒绝测试plan；必须经此入口核对原登记、模型身份、非训练/验证/开发seed、标签及配对后才显式准入。不把最终测试结果用于追加训练或模型选择。

真实新布局元数据的CPU准备检查覆盖16测试机会、8独立family、开发seed误入拒绝；学生身份在此准备检查中stub，无真实仿真/更新。`learning_report.py`汇总已有validation/finished与完整行为，生成各自标签空间曲线；不得跨A/B标签比较loss或将离线下降写成成功。当前A80遍/1200更新完成，B仍在同预算训练；最终测试和DAgger尚未执行。

## 2026-11-16 新布局共享纠偏链准备（尚未执行）

`scripts/tight_layout_20261116/dagger.py`准备单轮4个原始训练family×A/B共8机会，复用现有approach/touch/contact因果接管，不改模型或控制；触发类型待首轮完整自主行为审阅后确定。原生reset私有流及首动作前配对检查复用；仅合法成功专家后缀并入两组同一冻结池，学生前缀/延迟过渡仍mask0，统计固定新布局train30。更新DP后重算后端输入，再继续同源Transformer，最后相同4开发初态完整自主；不用抓持门槛裁剪评估。

learn新增显式root解析及聚合续训路径，原随机初始化路径保持；两组仍同一40–80/20–40有限验证规则，串行资源交接。通用batch只补已存在early trigger的准入/分类入口，不扩展阈值。31项原触发/因果测试通过；真实新布局reset准备与非法兜底/改阈值拒绝检查通过，模型身份在该CPU准备检查中明确stub、0GPU/0仿真。第一次用run_cpu两核affinity触发原Kit16线程配置拒绝；保留16核affinity、torch/BLAS2线程后通过，未降低配置检查。活动8bb8e23训练不热改；新入口必须在首轮结束审阅后冻结到独立树。

## 2026-11-16 新布局36条示范完成，随机初始化训练已接续

10cm布局独立train30/validation6已全部闭合：36机会、36合法、36原生完整成功，0任务失败、0替换seed。此前两个启动前软件问题另存原root，不并入物理机会；原pilot关闭卡顿也保留。36份实际初始可观测状态哈希互异。新示范抓前24/36/48tick的碰撞盒基座表面净空14.87–24.05mm；这是实测物体—基座几何，不是手指—基座力归因。专家仍可完成抓取、搬运、对准与插入。

采集owner与monitor已退出，退出码0；数据字段与训练标签审计通过。冻结8bb8e23、隔离树layout_pipeline02已自动接续共同train30/validation6，A优先，DP与Transformer从随机权重开始；尚无新学生自主结果。维持预登记40–80遍前端/20–40遍后端独立验证停滞规则、原总截止、相同A/B预算。没有重启旧15:30任务或混入旧布局数据。

`scripts/tight_layout_20261116/collection_report.py`重算闭合numeric的实际几何并绑定已审计数据manifest。首次汇总误读capture内不存在的sha256字段，改为读取已审计manifest的source_sha256并核对真实路径；不改原始数据或运行worker。报告[collection_complete.json](evidence/tight_layout_20261116/collection_complete.json)记录全部36机会。

## 2026-11-16 运行前确认混合策略评估路由

learn评价准备显式使用既有可靠评估的GPU队列、实际配对检查与真实视频hook，避免缺省role分波路径绕过这些hook。新增check_plan.py用真实准备代码、替换模型身份读取做CPU检查：8case/4family、同family A/B完全相同reset、旧GPU后端配置、首family两视频均通过，0仿真/0更新。首次两核检查显式拒绝旧Kit16线程配置；服务按已有执行配置保留16核affinity、torch/BLAS仍2线程。未改物理/控制或门槛。

仅停止尚在等待且未准备学习root的旧wait01，保存superseded.json；新wait02使用新冻结源码，当前采集worker/monitor未动，所有截止保持。不是训练重启，也未重复任何物理机会。

## 2026-11-16 重采进行中与自动串行交接

独立验证6/6原生成功；训练前三条均成功，仍在顺序采集，不将计划配额记为完成。首新训练源296tick通过原loader、178D真实历史、29D命令及q[t+1]逐值对齐；初次额外检查误用浮点精确相等，改用原loader既有1e-8精度后通过，控制dt未改。新增只读readout与learn.wait_and_run：等待当前采集owner/monitor退出且数据审计完成后才准备并运行新学习owner，失败或原截止即停止，不并发抢占Kit。

## 2026-11-16 新布局完整学习与评估编排

新增scripts/tight_layout_20261116/learn.py：仅在36成功数据审计和采集owner退出后准备独立learning01，从随机DP/Transformer开始；所有24h/22h截止继承原17:02登记，不重置。A优先顺序两组DP，分别刷新后端计划/特征再训练后端，避免此前并行OOM；两组共同train30/validation6及相同40–80/20–40遍验证停滞规则。原随机合法起点/半数几何/B2/目标合同保持，不混旧布局。

自主开发固定预登记28820000–003，与28817000–039训练、28818000–005验证、28819000–007最终测试全部隔离；每个初态A/B共享原生reset随机流并核查实际状态，八lane一整波无复用、完整原生任务、首初态两视频。复用已通过的paired_eval与video_eval；监控覆盖所有串行stage，失败保存、不自动重试，1200s无学习/cache/control进展有界停止。主Agent审阅自主结果后继续DAgger，不把脚本完成等同科学结论。目前重采继续，训练未启动。

## 2026-11-16 采集监控根目录绑定修正

recovery01在Kit启动前发现子目录默认lease读取错误的shared_resources路径，0新物理机会，现场保留。显式传入同一owner监控根的lease，独立recovery02接续原未执行任务和原截止。没有关闭资源检查。

## 2026-11-16 新布局采集退出门已通过，修复下一进程启动目录冲突

10cm首两条独立验证示范均完成、完整capture关闭；采集子进程退出0、GPU PID消失且monitor仍活。第二阶段尚未启动Kit即因StartupDiagnostics独占创建已有startup_tracebacks.log而失败；没有新的物理机会。原root与两条有效源记录全部保留。恢复采用新root tight_layout10_20261116_01_recovery01、每worker独立startup目录，复用已通过gate与结果，跳过已完成identity，所有原session截止不变。未开始的其余34成功目标继续，未宣称完成。

新增显式fresh_layout_budget_v1：DP40–80遍，后端20–40遍，独立验证停滞先减半LR再有界停止；模型/采样/动作保持最近已验证设置，A/B相同规则，从随机学生权重开始。统一CPU检查覆盖仅XY平移、碰撞盒距离、历史及新调度/恢复状态，无GPU/optimizer更新；真实训练尚未开始。此前pilot全部结果已落盘，但Kit关闭无进展420秒后owner终止，不能称SDK正常退出。

## 2026-11-16 抓前真实几何补充

闭合numeric证据：原位seed01/03的tick24–36在桌面约0.001mm间隙的同时距基座仅0.009/0.044mm，根部静置高度比10cm约高29mm；5cm的seed03仍贴基座。10cm三例tick24–48基座净空15.45/17.91/18.23mm、桌面约0.001mm，静置z变化0.06–0.24微米。这支持减少自然落桌后的基座几何干扰；没有把非物体手指合力归因特定基座，也不据此声称学生成功。first_lift诊断排除初始自由下落（从48tick后计算），原生成功判据不改。contacts.py可重算；保留原numeric与三视频。

新采集owner先两条实际完整capture并检查进程/GPU PID消失、monitor仍活，再继续剩余配额。所有写入关闭后逐文件fsync再os._exit，明确不是Kit SDK优雅关闭。source将在新的冻结工作树运行；当前pilot关闭故障现场不覆盖。

## 2026-11-16 试摆9机会已保存，选择10cm新布局

原位3/3成功；5cm2/2合法成功、1保守准入拒绝；10cm3/3合法完整成功，表面净空89.51–92.35mm（原34.81–35.69）。三视频已审，10cm首例629步其余316/345，不能宣称专家速度提高。旧基座接触因果/学生收益尚未证实。Kit完成后关闭仍未退出；不抢占，等待原有限owner。新collect/owner使用既有关闭记录器后进程退出方式，首两条独立退出gate后才继续其余。配置/任务/训练初始权重隔离；详细表在集中诊断报告与layout evidence。

## 2026-11-16 当前：仅平移插销起点的布局假设验证

当前授权覆盖此前禁止新采集/训练与已结束15:30截止：先实际清理，原位/远离基座5cm/10cm小规模专家全任务比较；可用后新版本约30独立成功示范，A/B共同划分、DP与Transformer随机初始化，再完整DAgger共同冻结池。原孔位/资产/物理/120-60-5Hz/任务成功不改，不大改模型/采样/控制。A=q[t+1]+原几何，B=u_expert，后端均最终专家命令；学生前缀只是真实历史。

现场在continuous_vision_20261114，HEAD e6d48db与origin/feature/tight-state-foundation一致；narrow_followup同名本地分支落后且脏，保留。16:46无活动研究worker；旧可靠评估16次均合法、四组各0/4原生插入已结束。

新root tight_layout_20261116_01。已先登记并执行256份闭合失败RGB-D退休，保留数值/控制/结果/已有视频和36份完整代表失败，所有成功及1363个manifest源保护；净释放135318011904bytes。不是损坏数据删除，不重算旧清理。清单/逐文件SHA在root/cleanup。

新pilot登记3初态×3布局，共9机会、原有慢速专家/真实观察/原生完整成功；独立2h只界定pilot。位移只改native fixed_start_pose XY，孔位按原相同seed生成后冻结；超旧object采样框是明确新初始化，不裁剪。每初态保持z/旋转/关节/孔/事件seed，隐藏状态不称精确配对。真实碰撞盒表面距离与合法性、首组3视频后审查；偏移布局至少2/3原生全任务成功且合法、可见、支撑通过才进入重采，允许有依据的一次小调整。尚未启动/无新结果，不把准备写成有效。

## 2026-11-16 收尾 — 终态、零重跑续接及完整证据

reliable_eval补明确complete/failed协调器终态；已完成root提前只读核对，不启动monitor/Kit、不重写原complete/result。实际再次调用：4个完整作业跳过、0worker，SHA均保持。四套backend内嵌DP身份与执行回执一致，真实配对评估四组0/4。reliable_report/review_videos生成九段视频的事件、动作分解、小图和完整评估分母；现有集中报告/README/AGENTS/handoff统一更新。所有owner/monitor/worker已退出、资源保留。实际仿真源码c8d4c01，后续终态修改只做零GPU续接验收；不伪称新训练或插入能力改善。

## 2026-11-16 16:05 — 有界四套完整策略评估实际完成

接续8b946ca。scripts/tight_fulltask_20261116/reliable_eval.py仅运行既有四套完整模型；独立作业状态、唯一监控owner、串行Kit、固定四个开发初态，无新数据/训练。paired_eval.py扩展只读初始状态验收到相对基座的物体/孔/目标/手掌/桌面位姿和物体/目标速度；原生随机化范围、动力学和控制不变。四组实际完整运行通过交接，共16合法，四模型各0/4原生成功，0软件中断；监控在最后worker退出后正常结束。A/B关节、速度和初始控制目标差0，其他位置误差小于1微米。review_videos/reliable_report读取原9段视频和动作记录，产出实测失稳证据；无新视频替代分析。原2纠偏和所有训练断点保留；不强行续训独立验证已变差的两轮DP。集中报告与机器证据随后收尾；不是能力达标或长期无人值守可靠性声明。

## 2026-11-15 22:32 真实早期纠偏对照4/4；队列启动失败已定位

旧ba8ee36路径2lane、同A11/B09与grasp_early：4机会全部成功入库，850合法专家动作，445.893秒/32.295有效轨迹每小时。A67/A91/B67/B91在71/74/70/65步接手，都是学生真实接触偏差；属于专家救回，不是更新后学生改善。

b0e7da0新队列首次在动作前因错误假设专家无RNN而显式拒绝，16.03秒退出，原现场完整。实际专家为LSTM1024；新实现只清零重分配lane的h/c，按active lane更新专家，未分配lane不推理。新attempt将独立编号，不覆盖旧失败。其余33项短检查已通过；正式速度/训练收益仍待。

## 2026-11-15 22:25 采集吞吐优先，旧训练保留

用户最新要求已将正式采集吞吐列为工程优先级。04训练保持冻结；未启动的05边界launcher已停止并保留priority_superseded.json，不启动旧的大量完整评估安排。新增GPU观察/历史/控制合批、批量CPU快照、向量专家队列映射、独立lane补任务、空闲标准相机停更、固定外参缓存及抓持阶段评估入口。实际队列/相机验收与正式加速尚待；当前不是成功或提速结论。

11项短检查通过，正式FP32下真实A11权重/记录输入配冻结patch fixture的最大目标差5.18e-7rad；初次CUDA检查遗漏正式precision设置的失败已说明，没有放宽最终2e-5rad检查。旧路径4机会真实早期纠偏对照已启动于04/throughput_baseline15_01，来源A11/B09；新队列将用同负载比较。全部保留04固定02:05:20截止。详情见scripts/tight_fresh_throughput_20261115/README.md与对应证据目录。

## 2026-11-15 21:30 实际边界服务与补充空间记录

ba8ee36已推送并冻结，边界service实际waiting_current_cycle；A14已20遍3300更新，B14训练继续。保存精确启动命令/截止/源SHA，追加两份旧已结束非best端点退休清单与执行脚本，保留全部best/raw。未声称第十五批已经采集或自主改善。

## 2026-11-15 21:20 抓持优先候选实现与现场衔接

新增grasp_curriculum、rear_episode_sampling；takeover_trigger/training_expansion/expansion/stage_takeover/report/batch_run接入每tick早期接管、失败训练family重访；backend/anchor_continue/cycle接入完整后端latest重复与真实曝光。train.metrics增加可选150步前误差，grasp_fit/evidence/review记录两网络及持续持物。research_loop/common launch/followup支持行为复核及继承绝对截止；scripts/tight_fresh_20261115_05含固定20机会、等待交接、完整运行、统一检查及精确退休脚本。

输入不变：每60Hz真实RGB-D/本体/动作历史，A q[t+1]、B及rear专家绝对目标；输出原29D目标。采集诊断不成为策略输入；源模型A11/B09，当前14冻结进程未热改。6项短检查通过，历史12失败可更早触发，尚未证明新专家救回或自主改善。14已新增23条，池306/162family；仅16是fixed1补充示范。实际结果与限制见最新handoff和grasp_followup05小报告。

## 2026-11-15 第十四批真实启动

冻结3d148de的独立session04于20:05:20建立，固定次日02:05:20截止，旧03不改；20:07已观察16环境第50控制步。新增只读report_startup绑定实际session/plan/进程回执，不轮询模型或重跑仿真。后续新数据量、8worker吞吐及自主结果待完成，不能把启动写成研究成功。

## 2026-11-15 第十三轮完整结果与早期/后段平衡接续

- report_completed13汇总已闭合15个合法numeric的现有统一分析，A0/7、B0/8插入，A持物下降，B一个8.4°近孔持物状态最终掉落；不整体晋升，下一采集候选A11/B13。正式整轮204.279分钟，预取前端较快但整周期/采集未提速。
- training_expansion和CLI新增fixed1，沿原完整学生/专家bridge执行1步学生前缀及合法队列mask；用于补接近和抓起，不把专家成功当学生成功。
- followup_session新增显式新session磁盘reserve参数，旧参数缺省不变，源session字节不改；启动脚本透传。新04包32机会、8校验worker、repeat2、20/15遍，独立68GiB准入/32GiB保留，尚待启动。
- 按固定清单退休30个完成训练的非best last，释放4.901GB；raw/all best/原05与当前冻结特征保留。统一2项短检查1.51秒通过，配置实际验收通过；未扩大旧检查矩阵或重跑旧仿真。

## 2026-11-15 第十三轮前端完成，后端正常接续

复用report_progress13小回执入口发布19:24快照：A/B各20遍3800更新，选18/14遍，新32条分别72/56 visit。两组前端90.509分钟，相比上轮98.055分钟减少7.70%，工作量不同；新后端输入10.075分钟已经完成，后端各约8/15遍继续，尚无自主结果。仅更新小报告和进度文档，活动代码/训练不改，无新增检查矩阵。

## 2026-11-15 第十三批闭合与预取正式训练进度

- 新增32成功纠偏/8325合法动作，含24条持物相关/5609动作，已入283条/149训练family池；采集58.752分钟、32.680成功/h，未实现采集提速，写入和结束等待合占36.83%。
- A完成20遍/3800更新，selected18遍的新32条各72 visit；几何覆盖75,824不同轨迹时刻。正式前端阶段47.518分钟，上轮57.839分钟/2880更新；数据和重复量不同，不作单因素或整周期加速结论。B在18:28快照第6遍，尚无新自主结果。
- 新report_progress13复用闭合采集与小JSON/validation回执，记录实际预取命中/等待/内存回退及服务状态，已真实生成报告。同步handoff/README和启动包说明，0新模型/原始轨迹读取，活动冻结源码不改，不重复已通过检查。

## 2026-11-15 预取接续时段实际启动

19736be独立冻结运行，session03为16:23:23至22:23:23，旧02文件/截止保持；已观察16环境批次第25个控制tick，owner/monitor正常。实际source pair、64机会和启动小回执公开；预取正式训练收益与新自主结果待完成，活动源码不热改。

## 2026-11-15 第十二轮收尾与有界批量预取

- 完整36新增/21持物相关成功纠偏进入251条池并训练；A0/7、B0/8自主插入，持物比实际来源退步，12未采用。发布采集、曝光、行为、整轮196.576分钟与每组约2.94TB实际读取记录；旧报告入口增加可选round参数，默认11保持，0重采/重训。
- frozen_visual_io新增v2，trajectory_prefetch接入正式B2前端，单worker最多下一批3GiB，12GiB RAM保留与前台回退；保留完整独立历史、顺序、张量值和更新语义。统一真实CPU数据/顺序/内存回退/退出检查2.568秒通过，0GPU/更新；正式性能尚待完整流程。
- 570个精确分类已结束派生payload释放38.2277GiB选定块，原始数据/selected best/05/最新12保护；清单SHA与执行日志公开小报告，不重扫原始payload。
- scripts/tight_fresh_20261115_03复用有限owner，登记64机会中56持物后段、前端新纠偏repeat4、20遍B2/15遍后端与同开发初态。准备不等于启动，旧时钟不延长。

## 2026-11-15 后段重点时段实际启动

- 已提交启动包1d4028c在独立冻结树运行，session02固定12:24:06至18:24:06，旧session01文件与截止未改。
- new_family_12实测wave0/control tick100，A11/B09来源与64机会实际plan一致；发布小型启动回执，未提前计入成功数据或训练结果。原第十一轮完整结果已公开，活动源码不热改。

## 2026-11-15 第十一轮完整收尾与持物重点接续配置

- report_cycle11复用原09完整结果，仅分析本轮15个合法闭合numeric，不开图片/权重或重跑仿真。A持物2/7→5/7，首次原生near累计7/10；B持物5/8→4/8，完整插入仍0/7与0/8。保留非法初态及原失败/恢复时间。
- 发布各阶段真实耗时、后端15遍实际监督和自主逐例结果；A的两个最佳姿态时刻分别记录，不拼成原生成功。
- scripts/tight_fresh_20261115_02登记A11/B09、共用215数据、held占48/64机会的独立6小时单轮；复用既有owners，统一配置检查通过，尚未启动，不修改旧截止。

## 2026-11-15 前端正式学习完成与实际覆盖

- A/B各20遍、2600更新已完整结束，已自动进入新权重后端输入刷新；A选20/B选19。新增45条每条末遍40次，B选中模型38次。
- 新report_frontend11仅复用小回执，记录51,012个不同随机起点、A58,821个不同轨迹时刻的几何覆盖、离线验证及实际86.032分钟前端耗时。B selected缺少纠偏cohort指标保持null，不冒充末遍，不宣称自主收益或整周期提速。
- 更新前端报告与handoff/README；无活动源码修改、无新增测试/仿真/重训。

## 2026-11-15 同类报告读取常数修复

只读summary中还残留同一185字段假设，改为已完成回执的正整数字段计数；成功与来源绑定不变。现有live_snapshot测试扩展CPU185/GPU186，两例通过，未重跑前述42项或正式数据审计。活动训练冻结树d0b7b5f保持，修改仅用于后续报告读取。

## 2026-11-15 固定时钟恢复实际启动

- d0b7b5f已冻结运行，复用45条完成纠偏，aggregate23.509秒后A/B池各215条/103family；正在生成缺失冻结视觉，前端/自主尚未完成。新启动小证据公开，旧失败不改写。
- report_collection11从已有四波小回执追加完整耗时分解，主要是含传感/渲染的环境推进41.45%、写入及验收合30.82%；0大payload读取。

## 2026-11-15 已完成GPU采集的训练交接修复

- tight_fresh_collect_cycle_owner使用已完成source_verified回执与精确capture绑定，替代CPU特定185字段常数；GPU186个真实字段仍逐项由原verifier校验。tight_fresh_research_loop把内层异常传到顶层状态。
- 42项统一短测试2.18秒通过，实际64机会/45成功回执准入通过，未重扫payload、未重跑采集。新增可复现小报告和固定原截止recover_cycle01入口；旧失败及冻结采集证据保留。
- 正式全批产出45成功/11,495合法动作/27新family，57.61分钟/46.87成功每小时；新训练仍待恢复启动，不将专家救回计为自主成功。

## 2026-11-15 GPU正式恢复首波已完成并接续

- 01453ae实际冻结运行，首波16合法机会/11接管/10成功纠偏（2643合法动作），含held0两条434动作；16 capture原186字段校验完成，无RGB错误，已自动进入下一波。
- 原12小时时钟不变，归档首次monitor后恢复，缺测496.259秒明确记录。startup/first_wave小报告与现有脚本公开；成功纠偏尚未训练，新随机/B2完整周期结果待收取。
- 第九轮完整周期与下一轮实际交接已观察，依用户许可由有限普通流程后台继续；不是自主插入成功结论。首波含admission33.269成功/h，未达到5分钟50条目标。

## 2026-11-15 修复GPU相机RGB严格保存

- sim/environment.py + sensor_rgb.py：CUDA真实uint8采用1KiB canonical lookup，CPU旧路不动；metadata显式身份。compact/recording原严格lossless判据未改。
- tools/tight_rgb_storage_check统一CPU/CUDA全灰度及真实compact写入读取检查通过，0Kit/0训练；旧GPU126/256灰度1ULP不一致已实证，修复全部逐位/原像素一致。只执行一次必要检查，不重复控制/历史/训练验收。
- 原new_family_10首波16环境各1控制步后软件中断，0完整轨迹/0入池，不能算任务失败；新attempt沿用相同初态和固定14:58:38.966截止。

## 2026-11-15 第九轮完整行为与周期结果

- 发布round09_behavior逐次结果/实际分析脚本、cycle_throughput_review/round09整批耗时及round09_front_learning/fit_report；复用既有结果，不重读模型/RGB或重跑旧评估。
- 新52成功纠偏入170条训练池；A0/7、B0/8完整自主成功，持物机会结果混合、仍无对孔/子目标，不整体晋升。前端、后端输入、后端、自主及全部周期时间分别报告。
- 新独立12小时session已实际启动，但GPU16采集因RGB浮点归一化的严格无损保存错误停止，0新完整轮/0新训练。原失败现场和固定截止保留；后续修复采用新身份。

## 2026-11-15 02:31 第九轮前端学习与统一读取证据

新52/held13的selected及last实际监督曝光、验证selection及B selected无纠偏cohort指标分别记录；复用小JSON，不重读模型/旧numeric。统一865帧读取对比完成，后端M32刷新176条各263.19秒已进入正式流程；完整周期和自主结果待收尾。

## 2026-11-15 02:26 冻结视觉文件读取优化候选

- native_data.Episode及新frozen_visual_io：opt-in直接eager读取当前dense frozen visual shard；不clone、不改tensor、不缓存learned state。frontend B2/metrics及rear cache配置接通并记录真实读取时间/字节；每批/轨迹释放。
- fresh_cycle/collect_cycle_owner/research_loop及实际启动包完整绑定profile，下一轮顺序前端、并行后端。旧运行/默认mmap路径保持，未宣称已获正式收益。
- 统一training_check增加CPU单train-shard字节/layout检查，复用已完成随机训练/控制证据。实际60秒I/O等待证据与完整周期计时分开；真实865帧统一CPU检查0.756秒，四类tensor/layout全部逐位相同，0GPU/0更新。首次exec权限拒绝0worker，独立bash启动attempt完成；后续用真实新周期判断效率。

## 2026-11-15 第九批完整收尾与独立连续运行启动包

collection09最终128机会/125合法/56接管/52成功入池，14572合法动作、34新family，完整池170条/76family。耗时6723.692秒、27.84有效轨迹/小时；held60/120新增13条/2049动作，fixed80为24/24，专家救回与学生自主0分开。复用旧审计及仅38个成功closed文件的小numeric字段，无大SHA/RGB/模型重读。README与研究入口已更新。

新增scripts/tight_fresh_20261115_01启动包：登记GPU16采集/CPU8评估、每轮32新family、持物即接管、20遍B2/15遍接续后端、原源视觉index复用及半数几何。默认纯stdlib静态预览；明确--launch且旧09完整结束/自有worker与monitor退出后，才新建独立12小时session与有限3轮服务。当前仅准备，不重置06截止或声称已启动/已交接。原运行源码87577e8保持。

## 2026-11-15 统一检查后接入持物即接管及有限连续流程

- 新held0复用真实完成transition的持续12tick持物条件，min_delay=0后下一动作接管；原60/120语义不变。修改takeover_trigger、training_expansion、正式expansion与collection_summary；27项短CPU测试通过，尚无held0正式数据。
- tight_fresh_followup_session显式allow_illegal_starts复用已有逐case审核，保留零动作非法初态与合法自主分母；不替换seed。默认要求全合法的旧行为保留，现有55项相关测试通过。
- tight_fresh_research_loop登记有限轮次/独立采集评估后端/原session时钟、失败保全与新身份衔接；普通脚本不宣称科研判断。源码与实际脚本一同发布，尚未启动。
- 统一真实检查4次更新完成，采用legal_tick_half_phase_v1；不同构型与真实梯度见sampling_training_review/check01.json，非正式学习或自主收益。
- tight_retired_cache_cleanup默认dry-run，按已审阅manifest精确删除退役衍生.pt；6873文件/53.773GiB实际删除，raw/fresh/best/统计与所有目录保留。修正重复别名候选，旧拒绝记录保留。

## 2026-11-15 01:42 — 随机合法起点、几何覆盖和正式采集校验复用

- anchor_sampling/continuous_vision/trajectory_batch及fresh_train/cycle：独立visit种子、完整history一次编码、真实非grid起点；fresh_visit_sampling记录不同anchor与几何位置并进入checkpoint恢复。旧默认及部署5Hz不变。
- geometry_sampling/anchor_continue/state_image_loss：仅train诊断阶段分层、不同合法构型去重、约半抽样/显式cap、所选均值及阶段实际比例；默认旧2构型保留用于复现。实际耗时/全模型梯度尚待统一4batch检查。
- training_receipts/learning_report/stage_exposure按实际visit起点累计监督与去重覆盖；固定5Hz离线评估保持。新增统一tools.tight_fresh_training_check，0heldout/最多4更新。
- state_benchmark/narrow_data/batch_run/expansion/batch_owner：opt-in capture-audit-memo，bounded SHA/模型匹配元数据复用、文件变化失效；下一正式pending16，当前冻结采集不变。测试复用各模块短check，root接线22项通过。

## 2026-11-15 01:25 — B首波后段与实际脚本可复查归档

- `collection09_registration/B16.json`：8真实恢复/2493步、两held恢复227步、未触发原因、原生子目标时钟及A/B非严格配对，全部来自closed numeric/真实label审核。
- `reproduction_scripts/session06`：10个实际CPU审计脚本原字节、manifest/README/真实启动核验；原输入和输出SHA、分析代码与collector不同依赖、覆盖风险、未知原HEAD/argv分别保留。主Agent10份逐字节/SHA检查通过，无重跑、无大数据发布。

## 2026-11-15 01:20 — 恢复后数值差异的独立定位工具

- `tools/tight_trajectory_recovery_probe.py`及测试：固定原train singleton与完整磁盘边界，最多5更新，分别同对象/新对象/default/确定性；逐参数裁剪后梯度、模型/EMA/Adam/RNG的exact与误差描述及分项计时，无放宽原gate或正式trainer修改。
- 主Agent20项CPU检查通过（2.37s）；真实输入check-only2.571秒通过，CUDA0/更新0。原失败源与辅助代码逐SHA绑定，原deadline不变，真实GPU运行待资源准入。

## 2026-11-15 01:14 — 未完成周期的真实阶段回执与B2失败小证据

- `tight_fresh_cycle_throughput.stage_receipts`原partial仅查不存在的ledger子目录；现读取真实stages根，兼容历史ledger，拒绝重名阶段/错文件名，worker/status不计完成，失败区间保留且不伪造completion。完整finished路径保持；真实第八轮除报告创建时间外逐字段一致。主Agent两文件30项通过，Agent扩展含读者41项通过。
- `trajectory_training_review/B_gate01.json`保留5次实际更新、恢复边界通过而后续exact失败的分组计数、原输入/输出SHA及退出状态；不改变原门槛，不将局部计时当完整训练加速。

## 2026-11-15 01:09 — 独立持物root近目标入口与真实恢复失败记录

- takeover_trigger/training_expansion与fresh_expansion、stage_takeover、report、batch_run、batch_collection_review新增显式held_root60；已完成post rigid-root/当前goal前后一致/held12，方向不设门槛，原near60与成功语义保持。
- 新test_held_root_goal_trigger覆盖双子步接触、goal切换、边界、lane独立、真实HDF回放、登记防漂移和末帧无下一动作；主Agent组合137项通过。near_root_takeover_review绑定真实8条train审计和A52假设触发，无新采集。
- 冻结761f53b B2实际更新/磁盘恢复gate完成但逐值后续更新不通过，明确未采用并保留5诊断更新/15.231秒现场。当前87577e8完整采集/下一轮未改。

## 2026-11-15 00:57 — A半批成功后段覆盖证据

- `collection09_registration/A64.json`：64合法/26接管/23成功、6098合法专家步；held24/7/6/957与5个旧stable排除但成功边界/763步，实际来源和私有审计SHA齐全。
- 数据已准入但尚未用于round09训练，B半批仍运行；不将半批吞吐、专家恢复或窗口判据混为完整自主成功。新B2真实更新gate尚未资源准入，活动作业不变。

## 2026-11-15 00:51 — 显式B2完整轨迹更新与真实恢复验收入口

- `trajectory_training.py` / `training_receipts.py`、fresh trainer/cycle：独立B2身份、完整历史、实际batch均值、逐visit/更新预算与deck恢复绑定；旧单轨默认保留，互斥pairedIO/prefetch。
- learning_report/stage_exposure/stage_fit/cycle_throughput和专属测试：按真实visit展开，selected/last计数各自核验，未知I/O/未完成周期不补造结果。
- `tools/tight_trajectory_training_gate.py`与测试：两条真实train最多5次诊断AdamW/EMA更新、serial均值参考、磁盘checkpoint恢复、明确有限边界；真实CPU准入通过，CUDA尚未运行。
- 主Agent292+15项CPU通过；完整epoch且selected/validation凭据完整的恢复获验证，多文件收尾窄窗口限制明确保留。活动冻结第九轮未改、未重启。

## 2026-11-15 00:40 — 已闭合后段恢复与目标可观测性证据

- `evidence/tight_fresh_20261114/target_observability_review/summary.json`：5条真实closed train、冻结源码SHA/行号、原目标序列、真实三视角检查及接管剩余时钟；0新仿真/0更新，不公开原RGB/大轨迹。
- 区分固定阶段部分可观测、尚无完整历史混叠实证，以及第一目标阶段的实际转正失败；晚接管超时不能证明物理不可恢复。正式输入/成功判据/活动冻结源码不变。

## 2026-11-15 00:38 — 完整采集、训练、缓存、自主吞吐核算

- `tools/tight_fresh_cycle_throughput.py` / `tests/test_cycle_throughput.py`：CPU JSON/文件stat报告，严格完成、输入绑定和累计计数，实际并行区间并集、读盘未知值和数据量差异；未完成或失败不能宣称完整周期。
- 主Agent定向29项测试通过；真实第八轮76.15分钟训练缓存评估、19条采集45.07条/小时与阶段工作量写入 `cycle_throughput_review/round08.json`。未增加GPU工作、未修改当前冻结87577e8作业。

## 2026-11-15 00:24 — 采集阶段汇总与B前端组件诊断

- `tools/tight_collection_summary.py` / `tests/test_tight_collection_summary.py`：JSON-only partial和完整审核分开，严格accepted manifest/policy/source绑定，held重查/实际专家阶段分母；15项CPU通过，首波真实小证据发布。
- `visuotactile/batched_frontend_diagnostic.py`、`tools/tight_fresh_frontend_component_gate.py`及专属测试：实际trained B EMA、独立M8历史/反馈、原确定性投影；train-only同输入准入后有限两臂自然终止，清晰组件身份与诊断资格，原deadline和失败清理。CPU已验、实际Kit未跑。

## 2026-11-15 00:21 — 真实合批门控与按角色历史来源比较

- `trajectory_batch_review/gate01.json`：冻结d17b1f8 A/B真实CUDA前向/梯度通过、实际命令/完整报告SHA/原截止/退出状态；局部速度与内存上升均实测，正式trainer未改。
- `tools/tight_fresh_round_compare.py` 与专属测试：显式每role资源根/执行目录/冻结树清单，严格原计划、实际wave/lane和合法交集，保留旧CLI。49项CPU通过，A08/B06真实完整预检4.02秒完成；`round09_source_review`保留小摘要和可复用清单。

## 2026-11-15 00:16 — 独立完整轨迹合批候选

- 新增 `visuotactile/trajectory_batch.py`、`tools/tight_trajectory_batch_gate.py` 与两份专属测试：逐轨完整历史/独立GRU/逐轨归一化与几何、保留噪声抽样次序；只作同参数前向与梯度验收，正式trainer不变。
- 12项CPU测试及真实B train-only CPU准入通过；CUDA尚未运行，不能报告提速或训练收益。停止/截止检查涵盖计算结束和最终落盘前。

## 2026-11-15 00:00 — 第八轮自主结论与128机会正式扩充

- `evidence/tight_fresh_20261114/round08_review`：同开发队列完整CPU审计、真实selected曝光、合法交集与原生判据。A0/7/B0/8，持物局部延长但无对孔/插入，回退保留。
- `collection09_registration/startup.json`：实际冻结87577e8、A08+B06来源和118池独立绑定、64新family/128机会、持久owner与完整后续cycle精确argv/原截止/首波启动。CPU owner/来源联合52项通过，真实采集结果待。当前正式代码不启Tiled、exact-tail或inactive-render。

## 2026-11-14 23:51 — 持久采集→完整训练评估 owner

- `tools/tight_fresh_collect_cycle_owner.py`及24项专属CPU测试：冻结registration、显式命令与输入绑定，有限采集/完整机会与成功数据准入，预留训练时间，逐阶段日志/完成SHA/PID与跨组子孙清理。不启新monitor、不重置截止、不自动retry；实际运行待模型选择。

## 2026-11-14 23:46 — 后端批量缓存 CUDA 全字段验收

- `evidence/tight_fresh_20261114/backend_cache_planning_review/gate01.json`记录冻结3e53273 A/B原始与纠偏真实全缓存对比、原门槛/计时混杂/owner清理。两组通过，下一正式cycle采用显式M32；当前运行82d5052不改，正式收益尚待。

## 2026-11-14 23:42 — 纠偏动作曝光、同session来源与独立离线批量规划

- `tools/tight_fresh_stage_exposure.py`及专属测试：冻结H24 masks与实际逐更新/selected/last计数核对；A/B真实CPU结果与34条四臂拟合已公开，旧注册不重写。
- `tight_fresh_collection_source`及expansion/training_expansion hooks：原session内独立绑定扩充数据池与有自主证据的真实A/B pair，非法零动作保留、来源writer退出核验，防止模型数据身份偷换。
- `tight_fresh_train/cycle`与offline metrics测试：显式M32 metrics接入、阶段计时/模型配置绑定；前端门控已通过，正式使用待。`paired_episode_io`与测试增加三处STOP同步检查。
- `native_data/anchor_continue/tight_finish/tight_fresh_backend`及full-cache gate/测试：显式离线条件批量计划，配置/index/记录profile一致，逐tick视觉/历史/监督合同不变；真实全字段门控尚待，不视为已采用。
- `round08_stage_fit_review`与`inactive_render_gate_review/gate02.json`记录实际拟合收益与相机跨波空图失败；不能代替自主成功或正式吞吐。

## 2026-11-14 23:25 两组离线批量规划实际门控与渲染诊断包装修复

A_gate01/B_gate02均用第八轮已选best EMA、各原始/新增两条train实际完成M1/M8/M32 CUDA对照，固定raw/bounded 2e-5rad全通过；无参数更新、无验证/测试轨迹。原B_gate01的split失败仍保留。B两条serial DDIM3.610/6.483秒、M32为0.169/0.466秒，最大目标差9.574e-7rad；同时有后端/阶段拟合作业，不能外推整轮速度。明确采用M32作为下一正式前端metrics的显式候选，尚未进入正式训练；后端逐tick视觉/完整计划缓存需另门控。详见offline_planning_review/live_gate02.json。

inactive_render_gate普通arm缺session的包装错误已修：verify frozen plan后exclusive byte-copy原session，源/本地SHA核验、保留精度/专家/后端/原deadline，环境前再验；不重写配置。23项新旧CPU检查通过；原01的0控制/无Kit证据在inactive_render_gate_review/gate01.json。修复尚未真实Kit验收。第八轮A后端15轮及离线已结束，B缓存准备中；四臂阶段拟合已完成A来源、正在A选中模型，未产生自主结论。

## 2026-11-14 23:21 实际门控失败与四臂阶段拟合监督

offline_planning_B_gate01在0.657秒CPU清单准入时因原始entry无split失败，0优化/0CUDA验收；当前state_data.splits才是权威分区。修复兼容缺省字段，存在矛盾字段仍拒绝，train ID/family隔离不放松；真实A/B各6validation拒绝检查通过，38项CPU测试通过。原01保留，后续只用新编号。inactive_render_gate01普通arm在2.271秒因缺少arm/session.json启动失败，未进入Kit或控制；这是诊断包装错误，不是渲染候选物理结论，原现场保留，修复尚待。

新增tight_fresh_stage_fit_owner绑定已准备的A/B v3注册和冻结eval08@2e38c91，四臂独立进程依次前向、再比较；原session截止与1800秒上限取较早，逐stage保留PID/starttick/PGID/命令/日志/结果SHA，STOP与残留子孙回收受测试。CPU21项检查通过，最后超时归因另5项定向复核；真实34条注册只读预检通过，尚未启动。GPU诊断将与现有后端训练适度重叠，计时不用于单变量加速宣称。旧monitor不认识run-arm参数，将依owner PID绑定补充归属说明，原遥测不改。

held_recheck_review公开摘要已审查：旧delay12共12次登记机会→6次接手→3次边界仍held；08 near60八合法→三曾held→一真恢复，回看held60额外只有A28400007未执行候选。专家截断后未知学生未来不外推；73.24%是早入口轨迹的完整专家后段占比，其中939步也held，不能叫73.24%早阶段动作。

## 2026-11-14 23:17 后段接管重查与下一轮吞吐候选

第八轮82d5052两前端均完成10轮/1370更新；A选epoch9，B选epoch5。cache_A已543.205秒完成，当前后端A训练，尚无第八轮自主结论。前端A/B进程read_bytes分别629.079/626.656GB（精确值见offline_planning_review），同一时段出现187.66/188.31秒consumer等待，slow row主要停在mmap clone之前/期间，resource pacing为0；这些是进程计数，不能当作独立物理盘读量。旧单epoch paired I/O整体慢的结论保留，下一较大规模需重新判断摊销，不自动采用。

新增held60/held120显式接管：首次12步持续抬升与双指接触后至少等待60/120控制步，真正接手前最新12步必须再次满足；允许期间失去条件后恢复、允许原生goal变化，不用距离/成功/掉落判断替代触发，无fallback。真实串/批入口和HDF仅重放学生控制段，完整保留专家后段。旧contact0/12和near60不改；回看delay12有6次实际接手，仅3次在接手边界仍满足held，旧成功记录不重分类为学生拿稳。

下一cycle可显式--parallel-backends：两前端完成后使用新遥测做22GiB显存/24GiB RAM联合准入，A/B各自prepare→cache→train→offline，全部结束才自主评估；不通过则顺序运行，剂量/参数不变，原deadline/STOP与systemd cgroup清理保留。仅候选，当前活动冻结树不改，尚无并行收益测量。

新增独立offline_planning helper/gate，完整60Hz历史重建后将连续5Hz anchors按M1/M8/M32推理，共享generator逐行原形状draw保留prefix/gap噪声消耗；固定raw/bounded目标差2e-5rad，train-only best EMA，默认训练路径不改。主Agent本批104项CPU检查通过（含离线门控、后段接管、并行后端和配对身份），CUDA验收尚待；metrics阶段28.62%/29.84%含编码/I/O/租约，不能全归DDIM。round_compare把inactive-render显式配置纳入执行身份，防止不同渲染策略冒充同配置配对。

## 2026-11-14 23:03 第八轮新增数据曝光与合集阶段拟合

已按不可变source SHA、源checkpoint train_ids与当前合法mask核实118条/42训练family/32920合法专家步。A05首次接触06/07/08共34条、9005步；B06首次接触07/08共25条、7155步，双方真正新增初态均12family。B06选中epoch3对第六批每条6次完整历史，last epoch10为20次，不能混算。详见evidence/tight_fresh_20261114/round08_exposure_review。

stage_fit显式--include-collection启用v3，保留v2默认：全部34、latest19、逐批、source已见/未见分层，同一真实历史分别由source/selected EMA重建，保持5Hz同槽q/实际target基线。特征SHA匹配后仅投影train，当前可学习GRU不跨模型复用；缺逐阶段优化器target槽记录仍标unknown。Agent41项CPU检查通过，主Agent27项阶段相关检查通过（集合重叠）；实际GPU拟合尚待完整cycle退出。当前B前端已完成10遍1370更新/33340anchor/764010目标槽，A10遍更新已结束、末次评估和保存中；尚无后端或新自主结论。

## 2026-11-14 23:00 已结束环境渲染候选接入正式入口

新增显式standard_inactive_render_products_v1，接入batch_run、expansion、cycle评估与真实run_wave。只在CPUstandard使用现有三视角产品；每波reset前恢复全部，首观测保留，非法/未分配lane或自然结束且末步已提交的lane才停用。真实活动lane继续每60Hz三视角、独立历史；缺省不绑定或切换相机。每tick验证新Hydra drawable、同当前物理clock及CameraParams姿态，相邻观测必须research+1/physics+2/time+1/60；停用画面显式invalid，不能作新观测。直接RGB/depth输出buffer与renderer frame的等同尚未证明，不声称硬件曝光时戳。

新tight_inactive_render_gate prepare→owner在独立Kit臂各执行A/B两波36步，分配槽位16→8→1；非法0动作不替换、lane0必须合法，两个退休区间需真实合法lane。普通臂全部产品保持绘制，候选只停非法与诊断退休；实际合法集合不同不作可比计时。固定prepare起算600秒（最多900）、原session截止、30秒清理储备、进程组绑定/恢复/失败凭据；诊断删减仅存在门控，正式没有人为裁尾。主Agent83项相关CPU测试通过；真实Kit门控与正式吞吐尚待，当前冻结82d5052训练不改。cycle的--payload-audit-memo已显式透传到缓存阶段。

## 2026-11-14 22:51 真实相机候选结论与缓存重复校验优化

GPU Tiled 同步候选的独立CPU复核已完成：两个固定相机米级位姿错误已修复，但第二视角仍全黑，腕部仅两lane有少量灰像素；首包24张raw/cache逐值一致。4次render-only仍未恢复三视角，未增加动作/训练数据。Tiled不采用，正式继续standard；详见evidence/tight_fresh_20261114/tiled_pose_sync_review，原失败现场保留。

冻结DINO缓存的A/B索引实际指向相同inode，旧round06/pool07的源码路径分别重复读取约47.32/50.08GiB进行SHA，非物理磁盘流量测量。新增显式--payload-audit-memo：同一进程首次完整读取，随后只对匹配dev/ino/size/mtime/ctime的已算SHA复用，每次仍核对预期SHA、source和帧合同；跨PID拒绝、改写/替换检查、有界4096元数据条目，不缓存可学习GRU。默认原路径不变；主Agent22项CPU测试通过，正式节省耗时待后续实际缓存阶段测量。当前运行的round08冻结82d5052不热改，已完成124条视觉缓存、并行DP正在推进，尚无新自主结果。

## 2026-11-14 22:42 第八轮完整流程运行，尾块严格门槛失败保留

冻结82d5052的tight-fresh-round08-cycle-20261114.service已启动，aggregate实际核对118训练轨迹/42训练family、6独立验证、原30归一化，新增19条。DP10遍、最新前端重复2、后端15遍、1GiB有界冻结特征预取；DP后刷新计划与可训练视觉依赖，再进行原M8/CPUstandard8完整自主。当前处于视觉缓存阶段，尚无第八轮行为结论；阶段凭据在session06/round_08_stages，固定截止未改。

exact_tail_gate01冻结a43f318实际CUDA检查在A第8个索引tick首次7lane时失败：frontend_memory/frames/2/0最大差4.11272049e-5，未过原atol/rtol=2e-5；此前8个相同M8 tick目标逐值相同。失败tick目标误差未写入v1，B未运行，不能宣称完整动作一致，也不能当作任务失败。候选不采用，原失败保留。新增单独--diagnose-exact-tail模式保留原阈值和strict失败，只继续记录数值历史差异；目标、非有限值、离散历史/数量、RNG、停用状态和真实反馈错误仍停止，始终adoption_eligible=false。新实现另保存失败tick已计算的目标误差及实际视觉计数，22项CPU检查通过；新诊断尚未运行。

## 2026-11-14 22:34 新初态来源自主完成与尾块实际门槛入口

来源A05/B06在8个独立development初态上完成16机会，A7合法/1动作前拒绝、B8合法；分别0/7、0/8插入。两组原生抬升各2例、旧拿稳各2例、允许转动的持物窗口各4例；所有合法执行601步超时，无原生近孔/对准/子目标。owner475.701秒。round_compare新增显式development路径、实际混合来源pair/冻结源码/批量后端/每帧历史及目标时钟核验，非法初态分母和两组都合法的交集分开；旧默认保持。27项CPU回归和完整实际来源审计通过。

新增fixed_m8_exact_tail_complete_student_gate_v1：相同train-only真实RGB-D、完整A/B学生、活动lane8→7→5→1、活动lane0独立历史reset，实际执行反馈teacher forcing明确。目标绝对误差<=2e-5，完整连续历史atol/rtol=2e-5，离散状态、RNG、停用lane及实际反馈严格一致；拦截实际backbone形状并保存独立算子CUDA计时。50项Agent CPU回归通过，主Agent发现并修复角色结果父目录缺失；真实CUDA门槛尚待，算子计时不等于正式吞吐。

## 2026-11-14 22:27 第八批实采完成与视觉尾块候选

冻结d168195的new_family_08全部32机会及文件准入完成，owner exit0、自有组无遗留进程。29合法、3初态包围盒准入拒绝；14固定接管和5条件接管全部专家恢复成功，19条/5795合法专家tick，覆盖12个新训练family。near60仅A05一例触发并救回，B四例均未触发；学生自主成功0。owner1517.518秒，即45.074有效轨迹/h、13747.45合法专家tick/h；初态/接管配比与旧批不同，不作纯并行因果结论。6条fixed40学生前缀不够完整高度参考，窗口记不可用。

来源A05/B06在8个独立新development初态上的完整批量自主已由tight-fresh-source08-eval-20261114.service启动，仍冻结d168195/CPU standard8/M8。已保留A seed28500003非法初态、0动作，不补种子或记为行为失败；完整结果待结束。下一第八轮将合并至118条、42个实际训练family（待aggregate核验），进行DP更新→缓存刷新→后端更新→同队列自主；原6validation保持独立。

新增显式current_observation_exact_tail_v1，贯通native_vision、batched_controller/rollout、batch_run/eval、expansion和cycle评估入口；默认合同保留。实际骨干核形状与零padding计数受核验，允许末尾少量活动环境减少重复计算。主Agent66项相关CPU回归通过，Agent88项集合有重叠；真实相同观测门槛和正式收益尚待，不热改当前冻结流程。

## 2026-11-14 22:16 一步构型标签与真正规划跨度的CPU核对

train99的27125合法专家tick核验q_before、同transition q_next、实际u和1/60s时钟。复制当前q的一步arm/hand RMSE为0.001197/0.017546rad；99.893%手臂单关节步差<=0.005rad，但手指仅49.042%，不能笼统称整手恒等。真实onset-aware 5Hz anchor固定当前q到未来24步，误差增到0.022976/0.281239rad，完整计划并非由一步小变化即可判断无意义。u-q_next包含动态跟踪、限制和接触，不是压力标签；本次没有据此贸然改变A监督。

已复用旧round03_frontend_baselines：同2062验证tick，A/B手指DP均胜过5Hz保持，手臂稍差；没有重新读取验证轨迹。新增train-only stage_fit v2在相同合法commit12槽加入anchor处q保持，B另有anchor处实际执行器target保持，不使用未来tick观测、不额外模型前向；供round08选中模型与来源逐阶段比较。26项CPU检查通过，实际模型比较尚待。小证据q_next_identity.json绑定完整本机报告及其SHA，原15条06/07阶段统计与99池总量分开；不以离线误差代替抓持/转正/插入行为。

## 2026-11-14 22:06 资源归属统计修正

state_resources只为明确batch_run run的唯一--output及batch_eval run的唯一--root/--output增加窄识别，解析真实路径且后者必须属于前者，相似模块/prepare/重复参数/越界路径拒绝；这是遥测分类，不是发信号授权。53项CPU检查通过，当前冻结monitor不热改。主Agent一次回归命令误写不存在test_state_resources.py而未执行，改为实际新测试后27项通过。

原session05资源审计：10461 shared样本、时间加权GPU34.15947%；含最后monitor样本34.15678%。最少free VRAM9461MiB、RAM18.404GiB；2条读数空值覆盖0.17603秒，最大采样间隔5.09828秒，声明开始至首shared还有16.95秒未覆盖。不能称全程零缺口。1064条external标记中1060条有项目进程绑定、1条仅同PID退出过渡推断、3条未解决；原记录不改、修正解释单列。full_gpu_v1 pacing全0，误标未恢复旧50%限制。详见session05_resources.json。

## 2026-11-14 22:00 新16初态正式批量采集与相机候选

session06固定窗口21:49:51至次日03:49:51，来源A05/B06及99条未全部训练池绑定，旧session05截止/现场不改。冻结d168195的new_family_08已由tight-fresh-newfamily08-20261114.service运行32机会、16新train family，配置与公开预注册逐项相等；CPU standard16/视觉micro8/512MiB异步writer/2个spawn校验进程、最多4待校验。初始化65.33秒，A波已真实执行，当前只是启动验收而非最终新增量。2个A初态碰撞非法、0动作，保留在机会分母；其余结果待完整结束和入池审核。旧monitor尚未识别batch_run的--output，将其PID误标external；owner独立PID/starttick/PGID证据正常，原遥测保留，不据误分类宣称外部争用。见session06_preregistration/startup.json。

显式backend v3候选只在CUDA+tiled启用两个固定相机实例的Fabric→USD位姿同步；三份安装源码SHA、类型/设备/原值和readback严格核对，腕相机与旧v1/v2不变，不修改共享Lab。单tick失败诊断可选1–4次render-only，先保留坏包，逐次核对可见物理/时钟/targets/queue/packet/frame/RNG；变化立即停止，原异常仍抛、画面不进控制/训练。安装源码支持此定位，真实Kit修复和速度仍待，当前正式继续standard。

新增tight_fresh_stage_fit的CPU prepare/独立source与selected EMA前向/compare：只投影本批新增train，按当前权重分别重算完整历史，按抓取前/持物/相对转动/近孔/原生阶段报告前端commit12误差。动作可见次数与实际训练target槽分开，后者无法逐阶段重建时记unknown；不测后端、不冒充自主收益。57项主Agent相关CPU回归通过，独立审查另71项（有重叠）；两候选均尚未GPU验收。

## 2026-11-14 21:49 session05 完整收尾与新初态阶段准备

原session05所有GPU worker和monitor已退出，原21:54:21截止未改。该时段实采第六/七批共32机会、16次接管、15条成功纠偏、3210合法专家步；累计99训练轨迹，仍30训练family，6验证不变。第六轮实际训练93条并完成完整自主，A/B各0/4插入；第七批新6只完成缓存尚未训练。B06两例持物转向/靠近改善而未到原15mm标准，下一采集保留A05/B06并保护原A/B05回退。

I/O独立恢复已完成缺失paired，各组同样102更新/2373anchor/54368目标槽。plain/paired共同训练区间66.40/54.85秒，合计进程read_bytes77.77/46.58GB；但完整arm181.38/196.03秒，且plain有134.76秒metadata扫描竞争、两者缓存温度不同。决定暂不采用paired，同步不能据此证明净收益。下一正式cycle用普通双前端与已实现有界预取；新fresh_stage_sample_wait_v1最多等30秒新遥测，原22GiB/24GiB、原deadline与STOP不变，避免上一阶段退出前采样误判；92项联合CPU检查通过（含混合采集审计，集合不与既报相加）。

已实现混合fixed40/fixed80/contact0/near60审计和预注册：16新train seeds28400000..15、8独立dev28500000..07配置及seed不交，原6validation保持；纯metadata不是已采数据。固定40步接管早于完整48步高度参考，其学生持物窗口记不可用，不利用专家后帧反证学生能力。新审计分别报告尝试/合法初态/条件满足/接管/救回/入池/新family与重叠阶段动作量，旧32机会的既有计数保持。采集16lane、开发8lane按各自样本数避免空闲渲染；无通用数量上限。新session06由已退出来源准备独立6小时窗口，不续期旧run。

Tiled CPU/GPU新单步诊断均0动作失败，raw/cache逐值一致。CPU校准与standard相同仍黑；GPU固定相机未写到目标位姿，与安装Tiled遗漏Fabric→USD同步吻合。证据和局限在backend_probe_review；正式使用standard，候选修复尚未运行验收。

## 2026-11-14 21:42 清理完成与缺失 I/O arm 的显式恢复

135个清单检查点已删除，原allocated blocks合计64,806,690,816字节（60.356GiB），文件系统available实际增加64,704,028,672字节，差值含同期记录写入。124保护模型stat前后不变；完整逐文件intent/删除/保护复核账本留本机，简表见evidence/tight_fresh_20261114/checkpoint_cleanup.json。旧中间optimizer/RNG恢复点明确退休，原始轨迹、缓存、选中模型未删。

I/O旧study已失败退出且原截止不改；ordinary双组各102更新完整，paired从未启动。新增tight_fresh_paired_io_recovery仅引用原plain并在新identity补缺失paired一轮，固定原b59baf3 trainer/输入/deck，独立最多360秒含30秒联合准入等待与20秒回收，受session05原21:54:21限制。只接受已确诊的16389MiB旧采样失败，等待晚于plain退出的新采样，保留22GiB/24GiB门槛；单次claim拒绝重复。32项CPU检查及原实验只读核验通过，尚未实际运行同步arm；134.76秒metadata扫描和暖缓存混杂保留，不能称公平因果提速。

## 2026-11-14 21:32 已完成后端短测与旧中间检查点清理入口

五个短probe逐lane审计完成：CPU16/M8为21.76 aggregate tick/s，CPU32为19.76且初始化155.63秒；下一正式新family批次选CPU16/M8。GPU16进入180tick接触段：15/16合法，2700步，15均有双指接触、5曾持续接触抬升、1旧拿稳；native lift/near/alignment/subgoal均0。不同长度与合法lane数不能构成公平速度比；GPU完整救回/入池尚待。结果见evidence/tight_fresh_20261114/backend_probe_review。

pool07_untrained的A/B冻结缓存均已完整105条（99train+6validation），新6纠偏仍未训练。paired_io_compare01普通组各102更新已完成，同步组在启动前因末次遥测仍为16389MiB可用而未过22528MiB联合准入，0同步更新；完整失败现场保留。只读清理审计在21:26:37–21:29:02.76与普通组重叠134.76秒，加上plain→paired顺序暖缓存，后续即使补齐也不能视为公平单变量速度比。未采用同步候选。

新增tight_checkpoint_cleanup默认只读，135精确候选SHA白名单、124保护模型stat、完成凭据、跨工作树/输出/PT metadata引用和FD/mmap消费者复核，删除使用逐文件持久intent/receipt与固定ledger，无递归删除。实际只读通过、24项CPU检查通过；3个明确身份的常驻非研究daemon不可读，记录可见性例外，其他未知不可读仍拒绝。候选64,806,690,816 allocated bytes，全部为已完成旧比较的中间/last检查点；删除会放弃其优化器/RNG中间恢复点。所有best、第五/六轮、原始数据、冻结视觉缓存受保护。当前尚未执行删除，实际释放量以执行ledger为准。

## 2026-11-14 21:26 有界训练 I/O 对照与初始化故障诊断

新增默认关闭的 paired_episode_io_v1：A/B 在相同完整轨迹访问的开始/结束会合，保留独立参数、完整历史重算、Adam/EMA/RNG 和更新预算；共享冻结缓存 inode 与 deck 受绑定。此为读取时序候选，不是跨轨迹梯度合批。工具 tight_fresh_paired_io_study 从原 round06 的93/6数据和第五轮起点分别运行普通/同步双前端，各组一完整轮102更新；两组均关闭预取，总窗口至多12分钟且不延长原session05。真实收益尚未运行验证，plain→paired 的缓存温度与顺序混杂必须披露。详见 TRAINING_PAIRED_IO.md。CPU 32项通过，包括两个 spawned 训练进程的参数/优化器/EMA/RNG 等价及真实 owner 停止回收；不等于 CUDA 或自主效果验收。

有限 tick probe 新增初始化阶段和120秒重复 Python 栈，仅诊断路径启用；正常/异常返回取消计时器。相机失败后保存原包，再读取现有 raw RGBA/cached buffer 和各 lane CPU tile 比较，披露异步读取时刻；不新增 render、不替换坏画面、不放松有效性。52项相关CPU检查通过。64环境第一次受进程FD限制，第二次只在该作业增加LimitNOFILE65536后越过FD错误，但renderer PerViewState分配失败并在原600秒启动边界退出，0动作；不是全局系统修改或可用64环境证据。128以上暂未运行，未设永久上限。

第七批99条池的冻结视觉缓存已从新冻结9e8c408开始补齐，只有新6条需要计算；尚无新训练或晋升。原session05 21:54:21截止不变。

## 2026-11-14 21:03 第七批真实批量结果、近孔接管与有界并行校验

第七批冻结2be0533已自然完成16/16机会与全部文件校验，owner exit0。7次触发、6条专家恢复成功并入池、1360合法专家控制步，覆盖5个已有训练family；另1次晚触发后原生超时、27合法专家步保留未入池。完整owner1059.375秒，相比串行第六批尝试/h提高15.0%、控制步/s提高30.6%，但有效轨迹/h下降23.3%、有效专家步/h下降15.5%，尚不能称有效采集提速。新增六条都从持物但方向不对的状态接起，未覆盖学生自己近孔后的精细偏差。详见[evidence/tight_fresh_20261114/collection07_review/README.md](evidence/tight_fresh_20261114/collection07_review/README.md)。累计未训练池pool07_untrained为99条、仍30训练family，原6验证及原30统计保持；新6尚未参加优化器训练。

已实现下一版本：`held_near_goal_then_delay_v1`在过去12步持续接触抬升且当前同目标最大关键点距离<=60mm时，从下一实际动作接管；目标切换拒绝，无fallback。60mm仅为采集入口，原15mm任务成功不改，真值不进入学生。串行、正式批量、HDF5回读共享因果适配器。B06开发轨迹CPU重放能在222/289步触发，另两例不触发；这些开发初态不转作训练，真实near60采集尚未执行。

`async_recording`新增默认关闭、显式2进程/4任务的spawn校验池。唯一HDF5写入线程关闭文件后传路径，独立进程执行原全部字段校验和完整SHA，成功才原子发布；保留排队/启动PID、等待、失败及结束证据。正式batch/expansion参数已贯通，并单列wave.admission_s。`tight_batch_owner`额外绑定PID启动时刻/PGID/SID/UID，清理本owner组内已退出leader遗留的验证子进程，拒绝PID重用或外来组。无热改旧worker，尚无正式2/4提速结果。150项联合CPU检查及后续23项spawn/进程组检查通过，主Agent另核验72项相关回归；测试集合有重叠，不相加。

后端探测继续独立记录：16环境标准相机576步短测21.76步/s；32环境1152步19.76步/s，启动155.63秒，增加并行没有继续提速。GPU物理8环境7合法lane共252步20.91步/s，1初始碰撞无动作，尚未完成真实接触/插入验收。Tiled失败实际帧已保存：所有lane第一RGB视角正常，另外两视角全黑、深度有效，0控制动作；不放松相机判据、不采用该后端。64环境CPU候选正在有界探测。后续依据真实数据与资源选择配置，旧session05截止21:54:21不变。

## 2026-11-14 Tiled真实失败现场保存

新增camera_failure_diagnostics及probe失败分支：原相机/时钟有效性检查异常后才保存实际RGB-D/valid/frame/time/calibration packet与逐lane/view数值，另读回真实camera prim、image/calibration RP布局、annotator形状及descriptor/tile配置；有界256MiB且超限拒绝、不截断。正常路径无额外同步/写盘；保存成功或失败均传播原异常，未放宽阈值。20项相关CPU检查通过，新一次有界8lane单步诊断仍待启动；旧首帧缺失不能倒推根因。

## 2026-11-14 20:47 第六轮完整自主审查与下一采集来源

第六轮完整训练→依赖刷新→后端→8自主已经自然结束，自有owner/Kit退出。A/B原生抬升各1/4、旧拿稳各2/4、允许转动的持续接触抬升各2/4，near/alignment/子目标/插入全部0/4。原生抬升与拿稳/接触抬升是不同分母；详见evidence/tight_fresh_20261114/round06_review。

B的seed01/02在真实持续持物时转向并接近首目标：同tick关键点最小17.439/16.045mm，root位置2.020/4.601mm、角差0.1831/0.2192rad；原阈值是关键点15mm，因此均没有对孔。对应持续接触抬升段74/61tick。来源B仅1持物例且最好关键点84.604mm；新B另两例仍未有效拿起，其中seed03较来源退化。A虽一例持物更久，但方向仍错且456tick跌落，另一例持物退化。决定下批采集用A05/B06，原A/B05完整回退保留；这是重复开发四例上的研究选择，不是新初态正式胜出。

已实现独立followup v3的逐角色策略来源和共享数据池：真实DP/后端/nominal依赖保留原训练来源，另核验严格追加训练池与原30统计/固定验证不变；允许只构造+缓存而未训练的数据池。源owner全部退出后才能登记新时段，不延长旧deadline。本批batch07仍冻2be0533，用旧A/B05、8环境/M8/CPU标准相机/Kit16/异步512MiB进行真实采集，不能中途替换。原8family的A波已4261真实tick/504.05秒，3专家插入；实际准入和B波尚待。

A波writer verify共188.27秒、compact追加187.45秒，主提交背压51.91秒、尾部等待132.09秒，存在重叠不可相加。证明单writer仍受逐lane完整校验阻塞；下一实现将显式使用有界独立验证进程，不跳过任一源字段或成功准入。当前尚无整批加速结论。Tiled容量v2两次readback匹配且跨过descriptor错误，但首次真实取图camera_valid失败，0动作；原现场保留，待记录具体lane/view与RGB/depth分量定位。

## 2026-11-14 正式批量采集独立资源时段入口

`tight_fresh_batch_run prepare --session-root`现在将已完成的旧来源pair与当前独立资源时段显式分开，沿用stage_takeover原有冻结来源/当前session机制。run同时核验已登记session副本和真实resource_root字节，禁止命令行在运行时改时段；旧默认保持。新增13项CPU检查覆盖真实prepare、过期来源/有效时段及漂移拒绝。下一批登记第五轮pair、原8训练family×A/B、contact-height0/12，8lane/视觉micro8、CPU标准相机/Kit16/PhysX2/512MiB writer；当前只准备接口，不等于已采集或入池。

## 2026-11-14 20:33 选中B新后段拟合与批量渲染资源候选

第六轮双DP、99条依赖刷新和双后端均已完成，8次完整自主正在原冻结3e648db运行，不能提前晋升。只读重新执行B实际选中epoch3 EMA对新9条训练后段的原冻结指标：arm/hand RMSE 0.00438868/0.0368985rad，来源起点0.00716645/0.0658879；选中时每条6次访问，最终epoch10的20次访问及更低误差不能赋给选中模型。完整真实历史重算、180源码及数据/缓存SHA核验，20.61秒、零optimizer/validation/test；见round06_selected_B_train_fit.json。

ff555ce的8lane视觉micro8对原串行micro2真实观测门槛通过：A/B最大目标差8.64267e-7/8.27014e-7rad，均小于原2e-5，每组200lane观察完整进入历史。不是新物理数据或速度结果。

Tiled8在15.80秒环境构造后首次reset取图发生renderer descriptor/ParameterBlock分配失败；standard16也在控制前发生同类错误。均0动作，已保留日志和operator_stop并只清理自有worker，不能计为任务失败。不是显存不足的证据。新增显式backend v2登记/两次严格读回descriptorSets360000、reservedDescriptors900000（取安装相机测试配置），画质/标定/物理不变；尚未真实测试，不宣称修复。异步capture wave新增实际payload/queue峰值收据；测试覆盖真实HDF5完整验证后的统计。

CPU新后端/readback/fit/writer检查通过；合并检查发现旧cycle失败fixture没有新增worker身份所需pid，已修正fixture而未放宽运行审计。原session05截止21:54:21不变。下一步验证容量候选，并把已通过的8环境标准路径用于真实完整接管采集。

## 2026-11-14 八环境真实观测通过与视觉批次候选

冻结7c27553的A/B各8lane×25真实训练观测完成完整DP＋后端逐lane/批量检查，lane0独立reset；最大目标差A5.55068e-7、B8.34465e-7rad，小于预登记2e-5，全部400个lane观察进入持续历史。仅teacher-forced记录输入门槛，不是物理自主或速度结果；证据batch8_observation_gate.json。

首CPU/standard、8env、Kit4/physics2短探针在300秒初始化上限失败，owner另10秒正常结束等待后SIGKILL，0动作/0完成机会。原目录及日志保留，未计任务失败。四tasking线程持续忙且GPU占用低，native stack因ptrace权限不可得，根因未定；只更改Kit16的独立对照已登记，仍原session截止。新增tight_batch_probe_owner保存源码/预算/退出/partial角色，不自动重试；2项完成合同测试通过。

Kit16对照现已完整结束：初始化17.92秒、288真实控制tick、loop含收尾27.30秒、含启动48.55秒，全部8lane传感时钟/完整DP与后端计数通过，随后明确诊断截断；无自主成功率/新数据。B依赖缓存并行运行，合计10.55tick/s不能作为无争用峰值或正式周期提升。原4线程启动失败保留，未为此改物理/任务。证据scaling_cpu_standard_n8.json。

发现native视觉按每2lane分块，新增显式current_observation_lane_microbatch_v1及--vision-microbatch传到门槛/探针/正式采集和cycle评估；普通单观察及冻结训练特征路径仍2，填充只算计算量不算新观测。39项CPU检查及14项入口复检通过，CUDA数值/速度仍待实际验收。修正GPU物理观测与CPU专家embedding拼接的设备边界，搬运计入expert_s；14项真实控制路径/FakeTensor设备检查通过，尚未声称GPU物理验收。

## 2026-11-14 可扩展完整学生采集与评估实现，待真实后端验收

新增tools/tight_fresh_batch_run.py、tight_batch_owner.py与visuotactile/batched_rollout.py：将仅开发用四lane批量路径扩成显式正整数并发的正式采集/自主整波执行；新cycle --evaluation-lanes保持原串行默认，更新DP及其全部依赖/后端后进入同一完整学生批量流程。case执行、文件校验与准入分开持久化，启动/绝对截止受ordinary owner监管，软件partial不覆盖、不冒充自然任务失败或完整结果。

新增batched_position_bridge.py只包装原ExpertPositionBridge，每lane真实前缀/接管/queue来源/flush/segment独立，不增原生随机draw。新增async_recording.py共享有界字节与唯一HDF5线程，独立快照、背压和失败传播；finish_async让其他lane继续，全部原字段SHA、相机与控制审核后才入训练。增加真实bridge/H5整波CPU集成和故障测试。batched_controller新增显式host scheduler参数；batch_eval/batch_gate/rollouts移除4/32机会硬上限，仍要求有限显式正数和实际资源验收。

新增execution_backend/camera_calibration模块；tight_state/native_evaluation/sim.environment/environment_rng接显式CPU或CUDA、standard或tiled和线程配置。旧缺省字节/CPU RNG路径保留；注册GPU只隔离对应device，不全局重播种。tiled按实际camera prim逐lane绑定当前renderer calibration，禁止stale USD fallback。允许后端改变不等于任务/资产/频率/成功标准改变。

当前验证：12项bridge/批量集成、22项writer/源字段、29项后端/RNG通过；后端另21项复检通过。CPU证明时钟/记录实现，不证明物理成功或实际加速；新真实GPU/Kit证据后补。活动round06仍冻结3e648db；公共仅代码/文档/精简人口，raw权重轨迹不上传。

新增episode_io.py/tight_episode_io_probe.py与FROZEN_EPISODE_IO：显式每trainer最多1GiB冻结CPU特征预取，维持10GiB可用RAM，完整真实历史每次按当前权重重算。tight_fresh_train/visual_aux_training接相同张量路径，cycle --prefetch-mib登记后才可启用，逐update/epoch记录等待与磁盘计数；缺省不变。52项相关CPU回归及新增cycle不可热替换合同通过；本次cycle/episode_io/batched_controller合并20项通过，尚无真实采用或加速结论。

新增training_expansion/tight_fresh_expansion：事前登记新train family及float32初态配置hash，排除已有训练与原验证/测试初态；原6验证和test22只读JSON分区、不读测试轨迹。支持明确fixed40/80与contact0/12混合真实学生前缀，registry贯穿采集元数据/合法专家mask/aggregate；未知family默认仍拒绝。68项隔离/实际HDF聚合/历史回归通过，24项正式runner/registry集成通过，真实元数据8候选干跑通过；还没有正式采集新family或新增完整专家模式。

batch_eval新增显式--max-control-ticks/--roles/--backend-json短探针，保留原自然全程缺省；截断机会只报censored，success=null且不能进入自主成功率或训练。每tick检查真实三视角时钟/帧更新、完整历史和控制次数，保存共享资源样本与含收尾wall。31项CPU检查通过；探针吞吐不能代替正式周期结果。

## 2026-11-14 第六批完整结案与新完整学生训练登记

16/16真实机会自然结束，冻结93eff18，owner/Kit正常退出：A救回5/8、B救回4/8，共9条成功专家后段，6训练family，1850合法专家步、3038真实历史tick。A/B共享新数据，累计93轨迹=原30+历史54+新9，仍原30训练family，固定6验证/test22不变。9条中8条在新接管前没有出现旧拿稳窗；A046同时满足旧窗。新入口确实补到了旧规则在这些前缀上漏掉的可救回状态，但不是旧/新触发的严格物理配对实验。

实际接手状态须分开：6条过去12tick持续双指且抬升；A031接手当刻双指但过去窗口有间断；007的A/B延后12tick接手时已经不满足双指条件，仍在抬升高度。专家全部从真实状态继续并插入，不能倒推它们接手时均拿稳。接手角差1.723–3.126rad，原生关键点237.7–386.5mm；物体/目标root距离106–194mm，故主要新增抬起后的转正与接触偏差，尚未补到贴孔的学生状态。合法专家后段实际包含188个native near-goal tick和18个alignment tick；这些阶段数不是独立轨迹，也不是学生自主进展。

7次未触发均601tick timeout：6次沉降后最大抬升不足2cm，A047虽短暂抬升33.48mm，但双指接触时最大只10.95mm，未出现12tick持续抬升/接触。它们属于未拿起或未持续持物，不能笼统叫插入失败。全部16仍在分母，失败现场保存；两个0动作启动超时另列软件尝试，未增加或替换机会。完整证据takeover06、takeover06_stage_audit、takeover06_pose_audit、takeover06_onset。

新cycle冻结3e648db，原session05截止21:54:21不变，service tight-fresh-round06-20261114。选择DP latest repeat2：旧84每遍一次，新9每遍两次，10遍/每组1020更新，其中新后段180次episode更新；动作合法步复用份额13.40%，不等于梯度份额。后端继承原完整pair、保持均匀15遍，预计每组1395episode访问、386475合法步，新9占7.18%合法步。两组先更新DP，刷新99条含固定6验证的计划与可训练视觉特征，再更新后端，最后各4次完整自主（仍开发seed28140000–3）。没有单独repeat1同预算控制，不能单独归因重采样收益。

启动已核验aggregate完成、图像缓存准备中；当前尚无新模型自主结论。仅冻结DINO缓存可复用，新DP依赖必须刷新。具体状态与任何失败由round_06_stages/status、各worker/阶段收据记录，root handoff_pointer提供入口；不是启动即完成。用户授权整卡，保留4GiB显存/10GiB内存与磁盘余量，双DP仅在联合启动余量准入后并行。

## 2026-11-14 新合法后段的可审计前端重复采样

已审阅接续时遗留的latest_takeover_sampling草稿并完成接入。显式latest_takeover_repeat_v1只允许1/2/4：每遍原示范与历史纠偏各一次，本次aggregate.new_training_ids按指定次数完整访问；新增必须是训练集中的真实专家接管，禁止原示范冒充、跨split、重复身份及学生标签。每次用当前权重重算全部真实学生/专家历史，动作和几何仍仅使用合法专家mask。缺省及repeat1的episode顺序与RNG字节均保持原路径；重复是旧轨迹复用，不是新样本。

cycle的--latest-repeat只登记前端，先验证A/B相同人口再写双方continuation；任何training目录出现后禁止修改。后端仍每epoch每episode一次，其当前计划/可训练视觉缓存必须在DP更新后全部刷新；明确不宣称后端也增加了重复倍数。sampling_exposure在选中和末尾checkpoint保存并验证完整遍数；恢复检查expanded deck游标，崩溃后未持久日志另存，不重复计入更新。

61项采样/恢复/并行检查及16项geometry/训练诊断通过（一次测试命令误指不存在文件，未执行测试；随后按真实文件完成）。这是采样实现验收，尚无新学生自主收益；实际重复倍数与预算在完整采集审查后的新cycle登记中确定，旧视觉对照和第五轮权重保持原样。

## 2026-11-14 第六批已真实接管，后续双DP并行入口准备

冻结93eff18的takeover_06_tasking4_01实际初始化72.213秒，Kit回读tasking4/PhysX2；同16机会、来源pair和session05原21:54:21截止保持。前两次0动作启动超时现场保留。第三次能启动不证明调度线程数是根因，不改变任何物理设置或宣称精确物理复现。截至本次代码登记已完成10/16：A4/B4条专家救回，分布5个训练family，1606合法专家步；另A050/B031未达到2cm抬升及持续双指条件，仍属拿起阶段失败。余6次正在执行，以上是部分现场，不是最终16次结论或学生自主成功。

接管姿态审计改为按冻结trigger版本调用make_trigger，同时保留原stable窗口分量；新contact-height触发端点与真正接手边界分别记录。阶段报告修复旧硬编码stable文件/分类，按已登记合同读取contact_height_stage_audit，独立重算学生窗口、真实控制authority、整队列flush、q[t+1]及实际专家目标。新入口没有给学生提供物体真值。真实完成案例复读通过，22项时间/authority/新旧trigger检查通过。

下一轮cycle增加显式--parallel-frontends：只在full_gpu_v1下、冻结图像缓存准备完成后启动独立A/B进程；各自数据/随机流/epoch预算不变，两个完成后才刷新各自DP计划/可训练视觉缓存并更新后端。联合启动要求遥测<=8s、空闲VRAM>=22GiB和可用RAM>=24GiB，否则记录不准入并使用原串行顺序；各worker原4GiB/10GiB余量检查继续。每stage单独保存真实PID、proc启动tick、命令、截止、结束码和完成SHA；一方普通失败保留另一方工作，任一失败禁止后端开始。实际本轮并行速度/峰值尚未测得，不外推历史探针收益。

本次只改审计和stage调度，活动采集树不热改。原latest_takeover_sampling及训练草稿保持未提交、未使用；新增数据最终数量与状态覆盖审查后再决定采样。仍须完成当前16机会及新的完整学生训练/自主审查。

## 2026-11-14 有界Kit启动调度诊断准备

两次contact06启动均卡在仿真初始化之前，未开始任何case；独立owner保持180秒启动界限，SIGTERM无响应再回收自有进程组。资源遥测无等待原因，native main futex等待/全进程约200%CPU；gdb取栈被系统ptrace限制拒绝。去掉本次额外PXR/OPENBLAS环境设置未解决，根因尚未确认。

tools/tight_state增加显式TIGHT_KIT_TASKING_THREADS=2/4，缺省与原Kit参数字符串逐字相同；仅carb.tasking调度线程可选4，PhysX配置仍2，原任务/资产/120Hz物理/60Hz控制/5Hz规划/模型不变。请求参数与Kit实际配置readback分别写eval_startup/kit_started，后者不等于完成物理初始化。7项参数/拒绝注入测试及18项collection_owner检查通过。拟在新冻结源码/新目录、同16机会/pair/deadline下仅做一次4线程启动诊断；成功不证明饥饿是根因，也不声称隐藏轨迹精确配对。失败则保留现场做更具体调查，不遍历线程数或重置截止。

## 2026-11-14 账号接续：视觉辅助对照完整结案，保留第五轮模型

机器实查session04已在17:58完成全部四cell训练/缓存/后端与16自主，账号中断没有中断它；没有重训或重跑。执行树fresh_visual_aux_20261114冻结6780cea，开发树continuous_vision_20261114/本地feature/tight-continuous-vision-20261114，公开目标仍origin/feature/tight-state-foundation；narrow_followup与原项目目录内交接较旧。开始时仅余monitor3670509，现已通过STOP_MONITOR正常退出，原实验现场保留。

完整report逐项核对冻结源码、保护数据/权重、四前后端实际预算与16逐次执行；每tick真实视觉/历史/后端次数均等于控制步，5Hz规划次数正确，零专家控制。16均合法，插入/原生对孔/子目标全部0；15个601步timeout，auxiliary A/28140002为411步fall。旧拿稳窗口source A/B为2/4、1/4；control为1/4、3/4；auxiliary为1/4、1/4。允许掌中转动的持续接触抬升窗source为2/4、1/4；control2/4、3/4；auxiliary2/4、1/4。新四cell原生lift均0，但这不等于没有物体抬升；旧窗口false也不等于掉落。

视觉辅助读出位置/朝向改善，DP动作验证无收益；后端control/auxiliary手指验证RMSE A0.0132750/0.0132739、B0.0148413/0.0148390rad，近乎相同。control B部分持物/转向更好：seed00在持续接触中角差0.2921rad、同tick关键点38.63mm；seed03同条件最小38.44mm，仍大于15mm容差。auxiliary B只1例有持续窗口，其同条件最小关键点117.12mm；不能以位姿读出改善宣布策略改善。每cell仅4个反复使用的开发机会，不是新终测，source为另一次较早执行，不声称隐藏状态精确配对。结论仅限本对照：不采用辅助或额外训练控制，保留第五轮完整A/B作为下一采集来源。

完整报告visual_aux_complete.json及两张曲线、简明接触运动复读visual_aux_closeout.json均在evidence/tight_fresh_20261114。报告运行于独立6780cea工作树，仅复制当前report脚本；所有依赖保持冻结字节，没有放松hash检查。新增tools/tight_fresh_visual_closeout以同一真实tick/当前原生目标报告持续持物时的角度和关键点；2项合成轨迹/换目标/数据漂移测试通过，实际24轨迹重读通过。未修改原生成功判定。

session04最终整卡均值31.499%、最大完整60s83.430%、missing0、最大相邻间隙1.056s；整卡授权下没有旧50%限制。该均值包括完成后约55分钟仅monitor运行，不能当有效训练平均；峰值显存8638MiB、最低可用RAM24.644GiB。权重/原轨迹仍留研究机。

## 2026-11-14 接触中转动纠偏：第六批登记与一次动作前恢复

独立session05固定18:54:21–21:54:21 Asia/Shanghai，冻结f4b3644；来源为round_05的原选中完整pair，84训练轨迹/原30统计。8训练family×A/B16机会，050/007/031延迟12，051/053/047/039/046立即；无固定时刻fallback。真实新数据及训练量必须等待结果，不把历史6候选当救回。

74项阶段/owner/触发/取证/session测试通过。真实CUDA A/B各25连续RGB-D帧、3个DP规划的首计划取证开关严格逐字节相等，零参数更新；forensics_gate/result.json保存结果。首次takeover_06在仿真初始化180秒超时，owner按已登记界限终止worker；0机会开始、0学生动作，旧现场保留。takeover_06_startup_recovery_01复用逐字节相同plan/session、同16机会与原截止；仅去除本次launcher额外添加的PXR_WORK_THREAD_LIMIT/OPENBLAS_NUM_THREADS环境项，恢复既有owner OMP/MKL=2，原Kit/物理线程设置不变。根因尚未确认，不称已修复。

当前实际进度看session05/takeover_06_startup_recovery_01/{owner_started,owner_worker_started,initialization,results,owner_finished}.json及worker.log；独立service tight-fresh-contact06-recovery-20261114，资源monitor另有时限。完整结果审查后再决定新增后段采样与完整训练。开发树原遗留latest_takeover_sampling草稿及train/recovery-log修改已识别并保留，未混入本次冻结采集代码或已有权重。

## 2026-11-14 下一批接触中转动纠偏的实际覆盖依据

CPU复读五轮全部84次采集机会，只取真实学生前缀，14条在9个family出现接触并抬升候选；这些均来自旧模型，当前第五轮完整pair的train-family前缀为0，不能预测其接管率。候选延迟12步仅4/14仍满足整个接触抬升窗，9/14当下仍双指接触；延迟24步只有13个可观察状态、其中2个仍双指，另1个被原专家接手截断。全部候选发生在首次原生near-goal之前。

待视觉对照完整结果审查后的采集预案为8不同训练family×A/B共16机会：050/007/031延迟12，051/053/047/039/046立即接管，无029、无固定时刻fallback。031仍只有原示范、无准入纠偏，其旧A在132至144步保持接触并转动，而原220步救援失败；047旧B在149步可接而12步后失双指，原360步救援失败。此选择旨在验证较早的实际偏差状态能否救回，不将历史轨迹当新模型必然路径。

新增tools/tight_fresh_next_collection_coverage及8项分界/authority/延迟/目标切换/失败分母测试。证据next_contact_height_collection_coverage.json绑定四个来源权重、data、实际family学习账本和84条numeric/result SHA，未读取validation/test轨迹、未启动GPU/Kit。延迟24只诊断，不扩展执行合同；正式来源及机会待主Agent审查当前完整对照后登记。

## 2026-11-14 首计划取证的真实输入一致性验收入口

新增tools/tight_fresh_forensics_gate，完整A/B各取原train30排序首条真实三视角RGB-D连续25tick，含0/12/24三次规划；并列开/关首计划取证，严格比较最终绝对目标、DP计划、完整前后端历史、实际输入及全局/独立Torch RNG。任何字节/形状/类型差异拒绝准入并保留首分歧，不放宽容差。两路都确认源轨迹真实目标作为teacher-forced历史，不把它称学生执行或物理重放。

9项CPU取证与差异注入测试通过；真实原示范HDF5的25步时钟、三视角及target_after到下一actuator_target逐位对齐已核验。GPU验收尚未执行，当前视觉对照仍冻结6780cea；该入口不启动Kit、不做参数更新，不能代替新采集真实接口验收或自主行为结果。对应执行前后证据需绑定实际源码/完整模型/数据与资源会话。

## 2026-11-14 新接管批次完成检查修复与前端监督人口复算

接触并抬升的采集入口按新合同写contact_height_stage_audit.json，管理脚本此前仍硬编码stable_stage_audit.json，会在实际成功结束后误报owner_error。现按登记合同读取对应审计，错误旧类型不能替代；原stable路径保留，32项管理/阶段测试通过。未改活动冻结6780cea，不修改采集或任务成功判据。

视觉辅助报告进一步从原始authority、合法mask、segment与recovery_onset_commit12_v1重算每个规划窗口及监督槽，四前端实际学习增量均一致；33项报告相关检查通过。公共visual_aux_frontends.json更新为v2，原v1字节及替换收据保留session04/cpu_audit_drafts。新一条第五轮纠偏仍仅占本次合法动作目标0.91%，不把对旧人口的追加训练算作新增后段覆盖。后端和完整自主仍待完整实物结果后判断。

## 2026-11-14 四组视觉前端完成与新的纠偏采集准备

冻结6780cea的control/auxiliary × A/B均完成5遍、420更新、10235规划anchor、235200合法动作目标槽。每cell原30/纠偏54的episode更新150/270，逐帧视觉164945tick，其中真实学生prefix44830、队列过渡540、合法专家67585、原示范51990；前两类没有动作监督。相同角色两路线逐episode次序、随机状态、预算和选模人口已核验。A两组均选第5遍，固定验证动作指标control0.007755423、auxiliary0.007755020；B两组均选第3遍，0.008168663/0.008171822。相较第五轮来源，两路线A约差0.5%、B约差1.4%，没有动作验证收益。

选中权重的验证位姿读出：A位置109.506→68.173mm、角98.691→95.616度；B位置133.978→95.878mm、角112.476→106.953度。它说明辅助目标被学习，不证明拿起/转正改善，也不证明来源缺少位姿信息。evidence/tight_fresh_20261114/visual_aux_frontends.json/png为明确INCOMPLETE的前端审计与曲线；四cell后端及16次自主仍在执行，未晋升模型。第一control A新DP缓存90/90、32989训练历史tick与23915合法tick已逐分片复核，后端完成5遍833更新，固定验证选第2遍；来源pair仍保留。

本视觉对照没有新增纠偏：第五轮唯一新轨迹在每cell最后5遍中仅参与5次episode更新、2560观察tick、95anchor、2130合法目标槽，分别占420更新1.19%和235200槽0.91%；B选中第3遍仅经历3次该轨迹。历史53条纠偏265更新/130890目标槽，原30示范150更新/102180槽，不能把本轮累计训练次数当作后半段新数据充足。下一步优先验证新的真实偏差状态接管覆盖。

新增tools/tight_fresh_expert_stability_coverage仅CPU重读原30与54成功纠偏，隔离学生prefix/队列/合法专家，并要求每个窗口12tick完全位于同一authority/segment。84条before/after连续性逐字节相等。原30对孔附近626个tick中411不满足原拿稳窗；合法专家后段1116个合格near-goal tick中832不满足。第一子目标首次稳定到随后near-goal区间，原示范7/29、纠偏15/49离开过窗口；不能把它们解释为掉落或意外滑动。4条早期接管的高度参考跨专家authority，已单列，不用于学生因果资格声明。报告只覆盖成功训练记录，不能估计全部失败状态恢复率。

新contact_height_then_delay_v1与contact_height_only_takeover_v1采集合同保留原12tick、双物理子步双指接触与抬升要求，允许掌中相对运动，原拿稳诊断及原生成功保持；新旧身份严格隔离，无固定时间fallback。它是准备中的采集候选，尚无真实新接管收益。first_plan_forensics默认关闭，新机会可保存首tick真实RGB-D/视觉/历史/独立采样器与计划指纹；不额外模型前向或随机抽样，小张量保存精确bytes。此前8对相同前端/seed、本体与相机帧龄相等仍首plan不同；原RGB-D/condition/noise缺失，原因仍未定位。新取证CPU检查通过，CUDA/Kit实物验收尚待，首次同步记录开销不能混为普通吞吐。

本次67项触发边界、历史兼容、首计划取证、专家覆盖和前端报告CPU tests通过。更改限开发树，当前worker仍使用冻结6780cea。已生成的早期覆盖草稿移至session04/cpu_audit_drafts保留，公共expert_stability_coverage.json为绑定冻结诊断函数的精简完整版本。原始数据、旧结果、成功标准与活动运行参数不变。

新触发的实际CPU验收另外覆盖第五批全部12机会（包含未接管失败轨迹），候选零/12步延后均从原1/12扩大至6/12：A1/6、B5/6。首次候选完成tick为B051135、B007123、B053124、A050104、B050132、B029218；这六刻均不满足原拿稳窗。独立逐tick计算、在线触发器及原12条回执复算一致，原stable函数/observe/receipt AST与冻结执行版本相等。新数据的可救回率仍未知；验收仅说明真实前缀可因果调度，专家后段未用于候选判断。证据takeover05_contact_height_coverage与takeover05_contact_height_acceptance；另有15项覆盖/分段测试通过，stage准备实际原/新身份各通过同一入口测试。run/report拒绝准备阶段参数，避免把冻结计划误认为可热改。

## 2026-11-14 第五轮完整自主的命令与持物运动分解

新增CPU只读tight_fresh_command_chain，对来源第五轮8条/4664控制tick逐项重建DP→受限名义→后端提案→共同可达域投影→issued/applied→两物理子步执行器目标→实际q。issued/applied/target/两个子步目标逐项相等，未发现第二次延迟或位置裁剪；提案链重建最大差5.96e-8rad。最终共同可达投影仍经常改变手指提案（各轨迹joint-tick约58–69%），这是约束生效的观测，不是限速错误证明。

三个曾满足拿稳窗的病例首次离窗时仍具双指接触与抬升，A02完成tick113为相对掌位移10.57mm超窗，A03 tick212和B03 tick319为相对转动0.259/0.224rad超窗。B03首次拿稳后12控制步，目标角差1.561→1.871rad；假设物体刚性固定于实际运动手掌的纯运动学终点为1.534rad。该比较只说明实际相对掌转动在此段出现，不能证明根因、动作错向或掉落；主动手内操作和中途绕转也可能离开窗口。

所有运动距离/角度显式使用窗口开始的动作前goal；原生关键点各用本transition真实目标。goal字节变化、四元数符号等价和native子目标变化分开记录。8条没有原生子目标；第1tick少量goal符号/roundoff差异不称任务目标切换。无自主专家真值，因此不能用A自身q偏差当专家误差或将后端反向修正称错误专家方向。6项定向测试通过；原8例/48窗口既有数字完全不变。证据round05_command_chain绑定原始数据/模型/源码及语义修订前SHA，零新GPU/训练/仿真。

新增visual_aux_report提供同源/selected/last真实学习量、动作/几何/位姿分项、固定train2与val6及完整自主统计；部分阶段明确INCOMPLETE，只绘已保存的完整epoch，不填零、不以辅助总loss跨组宣称改善。实际control A5遍验证较来源+0.504%，auxiliary B最优第3遍较来源+1.437%，尚无本研究闭环结果。报告检查与命令链合计25项定向CPU测试通过，完整四cell报告仍待实物齐全。

## 2026-11-14 视觉辅助真实验收通过，四组正式训练启动

冻结6780cea已推origin/feature/tight-state-foundation，独立session04为16:46:53–19:46:53 Asia/Shanghai；来源仍第五轮完整A/B，旧02/03时段与产物不变。训练集前6family的16条原示范/真实纠偏、A/B共32项真实梯度校准完成，零更新、未读validation/test；共同权重3.1796844675e-6，实际加权梯度比A0.0625–0.1663、B0.0475–0.1404，不能称每项均10%。

真实CUDA验收14.86秒完成：A/B原路径与detach控制的条件、预测、完整动作/几何目标、梯度和一步AdamW增量满足预先固定容差，候选视觉梯度非零；4次一次性更新未写部署权重，源模型/数据hash保持。此0正式更新只指验收，不是当前正式训练计数。当前四cell owner已真实启动control A前端并产生优化记录；各5遍DP、90条各自新缓存、5遍后端及16机会由已登记runner执行，主Agent继续审查阶段结果。尚无本对照完整学生行为结果。

相关129项CPU tests通过，独立17项cache/后端/单与批量控制兼容性检查通过；实际v7在线行为仍待新模型闭环。机器绑定见evidence/tight_fresh_20261114/visual_aux_study_launch.json，校准/验收全量记录留本机；不得把验收或启动写成拿稳/转正改善。

## 2026-11-14 视觉辅助监督与恢复日志准备

新增visual_pose_aux/visual_aux_training、calibration/acceptance/study入口：同动作前掌坐标物体位置和朝向仅作视觉训练标签，v7明确detach控制/shared-vision候选，在线不执行辅助头。因果readout复用每tick真实三视角路径；源A/B各90episode、35051tick的标签时钟与54条纠偏raw authority/mask/segment已CPU核验。原30统计、合法动作监督、5Hz规划、60Hz控制和任务不变。

一次训练集梯度尺度校准与固定容差CUDA验收均为正式前置：验收仅4次一次性AdamW更新、正式0更新，并绑定共同LR3e-5与完整源码。新owner管理四cell各5遍DP→各90条新计划/可训练视觉缓存→各5遍完整后端→16次交错自主；源第五轮完整后端不reset。旧source deadline保持，用显式session-root承载独立新时段。129项相关CPU测试通过；当前只实现/CPU检查，尚无实际CUDA验收、新训练或行为收益；结果以后续机器收据为准。

fresh trainer恢复时将真实crash-tail日志fsync保留到独立abandoned文件，只将last.pt完整epoch的更新计入恢复模型；缺失/重复durable行拒绝，恢复收据记原日志/保留/丢弃SHA与真实update IDs。USER_RESEARCH_DIRECTION和README去除重复快照，当前目标与单一状态入口明确。原后端对照A候选高度措辞改为约1.81mm（原1.810491mm），数字和结果不改。

## 2026-11-14 后端输出起点对照结案：离线改善，完整自主未改善

冻结cb47381的四cell全部完成15遍后端与16次登记自主，16:30:21 Asia/Shanghai软件完成。四组均4/4合法初态、4个601tick timeout、0拿稳/原生拿起/近孔/子目标/插入成功。每步真实视觉、历史及后端计数等于实际控制步，无额外专家前向。吞吐控制A/B为11.959/12.011tick/s，上一目标起点A/B为12.069/12.067；没有明显速度收益。来源第五轮另列，不混为本轮分母：A2/4、B1/4曾有拿稳窗，双方仍0/4成功。

候选的固定validation手指误差A0.010227、B0.010341rad，低于控制0.013343/0.014950。全部54条已学纠偏首12合法tick误差：来源A/B0.010550/0.009799、控制0.010311/0.009514、候选0.008259/0.008602rad；同刻保持上一目标基线0.019215。候选并非只输出保持目标，但预测手指增量RMS约0.0169rad，小于专家0.0192。真实自主候选A最大沉降后抬升约1.81mm（原值1.810491mm）、B不足0.13mm，主要卡在拿起；控制有最高88.6mm的抬升但没有拿稳。实际角度/关键点与阶段图分开呈现，不将触碰/抬高当拿稳。

结论：不晋升任一新后端，保留第五轮完整pair。这个对照证明给定接续条件下离线拟合更容易，未证明机制自主收益；控制组的head-reset也没保留来源短暂拿稳，不能从此单独归因输出坐标。下一步选择同来源完整模型的视觉辅助监督对照：每tick当前掌坐标物体位置/朝向仅为训练标签，同头detach控制与允许视觉梯度候选，不重置后端头。正式前需真实路径验收与train-only尺度校准；新DP后全部依赖缓存刷新。当前只完成代码与CPU准备，尚无新训练收益。

独立CPU复核16次新评估与8次来源评估，阶段快照及汇总与发布JSON逐项相等；收据见evidence/tight_fresh_20261114/round05_anchor_independent_outcome_audit.json。来源与control的8个观察初态相等，但首个DP计划最大绝对差0.00073–0.00698rad，原因未在此次有界审查中定位，不能称精确输入或动作配对；不改变本轮未拿稳、不晋升结论。

完整证据round05_anchor_study.json/png绑定源码、模型、曝光、缓存、实际16机会及来源8机会。session03 owner/Kit/monitor均已退出，GPU整卡观测均值38.565%、最大完整60s46.266%、missing0、相邻间隙最大约1.15s；整卡授权下无50%上限。session02另补资源结案：均值36.875%、最大60s84.455%，虽然missing_samples=0，monitor切换存在106.162s相邻采样间隙，不能称全程连续覆盖。两旧deadline和原证据保持，后续采用独立新时段。

## 2026-11-14 后端输出起点对照：四组训练完成，自主评估进行中

独立session03固定15:43:37–17:43:37 Asia/Shanghai，冻结cb47381。四cell均已完成84轨迹×15遍、2493更新、358725合法监督tick。控制组A选13遍、B选15遍；实际上一执行目标起点组A/B均选15遍。固定验证手指RMSE：控制A/B为0.01334299/0.01495001rad，候选为0.01022713/0.01034052；来源第五轮0.01332818/0.01488620。两个控制已恢复来源拟合水平，候选离线改善不代表拿稳/转正/对孔改善。16次交错完整自主已启动，结果尚未完整，不能提前晋升候选。

独立重演四cell全部优化窗口和末尾shuffle RNG，source trunk/buffers和共同清零head一致。报告修复了两类历史计数混用：494835完整顺序历史中，484755进入监督优化窗口，10080位于83个整块零监督前缀；后者真实更新了历史，不是训练遗漏。重复prefix重算由源码和准确排程另作推导，不称运行计时或额外独立数据。收据见evidence/tight_fresh_20261114/round05_anchor_training_exposure_audit.json。

报告工具同时核对预登记seed/domain/顺序、60Hz视觉/历史/后端计数，单列来源第五轮已有自主结果；转正用同tick物体与动作前目标姿态诊断，不把根距离当关键点距离，不把新目标切换后的姿态混入旧子目标。新增自主拿起/拿稳/对孔/插入及拿稳后角度图，19项定向测试通过；正式图仍待完整自主结果。

## 2026-11-14 拿起后转正的实际轨迹与朝向读出诊断

新增tight_fresh_grasp_trace及round04_round05_B_grasp_trace图/证据，绑定完整模型、登记机会、真实数值和结果。重复B/seed28140003两轮都能抬高并接近孔口；第四轮最小原生关键点距离48.710mm，第五轮84.604mm，均未达15mm容差。第四轮该时刻角差0.3447rad，第五轮1.5150rad；第五轮更晚出现的接触和稳定窗也不能说明转正改善。这是一例事后描述，不是总体差异或严格隐藏状态配对；两轮观察初态字段相等。图将真实接触指数量与过去12tick稳定窗端点分开，未用窗口false认定掉落。

新增tight_fresh_orientation_probe，在第五轮实际视觉512/robot178缓存上拟合固定λ=.01 ridge，预测同刻动作前R_palm^T R_object的前两列6D。每折排除6个完整训练family，学生编码器仍已学过这些轨迹，只有探针拟合排除；未读test。朝向不除形状对称，预测退化显式报告，不能替代原生对准判据。

学生接触后、按帧平均旋转测地角误差，纯视觉A1.2965→0.9730、B1.2956→0.9741rad（原示范拟合→其他family原示范+纠偏拟合）；视觉+本体A1.3130→0.9144、B1.3127→0.9114，本体共同1.5742→1.4309，常数2.0573→1.8938。全部预测旋转有效，无退化分母被隐藏。固定6验证纯视觉反而A0.3839→0.4147、B0.3834→0.4152rad，披露覆盖变化的取舍，不称全面改善。视觉含朝向信息但偏差状态仍难读出；不能因此确认策略使用得足够或视觉是自主失败根因。

4项坐标/符号/列顺序/退化测试通过；90轨迹35051帧目标与SciPy独立旋转组合最大矩阵差8.33e-16，全部有效。完整CPU探针12.62秒，零学生更新/模型前向/GPU/仿真。当前后端对照保持冻结cb47381，视觉辅助仍只为待选后续方案。

## 2026-11-14 阶段报告的参考点与关键点超差显式化

phases补充root_origin_distance_m、原生keypoint_distance_m及tolerance_m，同时保留历史center_distance与keypoint_gap数值；gap明确等于距离减容差。第五轮B近孔例实际关键点距离84.954mm、容差15mm、超差69.954mm，此前简写“关键点69.954mm”已纠正。沉降参考实际是root z，报告措辞修正；progression只允许已知旧“center z”文字与真实root定义等价，不接受其他参考范围或阈值。contact_motion/slip同样明确root参考点语义。

CPU重读第五轮8条numeric，全部既有数值/阶段字段逐项不变；六轮progression指标完全不变。新报告round05_phases_explicit_pose_semantics与核验收据保留旧报告SHA，旧报告不覆盖。此为解释修正与明确字段，没有更改物理、成功、稳定窗、训练或活动cb47381对照。

## 2026-11-14 独立后端输出起点对照已启动

第五轮完整学习、8例自主和新增后段拟合审查结束，自有worker与原monitor均退出后，建立tight_fresh_20261114_03独立2小时窗口。来源为第五轮固定验证选中的完整A/B，冻结cb47381，当前名义起点control与上一实际执行器目标previous四cell各15遍后端、各4完整开发机会。monitor先于worker启动，登记round05_anchor_study_launch；固定DP/84训练轨迹/90缓存、相同head-reset/fresh optimizer与真实监督曝光，尚未得到新行为结果。主Agent继续监督，不自动晋升候选。

独立姿态审核补全时间/authority/延迟标签边界、501完整窗口及原生子目标切换，原数值不变；相关38项测试通过。学生完成296步后专家从action296开始，首合法监督action298对应完成299步。原报告与源码版本绑定由更新证据明确记录，原始轨迹不变。下一阶段准备相关90项CPU回归通过。

## 2026-11-14 第五轮完整结果与下一后端输出起点对照

第五轮冻结32b3988在15:39:58（Asia/Shanghai）完整结束；84训练轨迹=原30+历史53+新1，仍30训练family，固定6验证/test22隔离。A/B各10遍DP、840更新、470400合法目标槽，选A第10遍/B第9遍；全部90条各自当前DP计划和可训练视觉缓存已刷新。后端各15遍2493更新、358725合法tick、494835真实历史tick，选A第13遍/B第15遍；新1实际后端曝光15次/3210合法tick，选中A权重只经历13次/2782tick，B为15次/3210tick。原30/历史53/新1实际合法后端曝光155970/199545/3210tick，不将最后权重账本混同选中权重。

固定6验证手指RMSE A0.0137064→0.0133282、B0.0152937→0.0148862rad；固定train2 A0.0128253→0.0116991、B0.0129640→0.0114917。唯一新增B实际专家后段沿完整真实历史重算，首12合法tick手指误差0.0284819→0.0188581rad（保持上一实际目标0.0210305），随后12为0.0217920→0.0131266，其余190为0.0178075→0.0133931。这是已学习记录上的拟合；全部12接管机会仍保留分母，A无合法own-role后段，不编造空均值或A改善。

完整自主8/8自然结束：A/B仍各0/4成功、0原生对准/near-goal/子目标。既定过去12tick拿稳窗口A2/4、B1/4；A的seed28140002在完成104–112tick满足窗口，seed28140003在207–211满足、最终457tick原生fall；其余7例601tick timeout。B seed28140003在315–380间有33个满足窗口时刻，最长连续15；最近目标root参考点距离10.849mm时角差1.524rad、原生关键点误差超出15mm容差69.954mm（实际距离84.954mm），未对孔。不能把窗口时刻数当持物总时长，也不能把参考点靠近当姿态正确。第四轮对应A1/4、B2/4，尚无可靠自主改善或A机制优势。

实际每例视觉/后端次数均等于控制tick，普通自主无额外专家前向；A/B吞吐11.886/12.120tick/s。round05_learning/family_exposure/increment_baseline/phases/runtime/onset与曲线、progression_through_round05绑定实际选模pair、数据、缓存及逐例结果；跨轮开发机会重复使用，不合并为独立test成功率。

下一阶段选择已准备的名义起点/上一实际执行器目标起点后端对照。依据是多轮离线误差下降没有稳定转化为拿稳/转正，既有同合法目标坐标诊断中上一目标坐标更小更平滑，重规划首tick预测也明显弱于同刻hold；这些只是待检验的优化解释，不是已确诊根因。固定第五轮DP、84训练轨迹及90当前缓存，两个候选都继承同组trunk/buffers并清零最后Linear，fresh optimizer，各15遍、固定6验证选模，四cell各4完整开发机会，共16；只改变输出起点，A仍q[t+1]+几何，B仍专家目标，完整DP计划仍输入后端。原名义坐标训练过的trunk偏置明确披露，原完整pair保持；若重置控制尚未恢复来源水平或候选自主无收益，不自动采用。所有第五轮GPU分析结束后建立独立2小时session，原02截止16:03:07不修改。此段记录选择，实际新阶段启动/完成以新机器收据为准。

## 2026-11-14 第五批接管姿态复核与诊断措辞修正

逐条件复核唯一成功纠偏B/formal_029：完成tick284满足原稳定窗，tick296接管时不满足；两时刻对应的过去12tick内，全部双物理子步均有至少两指接触，高度条件也保持。掌中参考点最大位移由5.228升至15.533mm，相对转角由0.19945升至0.28115rad，超出原10mm/0.2rad窗。窗口false只能说明该诊断条件不满足，不能单独区分主动转动、滑动或掉落；原“接管前再次失稳”措辞据此修正。

学生最后一步仍抬起并接触，但目标四元数角差3.03436rad、原生关键点最大误差63.099mm（原生容差15mm），尚未转正/对孔，学生前缀没有原生终止事件。专家继续使用新的真实观察执行，首次near-goal在完成tick458、首次对准及第一个子目标在467，第二个子目标及最终成功在512；这些均由专家完成，不能记作学生后段进展。四元数角差忽略形状对称，原生关键点判据仍为准。

新增tight_fresh_takeover_pose_audit和3项CPU测试；证据takeover05_pose_audit绑定原numeric/result、环境记录语义、触发器及审计源码。现有phases历史键center_distance实际是object/goal的root_state_w坐标原点距离，不是几何AABB中心距离；本例接管值12.959mm不等于已经对孔。保留原数值/阈值/成功判据/原始轨迹，不改活动训练。其他历史窗口false同样不能单独诊断掉落，需逐轨迹证据。

## 2026-11-14 固定family分组的位置读出诊断

新增tight_fresh_visual_family_probe：原30训练family按固定排序分五组，每次完整排除6family的原示范与全部纠偏，再以其余24原示范或其余family的原示范+真实纠偏观察拟合固定λ=.01线性位置读出。对被排除family的实际学生前缀，纯视觉RMSE A61.41→40.12/B61.22→40.06mm，视觉+本体68.97→38.45/68.80→38.41mm；接触后的组合83.97→45.45/83.76→45.41mm。读出拟合的状态覆盖能解释前述部分高误差，仍有较大残差，不能证明视觉已足够或信息已丢失。

此处只有线性探针排除了被评估family；冻结学生编码器早已学习过这些轨迹，不能称学生未见初态泛化或自主收益。当前几何仅作诊断拟合标签，未改变任何动作mask或策略参数；不读取validation/test payload。工具保存全部fold ID、实际标签/authority SHA、精确当前缓存及拟合系数身份；完整分片重hash，raw SHA仅引用manifest。4项family排除测试通过，正式CPU重算8.86秒与原诊断所有指标相同。证据[round04_visual_family_probe](evidence/tight_fresh_20261114/round04_visual_family_probe.json)。不据此直接切换视觉架构。

## 2026-11-14 可选后端输出起点的完整策略身份

previous-target候选的完整策略身份现在明确记录actual_pre_action_actuator_target_rad起点、输出合同和“上一实际目标加Transformer增量、完整DP计划仍输入、共同最终投影”的组合；配置与checkpoint输出版本不一致直接拒绝。当前名义起点的既有身份逐字段不变，避免误把候选记录成名义目标加修正。67项接管/输出/对照CPU测试通过，第四轮真实A/B身份重算与原plan完全一致；活动第五轮冻结源码未改，候选尚未训练/部署。

## 2026-11-14 后续时段的来源选模凭据

study_onset的source分析从登记checkpoint真实路径读取原selected.json并记录其SHA；独立session只链接best/finished，不再假设它复制了整个training目录。仍核验选中后端及前端/cache/data身份，不改来源收据或重新选模。此前第五轮与所有旧报告不受影响；此修改供完整pair审查后的新对照使用。

## 2026-11-14 第五轮两前端完成与后端报告人口修复

A/B各完成10遍840更新、20470规划anchor、470400合法目标槽，原30/历史53/新1实际episode更新300/530/10次，新增轨迹实际贡献190 anchor/4260目标槽，完整真实历史5120帧。固定6验证选择A第10遍、B第9遍，选择误差分别0.00812474→0.00771652（-5.02%）、0.00833612→0.00805608（-3.36%）；B第10遍略反弹，不把最后纠偏指标当作selected第9遍指标。证据round05_frontend_learning绑定实际学习收据；后端缓存与完整自主仍未完成，未宣称抓持改善。

准备中的anchor对照报告原硬编码第四轮53/12人口，现从真实注册manifest推导all/current correction ID并核验原30family、隔离split和合法takeover来源；本轮实际为54/1，旧权重对照仍只接受原53/12。图表按报告实际候选/遍数/分母绘制，防止沿用旧数量。39项人口/报告/恢复窗口CPU测试通过，真实A/B清单核验54/1一致；这是报告适配，不改变当前第五轮训练或启动新实验。

## 2026-11-14 第四轮接触运动与跨轮实际行为诊断

新增CPU工具tight_fresh_motion_probe，固定原train2/validation6、实际当前DP缓存和后端完整历史，绑定source/model/data/cache SHA。验证手指后端增量RMS为实际0.939/0.952，近零预测仅0.55%/0.45%，不支持整体静态保持捷径。首接触24tick的拇指CMC外展同向率A65.28%/B61.81%，误差为保持上一实际目标的1.023/1.226倍；食指及环指在部分family也差于保持，不能用统一收紧/松手规则解释。固定1e-4rad阈值同时报告运动能量覆盖与连续gain/cosine，全部12规划cursor及逐family分母保留。

前端整段手指RMSE A0.05941/B0.06923rad，优于保持5Hz规划起点的0.10652/0.11535；但刚重规划的cursor0仍为同刻保持基线的3.196/2.779倍，因此不能全归于计划过旧。A首目标是q(t+1)，B是当步专家提交后的实际绝对目标，均不要求等于当前观察。这里只是记录历史上的预测精度诊断，没有执行保持策略，也不证明自主失败因果。

新增tight_fresh_progression读取初始及四轮完整逐例回执、固定验证/训练诊断和selected完整pair，拒绝重复/未完成/专家接管/错配结果。后端验证误差持续下降而各轮A/B自主均0/4成功；稳定窗次数A为0/1/1/1/1，B为0/1/1/0/2。相同开发机会逐轮展示，不合并为独立test试验，也不把稳定窗当成功。原始报告和可复盘图为[progression_through_round04](evidence/tight_fresh_20261114/progression_through_round04.json)及[图](evidence/tight_fresh_20261114/progression_through_round04.png)，运动证据为[round04_motion_probe](evidence/tight_fresh_20261114/round04_motion_probe.json)。33项运动诊断测试通过，独立复核后补唯一ID/family、精确时间/缓存/归一化/标签绑定，原数值和prediction SHA保持；同向与近零比例明确非互斥。跨轮真实重算一致，重复阶段与其余8项异常输入被拒绝。零新GPU/仿真/优化，不修改活动第五轮。

## 2026-11-14 学生真实前缀的视觉位置分布诊断

新增tight_fresh_visual_prefix_probe及29项CPU测试；仅原30条10398帧拟合固定λ=.01 ridge，读取第四轮当前A/B精确缓存与53条已学纠偏的真实状态。元数据先验收family/split与A/B一致性，再核验完整模型/统计/data缓存身份、观察时钟、机器人特征、合法接管mask；不会把专家队列过渡或零标签行补成监督。当前几何中心标签仅供CPU诊断，不进入策略。证据round04_visual_prefix_probe绑定源码/模型/分片与实际读取的标签/时钟/authority张量；原大HDF5的SHA只引用manifest，不冒称重新全量校验。既存输出及悬空symlink拒绝覆盖。

固定验证6视觉位置RMSE A20.66/B20.73mm，学生前缀60.25/60.10mm，接触后学生状态72.73/72.52mm；本体在学生前缀82.35mm，组合68.18/68.00mm。按episode等权的前缀视觉均值34.10/34.09mm，保留两种口径揭示困难长轨迹权重。正式工具与先前CPU结果逐项差0，5.98秒、峰值0.93GiB，零学生更新/仿真/CUDA/test payload。纠偏已经参加学生训练，仅探针拟合排除它们；误差可来自原30探针拟合覆盖不足，不能断言视觉丢失几何、被策略忽略或是失败根因。探针不读持续视觉GRU。辅助位置监督仍为待选方案，本变更不修改训练核心或当前第五轮。

## 2026-11-14 第五批零标签报告与初段基准

onset_report/after保留全部12机会，未接管/无合法标签单独分组，空误差为null，只有真实成功且有合法标签的own-role后段进入前后拟合比较。A本批0条，仍核验完整A/B pair与当前缓存但不加载A推理；B为1条214tick。19项CPU测试通过，真实前报告绑定全部原轨迹及模型SHA。B最初12合法tick旧shadow手指RMSE0.02848rad，保持上一实际目标0.02103；余190tick为0.01781/0.02014。它是旧学生沿专家新观察的预测误差，不是自主表现。第五轮仍运行冻结32b3988，后报告须待新完整pair完成。

## 2026-11-14 后端输出起点对照代码准备（未选择运行）

新增可选reachable_previous_actual_target_linear_v1：零提案对应动作前真实执行器目标，当前名义计划仍完整输入后端；原受限名义输出合同不变。两种输出有相同最终可达域与区间内导数，候选只改变输出坐标和零输出先验，不是容量或物理修复。共同继承同组已训练trunk/buffers，最后Linear(128,29)同时清零、新optimizer、相同数据/缓存/shuffle/监督曝光/固定验证；未训练的重置head拒绝部署，旧权重禁止静默换坐标。

tight_fresh_anchor_study绑定来源完整自主证据、session/截止/监控归属、数据/统计/缓存与源码；四cell按交错完整DP＋后端队列比较，不自动晋升候选。report核验真实初始化、逐窗口曝光/更新、固定验证和每帧视觉；study_onset补增量方向及保持基线。51项相关CPU测试通过，已由子Agent对第四轮真实A/B各86个state tensor验证仅final head重置。当前round05运行于32b3988独立冻结树，未接入这些改动，也未启动此对照；是否选择和使用哪一组完整来源需依据round05真实结果。

## 2026-11-14 第五批稳定后接管完成，选择第五轮完整学习

第五批12机会全部自然结束：A6/6、B5/6未出现学生稳定窗而601tick超时，零合法专家标签；B/formal_029在首次稳定窗后继续12tick，于action296真实接管，512tick由专家成功完成。实际保留296学生历史tick、2队列过渡tick、214合法专家tick。接管时已不满足既定稳定窗，但两指接触和抬高条件仍成立；不能据此断定掉落。原生对孔/near-goal仍false，不能当学生自主插入或稳定后段覆盖充分。独立stage audit核验真实source、延迟queue、mask、q[t+1]与实际专家目标；所有失败现场保留。逐帧视觉/后端次数等于控制步，采集实际A7.66/B7.81tick/s，包括无损图像保存与校验。

选择round_05：原30+历史53+新1共84训练轨迹，仍30训练family；固定6验证、test22隔离。A/B共同接续第四轮选中完整pair，各10遍DP、刷新全部90条当前DP计划/可训练视觉缓存、15遍后端，再原4开发机会/组完整自主复测。冻结视觉同源分片可复用，参数相关缓存不得复用。沿用当前loss、采样、输出合同，不运行备选输出起点对照。本次数据与追加曝光共同变化，不声称单条数据的独立因果收益；原16:03:07截止保持，预计约60分钟。此条是决策，实际启动和完成另由stage收据证明。

聚合family准入从实际split与entry推导，拒绝未知/验证/test family及重复ID，修复旧默认train缺口。真实本批12均来自原训练family，未发现泄漏。小证据takeover_05、takeover05_stable_audit、takeover05_runtime绑定原采集与模型/源码；source人口不是优化曝光。

## 2026-11-14 聚合family准入修复

原families字典可能只记收到纠偏的训练family，缺少6个验证family；旧入口把未知family默认作train，存在准入缺口。现在从实际train/validation/test的split与entry元数据推导family，在读取新增纠偏payload前拒绝held-out、未知family、重叠split和重复ID。当前第五批严格登记原train30，实际12机会全部与validation隔离，未发现本批数据泄漏。44项相关CPU测试通过；真实A/B只读核查各30训练/6验证family，未读取test轨迹、未改原数据。

## 2026-11-14 独立第五批稳定后接管时段登记

已按持续研究授权建立tight_fresh_20261114_02，14:03:07至16:03:07（Asia/Shanghai）；原01截止13:59:40未变。来源为第四轮验证选出的完整A/B pair，未采用恢复初段三倍权重候选。来源训练/缓存/完整8次自主结果与模型hash由followup_session绑定；原始示范和旧结果只读链接保留，大raw/冻结视觉仅绑定已有manifest SHA及stat，未宣称重新全量hash。

第五批takeover_05已登记6个原训练family×A/B共12机会：051/007/050在首次既有拿稳窗后立即接管，046/053/029再真实执行12tick后接管。A/B交错，均完整DP＋后端，每60Hz真实视觉、5Hz规划。无固定时刻fallback；未拿稳或来不及切换就自然结束者保留全轨迹、零专家标签，不进入成功纠偏。不同family分配不同delay，不作delay的因果比较。监控已启动，采集尚未启动，训练量须依据这批实际合格后段决定；不把新session准备当行为改善。

前四轮后段覆盖的可复盘CPU工具tight_fresh_rear_coverage_report与证据已补齐：72机会、53成功，49从未拿稳、3曾稳但切换前失稳、1切换时仍稳；后者86/13303合法专家tick。完整训练曝光与validation-selected权重的曝光分开统计。40项相关CPU测试通过；本时段的真实采集验收尚待执行。

启动前补充独立collection_owner：核验原冻结worker源码与注册完整性，180秒初始化上限和原deadline不变，只回收自己创建的进程组，全部12机会/逐例稳定审计齐全才报完成。监控仅对明确stage_takeover run支持--output归属，仍拒绝路径逃逸和歧义；原无GPU监控已正常退出，记录保留，下一版监控在worker前接入。owner/ownership/eval_recovery/stage_takeover共78项CPU回归通过。

## 2026-11-14 完成匹配恢复权重对照、实际监督审计与阶段采集入口

冻结85ebd12的同预算对照于13:58:06（Asia/Shanghai）全部完成，原13:59:40截止未改。两候选从第四轮相同A/B后端开始，DP、89条缓存、统计和83训练episode完全相同；各3遍/490更新/71103合法监督tick、97431历史tick、249次episode曝光，首12步1908tick。全部优化窗口、实际shuffle顺序和Adam步数一致；权重1/3的loss质量分别71103/74919，后者不是更多数据。四候选均由固定6验证选第3遍；仅后端loss初段权重不同。

用当前DP缓存及完整真实前缀，在CPU两线程对三组后端重新计算全部53条已学纠偏，首12步手指RMSE：A原0.0117575、普通0.0114046、加权0.0108146；B原0.0112120、普通0.0110463、加权0.0105717rad。加权比普通降低5.17%/4.30%，但固定验证手指RMSE略差：A0.0137176→0.0137453、B0.0151767→0.0151878rad，normalized equal MSE也略差。此处全部53含两种学生来源，不与旧仅own-role的新12报告混比；三个候选相同CPU/完整历史，属于已学样本拟合。

完整自主两候选每组各4个原开发机会，合计16/16完成；四cell均0/4成功、0原生对孔/near-goal/子目标。普通与加权均A1/4短暂稳定窗、B0/4。加权A有2/4原生lift，其中seed28140000仅tick305满足稳定窗，tick355掉落；其余15例均601tick timeout。实际逐帧视觉/后端次数逐例等于控制步，普通A/B12.069/12.058tick/s，加权A/B12.169/12.042；普通自主无expert shadow。相同注册seed不是隐藏物理状态严格配对，四例不支持可靠收益或A机制优势。

决定不采用恢复初段三倍权重，不延长这个对照或自动重跑。下一轮保留第四轮选中完整pair，补学生自主拿稳后带偏差的真实状态。四批72接管机会中53被专家救回；53里49条从未有学生拿稳窗、3条曾拿稳但接手时已失稳、只有1条接手时仍稳。该条86个合法专家tick已在前端两轮更新20次、后端完成30遍，缺口是状态覆盖而非漏采样。新阶段入口仅原train30明确family、拿稳后0/12tick接管；未拿稳者完整自然结束并单列，不使用固定晚tick专家fallback。原任务/成功/60Hz/5Hz/绝对目标语义不变，尚未运行此新采集。

新增tight_fresh_study_onset/backend_study_report/study_plot绑定真实模型/数据/窗口、未加权训练与验证、全部自主分母；10项报告回归测试和完整真实报告通过。stage_takeover及既有trigger/DAgger共29项测试通过，真实Kit验收等待下一新session首例。slip_report复核第四轮A先相对滑动、B近孔例先相对转动再接触变弱，不能统一归为松手。证据round04_weight_study、round04_weight_independent_review、round04_slip_review及曲线均可复盘，test22未读。

## 2026-11-14 第四轮完整证据与已启动权重对照

发布第四轮学习/实际family曝光/命令保持基线/自主阶段/完整计时/曲线/接触变化/同轨迹恢复初段报告。83轨迹、89当前缓存、各830前端与2440后端更新核验；完整A/B各0/4成功，沉降参考短稳定A1/4、B2/4，B一例距目标约2cm后滑脱。选择并实际启动同当前pair后端3遍、恢复首12权重1/3的有限对照，固定85ebd12、原截止保持，尚无结果。monitor修复空闲接入并真实验证新归属，10.772秒遥测交接间隔明确记录。

## 2026-11-14 研究子目录的资源归属修复

state_resources.monitor的精确根目录匹配把本轮嵌套worker误标external。新增实路径子目录判断、worker相对cwd解析、兄弟/逃逸/歧义/坏symlink拒绝与启动源码SHA，测试test_state_resource_ownership及既有资源测试共24通过，真实当前worker归属复算通过。公开resource_ownership_fix，历史原监控保留，活动monitor尚未在空闲边界接入；full_gpu_v1当前不按该字段限速，未改变训练/物理/资源保留或任何无关进程。

## 2026-11-14 后端输出坐标的只读比较

新增tools/tight_fresh_anchor_coordinates.py及round03_anchor_coordinates.json。真实77缓存绑定当前DP/data/numeric，全部合法命令和观察精确匹配；报告当前受限名义起点与上一实际目标起点的所需raw幅度/相邻变化，不改后端、执行器或训练。A/B共同目标的候选指标精确相同，总训练合法tick20752与既有曝光一致；相邻差排除不连续mask/segment。首12纠偏的候选幅度和平滑性较好，但相同可达范围/内部导数不支持将其称为容量修复或自主收益。候选未选择。

## 2026-11-14 第四轮前端学习收据

公开round04_frontend_learning.json：两组实际830更新，原30/历史41/新12均每条10遍，所有计数与finished/learning收据相符；验证分别下降9.68%/10.56%，均选第10遍，后端缓存仍在刷新。没有把前端完成、已学纠偏误差下降或专家成功写成完整学生行为改善。

## 2026-11-14 视觉位置可解码性与诊断输入绑定

新增tools/tight_fresh_position_probe.py和round03_position_probe.json：原30拟合、固定6验证、固定ridgeλ=.01，比较纯视觉512/本体178/组合/训练均值，真实动作前物体中心转掌坐标作为诊断标签。组合验证三维误差A16.39/B16.43mm，低于本体19.66mm，但不作部署、视觉因果或自主成功声明。CPU独立全量重建结果精确一致；坐标旋转、ridge法方程、恒定维及帧/episode聚合检查通过，零学生训练/GPU/仿真。seed_spread补实际numeric/perception SHA及准确B目标时序说明，同模型全部数值精确不变。没有修改活动第四轮。

## 2026-11-14 训练初态诊断与扩散随机性实证

新增tools/tight_fresh_diagnostic_report.py与round03_training_diagnostic.json，按原4机会联合三个attempt，严格区分A完整0/2、B完整0/1加1删失；不打开/修复部分HDF5，不把progress当终止步。源码、pair、原计划、剩余队列、完整结果/控制频率/阶段指标校验通过。新增tools/tight_fresh_seed_spread.py与round03_seed_spread.json：两个原训练family、完整逐帧条件、完整过去5Hz噪声时钟、固定5种子CPU推理；限位后手指种子差异远低于实际误差，不支持仅调种子。两项均零优化/GPU/新仿真，保留CPU与CUDA随机序列不同、固定条件前端诊断不能说明自主因果的限制。当前入口同步第四轮实际训练中状态。

## 2026-11-14 robot178归一化实证与第四轮学习决定

新增tight_fresh_normalization_report与真实round03_normalization。77条numeric缓存与原始HDF5重建的canonical q/qd、实际target及严格过去issued/applied/age完全一致；原30全10398tick重算mean/std字节相同，178维没有触发1e-4 std下限。A/B四模型的DP model/EMA及后端共6套buffer全部正确。八次自主接触前max|z|<=5.18，首次>10均在接触后；初接触24观察最大A18.13/B20.08，主要小指DIP target-q 0.53–1.01rad，std0.04973。原示范同阶段max11.96、合法纠偏已覆盖26.5。大z只是实际接触偏差的描述，当前证据不支持缓存/归一化软件错误，不修改统计。

研究选择：当前早期偏差可由专家实际救回，新增12来自此前未接管的原6训练family；原30+历史41+新12共83episode，仍30初态family，独立6验证及test22隔离不变。下一轮round_04从当前完整pair接续，共同DP10遍、全计划/可训练视觉cache刷新、后端15遍，再原4开发机会/组完整自主复测。保持既有loss/采样/统计，首12加权候选不运行；该选择检验早期纠偏覆盖，不能将数据添加与追加学习剂量拆成单独因果效应。

预计由上一轮实测耗时及新人口估算约一小时，原13:59:40截止保持；剩余训练初态诊断释放GPU后才启动。原source53纠偏=41+12，不能称53新增初态。当前尚未优化新12条，不预告行为改善。

## 2026-11-14 训练初态诊断原生中断与未开始机会接续

pre-control恢复在11.08秒初始化成功，A/formal_050完整601tick timeout、无拿稳/对孔/子目标；B同family在执行中退出-11，最后progress为tick200。worker91.53秒退出，恰在90秒定时faulthandler dump处日志截断；目前不能确定定时dump与原生段错误的因果。没有把B部分轨迹算作完整任务失败，也不重跑A/B第一family。

恢复入口取消运行中定时dump，保留普通fault handler及owner180秒启动界限。新增tight_fresh_remaining只登记原plan中两次attempt均未出现case目录的机会，核验原plan字节、所有worker退出、session/pair/全部冻结源码身份；完成和partial均排除。41项CPU测试通过。选择仅继续第二family的原B/A两个机会，仍冻结4904336实际策略/执行源码。原4机会总分母与软件中断记录均保留，本条不预告结果。

## 2026-11-14 纠偏DP统计的完整规划噪声时钟修复

审查确认tight_fresh_train.metrics旧实现先过滤学生前缀anchor，再从seed7091重新取噪声，使首个合法纠偏规划错误使用tick0噪声；第三轮20条新纠偏首合法anchor为72–300，漏推进6–25次噪声。原30/固定6验证从tick0开始，选模数值、实际训练梯度、在线与后端cache均不受影响；旧纠偏DP指标不能声称与完整在线规划时钟配对，不将其与新口径直接作增益比较。

新增predict_on_native_clock只为跳过的真实5Hz规划消耗同形噪声，合法规划继续原DDIM16，不增加无用去噪；报告显式记录noise_schedule。11项聚焦CPU测试含原示范/验证逐项相同、学生前缀和mask洞正确推进，第三轮20真实mask核验通过。另用当前真实checkpoint在CPU核查25帧dense/逐帧condition最大差A1.31e-6/B1.67e-6，后端85tick分块/逐tick最终命令差均1.19e-7rad。没有新训练/仿真/GPU调用，未发现这些路径的新线上不一致。

## 2026-11-14 训练初态诊断启动超时与保持原登记恢复

冻结4904336训练初态诊断在Kit启动处卡住，无initialization、0case/0学生动作；180秒界限后SIGTERM无响应，由owner回收自有进程组，190.52秒结束、exit -9。它是软件启动失败，不计自主任务失败。物理初始化原生卡住的根因尚未确认，未修改共享模拟器/物理。

新增tight_fresh_eval_recovery --preserve-registered-plan：仅已退出且从未开始case的autonomous尝试可恢复；逐字保留原plan/session，核验原pair及冻结worker全部源码SHA/commit，原deadline与180秒界限保持。owner脚本与worker身份分别记录。27项CPU测试通过，覆盖source/pair/session漂移和只有started的部分case拒绝。选择在新输出round03_training_diagnostic_recovery运行原4机会一次；本条为恢复准备，不预告执行结果。

第四批新12纠偏的原完整学生shadow首12合法专家tick手指误差A0.0169216/B0.0138869rad，保持上一目标0.0196587/0.0192792；是在专家已经接手的真实历史上，不能代替自主恢复。公开takeover04_onset_before供本轮训练后同数据复算。

## 2026-11-14 第四批早期纠偏全部真实救回

冻结b581be9的最后6个原训练family×A/B、40/55/70/85/100/120tick接管自然完成，12/12合法且全部由专家完成原任务。12条全部通过实际捕获、相机、命令/标签边界审计，3913真实历史tick、2949合法专家tick；940学生前缀tick及24队列过渡tick不作监督。学生接管前无稳定窗/原生对准/子目标，不能把12次专家成功当学生拿稳或插入。

12.26秒初始化、578.71秒完整队列含收尾，实际采集吞吐A7.11/B7.70tick/s；每帧视觉/后端次数严格相等，持续lossless录制和最终重构审计成本单列。证据takeover_04/takeover04_phases/runtime绑定真实源/模型/数据。不同family分配不同接管时刻，不据此作接管时间因果比较；初态域未扩大。

冻结4904336训练初态诊断已登记原前2个训练family各完整A/B，单独分母，待实际结果判断再决定下一轮完整学习。当前旧41成功纠偏保留，新12尚未参加优化。准备的weight1/3候选只做CPU验收，没有选择运行。原13:59:40截止保持。

## 2026-11-14 接触阶段复查与训练初态诊断准备

新增tight_fresh_contact_motion_report及round02_round03_contact_motion证据，绑定同登记8例完整pair、numeric与可见初态。第三轮首次指尖接触51–67tick，6/8在80tick前相对接触前水平移动超过1cm；该量只是描述，不是新任务判据。A首例接触后24tick位移25.1→45.3mm、手指实际目标跟踪RMSE0.0356→0.1349rad；部分例子早期位移减少但仍未拿稳。未据此认定控制器故障或精确物理配对，支持当前更早接管的研究选择。

新增tight_fresh_train_diagnostic，登记原30中前2个family各A/B完整自主一次，使用原始成功记录的seed/event_seed/narrow_reset。它是单独训练初态诊断，不读test、不增加标签、不混开发分母；同种子不保证隐藏PhysX重现。审查先发现新cohort会误走旧状态恢复分支，已改为原生fresh重置路径、单独diagnostic_cohort标注；4项CPU测试含真实prepare入口和拒绝heldout/纠偏/变动source。此时未启动该仿真。

修复已准备但未选择执行的后端加权study：先准备A/B依赖再建立A后端，避免缺B公共命令统计；验证数量文字仅新显式fresh_validation_count_metadata配置启用，旧fresh配置身份不隐式改变。51项相关CPU测试通过，活动b581be9采集不热改。第四轮真实采集仍在进行，实际完成结果另记。

## 2026-11-14：第三轮完整结果与下一次早期接管

公开round03_learning/family_exposure/increment_baseline/phases/runtime/onset_after及曲线，77cache绑定与实际曝光通过；A/B仍0/4成功、A极短稳定1/4而B0/4。新增tight_fresh_frontend_baseline以当前准确cache和原numeric比较5Hz/60Hz观测保持，误差复算与实际DP选择验证一致。anchor_report改用模型实际ideal_execution_bounds，真实24分段统计逐项不变。CPU报告与实际GPU onset评估完成；后端加权候选暂不运行，决定当前pair在剩余6训练family上做12次40–120tick早期接管，尚未启动。

## 2026-11-14：归档过时任务授权

USER_RESEARCH_DIRECTION只保留本轮有效目标和实际状态；原分隔线后的历史任务文本逐字移至archive/USER_RESEARCH_DIRECTION_through_20261113.md，链接回当前入口，不再在活跃授权页堆叠旧禁令/期限。历史文字与Git证据保留，未改变运行配置或授权范围。实际逐字suffix校验通过。

## 2026-11-14：名义裁剪审计和可选恢复起点对照

新增tight_fresh_anchor_report.py及真实takeover03_nominal_projection.json，公开同合法状态下受限名义、旧完整shadow、上一实际目标比较与方向统计，不把离线作用称自主机制收益。新增visuotactile/fresh_recovery.py、tests/test_fresh_recovery.py；anchor_continue显式opt-in初段权重，setup/train二次准入、合法曝光与权重质量分开。tight_fresh_backend增加显式参数，默认关闭；新增tight_fresh_backend_study有限对照入口，原DP和全部cache仅在相同身份下复用，源完整自主未结束则不准入。当前无实际候选训练/GPU/新仿真，主Agent43项CPU测试通过。metadata的fresh validation4文字改为实际数量，历史权重不改。

## 2026-11-14：恢复起点对比与评估启动界限

新增tools/tight_fresh_onset_report.py、tight_fresh_onset_after.py与tests/test_fresh_onset_after.py，记录旧完整shadow和新完整pair在同合法专家恢复初段的误差、上一实际目标基线及精确数据/模型/缓存绑定。旧24轨迹报告已实际运行；新pair GPU对比待本轮完整循环结束。公开round03_frontend_learning.json记录实际10遍/710更新与分来源曝光，不将离线下降当自主收益。

tools/tight_fresh_cycle.py复用tight_fresh_eval_recovery.wait_owned，未来评估初始化限180秒；所有自有阶段进程组受原session deadline约束，超时写失败而不推进。保留旧失败与活动冻结源码。17项相关CPU测试通过，真实新受限Kit启动尚未运行；未来新worker才接入。

## 2026-11-14 当前任务入口清理

USER_RESEARCH_DIRECTION页首重写为当前已完成/正在运行/未证明的状态，移除原“新模型尚未训练”的过时快照；README同步第三轮20条真实纠偏及训练中状态。根AGENTS仍是唯一集中当前授权，下方历史证据保持，不叠加新冲突限制。仅文档，无活动worker变化。

## 2026-11-14 学习曝光按初态家族核算

新增tools/tight_fresh_family_report.py，依据逐更新learning收据与后端last实际episode_counts核算每family的历史、规划anchor、合法标签和后端tick；逐项与finished总量相等才发布。实际round02 A/B均51轨迹、30原初态family、0重复源文件SHA；每family前端10–50更新、4860–24950目标槽，后端2490–12710合法tick，总量与先前315580/160630严格一致。没有把同family真实新纠偏算成独立初态，也没有凭文件hash断言物理多样性。

当前第三轮71轨迹仍30family，episode数分布为10家族各1条、3家族各2条、14家族各3条、2家族各4条、1家族5条，固定6验证family无交叠。训练是逐episode等权，DAgger主动增加出现偏差family的曝光，不是逐family均匀抽样；该偏重已显式量化。本报告不改变活动训练或宣称它是失败根因。真实已完成两模型校验通过，第三轮实际曝光待完成后同工具核对。

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

### 2026-11-15 22:44 +08 — queue refill observation fix and shared DINO
- Real queue attempt02 passed startup and ran through the first indexed refill, then the unchanged lossless recorder rejected peer image post/pre mismatch. Preserve its partial records; do not count them as accepted training data.
- Cause: extra reset render refreshed continuing lanes at the same control time. `queued_rollout.py` now snapshots their already observed packet and expert input and replaces only the reset lanes. No previous physical state replay or synthetic image is used.
- `gpu_batch_controller.py` and `batched_controller.py` share one current DINO batch across A/B, avoiding per-role padded duplicate passes; sensor assembly excludes duplicate RGB transfer. Every active lane still consumes its real current view and updates its own history.
- Focused controller/bridge/refill tests:13 passed in1.82s. Real same-load attempt03 is required before formal adoption; attempt02 is not a throughput result. Scaling supervisor01 stopped without probes after attempt02 failure.

### 2026-11-15 22:49 +08 — eliminate duplicate render candidate
Source inspection confirms DirectRLEnv renders at every second120Hz physical tick, then the student image reader renders again. Added opt-in `current_render.py`, registered in new queue plans, to reuse a witnessed same-physical-tick render; reset/calibration invalidates it. One focused test covers current/stale/reset cases and cleanup; real collection acceptance is pending. Formal06 configuration now registers24 early-grasp opportunities: ten known hard training families plus two new narrow-domain families, A/B each. The original04 deadline, two-stage training, reserve and independent development remain unchanged. Scale supervisor02 follows completed queue03, using unchanged pre-registered16-tiled/32-standard probes. First queue03 service command failed before preparation (system Python lacked h5py); the corrected environment launched once as03b, with no simulation/data duplication.

### 2026-11-15 23:01 +08 — sequential same-deadline recovery
- Real queue03 completed all4 successful, audited suffixes. Queue pipeline is functional but raw successes/hour did not improve on this varying-length four-case load; actual per-control-step figures are preserved separately.
- Render01 and old04 autonomous development both hit RAM reserve when overlapping; original04 training had completed. Do not report04 as a complete cycle. Preserve partials and perform new executions for unresolved cases only, using saved14 weights.
- `scripts/tight_fresh_20261115_06/throughput_preflight.py` serializes capture/scale/recovery, records failed attempts, bounds startup memory, then selects formal concurrency from actual headroom. Launch02 gains an opt-in preflight hook; other bundles retain their prior behavior.06 sources evaluatedA11/B09 plus completedround14 pool; no deadline extension. Fourteen focused modified-path tests pass in2.21s; real render throughput remains pending.

### 2026-11-15 23:03 +08 — distinguish selected policy age from resource clock
`tools/tight_fresh_followup_session.py` accepts an explicit closed resource parent, retaining its exact deadline and task/checkpoint identities. New07 replaces pre-GPU failed06_launch; old launcher receipts remain. This fixes recovery from older reliable policies within current04's clock, without opening a new time window. Updated actual bundle and opt-in launch argument.


## 2026-11-15 throughput08 bounded recovery
- async_recording, queued_rollout, tight_fresh_batch_run: eager bounded spawned verifiers before native simulator/CUDA; real capture contract unchanged.
- tight_batch_owner/eval_recovery:180s stale-control progress dump and owned-group cleanup.
- collection_source/round_compare: recognize dynamic GPU queue receipts and censored grasp execution; actual15 recovered round14 numeric/history/counter audits passed.
- scripts tight_fresh_20261115_08: same04 deadline, A11/B09,14 DP epochs/latest4 then refresh/rear12/latest4;18 genuine early-grasp opportunities. No old training repeated.
- throughput scripts: actual five closed-file salvage (934 valid expert steps), hash-bound derived-cache retirement; raw data/weights preserved. Formal24 stalled and is not a completed throughput experiment.8 focused eager/lazy real-HDF5 tests passed.


## 2026-11-15 camera retirement/readback recovery09
- queued_rollout.RenderLanes: retiring render products now also excludes their image buffer and CameraParams reads; re-enabled lanes acquire fresh observations.08 thread dump identified stalled Warp annotator copy after retirement; verifier prestart alone was insufficient.
- recovered_captures/narrow_data/fresh aggregate/loop/owner: explicitly hash-bound admitted closed successes from interrupted attempts may join the next update under their original training registries. Eight real successes1477expertsteps registered; no partial prefix or unsuccessful trajectory admitted. New/current and recovered counts separate; all new identities receive front/rear latest4 sampling.
- New09 frozen run bundle retains original deadline and18 real early-grasp opportunities.12 focused camera/control checks and63 existing aggregation/owner checks passed; actual8 recovered entries checked for both consumers.


## 2026-11-16 reuse prepared data after storage admission failure
- fresh_prepared_reuse and cycle --prepared-from: new output only, same task/collection/deadline, refuse any prior optimizer run; copy metadata/numeric/loss-context and committed frozen feature indexes, retain original files.
-10 ordinary bounded launcher/owner: new16GiB disk reserve with original deadline, existing parallel A/B frontend headroom gate, selected-DP refresh then rear and behavior review. No recapture or repeated training.
-09 actual18/18 successes653.75s99.12/h;26new aggregate. Cache interrupted by32GiB reservation before training. Camera retirement/readback fix passed real whole collection.

Prepared-data continuation follow-up: preserve an already identical sampling contract only in explicit reuse mode; no resampling/reweighting of started learners.10 failed before GPU/optimizers;11 is separately registered.

Prepared reuse now materializes validated per-identity links required by committed-cache admission, with conflict rejection; original payload/index bytes remain untouched.11 failed before updates;12 uses a new identity.


## 2026-11-16 bounded CUDA allocator reclamation
- FullLease opt-in reclaim_unused_on_low_headroom_v1 releases only unused process-local CUDA allocations; checks same6GiB reserve using live free memory and records allocator telemetry. Global/driver settings untouched.
- Front trainer preserves diagnostic-only partial-epoch state on Python exceptions; no partial checkpoint is treated as selected or complete.
- Prepared-data reuse may import independently completed frozen cache indexes from12 while retaining untrained09 data; rejects learned projection cache/data hash mismatch.
-13 budget12 front epochs/2460updates each, rear12/latest4, original deadline.12 A71/B165 uncheckpointed updates lost after VRAM guard; full cache338 retained.10 focused checks passed.


### 2026-11-16 — focused grasp diagnosis and supervised night recovery preparation

Added tools/tight_fresh_supervision_audit.py for actual numeric labels and simulated timing, tools/tight_fresh_pair_fit.py for selected complete rear command fit on original2/latest2. Added versioned grasp_touch trigger across grasp_curriculum/takeover_trigger/training_expansion/report/stage_takeover/expansion CLI. Added same-source explicit cycle resume, owner heartbeat/stall guard and finite tools/tight_fresh_night_owner.py with two-real-cycle review barrier, one resource retry and no-gain stop. Current13 frozen code unaffected.43 focused CPU checks passed; real new cycles and pair-fit execution pending. scripts/tight_fresh_night_20261116/retire_last.py retires only named closed final optimizer states, keeps every best;9.753GiB logical retired. SESSION_HANDOFF corrects resource threshold from actual JSON.

Added recovered-family review lookup, original train2 full-policy diagnostic registration and explicit completed-DP resource-boundary continuation. Latest focused run had46 passing checks plus one new test-fixture scope error; that fixture was corrected and its3 targeted tests pass. No change to active13 implementation.17 obsolete Adam states archived with exact model/EMA preservation; current baselines excluded. Background readiness remains unproved pending two real cycles.


## 2026-11-16 01:49 +08 — B15 complete; slow storage and supervised-loop hard-family checks

B15 completed12epochs/2460optimizer and EMA updates,4920 complete-history visits; each latest26 correction48visits. B final validation arm/hand RMSE .005069/.055638rad; latest correction .003868/.040536rad. These are frontend command errors, not complete-policy grasp success. A is in epoch12, still updating. Around01:37 the NVMe showed97–98% utilization at55–57MB/s reads and29–35ms read latency; own trainers each read about29MiB/s. Individual436MB frozen-feature loads reached14.36s while ordinary GPU updates were under1s. This is measured I/O limitation; no hardware-failure or unrelated-job blame is established. No unrelated process changed.

Prepared finish26 fallback can resume an incomplete DP from the last durable full epoch in its ORIGINAL6938b5c trainer after the old temporary lease closes. It preserves AdamW/EMA/RNG/decks/2460budget, copies append-only logs, retains old partial rows separately and reuses completed B without updates. Normal completed-DP handover remains preferred. This fallback has two focused checks, not actual resume acceptance yet.

Next night cycles now append the newly corrected training families to the same queued autonomous grasp batch. Rechecks have no takeover, retain their actual registered resets, and are labeled separately from fixed development; no mixed generalization rate. Added numeric-only closed-gap report for the original train2 probes: actual issued/actuator parity, initial observed-state differences and descriptive earliest3-tick command/state gaps versus the old successful trajectory. After state divergence, that old expert path is explicitly not a counterfactual expert label. No extra simulation/model calls. Six focused handover/night/recheck checks pass; one typo named a nonexistent test path and executed0tests before correction. Two real full night cycles have still NOT passed.


### 2026-11-16 01:54 +08 — both DP complete; real rear refresh running

A/B15 both completed12epochs/2460updates/4920visits; latest26 each48visits. Normal completed-DP boundary succeeded, old13owner/monitor exited, new00 owner started at01:50:32 under09:00 deadline. Actual source01ecfe4; no DP resumption or repeated updates was needed. Both current-DP rear caches now advance (A42/338 at01:52:34); no simulator overlaps. See throughput15/round15_frontends.json for exact selection, errors and geometry/anchor coverage.

Found two prospective unattended-recovery defects before releasing the barrier: cycle resume rejected an already registered sampling contract as soon as a training directory existed, and the owner watchdog omitted backend/cache_progress.json. Fixed identical sampling resume to be read-only and reject changed receipts; added real cache/train/offline progress paths. The running frozen00 owner is not hot-edited. A narrow, recorded progress relay will mirror only changed actual backend progress into its otherwise unused role-level progress index to avoid a false900s timeout; it never marks completion or manufactures progress. Real two-cycle readiness remains unproved.


### 2026-11-16 02:03 +08 — cache complete; declared recovery exercise prepared

00 A/B refreshed338/338 each in628.3seconds, then entered real rear training. Runtime progress relay is active and copies only changed actual backend records; the frozen owner's stale timer now correctly follows work. No additional model updates or simulation runs are caused by the relay.

The first supervised small cycle is configured to exercise one coordinator recovery after BOTH real DP stages finish. It records a distinct `controlled_completed_frontend_resume_probe`, not a false resource/model failure. The ordinary one-retry path must reuse both exact completed frontend receipts, with no repeated updates, then refresh/train rear and evaluate. Second cycle uses ordinary automatic transition. Two targeted checks prove the new coordinator path skips both mock trainers; real acceptance is still pending. This is part of the user's requested two real end-to-end supervision cycles, not permission to release the unattended barrier early.


### 2026-11-16 02:14 +08 — lossless checkpoint write reduction

New fresh DP training saves each completed epoch snapshot once, then atomically links the selected-best name to that immutable snapshot. A later last save replaces its inode, so an earlier selected best remains unchanged. This removes redundant tensor copying/Adam serialization and duplicate disk allocation when last and best are the same epoch; optimizer updates, EMA and RNG are unchanged. One focused test covers first promotion, later nonselected last replacement, reselection and optimizer/RNG preservation. This path is for future frozen cycles; current00 training stays untouched.

Prepared a narrowly bounded CPU operation for the now CLOSED13 frontends: only retire each last.pt if its ENTIRE payload equals the retained selected best (including model, EMA, Adam, RNG, histories, epoch12/update2460). Original SHA/stat and retained-best SHA are inventoried, open descriptors reject retirement. This is duplicate serialization retirement, not removal of unique optimizer state. No raw data, current00 rear checkpoint, sourceA11/B09 or frozen feature is selected. Actual outcome is recorded under13/exact_duplicate_last_retirement.


### 2026-11-16 02:20 +08 — exact duplicate retirement passed

Closed13 A/B last and selected-best payloads match exactly, including Adam/RNG/history. Removed only those two redundant last serializations,2,330,423,296 allocated bytes; full selected-best snapshots remain. Actual free bytes35,662,209,024→37,992,615,936. Also preparing a narrow four-file optimizer-only archive for the CLOSED Nov11 G11 pool's comparison201/402/603 and last endpoints, retaining its original selected best.pt unchanged and preserving every model/EMA tensor plus all nonoptimizer fields in new explicit archives. This supplies headroom for the two requested real cycles without deleting observations or current caches. Its exact manifest/apply outcome is authoritative, not this preparation note.

The original-family closed-gap analysis now compares object positions in the robot base frame, removing different environment-lane world origins; it still discloses hidden-state replay limitations. No new physical run or model inference is added.


### 2026-11-16 02:32 +08 — compare learning under the same execution backend

Confirmed historical A11/B09 source evaluations use CPU physics/whole waves; current15 uses GPU physics/queued execution. Their old rates cannot isolate learning from the backend change. New night01 first measures the retained A11/B09 complete pairs once on the same current GPU queue and fixed development seeds, then compares15 before choosing a collection source. Adoption rejects differing backend/queue/render contracts. Lane populations may still differ; no exact hidden-state replay or statistical reliability claim. Current frozen00 evaluation and its historical comparison remain preserved, with this later bridge explicitly authoritative for model selection.

Found a prospective next-round handoff error: the source auditor accepted only development cases while the new evaluation also contains registered train-family rechecks. It now verifies each optional recheck against the exact completed collection plan/reset and reports it separately; adoption uses only shared development seeds.33 focused collection-source/night checks pass. No new physical cycle has completed yet.

Four old Nov11 G11 comparison/last endpoint archives completed with every model/EMA tensor and nonoptimizer field preserved, original selected best unchanged. Removed only obsolete Adam state,2,330,959,228 logical bytes; actual free disk at02:30 was40,318,574,592 bytes. All current trajectories, caches and source pairs retained.


### 2026-11-16 02:47 +08 —26 update completed; contact-state mismatch persists

00 owner completed02:43:32, monitor and owned children exited. Both rear trainers completed12 control-tick epochs/9200updates/1,348,476 supervised ticks/4920 real history visits. Latest26 each48 visits in final training; selected A9/B11 contain36/44 visits respectively. Before150 legal-expert command hand RMSE fell .010631→.008376rad(A), .011899→.008719(B). Actual previous-target baseline is also beaten offline; this is not autonomous success. Rear refresh pair took631.4s including process startup, rear training pair span2016.8s, main16 autonomy228.38s. No DP work was repeated in00.

CurrentGPU fixed development first300ticks: A hold60=0/6 (2 illegal starts), B2/8; contact+height lift3/6 and5/8, native lift1 each. Only4 legal executions ran full task, insertion0/4;10 were deliberately grasp-censored, not full-task failures. Originaltrain2: A0/2,B1/2 hold60. Hard families67/78: A1/2 (78), B0/2. New independent dev2: A0/2,B0/2. No90% claim or move to late-stage focus. Old frozen review's CPU source comparison is preserved but superseded for selection by01's current-GPU source bridge.

Numeric original-family check shows submitted targets equal both actual physics-substep targets exactly. Initial joints already differ from historical demonstrations by .0104–.0373rad despite matching object reset positions; time-aligned old command divergence from tick0 cannot isolate model error. New CPU-only tool tight_fresh_expert_counterfactual queries the released expert on the SAVED actual student observations and full LSTM history, using recorded native delay draws, native mapper and real previous target. On one already closed DAgger stream,150ticks reproduce expert raw means within5.126e-6 and mapped targets within2.384e-7rad (predeclared limits1e-4/2e-6). Four real autonomous traces show hand error .0078–.0096rad before48, rising to .0146–.0198 around48–100. These are diagnostic suggestions, not executed expert labels or new data. This supports earlier true-state contact correction; it does not establish a sole cause. First prototype hit metadata singleton limit-axis shape and produced no report; reshape fixed it before the successful actual validation.

01 launched from frozen27baa3e with owner PID516082, services tight-fresh-night-20261116-01 and tight-fresh-night-monitor-20261116-01. At02:46 same-backend old-pair evaluation tick75/14active; zero completed new DAgger cycles. Absolute night guard remains09:00 +08 (owner admission cutoff08:58). No unattended review release exists. See throughput15/round15_backends.json and round15_diagnosis.json.


### 2026-11-16 02:55 +08 — same-GPU selection and first earlier-contact collection

01 source bridge complete: on the shared current-GPU fixed seeds, A11 hold60=3/6 versus A15=0/6, so retainA11. B09=0/8 versus B15=2/8 (28500003/28500005 now hold1s), so tentatively useB15 for further collection. Both complete pairs/baselines retained; this small operating decision does not establish broad reliability. The CPU historical comparison alone would have missed B's current-backend improvement.

Round16 collected4/4 valid successful rescues from two existing hard families67/78 using completeA11/B15. Actual first-contact→takeover intervals1,1,1,3 control ticks; takeover59,61,62,67 (0.983–1.117 simulated seconds), no fixed first-step or12-tick grid. Added1017 legal expert steps; pool336trajectories,170 registered train families. All4 closed data passed admission in163.786seconds,87.92valid/h. This is a smaller/different batch than prior18/653.75seconds(99.12/h), not an isolated speedup claim, and still below600/h goal. Cache reuse and DP updates started automatically, B40updates observed02:54:43; no full supervised cycle has completed yet. Source/runtime and compact results:throughput15/night20261116.json.


### 2026-11-16 03:26 +08 — actual resume and command decomposition

Round16 real A/B DP stages each completed4epochs/696optimizer updates/1392 complete-history visits. The declared completed-DP coordinator recovery then ran; resumed finished receipts retain the exact recorded hashes and696updates each. A rear refresh342/342 completed463.58s and rear training is advancing. This validates completed-stage reuse, not yet a full recovered cycle or unattended readiness.

Extended the CPU actual-state expert diagnosis to compare saved nominal, rear preprojection and final commands. Reconstructing the real projection matches final targets within1.863e-8rad. In all12 case/phase hand comparisons, the complete final command is closer to the same-state expert than the nominal with that SAME projection. For B contact controls48–100, projected-nominal hand error .024118/.019646rad becomes .016650/.014880 with the complete policy. Thus these traces do not support simply removing the rear as the next fix. A's q[t+1] domain remains distinct; its nominal-to-command difference is not a forecasting error. This is a saved-history one-step diagnosis, not a new executed rollout or proof that rear behavior is always beneficial.

Freed4.3397GiB by retiring four CLOSED Nov12 validation_best containers only after comparing the ENTIRE payload to each retained best (model/EMA/Adam/RNG/all metadata exactly equal), binding both original hashes and checking no open descriptor. Every unique training state remains; raw data untouched. Script retire_closed_matches.py and local01/closed_checkpoint_dedup receipts reproduce the scope. Available disk33.7GiB after operation, preserving the29GiB next-round admission and16GiB runtime reserve. Two complete supervised cycles still pending.


### 2026-11-16 03:54 +08 — evaluation resume guard defect, explicit bounded recovery

Both round16 rear trainers completed4epochs/2563updates; no optimizer work remains. The original frozen coordinator incorrectly rejected its newly prepared development directory during --resume simply because that directory existed. It contains ONLY plan.json/session.json, no run_started, simulator initialization or case output: zero evaluation starts occurred. Owner and monitor stopped safely with exact failure retained. Four focused checks now distinguish prepared-only from any actual startup evidence; pending/partial physical cases remain protected.

recover_prepared16.py explicitly launches the already-bound plan in the ORIGINAL frozen worker tree, under the unchanged09:00 deadline and a new monitored service. It may write the missing development stage receipt only after real evaluation succeeds, then the old coordinator reuses all completed receipts to finish16 and advances17. Original failures are archived before current operational status changes; all four DP/rear completion hashes must stay unchanged. This is a supervised repair, not evidence of unattended readiness. The repaired guard applies to subsequent frozen versions; the old tree is never hot-edited.


### 2026-11-16 04:03 +08 — actual camera refill failure, preserved evaluation attempt

The prepared-only recovery launched real evaluation, but it failed after the first control step when a rejected refill left a lane disabled and the next task re-enabled its Hydra products. Stock Camera attempted to copy an empty RGB annotator buffer into[252,448,3]. The failed worker exited and the recovery monitor stopped; two pre-action illegal cases and seven partial one-step traces remain in original development. This is software failure, no student failure/success conclusion. No completed cycle or unattended readiness is claimed.

The collector now keeps a vacant render product enabled while queued work will immediately refill it, avoiding one-tick disable/enable churn. Once the queue empties, inactive products still retire. A bounded fallback checks reactivated RGB-D shapes and allows at most4 additional real renders at the SAME physical step before failing explicitly; no stale/zero images or extra physical steps. Four focused camera/render tests passed. A fresh execution of the original12-case small evaluation, with unchanged weights and reset specs, is needed to verify this exact rejected-refill path. Original interrupted evidence will remain separate; the two zero-action rejection opportunities are repeated for software verification, not replaced to improve a success rate.


### 2026-11-16 04:25 +08 — repaired evaluation complete; retain A11/B15, change early learning exposure

The separate `01/round_16/development_camera_repaired_01` completed all12 registered cases in151.99 worker seconds, including the previously failing rejected-refill path.10 legal,2 pre-action rejects; original failed evaluation and partial traces remain untouched. Fixed development hold60 A0/3,B0/4; same-backend shared-source subsets A11=1/3,B15=1/4. Reject both16 candidates for collection. Training-family rechecks A0/1,B1/2 hold60; B family78 held once, but initial q differs by0.01668rad from the earlier hard-family probe, so this is not a matched-state improvement. Four full-task legal executions: insertion0/4; six short grasp executions are censored for insertion. No alignment/subgoal observed.

Round16 real DP A/B each4epochs/696updates; selected A3/522,B4/696. Rear each4epochs/2563updates; latest4 each16 visits in selected rear,12/16 in selected A/B DP. The legal latest early-before100 exposure is only0.458%/0.442% of all selected DP target slots and0.609% of rear ticks. No student-prefix labels or action-mapping error was found. Corrected contact timing alone did not yield a development grasp gain. Actual intervals: collection163.79s, frozen cache124.04s, paired DP1011.85s, serial rear preparation1157.39s, rear training971.77s; full collection-to-repaired-eval4673.83s includes two repairs. Unified readout explicitly reports coordinator_complete=false; this is not a clean reliability cycle.

New explicit learning profiles are implemented and54 focused tests pass, but have not yet performed optimizer updates: DP samples half its unchanged per-trajectory anchor budget from legal ticks<150 where available, filling shortages without replacement; full real history once per visit, every legal time remains eligible, actual deployment/rear plans remain commit12. Rear assigns weight3 to all legal expert ticks<150, including original approach demonstrations; prefix and queue-transition masks remain excluded, later legal targets weight1. On the16 pool,19017/90941 legal ticks are early; their weighted mass is44.23%, versus20.91% raw. This intentionally changes the early gradient contribution under the existing fixed128 optimizer denominator; weights are not extra observations and unchanged LR does not imply equal total gradient scale. Old first12 takeover-weighting control is preserved and unchanged.

`launch_after_camera_repair.py` prepares a NEW02 identity from complete16 cumulative data and retained A11/B15, with the same absolute09:00 deadline. Clean supervised count resets to0; actual rounds17 AND18 must pass after these repairs before any unattended release. No completed26/16 training is repeated and no original raw data are removed. Storage admission remains29GiB/runtime16GiB. Current stage/PIDs are authoritative only in new launch and night/state receipts, not this prepared entry.


### 2026-11-16 04:28 +08 — new17 capture admitted, two clean cycles still pending

02 launched from frozen30aeb80 at04:22:21. Ordinary owner `tight-fresh-night-20261116-02.service` PID561430 and monitor PID561422; source tree `bc+rl_RSS_fresh_grasp_20261116`. No concurrent simulator. Round17 four genuine full A11/B15 prefixes on existing families76/86 all reached first persistent contact, takeover60/62/68/87 controls, exactly1 tick after first contact; all4 expert rescues admitted,1262 legal expert controls. Collection173.052s,83.212valid/h, no speed claim versus differently sized historical batches. Cache/DP/rear/evaluation follow automatically.09:00 deadline unchanged; two new clean supervised cycles still0/2.

For disk headroom, two CLOSED Nov13 adaptive validation_best containers were retired only after ENTIRE payload equality against retained best (including Adam/RNG):2.16991GiB. Four obsolete initial/round1v3 frontends were archived without Adam only, preserving model and EMA exactly and every nonoptimizer field:2.16929GiB logical. Selected A11/B09/B15,26/16 training, all current frozen features and raw observations remain. Per-file original SHA/new SHA and manifests under02/closed_adaptive_duplicate_retirement and closed_initial_optimizer_archive; explicit scripts recorded. Available disk34.39GiB at04:27:42, next-round admission29GiB/runtime16GiB unchanged.

Use ONE read-only entry `python -m tools.tight_fresh_night_status --root <02>` for stage heartbeat, actual update counters, capture completion and resource headroom. A missing supervised_release is intentional. Readout additionally separates rear's real early action exposure from weighted loss mass; no large image re-audit.


### 2026-11-16 04:47 +08 — real I/O stalls; bounded fixed-sample diagnosis prepared

17 DP is advancing; B has entered its fourth epoch and A its third. Actual eager feature loads occasionally take11–21s per326–619MB shard, making a B2 update28–32s versus ordinary<1s. Around04:35, A6/B5 slow updates each consumed~151s. NVMe later showed2.7–3.0GB/s, then again~60MB/s with35ms read latency at96.5% utilization. Worker prefetch threads waited on file pages; cgroup high/max/OOM events were0, SSD composite53.85C (reported max89.85C), no memory/thermal cause established. No external process or system setting changed. These are measured stalls, not a model hang or a claim of storage hardware failure.

A one-container lossless Zstd1 probe retained exact entire-file bytes but saved only8.28% (348.19MB→319.34MB) and added0.47s decompression. Do not adopt or claim formal speedup from it. Existing DINO payloads already use FP16; naive half-precision conversion is not a new optimization. Full every60Hz visual coverage remains.

Six closed pre-fresh Nov11 pool round2/3 and Nov12 matched round2 frontends now retain every model/EMA and nonoptimizer field in explicit weights_only archives; only obsolete Adam removed,3.254GiB logical. Original SHA/archive SHA/exact field comparisons are in02/closed_pre_fresh_optimizer_archive. Raw captures, current A11/B09/B15/16/17 states and current feature caches remain. This restores headroom for18 without lowering reserve guards.

Prepared `tools.tight_fresh_fixed_rear_fit`: if these early-learning cycles fail to gain holding, freeze the real selected DP and diagnose whether the COMPLETE policy's final commands can fit two original successful training trajectories. Reuse only that exact DP's bound plans/features; optimize rear with the established current-weight history replay and fixed128 tick denominator, bounded80 passes/15min per role. This is train-only capacity diagnosis, not a new formal candidate or autonomous success. It has only compiled at this point; no diagnostic optimizer update or rollout yet.

Metadata erratum: frozen30aeb80's rear configuration still contains the historical free-text `recovery_weighting_scope` wording about first contiguous correction ticks. The new versioned `fresh_recovery_weighting` contract, actual masks and population explicitly weight all legal controls<150 including original demonstrations, as intended and tested. Future source corrects that description; running17/18 frozen code is untouched. No inference about the math is made from the stale descriptive string.


### 2026-11-16 05:11 +08 — bounded fitting actually executed

`tools/tight_fresh_fixed_fit_owner.py` fit-only mode completed real A/B80passes/560updates using frozen A11/B15 plans on two original train histories. Selected early hand RMSE .002829–.003860rad versus .008195–.008922 before; no autonomous result yet. `scripts/tight_fresh_night_20261116/launch_fixed_fit_comparison.py` prepares serial source/fitted evaluation only after17/18 and main shutdown, retaining the original morning deadline and reusing fits. `tools/tight_fresh_night_status.py` includes independent fitting; `tight_fresh_cycle_readout.py` includes distinct legal-time coverage. Python compilation passes; running frozen30aeb80/be8a4da sources are unchanged. Actual results added to existing throughput15/night20261116.json.


### 2026-11-16 05:17 +08 — real cycle17 complete,18 automatic transition

One supervised real loop passed in51.74min,4new successful rescues; autonomous fixed development holdingA0/3,B0/4, training rechecks0/4, insertion0/4 complete tasks. Reject17, keepA11/B15. Actual selected exposure and timing in existing night report. `tight_fresh_cycle_readout.py` preserves nested rear metrics (real selected validation hand RMSE .01136/.01151rad), `tight_fresh_night_status.py` labels completed fixed fitting as complete instead of displaying its last heartbeat as live. Readout executed on completed17; original report retained, no physical/training repeat.


### 2026-11-16 05:26 +08 — optional state-based pre-contact recovery

Added `tools/tight_fresh_approach_timing.py` (numeric-only original30/latest8) and versioned approach screen in grasp_curriculum/takeover_trigger/training_expansion plus collection CLI/admission/report dispatch. Six of8 genuine prefixes offer4–21controls earlier help; no new rescued data implied.25 focused tests pass, actual saved-prefix replay agrees. Also corrected contact-v2 textual capture category to early-grasp (underlying trigger/admission was already correct); historical artifacts unchanged. Counterfactual diagnosis can compare multiple completed directories with one expert-validation call and a descriptive sustained-error onset. Main17/18 frozen implementation unchanged; new approach collection is not yet run.


2026-11-16 05:29 +08 clarification: the proposed approach screen excludes ANY prior>0.1N object-contact sample in either120Hz substep, including a one-substep transient. It does not infer absence of touch merely from the stricter two-substep holding/contact count. Eight targeted grasp checks pass, including a transient-contact case; actual8prefix replay still yields the same6earlier/2unchanged times. Authoritative retrospective receipt:02/approach_timing_v3_any_contact.json. No simulator has yet used v3;18 remains unchanged.


### 2026-11-16 05:48 +08 — bounded phase selection comparison and future exposure configuration

`tools/tight_fresh_fixed_rear_fit.py --compare-completed` ran read-only B17 best3/final4 validation6 comparison: early and all-time errors both favor best3, no model/selection change. `tight_fresh_night_owner.py` forwards separately registered DP/rear repetition counts, defaults4 preserved, finite1..32. Prepared a closed-root exact-entire-payload duplicate retirement script; no data/checkpoint removal yet. Active frozen30aeb80 and fittedbe8a4da unchanged. Python compilation passes; actual B17 metric comparison completed without simulator or optimizer work.


2026-11-16 05:54 +08 — Unified read-only status now prefers finished receipts over stale last-progress counters, reports the actual root-bound live owner, omits completed-round detail by default (--all-rounds restores it), and supports the forthcoming fixed-fit comparison owner and its serial queues. Verified on active18; no training or simulator state changed.


### 2026-11-16 06:30 +08 — two real loops passed, bounded fitting diagnosis, new pre-contact cycle prepared

17 and18 finished cleanly without manual stage recovery;18 began automatically after17. All owned02/03 simulation/training workers and monitors have now exited. TWO consecutive actual capture/admission/DP/new-DP-input/rear/autonomy loops were supervised.02 stopped safely for two rounds without clear development hold gain, not a software failure. No unattended release has been written. Monitor5716 samples, no missing, minimum free VRAM4286MiB and RAM18.99GiB; fullGPU permitted. This establishes these actual loops, not immunity to future faults.

18 admitted4/4 rescues on existing81/94,1118 legal expert controls. Cumulative344train/170registeredtrainfamilies plus6validation; trajectories are not independent initial families. DP both4epochs/712updates, selected A3/534,B4/712. Rear both4/2641updates, selected A3/1988,B4/2641; new4 each12/16 visits in selected A/B networks. Selected latest early hand command RMSE .010166/.009983rad. Early rear loss mass44.73%, raw~21.24%. Development hold60 A0/3,B1/4 versus retained-source1/3,1/4: reject both. Training rechecks A1/2,B0/2; A81 held and achieved one native subgoal. Four full-task insertion0/4, seven other legal trials grasp-censored, one pre-action illegal. Historical A81/A94 initial q differs .02942/.02040rad, so old-to-new family observations are NOT matched-state learning proof. Stage union seconds: capture169.16, frozen cache133.05, DP624.20, rear inputs828.17, rear training593.98, evaluation170.06; entire42.56min. Collection85.13valid/h remains far below600/h target.

03 actual fixed-original-train2 comparison reused the finished80pass/560update rear fits with unchanged A11/B15 DP. Four source then four fitted300tick full-policy trials completed serially (60.73/82.40worker seconds). Every pair's initial observable q/object position matches exactly, not hidden PhysX state. Source A0/2,B0/2 hold60; fitted A1/2,B0/2. Original formal029 A now holds73ticks after contact71/lift110 versus source contact88/no sustained lift; formal050 A remains unstable. B lost prior brief lifts despite lower saved-history error. All8 short trials are censored for insertion, no subgoal/alignment. This is one real fixed-training-case A improvement, not general grasp stability or adoption of the diagnostic weights.

Counterfactual actual-state expert queries passed the existing150tick mapping validation, without simulation/updates/admission. Fitted B's100:150 hand deviation is .03181/.02315rad versus source .01337/.00966; better offline fit did not preserve favorable closed-loop states. A029 holds despite an earlier command-deviation threshold crossing: do not make command mismatch alone a failure trigger. Evidence supports genuine pre-contact coverage plus better fit; no single exclusive root cause is established. The previous B17 early-validation check favors the same epoch3 as global validation, so selection-metric replacement is unsupported here.

Storage: attempted closed-last dedup stopped before deletion because best/last were already hardlinks. Preserved failure; no duplicated-byte saving claimed. Explicit archive_closed_grasp_artifacts.py then finished:8 obsolete Adam payloads retired, every model/EMA and other field exact; six unused learned-plan caches losslessly zipped, every original-byte SHA verified, exact restore manifests retained. Current A11/B09/B15/05/06 and selected18 SHA unchanged, raw histories/frozen DINO untouched. Actual available disk22.123→27.695GiB (+5.573GiB); old16/17 optimizer resume intentionally unavailable, their weights retained with explicit archive identities. No active file removed.

Prepared launch_precontact19.py creates NEW04 identity with retained A11/B15, cumulative18 data, real v3 approach opportunities on hard76/86,8epochs and latest12 for BOTH DP and rear. The6-earlier-of8 retrospective result remains zero new rescues until live collection. Combined timing/exposure change is an evidence-driven recovery experiment, not component causality. One finite cycle only, same09:00 deadline and16GiB runtime reserve;27.5GiB admission accounts for measured recent~9GiB output plus~2GiB checkpoint temporary. Need actual startup/collection evidence before claiming it runs.


### 2026-11-16 06:29 +08 —04 actually started from frozen9772a5c

NEW04 launched, owner tight-fresh-night-20261116-04.service PID623880 and separate monitor. Source tree bc+rl_RSS_fresh_precontact_20261116 @9772a5c9e281e5b694a083f94ca87a6d416f4673; one registered19 cycle,09:00 absolute deadline/08:58 owner cutoff, current stage source_19. Prior17/18 clean reliability evidence and actual03 diagnosis bound in registration.12 focused grasp/owner checks pass. Initial launcher invocation with system Python failed on missingh5py before creating a run; corrected invocation uses existing Isaac environment, no simulator duplicate. Monitor confirms no concurrent GPU experiment,~28.47GiB free disk. No v3 rescue/update yet and no unattended release.


### 2026-11-16 06:35 +08 — live v3 rescued4/4; repair repeat12 before any optimizer work

04/new_family_19 completed4/4 natural successful rescues and admission in146.537s (98.269valid/h),1022 legal expert controls. Real prefix takeover A76=67(contact formation),B76=64,A86=41,B86=45; last three are approaching_first_contact before any prior finger touch. Earlier-timing retrospective is now supplemented by actual recovered trajectories, not counted twice. All120/60/5Hz semantics and actual delays/labels preserved. Compared with17/18 the phase and recovery durations differ; this98/h is actual throughput, NOT isolated engineering speedup.

The subsequent cycle exited2 at argparse: lower --latest-repeat/--backend-latest-repeat and sampling decks still accepted only1/2/4 despite owner accepting1..32. No aggregate, cache, trainer, optimizer update or candidate output was created. Owner/monitor safely exited and preserved the full rejection. Repaired all callers/decks to use one bounded REPEAT_FACTORS1..32; old1/2/4 RNG/order unchanged.63 focused sampling/cycle/owner checks pass, covering repeat12/32 real deck counts and resume exposure. One old test still expecting3 invalid was corrected to33; no physics check broadened.

recover_precontact19.py is an explicit one-time new-frozen-source recovery, using the four existing admitted trajectories and unchanged8epoch/repeat12 budget. Original collection/source9772a5c, failed owner/configuration/monitor and failure logs remain; new operational config separately names the repaired commit. Remaining-stage admission26.5GiB accounts for the already-spent~1GiB capture, runtime reserve16GiB/deadline09:00 unchanged. Must observe actual recovery before any unattended release. Numeric supervision audit adds --latest-only to reuse unchanged original30 evidence and names the latest population correctly rather than hardcoding26.


### 2026-11-16 06:38 +08 — corrected19 aggregate/cache advancing; new pre-contact labels verified

Explicit recovery service tight-fresh-repeat12-recovery-20261116-04/PID627393 uses frozen4e48da52368b2b565c56d99d74cb3311474ef231 in bc+rl_RSS_fresh_precontact_fix_20261116. Original9772a5c collection and failed interface attempt unchanged. Reused4/4 admitted captures, no repeated rollout/update; actual aggregate348train/170families+6validation complete, frozen visual cache now advancing. First invocation's operational state/monitor/config are preserved under04/night/repeat12_recovery; same absolute deadline09:00, owner cutoff08:58.

Existing numeric audit --latest-only completed in0.129s on JUST the4new captures (no RGB rescan/original30 repetition). Exact A q[t+1], B/rear actual executed absolute targets, prior-command alignment, both physical-substep targets and prefix/queue masks pass. New4 have1022 legal expert controls,175before100/375before150;44 truly pre-contact labels (A86=21,B86=21,B76=2,A76=0). Switch41/45/64/67, first legal expert43/47/66/69, respectively; all saved student history remains input-only. Expert suffixes complete in3.37–5.00sim seconds; capturewall146.537s is separate. Planned DP budget each8epochs/1568updates/3136trajectory visits, latest4 each96visits. These are PLANNED, not yet completed optimization or autonomous benefit.

Prepared CPU-only completion_report.py waits for the actual19 completion receipt and writes its existing cycle_readout.json (including both-network selected exposure, true autonomous stages and full intervals). It never starts GPU work, retries a failed experiment or adopts a model; safe failure/deadline is recorded instead. Finite ordinary owner already handles stage handoff, one declared transient-resource resume and safe cleanup. No unattended release yet at this entry.


### 2026-11-16 06:41 +08 — bounded19 released after actual17/18 supervision and verified recovery

The agent has reviewed17/18 consecutive actual complete loops, their automatic transition/checkpoints/admission/autonomy/cleanup, the real19 v3 captures and the repaired repeated-training path.04/night/supervised_release.json explicitly permits ONLY the already-registered single19 cycle; it does not claim19 finished or authorize an unbounded next loop. Original interface failure remains visible. Latest release-time actual DP learning counters A90/B196; new4 have A4/5/7/4 visits and B12each. B's first full deck and both real optimizers therefore exercised repeat12. Planned final8epochs/1568DPupdates and rear96visits per new trajectory remain incomplete; selected-checkpoint exposure will be reported separately.

Active owner tight-fresh-repeat12-recovery-20261116-04.service PID627393, monitor PID627428; training/evaluation source4e48da5. CPU-only tight-fresh-completion-report-20261116-04.service PID629416 waits independently for actual completion, then writes04/round_19/cycle_readout.json. Owner state/heartbeat and exact current stage log are under04/night/state.json and its stage_ledger; use python -m tools.tight_fresh_night_status --root outputs/visuotactile/run/tight_fresh_20261116_04. Shared resource reserves4GiBVRAM/10GiBRAM/16GiBdisk, serial simulation, finite transient retry,900s progress watchdog and camera180s watchdog remain;08:58 owner cutoff within09:00 session window. Ordinary faults preserve completed receipts and stop clearly if recovery is unsafe. No additional model success is claimed.

Completed scientific result remains: sourceA11/B15 retained;17/18 fixed development not improved; diagnostic fittedA029 newly holds73ticks from identical initial observable q/object, B train2 not improved. New19 now provides44 actual pre-contact expert labels; its post-update autonomy is still pending. Do not equate4expert rescues or early loss with student success.


### 2026-11-16 morning review —19 completed08:02; candidates rejected, all owned services exited

04 cycle19, independent monitor and CPU completion report all completed/exit0 by08:03, before09:00 deadline. Current no active owner/trainer/simulator; stale heartbeat in read-only status is historical completion, not a live hang. Actual local readout04/round_19/cycle_readout.json. Both frontends completed8epochs/1568updates/3136visits, both rears8epochs/5720updates/844680legal target exposures. Latest4 were visited96times in each completed DP/rear run. Validation selected DP A6/1176 (latest72each), B7/1372 (84each); rear A8/5720 (96each), B4/2864 (48each). Do not substitute final exposure for selected-checkpoint exposure.

Fixed development: all legal starts contacted, A3/3,B4/4; contact+height lift A1/3,B0/4; sustained1s A0/3,B0/4, versus sources1/3,1/4. One A pre-action illegal is separate. Training-family rechecks A1/2,B0/2 sustained1s. A76 first contact63, hold60 first completed264, longest uninterrupted contact-height60ticks; final window NOT held. A86 briefly holds25ticks and native lifttrue, no1s hold. A76 native liftfalse, illustrating distinct criteria. Four full-task executions0/4 insertion; seven other legal trials grasp-censored. Zero alignment/subgoals. Both19 candidates rejected; retain completeA11/B15, no promotion toward later-stage collection.

Compared17→19, all four same-family recheck initial observable joint arrays and object-in-base positions match exactly. A76 changes from no1s hold to one qualifying hold, but fails final-window holding. Different event seeds and unobserved simulator state remain; one execution is not reliable-grasp proof or hidden-state exact replay. First attempted comparison incorrectly filtered nested spec.diagnostic_cohort and returned empty; corrected top-level selector asserts4matches in round17_initial_comparison_checked.json, superseding the preserved empty receipt. No simulation repeated.

Latest early-before150 hand command RMSE improves A .010087→.007149rad, B .009338→.007044rad, but fixed development worsens. Therefore latest labels were actually trained in BOTH networks and early fit improved, yet this four-example timing+exposure combination does not yield general grasp improvement. Contact-before-help coverage is real (44new expert ticks) but not sufficient evidence to repeat blindly. Next diagnostic priority is why saved-history fitting fails to preserve favorable closed-loop states, and current slow feature reads; do not claim a unique root cause or start another identical cycle by default.

Actual stage unions minutes: collection2.44, frozen cache2.12, DP49.42, new-DP inputs10.06, rear20.59, autonomy3.46; capture→evaluation92.56min including preserved pre-optimizer failure/recovery. Capture98.27valid/h, still below600/h. A/B median optimizer batch .673/.621s, but66/62 batches>5s consume1753/1683s; a largest batch loads two frozen tensors for29.30+23.04s (A),28.28+22.32s (B). These measured load stalls dominate the long tail, not evidence that more simultaneous learners alone improved total throughput.

Monitor retained a222.09s shared-log gap06:32:07–06:35:49 across the documented failed-CLI safe shutdown and explicit recovery; it is not an active-training freeze and not continuous whole-session telemetry. Runtime recorded peakused19171MiB (free>=4913MiB), minavailableRAM18.79GiB. At09:05 disk18.61GiB, insufficient for another same-sized cycle's admission; original data/current rollbacks remain. No new resource window, collection or optimizer run launched by this status review. Reviewed result/diagnostic summaries published to existing origin feature.


## 2026-11-16 — Classified cleanup, complete16–19 curves and causal limits

- Code: visuotactile/frozen_visual_io.py and native_data.py add explicit mapped-current v3 immutable reads to existing DP/rear preparation; old eager/prefetch/default behavior unchanged. tools/tight_fresh_train.py records input wait and update-host intervals without changing sample/order/loss/update budget. Exact bool configuration validation retained. tests/test_frozen_visual_io.py covers both profiles through real CPU batch transactions.
- Actual: retired14.873GiB with pre-deletion manifest, process-reference checks, exact preserved model/EMA and354 readable current records. Raw trajectories/current169.30GiB visual pool/recent16–19/currentA11/B15 protected.
- Results: TIGHT_TRAINING_DIAGNOSIS_20261116.md centralizes actual7360DP/27062rear update curves,34historical role-cohort stage rows, common300tick source comparison, new/old endpoint fit, exact execution-chain reconstruction and inference-only expert diagnostics. New4 fit improves while original2 worsens; rearB19 selection4 vs8 is a0.93% normalized metric difference. No20thDAgger or simulator starts.
- I/O: real disposable coldB2 same1.56GB inputs1.244→.905s; warm1.018→.445s. No full-cycle speed claim.27focused tests pass; source weights untouched. Scripts/configs/small data+figures under scripts/tight_diagnosis_20261116 and evidence/tight_diagnosis_20261116. Initial failed bitwise GPU probe and corrected diagnostic-window attempts explicitly retained locally/public receipt, not presented as passing results.


### 2026-11-16 固定分布预算诊断登记
新增bounded_extension与tight_budget_20261116脚本：严格DP末检查点续训，独立验证平台减LR/停止，固定cohort逐epoch物理阶段诊断，后端对应DP刷新与同序列8轮比较快照。旧路径未启用时不变；无新增数据/采样/架构改变。31项聚焦检查通过，实际训练/行为尚待。

预算诊断真实数据preflight：先拦住几何抽样未显式seed、再拦住验证记录缺训练专用阶段标签；均0正式更新。固定诊断采用原两构型几何估计，训练半抽样不变。修复后A/B各18条逐阶段/固定噪声指标通过，CPU/CUDA训练RNG逐位不变，耗时5.46/3.78秒（不含setup）。允许登记的平台停止作为完整有界训练完成，不伪称达到24轮上限。


预算对照正式第12轮遇到并行长轨迹GPU峰值OOM，原失败/partial/log保留；退回完整11状态，已恢复A12真实更新。新增有界顺序GPU恢复与独立当前/下一B2顺序文件预读；数据、B2、算法、RNG与窗口截止不变。对齐并发方案弃用于后续DP，不能以初期读盘提速掩盖OOM。更多环境不适用于离线轨迹训练；闭环已预登记16有效机会/组。

预算诊断收尾工程：scripts/tight_budget_20261116/finish.py在现有训练完整结束后顺序运行只读端点、预登记短闭环与CPU曲线，失败/截止明确退出，不自动重训。report.py新增实际optimizer横轴、独立验证选点、固定噪声分量与各cohort阶段CSV；端点记录选中模型实际曝光，观测到的插入与完整任务分母分开。真实DP曲线渲染检查通过，当前B训练/后端/闭环未完成，不提前宣称效果。

期末预算端点增加可选cohort_ids，只读覆盖全部原始30，其余仍历史4/最新4/val6；冻结76352bd正式训练和默认每epoch原始4完全不变。原始样本保留结论不从4条擅自扩大到30条。实际DP A22/B24已完成，后端新DP输入正在刷新，行为结果尚待。

预算后端354/354×2刷新完成并实际进入优化器。只读status补首次验证前更新数，behavior_report补每例DP/rear哈希、共同合法初态交集及逐初态持物获益/退化；不改变推理、数据分布、监督或成功标准。

预算图表明确标出8epoch和降LR位置；保存时包含完整标签边界。短闭环只读对照同时量化初始q/qdot、物体速度、base/object/goal/hole朝向、目标相对基座位置及离散阶段标志，仍明确不宣称隐藏物理状态精确重放；不增加仿真试验或改变任何训练。

预算曲线同时导出原并行OOM各140次未提交更新的小型指标，并在optimizer图中单独标橙色分支；它们不接入恢复模型的有效曲线/曝光，但计入整批耗时。原始日志与部分checkpoint未删除。

### 2026-11-16 — 固定分布延长预算与三组闭环完整收尾

- 实际训练：DP A22选14/B24选23，后端各24选22；原8预算DP验证改善6.17%/8.18%，同新DP后端8内best改善7.29%/5.07%。无新数据/采样/架构变更；DP完整恢复、后端新DP缓存绑定与匹配初始化。31聚焦检查的结果复用；实际优化、检查点、各阶段交接及42合法短闭环均正常结束，初始OOM/280丢弃更新继续单列。
- 实际行为：16登记初态×3组，完全相同非法起点分离；原19重跑→DP延长+短rear→都延长，A hold60 1/7→0/7→4/7，B2/7→3/7→1/7。A原先未持物的训练86、开发001/005新增1秒窗口，但末尾未保持同判据；A76对准/首子目标，不是完整插入。保留A候选，B长rear不采用，A11/B15回退保留，无长期循环。
- 报告：集中MD首屏重写当前结论，原16–19历史不覆盖；全部原始30端点显示B原8无整体遗忘、延长可同时改善原始/纠偏，纠正原2条外推范围。evidence/budget_extension保留真实epoch/update/分量/分段/曝光与图，无插补/伪数据；读者可区分训练拟合和独立验证。
- 公共逐epoch指标无损压成四份JSONL.gz（22/24/24/24个实测点），重读逐值相等，约2.81MB→.28MB；研究机原始日志不动，不重复保留派生JSON副本。report.py后续再生使用同格式。
- 收尾代码：behavior_report.py增加共同合法交集、初态比较和全部四训练family物理曲线，标签修正重绘的数值SHA保持；performance_report.py增加整阶段实际活动环境/有效控制步，189.148分钟完整成本含失败恢复，不冒充提速。closed_command_gap.py实际仅CPU1.80秒查询8条B真实状态，复用同SHA验证；plot_closed_gap.py只读绘图。B接触前命令差翻倍、执行目标误差0，但初始DP计划小差异保留因果边界；0新仿真/更新/入库。
- 运行退出：13:34:57收尾完成，owner/finish成功退出，监控已停、MainPID0。脚本README与SESSION_HANDOFF明确禁止重启已完成root，原6h截止未修改。当前约24GiB可用，未再删有效数据。训练源码冻结76352bd，闭环plan记录a7df427及逐文件SHA；只推代码/配置/小证据到原origin feature分支。
## 2026-11-16 延长训练后持物诊断：实施入口

新增`scripts/tight_stability_20261116`：读取全部7条A长训轨迹、相同真实专家历史下短/长后端分解、只读捕获完整后端输入与16环境有限新开发评估。未修改网络、损失、采样、控制或物理；没有优化器更新/新DAgger。A已有两条明确失去接触落下，一条变单指但仍抬高，一条仍持物并首子目标；不能把末窗口不满足一律写成掉落。实际新闭环结果尚待。公共精简证据集中于原诊断evidence/stability。


### 2026-11-16：有界完整任务补充与真实视频
- 新增scripts/tight_fulltask_20261116：真实观察视频、共同训练池更新、有限阶段恢复、成对native reset RNG及动作前q/qdot/目标核对、实际曲线/结果。继承120/60/5Hz与完整DP+Transformer。
- 2完整成功专家纠偏/4机会，9完整视频；DP两组372更新，rear A683完成/B183部分，截止停止，未获得新模型公平完整闭环。监控生命周期故障和status临时文件冲突均保留；后者加锁，前者尚未完成统一owner验证。
- stability补充真实同输入重算与新开发结果；明确initial q混杂与确定性recenter残差语义。详见集中报告。未修改原生资产、物理、动作映射或最终成功判据。
