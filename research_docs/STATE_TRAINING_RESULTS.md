# State observation/history repair — actual results

## 2026-11-03 00:19 UTC+8：训练完成，评估不完整，尚无学生完整成功

当前服务inactive，owner/全部自有worker/monitor于11月2日23:21:53退出。
机器状态为`training_complete_evaluation_incomplete`，不是整个实验complete。
StateBC完成10000updates，StateDP完成19000updates；两者均train320/validation40，
覆盖全部320训练身份，按验证集选择10000/19000的EMA，未使用test选择。

|模型/评估|完整成功/已完成|其他实测结果|缺失项|
|---|---:|---|---|
|BC所选10000|0/12|fresh0/4、train0/4、val0/4；全601timeout，无lift/子目标|无|
|DP19000里程碑|0/4|train1条lift后254步fall；另3条601timeout；无子目标|无|
|DP所选19000|0/5|fresh4条601timeout；train1条lift后273步fall；无子目标|第6条启动失败，后6条未启动|

DP里程碑与所选模型诊断分开计数，不合并为9个独立起点。
第6条`dp_single_05`在建立场景前发生FileThread/std::system_error SIGABRT(-6)，
只有runtime_controls.json，无动作或轨迹；该条未知任务结果，不能记为任务失败或成功。
另有DP10000里程碑第4条启动SIGSEGV缺失。三次自动Kit恢复已消耗，未追加重试。
因为两项deferred未解决，旧State12条对照、新test40离线和各组正式20条均未开始。
因此当前不能给出完整正式成功率或新旧State在同一新cohort上的最终改善幅度。

最终validation前12步arm/hand RMSE：BC0.617104/0.491214，DP0.715060/0.500374。
DP从10000步0.798099/0.559021继续下降，仍未形成完整成功。
训练指标仅固定前16条：BC0.540785/0.468841，DP0.612155/0.436233；
优化器实际使用全部train320，不能将16条指标当成全部训练分布指标。

所选DP已完成5条的reset Actor与对应专家均逐字节相同，第0步arm/hand命令已偏离；
reset动作RMSE跨轨迹平均0.619776/0.275060。chunk边界/内部跳变量RMS平均
0.455739/0.188447，提示重规划不连续；尚无因果干预证明其为唯一原因。
在线专家12/12成功与fresh replay2/2 exact16仍支持可运行场景下的控制基础设施。
软件启动不稳定解释缺失评估，不能解释已经实际完成的学生闭环失败。

只读收尾核验169源码哈希、88已提交stage结果、两份best权重SHA；
复用14条视频完整解码结果并重新核对文件哈希，全部通过，未新增训练或仿真。
资源报告含13279.81秒监测间隔（包括停机），整卡采样均值36.599%、最大完整60秒69.208%；
当前协议无固定利用率上限，且无另一项目吞吐指标，不能据此声称零干扰或全时段覆盖。
证据：[训练完成核验](evidence/state_shared_20261102/training_complete_review.json)、
[学习曲线](evidence/state_shared_20261102/results/learning_curves.png)、
[逐轨迹诊断](evidence/state_shared_20261102/results/summary.json)。
以下启动状态均为历史快照；本节没有宣称原生任务已学会。

**22:58:28 UTC+8实际恢复核验：** `state-full-20261102-01.service` active，
owner1297238、worker1297395存活，当前`dp_train_19000`。
resume.json确认从10000 checkpoint恢复optimizer/EMA/RNG/sampler；此时最新已提交进度仍10000，
不能把19000阶段已启动写成19000更新已完成。无资源暂停，freeVRAM18661MiB、availableRAM28.76GiB。
运行源码`fae5a0497377668f36440fd9ffaaf6a0788592f5`已push并核对origin同SHA。
原失败评估仍deferred，尚无学生完整成功。独立证据：
[runtime_configuration_resume.json](evidence/state_shared_20261102/runtime_configuration_resume.json)。
后续由普通服务推进，未使用对话持续轮询监督。

## 2026-11-02 22:52 UTC+8：恢复校验发现运行代码与训练身份耦合

`44e9424`恢复核验正确延期了场景前SIGSEGV，但DP19000在更新参数前报
`Training configuration drift`。该版本训练配置冻结了全部Python文件哈希，
包含此次只修改启动分类的state_full_run.py；因此保护检查阻断了数值配置未变的恢复。
两次尝试均只有空stage目录，没有加载模型/优化器或新增更新；仍是DP10000。
原日志、失败状态、自动恢复记录与空目录归档runtime_recovery_04，不把启动当完成。

本次将兼容处理限定为显式登记的“旧配置SHA + 精确源码before/after SHA”对照。
配置文件和checkpoint保持原字节；数据划分、网络合同、loss、优化参数仍逐字段严格相等。
网络源码变化或未登记的新trainer哈希均拒绝。trainer仅新增这段配置验证，
从数据加载到训练/EMA/RNG/sampler/保存的原实现逐字节不变，已实际核验。
10项针对性回归通过，实际last.pt确认updates10000及原config SHA一致。
此次是修正校验后的显式恢复；已消耗的自动训练恢复额度和3次Kit额度均保留，截止不变。
后续能否运行以独立startup check为准，不能因systemd active就称训练进行中。

## 2026-11-02 22:43 UTC+8：BC完成，DP10000与启动崩溃

StateBC与StateDP均已在train320/validation40完成10000updates，各覆盖全部320身份。
BC预算已经完成，验证选择10000；最终诊断fresh0/4、train0/4、validation0/4，
共0/12，全部601步timeout，无lift/alignment/子目标。不是完整任务成功。
DP的1000/3000/6000里程碑各0/4；10000里程碑已完成0/3，
其中train第一条native lift后第227步fall，另两条601timeout；均无子目标。
第4条在场景建立前崩溃，未知任务结果，不能写成0/4。DP最终选择/12条诊断与test尚未完成。
两组300update pilot及各自4条闭环已完成，均0/4；不与上述正式训练诊断混合。

DP validation arm/hand RMSE从1000步0.870098/0.647178降到10000步0.798099/0.559021。
6000→10000的selection MSE为0.552337→0.474733，改善14.05%，未触发既定早停。
因此原预算允许从合法checkpoint继续至19000检查点，不能宣称DP全部训练已经结束。
BC validation终值0.617104/0.491214；这些均为teacher-forced离线指标。

已重读新模型逐步诊断：BC12条reset Actor最大误差均0，但第0步arm/hand命令已偏离专家；
reset命令RMSE跨轨迹平均0.522935/0.424145。DP10000已完成3条同样reset Actor完全相同，
第0步动作误差平均0.590561/0.319983。初始决策误差在学生历史累积之前存在，
不能仅把失败归因历史恢复或闭环漂移。BC chunk边界/内部跳变量RMS跨轨迹平均
0.182732/0.104591；DP为0.532526/0.175608，提示DP重规划不连续值得诊断，
尚未证明它是唯一原因。专家12/12及fresh replay2/2证据仍有效；新旧模型最终对照尚缺。

22:36:20 owner因`dp_loop_10000_single_03` exit=-11停止；owner/worker/monitor均退出。
只有runtime_controls.json，无场景/轨迹文件；native fatal堆栈为
`libomniclient.so!omniClientFreeContent`与`omniclient::provider_file::FileThread`。
这说明启动软件故障位置，不是已证明的底层根因。此前分类只识别SIGABRT，未覆盖这次SIGSEGV。
本次仅补充“明确exit=-11 + 完整特定堆栈 + 场景前”分类，并在恢复时记录具体未完成stage。
三次启动重试已用尽，不扩量重试；该条保留为缺失评估，继续既定数值训练及其后新阶段。
任何已开始的轨迹/控制错误仍阻断；存在deferred时正式test保持未启动。
原模型/控制/数据/优化/截止不变；训练截止11月3日14:12:42、绝对截止17:12:42 UTC+8。
9项针对性回归通过，包括未知崩溃拒绝、已建场景拒绝、耗尽重试预算的恢复不启动新attempt。

本次停止时的资源报告含11997.15秒未监测间隔（含此前停机），不能把0 missing_samples
解释为连续覆盖。整卡已采样峰值占用21022MiB；6GiB是准入阈值，不是全程硬预留。
另一项目吞吐不可得，不声称全程资源无干扰。原始日志与收尾表均保留，恢复后以status.json为准。

## 2026-11-02 20:42 UTC+8：基础设施通过，BC首次真实学习

新冻结cohort专家12/12合法完整成功（fresh4、train4、validation4）；
两条same-command replay均完整成功，Actor/command/delayed/delay index/target/q/qd/
object/contact/object_contact/force/torque/lifted/subgoals/retract/alignment共16字段逐字节相同。
foundation正式通过。此结果支持当前单环境条件下的控制/随机流一致性，不证明历史4lane精确重放。

持续StateBC pilot已实际300updates、train4/val4，全部4条训练身份被采样；
首25步平均loss0.881002，末25步0.562472（下降36.16%），GRU梯度非零。
teacher-forced prefix12 arm/hand RMSE：train0.824452/0.727427，
validation0.939361/0.769086。模型尚未充分拟合，不能据此判定完整320数据训练能力。
首条train初态自主闭环0/1、601步native timeout、无lift/子目标/alignment；
Actor normalization maxerror0。其余3条尚未完成，不能报0/4。
逐步证据显示reset Actor与专家完全一致，但第0步arm/hand动作RMSE已为0.789460/0.530385；
q/target第0步即超过1mrad，物体到专家首次接触第51步才偏离超过1mm。
这条pilot主要是接触前命令拟合不足；chunk边界跳变量RMS0.180296，
小于chunk内部0.193631，未发现边界独有的尖峰证据。不能用这一个pilot归因正式模型。
截至本次检查，DP0updates、full320两组均未开始。

19:55:15第二条BC闭环的Kit在动作前SIGABRT(-6)，输出仅runtime_controls.json。
stderr只有std::system_error/Invalid argument，没有FileThread堆栈，因此上轮严格匹配未恢复。
保留stdout和对应原Kit日志，不能将这个缺栈崩溃唯一归因FileThread。
修复识别需要明确exit-6、精确错误签名、且没有任何场景/轨迹产物；仍每stage一次、总计三次。

调度修正：先完成BC/DP两个数值pilot，再做pilot闭环。基础设施通过之后，若学习阶段
仿真仅因上述动作前启动问题不可用，记录deferred并继续授权的train320；
任何已开始轨迹或真实控制一致性错误仍阻断，缺失闭环不算成功，正式test等待补齐。
不改网络/损失/优化器/初始化/物理/成功判据/截止；不重跑已完成更新和轨迹。
7项针对性回归通过，含截断日志必须有native退出证据、已开始轨迹不得延期放行。

## 2026-11-02 19:40 UTC+8：真实中断与恢复

17:19:29流水线在第8个Kit启动时停止，错误为OmniClient
`provider_file::FileThread`/`std::system_error: Invalid argument`，SIGABRT(-6)。
失败目录只有runtime_controls.json，无环境、轨迹或动作。已完成7/12登记专家：
fresh4/4、train3/3，全部合法且原生完整成功，分别168/153/206/161/145/171/181步。
尚余train1/validation4；新BC和DP均0updates、没有学生闭环结果。

这次不是GPU准入阻塞：中断前全卡最近60s12.12%、最近5min15.33%，
freeVRAM19.20GiB、availableRAM约30.67GiB，pause reasons为空。
不能用7/7替代完整12条基础设施验收，更不能据此宣称学生学会。

恢复仅改owner/资源报告/收尾，不改模型、控制、物理、数据、种子、预算和截止。
保存原session/status/失败log；已完成7条跳过。已知动作前FileThread abort允许每stage一次、
全session最多3次；任何started/result/轨迹文件都拒绝此自动恢复。
这不是FileThread内部根因已修复的证据，重复崩溃达到界限会继续保留并停止。
另修复报告丢失partial专家统计及PATH无ffmpeg导致收尾失败，改用已有PyAV完整解码。
5项针对性回归通过；专家视频实际168帧全部解码成功。原截止11月3日17:12:42UTC+8不变。

启动核验（2026-11-02 17:16 UTC+8）：`state-full-20261102-01.service` active，
root `state_full_20261102_01`已推进到独立专家第4条；新学生训练尚未开始。
代码 `c2e9895` 已push并核对origin同SHA。训练截止11月3日14:12:42，绝对截止17:12:42（UTC+8）。
自有进程识别通过；外部worker当时约8.33GiB显存，整卡free14.27GiB、availableRAM35.25GiB，
没有资源暂停。没有外部吞吐指标，不作零干扰承诺。后续阶段由普通脚本自动推进。

## 2026-11-02 最新任务：完整train320，新增独立窗口

最新附件147b678c授权新root `outputs/visuotactile/native_dp_v2/state_full_20261102_01`，
24小时绝对窗口含等待，前21小时训练、最后3小时收尾；旧root02和21:08UTC+8截止保留。
以下旧「正在运行」段落是历史快照：旧服务均已退出，BC/DP均0updates，最终阻断在初态查询。

已查明两个具体问题，并保存原证据：

1. `overlap_shape`读取USD GPrim位姿，而本机CPU PhysX/Fabric没有同步USD变换。
   fresh982501第一object box的USD中心是(0,0,0)，实际中心是
   (0.049823787,−0.028697385,0.619875848)m，旧查询因而碰到ground。
   新查询将原始两个box放到实际物体位姿，先验证USD/URDF半边长与局部偏移完全对应，
   再查询全部场景shape且逐box要求self-hit。没有忽略ground/hand、修改碰撞或推进物理时间。
2. 持久Kit先完成专家轨迹后，3/12重置初态的query返回手部overlap；其中fresh982506
   的全部保存diagnostics与screen逐字段相同，单独新Kit则无重叠、专家161步完整成功。
   这支持查询/场景状态受前序轨迹影响，尚未唯一定位PhysX内部缓存。新对照逐条新Kit，
   保持原reset和控制，用统一初始化与相同命令replay验证，代价是额外启动时间。

前置实测为12个登记初态、9条实际执行且9条成功、3条未执行；不能写成12/12。
另一个独立Kit诊断fresh982506为1/1成功。两组不混入新冻结的专家12分母。
小证据：`evidence/state_shared_20261102/collision_preflight.json`及两份preflight结果。

### 调度与预算

`state_resources.py`新opt-in策略不使用整卡45/50%准入或历史累计均值暂停。
单GPU worker，freeVRAM≥6GiB、availableRAM≥12GiB、disk≥12GB，缺测/内存不足才等待。
监控每秒记录整卡负载、进程PID/显存、自有CUDA峰值和工作/等待时间；最近1min/5min用于调速。
存在其他GPU进程时sleep/work=.25；最近1min>85%为1；1min>92%且5min>85%为2；自动恢复。
这些阈值只调节本任务节奏，不修改sim.dt，不宣称显存代表算力份额。
观察到另一项目FingerEyeLab的短时cuRobo/rollout worker，PID随任务更替；只读查看，无信号/配置修改。
没有稳定的外部吞吐接口，因此不能证明完全无干扰。原root02全卡max60s62.295%仍按旧规则记违规。

### 固定学习与评价

- 两种持续历史pilot各300updates、原train4/val4；检查有限loss、非零GRU梯度和下降。
  小样本闭环0成功不阻止full320。原模型、LossB、AdamW、warmup500、EMA.75不变。
- 完整BC320/val40里程碑1000/3000/6000/10000，完整DP320/val40里程碑
  1000/3000/6000/10000/19000；≥6000后连续两里程碑无1%验证改善则停止。
  每里程碑train2/val2自主闭环，最终best在fresh4/train4/val4闭环。
- 训练指标使用固定前16train作为低成本曲线并明确标注；优化覆盖全部train320，
  验证全部40。完整当前权重prefix重算+96tick TBPTT、batch两episode segment不变。
  数值cache避免每步HDF5读取，没有缓存旧权重hidden或读取RGB。
- 选择按validation40的prefix12 arm/hand等权命令MSE冻结EMA；BC/DP选择均冻结后，
  test40各一次离线；sorted first20 test IDs对expert/old/BC/DP闭环，非法初态原样报告，不替换。
  此test集曾用于历史实验，不能声称从未接触的全新外部测试集。
- checkpoint包含model/EMA/EMA计数/AdamW/全局与独立RNG/采样顺序cursor/源码及配置hash；
  完整更新边界每250保存，同里程碑恢复一次，已完成轨迹不重复。早停不是任务成功。

11项针对性测试通过，包括优化器+EMA+采样/RNG恢复逐位一致；既有81894 Actor验收复用，
不改变Actor/GRU/normalizer源码。新完整训练和闭环数值以新root结果为准，本段不预报成功。


本轮新root `outputs/visuotactile/native_dp_v2/state_history_20261102_02`，base `ec84bd3`。
这份文档区分已测事实和正在运行的阶段，不能用代码通过或离线拟合代替任务成功。

## 修改前

旧正式State19000updates：正式0/20；前轮fresh0/8。StateBC4000 saved train0/4、validation0/4，
全部601timeout、无lift/子目标；teacher-forced train arm/hand RMSE .02365/.03965，validation .98176/.71937。
短窗ActorBC也没有建立闭环能力。前轮single expert6/8、four8/8，但single6/8初始盒重叠，不能叫合法基准。
历史replay3/8和新same-condition2/2精确重现保持原分母；不覆盖旧结果。

## 新实验登记

绝对6h窗口包含资源等待；不自动重置截止。最多38条控制轨迹：expert12、same-command replay2、旧State8、
recurrentBC8、recurrentStateDP8。先固定合法初态，再观察结果。train4/val4身份不变，test不用于本轮学习/选择。
BC≤2000、DP≤1000；各在250及终点保存、按validation前12步命令MSE选诊断权重，再闭环。
离线best只是固定的小预算选模方法，不是闭环best或成功证据。没有自动扩大19000或ImageDP训练。

## 已测

- 92项新旧回归通过（8项新契约+84项既有；最后运行见verification.json）。
- 原400/旧numeric400/旧权重和native源保护hash核验通过。
- 81,894条实际Actor观测标准化误差0，静态group embedding完全相同。
- 8条真实teacher-forced controller/完整前缀/逐步history数值验收通过；细项见输入/历史文档。

## 正在执行，尚无任务结论

普通persistent owner已启动，依次执行初态screen、在线专家、同命令控制/随机流对照、旧State、BC、StateDP。
仿真闭环成功率、训练曲线、首个实质偏离和资源总结待实际产物；当前不能声称新学生成功或训练完成。
machine-readable `status.json`、各stage `result.json`为进度来源，不从服务active推断研究结果。

## 初始查询配置缺陷与保留的前置尝试

root01的headless配置`enable_scene_query_support=false`导致PhysX overlap查询返回空。
虽然OBB合法，空query不能支持实际shape无重叠结论。发现后停止自有owner，保留全部产物：
fresh982501专家168步成功，fresh982502已开始后被停止；二者均不纳入修正后的合法基准分母。
没有训练。root02仅打开read-only scene query，并用包围真实object的box查询作阳性对照；
原deadline继承不重置，总控制starts上限40=原38+两个失效前置尝试。候选顺序、原任务/成功/物理参数不变，
不利用前置结果选择种子。新的实际query可能改变几何合格集合，所有拒绝保留。

## 本次代码交付时的资源阻塞

修正后root02的preservation/parity已完成，freeze尚未执行初态screen，BC/DP均0更新。
整卡后台负载使资源监控保持RESOURCE_PAUSE；自有Kit停在资源等待，没有修改其他作业。
完整60s已经观测到>50%窗口，因此不能称全程资源合规，也不能将整卡利用率归因到本进程。
绝对截止仍为2026-11-02 21:08:22 UTC+8。owner会在满足原准入条件后自动继续，
到期则停止自有worker并保存阻断原因；没有扩大预算。
`state-history-20261102-finish.service`在owner退出后生成实测表/首差/曲线和完整视频decode结果，
保存至`evidence/state_history_20261102/results`，不自动提交未经审查的结果。
所以本轮当前交付是实现和数值验收，**真实合法专家基准、BC/DP学习及闭环目标尚未完成**。








## 自动收尾实测表（待人工复核与Git提交）

此表只读取普通owner结束后的产物，不新增训练或rollout。旧/前置失效样本不混入新分母。

|组|cohort|已尝试|合法|完整成功|native lift|至少1子目标|
|---|---|---:|---:|---:|---:|---:|
|bc|fresh|4|4|0|0|0|
|bc|train|4|4|0|0|0|
|bc|validation|4|4|0|0|0|
|bc_loop_1000|train|2|2|0|0|0|
|bc_loop_1000|validation|2|2|0|0|0|
|bc_loop_10000|train|2|2|0|0|0|
|bc_loop_10000|validation|2|2|0|0|0|
|bc_loop_3000|train|2|2|0|1|0|
|bc_loop_3000|validation|2|2|0|0|0|
|bc_loop_6000|train|2|2|0|0|0|
|bc_loop_6000|validation|2|2|0|0|0|
|bc_pilot_loop|train|2|2|0|0|0|
|bc_pilot_loop|validation|2|2|0|0|0|
|dp|fresh|4|4|0|0|0|
|dp|train|1|1|0|1|0|
|dp_loop_1000|train|2|2|0|0|0|
|dp_loop_1000|validation|2|2|0|0|0|
|dp_loop_10000|train|2|2|0|1|0|
|dp_loop_10000|validation|1|1|0|0|0|
|dp_loop_19000|train|2|2|0|1|0|
|dp_loop_19000|validation|2|2|0|0|0|
|dp_loop_3000|train|2|2|0|0|0|
|dp_loop_3000|validation|2|2|0|0|0|
|dp_loop_6000|train|2|2|0|0|0|
|dp_loop_6000|validation|2|2|0|0|0|
|dp_pilot_loop|train|2|2|0|0|0|
|dp_pilot_loop|validation|2|2|0|0|0|
|expert|fresh|4|4|4|2|4|
|expert|train|4|4|4|2|4|
|expert|validation|4|4|4|4|4|
|replay|fresh|2|2|2|0|2|

|组|updates|train arm/hand RMSE|validation arm/hand RMSE|
|---|---:|---|---|
|bc|1000|0.720295/0.581555|0.749026/0.573412|
|bc|3000|0.617310/0.514808|0.661225/0.525105|
|bc|6000|0.561306/0.479300|0.623675/0.498207|
|bc|10000|0.540785/0.468841|0.617104/0.491214|
|dp|1000|0.858414/0.639398|0.870098/0.647178|
|dp|3000|0.848297/0.611395|0.866523/0.629899|
|dp|6000|0.834543/0.583815|0.858557/0.606261|
|dp|10000|0.744861/0.519100|0.798099/0.559021|
|dp|19000|0.612155/0.436233|0.715060/0.500374|

最终owner状态：`{"state": "training_complete_evaluation_incomplete", "utc": 1793632912.7466462, "deferred": ["dp_deferred.json", "dp_loop_10000_deferred.json"], "formal_test_started": false, "reason": "Preserved startup failures need resolution; no success inferred"}`。

逐条reset误差、接触前后首差及阶段误差见公开`summary.json/trace_diagnostics`。
这些观测不唯一证明网络失败机制；同输入/同命令验收与完整成功分开报告。
