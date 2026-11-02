# Native DP 环境与学习归因修复（2026-11-02）

本轮按用户新授权拆分 G0，保留旧28条诊断和原400；旧协议的8/8历史开环成功不再是训练门槛。
本文件保留事前登记并追加实测结果；60/60主轨迹与两种BC各4000updates完成，但本轮没有取得学生原生完整任务成功。
所有自有worker/monitor于2026-11-02 12:01:30 UTC+8前退出，耗时1h17m55s；无自动后续训练。
公开小统计/曲线见[evidence索引](evidence/native_dp_repair_20261102/README.md)，本地视频见
`outputs/visuotactile/native_dp_v2/repair_20261102_01/README.md`。

## 已读证据与具体首差

已读旧28条的trace/restoration/result，8条回放重新与HDF5逐字段对齐；20条学生另读plans与expert_comparison。
机器证据在 `outputs/visuotactile/native_dp_v2/forensics_20261102_01`。
旧8条为3成功、4动作耗尽删失、1原生fall；没有native timeout。
command/delay index/delayed/target均逐值相同；外力/力矩通过原容差，但非零请求在不同姿态下有末位差。
所有8条q/qd第0控制tick已有首差，最小q约1.16e-10rad，最大初差1.19e-7rad；不是第一次接触才开始。
物体在tick0的XY或tick1的Z已有差；两条lane0的tick1 Z差为1.1920928955078125e-7m。
各条原/当前指端接触首次tick完全一致：24、85、120、95、24、65、79、54。
这只定位接触前状态积分差和随后接触放大，不能从中确认具体PhysX内部变量。

完整cfg递归比较仅4处差异：scene.num_envs4→1、seed931005→20261031、两条URDF绝对路径。
两条URDF内容hash相同；control/physics/DR/task/reset/reward/termination配置无变化。
实际原始质量/惯量/材料/COM未随旧400导出，不能把cfg相同写成实际PhysX属性已测相等。
来源世界坐标(±4,±4,0)与single0并列保存，源lane0/1/2/3都在本8条中。
早先T0同源world对照仍有tick1 Z差，单纯平移不是充分解释；新鲜seed17曾220步逐位复现。

## 不改成功判据的Gate

- G0-A：原400数据、train-only统计、原生命令数值审计全部通过。
- G0-B：预登记8个独立native reset（982101–982108），真实released expert闭环，至少6/8原生完整成功。
  6/8是本轮有界诊断的最低可用性证据，参考既有在线专家15/20；不能称总体稳定成功率已证明。
  所有8条必须软件完整、原生终止、同checkpoint与配置，失败不替换、不挑种子。阈值在新实验前登记。
- G0-C：历史重放的控制与物理差异独立保留；低开环成功率不自动否定示范或阻断小预算学习。
- G0合取A/B；任一失败不训练。原G1–G5的任务成功要求不由新门槛冒充通过。

## 事前固定实验与停止

新根 `outputs/visuotactile/native_dp_v2/repair_20261102_01`，从登记起8h包括资源等待；最多52主轨迹。
1. single在线expert8，保留每次native初态、env tensor/物理属性、141D Actor输入和完整控制事件。
2. 原4env/8m配置，982201和982202各4lane，共8；共享流/不同lane不是与single严格同初态配对。
3. single首两条新鲜轨迹，在新的Kit重放，不挑成功；另历史失败首两条做源世界坐标对照。
4. 旧State正式EMA在相同8个fresh seed上控制，逐字段检查初态，不能仅靠同seed宣称精确配对。
5. A/B通过后，StateBC与ActorBC各1000updates、相同4条train和4条validation诊断身份。
   相同两层256 Mish、H64/commit12/LossB/AdamW1e-4/warmup500/EMA、batch32与采样顺序；只改输入表示。
   State原2601D，Actor为13×(141+age+valid)=1859D；不是三组正式模型输入变更。
   Actor归一化仅原train320，标签仍原native pre-delay，141维最后一维为常量group50。
   可训练参数State1,208,896、Actor1,018,944，差异来自输入首层；这是隐藏层/输出匹配，不是完全等参数比较。
6. 两个BC各在8fresh初态自主执行；StateBC另train4/val4可见状态恢复，共8。
7. 真实结果审查后决定DP诊断；后台不会直接运行19000updates，不调用test40。

没有通过单步写历史物体轨迹、绕过target/native controller、改timeout或选择成功种子来获得验收。
四lane的只读object contact记录改为按真实num_envs检查N×1×1×3，未改物理/随机调用。
所有GPU阶段由独立monitor/lease约束，双50%上限、4GiB/10GiB硬余量、12GB磁盘余量。

## Actor与State输入审计

原Actor obs_list合计140D，再加group50常量得到记录141D；Actor不直接读取progress/success/reward，这些是critic字段。
q按原关节限位缩放；qd加σ=.1噪声；target为真实prev target；palm中心偏移(0,-.02,.16)，
tip pad偏移(.02,.002,0)，姿态xyzw，palm位置使用env原点而非robot base。
object pose先0–9tick延迟和1cm/5°噪声，再组装keypoints_rel_palm/goal；全部Actor观测再0–2tick延迟和±10裁剪。
State为当前无上述Actor噪声的q/qd/target、严格过去issued/delayed和elapsed；几何在robot base下用wxyz及符号规范。
State不直接提供pad/palm中心相对关键点，但固定变换和asset尺寸可推导；不提供Actor具体随机观测扰动及延迟队列。
因此不能把State真值与Actor随机输入视作逐值等价，也不能仅因85D几何就断言缺任务阶段标量。
同一真实State下，带噪/带迟的专家动作监督可能存在无法确定复原的条件方差；需ActorBC/StateBC对照实测，不能先当根因。

## 当前完成范围

原52条及补充8条已完成，两个BC各1000+3000updates；未启动新DP/视觉训练或test。
原生控制、原400、旧checkpoint、正式0/20全部保留。

## 在线专家与重放实际结果（先于学生训练）

| 对照 | 原生完整成功 | 具体结果 |
|---|---:|---|
| 当前single在线expert，982101–982108 |6/8|170、493、52fall、210、172、27fall、172、282步；失败进入分母|
| 四lane在线expert，982201/982202 |8/8|首batch195/211/135/162，次batch204/243/176/177步|
| 新鲜single首两条→新Kit同条件重放 |2/2|170/493步，command/delay/index/target/q/qd/子步q/object/wrench/stage比较误差全部0|
| 历史两条失败→源world对照 |1/2|e00_p000036成功180；e02_p000095仍动作耗尽214|

新single/4lane实际导出的robot/object/table/hole/goal质量、惯量、材料、COM、robot stiffness/damping逐项误差0。
两配置递归比较只有num_envs1/4差；seed是独立预登记reset，不宣称统计配对。6/8的Wilson95%约40.9–92.9%，
8/8约67.6–100%；当前只证明可靠任务能力的初步证据，不能据小样本证明环境数无影响或single一定劣于four。

源world两条在首次接触前q/qd逐位相等；首差分别从tick0移至tick85/tick120，和指端首次接触吻合。
然而object的tick1 Z仍差1.1920928955078125e-7m；e00_p000036源z=.64849853515625，重放=.6484986543655396；
e02_p000095源z=.6360113024711609，重放=.6360114216804504。
因此坐标平移是接触前robot差和至少一条结果变化的实证因素，但不是历史object首差的充分解释。
历史完整四lane内部状态缺失仍限制精确根因；新鲜同配置2条精确复现排除了“当前所有控制/物理接口普遍损坏”。

追加逐字节核验：fresh170/493步的command/delayed/index/target/q/qd/object/force/torque/contact、
两物理子步q/qd/object、subgoals/retract共15字段dtype与原始字节均一致，见本地`report/fresh_bytewise.json`。
这是两条已记录观测/事件的经验复现，仍不是PhysX全部内部状态可序列化的证明。

## 冻结Actor记忆干预（真实checkpoint，零更新）

`native_v2_actor_audit.py`直接用原网络构建器及全部checkpoint tensor，在8条保存Actor输入上递归推理。
输入140D以checkpoint running_mean/var、eps1e-5标准化再clip±5；group50先查32Dlearned embedding，
送入172D→LSTM1024hidden+1024cell→LayerNorm→ELU MLP[1024,1024,512,512]→29Dmu。
完整episode只在真正reset清零；子目标不得清零。固定网络与输入，比较完整/每步清零/每12步清零/滑窗13和25。

| 源后缀 | 完整历史arm RMSE | 仅最近13步arm/hand | 仅最近25步arm/hand |
|---|---:|---:|---:|
| e00_p000158 |6.21e-7|.49984/.23444|.16194/.06106|
| e00_p000036 |6.81e-7|.32866/.31608|.07317/.08330|
| e02_p000095 |5.99e-7|.28145/.23426|.13554/.07033|
| e01_p000097 |5.91e-7|.37635/.21901|.23386/.12613|
| e01_p000156 |4.81e-7|.47492/.24276|.15633/.13717|
| e00_p000055 |6.86e-7|.35517/.27092|.12637/.08451|
| e00_p000138 |6.06e-7|.40132/.20758|.17032/.09692|
| e03_p000070 |6.43e-7|.61943/.40978|.31470/.21393|

完整历史行使用全部tick；13/25滑窗在原5Hz replanning anchors测量。原网络的历史确实影响动作，
但这不证明不同结构无法从更少观测学会另一个成功策略。不能将输入变更和任务成功混为一谈。
原源Actor输入与持续历史能复现命令，也为专家checkpoint/记录标签身份提供独立数值检查。

## 真实部署特征边界核验

360条train/validation、6286anchors，逐tick真实StateController从保存packet构造输入，
过去动作明确teacher forcing，以源ack驱动ExecutionHistory；没有执行随机BC计划，没有optimizer。
2601维对训练缓存最大归一化误差5.1975e-5，最大项为target−q的手指分量；
源缓存读真实q，部署由限位归一化proprio逆变换，FP32往返不完全相等。该大小本身不是失败根因证据。
数据与逐特征.npy在 `deployment_audit_20261102_03`；两个早期CPU记录器缺ack字段错误尝试保留，不计闭环。

## 启动记录

原8小时端点固定。第一次启动未生成seed目录/HDF5；代码检查定位到reset之前读取camera metadata、
packet尚不存在的路径，但该次没有捕获Python traceback，不能冒称直接记录到了具体异常。
Kit关闭会掩盖Python异常退出码，因此owner以缺result正确阻断。
将metadata延后到reset之后，增加异常先写worker_error再关Kit；一次动作前恢复保留preaction_failure_1。
原deadline/已分配seed/轨迹上限不变，没有失败episode替换。此前资源等待中全卡窗口超50%的历史保留。

## 历史8条逐轨迹首差

下表tick从0起，是动作后60Hz状态；q/qd的分量是原canonical关节索引。
全部8条command、delay index、delayed command、target在共同前缀逐值一致。
接触为指端合力>0.1N，既不是所有碰撞的起点，也不能识别接触对象。
`eXX_pYYYYYY`均带原前缀`r931005_1793209376_`；eval始终lane0/origin(0,0,0)。

| 源后缀 | 源lane / world origin | q首差 tick:joint / rad | qd首差 tick:joint / rad/s | object首差 tick:轴 / m | 原/当前首次指端接触 | 结果 |
|---|---|---|---|---|---|---|
| e00_p000158 |0 / (4,-4,0)|0:4 / 3.72529e-9|0:0 / 3.72529e-8|1:z / 1.19209e-7|24/24|成功118，源119|
| e00_p000036 |0 / (4,-4,0)|0:8 / 1.86265e-9|0:0 / 2.98023e-8|1:z / 1.19209e-7|85/85|动作耗尽180|
| e02_p000095 |2 / (-4,-4,0)|0:8 / 9.31323e-10|0:0 / 7.45058e-9|0:x / 3.72529e-9|120/120|动作耗尽214|
| e01_p000097 |1 / (4,4,0)|0:0 / 1.19209e-7|0:0 / 3.42727e-7|0:y / 3.72529e-9|95/95|fall334，源509|
| e01_p000156 |1 / (4,4,0)|0:7 / 7.45058e-9|0:0 / 7.45058e-9|0:y / 3.72529e-9|24/24|成功136|
| e00_p000055 |0 / (4,-4,0)|0:7 / 7.45058e-9|0:0 / 4.47035e-8|0:y / 1.86265e-9|65/65|成功180|
| e00_p000138 |0 / (4,-4,0)|0:13 / 1.16415e-10|0:0 / 8.94070e-8|0:x / 1.86265e-9|79/79|动作耗尽216|
| e03_p000070 |3 / (-4,4,0)|0:1 / 1.19209e-7|0:0 / 9.68575e-8|0:y / 1.86265e-9|54/54|动作耗尽539|

q首次超过1mrad分别为48/91/126/107/35/74/94/65。全部q/qd首差早于指端接触；
下表列出外力与接触的首次非零差，`—`表示共同前缀全等，不代表该轨迹没有外力。
source-world物体值和原点0值均保存在每条JSON的`object_source_world`与`object`中。

| 源后缀 | force首差 tick / N | torque首差 tick / Nm | 指端contact首差 tick / N |
|---|---|---|---|
| e00_p000158 |—|—|24 / 2.18749e-4|
| e00_p000036 |—|—|85 / .00721097|
| e02_p000095 |—|—|120 / .00385809|
| e01_p000097 |271 / 2.38419e-7|—|95 / .00226846|
| e01_p000156 |62 / 2.98023e-8|—|24 / 9.87809e-7|
| e00_p000055 |—|136 / 1.21072e-8|65 / .00562930|
| e00_p000138 |153 / 3.57628e-7|156 / 7.45058e-8|79 / .00813437|
| e03_p000070 |152 / 8.34465e-7|457 / 2.60770e-8|54 / 2.29496e-7|

这回答的是8条抽样诊断，不是对400条逐一重放后的失败率。五条失败无法全部归给一个已证实原因：
源world干预实证恢复一条成功，另一个历史样本的object积分差在相同世界坐标下仍出现；
其余三条没有新增干预重复。缺少旧PhysX实际属性与内部状态，使“质量差异”或“solver缓存”仍为待区分假设。
已有新鲜同配置2/2零比较误差与此前seed17证据，不支持全盘否定数据或所有当前控制接口。

## 逐字段输入对照

Actor先整体clip±10，再用checkpoint RMS对前140维标准化并clip±5。
State机器人/几何标准化只使用train320的逐维均值和标准差；past-valid保留布尔，缺历史填0，age/valid另附。

| Actor字段（按实际obs_list顺序） | 维度 | 原Actor语义 | 当前State对应与差异 |
|---|---:|---|---|
| joint_pos |29|canonical q，限位线性映射[-1,1]，随后整体obs delay|robot0:29原始rad；live由proprio逆映射后train320标准化|
| joint_vel |29|canonical qd加σ=.1，再整体delay|robot29:58当前真实rad/s，无Actor噪声|
| prev_action_targets |29|真实_prev_targets，rad，再整体delay|robot58:87当前真实target，另87:116为target−q|
| palm_pos |3|palm body加中心offset(0,-.02,.16)，world−env origin|geometry28:35是base下palm body pose，中心可用固定offset推导|
| palm_rot |4|world xyzw，再整体delay|geometry31:35 base相对wxyz，w>=0符号规范|
| object_rot |4|world xyzw，object队列0–9tick及5°噪声，再整体delay|geometry3:7当前base相对wxyz，无同一随机扰动/队列|
| fingertip_pos_rel_palm |15|tip pad offset(.02,.002,0)，相对palm center的world方向向量|geometry47:62 tip body位置及62:82姿态，可推导pad向量；非直接同值输入|
| keypoints_rel_palm |12|带1cm/5°噪声的object关键点−palm center|object/palm pose和固定asset关键点可推导clean量，不能复原随机队列|
| keypoints_rel_goal |12|带噪object关键点−goal关键点|geometry0:14 object/goal pose，存在几何任务关系，无同一噪声历史|
| object_scales |3|物体scale乘DR scale multiplier，当前range(1,1)|geometry82:85为URDF包围盒尺寸，固定资产下相关但非同一数值定义|
| group ID |1|固定50→learned32D embedding|State无此字段；同任务常量不提供额外阶段信息|

State几何其余量为hole/table pose（14:28）、object/goal线/角速度（35:47）；robot116:174为
严格过去issued/delayed各29维、174:176为两种age、176 valid、177 elapsed。
State在13个60Hz机器人时刻及3个5Hz几何anchor上保留窗口；原Actor每60Hz更新persistent LSTM。
Actor整体obs delay0–2tick作用于所有140D；State记录当前观测，不增加人工延迟。
Actor无progress/successes/reward直接输入；目标pose随原任务变化且LSTM保存历史，不能把critic字段误作缺失Actor字段。
当前State包含clean任务几何，但“足以复现原专家决策信息”尚未成立：带噪队列及更早LSTM记忆未提供。
原Actor的历史干预已证明记忆依赖；是否必须用同一记忆结构才能学到成功策略，仍未证明。

## 旧State与小预算BC的实测学习状态

旧正式EMA（SHA256 `490dec2525504f7c7603a3d7e282b6989ecf040eaaf454a51e15f68548baea9e`）
在fresh982101–982108为0/8：601timeout、601timeout、54fall、546fall、289fall、27fall、601timeout、41fall。
其中982103/104/105有native lift，八条均0首子目标/0alignment。
与expert初始保存的所有可见diagnostics逐字段误差0；隐藏PhysX状态不作相等声明。
专家deterministic推理仍采样分布并消耗CPU RNG，学生推理不同，后续随机事件不能靠相同seed当成匹配。
实测八条step前CPU RNG均从tick0不同；delay index首次不同为0/0/2/0/0/0/1/0；
Actor观测首次不同为2/2/1/3/1/1/1/1。初始Actor观测一致，后续不同不能单独归给观测噪声或物理轨迹。
因此能确认“学生在专家具备任务能力的当前环境仍失败”，不能量化旧0/20中多少条由评估条件造成。

两种BC均4条train、4条val，各1000updates；无test、未改变三组正式输入。
State训练arm/hand RMSE从.95379/.92674降为.25960/.30907，验证为.95377/.69939；
Actor训练从.95299/.92541降为.31034/.34737，验证为.97048/.73296。
验证arm没有显示有效改善；Actor观测短窗口也没有解决这一小样本学习问题。
这只证明有限拟合，不能称收敛、充足训练或完整操作能力。

| source阶段 | State train arm/hand | State val arm/hand | Actor train arm/hand | Actor val arm/hand |
|---|---|---|---|---|
| 指端接触前 |.29570/.27837|.92457/.55727|.31056/.29345|.95478/.60853|
| 接触、lift前 |.24757/.37863|.98552/.76263|.28799/.40382|.98452/.79834|
| lift、首子目标前 |.27630/.30791|1.02083/.71764|.33722/.34327|1.04998/.69138|
| 首子目标后 |.22970/.28833|.93286/.72687|.31499/.35331|.94698/.76993|

上表为teacher-forced源状态上每次commit12误差，train/val分别884/1071条命令；retract观测0，故不虚构该阶段误差。
阶段按动作前原生标志和指端合力统计，优先级为after-subgoal > lift > contact > precontact。
最终真实自主结果、history-only影子干预和DP决定见下文；本表仅离线误差。

## 有据追加登记：BC拟合预算敏感性（闭环结果公布前固定）

StateBC fresh8已全部0成功；1000updates时train arm/hand=.25960/.30907，非EMA=.23830/.29154，
不能排除小预算欠拟合。history-only干预的117anchors/1339共同ticks上，固定student q/geometry，
仅替换strict-past issued/delayed/age，参考arm/hand RMSE从.95826/.76315变为.90929/.69616，
plan变化RMSE=.25114；CPU重建原student命令最大差5.96e-7。这个变化不是成功干预，
也不能把专家另一条轨迹同tick命令当作当前student状态的唯一正确动作。

基于“训练4条本身拟合仍不足”，在用户已授权的小预算学习诊断内，增加**一次**独立补充记录：
两种BC从原1000 checkpoint继续各最多3000updates（总4000，不是19000），同train4/val4、网络、
LossB、H64/commit12/60Hz；恢复AdamW状态，单独登记新的3000步warmup500/cosine学习率段，
EMA恢复原累计步数。不改变三组正式checkpoint或输入。
固定endpoint以train commit12 arm<=.05、hand<=.08作为“值得追加拟合初态闭环”的操作性门槛；
不是成功判据，val误差照报。若达标，StateBC仅增加同一train4/val4各一次自主原生终止，共最多8；
未达标不追加闭环、不延长。不得重选身份、循环延长、调用test或启动DP。
补充root独立，沿用本轮最初started/deadline，不重置8h；总主轨迹上限由52明确增为60。
必须等原52条owner/monitor退出，再由普通脚本接续资源监控和受控阶段；中间无GPU训练/Kit。
此项只改变训练预算以区分“1000步没拟合”与“拟合充分仍无法闭环”，不会同时变结构/目标/频率。

## 学生真实状态上的冻结专家旁路

读取每条student实际141D Actor观测，从真实reset连续递归原expert LSTM；不在模拟器执行旁路命令，
没有把源expert另一条轨迹同tick动作当作当前状态oracle。输出在`closed_loop_audit`，包含每步命令数组和阶段统计。

| 自主轨迹来源 | 原生完整成功 | 实际记录步数 | student对同观测冻结expert arm/hand RMSE | tick0 arm RMSE范围 |
|---|---:|---:|---|---|
| 旧正式State |0/8|2760|1.21752/.86337|.05620–.75207|
| StateBC1000 |0/8|3129|1.03407/.76554|.58239–1.15649|
| ActorBC1000 |0/8|2572|1.15900/.85230|.57846–1.11839|

tick0的Actor输入与对应expert真实控制初值相同、历史均空，student已经与expert明显不同，
所以“之后历史重放物理不一致”不是这些最初动作差的解释。后续旁路误差显示策略偏离，
但不证明expert若接管这些离分布student状态就一定能成功，也不能把命令RMSE替代成功。
旧State在precontact/contact/lift阶段arm分别1.24088/1.18242/1.22358；未进入首子目标后阶段。
StateBC1000的3129步中3079步为指端接触前，不能把它称为成功抓取策略。

## 源码和本地证据入口

| 内容 | 源码 | 本地相对 `outputs/visuotactile/native_dp_v2/` 的产物 |
|---|---|---|
| G0科学准入 | `visuotactile/native_v2_gates.py` | `repair_20261102_01/gates/G0-A.json`、`G0-B.json`、`G0-C.json` |
| 历史28条全部重读 | `native_v2_forensics.py` | `forensics_20261102_01/summary.json`、`replay_00..07.json` |
| fresh expert/4lane/重放/旧State | `native_v2_expert.py` | `repair_20261102_01/{expert1,expert4,fresh_replay,historical_world,old_state}` |
| 原Actor历史干预 | `native_v2_actor_audit.py` | `actor_audit_20261102_02/result.json`及逐轨迹npz |
| 真实State特征边界 | `native_v2_deployment_audit.py` | `deployment_audit_20261102_03/result.json`、`per_feature_max_error.npy` |
| 成对小预算BC | `native_v2_bc_compare.py` | `repair_20261102_01/bc_compare/{state,actor}/train.jsonl`与checkpoint |
| 纯历史替换 | `native_v2_history_audit.py` | `repair_20261102_01/history_only/result.json` |
| 同student状态expert旁路 | `native_v2_closed_loop_audit.py` | `repair_20261102_01/closed_loop_audit/result.json`及24份npz |
| 分阶段离线误差 | `native_v2_phase_metrics.py` | `repair_20261102_01/phase_metrics/result.json` |
| 有界owner/补充预算 | `native_v2_repair.py`、`native_v2_fit_extension.py` | 两root的`session.json/status.json/owner_exit.json` |
| 保护与小统计导出 | `native_v2_preservation.py`、`native_v2_repair_export.py` | `repair_20261102_01/preservation/result.json`，836checks全通过 |

source-world试验含原有`trace.npz/restoration.json/result.json`；fresh每条另有`lane_00.h5`、
`lane_00_trace.npz`、`initial_export.pt`、`initial_observable.json`、`metadata.json`、`result.json`。
所有大文件、原始HDF5、轨迹数组、checkpoint、视频保留本地，不加入公开Git。
代码基线为`f94b3c05abd7ba61fdcba7122a71894654215d99`；运行时含本轮未提交增量，
每root的source SHA逐文件登记，不能将基线hash冒充整个运行实现。最终交付commit另记录。

## 新鲜reset分布的额外审计与边界

本轮fresh seeds事先登记为原生reset，未按盒碰撞几何筛选；不能称为已验收合法初态cohort。
事后只读原URDF OBB检查：single8中6条object/table或hole正重叠，four8中3条；所有失败保留原分母。
single无重叠只有982104/982105，expert2/2、旧State/StateBC1000/ActorBC1000均0/2。
这仍为极小描述性子集，不是重新筛选后的正式结果；手部mesh和PhysX margin不在本盒检查范围。
982103物体最低z=.516737低于table上表面.539649，982106为.522984对.534390；
两条expert失败均有重叠，但其他重叠初态也成功，不能声称重叠就是两条fall的唯一原因。
StateBC已登记的历史train4/val4全部无上述box重叠（8/8）；仍受历史内部状态恢复缺口限制。
G0-B的6/8保留其“原生reset下有完整任务能力、可进入有界诊断”解释，不能升级为合法初态分布的可靠性验收。
大规模训练/正式性能验收前应独立预登记足够合法fresh起点并完整报告，不从本次事后子集挑成功种子。

## 补充阶段启动故障

4000updates两BC已完成；State train arm/hand=.023649/.039651、Actor=.029892/.050888，
State validation=.981758/.719374、Actor=1.001846/.764277；验证均未随拟合改善。
补充Kit第一次启动在`initializing`即SIGSEGV(-11)，栈为libomniclient provider_file::FileThread，
没有环境ready、trial目录、HDF5或模拟动作。`testSetWatchesEnabled(false)`已调用，不能声称该开关根治启动故障。
一次审查后动作前恢复保留原目录/log/session/status，固定原deadline、同8条身份、旧checkpoint，
不重复任何optimizer或已启动轨迹。若第二次失败则保留阻断，不循环重启；不修改系统inotify或共享安装。

## 对六个问题的当前证据回答

1. **为什么历史只重放3/8？** 这是8条样本，不是400条完整重放结论。五条失败为4动作耗尽+1fall；
   命令/index/target全等，物理首差早于指端接触。源world干预消除两条接触前q/qd差并恢复其中一条成功，
   已证实坐标处理会影响结果；tick1物体z差仍在。旧4lane完整内部状态缺失，剩余失败尚未唯一归因。
2. **新在线专家能否可靠成功？** 预登记原生reset下single6/8、four8/8；fresh重放2/2、15记录字段逐字节一致。
   环境具备完整任务能力。小样本置信区间宽且single6条有初始盒重叠，不能宣称合法初态分布可靠性已充分证明。
3. **旧State多少失败来自评估差异？** 无法给旧正式0/20精确分摊比例。fresh0/8，24组expert/student可见初态均匹配；
   其中无盒重叠的两个fresh起点expert2/2而旧State0/2。tick0即有动作偏离，支持模型问题存在；
   后续CPU RNG/index不严格配对以及历史恢复缺口阻止精确因果百分比。
4. **State输入能否复现Actor信息？** clean几何大部分可推导，但不含Actor具体噪声/延迟队列及persistent LSTM完整历史。
   同冻结网络改短历史显著改变动作，不能声称信息等价；并不证明当前几何原则上不能学到成功策略。
5. **简单BC能否学会操作？** 1000updates可有限拟合但State/Actor fresh各0/8，State历史train/val各0/4；
   4000updates将train RMSE降到State.02365/.03965、Actor.02989/.05089，但val未改善。
   4000endpoint独立train4/val4也各0/4、全601timeout，无lift/子目标；低误差没有转化为操作能力。
6. **StateDP是否需要从头训练？** 目前无证据支持立即原配置重跑19000步，也无标签/命令实现错误要求废弃旧权重。
   应先取得小预算State自主能力、明确历史/监督信息问题，再决定新输入合同或训练方案；
   当前大规模DP、视觉、test均保持关闭。这是证据不足的决定，不是证明DP永远不可学。

## 已实现的准入与证据保护修复

历史G0不再因3/8物理结果否定400。新训练必须同时重算A/B并核对原400HDF5、numeric400、
statistics/data索引、native action源码、released expert checkpoint与实际cohort/config；
只复制`acceptance_passed=true`文件不能绕过源证据。该加固在本轮optimizer结束后实测，
不追溯声称它是此前训练实际执行过的代码。此前训练有登记source hash，事后完整836项保护检查通过。
旧protocol2 run/resume在任何artifact写入/移动前拒绝，测试验证旧status/blocked字节不变。
新/旧学生输入、控制器、物理、成功判据、400与320/40/40划分没有被本修复更改。

4000步平均拟合还掩盖了reset处残差：四条train的第一条命令arm RMSE分别
.114865/.032373/.012245/.064414，hand=.112365/.104801/.071615/.112103；
首commit12 arm分别.065904/.042089/.018221/.073682。
所以即使全train平均arm=.02365，也不能说每个起点已精确模仿或把随后失败直接当纯泛化问题。
实际live初始plan与保存源输入影子比较必须双方clip[-1,1]，`plans.npz/student`保存原始未裁剪plan，
实际提交命令已clip。前三条live/source首plan最大差分别2.04e-5/4.76e-5/2.06e-5，
未发现足以解释大动作偏差的输入接线错误；初始监督残差与闭环历史漂移仍需分开分析。

可重复的CPU/报告入口（在V2 worktree执行；新审计输出必须新目录，不覆盖原始证据）：

```bash
bash visuotactile/run_cpu.sh -m pytest -q tests/test_native_v2_repair.py tests/test_native_v2.py tests/test_recorded_replay.py tests/test_native_dp.py tests/test_system_audit.py
bash visuotactile/run_cpu.sh -m visuotactile.native_v2_repair status --root outputs/visuotactile/native_dp_v2/repair_20261102_01
bash visuotactile/run_cpu.sh -m visuotactile.native_v2_repair_report --root outputs/visuotactile/native_dp_v2/repair_20261102_01
```

不会自动重启已完成原52条或补充cohort。完整实验配置在公开`configuration.json`及`fit_extension.json`，
原Actor定义`isaacsimenvs/tasks/play/play_env_cfg.py:126`/`utils/obs_utils.py:230`，
State allowlist在`visuotactile/native_data.py:40`及`:83`，实际部署在`native_controller.py:28`。

未执行的下一步必须有独立设计：在不改reset数学的前提下预登记合法初态采样流程与全部拒绝日志，
做专家和学生的当前环境对照；只改一项历史表示或训练条件，先验证reset/接近阶段及少量自主成功。
原专家LSTM短历史干预支持检验持续历史，但不支持直接把专家hidden state当学生部署输入。
本轮不据已见test/validation反复调参，不引入新任务阶段真值，不将建议自动转成新训练或新轨迹。

## 最终执行、资源与交付

| 实验 | 完整成功 | 原生结果 |
|---|---:|---|
| single fresh expert |6/8|6完整成功、2fall；原生reset含6条box重叠，不称合法分布验收|
| four-lane fresh expert |8/8|8完整成功；含3条box重叠|
| fresh replay |2/2|170/493步，15记录字段逐字节一致|
| historical source-world |1/2|180成功、214动作耗尽；原3/8证据不改写|
| 旧State fresh |0/8|3timeout、5fall；3lift、0子目标|
| StateBC1000 fresh |0/8|5timeout、3fall；1lift、0子目标|
| ActorBC1000 fresh |0/8|4timeout、4fall；1lift、0子目标|
| StateBC1000 saved train/val |0/4、0/4|全601timeout、无lift/子目标|
| StateBC4000 saved train/val |0/4、0/4|全601timeout、无lift/子目标|

ActorBC4000只做匹配预算离线对照，没有新闭环分母；不能把1000闭环结果贴给4000权重。
4000 State checkpoint SHA256=`91a57d4c6a19d15b8736cc146dbb1696d9e7e31c984bef81e17378352a343fdd`，
Actor=`51b3e27a2bed9e4610e39d0627e283c1ab2657da16151b4ddc861a35fdf30d74`。
8条4000初始clip后plan对源输入影子最大误差6.00964e-5；数据接线验收不证明成功。

两root合并保留全卡监控历史：有测区间加权均值28.37849%，最大完整60s均值54.62724%；
初段确有超50%窗口，不能因平均值较低而称双上限合规。
缺失采样计数0，但两次恢复停监控间隔合计438.47688s，最大间隔250.93948s；
加上登记至首个监控样本约13.015s无采样，不将空档视作0%利用率。
实测最低free VRAM11.50977GiB、available RAM24.32213GiB；未调整其他任务或系统配置。
监控中的wait_seconds=0不是worker没有等待；worker独立work.jsonl记录admission/pacing。
原52条与supplement全部owner/Kit/monitor退出，三个systemd unit均inactive/dead；清理删除0。

原400HDF5、numeric400、旧三组best/last/results/test/evaluation及原生任务/wrapper共836hash检查通过。
84项CPU/真实产物回归通过（8.60s），8份MP4逐帧解码，全部60fps；fresh首视频高位栏缺少有效画面，
侧/腕两视可用且State/Actor未读取图像。无原始数据/权重/视频上传，仅代码、文档、小JSON和4张曲线PNG。
本轮新增代码是在基线f94b3c0之上运行，runtime source SHA与最终源码中事后准入加固的差异明确保留；
最终commit由当前分支Git历史定位，不能把某一hash未经核对称为所有阶段实际运行源码。
