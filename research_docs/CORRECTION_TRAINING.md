# 独立残差 Correction v1（2026-10-30）

后续观看录像与B0/B1/A/B/C完整数据/参数统计见
[实验组对照报告](EXPERIMENT_GROUPS_AND_PARAMETERS_20261030.md)。观看重跑不改本报告训练、test或12次诊断分母。

本轮明确授权实际训练、最多12条诊断和feature分支发布。原400条数据、B0/B1 best、正式100条
及各0/50结果不变。原生环境与`sim/environment.py`只读。所有新产物在
`outputs/visuotactile/run/residual_20261030_01`，不发布原数组或权重。

## 实际架构与权限

| 组件 | 参数 / 冻结 | 输入与输出 |
| --- | --- | --- |
| 正式B0 v2 | 2,003,085；全部冻结 | 三448×252 RGB-D、58D本体、任务/外部goal；128D持续GRU；H12/DDIM8，每6步重规划；无触觉、无mixed history |
| 每个Residual v1 | 1,344,597总参数；489,125可训练、855,472冻结 | 独立复制B0已训练逐帧编码器并冻结；视觉128、触觉64、intent64、motion64 → MLP256/128/29 raw delta |
| Residual视觉GRU | 246,528可训练，独立复制自B0 | 冻结逐帧512D特征，逐60Hz更新；不是缓存随机网络或冻结全部Correction |
| 触觉分支 | 39,880可训练 | 五指world/local法向合力、encoder FK/旋转/速度、120Hz时间及因果差分；没有taxel、压力图、slip或partner身份 |
| intent / intent-memory | 18,240 / 24,960可训练 | 当前真实B0剩余chunk及mask，完整episode持续intent状态 |
| motion | 40,704可训练 | q/qd、实际t−1 command/valid、当前actuator target/tracking error |
| 输出MLP | 118,813可训练，末层零初始化 | `clip(base + gate * scales * tanh(raw), -1, 1)`；无接触逐位等于base |

旧`CorrectionPolicy`仍是完整candidate+替换/blend接口，新版不重新解释它的checkpoint。
新`residual_checkpoint_version=1`，输入`residual_frozen_frame_memory2_v1`，动作
`gated_additive_native_command_v1`；拒绝无有效更新、错误base hash、B1、v3 issued模式。
B0 SHA256为`fd1bd263de38ea6c6e0082d2ba55121adf3908b31c78b55d1cf39477ad6ff9d5`。

## 数据、参数化与学习目标

沿原专家完整观察序列，从step0持续更新冻结B0、私有噪声seed、真实重规划/cursor。
缓存逐帧特征、剩余plan/mask/base、生成step、time、原文件hash、来源，不把未来专家动作当plan。
专家t−1命令、target/tracking来自真实执行，但属于**teacher forcing**；部署改用学生真实ack，
存在不可忽略的状态/执行历史分布偏差。Correction实际反馈不回填B0。

原best固定320/40/40划分；先固定train前16和val前8完整episode。仅train16选幅度，之后固定：
arm和hand的contact `|expert-base|` p99均为2，因此两组scale均为2，不裁残差标签。
这是native命令幅度：arm是原velocity-style积分、hand是原绝对关节目标映射，**不是**统一米/秒。
scale=.1只覆盖arm19.20%、hand33.13%接触标签；对应contact RMSE可达下界.9064/.6421。
scale2接触误差下界为0的infimum；严格无接触旁路仍使全train16 RMSE下界.5467/.3453。
有限tanh不能精确达到端点，这些下界不代表可学到或安全保证。

监督为`command_clipped_pre_delay`，loss为contact时刻`.5*arm_MSE+.5*hand_MSE+1e-4*delta_L2`，
分组内取均值，防止22手关节仅因维度主导。无接触不更新，仍因果更新memory；无padding，尾chunk取实际长度。
完整episode持续memory，TBPTT32仅detach；CPU FP32、batch1 episode、AdamW1e-3、weight_decay1e-4、
grad norm1，无EMA/LR schedule。全划分重新从相同零头初始化B/C，最多5轮，不把pilot算成额外全数据epoch。

C与B结构、初始化、顺序和有效更新量匹配；gate/contact loss mask在真实未屏蔽数据上预先计算。
C仅在网络入口把world/local力置零，力差分、norm、contact/onset/release随之消失，FK/time保留。
**C仍使用真实触觉gate，不是严格无触觉对照。** B的validation选epoch，C取相同epoch，保持更新量可比。

## 已完成验收与阶段状态

- 全套80项通过：包括零残差、无接触exact-base、[-2,2]可达、独立参数/冻结、持续memory、完整/逐步等价、C力派生屏蔽、v3拒绝、未训练checkpoint拒绝、真实mixed反馈隔离与资源等待异常保存。
- 24条pilot缓存完整：train16共3226控制观察、1894个contact；val8共1758观察、1088个contact。
- 两轮pilot B/C各160有效更新、32次完整train episode遍历。val加权MSE从.727845降至B .623025、C .620782；通过学习信号门槛，已进入320/40全划分缓存。不能称触觉收益。
- 真实首条示范13步只读回放：step0/6/12采样，memory计数1/7/13；cache与online base/plan最大差1.91e-6，raw residual8.94e-8，过去真实专家命令逐位相同。没有为凑接触扩展此检查。
- 完整训练、一次held-out test和12个诊断分配均结束：11条自然超时、1条switch核验中止。
  下表是离线监督指标，不能当任务成功率；最终全套82项及真实产物核验通过。

## 完整离线结果

2026-10-30 06:16:18–06:33:12 UTC运行5完整epoch。每轮320条、64,825控制观察，
37,224个contact观察、1,575有效optimizer updates；最终B/C各7,875更新、1,600次完整episode访问，
共324,125观察、186,120 contact观察。两个模型初始化/数据顺序/有效更新相同，参数独立。
没有追加第6–10轮；使用首阶段固定5轮结果进入闭环。pilot各160更新是独立验证模型，不计入最终权重更新数。

B validation按epoch的加权command MSE为`.387972, .355340, .356157, .344325, .339453`；
C为`.381940, .357386, .347283, .342369, .340667`。B选择第5轮，C同轮匹配。
validation40有8,670观察/5,059 contact；test40只计算一次，有8,399观察/4,848 contact。

| 数据 / 模型 | 全程arm RMSE | 全程hand RMSE | contact arm RMSE | contact hand RMSE | 加权全程MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| val A（零残差） | .954809 | .659734 | .992612 | .737060 | .673454 |
| val B（真实力特征） | .682415 | .461754 | .470090 | .403417 | .339453 |
| val C（屏蔽力特征） | .679812 | .468177 | .463582 | .415910 | .340667 |
| test A（零残差） | .947055 | .650447 | .989277 | .721658 | .659997 |
| test B | .676828 | .463344 | .467370 | .399695 | .336392 |
| test C | .674974 | .469088 | .462700 | .411137 | .337817 |

单位均为native clipped pre-delay command。模型仅在contact时可改变动作；所有无接触旁路逐位一致。
train320事后描述统计（不重选scale）仍p99=2；严格旁路使全程arm/hand RMSE下界为.561776/.338660。
val输出饱和元素占比B50.16%、C50.59%。B与C差异很小且不是独立随机种子的重复实验，不能称触觉因果收益。

最终B checkpoint SHA256：`0ce3ce5c7df77abfb79c557528b2869cbef38321ad9d1d6047046d73036ada8b`；
C：`8750ebb9f2595ee0af96abcbc44a414a73035a3e06f24e99093c777cce3899be`。
训练时360条缓存索引SHA256为`14be455a5014f2c58ae64ac8b4c167e994e4be72dc62013434d8c735dc3549b8`，
同字节快照保存在`cache/train_validation_index.json`。之后仅追加test40，原360条缓存不改。
训练launch commit为`b88d685a5880aca0fe88b91dd38a857b179b2c61`，精确源码hash与完整配置保存在checkpoint及config.json。
B0参数全程无梯度、张量逐项不变；没有优化base或偷偷采用B1。

## 实际闭环结果与失败现场

固定12次启动全部使用，**不是12条全部成功完成的评估**：11条自然601步timeout，1条在专家
roll-in第120步、学生接管前校验中止。没有替换种子、重跑第8条或增加正式分母。
两个进程分别写`closed_loop`和`closed_loop_remaining_01`；原记录保持原样。
最终小型去敏聚合见[evidence/residual_20261030.json](evidence/residual_20261030.json)。

| 模式 / seed | A | B | C | 接触与任务进展 |
| --- | --- | --- | --- | --- |
| 自主981001 | 601步timeout | 同左 | 同左 | 三者gate均0；无首子目标/抬升；最近平均指端到物体中心约.115m |
| 自主981002 | 601步timeout | 同左 | 同左 | 三者gate均0；无首子目标/抬升；最近平均指端到物体中心约.0945m |
| roll-in981001 | step120交接；481学生步；gate33步 | step120校验中止，0学生步 | step120交接；481学生步；gate26步 | A/C均timeout，无首子目标/抬升；C明确非配对，不比较收益 |
| roll-in981002 | step88交接；513学生步；gate6步 | 同时交接；gate10步 | 同时交接；gate7步 | 专家prefix SHA及switch所有保存字段逐位相同；三者均timeout，无首子目标/抬升 |

自主A/B/C各0/2。局部自然终止结果A0/2、B0/1、C0/2，**不能把第8条软件/准入中止算B的任务失败**。
唯一完整严格匹配的局部A/B/C三联是981002；这些样本不能给出可靠成功率或触觉因果收益。
所有接触是五指aggregate normal force；没有partner身份，不能据此称抓住工件或识别slip。

第8条专家prefix完全一致，但旧比较的混合单位最大差4.085026>1e-5。旧错误路径没有保存
逐字段switch state或内存prefix数组；只保留error/status/log，具体差异来源未知。不是缺失填0。
接续前修复诊断保存路径，**只执行尚未启动的9–12**，不放宽阈值；不匹配仍完成自然诊断但标非配对。
第9条相对旧A的专家prefix不同、物体位置差.191mm、旋转差.007984rad、角速度距离2.4275rad/s；
raw goal quaternion差1.414主要是q/−q等价，物理姿态差约1.69e-7rad，不能解释为1.414m或大旋转。
原生日志中的kinematic velocity报错保留；未修改共享仿真栈，也未证明这些日志就是差异根因。

同seed不是逐位图像/策略重放。自主981001 A/B首命令已差.011999、全程最大差.096404，
而两者完全没有触发gate；这些差异不能归因于Correction。逐像素初始输入未保存，来源未完全定位。
981002局部switch物理字段与专家prefix精确一致，但B0私有预测仍分别相差约.00601/.00643，
说明物理匹配不等于视觉输入/模型计划逐位匹配；没有人为强写相同plan或memory。

### 执行指标：完整匹配的981002局部三联

| 指标（arm / hand） | A | B | C |
| --- | ---: | ---: | ---: |
| 实际纠正command RMSE（全学生段，含旁路零值） | 0 / 0 | .14500 / .10018 | .12885 / .08022 |
| actuator target跟踪RMSE（rad） | .00466 / .04746 | .00506 / .04897 | .00484 / .04806 |
| 专家→学生一次command跳变RMSE | 1.17140 / .76344 | .35956 / .20864 | .62432 / .20864 |
| 学生重规划边界command跳变RMSE（排除交接） | .85590 / .60824 | .85069 / .60574 | .84924 / .60181 |
| 学生段内command跳变RMSE | .63854 / .39824 | .63793 / .39913 | .64257 / .40001 |
| 学生输出饱和元素比例 | 6.82% | 9.19% | 8.21% |

B/C确实产生并执行非零残差，交接命令更接近此前专家命令，但并未转化为抬升或子目标。
A/B/C最后gate分别在native step96/99/97，之后严格旁路，不能把短暂接触称为持续局部控制成功。
以上command跳变不是物理关节速度/加速度；arm/hand仍是两种原生命令语义。

11条完整trace逐条验证：native ack、t−1真实feedback逐位一致，首次history无效、之后有效；
memory连续计数1..601，cursor循环0..5，generation step每6步更新，交接不清memory/plan。
无接触学生命令逐位等于当步B0，policy未改变CPU仿真RNG；事件时钟误差最大6.67e-12s。
原100正式NPZ与前一审查hash全部相同，原400 manifest及B0/B1 best不变；新诊断不写入旧0/50。

## 资源、预算、退出与可复现性

连续监测2026-10-30 **05:42:30–07:15:56 UTC**，共5,606.67秒（93.45分钟）；总训练/诊断窗口
上限21,600秒。首次GPU到已完成离线/test并启动闭环的保守上界3,217.10秒（53.62分钟）<90分钟；
后续闭环属于同一个6小时session，没有换root重置预算。

- 整卡时间加权均值**22.0602%**；5,414个完整60秒窗口最大**37.1971%**，均低于50%。
- 5,473个样本；缺测样本/时长均0；最大采样间隔1.26857秒。包含外部负载，不是进程归因或硬quota。
- 最低空闲显存11,875MiB；最低可用RAM26.227GiB。最终磁盘空闲约107.05GB（十进制）；
  每次工作前检查12GB余量，但没有连续磁盘最小值日志，不能把最终值冒充最低值。
- 原闭环与接续进程实测门控/节流总量741.858/455.406秒，其中主动pacing740.380/454.563秒；
  test缓存85.718秒，test门控.075秒；pilot/full CPU训练门控.097/3.587秒。
- 早期pilot/full缓存48.272/686.944秒包含实测准入加**请求**sleep，未分离调度超时；13步只读probe
  未持久化独立wait累计，保持未知。连续GPU遥测覆盖该区间，不能把wait归属缺口说成GPU缺测。

全部自有训练、缓存、两次诊断与monitor进程已核实退出；未向其他作业发送信号、改优先级或全局设置。
nice19、两计算线程、CPU22/23及有界chunk/队列贯穿本轮；CUDA/Kit辅助OS线程不等于只有两条OS线程。
原控制器、资产、120/60Hz、动作缩放、DR/reset/reward/timeout与研究wrapper相对fe9a634无diff。
82项测试通过；最后报告真实生成、400缓存/11完整trace hash、因果链及旧100 hash核对通过。
既有requests依赖警告未通过升级共享环境处理。完整配置、normalizer、base/derived/source hash与有效更新数
均随checkpoint保存；公开仅代码、MD和约75KB统计，没有权重、原数组、日志或视频。

## 结论与有边界的后续假设

**已实现并实测：** 独立监督残差训练、冻结B0、匹配力特征屏蔽对照、真实ack执行、持续memory及一次test。
**已观察：** 离线command误差显著降低；自主无接触，严格gate使残差无从改变接近；局部接触后真实残差
也没有推进任务。**未证明：** 触觉收益、成功率提高、压力去预加载、实机实时能力或原生重放差异根因。

若继续研究局部纠正，可建立明确版本的学生状态/真实反馈上的在线专家纠正或DAgger，减少teacher-forcing
分布偏差；或冻结base后另建SAC的critic、replay、native reward训练与交互协议。本轮没有这些组件，
**不是完成RL**。自主失败发生在接触前，应独立检验固定H12权重commit6/12等计划持续性假设；
本轮12个分配已用完，未执行此可选对照，不弱化旧loader、不改变严格gate或偷偷改动作/物理。
以上是后续候选，不自动启动新训练、补seed或扩展到50epoch/100条正式评估。

## 固定闭环与资源计划

预分配seed981001/981002，autonomous和expert_rollin各A/B/C，合计12条，不替换、不并入正式统计。
A明确base bypass；B训练残差；C匹配更新的force-masked残差。roll-in由released expert真实env.step，
首次连续3个有效观察的max finger force>.1N交接（最多前240步寻找）；各组从reset因果预热全部memory/plan，
比较原始初态、专家命令prefix SHA及switch状态，不能清memory、瞬移或提供未来专家动作。
无switch单列不可用；成功条件/原生timeout不变，保留完整自然终止。

独立`residual_session`进程连续采样ResourceGuard，覆盖worker导入、CUDA/Kit启动、等待、异常和退出。
完整60秒出现且batch/window均值≤45%才准入；4GiB显存、10GiB RAM、12GB磁盘；只睡自己的chunk/控制调用。
首次GPU计算起90分钟、整个session6小时不可换目录重置，计入等待。所有worker经run_cpu.sh，nice19、CPU22/23、两线程。
整卡采样包含外部负载，不能称硬quota或进程归因；不改变120/60Hz模拟时间。

等待记录限制：早期pilot/full缓存累计包括实测准入等待和请求的pacing sleep，未单独计调度超时；
13步causal检查未持久化独立wait总量，标为未知。后续worker独立保存实测准入/节流时间，
包括预算异常退出。连续整卡GPU遥测覆盖所有这些阶段；wait归属缺口不等同于GPU缺测。
