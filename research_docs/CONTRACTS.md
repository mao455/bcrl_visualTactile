# 当前训练与执行合同

2026-11-05新授权：仅收窄分布、冻结RL闭环新采集和一次性微型StateDP验收；采集完成后停止，不自动正式训练。原State143条探索评估继续暂停。当前阶段配置见[SLOW_EXPERT_RESULTS](SLOW_EXPERT_RESULTS.md)。历史320/40/40及旧权重续训只描述旧实验，不约束新数据数量，也不授权恢复旧任务。

本合同描述实际H24 position版本，不把旧H32/H64或原生command checkpoint当作兼容模型。旧实现见Git历史；此次整理没有改变已训练参数、监督、DDIM或底层控制。

## 模型与数据

- State：`PersistentState`，声明的Actor141、本体178、几何85及持续历史GRU；Image：`PersistentImage(I-HistoryDP)`，三视角448×252 RGB-D、冻结且固定revision的DINOv2-S/14及持续历史，不读特权物体几何或触觉。
- 历史原400维持320/40/40不变。新采集按真实初态家族先划分、再扩速度；数量动态、尾批不丢不复制，归一化只用新train家族。H24、每12个60Hz tick重规划并执行12、原物理120Hz。
- `position_rad_v1`、合同version3。29D canonical位置弧度，标签`transition/target_after_rad[t]`。观测t在动作之前，`obs.actuator_target_rad[t]=target_before[t]=target_after[t-1]`。padding只屏蔽无效损失，不能复制成有效监督。
- 训练GRU从真实episode开头用当前权重前向，原96tick TBPTT边界保持；每native tick更新，replan/chunk不清零。训练严格过去专家位置历史是teacher forcing，部署使用自身实际请求/应用反馈。current/future标签不是输入。
- DINO revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`；实际权重/缓存/统计SHA记录在data.json及checkpoint。RGB uint8 roundtrip、depth validity、相机时钟和固定batch规则保持。

## 动作与原控制

原生RL无量纲command先经过原delay，Arm增量/Hand绝对映射、EMA和限位。当前DP学习的是这个处理链之后的29D target。DP反归一化后仅按真实关节限位裁剪、canonical→Lab，写原目标缓冲，由原执行器施加。不能再次加delay/EMA，不能强制真实q/物体状态或用专家动作接管。

保留一个不参与目标计算的原delay随机抽样，使其后随机流调用顺序保持。原资产、碰撞、摩擦、增益、120/60Hz、reset、DR、外力、reward/success/timeout和研究sim wrapper均未改。

robot178中116:145为严格过去请求位置、145:174为过去应用位置，174:176为age，176为valid；初始清零。当前策略仍采用原proprio恢复路径；此前可选精度修复未暗中启用。

专家处理后target经此接口独立回放2/2完整成功（168/153ticks），14类控制/物理字段逐字节相同，提交/执行器目标误差0。该证据是接口验收，不是DP成功；历史四lane任意重建的PhysX内部状态仍不可精确恢复。

## 现有续训配置

原State/Image10000各采样133354 anchors，即24.04074 anchor等效epoch。全集5547 anchors，832旧片段。当前有效batch64个不同episode各一个anchor、microbatch4×累积16；episode deck与各episode的anchor deck无放回洗牌。它是episode均衡，不是全局anchor等概率，也不是64个统计独立样本。

base/FK/rotation/fingertip按全batch有效目标数归一化，temporal按全batch有效邻接对数。当前Loss B及已经train-only校准的辅助权重保持，不增加新损失。DDIM16、eta0、clipFalse，sample/x0；不能把epsilon loss绝对值与sample loss比较。

AdamW betas(.95,.999)、wd1e-6、EMA power.75沿用完整状态。新续训scheduler明确为3e-6→3e-5 warmup200，按新增23918 updates规划cosine到3e-6；不继承旧终点零LR，不重置optimizer/EMA。采样算法改变已有独立版本与审计，不是本次清理新增。

|累计anchor等效epoch|新增update|累计update|
|---:|---:|---:|
|50|2250|12250|
|100|6584|16584|
|150|10918|20918|
|200|15251|25251|
|300|23918|33918|

累计epoch=(旧133354+新实际抽样数)/5547；不把anchor曝光写成完整episode或96tick片段遍历。150后是否延长必须结合当前验证与闭环证据；旧queue的自动扩展规则受本轮明确review hold覆盖。Image仍需State实质进展及严格live Image输入配对通过。

## 评估与恢复

EMA，固定同数据、标准化和DDIM种子；端点10000/16584/20918统一train320与validation40，历史曲线train16明确标注。离线teacher-forced指标不能充当闭环成功。冻结validation seeds984101–984104；train拟合诊断982401–982404；不读test40、不改成功判据。

闭环每初态独立Kit；真实成功、native lift/subgoal/fall/timeout与软件缺失分别计数。稳定抓持无native离散定义，N/A。相同可观察初态、导出状态和事件RNG不等于证明PhysX所有隐藏状态完全相等。SDK fast exit即使code0，也不能称完整正常关闭。

一个owner、有限启动预算、持久日志和逐episode提交。恢复必须验证model、EMA、EMAstep、AdamW moments/step、Python/Torch/CUDA和采样RNG、sampler进度、学习率配置。冻结manifest可因审查后的纯组织迁移更新，但保留旧manifest和每项before/after SHA，不伪造旧源码匹配。

## 本轮随机评估补充

最新授权使用固定State150 checkpoint进行historical100（train320+validation40联合池均匀无放回抽样）及fresh100原reset分布。预注册seed20261105，历史主清单实际91train+9validation，后分层报告；test40排除。原生完整成功仍需两个插入子目标按顺序累计10个合格tick、最终7.5mm几何和撤手距离>0.1m。最终位姿/撤手在未完成前置阶段时仅作独立诊断，不增加原生成功分子。Kit门控针对完整运行、第一plan一致性、可靠落盘和资源回收，不要求重复三次SDK优雅退出。

## 独立慢速专家实验

只在expert_speed显式任务启用。A保持60Hz原生RL和一次delay/EMA/limit，在后处理目标上按Arm/Hand独立系数推进，并将执行目标写回原历史；原控制默认不变。B对已处理位置目标在60Hz重参数化，仅走position_rad_v1，不再次滤波。状态/图像只能来自真实重新仿真；新数据监督仍为target_after_rad。原生顺序成功、120Hz物理与阶段超时不变；新慢速数据与400原始数据独立，未通过数据合同前不续训。

## 新窄分布数据与随机初始化入口

选择只使用初态局部位置、物体—孔座相对位置、SO(3)旋转角距和手臂初态RMS；
不读取学生结果。保留原手指初态、初速、桌高、观测噪声、延迟、EMA和外力分布。
摩擦及物体尺度原配置为固定倍率1，本轮不新增随机化。原旧轨迹不作为新原速轨迹复用：
单lane事件流、历史可观察恢复与原四lane隐藏状态不等同，所有速度均重新闭环采集。
历史初态只有可观察字段恢复，不声称完整PhysX精确重放。

速度alpha每episode固定；一次native delay/EMA之后的目标推进只作用于专家采集。
StateDP标签仍是实际执行target_after_rad；学生position接口不会获得这个减速器。
完整三路448×252 RGB-D、五指120Hz力/FK、实际状态和命令保存；终止后自动reset图像
通过available mask排除。compact所有源字段逐字节hash验收，原400不写入。

StateDP仍使用PersistentState、sample/x0、Loss B及既定四个辅助权重；不重新标定loss。
新train统计沿原mean/std(clamp1e-4)和min/max affine算法；专家RMS与group embedding
保留冻结预处理。`anchor_continue fresh-smoke/fresh-train`复用当前模型、目标、采样和恢复工具。
正式新run从随机模型/全新AdamW、EMA、计数和schedule起步；只允许同一新run按配置hash恢复。
采集程序没有调用正式训练入口。
