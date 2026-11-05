# 当前训练与执行合同

## 2026-11-06 有限条件化后续

原200成功/201尝试、60家族44/8/8已通过数据门控。原run训练/闭环完成且所有自有进程、monitor、owner锁退出后，独立新run才启动GPU。原截止2026-11-06 13:57:39 UTC+8及统计、全部数据、best/last只读。新预算从实际接手开始固定8h（所有阶段和资源等待计入）；原任务等待上限为原截止+15min。新run45GiB文件上限、60GiB磁盘余量，沿用6GiB/12GiB和shared_recent_v1目标40%。每worker仅一次启动；软件/合同失败不自动重试。

调度依据末4个相隔≥400更新的验证点、train8/validation误差、Arm/FK和早期方向/幅值/hold，以及train4/unseen12完整成功。>3%验证改善且Arm/FK改善、没有快速扩大的泛化差距时允许一次有界续训；平台且训练早期拟合良好、未见早期偏差和泛化差距明显时，只有数据/缓存/在线重算/执行目标审核全通过才允许一次覆盖扩采。约50%以上小诊断成功优先进入独立最终验证，避免为追逐百分比自动加数据。这些是研究费用调度规则，不是任务成功硬阈值或数据不足因果证明；证据不足则停止。

续训从原last完整恢复model、AdamW moments/step/param groups、EMA/step、采样deck与Torch/CUDA/noise RNG；原cosine日程不拉伸、不重新warmup，到达原终点后维持原3e-6 floor。按实测吞吐/原更新数分配最多约半程新增更新及1h；保留原best，validation没有改善就不替换。平台/过拟合提前结束，最多一次train4+新开发unseen8检查。

扩采只一次：同一窄域直接采样80候选，合法且不重复者中对原train44做几何max-min选择32新家族；尺度仅原train拟合，选择不看学生结果。几何选择完成后固定24train/4val/4test，再展开每家族原速+两个连续随机alpha∈[.58,.95]（其中一部分在[.58,.65]）真实专家闭环。最多96次完整传感器采集；目标296总量，时间不足可在≥260且覆盖达标停止；28个以上新家族成功和≥60条新成功为最终门控。所有旧200只引用、不复制覆盖，旧test8及其速度版本仍test；真实任务失败、准入失败和软件失败单列。

扩充后唯一新模型使用独立seed2026110601、随机参数/AdamW/EMA/调度器。H24、5Hz/60Hz、29D目标、网络、Loss B及辅助权重、DDIM和因果历史全部保持。train-only统计按合并数据重建。更新预算由原平台出现时anchor曝光、合并anchor数/平均batch和实测吞吐确定，不超过12000及2.5h；每500验证，达到足够曝光后的稳定平台/过拟合提前结束。仍只按validation选best。原26条validation与train8上的离线同口径比较另列，绝不读取test训练。

原unseen12已用于决策，明确为开发诊断。独立最终16个初态在后续prepare时冻结；若扩采则原200与新模型在同一计划上各train4+unseen16，报告可观察初态差异，不声称隐藏物理状态严格配对。整个新run最多52学生starts（含可选续训12和最终40），非法初态不按结果补种子。最终测试结果仅报告、不触发第二轮训练/扩采。ImageDP仍需严格live输入配对；本轮不自动训练Image或触觉。

2026-11-05最新授权：新独立窄域数据验收后，自动从随机参数训练StateDP，并执行train4诊断和12个预注册未见初态闭环。旧State143探索队列保持暂停，旧隔离root不恢复。旧400及其split不改，新数量由实际数据决定。

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

## 历史续训配置（本轮已替代，不再运行）

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

## 历史评估与恢复

EMA，固定同数据、标准化和DDIM种子；端点10000/16584/20918统一train320与validation40，历史曲线train16明确标注。离线teacher-forced指标不能充当闭环成功。冻结validation seeds984101–984104；train拟合诊断982401–982404；不读test40、不改成功判据。

闭环每初态独立Kit；真实成功、native lift/subgoal/fall/timeout与软件缺失分别计数。稳定抓持无native离散定义，N/A。相同可观察初态、导出状态和事件RNG不等于证明PhysX所有隐藏状态完全相等。SDK fast exit即使code0，也不能称完整正常关闭。

一个owner、有限启动预算、持久日志和逐episode提交。恢复必须验证model、EMA、EMAstep、AdamW moments/step、Python/Torch/CUDA和采样RNG、sampler进度、学习率配置。冻结manifest可因审查后的纯组织迁移更新，但保留旧manifest和每项before/after SHA，不伪造旧源码匹配。

## 历史随机评估补充（已暂停）

最新授权使用固定State150 checkpoint进行historical100（train320+validation40联合池均匀无放回抽样）及fresh100原reset分布。预注册seed20261105，历史主清单实际91train+9validation，后分层报告；test40排除。原生完整成功仍需两个插入子目标按顺序累计10个合格tick、最终7.5mm几何和撤手距离>0.1m。最终位姿/撤手在未完成前置阶段时仅作独立诊断，不增加原生成功分子。Kit门控针对完整运行、第一plan一致性、可靠落盘和资源回收，不要求重复三次SDK优雅退出。

## 历史独立慢速专家实验（已结束）

只在expert_speed显式任务启用。A保持60Hz原生RL和一次delay/EMA/limit，在后处理目标上按Arm/Hand独立系数推进，并将执行目标写回原历史；原控制默认不变。B对已处理位置目标在60Hz重参数化，仅走position_rad_v1，不再次滤波。状态/图像只能来自真实重新仿真；新数据监督仍为target_after_rad。原生顺序成功、120Hz物理与阶段超时不变；新慢速数据与400原始数据独立，未通过数据合同前不续训。

## 当前直接窄域与新随机StateDP

新root `narrow_state_20261105_02`：全新60家族目标，最多80个直接候选，至少50家族；先按初态准入并冻结split，再展开速度。8 validation、8 test，其余train。同家族所有速度同split。

物体XY围绕已存密集初态中心±25mm；孔−物体XY围绕同一中心±35mm，并直接与原孔范围求交。SO(3)旋转球半径20°；桌原点z∈[.376,.384]m，物体高于桌原点[.244,.256]m。原arm/hand随机joint-limit lerp由.1收窄至.03，速度由±.5至±.15rad/s。每家族采样配置传给原native reset写入，不修改原reset函数或物理。原延迟、噪声、外力和固定摩擦保持。

新准入`live_collision_mesh_bounds_v1`：实际CollisionAPI下所有网格顶点的保守包围盒，以实时body_link_pose变换，额外1mm边界；物体/桌/孔仍用原box SAT。近边界可能保守拒绝，不放过网格包围盒相交。旧scene-query在相同实时位姿下产生11个错误命中，新运行只把它作为诊断保存。不增加物理积分或预动作，不用视觉网格或USD世界位姿判定。

每episode冻结一个alpha，原native command/delay/EMA后做真实目标推进。两train原速/.58校准，唯一允许上调到.65；后续连续区间与原速约220次计划，目标200条成功，最低150且train≥32、val/test各≥6家族。保留所有失败、软件中断和部分文件，不按结果更换家族。RGB-D、触觉、状态、目标/实际运动和120/60Hz时钟全部真实重录。成功校准数据按同合同计入，旧3条仍隔离。

H24/commit12、PersistentState、sample/x0、DDIM16、Loss B及既定四辅助权重保持。新train重新拟合robot逐tick及geometry每12tick统计，State历史每60Hz更新，target min/max affine不变。随机参数、全新AdamW/EMA/调度器/计数，只允许同一新run恢复。CPU/GPU可丢弃smoke与正式更新分开。

普通owner自动执行数据验收→GPU一次性前后向/恢复→正式训练→最终闭环。训练按接受数据的anchor数预算100 anchor等效epoch、最多12000更新；每500更新验证并记录train8对照及首12tick方向/幅值/hold/FK，最优checkpoint只按validation选择。测试不进入训练或选权；最终train4与12个预先冻结未见初态分列，保留原生完整成功。60–70%仅为愿望，不是验收阈值或续跑理由。

新运行固定16h（等待计入），采集6h、训练最多9h并预留末1h评估；每阶段最多2次经审查启动，无自动retry。6GiB VRAM/12GiB RAM、shared_recent_v1目标40%、60GiB磁盘余量、75GiB新产物上限。只在实际数据完整验收后写DATA_READY；正式更新和成功率以机器状态为准。
