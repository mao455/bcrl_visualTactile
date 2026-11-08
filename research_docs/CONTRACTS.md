# Tight-Insertion command StateDP 当前合同

## 2026-11-08 本轮覆盖：DP3风格条件诊断，等价路径复用

本轮起点`0caa908256da4863968208d9c040f7564a2a69af`；实际工作树与feature分支同下文，起始干净。
IDE原工作树停留`3e11d62`及其两份未提交历史文档原样保留。范围仅本项目Tight-Insertion。
独立配置[configs/tight_dp3_diagnostics.json](../configs/tight_dp3_diagnostics.json)，
`condition_mode=existing_robot_geometry_short_windows_v1`明确复用现有直达路径；不创建等价新模型。
本轮名称为“借鉴DP3条件编码的StateDP”，不声称完整视觉DP3复现。

参考DP3固定版本`47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97`的
[DP3Encoder](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py)与
[条件策略](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py)，只借鉴分支编码及条件化思路；未移植代码、点云或超参数，未增加第三方许可证负担。

### 实际输入字段与condition通路

源文件：`narrow_data.record`读取本批numeric缓存；`native_data.robot_rows/geometry`共用变换；
`state_history.PersistentState.encode`训练，`HistoryController.act → HistoryBank.observe`部署。
索引均为Python半开区间。时间t为当前60Hz动作前观察；Actor观测保留原生噪声/延迟，因此不等同无延迟真值。
State几何是本模型既有允许的模拟器State输入，不冒称真实视觉估计；没有新增特权字段。

|字段 / 切片|原始路径或来源|维度、单位、坐标|时间、归一化|实际网络分支|
|---|---|---|---|---|
|Actor q/qd/previous target `[0:87]`|`transition/expert_actor_input_diagnostic`，原`obs_utils.build_observations`|29归一化q、29 rad/s、29 rad；canonical Arm7+Hand22|原生延迟/噪声观察；前140维冻结专家RMS、clip5|当前Actor172→MLP128直接condition；另入GRU|
|Actor掌心位置/朝向 `[87:94]`，物体朝向`[94:98]`|同上；掌心含原PALM_CENTER_OFFSET|3m、4+4 quaternion **xyzw**；位置减env origin，姿态世界轴|同Actor预处理；不新增变换|同Actor直达及GRU|
|Actor指尖相对掌心`[98:113]`|同上；含原指腹offset|5×3 m，世界轴方向的位置差，非掌坐标旋转|同上|同上|
|Actor物体关键点相对掌心`[113:125]`、相对目标`[125:137]`|同上；4 object keypoints−掌心、4 object keypoints−对应goal keypoints|各12m；世界轴方向差；关键点携带既有形状/朝向信息|同上；保留原噪声/延迟，非专家隐藏态|同上；已明确表达物体—掌/目标关系|
|Actor尺度/group `[137:141]`|同上；140维观察+group ID|3无量纲尺度+1类别|RMS140；group匹配6类冻结32D embedding|合为172D→128D|
|robot q `[0:29]`、qd`[29:58]`|`transition/diagnostics_before/joint_position_rad,joint_velocity_rad_s`；部署`controller_proprio(canonical_joint_state_v1)`|29 rad、29 rad/s，canonical关节序|当前t；冻结train156 mean/std|当前及过去13帧直接robot_encoder→256；另native_encoder→128→GRU|
|robot执行器目标`[58:87]`及target−q`[87:116]`|`transition/target_before_rad == obs/actuator_target_rad`；`robot_rows`计算差|各29 rad；当前动作之前已生效目标|当前t；train156 mean/std|同robot两条路径；不读取当前标签target_after|
|robot过去requested/applied `[116:174]`、age/valid/elapsed`[174:178]`|`position_robot_history`；部署`observe_execution`反馈|29+29 rad，2s+1bool+1s|严格t−1及更早；首tick缺失置零，valid不标准化|同robot两条路径；训练明确expert teacher forcing|
|geometry object/goal/hole/table/palm `[0:35]`|`transition/diagnostics_before/*_pose_w`；部署相同`native_data.geometry`|5×(3m+4 **wxyz**)；共同robot base坐标、四元数w≥0|当前t；train156 mean/std|t−24,t−12,t窗口直接state_encoder→256；另当前geometry_encoder→128→GRU|
|geometry object/goal速度`[35:47]`|`*_velocity_w_m_s_rad_s`|各3m/s+3rad/s，旋转到base|当前t；同统计|同geometry双路径|
|geometry指尖`[47:82]`、尺寸`[82:85]`|`finger_position_w,finger_quat_wxyz`；既有object_dimensions URDF工具|5×3m+5×4wxyz，base坐标；3m尺寸|当前t；同统计|同geometry双路径；是link/body位姿，非触点|
|输出/监督|`transition/target_after_rad`|29D Arm7+Hand22**绝对关节目标rad**|当前及未来H24仅标签；train156仿射action_scale/offset|sample/x0 U-Net输出，DDIM16反归一化后原关节限位；仅执行前12|

geometry没有单独存储“物体相对掌/孔”的7D位姿；共有base中的成对位姿和Actor相对关键点已经直达condition。
不追加重复派生字段，也不把可推导关系说成显式相对四元数。
实际拼接为 `cat(robot_encoder[13×(178+age+valid)]256, state_encoder[3×(85+age+valid)]256,
actor_encoder128, persistent_GRU256)`，即896→现有MLP512→conditional U-Net `global_cond`。
当前帧在两个窗口末端，**不只经GRU压缩**；GRU仍每个native tick消费一次，仅真episode reset清空。
H24/commit12、5/60/120Hz、模型规模、sample/x0、损失/优化器、DDIM16/eta0、归一化均原样保留。

### 有界诊断与不训练依据

Core208/50家族、Aux15/11家族不动；只用既有Core train156统计及train8/validation30固定诊断身份。
旧8500更新/442012 anchor曝光、17次每500更新完整采样曲线直接复用；旧采样为episode-batch随机流，
新补充指标按部署batch1顺序、每episode seed7091、完整真实前缀和前12步执行窗口采样，不宣称两种噪声逐位相同。
新旧指标分别落盘，不能把旧曲线标作本轮训练结果。无test选择、无Aux加入。

已有等价结构且最近四次验证采样MSE改善0.9386%<1%、验证/train8差距35.52倍，旧有限patience4已触发。
按用户本轮条件分支，冻结**新增训练0更新/0小时**，不创建重复方案或续训；仅一次可丢弃CPU验收更新，不用于任何选模或部署。
后台最多**1小时评估总墙钟（含等待）**，固定38条离线诊断及旧27条闭环结果复用，新Kit/新初态启动0。
不重跑旧失败rollout，不扩100，不自动调参。预算耗尽写BUDGET_STOP，错误/NaN写FAILED，不补样本。

阶段直接复用`tools.narrow_repair_audit.stages`：无物体接触接近、低位接触、升高>3cm运输、原生对齐、第二插入、完成后回撤。
接触/对齐含transition事件，**仅事后分组和画图**，不进policy。保持与运动按相对anchor执行器目标的Arm/Hand RMS>.01rad分箱，
沿用现有评估的运动诊断尺度；不当成功阈值。零样本阶段标未测。末端/指尖误差复用GeometryObjective的URDF FK，报告mm/deg。
新目标叠图保存全部29关节数值，以预先固定canonical joint3与joint7示例；每次replan的变化量从真实anchor执行器目标起算。
闭环仅比较自身命令接口/实际关节tracking及事件，不能用不同进度专家轨迹的同tick目标作标签误差。

沿用整卡批次与完整60s≤50%，目标35%，准入45%；6GiB显存、14GiB RAM（比原12更保守）、60GiB磁盘储备。
复用现有普通资源监控及严格Lease；压力暂停自身chunk，固定deadline不延长。共享卡外部突发不能由采样准入硬保证，报告实际越界/缺测。
普通独立systemd脚本串行完成诊断、CSV/JSON→PNG、原位结果MD和终态；Restart=no、锁防重复、stdin关闭，Codex不驻守。
启动之前提交源码并校验acceptance/runtime source SHA；启动之后不修改对应代码。


2026-11-07最新用户授权：将全体逐条1.5–3倍改为Core/Auxiliary组成，完成两项确定性实现检查后，继续同一固定方案生产及Core优先训练。此授权取代此前65%严格准入率生产停机及路径进度机制提案，不追改历史结果。现有工程验收不重开；真实阶段通过后自动继续，启动不算完成。

执行结案（2026-11-08）：208Core＋15Aux冻结，Core-only fresh8500更新/100.21 anchor等效epoch；闭环train0/10、validation0/7、新初态0/10。预登记有限训练停止条件满足，但学生没有完整成功。本轮无待自动追加阶段；资源存在实测越界，详见[当前结果](STATE_DP_RESULTS.md)，不降低以下资源约束来改写历史。

## 固定实现

工作树`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`、分支`feature/tight-state-foundation`；本轮起点完整HEAD `e26e3d92e7daa39bc695a9bef3bce64b49ca7538`，起始工作区干净。正式运行使用本轮提交与SHA清单，不返回IDE原工作树。唯一配置[configs/tight_state.json](../configs/tight_state.json)，入口tools.tight_finish，不新建实验体系。

监督始终是专家实际处理后的`transition/target_after_rad`，canonical Arm7+Hand22绝对关节目标，单位rad，不是q_after或nominal。Actor141、robot178、geometry85、持续GRU256、原网络、sample/x0 Loss B及辅助权重、DDIM100训练噪声步/16推理步/eta0/不clip、overlap0保持。H24/commit12，5Hz规划、60Hz控制、120Hz物理。StateDP只读既定State输入，RGB-D/触觉用于准入，不进入学生训练。

不改专家权重、资产、PD、物理、原生DR/观测噪声/延迟/外力、任务成功或timeout。Arm=.50、Hand=1.00、Hand逐关节目标cap=2rad/s只在采集侧原生处理后施加一次；学生处理后目标不再次alpha/delay/EMA/cap。按原限位裁剪后canonical→Lab送实际执行器。

## 两项实现检查与物理时间

参考官方固定revision `5ba07ac6661db573af695b419a7947ecb704690f`的[Robomimic lowdim adapter](https://raw.githubusercontent.com/real-stanford/diffusion_policy/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/robomimic_replay_lowdim_dataset.py)和[lowdim policy](https://raw.githubusercontent.com/real-stanford/diffusion_policy/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_lowdim_policy.py)。其abs_action旋转转换针对末端位姿，不适用于本29关节标签。我们的动作仅作逐关节仿射归一化/反归一化；观察geometry的坐标/四元数变换不作用于动作。已知29D HDF标签→实际loss归一化→predict反归一化→canonical/Lab执行器目标往返测试需通过。

当前实现不是截断的64tick历史：GRU从本episode第0tick持续到当前tick；每次训练重算完整prefix，96tick TBPTT只界定梯度窗口。短窗口是13个native robot观察(a−12…a)和3个geometry观察(a−24,a−12,a)。旧native模型配置中的64是旧预测horizon，不是当前历史长度；不因此改模型或把动作从索引63开始切。

anchor a为12的倍数，obs[a]在a/60秒、动作前。未来24目标是y[a…a+23]，依次作用于[a/60,(a+24)/60]的24个控制区间；本次只提交前12目标，a+12时观察后重规划，从新plan索引0执行。首tick从0开始，不跳历史尾部；尾部越界label为0且mask=false，左侧缺失观察为0且valid=false，真实episode切换才reset。严格过去目标为y[a−1]，不是当前标签。人工tick轨迹覆盖episode首尾、63/64/65及replan边界。

上述实现若测试通过则保持不变，不新增模型实验。168项/完整510tick GPU FP32 normalized-input、hidden、condition及固定噪声动作一致性、microbatch/EMA/恢复、实际执行器/模态证据按源码SHA复用；仅改准入/队列则只运行受影响回归。新数据仍逐条验收并拟合train-only统计。

## 新版数据组成（新物理尝试前冻结）

Core目标约200条：完整成功、所有路径/真实运动/输入输出/模态条件通过，总时长比[1.5,3]。Auxiliary最多100条：完全相同的非时长要求，倍率只允许[1.35,1.5)或(3,3.30]，即原边界外最多10%相对偏差。选界依据是已有分布和保守10%上限，不跨过3.365等更远样本来提高准入。阈值在新采集前冻结，不事后挑选。

旧formal_05仍为20/20任务成功、10/20旧严格准入；新分类数值上10Core+2Aux。两条潜在Aux(formal_014/017)的RGB-D已按旧规则删除，不能复用成完整多模态集，不能补假图、复制帧或重采相同案例抹去历史。10条完整Core复用。14条开发Core及另外1条开发Aux候选保留独立开发身份，不混入独立validation/test；本次正式训练视图不混开发筛选案例。

Auxiliary全局≤Core/2，每家族最多2条且不超过本家族Core数量；按家族排序逐轮取身份排序条目，不按有利速度区间挑选或补采，记录上下侧实际分布。其余合格候选保留但不进入冻结组成。Core不足不能用Aux替代200目标；总约300是软目标。

所有准入轨迹必须同家族同配置原速参照、原生完整成功、post_success_actions=0，实际抓取/抬升运输/插入joint RMS降低至≤原速.98；另要求Hand全程及Hand/末端/物体插入实际均速≤原速.90，三者插入路径≤1.25。不是每关节或每子阶段都必须1.5–3倍。严重时长偏差、等待、绕路、任务失败、IO错配、State/三路RGB-D/触觉故障一律拒绝。

三路RGB-D逐帧真实内容/depth/mask/时钟/frame counter、触觉首tick无效后续有效/120Hz两子步、处理后目标实际执行一致性全通过。camera_valid全True不代替内容检查。每条保存时长比、阶段时长、Arm/Hand/末端/物体路径与速度分解、family、initial/event seed、split、原始成功/严格准入及新版分类。原速参照不进入训练，时间戳不改、帧不复制。

## 有限生产、冻结与训练

复用formal_05的55原速尝试与20慢速尝试，旧root只读。55家族41/7/7 split和初态范围保持：物体XY±2mm、相对XY±3mm、旋转半径1.5723103°、joint lerp.0075、qd±.03rad/s；完整范围以配置为准。保留formal_033原速动作前拒绝，不换家族；该家族缺成功参照的慢速任务预先记为依赖未启动，不计为任务失败或成功，不用删除ID声称全原分布可靠。

最多8个固定家族轮次，即含旧20在内440个慢速计划身份、另55个原速身份，总上限495。原先4轮改为8轮是本次明确有限生产预算；seed公式保持，已完成身份绝不重复。达到200Core且≥40家族/三split，在完整轮次边界停止；否则按有限队列或继承deadline停止并报告Core实际效率/缺口。Core冻结训练最低仍为150条/40家族/三split，不声称低于目标已经凑齐200。

5次连续真实任务失败停止；每20个慢速尝试检查合法执行累计成功率≥85%，非法初态/软件/传感器失败分别报告，后两者立即停。旧65%逐条严格产出率不再是新生产停机条件。新旧尝试/拒绝/依赖未启动全部留账，不换seed凑数。旧10/20结果及失败表不可重写。

采集截止仍为2026-11-08 09:08:21 Asia/Shanghai，结束前预留30min验收冻结；整体截止2026-11-09 03:08:21，不重置。独立root仅用于本次合同版本及不可变出处登记，仍使用原监督/恢复入口。到期不叫收敛。

冻结composition_manifest保存Core+按配额选出的Aux；训练manifest仅Core，按既有family split生成train-only归一化/statistics/cache。新源数据CPU一致性验收通过才训练；模型/optimizer/EMA/scheduler/RNG/计数全新，不载入旧学生。Auxiliary是否训练留待独立受控组成选择，本次不额外启动第二模型。

实际anchor预算：初始120、最低80、最多200 anchor等效epoch；每500更新验证，4次验证相对改善<1%视为平台，仍改善按40等效epoch完整状态续训。原episode-balanced AnchorSampler/effective64/micro4/短尾加权/EMA/完整恢复不变。时间或步数耗尽不能证明充分收敛。最终test不选模型。

闭环的train/validation家族必须实际存在于最终Core manifest，按字典序取既定最多10/7家族，不将原计划中零Core家族冒称训练家族；少于上限保留真实分母。闭环报告训练家族、validation及冻结后新初态；先10个新初态，出现真实非零成功才扩大到100，扩展前50有效例全失败早停；实际分母、非法初态、任务阶段和失败类型分别记。没有运行的训练/闭环不记0%。旧275混速State baseline的train1/4、新初态0/10合法+2拒绝保持独立，不当作新严格Core结果。

## 保护与交付

原400/275、专家/资产、当前必要成功模型、开发证据与失败分母只读保护。共享GPU整卡均值/完整60s≤50%，目标35%，6GiB VRAM/12GiB RAM/60GiB磁盘储备；普通systemd脚本负责监控/阶段推进，Restart=no，实际验完后才报告完成。不修改无关进程。

不新建重复MD；更新本合同、STATE_DP_RESULTS、简明README/SESSION_HANDOFF及CHANGELOG。已有清理7.510GiB与2.705GiB分别保留，不重复计算；新合格Core/Aux载荷不得按旧严格时长规则误删。不上传数据/checkpoint、不重写共享Git历史；审查后的代码/小证据普通提交当前origin feature。
