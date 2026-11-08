# Tight-Insertion command StateDP 当前合同

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
