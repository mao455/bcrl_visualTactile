# 持续历史、动作来源与时间合约

## 2026-10-30：Residual v1的独立执行反馈

B0固定command v2/none；Correction v1独立保存实际native ack并在下一观察核对episode/step/time。
训练的previous command/target/tracking来自过去专家执行，显式teacher forcing；部署来自实际学生mixed命令，
或标明的released-expert roll-in。B0自己的plan/cursor/RNG不接受这条反馈，v3 issued不接入此链。
cache1沿完整step0前缀生成真实B0计划，保存生成step/cursor/time/source/base hash；未来专家命令从未成为计划。
每60Hz更新四路memory、两120Hz触觉样本；TBPTT32只detach，episode ID才reset，roll-in交接也不清状态。
详情与真实回放误差见[Correction训练报告](CORRECTION_TRAINING.md)。

## 最新2026-10-30：真实已执行命令历史

新增command checkpoint3/history1，显式`--action-history-mode issued`；默认none继续复现旧B0/B1。
obs[t]额外读29D previous_issued_command、29D previous_delayed_command、两项float64模拟秒时间与bool有效性。
两个时间年龄乘60成为native ticks；61维输入经32维MLP进入context与持续GRU。t0无效，真实episode reset清理。
离线使用专家已经执行的transition[t−1]，明确teacher forcing；部署通过实际env.step后ack构造学生自身历史。
两者时序/字段定义一致，但轨迹分布不相同；本轮没有生成虚假学生未来计划或改旧权重flag。
已有q/qd历史不等于命令历史；target/error与混合命令未放入B0/B1或未来nominal allowlist。
7类输入权限、因果测试、真实24步前缀拟合及限制见[独立审查](INDEPENDENT_AUDIT_20261030.md)。
下面未训练/v4尚未完成等文字为历史；400条和旧50epoch/50组评估均已完成，本轮仅有界诊断。

> **2026-10-29更新：** v4三相机400条已完成；B0/B1现已获正式命令训练授权。
> `train_dp`以B=2/T=32完整有序前缀训练/验证，各行visual/tactile/fusion状态独立，
> padding不更新state，组边界保存游标后可恢复；不在episode中间用零状态继续。
> 本轮两组关闭自身计划且在线一致。下文关于未训练、单episode训练、尚无memory2验证的
> 文字是以前阶段；当前证据见[BASELINE_TRAINING.md](BASELINE_TRAINING.md)。

本文件对应 `memory_version=2` 与 `trace_version=2`。当前采集入口的
`perception_version/schema_version=2` 使用三路 RGB-D（448×252，固定高位、
固定侧视、link7 腕部）；已提交的三条生产记录仍是独立的 v3
`wrist_bracket` 构图，v4 `wrist_bracket_close` 尚未通过两条成功门槛。
历史双相机表格明确标为旧基线，不能与当前布局混读。用户方向见
[USER_RESEARCH_DIRECTION.md](USER_RESEARCH_DIRECTION.md)。本轮没有参数训练。

## 从真实源码发现的问题与实现

此前 Nominal 用4帧、stride3，每次重规划重新对窗口跑零初始 GRU；Correction 也对4帧视觉、12步运动与24个力样本重置 GRU。窗口虽未跨 episode，却没有持续的 episode 记忆。文档中的“完整轨迹”只指保存，不能等同网络使用完整历史。

现在 `NominalController.act` 每次收到新控制观测，先构造**自己**上一动作和旧计划，编码当前双视图/关节/目标/计划，更新128D状态。每6步才用当前状态调用 diffusion；这六步中每步均更新记忆。无须反复编码整段原始图像。4帧窗口仍在，用于旧模型兼容和诊断，不再是 memory2 的记忆上限。

Correction 分别持有视觉128D、motion64D、intent64D、五指各32D循环状态。五指融合后64D，不把融合后的维度解释为某个固定手指或物理量。当前预测序列内部的 intent GRU 每次从头总结这条计划，随后另一个跨控制步 GRU 更新计划历史；前者的重置没有清空 episode 历史。触觉状态另外保留上一个真实物理样本及时间/有效性，保证跨块的差分、接触建立/消失连续，上一帧不被重复送入循环网络。

`reset_state` 只清空 episode ID 改变的 batch 行，包括之前物理样本有效性。普通 task 子目标、重新规划、训练块边界均不清空。各 env 独立，`OnlineHistory` 拒绝丢步/重复步。固定128/64D压缩状态可能遗忘细节、衰减长时信息；这提供持续通路，**不保证所有历史均被记住或可恢复**。完整原图与数值另外保存。

## 与训练一致的读取

- `EpisodeSequenceDataset.chunks` 从真实 step0开始，包含每个控制观察。`encode_episode_chunks` 携带状态，拒绝缺前缀或乱序；块内返回 `(B,T,128)`，监督为当前起的 `(T,12,29)`，尾部mask。可调chunk size不改变模拟频率或记忆长度。
- `detach_between_chunks=True` 截断梯度，数值记忆保留。默认未来训练入口按episode顺序TBPTT，一次一条episode；本轮没有执行优化器。权重在块间变化时，携带的是旧权重生成的状态，这是TBPTT近似，不等于用最新权重重算全前缀。
- `CorrectionSequenceDataset` 每步只给一个新视觉/运动观察和两个新的物理样本，计划来自记录中真正计算的Nominal。`replay_correction_chunks` 顺序携带各分支状态；当前专家标签另置 `supervision`，没有实现Correction损失。
- 旧 `EpisodeDataset/CorrectionEpisodeDataset` 是明确的局部窗口接口，供memory1/诊断使用。memory2的 `loss` 不接受默认随机短窗口零初始训练，需使用顺序条件的 `loss_from_condition`。旧checkpoint缺少memory_version按1解释，不自动“升级”为持续历史。
- 保存恢复包含权重/配置/optimizer/RNG，但未来训练恢复会**从episode起点重放**，不声称恢复到了中断的块或在线隐藏状态。本轮检验固定参数保存重载和梯度；未验证长期优化效果。memory2验证损失尚未在CLI报告中实现，值为null，不伪造旧窗口验证指标。

## 动作来源：已执行的过去与未执行的预测不同

| 数据 | 何时真实可知 | 用途/限制 |
| --- | --- | --- |
| 专家 `command[t-1]` | 前一控制步已提交后 | 可作为明确的专家历史/teacher-forcing；Correction可作为真实执行历史；不能冒充模型预测 |
| 专家 `command[t:t+H]` | t时当前专家可以另行被调用；后续动作尚未知 | BC监督，禁止作为t决策输入；不因轨迹成功而例外 |
| `previous_nominal_action` | Nominal在t−1实际输出的当前项 | Nominal自己的预测上下文，可能没有执行；由controller私有状态产生 |
| `prior_plan` | 某次过去Nominal重规划实际生成 | 在当前推理之前取仍覆盖当前/未来的部分；尾部无效。不能用后来专家序列填充 |
| 本次Nominal plan/current action | 当前观测处理并推理后 | Correction输入、候选执行来源；记录模型hash/配置、生成步/时间、覆盖区间和执行选择 |
| Correction candidate | 当前独立策略前向后 | Shadow诊断，不执行，不作“真实专家”标签 |
| `submitted_action` | 最终选择控制来源后 | 进入env.step的29D命令；并非网络clip之前的原始均值 |
| `command_clipped_pre_delay` | 入口裁剪后 | 真正已提交的归一化命令历史；BC目前学习这个字段 |
| `control_trace/delayed_command` | 原随机队列实际抽样后 | 这步真正用于目标计算的命令；结合source step/lag追溯以前命令 |
| `target_before/after_rad` | 原控制管线运行前/后 | 实际控制器目标；与实际q不同，可能包含预加载 |
| post-state / contact / success | 本步物理执行后 | 结果/监督/诊断；不能进入同一步当前推理 |

默认部署Nominal不接受专家命令或Correction修正命令。顺序Dataset额外返回独立的 `expert_history.previous_issued_command/valid/source`，使将来的明确teacher-forcing方案有真实数据，不把它强行加入当前allowlist。教师历史可能含预加载；部署自主预测、教师执行状态、Correction执行状态有分布差异。旧数据中自己计划缺失时mask=false是“没有真实预测”，并不是“专家过去动作不能记录”。正式利用计划条件应来自明确模型的旁路/因果回放并考虑训练—部署分布差异。

本轮真实采集是专家执行、未训练Nominal/Correction旁路。结果属于专家选中命令经延迟处理后的执行，不能归因于旁路计划。`plan_trace.selected_for_execution` 只说明当前计划项是否提交，不承诺整段12步计划都执行；仍须查看delay和目标。模型hash识别参数；当前checkpoint有路径/配置，原专家另外记录文件SHA256，最新capture也记录Correction参数hash/config。`expert_actor_input_diagnostic`是确实给player的输入（内部归一化之前，含privileged信息和SAPG身份），不在学生allowlist。player返回的是clipped command，clip之前的均值仍缺失。

## 频率和公式

| 层 | 当前模拟频率/跨度 | 实际更新 |
| --- | --- | --- |
| physics / scene update | 120Hz，dt=1/120s | 原物理、隐式执行器；同一控制目标施加两次 |
| 原生动作管线 | 60Hz，dt=1/60s | 每步新命令、延迟抽样、积分/绝对映射和平滑 |
| 双相机 / proprio / 控制目标反馈 | 60Hz | 每控制观察新帧；图像时间是控制观察可得时刻，非硬件曝光时间 |
| ContactSensor / FK | 120Hz | 一包两个有时间戳的物理样本；reset初包无效 |
| Nominal视觉/状态/旧计划记忆 | 60Hz | 每控制步一次，无额外下采样 |
| Nominal diffusion重新规划 | 每6步，10Hz | horizon12个原生动作，覆盖0.2s；执行其中6步后更新计划 |
| Correction视觉/运动/计划历史与候选调用 | 60Hz | Shadow每步调用；触觉GRU每次消化两个新样本，相当于120Hz |

10Hz指**重新规划频率**，不指机器人每0.1s只执行一个动作。只改重新规划周期、继续逐步执行60Hz原生命令时，不乘6或除6。若将来改实际执行dt，arm积分、hand平滑、delay时间、horizon、差分和归一化都须定义新控制合约并对照验证，不能统一缩放全部29维。

Lab顺序前7关节是arm，剩余22是hand。设p为上一目标，a为**实际延迟后命令**：

`arm_raw=clip(p+1.5*(1/60)*a, limits)`；`arm_target=clip(.1*arm_raw+.9*p,limits)`。

`hand_raw=lower+.5*(a+1)*(upper-lower)`；`hand_target=clip(.1*hand_raw+.9*p,limits)`。

原队列默认长度3，抽index0..2。首步各槽填当前命令，故index2不一定表示实际2步旧命令；另外记录selected source step与effective lag。新增原action helper中的可选只读hook取得**实际选择结果**，不重抽随机数、不重实现控制数学。已知正弦命令数值测试比较hook有无的目标和RNG。

模拟同步执行会等待网络，不等于满足实时截止期。此次2env CPU双线程完整旁路中位约100ms/控制步，而60Hz预算16.67ms；memory编码约44ms、Correction约49ms。详见证据，不宣称实时10/60Hz。

## 触觉和视觉语义审计

五指身份顺序 index/middle/ring/thumb/pinky。力来自ContactSensor整个末节刚体的normal contact合力，世界系N；没有接触对、触点、切向力、力矩、压力图或可靠滑移。`contact`为模长>.1N的工程阈值。onset/release仅在前后样本均有效时比较，reset后的第一有效样本不猜测接触建立时刻。

FK实际输入是 `robot.data.joint_pos`，Lab名字映射的**弧度**，不是归一化proprio前29维或命令目标。模型是原URDF含固定关节。保存每个物理样本canonical q用于重算审计；位置是robot_base中的末节link原点（不是接触点），姿态 `R_base_link`。当前固定基座world方向identity且启动时断言；非identity明确拒绝。一般情况下应先 `F_base=R_world_baseᵀF_world` 再 `F_link=R_base_linkᵀF_base`，本机第一项恰为identity；不能忽略将来基座旋转。

每个scene.update之后同时采样q→FK，Sensor在同物理更新刷新history。120Hz时间戳与原始q一起记录。link速度为 `(p[t]-p[t-1])/实际dt`，单位m/s，固定base系；局部力变化为 `F_link[t]-F_link[t-1]`，单位N，**没有除dt，不是N/s变化率**。局部轴旋转也会导致变化，不可直接解释为世界力变化或滑移。dt和pair_valid本身是特征。Correction输入的 `target−q` 是当前已施加目标与当前实际编码器的误差，t时只能说明此前执行，不包含t尚未执行的后果。

视觉的两相机RGB、米制Depth、depth_valid和camera_valid均真实进入独立分支。16个空间softmax位置/方差/加权深度以及3×5网格保留可学习的空间结构，**不是标注的物体/手指语义关键点**。融合前每相机128D可按来源识别；拼接256D视觉+64D本体+64D上下文后，GRU输出128D混合隐变量，不给这些维度指定固定物理含义。

Nominal视觉/历史参数未来经真实动作序列的diffusion epsilon loss反向学习，不能根据随机初始化特征宣布几何能力。Correction触觉/视觉/运动参数需要未来有依据的Correction目标；本轮只做前向与梯度通路检查，未选择损失或更新参数。掩码测试检查无效Depth/相机像素扰动不影响输出且梯度为零；有效RGB/Depth梯度非零。这个证据比“形状相同”强，但仍不是任务感知性能。

## 存储选择与实际参考实现

源码快照只在忽略的 `outputs/visuotactile/history_refs/`；没有复制新第三方代码或引入依赖。以下按固定版本读过，而非只引用论文名称：

| 实现 | 实读路径和取舍 |
| --- | --- |
| [Robomimic d309eae](https://github.com/ARISE-Initiative/robomimic/tree/d309eaecc18acf4152a830a895a6984b8ac71b05)，MIT | `models/base_nets.py` 显式传入/返回RNN state与forward_step；采用这个状态接口思路。`algo/bc.py` 的BC_RNN会每rnn_horizon重置，**本项目不采用周期清空**。`utils/dataset.py`按episode HDF5读序列。 |
| [Stanford DP 5ba07ac](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)，MIT | `common/replay_buffer.py`使用Zarr+episode_ends，`common/sampler.py`有限窗口采样；数据边界有参考价值，但窗口不等于持续记忆。 |
| [LeRobot dd5313e](https://github.com/huggingface/lerobot/tree/dd5313e305ff7aaa5b6c7b4769d1b26650ce250b)，Apache2 | `datasets/lerobot_dataset.py`及video_utils：Parquet元数据/数值与分片视频、timestamp对齐。适合大规模RGB共享，但需要另行解决米制Depth无损与跨文件提交，不为当前小样本直接迁移。 |

| 方案 | 完整性、读取与维护成本 |
| --- | --- |
| 当前HDF5 | 嵌套字段+单位/来源metadata，float32 Depth/force无损gzip；episode内按step随机切片。已改逐步append/flush，提交length最后更新。单writer，断电/进程中断仍可能需HDF5恢复，不声称事务数据库。 |
| 一episode一PKL | Python内嵌dict直观、保留dtype与精度；通常读取整个episode，不利随机片段、大图内存和语言互通，不能加载不可信pickle。新增导出供检查，不替代主存储。 |
| Zarr | 分块数组、episode_ends，便于并行/远端切片；文件/对象多，需要新的schema迁移与一致性维护，当前收益不足。 |
| Parquet+video | 数值列/大规模RGB高效；有损视频、Depth精度、解码时钟及跨文件一致性需专门设计，当前保留研究空间。 |

`EpisodeReader.read/steps`只读已提交length；PKL导出保存metadata、attrs、schema/trace版本和按step为第一轴的完整嵌套数组，临时文件fsync后原子发布，不覆盖。导出可能包含privileged诊断，不能把整个字典直接送模型。原始RGB-D、力、q、动作都保留，网络特征不能替代它们。

尚缺：terminal图像、player clip前均值、实机曝光/通信时间、学生异步/延迟/噪声模型、Correction监督、长期记忆效用。新metadata的missing_fields和逐步available标志说明缺失；不能猜测补齐。

## 本轮实际证据（固定参数）

公共数值摘要：[evidence/causal_history.json](evidence/causal_history.json)。原始轨迹、权重、参考源码快照和日志留在忽略的outputs中。

- `causal_trace02/causal_checks01`：2env×240控制步，480行、3完成+2尾段；每行delay来源/时间、图像post/pre对应、FK/局部力重算和3个terminal图像缺失检查通过。15个专家字段与已有有效渲染轨迹完全一致。
- 最长225步：完整recurrence、chunk7、逐步在线一致。Nominal最大误差4.66e−7（不同CNN batch的浮点算术）；Correction chunk7与step1精确一致，与实际2env在线日志最大差4.47e−8。不是225步无损记忆质量测试。
- `causal_gradients01`：先因果回放37步形成状态，再在真实接触帧做Jacobian检查；两视角RGB/Depth输入、Nominal视觉/时序、Correction视觉/五指/motion/plan/intent-memory/head均非零有限梯度。没有标签拟合或optimizer；hash验证参数未变。权重与当前隐藏状态保存恢复后同一输出精确一致。这是张量重载测试，不是模拟器中途恢复功能。
- `causal_trace03/causal_checks02`：64行最终物理telemetry，真实关节限位和arm/hand目标公式、专家输入与hash通过。延迟时钟float64修正后最大差5.55e−17s。PKL114字段dtype/shape/value精确往返。
- `causal_trace04/causal_checks04`：最后2env×8步检查Correction参数hash/config、专家输入在env.step前冻结、全部目标公式和15字段回归；16行没有接触，只核验最终来源信息。此短样本不能代替上面的接触/terminal证据。
- `causal_regression01`：重新启动原渲染环境与新路径各32步，initial13项、逐步15项bitwise相等，包含RNG、物体、目标、关节、reward、done。已知kinematic setVelocity告警原路径也有；场景健康断言通过，不宣称整个安装环境无缺陷。
- `causal_unit_tests_final.log`：**23 passed，4.52s**，1个已有requests依赖warning；没有更改共享依赖。覆盖超出旧窗口的历史影响/梯度、每步记忆与重规划分离、异步reset、跨块力变化/事件、mask零梯度、真实原action helper的已知正弦序列与RNG、存储/版本边界。`compileall`与`git diff --check`另作最终静态检查。

当前训练入口只验证了顺序条件与loss反传接口，没有运行完整optimizer训练/长期恢复；memory2验证loss仍未实现。公共代码没有附带本轮训练数据或模型。

## 2026-10-29 recording throughput extension

New production stores one independently finalized episode per file with original143 logical fields; no raw observations replaced by network features. Native per-env reset clears only that env's actor RNN/queues; subgoal changes do not clear it. Nominal/Correction remain absent and masked. A four-batch writer queue and at most eight pending CPU audits decouple I/O from simulation without changing simulated timestamps or action dt. Original per-step RNG trace is a global run stream, repeated across env rows by design; episode identity is run/env/native-reset ID. Exact reconstruction and source hashes are checked before manifest publication. Full details/evidence in NIGHT_COLLECTION.
