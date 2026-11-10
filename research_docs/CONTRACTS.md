# Tight-Insertion 现行合同：成功筛选的专家接管式 DAgger

本页是当前实现与授权的统一约定。最新用户更正要求聚合后**先训练前端，再训练后端**，覆盖附件最初的后端限定。原实验 `tight_transformer_20261110_02` 的冻结合同继续约束它自己的运行；不能用新授权热改旧进程。旧阶段详细结果留在 [STATE_DP_RESULTS](STATE_DP_RESULTS.md)，过时预算/授权不再重复充当当前规则。

## 范围、来源与执行顺序

配置在 [tight_state.json](../configs/tight_state.json) 的 `dagger`，版本 `successful_takeover_two_stage_v1`。首轮只保留两组：

|组|采集起始前端|前端训练标签|后端起点与标签|
|---|---|---|---|
|G11|既有 G10 selected EMA；RGB-D+robot178+直接触觉|命令 `u_expert`|当前 `_02/command_residual/G11` 选定Transformer；最终 `u_expert`|
|QG01|当前 `_02/configuration/QG00` selected EMA；RGB-D+robot178|实际配置 `q_reached`，保留既定几何loss|当前 `_02/configuration_residual/QG01` 选定Transformer；最终 `u_expert`|

共享监督类型映射完整覆盖：G00/G10 前端为 `u_expert`；Q00/QG00 前端均为 `q_reached`；G01/G11/Q01/QG01 后端均为 `u_expert`。本轮只调度表中两组，数据层按显式监督类型选择字段与统计，不按组名猜测。

最终来源必须核验 checkpoint SHA、已完成更新、前后端配对和各自统计身份；不能用 disposable probe 或缺失/损坏权重代替。旧 G10 v4 隐含 command 合同可显式声明为 v5 `u_expert`，所有 EMA 参数和统计逐 tensor 不变；不能把旧命令模型直接解释为已训练的配置模型。新 round 使用新的优化器/日程/计数，同 round 恢复保留完整训练状态。

顺序为：两组有限采集 → 各自成功纠正与原 Core 聚合 → 现有前端全部可训练模块微调 → 固定 validation 选择前端 → 冻结该前端并刷新全池名义缓存 → 从原选定后端微调 → 无接管自主评估。没有足够有效前端标签时明确报告不足，不悄悄退为仅后端实验。两组比较包含前端模态和监督方法差异，不能单独证明几何因果贡献。

## 原生执行与专家接管

H24/commit12、sample/x0、DDIM16/seed7091、overlap0；5Hz规划、60Hz控制、120Hz物理。动作是 canonical Arm7+Hand22 的29D绝对关节目标 rad，经原关节限位和位置执行接口提交；不把归一化原生 expert action 直接当 rad。原资产、PD、物理、任务成功、timeout、reset 与 DR 不改。

每次从登记训练家族初态运行学生。专家 recurrent shadow 在每个真实观察上连续推进；按预定第60/120/180/240控制tick接管，不在终止后接管、不恢复理想姿态或拼接 reset。专家动作经原生映射后施加已有 Arm=.5、Hand=1、Hand目标变化上限2rad/s；学生目标不再重复这一限速。两者通过同一位置执行器，并记录两个物理子步实际目标。

专家独立建议队列从学生前缀开始更新，复用位置执行器原有的一次 delay RNG draw，不额外抽样或修改共享 native queue。接管时全队列尚含学生控制期建议的过渡段不监督；干净专家实际执行后缀才有有效标签。连续 shadow、映射和时序需新阶段真实采集验收，CPU/mock 通过不等于物理接管通过。

只有原生完整成功的接管 episode、且内容/时序/标签有效，才能加入成功纠正主池。学生提前自主成功、接管失败、超时和软件中断各自保留分母；它们不自动变为专家训练数据。不会为了凑成功补 seed 或扩大机会。

## 记录、标签与历史的分离

标准 HDF5 `transition/target_after_rad`、`diagnostics_before/after`、issued/applied 和观测时钟保留实际执行。额外 `transition/dagger/` 包含专家 raw/clipped/delayed action、来源 tick、control_source、segment_id、expert_label_valid、switch_clean、previous_issued、previous_nominal/valid、u_expert、q_reached。原始 q_after 可保存全段；它只有在专家真正执行且 mask 有效时才是专家配置监督。

|共享字段或语义|来源与用途|约束|
|---|---|---|
|`robot` / `command`|完整真实 q/qd、目标和过去实际执行；command保持实际 target_after|学生前缀也真实保留；不被零标签或重预测替代|
|`clean_sample`|前端按 supervision 类型选 u_expert 或 q_reached|只用于加噪 clean sample 和同语义目标/FK loss|
|`expert_command`|后端最终命令标签 u_expert|不因前端采用 q_reached 而改变|
|`supervision_mask[T]`|映射 expert_label_valid；原 Core 全 true|学生/切换过渡为 false；是 loss/exposure mask|
|`control_segment[T]`|映射 segment_id|控制权或有效性切换递增；H24不能跨段|
|`inputs.valid`|真实历史行存在位|保留学生前缀；不能与监督 mask 混用|
|`previous_nominal`|上一 tick 的名义预测|与严格过去 issued/applied 独立；首tick无效|
|`loss_context`|真实后态物体/接触等几何监督|独立文件；不进入学生 encoder、接管触发或成功 gate|

`u_expert` 是专家实际最终执行目标；`q_reached` 是同一专家 transition 积分后的真实配置。学生执行后的 q_after 不能作为专家建议的反事实配置，也不能从原示范“相同 tick”取标签覆盖偏离状态。

规划 anchor 是12的倍数。H24 从 anchor 当前 transition 开始，遇首次无效、segment变化或 episode尾部即截断余下标签；填充为零且 mask=false。训练只抽有专家监督的 anchor，曝光按实际有效人口计。观察历史可跨接管边界，GRU仍从真正 episode起点推进；96tick TBPTT或后端32tick detach只限制梯度，不重置记忆。未来观察/动作只作允许的标签，不作策略输入。

## 聚合、统计、缓存与损失

原 Core train166 加各组成功纠正；validation30 和 test22 身份保持，test22 本轮不读取轨迹、不评估、不选模。原 Core 的55,392训练tick、4,690规划anchor是旧人口；聚合后的分母由实际 mask 重新计数，不能用历史 epoch 或学生前缀冒充新增监督。

`normalization_train_ids` 固定为原训练人口；`statistics.pt` 与 G10 的 `tactile_statistics.pt` 按原字节复用。Q 前端保持 q affine；G 前端保持 u affine；所有后端保持共同 `u_expert_train166_affine_v1`。不从纠正数据/validation/test重新拟合。G10六个、G00四个观测/动作统计 buffer 在同语义续训时逐 tensor 校验。

前端保持现有 Conditional U-Net 和持续历史；DINO backbone 冻结，既有视觉投影/深度/历史/策略参数可训练。QG沿用已验收的 signed-surface soft geometry：分离 d>0、穿透1mm容许、指定实际末节接触距离2mm；额外梯度仅到手22D，arm/object上下文 detach。既有校准权重/选点规则保持，不执行逐配置SLSQP，不新增未经建模的内力 loss。

后端仍为 `target_residual_transformer60_v1`：756,581参数，2层d128/4头/FF256，过去32tick与持续GRU、当前H24计划/cursor/mask和独立上一名义输出。无失触立即清零门；监督只作用本轮有效专家tick，Arm/Hand等权，既定残差幅度与关节限位不变。报告所需修正超界比例、实际幅度/饱和和观测失触→再接触；后者不证明已学会恢复。

新前端选定后，对全部原Core/纠正train和validation重放完整实际历史，重新计算持续隐藏态、视觉读出、H24/cursor/mask及 previous_nominal。不能复用旧 DP 的预测或可训练表征。每个后端缓存绑定 `base_checkpoint_sha256/statistics_sha256/data_sha256/residual_statistics_sha256`，index另绑定实际监督mask+segment SHA。只允许复用源轨迹、backbone与内容SHA都一致的冻结DINO特征；新纠正图像重新提特征。

## 有限预算、停止与自主评估

|项目|本轮登记|
|---|---|
|round|1；新阶段实际起点起12h硬上限，含等待/探针/报告；最后10min预留收尾，不继承旧deadline|
|采集机会|每组12；相同训练家族分布/接管tick规则；每组采集上限1.5h，无替换尝试；在原12机会内累计首2个真实且标签合法的接管作接口验收，自然提前终止继续原计划|
|进入训练最低新增监督|至少1个成功纠正episode、48有效专家tick、4有效DP anchor；不足如实停止该分支|
|前端|最多20有效anchor等效遍历；10后才允许验证早停，约每5遍验证；LR峰值1e-5；每组最多4h|
|后端|最多30有效监督tick遍历；10后才允许验证早停，每5遍验证；LR3e-4；每组最多1.5h|
|实际分配|先测真实吞吐；各阶段上限和12h全局截止优先，未达曝光须披露|
|开发自主评估|各10个既有固定合法初态机会，前2身份视频；拒绝/中断不补；无专家替换|
|资源|充分使用空闲GPU；至少6144MiB空闲显存、14GiB RAM、60GiB磁盘；只管理自有进程|

验证使用原固定数据；保存选定模型和最近可恢复状态，报告真实更新/有效曝光/训练与等待墙钟。少量 acceptance/probe 更新是可丢弃验收，不计正式训练。预算耗尽或少量成功不能被称作收敛/稳定改善。最终以原生完整成功及阶段/掉落/timeout报告；接管成功、自主成功和软件有效分母分开。原协议相同且模型未变的已有结果可引用，不重跑以挑选有利结果。

## 复用的模块与新增职责

|职责|原位复用/新增|本轮变化|
|---|---|---|
|配置与入口|现有 `configs/tight_state.json`、`tools.tight_finish`|增加 dagger 配置与子命令，不按组复制脚本|
|后继协调|新增小模块 `tools.tight_dagger`|等待旧进程真实退出、登记/锁/阶段和凭据；不实现模型或优化循环|
|采集/动作|现有 `state_benchmark.trial`、`position_control`、recording/compact|opt-in expert bridge和同episode接管字段|
|数据与缓存|现有 `narrow_data/native_data/anchor_sampling`|同loader聚合、两监督类型、mask/segment、DINO共享与新DP缓存绑定|
|前端与后端|现有 `state_history/native_dp/native_controller`|严格同语义EMA/后端参数续训；网络种类不变|
|优化/评估/发布|现有 `anchor_continue`、`tight_finish`、`tight_diagnostics`|有效曝光与恢复、无接管评估、当前结果汇总|

官方 Diffusion Policy 的 image workspace 分别构造 policy、dataset/validation、normalizer、optimizer/EMA 和 env runner；本项目据此保持数据、模型、训练循环与运行协调的职责分离，没有迁移其框架或套用其任务动作语义。[官方 workspace 源码](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py)

robomimic 的 SequenceDataset 将 HDF5 观察键、动作键、序列长度、缓存与padding mask作为数据层参数；这里沿用统一 episode 读取和显式 mask，额外保证接管控制段不被H24跨越。[官方 SequenceDataset](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/dataset.py)

imitation 将交互收集、轨迹保存与后续 BC 训练分离；其 InteractiveTrajectoryCollector 可在学生实际执行时仍保存专家建议。本轮只监督成功接管后的专家实际执行后缀，明确是成功筛选的接管变体，不声称与标准全学生状态标注 DAgger 等价。[官方算法说明](https://imitation.readthedocs.io/en/latest/algorithms/dagger.html) · [官方 collector/trainer 源码](https://imitation.readthedocs.io/en/latest/_modules/imitation/algorithms/dagger.html)

## 接续、发布与清理

准备在隔离 `bc+rl_RSS_dagger`，旧 `bc+rl_RSS_palm_eval` 不动。普通 `dagger_follow` 核验旧状态、owner/worker退出、有效选定模型，再整合原owner发布并进入新轮。等待锁、registration、实际进程和status共同证明已排队；仅有脚本/配置不证明注册。不得由LLM持续轮询。原轮零成功不阻止采集；真正损坏/缺失权重须明确报告。

只推 `origin/feature/tight-state-foundation`；新小型结果写现有文档，大数据/权重/视频留本地。按实际依赖清理不再使用的旧代码、checkpoint、缓存和报告，记录文件数与实际释放量；保护原400/仍用Core、专家/资产、活动/排队依赖、选定及恢复模型和必要比较结果。不能整树删除仍被manifest引用的历史run，不通过归档搬家冒充删除。

当前已完成的CPU数据与接口验收见 [SESSION_HANDOFF](SESSION_HANDOFF.md)。它不构成真实接管成功、正式训练完成或自主成功证据。
