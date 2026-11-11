# Tight-Insertion 现行合同：成功筛选的专家接管式 DAgger

本页约束新授权的多轮共享池DAgger：每轮先训练现有前端，再刷新缓存并训练后端。旧 `_02` 与 `tight_dagger_20261111_01` 已有完成凭据；新工作在隔离 `bc+rl_RSS_qg_report`，不修改旧产物或历史结果。当前实现与CPU验收不等于新GPU运行已启动，实际状态由owner凭据说明。[历史结果](STATE_DP_RESULTS.md)。

## 范围、来源与执行顺序

配置在 [tight_state.json](../configs/tight_state.json) 的 `dagger`，版本 `successful_takeover_pool_v1`。按QG01优先、G11随后推进最多三轮：

|组|采集起始前端|前端训练标签|后端起点与标签|
|---|---|---|---|
|G11|已完成DAgger G11 selected EMA；RGB-D+robot178+直接触觉|命令 `u_expert`|同一已完成DAgger配对的Transformer；最终 `u_expert`|
|QG01|已完成DAgger QG01 selected前端EMA；RGB-D+robot178|实际配置 `q_reached`，保留既定几何loss|同一已完成DAgger配对的Transformer；最终 `u_expert`|

共享监督类型映射完整覆盖：G00/G10 前端为 `u_expert`；Q00/QG00 前端均为 `q_reached`；G01/G11/Q01/QG01 后端均为 `u_expert`。本轮只调度表中两组，数据层按显式监督类型选择字段与统计，不按组名猜测。

最终来源必须核验 checkpoint SHA、训练来源、前后端配对和各自统计身份；不能用disposable probe或缺失/损坏权重代替。选定update0必须有真实训练过的源checkpoint及SHA链，new_optimizer_updates仍记0，不能把它算新增训练完成。旧 G10 v4 隐含 command 合同可显式声明为 v5 `u_expert`，所有 EMA 参数和统计逐 tensor 不变；不能把旧命令模型直接解释为已训练的配置模型。新 round 使用新的优化器/日程/计数，同 round 恢复保留完整训练状态。

顺序为：本轮当前配对学生有限采集 → 两组成功纠正进入同一持续池并冻结共同快照 → 原Core与共享纠偏分层微调现有前端 → 原validation与correction_dev联合选模 → 冻结选定前端并刷新全池名义缓存 → 原选定后端微调 → 无接管自主评估 → 完整pair选择/回退并进入有限下一轮。没有足够有效前端标签时明确报告不足，不悄悄退为仅后端实验。两组比较包含前端模态和监督方法差异，不能单独证明几何因果贡献。

## 原生执行与专家接管

H24/commit12、sample/x0、DDIM16/seed7091、overlap0；5Hz规划、60Hz控制、120Hz物理。动作是 canonical Arm7+Hand22 的29D绝对关节目标 rad，经原关节限位和位置执行接口提交；不把归一化原生 expert action 直接当 rad。原资产、PD、物理、任务成功、timeout、reset 与 DR 不改。

每次从登记训练家族初态运行学生。专家 recurrent shadow 在每个真实观察上连续推进；按冻结的本轮计划接管：新计划switch=12k−2，原3项delay队列清洁后缀从12k规划anchor开始；默认switch为46/70/94/118/154/190/238/298tick，后续仅按登记有限规则调整，不在终止后接管、不恢复理想姿态或拼接 reset。专家动作经原生映射后施加已有 Arm=.5、Hand=1、Hand目标变化上限2rad/s；学生目标不再重复这一限速。两者通过同一位置执行器，并记录两个物理子步实际目标。

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

## 聚合、统计与缓存

原Core train166加两组/各轮共享成功纠偏train；已有21条旧纠偏均保留为历史train，新采集的correction_dev在collection_plan整episode预登记并永久隔离。validation30和test22身份保持，test22 本轮不读取轨迹、不评估、不选模。原 Core 的55,392训练tick、4,690规划anchor是旧人口；聚合后的分母由实际 mask 重新计数，不能用历史 epoch 或学生前缀冒充新增监督。

`normalization_train_ids` 固定为原训练人口；`statistics.pt` 与 G10 的 `tactile_statistics.pt` 按原字节复用。Q 前端保持 q affine；G 前端保持 u affine；所有后端保持共同 `u_expert_train166_affine_v1`。不从纠正数据/validation/test重新拟合。G10六个、G00四个观测/动作统计 buffer 在同语义续训时逐 tensor 校验。

池的全局身份包括run/group/round/source与内容SHA；同内容去重，禁止改变既定cohort。每轮snapshot冻结之后再训练，两组读同一快照。本轮层必须同时匹配run_id与round_index，不能把旧run相同轮号混入本轮。采样按Core .5、历史纠偏 .2、本轮纠偏 .3，空纠偏层权重移到另一纠偏层；DP按有效anchor，后端按有效监督tick分配完整episode。记录实际各层比例、重复样本、唯一episode/anchor/tick覆盖与等效曝光，不把重复抽样称为新增数据。

前端保持现有 Conditional U-Net 和持续历史；DINO backbone 冻结，既有视觉投影/深度/历史/策略参数可训练。每轮先恢复这些前端参数参与微调，再固定本轮选定前端训练后端；不能把整个 DAgger 轮次称为“冻结前端”。前后端分别优化，没有端到端联合反传。

后端仍为 `target_residual_transformer60_v1`：756,581参数，2层d128/4头/FF256，过去32tick与持续GRU、当前H24计划/cursor/mask和独立上一名义输出。无失触立即清零门，既定残差幅度与关节限位不变。报告所需修正超界比例、实际幅度/饱和和观测失触→再接触；后者不证明已学会恢复。

新前端选定后，对全部原Core/纠正train、validation和correction_dev重放完整实际历史，重新计算持续隐藏态、视觉读出、H24/cursor/mask及 previous_nominal。不能复用旧 DP 的预测或可训练表征。每个后端缓存绑定 `base_checkpoint_sha256/statistics_sha256/data_sha256/residual_statistics_sha256`，index另绑定实际监督mask+segment SHA。只允许复用源轨迹、backbone与内容SHA都一致的冻结DINO特征；新纠正图像重新提特征。

<a id="qg-loss-details"></a>

## QG 数学目标、量纲与梯度

以下是已完成离线 QG00 与本轮 DAgger QG01 前端实际沿用的 `hand_surface_soft_hinge_v1` 目标。Q00 使用相同配置监督但没有手部几何项；G00/G10 保持命令监督。`loss_id=palm_command_sample_commit12` 是历史标识字符串，不能据此把 QG 的真实 `q_reached` 标签解释成 `u_expert`。实际标签由合同中的 `supervision/clean_sample_field/normalization_identity` 决定。

令 $q^*$ 为有效专家 transition 完成后的配置（rad），$z^*=s_q\odot q^*+b_q$ 为其固定 train166 affine，$\hat z$ 为 U-Net 对加噪 $z^*$ 的 sample/x0 预测，$\hat q=(\hat z-b_q)/s_q$。clean sample、加噪输入和拟合目标都使用这一配置语义。基础拟合覆盖全部有效 H24 目标：

$$
L_{\rm fit}=\frac12\operatorname{mean}_{V,\,j\in Arm7}(\hat z_j-z_j^*)^2
+\frac12\operatorname{mean}_{V,\,j\in Hand22}(\hat z_j-z_j^*)^2.
$$

$V$ 排除 padding、学生前缀和切换过渡；窗口在首次无效监督或控制段边界处截断。两个关节组各占一半，不能按29个关节直接混合求均值。掌部项仅覆盖 $C=V\cap\{0,\ldots,11\}$，用还原后的真实弧度做 FK：

$$
L_{\rm pos}=\operatorname{mean}_{C}\|p(\hat q)-p(q^*)\|_2^2,\qquad
L_{\rm rot}=\frac12\operatorname{mean}_{C}\|R(\hat q)-R(q^*)\|_F^2.
$$

位置项单位是 **m²**，旋转矩阵项无量纲，不能把毫米或角度误差直接代入这些训练项。这里的掌部位姿是预测关节配置的 FK，不是实际自主轨迹偏离。代码见 [sample/x0 与掌部目标](../visuotactile/state_image_loss.py#L59)、[Arm/Hand 等权拟合](../visuotactile/dp_objectives.py#L8)、[有效窗口](../visuotactile/anchor_sampling.py#L273)。

对抽中的配置，$d_c$ 是手部碰撞表面与物体之间的有符号距离（m，分离为正、穿透为负），$d_f$ 是指定手指实际末节/弹性体碰撞表面的最小距离，$m_f$ 是要求接触的五指 mask。令 $[x]_+=\max(x,0)$：

$$
\ell_{\rm geom}=
\operatorname{mean}_{c}\left(\frac{[-d_c-0.001]_+}{0.001}\right)^2
+\frac{\sum_f m_f\left(\frac{[d_f-0.002]_+}{0.002}\right)^2}{\max(1,\sum_fm_f)}.
$$

穿透容许1mm、要求接触的表面间距容许2mm；平方 hinge 除以各自容差平方，因此几何项与其乘子均无量纲。接触安排来自同 transition 最后120Hz采样的 object-filtered force，范数超过0.1N的指头才有 $m_f=1$。接近、释放及未要求接触的指头没有吸附项。after-state 物体位姿、接触 mask 和阶段只进入 loss，不进入策略观测；它们与 `q_reached` 对齐。[软几何实现](../visuotactile/state_image_loss.py#L162) · [几何标签时序](../visuotactile/state_image_loss.py#L354)

每个完整 optimizer update 最多抽两个合法 commit 配置，有候选时接触/非接触各一个，否则从剩余候选补齐。$L_{\rm geom}$ 是这两个配置的等权均值；分母在整个 update 内固定，各微批直接累加贡献，不能误乘成每微批两个配置。该分层抽样均值**不是整个目标人口的无偏估计**；基础拟合仍使用全部合法目标，困难几何样本不按可行性剔除。[抽样](../visuotactile/anchor_continue.py#L125) · [全 update 归一化](../visuotactile/state_image_loss.py#L113) · [微批累加](../visuotactile/anchor_sampling.py#L221)

实际总损失与权重为：

$$
L=L_{\rm fit}+g_p(k)\,[0.3342396121814294L_{\rm pos}
+0.029946394754032265L_{\rm rot}]
+g_g(k)\,2.355370821894758\times10^{-6}L_{\rm geom},
$$

其中 $k$ 是本训练阶段从1开始的 optimizer update，$g_p(k)=\min(1,k/500)$，$g_g(k)=\operatorname{clip}((k-100)/400,0,1)$。前100次更新跳过几何距离计算，第101次开始渐入，第500次达到全权重；掌部权重独立渐入，手部几何不会再重复乘 $g_p$。位置权重对应上面的 m² 标度。[实际训练累加与调度](../visuotactile/anchor_continue.py#L717)

几何权重经过训练集六状态检查和**一次完整 update**校准：基础手部输出梯度范数0.001015357876518895，未加权几何手部输出梯度范数86.21639251708984，最终加权比0.2，手臂几何输出梯度为0。这个0.2是该次校准的输出梯度比，既不是模型参数梯度比，也不是后续每次更新的保证；没有使用 validation/test 调权重。[原始软几何验收](evidence/tight_reuse_20261110/geometry_soft_acceptance.json) · [吞吐与拟合短测](evidence/tight_reuse_20261110/training_readiness.json)

距离计算复用 `HandCollisionGeometry.distances()` 的局部手关节 Jacobian；碰撞 witness 不反向传播，arm/object 上下文 detach，几何梯度只接到输出的 Hand22。共享网络参数会收到这些手部梯度，不能声称共享参数更新绝不会影响手臂预测。当前正式训练不调用 SLSQP。此前 `projection_v1` 则是求局部可行投影后回归投影手部目标，只有可行样本贡献；两者数学目标与权重单位不同，不能把旧投影权重直接沿用或混称同一方法。精确投影代码只留诊断。**内部力项未启用**，软惩罚也不保证几何可行或任务成功。

几何训练日志记录的是抽样 x0 输出在记录物体场景下的距离与局部梯度；`maximum_penetration_violation_m` 是**超过1mm后的违约量**，绝对穿透深度应取 $\max(0,-\min_c d_c)$。离线几何诊断使用专家观测前缀下的完整 DDIM 预测。两者都不能冒充自主 rollout 的实际物理穿透或失败因果证据。

## 后端目标与验证选择

所有 Transformer 后端都监督最终 `u_expert`，包括接在 Q00/QG00 后面的 Q01/QG01。设共同命令 affine 的 scale 为 $s_u$，$\hat u$ 为限位后的组合命令，$\Delta$ 为归一化命令空间残差，则每个合法监督 tick：

$$
\ell_{\rm back}=\frac12\operatorname{mean}_{Arm7}[s_u\odot(\hat u-u^*)]^2
+\frac12\operatorname{mean}_{Hand22}[s_u\odot(\hat u-u^*)]^2
+10^{-4}\left(\frac12\operatorname{mean}_{Arm7}\Delta^2+\frac12\operatorname{mean}_{Hand22}\Delta^2\right).
$$

最终对合法专家监督 tick 求均值，接触/非接触都参与；学生前缀仍更新历史但不监督。后端没有新增 QG 几何项，也不把命令 affine 换成前端配置 affine。组合为 `base + delta / action_scale` 后施加关节限位，残差幅度沿用既有校准。[组合与后端目标](../visuotactile/models/native_dp.py#L549)

前端按固定 validation4、真实完整前缀、部署 batch1 DDIM16/seed7091 的 commit12 误差选择最低分 EMA：

$$
S=\frac14\left[(\mathrm{ArmRMSE}_{rad}/0.01)^2+
(\mathrm{HandRMSE}_{rad}/0.1)^2+
(\mathrm{PalmRMSE}_{mm}/10)^2+
(\mathrm{MeanAngle}_{deg}/5)^2\right].
$$

四项物理尺度固定，不是训练loss或几何可行率。H24全长误差另报。前端旧验证、纠偏dev及其每episode最初24个有效commit目标沿用同一物理分数；后端使用命令归一化Arm/Hand等权MSE，不加残差正则。学生前缀与跨控制段窗口均不计误差，仍完整推进历史。

两阶段都真实评估update0；三部分各自除以对应update0分数（数值分母下限1e-12），按旧validation .5、correction_dev .25、纠偏初24 .25加权。较差的训练checkpoint可以全部不入选；选中update0不表示未进行训练，正式完成凭据另报告真实新增更新。保存baseline、各项原分数/归一化比、best和last。后端checkpoint始终绑定产生其缓存的选定前端SHA。新pair若自主表现劣化，回退上轮完整配对；不能任意拼装两个独立best。

## 有限预算、停止与自主评估

|项目|本次登记|
|---|---|
|round/总时限|最多3轮；新root实际起点起18h，包含等待、探针和报告，最后10min收尾；resume不重设deadline|
|采集机会|每轮每组默认32，有限规则可24；每组最多96、两组共192；4个dev编号1/10/20/31预登记；每组每轮最多1.25h，无失败补seed|
|最低新增监督|成功纠偏episode、至少48有效专家tick和4个有效DP anchor；新增数据不足如实保留原pair，不偷用dev或退为仅后端|
|前端|最多10有效anchor等效遍，6后可验证停止，约每2遍验证；LR峰值1e-5；每组每轮最多2h|
|后端|最多12有效监督tick等效遍，6后可验证停止，每3遍验证；LR3e-4；每组每轮最多.75h；完整episode组取整的超出tick明示|
|开发自主评估|先固定4例；>=1原生成功、>=1子目标或>=2原生lift时才按同一登记队列扩至10；每组1视频；拒绝/中断不补，始终无专家替换|
|资源|充分使用空闲GPU；至少6144MiB空闲显存、14GiB RAM、60GiB磁盘；只管理自有进程|

正式LR日程先固定，短探针只提前结束且明确disposable；不把少量验证当作收敛。DP与backend使用各自新增更新/有效曝光；重复轮次不重做已有SHA凭据完整的阶段。0自主成功本身不取消已授权有限下一轮；数据错误、持续工程错误、资源/绝对时限仍会停止。旧结果可在相同模型SHA、协议和初态spec下引用，不能重跑挑选有利结果。接管成功、自主成功和软件有效分母分开，早停理由及真实耗时均公开。

## 复用的模块与新增职责

|职责|原位复用/新增|本轮变化|
|---|---|---|
|配置与入口|现有 `configs/tight_state.json`、`tools.tight_finish`|增加 dagger 配置与子命令，不按组复制脚本|
|后继协调|新增小模块 `tools.tight_dagger`|持久共享池/每轮snapshot、登记/锁/阶段、有限决策和恢复凭据；不实现模型或优化循环|
|采集/动作|现有 `state_benchmark.trial`、`position_control`、recording/compact|opt-in expert bridge和同episode接管字段|
|数据与缓存|现有 `narrow_data/native_data/anchor_sampling`|同loader聚合、两监督类型、mask/segment、DINO共享与新DP缓存绑定|
|前端与后端|现有 `state_history/native_dp/native_controller`|严格同语义EMA/后端参数续训；网络种类不变|
|优化/评估/发布|现有 `anchor_continue`、`tight_finish`、`tight_diagnostics`|有效曝光与恢复、无接管评估、当前结果汇总|

官方 Diffusion Policy 的 image workspace 分别构造 policy、dataset/validation、normalizer、optimizer/EMA 和 env runner；本项目据此保持数据、模型、训练循环与运行协调的职责分离，没有迁移其框架或套用其任务动作语义。[官方 workspace 源码](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py)

robomimic 的 SequenceDataset 将 HDF5 观察键、动作键、序列长度、缓存与padding mask作为数据层参数；这里沿用统一 episode 读取和显式 mask，额外保证接管控制段不被H24跨越。[官方 SequenceDataset](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/dataset.py)

imitation 将交互收集、轨迹保存与后续 BC 训练分离；其 InteractiveTrajectoryCollector 可在学生实际执行时仍保存专家建议。本轮只监督成功接管后的专家实际执行后缀，明确是成功筛选的接管变体，不声称与标准全学生状态标注 DAgger 等价。[官方算法说明](https://imitation.readthedocs.io/en/latest/algorithms/dagger.html) · [官方 collector/trainer 源码](https://imitation.readthedocs.io/en/latest/_modules/imitation/algorithms/dagger.html)

## 接续、发布与清理

准备在隔离 `bc+rl_RSS_qg_report`，旧树/旧产物不动。普通owner核验源冻结、原配对SHA与自有worker实际退出后推进。恢复命令：`bash visuotactile/run_cpu.sh -m tools.tight_finish dagger_resume --root <root>`；完整阶段按凭据/SHA跳过，中断训练从last的optimizer/EMA/RNG/deck/cursor恢复。STOP只请求本任务在optimizer或完整episode组边界保存；未持久化日志保留为`*.uncommitted.jsonl`，不重复纳入正式计数。恢复不重设总deadline。注册/状态/真实进程共同证明启动，仅有脚本或CPU测试不构成GPU成果；不得由LLM持续轮询。

只推 `origin/feature/tight-state-foundation`；新小型结果写现有文档，大数据/权重/视频留本地。按实际依赖清理不再使用的旧代码、checkpoint、缓存和报告，记录文件数与实际释放量；保护原400/仍用Core、专家/资产、活动/排队依赖、选定及恢复模型和必要比较结果。不能整树删除仍被manifest引用的历史run，不通过归档搬家冒充删除。

当前已完成的CPU数据与接口验收见 [SESSION_HANDOFF](SESSION_HANDOFF.md)。它不构成真实接管成功、正式训练完成或自主成功证据。
