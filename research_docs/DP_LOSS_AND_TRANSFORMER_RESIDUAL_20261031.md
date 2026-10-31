# DP损失对照与Transformer触觉残差：2026-10-31

## 开始状态与预登记

接手HEAD及origin功能分支均为`7a592ad86377ef29fac8b5b694e883ecdacc24a0`，工作树干净，旧任务已退出。
本轮用户重新授权真实续训、独立Transformer残差、有界诊断、曲线及审查后的origin feature推送。
原400条、320/40/40、旧best及正式0/50保留；原controller/physics/DR/reset/reward/120/60Hz及sim wrapper只读。
本文件先于新训练代码建立，结果逐阶段追加。状态：接口验收和短pilot通过，G00-A/B正式续训运行中。

### 配对设计与停止条件

- 新run ID `dp_loss_trt_20261031_01`，唯一总计算窗口6小时（含检查、等待、恢复和诊断），起止由session.json固定。
- G00-A/B优先，随后G10-A/B；每对从对应原epoch40 EMA best的相同权重和normalizer出发。
- 四线各固定额外12完整train320遍历、9,984更新（832/遍），不是临时缩短20遍以宣称完成。
  原最后10遍全时域validation selection MSE仍改善：G00从0.914181至0.894374，G10从0.916017至0.907204，
  但改善有限；给定四线、详细验证和残差的6小时总预算，预留至少90分钟残差及60分钟闭环/交付。
- 每对同seed/episode顺序/噪声，独立模型/EMA/optimizer/RNG；AdamW1e-4、betas(.95,.999)、wd1e-6，
  新cosine schedule共9,984更新/warmup500，不接续已衰减至终点的旧optimizer。
- 固定原架构/DINO/历史/H64/commit12/sample预测/normalizer/BF16/native96tick TBPTT。
  每组20遍只是用户给出的上限候选，本轮预登记12遍。资源/软件错误/截止时停并保留可恢复断点；
  若某对未完成相同有效预算，不把它作为公平损失结论，不使用test决定停止或参数。
- validation选模固定为EMA采样的前12步`0.5*arm_native_MSE+0.5*hand_native_MSE`，保留H64指标；
  两种训练loss的值不可直接排名。每线最终选定后test40一次。

### Loss A/B的确切边界

`A=MaskedMean((pred_x0-target_x0)^2)`，arm总权重7/29、hand22/29。
`B=0.5*MaskedMSE(arm7)+0.5*MaskedMSE(hand22)`，hand内部22维均权。
不加入时间权重、前缀辅助loss、epsilon目标或新网络。尾部mask采用实际有效目标分母。
原canonical关节中thumb7–11五维、index12–15四维、middle16–19四维、ring20–23四维、pinky24–28五维
（这里是0-based索引）。本轮不加五指等权设计，避免再引入一个变量；逐指指标全部单列。

### 指标与原曲线缺失

每轮记录真实train diffusion目标、固定train16诊断子集与完整val40的同定义loss/采样指标、
arm/hand/五指的normalized-reference及native-command误差、rad参考误差、前12与H64、
逐未来位置、饱和、lr、梯度范数、更新量和样本数；固定val第一条的代表关节轨迹另存。
原40遍metrics没有逐epoch train loss、前12arm/hand拆分、五指或逐horizon曲线，不能补造历史。
补测只用train/validation的最终旧checkpoint；不重测旧test，不把补测的单点画成40遍历史。
EMA/raw对照限固定train/val样本，模型选择始终使用EMA。

### Transformer阶段与有界闭环

优先G01：选择验证前缀指标最好的G00基座并冻结，先以共同合法seed981001/981002/981004验证自主接触机会。
基础最低门槛：验证前缀arm/hand均优于零命令基线且有限、无接口错误；若未达门槛不伪称通过，
根据用户授权可用明确专家roll-in局部接触试验继续考察独立残差，并分开解释自主适用性。
从完整专家轨迹因果生成base计划，实际历史是teacher forcing；未来专家仅为监督。
新残差初版2层/128hidden/4heads，当前29D u_base独立token，剩余绝对参考plan token，
视觉上下文、q/qd、过去实际issued/delayed、controller target/tracking、五指力/FK/时间/valid，
近期原120Hz力窗口与每60Hz更新的持续记忆并存。频率先固定10Hz（每6tick输出6个残差）；原60Hz执行。
delta标签为同tick expert_clipped−base，允许[-2,2]；无接触逐tickexact-base，真实gate>.1N，
不能将聚合接触称工件抓握。连续力屏蔽对照保留相同真实gate和冻结base。
名义历史与混合反馈的边界已单独询问，未收到新授权，采用严格隔离：G01使用单独版本的
`private_proposals_with_native_delay_timing_v1`，名义历史只记录自身base建议及按原生延迟时间选择的base建议，
不读混合/专家动作值；真实issued/delayed只送残差。它是反事实私有建议历史，不冒称真实执行。
原G00/G10 DP对照仍保留原actual history；新G01协议须验证无修正时与实际history数值一致。
缓存中的G01名义历史同样由因果自身建议生成；残差真实历史是明确专家teacher forcing。
残差训练预算在实际缓存规模/短pilot后、正式更新前固定；至少一组真实训练优先于G11扩展。
闭环最多18次预分配启动：四条DP线×3seed=12，选定G00 base的TRT/连续力屏蔽×3seed=6；
若基础接触不足，最后6次改为前两seed×(base/TRT/连续力屏蔽)的同专家前缀roll-in，
保留局部base控制组，提前写入plan，不额外增加或替换失败启动。
旧GRU残差只作历史比较，因基座/训练目标不同不能宣称同条件优越。成功/子目标/timeout保持原生。

### 资源与交付

唯一调度器和GPU互斥短块；目标38%，准入42%，整卡批次和完整60秒上限50%；
准入空闲VRAM6GiB/RAM16GiB，硬余量4GiB/10GiB及12GB磁盘。最多两个DP trainer并行，
监测从首个GPU工作之前持续到全部子进程退出，越界记录并暂停自有工作，不靠闲置稀释历史。
RAM/显存压力时保存已有CPU断点，不在低余量时再复制大型checkpoint。STOP/截止可从完整episode边界恢复。
独立轻量曲线生成只读统计，不反复加载模型。公开仅代码/配置/MD/小JSON/CSV/必要PNG，不上传原始产物。

## 文献与实际源码核查

| 方法 | 公式/实际实现证据 | 与本项目的必要区别及本轮处理 |
| --- | --- | --- |
| [官方DP](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py#L261) | `compute_loss`支持epsilon/sample，逐元素MSE，normalizer；mask后flatten mean，明确拒绝传入valid_mask | 保留sample及原limits normalizer；我们的变长尾部必须按实际有效数归一，不能照搬固定H分母。A是本地原目标续训，B只改分组权重。 |
| [UVTA论文式7](https://arxiv.org/html/2609.34182v2#S4.SS4)及[固定源码](https://github.com/uni-vta/UVTA/tree/74f4270142cf3778a7103f04bbb922daa2597fd4) | `scripts/train.py::_weighted_action_loss`组内MSE再按归一组权重求和；`configs/flip_book.yaml`启用balanced_action_loss、action_eef=action_hand=1；训练入口确实替换backprop scalar | 论文是9D相对腕位姿+22D手指、epsilon，并有未来触觉head。我们是7D绝对arm参考+22D手指、sample，无未来触觉loss；B只借鉴均衡思路。模型类默认全维MSE不代表训练入口最终目标。 |
| [TouchWorld式7–9及附录8](https://arxiv.org/html/2607.07287v2#S8) | 名义lookahead+历史+context→残差；冻结nominal，58/120维mask、MSE与1e-4正则；H32/W16/C4，8层512/8heads | 官方项目页本次仅给论文，未取得可核对训练源码；实际Hz未由C4唯一确定。我们29D native命令、五指合力/FK，2层128/4heads、10Hz/C6，不能称复现VLA或taxel。 |
| [RDP](https://reactive-diffusion-policy.github.io/)及[固定代码](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/blob/824c5e8de1fd1811106907a04b5f0186e0138c0b/reactive_diffusion_policy/policy/latent_diffusion_unet_image_policy.py) | 先训练AT，再latent diffusion；`predict_from_latent_action`以快反馈驱动AT decoder，损失支持epsilon/sample | 是动作潜变量与快速解码，不是当前命令上加delta；本轮只借鉴慢快分工，不改DP去噪目标或冒称RDP残差。 |
| [ReTouch式3–8及附录A](https://arxiv.org/html/2608.01824v2) | 训练期未来触觉HAE监督FAE、stop-gradient latent alignment与flow matching；9Hz上下文/36Hz动作更新，更新尚未执行后缀 | 不是独立加法残差；未来触觉只允许其训练privileged分支。我们不建该分支、不伪造taxel。未取得公开训练源码，代码级实现一致性未知。 |

UVTA固定源码许可证MIT；本次只阅读、未复制实现。GitHub API触发403限流后，用稀疏只读clone核查固定提交。
TouchWorld的58维mask具体索引未在已读材料中展开，保持未知；不把论文参数直接填入本项目配置。

## 实际结果

第一阶段：真实旧epoch40 EMA权重、145个完整连续观察跨两秒边界，CPU历史前向+实际GPU DDIM采样，
使用明确teacher-forced原生ack回放；不是新自主轨迹。G00/G10条件最大差1.90735e−6/1.43051e−6，
持续状态2.38419e−7/2.98023e−7；归一化往返1.19209e−7、native command往返2.38419e−6。
两帧原三视图RGB重算DINO与冻结缓存最大差0；原split/camera顺序/权重身份通过。
旧最终权重val前8条补测：EMA前12arm RMSE0.966626/1.000643，零命令0.956411；hand0.531262/0.533749。
raw分别0.965712/1.001787及0.529436/0.531897，未发现EMA与raw巨大分离；这不是完整val40的最终比较。
原40遍没有的历史分项仍标缺失，新指标从续训前的epoch0基线开始。

19项定向测试通过（loss梯度权重、尾mask、手指分组、资源违规和既有long-horizon接口），不与旧测试数相加。
G00两条短pilot各11个真实更新完成；03:11:59 UTC开始G00正式配对训练。新root的monitor先于GPU工作启动，
独立watchdog与恢复checkpoint已实际运行，后续阶段以pipeline/status为准。最终结果未完成，不能据此宣称收益。
启动HEAD为7a592ad，但新增实现当时尚未提交；实际运行源码SHA保存在worker/config文件，不能把旧HEAD当作新增源码版本。
