# DP损失对照与Transformer触觉残差：2026-10-31

## 开始状态与预登记

接手HEAD及origin功能分支均为`7a592ad86377ef29fac8b5b694e883ecdacc24a0`，工作树干净，旧任务已退出。
本轮用户重新授权真实续训、独立Transformer残差、有界诊断、曲线及审查后的origin feature推送。
原400条、320/40/40、旧best及正式0/50保留；原controller/physics/DR/reset/reward/120/60Hz及sim wrapper只读。
本文件先于新训练代码建立，结果逐阶段追加。状态：G00-A/B各12遍已完成，G10-A/B运行最后一遍；闭环与残差待完成。

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
用户在残差缓存/训练开始前明确回复：“允许新版本读取过去实际混合命令，明确标注间接影响”。
新G01使用`past_actual_mixed_commands_v1`：名义DP和残差均读取严格t−1实际issued/delayed命令及时间。
G00名义仍不直接读取力，但新G01的名义DP会通过混合命令历史间接受到残差/触觉影响，不能称严格触觉隔离。
冻结DP缓存在专家观察上逐native tick回放，双方过去实际命令均为明确teacher forcing；
部署则来自真实学生混合命令ack。该分布差异保留，不把专家命令称为学生预测。
原G00/G10 DP对照不变。授权亦覆盖未来G11，但本轮优先G01；此前私有建议预检仅为已取代设计的工程记录。
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

官方DP同pin的`config/train_diffusion_unet_hybrid_workspace.yaml`还明确AdamW1e−4、betas(.95,.999)、
wd1e−6、cosine/warmup500、EMA inv_gamma1/power.75/min0/max.9999；
`model/diffusion/ema_model.py`的step计数与本地`update_ema`核对一致。这些是所选公开配置，不是所有DP实验通用超参。
官方该例为H16/commit8、epsilon/100步推理、clip_sample=True；本轮保持原候选H64/commit12、sample/DDIM16、
clip_sample=False及既有limits归一化。绝对arm参考可能略越限，执行经原生命令/控制器限制；
本轮不同时改这些因素。官方checkpoint monitor配置不是本轮选模规则，本轮只用固定validation前缀指标。

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

### Transformer实现检查（尚未完成训练）

独立模型/缓存/训练/控制/闭环代码已实现。单独6项测试覆盖非零head的未来隔离、batch/逐tick持续记忆、
当前base独立输入生效、force屏蔽先于memory、无接触exact bypass、clip外raw监督梯度、6/12tick窗口对齐、
私有建议与实际历史的无修正等价以及未来触觉拒绝。含DP相关项的定向18项通过，不与之前测试数相加。
真实244步示范的actual命令作为私有建议回放时，history最大差0；真实tactile输入为244×2×5×20且有限。

首次新缓存预检发现BF16将中间11步GRU合批与在线逐tick执行有数值漂移：首计划完全相同，
step12条件最大差0.014226，step24参考差0.172144rad，已执行前缀命令最大差0.100141。
失败报告保留，没有放行该缓存或Transformer训练；改为与在线相同的逐native tick冻结DP回放，再验收。
这也说明此前CPU FP32语义验收不等于GPU BF16任意分批的逐位一致；DP A/B的公共离线分批口径保持不变，
其真实控制质量仍必须由逐tick闭环检验，不能用离线下降替代。
修正后旧epoch40权重的25个真实观察（含step0/12/24三次计划）命令、reference、condition、memory、
私有history最大差均0；非零残差head的CPU batch/streaming差1.19209e−7、memory差3.57628e−7。
小残差网络训练/验证/部署统一FP32，并额外检查GPU batch/streaming，不把CPU结果充当GPU证明。
同一非零head在GPU的输出/持续状态batch-streaming最大差1.78814e−7/3.57628e−7；
独立残差实际828,189参数（不含冻结DINO/nominal）。最终选定G00权重后还会再次验收，预检不替代正式缓存验收。
进一步扩大到145个真实观察跨两秒窗口：名义命令/reference/私有history最大差仍0；
CPU残差输出/记忆差1.78814e−7/3.27826e−7，GPU差1.19209e−7/4.17233e−7，均通过预定阈值。

用户授权实际混合反馈后重新执行145步真实预检（past_actual_mixed_commands_v1）：
名义command/reference/actual-history与独立延迟重建最大差均0；CPU残差输出/记忆差
1.49012e−7/3.27826e−7，GPU差1.49012e−7/4.17233e−7。7项残差测试含真实ack接收混合命令、
错误提交/重复ack拒绝；合并DP/long-horizon共19项通过。此预检使用原G00 epoch40，
后续正式缓存仍必须用validation实际选定的G00再次验收。

### G00配对续训完成（G10及闭环仍运行/待运行）

两条均完成12完整train320遍历、9,984更新；A/B各自耗时1698.38/1709.09秒（并行共享lease，不能相加当wall）。
原epoch40 EMA的val40前12arm/hand为0.962260/0.504293，零命令0.953035/0.924980。

| G00线 | validation选定额外epoch | 选定额外更新 | 前12 arm RMSE | 前12 hand RMSE | H64 arm RMSE | H64 hand RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 8 | 6656 | 0.951643 | 0.517129 | 1.135070 | 0.728337 |
| B | 12 | 9984 | 0.917858 | 0.508648 | 1.115963 | 0.717122 |

表均为EMA、validation40、归一化native命令单位；完整训练预算相同，validation选择的checkpoint时刻可不同。
B相对A的前12arm下降约3.55%，hand下降约1.64%；相对续训起点，B的arm改善约4.61%，hand仍略差约0.86%。
不能称继续训练让每个动作分量都改善。A后段接近平台，B的arm也在末几轮趋于平台，hand仍缓慢下降。
B选定时固定train16/val40前12arm为0.809004/0.917858、hand0.449987/0.508648，存在泛化差距；
这是固定16条诊断子集，不是整个train320的评估，不能单凭该差距断言整体过拟合。
G10已接续；本小节不包含尚未生成的test或自主成功结果。
