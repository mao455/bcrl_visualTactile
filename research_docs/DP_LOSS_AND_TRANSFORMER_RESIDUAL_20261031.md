# DP损失对照与Transformer触觉残差：2026-10-31

[公开图表与CSV索引](evidence/dp_loss_trt_20261031/README.md)

## 开始状态与预登记

接手HEAD及origin功能分支均为`7a592ad86377ef29fac8b5b694e883ecdacc24a0`，工作树干净，旧任务已退出。
本轮用户重新授权真实续训、独立Transformer残差、有界诊断、曲线及审查后的origin feature推送。
原400条、320/40/40、旧best及正式0/50保留；原controller/physics/DR/reset/reward/120/60Hz及sim wrapper只读。
本文件先于新训练代码建立，结果逐阶段追加。本轮预登记执行已全部结束，GPU阶段于2026-10-31 05:01:00 UTC
（13:01:00 UTC+8）退出；四线DP、force/masked残差、各自最终test及18条原生诊断全部完成。
Loss B改善动作误差，但任务仍未成功；残差离线改善主要在arm，连续力输入未显示优势。
闭环只有保存物理初态匹配，接触前命令已不同，不能用轨迹差异单独建立残差/触觉因果收益。

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
无接触exact-base指当前输出严格等于当前名义预测；过去残差可经历史改变后续名义计划，
因此这不保证其整条轨迹与单独运行冻结base相同，闭环效应包含该反馈路径。
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
独立watchdog与恢复checkpoint当时已实际运行；这是启动历史快照，当时尚无最终结果，不据启动状态宣称收益。
启动HEAD为7a592ad，但新增实现当时尚未提交；实际运行源码SHA保存在worker/config文件，不能把旧HEAD当作新增源码版本。

### 训练前的Transformer实现检查（历史记录）

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

### 中间记录：G00先完成的配对结果

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
此小节记录G00先完成时的数据；四线最终test与闭环结果见后续章节。

### 四线续训与一次性test已完成

四条训练线均完成预登记12完整遍历/9,984次额外更新，总39,936次正式更新；另每线pilot11次。
G10-A/B分别耗时1851.30/1862.41秒，均选额外epoch12（9984更新）。两种损失内选模后，
G00与G10的统一validation前缀指标均选B；冻结G00-B供残差阶段使用。

| 训练线 | 选定额外epoch | val40前12 arm/hand | test40前12 arm/hand | test40 H64 arm/hand |
| --- | ---: | --- | --- | --- |
| G00-A | 8 | 0.951643 / 0.517129 | 0.952267 / 0.523374 | 1.141666 / 0.726647 |
| G00-B | 12 | 0.917858 / 0.508648 | 0.921222 / 0.509035 | 1.118779 / 0.713937 |
| G10-A | 12 | 0.951744 / 0.499069 | 0.967824 / 0.504618 | 1.140970 / 0.710218 |
| G10-B | 12 | 0.908257 / 0.496741 | 0.933257 / 0.501724 | 1.117516 / 0.707535 |

均为选定EMA、归一化native命令RMSE。离线转换使用专家真实controller target作参照，
未来target仅参与指标还原，不进入模型；因此该表不是自主执行误差或预测未来controller target的结果。
test RNG与validation固定但不同；四线test各一次，未据test调参。零命令test前12 arm/hand为0.956817/0.921994，
H64为0.955774/0.921256。B相对A的test前12arm改善G00约3.26%、G10约3.57%；hand约2.74%、0.57%。
两组B的arm前缀均优于零命令，H64 arm仍差于零命令，远期动作精度不足。

G10-B的val前12arm从起点0.962992降至0.908257，hand从0.501030降至0.496741。
末三遍arm0.910361→0.908408→0.908257已明显趋平，hand0.501194→0.498640→0.496741仍缓慢下降。
G10-A/B选定权重train16前12arm为0.866395/0.803964，val40为0.951744/0.908257；
存在泛化差距，但末期validation没有持续反弹，不能把全部问题归为过拟合或简单增加训练遍数。

一个控制尺度上的关键限制：原arm转换每tick满幅参考增量仅1.5×(1/60)=0.025rad。
G00-B test前缀arm参考RMSE仍0.036111rad，除该尺度后未clip命令误差约1.44；
因此“弧度误差只有几百分之一”并不意味着命令足够精确。实际clip后前缀RMSE仍0.921222。
本轮没有为改善表面指标改原action scale、积分/平滑或120/60Hz；未来若研究前缀权重/动作辅助loss，须单独对照。

预测位置诊断同样显示误差随未来位置变差：G00-B test arm在offset0/11/31/63分别为
0.840325/0.990436/1.165412/1.259874，G10-B为0.855247/1.005582/1.157040/1.263452。
即使在实际commit12内，也不能把首位置的改善外推为整段动作精确。四线全H64/29D命令饱和率分别
43.16%、42.50%、43.24%、42.87%，含合法专家接近限幅的动作，单凭饱和率不能判错误方向。
G00-B五指test前缀RMSE（thumb/index/middle/ring/pinky）为0.625221/0.525188/0.600523/0.246491/0.437547，
G10-B为0.617369/0.511242/0.593221/0.255248/0.428616；拇指与中指误差仍明显。

### DP真实闭环：12条全部自然结束

三共同合法seed按981001/981002/981004排列，四线各0/3；全部601步原生timeout、0子目标、无原生lift标记。
12条保存初态不仅本轮逐项一致，与旧epoch40对应初态也完全一致；两次evaluator均用seed+1000000的私有采样种子。
不声称渲染图像或轨迹逐位一致，不把这些诊断加进旧正式0/50。有限值、60/120Hz、原始plan/命令对齐、
专家prefix和完整601帧G10-B视频解码已逐artifact核查。

| 线 | 观测接触步数（三seed） | 最长连续接触步数（三seed） | 物体进展 |
| --- | --- | --- | --- |
| G00-A | 2 / 6 / 152 | 1 / 6 / 46 | 无正向reset-relative高度提升 |
| G00-B | 7 / 4 / 336 | 4 / 4 / 139 | 无正向reset-relative高度提升 |
| G10-A | 69 / 0 / 253 | 28 / 0 / 51 | 无正向reset-relative高度提升 |
| G10-B | 41 / 62 / 465 | 9 / 50 / 267 | seed981004最高+0.076863m，高于5cm最长0.30s，随后掉落 |

G00-B在第三seed更接近且接触更久，前两seed仅4步连续接触；不能称已稳定掌握接近/抓握。
G10-B第三seed step216达到高度峰值，step231降回reset高度+2cm以下；仍无稳定保持/插入。
原G10 epoch40同seed曾+0.107093m并触发原生lift，最终fall；本轮B没有重现该原生lift，
因此更低离线误差没有转化成可靠任务进展，不能声称续训已解决抓取。
新12条实际composer/world外力与力矩记录均为0；不构成有外力暴露的鲁棒性证据。
G10-B掉落前后确有接触消失和名义动作变化（step216/228的29D命令变化RMS约0.607/0.509），
但仅为时间关联；不能据此区分主动松指、滑脱或重规划冲击。旧10.7cm轨迹缺actual-wrench日志，旧掉落原因仍未知。
五指合力没有碰撞对象身份；录像显示工件移动，仍不能把全部gate开启都当工件有效抓握。
触觉槽位顺序已按metadata核实为index/middle/ring/thumb/pinky，区别于thumb先行的动作维度分组。
峰值附近主要为thumb传感slot3读数，step228时五指读数均0。事件PNG/CSV分别记录观察前力、动作后高度、
步内实际外力和arm/hand命令变化，明确时间口径，不能把所有量误当同一采样瞬间。
政策前向median约20ms、p95约106–112ms，且有渲染/资源等待；60Hz是保持不变的仿真时钟，不能宣称整套栈实时60Hz。

冻结G00-B满足预登记离线前缀优于零命令及自主连续观测接触至少3步的门槛。
因此残差最后6次固定为force/masked各三seed自主；本轮不额外加专家roll-in或G11。
实际选定G00-B的145步缓存/在线验收通过：名义command/reference/history/延迟重建差0；
GPU残差输出/记忆差1.49012e−7/4.76837e−7。该验收先于400条缓存与残差训练，最终结果如下。

### Transformer实际训练与离线结果

400条冻结G00-B缓存全部完成，保留320/40/40。train320共64825个native时刻，其中37224个接触时刻（57.42%）。
force与masked各先跑4 train/4 val完整episode pilot、11更新，再独立从相同初始化训练5完整train320遍历，
每遍746个有接触标签的更新，共3730更新；不含pilot权重。无接触chunk仍递推记忆，只不反传接触目标。
两组都按validation接触组合命令的arm/hand等权MSE选epoch5，之后各test40一次。
优化器AdamW 3e−4、betas(.9,.999)、wd1e−4，新cosine/warmup50；FP32、dropout0、无EMA。
DP与残差均为单episode顺序TBPTT最多96tick；DP每更新最多8个H64 anchor，残差最多16个六步anchor，
跨chunk仅detach，episode结束才清空记忆。正式残差参数828189，不含冻结DP/DINO。

| 相同缓存的接触时段 | validation40 arm/hand RMSE | test40 arm/hand RMSE |
| --- | --- | --- |
| 冻结G00-B base | 0.949837 / 0.578360 | 0.947673 / 0.587045 |
| force Transformer | 0.714952 / 0.568294 | 0.711644 / 0.577887 |
| masked Transformer | 0.711823 / 0.566785 | 0.710323 / 0.576170 |

均为组合后native命令单位。force相对base的test接触arm改善约24.91%、hand约1.56%；
masked的两项略好，不能宣称连续力特征带来收益。只有一个配对训练随机种子，不作统计显著性推断。
本表base使用逐native tick、每episode固定采样种子的残差缓存；与DP表的BF16分批/采样种子口径不同，
不得跨表直接相减。双方在同一缓存上比较，未来专家命令/力仅用于监督或评价mask，不进入输入。
两组末期validation仍缓慢下降并趋平，未见持续反弹；force固定train16/val40接触arm为0.665095/0.714952，
hand为0.528791/0.568294，存在泛化差距。离线拟合不等于学生自主分布下可靠控制。

test40有效8399步，接触4848、非接触3551，触觉有效率99.5238%；force/masked原始delta平均绝对值
0.132347/0.137681，接触执行组合裁剪比例30.3972%/30.0323%，错误方向饱和3.1567%/3.0478%。
这些裁剪只发生在执行组合，raw delta监督仍可反传；raw幅度包含不会执行的非接触预测，不能当实际施加幅度。
旧GRU＋MLP使用不同B0基座、时域、历史和训练目标，保留作历史参照，未进行可归因的架构优劣比较。

### Transformer自主结果与配对限制

六条均自然601步timeout、零子目标、无原生lift或正向reset-relative高度提升；force/masked各0/3。
同一seed的冻结base数据来自本轮DP诊断，未额外增加启动。三seed顺序仍为981001/981002/981004。

| 模型 | gate步数 | 最长连续gate步数 | 实际施加修正平均绝对值（全601步） |
| --- | --- | --- | --- |
| 冻结base | 7 / 4 / 336 | 4 / 4 / 139 | 0 / 0 / 0 |
| force | 6 / 10 / 191 | 4 / 6 / 83 | 0.000597 / 0.001107 / 0.033785 |
| masked | 5 / 13 / 182 | 4 / 8 / 42 | 0.000489 / 0.001514 / 0.030470 |

残差确实获得控制机会；前两seed很稀少，第三seed的force有191步、最长83步（1.38s）。
没有观测到更稳定抬升或任务成功，不能把它描述成“完全没有gate所以没测到”。
但这些接触差异不能严格归因于残差：保存物理初态全部差0，首次接触之前命令却已经不同。
force/masked相对base的接触前最大命令差分别为0.793103/0.711663、2.0/2.0、1.222656/1.030016；
force与masked之间分别为1.018870、2.0、0.966797。全部strict_precontact_match=false。
原run中的paired字段只表示保存初态/切换状态检查，不表示相机输入或执行前缀逐位相同。
本轮没有保存这些时刻的完整模型输入快照，无法事后定位具体观察/渲染/数值来源，不能武断归咎于某一项。

为检查封装路径，使用同一段真实观察、已核实DINO缓存、同一私有采样种子，以及实际训练后的残差权重：
13步跨一次重规划时，独立DP与残差封装内部DP的command/plan/condition最大差全0；这不替代真实轨迹配对。
最终force/masked权重145步CPU batch/streaming输出最大差5.96046e−7/3.57628e−7，记忆差均5.96046e−7。
18条实际trace的时间/命令/plan/残差cursor、无接触当前base精确bypass、过去实际混合命令反馈均通过核查。
DP原有BF16合批训练与逐tick推理仍不保证数值等价；残差缓存已按实际部署逐tick生成，不能将其验收推广成所有DP路径零误差。
残差refresh前向p95为2.27–2.65ms，整套policy median22.76–27.12ms、p95 121.18–136.81ms；
支持本次10Hz仿真反馈安排，不宣称整套60Hz墙钟实时，也没有运行20Hz版本。

### 资源、退出与保留项

固定窗口起点03:01:47 UTC，GPU流程/monitor在05:01:00 UTC结束，用时1h59m13s，未重置6小时deadline。
监控覆盖全部GPU阶段，持续6650.96s；整卡时间加权均值33.1988%，最大完整60秒均值39.4259%，超限样本0、缺测0。
最低空闲VRAM10.6592GiB、最低可用RAM28.8657GiB，余量均满足要求；最终磁盘约119.0GB空闲。
磁盘在每次自有lease准入检查12GB余量，未记录连续磁盘最低值。软件节流不是硬件配额，也不提供逐进程GPU归因。
各worker锁/准入等待累计2491.38s、节流1014.81s；并发等待重叠，不能与工作时间简单相加当总wall。
独立监控、owner、训练、缓存、仿真和额外同输入复验进程均退出；最终CPU artifact审计通过。
早期BF16缓存合批预检失败记录完整保留，修正后才放行残差缓存；正式训练和18条轨迹无软件中断或重试。
原400 manifest/HDF5身份、B0/B1 best、原正式0/50报告、原G00/G10 epoch40权重保持不变；原生代码及sim wrapper无diff。
未运行G11、20Hz残差、新专家roll-in或额外诊断；这些不混入已完成结果，不自动追加训练/试验。

## 最终12问

1. **损失与依据：** A为有效位置29维sample/x0 MSE；B为arm7/hand22组内均值各50%。官方DP支持A的逐元素目标，UVTA支持组间平衡思路，动作空间差异见文献表；TouchWorld支持冻结基座残差思路，RDP/ReTouch并非同一种加法残差。
2. **权重与五指：** A总arm/hand为7/29、22/29；B为1/2、1/2。五指未等权，B中thumb/pinky各5/44，index/middle/ring各4/44，逐指误差独立记录。
3. **训练与选模：** 四线各额外12遍/9984更新；G00-A选额外epoch8，其余选12。两个残差各5遍/3730更新，均选epoch5；全部按预登记validation标准选，test不参与选择。
4. **曲线判断：** 机械臂末期趋平，手指还有小幅下降，存在train/val差距；没有证据支持“只需无条件多训”或“已明显持续过拟合”的单一解释。
5. **前12步误差：** B相对A的test arm改善G00约3.26%、G10约3.57%；hand约2.74%、0.57%，但H64 arm仍差于零命令参照。
6. **任务进展：** 四线均0/3；部分接触更久，G10-B短暂+7.69cm后掉落，没有稳定保持、原生lift或插入。
7. **残差输入：** 当前29D native base独立token、剩余H64绝对参考、冻结视觉上下文、q/qd、target/tracking、真实过去issued/delayed、五指力/FK/时间/valid；近期高分辨率窗口与持续GRU并存。新G01允许混合命令历史，名义DP间接受残差影响。
8. **残差是否更好：** 两组真实训练完成，离线arm明显改善、hand改善小；masked略好，未证明连续力收益。自主均失败，也不能与不同基座的旧GRU作架构因果比较。
9. **是否真正控制：** force在三个seed分别6/10/191步开启gate并施加非零修正，无接触时精确返回当前base；未观察到抬升收益，接触差异受非严格前缀配对限制。
10. **接口验证：** 真实数据、缓存、checkpoint、动作往返、因果历史、训练后递推和实际ack均有数值核查；DP BF16分批差异与闭环输入/前缀不等价保留为明确限制，不宣称全链路逐位一致。
11. **资源与预算：** 整卡均值33.20%、最大完整60秒39.43%，VRAM/RAM最低余量10.66/28.87GiB，缺测/资源越界0；GPU工作在两小时内结束，自有进程已退出。
12. **交付位置：** [全部图表/CSV索引](evidence/dp_loss_trt_20261031/README.md)、[最终去敏统计与逐artifact审计](evidence/dp_loss_trt_20261031.json)、[origin功能分支](https://github.com/mao455/bcrl_visualTactile/tree/feature/trained-tactile-residual)。本地视频索引为`outputs/visuotactile/run/dp_loss_trt_20261031_01/README.md`，权重/轨迹/视频不上传。
