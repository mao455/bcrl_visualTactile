# 独立policy比较：源码、论文与本轮实验分开

新增[实验组、视频与完整参数统计](EXPERIMENT_GROUPS_AND_PARAMETERS_20261030.md)及
[逐模块计数/57个上游文件SHA核验](evidence/parameter_census_20261030.json)。本次已在CPU/meta构造
固定DP/RDP/ImplicitRDP示例的完整神经网络；没有运行它们的机器人任务或取得作者checkpoint。
后文先前“未实例化/完整参数未知”范围由以下计数更新，论文未公开实现仍保持未知。

四组 canonical 语义为 G00=B0/A、G10=B1、G01=Residual B、G11=尚未执行。
本轮 C 仅是保留真实触觉 gate 的连续力屏蔽诊断，不能替代 G11；因此报告不伪造 G11 参数、训练量或闭环结果。

本轮实际训练/执行已结束：Residual B/C各5完整epoch、7,875有效更新；一次test加权command MSE
A .659997→B .336392/C .337817，但自主各0/2且完全无gate。12个预分配启动中11条自然timeout、
1条switch核验中止；唯一完整严格匹配的局部981002三联也全失败，无首子目标/抬升。
详见[完整训练、交接失败与资源证据](CORRECTION_TRAINING.md)。这支持“离线监督学会了部分命令映射”，
不支持触觉收益、闭环成功或RL已经实现；严格action gate确实阻止了无接触接近阶段的任何修正。

本报告读取固定版本的policy、训练workspace、task配置和真实部署runner；外部只做构造与参数核算，
没有执行其训练/任务或下载checkpoint。源码选择不等于作者所有实验的最终resolved配置。没有公开/没有实际读取的
参数和实现明确记为未知，不以摘要或名称推断。参考文件留在本轮ignored `references/`，不vendoring新代码。
主要源码和论文的SHA256及补充README版本见[evidence/policy_sources_20261030.json](evidence/policy_sources_20261030.json)。

## 来源与生效配置

| 方法 | 固定来源 / 本次实际检查入口 |
| --- | --- |
| DP | [5ba07ac](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)：`train_diffusion_unet_ddim_hybrid_workspace.yaml`、`train_diffusion_unet_real_image_workspace.yaml`、image/hybrid policy、workspace、normalizer、sampler、`eval_real_robot.py` |
| RDP | [824c5e8](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/tree/824c5e8de1fd1811106907a04b5f0186e0138c0b)：`eval.sh`当前启用peel/wrench LDP24fps+at_peel；AT/VAE、latent policy、workspace和`env_runner/real_runner.py` |
| ImplicitRDP | [112d685](https://github.com/Chen-Wendi/ImplicitRDP/tree/112d6857d914c5dee4c4c172790088adc6bdc3a5)：`eval.sh`当前启用flip/wrench implicitrdp10fps、v_prediction；reactive transformer配置、policy、workspace和`real_stable_reactive_runner.py` |
| OmniTacTune | [论文v1](https://arxiv.org/html/2607.03723v1)，特别是§3.3与附录A.3/A.5/A.8；[作者项目页](https://colinyu1.github.io/omnitactune-site/)本次未找到可检查的官方训练实现。以下为论文描述，非源码确认 |
| 我方 | 原正式B0/B1 best与当前`visuotactile`源码；本轮Residual v1独立实现，实测见[训练报告](CORRECTION_TRAINING.md) |

**DDIM8不是bug判据。** 官方DP的`train_diffusion_unet_ddim_hybrid_workspace.yaml`明确8，
real_hybrid/transformer_real_hybrid也有8；RDP `eval_real_robot_flexiv.py`把DP和LDP分支设成8。
DP real_image训练yaml写100，但该仓库real部署脚本override成16；不能只读yaml或把一种示例当全项目设置。
ImplicitRDP transformer部署override成5，UNet分支8。当前B0使用8有明确同类先例，增加步数不自动修复能力。

## 传感、权限、历史和动作

| 维度 | DP | RDP：当前peel/wrench示例 | ImplicitRDP：当前flip示例 | 我方B0＋Residual v1 |
| --- | --- | --- | --- | --- |
| 视觉/本体 | RGB+task指定lowdim；obs2窗口 | 腕RGB240×320、TCP xyz、夹爪width、6D wrist wrench | 腕RGB360×640、TCP pose、6D wrench；ResNet34 | 三448×252 RGB-D、q/qd58、外部任务/goal；五指合力/FK仅给Correction |
| 触觉进入何处 | 所选示例无触觉 | wrench既在普通obs，也在AT的extended temporal condition；其他task有GelSight/PCA15 | wrench进观察与temporal RNN，直接调制扩散生成 | B0禁止；Correction自己的逐指GRU。B1本身读触觉，不能当视觉nominal |
| 历史 | 固定obs窗口；源码assert不支持past_action | obs2（dataset downsample2），RNN解码新的力历史；无显式past_action | 2个视觉观察+chunk内增长的force prefix；源码assert无past_action | B0完整episode持续记忆但own-plan/issued均none；Correction持有真实t−1 clipped command、target/tracking及独立memory |
| 计划 | 原生预测action chunk | 先预测latent action，再用新力解码动作；不是简单加性residual | 缓存视觉tokens和同一initial noise，重新用force prefix扩散；不是我方外置delta | 当前/未来计划来自冻结B0真实推理，绝不以专家未来动作替代；剩余chunk及valid进入intent GRU |
| 动作 | task相关；real PushT是2D绝对EEF位置，其他含rotation6/gripper | 当前peel action4=xyz+gripper；dataset.relative_action=True，非29D关节命令 | 存储19D TCP/virtual target/rotation6/stiffness，model13D改RPY；eval.sh action_type又override为right_arm_6DOF | 原native29D [-1,1]：7 arm velocity-style积分、22 hand绝对映射+平滑；delta不能解释为统一笛卡尔位移 |
| 归一化 | 训练集LinearNormalizer，预测反归一化；RGB crop/GroupNorm/ImageNet scaling | action/lowdim/latent normalizer；AT act_scale1；RGB随机crop；PCA版本另按task | action/lowdim normalizer，relative/RPY变换；图像旋转/ColorJitter，非我方默认 | B0训练集q/qd mean/std、RGB[-1,1]、metric depth/2m+valid；触觉signed log1p N；targets/tracking保留rad；命令本就[-1,1] |
| 特权/监督 | 依任务所选观测，不自动等于可实机传感 | 实机RGB、pose/wrench/width | 实机RGB、pose/wrench；virtual-target/stiffness标签不是我方执行空间 | 物体/goal真值、reward、stage、扰动只诊断；专家当前命令仅监督，过去专家执行明确teacher forcing |

所有方法的“历史”不可混称同一能力：固定窗口、chunk内force prefix、完整episode持续GRU、实际提交/延迟反馈、
私有计划是不同信息。RDP/ImplicitRDP公开runner没有我方trace2式native ack契约；未知硬件底层反馈不应补写为已实现。
我方v3 issued是独立变体；旧CorrectionContext没携带其完整allowlist，且v3禁止mixed覆盖。新Residual明确拒绝v3，
用v2 B0和Correction自己的实际ack；未通过静默加字段让二者“兼容”。

## 时钟、模型和实际优化

| 维度 | 官方DP示例 | RDP示例 | ImplicitRDP示例 | 原B0/B1与本轮残差 |
| --- | --- | --- | --- | --- |
| H / commit / 秒数 | H16；hybrid action8。real runner默认10Hz、每6步inference：1.60s计划、.60s提交；hybrid仿真Hz须读任务metadata | H32@24Hz=1.333s，obs2×downsample2后可用29步；planner inference6Hz，latent更新interval16 native ticks=.667s；力条件解码24Hz。不能把6Hz当control | H16@10Hz=1.60s；obs2→可用15步；vision/noise cache每6tick=.60s刷新，force-conditioned重采样10Hz | B0 H12@60Hz=.20s、commit6=.10s；全部60Hz观察更新memory；Correction每60Hz一次，触觉两120Hz样本。原物理120Hz不变 |
| 网络尺寸 | hybrid UNet[256,512,1024]；real_image[512,1024,2048]；ResNet18 | UNet[512,1024,2048]、ResNet18；AT conv32、RNN32、peel latent4（其他task8）、n_embed16 | Transformer d768×7层×4heads、ResNet34，force RNN512×1 | B0/B1 denoiser[64,128]，visual GRU128；2,003,085 / 2,166,613参数。Residual每个1,344,597总 / 489,125可训练 |
| 神经参数量（指定示例） | hybrid lift89,317,578；real PushT328,915,202 | LDP198,169,296＋独立AT8,588＝198,177,884 | flip/wrench95,006,541；作者checkpoint未知 | B0 2,003,085；B1 2,166,613；Residual1,344,597，其中训练489,125；完整B0＋Residual3,347,682 |
| loss | masked epsilon MSE，取决于scheduler prediction_type | AT动作重建+KL1e-6（当前use_vq=False）；LDP latent epsilon MSE | train_implicitrdp.sh与eval.sh均指定v_prediction；workspace调用compute_diffusion_loss，13D输出含辅助标签 | 原epsilon MSE；新残差监督clip后的command，contact mask，arm/hand各.5，delta L2 1e-4 |
| optimizer / EMA / LR | AdamW1e-4、wd1e-6、betas(.95,.999)；EMA、cosine、warmup500 | AT AdamW1e-3/wd1e-4、cosine/warmup100、无EMA；LDP AdamW1e-4/wd1e-6、cosine/warmup500、EMA | 所选脚本model LR1e-4覆盖YAML默认3e-4；backbone独立3e-5、wd1e-6、betas(.95,.999)；EMA、cosine、warmup2000 | 原AdamW1e-4无EMA/schedule；本轮CPU FP32 AdamW1e-3/wd1e-4、gradclip1，无EMA/schedule，显式有限基线 |
| 训练量 / 顺序 | batch64随机window/shuffle；hybrid配置3000epochs、real600；实际有效更新数未知 | batch64/shuffle，AT601、LDP401配置epochs；实际有效更新数未知 | batch64/shuffle、配置600epochs；actual checkpoint有效更新数未知 | 原batch2×TBPTT32、50完整epochs，各61,050更新，3,241,250有效控制时刻（含重复）；固定episode顺序。本轮pilot16/val8后最多5完整epochs，B/C匹配顺序与更新 |
| 冻结 / 梯度 | 所选配置端到端；EMA另存影子权重 | AT是非nn.Module容器、另加载权重，LDP optimizer不注册AT；`.eval()`本身不是冻结梯度，latent loss路径未显式detach AT输出。本次不替作者声称全部梯度截断 | force encoder、视觉和Transformer共同优化；不是冻结视觉base外接残差 | B0 requires_grad=False/no_grad；独立复制已训练frame encoder冻结，四路GRU/MLP可训练；head零初始化 |
| 噪声 / 缓存 | 高斯扩散噪声；重新采样chunk | latent计划更新；新力不断解码，不是每tick重新抽latent | vision tokens与initial noise缓存跨6tick，新force改变扩散输出 | B0每6步私有RNG新噪声；cache每示范固定一个私有seed和一条完整采样序列。跨epoch重用训练过的冻结特征/真实计划，不是随机latent |

batch2×TBPTT32是两个相关episode流中的64个control observation loss位置，不等于64个独立random windows，
也不能直接称window64的32倍训练量。需要同时报告独立episode数、有效观察数、有效target数、optimizer更新和遍历次数。
原模型无EMA/schedule、固定顺序、较短H和较小网络是明确差异；本轮没有将它们未经对照定性为唯一根因。

精确生效配置补充：RDP非VQ的AT conv latent4经Gaussian projection变为16通道，H32经过两次
stride2得到8；实际latent diffusion为8×16、kernel3，不能照yaml的dummy kernel5计数。
ImplicitRDP存储360×640图在encoder内变为224²，产生102×768观察tokens；其3e-5仅用于
视觉backbone，projection按train_implicitrdp.sh覆盖后为1e-4。RDP非timm路径LR为1e-4乘accelerator进程数，实际进程数未知。
我方AdamW beta1=.9，三类公开DP示例为.95；DP hybrid的10fps字段是录像fps，真实控制Hz依dataset metadata。
以上新增事实及计数环境/未知范围见完整参数报告，不据容量倍数推断任务表现。

## OmniTacTune：相似目标，不同训练与门控

以下是论文v1明确描述，未由公开代码验证：冻结视觉base；先12分钟自主base rollout初始化replay、
训练双critic并适配触觉encoder，然后SAC更新residual。critic为三层MLP；触觉latent32，marker用两层MLP，
预训练触觉backbone冻结、projection/adapter更新。包含flow/keypoint信息，与我方五指合力/FK不同。

门控作用于**触觉feature** `g*z_tactile`；动作是`base + scale*residual`。这不蕴含无接触时残差为零。
我方严格action gate是本轮用户指定的更强限制，不能说与论文等价。论文EEF frame归一化[-1,1]、10Hz，
flow chunk K8约.8s、最长300tick。SAC batch256、每step一次update、actor/critic LR3e-4、encoder1e-4、
gamma.99、target tau.005、replay1e4；scale按500 online steps从.05→.10→.15。
这些scale不能照搬到我方7 arm+22 hand混合命令。论文warm-start每条生成两条ControlTac增强轨迹；
我方不生成触觉、不改变原数据，没有critic/replay/新reward，因此**本轮是监督残差，不是已完成RL**。
论文总训练40–80分钟、任务结果来自实机；本轮墙钟/成功率不应直接与它比较。精确参数量、optimizer实现、
EMA、执行ack、全局历史长度及实际checkpoint配置在已读公开材料中未知。

## 补充方法：只保留与当前决策有关的差异

- [TouchWorld](https://arxiv.org/pdf/2607.07287)：该无版本URL本次返回v2（2026-07-09），不是暗中当v1。
  附录明确nominal冻结，TRT d512/8层/8heads，H32、lookahead W16、commit C4，120D动作仅58D子空间纠正；
  AdamW1e-4/wd1e-4，corrected-action masked MSE+1e-4正则。nominal本身是**visuo-tactile goal-conditioned**，
  还有规划/world-model/上下文token，违反我方tactile-free nominal边界，不能整套移植。图示频率不能代替Hz/commit实测。
- [ViTaL：Touch begins where vision ends](https://github.com/Exiam6/ViTaL/tree/4f349cfc5d33d0155b6bd66680d9daf401af6449)：
  视觉背景增强、BC encoder和真实visuotactile residual RL，JAX实现；不是同名的视觉token论文或inference-time steering。
  本轮只核对官方README与论文/项目定位，未运行其训练配置，精确参数/更新量未知。
- [T-Rex](https://github.com/ZhuoyangLiu2005/T-Rex/tree/ba4047ce3e9f3d2013d60de600e136897e90ccb8)：
  Qwen3-VL-2B基础上的variable-rate MoT、temporal tactile VQ-VAE、约5Hz action/20Hz tactile refinement，
  100小时数据与多阶段BC/flow matching；不是GRU外挂或本轮已实现的RL。main与full-pipeline发布范围不同，
  没有在本机复现实验或核算完整参数，不能把“大模型触觉分支”直接当本项目效果证据。
- [ResiP](https://github.com/ankile/robust-rearrangement/tree/cf80d12d4acae9e4751a49407a96f85f07b342e3)：
  frozen chunked BC+on-policy residual RL，利用仿真任务信息与后续视觉distillation；不是触觉特征门控方法。
  稀疏成功reward与PPO路线说明RL残差可与冻结DP分开，但不能把它的状态权限/物理任务等同我方。
- [ResFiT](https://github.com/amazon-far/residual-offpolicy-rl/tree/66262b06a0caf159fb001504949b7b48ccda5f64)：
  官方入口是residual TD3/off-policy finetuning，需replay/critic与交互；不是仅训练expert−base就完成SAC。
  本轮未移植这些组件，未声称相关RL收益。补充仓库只读取README，配置/参数/实际运行量未知。

## 对当前代码的判断

已证实的缺口是旧Correction没有目标、训练checkpoint或真实执行链，且输出完整candidate而非delta；
本轮实现已补齐，不把B1重命名为视觉nominal。旧数据没有真实nominal计划，需要冻结checkpoint因果派生，
不能沿用空计划或示范未来动作。无接触阶段错误受strict gate限制：残差无法修复尚未接触前的接近失败。
离线teacher-forced残差学习与学生闭环之间存在分布偏差；即使验证改善，也必须看自主与真实roll-in各自结果。

如果监督残差已降低离线误差但闭环仍失败，候选是明确版本的在线专家纠正/DAgger，或冻结base的SAC
（另实现critic/replay/reward、资源和干预协议）。若主要失败在接触前，应另做固定H12权重commit6/12等
计划持续性实验；本轮12条已分配A/B/C，自主不擅改严格门控、动作空间或原生控制以取得成功。
