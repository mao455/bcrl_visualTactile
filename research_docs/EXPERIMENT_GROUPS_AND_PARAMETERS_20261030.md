# 实验组、观看视频与DP完整参数对照（2026-10-30）

本轮沿用已完成权重，新增独立观看录像与CPU参数统计，没有重新训练。
先前正式B0/B1仍各0/50；残差诊断仍为12次启动、11次自然超时和1次交接校验中止。
离线误差降低没有转化为自主或局部任务成功，不能宣称触觉因果收益。
本轮视频的实际结果和资源统计在完成段记录，不混入以上分母。

## Canonical 四组映射

用户定义的组名与本轮已有实验必须分开标记，避免把诊断控制误当成新对照：

| canonical 组 | 定义 | 本轮对应物 | 状态 |
| --- | --- | --- | --- |
| G00 | DP 不直接读触觉，外接 Correction 关闭 | B0；录像别名 A | 已训练、已录像 |
| G10 | DP 直接读触觉，外接 Correction 关闭 | B1 | 已训练、已有独立 B0/B1 录像 |
| G01 | DP 不直接读触觉，冻结 B0 后接触残差读取触觉 | Residual B | 已训练、已录像 |
| G11 | DP 直接读触觉，并同时接触触觉残差 | 当前没有合法 checkpoint/训练/录像 | 未执行 |

本轮 C 是“屏蔽残差网络的连续力特征、但保留真实力 gate 和 contact mask”的诊断控制，
因此不等于 G11，也不是严格无触觉组。不能把 B1 与 B 的权重拼接后声称完成 G11；
G11 的数据、冻结边界、训练量和闭环分母均为空，不进入本轮成功率比较。

## 每组究竟是什么

| 组 | 学习/执行内容 | 力输入 | 数据与权重 |
| --- | --- | --- | --- |
| B0 / A | 无触觉视觉本体DP；A是同一B0零残差 | 无 | 原400成功示范固定320/40/40；训练50轮，validation选49轮best |
| B1 | 视觉、五指力/FK、融合GRU共同预测完整命令的DP | 网络直接读取 | 同划分、同50轮预算、独立参数；best49；不是视觉nominal，也不是外接Correction |
| B | 冻结B0＋独立训练的显式加性残差 | 真实力及其派生特征 | 同train320/val40；5轮，B验证选epoch5；test40仅一次 |
| C | 冻结同一B0＋匹配残差架构；屏蔽网络可见力/派生特征 | **真实力仍决定gate与训练mask** | 与B相同初始化、顺序、有效更新数、epoch5；参数独立，保留FK/时间 |

A不训练新模型。录像runner为了匹配同一执行通路仍加载/计算残差B，但显式bypass，
最终执行命令就是B0；这个额外计算成本不应混同数学上A仅需B0的参数量。
C不是严格无触觉组；B比C的差异不能仅凭一个训练seed就称触觉收益。
v3 issued历史短拟合是独立诊断变体，未被本次视频或正式best静默采用。

400条是审计后的专家**自然成功示范**，不是400次独立原生reset成功率。
训练监督是29D `command_clipped_pre_delay`，非joint target、非实际延迟后命令、非关节角度。
所有模型的q/qd来自编码器；物体/目标真值、阶段、reward只用于诊断。
没有做示范去预加载或通过修改原物理/任务条件改善结果。

## 有效训练量与离线结果

| 项目 | B0 | B1 | B / C（各自） |
| --- | ---: | ---: | ---: |
| 独立train / val / test episode | 320 / 40 / 40 | 同左 | 同左 |
| 完整训练轮数 / 完整episode访问次数 | 50 / 16,000 | 50 / 16,000 | 5 / 1,600 |
| 全部训练optimizer updates | 61,050 | 61,050 | 7,875 |
| **实际部署checkpoint**轮数 / updates | 49 / 59,829 | 49 / 59,829 | 5 / 7,875 |
| 全部训练有效控制观察（含重复） | 3,241,250 | 3,241,250 | 324,125 |
| 有效未来目标位置（含重复） | 37,839,000 | 37,839,000 | 当前contact命令186,120 |
| batch / TBPTT / precision | 2 / 32 / BF16 | 2 / 32 / BF16 | 1 / 32 / CPU FP32 |
| loss | epsilon MSE，按有效未来位置mask | 同左 | contact-only，arm/hand各.5组内MSE＋1e-4 delta正则 |
| best validation | .176505 epsilon MSE | .147740 epsilon MSE | B .339453 / C .340667 command MSE |
| 一次test | .177786 epsilon MSE | .148961 epsilon MSE | B .336392 / C .337817 command MSE |

Residual pilot是独立train16/val8完整episode、2轮、各160更新；这些权重没有继续用于full，
不能把pilot更新加到最终checkpoint的7,875。B/C每轮64,825观察、37,224 contact、1,575更新。
test40为8,399观察/4,848 contact；val40为8,670观察/5,059 contact。
batch2×TBPTT32是两个相关episode流中的64个loss时刻，既不是64个独立窗口，也不是官方batch64的32倍。
原DP未来尾部mask；batch补齐行不更新memory。Residual batch1使用变长尾块、不padding。
所有持续memory跨TBPTT detach和重规划保留，episode才reset。

| 同一Residual逐观察/cursor测试口径 | A | B | C |
| --- | ---: | ---: | ---: |
| test加权全程command MSE | .659997 | .336392 | .337817 |
| test全程arm / hand RMSE | .947055 / .650447 | .676828 / .463344 | .674974 / .469088 |
| test contact arm / hand RMSE | .989277 / .721658 | .467370 / .399695 | .462700 / .411137 |

原B0/B1另一个test协议是在每6观察重规划时统计整个预测H12，arm/hand分别
.969639/.682984和.895864/.639175；它与上表逐观察选中cursor及私有随机序列不同，不能直接相减。
epsilon MSE、加权command MSE和RMSE也不是同一指标。

## 精确参数量与冻结边界

计数入口`visuotactile.policy_census`实际构造模型并求唯一`Parameter.numel()`；不按宽度估算。
外部是固定**公开示例配置**，未加载作者训练checkpoint，未运行其机器人任务。
FP32列只是4 bytes/parameter，排除buffer、激活、optimizer和EMA副本，不是实测峰值显存或延迟。

| 模型及固定示例 | 视觉/本体编码 | 扩散器 | 额外时序/AT | 总参数 | 对B0总量 | FP32参数MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 我方B0 | 1,102,000 | 901,085 | 已含持续GRU | **2,003,085** | 1.00× | 7.64 |
| 我方B1 | 1,102,000 | 901,085 | tactile39,880＋fusion123,648 | **2,166,613** | 1.08× | 8.27 |
| 我方Residual B或C单独 | 1,102,000 | 无 | 242,597 | **1,344,597** | — | 5.13 |
| 冻结B0＋Residual B或C | 独立副本，无共享参数 | 同B0 | 同Residual | **3,347,682** | 1.67× | 12.77 |
| DP hybrid / lift_image_abs | 22,394,176 | 66,923,402 | 无持续RNN | **89,317,578** | 44.59× | 340.72 |
| DP real_image / real_pusht | 22,353,024 | 306,562,178 | 无持续RNN | **328,915,202** | 164.20× | 1,254.71 |
| RDP / peel+wrench+at_peel | 11,176,512 | 186,992,784 | 独立AT8,588 | **198,177,884** | 98.94× | 755.99 |
| ImplicitRDP / flip+wrench | 21,689,408 | 72,124,429 | force GRU＋投影1,192,704 | **95,006,541** | 47.43× | 362.42 |

只比扩散器参数时，DP hybrid、DP real、RDP、ImplicitRDP分别是B0的74.27、340.21、207.52、80.04倍。
任务、相机数、输出维度及归纳结构不同，这些不是公平性能倍数，更不是失败根因的因果证据。

| 我方模块 | 注册参数 | Residual训练时可优化 |
| --- | ---: | ---: |
| 共享三视图RGB-D空间编码 | 839,552 | 0 |
| proprio / task / 私有plan / context | 3,776 / 16 / 6,048 / 6,080 | 0 |
| 视觉持续GRU（512→128） | 246,528 | 246,528 |
| 逐指触觉＋融合输出64 | 39,880 | 39,880 |
| 当前chunk intent GRU29→64 | 18,240 | 18,240 |
| 跨观察intent GRU64→64 | 24,960 | 24,960 |
| motion GRU146→64 | 40,704 | 40,704 |
| head320→256→128→29 | 118,813 | 118,813 |
| Residual合计 | 1,344,597 | **489,125** |

Residual另有855,472冻结参数；连同冻结B0，完整B/C组合冻结2,858,557，可训练489,125。
B0/B1原训练注册参数均可优化，但`own_plan=none`下6,048参数plan模块被全false mask屏蔽；
注册量不等于有效信息维度或每个参数均获非零任务梯度。三视图共用一个RGB-D空间encoder，不能乘三。
执行时全部eval/no_grad；Residual复制已训练B0逐帧encoder并冻结，自己的视觉GRU另训。

我方逐帧结构：RGB ResNet18截到layer2/stride8，depth＋valid走2→32→64→128卷积，
256→64融合；16个soft spatial moments各6维＋16通道3×5网格共336维→128D/视图。
三视图384D＋proprio64D＋context64D形成512D，进入128D持续GRU。
触觉每指27个测量/派生量＋8D身份→MLP64/32→共享权重、独立指状态的GRU32；
五指160D→128→64。它是聚合法向力与FK编码，不是触觉图像ResNet或可观测slip估计器。

RDP的AT不是`nn.Module`容器，因此LDP的`parameters()`仅198,169,296，必须另加8,588。
AT细分encoder1,316、decoder7,044、quant160、post_quant68；在线解码实际需要后两者中的
decoder＋post_quant共7,112，完整已加载AT仍8,588。AT不在LDP optimizer中；`.eval()`本身不截断梯度，
其训练latent loss路径没有显式detach，不能把“不更新AT”写成“AT所有梯度均被切断”。
ImplicitRDP视觉backbone21,284,672＋投影404,736，force GRU含输出投影1,192,704。
其Transformer包含16×768 action位置与800×768 condition位置共626,688直接参数，不能只加子模块漏算。

## 实际生效超参数：与论文名称分开

固定来源：DP `5ba07ac6661db573af695b419a7947ecb704690f`；RDP
`824c5e8de1fd1811106907a04b5f0186e0138c0b`；ImplicitRDP
`112d6857d914c5dee4c4c172790088adc6bdc3a5`。
输入权限、传感器来源、动作映射和补充方法逐项对照见[完整policy比较](POLICY_COMPARISON_20261030.md)。

| 项 | 我方B0/B1 | 官方DP hybrid / real | RDP peel/wrench | ImplicitRDP flip/wrench |
| --- | --- | --- | --- | --- |
| U-Net/Transformer | [64,128]，kernel3，time embedding64 | [256,512,1024] / [512,1024,2048]，kernel5，time128 | [512,1024,2048]，**生效kernel3**，time128 | d768×7层×4heads，FFN3072，condition MLP，无额外condition Transformer |
| 视觉 | 3×448×252 RGB-D；ResNet18到layer2，无预训练；depth独立 | hybrid双84² RGB、空间softmax；real双240×320→crop216×288；完整ResNet18 | 单240×320→随机90%crop；完整ResNet18 | 单360×640存储→resize/center crop224²＋rotation/jitter；ResNet34，无预训练 |
| condition | 持续128D；B1额外fusion128 | hybrid137×2=274；real1026×2=2052 | obs522×2=1044 | 102个768D观察tokens＋force GRU512→768 |
| 动作/latent | 原native29D | hybrid10D abs；real2D EEF | 实际动作4D；AT内部conv latent4，Gaussian channels16；**扩散8×16 latent** | 存储19D→训练13D RPY；部署action_type还override为right_arm_6DOF |
| H / 执行节奏 | H12@60Hz=.20s；commit6=.10s | H16；hybridcommit8，实际Hz依dataset env metadata；real10Hz/commit6→1.60/.60s | 原动作H32@24Hz=1.333s；latent更新16tick=.667s，planner6Hz，decode24Hz | H16@10Hz=1.60s；vision/noise cache6tick=.60s；force重采样10Hz |
| 训练扩散 / 推理 | cosine100 / DDIM8 eta0 | cosine100 / hybrid8；real yaml100→部署16 | cosine100 / yaml100→部署8 | cosine100 / 部署5；eval.sh v_prediction覆盖yaml epsilon |
| optimizer / beta / wd | AdamW1e-4 / (.9,.999) / 1e-6 | AdamW1e-4 / (.95,.999) / 1e-6 | LDP同DP；非timm路径LR再乘accelerator.num_processes | AdamW3e-4 / (.95,.999) / 1e-6；**视觉backbone**3e-5，projection仍3e-4 |
| EMA / LR schedule | 无 / 无 | EMA .9999上限 / cosine warmup500 | LDP同DP；AT无EMA、cosine warmup100 | EMA .9999上限 / cosine warmup2000 |
| batch / 配置轮数 | 2×连续TBPTT32 / 实测50轮 | random window64 / 3000、600 | window64 / AT601、LDP401 | window64 / 600 |
| 数据顺序 / val | 固定episode顺序 / 10% | shuffle / hybrid2%、real0%示例 | shuffle / 当前peel示例0% | shuffle / 10% |
| 训练噪声 / 缓存 | 每loss新高斯；每6tick私有RNG重采样chunk | 高斯diffusion窗口 | AT Gaussian posterior采样再latent diffusion；新触觉持续decode | input perturbation .1；chunk内缓存视觉及同一initial noise |
| 准确训练样本/更新量 | 本报告实测 | 作者checkpoint未取得，未知 | 作者checkpoint未取得，未知 | 作者checkpoint未取得，未知 |

DP hybrid runner的`fps=10`是录像频率，源码另写robosuite_fps20；未取得该示例HDF5 env metadata，
不能据录像fps断言真实控制10Hz。表中的real10Hz来自其真实runner默认，仍不是全部作者实验的Hz。
官方DP和RDP都存在DDIM8，**不能把8步直接认定为bug**。
RDP的encoder_lr_coefficient=.1只在特定pretrained timm分支生效，当前ResNet18示例不套用。
本项目history4/stride3字段在memory2下不是仅记忆.15秒；持续GRU每60Hz观察更新。

Residual本身不扩散：AdamW1e-3、betas(.9,.999)、wd1e-4、gradclip1，无EMA/schedule，
`clip(B0 + observed_gate × scale × tanh(raw), -1, 1)`。
train16 p99决定arm7/hand22的scale均2，之后不利用val/test调幅度；命令差可达[-2,2]，标签未盲裁[-1,1]。
arm是原生积分＋平滑，hand是绝对映射＋平滑，不把scale2解读为2米或统一角速度。
当前专家命令仅监督，t−1专家执行反馈明确teacher forcing；部署Correction只读真实学生ack，mixed不回填B0。

## 论文补充、可核实范围与结论

OmniTacTune论文v1的gate作用于**触觉feature**，不等价于我方无接触强制零动作残差。
它描述冻结base、SAC actor/双critic/replay、12分钟warmup、10Hz/K8、latent32、batch256、
actor/critic LR3e-4、encoder1e-4、gamma.99/tau.005/replay1e4、scale .05→.1→.15。
未找到可检查官方训练源码或checkpoint，完整参数量/逐模块冻结数不能精确核算；不将论文配置标为源码实测。
TouchWorld的nominal本身读触觉；ViTaL为视觉BC＋触觉residual RL；T-Rex为VLA/MoT；
ResiP为PPO式残差与distillation；ResFiT为TD3 off-policy。来源pin和已知/未知详见比较报告，
它们不是与本项目29D关节命令、五指合力、监督残差完全同类的DP基线。

**源码事实：** 我方模型显著较小、时间跨度较短、无EMA/schedule、固定episode顺序；
Residual有真实训练和native ack执行链，B0冻结且无触觉/mixed输入。
**实验观察：** 监督残差降低离线误差，自主没有接触gate，局部交接后接触短暂、无子目标/抬升。
**推断：** 严格门控确实不能修复无接触接近失败；离线teacher forcing和学生状态存在分布偏差。
**未验证：** 大模型、EMA、schedule、较长commit或动作历史中哪一项能提高成功率，及接触力因果收益。
本轮未更换模型、训练RL或调整原控制；后续计划持续性/在线专家纠正/SAC是独立研究选项，不冒充本次实现。

## 统计方法与失败记录

真实checkpoint在CPU加载；外部视觉/AT在CPU构造，大型denoiser在meta device构造，仅计张量形状。
DP/RDP full policy中仅用局部factory指定denoiser设备，未改参数拓扑或上游源码。
57个实际导入的上游Python文件逐一从固定GitHub revision下载并核对SHA256，全部一致。
原DP robomimic0.2.0与timm1.0.15/loguru/dill仅装在ignored隔离目录，未改共享环境；
RDP/ImplicitRDP未pin timm，故1.0.15是本次核算依赖，不冒充作者实际训练版本。
未使用的LoRA分支只有PEFT导入sentinel，若调用任何PEFT功能立即报错；所选架构没有LoRA。
Torch2.7.0、torchvision0.22.0、diffusers0.39.0、einops0.8.2，不声称重建了作者训练环境。

核算途中出现过B1属性名称、安装版robomimic CropRandomizer位置、meta默认设备影响随机crop、
未安装可选PEFT等工程错误；保留各attempt日志，以完成后的完整计数为准，未通过删模块或套宽度公式补数。
外部模型未下载权重、未训练、未推理任务，只有本项目权重和录像可提供本机实验结论。

## 本轮观看录像与资源完成记录

预分配：seed981002 × A/B/C × autonomous/expert_rollin，共6次启动；roll-in首次连续3个有效观察
max finger force>.1N时切换，最多240步寻找条件；真实env.step、从reset因果预热，不瞬移、不清memory。
三相机逐观察RGB、60fps、原生终止，字幕包含来源、gate和cursor；终止动作后图像不可用，不录reset帧。
既有B0/B1四视频入口为`outputs/visuotactile/demos/baselines_20261030/README.md`。
新视频入口为`outputs/visuotactile/demos/residual_20261030_01/README.md`。
六条均完成原生601步timeout；新6＋旧4视频均完整解码、601帧、60fps、1344×316、10.0167秒，PTS连续。
新视频共13,582,728bytes；旧4视频8,892,289bytes，旧SHA再次匹配。预览亲看初始、交接/中段和末帧，
三视图/字幕正常，未将关节运动、接近或短暂接触误称抓取/抬升成功。原始视频不上传仓库。

| 观看模式 | A | B | C |
| --- | --- | --- | --- |
| 自主 | 601学生步，gate0 | 同左 | 同左 |
| 专家roll-in交接 / 学生步数 | 88 / 513 | 88 / 513 | 88 / 513 |
| 局部gate步数 / 占学生比例 | 6 / 1.170% | 10 / 1.949% | 7 / 1.365% |
| 局部首次 / 末次gate step | 88 / 96 | 88 / 99 | 88 / 97 |
| 局部实际delta arm/hand RMSE | 0 / 0 | .145012 / .100152 | .128905 / .080208 |
| 交接command jump arm/hand RMSE | 1.172866 / .762103 | .363496 / .208750 | .621667 / .208750 |
| 学生重规划jump arm/hand RMSE | .855927 / .608070 | .850694 / .605714 | .849225 / .601784 |
| 学生段内jump arm/hand RMSE | .638592 / .398216 | .637908 / .399110 | .642571 / .399991 |
| 局部command饱和元素比例 | 6.829% | 9.195% | 8.194% |
| 自主/局部任务成功、子目标、原生lift | 全部0 | 全部0 | 全部0 |

三条局部专家prefix SHA与每个保存的switch物理字段精确一致；不是瞬移或强写状态配对。
仍非逐位视觉/计划重放：switch处B0最大命令差A/B=.004771、A/C=.010982；自主首命令差
.008798/.008998，整段最大差.030882/.030030，而自主gate始终0。这些自主差异不能归因于残差。
观看结果单列为每组自主0/1、局部0/1，不更新原正式0/50或旧诊断0/2等分母。

完整小聚合：[视频/执行/保留原产物/资源证据](evidence/experiment_videos_20261030.json)。
6条trace的native ack、t−1真实反馈、memory1..601、cursor0..5、generation每6步、无接触旁路全部通过；
交接不清memory，A全部学生动作逐位等于当步B0。原100正式trace与旧11完整残差trace SHA未变，
manifest、两份best、正式report未变，原生控制/资产/研究wrapper相对fe9a634无diff。
原生日志仍有kinematic setLinearVelocity/setAngularVelocity错误与deprecated setter告警，未修共享栈；
本次无软件中止、无替换种子，不能据日志直接推断失败根因。

### 本轮资源与限制（UTC）

连续监测09:02:23.283–09:27:59.589，1,536.325秒（25.61分钟），低于含等待2小时预算。
整卡时间加权均值**27.9960%**；最大完整60秒均值**60.3794%**，因此**没有全程满足双50%约束**。
1505样本、1446完整窗口，缺测样本/秒数0，最大间隔1.1217秒；最低空闲VRAM11,869MiB、RAM26.0815GiB。
结束磁盘空闲100.43GB；12GB余量每次工作前检查，连续磁盘最低值未知。

| 完整60秒滚动均值>50%的样本区间 | 区间最高均值 |
| --- | ---: |
| 09:06:28.532–09:06:44.877 | 60.3794% |
| 09:08:16.762–09:08:32.077 | 58.8462% |
| 09:10:06.097–09:11:06.351 | 57.6815% |
| 09:11:17.566–09:12:14.777 | 55.9423% |

这些是重叠窗口超限的采样端点区间，不是按进程归因的超额GPU时间。自身使用45%准入、
实测准入等待250.644秒、主动节流763.655秒，总1014.298秒；未修改外部作业、驱动、全局配置或模拟dt。
采样门控会在当前窗口不安全时暂停本任务，却无法事先阻止整卡负载突发；不能用最终均值掩盖上述违反。
小报告另列recorder启动到最后trial报告的资源窗口，避免把monitor尾段与执行阶段混称。
全部录像、monitor、参数核算和测试worker已退出，没有新的训练进程。
全套85项通过，随后新增录制窗口/缺测边界测试在针对性12项中通过；原requests依赖warning保留。
