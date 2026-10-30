# B0/B1独立审查与行动历史（2026-10-30）

本轮用户授权审查、必要修复、少量诊断和小演示拟合，以及审核后推送功能分支。
原400条数据、两份best、100条正式评估保持原样；没有启动新的完整训练或采集。
时间统一上海/新加坡UTC+8。本报告区分用户要求、实现选择和实测，不以旧文档代替原始证据。

## 实际身份与研究含义

接手HEAD为`8ed8857`，工作树干净，原分支`feature/command-training-pipeline`；
本轮分支`feature/baseline-audit-action-history`。用户fork是origin，upstream只读。
训练launch记录HEAD为`4714c96`，但两份best内10个训练源码SHA256逐个匹配`76bb58e`；
正式evaluation/plan记录`dc77a6b7b00621cef6b0607d3174d5d55f86b3c7`，其13个源码hash也逐项匹配该提交，不能拿当前HEAD替代。
PyTorch2.7.0+cu128、torchvision0.22.0+cu128、Diffusers0.39.0、NumPy1.26.0、h5py3.16.0。

| 原始文件核验 | B0 | B1 |
| --- | --- | --- |
| report完整epoch / best内epoch | 50 / 49 | 50 / 49 |
| best SHA256 | `fd1bd263de38ea6c6e0082d2ba55121adf3908b31c78b55d1cf39477ad6ff9d5` | `bb23694f94294b20a2bb21b0c260b3ebbd30f2e3ed6a6e22b2465a501b1c5cc2` |
| test epsilon MSE | .17778561 | .14896128 |
| sampled arm / hand RMSE | .969639 / .682984 | .895864 / .639175 |
| sampled horizon0 → horizon11 RMSE | .690343 → .811935 | .642791 → .790874 |
| 正式成功 / 原生timeout | 0/50 / 50 | 0/50 / 50 |

每条601步、successes始终0、39非法初始化前置排除，50组初态配对差0。
B1 zero-force离线RMSE=.724559；它保留FK/时间/结构，不能解释成只去掉全部触觉分支。
loss不是关节角误差；表内命令RMSE单位为原生[-1,1]命令。

B0是三路RGB-D+编码器/外部任务命令的持续128D视觉本体GRU，再条件化扩散U-Net。
B1另有五指力/FK的逐物理样本GRU与融合GRU，各自独立训练。它不是触觉Correction。
研究目标仍是几何意图与触觉执行分离；当前监督是普通专家命令，包含可能的压力预加载。
B2/B3、去预加载、Correction学习目标与正式训练尚未完成。

训练曲线实际读取best内1–49轮以及train.log第50轮：B0 train .773291→.170946、
val .634335→.178691；B1 train .767739→.141926、val .618740→.151395。
第49轮val .176505/.147740优于50，因此best49身份合理；下降的epsilon曲线与较差采样可同时存在。

## 已确认与尚未确认

已确认的实现缺口：原B0/B1没有行动条件，`own_plan_mode=none`在训练与部署都屏蔽私有计划；
Dataset虽另返expert_history，batch/encoder不使用它。q/qd持续记忆不是已发命令历史。
这是本轮设计要求的缺口，尚不能据此认定所有DP必须用行动历史才能成功。

已确认的评估检查缺口：原`assert_policy_compatible`主要验证传感输入，没有完整比较原生控制/DR/reset/物理。
新增`native_execution_contract/assert_execution_compatible`检查这些配置、原timeout/decimation和实际关节限位。
实际这次数据与评估resolved config只差`scene.num_envs=4→1`、seed，未发现控制参数漂移。
这些有意不同的实例参数不是动作语义变化。源码/资产仍需另外核查，配置比较不是全栈确定性证明。

100个原NPZ重算表明模型提交动作与submitted完全一致；延迟后动作按原arm/hand公式产生目标，
最大误差B0=1.55e-7rad、B1=2.15e-7rad。未发现动作乘六、重复积分或6步只执行一个动作。
原手臂每tick最大未饱和目标增量约.0025rad；手指是绝对映射+平滑，不能一起解释为速度。

| 全程每条统计再取平均 | B0 arm | B1 arm |
| --- | ---: | ---: |
| 重规划边界命令jump RMSE | .859586 | .791222 |
| 段内命令jump RMSE | .608543 | .345893 |
| 相邻6步平均arm向量点积<0比例 | .223030 | .156970 |
| 命令净向量长度 / 路径长度 | .238855 | .313814 |
| 目标−实际q RMSE(rad) | .005493 | .007272 |

这些是关节空间运动/抵消诊断，不把全任务要求为单调前进。前60步与全程统计分别保存在
`outputs/visuotactile/run/audit_20261030_01/traces.json`。
B0有21/50、B1有40/50轨迹出现>0.1N指端法向合力；两组均0/50达到初始对象高度+5cm。
指端可能接触桌子/其他手指，最大值也包含很大的碰撞脉冲；无接触对标签，不能称为工件抓持。
初态没有settling，物体会先下落；高度相对初态的阈值不能代替原lift flag或证明完全没有短暂抬升。
实际原lift flag计算为`.05+z−initial_z > .15`，即相对初态超过10cm才锁存；
后述成功专家最大约+8.99cm也没有该flag，不能把native_lift=false解释为未抓起。
因此本轮主要是接近/抓持/抬升前的问题证据，不能由0子目标断言全部没接触。

## A–F线索逐项复核

| 线索 | 结论与依据 |
| --- | --- |
| A 时间/索引/记忆 | Dataset用obs[t]监督command[t:t+12]；部署cursor0..5，每6步新plan，每个60Hz观察更新memory。T32仅detach，不清状态；episode ID才reset。128D是有损压缩通路，未证明长历史任务效用。 |
| B DDIM | 实际leading8为[84,72,60,48,36,24,12,0]，alpha_bar84=.05365923、99=2.42854e-7；t84仍有sqrt(alpha)=.2316信号。冻结噪声/权重/完整前缀对比8/20/50/100，未显示100稳定改善，不把起点差异判为唯一根因。 |
| C 段间切换 | 原NPZ边界jump更大，段内也存在明显变化。同初始condition四种noise的前6步arm平均标准差B0约.59–.62、B1约.54–.62，随机性大；合理多模态与质量不足仍需区分。 |
| D 动作/timeout/阶段 | 原native counter>=max_episode_length终止，当前max=601；不是GPU墙钟超时。第一计数目标是final沿孔局部+Z回退.05m的pre-insert，不是抓起。 |
| E 学习质量 | 两条训练与两条测试episode完整前缀采样，训练arm全程RMSE仍约.84–1.03，不能用epsilon loss下降证明动作已学会。无EMA、无LR调度、固定episode顺序、较小网络是需对照的选择，不自动定为bug。 |
| F 配置核对 | 原兼容检查覆盖不足已修；本轮原始配置未见除种子/环境数外差异。world_hold是历史研究wrapper的真实施力兼容行为，必须单独声明，不能把它说成纯遥测。 |

离线对照固定best与每条episode的噪声seed20261030，各sampler使用完全相同initial_noise；
从step0按8行读到自然记录结束，所有观察更新memory，选择step%6==0的条件。保留真实三路448×252，
不把随机零状态窗口称完整前缀。样本是各split保存顺序前两条，未按结果选择。

| 全程arm RMSE，按4条episode的逐条RMSE均值 | DDIM8 | DDIM20 | DDIM50 | DDIM100 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 1.010736 | .984567 | .968690 | .984860 |
| B1 | .945559 | .937625 | .941508 | .967699 |

逐joint、horizon、前60步、前6执行步、方向误差和计划重叠误差都保存在该目录。
DDIM8前60步内的前6动作arm RMSE（两train、两test顺序）B0=.9818/.8736/.7753/1.0619，
B1=.8081/.8086/.7510/.9513；问题已在初始接近阶段出现。各关节误差没有只集中于单个关节。
初版输出的observations字段误报最后重规划step+1，实际扫描完整记录；后续仅修正该计数字段，
未重跑GPU实验、未改旧JSON，launch源码hash仍表示真正运行的版本。

## 官方实现对照

本轮实际读取固定上游Play2Perfect `ff2fc62873f6c6cd93164123c8358e9ce2d65c9b`，
以及官方DP `5ba07ac6661db573af695b419a7947ecb704690f`。
固定源码入口：[Play2Perfect](https://github.com/kushal2000/play2perfect/tree/ff2fc62873f6c6cd93164123c8358e9ce2d65c9b)、
[DP image policy](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_image_policy.py)、
[DP训练配置](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/train_diffusion_unet_image_workspace.yaml)。
联网`ls-remote`核实两者当前HEAD恰为该版本；下载文件SHA保存于
`outputs/visuotactile/run/audit_20261030_refs/identity.json`，参考源码留本地，不新增第三方依赖。

| 官方路径与实现 | 本项目差异及判断 |
| --- | --- |
| `policy/diffusion_unet_image_policy.py` | 官方使用n_obs_steps窗口，动作取start=To−1；本项目标签从当前t开始，执行0..5，两种索引分别匹配自己的数据，不能直接搬To−1。官方明确assert无past_action，不证明缺行动历史必错。 |
| 同文件采样/损失 | 官方高斯初始化、每次重新采样、epsilon MSE及scheduler.step；本项目同类目标，DDIM eta0而该官方配置用DDPM。新噪声不是单独的bug证据。 |
| `model/common/normalizer.py` | 官方训练集拟合动作/低维归一化并反归一化；本项目命令本来就是native[-1,1]，输出直接clip，q/qd训练集mean/std，RGB映射[-1,1]、depth/2m固定。不能把命令当rad再次反归一化。 |
| `model/vision/multi_image_obs_encoder.py` | 官方多相机/RGB与lowdim拼接；本项目共享每视图RGB-D空间编码+每步GRU，无ImageNet预训练、无图像augmentation。都是实现选择，需小样本功能验证。 |
| `config/train_diffusion_unet_image_workspace.yaml` | 该官方示例100推理步、H16/obs2、denoiser[512,1024,2048]、EMA、cosine LR和500 warmup；本项目H12、[64,128]、常数AdamW1e-4、无EMA。任务/预算不同，不整体照搬。 |
| `workspace/train_diffusion_unet_image_workspace.py` | 官方有EMA更新、rollout和采样动作误差；本项目原best只按随机epsilon validation选，任务能力验收过晚。本轮补冻结采样和有界功能检验；下一轮应提前设置动作质量/闭环门槛。 |

原生源码diff：整个isaacsimenvs相对该上游仅action_utils.py多7行可选只读trace hook，
来自`6fc04a8`；hook在原delay抽样之后克隆数据，没有修改公式/RNG抽样。当前测试仍核对hook有无的目标及RNG。
`sim/environment.py`是新增研究wrapper；历史world_hold每物理子步重表达原world请求，确实改变了安装版composer的施力方向行为；
原生产与本次评估均为world_hold，未将legacy结果混为同协议。本轮不修改原生控制器或该wrapper。

## 新行动历史：契约与权限

新增`PolicyConfig.action_history_mode="issued"`、`command_checkpoint_version=3`、`action_history_version=1`。
旧权重缺字段解释为none/version2；不能只改config就加载成新版。新CLI显式
`--action-history-mode issued`；新32D命令MLP进入context，再由每步GRU累计。没有参数跨B0/B1共享。

| 类别 / 来源、维度、单位、时钟 | B0 v3 | B1 v3 | 未来B2 nominal | 未来B2 correction |
| --- | --- | --- | --- | --- |
| 观测：三RGB-D/V4HW、58D编码器q/qd、operator task/可选9D goal；t观察可得，图像/goal各自mask；RGB[-1,1]、depth/2、proprio训练集统计 | 读 | 读 | 允许 | 自己的编码器允许 |
| 实测运动：q/qd每60Hz，FK位置/姿态与差分每120Hz；q是限位归一化/qd rad/s，FK米/旋转、速度m/s；reset首物理包无效 | q/qd持续状态 | 另读五指FK/响应 | 可读合法编码器/FK；物理运动含接触影响 | 允许 |
| 过去提交：`command_clipped_pre_delay[t−1]` canonical29D[-1,1]；raw/submitted另存，当前/未来专家命令仅监督 | 读 | 读 | 混合命令禁止；要用独立私有nominal输出 | 可读实际提交，必须区分来源 |
| 过去实际采用：`control_trace.delayed_command[t−1]`29D、原抽样source timestamp；两种命令年龄按秒×60转为native ticks，未把slot index当实际age | 读 | 读 | 可能包含correction，不自动授权 | 允许 |
| 控制目标/跟踪：obs actuator_target与target−q，各29D rad，t时反映前次执行；目前原Correction未额外拟合归一化 | 本版不读 | 本版不读 | 禁止未经区分的混合控制目标 | 已有独立shadow接口可读 |
| 私有模型计划：previous_nominal29D、prior_plan12×29/mask、生成step/time/cursor；由模型真实推理 | 本版mask关闭，仍记录预测段 | 同左 | 可读私有预测；须配套真实因果训练构造 | 可读当前/未来nominal意图 |
| 触觉：两120Hz样本×五指×3 world/local N、有效性/time、FK；力signed log1p，差分N，速度m/s；仅<=当前obs时刻 | 禁止 | 读并保持五指/融合state | 禁止直接读 | 允许，仍无已定学习目标 |
| Correction/混合：29D native候选、混合submitted、source/执行标志 | 不存在 | 不存在 | 私有输出与混合严格隔离 | 单独记录/使用，未训练不执行 |

新增动作字段为`previous_issued_command`、`previous_delayed_command`、两项float64模拟时间戳与bool有效性。
默认t0零值/false；每个完整episode单独清状态，子目标/重规划/chunk不清。在线必须在env.step结束后调用
`observe_execution`，核对episode/step、实际clipped命令、trace.valid与前次完成时间；缺ack、重复ack或混合覆盖命令明确拒绝。
任何packet伪造的command history都会被私有执行记录替换。没有修改env/原action helper以实现此输入。

离线从原HDF5 transition[t−1]构造相同字段，保留完整前缀；来源是**专家已执行历史teacher forcing**。
在线是**学生真实提交/延迟执行历史**，无需专家。相同定义不等于相同状态/命令分布，仍存在covariate shift。
本版没有生成或训练“学生在专家轨迹上的私有未来计划”，也没有把这种旁路推理说成真实闭环经验。
用户要求的自身行动上下文通过部署的实际提交历史实现；私有剩余计划是另一种待独立实验的设计，不能拿未来专家标签填补。

## 验证与后续门槛

新增测试覆盖：t时当前/未来标签修改不影响t输入、跨chunk过去命令对齐、真实执行ack与异步reset、
B0/B1逐步/整段条件一致、历史扰动影响未来条件、有限非零梯度、v3保存加载、错误版本/混合/缺反馈/未来时间拒绝、物理配置漂移拒绝。
当前全套73项通过，已有requests依赖warning未修改共享依赖。

### 单演示24步前缀：已完成的有界拟合

各模型仅使用保存train split第一条`r931005_1793209376_e00_p000000`前24个观察，
每次从step0完整重算前缀，全部RGB-D参与反向；没有缓存随机编码器latent。
明确warm-start旧best已有参数；新增history MLP随机、context新增列初始零，随后所有参数可更新。
FP32、AdamW3e-4、clip1、各240次更新；3个评估噪声seed190/191/192与训练噪声分开。
这是短前缀拟合，不是完整episode过拟合成功，更不是正式重训。

| first6 arm RMSE（3噪声均值） | B0 | B1 |
| --- | ---: | ---: |
| 拟合前 | .974926 | .794000 |
| 拟合后 | .584939 | .331831 |
| 拟合后屏蔽历史 | .584900 | .331859 |
| history MLP梯度范数最大值 | .000382752 | .000119443 |

误差有下降但仍明显；屏蔽历史几乎不变，**不能把拟合改善归因于行动历史**。
真实记录逐步执行反馈与Dataset历史字段逐位相同；整段/逐步条件最大误差2.68e-7/2.98e-7，
固定噪声保存重载结果精确相同。诊断权重标记`diagnostic_only=True`、v3，并保存warm-start SHA，
正式评估入口拒绝这些权重。当前probe未保存完整运行源码hash，不能补写成原始launch证据。
本地输出为`outputs/visuotactile/run/history_probe_20261030_01/`。

原正式0/50只描述旧模型，不将有界诊断加入分母。新的历史输入没有被宣称已改善接近或成功率。

### 同入口原专家参考

`audit_rollout_old8_20261030_01`只用预定合法seed981001，原best B0/B1与released expert
共享原reset/env.step、三路质量和world_hold，初态记录最大差0；不重放旧数值轨迹。
原专家SHA=`7e61015dad660430ab07bab4f715c90c6f76b84cc358da6be81319a6aee09990`，
SAPG group0/CPU确定性actor，特权观察仅给专家，B0/B1不读。
专家217步完成两目标和retract；B0/B1各601步timeout、该seed均无>0.1N接触、物体未被抬起。
这是同入口可以成功的直接证据，仅1个合法起点，不是专家鲁棒性估计；该组实际外扰曝光0。
专家deterministic actor内部仍消耗global RNG（与原采集一致），不宣称与学生随机流逐位相同。

旧DDIM8的B0/B1 plan重叠arm RMSE=.859545/.839385；所有实际command与对应cursor预测逐位一致，
generation step匹配，重规划确实执行0..5。finger-link均值到对象原点最小距离=.1150/.1034m，
专家=.0486m；这是诊断几何指标，不是指尖表面距离或抓持判据。旧视频step0/120/360/600三视图
已亲看，接近后撤退且对象留桌面与数值相符；稀疏图不用于证明全程没有短暂接触。

### 冻结旧权重：DDIM100闭环

`audit_rollout_old100_20261030_01`使用同seed981001、同best、同策略noise seed，
仅开发推理步数改100（每次重规划的initial_noise相同定义；分叉后的观察条件自然不同）。
B0/B1仍601步timeout，无子目标、无+5cm抬升；B0无>0.1N接触，B1有3/601个控制时刻接触。
B0边界/段内arm jump=.7098/.3810、B1=.7714/.3820；计划重叠arm RMSE=.7563/.8295。
最小finger-link均值距离B0=.1383m、B1=.0708m；B1更靠近且短暂接触，但没有任务推进证据。
100步没有解决该起点失败，也不证明8步在所有种子上等价；不更改旧正式默认sampler。
两组诊断B0/B1的所有已保存初态字段与原正式seed981001也逐项差0；这不包含完整渲染内部状态，
不能升级为跨运行逐位重放承诺。交付前再次SHA核验原100个NPZ、两份best和manifest均未改变。

历史权重闭环首次目录`audit_rollout_history_20261030_01`在AppLauncher后、scene/策略加载前
发生native SIGABRT/exit134：`std::system_error: Invalid argument`，栈位于
libomniclient FileThread/TickThread。没有任何动作或trial，不能算策略失败。
当时RAM可用约35.5GiB、VRAM空闲约21.3GiB，不能从该栈认定OOM或本轮inotify耗尽。
完整日志和目录保留，按预动作SIGABRT规则只在新目录`_02`同seed做一次恢复；未改共享库、
系统watch设置或其他进程。启动崩溃的遥测覆盖最后样本之前约11.65秒，之后崩溃尾段未覆盖，
没有完整60秒窗口，不能把它写成“整段资源均值已证明合规”。

### 新版真实执行历史闭环：完成与局限

恢复目录`audit_rollout_history_20261030_02`已完成，B0/B1均601步原生timeout。
两份权重仅经过上述24步前缀拟合，标为diagnostic；没有被称为新正式baseline。
全程每次真实env.step后ack、每次下一观察读取，t0 history_valid=false、其后600次true；
历史issued/delayed与实际前一步trace最大差均0，提交时间差0；两组command/plan/cursor一致。
这证明真实学生闭环反馈链可运行，不依赖专家回填；不意味着teacher forcing与自主分布一致。

| seed981001诊断 | 旧B0 DDIM8 | 新历史B0 | 旧B1 DDIM8 | 新历史B1 |
| --- | ---: | ---: | ---: | ---: |
| 最小finger-link均值到对象原点距离(m) | .1150 | .2617 | .1034 | .1745 |
| >.1N接触控制时刻数 / 601 | 0 | 366 | 0 | 191 |
| arm边界jump RMSE | .8293 | .7229 | .7361 | .5631 |
| arm段内jump RMSE | .5805 | .6263 | .3360 | .4195 |
| 实际任务成功 / +5cm抬升 | 否 / 否 | 否 / 否 | 否 / 否 | 否 / 否 |

新版边界jump较小，但段内更大、接近距离更差、物体仍留桌面。step480三视图已查看，手在桌边/桌面区域；
法向力传感器没有接触对象身份，更多接触不能叫抓持改善。这个短前缀拟合没有证明连贯接近、
有效接触或抬升改善；也不是单独历史消融，所有参数都更新且不同于正式训练协议。
完整记录保留，7条成功/失败诊断不扩充原100条正式样本；全部诊断实际外扰曝光0。

### 资源与结束状态

GPU任务串行，offline/fit每短计算块后约2倍计算墙钟sleep（单次最多2秒）；
闭环每原生控制调用间sleep100ms，DDIM100为150ms，原120/60Hz及动作幅度不变。
ResourceGuard用50%上限/45%准入、4GiB显存/10GiB主存余量。交付前磁盘读数超过100GiB，
新增评估按起点检查12GB保留量。没有删除原数据、改系统watch/驱动或触碰其他作业；
不能把离散磁盘检查称连续最小值。

| 监测任务 | 时间加权整卡均值% | 最大完整60秒均值% | 工作秒 | 短节流秒 | 准入等待秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原trace/冻结离线 | 12.497 | 15.366 | 39.174 | 78.425 | 1.000 |
| 24步前缀拟合 | 27.000 | 35.288 | 108.610 | 217.280 | 1.000 |
| 旧DDIM8＋专家 | 18.443 | 20.007 | 105.460 | 142.032 | 2.000 |
| 旧DDIM100 | 19.660 | 28.115 | 134.717 | 180.415 | 2.000 |
| 历史首次启动崩溃前 | 25.436 | 无完整窗口 | 未分解 | 0 | 2.000 |
| 历史恢复闭环 | 17.798 | 21.114 | 91.654 | 120.310 | 2.000 |

offline/fit工作秒是计算块墙钟；闭环工作秒是policy+env.step，启动/图像写入等另在原report计时。
两者不是GPU忙时或每进程GPU占比。完整闭环进程墙钟分别272.322/341.496/239.771秒。
已监测区间合并1332.345秒，按测得时长加权均值20.225%；各任务内完整60秒最大35.288%。
区间内缺测样本/秒均0，最大采样间隔12.303秒发生在启动，不能声称瞬时硬配额。
读文档等任务间隙与崩溃末尾未连续监测，未当0稀释，不能声称整个会话任意60秒都有证据。
最小采样空闲显存17577MiB、可用RAM31.814GiB；全部受控任务实际已退出，没有后台训练/评估继续运行。

公开聚合为[evidence/independent_audit_20261030.json](evidence/independent_audit_20261030.json)，
仅少量统计/身份/资源，不含完整trajectory、权重、视频或日志。原MIT与vendor MIT/provenance保留。
本轮以`feature/baseline-audit-action-history`聚焦提交到用户fork；确切commit见Git历史与交付消息，
不向upstream推送、不强推、不合并main。代码验证为全套73项及后续相关22项通过，compileall/diff检查通过。

### 下一轮建议及预算（本轮未启动）

优先完成**完整演示拟合及历史消融**：固定train前2条完整episode，B0/B1各做issued/none独立对照，
保持相同warm-start、噪声、数据顺序，B1保留触觉。B=1/T=32、完整有序前缀/TBPTT、H12/replan6、
FP32/AdamW3e-4、clip1，4组各最多600更新；墙钟总上限90分钟、GPU整卡双均值<=50%、
4GiB VRAM/10GiB RAM/12GB磁盘，最多4个末权重和小报告。时间是停止预算，不是速度保证。
按相同3个离线采样噪声比较完整轨迹与前60步/first6的arm、hand、方向与overlap误差，
另在预先声明的2合法种子最多8条闭环观察接近/接触/高度；不调原控制器、成功阈值或timeout。
若仍无法拟合完整示范或历史消融无影响，不进入大训练；下一次只改变一个容量/优化/条件设计因素。

通过小验收后，正式候选从固定320/40/40上进行**5epoch×2的阶段训练**，版本3/issued、
B=2/T=32/BF16、prefetch1、H12/replan6/DDIM8、独立参数、AdamW1e-4，先保持其他设置可比；
串行GPU任务、墙钟上限6小时（依据旧epoch计时保守外推，非本轮测得ETA），每epoch除了epsilon
还固定噪声报告执行前6步动作误差。预定小闭环只作开发门槛，最终选模型只用validation，test保留作终评。
只有验证与闭环能力出现可解释进步再讨论完整50epoch和正式50配对；EMA、LR schedule、
episode shuffle、容量与采样步数分别做独立对照，不把它们全部同时改后归因。
无接触区间的no-contact correction不能主动救回接近失败；有接触但无抓持的case才支持进一步研究接触修正。
