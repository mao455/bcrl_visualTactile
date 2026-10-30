# 独立残差 Correction v1（2026-10-30）

本轮明确授权实际训练、最多12条诊断和feature分支发布。原400条数据、B0/B1 best、正式100条
及各0/50结果不变。原生环境与`sim/environment.py`只读。所有新产物在
`outputs/visuotactile/run/residual_20261030_01`，不发布原数组或权重。

## 实际架构与权限

| 组件 | 参数 / 冻结 | 输入与输出 |
| --- | --- | --- |
| 正式B0 v2 | 2,003,085；全部冻结 | 三448×252 RGB-D、58D本体、任务/外部goal；128D持续GRU；H12/DDIM8，每6步重规划；无触觉、无mixed history |
| 每个Residual v1 | 1,344,597总参数；489,125可训练、855,472冻结 | 独立复制B0已训练逐帧编码器并冻结；视觉128、触觉64、intent64、motion64 → MLP256/128/29 raw delta |
| Residual视觉GRU | 可训练，独立初始化自B0 | 冻结逐帧512D特征，逐60Hz更新；不是缓存随机网络或冻结全部Correction |
| 触觉分支 | 可训练 | 五指world/local法向合力、encoder FK/旋转/速度、120Hz时间及因果差分；没有taxel、压力图、slip或partner身份 |
| intent / motion | 可训练 | 当前真实B0剩余chunk及mask；q/qd、实际t−1 command/valid、当前actuator target/tracking error |
| 输出头 | 零初始化，可训练 | `clip(base + gate * scales * tanh(raw), -1, 1)`；无接触逐位等于base |

旧`CorrectionPolicy`仍是完整candidate+替换/blend接口，新版不重新解释它的checkpoint。
新`residual_checkpoint_version=1`，输入`residual_frozen_frame_memory2_v1`，动作
`gated_additive_native_command_v1`；拒绝无有效更新、错误base hash、B1、v3 issued模式。
B0 SHA256为`fd1bd263de38ea6c6e0082d2ba55121adf3908b31c78b55d1cf39477ad6ff9d5`。

## 数据、参数化与学习目标

沿原专家完整观察序列，从step0持续更新冻结B0、私有噪声seed、真实重规划/cursor。
缓存逐帧特征、剩余plan/mask/base、生成step、time、原文件hash、来源，不把未来专家动作当plan。
专家t−1命令、target/tracking来自真实执行，但属于**teacher forcing**；部署改用学生真实ack，
存在不可忽略的状态/执行历史分布偏差。Correction实际反馈不回填B0。

原best固定320/40/40划分；先固定train前16和val前8完整episode。仅train16选幅度，之后固定：
arm和hand的contact `|expert-base|` p99均为2，因此两组scale均为2，不裁残差标签。
这是native命令幅度：arm是原velocity-style积分、hand是原绝对关节目标映射，**不是**统一米/秒。
scale=.1只覆盖arm19.20%、hand33.13%接触标签；对应contact RMSE可达下界.9064/.6421。
scale2接触误差下界为0的infimum；严格无接触旁路仍使全train16 RMSE下界.5467/.3453。
有限tanh不能精确达到端点，这些下界不代表可学到或安全保证。

监督为`command_clipped_pre_delay`，loss为contact时刻`.5*arm_MSE+.5*hand_MSE+1e-4*delta_L2`，
分组内取均值，防止22手关节仅因维度主导。无接触不更新，仍因果更新memory；无padding，尾chunk取实际长度。
完整episode持续memory，TBPTT32仅detach；CPU FP32、batch1 episode、AdamW1e-3、weight_decay1e-4、
grad norm1，无EMA/LR schedule。全划分重新从相同零头初始化B/C，最多5轮，不把pilot算成额外全数据epoch。

C与B结构、初始化、顺序和有效更新量匹配；gate/contact loss mask在真实未屏蔽数据上预先计算。
C仅在网络入口把world/local力置零，力差分、norm、contact/onset/release随之消失，FK/time保留。
**C仍使用真实触觉gate，不是严格无触觉对照。** B的validation选epoch，C取相同epoch，保持更新量可比。

## 已完成验收与阶段状态

- 7项针对性测试通过：零残差、无接触exact-base、[-2,2]可达、独立参数/冻结、持续memory、完整/逐步等价、C力派生屏蔽、v3拒绝及资源基础测试。
- 24条pilot缓存完整：train16共3226控制观察、1894个contact；val8共1758观察、1088个contact。
- 两轮pilot B/C各160有效更新、32次完整train episode遍历。val加权MSE从.727845降至B .623025、C .620782；通过学习信号门槛，已进入320/40全划分缓存。不能称触觉收益。
- 真实首条示范13步只读回放：step0/6/12采样，memory计数1/7/13；cache与online base/plan最大差1.91e-6，raw residual8.94e-8，过去真实专家命令逐位相同。没有为凑接触扩展此检查。
- 完整训练、held-out test与12条闭环尚在本轮执行中；最终结果另追加，不能把当前pilot当最终交付。

## 固定闭环与资源计划

预分配seed981001/981002，autonomous和expert_rollin各A/B/C，合计12条，不替换、不并入正式统计。
A明确base bypass；B训练残差；C匹配更新的force-masked残差。roll-in由released expert真实env.step，
首次连续3个有效观察的max finger force>.1N交接（最多前240步寻找）；各组从reset因果预热全部memory/plan，
比较原始初态、专家命令prefix SHA及switch状态，不能清memory、瞬移或提供未来专家动作。
无switch单列不可用；成功条件/原生timeout不变，保留完整自然终止。

独立`residual_session`进程连续采样ResourceGuard，覆盖worker导入、CUDA/Kit启动、等待、异常和退出。
完整60秒出现且batch/window均值≤45%才准入；4GiB显存、10GiB RAM、12GB磁盘；只睡自己的chunk/控制调用。
首次GPU计算起90分钟、整个session6小时不可换目录重置，计入等待。所有worker经run_cpu.sh，nice19、CPU22/23、两线程。
整卡采样包含外部负载，不能称硬quota或进程归因；不改变120/60Hz模拟时间。

等待记录限制：早期pilot/full缓存累计包括实测准入等待和请求的pacing sleep，未单独计调度超时；
13步causal检查未持久化独立wait总量，标为未知。后续worker独立保存实测准入/节流时间，
包括预算异常退出。连续整卡GPU遥测覆盖所有这些阶段；wait归属缺口不等同于GPU缺测。
