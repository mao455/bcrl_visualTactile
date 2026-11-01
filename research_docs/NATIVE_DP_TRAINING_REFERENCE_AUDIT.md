# Native DP 训练源码审计（2026-11-02，protocol V2）

基线为 `3e11d62`；新实现位于独立 `feature/native-dp-training-v2` worktree。旧 run02 不停机、不换源码、不修改结果。
本报告区分源码事实、已测结果和候选方案。没有把官方默认超参数或训练结束视为成功证明。

## 参考身份与读取范围

- [DP 固定 revision](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)：
  `workspace/train_diffusion_unet_lowdim_workspace.py`、`policy/diffusion_unet_lowdim_policy.py`、
  `common/sampler.py`、`model/common/normalizer.py`、`dataset/robomimic_replay_lowdim_dataset.py`、
  `env_runner/robomimic_lowdim_runner.py`、`config/train_diffusion_unet_lowdim_workspace.yaml`。
- [robomimic 固定 revision](https://github.com/ARISE-Initiative/robomimic/tree/d309eaecc18acf4152a830a895a6984b8ac71b05)：
  `config/diffusion_policy_config.py`、`exps/templates/diffusion_policy.json`、`examples/train_bc_rnn.py`、
  `utils/train_utils.py`、`utils/dataset.py`、`docs/tutorials/{training_diffusion_policy,using_pretrained_models,dataset_contents}.md`、`docs/modules/dataset.md`。
- 完整 URL/SHA 清单：`evidence/native_dp_v2/reference_sources.json`。本机源码在
  `outputs/visuotactile/native_dp_v2/references/`。最初请求的 robomimic `docs/tutorials/training.md` 返回404，
  已保留该记录并读取实际 `training_diffusion_policy.md`，不将缺失文档冒称已读。

## 当前训练设置审计表

| 项目 | 旧 Native 实际设置与代码位置 | 审计结论 |
|---|---|---|
| 数据/成功 | `native_data.prepare`；400条完整原生 `max_successes` + retract；320/40/40 | 本轮重读400原文件SHA及81,894步数值；400成功，不是只相信manifest |
| 长度 | train119–509/中位195.5；val136–539/202.5；test137–351/207 | 不等长；不可把一次segment称一条完整episode |
| 输入 | `native_data.robot_rows/geometry`；178D robot，S独有85D geometry | 当前target合法，issued/delayed严格t−1；详见protocol维度表 |
| 标签 | `native_data.labels`，同tick起H64 clipped pre-delay29D | 不是到达q、延迟后command或平滑target；部署原生delay仅一次 |
| 归一化 | robot/geometry train-only mean/std，std≥1e−4；action limits affine | 重算全部6个统计buffer误差0；没有验证/test污染证据 |
| 抽样 | `native_training.segments`：episode洗牌，episode内顺序96tick段；每12tick一个anchor | 每遍832段/5547 anchors；不是真正episode-balanced，也非随机独立anchor batch |
| 有效batch | 4段梯度累积；每段最多8 anchors | 最多32 anchors；每遍208更新，平均26.668 anchors/update；按有效future targets归一化 |
| objective | `NativePolicy.loss`：sample/x0，Loss B，arm7/hand22各50% | `dp_objectives.masked_objectives`排除无效尾部；不存在对padding直接计算MSE的错误 |
| 扩散 | 100步 squaredcos_cap_v2；DDIM16、eta0、clip_sample=False | 首选保留；最终原生命令clip[-1,1]。sample内部clip取舍需单变量实测，不凭注释改权重 |
| 优化 | AdamW lr1e−4、betas(.95,.999)、wd1e−6；500 warmup+cosine；grad clip1 | 与官方低维配置接近，暂无必须修改的证据 |
| EMA | 官方vendor EMAModel power.75，每optimizer step更新 | 模型/EMA/optimizer/RNG/游标均存checkpoint |
| 预算 | 三组各19000；每1000固定seed7091验证；≥5000且≥20遍、6次无0.5%改善早停 | 19000约91.35遍，不是19k个episode；最终是否收敛需结合闭环 |
| 选模 | EMA前12步native arm/hand均权MSE最小；单一采样seed | 需多seed报告不确定性及validation自主选模，不能只靠RMSE |
| 保存恢复 | 每100更新last、改善时best；精确配置比对、RNG/optimizer/cursor恢复 | 中间里程碑未保留；不能事后重建旧1000/3000/...权重 |
| Gate顺序 | data→同一新expert oracle两模式→各组network/fit1/fit4/诊断/pilot→三组formal→test40→train4→test20×3 | fit闭环 `passed=True`指正常完成，不要求success；是确定的放行缺陷 |
| fit真实门槛 | fit1≤1000updates arm≤.20/hand≤.10；fit4≤2000 arm≤.35/hand≤.20，均需≤零基线70% | fit1实际400、fit4实际500；这些是离线拟合门槛，不是任务验收 |
| 并行/瓶颈 | 各组顺序；单env Kit；episode反复读取；History前缀按当前权重重算；lock含wait/pacing | 新State只缓存原始数值窗，不缓存可训练encoder输出；新lease把wait/pacing移到锁外 |

train共有355,008个H64位置，其中53,675无效（15.12%），实际监督301,333；
同一原生动作可被重叠future窗口多次覆盖，近起点权重低，非“每个tick均权”。
当前没有接触阶段额外采样；本轮保留Loss B和物理时域，先改变batch构成与数据加载一种设置。

## 与成熟实现的逐项比较

| 当前项目 | 官方源码设置/位置 | 差异判断 | V2决策与验证 |
|---|---|---|---|
| H64/执行12/native60Hz | DP lowdim YAML H16/O2/A8；policy `predict_action` 从O−1取action | 数据时间轴不同，不能机械改16/8 | 保留1.067秒预测/.2秒规划；保存边界不连续、推理超200ms次数 |
| future-only标签从t开始 | DP sampler序列含观察前缀，`oa_step_convention`控制O−1 | 我方无额外O−1偏移是合理的 | 真实整episode逐步条件 vs 缓存条件测试；逐tick计划cursor |
| 4段累积、相邻anchor | DP YAML batch256、shuffle=True、accumulate1 | 有效相关样本较少，但不是已证实失败根因 | 新State固定batch32随机anchor无放回；optimizer/模型/LR不变；新目录、不可无损续旧训 |
| train-only统计 | DP指定robomimic lowdim dataset `get_normalizer`使用整个 `replay_buffer` 统计 | 官方此路径没有按train_mask裁统计；不能盲目照搬 | 保留我方严格train-only；400真实重算误差0 |
| 尾部0+valid mask | DP `SequenceSampler.sample_sequence`复制边缘；lowdim `compute_loss` assert无valid_mask | 我方适合可变长度自然成功数据；不是必须复制padding监督 | 测试无效位置对loss/梯度无贡献；记录各阶段窗口覆盖权重 |
| sample + DDIM16 | DP配置epsilon/DDPM100，代码支持sample；robomimic默认epsilon/DDPM100，可选DDIM10 | 都是可支持配置，不能从默认推断我方错误 | 暂不换objective/采样步数；旧权重先多seed离线/自主诊断 |
| Loss B组均权 | DP默认全维MSE；robomimic通用action MSE | 29D的7/22维不对称合理适配 | 保留原生尺度与B；禁止同时改LR、结构和控制 |
| AdamW/EMA/warmup | DP YAML与我方同lr/betas/wd/warmup/power；robomimic config lr1e−4/wd1e−6/warmup500 | 没有“优化器明显配错”的证据 | 保留，增加有效batch、遍历、grad norm和阶段计时日志 |
| 仅离线best | DP workspace执行runner、val loss、action sample，TopK配置monitor `test_mean_score`；robomimic例子保存best validation及best rollout success | 官方runner名为test不代表本项目可用test调参 | 我方只validation rollouts选模，分别保留offline/rollout best |
| 每段HDF5/episode加载 | DP dataset创建内存ReplayBuffer；robomimic SequenceDataset支持all/low_dim缓存 | 明确可避免的I/O | StateWindows一次加载原数值；与原条件逐点对比；视觉缓存只冻结DINO可复用 |
| History每段从0预热 | robomimic BC-RNN提供序列模型；DP默认固定窗，非持续GRU | 不能把DP短窗或滑窗叫完整历史 | 本轮不改变History状态梯度语义；State未过G4不花大预算重写它 |
| Official ResNet18 | `OfficialVision`: `weights=None`、GroupNorm、共享三视角；DP hybrid依赖任务encoder配置 | 确认从零初始化，无随机裁剪/增强；不是偷偷采用ImageNet | 保留旧baseline身份；预训练必须独立配置对照，当前未启动 |
| 并行环境 | DP runner可以多环境；我方原4lane收集却缺完整物理/RNG快照 | 不能由采集4env等价推出评估4env等价 | 当前单env持久Kit；4env需先逐lane一致性验收 |

官方代码也有具体版本与任务差异。例如指定DP workspace的累积更新条件以global_step%accumulate判断，
默认accumulate1时没有此歧义；我方不会移植该写法来改变现有四段累积行为。
