# B0/B1 命令模仿：实现、验收与后台训练

2026-10-29 当前阶段。400条正式数据已交付；本轮按用户新授权训练两个普通命令DP，
之后评估50组新合法初态。用户允许全部GPU算力，历史50%限制不再用于本轮训练。
进度会变化，实时入口见下文；本文件不把启动、短诊断或早期loss下降当作成功率。

**最新审查入口：** 本文末尾“18:54审查补充”及
[`evidence/baseline_review_20261029.json`](evidence/baseline_review_20261029.json)。
前面的17时进度和15小时外推保留为历史，不能作为最新ETA。下文时间均为Asia/Singapore。

## 研究定义与调用关系

| 项目 | B0 | B1 |
| --- | --- | --- |
| 类 | `models/nominal.py::NominalDiffusionPolicy` | `models/tactile_dp.py::TactileCommandDiffusionPolicy` |
| 共同输入 | 三相机RGB-D、depth/camera valid与时间，58D关节位置/速度，类别task ID、可选外部goal及mask | 相同输入，独立参数 |
| 额外输入 | 无触觉 | 五指世界/局部聚合法向力、编码器FK末节位置/姿态、两个物理样本时间与有效性 |
| 条件状态 | 视觉/本体/context GRU，128D | 独立视觉128D、五指各32D→64D、融合GRU128D |
| 参数量 | 2,003,085 | 2,166,613 |
| 输出 | 12×29原生命令序列，[-1,1] | 相同 |

`ManifestSequenceDataset(stage="production") → EpisodeSequenceDataset.chunks →
train_dp._batch_episode_chunks → _condition → loss_from_condition` 是训练路径。
部署由 `rollout.load_command_controller` 选择 `NominalController` 或
`TactileCommandController`，最终 `compose_action → 原env.step`，只走一次原动作链。

三相机顺序固定 `fixed_high/fixed_side/wrist_rgbd`，v4、448×252。
`RGBDSpatialEncoder` 共享同一模型内部的相机编码权重：RGB为ResNet18到layer2/GroupNorm，
无预训练；Depth+有效性为独立卷积；空间矩与3×5网格保留位置，每视图128D。
这些是待训练的空间特征，不是已有物体/孔口语义关键点。B0/B1之间不共享可更新权重、
optimizer或视觉缓存。

B1额外拥有FK特征和融合GRU，比较结果不能归因成“只增加触觉力”。触觉仍不是压力图、
切向摩擦力或可靠滑移；局部力差单位N，位置差/实际dt为base系m/s。
B1的触觉编码器由真实命令扩散损失训练。Correction是另一个独立网络，目前仍未训练，
不受本轮授权驱动。未来B2/B3冻结相应base再研究残差；去预加载和实际到达位置监督暂缓。

## 监督、时钟与信息边界

`obs[t]`监督 `transition/command_clipped_pre_delay[t:t+12]`，尾部mask。
29维按canonical关节顺序，是入口裁剪后的命令，不是角度/执行器目标/延迟后旧命令。
专家当前/未来命令不进入condition；两组均 `own_plan_mode=none`，在线也强制自身旧计划与
上一预测mask=false。纯专家数据没有自身计划，不能用专家未来动作冒充。

保留120Hz物理和60Hz动作处理。每个60Hz观测更新视觉/本体记忆；B1再按顺序处理两个
120Hz接触/FK样本。每六步重规划，仍逐步执行60Hz原生命令，不乘六、不改decimation。
arm增量积分和hand绝对目标映射保持原规则及延迟/限位/平滑。仿真等待网络不代表实时控制。

## 多轨迹与训练协议

- 原400条/81,894步不重采。固定seed41按pair_id划分320/40/40；训练集64,825步。
  proprio均值/标准差只拟合训练集。50个完整训练epoch，不缩减预算。
- 每模型B=2、T=32，图像 `[2,32,3,4,252,448]`；condition `[2,32,128]`，
  标签 `[2,32,12,29]`。一组两个独立episode从step0开始，直至较长者结束；短者尾部mask，
  组内不装入新episode。所有分支状态按行独立。chunk边界只detach，组结束才清空。
- B1同时保留visual/tactile/fusion state和上一个**真实有效**物理样本。
  padding不能成为下次差分参考；首包无效不虚构力。时间晚于obs的触觉显式拒绝。
- loss同时mask当前观测和未来目标。有效步、目标数、padding和optimizer更新分别计数。
  无梯度累积；AdamW `lr=1e-4, weight_decay=1e-6`，clip norm1，B0/B1相同。
  B=2减少了更新次数，不能把所有耗时变化都称为硬件提速。
- BF16 autocast；原始时间float64、mask bool。FP16可选时先unscale再裁剪，保存GradScaler。
  图像每12张做activation checkpoint，反向重算，未detach/冻结视觉、未缩短T。
- 每模型一个CPU预取线程，最多一个待消费chunk，独立HDF5句柄；没有fork继承句柄，
  不缓存学习中的视觉隐变量。当前不使用pinned memory：额外复制尚无测得收益。
- 每8组原子保存 `last.pt`，收到停止请求则在下一整组结束保存；保存完成组游标、optimizer、
  CPU/CUDA RNG、AMP、划分与输入契约。预取不算已消费。中途恢复最多重做尚未保存的组，
  不用零状态跳到episode中间。验证阶段中断可从完成训练的保存点重做验证。
- epoch验证按固定权重完整因果展开；依据validation loss保存有限的 `best.pt/last.pt`。
  正式test先真正加载best，之后才计算命令误差与闭环评估。17:45已保存首个完整epoch的best，
  这不是最终选定权重。

## 实际验收证据与局限

可公开小统计见 `evidence/command_training_acceptance.json`；原始数据/权重留在outputs。

1. 实际训练split中的119步与509步轨迹，共628个有效控制时刻：B0/B1逐条、B=2、交换行、
   尾部padding及各分支state最大误差均≤2.99e-7（FP32固定权重）。早期报告539是全400的
   另一选样口径，不能替代本次训练split中的509步证据。
2. 实际obs[t]/未来12命令逐值对齐；两个力样本时间≤当前obs且有效时严格递增。
   单元测试另覆盖未来观测编辑不改前缀、单行reset不改另一行、B0额外字段拒绝。
3. 同一真实诊断块、固定噪声8次更新：B0 loss 1.132095→1.019368，B1 1.176422→1.025207。
   RGB/Depth/history/denoiser和B1 tactile/fusion均有有限非零梯度及参数变化。
   这是有预算的训练链诊断，不是正式表现。
4. 新checkpoint由真实仿真controller加载，8步因果回放条件误差≤1.79e-7，输出29D有界，
   no-own-plan保持关闭，推理没有改变环境全局torch RNG。真实训练last也已重新加载成功。
5. 开发仿真完成2组合法配对（另1非法起点排除），共4×600控制步、544.25s；初态匹配，
   都是预算删失，不能据此发布正式成功率。训练后50组使用不同seed区间981000起。
6. 全部回归测试61项通过（一个已有requests依赖warning）；编译与diff whitespace检查通过。
   其中新增指标oracle只核对第0/6步重规划和目标尾部mask，不是训练标签或演示数据。

正式评估由 `evaluate_baselines` 用一个渲染的CPU物理实例、原v4相机/120/60Hz、无专家接管。
每个预定候选seed先检查几何合法性，再比较实际关节/目标/物体等初态，随后评估两模型。
合法任务失败进入分母；预算删失单列，不能宣称自然终止率；同初态不代表不同策略有相同
后续扰动曝光（原扰动门控依赖轨迹）。加载真正best，不用last替代。正式评估尚未执行。

## 吞吐与资源：测得什么

旧任务在epoch1退出且无checkpoint，不能恢复。旧日志100块约180s但块有效步口径不完整，
先前“后续块0.04s”等快照不代表完整训练吞吐。

同12条真实轨迹、2,380有效步的短测：

| 配置 | 墙钟秒 | 有效步/s | 更新数 | padding步 | 峰值CUDA allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 batch=1、无预取 | 37.87 | 62.84 | 81 | 212 | 1.00GiB |
| B0 batch=2、预取1 | 39.77 | 59.85 | 47 | 628 | 1.17GiB |
| B1 batch=2、预取1 | 39.18 | 60.75 | 47 | 628 | 1.17GiB |

这组短测不足60秒，不能报告完整60秒均值，也未证明batch=2更快。选择B=2用于已经验收的
独立轨迹batch，未把B=4当必要目标：读取占主导，增加batch会提高内存和改变更新分组。
未完成严格同条件的顺序/并行全周期对照；两个模型已后台并行，不宣称达到最优调度。

首次正式并行100块：5,305有效步，约178s，读取等待134–139s、compute37–43s。
定位到RGB `astype/divide/concat`重复大数组；改为一次分配RGB-D后直接除255写入。
相同8块CPU读取短测13.073→8.764s；真实8帧三视角RGB/Depth逐位相等。
缓存/其他负载未控制，不能把1.49倍当完整训练提速。

2026-10-29 17:34新加坡时间，在两组均完成51组/372次更新后恢复优化版，无已保存工作丢失。
17:36快照两组均525次更新、27,899/64,825训练步（epoch1）；累计约43有效步/s。
按累计速率加15%验证/保存/test暂估尚需约24小时；这个估计尚未经过完整epoch校准，
后续状态会继续更新。开发仿真约4.41控制步/s，50组×两模型×2400步上界外推约15小时，
早终止可能减少，加载/非法起点/资源等待会增加。训练与正式评估ETA必须分开。

**完整首轮修正（17:46实查）：** B0/B1均完成epoch1并进入epoch2，每组真实训练64,825步、
756,780个有效horizon目标、1,221次更新、13,319个padding位置，更新跳过0。
B0训练1111.72s＋验证103.30s，B1训练1118.28s＋验证100.96s；包含重启前的累计有效训练耗时，
不含停机间隔。日志`epoch_s=681/685`仅是恢复后进程本轮耗时，不能据此低估整epoch。
验证loss B0=0.634335、B1=0.618740；两份best均由实际在线controller加载且参数有限。
剩49轮按完整首轮估16.6小时，离线最终test另留数分钟，约10月30日10:30新加坡时间。
首轮前段为旧读取实现，后续完整周期将进一步修正ETA；暂不承诺时间或策略表现。

后台unit `p2p-dp-command-20261029.service`；runner2429427、B0 2429429、B1 2429455为17:35实查PID。
进程组MemoryHigh12GiB/MemoryMax16GiB、禁用该组swap；每模型CUDA allocator上限36%，
块前要求整卡至少4GiB空闲和系统至少10GiB可用。不限利用率；内存紧张只等待本任务。
这些保护降低风险，不保证其他进程突发分配时永不OOM。现有系统swap已满，不能再放大无界队列。
两进程各4CPU线程/独立CPU亲和性；未操作外部抓取生产任务。旧异常C++索引子进程已暂停，
项目本地忽略的`.vscode/settings.json`禁用C++索引，VS Code主进程保留；操作记录位于
`outputs/visuotactile/run/ide_indexer_memory_action.json`，不能无确认地恢复持续膨胀索引。

恢复进程17:46累计整卡采样均值分别53.1%/57.3%，完整60秒峰值77.5%/79.5%，显存采样峰值
4201MiB，主动内存等待/缺测均0；两条是不同采样相位的整卡测量，不能相加或称本任务独占利用率。
该时段systemd组内存峰值约10.7GiB、系统可用约36GiB，组内OOM事件0。资源JSONL保留重启前后
记录，最新resources.json仅描述当前进程生命周期，不冒充跨重启全程合并统计。

## 18:54审查补充：运行身份与已解决问题

用户明确要求不中断当前训练。本轮未停止、重启或改变两组训练配置，未增加正式实验。
`76bb58effc7be462dbad8cbe6bda4905aa520f5f` 已在本轮审核后推送到 `origin/main`；
检查了此前未公开的两个提交（28个路径），只含代码、测试、文档，保留MIT与vendor许可。
没有上传数据、权重、日志、私有IDE配置或凭据。新评估改动单独提交。

运行身份不能只看当前HEAD：17:34启动的 `launch.json` 写的是 `4714c96` 加当时工作树；
其中10个训练相关源码SHA256逐个与76bb58e的Git对象比对，全部一致。具体hash在本次公开小证据中。
当前trainer已导入的模块不因磁盘文件更新自动替换。本轮只改controller/evaluator及测试、文档，
没有改运行中的train_dp/模型/reader。runner完成两组训练后新建的评估进程才会加载本轮代码；
评估plan会保存实际源码commit及文件hash，不能把本轮评估版本反写为训练运行版本。

| 此前问题 | 真实修复位置与现在行为 |
| --- | --- |
| 读取慢、大数组复制和低效索引 | `compact.py::VirtualArray.__getitem__` 用连续切片/唯一索引重建；RGB-D一次分配。`train_dp.py::_prefetch` 只有一个待消费chunk，无无限缓存。真实RGB/Depth逐位往返与索引测试已通过。 |
| 只在epoch结束保存，重启丢失大量工作 | `train_dp.py::_save/main/group_checkpoint` 原子保存每8个完整episode组；保存optimizer、CPU/CUDA RNG、AMP、组游标/累计统计。resume核对split、数据hash和模型配置。17:34两组从group51/372次更新真实恢复。 |
| 训练权重格式不能由在线接口直接解释 | `_save` 写 `command_checkpoint_version=2`、输入/历史契约、metadata、归一化buffer；`rollout.py::load_command_controller` 严格验证并分派B0/B1，拒绝旧格式或错分支。 |
| 在线自身计划、归一化或B1记忆与离线不一致 | `NominalController.act` 强制no-own-plan；`TactileCommandController.encode_current` 使用独立visual/tactile/fusion持续状态；归一化从state_dict恢复。非零均值/非单位std的实际加载测试已补强。 |
| best/last身份混用 | `train_dp.main` 完成后明确重载best再做test；本轮 `evaluate_baselines.checkpoint_identity` 核对完成report、best路径、variant、selected_epoch、manifest与SHA256，加载后再次核对。测试拒绝last和错epoch。 |
| CPU网络推理、memory2仍复制整段图像窗口 | `load_command_controller(device=...)` 与 `act/encode_current` 支持CUDA；memory2只保留生命周期检查与持续状态，去掉未使用的原图窗口复制，不跳过任何观测。固定权重9步、两行异步reset前后输出/状态完全相同。 |
| Kit可能掩盖异常退出，未满50组却出现report | `evaluate_baselines.publish_report` 只有完整cohort才写report.json；候选耗尽保存incomplete_report.json，后台不能误认完成。保留软件错误与部分结果。 |

## 真实训练配置与完整周期吞吐

下表采用最新完整epoch的稳定统计，原始计数见本次证据JSON。参数量直接从实际配置模型计数，
没有冻结参数；no-own-plan的未启用条件不等于冻结编码器。

| 项目 | B0 | B1 |
| --- | ---: | ---: |
| 轨迹batch B / 时间块T / 梯度累积 | 2 / 32 / 1 | 2 / 32 / 1 |
| 相机数 / 图像H×W | 3 / 252×448 | 3 / 252×448 |
| 参数device/dtype；autocast | cuda:0 / FP32；BF16 | cuda:0 / FP32；BF16 |
| 可训练 / 冻结参数 | 2,003,085 / 0 | 2,166,613 / 0 |
| 每epoch有效控制时刻 | 64,825 | 64,825 |
| 每epoch有效Horizon目标 | 756,780个29维命令 | 756,780个29维命令 |
| 每epoch实际optimizer更新 / 跳过 | 1,221 / 0 | 1,221 / 0 |
| 每epoch验证控制时刻 / 目标 / 更新 | 8,670 / 101,400 / 0 | 8,670 / 101,400 / 0 |
| 稳定窗口 | epoch3–5 | epoch3–5 |
| 完整训练 / 验证 / epoch末保存 秒 | 831.15 / 106.10 / 0.074 | 865.26 / 105.77 / 0.079 |
| 训练有效控制时刻/秒；目标/秒 | 77.99；910.52 | 74.92；874.63 |
| 含验证/末保存的控制时刻/秒 | 69.16 | 66.75 |

训练墙钟包括读取等待、周期性组保存；`save_seconds`只测epoch末best/last，不包含所有组保存。
组保存与训练末验证前的额外last保存没有独立完整计时，不能编造“全部保存总耗时”。
每个有效目标是一个29维向量，不能与标量分量数混用。两组数据和更新预算一致，未降分辨率或缩短历史。

**约4.1GiB显存的解释：** 双trainer常驻整卡采样为4201MiB；BF16减少激活，
每12张图activation recomputation用额外计算换取更少保留激活，及时释放已反传的图和旧chunk引用，
预取有界。模型本身仅约200万参数。CPU reader一次分配主要节省主存和复制，不是显存降幅的全部原因。
没有冻结视觉或缓存脱离梯度的编码。短诊断单模型CUDA max allocated约1.17–1.20GiB；
4201MiB是含CUDA缓存及其他任务的全卡采样，不是每个kernel的精确活跃张量峰值。

当前进程持续两秒左右采样，全卡加权均值约53.9%/54.4%，最大完整60秒均值78.29%/79.52%；
两条采样相位不同，不能相加。短Isaac检查与训练同时运行时全卡峰值增至7381MiB，检查退出后恢复4201MiB。
trainer缺测时间/ResourceGuard主动等待均0；短检查自身有19.87s启动采样间隙，不能隐藏或计算伪60秒均值。
组内内存峰值约12.0GiB、系统可用约35GiB。保留12/16GiB组内高/硬阈值、禁swap、4GiB显存和10GiB
主存余量。18:58核查cgroup的MemoryHigh回收事件超过22万次，OOM/oom_kill均0；
这是软阈值下的内核回收，不等同于OOM，但可能增加读取/计算墙钟。没有独立测量回收耗时，
不能将ResourceGuard等待0解释为完全没有内存压力；本轮未改阈值或打断训练。
无利用率上限，未碰外部有效任务；并未证明当前配置达到最佳硬件效率。

### 已有真实分支更新证据（复用，不再诊断训练）

8次固定噪声真实诊断块，数字是L1总量，不是性能指标：

| 分支 | B0梯度 / 参数变化 | B1梯度 / 参数变化 |
| --- | ---: | ---: |
| RGB | 12.404 / 166.421 | 7.699 / 196.603 |
| Depth | 4.814 / 23.837 | 2.825 / 25.987 |
| 视觉/本体历史 | 8.663 / 90.611 | 7.482 / 82.483 |
| Denoiser | 134.350 / 337.589 | 159.366 / 313.777 |
| 五指触觉 | 不属于B0 | 6.981 / 11.938 |
| 融合GRU | 不属于B0 | 10.400 / 42.550 |

## 关键正确性证据索引

本轮65项通过（此前61＋4项），一个已有requests依赖warning。以下分别说明覆盖内容，
不是用测试总数替代正确性；完整原轨迹不公开，允许公开的聚合数值位于两份evidence JSON。

| 关键项 | 已执行证据与边界 |
| --- | --- |
| obs[t]→command[t:t+H] | `validate_command_training.main` 用真实119/509步训练轨迹逐值对照 `command_clipped_pre_delay`、H12及尾mask；`test_sampled_metrics_replan_phase_and_episode_tail_masks` 另检验指标重规划相位。后者的人工oracle不是数据标签。 |
| B0无触觉/专家答案 | `test_future_tactile_rejected_and_b0_rejects_extra_fields` 拒绝expert字段；`test_nominal_rejects_tactile_and_privileged_keys`及`test_nominal_inference_is_invariant_to_tactile_and_diagnostics` 检验allowlist和输入扰动不变。 |
| B1触觉因果性 | 真实文件时间逐值≤obs；上述future测试拒绝未来力；`test_tactile_reference_with_initial_missing_and_padded_tail` 保留最后有效物理样本。 |
| 独立状态、尾mask、局部reset | `test_causal_packing_swap_padding_and_row_reset`（两模型）比较完整/分块/单条/交换/未来编辑/单行reset；`test_controller_updates_between_replans_and_only_episode_reset` 检验每步更新、子目标不清零。 |
| 固定权重单条/批处理 | 真实628时刻最大误差≤2.99e-7；各visual/tactile/fusion分支也比较，非只检查shape。 |
| 保存恢复 | `test_resume_checkpoint_restores_optimizer_rng_and_next_update` 真正保存/读回后下一次更新一致；两份实际last在17:34恢复继续，已保存372次更新不丢失。不是任意episode中途的零记忆恢复。 |
| 在线加载/动作/原执行链 | `test_checkpoint_actual_controller_online_no_plan_rng_and_causal_state`，两轮真实开发仿真；本轮CUDA两条各96步，29D命令与submitted逐位相同，裁剪逐位相同，延迟索引0/1/2，target有限，关节真实运动。`test_action_delay_trace_rng_and_formulas`与`test_native_arm_hand_semantics_and_canonical_mapping` 检查原公式。 |
| no-own-plan与归一化 | actual-controller测试已加入非零mean/非单位std，并逐步对照离线condition；在线强制计划mask=false，不把外部packet动作注入。 |
| 新GPU路径 | 小输入固定权重CPU/CUDA condition最大差1.08e-6；使用相同初始噪声的8步DDIM动作最大差3.94e-6，CPU/CUDA全局RNG不变。CPU/CUDA私有生成器算法不同，不宣称同数字seed产生逐位相同动作。 |
| best/完成标记 | `test_evaluation_requires_actual_best_identity`、`test_incomplete_evaluation_never_publishes_completion_marker`。 |

## 正式评估协议2、计时与ETA

仍是50组新合法起点、两模型各50条，不筛成功、不专家接管、不改物理或原超时。
一个持久Isaac实例，两个checkpoint各加载一次；固定8m配置，目前实际只用一个环境，
没有把四环境采集的吞吐冒充策略评估吞吐。多环境需要独立RNG配对与异步生命周期验证，
本轮先做直接可验证的GPU推理优化，不为一次评估扩大结构。

CPU物理、GPU RTX；模型参数/三路图像/视觉和触觉GRU/扩散全在cuda:0，FP32推理、TF32关闭。
每步更新记忆、每6步按checkpoint的8个DDIM步重规划H12，输出搬回CPU原动作链。
仍120/60Hz模拟时钟，墙钟并不实时。逐条NPZ只保留动作/目标/关节/状态/接触/外力/时钟；
首个配对每120步保存一次三路并排PNG，其余不写RGB-D全轨迹、不默认重复导出MP4。

**旧15小时从何来：** 旧CPU推理开发运行4×600控制步共544.250s（含reset等循环开销，
不含App/场景启动），544.250/2400×100×2400=15.12小时，是每条跑满外部上限的线性外推。
旧policy.act中位140–151ms；场景日志3.204s、启动仿真1.332s；未分解其视觉/扩散/reset/写盘耗时。
末段与训练重叠，不能当无争用benchmark。原生dt浮点ceil导致 `max_episode_length=601`，
不是600；旧600步全部在首次自然超时前截断。这只支持接线检查，不支持自然失败率。
当前2个子目标加撤手，子目标命中重置阶段计数；保守总原生允许量3×601=1803步，
外部2400保持不变，并显式验证不短于原生允许量；无进展一般601步超时，掉落等可更早结束。

**本轮短CUDA实测（与两trainer竞争资源）：** 一组合法配对，各96步；另1非法起点前置排除。
首个尝试在reset前取metadata产生TypeError，零有效轨迹，证据保留；修为reset后读取，再在新目录验证。
重试运行App11.041s、场景4.831s、两权重加载0.934s，全部reset0.769s；完整进程54.260s，
报告循环含reset共34.080s。仍是诊断权重/主动删失，未声称学会任务或正式成功率。

| 96控制步的墙钟分解 | B0 | B1 |
| --- | ---: | ---: |
| 视觉/本体（B1含触觉与融合）记忆更新 | 2.959s | 3.249s |
| 16次DP重规划 | 0.997s | 1.221s |
| policy.act总计，含转移/包装/同步 | 4.238s | 5.902s |
| env.step（物理＋渲染＋wrapper） | 11.533s | 11.057s |
| NPZ写入 / 少量PNG | 0.0033 / 0.0473s | 0.0032 / 0.0457s |
| 完整控制循环 | 16.052s | 17.228s |
| act中位 / p95 | 27.38 / 86.07ms | 37.67 / 215.78ms |

物理与渲染没有可靠独立计时，只报告组合；没有以同步仿真宣称满足60Hz墙钟实时。
短循环均值0.1733s/控制步，不足以给出成熟策略的真实平均episode长。
若100条都在首阶段601步结束约2.9h；若都耗完原生保守1803步约8.7h；
机械套用外部2400则11.6h。加载/非法初始化/资源等待另计；训练结束后无双trainer竞争，仍不预先承诺提速。
正式前两组完整配对计入50并更新剩余ETA，不另挑好结果试跑。

当前以完整稳定周期估B0在10月30日06:31、B1约06:58完成50轮，最终离线test/采样指标暂各留10分钟，
保守对外报告**训练及离线指标约10月30日07:00–07:30，正式评估约10:00–16:30**。
这是有假设的时间范围，后者依赖实际自然终止长度，不是15小时定值或硬保证。
真实best的epoch/hash只有验证完成后冻结；formal report记录它们，不按test或闭环结果换模型。

## 日常入口

见 [RUNBOOK当前命令](RUNBOOK.md#b0b1-后台训练与恢复2026-10-29-最新)。数据、权重和日志均不公开。
`run_baselines --status`不初始化CUDA，不启动任务。停止用根目录STOP，恢复前检查状态、删除STOP、
重新启动同一unit；runner从last恢复，完成训练后自动调用已授权50组评估。评估半成品保留并报错，
不静默覆盖。只有两份训练report和完整评估report都存在且标明完成，才算本轮全部完成。
