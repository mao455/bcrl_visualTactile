# B0/B1 命令模仿：实现、验收与后台训练

2026-10-29 当前阶段。400条正式数据已交付；本轮按用户新授权训练两个普通命令DP，
之后评估50组新合法初态。用户允许全部GPU算力，历史50%限制不再用于本轮训练。
进度会变化，实时入口见下文；本文件不把启动、短诊断或早期loss下降当作成功率。

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

## 日常入口

见 [RUNBOOK当前命令](RUNBOOK.md#b0b1-后台训练与恢复2026-10-29-最新)。数据、权重和日志均不公开。
`run_baselines --status`不初始化CUDA，不启动任务。停止用根目录STOP，恢复前检查状态、删除STOP、
重新启动同一unit；runner从last恢复，完成训练后自动调用已授权50组评估。评估半成品保留并报错，
不静默覆盖。只有两份训练report和完整评估report都存在且标明完成，才算本轮全部完成。
