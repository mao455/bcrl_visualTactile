# Tight-Insertion：接管入口相位修复、匹配重训与开发验证

本报告截至 2026-11-22（Asia/Shanghai）。入口修复已经进入实际训练，A/B 选中 DP 的新增 50 条纠偏、334 个合法入口提交槽全部达到 50 次不同成功 optimizer step；完整 DP + Transformer 的 96 次开发 rollout 已完成。入口专属同历史误差、补录短诊断及冻结后终测尚未完成，因此本报告是阶段交付，不能当作原任务全部完成或候选已晋升。

开发结果出现更多自主成功，同时丢失旧成功。独立开发 A 从原 R2 的 0/8 到入口候选 2/8，B 从 3/8 到 4/8；两组合计只保留原 R2 的 1/3 个旧成功。历史开发回归的原 R2 三个旧成功全部丢失。现有结果不支持“入口修复已保证闭环恢复能力和旧能力保持”的结论。

## 实验身份和证据边界

唯一工作区为 `/home/CNS2025331827/docs/bc+rl_RSS`。本轮训练 owner、恢复 owner、后端配置和开发 plan 均记录实际源码提交 `90b0a7ccebd350de06d2f7c9c83bf2a652c80f21`；实际文件身份另由各配置中的 `source_sha256` 固定。原 130-50 发布提交为 `9417461`，两轮 adaptive DAgger 发布提交为 `248aec3`，不能用这两个发布提交替代各历史阶段的执行源码：旧独立审计记录 R1 训练 `e7502d78f4`、R2 训练 `a076a8274c`、R2 恢复 `fed146f224`、原终测 `0e914563c2`。

本轮训练 root 为 `outputs/visuotactile/run/tight_entry_phase_20261121_01`，恢复 owner 为同名 `_recovery01`，开发 root 为其下的 `development_20261122_01`。训练 owner 于 11 月 22 日 05:56:04 正常退出；开发 owner 于 09:45 启动、10:44:48 正常退出，16 波全部结束，子进程均已回收。早期 `completed_training_status_20261122.json` 是 09:38 的训练快照，其 `closed_loop_rollouts_completed=0` 已由之后的开发汇总更新，不能解释成当前仍未跑开发。

公开证据位于 [本轮证据目录](evidence/tight_entry_phase_20261121_01/)；其中 `audit/` 保存独立核验及逐入口表，`run/` 保存运行合同、训练 epoch 验证和选模，`development/` 保存 16 波 plan、初态检查及逐例结果，`recovery/` 保存中断与恢复回执。原始 HDF5、视觉载荷、checkpoint、nominal 缓存及视频保留在本机，按 manifest 和 SHA 取用，不放入 Git。原审计另见 [旧独立审计](evidence/tight_adaptive_audit_20261121_01/REPORT.md)。

## 独立复现：缺失的是入口条件和提交槽

[independent_phase.py](evidence/tight_entry_phase_20261121_01/audit/independent_phase.py) 在修改采样器前，显式导入 `outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS` 中的历史生产 loader 和采样器，读取实际 HDF5。它以原 130 池与 R1/R2 train identity 的集合差重算新增 29 + 21 = 50 条纠偏，再分别读取 A/B 的选中 checkpoint 访问记录。结果为 50 个唯一纠偏、100 条角色核验行；48/50 首合法标签不在规划边界，历史验证器拒绝 96/100 条角色入口行。分母不是 100 条独立纠偏。

代表原始轨迹为 `r10_B_layout10_train_28817004_contact`，位于旧 R1 `collection00_recovery02/.../demonstration/data.h5`：

| 量 | 实际值 |
| --- | --- |
| switch / first_legal | 108 / 110 |
| 原生 12-tick anchor / 首合法提交槽 | 108 / slot 2 |
| 轨迹长度 / 合法标签数 | 266 / 156 |
| anchor108 的 24-slot target mask | slot0/1 为 false，slot2..23 为 true |
| 入口提交槽 | slot2..11，对应绝对 tick110..119 |
| 首合法槽 target 与原始对应标签误差 | A/B 均 0 |
| 历史 epoch1 覆盖补齐 | 加入 anchor110、172，没有108 |
| 历史选中 checkpoint 的最早 anchor | 110；108 未出现 |

在线每 12 个控制 tick 生成一次 24-tick 计划，前 12 tick 进入提交队列；tick108 的计划因而以 slot2 控制 tick110。`recovery_onset_commit12_v1` target-window 已允许以真实学生前缀为条件，在 anchor108 的 slot2 开始监督合法专家后缀。旧 `validate_training_anchors` 却要求 anchor 自己的监督 mask 为 true，抛出 `Selected training starts must have true action supervision`；随机 anchor 的总体和 greedy 覆盖补齐也只从合法绝对 tick 取起点。这使 tick110 在较晚历史条件下反复受训，而在线入口 anchor108 的 slot2 没有受训。

这不否定旧的绝对 tick 曝光结论。绝对 tick 覆盖与“原生入口 anchor、提交槽、绝对 tick”覆盖是两个口径；[entry_coverage_before.csv](evidence/tight_entry_phase_20261121_01/audit/entry_coverage_before.csv) 和 [independent_phase.json](evidence/tight_entry_phase_20261121_01/audit/independent_phase.json) 保存逐条原始来源、mask、target、采样回执和历史访问数。

## 采样修复及实际训练准入

[anchor_sampling.py](../visuotactile/anchor_sampling.py) 新增 `grasp_random_legal_cover_native_entry_v2`：先执行完全相同的旧随机采样和覆盖补齐，再为每条纠偏追加 `floor(first_legal/12)*12`，已有则去重。只放行这个确切原生入口，不任意放行其他无标签 anchor。随机 anchor 增广继续保留，原始非纠偏轨迹的采样不变。

[entry_exposure.py](../visuotactile/entry_exposure.py) 的 `EntryLedger` 从生产 `labels(...,24)` 取入口前 12 槽的实际合法 mask。贡献只来自精确原生入口、正监督 mask 与提交槽；同一 optimizer step 内重复 identity/tick 只计一次。生产 B2 训练在 `execute_update` 成功返回后推进更新号与 ledger，台账随 checkpoint 保存、恢复，并参与选模准入。祖先、未提交中断更新、较晚 anchor 对同一绝对 tick 的覆盖都不能补入口年龄。

预测窗口仍为 24 tick、提交仍为前 12 tick，控制仍为 60 Hz、规划仍为 5 Hz。A 监督实际到达位置 `q_reached`；B 监督实际专家最终绝对命令 `u_expert`。两组保留完整真实学生历史，prefix、两个切换过渡 tick、padding 均屏蔽 loss；网络、目标、专家裁剪、可达投影和执行器规则保持原合同。

[生产路径回归](../tests/test_native_entry_sampling.py) 直接经过采样、target-window、`execute_update` 和梯度检查，包含 first_legal110/anchor108、恰在108边界、长度111的末尾截断；检查 slot0/1 无 loss 梯度、slot2 对应原始 tick110 并有梯度、完整真实历史被读取。另检查同一步去重、恢复、无关 anchor 不增入口计数，以及旧随机样本与 geometry RNG 保持。其存在证明回归覆盖范围，实际训练证据仍以下面的日志和选中 checkpoint 台账为准。

训练前的 [sampling_preflight.json](evidence/tight_entry_phase_20261121_01/audit/sampling_preflight.json) 重放双方全部 60 个实际 deck，B2 每 epoch240访问/120更新，预算7200更新。两种采样均达到绝对 tick 最低60次；旧采样对150条入口956槽仍有860槽低于50，新采样全部60次。新增入口目标50次在 epoch50/update6000可达，且入口门槛没有改变旧绝对 tick 门槛下的可选 epoch50..60。

## 匹配重训、选中模型与来源

双方使用同一个 train180/validation6/test0，冻结原统计与 family 划分。匹配起点取旧 R2 **实际开始训练时**的各自 R1 选中 EMA DP 与匹配后端，使用 fresh AdamW/EMA；不是从旧 R2 最终权重再多训一轮。DP 父 SHA 为 A `5df541a70b9eab75800fe3ae9cae4700290fbc557605f4f5497523e9d98fa446`、B `624dae38c80c45ab07f808e8c916b8c2683ac598db4892f04366da408e68755a`。后端父 SHA、数据/统计/视觉缓存 SHA 和完整来源保存在配置及复用审计中。

学习率从旧训练实际 epoch 验证日志读取并显式回放：DP epoch1..48 为1.5e-5、49..60为7.5e-6；后端1..48为1e-4、49..60为5e-5。A/B 的初始化 seed 为2026111401、deck seed2026111402、训练扩散噪声 seed2026111403、anchor seed2026111501；checkpoint 的推理 sampler seed 为7091，7093只是历史代码的回退默认值。seed 相同不能证明修改 anchor 数量后的逐visit训练扩散噪声仍相同。

| 组件 | 实际新增预算 | 选中模型 | 选中模型的有效曝光 |
| --- | --- | --- | --- |
| A DP | 60 epoch / 7200更新 | epoch50 / update6000 | 原生150条/956槽最低与最高均50；绝对35863合法tick最低50 |
| B DP | 60 epoch / 7200更新 | epoch50 / update6000 | 原生150条/956槽最低与最高均50；绝对35863合法tick最低50 |
| A Transformer | 22763更新 | epoch53 / update20108 | 180轨迹/35863合法tick最低53次，0低于50 |
| B Transformer | 22763更新 | epoch54 / update20490 | 180轨迹/35863合法tick最低54次，0低于50 |

两份 DP 实际 `learning.jsonl` 各有7200条更新，后端 `loss.jsonl` 各22763条；epoch验证与选中 checkpoint 台账相符。新增50条在 A/B 各自选中 DP 的 [A逐槽表](evidence/tight_entry_phase_20261121_01/audit/A_new50_entry_selected_20261122.csv)、[B逐槽表](evidence/tight_entry_phase_20261121_01/audit/B_new50_entry_selected_20261122.csv) 中各334行、50个identity，所有 `optimizer_steps=50`，低于50为0/334；slot均在0..11，且 `native_anchor+slot=absolute_tick`。这是选中best本身的50次，不使用末轮last的60次替代。

| 本轮选中组件 | SHA256 |
| --- | --- |
| A DP | `a3a35e465723318dbb352e96b696af91c58440d99923205590888e85f97e3351` |
| A Transformer | `a332df55c9f9f7ea623da02512946944e3a103aa9e0989cd24ca12ad25940585` |
| B DP | `91a649ebb652635b212be11eadd96a18c38a0a716acaae2ba8cc7276edbbc42d` |
| B Transformer | `21ef1a01a0734ca6890430d1a182d595aff4d47adb884d503946737fcb24380f` |

每个后端都在本组选中 DP 之后重新生成186集 nominal 缓存，缓存索引和每个载荷验证绑定本组 DP SHA；A 索引 SHA 为 `82a37b6cbbbd44427c76cd67d0121d277685b138f1995d9980a19197281ee6af`，B 为 `98b350d9a7fd1773091882c5612bf386038753581d10548239732067568c0baf`。没有用旧 DP 的 nominal 来训练新后端。

原 attempt 在 A epoch1/update120之后发生资源中断。恢复从 durable120完整还原 model、Adam、EMA、deck、CPU/CUDA和噪声RNG，121..129共9条非durable更新单独归档、计数为0；重执行后只计一次。[恢复核验](evidence/tight_entry_phase_20261121_01/audit/recovery_nine_update_check.json) 与 `recovery/` 保留原失败、日志和原截止，不把一次恢复说成额外9次新曝光。

## 原 R2 对照复用与视觉合同

[r2_control_equivalence_20261122.json](evidence/tight_entry_phase_20261121_01/audit/r2_control_equivalence_20261122.json) 对两组逐项确认：train180/validation6、真实视觉载荷引用、父DP/后端、60实际deck、7200/22763更新、实际epoch LR、可选epoch50..60、验证指标、后端数学及各自DP缓存绑定匹配。因此复用冻结原 R2 完整pair作为旧采样器组，未另造一个新训练对照。旧 R2 选中DP两组均epoch53/update6360，后端A epoch50/update18960、B epoch54/update20490；各组按相同选模规则选择，选中更新数无需恰好相同。

历史采集真实 DINO microbatch=1，缓存和原终测为2。本轮采集、缓存、评估合同及缓存标识显式记录 `native_dino_arithmetic_v1`、microbatch2、尾lane复制、相机顺序、RGB/深度尺度、精度与kernel设置。现有186集缓存逐文件引用、时间戳/帧数和声明SHA在两训练组间一致；另从186集抽取672个实际帧，按microbatch2重新编码验证通过。验证范围是抽帧，不是全缓存逐帧重编码。

四条既有轨迹按各自历史采集路径、完整真实历史、实际规划相位与执行反馈完成同模型重放：R1 A283帧、B266帧，R2 A292帧、B280帧；最终命令最大误差分别为7.15e-7、7.15e-7、6.37e-7、7.15e-7rad，均小于1e-6；机器人历史误差为0。见 [four_replay_receipt.json](evidence/tight_entry_phase_20261121_01/audit/four_replay_receipt.json)。这四条旧轨迹的重放不等于本轮新补录短诊断已经完成。

复用审计的限制必须保留：历史视觉合同缺少显式算术身份，训练组间相等依靠相同缓存载荷及抽帧核验闭合；anchor数量变化导致扩散noise消费发生变化，不能声称逐visit训练噪声相同。此审计只建立对照可比性，不独自证明入口误差或自主恢复改善。

## 96次完整自主开发及旧成功保留

所有主结果均来自完整 DP + Transformer，自主执行时没有专家接管。预登记8个独立开发初态28850000..007与8个历史开发回归初态28826000..007，各初态同波运行原130-50、原R2、入口修复A/B共6套完整策略。实际16波/96次全部合法且软件完成，16/16波可观测真实初态配对检查通过；不主张隐藏PhysX状态及所有随机流精确相同。DINO统一microbatch2，主评估源码为90b0a7c。

| 开发组 | 原130-50严格成功 | 原R2严格成功 | 入口修复严格成功 | 入口修复至少1子目标 |
| --- | --- | --- | --- | --- |
| 独立开发 A | 1/8 | 0/8 | 2/8 | 6/8 |
| 独立开发 B | 0/8 | 3/8 | 4/8 | 5/8 |
| 历史开发回归 A | 1/8 | 2/8 | 2/8 | 3/8 |
| 历史开发回归 B | 1/8 | 1/8 | 3/8 | 4/8 |

首子目标、持物和近孔只是诊断，不替代原生严格成功；成功判据、60Hz控制及首子目标后的原生时钟规则未修改。不同cohort不作为同一个新终测池解释。

| 对原R2的配对比较 | 旧成功保留 | 旧成功丢失 | 在原失败中新增成功 |
| --- | --- | --- | --- |
| 独立开发 A | 0/0（无旧成功） | 0/0 | 2/8 |
| 独立开发 B | 1/3 | 2/3 | 3/5 |
| 独立开发合计 | 1/3 | 2/3 | 5/13 |
| 历史开发回归 A | 0/2 | 2/2 | 2/6 |
| 历史开发回归 B | 0/1 | 1/1 | 3/7 |
| 历史开发回归合计 | 0/3 | 3/3 | 5/13 |

独立开发B仅保留28850002，丢失28850000/003，新增28850005/006/007；A新增28850005/006。历史开发A丢失28826003/004、新增002/007；B丢失007、新增000/002/004。相对原130-50，历史开发两个旧成功同样均未保留（0/2），独立开发唯一旧成功也未保留（0/1）。

逐例分母、成功seed、原生子目标、步数与本地result路径见 [development_episodes_20261122.csv](evidence/tight_entry_phase_20261121_01/audit/development_episodes_20261122.csv)，两种参考策略的保留/丢失/新增见 [development_summary_20261122.json](evidence/tight_entry_phase_20261121_01/audit/development_summary_20261122.json)。全部96条原始结果携带真实完整模型路径和SHA，与所在wave的plan逐项核对。

这些小样本支持“当前开发的成功总数有所增加，同时存在明显旧成功丢失”。它们不能证明统计显著性，不能把新增自主成功都叫作从相同真实偏离状态救回，也不能用接触重新出现作为因果恢复能力的替代指标。

## 命令、复核和历史身份保护

以下是实际执行命令的登记形式；完整实际子命令数组保存在训练/开发 `project_job_started.json` 中。它们用于追溯，**不可在已尝试的原root上重启**。统一入口拒绝复用 `project_job_started.json`、`STOP_MONITOR` 或过期session；复跑必须登记新的独立run、有限session和显式job，核对父模型/数据/视觉SHA后再启动，不能删除旧回执绕过守卫。

```bash
cd /home/CNS2025331827/docs/bc+rl_RSS

# 实际原训练作业；资源中断后以独立owner恢复同一训练状态
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python -m tools.tight_project train --job configs/experiments/entry_phase/train.json
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python -m tools.tight_project train --job configs/experiments/entry_phase/train_recovery01.json

# 实际完整开发作业
/home/CNS2025331827/docs/bc+rl_RSS/.venv_isaacsim/bin/python -m tools.tight_project evaluate --job configs/experiments/entry_phase/development.json

# 只检查登记的子命令，不启动训练或仿真
python -m tools.tight_project train --job configs/experiments/entry_phase/train.json --dry-run
python -m tools.tight_project evaluate --job configs/experiments/entry_phase/development.json --dry-run

# 最小生产路径回归（CPU）
python -m pytest -q tests/test_native_entry_sampling.py tests/test_recovery_onset_windows.py tests/test_visual_arithmetic_contract.py

# 无本机数据/GPU依赖：复核公开文件SHA、实际step入口曝光和96例配对结果
python research_docs/evidence/tight_entry_phase_20261121_01/verify_published.py
```

历史独立复核命令为 `python outputs/diagnostics/tight_entryfix_20261121_01/independent_phase.py`，训练预算预检为 `python -m tools.tight_entry_phase preflight --config configs/experiments/entry_phase/experiment.json`，开发汇总为 `python outputs/diagnostics/tight_entryfix_20261121_01/summarize_development.py`。这些脚本会写回各自固定输出，因此复核时先复制到新的诊断root并调整输出路径，保留现有原始证据；公开副本是原样导出，目录层级变化后不能假设硬编码路径已自动迁移。

## 尚缺的验收及对三个问题的直接回答

1. **入口是否真的进入训练：是。** 新增50条在A/B选中DP各334个合法入口提交槽全部为50个不同成功optimizer step；全池150条/956槽全部50次。生产更新、epoch验证、选中checkpoint台账和逐槽表互相对应，未计祖先或9条弃用更新。
2. **入口误差是否下降：尚无可下结论的本轮结果。** 同一真实历史、同一采样噪声、同一执行反馈下，首合法槽及后续提交槽的DP nominal、Transformer residual、最终投影命令误差比较尚未完成。通用离线验证分数和四条旧模型自重放不能回答这个问题。
3. **闭环恢复和旧成功怎样变化：完整自主开发成功增加，旧成功保持不足。** 对原R2，独立开发由3/16到6/16、保留1/3旧成功；历史开发由3/16到5/16、保留0/3。新增分别为5/13、5/13个原失败。真实偏离状态的因果恢复能力尚未单独验证，不能从成功总数代替推断。

仍须完成入口同历史分层误差、本轮少量短诊断的精确视觉/反馈/相位/噪声记录与同模型自重放、首次差异定位，以及开发选模与冻结后的独立终测。预留终测28851000..007尚未打开，原已打开终测只作历史审计与回归，不参与反复调参。当前证据不足以宣布完成原任务、晋升候选或清除必要的原始数据、模型及失败记录。
