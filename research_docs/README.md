# 视觉与触觉模仿学习研究工程

## 最新系统与实验审计 — 2026-10-31

[完整系统说明与控制链审计](SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md)、
[逐文件地图](PROJECT_FILE_MAP.md)、[实验有效性矩阵](EXPERIMENT_VALIDITY_MATRIX.md)
是本轮审阅入口。400条/81,894步数值链重读、147条既有评估队列核验、T0源RNG重建均已完成。
原生控制仅有只读hook；近似T0不能否定原示范成功。文档首版`6022330`已推送，随后Loss B默认与诊断record2修复完成，
26项定向CPU测试通过。未启动新GPU实验；记录器尚无新Kit验收，完整物理精确重放仍未解决。

## 最新完成 — 2026-10-31网络核验与近似训练初态诊断

[中文核心结论与完整逐层说明](NETWORK_ARCHITECTURE_EXACT.md)包含真实checkpoint的594行模块审计、
743行参数/optimizer、5张网络结构图、Force同权重干预、400条分布和12条T0/T1诊断。
[所有关键训练曲线直接阅读](evidence/dp_loss_trt_20261031/README.md)。无新参数训练；自有GPU进程已退出。
T0记录命令未复现成功，T1近似初态G00-B/G10-B各0/4；不将其称为精确训练环境失败或泛化差。
Force确实响应连续力，但未证明收益；五遍未证明收敛。下方“当前”段落为历史阶段记录。

## 当前新候选 — 2026-10-30

[G00-new/G10-new长预测训练记录](LONG_HORIZON_20261030.md)：新授权实现、训练、有限闭环与产物清理。
5Hz/H64、冻结DINOv2-S/14、显式两秒历史、60Hz持续记忆、绝对参考转原生命令。
下方已完成阶段的“不再训练”等是历史范围；原B0/B1/A/B/C实验身份和结果不改写。

## 实验组录像与精确参数统计 — 2026-10-30

[完整对照报告](EXPERIMENT_GROUPS_AND_PARAMETERS_20261030.md)汇总B0/B1/A/B/C的数据、best更新数、
各类离线/闭环分母、逐模块参数和官方生效配置；[可核查小统计](evidence/parameter_census_20261030.json)
包含57个实际导入上游文件的固定revision SHA核验。原始视频保留本机ignored demos，未上传公开仓库。
本轮仅观看重跑与CPU核算，不重新训练；原400数据、正式0/50和旧12次诊断保持不变。
6条新视频已完成并与旧4条全部解码核验，均601步timeout，无子目标/抬升；全部自有进程退出。
观看索引在本机`outputs/visuotactile/demos/residual_20261030_01/README.md`。
整卡均值27.9960%，完整60秒最大60.3794%，四段超限已记录，不能称本次全程满足50%资源限制。

## 当前独立残差Correction阶段 — 2026-10-30

本轮已授权真实Correction训练、固定最多12条诊断和feature发布，覆盖下方历史限制。
[Correction训练与执行报告](CORRECTION_TRAINING.md)记录实际版本、输入权限、冻结、pilot学习及运行预算。
[官方方法独立比较](POLICY_COMPARISON_20261030.md)区分固定源码配置、部署override、论文描述与未知实现。
正式B0保持冻结、B1不充当nominal；原400条、best与各0/50保持不变。
本轮已完成并退出：5完整epoch、一次test40，B/C各7,875更新；12个诊断分配中11条自然timeout、
1条交接校验中止。自主各0/2；唯一完整匹配的局部三联也失败，无首子目标/抬升。
离线误差下降不等于触觉收益或RL。详见[去敏统计](evidence/residual_20261030.json)，不自动扩量或重启。

## 最新独立审查 — 2026-10-30

本轮用户授权审查、真实行动历史实现、小拟合/诊断闭环与功能分支推送，覆盖下面观看阶段限制。
详见[独立审查与A–F核查](INDEPENDENT_AUDIT_20261030.md)、[输入契约](CONTRACTS.md)、
[复现命令](RUNBOOK.md)。旧best和正式B0/B1各0/50不变；没有完整重训、重采或再做100条评估。
新v3 `action_history_mode=issued`训练取过去专家已执行命令teacher forcing，部署取学生真实执行反馈；
旧v2默认none。历史接口已实现，收益仍需实证，不能把小前缀loss下降称任务改善。
GPU整卡时间加权与完整60秒均值<=50%、内存/显存余量和原控制器只读继续有效。
以下“仅视频”“不新增模型”是先前阶段快照，不覆盖本轮授权。

## 最新结果 — 2026-10-30

B0/B1各50epoch已完成；验证选epoch49。50组配对各0/50，全部601步原生超时、未完成第一子目标。
服务已退出，不再按旧ETA视为仍在训练。详见[训练与正式评估结果](BASELINE_TRAINING.md)和
[最终小统计](evidence/baseline_final_20261030.json)。用户本次要求各2条真实视频，并恢复整卡
GPU时间加权批次与完整60秒均值<=50%；保留主存/显存余量，不启动追加训练。
4条视频已完成，目录为 `outputs/visuotactile/demos/baselines_20261030/`（目录内README为观看入口）。
每条601帧/60fps约10秒；录制GPU均值29.50%、完整60秒最大38.15%，service已退出。

下文均为按日期保留的先前阶段记录；“正在训练”“不限GPU”及旧ETA不覆盖上述最新状态。


> **当前工作（2026-10-29下午）：** 400条生产数据保持不变，用户已授权B0/B1普通命令DP各50epoch训练和50组配对评估，全部GPU算力可用，仍保护RAM/VRAM。两模型在后台独立并行训练，尚未完成正式评估。入口见 [BASELINE_TRAINING.md](BASELINE_TRAINING.md)、[RUNBOOK.md](RUNBOOK.md) 和交接顶部；下方“不训练”属于采集/感知阶段历史。

> **当前交付（2026-10-29）：** 四环境、8m、v4三相机生产批次已完成 **400/400 条审计通过的自然成功轨迹**，采集器已退出。最终统计见 [`NIGHT_COLLECTION.md`](NIGHT_COLLECTION.md) 和 [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md)；读取正式数据时使用 `outputs/visuotactile/data/close_v4/manifest.json` 并显式指定 `stage="production"`。本轮未训练。

> **历史快照（已被上方最终交付覆盖）：** [NIGHT_COLLECTION.md](NIGHT_COLLECTION.md)记录了四环境＋8m生产过程、原1.2m邻居入镜排除和运行证据；下面旧快照不可当作当前状态。

> **历史状态（已完成）：** 2026-10-29 00:05时曾是2/400并遇到启动错误；该快照后来恢复生产并已被400条最终交付覆盖。一次受限strace确认底层inotify_add_watch ENOSPC；未改系统/其他进程。

> **2026-10-28 最新纠正：** v4近高位三视角已完成seed17/920101两条真实验收（220/214步），与各自v3参考128个非图像字段逐位相同；143字段无损转换和三路模型接口检查通过，零训练。此前“零v4成功/资源保护已完整”的文字为当时快照：上一提交的容量统计仅单root、v4仍有两处版本漏判，本轮已修正。实际背景批次看最新SESSION_HANDOFF/status；不要把下面历史状态当实时。源码通俗说明见[PROJECT_GUIDE_ZH.md](PROJECT_GUIDE_ZH.md)。


**Objective:** a tactile-free nominal Diffusion Policy for geometry and an independently learned tactile correction controller for contact execution, integrated with the original Play2Perfect assets/control/tasks.

## Reading order
0. `USER_RESEARCH_DIRECTION.md` 与 `HISTORY_AND_TRACE.md`: 用户方向与当前历史/数据合约。
1. `CONTRACTS.md`: audited control, observation provenance, timing and reset contracts.
2. `ARCHITECTURE.md`: proposed components, information boundaries, phases and acceptance gates.
3. `RESEARCH.md`: primary literature/code, licensing and choices (written as reviewed).
4. `CHANGELOG.md`: append-only implementation/test evidence.
5. `RUNBOOK.md`: 可复制的低资源命令与实际结果。
6. `PERCEPTION.md`: 双视角/逐指编码、来源实现与当前验证。
7. `AUDIT.md`: 原任务调用链、物理参数、本机 CPU/渲染限制和图像审计。

**最新相机阶段（2026-10-28晚）：** [三视角验收与400条协议](CAMERA_VALIDATION.md)。已验证两个固定＋腕部相机、同步动态标定、原专家配对、三路存储/模型接口；400批次真实状态见 [SESSION_HANDOFF](SESSION_HANDOFF.md)。以下有限实验结果为历史，不能当作新批次数量。

## 当前状态 — 2026-10-27（持续历史 / trace2）

**2026-10-28 最新入口：[有限插入数据集报告](INSERTION_DATASET.md)、[会话交接](SESSION_HANDOFF.md)。** 本轮20次尝试，17条有效成功记录（验收4＋扫描13），两次动作前非法初始化、一次Kit启动SIGSEGV后停止。无参数训练。新storage1三文件、manifest顺序读取、逐字段无损转换及真实411步零更新检查通过；35项测试通过。扫描只有一个有效起点且零外力曝光，不能声称方向鲁棒性。旧pilot20/40有初始box重叠，不能直接外推旧75%。报告区分现有读取链与尚待接通的正式训练。

2026-10-28补充：[原专家配对扰动实验](RL_EXPERT_TRIALS.md)记录本轮无训练采集、验收/pilot、实际外力提交与本机composer限制。新增`run_rl_trials`/`analyze_rl_trials`，保持原感知/模型/动作接口；精确结果和错误数量以该实验报告为准，不能用下面的历史检查替代本轮验收。

- 先读 [用户研究方向](USER_RESEARCH_DIRECTION.md)：用户明确要求、开放问题、Codex暂定决策和已测事实分别记录，方向变更追加历史。
- [HISTORY_AND_TRACE.md](HISTORY_AND_TRACE.md) 是当前记忆/动作/频率合约。Nominal每60Hz观察更新持续128D状态，每6步才重规划；Correction保留视觉、运动、计划和五指状态，触觉每120Hz样本更新。只在真实episode reset清空对应环境。
- 双固定RGB-D、空间编码、五指聚合法向力/FK仍使用perception2。新增memory2与trace2独立版本；旧checkpoint缺memory版本按1解释。过去专家命令可独立作为teacher-forcing历史，绝不冒充Nominal预测；默认Nominal仍只读自己的预测上下文。
- HDF5逐控制步追加/flush，明确submitted/clipped/delayed command、实际delay、target、post-state、计划来源与缺失终止图。提供逐步reader与单episode PKL导出；不公开轨迹或权重。
- **23项单元测试通过**；真实480步记录审计、225步完整/分块/逐步回放通过（Nominal最大差4.66e−7，Correction分块/逐步一致）；真实接触前缀双相机RGB/Depth、触觉、motion与intent梯度及权重/状态恢复检查通过。原渲染环境32步15字段逐位一致。证据见 [causal_history.json](evidence/causal_history.json)。
- **本轮0次参数更新，Correction始终Shadow**。保持原物理/控制数学；原action helper仅增加可选只读trace hook。近期双相机真实图与覆盖限制仍见 [PERCEPTION.md](PERCEPTION.md)。固定维度记忆不是无损轨迹，也未验证长期任务效用。
- 未来训练入口支持整episode顺序TBPTT，当前没有执行；memory2验证loss尚未接入，恢复从episode边界开始。去预加载、Correction目标、目标估计、异步传感器和实时控制仍未解决。当前双核2env完整旁路中位约99.5ms/步，不能满足60Hz墙钟期限。

### 本机兼容性的重要限制

CPU 无渲染原路径在第一步后把运动学桌子、孔座、目标移回了初始原点。这解释了之前回归的巨大坐标差异；不能归因于普通随机噪声。补上与相机路径一致的渲染调用后，原环境与新接口在 32 步内所有被测字段逐位相同，包含物体/孔座/目标、奖励、观测和 RNG。完整证据见 `AUDIT.md`。此前无渲染 CPU 的 0/3 评估不能再用于判断专家质量。

当前采用 CPU physics + GPU 小分辨率渲染，保持 120 Hz 物理和 60 Hz 原生动作处理；nominal 以每 6 步重规划实现约 10 Hz 的模拟更新节奏。真实时间截止期未保证，后台 GPU 任务始终受保护。

Work proceeds through audit → sensor/recording vertical slice → leakage-safe dataset/history → nominal diffusion training/reload/rollout → correction interface and justified learning experiments. A completed software smoke test is not evidence of assembly success or contact improvement.
