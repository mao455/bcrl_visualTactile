# 视觉与触觉模仿学习研究工程

> **最新实测状态（2026-10-29 00:05）**：v4两条验收通过；随后后台提交2条新成功后在seed930017启动崩溃而停止（2/400，缺398，当前无采集进程）。一次受限strace确认底层inotify_add_watch ENOSPC；未改系统/其他进程。详见SESSION_HANDOFF最后一节。

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
