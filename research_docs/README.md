# 视觉与触觉模仿学习研究工程

**Objective:** a tactile-free nominal Diffusion Policy for geometry and an independently learned tactile correction controller for contact execution, integrated with the original Play2Perfect assets/control/tasks.

## Reading order
1. `CONTRACTS.md`: audited control, observation provenance, timing and reset contracts.
2. `ARCHITECTURE.md`: proposed components, information boundaries, phases and acceptance gates.
3. `RESEARCH.md`: primary literature/code, licensing and choices (written as reviewed).
4. `CHANGELOG.md`: append-only implementation/test evidence.
5. `RUNBOOK.md`: 可复制的低资源命令与实际结果。
6. `PERCEPTION.md`: 双视角/逐指编码、来源实现与当前验证。
7. `AUDIT.md`: 原任务调用链、物理参数、本机 CPU/渲染限制和图像审计。

## 当前状态 — 2026-10-27（Perception v2）

- 本轮已接通双固定 RGB-D 相机（224×126）、深度有效掩码和标定/时间记录。经过三次安装候选的真实渲染检查，保留原全局视角并增加左前上方互补视角。详见 [PERCEPTION.md](PERCEPTION.md) 和图示。
- 视觉升级为共享 ResNet18 前半段 RGB + 独立 Depth 分支、空间 keypoint/grid + masked GRU。Nominal 接收合法 task/可选 operator goal 与自身因果旧计划；不接收触觉、专家动作或 correction 命令。
- 五指逐指编码合力、身份、编码器 FK 位姿和时序变化；Correction 独立融合视觉/运动/执行目标/触觉及 nominal 未来预测。**始终 Shadow，未训练候选从不执行。**
- 真实 2 env × 240 ticks 专家旁路：记录/读取、前向、有限梯度、世界/局部力转换、离线/在线历史一致和模型精确重载通过。15项CPU单元测试通过。原渲染环境32步逐位回归通过；有无新模型旁路的480个专家transition动作/目标/物体/奖励等字段逐位相等。
- **本轮0次参数更新，无正式训练、RL或成功率优化。** 历史v1的1000步nominal模型仍保留，尚无任务能力证据；v2未训练，不能根据接口测试推断精细操作能力或触觉收益。
- schema/checkpoint v1/v2 明确隔离。旧单相机命令需 `--perception-version 1`；新默认采集v2。当前task ID可用，但目标pose默认缺失；视觉目标估计和自动阶段命令仍待研究。当前CPU完整旁路约200ms/步，未满足实时要求。

### 本机兼容性的重要限制

CPU 无渲染原路径在第一步后把运动学桌子、孔座、目标移回了初始原点。这解释了之前回归的巨大坐标差异；不能归因于普通随机噪声。补上与相机路径一致的渲染调用后，原环境与新接口在 32 步内所有被测字段逐位相同，包含物体/孔座/目标、奖励、观测和 RNG。完整证据见 `AUDIT.md`。此前无渲染 CPU 的 0/3 评估不能再用于判断专家质量。

当前采用 CPU physics + GPU 小分辨率渲染，保持 120 Hz 物理和 60 Hz 原生动作处理；nominal 以每 6 步重规划实现约 10 Hz 的模拟更新节奏。真实时间截止期未保证，后台 GPU 任务始终受保护。

Work proceeds through audit → sensor/recording vertical slice → leakage-safe dataset/history → nominal diffusion training/reload/rollout → correction interface and justified learning experiments. A completed software smoke test is not evidence of assembly success or contact improvement.
