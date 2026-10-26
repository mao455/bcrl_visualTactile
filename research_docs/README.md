# 视觉与触觉模仿学习研究工程

**Objective:** a tactile-free nominal Diffusion Policy for geometry and an independently learned tactile correction controller for contact execution, integrated with the original Play2Perfect assets/control/tasks.

## Reading order
1. `CONTRACTS.md`: audited control, observation provenance, timing and reset contracts.
2. `ARCHITECTURE.md`: proposed components, information boundaries, phases and acceptance gates.
3. `RESEARCH.md`: primary literature/code, licensing and choices (written as reviewed).
4. `CHANGELOG.md`: append-only implementation/test evidence.
5. `RUNBOOK.md`: 可复制的低资源命令与实际结果。
6. `AUDIT.md`: 原任务调用链、物理参数、本机 CPU/渲染限制和图像审计。

## 当前状态 — 2026-10-27（Asia/Shanghai）

- 已审计并复用原资产、任务、控制、随机化和 reset 路径；原 task/RL 源文件未改动。`AUDIT.md` 补充源码调用链、物理参数和本机运行限制。
- 已实际采集 RGB-D、58 维编码器状态、五个末节刚体的法向接触合力、原生动作和执行器目标，按独立 episode 保存。原默认相机画面为空；新增可选斜视安装位置可看到场景，但遮挡和插入细节仍未解决。
- 已用真实专家轨迹训练并恢复到 1,000 步 nominal diffusion，完成保存、精确重载和独立仿真 rollout；验证 loss≈.736。最新 2 env × 660 ticks 中专家 5/6 成功，nominal 0/2、未产生接触，**尚未得到有任务能力的 BC 模型**。普通 BC 仍可能学习专家的预加载意图。
- Correction 已有独立视觉/运动/动作/触觉时序输入、预测 nominal chunk 编码和候选命令输出。旁路运行仅记录未训练候选，绝不执行。**尚无训练过的 correction，也未证明触觉带来改进。**
- 单元测试已扩至 8 项通过；真实轨迹对齐检查通过。后续 rollout/评估的最新数据见 `CHANGELOG.md` 与 `RUNBOOK.md`。

### 本机兼容性的重要限制

CPU 无渲染原路径在第一步后把运动学桌子、孔座、目标移回了初始原点。这解释了之前回归的巨大坐标差异；不能归因于普通随机噪声。补上与相机路径一致的渲染调用后，原环境与新接口在 32 步内所有被测字段逐位相同，包含物体/孔座/目标、奖励、观测和 RNG。完整证据见 `AUDIT.md`。此前无渲染 CPU 的 0/3 评估不能再用于判断专家质量。

当前采用 CPU physics + GPU 小分辨率渲染，保持 120 Hz 物理和 60 Hz 原生动作处理；nominal 以每 6 步重规划实现约 10 Hz 的模拟更新节奏。真实时间截止期未保证，后台 GPU 任务始终受保护。

Work proceeds through audit → sensor/recording vertical slice → leakage-safe dataset/history → nominal diffusion training/reload/rollout → correction interface and justified learning experiments. A completed software smoke test is not evidence of assembly success or contact improvement.
