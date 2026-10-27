# 可复现命令与验证结果

从仓库根目录运行。所有命令使用项目 .venv_isaacsim。run_cpu.sh 只限制它新启动的进程：nice 19、两线程、CPU 22/23（可用 RESEARCH_CPUS 指定），不操作其他任务或 GPU 配置。CPU physics 仍需 GPU 渲染相机，不能承诺零资源竞争。这些是短实验命令，不是论文规模训练。

输出目录必须不存在；使用新的 RUN 前缀保留历次证据。退出码不是充分证据：Kit 关闭曾掩盖异常，必须确认 result.json 存在且没有 FAILED.txt，训练确认 report.json，并运行数据审计。

## 当前 memory2 / trace2：只验证，不训练

```bash
RUN=outputs/visuotactile/history_$(date +%Y%m%d_%H%M%S)

# 专家执行；双相机、持续记忆Nominal/Correction旁路。未训练候选不执行。
bash visuotactile/run_cpu.sh -m visuotactile.collect --perception-version 2 \
  --steps 240 --num-envs 2 --perception-shadow --output "$RUN/sensors"

# 逐步来源/时间/延迟/FK/terminal、完整/分块/在线记忆、PKL往返。
# check-gradients在真实接触前缀上只做Jacobian与权重/隐藏状态重载，无optimizer。
bash visuotactile/run_cpu.sh -m visuotactile.probe_history \
  --data "$RUN/sensors/trajectories.h5" --output "$RUN/checks" --check-gradients

bash visuotactile/run_cpu.sh -m pytest -q \
  tests/test_visuotactile.py tests/test_perception_v2.py tests/test_causal_memory.py
```

`probe_history`完整记忆复现只接受有种子/版本的未训练Shadow记录；它验证固定参数，不宣称任意checkpoint可重建。`--trace-only`跳过225步式记忆回放；`--reference`可指定同种子、配置和env数的先前专家capture，精确比较15字段。输入不满足条件即失败，不用无渲染数据补齐。检查`sensors/result.json`、`checks/report.json`且没有`FAILED.txt`。`--check-gradients`要求真实非零接触；极短轨迹可能没有，不能伪造接触。

### 逐步读取与单episode PKL导出

```bash
bash visuotactile/run_cpu.sh -m visuotactile.recording \
  --data "$RUN/sensors/trajectories.h5" --episode env000_ep000000 \
  --pkl "$RUN/episode.pkl"
```

Python中`EpisodeReader(path).episodes`列出实际ID，`.read(id,t)`读一个控制步，`.steps(id)`迭代原始嵌套字段。导出不覆盖已有文件；PKL只用于可信本地检查，HDF5保持主格式。完整/尾段标志与缺失标记保留。不要把包含privileged诊断的整个step字典输入模型。

### 当前实际验证

- `causal_trace02`：2env×240tick，480transition，3完成+2尾段；`causal_checks01`检查每步delay/source/time、FK/局部力、post/pre图像与terminal缺失。15专家字段与旧`perception_final01`完全一致。
- `causal_checks01`：225步全前缀、chunk7、step1固定参数结果一致；Nominal最大差4.66e−7（图像batch浮点差），Correction分块/逐步精确一致、与原2env在线候选最大差4.47e−8；113字段PKL无损往返。
- `causal_trace03/causal_checks02`：64transition补充实际限位/目标公式、专家actor输入与checkpoint hash、float64延迟时间；FK/force重算误差0，时间差5.55e−17s，114字段PKL精确。
- `causal_trace04/causal_checks04`：最后2env×8tick核验Correction hash/config和冻结专家输入快照；无接触/完成episode，不作为触觉或成功证据。
- `causal_gradients01`：真实接触t37前顺序回放形成记忆；双视图RGB/Depth、触觉、motion、intent及Nominal时序分支梯度有限非零，参数hash未变；权重+隐藏状态重载输出精确一致。
- `causal_regression01`：原渲染环境32步对照，initial13项/trace15项逐位相等，包括RNG。日志有与原路径相同的kinematic setVelocity告警；未把退出码或无告警当成有效场景依据。
- 最终23项CPU测试：远于旧窗口的输入有影响/梯度、跨块与在线一致、reset分env、视觉mask与非零梯度、真实控制helper数值/RNG、数据往返、旧版本/重载。梯度检查不更新参数。
- 当前2env双核旁路中位99.49ms、memory44.11ms、Correction48.75ms。120Hz物理/60Hz命令/10Hz重规划为**模拟时间**安排，非实时能力证明。没有操作后台任务或共享GPU设置。

完整数值证据在 [evidence/causal_history.json](evidence/causal_history.json)。真实数据/日志/checkpoint仅在忽略的outputs中，不推送。

`--perception-shadow` 数据标为`untrained_perception_probe`，普通正式训练入口拒绝；单纯专家采集去掉此参数，自己计划mask=false，专家过去命令仍保存。当前只有categorical tight_insertion task ID；可选`--goal-command command.json`格式仍为：

```json
{"source":"operator_command","frame":"robot_base","position_m":[0.0,0.0,0.7],"rotation_columns":[1,0,0,0,1,0]}
```

该值只能是外部命令，不是仿真目标估计，也不自动重写专家原任务。目标pose缺失保持mask=false。

### 未来训练接口（本轮没有执行）

默认`memory_version=2`，`batch_size=1`，`chunk_size=8`。`EpisodeSequenceDataset`从episode0步逐块读取，`encode_episode_chunks`传递状态；TBPTT detach保留记忆值。旧window dataset不是新memory训练入口。未来`train`恢复权重/optimizer/RNG后从episode起点重放，不是精确中断点恢复；memory2验证loss目前为null。Correction只有顺序输入/标签接口，没有训练目标和优化器。专家teacher-forcing历史单独暴露，默认Nominal不读它。

`probe_perception`明确使用memory1作历史窗口/相机构图诊断；旧结果如下。当前持续历史检查使用`probe_history`，两者不能混称。

### 先前 perception2 / memory1 的真实结果（保留）

- `perception_camera01/02/03`：三个安装候选，分别1×300、2×260、2×240 native ticks；每次都开启渲染/场景健康断言。
- `perception_final01`：最终224×126，2env×240ticks，73.10s循环；480条原专家transition，3完整+2尾段。成功标记只用于选择检查画面，未用于策略优化。
- `perception_checks02/report.json`：全轨迹传感器有限性、坐标转换、FK与truth只读对照、时间和历史对齐；前向/有限非零梯度、精确保存恢复全部通过。0次参数更新。
- 五指FK最大位置误差6.60e−7m、旋转矩阵元素误差9.82e−7。最终两段长episode全局有效深度约68.6–69.3%、局部约93.7–94.1%。ring在最终短样本里没有接触；camera01里它有13.73N非零信号，不能把零力说成传感器损坏。
- `perception_regression01/comparison.json`：原渲染环境32步15字段逐位一致；camera03与final01有无旁路的480条动作/执行目标/奖励/关节/物体/孔座/成功/终止字段逐位一致。
- 15项单元测试通过；旧v1的真实1000step checkpoint和schema1 Dataset仍可加载/前向，v1/v2混用明确拒绝。
- 2env完整旁路median206.94ms，Correction部分median199.16ms；单样本冷启动nominal333.17ms（当时并行有本轮回归启动）。**当前限制CPU配置不满足实时10/60Hz**；同步仿真能运行不代表实时可部署。未占用GPU训练，也没有更改后台任务。

### 旧 v1 命令与证据（保留，非本轮训练安排）

以下历史流程继续使用schema1，所有仿真采集/旧checkpoint rollout显式加`--perception-version 1`。新schema2不沿用旧模型权重。

## 完整的小规模实验链

    RUN=outputs/visuotactile/run_$(date +%Y%m%d_%H%M%S)

    # 原专家 + 真实相机/接触数据，保留原 DR 和执行器路径
    bash visuotactile/run_cpu.sh -m visuotactile.collect --perception-version 1 --steps 660 --num-envs 2 --teacher-group 0 --output "$RUN/teacher"

    # 默认只使用完成所有目标和 retract 的专家 episode
    bash visuotactile/run_cpu.sh -m visuotactile.train --memory-version 1 --data "$RUN/teacher/trajectories.h5" --output "$RUN/train" --steps 100 --batch-size 4

    # 模型独立执行；shadow 只记录未训练候选，绝不执行候选
    bash visuotactile/run_cpu.sh -m visuotactile.collect --perception-version 1 --steps 660 --num-envs 2 --policy-checkpoint "$RUN/train/checkpoint.pt" --correction-shadow --output "$RUN/nominal"

    # 对齐/有限数/reset/输入及配置一致性审计和成功统计
    bash visuotactile/run_cpu.sh -m visuotactile.evaluate --data "$RUN/teacher/trajectories.h5" "$RUN/nominal/trajectories.h5" --output "$RUN/evaluation.json"

若本次没有成功 episode，普通训练会拒绝；应检查专家/场景或多采集。--smoke-only 仅显式允许真实失败/不完整轨迹用于软件测试，不生成标签。该标志为 false **不表示模型已有任务能力**。

## 用已保存的真实数据继续训练

    RUN=outputs/visuotactile/resume_$(date +%Y%m%d_%H%M%S)
    bash visuotactile/run_cpu.sh -m visuotactile.train --memory-version 1 --data outputs/visuotactile/episodes02/trajectories.h5 --resume outputs/visuotactile/train02/checkpoint.pt --output "$RUN/train" --steps 100 --batch-size 8
    bash visuotactile/run_cpu.sh -m visuotactile.collect --perception-version 1 --steps 660 --num-envs 2 --policy-checkpoint "$RUN/train/checkpoint.pt" --output "$RUN/rollout"

恢复要求相同数据哈希、模型配置和 smoke 筛选模式，恢复优化器与 torch RNG。PolicyConfig 保存历史长度/stride、预测 horizon、重规划周期和 diffusion 步数。配置实验不要通过改变环境 decimation 模拟低频网络。历史memory1的默认history=4、stride=3；当前memory2无此记忆上限，horizon=12、replan=6保持，即 120 Hz 物理、60 Hz native control、10 Hz nominal replan。同步仿真等待推理，尚不是带真实墙钟延迟/超时回退的实时系统。

## 在真实接触轨迹上检查 correction

    RUN=outputs/visuotactile/contact_probe_$(date +%Y%m%d_%H%M%S)
    bash visuotactile/run_cpu.sh -m visuotactile.probe_correction --data outputs/visuotactile/teacher03/trajectories.h5 --checkpoint outputs/visuotactile/train02/checkpoint.pt --episodes 2 --output "$RUN/replay.json"

仅 CPU 重放真实传感器。nominal chunk 来自冻结模型的因果预测；记录命令在当次观测处理结束后才作为下一步已发命令历史提交。没有生成标签、没有训练 loss、没有执行 correction。在线 --contact-threshold-N 默认 .1 N 仅是待校准的门控候选，不是压力安全阈值。

## 回归与单元测试

    bash visuotactile/run_cpu.sh -m pytest -q tests/test_visuotactile.py

    RUN=outputs/visuotactile/regression_$(date +%Y%m%d_%H%M%S)
    bash visuotactile/run_cpu.sh -m visuotactile.regression --mode rendered --steps 32 --output "$RUN/reference"
    bash visuotactile/run_cpu.sh -m visuotactile.regression --mode instrumented --steps 32 --output "$RUN/instrumented"
    bash visuotactile/run_cpu.sh - "$RUN" <<'PY'
    import sys
    from pathlib import Path
    import numpy as np
    root = Path(sys.argv[1])
    for filename in ('initial.npz', 'trace.npz'):
        a = np.load(root/'reference'/filename)
        b = np.load(root/'instrumented'/filename)
        assert a.files == b.files
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    print('Rendered original / instrumented regression passed')
    PY

--mode original 保留无渲染原类用于复现本机问题，不能作为有效 CPU 物理评估基线。无渲染时运动学桌子/孔/目标在首步移回原点；相同渲染调用下，32 步全部 15 个字段逐位一致。见 AUDIT.md。原task/RL/控制数学未变；当前action_utils只有可选只读trace hook，回归已覆盖该路径。

## 实际结果（不是性能承诺）

| 证据（outputs/visuotactile 下） | 真实结果 |
| --- | --- |
| probe04 | 原相机空画面；斜视约 69% 深度有效；arm 每 tick 最大目标差 .00250006 rad |
| episodes02 + episodes02_audit.json | 2 env × 660 ticks，7 完整/6 all-goal+retract 成功，2 不完整尾段；五指力非零；对齐通过 |
| train01、resume01 | 真数据 100 次更新与恢复到 102；精确重载通过 |
| train02 | 续训 900 次至 1000；验证 epsilon loss .73599；精确重载；142.9 s CPU 训练 |
| teacher03 | SAPG group 0，2 env × 660 ticks，6 完整/5 成功；23.44 s（不含启动） |
| nominal_shadow02 | 100-step 模型，同预算，2 完整/0 成功，无接触；旁路候选未执行 |
| nominal_shadow03 | 1000-step 模型，同预算，2 完整/0 成功，无接触；没有任务能力证据 |
| correction_replay01.json | 两段真实接触：225/230 步，184/134 接触帧，28.71/44.92 N 峰值；历史/有限数/reset 检查通过；无训练/执行 |
| unit_tests_final.log | 最终 8 passed / 2.60 s；requests 依赖版本 warning，未修改共享依赖 |
| comparison_final.json | 三个同配置/种子/预算 run 全轨迹审计通过；shadow 执行动作逐项等于 nominal，前序命令未跨 reset |
| regression_rendered_comparison.json | 渲染原类与 instrumented 32 步全部字段逐位一致 |

旧 outputs/compat_check 中 PPO/SAPG/finetune 测试只证明入口曾运行；旧无渲染 CPU 0/3 评估被新的场景健康发现降级，不能据此判断专家质量。

nominal_shadow03 的两环境重规划中位数 28.44 ms、最大 35.73 ms，但每步调用加旁路最大 171.48 ms，shadow 自身最大 158.80 ms。同步仿真结果不能证明 60 Hz correction 或严格 10 Hz 实时截止期。Teacher 原概率网络即使选均值，也采样未使用的动作并消耗全局 RNG；nominal 使用独立 generator。相同起始种子/配置不意味着完全相同的逐步扰动或后续 episode 初值。当前同种子小样本比较不是统计性能或泛化结论。

## 明确未完成

训练过的 correction、demonstration nominalization、视觉目标估计、充分覆盖的多视角泛化比较、接触校准/滑移估计、真实时延/截止期与实机验证均未完成。学生传感器延迟/噪声/随机安装位当前显式拒绝，原 teacher/action DR 保留。不得把全末节法向合力称为触觉图像、切向力或压力。
