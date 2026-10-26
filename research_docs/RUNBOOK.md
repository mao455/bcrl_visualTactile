# 可复现命令与验证结果

从仓库根目录运行。所有命令使用项目 .venv_isaacsim。run_cpu.sh 只限制它新启动的进程：nice 19、两线程、CPU 22/23（可用 RESEARCH_CPUS 指定），不操作其他任务或 GPU 配置。CPU physics 仍需 GPU 渲染相机，不能承诺零资源竞争。这些是短实验命令，不是论文规模训练。

输出目录必须不存在；使用新的 RUN 前缀保留历次证据。退出码不是充分证据：Kit 关闭曾掩盖异常，必须确认 result.json 存在且没有 FAILED.txt，训练确认 report.json，并运行数据审计。

## 完整的小规模实验链

    RUN=outputs/visuotactile/run_$(date +%Y%m%d_%H%M%S)

    # 原专家 + 真实相机/接触数据，保留原 DR 和执行器路径
    bash visuotactile/run_cpu.sh -m visuotactile.collect --steps 660 --num-envs 2 --teacher-group 0 --output "$RUN/teacher"

    # 默认只使用完成所有目标和 retract 的专家 episode
    bash visuotactile/run_cpu.sh -m visuotactile.train --data "$RUN/teacher/trajectories.h5" --output "$RUN/train" --steps 100 --batch-size 4

    # 模型独立执行；shadow 只记录未训练候选，绝不执行候选
    bash visuotactile/run_cpu.sh -m visuotactile.collect --steps 660 --num-envs 2 --policy-checkpoint "$RUN/train/checkpoint.pt" --correction-shadow --output "$RUN/nominal"

    # 对齐/有限数/reset/输入及配置一致性审计和成功统计
    bash visuotactile/run_cpu.sh -m visuotactile.evaluate --data "$RUN/teacher/trajectories.h5" "$RUN/nominal/trajectories.h5" --output "$RUN/evaluation.json"

若本次没有成功 episode，普通训练会拒绝；应检查专家/场景或多采集。--smoke-only 仅显式允许真实失败/不完整轨迹用于软件测试，不生成标签。该标志为 false **不表示模型已有任务能力**。

## 用已保存的真实数据继续训练

    RUN=outputs/visuotactile/resume_$(date +%Y%m%d_%H%M%S)
    bash visuotactile/run_cpu.sh -m visuotactile.train --data outputs/visuotactile/episodes02/trajectories.h5 --resume outputs/visuotactile/train02/checkpoint.pt --output "$RUN/train" --steps 100 --batch-size 8
    bash visuotactile/run_cpu.sh -m visuotactile.collect --steps 660 --num-envs 2 --policy-checkpoint "$RUN/train/checkpoint.pt" --output "$RUN/rollout"

恢复要求相同数据哈希、模型配置和 smoke 筛选模式，恢复优化器与 torch RNG。PolicyConfig 保存历史长度/stride、预测 horizon、重规划周期和 diffusion 步数。配置实验不要通过改变环境 decimation 模拟低频网络。当前默认 history=4、stride=3、horizon=12、replan=6，即 120 Hz 物理、60 Hz native control、10 Hz nominal replan。同步仿真等待推理，尚不是带真实墙钟延迟/超时回退的实时系统。

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

--mode original 保留无渲染原类用于复现本机问题，不能作为有效 CPU 物理评估基线。无渲染时运动学桌子/孔/目标在首步移回原点；相同渲染调用下，32 步全部 15 个字段逐位一致。见 AUDIT.md。原 task/control/RL 源码没有修改。

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

训练过的 correction、demonstration nominalization、视觉目标估计、多视角比较、接触校准/滑移估计、真实时延/截止期与实机验证均未完成。学生传感器延迟/噪声/随机安装位当前显式拒绝，原 teacher/action DR 保留。不得把全末节法向合力称为触觉图像、切向力或压力。
