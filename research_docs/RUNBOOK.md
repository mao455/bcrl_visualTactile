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

## 2026-10-28：原论文专家配对扰动采集（无训练）

先读[RL_EXPERT_TRIALS.md](RL_EXPERT_TRIALS.md)：包含本机world-request与实际body-buffer施力差异。本轮使用`wrench_trace_version=3`，感知schema2/trace2不变。以下输出路径必须尚不存在，不覆盖已有实验。

单条自然episode（external budget不是原timeout）：

```bash
bash visuotactile/run_cpu.sh -m visuotactile.collect \
  --checkpoint pretrained_assembly/tight_insertion/model.pth \
  --teacher-group 0 --condition original --pair-id manual17 --seed 17 \
  --episodes 1 --max-steps 2400 --num-envs 1 \
  --perception-version 2 --camera oblique \
  --output outputs/visuotactile/manual_exposure_new
```

`--condition force_1p5`仅将config中的force_scale乘1.5；不改torque、概率、门控、动作频率、成功判据。`--episodes`模式限1env；旧`--steps`固定tick模式保留1–2env。不要添加`--policy-checkpoint`或Shadow来混入本轮纯专家对照。

预先声明两个阶段，自动通过验收后继续pilot：

```bash
bash visuotactile/run_cpu.sh -m visuotactile.run_rl_trials \
  --output outputs/visuotactile/paired_experiment_new \
  --exposure-evidence outputs/visuotactile/rl_experiment_20261028/gate_seed17_applied
```

默认验收910101/910102/910103；pilot910201–910220；2400步/480s每trial，逐条独立进程并交替条件顺序。exposure-evidence是已实测非零物理提交的独立诊断，不放进cohort统计；本轮三个验收seed恰好零暴露不意味着原扰动永远不触发。每条完成就审计；runtime/data/initial-state失败会保留记录并停下。

若只在创建轨迹之前发生SIGABRT，可在检查日志后同样命令加`--resume-startup-errors`：复用已完成的attempt，失败seed只允许在`_attempt02`目录再试一次；不会重采已有成功/失败episode，不允许自动重试含HDF5的部分轨迹。原status/log/空目录都保留。不要以重试为由挑选机器人任务结果。

独立重分析（不启动Isaac Sim、不训练）：

```bash
bash visuotactile/run_cpu.sh -m visuotactile.analyze_rl_trials \
  --root outputs/visuotactile/rl_experiment_20261028/cohort_v3/pilot \
  --output outputs/visuotactile/pilot_reanalysis_new
```

`report.json`包含逐episode状态、受力/力矩、五指接触、tracking、缺失字段、配对初始条件/共同前缀误差和分母；`report.md`为摘要；每对一张静态PNG。计划中未启动trial仍列出，运行错误（包括未启动episode的失败attempt）单列。命令保存于每个`.status.json`；stdout/Kit错误保存于邻接`.launcher.log`。

逐episode导出PKL（可选，HDF5继续为主）：

```bash
bash visuotactile/run_cpu.sh -m visuotactile.recording \
  --data outputs/visuotactile/rl_experiment_20261028/cohort_v3/acceptance/pair01_original/trajectories.h5 \
  --episode env000_ep000000 --pkl outputs/visuotactile/episode_export_new.pkl
```

本轮38步v3真实episode已验证139字段及metadata逐项相等：HDF5=20,330,072 bytes，PKL=82,440,908 bytes，不批量导出。纯专家258步`EpisodeSequenceDataset`逐块读取成功，Nominal plan mask始终false；CorrectionDataset正确拒绝没有Nominal预测的数据。该拒绝不代表专家轨迹不合格，只表示尚不满足Correction模型输入前提。

本轮实测完成目录是`outputs/visuotactile/rl_experiment_20261028/cohort_v3/`，最终以两个阶段的`analysis_final/report.json`及[完整中文报告](RL_EXPERT_TRIALS.md)为准。原自动生成的`analysis/`和中途resume分析都保留，后续`analysis_final`增加完整聚合/接触时曝光/来源hash核对；没有重采或改写HDF5。pilot40条均完成，两次无HDF5的启动SIGABRT在同seed的`_attempt02`恢复。27项测试通过；全程没有训练。

## 2026-10-28 有界数据集：已停止，仅回读/审计

先看[INSERTION_DATASET.md](INSERTION_DATASET.md)和[SESSION_HANDOFF.md](SESSION_HANDOFF.md)。本轮根目录是`outputs/visuotactile/insertion_dataset_20261028`，20次尝试后停止，无自动续跑授权。`run_insertion_dataset --stage scan`需已通过acceptance_gate；`--continue-unaffected-initializations`只允许跳过已确认动作前非法组继续原计划，不重试其他错误。不要对当前停止批次反复启动。

可复现读取（不启动仿真/训练）：

```bash
bash visuotactile/run_cpu.sh - <<'PY'
from visuotactile.manifest import ManifestReader, ManifestSequenceDataset
path='outputs/visuotactile/insertion_dataset_20261028/manifest.json'
reader=ManifestReader(path, verify=True)
step=next(reader.steps('s920302_baseline'))
print(step['obs']['image'].shape)  # (2,4,126,224), RGB float32 + metric depth
print(step['transition']['submitted_action'].shape)  # (29,)
dataset=ManifestSequenceDataset(path)
chunk=next(dataset.chunks('s920302_baseline',chunk_size=7))
print(chunk['obs']['image'].shape)  # (1,7,2,4,126,224)
print(chunk['action_target'].shape) # (7,12,29), supervision only
PY
```

独立零更新回放（没有optimizer）：

```bash
bash visuotactile/run_cpu.sh -m visuotactile.check_dataset \
  --manifest outputs/visuotactile/insertion_dataset_20261028/manifest.json \
  --ids s920302_baseline s920302_yminus05cm \
  --output outputs/visuotactile/insertion_dataset_20261028/zero_update_manifest_repeat.json
```

真实旧/新转换验证见`zero_update_conversion.json`；新manifest两episode411步检查见`zero_update_manifest.json`，参数和buffer hash前后相同。不要运行train.py --smoke-only假装零更新；其仍会optimizer.step。正式manifest训练尚有明确预检拒绝及接口缺口，详见报告。

统计重读：

```bash
bash visuotactile/run_cpu.sh -m visuotactile.analyze_insertion_dataset \
  --root outputs/visuotactile/insertion_dataset_20261028
```

`--previews`仅在需要重新导出最多4个拼接视频时使用，会覆盖该批的同名派生预览；正式三文件不改。当前只3个视频，旧失败已标非法初始诊断。图/JSON在analysis，逐字段压缩大小在storage_audit.json，全部原始产物ignored，不公开上传。

## 新三视角验收及已授权400批次（2026-10-28晚）

先读CAMERA_VALIDATION及SESSION_HANDOFF。旧批次停止记录是历史；新400已授权，但必须先通过gate。原专家权重/资产/物理不变；以下CLI不训练。示例路径必须不存在，旧输出不能覆盖。

```bash
bash visuotactile/run_cpu.sh -m visuotactile.collect \
  --output outputs/visuotactile/new_camera_probe \
  --camera-rig wrist_bracket --image-size 448 252 --render-mode quality \
  --seed 17 --episodes 1 --max-steps 2400 \
  --experiment-condition camera_validation --wrench-frame-mode world_hold --camera-diagnostics
bash visuotactile/run_cpu.sh -m visuotactile.camera_review review outputs/visuotactile/new_camera_probe
bash visuotactile/run_cpu.sh -m visuotactile.camera_review clearance outputs/visuotactile/new_camera_probe
bash visuotactile/run_cpu.sh -m visuotactile.camera_review aperture outputs/visuotactile/new_camera_probe
```

相机诊断末尾render-only对照可能已在reset状态，不能当终止曝光。若camera.data位姿不动而图像在动，使用本轮CameraParams同步字段；不修改共享Lab。原双路仍为`--camera-rig legacy_dual`。

400新driver为`python -m visuotactile.collect_success_batch --output <fresh_root> --gate <camera_gate.json>`，由低优先级run_cpu启动。只启动一个driver；运行前先读实际status/PID/log，避免重复采集。本轮gate在`outputs/visuotactile/camera_v3_20261028/camera_gate.json`，生产实际路径见SESSION_HANDOFF。driver冻结代码hash/相机/权重、逐条完整审计/转换发布，5秒间隔。`run_status/status.json`为当前进度，`manifest.json`记录尝试分母与正式episode。`STOP`文件只让本driver在下一尝试前停止；不可操作其他GPU进程。任何blocked必须先阅读error/日志与现场，不自动循环重启。

读取三视角生产数据：

```python
from visuotactile.manifest import ManifestReader, ManifestSequenceDataset
manifest = 'outputs/visuotactile/insertion_triple400_20261028/manifest.json'
r = ManifestReader(manifest, verify=True)
print([(k, v['outcome']) for k,v in r.entries.items()])
ds = ManifestSequenceDataset(manifest, stage='production')
episode = ds.episodes[0]
chunk = next(ds.chunks(episode, chunk_size=8))
print(chunk['obs']['image'].shape)  # (1,T,3,4,252,448)
```

此读取/顺序记忆接口可用，不等于正式manifest训练已就绪。纯专家缺自身Nominal预测，不要用专家未来命令补prior_plan，也不要启动train.py做“零更新”检查。

实际启动修复追加：本机inotify额度紧张，新400使用`--disable-extension-watch`（Kit进程内扩展热重载false），经两个完整配对probe验证；不修改全局sysctl/IDE。新gate为`camera_gate_watchoff.json`，新root为`insertion_triple400_watchoff_20261028`，需要显式`--predecessor outputs/visuotactile/insertion_triple400_20261028`保留首次启动失败并从930001继续。该旧root没有episode；通用中途有数据resume尚未实现，不可绕过guard删除既有数据。上面的读取示例manifest路径应改为这个新root。

已实际启动：watchoff driver PID1996734；首条`episodes/s930001/data.h5`完成143字段审计与SHA验证，manifest/production Dataset读取成功。当前统计必须读`insertion_triple400_watchoff_20261028/run_status/status.json`及manifest，文档快照不是实时进度。详见SESSION_HANDOFF。原root仍保留启动失败，不要对它再次启动相同seed。

最新状态纠正（21:18 CST）：PID1996734已退出；新root blocked at3/400，s930005是LLVM启动allocation错误，暂无场景/H5。三条正式文件SHA再检查通过。不要仅凭上面的启动记录判断仍在跑；先读status及blocker_llvm.json。已有3条数据的通用恢复未实现，现有predecessor选项会拒绝非空episode，不能删数据绕过。400授权保持，但先诊断启动阻断。
