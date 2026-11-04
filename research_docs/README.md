# 当前 StateDP / ImageDP 实验系统

当前结果与训练决策：[STATE_DP_RESULTS.md](STATE_DP_RESULTS.md)。清理、保留项和回归：[CLEANUP_AUDIT.md](CLEANUP_AUDIT.md)。接口与训练配置以 [CONTRACTS.md](CONTRACTS.md) 为准；历史实验只读摘要见 [RESEARCH_HISTORY.md](RESEARCH_HISTORY.md)。

## 当前入口

|功能|唯一维护入口|说明|
|---|---|---|
|可恢复 State/Image 训练|`visuotactile.anchor_continue`|`--job state/image`；恢复完整状态，参数和数据合同不变|
|串行队列及里程碑|`tools.anchor_continue_queue`|一个owner；按已提交stage恢复，禁止重复任务|
|里程碑闭环评估|`tools.anchor_interim_eval`|epoch/cohort由配置给出；等待当前trainer自然完成，独立Kit worker，持有后续训练直到证据审查|
|独立Kit模型worker|`tools.anchor_continue_worker`|历史/执行审计、视频、生命周期记录和可选Image严格输入捕获|
|统一结果分析|`tools.anchor_interim_analysis`|train/validation分列；三checkpoint端点、接触前增量、曲线、视频位置|
|不可变数据与模型|`visuotactile.native_data`、`state_history`、`models.native_dp`、`models.native_vision`|State/Image参数独立；共享合法基础组件|
|共享工具|`checkpointing`、`state_resources`、`video`、`tools.runtime`、`tools.trajectory_metrics`|恢复、资源、进程、视频和统计不再借用旧实验runner|
|依赖与清理|`tools.project_maintenance`|sources→reviewed plan→apply；保护路径、打开文件、SHA、硬链接计数|

命令使用本仓库 `bash visuotactile/run_cpu.sh`，它进入现有Isaac/PyTorch环境并限制CPU线程。离线训练不加载Kit；真实仿真由独立进程执行。原环境入口 `evaluation/eval_isaacsim.py` 和 `isaacsimenvs/` 保持原项目用途。

首先只读检查已有服务和状态：

```bash
systemctl --user status anchor-continuation-20261104-01.service anchor-assessment-20261105-01.service anchor-assessment-analysis-20261105-01.service
cat outputs/visuotactile/native_dp_v2/anchor_continue_20261104_01/state_train/progress.json
cat outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01/status.json
```

评估配置在run的`session.json`；checkpoint、实际PID、SHA、阶段结果均在run内。不能照抄历史启动命令重开训练。当前150 epoch之后需要结果决策，不能仅执行`release`就继续训练。`--hold-parent`是明确的证据审查保持状态。

```bash
# 清理后CPU回归（不启动Kit、不做正式优化）
bash visuotactile/run_cpu.sh -m pytest -q tests
# 仅重新生成已有结果的报告
bash visuotactile/run_cpu.sh -m tools.anchor_interim_analysis --root outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01
```

## 本地数据

400条原始成功记录在`outputs/visuotactile/data/close_v4`。当前split和视觉索引仍在`native_dp_3way/native_20261101_02`；数值缓存仍在`native_dp_v2/position_20261103_01`；DINO缓存仍在`run/long_horizon_20261030_v2/cache`；位置统计仍在`native_dp_v2/h24_parameterization_20261103_01`。这些旧名字是保留依赖，不表示旧训练仍有效，禁止整目录按日期清理。具体保护理由和清单见清理报告。

raw数据、权重、视频仅本地；Git保留代码、精简数值、图与来源SHA。旧实现与完整旧报告可从[清理前版本](https://github.com/mao455/bcrl_visualTactile/tree/48a3516)读取。
