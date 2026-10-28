# 初始化与扰动：配置与真实记录分别解释

**Scope.** This is the measured reset/DR report for the released
`tight_insertion` expert and the three committed version-3 production episodes
(`s930001`--`s930003`). It is not a new reset policy and it does not describe
the separate cuRobo/small-object project. No parameter was trained here.

## How the values are selected

The task is registered as `Isaacsimenvs-PreciseAssembly-Direct-v0` in
`isaacsimenvs/tasks/precise_assembly/__init__.py:28-39`. The typed defaults are
in `isaacsimenvs/tasks/play/play_env_cfg.py` and the task overlay is
`isaacsimenvs/cfg/task/PreciseAssembly.yaml`. `visuotactile/collect.py:84-119`
loads that registered configuration, applies only the requested experiment
label's `force_multiplier`, selects the released checkpoint, and passes the
camera rig, seed, zero XY offset and wrench mode to
`InstrumentedAssemblyEnv`. The SAPG YAML at
`isaacsimenvs/cfg/train/PreciseAssemblySAPG.yaml` configures the RL player; it
does not replace the task reset or physics values.

| source / field | resolved value and meaning |
| --- | --- |
| robot / table / object | `assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf`, `table_narrow.urdf`, `lpeg_matchedmass`; object scale `[6.25,.75,.5]`; table XY scale `[1,1]`; base `(0,.8,0)` with identity quaternion |
| table/object reset | table z `0.38 +/- 0.01 m`; object z offset `0.25 m`; native object XY noise each in `[-0.1,0.1] m`, z noise `[-.02,.02] m`; table XY and yaw randomization are both zero in production |
| object orientation | installed `math.random_orientation` draws four independent standard Gaussians and normalizes the quaternion (uniform 3D orientation); not yaw-only |
| joint reset | arm and fingers use a blend fraction `0.1` between the default pose and a random joint-range sample (`reset_utils.py:198-209`); it is **not** ±0.1 rad. Velocity interval is ±0.5 rad/s. |
| geometry / target | fixed object dimensions `[.141,.03025,.0271] m` in the reward; `tight_insertion`, `preInsertAndFinal`, delta goal distance `.1 m`, delta rotation `90°`, hole x/y ranges `[-.1875,.1875]` / `[-.1,.1] m`, no random-goal curriculum (`random_goal_fraction=0`) |
| observation DR | `use_obs_delay=true`, `obs_delay_max=3`: actual queue lags are 0/1/2 control steps; object queue length10, actual lag0–9, XYZ noise std `.01 m`, rotation noise `5°`, joint velocity noise std `.1 rad/s`. These are teacher/diagnostic fields unless a student estimator explicitly consumes them. |
| action DR | `use_action_delay=true`, `action_delay_max=3`: actual lag 0/1/2, measured in `transition/control_trace`; native arm/hand moving averages remain `.1`, arm speed scale `1.5`. |
| external wrench | force and torque probabilities independently log-uniform in `[.001,.1]` at reset; scale 20.0 is used in `randn(3) * object_mass * force_scale`, not a fixed or maximum 20 N; torque scale 2.0. Both are gated by lifted/retract state, decay is zero. `world_hold` only changes the audited per-physics-step world-to-body submission wrapper; native DR generation is unchanged. A trajectory may have no nonzero wrench. |
| collector overrides | production uses released checkpoint SHA `7e61015d...09990`, one environment, native reset, zero extra XY, `world_hold`, 120 Hz physics / 60 Hz control, `wrist_bracket` v3 for the three old records. The version-4 probe used the same task and control settings but the new close camera calibration. |

There is no separate settling phase. The code resets and immediately begins the
native episode; calling the initial pose “stably settled” would be inaccurate.
Startup failure before the scene or metadata is created has **no initial state**;
it must not be represented by zeros.

## Three real initial states from committed production successes

Values are read from each episode's `data.h5` `metadata_json` and
`transition/diagnostics_before` arrays. Object quaternions are internal `wxyz`.
`arm q[0:7]` is canonical radians and `||dq||` is the 29-joint rad/s norm.

| episode / seed | object pose `(x,y,z,qw,qx,qy,qz)` | hole position (m) | arm q[0:7] (rad) | `||dq||` | steps / outcome |
| --- | --- | --- | --- | --- | --- |
| s930001 / 930001 | `(0.035756,-0.000229,0.632480,0.142542,0.305092,-0.925696,0.172302)` | `(-0.071171,-0.023463,0.526625)` | `(-1.32441,1.32564,-.13353,1.12057,-.01591,1.17425,1.30305)` | 1.4063 | 160 / success |
| s930002 / 930002 | `(-0.000055,0.056078,0.631235,-.113940,.417388,-.888115,-.155104)` | `(-.046041,.060124,.536649)` | `(-1.14950,1.55076,.20059,1.42314,.00232,1.36413,1.39754)` | 1.4558 | 163 / success |
| s930003 / 930003 | `(-.046796,.030519,.645310,.356912,.193447,.662153,.629878)` | `(-.046859,.018598,.526997)` | `(-1.60563,1.41761,.00892,1.41686,-.01663,1.54125,1.06471)` | 1.6247 | 115 / success |

These three rows are the only initial-state sample shown here; they do not
estimate the native reset distribution or justify a success rate.

## Measured wrench and action-delay exposure

The three episodes contain 438 control steps. Nonzero applied world-force
samples were 8 (s930001), 2 (s930002), and 0 (s930003), with peaks 2.6025 N,
1.0096 N, and 0 N. The corresponding nonzero exposure time at 120 Hz is
`0.0667 s`, `0.0167 s`, and `0 s`. Contact sensor readings were present even
when the external DR wrench was zero; they are aggregate normal-force signals,
not pressure images or complete friction/slip measurements.

The actual selected action-delay proportions were:

* s930001: lag 0/1/2 = 43/52/65 (26.9/32.5/40.6%);
* s930002: 48/54/61 (29.4/33.1/37.4%);
* s930003: 39/42/34 (33.9/36.5/29.6%).

The stored `external_force_after_dr_w_N`, `applied_wrench` and
`control_trace/effective_lag_steps` are the evidence. Noise empirical
distributions not recorded by the collector remain **unmeasured**; they are
not reconstructed from configuration values. The version-4 seeds 930006,
930008 and 930009 were rejected before any action because the native box
initialization overlapped; 930010 failed during Kit startup with LLVM OOM and
therefore has no initial state.

## Current interpretation

The three records are version-3 `wrist_bracket` data and must remain separate
from the version-4 `wrist_bracket_close` calibration. The latter has not yet
passed the two-success gate: one probe accidentally omitted the wrist view due
to a name-dispatch bug (fixed before the next probe), and the next startup hit
LLVM OOM. No claim about the new camera's full-trajectory coverage or 400-item
batch is made here.

## 2026-10-28 补充实测与源码纠正（中文入口）

上述三行初态来自旧v3生产数据，不是新v4的400条。源码进一步追踪：`reset_utils.py::_randomize_robot_dof_state`对默认姿态和关节限位内均匀样本作0.1混合，速度均匀±0.5rad/s；桌z均匀0.38±0.01m，物体z=桌原点z+0.25+均匀±.02m，XY独立均匀±.1m。物体姿态由安装版`math.py::random_orientation`的四维标准正态归一化得到，是全三维均匀旋转。没有沉降等待。

实际task alias在`evaluation/problems/__init__.py:145`把tight_insertion映射到`Lpeg_matchedmass.tol0p5mm`，不是另一个普通peg任务。`PreciseAssemblyEnv::_configure_problem`选择真实URDF、插入序列；`_reset_peg_episode`令孔xy独立均匀采样、孔z=桌z+.15+problem.hole_z_offset，yaw范围0。preInsertAndFinal用该问题两项目标；不使用play随机delta目标值去解释实际插入轨迹。`_get_dones`保留原插入/撤手阈值。

教师对象位置噪声为独立高斯std.01m；姿态是随机单位轴+角度均匀±5°（不是5°高斯std）。对象队列长度10→0…9步；整teacher与动作队列长度3→0/1/2步。关节速度高斯std.1rad/s。目标xy/yaw按实际config是否非零决定；学生相机augmentation、延迟、pose randomization未开启；当前events=None，摩擦/质量乘子范围[1,1]，不能把未启用项当实采扰动。

外力/矩概率是reset时log-uniform [.001,.1]，每控制步Bernoulli抽取；向量为randn×质量×scale，force20不是20N上限。衰减0、lift锁存门控且撤手阶段关闭。world_hold按每个120Hz子步当前姿态提交原world请求；不是新扰动分布。每条不保证有力。

| 旧v3 seed | 实际force概率 | torque概率 | 非零force物理样本 / 峰值N | 非零torque样本 / 峰值Nm | 有force且之前有指接触的控制步 |
|---|---:|---:|---|---|---:|
|930001|.06679349|.04215763|8 / 2.60249|4 / .335605|3|
|930002|.00277753|.00107717|2 / 1.00962|0 / 0|1|
|930003|.00196456|.00674930|0 / 0|0 / 0|0|

旧三条零force比例1/3；438控制步实际lag0/1/2=130/148/160（29.68/33.79/36.53%）。三条完整初态无精确重复；物体XYZ范围[-.046796,.035756]、[-.000229,.056078]、[.631235,.645310]m，样本太少不能代表覆盖。候选包括旧930004非法、930000/005启动未产生初态；新930006/008/009非法、930007两视图错误不合格、930010启动未产生初态。不能把只保留成功的表当无条件reset分布。

小CSV和实测分位数：`outputs/visuotactile/camera_v4_close_20261028/old_v3_initialization_report/`；由`python -m visuotactile.initialization_report <manifest> --output <directory>`重建。程序审计真实applied_wrench与lag，不从配置伪造经验噪声分布。新v4验收seed17/920101单列；后续新生产manifest用同命令另行更新，不合并版本。
