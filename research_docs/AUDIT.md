# 原仓库运行链路与本机兼容性审计

基线 commit：`ff2fc62873f6c6cd93164123c8358e9ce2d65c9b`。不是对整个 Isaac Lab 的兼容性认证；以下区分源码事实与实际执行证据。

## 入口、资产与控制

- 原 README 的 Stage 1 入口是 `isaacsimenvs/train.py` + `Isaacsimenvs-Play-Direct-v0`；Stage 2 使用 `Isaacsimenvs-PreciseAssembly-Direct-v0`。PPO 与 SAPG YAML 分开。交互评估和离线 first-episode 评估分别在 `evaluation/eval_isaacsim.py`、`evaluation/eval_offline.py`。四个注册任务为 tight insertion、beam step 1/2、screwing；新增链路目前仅实测 tight insertion。
- `PreciseAssemblyEnv._configure_problem` 从 `evaluation/problems` 和 `evaluation/objects` 解析 URDF、插入姿态与子目标；`precise_assembly/scene_utils.py` 复用 Play 的 URDF→USD、属性烘焙和机器人构造。所有实体注册进同一个 `InteractiveScene`。
- 机器人资产：`assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf`。当前 L-peg：`assets/urdf/peg_in_hole/lpeg_matchedmass/lpeg_matchedmass.urdf`；孔：`holes/hole_tol0p5mm/hole_tol0p5mm.urdf`；桌子：`assets/urdf/table_narrow.urdf`。限位/质量/惯量继续由这些资产与原转换器确定，未重写。
- 原机器人固定基座，禁用自身碰撞和重力；物体动态且有重力；桌子、孔座、GoalViz 是运动学刚体，GoalViz 无碰撞。原烘焙 contact offset=.002 m、rest offset=0；原机器人 solver position iterations=8，物体/孔/GoalViz=4，velocity iterations=0。新增只在现存五个 distal 刚体启用接触报告，不改几何。
- 原 ImplicitActuator 继续通过 `build_robot_articulation_usd_cfg` 写入原增益。驱动必须先由 URDF 转换建立 force/position DriveAPI；转换时零增益不代表运行增益为零。
- 当前材料路径 `apply_physx_material_properties` 给普通机器人、桌子、物体、孔座设置 `robot_friction=.5`，指定指尖形状为 `finger_tip_friction=1.5`；另有配置字段不等于它实际被该运行路径使用。恢复/修改摩擦应先查这个函数，不要仅改配置字段。
- 每 tick：29 维 canonical 命令 → 原动作延迟队列 → canonical/Lab 排列 → 前 7 臂关节速度式目标积分与平滑、后 22 手关节绝对归一化位置映射与平滑 → 两个物理子步写同一目标。具体公式与坐标约定见 `CONTRACTS.md`。

## 任务、噪声、reset 与监督来源

当前实测 config 保留 1/120 s physics、decimation=2、gravity=(0,0,-9.81)、动作/总观测延迟队列长度 3、物体状态延迟长度 10、物体位置噪声 .01 m/转角 5 度、actor 关节速度噪声 .1，及原随机外力/力矩。队列长度不是最大 lag：长度 3 的实际 lag 为 0..2。学生编码器速度没有继承 actor 的噪声。

桌高复位基准 .38 m、随机半宽 .01 m；孔 XY 随机范围保留原配置。当前 `goal_mode=preInsertAndFinal`、两子目标、`random_goal_fraction=0`、`enable_retract=true`。达到子目标会把原 `episode_length_buf` 清零，因此它不能作为完整 episode 身份。任务插入阈值 .01 与 retract 阈值 .005 还乘 `reward.keypoint_scale`；不能把 README 的几何间隙当成成功判据。retract 还要求平均指尖距离 >.1 m。

`_get_dones` 内的 `compute_intermediate_values` 会更新近目标计数；随后 `_get_rewards` 使用同一缓存；终止后 DirectRLEnv 先 reset 再返回新观测。记录器在奖励 hook 保存终止状态，使用独立 episode/step ID。不额外调用 task 中间量函数。新数据筛选当前要求 all-goals 与 retract 都完成；原离线严格连续成功规则与训练配置不可混为一个指标。

Teacher 的 140-D noisy actor 和 162-D privileged critic 含模拟物体/目标信息，噪声不使其成为可部署估计器。新 nominal 只用 RGB-D + 58-D 编码器；原 student 87-D 中最后 29 维执行目标可能携带 correction 输出，只存诊断。训练目标保留专家原动作，不能称为去压力/去预加载后的动作。

## CPU 无渲染回归问题：已定位行为，未修改引擎

环境：Python 3.11.15、Isaac Sim 5.1.0.0、本地 Isaac Lab 2.3.2 fork、torch 2.7.0+cu128。使用项目 `.venv_isaacsim`。

初次 original/no-render vs instrumented 比较中，关节/目标看似一致，但 actor 最大差 .931697、reward 最大差 .6378815。后续增加原始 PhysX 根变换和 RNG 跟踪，发现：

1. reset 后两条路径初值完全相同。原 CPU 无渲染路径第一步后 table z 从 .38666487 变为 0，hole z 从 .53666484 变为 0，GoalViz z 从 .72266483 变为 0；PhysX 原始查询也如此，不只是观测缓存。
2. 给原类的 `_get_observations` 仅增加 `sim.render()`，不加传感器、不改控制/任务，运动学位置保持 reset 值。
3. 这个 rendered-original 与 instrumented 路径的 32 步 trace 所有 15 个字段逐位相同：关节、目标、物体/孔/桌/GoalViz 根状态及 PhysX 根变换、RNG、观测、reward、done、success。这把差异定位到本机 CPU 的无渲染运行方式，而非新增触觉网络或普通观测噪声。底层引擎为何出现这个行为尚未进一步修改/证明。

证据：`outputs/visuotactile/regression_v2_original/`、`regression_v2_rendered/`、`regression_v2_instrumented/`、`regression_v3_rendered/`、`regression_rendered_comparison.json`。新的 adapter 每步校验 table/hole 未偏离 reset 位置，失败则中止保存成功结果。没有向原源码打补丁、强行覆写刚体位置或更改动力学。

**撤回先前把 reward/actor 差异解释成随机噪声的说法。** 旧无渲染兼容性测试只能说明入口执行结束，不能证明场景物理正确；旧 CPU 0/3 专家评估不作为专家能力证据。仍未实测原 GPU physics 或所有任务/配置。

## 相机与触觉实际可见内容

`outputs/visuotactile/probe04` 与 `episodes02` 保存 PNG 和 raw depth 示例。原相机 RGB 标准差为 0、有效深度比例为 0；可选斜视 eye=(.6,-.8,1.1)、look-at=(0,.15,.65) 相对环境原点，约 69%–70% 有效深度。实际相机内参 fx=fy≈115.6718，cx=80、cy=45；本机忽略了请求的 aperture offset。新的元数据记录实际 K、ROS wxyz 姿态和相对环境的位置。

人工查看可见机器人、桌面、孔座/物体区域，但 160×90 细节少、手部遮挡明显；单视角足够性未证明。GoalViz 隐藏，不能把彩色真值目标通过图像泄漏给学生。固定 checkpoint 对应外部选择的 task identity，目前没有显式合法目标姿态输入；下一步应比较相对的第二视角/更高分辨率与外部目标命令，而非直接喂真值。

五个接触 sensor 对应 `left_{index,middle,ring,thumb,pinky}_DP`，不是虚构 taxel 图像。每 1/120 s 更新、记录连续两子步、world axes、单位 N；合并固定关节后一个 distal 刚体可能覆盖多种接触表面，未区分接触对象、切向力、力矩或局部压力。实测五通道都非零。

当前明确拒绝 student camera delay、bundle delay、pose randomization、depth augmentation 与非零 camera skip period：这些的曝光时间/独立 RNG 尚未验证。原 pose randomization helper 实际不检查 student enabled，所以并非原先猜测的 no-op；手工斜视安装若直接启用它会复位到 config 的旧安装位，因此先阻止这种混用。原 teacher/action DR 不受这个限制影响。

## 2026-10-27 — v2感知回归补充

新增固定相机、post-scene-update编码器FK采样和未训练网络旁路后，原渲染路径与instrumented路径32步的15个trace字段仍逐位相同（含RNG、原actor观测和PhysX变换）。另对camera03与final01的2env×240ticks记录比较，480条专家动作、执行器目标、奖励、关节、物体、孔座、success和done逐项一致。证据：`outputs/visuotactile/perception_regression01/comparison.json`，公开汇总在`evidence/perception_v2.json`。

仍保留原CPU渲染约束；没有修改引擎、碰撞、控制或任务成功判据。原路径仍会输出运动学刚体不能设置速度的PhysX日志，本轮以实际位置断言/状态回归作为有限范围内的有效性依据，不能宣称所有引擎问题已消除。当前观测/图像/传感器源见`PERCEPTION.md`，terminal图不可用及同tick插入/retract成功现象已明确记录。

## 2026-10-27 持续历史与trace2补充

重新检查原action pipeline后，只增加7行可选只读hook来观察其**实际**delay选择；原数学/RNG/资产/RL未改。新`causal_regression01`再次对比rendered原环境与instrumented32步，initial13项/trace15项bitwise相等。真实480行记录补充逐步延迟源、FK实际弧度、终止缺失和持续历史检查；具体范围和误差见 [HISTORY_AND_TRACE.md](HISTORY_AND_TRACE.md)。历史段落的“原源码未修改”描述的是当时里程碑，不能替代当前7行hook的审计。CPU无渲染限制仍在。
