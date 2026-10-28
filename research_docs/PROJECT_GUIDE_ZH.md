# 当前项目怎样运行：从专家到三路图像与逐步数据

本说明以2026-10-28源码为准。本轮**操作者是原论文已发布的RL专家**，不是视觉BC，也不是触觉Correction。新增网络可以前向、反向检查接口，但没有通过训练获得能力。当前采集配置为v4 `wrist_bracket_close`，三路448×252 RGB-D；perception/schema2、memory2、trace2、storage2是不同层的版本，不能互换。旧v3三条生产数据单列。

## 一次控制步的实际顺序

1. `collect.py::main`用注册任务配置构建`InstrumentedAssemblyEnv`，`Runner.create_player → _load_checkpoint_weights → player.set_weights`加载217,700,467字节的released `tight_insertion/model.pth`。SAPG group0为50。它仍接收140维teacher observation；研究图像不会拼进此专家输入。
2. `packet = env.research_packet`冻结本步动作前研究观测。`player.get_action(actor_obs, is_deterministic=True)`给出29维命令；确定性均值推理仍可能消耗内部被弃用采样的RNG，因此记录RNG并做真实配对。
3. `env.step(action)`仍进入原`play/utils/action_utils.py::apply_action_pipeline`。先clip[-1,1]、canonical到Lab顺序、随机延迟队列，再做关节目标处理。前7维是臂：`target = .1*clip(previous+1.5*(1/60)*action)+.9*previous`；后22维手是限位区间内绝对映射再平滑。29维命令不是29个角度，也不是所有维度统一增量。
4. 原执行器在两个1/120秒物理步使用目标；原`PreciseAssemblyEnv::_get_dones`判断阶段/成功，再奖励、自动reset、生成下一观察。任务的全部插入目标完成且撤手成功才计成功；子目标变化不等于新episode。
5. `EpisodeWriter.append`将动作前packet与本步transition对应写入同一行，最后发布committed length。`diagnostics_after`和接触是reset前末状态；真正终止图像目前取不到，`observation_after/available=false`，不能拿下一episode的图填上。

BC标签是专家本步/后续步骤实际提交的**29维命令序列**；`target_before/after_rad`是经延迟、积分、限位、平滑后的控制器目标；`joint_position_rad`是机器人实际角度。它们不是同一个量。未来专家动作是监督，不能喂给当前策略。普通专家示范还可能包含接触预加载，尚未做nominalization。

## 读代码的入口（函数名可直接搜索定位）

| 文件与入口 | 当前职责与关键语句 |
|---|---|
| `isaacsimenvs/tasks/play/play_env.py`；`utils/action_utils.py::apply_action_pipeline` | 原物理/执行链，29维命令最终调用原joint target接口；本轮不改 |
| `isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py::_reset_peg_episode/_get_dones` | 孔位、任务目标、阶段及自然成功；不使用新视觉网络判成功 |
| `visuotactile/collect.py::main` | 加载专家、`player.get_action`、`env.step`、逐步记录；可选global输出预留只检查磁盘，不改动作或RNG |
| `visuotactile/sim/environment.py::_get_observations` | 保留teacher路由，同时构造`research_packet`；相机/关节/触觉与特权诊断分开 |
| 同文件`_setup_scene`；`camera_rigs.py::rig_contract` | 两台固定＋`iiwa14_link_7/ResearchWristCamera`；v4仅移动高位光心，局部腕部变换固定 |
| `camera_rigs.py::renderer_pose_ros`；环境CameraParams annotator | 从与RGB/Depth同一个render product的view transform取得动态相机位姿。本机CPU的camera.data缓存会过期；修复的是标定记录，不是机器人控制 |
| 环境`wrench_frame_mode='world_hold'`的write wrapper | 原DR按世界系生成力；每个物理子步用当前姿态转换成body力再提交，避开安装版composer过期姿态缓存。不改原概率、强度、门控；它是明确版本化的兼容模式，不能与legacy混算 |
| `models/vision.py::RGBDSpatialEncoder` | RGB用ResNet18前半段，**`weights=None`**；Depth＋有效mask独立卷积。学习的空间moments和3×5网格保留位置线索；不是已经识别的物体/孔口语义点 |
| `models/vision.py::MultimodalHistoryEncoder` | 按相机ID顺序融合每路128D，再加关节、类别task ID、合法目标/自身计划，送持续GRU。融合后隐变量不再逐维对应某一路或物理量 |
| `models/nominal.py::NominalDiffusionPolicy` | 非触觉条件生成H×29动作；未来训练loss是噪声预测误差，经动作有效mask计入。视觉参数届时通过该损失学习；本轮没有优化器更新 |
| `models/tactile.py::FingerTactileEncoder` | 五指各有身份embedding和GRU，输入聚合法向力及编码器FK，融合成触觉特征；不是GelSight图像或压力阵列，也不能测完整摩擦/可靠滑移 |
| `models/correction.py::CorrectionPolicy` | 独立视觉、触觉、motion、Nominal意图分支；输出候选29维。Correction不等于触觉编码器；最终损失/动作组合未确定 |
| `history.py::OnlineHistory/recurrent_scan`；`rollout.py::NominalController.act` | memory2每60Hz观察更新持续状态，每6步重规划；只在真实episode ID变化时清空。自身旧预测计划与已执行专家历史不同；原轨迹另存，GRU不无损记住全部过去 |
| `recording.py::EpisodeWriter/EpisodeReader` | 原始字段逐控制步写/读、明确缺失与来源；不是只存网络特征 |
| `compact.py::convert_episode/EpisodeFile` | RGB精确还原uint8，Depth保留float32米，去除重复post图；逐字段无损重建验证。视频只是观看副本，生产可不写 |
| `manifest.py::ManifestReader/ManifestSequenceDataset`；`dataset.py` | hash/长度、成功筛选、布局和实际标定兼容、整episode顺序分块读取；禁止新旧相机静默混读 |
| `train.py::main` | 已有单H5顺序TBPTT骨架；正式manifest分组训练、memory2验证loss尚未接通。纯专家缺自身预测计划时正式self-plan训练明确拒绝；需先定义无计划baseline或真实因果模型回放。没有Correction正式trainer |
| `collection_resources.py::Supervisor`；`collect_success_batch.py::run` | 整卡GPU采样、全项目磁盘、绝对截止时间；只墙钟暂停自己创建的进程组，不能暂停/改动他人任务。未知软件/审计错误停止，普通非法起点分类后下一seed |

## 触觉与信息边界

五指normal force来自安装版本ContactSensor的`net_forces_w`，每控制步两个120Hz样本。实际canonical关节角以弧度进入原URDF FK；本机基座方向必须identity，world→指末节局部用旋转的逆。位置是末节原点，不是假定的接触点。局部力差包含手指坐标系转动的影响；当前位置差除真实dt才是速度，力差本身不是变化率或滑移标签。

Nominal只读白名单：视觉、58维q/dq、暂定类别task ID、带mask的外部目标命令、自己的旧动作/计划。仿真孔位/成功/扰动力仅供诊断和标签；不变成学生目标。Correction可读真实历史执行命令、控制目标与跟踪误差、触觉和Nominal意图。纯专家正式采集没有额外跑Nominal/Correction；其computed=false，不伪造预测。可选检查中Correction保持Shadow，输出从未执行。

## 三台相机、连接空隙及目前局限

左fixed_high、中央fixed_side、右wrist_rgbd已经对原交付seed17/920101/920302同步视频和单路视频逐帧进行原图匹配；不只是依名称推断。相机只是观测节点，没有新增外壳、支架、质量或碰撞。实体KUKA左手支架适配仍未验证。

URDF `iiwa14_joint_ee`有0.045m固定偏移，空`iiwa14_link_ee`接无visual/collision的`sharpa_mount`，再经0.05m固定偏移接手掌。转换后合并到link7刚体，连接仍在；0.095m原点位移不是可见缝隙长度。保持原资产，不为补画面改物理。

v4 high eye=(.288,-.405,1.463)，target=(0,0,.68)，沿旧视线靠近10%；共同焦距40、侧视(-.58,-.06,1.05)、腕部link7局部(.12,.16,.07)均不变。两条完整轨迹的高位构图未发现关键裁切；腕部仍受手背遮挡、长物体局部出画，必须配合外部视角。终止图缺失、真实硬件、泛化可见性、去预加载、Correction损失和正式训练均未解决。
