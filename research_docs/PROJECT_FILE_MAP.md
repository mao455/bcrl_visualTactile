# 研究项目逐文件地图

审计基线为 a07d0aa；本轮工具 audit_system.py 也列入。此表覆盖全部 visuotactile Python/shell、研究测试、实际原生任务/配置及专家加载入口。
每项列出路径、源码定义行、静态模块依赖/调用者、潜在 I/O 调用及关联文档/测试。依赖是 AST import 图，不冒充完整动态调用图；CLI、注册表和 shell 子进程调用另见下方主链。
所有源文件 SHA256、完整定义和依赖在 [file_inventory.json](evidence/system_audit_20261031/file_inventory.json)。第三方完整源码不逐文件复制。

## 从入口追踪系统

| 环节 | 入口 → 被调用模块 | 输入 → 输出/副作用 |
|---|---|---|
| 原专家采集 | collect_success_batch → collect → evaluation.eval_isaacsim / rl_games → sim.environment → recording → compact/storage_audit → manifest | 原生观察→专家命令→真实 env.step→成功审核→本地 HDF5/manifest；可启动 Kit/审核进程 |
| 旧命令DP | run_baselines → train_dp → dataset/history/models.nominal或tactile_dp；evaluate_baselines → rollout | 固定320/40/40→训练checkpoint；GPU策略/CPU物理→原生轨迹 |
| H64 DP | run_long → long_data/train_long/validate_long；evaluate_long → long_controller | 冻结DINO缓存→完整episode训练→绝对reference→共享action_reference→native command |
| A/B历史对照 | run_loss_study → train_loss_study/study_metrics → evaluate_study | 相同epoch40 EMA→四条受控续训；只作为显式历史复现 |
| GRU残差 | residual_session → residual_data/train_residual/evaluate_residual → residual_controller | 冻结B0因果计划→专家teacher forcing→contact-gated delta；不回填B0 |
| Transformer | transformer_data/train_transformer_residual → transformer_controller → long_controller | 专家过去实际动作/DP计划→独立残差；部署ack回填实际混合命令 |
| T0/T1 | evaluate_training_initials → restore_observable_state → native env.step；audit_training_initials | 近似初态+原命令或DP→删失/timeout独立报告；不是精确重放 |
| 资源 | collection_resources / training_resources / long_resources / study_resources | 只管理自有进程、锁、墙钟等待；不改模拟dt或他人任务 |

## 状态和边界

`train.py`/早期 probes 是保留的历史接口；当前正式候选从 `train_long.py` 进入。`models/correction.py` 的 full-candidate shadow 与 `models/residual.py` 的已训练 additive delta 不可混称。
`private_history.py` 保存隔离历史的历史实现/检查；当前 Transformer 使用已授权 actual mixed feedback。禁止把旧版本权重切换新输入语义。
`models/vendor` 是固定 MIT DP源码；DINO来自固定torch hub revision。原生任务只多 action_utils 的只读trace hook；环境wrapper是本地扩展，world_hold是显式兼容模式。

## 逐文件索引

### `evaluation/eval_isaacsim.py`

Interactive (viser) evaluation of a precise-assembly policy in Isaac Sim.

- 定义（函数/类:行）：PolicySpec:86, PopenAdapter:100, _resolve_path:120, _default_isaacsim_python:127, _default_policies_dir:137, _load_metadata:148, discover_policies:154, _choose_policy:171, _iter_policy_lines:185, _coerce_override_value:191, _set_attr_path:210, _set_dict_path:219, _wxyz_to_xyzw_np:228, _tensor_bool:232, _tensor_float:236, _tensor_int:240, _done0:244, _load_env_cfg:250, _instantiate_env:266, _apply_env_overrides:276, _configure_agent:314, _sim_get_state:340, _sim_episode:428, sim_worker:479, _worker_main:599, _build_parser:628, _collect_policies:675, _run_viewer:703, main:912, problem:93, __init__:103, is_alive:107, join:110, kill:116, IsaacSimPegDynamicDemo:736, __init__:737, _load_env:752, run:879。
- 来源：upstream identical；依赖：__future__, argparse, dataclasses, evaluation._viser_demo, gymnasium, importlib, isaaclab.app, isaaclab.utils.math, isaaclab_tasks.utils.parse_cfg, isaacsimenvs, isaacsimenvs.tasks.play.utils.obs_utils, isaacsimenvs.utils.rlgames_utils, json, math, multiprocessing.connection, numpy, os, pathlib, rl_games.torch_runner, subprocess, sys, time, torch, traceback, typing, yaml。
- 静态调用者：visuotactile/collect.py, visuotactile/evaluate_baselines.py, visuotactile/evaluate_long.py, visuotactile/evaluate_residual.py, visuotactile/evaluate_study.py, visuotactile/evaluate_training_initials.py, visuotactile/expert_reference.py, visuotactile/regression.py。
- I/O或状态操作候选：PopenAdapter, json.dumps, player.env_step, self._kill_subprocess, self.popen.kill, self.popen.poll, self.popen.wait, subprocess.Popen。
- 文档：PROJECT_FILE_MAP.md, INSERTION_DATASET.md, AUDIT.md；测试：未检出直接import测试；不表示已验证。

### `evaluation/objects.py`

Object registry shared by the assembly problem definitions.

- 定义（函数/类:行）：get_repo_root_dir:18, Object:24, rescale_by_factor:76, __post_init__:36, get_object_mesh_path_and_scale:39, get_object_mesh:69。
- 来源：upstream identical；依赖：dataclasses, numpy, pathlib, trimesh, typing, yourdfpy。
- 静态调用者：evaluation/problems/peg/objects.py, isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `evaluation/problems/__init__.py`

Unified registry for precise-assembly problem instances.

- 定义（函数/类:行）：make_pre_insert_sequence:19, make_transport_pre_insert_sequence:44, Problem:75, __post_init__:102, _normalize_pose_sequence:107, final_insert_pose_rel_receptive:122。
- 来源：upstream identical；依赖：dataclasses, evaluation.problems, math, typing。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `evaluation/problems/peg/__init__.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：evaluation.problems.peg。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `evaluation/problems/peg/objects.py`

Register the L-peg (matched-mass) inserter in the global NAME_TO_OBJECT registry.

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：evaluation.objects。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `evaluation/problems/peg/problems.py`

Register peg-in-hole Problem entries.

- 定义（函数/类:行）：_dense_insert_sequence:30, _playnatural_insert_sequence:55, _register_all:66。
- 来源：upstream identical；依赖：evaluation.problems, pathlib。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：本系统审计/文件地图；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/cfg/__init__.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/cfg/task/Play.yaml`

原生任务/训练配置；由 Hydra / _load_env_cfg / _configure_agent 解析

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/cfg/task/PreciseAssembly.yaml`

原生任务/训练配置；由 Hydra / _load_env_cfg / _configure_agent 解析

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md, INSERTION_DATASET.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/cfg/task/__init__.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/cfg/train/PlayPPO.yaml`

原生任务/训练配置；由 Hydra / _load_env_cfg / _configure_agent 解析

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/cfg/train/PlaySAPG.yaml`

原生任务/训练配置；由 Hydra / _load_env_cfg / _configure_agent 解析

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/cfg/train/PreciseAssemblyPPO.yaml`

原生任务/训练配置；由 Hydra / _load_env_cfg / _configure_agent 解析

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/cfg/train/PreciseAssemblySAPG.yaml`

原生任务/训练配置；由 Hydra / _load_env_cfg / _configure_agent 解析

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/cfg/train/__init__.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/__init__.py`

Play task registration.

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：__future__, gymnasium, isaacsimenvs.tasks.play.play_env, isaacsimenvs.tasks.play.play_env_cfg, pathlib。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/play_env.py`

Thin DirectRLEnv wrapper for Play.

- 定义（函数/类:行）：PlayEnv:31, __init__:34, _setup_scene:46, _reset_idx:49, _pre_physics_step:58, _apply_action:62, _get_dones:66, _get_rewards:71, _get_observations:76, get_student_obs:79。
- 来源：upstream identical；依赖：__future__, isaaclab.envs, isaacsimenvs.tasks.play.play_env_cfg, isaacsimenvs.tasks.play.utils.action_utils, isaacsimenvs.tasks.play.utils.logging_utils, isaacsimenvs.tasks.play.utils.obs_utils, isaacsimenvs.tasks.play.utils.reset_utils, isaacsimenvs.tasks.play.utils.reward_utils, isaacsimenvs.tasks.play.utils.scene_utils, isaacsimenvs.tasks.play.utils.termination_utils, torch。
- 静态调用者：isaacsimenvs/tasks/play/__init__.py, isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py。
- I/O或状态操作候选：log_step_metrics。
- 文档：PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/play_env_cfg.py`

PlayEnvCfg — typed defaults for the Play goal-pose-reaching task.

- 定义（函数/类:行）：AssetsCfg:56, ObsCfg:102, StudentObsCfg:148, ActionCfg:311, RewardCfg:325, ResetCfg:353, TerminationCfg:411, DomainRandomizationCfg:443, _default_sim_cfg:505, PlayEnvCfg:525。
- 来源：upstream identical；依赖：__future__, isaaclab.envs, isaaclab.scene, isaaclab.sim, isaaclab.utils。
- 静态调用者：isaacsimenvs/tasks/play/__init__.py, isaacsimenvs/tasks/play/play_env.py, isaacsimenvs/tasks/precise_assembly/precise_assembly_env_cfg.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/pose_viewer.py`

Pose-only interactive HTML capture for Play training.

- 定义（函数/类:行）：_to_numpy:32, _quat_wxyz_to_xyzw:40, _pose_xyzw:45, _git_output:52, _github_slug_from_remote:63, _derive_github_raw_base:86, _normalize_raw_base:112, _check_url:117, _raw_url_for_repo_path:133, _rewrite_embedded_urdf_mesh_urls:141, object_urdf_for_env:185, object_urdf_text_for_env:201, table_urdf_for_env:205, table_urdf_text_for_env:215, hole_urdf_for_env:219, capture_pose_viewer_frame:229, build_pose_viewer_html:265, PlayPoseViewerWrapper:338, __init__:341, step:397, _capture_student_image:419, _capture_student_image_clean:440, close:457, _finalize_capture:462, _log_wandb:491, _to_uint8_grayscale_rgb:511。
- 来源：upstream identical；依赖：__future__, gymnasium, isaacsimenvs.tasks.play.utils.scene_utils, isaacsimenvs.utils.interactive_viewer, numpy, pathlib, subprocess, time, typing, urllib.parse, urllib.request, wandb, xml.etree.ElementTree。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：_rewrite_embedded_urdf_mesh_urls, html_path.write_text, self.env.step, self.output_dir.mkdir, subprocess.check_output, subprocess.check_output(['git', '-C', str(REPO_ROOT), *args], stderr=subprocess.DEVNULL, text=True).strip。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/utils/__init__.py`

Helper modules for the Play task.

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/utils/action_utils.py`

Action processing and random object wrench helpers.

- 定义（函数/类:行）：sample_log_uniform:10, apply_action_pipeline:18, apply_wrench_dr:84。
- 来源：modified upstream；依赖：__future__, math, torch。
- 静态调用者：isaacsimenvs/tasks/play/play_env.py, isaacsimenvs/tasks/play/utils/reset_utils.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, INDEPENDENT_AUDIT_20261030.md, PROJECT_GUIDE_ZH.md, CONTRACTS.md, PROJECT_FILE_MAP.md, RL_EXPERT_TRIALS.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/utils/generate_objects.py`

Procedural handle-head URDF generation for Play.

- 定义（函数/类:行）：_cuboid_urdf_constant_density:47, _cylinder_urdf_constant_density:80, generate_handle_urdf:115, _compute_mass_and_inertia:133, _handle_head_urdf_variable_density:166, generate_handle_head_urdf:262, _scale_to_3d:293, generate_handle_head_urdfs:302。
- 来源：upstream identical；依赖：__future__, isaacsimenvs.tasks.play.utils.object_size_distributions, math, numpy, os, pathlib, typing。
- 静态调用者：isaacsimenvs/tasks/play/utils/scene_utils.py。
- I/O或状态操作候选：p.unlink, path.write_text。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/utils/goal_sampling.py`

Pure-torch goal-pose samplers for Play.

- 定义（函数/类:行）：_scale_workspace_bounds:12, sample_absolute_goal_pose:21, sample_delta_goal_pose:36。
- 来源：upstream identical；依赖：__future__, isaaclab.utils.math, math, torch。
- 静态调用者：isaacsimenvs/tasks/play/utils/reset_utils.py, isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/utils/logging_utils.py`

Task metric publishing for Play.

- 定义（函数/类:行）：log_step_metrics:8。
- 来源：upstream identical；依赖：__future__, torch。
- 静态调用者：isaacsimenvs/tasks/play/play_env.py, isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/utils/object_size_distributions.py`

Per-handle-head-type procedural-asset size distributions for Play.

- 定义（函数/类:行）：ObjectSizeDistribution:25, __post_init__:36, shape:48, sample_handle_scales:51, sample_head_scales:58, sample_handle_densities:67, sample_head_densities:72。
- 来源：upstream identical；依赖：__future__, dataclasses, numpy, typing。
- 静态调用者：isaacsimenvs/tasks/play/utils/generate_objects.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/utils/obs_utils.py`

Observation assembly and step-shared geometry caches for Play.

- 定义（函数/类:行）：compute_obs_dim:57, _stack_obs_dict:62, _perturb_quat:75, _apply_local_offset:88, _keypoints_world:103, _episode_start:115, _sample_delay:119, _canonical_joint_obs:136, compute_intermediate_values:153, _apply_object_state_dr:210, _apply_obs_delay:223, build_observations:231, _student_proprio_dict:352, _apply_student_tensor_delay:361, _apply_student_bundle_delay:394, build_student_observations:423。
- 来源：upstream identical；依赖：__future__, isaaclab.utils.math, isaacsimenvs.tasks.play.utils.reward_utils, isaacsimenvs.tasks.play.utils.scene_utils, math, torch。
- 静态调用者：evaluation/eval_isaacsim.py, isaacsimenvs/tasks/play/play_env.py, isaacsimenvs/tasks/play/utils/reset_utils.py, isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py, visuotactile/sim/environment.py。
- I/O或状态操作候选：update_near_goal_steps。
- 文档：CONTRACTS.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/utils/reset_utils.py`

State allocation and reset helpers for Play.

- 定义（函数/类:行）：allocate_state_buffers:23, _randomize_robot_dof_state:200, _reset_table_pose:223, _reset_object_pose:271, _reset_goal_pose:304, _clear_goal_trackers:372, reset_goal_trackers:378, reset_env_state:384。
- 来源：upstream identical；依赖：__future__, isaaclab.utils.math, isaacsimenvs.tasks.play.utils.action_utils, isaacsimenvs.tasks.play.utils.goal_sampling, isaacsimenvs.tasks.play.utils.obs_utils, isaacsimenvs.tasks.play.utils.scene_utils, json, pathlib, torch。
- 静态调用者：isaacsimenvs/tasks/play/play_env.py, isaacsimenvs/tasks/play/utils/termination_utils.py。
- I/O或状态操作候选：env._traj_step[env_ids].clamp, env.goal_viz.write_root_pose_to_sim, env.object.write_root_pose_to_sim, env.object.write_root_velocity_to_sim, env.robot.write_joint_state_to_sim, env.table.write_root_pose_to_sim。
- 文档：CONTRACTS.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/utils/reward_utils.py`

Pure-torch reward helpers for Play.

- 定义（函数/类:行）：lifting_reward:8, distance_delta_reward:26, keypoint_reward:40, action_penalty:54, update_near_goal_steps:67, reach_goal_bonus:79, compute_rewards:92。
- 来源：upstream identical；依赖：__future__, torch。
- 静态调用者：isaacsimenvs/tasks/play/play_env.py, isaacsimenvs/tasks/play/utils/obs_utils.py, isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/utils/scene_utils.py`

Scene construction, asset conversion, and runtime material setup.

- 定义（函数/类:行）：build_robot_articulation_usd_cfg:150, build_rigid_object_cfg:179, _log_scene_step:187, _student_camera_data_types:191, hide_goal_viz_for_student_camera:205, _quat_wxyz_to_rotmat:224, _stereo_right_pose_from_left:239, _diagnose_target_expr:255, _expand_link_wildcard:303, setup_student_camera:371, _apply_depth_noise:613, _draw_depth_sticks:670, _apply_camera_pose_rand_at_reset:759, _preprocess_student_depth:812, _validate_student_image_shape:844, _crop_student_image:856, _run_foundation_stereo:874, read_student_camera_image:933, _set_usd_attr:1061, _UrdfSdfCollisionMarker:1072, _usd_safe_identifier:1081, _resolve_urdf_mesh_path:1089, _prepare_urdf_for_isaacsim:1096, _parse_optional_int:1157, _parse_optional_float:1161, _parse_urdf_sdf_collision_markers:1165, _apply_urdf_sdf_collision_markers:1197, _generate_scaled_table_urdfs:1268, _convert_urdf_to_usd:1322, _robot_joint_drive_cfg:1354, _bake_usd:1362, apply_physx_material_properties:1447, _materialize_env_prims:1532, _build_object_scale_tensor:1539, setup_scene:1559, _build_stereo_cam_cfg:549。
- 来源：upstream identical；依赖：__future__, dataclasses, isaaclab.actuators, isaaclab.assets, isaaclab.sensors, isaaclab.sensors.ray_caster, isaaclab.sim, isaaclab.sim.converters, isaaclab.sim.schemas, isaaclab.sim.spawners.from_files, isaaclab.sim.spawners.wrappers, isaaclab.sim.utils, isaaclab.utils.math, isaacsimenvs.tasks.play.utils.generate_objects, numpy, pathlib, pxr, re, shutil, tempfile, time, torch, torch.nn.functional, xml.etree.ElementTree。
- 静态调用者：isaacsimenvs/tasks/play/play_env.py, isaacsimenvs/tasks/play/pose_viewer.py, isaacsimenvs/tasks/play/utils/obs_utils.py, isaacsimenvs/tasks/play/utils/reset_utils.py, isaacsimenvs/tasks/precise_assembly/scene_utils.py, visuotactile/sim/environment.py。
- I/O或状态操作候选：_log_scene_step, alias_dir.mkdir, baked_usd_path.parent.mkdir, obj_path.rsplit('/', 2)[-2].removeprefix, out_dir.mkdir, path.write_text, prim.RemoveProperty, stage.GetRootLayer().Save, tree.write, usd_work_dir.mkdir。
- 文档：CONTRACTS.md, PROJECT_FILE_MAP.md, AUDIT.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/play/utils/termination_utils.py`

Termination and success-tolerance curriculum helpers.

- 定义（函数/类:行）：update_tolerance_curriculum:10, compute_terminations:39。
- 来源：upstream identical；依赖：__future__, isaacsimenvs.tasks.play.utils.reset_utils, torch。
- 静态调用者：isaacsimenvs/tasks/play/play_env.py, isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/precise_assembly/__init__.py`

Precise-assembly task registration.

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：upstream identical；依赖：__future__, gymnasium, isaacsimenvs.tasks.precise_assembly.precise_assembly_env, isaacsimenvs.tasks.precise_assembly.precise_assembly_env_cfg, pathlib。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py`

Dynamic peg-in-hole task built as a PlayEnv variant.

- 定义（函数/类:行）：_xyzw_to_wxyz:37, _obs_field_slice:41, _resolve_asset_path:51, PreciseAssemblyEnv:64, __init__:67, _configure_problem:204, _setup_scene:290, _reset_idx:293, _reset_peg_episode:303, _write_goal_pose:416, _clear_goal_trackers:517, _keypoint_success_tolerance_m:522, _curriculum_eligible_mask:562, _curriculum_success_threshold:571, _wrench_dr_active_mask:581, _get_dones:590, _get_rewards:678, _log_peg_metrics:748, _get_observations:797。
- 来源：upstream identical；依赖：__future__, evaluation.objects, evaluation.problems, gymnasium, isaaclab.utils.math, isaacsimenvs.tasks.play.play_env, isaacsimenvs.tasks.play.utils.goal_sampling, isaacsimenvs.tasks.play.utils.logging_utils, isaacsimenvs.tasks.play.utils.obs_utils, isaacsimenvs.tasks.play.utils.reward_utils, isaacsimenvs.tasks.play.utils.termination_utils, isaacsimenvs.tasks.precise_assembly.precise_assembly_env_cfg, isaacsimenvs.tasks.precise_assembly.scene_utils, math, numpy, pathlib, sys, torch。
- 静态调用者：isaacsimenvs/tasks/precise_assembly/__init__.py, visuotactile/regression.py, visuotactile/sim/environment.py。
- I/O或状态操作候选：log_step_metrics, self._write_goal_pose, self.goal_viz.write_root_pose_to_sim, self.goal_viz.write_root_velocity_to_sim, self.hole.write_root_pose_to_sim, self.hole.write_root_velocity_to_sim。
- 文档：PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/precise_assembly/precise_assembly_env_cfg.py`

Config for the peg-in-hole Play variant.

- 定义（函数/类:行）：PreciseAssemblyAssetsCfg:27, PreciseAssemblyCfg:37, _default_precise_assembly_sim_cfg:83, PreciseAssemblyEnvCfg:108。
- 来源：upstream identical；依赖：__future__, isaaclab.sim, isaaclab.utils, isaacsimenvs.tasks.play.play_env_cfg。
- 静态调用者：isaacsimenvs/tasks/precise_assembly/__init__.py, isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/tasks/precise_assembly/scene_utils.py`

Peg-in-hole scene setup.

- 定义（函数/类:行）：_asset_path:30, _pih_rigid_object_cfg:37, setup_scene:51。
- 来源：upstream identical；依赖：__future__, isaaclab.assets, isaaclab.sim, isaaclab.sim.spawners.from_files, isaacsimenvs.tasks.play.utils.scene_utils, pathlib, tempfile, time, torch。
- 静态调用者：isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py。
- I/O或状态操作候选：_log_scene_step, usd_work_dir.mkdir。
- 文档：CONTRACTS.md, PROJECT_FILE_MAP.md, AUDIT.md；测试：未检出直接import测试；不表示已验证。

### `isaacsimenvs/utils/rlgames_utils.py`

Glue between Isaac Lab DirectRLEnv and our vendored `./rl_games/`.

- 定义（函数/类:行）：_flatten_dict:19, _is_scalar:30, _as_float:36, _mean_float:42, _value_at:50, _has_env_axis:62, _slice_env_axis:72, _remove_env_boundary_from_info:78, EnvStatsAlgoObserver:102, MultiObserver:212, register_rlgames_env:252, teacher_env_info:309, __init__:105, after_init:115, process_infos:119, _process_episode_cumulative:139, _process_episode_final:158, _process_vector_summaries:169, after_clear_stats:187, after_print_stats:193, __init__:218, _call_multi:222, before_init:233, after_init:236, process_infos:239, after_steps:242, after_clear_stats:245, after_print_stats:248, _DAggerRlGamesVecEnvWrapper:273, _process_obs:274, teacher_obs_space:281。
- 来源：upstream identical；依赖：__future__, collections, copy, gymnasium, isaaclab_rl.rl_games, numpy, rl_games.common, rl_games.common.algo_observer, torch。
- 静态调用者：evaluation/eval_isaacsim.py, visuotactile/collect.py, visuotactile/expert_reference.py。
- I/O或状态操作候选：_remove_env_boundary_from_info, self.writer.add_scalar。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `rl_games/rl_games/algos_torch/models.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：BaseModel:13, BaseModelNetwork:36, ModelA2C:64, ModelA2CMultiDiscrete:120, ModelA2CContinuous:187, ModelA2CContinuousLogStd:245, ModelMultiA2CContinuousLogStd:302, ModelCentralValue:440, ModelSACContinuous:478, __init__:14, is_rnn:17, is_separate_critic:20, get_value_layer:23, build:26, __init__:37, norm_obs:53, denorm_value:60, __init__:65, Network:69, __init__:121, Network:125, __init__:188, Network:192, __init__:246, Network:250, __init__:303, Network:307, build:428, __init__:441, Network:445, __init__:480, Network:484, __init__:70, is_rnn:74, get_default_rnn_state:77, get_value_layer:80, kl:83, forward:88, __init__:126, is_rnn:130, get_default_rnn_state:133, get_value_layer:136, kl:139, forward:144, __init__:193, is_rnn:197, get_default_rnn_state:200, get_value_layer:203, kl:206, forward:211, __init__:251, is_rnn:255, get_value_layer:258, get_default_rnn_state:261, forward:264, neglogp:297, __init__:308, split_input_dicts:314, concat_results:348, is_rnn:359, get_value_layer:362, get_default_rnn_state:365, forward:368, neglogp:423, __init__:446, is_rnn:450, get_value_layer:453, get_default_rnn_state:456, kl:459, forward:462, __init__:485, critic:489, critic_target:492, actor:495, is_rnn:498, forward:501。
- 来源：upstream identical；依赖：numpy, rl_games.algos_torch.layers, rl_games.algos_torch.moving_mean_std, rl_games.algos_torch.running_mean_std, rl_games.algos_torch.sac_helper, rl_games.common.divergence, rl_games.common.extensions.distributions, torch, torch.distributions, torch.nn, torch.nn.functional。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：本系统审计/文件地图；测试：未检出直接import测试；不表示已验证。

### `rl_games/rl_games/algos_torch/network_builder.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：_create_initializer:13, NetworkBuilder:17, A2CBuilder:183, Conv2dAuto:564, ConvBlock:570, ResidualBlock:585, ImpalaSequential:617, A2CResnetBuilder:632, DiagGaussianActor:860, DoubleQCritic:889, SACBuilder:912, __init__:18, load:21, build:24, __call__:27, BaseNetwork:30, __init__:184, load:187, Network:190, build:560, __init__:565, __init__:571, forward:578, __init__:586, forward:601, __init__:618, forward:625, __init__:633, load:636, Network:639, build:855, __init__:862, forward:871, __init__:891, forward:902, __init__:913, load:916, build:919, Network:923, __init__:31, is_separate_critic:57, get_value_layer:60, is_rnn:63, get_default_rnn_state:66, _calc_input_size:69, _noisy_dense:76, _build_rnn:79, _build_sequential_mlp:87, _build_mlp:114, _build_conv:128, _build_cnn2d:138, _build_cnn1d:156, _build_value_layer:171, __init__:191, forward:317, is_separate_critic:485, is_rnn:488, get_default_rnn_state:491, load:515, __init__:640, forward:726, load:800, _build_impala:833, is_separate_critic:841, is_rnn:844, get_default_rnn_state:847, __init__:924, _build_critic:975, _build_actor:978, forward:981, is_separate_critic:987, load:990。
- 来源：upstream identical；依赖：os, rl_games.algos_torch, rl_games.algos_torch.d2rl, rl_games.algos_torch.sac_helper, rl_games.common, rl_games.common.layers.recurrent, rl_games.common.layers.value, torch, torch.nn。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：本系统审计/文件地图；测试：未检出直接import测试；不表示已验证。

### `rl_games/rl_games/algos_torch/players.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：rescale_actions:13, PpoPlayerContinuous:20, PpoPlayerDiscrete:115, SACPlayer:219, __init__:22, get_action:62, restore:96, reset:111, __init__:117, get_masked_action:145, get_action:176, restore:205, reset:215, __init__:221, restore:246, get_action:258, reset:268。
- 来源：upstream identical；依赖：gym, numpy, os, rl_games.algos_torch, rl_games.algos_torch.running_mean_std, rl_games.common.custom_utils, rl_games.common.player, rl_games.common.tr_helpers, torch。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：本系统审计/文件地图；测试：未检出直接import测试；不表示已验证。

### `rl_games/rl_games/torch_runner.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：_restore:19, _load_checkpoint_weights:37, _override_sigma:46, Runner:57, __init__:59, reset:78, load_config:81, load:130, set_vec_env:135, run_train:138, run_play:145, create_player:152, reset:155, run:158。
- 来源：upstream identical；依赖：copy, numpy, os, random, rl_games.algos_torch, rl_games.common, rl_games.common.algo_observer, time, torch。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：本系统审计/文件地图；测试：未检出直接import测试；不表示已验证。

### `tests/test_action_history.py`

Causal execution feedback, native configuration and version admission tests.

- 定义（函数/类:行）：transition:19, test_offline_history_is_past_execution_across_chunks_and_label_mutation:27, test_history_actual_controller_sequence_gradients_reset_and_version:53, test_execution_acknowledgment_rejects_missing_mixed_duplicate_and_future:101, test_native_contract_rejects_control_dr_reset_timeout_changes:128。
- 来源：local addition；依赖：copy, h5py, pytest, test_causal_memory, test_perception_v2, torch, types, visuotactile.action_history, visuotactile.contracts, visuotactile.dataset, visuotactile.models.nominal, visuotactile.observations, visuotactile.rollout, visuotactile.train_dp。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：_save, torch.save。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_camera_rigs.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：test_renderer_row_matrix_ros_roundtrip:6, test_mount_looks_along_declared_parent_optical_direction:16, test_close_rig_moves_only_high_camera_preserves_real_wrist:24。
- 来源：local addition；依赖：numpy, scipy.spatial.transform, visuotactile.camera_rigs。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_causal_memory.py`

Fixed-parameter causal tests; fixtures never used for training.

- 定义（函数/类:行）：sequence:25, test_episode_sequence_chunk_online_and_detach:32, test_controller_updates_between_replans_and_only_episode_reset:60, test_tactile_chunk_boundaries_preserve_derivatives_and_events:85, test_shadow_state_matches_direct_step_carry_and_reset:107, test_masked_depth_and_invalid_camera_do_not_influence_or_receive_gradient:128, test_action_delay_trace_rng_and_formulas:145, fixture_file:174, test_hdf5_step_pickle_and_sequential_training_roundtrip:187, test_legacy_memory_contract_and_checkpoint_state:218。
- 来源：local addition；依赖：copy, dataclasses, importlib.util, numpy, pathlib, pickle, pytest, test_perception_v2, torch, types, visuotactile.correction, visuotactile.dataset, visuotactile.history, visuotactile.models.correction, visuotactile.models.nominal, visuotactile.models.tactile, visuotactile.observations, visuotactile.recording, visuotactile.rollout。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：EpisodeWriter, controller.plan_generation_step.tolist, reader.steps, steps.append, torch.save, writer.append, writer.close。
- 文档：CHANGELOG.md, RUNBOOK.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_collection_resources.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：test_inventory_counts_other_batch_and_temporary_never_links:7, test_irregular_samples_are_weighted_and_missing_not_zero:16。
- 来源：local addition；依赖：os, pathlib, pytest, visuotactile.collection_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(outside / 'large').write_bytes, outside.mkdir, p.parent.mkdir, p.write_bytes。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_command_training.py`

Control, causal batching and reload regressions; fixtures are not demonstrations.

- 定义（函数/类:行）：chunk:18, sliced:28, compare_state:33, test_causal_packing_swap_padding_and_row_reset:41, test_tactile_reference_with_initial_missing_and_padded_tail:76, test_checkpoint_actual_controller_online_no_plan_rng_and_causal_state:89, test_command_history_fast_path_matches_retained_images:119, test_evaluation_requires_actual_best_identity:134, test_incomplete_evaluation_never_publishes_completion_marker:148, test_activation_recomputation_keeps_features_and_parameter_gradients:158, test_future_tactile_rejected_and_b0_rejects_extra_fields:170, test_resume_checkpoint_restores_optimizer_rng_and_next_update:180, test_sampled_metrics_replan_phase_and_episode_tail_masks:209, step:188, Oracle:223, unchanged:67, eval:225, predict_from_condition:226, select:57。
- 来源：local addition；依赖：copy, dataclasses, json, numpy, pytest, test_perception_v2, torch, types, visuotactile.command_metrics, visuotactile.evaluate_baselines, visuotactile.history, visuotactile.models.nominal, visuotactile.models.tactile, visuotactile.observations, visuotactile.rollout, visuotactile.train_dp。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(tmp_path / 'report.json').write_text, _save, json.dumps, o.step, step, torch.save。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_compact.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：fixture:13, test_lossless_storage_and_terminal_boundary:36, test_refuse_noninvertible_rgb_and_misaligned_frames:51, test_optional_viewing_videos_keep_lossless_hdf5:59, test_fresh_and_resume_share_result_identity_gate:72, test_compact_repeated_reversed_and_empty_indices:87。
- 来源：local addition；依赖：h5py, json, numpy, pathlib, pytest, unittest.mock, visuotactile.compact, visuotactile.recording, visuotactile.run_rl_trials。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(path / 'FAILED.txt').write_text, (path / 'result.json').write_text, (tmp_path / 'trial.status.json').write_text, json.dumps, path.mkdir, {'image': images, 'depth_m': depth, 'depth_valid': valid, 'camera_time_s': np.arange(4)[:, None].repeat(views, 1) / 60, 'camera_frame': np.arange(4)[:, None].repeat(views, 1), 'episode_id': np.zeros(4, dtype='i8'), 'step_id': np.arange(4)}.items。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_insertion_dataset.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：test_box_geometry_rotation_touch_and_penetration:13, manifest_fixture:23, test_manifest_checksum_missing_files_and_group_split:36, test_manifest_success_filter_and_self_plan_guard:48, test_frozen_design_and_storage_accounting:65。
- 来源：local addition；依赖：json, numpy, pytest, unittest.mock, visuotactile.compact, visuotactile.initial_geometry, visuotactile.manifest, visuotactile.run_insertion_dataset, visuotactile.storage_audit。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(tmp_path / 'a/data.h5').write_bytes, directory.mkdir, file.write_bytes, json.dumps, path.write_text。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_long_horizon.py`

Causality, boundary coverage and native representation tests for candidate2.

- 定义（函数/类:行）：record:16, test_roundtrip_unclipped_arm_limits_and_nonidentity_hand_order:33, test_current_and_future_commands_never_enter_conditions:47, test_short_episode_and_two_second_boundaries:61, test_padding_does_not_change_masked_history:78, test_sample_prediction_preserves_unclipped_reference_at_terminal_noise:88, test_pilot_admission_rejects_angle_only_improvement:103, test_adapter_uses_unchanged_native_delay_and_smoothing:125, test_whole_prefix_chunk_and_online_memory_match:165。
- 来源：local addition；依赖：copy, importlib.util, json, pathlib, pytest, torch, types, visuotactile.action_reference, visuotactile.long_data, visuotactile.models.long_horizon, visuotactile.run_long。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(output / 'config.json').write_text, (output / 'fit.json').write_text, (output / 'report.json').write_text, (tmp_path / 'cache_index.json').write_text, (tmp_path / 'causal_acceptance.json').write_text, json.dumps, output.mkdir, scheduler.set_timesteps, scheduler.step。
- 文档：CHANGELOG.md, LONG_HORIZON_20261030.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_long_resources.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：sample:8, test_shared_feedback_does_not_adjust_twice_for_same_sample:12, test_feedback_targets_forty_and_reacts_to_increased_load:24, test_missing_short_window_never_becomes_zero_utilization:42。
- 来源：local addition；依赖：math, pytest, visuotactile.long_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_loss_study.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：test_group_weight_gradient_and_tail_mask:8, test_per_finger_mapping_respects_unequal_counts:26, test_resource_violation_uses_whole_card_and_memory:37。
- 来源：local addition；依赖：pytest, torch, visuotactile.study_metrics, visuotactile.study_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_network_audit.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：test_force_interventions_preserve_nonforce_fields_and_past_support:8, test_parameter_audit_rejects_partial_optimizer_scope:25, test_first_deviation_uses_time_axis_and_tolerance:32。
- 来源：local addition；依赖：pytest, torch, visuotactile.audit_network, visuotactile.audit_training_initials。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_perception_v2.py`

Boundary/numerical checks, not synthetic learning or policy evaluation.

- 定义（函数/类:行）：packet:15, test_v2_full_interfaces_gradients_and_untrained_roundtrip:33, test_v2_nominal_private_plan_and_no_tactile_leak:66, test_goal_sources_masks_and_camera_time:88, test_force_frame_rotation_and_reset_mask:104, test_v2_modalities_and_view_identity_reach_encoder:119, test_legacy_config_explicit_interpretation:134, test_v2_correction_async_reset_preserves_other_environment:140。
- 来源：local addition；依赖：copy, dataclasses, pytest, torch, visuotactile.correction, visuotactile.kinematics, visuotactile.models.correction, visuotactile.models.nominal, visuotactile.models.tactile, visuotactile.observations, visuotactile.rollout。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：torch.save。
- 文档：CHANGELOG.md, RUNBOOK.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_persistent_collection.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：rows:8, test_stream_exact_fields_and_independent_episodes:26, test_stream_rejects_misalignment:39, test_stream_graceful_censor_preserves_real_last_post_frame:47, test_mount_env_instance_is_not_a_new_calibration:59。
- 来源：local addition；依赖：copy, hashlib, json, numpy, pytest, visuotactile.camera_rigs, visuotactile.contracts, visuotactile.recording。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_residual.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：fixture_model:10, inputs:16, test_zero_residual_bypass_and_reachable_two:29, test_independent_freeze_and_full_prefix_equivalence:45, test_force_mask_removes_force_derived_features_but_retains_fk:72, test_v3_is_not_silently_connected:88, test_untrained_checkpoint_cannot_execute:94, test_acknowledged_mixed_feedback_is_private_to_correction:105, test_diagnostic_continuation_never_repeats_a_started_allocation:128, test_pose_comparison_respects_quaternion_double_cover_and_units:148, test_viewing_allocation_is_separate_and_never_continued:160, test_video_status_does_not_modify_policy_images:170, test_resource_overlimit_intervals_preserve_unknown_gap:189, test_recorder_resource_window_excludes_monitor_tail_and_keeps_missing:204。
- 来源：local addition；依赖：av, copy, json, numpy, pytest, torch, visuotactile, visuotactile.action_history, visuotactile.evaluate_baselines, visuotactile.evaluate_residual, visuotactile.history, visuotactile.models.nominal, visuotactile.models.residual, visuotactile.residual_controller, visuotactile.residual_data, visuotactile.residual_report, visuotactile.viewing_report。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(previous / '02_failed_switch_error.json').write_text, (previous / 'status.json').write_text, (previous / 'trials.json').write_text, json.dumps, path.write_text, previous.mkdir, torch.save。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_rl_trials.py`

Experiment denominator/gating checks; no Isaac Sim or training required.

- 定义（函数/类:行）：native_wrench:16, fake_env:22, test_native_scale_changes_only_force_with_same_rng_and_gates:38, test_censor_errors_remain_unknown_not_robot_failure:52, test_alternation_and_impulse_units:72, test_startup_retry_retains_failed_attempt_and_rejects_partial_data:82, setter:24。
- 来源：local addition；依赖：importlib.util, json, numpy, pathlib, pytest, torch, types, visuotactile.analyze_rl_trials, visuotactile.run_rl_trials。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(tmp_path / (name + '.status.json')).write_text, (tmp_path / name).mkdir, json.dumps。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_success_batch.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：test_rejection_requires_explicit_diagnostic_and_no_started_trajectory:5, _recovery_fixture:15, test_recovery_deduplicates_and_respects_target:39。
- 来源：local addition；依赖：json, pathlib, visuotactile.collect_success_batch, visuotactile.finalize_collection。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(root / 'manifest.json').write_text, (root / 'reference_metadata.json').write_text, (run / 'progress.json').write_text, (source / 'data.h5').write_bytes, (tmp_path / 'FAILED.txt').write_text, (tmp_path / 'metadata.json').write_text, (tmp_path / 'trajectories.h5').write_bytes, json.dumps, root.mkdir, run.mkdir, source.mkdir。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_training_resources.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：row:4, test_average_gate_ignores_isolated_peak_but_checks_both_means:9, test_average_gate_requires_telemetry_and_keeps_memory_reserve:20, test_session_waits_measure_sleep_and_survive_budget_exit:29, test_stream_video_records_each_view_and_frame_without_raw_frame_cache:55, delayed_sleep:38。
- 来源：local addition；依赖：av, json, numpy, pytest, torch, visuotactile, visuotactile.evaluate_baselines, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(tmp_path / 'session.json').write_text, gate.save_waits, json.dumps。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_transformer_residual.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：record_fixture:14, test_native_memory_matches_streaming_and_future_is_not_input:26, test_force_mask_before_memory_and_no_contact_exact_bypass:52, test_raw_delta_loss_retains_gradient_at_execution_clip:67, test_plan_and_target_alignment_at_six_and_twelve_ticks:82, test_private_proposals_equal_actual_without_correction_and_ignore_mixed:96, test_tactile_rejects_future_sensor_sample:122, test_authorized_mixed_feedback_reaches_nominal_after_actual_ack:130。
- 来源：local addition；依赖：copy, pytest, torch, types, visuotactile.action_history, visuotactile.models.transformer_residual, visuotactile.private_history, visuotactile.transformer_controller, visuotactile.transformer_data。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `tests/test_visuotactile.py`

Contract tests. Synthetic sentinels test boundaries; they are not training data.

- 定义（函数/类:行）：test_async_history_matches_offline_indices:13, test_duplicate_observation_is_rejected:33, test_native_arm_hand_semantics_and_canonical_mapping:41, test_nominal_rejects_tactile_and_privileged_keys:61, test_correction_is_independent_and_cannot_execute_untrained:71, test_checkpoint_load_preserves_simulator_rng:91, test_correction_real_time_alignment_and_async_reset:102, test_nominal_inference_is_invariant_to_tactile_and_diagnostics:126。
- 来源：local addition；依赖：dataclasses, importlib.util, numpy, pathlib, pytest, torch, types, visuotactile.correction, visuotactile.history, visuotactile.models.correction, visuotactile.models.nominal, visuotactile.rollout。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：torch.save。
- 文档：CHANGELOG.md, RUNBOOK.md, GIT_WORKFLOW.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/__init__.py`

Visual–tactile research extensions; importing this package never boots Isaac.

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：local addition；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/action_history.py`

Past native execution feedback for command baselines, never private plans.

- 定义（函数/类:行）：history_contract:5, execution_context:17, ExecutionHistory:36, __init__:39, observe:43, context:62。
- 来源：local addition；依赖：torch。
- 静态调用者：tests/test_action_history.py, tests/test_residual.py, tests/test_transformer_residual.py, visuotactile/dataset.py, visuotactile/long_controller.py, visuotactile/private_history.py, visuotactile/probe_action_history.py, visuotactile/residual_controller.py, visuotactile/rollout.py, visuotactile/train_dp.py, visuotactile/transformer_controller.py。
- I/O或状态操作候选：packet['step_id'].cpu, step_ids.cpu, step_ids.cpu().clone, transition['step_id'].cpu。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_action_history.py, tests/test_transformer_residual.py, tests/test_residual.py。

### `visuotactile/action_reference.py`

Reversible research adapter between native commands and joint references.

- 定义（函数/类:行）：_validate:12, command_to_reference:23, reference_to_command:35, roundtrip_metrics:48。
- 来源：local addition；依赖：__future__, torch。
- 静态调用者：tests/test_long_horizon.py, visuotactile/audit_system.py, visuotactile/long_controller.py, visuotactile/long_data.py, visuotactile/study_audit.py, visuotactile/study_metrics.py, visuotactile/train_long.py, visuotactile/transformer_data.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：tests/test_long_horizon.py。

### `visuotactile/analyze_insertion_dataset.py`

Descriptive signed-offset results, storage, force exposure and <=4 previews.

- 定义（函数/类:行）：wrench_events:23, preview:38, analyze:60, main:222, frames:44。
- 来源：local addition；依赖：PIL, argparse, collections, datetime, h5py, json, matplotlib, matplotlib.pyplot, numpy, pathlib, visuotactile.analyze_rl_trials, visuotactile.compact, visuotactile.initial_geometry, visuotactile.manifest, visuotactile.run_insertion_dataset, visuotactile.trial_utils。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(output / 'report.md').write_text, fig.savefig, json.dumps, output.mkdir, preview_dir.mkdir, write_json, write_video。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/analyze_rl_trials.py`

Audit and compare bounded, single-environment RL trials, without training.

- 定义（函数/类:行）：episode_outcome:21, wilson:27, maxdiff:36, exposure:40, load_trial:52, config_differences:251, compare_pair:260, aggregate:313, condition_metrics:333, plot_pair:372, analyze:399, main:466, read:66, check:75, quantiles:347。
- 来源：local addition；依赖：__future__, argparse, h5py, json, matplotlib, matplotlib.pyplot, numpy, pathlib, scipy.spatial.transform, visuotactile.compact, visuotactile.trial_utils。
- 静态调用者：tests/test_rl_trials.py, visuotactile/analyze_insertion_dataset.py, visuotactile/evaluate_baselines.py, visuotactile/run_insertion_dataset.py, visuotactile/run_rl_trials.py。
- I/O或状态操作候选：(output / 'report.md').write_text, fig.savefig, json.dumps, output.mkdir, write_json。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_rl_trials.py。

### `visuotactile/audit_baselines.py`

Bounded read-only diagnostics; reports never replace formal evaluation.

- 定义（函数/类:行）：DiagnosticBudget:23, rms:57, trace_statistics:62, audit_traces:90, audit_development:132, sample_frozen:180, command_errors:190, audit_offline:203, main:268, __init__:26, block:32, summary:50。
- 来源：local addition；依赖：argparse, contextlib, diffusers, importlib.metadata, json, numpy, pathlib, shutil, subprocess, time, torch, visuotactile.compact, visuotactile.contracts, visuotactile.manifest, visuotactile.rollout, visuotactile.train_dp, visuotactile.training_resources。
- 静态调用者：visuotactile/probe_action_history.py。
- I/O或状态操作候选：args.output.mkdir, json.dumps, scheduler.alphas_cumprod[scheduler.timesteps].tolist, scheduler.set_timesteps, scheduler.step, scheduler.timesteps.tolist, steps.extend, subprocess.check_output, subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/audit_distribution.py`

Read-only fixed400 diversity census and outcome-independent diagnostic selection.

- 定义（函数/类:行）：summary:15, census:21, read:35。
- 来源：local addition；依赖：argparse, csv, h5py, hashlib, json, numpy, pathlib, visuotactile.long_data, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：csv.DictWriter, json.dumps, output.mkdir, writer.writeheader, writer.writerows。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/audit_loss_study.py`

Read-only final verification and small sanitized result export for this study.

- 定义（函数/类:行）：require:14, compact_metrics:19, precontact_difference:24, audit_traces:37, main:127。
- 来源：local addition；依赖：argparse, av, hashlib, json, numpy, pathlib, shutil, visuotactile.compact, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：args.output.parent.mkdir, json.dumps。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/audit_network.py`

Actual-checkpoint CPU hooks, parameter census and paired force interventions.

- 定义（函数/类:行）：describe:26, Hooks:38, parameter_rows:74, write_csv:97, intervene:103, residual_audit:120, main:214, __init__:39, finish:53, capture:44。
- 来源：local addition；依赖：argparse, csv, gc, h5py, json, pathlib, torch, torch.nn.utils.rnn, visuotactile.compact, visuotactile.long_data, visuotactile.manifest, visuotactile.models.long_horizon, visuotactile.models.residual, visuotactile.models.transformer_residual, visuotactile.residual_data, visuotactile.study_metrics, visuotactile.train_residual, visuotactile.train_transformer_residual, visuotactile.training_resources, visuotactile.transformer_data。
- 静态调用者：tests/test_network_audit.py。
- I/O或状态操作候选：LongHorizonPolicy(LongHorizonConfig(**saved['config'])).eval, ResidualCorrection(base.model.encoder, base.model.cfg, saved['scales']).eval, TransformerResidual(TransformerResidualConfig(**saved['config'])).eval, args.output.mkdir, csv.DictWriter, handle.remove, json.dumps, write_csv, writer.writeheader, writer.writerows。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_network_audit.py。

### `visuotactile/audit_system.py`

CPU-only control, dataset, RNG provenance and repository inventory audit.

- 定义（函数/类:行）：digest:23, sanitized:27, source_audit:37, inventory:50, dataset_audit:134, rng_audit:225, historical_sources:257, evaluation_audit:290, parameter_contracts:325, main:354, read:151。
- 来源：local addition；依赖：argparse, ast, collections, csv, h5py, hashlib, json, numpy, pathlib, subprocess, torch, visuotactile.action_reference, visuotactile.long_data, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(delayed - command[source_steps]).abs, (delayed - command[source_steps]).abs().max, (source_steps <= torch.arange(length)).all, (source_steps >= 0).all, Path('research_docs/PROJECT_FILE_MAP.md').write_text, args.output.mkdir, csv.writer, json.dumps, json.dumps(code, sort_keys=True).encode, json.dumps(configuration, sort_keys=True).encode, read('transition/control_trace/selected_command_step').long, saved['model'].items, subprocess.check_output, subprocess.check_output(['git', 'ls-files', 'isaacsimenvs'], text=True).splitlines, subprocess.check_output(['git', 'ls-files'], text=True).splitlines, subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip, subprocess.run, writer.writerow。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/audit_training_initials.py`

Read-only analysis of completed approximate-initial-state diagnostics.

- 定义（函数/类:行）：first_difference:19, write_csv:27, audit:34。
- 来源：local addition；依赖：argparse, csv, h5py, json, matplotlib, matplotlib.pyplot, numpy, pathlib, subprocess, visuotactile.compact, visuotactile.long_data, visuotactile.training_resources。
- 静态调用者：tests/test_network_audit.py。
- I/O或状态操作候选：csv.DictWriter, figure.savefig, frame_figure.savefig, json.dumps, np.linalg.norm(trace['force'][step], axis=-1).max, np.savez_compressed, subprocess.check_output, subprocess.check_output(['git', 'diff', '--name-only', '--', *protected], text=True).splitlines, subprocess.check_output(['git', 'ls-files', 'isaacsimenvs'], text=True).splitlines, trial['restoration']['saved_observable_errors'].values, write_csv, writer.writeheader, writer.writerows。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_network_audit.py。

### `visuotactile/camera_diagnostics.py`

Render-only diagnostic after a rollout. No physics steps or task observation calls.

- 定义（函数/类:行）：capture_static:11。
- 来源：local addition；依赖：carb, json, numpy, omni.usd, os, pxr, subprocess, time, torch, visuotactile.camera_rigs, xml.etree.ElementTree。
- 静态调用者：visuotactile/collect.py。
- I/O或状态操作候选：(output / 'camera_diagnostics.json').write_text, json.dumps, np.savez_compressed, subprocess.check_output, subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/camera_review.py`

Offline camera evidence from committed raw frames; no simulation or training.

- 定义（函数/类:行）：data_path:19, compare:24, review:51, clearance:94, aperture_diagnostics:147, main:186, visit:30, frames:59。
- 来源：local addition；依赖：PIL, argparse, json, numpy, pathlib, scipy.spatial.transform, torch, trimesh, visuotactile.compact, visuotactile.initial_geometry, visuotactile.kinematics, visuotactile.trial_utils, xml.etree.ElementTree。
- 静态调用者：visuotactile/review_delivery.py。
- I/O或状态操作候选：Image.fromarray((np.clip(depth / 1.5, 0, 1) * 255).astype('u1')).save, Image.fromarray((valid * 255).astype('u1')).save, Image.fromarray(rgb[j]).save, json.dumps, np.save, out.mkdir, sheet.save, write_json, write_video。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/camera_rigs.py`

Explicit observation-only camera layouts. No object state or task stages.

- 定义（函数/类:行）：optical_quat:21, rig_contract:29, renderer_pose_ros:46。
- 来源：local addition；依赖：numpy, scipy.spatial.transform。
- 静态调用者：tests/test_camera_rigs.py, tests/test_persistent_collection.py, visuotactile/camera_diagnostics.py, visuotactile/collect.py, visuotactile/sim/environment.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, CAMERA_VALIDATION.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md；测试：tests/test_camera_rigs.py, tests/test_persistent_collection.py。

### `visuotactile/check_camera_interfaces.py`

Bounded real three-camera causal replay and Jacobian probe; zero updates.

- 定义（函数/类:行）：main:16。
- 来源：local addition；依赖：argparse, dataclasses, pathlib, torch, visuotactile.check_dataset, visuotactile.compact, visuotactile.correction, visuotactile.models.correction, visuotactile.models.nominal, visuotactile.rollout, visuotactile.trial_utils。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：a.output.mkdir, torch.save, write_json。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/check_dataset.py`

Real data zero-update checks; reads expert labels, never creates an optimizer.

- 定义（函数/类:行）：fingerprint:13, check:20, check_manifest:54, main:95, sequence:66, encode:69, encode:33。
- 来源：local addition；依赖：argparse, hashlib, json, pathlib, torch, visuotactile.dataset, visuotactile.manifest, visuotactile.models.nominal, visuotactile.trial_utils。
- 静态调用者：visuotactile/check_camera_interfaces.py。
- I/O或状态操作候选：NominalDiffusionPolicy(PolicyConfig(inference_steps=1, views=len(first.metadata['camera_ids']))).eval, NominalDiffusionPolicy(PolicyConfig(inference_steps=1, views=len(new.metadata['camera_ids']))).eval, json.dumps, write_json。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/cleanup_long.py`

Explicit one-session cleanup of invalid caches and redundant failed videos.

- 定义（函数/类:行）：main:9。
- 来源：local addition；依赖：json, pathlib, visuotactile.compact, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：json.dumps, path.unlink。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/collect.py`

Bounded CPU collection with original teacher/control; see research_docs/RUNBOOK.

- 定义（函数/类:行）：main:11, save_frame:143。
- 来源：local addition；依赖：PIL, argparse, carb, dataclasses, evaluation.eval_isaacsim, hashlib, isaaclab.app, isaacsimenvs, isaacsimenvs.utils.rlgames_utils, json, numpy, os, pathlib, rl_games.torch_runner, time, torch, traceback, visuotactile.camera_diagnostics, visuotactile.camera_rigs, visuotactile.collect_success_batch, visuotactile.collection_resources, visuotactile.contracts, visuotactile.correction, visuotactile.initial_geometry, visuotactile.models.nominal, visuotactile.observations, visuotactile.recording, visuotactile.rollout, visuotactile.sim.environment, visuotactile.trial_utils。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(args.output / 'FAILED.txt').write_text, (args.output / 'metadata.json').write_text, (args.output / 'result.json').write_text, EpisodeWriter, Image.fromarray((im[3].clip(0, 1) * 255).astype(np.uint8)).save, Image.fromarray((im[3].clip(0, 2) / 2 * 255).astype(np.uint8)).save, Image.fromarray((im[:3].transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)).save, Image.fromarray((valid * 255).astype(np.uint8)).save, args.output.mkdir, env.last_transition['contact_substeps_after_w_N'].norm, env.step, intent.plan_generation_step.clone, json.dumps, np.save, rng_before_step[None].expand, save_frame, writer.append, writer.close。
- 文档：CHANGELOG.md, NIGHT_COLLECTION.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md, ARCHITECTURE.md；测试：tests/test_success_batch.py, tests/test_collection_resources.py。

### `visuotactile/collect_success_batch.py`

Persistent released-expert capture, bounded audit pipeline and resumable manifest.

- 定义（函数/类:行）：audit_worker_init:26, classify_pre_action_rejection:33, episode_metadata:42, collect_persistent:68, process_alive:275, recover_pending:282, main:357, identity:99, publish:101, consume_resets:122, write_batch:133, reap_writes:144, reap_audits:151, discard_recovered_payload:299, close_partial:253。
- 来源：local addition；依赖：argparse, collections, concurrent.futures, copy, datetime, fcntl, json, multiprocessing, numpy, os, pathlib, shutil, subprocess, sys, threadpoolctl, time, torch, visuotactile.collection_resources, visuotactile.compact, visuotactile.contracts, visuotactile.finalize_collection, visuotactile.initialization_report, visuotactile.recording, visuotactile.run_insertion_dataset, visuotactile.trial_utils。
- 静态调用者：tests/test_success_batch.py, visuotactile/collect.py。
- I/O或状态操作候选：(run / 'STOP').unlink, dest.parent.mkdir, env.step, json.dumps, reap_writes, rng_before_step[None].expand, run.mkdir, subprocess.Popen, tmp.mkdir, write_json, writer.shutdown, writer.submit, writer.submit(close_partial).result, writes.append, writes.popleft, writes.popleft().result, writes[0].done。
- 文档：CHANGELOG.md, NIGHT_COLLECTION.md, CAMERA_VALIDATION.md, RUNBOOK.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md；测试：tests/test_success_batch.py。

### `visuotactile/collection_resources.py`

Wall-clock supervision of ONLY owned children; no simulation-time changes.

- 定义（函数/类:行）：output_inventory:19, time_mean:38, gpu_sample:54, Supervisor:63, main:192, __init__:70, sample:81, check:127, wait:136, run:141。
- 来源：local addition；依赖：argparse, json, os, pathlib, shutil, signal, stat, subprocess, time, visuotactile.trial_utils。
- 静态调用者：tests/test_collection_resources.py, visuotactile/collect.py, visuotactile/collect_success_batch.py。
- I/O或状态操作候选：a.output.mkdir, f.write, json.dumps, self.root.mkdir, subprocess.Popen, subprocess.check_output, subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=3).splitlines, subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=3).splitlines()[0].split, write_json。
- 文档：CHANGELOG.md, NIGHT_COLLECTION.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md；测试：tests/test_collection_resources.py。

### `visuotactile/command_metrics.py`

Offline sampled-command error with one fixed DDIM draw per replan condition.

- 定义（函数/类:行）：sampled_command_metrics:7。
- 来源：local addition；依赖：time, torch, visuotactile.train_dp。
- 静态调用者：tests/test_command_training.py, visuotactile/train_dp.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_command_training.py。

### `visuotactile/compact.py`

Lossless storage v1: uint8 RGB, float32 depth, explicit post-frame references.

- 定义（函数/类:行）：sha256:20, rgb_u8:25, VirtualArray:32, GroupView:92, EpisodeGroups:123, EpisodeFile:136, write_video:161, convert_episode:192, __init__:33, __len__:41, __getitem__:44, __init__:94, __getitem__:97, __contains__:104, items:107, visititems:114, __init__:124, __iter__:126, __getitem__:128, items:130, __len__:132, __init__:137, __enter__:145, __exit__:147, __getitem__:149, get:157, visit:115, copy:220, compare:262。
- 来源：local addition；依赖：av, h5py, hashlib, json, numpy, os, pathlib。
- 静态调用者：tests/test_compact.py, tests/test_insertion_dataset.py, visuotactile/analyze_insertion_dataset.py, visuotactile/analyze_rl_trials.py, visuotactile/audit_baselines.py, visuotactile/audit_loss_study.py, visuotactile/audit_network.py, visuotactile/audit_training_initials.py, visuotactile/camera_review.py, visuotactile/check_camera_interfaces.py, visuotactile/cleanup_long.py, visuotactile/collect_success_batch.py, visuotactile/dataset.py, visuotactile/evaluate_baselines.py, visuotactile/evaluate_long.py, visuotactile/evaluate_residual.py, visuotactile/evaluate_study.py, visuotactile/evaluate_training_initials.py, visuotactile/finalize_collection.py, visuotactile/initialization_report.py, visuotactile/inspect_long.py, visuotactile/long_data.py, visuotactile/manifest.py, visuotactile/plot_network_audit.py, visuotactile/policy_census.py, visuotactile/probe_action_history.py, visuotactile/recording.py, visuotactile/residual_controller.py, visuotactile/residual_data.py, visuotactile/residual_report.py, visuotactile/review_delivery.py, visuotactile/run_insertion_dataset.py, visuotactile/run_long.py, visuotactile/run_loss_study.py, visuotactile/run_rl_trials.py, visuotactile/storage_audit.py, visuotactile/study_audit.py, visuotactile/train.py, visuotactile/train_dp.py, visuotactile/train_long.py, visuotactile/train_loss_study.py, visuotactile/train_residual.py, visuotactile/train_transformer_residual.py, visuotactile/transformer_controller.py, visuotactile/transformer_data.py, visuotactile/validate_command_training.py, visuotactile/viewing_report.py。
- I/O或状态操作候选：destination.mkdir, json.dumps, write_video。
- 文档：CHANGELOG.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md, BASELINE_TRAINING.md；测试：tests/test_compact.py, tests/test_insertion_dataset.py。

### `visuotactile/contracts.py`

Serializable sensor/execution contracts, independent of Isaac Lab imports.

- 定义（函数/类:行）：policy_input_contract:7, assert_policy_compatible:68, native_execution_contract:77, assert_execution_compatible:91, evaluation_contract:104, contract_hash:120。
- 来源：local addition；依赖：hashlib, json, numpy, re。
- 静态调用者：tests/test_action_history.py, tests/test_persistent_collection.py, visuotactile/audit_baselines.py, visuotactile/collect.py, visuotactile/collect_success_batch.py, visuotactile/evaluate.py, visuotactile/evaluate_baselines.py, visuotactile/evaluate_long.py, visuotactile/evaluate_residual.py, visuotactile/evaluate_study.py, visuotactile/finalize_collection.py, visuotactile/manifest.py, visuotactile/probe_correction.py, visuotactile/probe_perception.py, visuotactile/train_dp.py。
- I/O或状态操作候选：hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest, json.dumps, json.dumps(contract, sort_keys=True).encode。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：tests/test_action_history.py, tests/test_persistent_collection.py。

### `visuotactile/correction.py`

Causal correction context and NON-CONTROLLING real-data integration probe.

- 定义（函数/类:行）：remaining_nominal_chunk:15, CorrectionContext:26, CorrectionShadow:77, latest_correction_inputs:118, __init__:27, observe:39, commit:69, __init__:79, observe:94。
- 来源：local addition；依赖：math, time, torch, visuotactile.history, visuotactile.models.correction, visuotactile.observations。
- 静态调用者：tests/test_causal_memory.py, tests/test_perception_v2.py, tests/test_visuotactile.py, visuotactile/check_camera_interfaces.py, visuotactile/collect.py, visuotactile/dataset.py, visuotactile/probe_correction.py, visuotactile/probe_perception.py, visuotactile/residual_controller.py, visuotactile/residual_data.py。
- I/O或状态操作候选：feedback['contact_substeps_local_N'].reshape。
- 文档：CHANGELOG.md, PROJECT_GUIDE_ZH.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：tests/test_causal_memory.py, tests/test_visuotactile.py, tests/test_perception_v2.py。

### `visuotactile/dataset.py`

Strict nominal-input allowlist; future actions are supervision only.

- 定义（函数/类:行）：EpisodeDataset:11, CorrectionEpisodeDataset:81, EpisodeSequenceDataset:127, encode_episode_chunks:197, CorrectionSequenceDataset:219, replay_correction_chunks:247, __init__:12, __len__:55, __getitem__:58, proprio_statistics:74, __init__:87, __getitem__:94, __init__:135, chunks:142, __init__:226, chunks:232, read:99。
- 来源：local addition；依赖：json, numpy, torch, torch.utils.data, visuotactile.action_history, visuotactile.compact, visuotactile.correction, visuotactile.history, visuotactile.observations。
- 静态调用者：tests/test_action_history.py, tests/test_causal_memory.py, visuotactile/check_dataset.py, visuotactile/manifest.py, visuotactile/probe_history.py, visuotactile/probe_perception.py, visuotactile/train.py, visuotactile/train_dp.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：HISTORY_AND_TRACE.md, CHANGELOG.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：tests/test_action_history.py, tests/test_causal_memory.py。

### `visuotactile/evaluate.py`

Audit recorded trajectories and compare only identical recorded protocols.

- 定义（函数/类:行）：audit_trajectory:14, main:95。
- 来源：local addition；依赖：argparse, h5py, json, numpy, pathlib, visuotactile.contracts。
- 静态调用者：visuotactile/probe_perception.py。
- I/O或状态操作候选：args.output.parent.mkdir, json.dump, json.dumps。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：tests/test_training_resources.py, tests/test_command_training.py, tests/test_residual.py。

### `visuotactile/evaluate_baselines.py`

Paired B0/B1 command-controller evaluation on prespecified fresh legal resets.

- 定义（函数/类:行）：checkpoint_identity:16, publish_report:40, RolloutVideo:55, main:102, __init__:57, append:68, close:89, initial:212。
- 来源：local addition；依赖：PIL, argparse, av, evaluation.eval_isaacsim, isaaclab.app, isaacsimenvs, json, numpy, os, pathlib, shutil, subprocess, time, torch, traceback, visuotactile.analyze_rl_trials, visuotactile.compact, visuotactile.contracts, visuotactile.expert_reference, visuotactile.initial_geometry, visuotactile.observations, visuotactile.rollout, visuotactile.sim.environment, visuotactile.training_resources。
- 静态调用者：tests/test_command_training.py, tests/test_residual.py, tests/test_training_resources.py, visuotactile/evaluate_long.py, visuotactile/evaluate_residual.py, visuotactile/evaluate_study.py。
- I/O或状态操作候选：(args.output / 'error.txt').write_text, Image.fromarray((np.concatenate(im, axis=1).clip(0, 1) * 255).astype('uint8')).save, args.output.mkdir, env.step, json.dumps, np.savez_compressed, saved.get, subprocess.check_output, subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_training_resources.py, tests/test_command_training.py, tests/test_residual.py。

### `visuotactile/evaluate_long.py`

Six preregistered autonomous candidate diagnostics, unchanged native simulator.

- 定义（函数/类:行）：main:14, block:24。
- 来源：local addition；依赖：argparse, evaluation.eval_isaacsim, isaaclab.app, isaacsimenvs, json, numpy, os, pathlib, subprocess, time, torch, visuotactile.compact, visuotactile.contracts, visuotactile.evaluate_baselines, visuotactile.initial_geometry, visuotactile.long_controller, visuotactile.long_resources, visuotactile.models.long_horizon, visuotactile.observations, visuotactile.sim.environment, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：env.step, json.dumps, np.savez_compressed, output.mkdir, subprocess.check_output, subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/evaluate_residual.py`

Twelve prespecified diagnostic trials, autonomous and real expert roll-in.

- 定义（函数/类:行）：diagnostic_allocation:15, main:43。
- 来源：local addition；依赖：argparse, evaluation.eval_isaacsim, hashlib, isaaclab.app, isaacsimenvs, json, numpy, os, pathlib, subprocess, time, torch, traceback, visuotactile.compact, visuotactile.contracts, visuotactile.evaluate_baselines, visuotactile.expert_reference, visuotactile.initial_geometry, visuotactile.observations, visuotactile.residual_controller, visuotactile.residual_session, visuotactile.sim.environment, visuotactile.training_resources。
- 静态调用者：tests/test_residual.py。
- I/O或状态操作候选：(output / 'error.txt').write_text, env.step, errors[0].name.removesuffix, gate.save_waits, json.dumps, np.savez_compressed, output.mkdir, subprocess.check_output, subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_residual.py。

### `visuotactile/evaluate_study.py`

Preregistered native DP comparisons and matched local/autonomous residual trials.

- 定义（函数/类:行）：longest_streak:16, allocation:24, main:52, interrupted:57。
- 来源：local addition；依赖：argparse, evaluation.eval_isaacsim, hashlib, isaaclab.app, isaacsimenvs, json, numpy, os, pathlib, signal, subprocess, time, torch, visuotactile.compact, visuotactile.contracts, visuotactile.evaluate_baselines, visuotactile.expert_reference, visuotactile.initial_geometry, visuotactile.long_controller, visuotactile.models.long_horizon, visuotactile.models.residual, visuotactile.observations, visuotactile.sim.environment, visuotactile.study_resources, visuotactile.training_resources, visuotactile.transformer_controller。
- 静态调用者：visuotactile/transformer_data.py。
- I/O或状态操作候选：env.step, json.dumps, np.savez_compressed, output.mkdir, subprocess.check_output, subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/evaluate_training_initials.py`

Bounded native diagnostics from explicitly approximate saved demonstration states.

- 定义（函数/类:行）：initialize:15, restore_observable_state:29, evaluate:88, main:229, interrupted:259。
- 来源：local addition；依赖：argparse, evaluation.eval_isaacsim, h5py, isaaclab.app, isaacsimenvs, json, numpy, os, pathlib, signal, subprocess, sys, time, torch, visuotactile.compact, visuotactile.long_controller, visuotactile.long_data, visuotactile.models.long_horizon, visuotactile.models.residual, visuotactile.observations, visuotactile.sim.environment, visuotactile.study_resources, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：body.write_root_pose_to_sim, body.write_root_velocity_to_sim, env.robot.write_joint_state_to_sim, env.scene.write_data_to_sim, env.step, group['transition/diagnostics_after/object_pose_w'][step].copy, json.dumps, np.savez_compressed, restoration['saved_observable_errors'].values, root.mkdir, saved_input.create_dataset, subprocess.Popen, subprocess.check_output, subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/expert_reference.py`

Released privileged expert, solely for explicit diagnostic reference runs.

- 定义（函数/类:行）：ExpertReference:6, __init__:7, act:38, observe_execution:48。
- 来源：local addition；依赖：evaluation.eval_isaacsim, isaacsimenvs.utils.rlgames_utils, rl_games.torch_runner, torch, types。
- 静态调用者：visuotactile/evaluate_baselines.py, visuotactile/evaluate_residual.py, visuotactile/evaluate_study.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/finalize_collection.py`

Audited conversion worker; supervisor monitors its complete wall time.

- 定义（函数/类:行）：finalize_stream:11, finalize:44, main:65。
- 来源：local addition；依赖：argparse, json, numpy, pathlib, visuotactile.compact, visuotactile.contracts, visuotactile.recording, visuotactile.run_rl_trials, visuotactile.trial_utils。
- 静态调用者：tests/test_success_batch.py, visuotactile/collect_success_batch.py。
- I/O或状态操作候选：write_json。
- 文档：CHANGELOG.md, NIGHT_COLLECTION.md, PROJECT_FILE_MAP.md；测试：tests/test_success_batch.py。

### `visuotactile/history.py`

Shared causal indexing for dataset windows and independently resetting envs.

- 定义（函数/类:行）：history_indices:7, OnlineHistory:14, detach_state:45, detach_state_rows:54, reset_state:71, recurrent_scan:81, __init__:15, append:21, batch:32。
- 来源：local addition；依赖：collections, numpy, torch。
- 静态调用者：tests/test_causal_memory.py, tests/test_command_training.py, tests/test_residual.py, tests/test_visuotactile.py, visuotactile/correction.py, visuotactile/dataset.py, visuotactile/models/correction.py, visuotactile/models/nominal.py, visuotactile/models/residual.py, visuotactile/models/tactile_dp.py, visuotactile/models/vision.py, visuotactile/probe_history.py, visuotactile/residual_controller.py, visuotactile/residual_data.py, visuotactile/rollout.py, visuotactile/train_dp.py, visuotactile/train_residual.py, visuotactile/validate_command_training.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：tests/test_command_training.py, tests/test_causal_memory.py, tests/test_residual.py, tests/test_visuotactile.py。

### `visuotactile/initial_geometry.py`

Diagnostic OBB overlap of the original box colliders (no scene mutation).

- 定义（函数/类:行）：rotation:10, boxes:16, overlaps:28, inspect:40。
- 来源：local addition；依赖：numpy, xml.etree.ElementTree。
- 静态调用者：tests/test_insertion_dataset.py, visuotactile/analyze_insertion_dataset.py, visuotactile/camera_review.py, visuotactile/collect.py, visuotactile/evaluate_baselines.py, visuotactile/evaluate_long.py, visuotactile/evaluate_residual.py, visuotactile/evaluate_study.py, visuotactile/sim/environment.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md, INSERTION_DATASET.md；测试：tests/test_insertion_dataset.py。

### `visuotactile/initialization_report.py`

Small measured reset/wrench report from audited episodes, not inferred noise.

- 定义（函数/类:行）：manifest_statistics:12, report:58, main:89。
- 来源：local addition；依赖：argparse, collections, csv, json, numpy, pathlib, visuotactile.compact, visuotactile.run_rl_trials, visuotactile.trial_utils。
- 静态调用者：visuotactile/collect_success_batch.py。
- I/O或状态操作候选：csv.DictWriter, json.dumps, output.mkdir, write_json, writer.writeheader, writer.writerows。
- 文档：CHANGELOG.md, NIGHT_COLLECTION.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/inspect_long.py`

Bounded training-only sampling checks on a completed long-horizon pilot.

- 定义（函数/类:行）：main:15。
- 来源：local addition；依赖：argparse, json, pathlib, torch, visuotactile.compact, visuotactile.long_resources, visuotactile.models.long_horizon, visuotactile.train_long, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：LongHorizonPolicy(LongHorizonConfig(**saved['configuration']['model'])).to, json.dumps。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/kinematics.py`

Encoder-only URDF forward kinematics. No simulator body state is an input.

- 定义（函数/类:行）：rpy_matrix:12, EncoderKinematics:19, rotate_force_to_link:86, __init__:25, __call__:67。
- 来源：local addition；依赖：hashlib, math, torch, xml.etree.ElementTree。
- 静态调用者：tests/test_perception_v2.py, visuotactile/camera_review.py, visuotactile/probe_history.py, visuotactile/sim/environment.py。
- I/O或状态操作候选：needed.remove。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md, PERCEPTION.md, ARCHITECTURE.md；测试：tests/test_perception_v2.py。

### `visuotactile/long_controller.py`

Native-tick deployment for candidate2, with actual execution acknowledgments.

- 定义（函数/类:行）：LongController:10, __init__:11, act:38, observe_execution:86。
- 来源：local addition；依赖：torch, visuotactile.action_history, visuotactile.action_reference, visuotactile.long_data, visuotactile.models.long_horizon。
- 静态调用者：visuotactile/evaluate_long.py, visuotactile/evaluate_study.py, visuotactile/evaluate_training_initials.py, visuotactile/study_audit.py, visuotactile/transformer_controller.py, visuotactile/transformer_data.py, visuotactile/validate_long.py。
- I/O或状态操作候选：inputs_for(record, step, step + 1, self.model.cfg.tactile, torch.tensor([step])).items, packet['contact_substeps_local_N'][0].cpu, packet['contact_substeps_w_N'][0].cpu, saved.get。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/long_data.py`

Versioned frozen feature cache and identical causal offline/online windows.

- 定义（函数/类:行）：depth_grid:24, proprio_radians:35, tactile_values:41, shift_actions:51, window:62, inputs_for:70, supervision:107, load_record:115, cache:122, main:221, read:159。
- 来源：local addition；依赖：argparse, h5py, json, numpy, pathlib, time, torch, torch.nn, visuotactile.action_reference, visuotactile.compact, visuotactile.long_resources, visuotactile.models.long_horizon, visuotactile.training_resources。
- 静态调用者：tests/test_long_horizon.py, visuotactile/audit_distribution.py, visuotactile/audit_network.py, visuotactile/audit_system.py, visuotactile/audit_training_initials.py, visuotactile/evaluate_training_initials.py, visuotactile/long_controller.py, visuotactile/study_audit.py, visuotactile/study_metrics.py, visuotactile/train_long.py, visuotactile/train_loss_study.py, visuotactile/transformer_controller.py, visuotactile/transformer_data.py, visuotactile/validate_long.py。
- I/O或状态操作候选：directory.mkdir, torch.save。
- 文档：PROJECT_FILE_MAP.md；测试：tests/test_long_horizon.py。

### `visuotactile/long_resources.py`

One monitored GPU lease shared by all workers of a long-horizon session.

- 定义（函数/类:行）：update_pacing:15, Lease:48, main:132, __init__:49, memory:56, check:67, block:77。
- 来源：local addition；依赖：argparse, contextlib, fcntl, json, math, os, pathlib, shutil, time, torch, visuotactile.training_resources。
- 静态调用者：tests/test_long_resources.py, visuotactile/evaluate_long.py, visuotactile/inspect_long.py, visuotactile/long_data.py, visuotactile/study_resources.py, visuotactile/train_long.py。
- I/O或状态操作候选：json.dumps, stream.write。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_long_resources.py。

### `visuotactile/manifest.py`

Explicit multi-episode index over legacy or compact lossless HDF5 records.

- 定义（函数/类:行）：ManifestReader:10, ManifestSequenceDataset:43, assert_group_disjoint:104, __init__:11, reader:36, steps:39, __init__:53, chunks:92, require_self_plan_training:97。
- 来源：local addition；依赖：json, pathlib, visuotactile.compact, visuotactile.contracts, visuotactile.dataset, visuotactile.recording。
- 静态调用者：tests/test_insertion_dataset.py, visuotactile/analyze_insertion_dataset.py, visuotactile/audit_baselines.py, visuotactile/audit_network.py, visuotactile/check_dataset.py, visuotactile/probe_action_history.py, visuotactile/residual_data.py, visuotactile/run_insertion_dataset.py, visuotactile/train_dp.py, visuotactile/validate_command_training.py, visuotactile/validate_residual.py。
- I/O或状态操作候选：self.reader(episode_id).steps。
- 文档：CHANGELOG.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md；测试：tests/test_insertion_dataset.py。

### `visuotactile/models/__init__.py`

Pure PyTorch policies; no simulator imports.

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：local addition；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/models/correction.py`

Replaceable correction research interface. Untrained networks cannot execute.

- 定义（函数/类:行）：TactileEncoder:8, CorrectionPolicy:25, compose_action:76, __init__:10, forward:16, __init__:32, forward:49。
- 来源：local addition；依赖：torch, visuotactile.history, visuotactile.models.nominal, visuotactile.models.tactile。
- 静态调用者：tests/test_causal_memory.py, tests/test_perception_v2.py, tests/test_visuotactile.py, visuotactile/check_camera_interfaces.py, visuotactile/correction.py, visuotactile/models/residual.py, visuotactile/probe_history.py, visuotactile/probe_perception.py, visuotactile/rollout.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, PROJECT_GUIDE_ZH.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：tests/test_causal_memory.py, tests/test_visuotactile.py, tests/test_perception_v2.py。

### `visuotactile/models/long_horizon.py`

H64 reference policy with explicit recent histories and native-rate memory.

- 定义（函数/类:行）：LongHorizonConfig:16, FrozenDinoSpatial:42, MaskedHistory:64, LongHorizonPolicy:80, __post_init__:28, tactile:38, __init__:43, forward:52, __init__:65, forward:69, __init__:81, project_frames:105, encode:113, loss_from_condition:141, predict_from_condition:153。
- 来源：local addition；依赖：dataclasses, diffusers, torch, torch.nn, visuotactile.models.vendor.conditional_unet1d。
- 静态调用者：tests/test_long_horizon.py, visuotactile/audit_network.py, visuotactile/evaluate_long.py, visuotactile/evaluate_study.py, visuotactile/evaluate_training_initials.py, visuotactile/inspect_long.py, visuotactile/long_controller.py, visuotactile/long_data.py, visuotactile/study_audit.py, visuotactile/train_long.py, visuotactile/train_loss_study.py, visuotactile/transformer_data.py, visuotactile/validate_long.py。
- I/O或状态操作候选：self.sampler.set_timesteps, self.sampler.step。
- 文档：CHANGELOG.md, LONG_HORIZON_20261030.md, PROJECT_FILE_MAP.md；测试：tests/test_long_horizon.py。

### `visuotactile/models/nominal.py`

Tactile-free history-conditioned action diffusion, not preload nominalization.

- 定义（函数/类:行）：PolicyConfig:11, LegacyVisualHistoryEncoder:44, VisualHistoryEncoder:79, NominalDiffusionPolicy:88, __post_init__:28, from_dict:38, __init__:46, forward:60, __init__:89, loss:103, loss_from_condition:111, predict:121, predict_from_condition:126。
- 来源：local addition；依赖：dataclasses, diffusers, torch, visuotactile.history, visuotactile.models.vendor.conditional_unet1d, visuotactile.models.vision。
- 静态调用者：tests/test_action_history.py, tests/test_causal_memory.py, tests/test_command_training.py, tests/test_perception_v2.py, tests/test_residual.py, tests/test_visuotactile.py, visuotactile/check_camera_interfaces.py, visuotactile/check_dataset.py, visuotactile/collect.py, visuotactile/models/correction.py, visuotactile/models/tactile_dp.py, visuotactile/probe_action_history.py, visuotactile/probe_history.py, visuotactile/probe_perception.py, visuotactile/rollout.py, visuotactile/train.py, visuotactile/train_dp.py, visuotactile/validate_command_training.py。
- I/O或状态操作候选：self.sampler.set_timesteps, self.sampler.step。
- 文档：CHANGELOG.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md, BASELINE_TRAINING.md, ARCHITECTURE.md；测试：tests/test_action_history.py, tests/test_command_training.py, tests/test_causal_memory.py, tests/test_residual.py, tests/test_visuotactile.py, tests/test_perception_v2.py。

### `visuotactile/models/residual.py`

Version 1 contact-gated native-command residual with independent parameters.

- 定义（函数/类:行）：observed_contact:10, compose_residual:14, ResidualCorrection:26, residual_loss:69, __init__:29, forward_cached:43。
- 来源：local addition；依赖：copy, torch, visuotactile.history, visuotactile.models.correction。
- 静态调用者：tests/test_residual.py, visuotactile/audit_network.py, visuotactile/evaluate_study.py, visuotactile/evaluate_training_initials.py, visuotactile/policy_census.py, visuotactile/residual_controller.py, visuotactile/residual_data.py, visuotactile/train_residual.py, visuotactile/transformer_controller.py, visuotactile/transformer_data.py。
- I/O或状态操作候选：(per_step[contact] + regularization * penalty[contact]).mean。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：tests/test_residual.py。

### `visuotactile/models/tactile.py`

Five aggregate normal-force sensors; per-finger identity, kinematics and time.

- 定义（函数/类:行）：FingerTactileEncoder:10, __init__:11, forward:21。
- 来源：local addition；依赖：torch。
- 静态调用者：tests/test_causal_memory.py, tests/test_command_training.py, tests/test_perception_v2.py, visuotactile/models/correction.py, visuotactile/models/tactile_dp.py。
- I/O或状态操作候选：encoded_steps.append。
- 文档：CHANGELOG.md, RUNBOOK.md, PROJECT_GUIDE_ZH.md, GIT_WORKFLOW.md, PROJECT_FILE_MAP.md, PERCEPTION.md, ARCHITECTURE.md；测试：tests/test_command_training.py, tests/test_causal_memory.py, tests/test_perception_v2.py。

### `visuotactile/models/tactile_dp.py`

Tactile conditioned command diffusion baseline (B1).

- 定义（函数/类:行）：TactileCommandDiffusionPolicy:18, __init__:21, encode:41, loss_from_condition:95, predict_from_condition:107。
- 来源：local addition；依赖：diffusers, torch, visuotactile.history, visuotactile.models.nominal, visuotactile.models.tactile, visuotactile.models.vendor.conditional_unet1d。
- 静态调用者：visuotactile/rollout.py, visuotactile/train_dp.py。
- I/O或状态操作候选：self.sampler.set_timesteps, self.sampler.step。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md, BASELINE_TRAINING.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/models/transformer_residual.py`

Causal native-command correction with explicit current and remaining-plan tokens.

- 定义（函数/类:行）：TransformerResidualConfig:12, combine:27, residual_loss:35, TransformerResidual:44, __post_init__:22, __init__:45, force_features:70, forward:78。
- 来源：local addition；依赖：dataclasses, torch。
- 静态调用者：tests/test_transformer_residual.py, visuotactile/audit_network.py, visuotactile/train_transformer_residual.py, visuotactile/transformer_controller.py, visuotactile/transformer_data.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：tests/test_transformer_residual.py。

### `visuotactile/models/vendor/__init__.py`

MIT Diffusion Policy modules; see PROVENANCE.md.

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：local addition；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/models/vendor/conditional_unet1d.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：ConditionalResidualBlock1D:14, ConditionalUnet1D:69, __init__:15, forward:46, __init__:70, forward:173。
- 来源：local addition；依赖：einops, einops.layers.torch, logging, torch, torch.nn, typing, visuotactile.models.vendor.conv1d_components, visuotactile.models.vendor.positional_embedding。
- 静态调用者：visuotactile/models/long_horizon.py, visuotactile/models/nominal.py, visuotactile/models/tactile_dp.py。
- I/O或状态操作候选：self.diffusion_step_encoder, timesteps.expand, timesteps[None].to。
- 文档：RESEARCH.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/models/vendor/conv1d_components.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：Downsample1d:7, Upsample1d:15, Conv1dBlock:23, test:43, __init__:8, forward:12, __init__:16, forward:20, __init__:28, forward:39。
- 来源：local addition；依赖：torch, torch.nn, torch.nn.functional。
- 静态调用者：visuotactile/models/vendor/conditional_unet1d.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/models/vendor/positional_embedding.py`

包入口/测试或辅助定义

- 定义（函数/类:行）：SinusoidalPosEmb:5, __init__:6, forward:10。
- 来源：local addition；依赖：math, torch, torch.nn。
- 静态调用者：visuotactile/models/vendor/conditional_unet1d.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/models/vision.py`

Shared multi-view RGB/depth spatial encoder; see PERCEPTION.md for sources.

- 定义（函数/类:行）：RGBDSpatialEncoder:17, MultimodalHistoryEncoder:58, __init__:18, forward:30, __init__:60, encode_frames:75, forward:145。
- 来源：local addition；依赖：torch, torch.nn, torch.utils.checkpoint, torchvision.models, visuotactile.history, visuotactile.observations。
- 静态调用者：visuotactile/models/nominal.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/observations.py`

Versioned, explicit inference inputs and operator-provided task commands.

- 定义（函数/类:行）：nominal_keys:15, empty_plan_context:25, TaskCommand:32, __init__:38, packet:56, metadata:61。
- 来源：local addition；依赖：torch。
- 静态调用者：tests/test_action_history.py, tests/test_causal_memory.py, tests/test_command_training.py, tests/test_perception_v2.py, visuotactile/collect.py, visuotactile/correction.py, visuotactile/dataset.py, visuotactile/evaluate_baselines.py, visuotactile/evaluate_long.py, visuotactile/evaluate_residual.py, visuotactile/evaluate_study.py, visuotactile/evaluate_training_initials.py, visuotactile/models/vision.py, visuotactile/probe_correction.py, visuotactile/probe_perception.py, visuotactile/residual_controller.py, visuotactile/rollout.py, visuotactile/sim/environment.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：tests/test_action_history.py, tests/test_command_training.py, tests/test_causal_memory.py, tests/test_perception_v2.py。

### `visuotactile/plot_loss_study.py`

CPU-only figures from recorded metrics and bounded traces; no model loading.

- 定义（函数/类:行）：plot_dp_lift_event:15, plot_dp_closed_loop:66, plot_residual:107, plot:215, finish:142, finish:254。
- 来源：local addition；依赖：argparse, csv, json, matplotlib, matplotlib.pyplot, numpy, pathlib, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：csv.DictWriter, figure.savefig, output.mkdir, writer.writeheader, writer.writerows。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/plot_network_audit.py`

Reproducible architecture diagrams and plots from audited statistics.

- 定义（函数/类:行）：curve_sources:18, diagram:83, architecture:128, plots:199, write:19。
- 来源：local addition；依赖：argparse, csv, io, json, matplotlib, matplotlib.patches, matplotlib.pyplot, numpy, pathlib, visuotactile.compact, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(output / (name + '.svg')).write_text, csv.DictWriter, figure.savefig, write, writer.writeheader, writer.writerows。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/policy_census.py`

CPU/meta parameter census of local checkpoints and pinned public examples.

- 定义（函数/类:行）：count:18, describe:25, meta_factory:32, own_models:39, public_models:78, main:194, create:33, unavailable:84。
- 来源：local addition；依赖：ImplicitRDP.model.diffusion.transformer_for_diffusion, ImplicitRDP.model.vision.transformer_obs_encoder, ImplicitRDP.policy.diffusion_transformer_image_policy, argparse, diffusers, diffusion_policy.model.vision.model_getter, diffusion_policy.model.vision.multi_image_obs_encoder, diffusion_policy.policy, diffusion_policy.policy.diffusion_unet_hybrid_image_policy, diffusion_policy.policy.diffusion_unet_image_policy, gc, importlib.metadata, json, omegaconf, pathlib, reactive_diffusion_policy.model.vae.model, reactive_diffusion_policy.model.vision.model_getter, reactive_diffusion_policy.model.vision.multi_image_obs_encoder, reactive_diffusion_policy.policy, reactive_diffusion_policy.policy.latent_diffusion_unet_image_policy, sys, torch, torchvision, types, unittest.mock, visuotactile.compact, visuotactile.models.residual, visuotactile.rollout, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：args.output.parent.mkdir, json.dumps。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/private_history.py`

Versioned private base proposals with observed native delay timestamps.

- 定义（函数/类:行）：PrivateProposalHistory:9, private_action_rows:41, __init__:10, observe:14, observe_proposal:17。
- 来源：local addition；依赖：collections, torch, visuotactile.action_history。
- 静态调用者：tests/test_transformer_residual.py, visuotactile/transformer_data.py。
- I/O或状态操作候选：(times[:step] - selected_times[previous]).abs, packet['step_id'].cpu, packet['step_id'].cpu().clone。
- 文档：PROJECT_FILE_MAP.md；测试：tests/test_transformer_residual.py。

### `visuotactile/probe_action_history.py`

Bounded real-prefix fit and executed-history replay; never formal training.

- 定义（函数/类:行）：initialize_history_model:21, assess:40, verify_real_execution_context:58, main:94。
- 来源：local addition；依赖：argparse, copy, dataclasses, pathlib, time, torch, types, visuotactile.action_history, visuotactile.audit_baselines, visuotactile.compact, visuotactile.manifest, visuotactile.models.nominal, visuotactile.rollout, visuotactile.train_dp, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：_save, args.output.mkdir, optimizer.step, saved.update, target.mkdir, torch.save。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/probe_correction.py`

Replay real contact episodes through causal correction inputs on CPU.

- 定义（函数/类:行）：main:19。
- 来源：local addition；依赖：argparse, h5py, json, numpy, pathlib, torch, visuotactile.contracts, visuotactile.correction, visuotactile.observations, visuotactile.rollout。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：args.output.parent.mkdir, json.dump, json.dumps。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/probe_history.py`

Audit real trace/memory at fixed weights; no optimizer and no simulation.

- 定义（函数/类:行）：audit_trace:25, audit_memory:139, audit_gradients:177, main:246, encode:149, replay:163。
- 来源：local addition；依赖：argparse, h5py, io, json, numpy, pathlib, pickle, torch, traceback, visuotactile.dataset, visuotactile.history, visuotactile.kinematics, visuotactile.models.correction, visuotactile.models.nominal, visuotactile.recording。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：((source >= 0) & (source <= step)).all, (args.output / 'FAILED.txt').write_text, (args.output / 'report.json').write_text, args.output.mkdir, json.dumps, torch.save。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/probe_perception.py`

Audit real v2 recordings, render stage sheets and check trainable I/O.

- 定义（函数/类:行）：check_tree:28, stage_sheet:36, main:88。
- 来源：local addition；依赖：PIL, argparse, dataclasses, h5py, json, numpy, pathlib, scipy.spatial.transform, time, torch, torch.utils.data, visuotactile.contracts, visuotactile.correction, visuotactile.dataset, visuotactile.evaluate, visuotactile.models.correction, visuotactile.models.nominal, visuotactile.observations。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(args.output / 'report.json').write_text, args.output.mkdir, json.dumps, sheet.save, torch.save。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/recording.py`

Episode-based HDF5 recorder; model inputs and privileged labels are separate.

- 定义（函数/类:行）：model_fingerprint:12, numpy_tree:20, flatten:28, EpisodeWriter:39, CompactEpisodeStream:90, verify_stream_fields:188, post_observation:207, EpisodeReader:221, main:284, __init__:40, append:48, close:86, __init__:97, append:111, _extend:143, flush:153, finish:167, __init__:227, attributes:235, read:239, steps:257, export_pickle:261, verify:194, visit:247。
- 来源：local addition；依赖：argparse, h5py, hashlib, json, numpy, os, pathlib, pickle, torch, visuotactile.compact。
- 静态调用者：tests/test_causal_memory.py, tests/test_compact.py, tests/test_persistent_collection.py, visuotactile/collect.py, visuotactile/collect_success_batch.py, visuotactile/finalize_collection.py, visuotactile/manifest.py, visuotactile/probe_history.py。
- I/O或状态操作候选：json.dumps, pickle.dump, self.directory.mkdir, temporary.unlink。
- 文档：CHANGELOG.md, NIGHT_COLLECTION.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：tests/test_compact.py, tests/test_causal_memory.py, tests/test_persistent_collection.py。

### `visuotactile/regression.py`

Compare original and instrumented tasks in separate Kit processes.

- 定义（函数/类:行）：main:12, RenderedOriginal:34, snapshot:42, _get_observations:35。
- 来源：local addition；依赖：argparse, evaluation.eval_isaacsim, isaaclab.app, isaacsimenvs, isaacsimenvs.tasks.precise_assembly.precise_assembly_env, json, numpy, os, pathlib, torch, traceback, visuotactile.sim.environment。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(args.output / 'FAILED.txt').write_text, (args.output / 'result.json').write_text, args.output.mkdir, env.step, json.dumps, np.savez。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/residual_controller.py`

Trained residual execution; mixed feedback belongs only to Correction.

- 定义（函数/类:行）：ResidualController:14, __init__:15, act:39, acknowledge:75。
- 来源：local addition；依赖：pathlib, torch, visuotactile.action_history, visuotactile.compact, visuotactile.correction, visuotactile.history, visuotactile.models.residual, visuotactile.observations, visuotactile.residual_data。
- 静态调用者：tests/test_residual.py, visuotactile/evaluate_residual.py, visuotactile/validate_residual.py。
- I/O或状态操作候选：saved.get, self.base.plan_generation_step.clone。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_residual.py。

### `visuotactile/residual_data.py`

Immutable, causal B0 replay caches over full expert observation sequences.

- 定义（函数/类:行）：load_base:27, splits:37, cache_episode:43, main:111。
- 来源：local addition；依赖：argparse, dataclasses, hashlib, json, numpy, pathlib, time, torch, visuotactile.compact, visuotactile.correction, visuotactile.history, visuotactile.manifest, visuotactile.models.residual, visuotactile.residual_session, visuotactile.rollout, visuotactile.training_resources。
- 静态调用者：tests/test_residual.py, visuotactile/audit_network.py, visuotactile/residual_controller.py, visuotactile/residual_report.py, visuotactile/train_residual.py, visuotactile/validate_residual.py。
- I/O或状态操作候选：cache.mkdir, json.dumps, torch.save。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_residual.py。

### `visuotactile/residual_report.py`

Small sanitized aggregate, recomputed from local residual artifacts.

- 定义（函数/类:行）：metric_summary:14, resource_summary:21, paired_trace_summary:44, pose_difference:68, switch_physical_summary:76, execution_summary:100, main:124。
- 来源：local addition；依赖：argparse, json, numpy, pathlib, shutil, visuotactile.compact, visuotactile.residual_data, visuotactile.training_resources。
- 静态调用者：tests/test_residual.py, visuotactile/viewing_report.py。
- I/O或状态操作候选：json.dumps。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_residual.py。

### `visuotactile/residual_session.py`

Continuous external telemetry and immutable deadlines for residual experiments.

- 定义（函数/类:行）：SessionGate:13, main:66, __init__:14, save_waits:23, remaining:28, before:35, pace:57。
- 来源：local addition；依赖：argparse, atexit, json, os, pathlib, shutil, time, visuotactile.training_resources。
- 静态调用者：visuotactile/evaluate_residual.py, visuotactile/residual_data.py, visuotactile/train_residual.py, visuotactile/validate_residual.py。
- I/O或状态操作候选：args.root.mkdir。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/review_delivery.py`

Small human-readable camera delivery; original training pixels untouched.

- 定义（函数/类:行）：rotation:15, source_audit:17, deliver:51, main:118, canvas:57。
- 来源：local addition；依赖：PIL, argparse, av, json, matplotlib, matplotlib.pyplot, numpy, pathlib, scipy.spatial.transform, visuotactile.camera_review, visuotactile.compact, visuotactile.trial_utils。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：canvas(n // 3).save, fig.savefig, image.save, out.mkdir, sheet.save, write_json, write_video。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/rollout.py`

Independent nominal checkpoint execution; no expert/privileged inference data.

- 定义（函数/类:行）：NominalController:11, TactileCommandController:133, load_command_controller:147, __init__:12, load:37, encode_current:57, condition_rows:60, synchronize:63, observe_execution:67, act:73, encode_current:135, condition_rows:143。
- 来源：local addition；依赖：time, torch, visuotactile.action_history, visuotactile.history, visuotactile.models.correction, visuotactile.models.nominal, visuotactile.models.tactile_dp, visuotactile.observations。
- 静态调用者：tests/test_action_history.py, tests/test_causal_memory.py, tests/test_command_training.py, tests/test_perception_v2.py, tests/test_visuotactile.py, visuotactile/audit_baselines.py, visuotactile/check_camera_interfaces.py, visuotactile/collect.py, visuotactile/evaluate_baselines.py, visuotactile/policy_census.py, visuotactile/probe_action_history.py, visuotactile/probe_correction.py, visuotactile/residual_data.py, visuotactile/validate_command_training.py。
- I/O或状态操作候选：(changed & (packet['step_id'].cpu() != 0)).any, packet['step_id'].cpu, saved.get。
- 文档：CHANGELOG.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md, BASELINE_TRAINING.md, ARCHITECTURE.md；测试：tests/test_action_history.py, tests/test_command_training.py, tests/test_causal_memory.py, tests/test_visuotactile.py, tests/test_perception_v2.py。

### `visuotactile/run_baselines.py`

Run two independent trainers, then their paired evaluation. Own children only.

- 定义（函数/类:行）：read_status:14, main:38, read:17, stop:54。
- 来源：local addition；依赖：argparse, fcntl, json, os, pathlib, signal, subprocess, sys, time, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(output / 'pid').write_text, args.root.mkdir, json.dumps, output.mkdir, subprocess.Popen。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/run_cpu.sh`

运行入口或显式配置；具体 shell / JSON 字段见源文件

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：local addition；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, NIGHT_COLLECTION.md, RUNBOOK.md, LONG_HORIZON_20261030.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, GIT_WORKFLOW.md, SESSION_HANDOFF.md, NETWORK_ARCHITECTURE_EXACT.md, PROJECT_FILE_MAP.md, CORRECTION_TRAINING.md, ARCHITECTURE.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/run_insertion_dataset.py`

Bounded serial expert collection. No optimizer, retries or outcome selection.

- 定义（函数/类:行）：tree_bytes:20, design:24, check_pairing:50, run:69, main:186。
- 来源：local addition；依赖：argparse, json, numpy, pathlib, shutil, subprocess, time, visuotactile.analyze_rl_trials, visuotactile.compact, visuotactile.manifest, visuotactile.run_rl_trials, visuotactile.trial_utils。
- 静态调用者：tests/test_insertion_dataset.py, visuotactile/analyze_insertion_dataset.py, visuotactile/collect_success_batch.py。
- I/O或状态操作候选：(root / 'run_status').mkdir, json.dumps, subprocess.Popen, subprocess.check_output, subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip, write_json。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_insertion_dataset.py。

### `visuotactile/run_long.py`

One bounded owner for pilot admission, training and the six diagnostics.

- 定义（函数/类:行）：pilot_admission:22, active_trainers:48, active_pilots:63, decode_videos:67, run:85, main:185, status:89, launch:91, wait:99。
- 来源：local addition；依赖：argparse, av, fcntl, json, math, os, pathlib, subprocess, time, visuotactile.compact, visuotactile.training_resources。
- 静态调用者：tests/test_long_horizon.py。
- I/O或状态操作候选：subprocess.Popen。
- 文档：PROJECT_FILE_MAP.md；测试：tests/test_long_horizon.py。

### `visuotactile/run_loss_study.py`

Bounded owner for paired loss experiments and registered downstream stages.

- 定义（函数/类:行）：initialize:15, run:33, status:39, launch:41, wait:50。
- 来源：local addition；依赖：argparse, datetime, fcntl, json, os, pathlib, subprocess, time, visuotactile.compact, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：root.mkdir, subprocess.Popen, subprocess.check_output, subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/run_rl_trials.py`

Sequential paired first-episode collection, with a durable plan and status.

- 定义（函数/类:行）：make_trials:18, utc:27, validate_completion:31, prior_attempt:67, run_stage:88, main:156。
- 来源：local addition；依赖：argparse, datetime, json, pathlib, shutil, subprocess, time, visuotactile.analyze_rl_trials, visuotactile.compact, visuotactile.trial_utils。
- 静态调用者：tests/test_compact.py, tests/test_rl_trials.py, visuotactile/finalize_collection.py, visuotactile/initialization_report.py, visuotactile/run_insertion_dataset.py。
- I/O或状态操作候选：(args.output / stage).mkdir, args.output.mkdir, json.dumps, stream.write, subprocess.Popen, subprocess.check_output, subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip, write_json。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_compact.py, tests/test_rl_trials.py。

### `visuotactile/sim/__init__.py`

Import simulation modules only AFTER Isaac Lab AppLauncher.

- 定义（函数/类:行）：无函数定义/配置值。
- 来源：local addition；依赖：无静态导入。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：CHANGELOG.md, SYSTEM_ARCHITECTURE_AND_EXPERIMENT_AUDIT.md, PROJECT_FILE_MAP.md, INITIALIZATION_AND_PERTURBATIONS.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/sim/environment.py`

Opt-in instrumentation; original task, control and RL classes are unchanged.

- 定义（函数/类:行）：InstrumentedAssemblyEnv:31, __init__:34, _student_enabled:137, _setup_scene:145, _contact_window:206, configure_camera_views:210, _sample_kinematics:220, _kinematic_packet:226, _camera_packet:242, _diagnostics:275, _reset_peg_episode:305, _reset_idx:321, _record_action_trace:366, _pre_physics_step:392, _get_rewards:436, _get_observations:465, research_metadata:496, write_with_wrench_sample:95, update_with_encoder_sample:130。
- 来源：local addition；依赖：contextlib, isaaclab.sensors, isaaclab.sim, isaaclab.utils.math, isaacsimenvs.tasks.play.utils.obs_utils, isaacsimenvs.tasks.play.utils.scene_utils, isaacsimenvs.tasks.precise_assembly.precise_assembly_env, numpy, omni.replicator.core, omni.usd, pxr, torch, visuotactile.camera_rigs, visuotactile.initial_geometry, visuotactile.kinematics, visuotactile.observations。
- 静态调用者：visuotactile/collect.py, visuotactile/evaluate_baselines.py, visuotactile/evaluate_long.py, visuotactile/evaluate_residual.py, visuotactile/evaluate_study.py, visuotactile/evaluate_training_initials.py, visuotactile/regression.py。
- I/O或状态操作候选：(self.research_step > 0).clone, (steps - selected_step).to, original_write, selected_step.clone, self.object.write_root_pose_to_sim, self.research_step.clone, steps.clone, steps[:, None].expand, steps[:, None].expand(-1, length).clone, super()._pre_physics_step。
- 文档：CHANGELOG.md, INDEPENDENT_AUDIT_20261030.md, NIGHT_COLLECTION.md, PROJECT_GUIDE_ZH.md, PROJECT_FILE_MAP.md, CORRECTION_TRAINING.md, ARCHITECTURE.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/storage_audit.py`

Measure HDF5 allocation, not decoded-array size. No simulator or training.

- 定义（函数/类:行）：category:12, allocation_groups:35, audit_file:43, main:63, visit:46。
- 来源：local addition；依赖：argparse, collections, h5py, json, numpy, pathlib, visuotactile.compact, visuotactile.trial_utils。
- 静态调用者：tests/test_insertion_dataset.py。
- I/O或状态操作候选：json.dumps, write_json。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_insertion_dataset.py。

### `visuotactile/study_audit.py`

Pinned real-checkpoint input, normalization and execution-history audit.

- 定义（函数/类:行）：audit:20, CachedVision:49, predict:61, __init__:50, __call__:52。
- 来源：local addition；依赖：argparse, gc, h5py, json, pathlib, torch, visuotactile.action_reference, visuotactile.compact, visuotactile.long_controller, visuotactile.long_data, visuotactile.models.long_horizon, visuotactile.study_metrics, visuotactile.study_resources, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(controller.last_condition[0] - full[step // 12]).abs, (controller.last_condition[0] - full[step // 12]).abs().max, LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to, LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to('cuda:0').eval。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/study_metrics.py`

Matched x0 objectives and sampled native-command metrics by group/horizon.

- 定义（函数/类:行）：joint_groups:12, masked_objectives:23, denoising_objectives:32, evaluate:44。
- 来源：local addition；依赖：json, math, torch, visuotactile.action_reference, visuotactile.long_data, visuotactile.train_long。
- 静态调用者：tests/test_loss_study.py, visuotactile/audit_network.py, visuotactile/study_audit.py, visuotactile/train_loss_study.py, visuotactile/train_transformer_residual.py。
- I/O或状态操作候选：未检出写入/step命名调用；以源码为准。
- 文档：PROJECT_FILE_MAP.md；测试：tests/test_loss_study.py。

### `visuotactile/study_resources.py`

Independent resource watchdog for the bounded loss/residual study.

- 定义（函数/类:行）：StudyLease:13, owned_workers:21, violation_reasons:39, monitor:54, check:14。
- 来源：local addition；依赖：argparse, json, os, pathlib, signal, time, visuotactile.long_resources, visuotactile.training_resources。
- 静态调用者：tests/test_loss_study.py, visuotactile/evaluate_study.py, visuotactile/evaluate_training_initials.py, visuotactile/study_audit.py, visuotactile/train_loss_study.py, visuotactile/train_transformer_residual.py, visuotactile/transformer_data.py。
- I/O或状态操作候选：(root / 'RESOURCE_PAUSE').unlink, json.dumps, stream.write。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_loss_study.py。

### `visuotactile/train.py`

CPU smoke or research training on recorded actions; never synthetic labels.

- 定义（函数/类:行）：main:15, ordered_batches:80。
- 来源：local addition；依赖：argparse, dataclasses, hashlib, json, pathlib, time, torch, torch.utils.data, visuotactile.compact, visuotactile.dataset, visuotactile.models.nominal。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(args.output / 'report.json').write_text, args.output.mkdir, json.dumps, optimizer.step, torch.linspace(0, len(validation) - 1, steps=min(16, len(validation))).long, torch.linspace(0, len(validation) - 1, steps=min(16, len(validation))).long().tolist, torch.save。
- 文档：CHANGELOG.md, RUNBOOK.md, PROJECT_GUIDE_ZH.md, CONTRACTS.md, PROJECT_FILE_MAP.md, ARCHITECTURE.md, DP_LOSS_AND_TRANSFORMER_RESIDUAL_20261031.md, INSERTION_DATASET.md, AUDIT.md；测试：tests/test_action_history.py, tests/test_training_resources.py, tests/test_command_training.py。

### `visuotactile/train_dp.py`

Train the command imitation baselines B0 and B1 on the production manifest.

- 定义（函数/类:行）：_move:35, _manifest_hash:41, _split:45, _assert_no_own_plan:66, _fit_proprio_stats:85, _make_model:103, _condition:111, _pad_time:121, _batch_episode_chunks:131, _amp_context:187, _prefetch:193, _run_epoch:209, _save:297, main:327, handle_stop:373, log:380, publish_progress:478, group_checkpoint:486。
- 来源：local addition；依赖：__future__, argparse, concurrent.futures, contextlib, dataclasses, hashlib, json, numpy, os, pathlib, signal, subprocess, time, torch, visuotactile.action_history, visuotactile.command_metrics, visuotactile.compact, visuotactile.contracts, visuotactile.dataset, visuotactile.history, visuotactile.manifest, visuotactile.models.nominal, visuotactile.models.tactile_dp, visuotactile.training_resources。
- 静态调用者：tests/test_action_history.py, tests/test_command_training.py, visuotactile/audit_baselines.py, visuotactile/command_metrics.py, visuotactile/probe_action_history.py, visuotactile/validate_command_training.py。
- I/O或状态操作候选：(args.output / 'config.json').write_text, _save, args.output.mkdir, json.dumps, log_file.write, optimizer.step, scaler.step, stream.write, subprocess.check_output, subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip, torch.save。
- 文档：CHANGELOG.md, CONTRACTS.md, SESSION_HANDOFF.md, PROJECT_FILE_MAP.md, BASELINE_TRAINING.md；测试：tests/test_action_history.py, tests/test_command_training.py。

### `visuotactile/train_long.py`

Sequential H64 training, EMA validation, resumable updates and held-out test.

- 定义（函数/类:行）：move:22, fit_statistics:26, update_ema:50, learning_rate:59, sampled_metrics:65, save:130, cpu_snapshot:136, train:146, test:334, main:350, checkpoint:218, inference_payload:227。
- 来源：local addition；依赖：argparse, copy, dataclasses, gc, json, math, pathlib, random, signal, time, torch, visuotactile.action_reference, visuotactile.compact, visuotactile.long_data, visuotactile.long_resources, visuotactile.models.long_horizon, visuotactile.training_resources。
- 静态调用者：visuotactile/inspect_long.py, visuotactile/study_metrics.py, visuotactile/train_loss_study.py, visuotactile/train_transformer_residual.py, visuotactile/transformer_controller.py, visuotactile/transformer_data.py。
- I/O或状态操作候选：LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to, LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to('cuda:0').eval, optimizer.step, output.mkdir, save, saved['cpu_rng'].cpu, saved['cuda_rng'].cpu, torch.save。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/train_loss_study.py`

Independent paired continuations from a frozen epoch40 EMA checkpoint.

- 定义（函数/类:行）：train:25, test:207, checkpoint:95, best_checkpoint:105, measurements:115。
- 来源：local addition；依赖：argparse, copy, dataclasses, gc, json, math, os, pathlib, random, signal, time, torch, visuotactile.compact, visuotactile.long_data, visuotactile.models.long_horizon, visuotactile.study_metrics, visuotactile.study_resources, visuotactile.train_long, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to, LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to('cuda:0').eval, optimizer.step, output.mkdir, save。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/train_residual.py`

Full-episode supervised residual training with matched force-masked control.

- 定义（函数/类:行）：read_record:23, amplitude_statistics:33, chunk_record:59, metrics_empty:67, accumulate:74, finish_metrics:89, run_epoch:100, save_checkpoint:141, main:157, boundary:227。
- 来源：local addition；依赖：argparse, dataclasses, json, math, numpy, os, pathlib, subprocess, time, torch, traceback, visuotactile.compact, visuotactile.history, visuotactile.models.residual, visuotactile.residual_data, visuotactile.residual_session, visuotactile.training_resources。
- 静态调用者：visuotactile/audit_network.py, visuotactile/validate_residual.py。
- I/O或状态操作候选：(output / 'error.txt').write_text, json.dumps, optimizers[name].step, output.mkdir, save_checkpoint, stream.write, subprocess.check_output, subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip, torch.save。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/train_transformer_residual.py`

Matched real-force and force-masked full-episode residual training.

- 定义（函数/类:行）：ResidualMetrics:24, evaluate:73, build_model:110, train:127, test:286, __init__:25, add:31, result:58, checkpoint:198, measurements:205。
- 来源：local addition；依赖：argparse, dataclasses, gc, json, math, os, pathlib, random, signal, time, torch, visuotactile.compact, visuotactile.models.transformer_residual, visuotactile.study_metrics, visuotactile.study_resources, visuotactile.train_long, visuotactile.training_resources, visuotactile.transformer_data。
- 静态调用者：visuotactile/audit_network.py。
- I/O或状态操作候选：optimizer.step, output.mkdir, save。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/training_resources.py`

Memory/GPU-average admission with bounded telemetry; no external signals.

- 定义（函数/类:行）：atomic_json:10, ResourceGuard:17, __init__:18, sample:40, memory_safe:96, utilization_safe:100, safe:111, before_chunk:114, summary:137。
- 来源：local addition；依赖：collections, json, os, pathlib, subprocess, time, torch。
- 静态调用者：tests/test_training_resources.py, visuotactile/audit_baselines.py, visuotactile/audit_distribution.py, visuotactile/audit_loss_study.py, visuotactile/audit_network.py, visuotactile/audit_system.py, visuotactile/audit_training_initials.py, visuotactile/cleanup_long.py, visuotactile/evaluate_baselines.py, visuotactile/evaluate_long.py, visuotactile/evaluate_residual.py, visuotactile/evaluate_study.py, visuotactile/evaluate_training_initials.py, visuotactile/inspect_long.py, visuotactile/long_data.py, visuotactile/long_resources.py, visuotactile/plot_loss_study.py, visuotactile/plot_network_audit.py, visuotactile/policy_census.py, visuotactile/probe_action_history.py, visuotactile/residual_data.py, visuotactile/residual_report.py, visuotactile/residual_session.py, visuotactile/run_baselines.py, visuotactile/run_long.py, visuotactile/run_loss_study.py, visuotactile/study_audit.py, visuotactile/study_resources.py, visuotactile/train_dp.py, visuotactile/train_long.py, visuotactile/train_loss_study.py, visuotactile/train_residual.py, visuotactile/train_transformer_residual.py, visuotactile/transformer_data.py, visuotactile/validate_command_training.py, visuotactile/validate_long.py, visuotactile/validate_residual.py, visuotactile/viewing_report.py。
- I/O或状态操作候选：json.dumps, self.output.mkdir, stream.write, subprocess.check_output, subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=3).splitlines, temporary.write_text。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_training_resources.py。

### `visuotactile/transformer_controller.py`

G01 actual mixed-command history and10Hz residual with native60Hz feedback.

- 定义（函数/类:行）：TransformerController:17, __init__:18, act:38, acknowledge:97。
- 来源：local addition；依赖：pathlib, time, torch, visuotactile.action_history, visuotactile.compact, visuotactile.long_controller, visuotactile.long_data, visuotactile.models.residual, visuotactile.models.transformer_residual, visuotactile.train_long, visuotactile.transformer_data。
- 静态调用者：tests/test_transformer_residual.py, visuotactile/evaluate_study.py。
- I/O或状态操作候选：packet['contact_substeps_local_N'][0].cpu, packet['contact_substeps_w_N'][0].cpu, saved.get。
- 文档：PROJECT_FILE_MAP.md；测试：tests/test_transformer_residual.py。

### `visuotactile/transformer_data.py`

Frozen selected G00 replay with explicitly teacher-forced actual history.

- 定义（函数/类:行）：audit_generated_episode:24, tactile_features:118, remaining_plans:130, residual_inputs:138, residual_labels:155, load_residual_record:163, generate_episode:171, main:229, CachedDino:28, __init__:29, __call__:32, at:44, read:180。
- 来源：local addition；依赖：argparse, h5py, hashlib, json, pathlib, time, torch, visuotactile.action_reference, visuotactile.compact, visuotactile.evaluate_study, visuotactile.long_controller, visuotactile.long_data, visuotactile.models.long_horizon, visuotactile.models.residual, visuotactile.models.transformer_residual, visuotactile.private_history, visuotactile.study_resources, visuotactile.train_long, visuotactile.training_resources。
- 静态调用者：tests/test_transformer_residual.py, visuotactile/audit_network.py, visuotactile/train_transformer_residual.py, visuotactile/transformer_controller.py。
- I/O或状态操作候选：(command[0] - record['base'][step]).abs, (command[0] - record['base'][step]).abs().max, (controller.last_condition[0] - record['nominal_conditions'][step // 12]).abs, (controller.last_condition[0] - record['nominal_conditions'][step // 12]).abs().max, (controller.plan[0] - record['plans'][step // 12]).abs, (controller.plan[0] - record['plans'][step // 12]).abs().max, (controller.records['action'][-1] - record['nominal_action'][step]).abs, (controller.records['action'][-1] - record['nominal_action'][step]).abs().max, (controller.records['depth'][-1] - original['depth'][step // 12]).abs, (controller.records['depth'][-1] - original['depth'][step // 12]).abs().max, (controller.state.float().cpu() - record['nominal_states'][step // 12]).abs, (controller.state.float().cpu() - record['nominal_states'][step // 12]).abs().max, LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to, LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to('cuda:0').eval, LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to('cuda:0').eval().requires_grad_, at('frames/depth_m', step).float, at('frames/rgb_u8', step).float, directory.mkdir, model.project_frames(original['patches'][step // 12:step // 12 + 1].to('cuda:0'), original['depth'][step // 12:step // 12 + 1].to('cuda:0'), original['camera_valid'][step // 12:step // 12 + 1].to('cuda:0')).float, model.project_frames(original['patches'][step // 12:step // 12 + 1].to('cuda:0'), original['depth'][step // 12:step // 12 + 1].to('cuda:0'), original['camera_valid'][step // 12:step // 12 + 1].to('cuda:0')).float().cpu, original['camera_valid'][step // 12:step // 12 + 1].to, original['depth'][step // 12:step // 12 + 1].to, original['patches'][step // 12:step // 12 + 1].to, save。
- 文档：PROJECT_FILE_MAP.md；测试：tests/test_transformer_residual.py。

### `visuotactile/trial_utils.py`

Small, simulation-independent provenance utilities for bounded RL trials.

- 定义（函数/类:行）：code_identity:8, write_json:20, git:10。
- 来源：local addition；依赖：hashlib, json, pathlib, subprocess。
- 静态调用者：visuotactile/analyze_insertion_dataset.py, visuotactile/analyze_rl_trials.py, visuotactile/camera_review.py, visuotactile/check_camera_interfaces.py, visuotactile/check_dataset.py, visuotactile/collect.py, visuotactile/collect_success_batch.py, visuotactile/collection_resources.py, visuotactile/finalize_collection.py, visuotactile/initialization_report.py, visuotactile/review_delivery.py, visuotactile/run_insertion_dataset.py, visuotactile/run_rl_trials.py, visuotactile/storage_audit.py。
- I/O或状态操作候选：hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest, json.dumps, json.dumps(hashes, sort_keys=True).encode, subprocess.check_output, subprocess.check_output(['git', *args], cwd=root, text=True).strip, temporary.write_text。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/validate_command_training.py`

Bounded real-production acceptance; writes local evidence, never full training.

- 定义（函数/类:行）：_slice:20, main:25, loss_fixed:64, state_error:91。
- 来源：local addition；依赖：argparse, copy, json, numpy, pathlib, time, torch, types, visuotactile.compact, visuotactile.history, visuotactile.manifest, visuotactile.models.nominal, visuotactile.rollout, visuotactile.train_dp, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：_save, args.output.mkdir, json.dumps, optimizer.step, torch.save。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/validate_long.py`

Actual-data causal alignment checks, without new simulator trajectories.

- 定义（函数/类:行）：main:15, CachedVision:35, __init__:36, __call__:38。
- 来源：local addition；依赖：argparse, h5py, json, pathlib, torch, visuotactile.long_controller, visuotactile.long_data, visuotactile.models.long_horizon, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(controller.last_condition[0] - full[step // 12]).abs, (controller.last_condition[0] - full[step // 12]).abs().max。
- 文档：PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/validate_residual.py`

Read-only real checkpoint/cache/feedback equivalence on an observed prefix.

- 定义（函数/类:行）：tensors:18, main:24, move:43。
- 来源：local addition；依赖：argparse, json, numpy, pathlib, time, torch, visuotactile.manifest, visuotactile.residual_controller, visuotactile.residual_data, visuotactile.residual_session, visuotactile.train_residual, visuotactile.training_resources。
- 静态调用者：CLI/动态调用或无静态调用；需结合主链。
- I/O或状态操作候选：(controller.last['base'][0] - record['base'][step]).abs, (controller.last['base'][0] - record['base'][step]).abs().max, (controller.last['plans'][0] - record['plans'][step]).abs, (controller.last['plans'][0] - record['plans'][step]).abs().max, (controller.last['raw'][0] - expected[0, step].cpu()).abs, (controller.last['raw'][0] - expected[0, step].cpu()).abs().max, expected[0, step].cpu。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：未检出直接import测试；不表示已验证。

### `visuotactile/viewing_report.py`

Decode every viewing frame and publish only small, sanitized statistics.

- 定义（函数/类:行）：check_video:14, check_trace:45, resource_intervals:66, preservation_checks:87, worker_resource_window:115, main:133。
- 来源：local addition；依赖：PIL, argparse, av, json, numpy, pathlib, shutil, visuotactile.compact, visuotactile.residual_report, visuotactile.training_resources。
- 静态调用者：tests/test_residual.py。
- I/O或状态操作候选：json.dumps, preview.save。
- 文档：CHANGELOG.md, PROJECT_FILE_MAP.md；测试：tests/test_residual.py。
