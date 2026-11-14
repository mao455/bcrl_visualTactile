# Tight-Insertion 执行后端候选合同

2026-11-14 用户授权可扩展批量采集/评估及后端测试。本合同说明已实现的可选入口；**GPU物理/TiledCamera的完整传感、接触和任务验收仍未完成**。真实吞吐与学生成功结论必须从登记run的结果取得；首次Tiled故障及后续候选见下。

`execution_backend` 使用 `tight_execution_backend_v1`：`physics_device` 为 `cpu` 或显式 `cuda:N`，`camera_backend` 为 `standard` 或 `tiled`，`kit_tasking_threads` 与 `physics_threads` 为不超过当前 CPU affinity 的正整数。配置及 SHA、实际环境数、启动请求、Kit 参数回读、构造后设备/dt 分别保存。环境数没有四环境或其他人为上限，显存/RAM/磁盘准入与实测吞吐决定采用值。Tiled 的进程级 `viewTile/limit` 随登记环境数设置，不能静默使用安装版 tile 上限阻挡更大候选。

未登记后端时，原 CPU/standard 相机与双物理线程路径保持。原 `TIGHT_KIT_TASKING_THREADS=2/4` 兼容；已登记配置与该环境变量冲突时启动前拒绝，不能静默覆盖。任务资产、频率、PD、动作目标、成功判据和图像内容检查均不由此配置修改。CPU 与 GPU 物理采用不同执行身份，不要求隐藏轨迹逐位相同。

## 三视角与移动相机标定

三视角均使用真实同 tick RGB-D；standard 仍从各 RGB-D render product 的 CameraParams 读取当前 ROS 光学位姿。安装版 TiledCamera 每个视角只有一个包含多相机的 render product，原单 product 对应单 lane 的标定读取不能直接沿用。

Tiled 候选按传感器真实相机 prim 顺序验证 `env_0 … env_N-1`，为每个同一相机 prim 创建一个独立 **1×1 标定 render product**，读取该产品 CameraParams；不使用已经发现陈旧的 `camera.data.pos_w/quat` 作为移动相机替代。相机路径、图像产品、标定产品、1×1 分辨率和标定来源进入 metadata，标定产品开销计入候选实测时间。缺失 lane、乱序、缺少变换或非有限变换会失败。真实移动父体与相机同步、接触、内容和任务执行仍须现场验收；仅通过这些结构检查不能宣布标定有效或采用 Tiled。

## 随机流和 lane 边界

旧未登记 CPU 合同不变。显式后端使用进程内拥有的 Python/NumPy/CPU Torch 随机流，GPU 候选同时保存/恢复指定 CUDA 设备的随机状态；不会使用 `manual_seed_all`。已登记的整批 reset 在拥有的 scope 内调用 `env.reset(seed=None)`，避免 native seed helper 重播其他 GPU 的全局随机流。独立 DP 采样器仍由各 lane 持有。

原生向量化延迟、扰动和观测噪声仍按**整批随机流**产生，不声称原生 event RNG 已逐 lane 独立，也不声称批量与逐例隐状态精确相同。显式 `reset_lanes` 只调用原生索引 reset，不全局 reseed；调用方仍负责场景 forward、观测和对应记忆边界。`LaneRandomStreams` 可用于显式逐 lane callback，重置某一流不改变其他流或外部 RNG，但未将它冒充为对原生所有向量化 draw 的自动拦截。

当前正式批量入口采用整 wave：所有有效 lane 结束后才换下一 wave，没有动态回收仍活动的 lane。每 lane 的历史、目标队列、takeover、writer、自然终止与分母须由 runner 单独验收。

## 已执行检查

CPU 检查覆盖默认参数字节兼容、配置拒绝、线程/后端身份、200/256 lane 无人为上限、CPU RNG 隔离、单 lane 流替换与索引 reset、CUDA 状态 mock 的指定设备隔离/异常恢复、相机 lane 顺序与当前矩阵拒绝。CPU及CUDA mock测试不算现场后端验收。

## v2：显式渲染描述符容量候选

首次`scaling_cpu_tiled_t16_n8_01`冻结`ff555cec5d3edd565a3cd7d24299a982611afefd`，登记CPU物理、Tiled、Kit16/PhysX2、8lane、视觉微批2。`startup_timing.json`实际为15.801168673秒；`A/`为空，未完成初态/任何控制tick。Kit日志在17.135秒开始报`Unable to allocate descriptor sets`，继而RTX denoiser/raytracing的ParameterBlock及RenderGraph error7。它是构造完成后首次reset取图故障，不是学生任务失败。owner保留现场，最终回收worker，退出码-9、总197.074秒。

只读安装代码核对：实际加载`/home/CNS2025331827/docs/IsaacLab_DexAssemble-sharpa/source/isaaclab/isaaclab/sensors/camera/tiled_camera.py`，每个视角一个8相机render product；本wrapper另外逐相机创建1×1标定产品。研究相机共3个图像产品+24个标定产品，合计48个camera-view绑定；standard路径为24个产品/24个绑定。这里不含Kit额外viewport，也不是对内部descriptor数量的测量。1×1标定仍是正常RTX render product，不能按像素尺寸认为它没有每view的renderer/denoiser状态开销。

安装包的`/home/CNS2025331827/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.sensors.camera/config/extension.toml`，`[[test]]`参数实际包含`--/rtx/descriptorSets=360000`及`--/rtx/reservedDescriptors=900000`。同安装`extscache/omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311/config/extension.toml`测试使用60000/500000。它们是已安装上游测试的显式设置，**不是当前worker默认容量的证据**；失败v1没有记录这些设置的实际值。容量不足与额外view状态是待检验假设，不能据此宣称根因已经证实。

新`execution_backend.version=tight_execution_backend_v2`单独启用两个进程级容量字段：`renderer_descriptor_sets`默认360000、`renderer_reserved_descriptors`默认900000，可显式登记其他正32位整数。它们进入配置SHA和Kit启动参数，在AppLauncher完成后以及环境构造后分别读取`/rtx/descriptorSets`和`/rtx/reservedDescriptors`；缺失、类型不符或数值不符均保存读回收据并拒绝继续加载专家/开始动作。两次收据分别位于`kit_started.json`和`execution_backend_observed.json`的`renderer_resources`。同时记录实际`viewTile/limit`供诊断，不把早期启动请求冒充渲染模式加载后的实际值。

v1/未登记路径不添加容量参数，也不额外读取这些设置；默认启动字符串保持原字节。v2不改变画质、AA、RGB-D分辨率、相机产品、动态标定、资产或物理控制。下一次同8lane/微批2的有限真实对照应只改变这两个容量；只有真实取图、内容、每tick时钟/推理计数及正常软件完成后，才能判断是否解决本次故障。即使该探针通过，完整Tiled移动标定/接触与任务执行仍须验收。

本次32项CPU检查通过，包括v1启动兼容、v2身份与可配置值、拒绝非法容量、真实入口mock两阶段读回、保存失败收据并在专家创建前终止。未由本次实现启动GPU或Kit；不宣称已修复现场或提高吞吐。
