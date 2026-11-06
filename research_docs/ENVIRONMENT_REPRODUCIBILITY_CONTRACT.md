# 仿真与复现边界

当前授权和门控只见[当前合同](CONTRACTS.md)，本页保留工程解释。运行采用已安装Isaac Sim5.1、CPU PhysX单lane、原120/60Hz；不升级SDK、不改资产、PD、碰撞、成功/终止定义。精确环境和资产SHA由实际run导出，不把配置文字当实测。

`IsolatedEnvironment`隔离环境的Python/NumPy/Torch CPU随机流；策略diffusion用独立CUDA Generator。专家和学生动作不同可改变lift/retract条件分支，因此同seed不保证逐tick外力/延迟事件等价。原四lane数据的solver/contact/sensor内部状态未保存；恢复observable不是历史精确物理重放。

合法性查询使用真实PhysX body pose上的原object geometry，检查self-hit及外部overlap。历史USD spawn-transform查询曾错误；历史拒绝和失败不因修复被删除。当前持久单lane场景每次重新原生reset并保存可观察初态SHA，不把不同初态拼成严格轨迹对。

监督target_after_rad是原生delay/EMA/减速之后的实际目标。学生经原限位裁剪及canonical→Lab映射直接设置执行器目标缓冲；保留原PD和积分。逐子步目标与提交目标必须真实验证。当前240tick H24 oracle的目标误差均0，但开环任务未完成且动作耗尽，因此记删失；数据准入不要求它成功。

共享GPU以当前合同为准：目标35%、实测整卡均值约束50%、6GiB VRAM/12GiB RAM/60GiB磁盘保留；只等待/终止本任务拥有的进程。分阶段监控存在的缺口必须报告，不声称所有时刻均已观测。
