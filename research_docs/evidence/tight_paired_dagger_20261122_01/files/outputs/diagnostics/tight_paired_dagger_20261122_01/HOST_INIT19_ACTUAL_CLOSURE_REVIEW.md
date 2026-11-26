# Host-init19 实际收尾独立复核

19 helper 已退出，**保留失败，不存在待恢复的活跃 affinity 扩展证据**。精确 service `tight-paired-host-init-capacity19.service`、InvocationID `34b4aabd29454f7ab57f414fbb67a41a`、PID3542303/birth1795506788.47 对应 `MainPID=0`、`ExecMainStatus=1`、`Result=exit-code`；unit 为 failed，cgroup 已不存在。完整 exact-module census 为空，helper 及三条实际触及 worker 的 PID/birth 均已不存在。

实际只有 wave46、47、48 三次持久 intent，均有 exit。46 只借 CPU5，134 次扩展和134次恢复，实际0.4397364秒；47 只借 CPU1，133次扩展和133次恢复，实际0.4398157秒。两次均因 external-activity 标记提前撤回，记录 `restored=true`、原 mask8–23、nice10、`placement_end_verified=true`，无 restore_error/infrastructure_abort。48 因 capacity/readiness race 未修改，扩展、恢复和总 affinity 写均为0。三个计划、launch、dispatch SHA及全部四项源码和五项登记绑定保持。

全局 exit 在 UTC1795508153.012024 记录 `ValueError('Raw PID/birth changed before clock mapping')`。精确 invocation 的 journal 将错误定位到 probe 第492行、创建新 intent 之前的 identity 校验；它本身不能区分退出、zombie 或 PID复用，不能称为已经证明PID复用。最后持久 attempt 仍是48，没有新增 intent 或未闭合的扩展。失败发生后未由本复核重启、发信号或修改任何进程。

46 已有独立的存活同birth worker 逐TID恢复复查；47 的历史全线程恢复依据冻结源码协议和持久 helper 摘要，本次没有伪造历史活体快照。最终两 worker 均已退出，故不存在其仍持借核状态运行的问题。逐TID扫描非原子，external撤回没有保存逐进程计数，不能据此独立归因竞争或宣称零干扰/提速。placement 可与物理初始化重叠；锁只约束后续 Python 初始化提交和控制，不证明隐藏物理状态相同。

本次只读审查 source、轻量运行回执、精确 service/journal、进程与cgroup；未读取 main64 科学结局或模型/缓存/轨迹载荷。该结论仅为19辅助进程带失败收尾，不能替代15 owner/watcher及主评估完整收尾。JSON含逐项核验和输入SHA，SHA256 `467919360fb179137906d0d81e56de502e8e72a559efb2429df0d81d50b2100c`。
