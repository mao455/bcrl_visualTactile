# Wave47 constructor：只读主机诊断

wave47 已自行完成 constructor，耗时536.132秒，其中日志的 simulation start 为531.800秒；相邻wave48 constructor 为72.882秒，随后也ready。收尾观察时owner已到wave50，因此没有仍需解锁的wave47卡死证据。

4次自动120秒Python栈都停在 `SimulationContext.play → reset` 的原生调用边界。旧wave41已有8个 `carb.tasking` 线程各约100%CPU、低I/O的只读记录；但既有native attach记录失败，没有可用native帧，不能定位死锁、线程池内部自旋或特定PhysX任务。Kit真实启动参数已经固定carb线程数8、physics线程数2；只增加affinity不保证增大这些池。此次没有attach、发信号或改进程。

发现两处可验证的host19调度局限：wave46/47都在约0.44秒因external aggregate撤回；判据统计所有affinity与借核相交进程的**全进程**CPU增量，不能证明增量发生在借出的核。回执未保留逐进程external原始计数，无法把这两次归因为editor。wave48则是明确排名问题：最初top2为CPU4/2，锁内top2变为CPU5/1，取交集后为空，尽管三个样本中四个候选核全部低于15%。其零写no-op不能解释为实际容量耗尽。

另一个独立资源问题是inotify watch接近上限：当前UID的fdinfo watch求和524180，上限524288；两个code进程分别约487716和36006。这个求和可能重复共享/继承instance，不能当作精确全局唯一数，但与systemd的watch ENOSPC一致；87个有watch的FD远低于4096 instance上限。没有证据把它因果连接到constructor慢，且Kit已设置extension fsWatcher=false；不据此改外部editor或全局limit。

最小的后续host调度候选是从**全部**通过原两次样本的候选里，在锁内重新挑最多两核，避免top2交集浪费；撤回同时记录借核实际压力与external原始计数，避免把其他核上的活动当成借核争用。它需要新的独立helper身份与验证，不能修改运行中的19或保证能缩短constructor。本诊断不启动20、不改线程池/物理/噪声/成功判据，也不重跑机会；没有足够新证据支持进一步native原因判断，本项诊断结束。

未打开result或轨迹载荷；最初读取stage30尾部时包含一行无success字段的aggregate性能profile，此后仅提取startup/stack行，未用于模型判断。0项新测试。原始阶段时间、自动栈边界、inotify逐进程计数、线程快照、实际19回执和SHA见同名JSON，SHA256 `cbce0bde0a38c167b0fe461bba7586e2914dd79845e73d5c7cb6c5854472621e`。
