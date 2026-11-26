# Wave41：限时只读启动诊断

结论：当前是native Kit启动路径停在 `SimulationContext.play()` 的 `_app.update()`，8个 `carb.tasking0..7` 线程持续各吃满一核；不是GPU容量不足或CPU完全未获得调度的表象。支持native tasking忙等／活锁式停滞的怀疑，但没有native栈，不能确诊死锁、锁拥有者或根本原因。本轮不再做affinity／nice／线程／配置干预；让原冻结1800秒启动规则决定本wave退出，再由原队列推进下一未消耗wave。

只读核验实际owner15 stage24，worker PID3472362、birth1795501423.39、命令／cwd与原wave41一致。约1200秒后仍为 `environment_constructor_started`，启动traceback连续多个120秒转储均停于：

```text
Isaac Sim SimulationContext.play:917 -> self._app.update()
SimulationContext.reset:644
IsaacLab SimulationContext.reset:519
DirectRLEnv.__init__:167 -> self.sim.reset()
```

该 `play()` 源码文档明确“内部走一步以传播physics handles”；因此当前并非可证明无物理的纯加载区间。freeze的构造／physics／线程配置不作修改。

2.010262秒独立计数区间内，进程CPU为801.89%；8个carb.tasking线程各增加2.01 CPU秒、均running／wchan0。主Python线程sleeping／futex，其余多数线程在futex或poll。133线程当时mask均8..23、nice10。此区间实际read_bytes与write_bytes均增加0。GPU只读单次样本为利用率2%、已用4873MiB／空闲19211MiB；RAM可用约23.2GB，进程VmSwap为0，系统CPU／memory／I/O PSI avg10均0。系统swap虽满，但本区间没有支持“当前页换入／磁盘等待使其卡住”的证据；不能由单次样本排除全部历史压力。

原stage日志只有常见启动／URDF警告，最后可见scene／camera注册，未发现新的Python exception。powersave警告是原环境状态；不是本轮更改全局调频的授权或必要证据。没有额外启动Kit、读取成功结果、信号抓栈、ptrace、perf附着、改affinity或动外部进程。

建议本轮只保留原1800秒有界启动中止和已登记软件未完成分母，不重试wave41、不延长截止、不用新host“修复”改变冻结执行。之后若另开工程定位，可在当前终测结束后登记独立非终测启动样本，采native task栈／调度器profiling并对单一配置作有界对照；这些不在本次执行。当前数据只能回答“8核忙、主线程在native update等待、GPU空闲、非明显I/O瓶颈”，不能证明一种干预有效。

结构化回执 `STARTUP_STALL41_READONLY17.json` 保存两次逐线程／资源样本、原status、初始化SHA和traceback尾部SHA。SHA256：`45660d0383f5db70a51e8a67d0b7213f6afccd032a2a69c88de1966a65a9802d`。当前诊断不替代wave41后续真实软件退出或原64家族／256机会终局统计。

本次未做只读prewarm：累计read_chars约5.35GB而实际read_bytes仅约1.3MB，当前2秒实际读盘为0，主栈亦非文件加载；额外读取文件制造page-cache／I/O竞争缺少当前收益证据，且不会释放native task等待。
