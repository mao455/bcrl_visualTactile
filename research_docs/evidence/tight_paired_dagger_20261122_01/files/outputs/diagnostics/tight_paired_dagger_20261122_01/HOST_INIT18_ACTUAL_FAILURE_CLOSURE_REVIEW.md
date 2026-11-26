# Host-init18 实际失败与收尾独立复核

18 已在任何 wave intent 或 affinity 路径前失败并退出，不需要执行恢复。独立检查到目录仅有 `started.json` 和 `exit.json`，`attempted=[]`、`durable_attempts=[]`；源码仍为 `7ac5bb977213647e8461fe8fc2e500eea1b48fa557a56dbeca2b3352983c3cc2`，其余三份冻结依赖也一致。

精确 unit 为 `tight-paired-host-init-capacity18.service`，InvocationID `11968f035ca74c34b6164b943fea402a`，ExecMainPID3529061、ExecMainStatus1、Result=exit-code、MainPID0。目标 helper PID 已不存在，精确模块进程 census 为空。journal 和 exit 均定位到 `Exact launch/birth fence differs`，在 probe 首次 identity 内、取锁和写 intent 前抛出。journal 中另有 inotify ENOSPC 警告，它不是这次 Python guard 异常的直接原因，也不能据此认定磁盘字节耗尽。

根因是 Linux psutil 的时间口径。其源码把 `/proc/stat` 的整数 `btime=1792940224` 加到 `/proc/3520447/stat` 的 `starttick=256493441` / `SC_CLK_TCK=100`，所以 raw create_time 为1795505158.41。实际 launch UTC 为1795505159.3052104，raw 差−0.8952103秒，违反原来的−0.05秒门限。launch 其余各项准确字段均通过。

五组 `wall_before → CLOCK_BOOTTIME → wall_after` 实测把 boot epoch 定位在1792940224.893455附近。计入 1/100 秒的进程起点 tick 量化后，映射的出生区间为 `[1795505159.303455, 1795505159.3134563]`；相对 launch 为 `[−0.0017552,+0.0082459]` 秒，与原门限兼容。run15 源码也确认先持久化 launch，再调用 Popen。因此问题是将整数 boot epoch 的 psutil birth 直接与 wall-clock launch 相减；不是 launch 顺序或 JSON 字段名错误。

19 的最小安全修复是新增源码和独立 run 身份，只把 chronology 改为有界校准后的 birth interval；原 raw PID/birth 对象身份比较、所有 status/immutable-launch/dispatch/owner/job/source 绑定、白名单、阶段及恢复逻辑保持。校准应保存并验证有限且窄的 wall/BOOTTIME 区间、starttick/HZ 和原 raw birth 一致性，包含 tick 量化，对时钟突变或无法界定的样本拒绝；不能把门限任意扩大成一秒。18 原源码、失败、激活和签署记录全部保留。

当前映射是事后测量，不能证明过去未发生 wall-clock step；19 必须在自身运行时登记并复核校准。原18合成源码签署没有覆盖主机 boot-epoch 量化，这是此次准入失败暴露的验证缺口。本次没有读取 main64 科学结局，没有发信号或修改任何实际 affinity/nice。逐项原始数字、psutil 源码、unit/journal、身份及哈希见同名 JSON，SHA256 `50881fbd8a4e85fd4233ec40cb6fe056aac075ae75ed0e429fc38eb206a188bb`。
