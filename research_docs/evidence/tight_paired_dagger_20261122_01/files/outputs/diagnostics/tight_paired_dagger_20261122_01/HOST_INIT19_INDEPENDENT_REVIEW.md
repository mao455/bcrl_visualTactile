# Host-init19 独立源码与时钟接口复核

结论：新19源码及合成夹具通过，无剩余源码阻塞项；没有激活19。冻结源码 SHA256：`c7374e679a2a67156fc1ce942852cc1aee9978d695e185e94d9870ae7dda56b9`。

三套19 CPU 合成测试 **149 passed in 0.25s**：作者75、本独立25、并行数学复核49。四个新 Python 文件以 `compile(bytes, path, 'exec')` 检查通过。独立 AST 检查确认：相对冻结18，原函数中仅 `source_guard`、`receipt_birth` 和 `identity` 改变，另加五个 clock helper；采样余量、借核范围、锁、逐线程扩展/恢复、异常/STOP、有限波次及主循环逐函数保持一致。

19修复18实际暴露的整数 boot epoch 口径差：`raw psutil birth` 仍只用于 PID/birth 身份，chronology 改用 `/proc/PID/stat` 的 starttick/HZ 和独立 `wall_before → CLOCK_BOOTTIME → wall_after` 校准后的出生区间。检查 raw birth 与整数 btime+tick/HZ 一致，且映射前后重新检查 exact live identity；完整 kernel 字段、时钟样本及映射上下界进入 identity evidence。没有任意一秒容差。

receipt 与 launch 都要求**整段**可行出生区间落在原18域内：receipt `[utc−120秒,utc+0.05秒]`，launch `[utc−0.05秒,utc+120秒]`。只相交的区间拒绝；数值/布尔类型、有限性、采样反向或超过50ms、错误 starttick/HZ/btime、raw birth/PID 变化均拒绝。合成回归重现约0.893秒整数 btime 偏差并通过新校准，同时保留旧status下新worker、owner复用、launch/dispatch/job/source改动的拒绝。

149项包含数学边界、整区间包含的反例、量化与样本宽度、完整配套身份和继承18的恢复/错误路径。审查只读取指定源码、说明和临时合成夹具；19阶段没有读取实际 lifecycle、科学结局、模型/缓存/轨迹载荷，未查询真实进程或修改 affinity/nice、发送信号。18实际失败审查单独保留，未改写18源码、签署或运行证据。

当前时钟采样界定的是**当前** realtime−BOOTTIME offset。把它用于历史 launch/status/dispatch UTC，仍以这些事件间 offset 稳定为适用前提。此前没有保存历史 clock anchor，已有 launch/status/owner双向检查能排除部分不一致，但不能证明任意历史 wall-clock step 都未发生；在容差内的较早跳变可能无法区分。正常 Linux NTP 频率调整本身不等同于这种 offset 跳变。这是证据适用范围，不是新增审批条件。

实际 registry/review/owner绑定及激活由 root 另核。实际19收尾应逐 intent 对照扩展、恢复或 exact worker exit、helper service及进程 census，保留失败并单独签署；本源码签署和 closure15 均不能代替19实际收尾。物理 constructor 重叠已由用户授权，lifecycle锁仍只保证 Python initialization commit/学生控制前恢复；没有增加物理前审批。既有 SIGKILL/主机丢失、非原子线程扫描、40秒扩展预算可能被恢复耗时超出的限制保留。
