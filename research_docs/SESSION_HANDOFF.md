# 当前交接：窄分布采集被准入一致性阻断

2026-11-05本轮已实际执行两次持久Kit启动，服务与所有自有worker/monitor均退出。
没有通过短监督，不能写“稳定后台采集”。完整事实与命令见[SLOW_EXPERT_RESULTS](SLOW_EXPERT_RESULTS.md)，
机器证据见[evidence/narrow_startup.json](evidence/narrow_startup.json)。

- 工作树`/home/CNS2025331827/docs/bc+rl_RSS_narrow_state`，分支`feature/narrow-state-data`。
- root`outputs/visuotactile/run/narrow_state_20261105_01`；服务`narrow-state-data-20261105-01.service`已退出。
- 条件域筛出15旧家族，补9个无动作native reset候选，共24（16/4/4）。预检暂准入23（15/4/4），随后一致性失败，不能认为准入已可靠。
- 6次记录：原速成功171/210ticks，alpha=.58成功366ticks（与210参考比1.743倍）；另3次0动作准入拒绝，非任务失败。
- 同一家族q/qd、物体/孔/桌、手掌/指尖8组初态完全相同，碰撞查询却从无命中变成手部多网格命中。原因未解决，禁止放宽合法性门槛或绕过物理。
- 3份RGB-D/触觉真实完整数据及所有失败/冻结证据保留；manifest.quarantined=true、data_ready=false、COMPLETE不存在。生产阶段未放行。
- 原StateDP一次CPU随机初始化优化/EMA/AdamW恢复检查通过，正式训练0更新；图像缓存/触觉训练/旧评估都未启动。
- 原4h截止2026-11-05 22:11:24 UTC+8，2次worker额度耗尽，不重置、不自动重试。最终保护代码会拒绝隔离root恢复，并在初态相同而准入矛盾时立即停止。
- 运行源码6a1a0aa、2e7ab94；其后的诊断/门控修改未进行新live验收。旧source_freeze留在root及maintenance，不回填历史身份。
- 原400存在/大小及manifest SHA核验通过。清理工具因未读进程句柄保护跳过一次性checkpoint，本轮释放0bytes。
- 整卡实测均值14.18%、最大完整60秒19.56%；202.11秒停机缺测保留。原IDE两处既有文档改动未改写、未提交。

只读状态：`bash visuotactile/run_cpu.sh -m tools.narrow_collection status --root outputs/visuotactile/run/narrow_state_20261105_01`。
原State143评估继续paused；旧expert capture blocked；不得恢复这些旧队列。
下一次先审查持久reset/PhysX query矛盾的具体修复，再决定新的有界live验收；不直接正式训练。
