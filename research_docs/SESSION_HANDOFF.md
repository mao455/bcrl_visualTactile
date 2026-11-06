# 当前交接：StateDP基础能力与动作块诊断

<!-- STATE_MECHANISM_START -->
## 2026-11-06：StateDP基础能力、采样与动作块诊断

实测 `running` / `oracle`；独立root `state_mechanism_20261106_01`，固定4h含等待，最多15次Kit轨迹启动、1条训练示范的独立随机模型≤4000更新/90min。正式数据/权重保持，A/B/C不改变H24/12/29D/损失/历史/物理。
只读审查275条/64,663tick：全体均值3.919s、中位3.433s、p10–p90 2.640–5.517s，范围2.117–12.900s。新增75均值4.676s、中位4.4s；尚不符合多数5–7s的设计预期，没有补等待或改时间戳。State-valid275、Multi-view-valid75、旧固定视角无效200；触觉数值有限及有效标记通过，不等于硬件校准。所有成功标签只在末transition出现，275条原始数值与缓存完全相等，q_after[t]=q_before[t+1]。

A/B/C尚未全部完成，不作机制结论；以已保存门控和错误为准。

实际A/B/C、采样曝光及最新状态见[StateDP统一结果](STATE_DP_RESULTS.md)和[机器统计](evidence/state_mechanism.json)。
<!-- STATE_MECHANISM_END -->

## 操作边界与实际状态

隔离worktree：`bc+rl_RSS_narrow_followup`，仅推送origin/feature/narrow-state-data。新root：`outputs/visuotactile/run/state_mechanism_20261106_01`。读取status.json、heartbeat.json、data_audit.json、gradient_gate.json、oracle_result.json、tiny_training/progress.json及closed_loop_result.json判断进度，不能用旧PID或旧“当前训练”段落推测。

后台服务：`state-mechanism-20261106-01.service`。只读查询用`systemctl --user show state-mechanism-20261106-01.service -p ActiveState -p MainPID -p Result`。启动入口为`bash visuotactile/run_cpu.sh -m tools.state_mechanism run --root outputs/visuotactile/run/state_mechanism_20261106_01`，只能首次使用；阶段目录/原deadline阻止重复owner和无审查重启。停止可写root/STOP；异常保留error/last，禁止直接重跑整个run。若需恢复，先审查完整checkpoint、已消费启动/更新预算和冻结源码，在同原截止内提出/实现一次明确的未完成后缀恢复；当前没有自动恢复命令。

顺序：CPU数据与sampler审查→真实梯度等价门控→A专家与位置Oracle→C离线随机性→B单轨迹独立训练→B/C共10次闭环→停止并发布简洁结果。一次性4h含等待，A5+B2+C8=15 starts；B≤4000更新/90min。启动后短监督，后续由普通独立进程完成，不由LLM持续轮询。失败不放宽门限、换种子或增加尝试。

## 已完成事实（不恢复旧任务）

- 原400条保持。原窄200条：201尝试/200成功，60家族44/8/8；5735更新，原train0/4、unseen0/10合法+2初态拒绝。固定相机空白，只有State-valid。
- 视觉/GPU精度修复后的扩充：76尝试/75成功/1掉落，32新家族，合并275；5000更新平台早停、best4500。最终新旧各train0/4、独立unseen0/10合法+2拒绝，32 starts完成，全部进程15:19:38退出。代码和收尾5604249。
- 历史末尾参数 formal_updates=0是汇总遗漏，已按真实trainer更正5000，原快照保留。两份一次性smoke已清理2.148GiB，正式best/last、全部成功及失败数据保留。
- 前轮整卡采样均值30.81%、完整60s最大52.19%；monitor交接空档2518.41s，不称完全合规/无缺测。旧预算和证据保持。
- 四条旧观看失败及六条修复后的最终视频位于原索引`outputs/visuotactile/demos/narrow_failures_20261106_01/README.md`。旧200+新75仍不能整体用于多视角训练。

更早的控制/初始化/采集审查和来源commit保留于[历史交接5604249](https://github.com/mao455/bcrl_visualTactile/blob/5604249/research_docs/SESSION_HANDOFF.md)、STATE_DP_RESULTS和SLOW_EXPERT_RESULTS。本轮不自动进入Image、触觉、DAgger或扩大数据。
