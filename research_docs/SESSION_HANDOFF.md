## 2026-11-06 数据可信度审计完成：停止并交由用户决定

2026-11-06当前审计已完成：`dataset_trust_20261106_01/status.json` 为audit_complete/await_user_decision。全275/64663tick及全部RGB-D像素审核，State/触觉数值275、三视角75；旧200空白保留。主56+fresh4=60/60轨迹，所有GPU服务已退出，0训练/0学生rollout/0扩采；不要自动追加任何旧/新实验。原速专家15/15、来源慢速8/9，tick/H24主队列6/15及4/15，额外两来源fresh各两模式全成功且18字段逐位相同，不能混池或称历史T0精确恢复。275/277是筛选后采集频率，不是连续分布成功率；实际慢速配对中位1.231倍、5–7s仅21/275。275数据技术准入StateDP，是否重训/补数据由用户决定。代码未见网络/损失/原生控制暗改，但短尾batch/fresh训练/TF32有真实变化，早期三版门控源码无法完整追溯。原400/275及比较模型保护通过。详见SLOW_EXPERT_RESULTS、STATE_DP_RESULTS及evidence/dataset_trust；两阶段间231.662s无自有GPU任务的遥测断点明确保留，不宣称整段无缺口。

## 审计启动时的授权与预算（已完成历史）

先重读研究目标与冻结历史，逐条审核275条原始数据全部模态、分布、减速、成功/失败及采样偏差；基点84cbb98至0c209b9做真实Git/AST/运行SHA核查。静态通过后独立分层专家/逐tick target/H24回放，root dataset_trust_20261106_01，固定3h含准备等待、最多60轨迹启动、8GiB新增/60GiB磁盘保留、共享目标35%及6GiB VRAM/12GiB RAM。仅CPU审核和冻结RL专家/记录命令，正式数据新增0、学生rollout0、optimizer更新0。所有原400/275、manifest、正式checkpoint和失败证据只读。审计报告和准入建议为终点，用户决定下一阶段；不自动扩采/重训/学生误差调参/Image/nominal/触觉。H24/12、5/60/120Hz、target_after_rad而非q_after、29D、网络/损失/归一化/历史/原生物理保持。新脚本独立，原始验收标记不改写。

启动快照已由上述完成状态取代。最终只读入口为status/final_audit/static_summary/source_audit/physical_details/fresh_pair_details；不运行旧owner。

## 以下为已完成机制诊断历史

# 当前交接：StateDP基础能力与动作块诊断

<!-- STATE_MECHANISM_START -->
## 2026-11-06：StateDP基础能力、采样与动作块诊断

实测 `complete` / `complete`；独立root `state_mechanism_20261106_01`，固定4h含等待，最多15次Kit轨迹启动、1条训练示范的独立随机模型≤4000更新/90min。正式数据/权重保持，A/B/C不改变H24/12/29D/损失/历史/物理。
只读审查275条/64,663tick：全体均值3.919s、中位3.433s、p10–p90 2.640–5.517s，范围2.117–12.900s。新增75均值4.676s、中位4.4s；尚不符合多数5–7s的设计预期，没有补等待或改时间戳。State-valid275、Multi-view-valid75、旧固定视角无效200；触觉数值有限及有效标记通过，不等于硬件校准。所有成功标签只在末transition出现，275条原始数值与缓存完全相等，q_after[t]=q_before[t+1]。

极小集离线拟合通过但对应初态仍未成功；精准目标回放接口检查已通过，优先研究执行后状态偏离teacher-forced训练轨迹这一机制。下一轮只做有限闭环分布偏移定位；本轮不自动DAgger或改方法。噪声/块跳变的因果归属仍受小样本限制。

实际A/B/C、采样曝光及最新状态见[StateDP统一结果](STATE_DP_RESULTS.md)和[机器统计](evidence/state_mechanism.json)。
<!-- STATE_MECHANISM_END -->

## 操作边界与实际状态

隔离worktree：`bc+rl_RSS_narrow_followup`，仅推送origin/feature/narrow-state-data。新root：`outputs/visuotactile/run/state_mechanism_20261106_01`。读取status.json、heartbeat.json、data_audit.json、gradient_gate.json、oracle_result.json、tiny_training/progress.json及closed_loop_result.json判断进度，不能用旧PID或旧“当前训练”段落推测。

本轮已于19:39:53正常结束，服务`state-mechanism-20261106-01.service` inactive/success，全部15启动、800次tiny更新已完成。只读查询可用`systemctl --user show state-mechanism-20261106-01.service -p ActiveState -p MainPID -p Result`；不要重新执行run、重设23:00截止或消耗剩余预算。没有待恢复阶段。

A映射通过；B离线拟合通过但0/2闭环；C0/C4各0/4、各2次native lift、零子目标。下一研究建议仅为闭环偏离/误差放大定位，尚未授权自动进入新实验。初态/前缀误差、来源字段说明、资源超50%窗口及保护hash详见STATE_DP_RESULTS“20:11复核”和final_review.json。保留tiny best/last及全部8视频；当前没有应删除的废弃模型。

## 已完成事实（不恢复旧任务）

- 原400条保持。原窄200条：201尝试/200成功，60家族44/8/8；5735更新，原train0/4、unseen0/10合法+2初态拒绝。固定相机空白，只有State-valid。
- 视觉/GPU精度修复后的扩充：76尝试/75成功/1掉落，32新家族，合并275；5000更新平台早停、best4500。最终新旧各train0/4、独立unseen0/10合法+2拒绝，32 starts完成，全部进程15:19:38退出。代码和收尾5604249。
- 历史末尾参数 formal_updates=0是汇总遗漏，已按真实trainer更正5000，原快照保留。两份一次性smoke已清理2.148GiB，正式best/last、全部成功及失败数据保留。
- 前轮整卡采样均值30.81%、完整60s最大52.19%；monitor交接空档2518.41s，不称完全合规/无缺测。旧预算和证据保持。
- 四条旧观看失败及六条修复后的最终视频位于原索引`outputs/visuotactile/demos/narrow_failures_20261106_01/README.md`。旧200+新75仍不能整体用于多视角训练。

更早的控制/初始化/采集审查和来源commit保留于[历史交接5604249](https://github.com/mao455/bcrl_visualTactile/blob/5604249/research_docs/SESSION_HANDOFF.md)、STATE_DP_RESULTS和SLOW_EXPERT_RESULTS。本轮不自动进入Image、触觉、DAgger或扩大数据。
