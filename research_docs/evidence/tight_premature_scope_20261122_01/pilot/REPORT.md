# 冻结入口模型：训练 family 对准停滞接管小试

| 组 | 同场自主完整插入 | 接管机会 | 实际触发 | 触发后专家救回 | 未触发 |
|---|---:|---:|---:|---:|---:|
| A | 1/4 | 4 | 0 | 0/0 | 4 |
| B | 2/4 | 4 | 1 | 1/1 | 3 |

Owner实际开始 2026-11-22T15:59:02.864206+08:00，关闭回执 2026-11-22T16:24:38.215986+08:00，历时 1535.35 秒。

自主对照离线复算同一alignment屏幕：A命中1/4，其中最终自主成功0例；B命中1/4，其中最终自主成功0例。屏幕命中不等于已证明有害错误。

本表专家救回不是学生自主成功。16机会均来自预先指定的4个原训练family；0训练更新、0入训练池。
触发规则直接复用既有adaptive alignment：连续12tick持物、距目标root≤90mm、keypoint>15mm、36tick关键点改善不足1.5mm、连续2tick确认，stage0后下一动作接管，无定时fallback。
registration 的 trigger_before_native_15mm_gate=True 只表示触发当下 keypoint>15mm 且 goal0，不保证此前从未进入15mm门；请看逐例 gate15mm_previously_entered。
该触发筛选当前 keypoint>15mm 的转正/对准停滞，不将短暂离门普遍视为错误。各次触发朝向、关键点、接触、是否曾入门及后续原生成功见opportunities.csv。
自主对照与接管臂均运行同一完整学生、专家shadow和记录路径；自主臂从未获得专家权限，逐tick命令链审计通过。两臂可观测初态配对通过，但不主张隐藏物理或后续随机流精确配对；pre-switch数值差异见summary.json。
资源遥测的known-entry识别未覆盖自定义pilot_entry，原始日志可将自有worker记为owned=false；生命周期按登记的owner/worker/monitor PID独立核验，资源采样见resource_summary.json。
小试family依据既有训练alignment覆盖选取，未用终测挑选，也不是泛化成功率估计。所有权重、数据manifest、统计、numeric及4条源原轨SHA前后不变。
