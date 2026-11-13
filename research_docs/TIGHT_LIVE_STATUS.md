# Tight-Insertion 自动进度

证据读取时间：2026-11-13T23:19:20+08:00。

本页由独立 CPU 脚本更新；只发布有凭据的新进展，不启动训练或解释科研结论。
每分钟检查，有变化时通常五分钟内推送；阶段完成/失败优先推送。原截止北京时间 11月14日 00:05:15 不变。

当前阶段：`rear_span_dose_02`；状态：`{"arm": "continued", "group": "QG01", "stage": "complete_student_development", "state": "RUNNING"}`。
完成凭据：False；失败凭据：False。

公平规则见 [AB_COMPARISON_POLICY](AB_COMPARISON_POLICY.md)。最近 contact 周期增量审计：通过；历史初始化/曝光仍不等同。

| 评估队列 | 范围 | 已完成/计划 | 合法 | 拒绝 | 已观察成功 | 整队列完成 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `ab_replication_completion_01/batch_00_contact/QG01` | new stage; A/B comparability requires its own audit | 10/10 | 10 | 0 | 0 | True |
| `ab_replication_completion_01/batch_01_contact/QG01` | new stage; A/B comparability requires its own audit | 10/10 | 10 | 0 | 0 | True |
| `adaptive_centered_01/formal/G01/backend/G01` | new stage; A/B comparability requires its own audit | 10/10 | 10 | 0 | 0 | True |
| `adaptive_centered_01/formal/QG01/backend/QG01` | new stage; A/B comparability requires its own audit | 10/10 | 10 | 0 | 0 | True |
| `adaptive_contact_recovery_01/round_01/G01/backend/G01` | A/B matched increment | 10/10 | 10 | 0 | 1 | True |
| `adaptive_contact_recovery_01/round_01/QG01/backend/QG01` | A/B matched increment | 10/10 | 10 | 0 | 0 | True |
| `adaptive_tick_window_01/formal/G01/backend/G01` | new stage; A/B comparability requires its own audit | 10/10 | 10 | 0 | 0 | True |
| `adaptive_tick_window_01/formal/QG01/backend/QG01` | new stage; A/B comparability requires its own audit | 10/10 | 10 | 0 | 0 | True |
| `rear_replay_balance_01/control/G01/backend/G01` | new stage; A/B comparability requires its own audit | 10/10 | 10 | 0 | 0 | True |
| `rear_replay_balance_01/control/QG01/backend/QG01` | new stage; A/B comparability requires its own audit | 10/10 | 10 | 0 | 0 | True |
| `rear_replay_balance_01/population/G01/backend/G01` | new stage; A/B comparability requires its own audit | 10/10 | 10 | 0 | 0 | True |
| `rear_replay_balance_01/population/QG01/backend/QG01` | new stage; A/B comparability requires its own audit | 10/10 | 10 | 0 | 0 | True |
| `rear_span_dose_02/continued/G01/backend/G01` | new stage; A/B comparability requires its own audit | 10/10 | 10 | 0 | 0 | True |
| `rear_span_dose_02/continued/QG01/backend/QG01` | new stage; A/B comparability requires its own audit | 8/10 | 8 | 0 | 0 | False |
| `rear_span_dose_02/reference/QG01/backend/QG01` | new stage; A/B comparability requires its own audit | 10/10 | 10 | 0 | 0 | True |
| `rear_span_metric_eval_recovery_02/reference/G01/backend/G01` | new stage; A/B comparability requires its own audit | 10/10 | 9 | 1 | 0 | True |
| `rear_span_metric_eval_recovery_02/reference/QG01/backend/QG01` | new stage; A/B comparability requires its own audit | 10/10 | 9 | 1 | 0 | True |
| `rear_span_metric_eval_recovery_02/span/G01/backend/G01` | new stage; A/B comparability requires its own audit | 10/10 | 9 | 1 | 0 | True |
| `rear_span_metric_eval_recovery_02/span/QG01/backend/QG01` | new stage; A/B comparability requires its own audit | 10/10 | 9 | 1 | 0 | True |
| `success_replication_01/batch_00_contact/G01` | B current/previous only; not A/B | 10/10 | 10 | 0 | 0 | True |
| `success_replication_01/batch_00_previous/G01` | B current/previous only; not A/B | 10/10 | 10 | 0 | 0 | True |
| `success_replication_01/batch_01_contact/G01` | B current/previous only; not A/B | 10/10 | 10 | 0 | 0 | True |
| `success_replication_01/batch_01_previous/G01` | B current/previous only; not A/B | 10/10 | 10 | 0 | 0 | True |

未完成队列只报告进度，不计算最终成功率。B 新旧模型复验不能用于与 A 排名，也不加入原 A0/10、B1/10 的分母。

训练收据可能由参考模型继承，不把跨目录重复收据累计为新增训练。

| 训练收据 | 更新 | 有效监督量 | 本阶段新增 |
| --- | ---: | ---: | --- |
| `adaptive_centered_01/formal/G01/backend/G01` | 2349 | 171133 | True |
| `adaptive_centered_01/formal/QG01/backend/QG01` | 2349 | 171133 | True |
| `adaptive_contact_recovery_01/round_01/G01/backend/G01` | 1362 | 193581 | True |
| `adaptive_contact_recovery_01/round_01/G01/dp` | 514 | 765303 | True |
| `adaptive_contact_recovery_01/round_01/QG01/backend/QG01` | 1362 | 193581 | True |
| `adaptive_contact_recovery_01/round_01/QG01/dp` | 514 | 765303 | True |
| `adaptive_tick_window_01/formal/G01/backend/G01` | 1191 | 171133 | True |
| `adaptive_tick_window_01/formal/QG01/backend/QG01` | 1191 | 171133 | True |
| `rear_replay_balance_01/control/G01/backend/G01` | 1363 | 193690 | True |
| `rear_replay_balance_01/control/QG01/backend/QG01` | 1363 | 193690 | True |
| `rear_replay_balance_01/population/G01/backend/G01` | 1314 | 193520 | True |
| `rear_replay_balance_01/population/QG01/backend/QG01` | 1314 | 193520 | True |
| `rear_span_dose_02/continued/G01/backend/G01` | 2620 | 387040 | True |
| `rear_span_dose_02/continued/QG01/backend/QG01` | 2620 | 387040 | True |
| `rear_span_dose_02/reference/G01/backend/G01` | 1314 | 193520 | False |
| `rear_span_dose_02/reference/QG01/backend/QG01` | 1314 | 193520 | False |
| `rear_span_metric_eval_recovery_02/reference/G01/backend/G01` | 1314 | 193520 | False |
| `rear_span_metric_eval_recovery_02/reference/QG01/backend/QG01` | 1314 | 193520 | False |
| `rear_span_metric_eval_recovery_02/span/G01/backend/G01` | 1314 | 193520 | True |
| `rear_span_metric_eval_recovery_02/span/QG01/backend/QG01` | 1314 | 193520 | True |

未通过凭据检查的条目：0；此类条目不发布成功统计。

[机器可读证据](evidence/tight_execution_learning_20261113/live_status.json)。脚本不上传原数据、模型或视频；网络失败保留待发布状态，恢复后重试。
