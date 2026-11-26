# A/B 配对 DAgger 一轮结果草稿（待主执行者审阅）

A 新版 0/64（0.0%），净差为零、仍有不确定性；B 新版 0/64（0.0%），净差为零、仍有不确定性。两组均达到预登记50%工程门槛：否；50%仅是本轮工程里程碑，不能直接宣称部署可用。A−B=+0.0 pp，配对95%区间[0.0, 0.0] pp。

下一步优先定位首子目标前的最终下发命令与实际反馈：A/B 尚未达首子目标各64/64例；具体纠偏类别应结合本轮开发记录复核后确定。该建议仅据阶段分母定位，未宣称因果瓶颈。

| 完整学生 | 严格完整插入 / 全部 | Wilson 95%区间 | 软件未完成 | 协议未核验 |
| --- | ---: | --- | ---: | ---: |
| A_new | 0/64（0.0%） | [0.0%, 5.7%] | 9 | 0 |
| B_new | 0/64（0.0%） | [0.0%, 5.7%] | 9 | 0 |
| A_parent | 0/64（0.0%） | [0.0%, 5.7%] | 9 | 0 |
| B_parent | 0/64（0.0%） | [0.0%, 5.7%] | 9 | 0 |

| 配对比较 | 净差 | 配对95%区间 | 双成功 | 新增/左胜 | 丢失/右胜 |
| --- | ---: | --- | ---: | ---: | ---: |
| A_new_minus_B_new | +0.0 pp | [0.0, 0.0] pp | 0 | 0 | 0 |
| A_new_minus_parent | +0.0 pp | [0.0, 0.0] pp | 0 | 0 | 0 |
| B_new_minus_parent | +0.0 pp | [0.0, 0.0] pp | 0 | 0 | 0 |

绝对率区间为Wilson；差值按64个家族整对重采样200000次、固定seed20261122。宽区间仍允许反向效果。软件未完成、协议缺失或初态不合格均按事前全体分母计失败；逐轨原因保留。专家恢复不计学生自主成功。

| 新版 / 全体64 | 稳定持物代理 | 曾进入15mm门 | 原生首子目标累计10tick确认 | 最终插入 |
| --- | ---: | ---: | ---: | ---: |
| A_new | 17/64 | 0/64 | 0/64 | 0/64 |
| B_new | 12/64 | 0/64 | 0/64 | 0/64 |
| A_parent | 19/64 | 0/64 | 0/64 | 0/64 |
| B_parent | 16/64 | 0/64 | 0/64 | 0/64 |

上表各列分母均为全部64，阶段不是严格嵌套漏斗。稳定持物为既有held12物理代理；15mm进门不等于累计10tick首子目标确认。保留曾进门又离开与从未进门标记。

共享专家示范接受24/24家族，实际尝试28次；失败、拒绝与重试见experts_shared/results.json。

| 组 | 学生机会 | 未触发 | 接管 | 专家恢复 | 有效纠偏 | 合法专家tick / 准入tick |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 48 | 18 | 25 | 24 | 24 | 7203 / 6721 |
| B | 48 | 17 | 28 | 27 | 27 | 8303 / 7801 |

采集可观测初态：47/48对已核验，1/48对未核验，失败0对；不声称48对全部通过。中断前15次机会均已消耗，只继续原登记尚未分配的81次，未重跑已启动机会。

| 组 / 全部48机会 | 软件未完成 | 初态非法 | 初态合法性未知 |
| --- | ---: | ---: | ---: |
| A | 5 | 0 | 0 |
| B | 3 | 0 | 0 |

上述中断、非法与未核验机会均保留原分母，未当作未触发。只有已核验完整捕获的合法专家后缀可进入训练；保存了RNG端点但执行记录不完整的机会，不据此认证完整噪声消费。

| 组 / 触发原因 | 机会 | 接管 | 恢复 | never-entry | 曾进门又离开 | 合法tick |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A / contact_stall | 16 | 16 | 15 | 16 | 0 | 4986 |
| A / registered_conditions_not_reached | 18 | 0 | 0 | 0 | 0 | 0 |
| A / software_incomplete | 5 | 0 | 0 | 0 | 0 | 0 |
| A / support_instability | 9 | 9 | 9 | 9 | 0 | 2217 |
| B / contact_stall | 21 | 21 | 20 | 21 | 0 | 6390 |
| B / registered_conditions_not_reached | 17 | 0 | 0 | 0 | 0 | 0 |
| B / support_instability | 7 | 7 | 7 | 7 | 0 | 1913 |
| B / software_incomplete | 3 | 0 | 0 | 0 | 0 | 0 |

实际数据可不同；薄弱阶段由上述实际分母判断，不因触发少而追加机会。学生真实前缀保留作历史输入，仅合法专家后缀进入动作监督。

历史entry本轮前端均epoch50，但祖先年龄未能对齐，因此按登记从共同180、相同初始化规则重建完整父本；历史130-50和entry均保留。父本与本轮各四组件均按登记部署epoch60，后端使用各自选中新DP重建的nominal缓存。

| 阶段 / 组 / 组件 | 实际更新 | 最低合法tick成功step | 最低原生入口slot成功step |
| --- | ---: | ---: | ---: |
| common180_parents / A / frontend | 7200 | 60 | 60 |
| common180_parents / A / backend | 22763 | 60 | 不适用 |
| common180_parents / B / frontend | 7200 | 60 | 60 |
| common180_parents / B / backend | 22763 | 60 | 不适用 |
| learning / A / frontend | 10200 | 60 | 60 |
| learning / A / backend | 38400 | 60 | 不适用 |
| learning / B / frontend | 10200 | 60 | 60 |
| learning / B / backend | 38400 | 60 | 不适用 |

曝光从实际部署checkpoint的逐成功更新ledger重建，重复窗口、padding、无标签交接和祖先不计。覆盖到监督不等于学会控制。两组分别保存旧完整/旧纠偏/共享新完整/本组新纠偏分区和实际采样，见pool_partitions.json。

| 组 / 数据分区 | 轨迹数 | DP每epoch实际登记访问数 |
| --- | ---: | ---: |
| A / old_full | 30 | 94 |
| A / old_correction | 150 | 150 |
| A / new_full | 24 | 72 |
| A / own_new_correction | 24 | 24 |
| B / old_full | 30 | 91 |
| B / old_correction | 150 | 150 |
| B / new_full | 24 | 72 |
| B / own_new_correction | 27 | 27 |

训练是否仍改善：curve_trends.json按相同组件、相同split比较epoch41–50与51–60的描述误差，完整家族/阶段曲线在各组件family_phase_curves.csv；该诊断不选模。DP训练与验证预测协议不同，不直接比较两条曲线绝对值；A学q_reached、B学u_expert，不用nominal损失给A/B排名。泛化是否改善只由上方独立64自主结果与不确定性回答。

编码器便宜定位没有拟合：旧36家族已参加读出；独立96条开发记录没有RGB-D/特征载荷；另两family短前缀有局部与聚合缓存但近孔tick为0，仅33个持物代理tick，无法完成合格的独立阶段比较。未启动四套编码器重训。

本轮检验“扩初态＋自身纠偏＋配对训练”整体；没有仅回放旧数据且等更新的A/B对照，不能单独估计新数据贡献。完整DP＋Transformer、原生120/60/5Hz、三视角逐帧RGB-D、五指力/FK、原可达投影/归一化/严格成功判据保持。初态只声称可观测量配对，不声称隐藏PhysX或未来随机流逐位相同。

| 本轮组件 / split | epoch41–50误差均值 | epoch51–60误差均值 | 相对变化 |
| --- | ---: | ---: | ---: |
| learning/A/frontend / train_evaluation | 0.0008286 | 0.000683074 | -17.6% |
| learning/A/frontend / train_updates | 0.00217129 | 0.00122794 | -43.4% |
| learning/A/frontend / validation | 0.00363707 | 0.00363995 | +0.1% |
| learning/A/backend / train_evaluation | 9.46875e-05 | 8.23738e-05 | -13.0% |
| learning/A/backend / train_updates | 0.000127161 | 0.000113356 | -10.9% |
| learning/A/backend / validation | 0.000269257 | 0.000275973 | +2.5% |
| learning/B/frontend / train_evaluation | 0.000811954 | 0.000681006 | -16.1% |
| learning/B/frontend / train_updates | 0.00222624 | 0.00131929 | -40.7% |
| learning/B/frontend / validation | 0.00407833 | 0.00406671 | -0.3% |
| learning/B/backend / train_evaluation | 8.38135e-05 | 7.34712e-05 | -12.3% |
| learning/B/backend / train_updates | 0.000124547 | 0.000112008 | -10.1% |
| learning/B/backend / validation | 0.000361014 | 0.000361053 | +0.0% |

负变化表示该组件该split的后10epoch描述误差下降，正变化表示上升；只是趋势，不等于收敛证明。验证为既有validation6；自主新64的泛化结论仍以上述严格插入率及区间为准。

登记开始2026-11-22T17:16:11.226184+08:00；终测汇总2026-11-26T09:51:03.320878+08:00；墙钟间隔88.58h（包含等待）。

| 可观测owner尝试 | 开始（+08） | 退出（+08） | 闭合墙钟小时 | 状态 |
| --- | --- | --- | ---: | --- |
| backend_recovery07 | 2026-11-23T22:33:56.961920+08:00 | 2026-11-23T23:03:10.478882+08:00 | 0.487 | closed |
| backend_recovery08 | 2026-11-23T23:37:59.164977+08:00 | 2026-11-23T23:38:22.240114+08:00 | 0.006 | closed |
| backend_recovery09 | 2026-11-24T01:38:38.391948+08:00 | 2026-11-24T01:41:28.494250+08:00 | 0.047 | closed |
| backend_recovery10 | 2026-11-24T01:50:00.982610+08:00 | 2026-11-24T06:54:00.434096+08:00 | 5.067 | closed |
| collection | 2026-11-23T06:02:35.987851+08:00 | 尚无闭合回执 | 未计入 | not_closed_or_unreaped |
| collection_attempt01_prephysics | 2026-11-23T06:00:19.084734+08:00 | 2026-11-23T06:00:26.158181+08:00 | 0.002 | closed |
| collection_recovery05 | 2026-11-23T09:53:24.747209+08:00 | 2026-11-23T12:37:06.156415+08:00 | 2.728 | closed |
| experts | 2026-11-22T17:17:05.889313+08:00 | 2026-11-22T17:38:42.236933+08:00 | 0.360 | closed |
| experts_admission01 | 2026-11-22T17:38:59.930445+08:00 | 2026-11-22T17:40:18.013309+08:00 | 0.022 | closed |
| main_cache | 2026-11-23T13:25:54.808870+08:00 | 2026-11-23T13:34:09.043662+08:00 | 0.137 | closed |
| main_train | 2026-11-23T13:36:40.671385+08:00 | 2026-11-23T22:14:27.795212+08:00 | 8.630 | closed |
| parent_train | 2026-11-22T18:01:10.923715+08:00 | 2026-11-22T22:45:08.132830+08:00 | 4.733 | closed |
| parent_train_recovery01 | 2026-11-22T22:50:22.629357+08:00 | 2026-11-23T05:56:13.787282+08:00 | 7.098 | closed |
| shared_demo_cache | 2026-11-22T17:44:27.446467+08:00 | 2026-11-22T17:54:30.643405+08:00 | 0.168 | closed |
| test_continuation15 | 2026-11-24T11:02:21.907558+08:00 | 2026-11-26T09:50:50.765986+08:00 | 46.808 | closed |
| test_fragment13 | 2026-11-24T10:20:09.168062+08:00 | 2026-11-24T10:38:20.481896+08:00 | 0.303 | closed |
| evaluation/development | 2026-11-24T06:55:47.435669+08:00 | 2026-11-24T08:03:41.461279+08:00 | 1.132 | closed |
| evaluation/test | 2026-11-24T08:16:15.358838+08:00 | 2026-11-24T09:53:44.775744+08:00 | 1.625 | closed |

已闭合owner的可观测时间区间并集79.352h；重叠owner壁钟只计一次，未闭合/未回收owner 1个不纳入该并集。该时间含准备、等待、缓存、验证和失败尝试，不能解释为GPU有效训练时间或GPU小时。

| 组件结束回执计时（单独诊断） | 日志wall小时 |
| --- | ---: |
| common180_parents/A/frontend | 0.368 |
| common180_parents/A/backend | 1.611 |
| common180_parents/B/frontend | 3.338 |
| common180_parents/B/backend | 1.620 |
| learning/A/frontend | 4.377 |
| learning/A/backend | 3.886 |
| learning/B/frontend | 4.239 |
| learning/B/backend | 3.778 |

DP finished.wall_seconds在恢复后重新计时；上表仅为结束回执提供的计时，各组件/各attempt计时范围可能不同，不相加、不与owner壁钟相加，也不称累计完整训练耗时。原parent_train与parent_train_recovery01已在owner表分列，其他后续recovery同样按独立attempt保留。

技术中断 common180_parents/A/frontend：从epoch57/update6840恢复；4步未持久化更新（ID 6841, 6842, 6843, 6844）确已消耗计算，原始弃用日志保留，但对最终选中模型训练年龄与曝光的贡献为0。其耗时包含在相应原owner尝试区间，未额外再加。

未采样的资源、能耗、有效GPU工作量不推算。独立运行、失败尝试、资源回执和历史池完整保留。

本机根：`/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_paired_dagger_20261122_01`。主结果与逐初态表：`evaluation/test/summary/primary_summary.json`、`per_initial_state.csv`、`rollouts.csv`。本草稿输入SHA与检查见assembly_status.json；实际曝光actual_exposure_and_age.json；attempt时间与弃用计算见attempt_wall_accounting.json；数据/预算见learning/budget_registration.json与各组state_data/data.json；编码器限制见独立AUDIT/encoder_probe_feasibility.json。

运行恢复：原06–10及12失败均保留，开发与原终测已消耗的软件失败机会不补跑。13在原生owner启动前失败，新增科学机会0；14复用owner13后在原波16软件失败，原波12和16均保留。15按预登记策略仅执行17–63剩余47波；每波软件退出后先回收子进程再继续下一未派发波，不重试或替换。队列完成只表示47波机会已消耗并回收，不表示各波软件完成或学生成功。成功率按原64初态全分母；任何后续软件失败、保留下来的合法已完成lane与缺失lane见逐初态表。各代driver、native owner及辅助汇总区间分别登记后合并，历史collection结束未知，不推算GPU活跃时间。预评估guard与终末曝光提取各读8套权重，组装和导出复用终末表。
