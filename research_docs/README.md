# Play2Perfect 当前StateDP实验

2026-11-06：200条数据已通过验收，原训练/闭环仍由原owner负责；独立`tools.narrow_followup`在其正常完成、进程与锁全部释放后条件化接续。新增工作最多一次续训、一次260–296总量扩采和一次随机新训练，最终独立16初态不参与调度。新源码隔离在`bc+rl_RSS_narrow_followup`，仍发布到`origin/feature/narrow-state-data`。实际状态使用：

```bash
bash visuotactile/run_cpu.sh -m tools.narrow_followup status --root outputs/visuotactile/run/narrow_state_followup_20261106_01
```

下段是原流水线入口说明；旧启动计数不代表当前进度。

当前任务是直接窄分布新数据→随机初始化StateDP→有界闭环，工作树`bc+rl_RSS_narrow_state`、分支`feature/narrow-state-data`。原400只读；旧State评估暂停；旧narrow v1三条数据隔离。以新root的机器状态为准，不把服务启动当作数据/训练完成。

- [实验与真实采集结果](SLOW_EXPERT_RESULTS.md)：初始化query误报、直接采样范围、速度与新运行预算。
- [训练/执行合同](CONTRACTS.md)：H24/commit12、train-only统计、持续历史、随机新训练及固定评估。
- [StateDP结果](STATE_DP_RESULTS.md)：新运行与原宽分布结果分别标注。
- [当前交接](SESSION_HANDOFF.md)、[清理](CLEANUP_AUDIT.md)、[变更](CHANGELOG.md)。

维护入口：`tools.narrow_collection`管理整个新流水线，`visuotactile.anchor_continue fresh-*`复用现有State模型训练，`state_benchmark`执行同一真实环境/记录器，`live_collision`只读初态碰撞网格。原生RL入口和assets不变。

```bash
bash visuotactile/run_cpu.sh -m tools.narrow_collection status --root outputs/visuotactile/run/narrow_state_20261105_02
```

具体stop/resume和日志位置见实验文档。旧anchor_continue_queue、anchor_interim_eval及evaluation_delivery仅维护历史结果，本轮不运行；旧150→200/300续训、固定三档速度及宽分布扩采方案已被新协议替代。

原400在outputs/visuotactile/data/close_v4。旧native_dp_3way/native_20261101_02的索引及其他旧目录仍含必要依赖，不能按日期整目录删除。原始HDF5、权重、视频留本地；Git只含审查后的代码、精简统计和来源SHA。[历史索引](RESEARCH_HISTORY.md)。
