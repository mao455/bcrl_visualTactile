# Play2Perfect 当前StateDP实验

2026-11-06：200条训练已完成5735更新，正式train0/4、unseen0/10合法（另2次动作前拒绝）。条件化后续在GPU历史一致性审查处停止，没有续训或扩采；不自动重启。用户授权的4个独立观看复跑已完成，均超时；4个原三视角及4个腕部版本已全帧验证，视频不并入正式计数。额外发现200条新数据的两固定RGB首帧均空白，ImageDP须先修复并验收；详见StateDP结果及相机证据。源码隔离在`bc+rl_RSS_narrow_followup`，发布到`origin/feature/narrow-state-data`。实际后续状态使用：

```bash
bash visuotactile/run_cpu.sh -m tools.narrow_followup status --root outputs/visuotactile/run/narrow_state_followup_20261106_01
```

下段是原流水线入口说明；旧启动计数不代表当前进度。

当前任务是直接窄分布新数据→随机初始化StateDP→有界闭环，工作树`bc+rl_RSS_narrow_state`、分支`feature/narrow-state-data`。原400只读；旧State评估暂停；旧narrow v1三条数据隔离。以新root的机器状态为准，不把服务启动当作数据/训练完成。

- [实验与真实采集结果](SLOW_EXPERT_RESULTS.md)：初始化query误报、直接采样范围、速度与新运行预算。
- [训练/执行合同](CONTRACTS.md)：H24/commit12、train-only统计、持续历史、随机新训练及固定评估。
- [StateDP结果](STATE_DP_RESULTS.md)：新运行与原宽分布结果分别标注。
- [当前交接](SESSION_HANDOFF.md)、[清理](CLEANUP_AUDIT.md)、[变更](CHANGELOG.md)。

新增数据代码：[采集流水线](../tools/narrow_collection.py)、[初态/数据合同和缓存](../visuotactile/narrow_data.py)、[真实闭环慢速执行](../visuotactile/sim/expert_speed_environment.py)、[条件化单次扩采](../tools/narrow_followup.py)。前者已实际生成200条，后者仅实施、尚未执行扩采；不要混同。观看入口：[narrow_videos](../tools/narrow_videos.py)，结果见StateDP主文档。

维护入口：`tools.narrow_collection`管理整个新流水线，`visuotactile.anchor_continue fresh-*`复用现有State模型训练，`state_benchmark`执行同一真实环境/记录器，`live_collision`只读初态碰撞网格。原生RL入口和assets不变。

```bash
bash visuotactile/run_cpu.sh -m tools.narrow_collection status --root outputs/visuotactile/run/narrow_state_20261105_02
```

具体stop/resume和日志位置见实验文档。旧anchor_continue_queue、anchor_interim_eval及evaluation_delivery仅维护历史结果，本轮不运行；旧150→200/300续训、固定三档速度及宽分布扩采方案已被新协议替代。

原400在outputs/visuotactile/data/close_v4。旧native_dp_3way/native_20261101_02的索引及其他旧目录仍含必要依赖，不能按日期整目录删除。原始HDF5、权重、视频留本地；Git只含审查后的代码、精简统计和来源SHA。[历史索引](RESEARCH_HISTORY.md)。
