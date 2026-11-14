# 同一截止内的 RGB 软件故障恢复

这是待启动入口，不是恢复已成功。原 `new_family_10` 因 RGB 转换软件错误失败；首波A16各只执行一个控制步，没有完整轨迹、专家纠偏或入池数据，其余机会未开始。原 `research_loop`、失败采集目录及冻结 d30 工作树全部保留。

`recover_rgb01.py` 不调用 followup，不重建 session，也不重置时钟。它要求现有 session 字节 SHA 和原注册 SHA 与 `recovery_rgb01.json` 一致，截止固定为 Unix `1794725918.9665225`（2026-11-15 14:58:38.9665225，Asia/Shanghai）。每个新服务的 `RuntimeMaxSec` 只取提交时原截止剩余秒数；既有 loop 自身仍受原 session 截止与60秒清理裕量约束。

恢复配置直接读取原 `launcher/loop_config.json`，保留所有 seeds、event seeds、schedule、I/O和训练参数，只改以下身份：

| 原索引 | 新索引 | 含义 |
|---|---|---|
| 10 | 11 | 原首轮同32家族、64机会的软件失败替代 attempt，不算独立新增家族 |
| 11 | 12 | 原未启动的第二组家族 |
| 12 | 13 | 原未启动的第三组家族 |

各 registry ID 添加 `rgb01`。初始学生与训练池仍来自本 session 的 A/B 链接；后续采用上一轮完整候选作 DAgger 来源。既有输入配置继续指向原只读冻结树，运行代码由本次新冻结 commit 独立绑定。不会导入开发树的动态源码，也不会重新读取大权重或扫描全部轨迹。

从 root 审阅并冻结、包含 RGB 修正的新工作树运行：

```bash
FROZEN_TREE=/absolute/path/to/new/reviewed/frozen/tree
FULL_COMMIT=replace_with_new_full_40_character_commit
"$FROZEN_TREE/.venv_isaacsim/bin/python" \
  "$FROZEN_TREE/scripts/tight_fresh_20261115_01/recover_rgb01.py" \
  --worker-tree "$FROZEN_TREE" --expected-commit "$FULL_COMMIT"
```

默认只预览参数与命令，不创建目录或服务。实际执行时在同一命令后加 `--launch`。执行一次即可：已有 `recovery_rgb01`、`research_loop_rgb01` 或同名恢复 service 都会拒绝再启动。不要用原 `launch.py --launch`，那是创建独立新时段的入口。

正式启动会先确认旧 owner、monitor 和本 session worker 已退出，再将旧 `monitor.json`、`monitor_exit.json`、`shared_resources.json`、资源摘要、三个追加日志及 `STOP_MONITOR` 的原字节复制到 `recovery_rgb01/previous_monitor/`，记录 SHA。追加日志保留原路径，恢复 monitor 会接上历史；旧 monitor退出标记和 STOP_MONITOR 在归档成功后才移除。`monitor_observed.json` 明确记录旧末样本至新首样本的缺测间隔，不填成零负载或声称连续无缺口。

新服务为 `tight-fresh-research-loop-20261115-01-rgb01.service` 和 `tight-fresh-monitor-20261115-01-rgb01.service`，均不自动重启，使用 control-group 清理和原 LimitNOFILE。owner 结束后只写本 session 的 STOP_MONITOR。唯一一次正式 schema/source 检查由原 `tools.tight_fresh_research_loop prepare` 执行；新注册写入 `research_loop_rgb01/`。启动器最多观察新 monitor 45秒及 owner 30秒，只确认启动，不声称整轮完成。

启动命令、旧进程结束证据、归档清单、映射、新源码身份和异常均保存到 `recovery_rgb01/`。启动失败保留现场；已提交owner时不盲目停止它或重试。磁盘、有效纠偏不足、软件错误或剩余时间不足仍由原有限流程停止，不另开时钟。
