# Affinity16 实际关闭独立核验

已确认该辅助工具及其两次目标 worker 的原 PID／birth 身份均不再存在，当前没有相同完整命令的辅助工具进程；原 service cgroup 路径已不存在。两份实际 intent 均有同身份的 exit，均记录 `restored=true` 和唯一线程mask `8..23`。本核验只读进程、systemd／cgroup、实际介入回执和原生launch／dispatch；未发送信号、未写affinity、未读科学结果、模型或缓存载荷，未修改已激活源码，主冻结评估继续由原owner执行。

辅助PID `3445717`／birth `1795499527.26`；实际只尝试wave34和wave35，global exit 的 attempted 集合与全部intent／exit精确一致，error为null。两波intent的PID分别与原owner15 dispatch相符、原launch命令为正确worker／output、birth先于launch；原status未绑定birth的源码漏洞在这两次实际材料中没有发现对应的PID误归属证据。

| 原wave | 目标PID | 改／恢复线程数 | intent至exit | 记录的恢复状态 |
| --- | ---: | ---: | ---: | --- |
| 34 | 3446579 | 135／135 | 40.093451秒 | `restored=true`，全部8..23 |
| 35 | 3455028 | 133／133 | 9.568260秒 | `restored=true`，全部8..23 |

wave34全部135个改动TID都出现在恢复列表。wave35有一个旧TID未出现在恢复列表、一个新TID进入恢复列表；源码按当时活跃线程恢复，记录覆盖当时133线程。原两个worker身份现在均已消失，因此不存在可继续独立读取的原worker线程mask；历史最终mask来自保存回执，不是本审查对历史逐线程操作的连续观测。

三个源码阻断仍成立：PID来源需更强绑定、readiness race零写入、恢复失败必须保留逐波exit。本次两波都实际保存了非空changed_threads和完整恢复exit，没有进入所审“未扩宽仍恢复”的标记分支，也没有发生缺exit的恢复失败表象；这不使旧源码获得通过。wave34实际intent到exit比名义40秒多约0.093秒，应保留这个超额，不能声称严格40秒上界。wave35结束时记录environment_ready只是启动生命周期信息，不用于自主成功或因果效率结论。

当前 `systemctl --user show tight-paired-startup-affinity16.service` 为 `MainPID=0 / ActiveState=inactive / LoadState=not-found / ControlGroup空 / InvocationID空`。瞬态unit已卸载，所以当前 `Result=success` 不能单独认证旧invocation的退出结果；独立关闭结论主要由确切PID／birth消失、exact-command census为空、原cgroup不存在及global exit支撑。原invocation `ad8547419d664bef83f4e22778d35003` 的journal只保留启动和两条cgroup inotify资源警告，没有保留退出消息；警告的 `No space left on device` 是inotify watch添加失败文本，不能据此推断磁盘占满或重新启动实验。

结构化证明见同名JSON，含每份实际输入SHA、原launch／dispatchSHA、PID核验、service结果、原cgroup路径、两波时长及上述限制。报告不保证恢复缺失的历史birth、不对所有历史thread调度操作作逐位审计，也不把回执中的external_mutations／signals字段当作独立连续监控。无需也未对外部任务作退出要求。
