# Watcher15 关闭工具：最终独立源码签核

通过，未发现剩余源码阻断。此结论替代 `WATCHER15_CLOSURE_SOURCE_REVIEW.md` 对旧脚本的阻断结论，同时保留旧发现记录。本签核仅认证所审关闭工具及合成测试；不证明实际队列、helper、服务、进程、恢复或最终实验已经闭合。

| 文件 | SHA256 |
| --- | --- |
| close_and_certify_watcher15.py | `c0474fcc086fc81bcec4aa36f5184ad6cf10418dfe66e0a596ce1bafe2ce5ca2` |
| test_close_and_certify_watcher15.py | `4b4dddb7933809b81ec817e96a71e34cf726cdf36c1d82806035d0106b4c4c92` |

此前两个路径问题已解决。marker 通过规范路径检查及 `O_CREAT|O_EXCL|O_NOFOLLOW` 创建并 fsync，拒绝悬空链接、祖先链接及覆盖；证明绑定 marker 字节、设备、inode、watcher PID／birth。backup 目录及每个 target 必须为规范绝对路径，解析后位于原 root，拒绝目录／祖先链接、dotdot 和 root 外路径；逐实际完整 backup 集合核对字节或原不存在状态。

actor 核验已不依赖稀疏快照：直接绑定 activation 的 driver PID／birth，并要求其消失；核对 native owner、monitor、最终 worker 和47个 launch／stage-exit 的时序与实际回收标志，所有可用 dispatch PID 都作历史生命周期窗口检查。原生 `visuotactile/state_resources.py` 的 monitor 写入字段确为 `started` 和 `ended`，本版读取已对应。稀疏快照只是附加证据。

尾部 helper 名单被限定为原登记的两项，逐项要求正常退出回执；缺失历史 birth 明确保留为未捕获，没有补造精确 birth。脚本再检查 driver 服务 inactive／MainPID0／success、其实际 cgroup 为空，并作 fresh、同用户、精确 module／路径／root 的 census，拒绝仍存的本轮 driver、worker、native monitor、两项 tail helper 和 root lifecycle monitor。外部测试存在不是否决条件；当前 watcher 被有意留到最后。`stop` 还要核对当前 watcher 的 birth、完整命令和 cwd；`certify` 要求其身份消失、自己服务成功关闭和 cgroup 为空，逐 backup 恢复后排他写证明。源码无 kill、terminate、signal 或 systemctl stop/kill 操作。

本轮只读实际源码和任务说明；没有读取实际生命周期、科学结果、权重或缓存，没有执行真实 stop／certify，没有查询真实进程／GPU、启动GPU工作或发送信号。测试只使用 CPU 临时文件与注入的假进程／服务接口。先运行作者23项全部通过；尾部名单 guard 最后更新后重跑作者23项并加入6项独立测试（缺失／减少／重复／倒序 helper名单，及两项真实命令的 live-helper census 反例），共29项全部通过，0.33秒。

复现命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/CNS2025331827/miniconda3/envs/isaaclab/bin/python -m pytest -q -p no:cacheprovider outputs/diagnostics/tight_paired_dagger_20261122_01/test_close_and_certify_watcher15.py outputs/diagnostics/tight_paired_dagger_20261122_01/test_independent_closure15_review.py
```

root 可在原47队列及原尾部汇总／noise helpers 真正闭合后，按顺序执行一次：

```bash
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python outputs/diagnostics/tight_paired_dagger_20261122_01/close_and_certify_watcher15.py stop
```

等待其 own watcher 自行退出后再执行：

```bash
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python outputs/diagnostics/tight_paired_dagger_20261122_01/close_and_certify_watcher15.py certify
```

任一实际检查失败都保留失败状态，不重建已写 marker／intent，不重启实验、不伪造 closure。当前脚本只能确认实际调用时的精确 census、cgroup 与历史窗口；它不恢复缺失的历史 birth 或对任意未记录且已迁离本轮 cgroup、命令也已变化的后代作无条件保证。root 应保留此范围并把实际 intent／certificate 及其 SHA 加入最终绑定，仍由 final15 工具核验全部原64家族／256机会分母、原失败和最终证据。
