# Startup affinity16：独立源码复核

结论：当前源码未通过；存在三个可复现阻断，须由 root 处理已激活工具的生命周期并由作者修复后再审。此审查没有修改启动工具，也没有读取或影响其实际运行；不得把本审查当作实际恢复证明。

所审源码 `startup_affinity16.py` SHA256：`cab175e633b7c485f8169f40ca55cfbe16977a8acf787655ca3a4af46e1c1aa1`。作者测试 `test_startup_affinity16.py` SHA256：`75a1858bdf8418ea8e6ec139f19106c998d294dc7522aa65747b1668914e2f34`。仅读取这两份源码、root任务说明及既有运行源码中的接口定义；未读实际 probe、outcome、weights、cache 或运行日志，未查询进程／GPU，未调用真实 sched affinity 或发送信号。

## 已复现的阻断

1. **出生时间没有绑定原 status／dispatch，PID 复用可被接受。** `probe` 从当前 `psutil.Process(status.pid)` 取得 birth，随后 `identity` 仅拿它和同一个当前对象比较。因此当前进程即使出生晚于旧 `status.utc`，只要命令／cwd／nice相同仍会被接受。独立假进程 birth=200、status.utc=100 的用例没有被拒绝。身份应绑定原 owner15及本波 dispatch／launch、注册plan，并至少要求出生不晚于原 status的合理窗口；沿用已审 `shared_capacity15.registered_worker` 的 owner祖先、登记SHA、stage／output／fence核验可避免另造较弱身份门。无需读取科学结果。
2. **readiness-race“未改动”分支仍写亲和性。** 初检未ready、写intent后第二次ready为真时，`return` 仍执行finally里的 `change(..., OLD)`。独立用例记录到一次OLD写入，而 `readiness_race_no_change` 同时为true。此时已经不属于构造停滞阶段，且原生初始化可能新建／调整过线程；不应无条件覆写。只有确已开始临时扩宽时才恢复，未扩宽的ready-race应零affinity写入。
3. **恢复失败会跳过逐波exit回执。** finally中的 `require(row['restored'])` 或恢复本身的异常发生在逐波 `write(..._exit.json)` 之前。独立用例令最终mask仍为WIDE，观察到只有intent，没有exit。main虽可写全局error，但未保留该波pid／birth、已改线程、失败恢复状态；需要嵌套finally确保逐波exit尝试落盘，明确restore_error，并使watcher停止、不再发生任何后续扩宽。

作者7项CPU测试全部通过（0.55s），但未覆盖上述三条。新增独立 `test_independent_startup_affinity16_review.py` 只用假进程、mock读写／affinity和CPU调用复现三个失败，3 failed，0.06s。它们是保留的回归用例，不是实际运行失败证据。

此外，本版没有task要求的Tgid复用反例；作者测试中的 `test_reused_pid` 只检查 `same()`，没有检查 `probe` 的完整身份门。`change` 在Tgid检查与sched调用之间仍有线程退出／复用窗口；修复版应明确每个thread的身份核验和失效处理，至少补充Tgid不符／消失、worker birth变化、部分扩宽后异常、恢复异常、ready-race零写入及40秒cap的合成反例。当前40秒计时也从 `change(WIDE)` 返回之后开始，严格cap应从首次扩宽之前记时。

## 可保留的行为与实际边界

源码没有信号发送、外部进程名称匹配变更或额外GPU／Kit启动；限定剩余33..63、60秒构造等待、原有限deadline、nice10和初始全线程8..23；有每波排他intent以及恢复后mask检查。这些方向可保留，但不能抵消上述身份／race／回执问题。

root已告知该工具曾获其签核并激活，本审查未去验证实际状态。root应仅用该工具自己的STOP机制让其结束，并核验任何已经写intent的波的恢复／退出证据和其实际PID／birth；不要由此向外部任务发信号。若修复新的host工具，保留旧源码和实际回执，不对已激活文件热改，不重跑消耗过的wave，不改变冻结runtime、原生控制或模型。任何实际cap／恢复缺口应按实记录，不能依据源码中的 `external_mutations=0` 或 `restored` 预期字段反推未观察事实。
