# Host-init18 独立源码复核

结论：冻结源码及合成回归通过；无剩余源码阻塞项。本签署只覆盖源码和 CPU 合成夹具，没有执行激活、查看实际 main64 结局、读取模型/缓存/轨迹载荷、查询活动进程或修改实际 affinity/nice。

`host_init_capacity18.py` SHA256：`7ac5bb977213647e8461fe8fc2e500eea1b48fa557a56dbeca2b3352983c3cc2`。

三套当前测试合计 **100 passed in 0.23s**：作者 53、另一独立复核 25、本次独立补充 22。七个被审源码文件另以 `compile(bytes, path, 'exec')` 检查，没有写入 pyc。只运行了 host-init18 测试，没有重复 final15/closure15 测试。

核验覆盖：

- 仅 wave43–63、每波一个 durable intent，worker/status/immutable launch/dispatch/owner job/祖先链与 PID/birth 绑定；source_guard 要求自身及三份冻结源码、五项 lifecycle 绑定的准确集合和摘要。
- 原始 CPU8–23、nice10；最多从 CPU1/2/4/5 借两核，初始两个 1.5 秒低于 15% 样本及锁内 0.25 秒样本均满足后才写。无余量、readiness/阶段竞态和未触发路径不做扩展或恢复 syscall。
- 当时及过去的 constructor 标记检查在锁内及逐线程写入前重复；每个 TID 均检查 Tgid 和 starttick，检查围绕 affinity syscall，nice 只验证、不更改。外部新身份或相交进程 CPU 增量超过阈值会撤回。
- 40 秒扩展预算从首轮逐线程写入前开始；部分写失败及后生线程纳入恢复。记录实际耗时，并在恢复未验证时标记 `placement_end_verified=false`。恢复本身可能使总耗时超过 40 秒，不能把 40 秒写成全部收尾的硬上限。
- `lifecycle13.lock` 保持到原 mask 恢复验证或 exact worker 退出；它阻止 `initialization.json` 提交及其后的 Python 学生控制，不阻止已获用户授权的物理 constructor 初始化。没有新增“先物理审批”条件。
- 恢复失败在尝试 wave exit 后仍执行全局 `OWNER/STOP`/infrastructure-abort；即使 exit/evidence/STOP I/O 出错或进程身份读取不确定，也不得据此把仍可能存活的目标判成退出并释放锁。若目标已退出，仍先尝试全局 STOP，且区分请求与 `owner_stop_committed`。没有伪造正常 startup1800 scoped-stop。
- SIGTERM/SIGHUP 只置协作停止标记，不能异步跳过恢复。主循环区分 pre-intent 调度竞态与 post-intent 错误；已记 intent 的波次不重试。

本次独立复现了旧16的三个反例类别：旧 status 下 PID 复用拒绝；readiness 竞态零 affinity/恢复写；恢复失败仍尝试 exit、全局中止并在锁内等待 exact target 退出。此外直接新增 constructor 阶段三处漂移、超白名单/超预算、exit+restore 同时失败、identity 读取失败、worker 已消失仍请求全局中止、TID/nice 漂移和 once-per-wave 主循环回归。提出的问题已在最终冻结版修复。

实际激活必须由 root 另核准确 registry/review/owner 绑定。实际收尾还必须逐 intent 对照 exit、扩展/恢复写、原 mask、birth、实际 elapsed、全局停止及 helper 退出；**closure15 不覆盖 host-init18**。本签署不能证明实际运行已恢复、未影响外部测试或实现提速。

边界：Python finally/flock 无法覆盖 SIGKILL、主机丢失或内核故障；线程 mask/CPU 使用率是逐次观测，非原子全进程快照。初始容量与外部增量是保守检测，不能证明外部负载未来不变。调度改变没有改写四套模型、原生频率、成功规则或噪声计划，但本审查没有重新执行科学结果一致性评估。所有 hash、命令和范围见同名 JSON。
