# Fragment13：仅源码准备，未激活

本预案不读取当前测试结局，不停止健康 recovery12，也不延长截止。准备代理只写了新13源码和合成fixture，未运行 prepare/driver/worker/watch，未作任何实际生命周期 census。

仅当 recovery12 真正失败，root 先核对并回收其准确自有 driver、test owner、monitor、worker 和 watcher。所有原失败、STOP 和已启动整波保持；任何已启动或不确定整波的四个机会都计入分母，不能补跑。原来64个wave、job、registration、plan、session、四完整pair、295源码、noise7091和deadline1795684571.2261844不变。

1. root 写 `ROOT_TEST_FRAGMENT13_BOUNDARY.json`。必须含 status=`actual_original12_boundary_closed_for_missing_only_fragment`、deadline、all_original12_actors_reaped=true、no_external_signals=true、`closed_identities:[{pid,created,...}]`，以及 `artifact_sha256` 绑定原 test/project_failure.json、project_exit.json、monitor.json、project_status.json、project_job_started.json、driver_eval_recovery12/failure.json、driver_exit.json、WATCHER12_ACTUAL_CLOSED.json；可再加入独立实际PID审计回执。watcher12实际关闭回执需原动态schedule逐项恢复核验，无冲突、MainPID0/inactive/Resultsuccess。此文件只能基于真实关闭，不由本预案代写。
2. 已审阅新13源码后，root 可运行注册解释器 `-m outputs.diagnostics.tight_paired_dagger_20261122_01.test_fragment13 prepare`。这是第一次实际 census/准备：只接受原 owner stage log 形成唯一已启动前缀，空log也消耗；任何后缀运行痕迹、未知文件、symlink或身份变化拒绝。未启动后缀必须每wave仅原plan/session。prepare 新建独立 owner/test_fragment13、job、dispatch目录和 TEST_FRAGMENT13_REGISTRATION；部分准备失败也保留，不在同身份重试。
3. root 对实际 registry/census/job/session/源码SHA独立复核，确认不能有活的旧角色或未解释 GPU 残留。写 `ROOT_TEST_FRAGMENT13_REVIEW.json`：status=`source_and_actual_missing_suffix_review_passed_inactive`、registration_sha256、复核理由与来源。再次做原wave/八权重实际SHA和原session/data保护核验。无需修改或删除原STOP_RESOURCE_WATCH：新watcher仅检查 STOP_RESOURCE_WATCH13。
4. 才可用 systemd 同环境、nice10/ionice idle/canonical cwd 启动 `shared_capacity13 --audit <AUDIT>/watch_shared_capacity13 --apply`，随后 `test_fragment13 driver`。driver保持taskset8-23用于evaluate，8-15用于小CUDA审计。当前三脚本均不写合成telemetry。原 E/test 的 lease 精确映射到新 owner C 的真实 native FullLease；wave session/plan完全不动。watcher按真实剩余容量共享GPU，外部进程存在本身绝不暂停。
5. driver仅跑已冻结全后缀，原生autonomy worker和原生 owner.execute均保留。每个wave在任何native工作前先排他fsync dispatch fence；既有fence和runtime痕迹都阻止二次dispatch。启动诊断只加traceback与既有StartupDiagnostics。1800秒只适用于 native initialization.json 前；durable timeout intent之后再加锁复核owner stage与worker initialization，变化则写abandoned、无STOP。owner stage写入、worker initialization写入和最终STOP均用同一锁序列化，避免receipt fsync后的陈旧停止。
6. 仅在原12尚未尝试的情况下，driver再运行原64 summarize+真实CUDA RNG核验及collection噪声核验；已rc0的跳过并绑定原exitSHA；有log但无rc0的拒绝自动重试。任何13本体失败也不重启原owner。所有未启动wave可供未来另一次独立census判断，已有dispatch fence/stage log和wave目录必须永久纳入消费台账。
7. 真正关闭后root核对driver/owner/worker/monitor已回收，再touch ROOT/STOP_RESOURCE_WATCH13，核验动态schedule原字节/不存在状态全部恢复及watcher服务inactive/MainPID0/Resultsuccess，写新实际关闭回执。原12失败不能伪造为closure。最终证据适配需另审阅真实12失败+13续段，当前final12 adapters不能直接冒充闭合。

源码：`test_fragment13.py`、`shared_capacity13.py`。CPU fixtures：`test_test_fragment13.py`。本地23项已通过；独立review另记。没有执行实际13 census、登记、采集、仿真、推理、训练或信号。
