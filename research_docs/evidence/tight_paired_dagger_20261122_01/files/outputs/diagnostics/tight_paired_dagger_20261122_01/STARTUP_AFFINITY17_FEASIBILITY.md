# 后续启动亲和性工具：可行性结论

**结论：在不修改冻结worker／原生环境、不开新的物理执行的约束下，现有lifecycle13锁无法保证亲和性扩宽及恢复都发生在首个物理阶段之前。本轮不建议实现或激活新的affinity17；继续原冻结队列与只读监控。** 此结论为源码可行性分析，未实现工具、未激活、未发送信号、未改affinity／schedule／runtime，未读取终测结果、权重或缓存载荷。

`test_fragment13.py` 的 `lifecycle_lock`（320行）及 `locked_atomic`（326行）只把指定JSON的原子提交包在锁内。实际worker在449行仅把这层锁接到 `tools.tight_fresh_batch_run.atomic_json` 的本波 `initialization.json`。owner的 `project_status` 提交及启动timeout决策也用同一锁，因而可以可靠地序列化“状态发布／是否已提交初始化／是否写scoped STOP”，但没有把整个构造函数或其内部native调用放在这个锁里。

实际顺序如下，均取自现有源码：

| 顺序 | 源码边界 | 对亲和性隔离的含义 |
| --- | --- | --- |
| 1 | `native_evaluation.py:159` 写environment_constructor_started，160行调用env_class | 只是进入native构造调用的前置标记；下一条Python标记要等调用返回 |
| 2 | `InstrumentedAssemblyEnv.__init__` 调 `PreciseAssemblyEnv`，后者调 `DirectRLEnv.__init__` | 共享该构造调用，不持lifecycle13锁 |
| 3 | 已安装IsaacLab `direct_rl_env.py:167` 调 `self.sim.reset()`，172行 `scene.update(dt=physics_dt)` | 构造期间已开始模拟器／物理句柄初始化和正dt场景更新；不能称作纯文件加载或无物理区间 |
| 4 | 构造返回后写environment_constructor_complete，配置相机，再创建expert | environment_constructor_started仍为最新值时，内部具体进度可能早已跨过物理初始化 |
| 5 | `StartupDiagnostics.__exit__` 写environment_ready | 此时构造和expert创建已完成；这个发布本身不受lifecycle13锁保护 |
| 6 | `tight_fresh_batch_run.py:310` 提交initialization.json | 只有这里会等待lifecycle13锁；发生在环境构造之后 |
| 7 | 随后DINO／backend加载，进入run_wave；`batched_rollout.py:247` reset，323行step | 阻止初始化JSON提交可暂阻后续Python控制路径，但不能倒退或隔离步骤3的物理初始化 |

因此，在观察到constructor_started后持lifecycle13锁40秒并扩宽线程，只能保证worker在步骤6不能越过JSON提交点。它可能在持锁期间完成步骤3，甚至步骤5；而辅助工具对environment_ready做轮询始终存在检测延迟。锁不能阻止native构造内部继续执行或新建线程，也不能把新的thread affinity继承、TID消失／复用和恢复异常变成原子事务。修复affinity16的三个实现缺陷仍不足以满足“严格在物理／控制前完成”的要求。

现有SharedLease的gpu.lock也不解决：当前生产session使用FullLease路径时没有该跨进程锁，即使引入新锁，worker也未约定在物理初始化之前等待它。写pause_groups只能影响后续lease准入，已进入的构造调用不会由外部标记中断。额外发送STOP／CONT、ptrace、suspend、调nice或修改共享systemd／CPU配置都会引入新的干预，且仍没有当前冻结代码内的可核验“首次物理前”合作点，本分析不建议这些替代。

将来若单独登记启动效率实验，要有worker内部、物理初始化前的合作屏障，并由该worker在屏障内验证全部线程已恢复后才进入 `sim.reset()`；必要时仅对与模拟器隔离的预加载进程优化。这会改变运行包装／依赖及登记，不能放进当前已冻结、已消耗部分wave的测试来声称完整保持合同，也不能据开发观察临时改变终测执行方式。若目标只是wall-clock效率，可以另定义允许启动阶段host调度变化的合同并验证，但应诚实保留其对隐藏物理初始化状态的潜在影响，不能称本轮原冻结合同不变。

wave32单次probe及affinity16的wave34／35介入历史保持。实际关闭核验已确认辅助工具和目标worker身份消失、两份exit记录恢复mask；这证明记录范围内的恢复与当前关闭，不证明介入完全避开物理初始化，不证明效率因果收益。wave34约40.093秒及wave35线程人口变化的限制继续保留；不得把本可行性结论用于重跑任何已消耗机会或抹除历史软件未完成。

本轮建议仅继续原owner15的有限队列、原容量watcher和只读生命周期监控，记录实际启动时间、软件退出、helper及actor闭合；保持全部64家族／256策略机会原分母与所有失败。root如需额外非干预性效率工作，可并行准备CPU证据工具、审核小文件、计算磁盘清单，但不要增加当前worker的调度变更。
