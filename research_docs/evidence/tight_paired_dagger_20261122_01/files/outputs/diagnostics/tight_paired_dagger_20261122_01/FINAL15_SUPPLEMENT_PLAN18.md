# Final15 补充证据计划18

本计划承接 `FINAL15_SUPPLEMENT_PLAN17.json` 的35项 `fixed_existing`，逐项核验原 SHA256 和字节数并原样保留；新增29项明确存在的小文件，共64项。八组 Markdown／JSON 证据对均保留，`missing_expected=[]`。本计划只列举现有可复核材料及以后应审核的路径，没有创建实际 final15 binding、签署真实闭合或导出证据。未读取 main64 结局、权重或缓存载荷，未查询或改动进程。

新增材料包括自主权限合同与配对优化配置核对、非结局短稿、训练事实／曲线生成方法、编码器缓存可行性方法、17版主机初始化诊断与当时限制、18版容量观测、独立源码、三个 CPU 测试、两份独立审阅、handoff、登记与 root review。完整绝对路径、SHA256、字节数及类别见同名 JSON 的 `fixed_existing`／`added_existing`。报告事实10、训练事实11、开发诊断17、辅助16真实关闭、权限合同17、stall41诊断17、CPU容量18和最终独立源码审阅18的 `.md/.json` 都成对列入。

历史审阅的范围和时间保持：`HOST_INIT_CAPACITY18_INDEPENDENT_REGRESSION_REVIEW.md` 记录较早源码快照；最终 `HOST_INIT18_INDEPENDENT_REVIEW.md/.json` 对应冻结 `host_init_capacity18.py` SHA `7ac5bb977213647e8461fe8fc2e500eea1b48fa557a56dbeca2b3352983c3cc2` 及100个 CPU 合成测试。此前17版调度限制和可行性说明保留原文，18版登记明确记录后续授权与范围修订，不能把历史较强限制当作当前待补审批，也不能删除历史判断。

六个 final15 工具只核验 SHA，未作任何修改；其身份在 `frozen_final15_tools` 中登记。root以后首次构建真实 binding 时，可将已经闭合、已实际审核的具体小文件合并到 `small_file_sha256`，完整保留 builder 返回的原有项。不要直接把计划标为 `actual_binding`，不要用源码签核替代实际身份、逐波恢复或退出证据。最终绑定本计划 JSON／Markdown 时，由 root 在两份文件字节固定后外部登记 SHA；文件内不嵌入自身 SHA，避免循环。

18版 activation、实际 `started.json`／`exit.json`、43–63有限范围逐波 `intent`／`expansion_intent`／`exit`，以及条件性的 `infrastructure_abort`／`abort_worker_exited`，均列为 pending，没有预造 SHA。哪怕文件已经出现，本计划也不打开其实际内容或将其当作闭合证明。root须按实际精确 census 加入全部已发生尝试，包括未扩展、错误和缺失退出；不能只挑成功恢复。实际停止／身份／关闭审阅应成对保留 Markdown 和 JSON，由 root确定具体文件名并核验。原15版待闭合清单也保留待实审状态；`existing_as_of_plan` 仅说明路径是否存在，不是认证。

实际审阅至少应对应精确 PID／birth、argv／cwd、服务 invocation 或进程退出、每个 intent 的实际处置、原 mask8–23／nice10、外部容量采样、40秒触发预算与实际耗时、所有缺口和关联 SHA。恢复失败时应区分“请求全局停止”“STOP已持久化”与“队列实际退出”。原64×4分母、已经消耗的波、软件失败和原1800秒 timeout 记录均保留；本计划不允许重试、替换或改写原机会。

曲线 CSV 已有冻结底层 exporter 支持：`common180_parents`／`learning` × A/B × DP／后端，共八组件的 `family_phase_curves.csv` 和 `family_objective_loss.csv`，以及对应 `family_phase_curves.png`、`validation.png`、`training.png`。精确16条 CSV 路径已列于 JSON。此次只核对 allowlist 源码，没有打开曲线载荷；底层规则是文件存在时加入，支持导出不证明八套实际文件齐全，真实完整性仍须 final15 科学门和 dry inventory 核验。CSV／PNG 不可塞入只接受 `.json/.py/.md/.txt` 的补充小文件映射。可选 `TRAINING_FACTS11` 汇总 CSV 也不自动升级到该映射。另一个边界是 binding允许`.txt`，但底层 exporter不允许；本计划新增文件仅用 `.json/.py/.md`，不因此变更冻结工具。

`local_retained_log_sha256` 完全不扩展。其集合必须精确等于 `ROOT_TEST_FRAGMENT13_BOUNDARY.json`、`ROOT_TEST_ADMISSION14_BOUNDARY.json`、`ROOT_TEST_CONTINUATION15_BOUNDARY.json` 的 `.log` artifact 并集，再加 mandatory `cycle_logs_fragment13/evaluate_fragment13.log`。辅助18和其他普通 stdout／stderr 留本机，root可在允许的 JSON 审阅中记录路径及 SHA，不能把这些日志硬塞进 local-retained 映射或公开小文件映射。裸 STOP 和 STOP_RESOURCE_WATCH15 只通过已绑定 intent／certificate 记录精确身份及 SHA，不直接导出裸 marker。

所有补充继续服从16MiB单文件、256MiB总量、规范普通路径、不覆盖既有输出，以及不公开权重、HDF5、视觉、缓存载荷、RNG tensor、视频和普通日志的冻结导出规则。root后续实际 binding／review、8载荷选中曝光提取、compute、assembly、dry inventory和apply回执分别实审；本计划不替代其中任何一步。
