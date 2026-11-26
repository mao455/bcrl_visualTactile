# Final15 补充证据计划19

本计划原样承接 `FINAL15_SUPPLEMENT_PLAN18.json` 的64项 `fixed_existing`，逐项重核 SHA256 与字节数；新增27项，共91项。12组 Markdown／JSON 证据对均完整，`missing_expected=[]`。它只登记当前具体小证据及后续应审核的有限路径，不创建真实 final15 binding，不签署实际最终闭合，不导出载荷。未读取 main64 结局、权重或缓存载荷，未查询或改动进程。

新增材料包括 PLAN18 两份文件、18版 activation／started／exit 及实际失败关闭复核对；19版冻结源码、三个 CPU 合成测试、handoff、独立源码与时钟复核对、登记、root review、实际只读身份 preflight、activation，以及首个实际 intent 审阅对。19版源码 SHA 为 `c7374e679a2a67156fc1ce942852cc1aee9978d695e185e94d9870ae7dda56b9`，三套 CPU 合成测试共149项通过；这项源码结论不能替代真实逐波身份、恢复、退出及停止审阅。当前采样不能证明历史 realtime−BOOTTIME 偏移始终稳定；既有 SIGKILL／host failure 和整数 TID 系统调用竞态限制保持。

18版实际复核记录：helper 因 raw psutil／realtime 时钟接口不匹配而在任何 wave intent 或 affinity 写入之前失败退出，实际目录只有 `started.json` 与 `exit.json`。本计划保留这次负结果，核验其具体回执与独立关闭复核；不重写18版文件，也不把失败丢弃。PLAN18 原 pending 列表以 `historical_pending_from_plan18` 原样保留为历史快照，其旧状态不覆盖本计划单列的实际18关闭记录。

19版实际目录只做43–63的精确有限 census，每波最多一次；JSON 的 `actual19_finite_census.rows` 列出21波各自 `intent`、`expansion_intent`、`exit` 及条件性 `infrastructure_abort`／`abort_worker_exited` 五种具体路径。当前有 intent 的波为 `[46, 47]`，expansion为 `[46, 47]`，exit为 `[46, 47]`；每份已存在小 JSON 只登记当前 SHA256 与字节数，并明确仍待 actual final closure。当前有 intent 而缺 exit 的波为 `[]`，没有把缺失行丢弃。尚未出现的文件保留 null SHA 和 pending，不能据此断言那些波永不发生或已闭合。

`HOST_INIT19_ACTUAL_FIRST_INTENT_REVIEW.md/.json` 只审核 wave46 的实际恢复；它明说 helper 仍运行，不是最终关闭。其他已存在波的回执即使完整，也仍待实际独立复核。helper最终 `exit.json`、后续 stop／身份／最终关闭复核对及未来逐波回执继续 pending，不造 SHA。实际存在只说明当前文件快照，不能替代 root 的最终 PID／birth、服务 invocation、原 mask8–23／nice10、容量样本、40秒触发预算及实际耗时、全部恢复／失败／STOP持久化／精确进程退出检查。原64×4分母、已消耗波、软件失败和原1800秒 timeout 保持，不允许重试、替换或改写原机会。

六个 frozen final15 工具仅重核原 SHA 和大小，未修改。root 在实际全部闭合、逐项审核后，才可将具体小文件合并到真实 `small_file_sha256`，并保留 builder 的全部原项。当前 pinned 并不等于已获最终绑定资格。PLAN19 两份文件固定后可由 root 外部登记 SHA，不嵌入自身 SHA。真正的 actual binding／root review、八组件选中曝光提取、compute、assembly、dry inventory 和 apply 回执都保留各自科学与身份门槛；本计划不替代任何步骤。

曲线导出规则完全继承18版：已有冻结底层 exporter 源码支持 `common180_parents`／`learning` × A/B × DP／后端的八组件，精确16个 `family_phase_curves.csv`／`family_objective_loss.csv` 路径，以及 `family_phase_curves.png`、`validation.png`、`training.png`。本轮不打开 CSV／PNG 载荷；allowlist 支持不证明实际曲线齐全，仍由 final15 科学门与 dry inventory 核验。CSV／PNG 不进入只允许 `.json/.py/.md/.txt` 的小文件映射。binding 可收 `.txt` 但冻结底层 exporter 拒绝 `.txt`；本计划没有新增 `.txt`。

`local_retained_log_sha256` 集合完全不扩：只能是三个 ROOT 边界 JSON 中的 `.log` artifact 并集，再加 mandatory `cycle_logs_fragment13/evaluate_fragment13.log`。辅助18／19 stdout、stderr 继续留本机；允许的 JSON 审阅可记录其路径与 SHA，但普通日志不进入公开小文件或扩大的 local-retained 映射。裸 STOP／STOP_RESOURCE_WATCH15 只由已绑定 JSON intent／certificate 保留精确身份与 SHA，不直接导出裸 marker。

继续遵守单文件16MiB、总导出256MiB、规范普通路径和不覆盖既有输出；不公开权重、HDF5、视觉、缓存、RNG tensor、视频或普通日志。当前 census 是在 `1795507443.2350924` UTC epoch 取得的文件快照；后续新回执须由 root 最终精确 census 补齐并审核，不能静默覆盖此计划快照。
