# Kit生命周期与有界恢复

## 2026-11-03 StateDP运动实验的当前smoke证据

新root`state_motion_20261103_01`使用同一已验证的独立worker模式，真实创建环境、reset、
执行1个专家控制步，有限值检查通过；父进程wait/reap后核实PID及nvidia-smi中均无该worker，
随后才发布committed marker。见[实际退出证据](evidence/state_motion_20261103/startup_review.json)。
本次smoke无重试；一个短probe不证明长期Kit稳定，也不代表FileThread根因被修复。
本轮最多2次已分类pre-scene启动重试（整个session共享），NaN/接口/数据/hash失败不自动重试。
SDK graceful shutdown仍未修复；新实验继续显式durable独立进程退出，未更改系统inotify或共享环境。


## 最终运行结果（2026-11-03 04:45 UTC+8已停止）

受控隔离退出缓解了关闭崩溃，未根治启动问题。本轮记录3次启动恢复：预检kit_startup_02，
以及native_h32_selected_single_05和position_h32_validation_1000_single_01，恢复后均完成。
正式学生阶段这两次分别为截断system_error/Invalid argument和FileThread启动abort。

position_h32_validation_6000_single_02仍在launching时exit-6，只有runtime_controls、
runtime_before及lifecycle，无场景/动作。Fatal stack经过libnrend.so、operator new、
libomniclient.so!omniClientFreeContent和provider_file::FileThread，缺少恢复分类器所要求的
std::system_error签名，未自动重试；不能仅凭operator new推断OOM或FileThread推断inotify耗尽。
该8条cohort为0/7已完成+1未知；随后独立评价继续，最终选定模型12条全部有原生结果。
因此服务完成不代表所有里程碑均无软件故障，也不能用此启动缺失解释最终学生0/12。
原始失败log/attempt保留，收尾无新启动；所有自有worker、owner、monitor已退出。
见[收尾核验](evidence/position_20261103/completed_review.json)和
[6000验证结果](evidence/position_20261103/results/position_h32_validation_6000.json)。

## 启动与关闭专项审计


当前证据不能把FileThread堆栈唯一归因为inotify耗尽，也不能宣称问题已经修好。
首次专项检查时系统max_user_watches=524288；用户VSCode两个主要进程约440384/34324 watches，
接近限额，但没有捕获本轮ENOSPC。未终止或修改这些进程，未更改系统限制及共享安装。

`position_20261103_01`最初4个startup probes：前3个最终exit0（其中第3个此前启动abort后恢复1次）；
第4个已经写出`app_ready.json`和`result.json`，随后SIGABRT(-6)，无`closed.json`。
这与“建立场景之前失败”不同；不能把存在result当作完整生命周期通过。
OmniClient setter的机器码经精确匹配后只读其存储标志，App就绪后仍为false。
因此watches关闭已生效，但不能排除SDK自身线程/关闭顺序问题。

专项对照只改进程生命周期：AppLauncher显式`fast_shutdown=False`，保留完整扩展清理；
启动前、App就绪、关闭前记录自身fds/threads/inotify、rlimit；明确launching、environment_ready、
results_committed、closing、closed。位置/原生物理和任务不因此改变。
4次新对照在原3小时Gate窗口内执行，保留原4次记录，不重置deadline。
要求零退出和closed证据；实际结论随真实产物更新，不把配置改动当根因证实。

新位置会话最多2次pre-scene retry/stage、12次/session，退避3/9秒，Gate只容许最多2次。
只有明确native崩溃签名、尚无场景/轨迹且状态launching时才可恢复；原attempt连同log归档。
已执行轨迹绝不重跑、替换seed或记成原生任务失败。已完成结果原子保存。
已有旧会话的3次恢复预算保持耗尽；新授权只适用于本次新位置研究。

当前仍用每受控轨迹独立Kit。此前持久reset的USD碰撞查询缓存问题已通过live PhysX pose查询
修正，但尚未证明重复reset能还原全部接触/solver状态；不为节省启动时间宣称跨episode精确。
共享GPU只有一个本项目worker、6GiB空闲显存/12GiB available RAM准入，近期1/5分钟负载pacing。
没有外部项目吞吐指标，因此不作零干扰承诺。

## 受控退出对照实测

完整清理对照`kit_graceful_00`已经进入closing，随后exit-11，无closed证据。
该进程before→ready→closing：fds21→248→247、threads4→123→123、inotify watches0→1→1；
nofile软限从1024由Kit升至2450、硬限1048576。没有观察到此进程fd耗尽；不能推定全用户inotify根因。

新的4次`kit_isolated_00..03`全部完成App更新、原子结果、文件fsync后`os._exit(0)`。
父进程wait/reap，再核查`/proc/PID`和nvidia-smi中均无该PID，才保存closed证据。
这避开已测SDK关闭故障，属于显式进程隔离/内核资源回收，不等于SDK graceful shutdown修复。
异常或未完成结果不能进入这条成功退出路径，原始启动崩溃仍按有界规则处理。
后续场景、专家和位置回放已由同一服务验收；4个短probe不是长期可靠性证明。

之后freeze、2条在线专家、2条位置重放均完成并正常受控退出；位置foundation已通过。
原生完整成功2/2与回放2/2，14字段exact，细节见POSITION_ACTION_CONTRACT。
初态控制/学生评价逐条保留进程错误；最终仍有1条启动缺失，详见顶部结论。
