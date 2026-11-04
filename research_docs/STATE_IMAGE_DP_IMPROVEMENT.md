# StateDP + ImageDP baseline 改进

## 用户批准后恢复：资源预检通过，完整Kit门控待执行

用户已明确批准父docs工作区的deliverables-only监视排除；两项配置已合并保存，未修改其他设置。
新的真实探针256/256成功，可读取watch FD引用34756/524288，资源预检通过。
会话间隔内其他进程状态也可能变化，缺少紧邻配置修改前的census，不能将全部下降归因于配置。
见[恢复资源证据](evidence/state_image_20261104/resource_resolution.json)。

新增持久独立Kit门控入口 `tools/state_image_kit_gate.py` 和用户service：
预定expert→Sample8000→重复expert，各为新Kit，最多5starts，当前登记3次、无自动重试。
采用真正env.close/app.close及closed marker，要求exit0、父进程回收、无存活自有子进程/GPU记录；
明确拒绝仅有result、关闭崩溃或os._exit的假通过。每2秒记录阶段/线程/FD/RSS/inotify，
每个episode前后再资源预检，独立连续整卡监控。60s关闭上限/15min单worker上限只约束本门控。
重复专家与原专家参照比较控制及物理字段；不声称未保存的PhysX隐藏状态完全一致。
两项新增生命周期判定测试通过，正式模型仍0updates；实际Kit进展读取kit_gate/status.json。
此门控本身不启动训练；只有真实通过后才继续新损失实现和监督验收。

## Kit监控器权限缺口修复（首个start仍计入预算）

首个expert在场景就绪前被owner终止：监控读取Kit辅助进程/proc/PID/fd返回PermissionError，
不是已证实的FileThread崩溃。原日志、exit−15、launching状态和owner traceback保留。
只修新监控：不可读FD/inotify计数记null及permission缺失，仍跟踪PID/starttime；
退出时再检查整个专用systemd cgroup，不能遗漏被重父化的子进程。
3项针对性测试通过。显式恢复登记只允许这个已识别的pre-scene监控错误，旧SDK/模型/物理不改。
原start计数1/5和原session deadline不变；后续expert/student/expert使用全新attempt目录，
不自动无限恢复，也不把已失败首次start覆盖成成功。

## 首次预检历史快照，2026-11-04（已由下述恢复状态更新）

新运行身份 `state_image_20261104_01`，基础提交 `2dc8b58533bba24be5e8f1905279f9fa3372669b`，
分支 `feature/native-dp-training-v2`。**当前 blocked；尚未启动 Kit 或训练。**
这不是后台训练启动验收，也不是 Kit 故障已经修复。

|阶段|真实状态|完成量|
|---|---|---|
|资源预检|blocked：实际 inotify_add_watch ENOSPC|两次有界探针均0/256|
|独立 Kit 完整 episode 门控|未开始|0/5次启动；要求至少连续3次，含专家和学生|
|S-H24-Improved|pending，改进损失/正式配置未实现冻结|0 updates，监督验收0/200|
|I-HistoryDP-00|pending|0 updates，监督验收0/200|
|持久训练 coordinator|未安装、未启动|无PID/service/heartbeat/checkpoint|

本轮只执行 CPU 现场审计、资源探针和相关测试，没有新 GPU/Kit 工作。
不能把已保存的 `queue_intent.json` 当作已经实现并运行的自动训练队列。

## Kit 资源故障：本轮新证据

系统 `max_user_watches=524288`，`max_user_instances=4096`。
两次读取到当前用户 **524190 个 watch FD 引用、86 个 inotify FD 引用**。
这是可读取 `/proc` 的引用总数，继承/共享FD可能重复，3个不可读取PID也可能隐藏用量；
**不宣称它等于内核唯一计数或真正尚余98个。**
新建私有 inotify 实例后，给第一个临时目录添加 watch 就返回 **errno28/ENOSPC**。
计划上限256项，实际分配0；探针FD及临时目录均释放。

|检查|主要结果|
|---|---|
|占用主体|VS Code PID2539563约489435项；PID2305957约34332项|
|目录归属|较大进程来自打开整个 `docs` 的另一窗口；其他项目的 `deliverables` 匹配254736个已监视目录|
|当前错误日志|VS Code renderer仍报告 inotify_add_watch ENOSPC；本项目视频文件的fs.watch也有同类错误|
|FD/线程/内存|按进程读取FD、线程、RSS；自身RLIMIT_NOFILE=1048576；未发现需清理的本轮Kit残留，因为尚未启动|
|磁盘与RAM|预检有约415GiB可用磁盘、约24GiB available RAM；ENOSPC探针来源是watch资源，非磁盘容量|
|生命周期|前轮独立进程 durable exit 缓解已知SDK关闭问题；不能冒称SDK graceful shutdown已修复|

这证明当前确有 inotify 耗尽；**不证明历史所有 FileThread/SIGABRT 都由同一原因造成**。
用户明确要求资源仍接近耗尽就不能通过 Kit gate，因此没有消耗启动预算去碰运气。

### 已实施的局部修复和保护

1. 仅在本项目原工作树及 `bc+rl_RSS_native_v2` 的本地 `.vscode/settings.json` 中补充
   outputs、虚拟环境、缓存、预训练产物、Git objects 的监视排除；保留所有其他设置，原文件和SHA已备份。
2. 项目级规则不能约束另一个以父目录 `docs` 为根的窗口。修改后的实际复查仍0/256、ENOSPC，
   因此该局部改动**尚未解决资源阻塞**。
3. 新增 `tools/kit_resource_preflight.py`：同UID FD/线程/RSS/inotify census、固定上限的实际分配探针、
   全部私有资源清理、不可覆盖的JSON证据。退出码2代表资源阻塞，0只代表资源预检允许继续。
4. 保守要求至少20%估算watch/instance余量且探针全部通过；这只是工程余量，不能代替独立Kit完整episode门控。
5. 五项针对性CPU测试通过：低估计计数下真实ENOSPC仍阻塞；小探针通过但近限仍阻塞；
   资源预检不冒充Kit验收；FD/临时文件清理；零watch实例计数与非唯一统计声明。

没有修改全局sysctl、驱动、共享SDK、其他生产进程，没有删除失败记录或停止VS Code。

### 待确认的最小外部改动

已经向用户请求确认：在父工作区 `/home/CNS2025331827/docs/.vscode/settings.json` 中，
只合并以下两个排除项（不覆盖其他设置）：

```json
{
  "files.watcherExclude": {
    "**/IsaacLab_DexAssemble-sharpa/deliverables": true,
    "**/IsaacLab_DexAssemble-sharpa/deliverables/**": true
  }
}
```

这是其他项目生成目录的**编辑器通知订阅**，不是文件删除或训练进程变更。
之所以单独确认，是因为作用域超出本仓库；目前未应用，也未以等待时间当作同意。
若IDE没有动态释放订阅，需要用户在该窗口重载配置/改为较窄的工作区；不能擅自结束其会话。
资源释放后应先保存一次新预检，再实施最多5次独立Kit验收，不能直接跳到正式训练。
若用户选择保留这些订阅，管理员可依据本次ENOSPC及占用证据评估提高watch上限；
例如1048576能提供更多余量，但不能保证FileThread故障消失，本任务不会自行修改sysctl。

## 模型和训练合同审查：已确认与未实施分开

- 旧H24实际模型为 `visuotactile.state_history.PersistentState`，不是短历史MLP。
  Actor141/robot178/geometry85分别编码，已有GRU256每60Hz tick持续更新。
  训练当前权重重算完整过去前缀，再对96tick段反传；replan不清记忆。
  因此没有重复增加GRU，也没有声称本轮新历史一致性验收已通过。
- 动作仍为 canonical29 的 `transition/target_after_rad`，已包含原专家delay/EMA结果；
  不是实际到达q。部署经真实限位进入原执行器，不再附加delay/平滑。
- 已有 `EncoderKinematics` 使用原URDF，支持link7及五个手指末端**link frame**；
  不是接触点。既有Arm FK梯度证据可复用，但新姿态/指尖损失及跨窗口一致性还未实现验收。
- 已有Image实现为 `PersistentImage('I-HistoryDP', image_contract(...))`：
  冻结DINOv2-S/14、三路448×252 RGB-D、depth融合、视觉GRU和native GRU、条件U-Net。
  Image训练allowlist剔除Actor和privileged geometry；本轮00也禁止tactile与Correction。
- 已有DINO原始缓存和尾帧overlay处理分别在 `native_data.prepare_native_perception` 与
  `models.native_vision.NativeDinoSpatial`；未来必须校验实际在线/离线精度、批处理、时间对齐及GRU，
  不能仅凭存在这些函数宣布本轮感知验收完成。
- 旧 `state_full_training` 仍有“State成功后才可Image训练”的历史门槛。
  新任务明确要求State正常结束后无论成功率如何都继续Image；后续需要新身份下显式调整该条件，
  保留旧run合同，不能当前直接调用旧入口或伪造旧image_gate。

## 已保存的串行训练要求，尚非可运行配置

顺序为 Kit gate → S-H24-Improved → I-HistoryDP-00 → 最终汇总。
两组均从头初始化，H24/commit12、sample/x0、DDIM16/eta0/clipFalse、原29D位置合同、原320/40/40。
保持2段×最多96ticks、Loss B Arm/Hand各50%、AdamW lr1e-4/betas(.95,.999)/weight_decay1e-6、warmup500。
新计划累计1000→3000→6000→10000，cosine统一10000，不在里程碑重置。
FK位置/姿态/指尖及连续性权重仍待train-only代表batch损失/梯度测量，未随意填写。
保留原train4+validation8初态清单及SHA作为以后共同评价参照，test40不作选择。

每组必须真实达到200有效updates及至少3个监测周期，完成loss/梯度/资源/心跳、
checkpoint读回恢复、validation和完整Kit rollout后才能宣布监督验收通过。
本轮这些检查尚未执行；无新训练曲线、逐关节指标、闭环成功率或10000录像可报告。
01/10/11和nominal-action设计均未启动。

## 原实验保护和接续入口

本轮重新读取并SHA核验，三项均与原manifest一致：

|checkpoint|SHA256|
|---|---|
|H24-Sample8000|00783c7db01b8b269746718a3322238beb9906a88eac1fa8e9adc58bfc5b4399|
|H24-Sample10000|9a7cc8c6123563f93b77256104e61e6b357bb38303ede4231fe19fddeb0e17ca|
|H24-Epsilon3000|48236f836b23e74ccbc4f078f8806089f8f08bd71fc70c2184ec4f9cb704e44e|

完整本地路径、大小、原配置SHA见新root的 `protected_checkpoints.json`；原轨迹/录像/故障证据保持。
本地root：`outputs/visuotactile/native_dp_v2/state_image_20261104_01`。
一条只读状态命令（在当前feature工作树执行）：

```bash
cat outputs/visuotactile/native_dp_v2/state_image_20261104_01/status.json
```

后续资源预检须用新文件名，避免覆盖当前两次失败证据：

```bash
python3 tools/kit_resource_preflight.py --output outputs/visuotactile/native_dp_v2/state_image_20261104_01/preflight/03_after_external_resolution.json
```

公开证据见 [preflight_summary.json](evidence/state_image_20261104/preflight_summary.json)。
当前只发布预检代码、测试、报告和轻量证据；不上传权重、原始数据、MP4或其他项目日志。
