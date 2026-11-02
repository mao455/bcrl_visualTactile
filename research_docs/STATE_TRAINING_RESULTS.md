# State observation/history repair — actual results

## 2026-11-02 19:40 UTC+8：真实中断与恢复

17:19:29流水线在第8个Kit启动时停止，错误为OmniClient
`provider_file::FileThread`/`std::system_error: Invalid argument`，SIGABRT(-6)。
失败目录只有runtime_controls.json，无环境、轨迹或动作。已完成7/12登记专家：
fresh4/4、train3/3，全部合法且原生完整成功，分别168/153/206/161/145/171/181步。
尚余train1/validation4；新BC和DP均0updates、没有学生闭环结果。

这次不是GPU准入阻塞：中断前全卡最近60s12.12%、最近5min15.33%，
freeVRAM19.20GiB、availableRAM约30.67GiB，pause reasons为空。
不能用7/7替代完整12条基础设施验收，更不能据此宣称学生学会。

恢复仅改owner/资源报告/收尾，不改模型、控制、物理、数据、种子、预算和截止。
保存原session/status/失败log；已完成7条跳过。已知动作前FileThread abort允许每stage一次、
全session最多3次；任何started/result/轨迹文件都拒绝此自动恢复。
这不是FileThread内部根因已修复的证据，重复崩溃达到界限会继续保留并停止。
另修复报告丢失partial专家统计及PATH无ffmpeg导致收尾失败，改用已有PyAV完整解码。
5项针对性回归通过；专家视频实际168帧全部解码成功。原截止11月3日17:12:42UTC+8不变。

启动核验（2026-11-02 17:16 UTC+8）：`state-full-20261102-01.service` active，
root `state_full_20261102_01`已推进到独立专家第4条；新学生训练尚未开始。
代码 `c2e9895` 已push并核对origin同SHA。训练截止11月3日14:12:42，绝对截止17:12:42（UTC+8）。
自有进程识别通过；外部worker当时约8.33GiB显存，整卡free14.27GiB、availableRAM35.25GiB，
没有资源暂停。没有外部吞吐指标，不作零干扰承诺。后续阶段由普通脚本自动推进。

## 2026-11-02 最新任务：完整train320，新增独立窗口

最新附件147b678c授权新root `outputs/visuotactile/native_dp_v2/state_full_20261102_01`，
24小时绝对窗口含等待，前21小时训练、最后3小时收尾；旧root02和21:08UTC+8截止保留。
以下旧「正在运行」段落是历史快照：旧服务均已退出，BC/DP均0updates，最终阻断在初态查询。

已查明两个具体问题，并保存原证据：

1. `overlap_shape`读取USD GPrim位姿，而本机CPU PhysX/Fabric没有同步USD变换。
   fresh982501第一object box的USD中心是(0,0,0)，实际中心是
   (0.049823787,−0.028697385,0.619875848)m，旧查询因而碰到ground。
   新查询将原始两个box放到实际物体位姿，先验证USD/URDF半边长与局部偏移完全对应，
   再查询全部场景shape且逐box要求self-hit。没有忽略ground/hand、修改碰撞或推进物理时间。
2. 持久Kit先完成专家轨迹后，3/12重置初态的query返回手部overlap；其中fresh982506
   的全部保存diagnostics与screen逐字段相同，单独新Kit则无重叠、专家161步完整成功。
   这支持查询/场景状态受前序轨迹影响，尚未唯一定位PhysX内部缓存。新对照逐条新Kit，
   保持原reset和控制，用统一初始化与相同命令replay验证，代价是额外启动时间。

前置实测为12个登记初态、9条实际执行且9条成功、3条未执行；不能写成12/12。
另一个独立Kit诊断fresh982506为1/1成功。两组不混入新冻结的专家12分母。
小证据：`evidence/state_shared_20261102/collision_preflight.json`及两份preflight结果。

### 调度与预算

`state_resources.py`新opt-in策略不使用整卡45/50%准入或历史累计均值暂停。
单GPU worker，freeVRAM≥6GiB、availableRAM≥12GiB、disk≥12GB，缺测/内存不足才等待。
监控每秒记录整卡负载、进程PID/显存、自有CUDA峰值和工作/等待时间；最近1min/5min用于调速。
存在其他GPU进程时sleep/work=.25；最近1min>85%为1；1min>92%且5min>85%为2；自动恢复。
这些阈值只调节本任务节奏，不修改sim.dt，不宣称显存代表算力份额。
观察到另一项目FingerEyeLab的短时cuRobo/rollout worker，PID随任务更替；只读查看，无信号/配置修改。
没有稳定的外部吞吐接口，因此不能证明完全无干扰。原root02全卡max60s62.295%仍按旧规则记违规。

### 固定学习与评价

- 两种持续历史pilot各300updates、原train4/val4；检查有限loss、非零GRU梯度和下降。
  小样本闭环0成功不阻止full320。原模型、LossB、AdamW、warmup500、EMA.75不变。
- 完整BC320/val40里程碑1000/3000/6000/10000，完整DP320/val40里程碑
  1000/3000/6000/10000/19000；≥6000后连续两里程碑无1%验证改善则停止。
  每里程碑train2/val2自主闭环，最终best在fresh4/train4/val4闭环。
- 训练指标使用固定前16train作为低成本曲线并明确标注；优化覆盖全部train320，
  验证全部40。完整当前权重prefix重算+96tick TBPTT、batch两episode segment不变。
  数值cache避免每步HDF5读取，没有缓存旧权重hidden或读取RGB。
- 选择按validation40的prefix12 arm/hand等权命令MSE冻结EMA；BC/DP选择均冻结后，
  test40各一次离线；sorted first20 test IDs对expert/old/BC/DP闭环，非法初态原样报告，不替换。
  此test集曾用于历史实验，不能声称从未接触的全新外部测试集。
- checkpoint包含model/EMA/EMA计数/AdamW/全局与独立RNG/采样顺序cursor/源码及配置hash；
  完整更新边界每250保存，同里程碑恢复一次，已完成轨迹不重复。早停不是任务成功。

11项针对性测试通过，包括优化器+EMA+采样/RNG恢复逐位一致；既有81894 Actor验收复用，
不改变Actor/GRU/normalizer源码。新完整训练和闭环数值以新root结果为准，本段不预报成功。


本轮新root `outputs/visuotactile/native_dp_v2/state_history_20261102_02`，base `ec84bd3`。
这份文档区分已测事实和正在运行的阶段，不能用代码通过或离线拟合代替任务成功。

## 修改前

旧正式State19000updates：正式0/20；前轮fresh0/8。StateBC4000 saved train0/4、validation0/4，
全部601timeout、无lift/子目标；teacher-forced train arm/hand RMSE .02365/.03965，validation .98176/.71937。
短窗ActorBC也没有建立闭环能力。前轮single expert6/8、four8/8，但single6/8初始盒重叠，不能叫合法基准。
历史replay3/8和新same-condition2/2精确重现保持原分母；不覆盖旧结果。

## 新实验登记

绝对6h窗口包含资源等待；不自动重置截止。最多38条控制轨迹：expert12、same-command replay2、旧State8、
recurrentBC8、recurrentStateDP8。先固定合法初态，再观察结果。train4/val4身份不变，test不用于本轮学习/选择。
BC≤2000、DP≤1000；各在250及终点保存、按validation前12步命令MSE选诊断权重，再闭环。
离线best只是固定的小预算选模方法，不是闭环best或成功证据。没有自动扩大19000或ImageDP训练。

## 已测

- 92项新旧回归通过（8项新契约+84项既有；最后运行见verification.json）。
- 原400/旧numeric400/旧权重和native源保护hash核验通过。
- 81,894条实际Actor观测标准化误差0，静态group embedding完全相同。
- 8条真实teacher-forced controller/完整前缀/逐步history数值验收通过；细项见输入/历史文档。

## 正在执行，尚无任务结论

普通persistent owner已启动，依次执行初态screen、在线专家、同命令控制/随机流对照、旧State、BC、StateDP。
仿真闭环成功率、训练曲线、首个实质偏离和资源总结待实际产物；当前不能声称新学生成功或训练完成。
machine-readable `status.json`、各stage `result.json`为进度来源，不从服务active推断研究结果。

## 初始查询配置缺陷与保留的前置尝试

root01的headless配置`enable_scene_query_support=false`导致PhysX overlap查询返回空。
虽然OBB合法，空query不能支持实际shape无重叠结论。发现后停止自有owner，保留全部产物：
fresh982501专家168步成功，fresh982502已开始后被停止；二者均不纳入修正后的合法基准分母。
没有训练。root02仅打开read-only scene query，并用包围真实object的box查询作阳性对照；
原deadline继承不重置，总控制starts上限40=原38+两个失效前置尝试。候选顺序、原任务/成功/物理参数不变，
不利用前置结果选择种子。新的实际query可能改变几何合格集合，所有拒绝保留。

## 本次代码交付时的资源阻塞

修正后root02的preservation/parity已完成，freeze尚未执行初态screen，BC/DP均0更新。
整卡后台负载使资源监控保持RESOURCE_PAUSE；自有Kit停在资源等待，没有修改其他作业。
完整60s已经观测到>50%窗口，因此不能称全程资源合规，也不能将整卡利用率归因到本进程。
绝对截止仍为2026-11-02 21:08:22 UTC+8。owner会在满足原准入条件后自动继续，
到期则停止自有worker并保存阻断原因；没有扩大预算。
`state-history-20261102-finish.service`在owner退出后生成实测表/首差/曲线和完整视频decode结果，
保存至`evidence/state_history_20261102/results`，不自动提交未经审查的结果。
所以本轮当前交付是实现和数值验收，**真实合法专家基准、BC/DP学习及闭环目标尚未完成**。



## 自动收尾实测表（待人工复核与Git提交）

此表只读取普通owner结束后的产物，不新增训练或rollout。旧/前置失效样本不混入新分母。

|组|cohort|已尝试|合法|完整成功|native lift|至少1子目标|
|---|---|---:|---:|---:|---:|---:|
|expert|fresh|4|4|4|2|4|
|expert|train|3|3|3|2|3|

|组|updates|train arm/hand RMSE|validation arm/hand RMSE|
|---|---:|---|---|

最终owner状态：`{"state": "blocked", "reason": "RuntimeError('expert_single_07 failed: exit=-6')", "utc": 1793611169.3465292}`。

逐条reset误差、接触前后首差及阶段误差见公开`summary.json/trace_diagnostics`。
这些观测不唯一证明网络失败机制；同输入/同命令验收与完整成功分开报告。
