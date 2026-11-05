# 当前交接：200条StateDP及条件化后续

2026-11-06：原run实际200/200成功、201尝试、43,621控制步、17,972,841,587bytes，60家族44/8/8。原服务已完成5,735/5,735更新并进入闭环；检查时前2条train均601tick超时，整体train4/unseen12尚未完成。best/last完整checkpoint已在CPU读回，EMA/176组AdamW step均5735、LR3e-6；不能把后续授权写成闭环完成。原绝对截止13:57:39 UTC+8不变。

用户新授权的独立源码在`bc+rl_RSS_narrow_followup`，独立root为`outputs/visuotactile/run/narrow_state_followup_20261106_01`。`tools.narrow_followup run`只等待原run正常complete、自有进程全部退出和owner锁释放，再启动新monitor/数据和已记录控制审核；依据实际结果选择最多一次续训、一次新增约60–96条真实数据、一次独立随机训练或结束。新执行期固定8h，等待原任务不使用GPU且有原deadline+15min上限。停止方式为在新root创建STOP；不删除预算、attempt或失败证据，不自动重启。

现有末段证据（3000→4500更新）验证MSE改善1.44%，train8改善42.27%，未见首块幅度约3.11倍、hold误差比1.56；只是平台/泛化差距的预警。是否扩采须等完整闭环和数据/在线控制审核；数据不足不是已证明原因。新的最终16个未见初态预先固定，不进入调度；原unseen12标记开发用途。

新owner实现普通后台等待、固定预算/单次启动/资源监控、阶段转换、最终停止和轻量自动发布，服务名`narrow-state-followup-20261106-01.service`。`tools.narrow_delivery`仅更新现有主文档生成区及一个JSON/曲线，发布前拒绝HEAD、人工正文或工作树变化；只push origin同feature。所有400+200及正式模型、失败证据保留，暂无可确认废弃的新增checkpoint可删。51项CPU回归通过；真实已记录学生tick0/12/95的本体读取误差2.38419e-7，未启动额外GPU/Kit。启动实证见evidence/narrow_followup_startup.json及新root机器状态，不能以此宣称后续阶段执行完成。

下面为原run启动时历史证据，不是当前进度。

分支`feature/narrow-state-data`，工作树`bc+rl_RSS_narrow_state`，root`outputs/visuotactile/run/narrow_state_20261105_02`。已准入60/68直接候选，44/8/8；4条真实pilot全部成功，199/157原速、455/264步alpha.58，实际时长比2.286/1.682。完整154字段hash/120–60Hz对齐及同家族初态一致通过。新几何准入使用34个实际碰撞网格包围盒和实时body pose，旧query误报继续留证。

一次性GPU随机模型优化1次、AdamW/EMA保存恢复、完整/在线历史与未来隔离通过；正式训练0更新，DATA_READY=false。最初GPU smoke在优化前的deadline参数错误保留，唯一补充检查通过。46项CPU回归通过。

普通owner将在生产结束且数据门控通过后自动GPU完整数据smoke→随机StateDP→train4/unseen12闭环。总截止2026-11-06 13:57:39 UTC+8不变；按实测共享负载在生产前重分配采集6h/训练最多9h/末1h评估。达到150–180条且覆盖合格可按预算结束，不无限补数。完整边界暂停/源码切换证据在maintenance/production_cutover；旧v1及旧State143队列不恢复。

首worker整卡均值43.69%、最大完整60s52.64%，缺测0，最低RAM25.66GiB；不称满足旧50%硬门限。资源用shared_recent_v1目标40%、6GiB/12GiB准入。原400逐文件存在/大小和manifest SHA通过，仍70,186,972,525bytes。

状态命令：`bash visuotactile/run_cpu.sh -m tools.narrow_collection status --root outputs/visuotactile/run/narrow_state_20261105_02`。真实后台是否继续、目前条数/更新数以该JSON为准。实验设置和日志/stop/resume见SLOW_EXPERT_RESULTS，训练合同见CONTRACTS。

生产源码b119470；当前服务`narrow-state-data-20261105-02-production.service`已实际完成额外2条自然成功，最近快照7/7、DATA_READY=false、正式0更新。owner/worker/monitor实际在运行。两个production样例已有推理、环境/渲染、numeric、compact和finalize耗时；详细机器证据见evidence/narrow_v2_startup.json。后续由普通程序接续，不再LLM轮询。清理共释放2,306,101,248bytes（2.148GiB），原数据/依赖保留。
