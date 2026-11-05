# 当前交接：新窄域短监督通过，后台生产运行中

分支`feature/narrow-state-data`，工作树`bc+rl_RSS_narrow_state`，root`outputs/visuotactile/run/narrow_state_20261105_02`。已准入60/68直接候选，44/8/8；4条真实pilot全部成功，199/157原速、455/264步alpha.58，实际时长比2.286/1.682。完整154字段hash/120–60Hz对齐及同家族初态一致通过。新几何准入使用34个实际碰撞网格包围盒和实时body pose，旧query误报继续留证。

一次性GPU随机模型优化1次、AdamW/EMA保存恢复、完整/在线历史与未来隔离通过；正式训练0更新，DATA_READY=false。最初GPU smoke在优化前的deadline参数错误保留，唯一补充检查通过。46项CPU回归通过。

普通owner将在生产结束且数据门控通过后自动GPU完整数据smoke→随机StateDP→train4/unseen12闭环。总截止2026-11-06 13:57:39 UTC+8不变；按实测共享负载在生产前重分配采集6h/训练最多9h/末1h评估。达到150–180条且覆盖合格可按预算结束，不无限补数。完整边界暂停/源码切换证据在maintenance/production_cutover；旧v1及旧State143队列不恢复。

首worker整卡均值43.69%、最大完整60s52.64%，缺测0，最低RAM25.66GiB；不称满足旧50%硬门限。资源用shared_recent_v1目标40%、6GiB/12GiB准入。原400逐文件存在/大小和manifest SHA通过，仍70,186,972,525bytes。

状态命令：`bash visuotactile/run_cpu.sh -m tools.narrow_collection status --root outputs/visuotactile/run/narrow_state_20261105_02`。真实后台是否继续、目前条数/更新数以该JSON为准。实验设置和日志/stop/resume见SLOW_EXPERT_RESULTS，训练合同见CONTRACTS。

生产源码b119470；当前服务`narrow-state-data-20261105-02-production.service`已实际完成额外2条自然成功，最近快照7/7、DATA_READY=false、正式0更新。owner/worker/monitor实际在运行。两个production样例已有推理、环境/渲染、numeric、compact和finalize耗时；详细机器证据见evidence/narrow_v2_startup.json。后续由普通程序接续，不再LLM轮询。清理共释放2,306,101,248bytes（2.148GiB），原数据/依赖保留。
