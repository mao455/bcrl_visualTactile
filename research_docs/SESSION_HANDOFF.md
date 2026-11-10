## 2026-11-10 接手：Transformer/实际配置方法实现与门控

起点8e67fb7，工作树bc+rl_RSS_palm_eval干净后开始；原主工作树/窄域工作树未提交文档未动。现有400/Core/旧成功保护。当前run为outputs/visuotactile/run/tight_transformer_20261110_01，机器status/各gate为准，正式更新暂0；102项定向CPU测试通过，详细启动进度随后以status/launch为准。

新旧差异、监督时序、网络/门控、实验矩阵和有限预算见CONTRACTS当前段。CPU验证实际train166+validation30共65630tick，test未读；新Transformer756581参数，旧GRU显式保留用于成功对照。清理398文件释放823.789063MiB，0模型权重删除。

首Kit gate完成2reset/4pose/0step、PhysX错误0，但geometry metadata header缺config导致worker异常。原失败完整保留；第二次physics_gate_repair同样完成零步功能检查，但SDK关闭停滞；只中断本次父进程并清理自有子进程，保留中断凭据。第三次physics_gate_exitfix复用既有durable-results OS退出，worker0/全日志PhysX0/进程与GPU census退出核验通过；不称SDK graceful close。第三次几何回调26/26明确INVALID_PARSING(2)：错误地把CollisionAPI包装Xform当Mesh传入。按具体API缺陷做一次第四复验，预算修订留账、原三小时期限不变，不改任何几何/物理容差。第四次physics_gate_meshfix实际通过：worker0、完整日志PhysX0、所有自有进程退出；26实际Mesh/属性同SDK重建凸包，当前姿态FK/查询一致。固定6训练状态6/6投影可行，λ_geom=0.7029496672279293，手臂新增梯度0、手部梯度有效；QG00/QG01准入。内部力前提未满足：现有5distal sensor接口可读但startup为零接触，缺全手/物体环境接触与所需wrench及动作响应；实际扰动0，未建立F(action,...)，λ_int禁用。不能把几何消融称完整内力方法。

root配置/数据已prepare；大训练尚须GPU真实DDIM/cache/controller、保存恢复、每profile8update吞吐门控。门控source冻结与绝对期限不重置。外层owner负责训练→验证→残差→固定开发→冻结后test22→CPU整理，任何依赖失败停后续、不补seed。最终状态及是否启动必须读机器文件。

以下历史handoff不作为当前任务队列。

# 当前交接：Core/Auxiliary有限任务已完成，学生未成功

2026-11-08，使用修复worktree `/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`，branch `feature/tight-state-foundation`。当前实验root `outputs/visuotactile/run/tight_state_20261107_01/composition_06`，status=complete/evaluated；11月8日13:21:54 owner/monitor/Kit退出。原训练PID2805678、owner2393723、monitor2393729、最终eval3127549与两版资源guard均已核实退出；不可用历史PID冒充活进程。

采集源码4d43a3975c9bd153ff90d7c13e03ca9a0e6aa917，训练/评估fdb7ea94a1457de57ff2a9fc21d1a9486742d5ab，仅后续闭环成员选择修订。源代码211文件/计划指纹及基础accepted gates有效；61项定向/准入回归＋1项成员回归通过，不重开168/510tick基础审计。完整episode持续GRU不是64tick截断；29D绝对处理后目标、H24/commit12与所有模型/物理条件不变。

最终55refs=54成功+1动作前拒绝；378slow=359成功+11任务失败+8动作前拒绝。208Core/50家族，train156/36家族、validation30/7、test22/7；15Aux/11家族另存，未进入训练或统计。另2完整Aux未选保留。旧20严格10/20不改，旧2数值Aux缺图不复用。第7轮达到Core软目标停止，62计划身份未启动；完整数据、阶段运动、原生结果与分母在当前结果/CSV/机器证据。

Fresh训练8500步、442012draws、100.2068 anchor等效epoch，全部4411anchor覆盖57–150次；最低80曝光已过，最后4验证改善.9386%且泛化差距触发预登记early stop。sufficiency=true仅有限曝光/趋势，不是全局收敛。完整恢复passed、旧学生loaded=false；best8500 EMA SHA6940b066cf8296d8e90203c0aa714f2ad780d5440c6128b89ddc5698bbbaa4da。test未选模。

闭环27/27合法：train0/10、validation0/7、新初态0/10；24timeout/3fall，1例第一子目标后fall。首10新初态零成功，剩余90未启动；不扩分母，不新增seed/训练/参数搜索。全部实际执行器与canonical actor一致性误差0。剩余是自主抓取/抬升/连续插入稳定性；不同于旧275数据，不作因果matched比较。

清理后225合格payload SHA、135拒绝numeric SHA、6保护SHA、manifest/stat/cache指纹及3CLI通过；135拒绝RGB-D+1重复checkpoint释放36.191101GiB。best保留原字节，last是完整状态相同的硬链接别名；旧last SHA映射保留，完整恢复读回8500/442012。原400/275、专家/资产、开发证据与全部失败分母不动，不重复计算旧7.510/2.705GiB。

整卡均值28.389%、最高完整60s59.632%、缺口0，最低VRAM4154MiB/可用RAM16.85GiB；本轮资源并非全程合规。v1 SIGSTOP导致比例限流重复等待，v2改原gpu.lock在计算外等待，不改训练状态、没重启trainer、没重置deadline；原日志和源码receipt都保留，guard正常exit0。不要再用SIGSTOP节流。

详情以CONTRACTS、STATE_DP_RESULTS、CLEANUP_AUDIT和evidence/tight_composition_results_20261108.json为准。不可变evidence/tight_composition_20261107.json是启动前登记，started=false不代表当前状态，勿改其SHA。没有新的待执行实验或ImageDP/触觉授权。本次只正常提交/推送审查后的当前origin feature内容，不上传数据/checkpoint或重写历史。
