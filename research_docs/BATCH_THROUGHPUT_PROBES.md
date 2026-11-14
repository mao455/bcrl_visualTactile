# 可扩展完整学生执行的短吞吐探针

`tools.tight_fresh_batch_eval` 支持显式 `--max-control-ticks 36`，先用有限真实控制tick选择环境数和执行后端，再进入正式采集/训练/自主评估。`--count`没有四环境上限；可登记8、16、32、64、128、200等正整数，实际资源准入和现场吞吐决定采用值。`--roles A`或`--roles B`只运行一个完整来源pair；缺省仍运行A/B。`--backend-json`使用[执行后端合同](EXECUTION_BACKEND_SCALING.md)，请求随plan冻结并真正传到环境构造。

短探针明确登记 `inference_scheduler_index_mode=host`，与新正式batch runner及其门槛使用同一调度器索引路径，实际传入完整学生controller；不把legacy额外同步开销误作新正式流程的成本。没有上限的旧缺省路径仍使用legacy，不静默修改旧评估身份。

每个合法活动lane使用其当前真实三视角RGB-D、60Hz持续历史、完整DP及后端；物理120Hz、控制60Hz、DP5Hz、29维绝对关节目标保持。探针检查三相机内容有效、当前时间戳一致、相机frame严格推进，以及图像/历史/本体/后端计数等于实际控制tick、规划数等于向上取整tick/12。数值trace与逐lane推理计数保存，短探针不是新训练数据。

达到登记tick上限仍活动的lane标记 `censored=true`、`censor_reason=diagnostic_control_limit`；没有伪造native timeout或原生终止理由。短探针所有lane的`success=null`、`autonomous_evaluation_eligible=false`、`accepted_for_training=false`。即使上限之前自然完成，真实完成保存在`observed_native_completion`，仍不并入自主成功率或正式评估分母。原生自然结束、诊断截断和非法初态分别报告。缺省不传上限时保留完整自然执行及原生终止界限检查。

保存共享资源monitor样本及原日志路径、worker RSS/CUDA allocator峰值（不含renderer/driver的独立分配）、启动时间、模型加载等role总wall、控制loop/数值flush、包含trace压缩和计数验收的wall吞吐。短探针显式在控制器及环境阶段末同步CUDA以减少时间错归，记录该额外同步；最终正式流程仍需实测。共享monitor原始完整时间线是资源判断依据，短样本不声称完整60秒均值。

本入口只能给出当前配置的短执行成本、传感器/推理计数与实际软件故障证据。36tick通常覆盖不到抓住、转正或插入；不能因此声称接触物理已验收或策略改善。Tiled移动标定/GPU物理仍需少量完整真实任务、对应接触/目标/成功检查；最终吞吐优化必须进入真实“采集→训练→自主评估”周期。新增CPU测试通过不等于Kit现场验收。

## 显式视觉lane微批候选

`tight_fresh_batch_gate`、`tight_fresh_batch_eval prepare`、`tight_fresh_batch_run prepare`及新训练初态入口`tight_fresh_expansion`现在接受`--vision-microbatch <正整数>`。它独立于物理/相机后端：正式与扩充计划写入`batch.vision_microbatch`，探针写入`vision_microbatch`，执行实际传到完整学生与`NativeDinoSpatial.forward_lanes`；旧计划缺省仍为2。候选不得热改当前冻结`7c27553`执行树。先以相同参数通过真实25tick完整命令门槛，再测实际吞吐，之后才决定正式流程是否采用。

每组处理M个环境、共3M张当前相机图像；末组不足M时复制最后一个真实环境的当前三视角，只保留真实lane的输出。M=2保持原六图像kernel，普通单观察`NativeDinoSpatial.forward`固定走此缺省，`FrozenDinoSpatial.forward`与既有缓存合同均未修改。显式M改变CUDA kernel形状，可能改变浮点结果；没有跨图像注意力不等于逐位等价，也不能以CPU模拟backbone检查替代真实命令门槛。原门槛的2e-5rad不放宽，串行参照始终M=2。

`vision_execution`记录版本`current_observation_lane_microbatch_v1`、M、每kernel图像数、实际kernel调用数、真实lane/图像数、padding lane数、总计算图像数，以及每lane真正进入历史的观察数。计数是整批合计，不能再按lane相加；padding不算观察、动作监督或新增数据。N活动lane每tick调用`ceil(N/M)`次DINO；例如N=200/M=128为2次、56个填充lane；当N小于M也会显式填满M，额外成本不会隐藏。该计数描述计算预算，不是VRAM上界或速度承诺；增加M是否值得由实际峰值内存和有效吞吐决定。

本候选39项CPU检查通过：微批1/2/4/8的真实输入路由、末组填充、默认六图像路径、正整数校验、计数及不活动lane、正式/探针登记、正式接管波次中不同终止/历史/标签边界。尚未启动新的CUDA/Kit比较，未宣称加速、缓存数值等价或自主行为改善。
