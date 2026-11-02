# State observation/history repair — actual results

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
