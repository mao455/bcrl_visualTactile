# 当前初始化与原生扰动

本页解释来源，不是新的运行授权。唯一数值配置为[configs/tight_state.json](../configs/tight_state.json)，当前验收状态见[结果](STATE_DP_RESULTS.md)。旧v3/v4分布和历史小样本报告在Git642f5a2保留。

|来源|当前解释|
|---|---|
|初始物体/孔位|配置中心周围物体XY±2mm、相对孔位XY±3mm、SO(3)旋转球1.5723103°；direct_configuration真实读取范围；孔位仍位于原生范围内|
|桌高/物体高|桌z[.379,.381]m，物体距桌原点[.248,.252]m，不加沉降等待|
|初始关节|原生默认姿态与限位内随机样本的lerp .0075；不是±.0075rad；初速度±.03rad/s|
|观测噪声与延迟|保留原生对象队列10、对象XYZ噪声std.01m、随机轴旋转±5°、qd噪声std.1rad/s、whole-observation队列3；配置不等于逐tick实测噪声|
|动作延迟|原生3槽随机选择，实际index保留于control_trace；专家native执行保持，学生已处理绝对目标不再次作用delay/EMA/alpha|
|外力|原生force20/torque2，reset概率log-uniform[.001,.1]；随机向量按质量缩放、lift/retract门控；20不是20N上限，也不保证每条受力|
|合法性|真实PhysX当前姿态的shape overlap与几何检查；动作前拒绝单列，不把查询结果冒称signed penetration depth|

每条attempt保存seed、完整narrow_reset、真实native config与初态SHA、source expert SHA。开发原速/慢速的可观察初态和native config逐条相同；不声称恢复PhysX/contact内部完整状态。同seed不同动作会改变事件分支，不能称后续扰动逐tick严格配对。

当前21个专家物理attempt全部合法；6个原速、15个慢速，任务结果和减速拒绝理由均保留。另一次错误解释器引起的启动中断发生在首条physical capture之前，计软件错误，不伪造初态。未冻结正式范围和split，因为减速方案未过门控。
