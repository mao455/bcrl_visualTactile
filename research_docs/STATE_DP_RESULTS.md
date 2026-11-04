# StateDP 150 anchor等效epoch：训练、闭环与清理

|模型|初态用途|成功/有效|接触诊断|原生抬升|首子目标|超时|
|---|---|---:|---:|---:|---:|---:|
|StateDP 10000|train|0/0|0|0|0|0|
|StateDP 20918|train|0/0|0|0|0|0|
|StateDP 10000|validation|0/0|0|0|0|0|
|StateDP 20918|validation|0/0|0|0|0|0|

计划8初态，实际启动0，有效0，软件缺失0，验收失败0，待执行/待获得有效结果8。最多9次启动。训练初态只作拟合诊断，不并入validation或test。原StateDP/ImageDP正式0/12不变，Image未新训。
原生环境没有离散稳定抓持判据，保持N/A；contact是已有力诊断，不作为稳定抓取。异常关闭与任务失败分开；SDK fast exit不称完整正常关闭。

![阶段对比](evidence/state_dp/stages.png)

|seed|cohort|首次5mm连续3tick偏离：旧/新|专家接触tick|终止|视频|
|---|---|---|---:|---|---|

![训练曲线](evidence/state_dp/training.png)

历史曲线固定train16 / validation40，EMA、H24执行前12。以下端点重算使用完整train320 / validation40，三个checkpoint、归一化、DDIM16/eta0/clipFalse seed7091、loss noise8091和辅助权重相同。Train总loss与Validation总loss可比较；它们不是完整任务成功率。

|更新|anchor等效epoch|cohort|总loss|Arm rad|Hand rad|FK mm|delta Arm rad/tick|动作方向cos|幅值比|
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|

同口径端点离线比较尚未完成，不提前给出150 epoch结论。

|更新|cohort|时段|Arm rad|增量rad/tick|hold基线rad|方向cos|幅值比|FK mm|EE积分误差mm·s|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|

逐关节位置/增量误差、末端方向/幅值、静态tick数量、Train/Validation gap及checkpoint SHA见[端点数值](evidence/state_dp/offline.json)。运动方向只排除专家增量≤1e-6rad或1e-6m的tick；这些tick仍计入RMSE。幅值和cos按episode宏平均，不把微小分母比值池化成任务成功。
原400记录缺少物体过滤的接触力，离线precontact使用现有任一指尖子步力>0.1N的首次诊断事件；闭环专家接触沿用物体过滤力连续3tick的既有定义。二者不混用，不称稳定抓持。
分析使用专家历史观测，明确为teacher-forced离线诊断；闭环始终为模型自主动作，没有专家接管、平滑或控制器修改。
评估source commit `23d250801a4d14433912dfc897529f8eec5a4c38`；运行目录 `/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01`。训练150 epoch后保持证据审查暂停；本次不会自动增加至200–300或启动Image训练。

清理及回归结果见 [CLEANUP_AUDIT.md](CLEANUP_AUDIT.md)。仍在使用的旧目录缓存与checkpoint不按日期删除。
