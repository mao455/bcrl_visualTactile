# StateDP 150 anchor等效epoch：训练、闭环与清理

|模型|初态用途|成功/有效|接触诊断|原生抬升|首子目标|超时|
|---|---|---:|---:|---:|---:|---:|
|StateDP 10000|train|0/4|4|1|0|2|
|StateDP 20918|train|0/4|4|2|0|2|
|StateDP 10000|validation|0/4|3|0|0|4|
|StateDP 20918|validation|0/4|4|0|0|4|

计划8初态，实际启动8，有效8，软件缺失0，验收失败0，待执行/待获得有效结果0。最多9次启动。训练初态只作拟合诊断，不并入validation或test。原StateDP/ImageDP正式0/12不变，Image未新训。
原生环境没有离散稳定抓持判据，保持N/A；contact是已有力诊断，不作为稳定抓取。异常关闭与任务失败分开；SDK fast exit不称完整正常关闭。

![阶段对比](evidence/state_dp/stages.png)

|seed|cohort|首次5mm连续3tick偏离：旧/新|专家接触tick|终止|视频|
|---|---|---|---:|---|---|
|984101|validation|3 / 3|68|{'fall': False, 'max_successes': False, 'hand_far': False, 'dropped': False, 'timeout': True}|[本地视频](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01/stages/state_interim_04/attempt_1/output/r931005_1793209376_e00_p000055/rollout.mp4)|
|984102|validation|6 / 5|43|{'fall': False, 'max_successes': False, 'hand_far': False, 'dropped': False, 'timeout': True}|[本地视频](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01/stages/state_interim_05/attempt_1/output/r931005_1793209376_e00_p000057/rollout.mp4)|
|984103|validation|7 / 7|46|{'fall': False, 'max_successes': False, 'hand_far': False, 'dropped': False, 'timeout': True}|[本地视频](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01/stages/state_interim_06/attempt_1/output/r931005_1793209376_e00_p000067/rollout.mp4)|
|984104|validation|4 / 5|65|{'fall': False, 'max_successes': False, 'hand_far': False, 'dropped': False, 'timeout': True}|[本地视频](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01/stages/state_interim_07/attempt_1/output/r931005_1793209376_e00_p000121/rollout.mp4)|
|982401|train|9 / 10|51|{'fall': True, 'max_successes': False, 'hand_far': False, 'dropped': False, 'timeout': False}|[本地视频](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01/stages/state_interim_00/attempt_1/output/r931005_1793209376_e00_p000000/rollout.mp4)|
|982402|train|3 / 9|63|{'fall': False, 'max_successes': False, 'hand_far': False, 'dropped': False, 'timeout': True}|[本地视频](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01/stages/state_interim_01/attempt_1/output/r931005_1793209376_e03_p000159/rollout.mp4)|
|982403|train|10 / 10|55|{'fall': True, 'max_successes': False, 'hand_far': False, 'dropped': False, 'timeout': False}|[本地视频](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01/stages/state_interim_02/attempt_1/output/r931005_1793209376_e02_p000059/rollout.mp4)|
|982404|train|6 / 16|88|{'fall': False, 'max_successes': False, 'hand_far': False, 'dropped': False, 'timeout': True}|[本地视频](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01/stages/state_interim_03/attempt_1/output/r931005_1793209376_e01_p000153/rollout.mp4)|

![训练曲线](evidence/state_dp/training.png)

历史曲线固定train16 / validation40，EMA、H24执行前12。以下端点重算使用完整train320 / validation40，三个checkpoint、归一化、DDIM16/eta0/clipFalse seed7091、loss noise8091和辅助权重相同。Train总loss与Validation总loss可比较；它们不是完整任务成功率。

|更新|anchor等效epoch|cohort|总loss|Arm rad|Hand rad|FK mm|delta Arm rad/tick|动作方向cos|幅值比|
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
|10000|24.0407|train|0.002942336|0.0115009|0.0538956|9.529|0.00842606|0.24291|3.2354|
|10000|24.0407|validation|0.009609138|0.0206615|0.111931|17.048|0.0100987|0.19904|3.5739|
|16584|100.0054|train|0.0002465709|0.00374974|0.0139912|3.032|0.00407255|0.49558|1.7937|
|16584|100.0054|validation|0.01833578|0.0193283|0.107983|16.47|0.00795393|0.32671|2.4888|
|20918|150.0101|train|0.0001122768|0.00237603|0.00955311|1.9572|0.00295834|0.62639|1.4595|
|20918|150.0101|validation|0.01997827|0.0192881|0.107426|16.366|0.00753911|0.38577|2.2159|

|更新|cohort|时段|Arm rad|增量rad/tick|hold基线rad|方向cos|幅值比|FK mm|EE积分误差mm·s|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
|10000|train|first12|0.011485|0.008143|0.016772|0.2415|3.2141|9.2229|1.631|
|10000|train|first24|0.011462|0.0081986|0.032048|0.25797|3.1826|9.4507|3.3509|
|10000|train|precontact|0.010786|0.0078554|0.10856|0.27742|3.217|8.7933|8.8194|
|10000|validation|first12|0.022148|0.0097697|0.016573|0.19484|3.6081|16.929|3.0422|
|10000|validation|first24|0.021511|0.010134|0.031965|0.21175|3.6725|18.104|6.4905|
|10000|validation|precontact|0.020811|0.0095031|0.10498|0.22691|3.6306|17.344|18.129|
|16584|train|first12|0.0034571|0.0038893|0.016772|0.5193|1.7715|2.6568|0.46054|
|16584|train|first24|0.0033563|0.0037531|0.032048|0.53838|1.727|2.5955|0.90188|
|16584|train|precontact|0.0034564|0.003579|0.10856|0.54823|1.7255|2.7133|2.6789|
|16584|validation|first12|0.020542|0.0070677|0.016573|0.33471|2.3543|16.162|2.8797|
|16584|validation|first24|0.019578|0.007649|0.031965|0.35906|2.4392|17.63|6.3289|
|16584|validation|precontact|0.018803|0.0070801|0.10498|0.37172|2.4516|16.657|17.413|
|20918|train|first12|0.0021967|0.0027915|0.016772|0.65286|1.4398|1.7269|0.29993|
|20918|train|first24|0.002112|0.0026837|0.032048|0.67177|1.4138|1.6716|0.58237|
|20918|train|precontact|0.002147|0.0025372|0.10856|0.67881|1.4093|1.724|1.6957|
|20918|validation|first12|0.02026|0.0065016|0.016573|0.39595|2.061|15.953|2.8267|
|20918|validation|first24|0.019372|0.0071099|0.031965|0.42556|2.1374|17.537|6.2629|
|20918|validation|precontact|0.018709|0.0066362|0.10498|0.43029|2.1786|16.492|17.203|

逐关节位置/增量误差、末端方向/幅值、静态tick数量、Train/Validation gap及checkpoint SHA见[端点数值](evidence/state_dp/offline.json)。运动方向只排除专家增量≤1e-6rad或1e-6m的tick；这些tick仍计入RMSE。幅值和cos按episode宏平均，不把微小分母比值池化成任务成功。
原400记录缺少物体过滤的接触力，离线precontact使用现有任一指尖子步力>0.1N的首次诊断事件；闭环专家接触沿用物体过滤力连续3tick的既有定义。二者不混用，不称稳定抓持。
分析使用专家历史观测，明确为teacher-forced离线诊断；闭环始终为模型自主动作，没有专家接管、平滑或控制器修改。
评估source commit `23d250801a4d14433912dfc897529f8eec5a4c38`；运行目录 `/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/anchor_assessment_20261105_01`。训练150 epoch后保持证据审查暂停；本次不会自动增加至200–300或启动Image训练。

清理及回归结果见 [CLEANUP_AUDIT.md](CLEANUP_AUDIT.md)。仍在使用的旧目录缓存与checkpoint不按日期删除。

![代表轨迹](evidence/state_dp/representative.png)
