# 有界 tight_insertion 专家数据集：2026-10-28

本轮完成代码接入、有限真实采集、无损转换和零更新检查；**没有训练参数**。因原生非法初始化及Kit启动崩溃，扫描提前停止，未补种子。此报告以实际记录为准，不把旧75%或文献结果移植到本轮。

## 实际结果与停止位置

|阶段|计划|启动尝试|合法动作episode|成功|任务失败|预算中止|非法初始化|软件错误|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|验收|4（上限8）|4|4|4|0|0|0|0|
|位置扫描|52|16|13|13|0|0|2|1|
|合计|56（总上限60）|20|17|17|0|0|2|1|

- `920301` baseline：原生物体盒与桌面重叠，专家动作前拒绝；剩余12变体跳过。
- `920302`：13条件全部合法开始并成功，2589控制步；每格只有**1个起点**。
- `920303` baseline：原生物体与孔座盒重叠，动作前拒绝；剩余12变体跳过。
- `920304` baseline：`libomniclient.so` 启动线程SIGSEGV，returncode=-11；未产生metadata/HDF5，不能判断起点是否合法，余12条件不启动。保留log和空现场目录，没有重试。
- 扫描36项未执行，其中24因非法初始组跳过、12因软件停止。未把非法初始化和启动崩溃塞入机器人成功率分母，也未悄悄删去它们。
- 17条记录3462控制步；验收中有1条legacy，其他16条world_hold。**验收与扫描、legacy与world_hold不能混算正式成功率或默默混入训练。** ManifestSequenceDataset默认只选scan成功记录。

扫描全部没有实际原生外力或力矩曝光（零曝光13/13）。因此本轮位置结果没有动态抗扰证据。验收的`accept_world`确有非零施力，见后文；它单列，不能替代扫描的曝光。

## 可观看内容与可见性限制

本机输出目录：`/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/insertion_dataset_20261028/`。

`previews/`只交付3个并排双相机MP4：

1. `s920302_baseline.mp4`：本轮world_hold基准成功，168帧/2.800s。
2. `s920302_yminus05cm.mp4`：相同起点额外−y5cm，243帧/4.050s，成功。
3. `old_failure.mp4`：**旧cohort_v3 seed910101 original**，38帧/0.633s，原判据fall；重新检查发现初始穿桌，标注为非法初始化诊断，不是本轮合法失败样本。

没有新任务失败可展示，也没有为了凑第4条重跑。视频1×模拟时间、60fps、完整原始图像序列，标明episode/条件/结果/时间和末图缺失。2倍最近邻放大仅用于观看，未改正式图像。真实终止前最后一张可用图是最后动作的pre-frame，没有伪造terminal/reset画面。

实际检查了三视频的首、中、末等帧以及解码帧数。全局与相反侧视角能看到红色L形件、手和孔座，另一视角补充背侧；手仍会遮挡接触。RGB有明显渲染颗粒、分辨率224×126，精细孔间隙不能清晰辨认；没有修改渲染质量或把这些图像称为足以证明精细视觉学习。图像可看与深度有限性已验证，不代表视觉策略有效。

## 原配置、评估覆盖与本轮变化

检查原作者upstream/main与本地原task/资产/config/evaluation，本轮未改这些文件。checkpoint：`pretrained_assembly/tight_insertion/model.pth`，217700467 bytes，SHA256 `7e61015dad660430ab07bab4f715c90c6f76b84cc358da6be81319a6aee09990`。使用原SAPG player deterministic均值、group0对应原embedding ID50；原player仍消耗Normal随机抽样，保留RNG追溯。没有假定存在checkpoint旁的config.yaml。

- 原`PreciseAssembly.yaml`及实际resolved：初始object x/y各±0.10m，z=随机table_z+0.25±0.02m，原随机姿态；table_z=.38±.01m，桌面盒.475×.4×.3m，XY/yaw固定。hole x±.1875m/y±.1m/yaw0，目标XY观测噪声.002m。
- 原DR：obs/action delay max3，object_state delay max10，xyz噪声.01m、rot5°、关节速度噪声.1；force_scale20、torque_scale2、概率loguniform(.001,.1)、decay0、lift及非retract门控。scale不是牛顿。原资产质量/惯量/摩擦/碰撞/动作/判据保留。
- 作者`evaluation/eval_isaacsim.py::_apply_env_overrides`默认关闭多项DR，设连续near-goal成功；这是评估协议。本采集不套用这些覆盖，不把差异自动当bug。
- 本轮仅新研究wrapper增加初始化平移、可关闭的world_hold兼容模式、记录/审计与存储。原120Hz物理、60Hz专家/控制/双相机，五指力及FK每120Hz两子步，均不降采样。

位置设计预先固定：13条件×seed920301–920304，额外0和±x/±y各1/3/5cm；奇偶组交错顺序，基准先行。不因为结果更换seed或幅度。额外5cm使中心落在原±.15m范围，仍在桌面半宽内，但**中心合法不保证倾斜长物体不穿桌/孔座**。

平移发生在PlayEnv原reset采样后、PreciseAssembly生成目标前；不额外抽随机数，z/quat/速度不变。验收和每组配对核验原pose、初始RNG、非object状态、配置、checkpoint一致。记录原pose、最终pose、额外XY和object−hole总偏差。坐标是固定桌平面轴，与worldXY同向；非零table yaw拒绝。例如920302基准object−hole约(18.849,−3.900)cm，并不等于额外0cm。

`initial_geometry.py`读取实际URDF多box碰撞体，用wxyz旋转及15轴OBB分离轴检测；允许接触，正重叠阈值1e−6m。它没有改碰撞，也不涵盖手mesh或PhysX contact_offset。当前任务URDF尺寸直接用于spawn；assets.object_scale用于keypoints而非再次缩放这些box。不能推广到任意缩放/mesh资产。

## 世界施力兼容模式：测过什么

安装Lab2.3.2本地fork/Sim5.1的permanent composer缓存world→body变换。原生DR请求是世界向量；原`legacy`路径在物体转动后实际提交方向可偏离。新`world_hold`在**每个120Hz提交时**用当前root quat的逆旋转转换原请求，再用公开setter `is_global=False` 更新permanent composer；原write照常调用。instantaneous composer活跃则拒绝。原随机分布、质量缩放、门控和抽样顺序不变，未改共享Lab。

seed17 legacy与旧同seed数据的16组数值数组逐位相等；包括动作、目标、关节、接触、物体、外力、延迟与三份RNG。legacy/world_hold首个非零外力前174控制步一致。实际世界力与请求最大向量误差：legacy5.606900N，world_hold9.541399e−7N。两条均成功220步，这不证明修复提高成功率。

world_hold验收：14个120Hz非零样本/7个burst，共0.116667s，峰值3.224302N，∫|F|dt=.271102N·s，世界向量积分约(−.022133,−.116334,−.052651)N·s；力矩曝光0。逐个方向、时刻和原任务阶段在`analysis/accept_world_wrench.json`。没有新增人为方向性施力实验；只验证原生请求按世界方向保持。非零力矩的动态回归仍缺样本。

## 有符号偏移结果

所有格子计划4起点，实际1个合法起点；均1/1成功，单格Wilson95约[20.65%,100%]，**仅探索，尚不能判断方向优劣**。13个变体不是13个独立起点。无短episode过滤，也未用生产成功示范筛选改变实验分母。

|额外位移|−x|+x|−y|+y|
|---|---|---|---|---|
|1cm|1/1，3.417s|1/1，2.717s|1/1，3.533s|1/1，2.750s|
|3cm|1/1，3.150s|1/1，3.600s|1/1，4.917s|1/1，2.850s|
|5cm|1/1，3.000s|1/1，3.850s|1/1，4.050s|1/1，2.517s|

基准1/1，2.800s。时间是模拟控制步数/60，不是墙钟实时性能。配对共同前缀的object距离、关节最大差、完成时长差均在`analysis/report.json`，这些描述运动变化，不解释滑移、卡孔等未测原因。方向图和初始覆盖图见同目录PNG及公开`evidence/insertion_*`。

### 旧数据重新审计的限制

旧cohort40条pilot仍是30成功/10失败的原生任务标记，但本轮新box检查发现20/40初始重叠（10原成功＋10原失败）；另6条旧acceptance中2条重叠。旧源记录保留不改，结果不重写为新成功率。其训练资格和质量需要另审，**不能直接把旧75%作为合法场景专家质量或下一批预算依据**。失败转换测试依然是有效的存储边界测试，不因此成为合法物理场景的策略失败证据。

## 真实存储与格式选择

旧manifest46个文件全部SHA256/字节数匹配。所称“40条约3.55GiB”仅指`cohort_v3/pilot/`40份trajectories.h5：**3,809,235,306 bytes，7060步，3.548GiB**，不含6份acceptance/视频/日志。每控制步平均539551.7bytes；成功30条6515步/3,512,768,258bytes（平均217.17步/117.09MB）；失败10条545步/296,467,048bytes（平均54.5步/29.65MB）。

以下用HDF5 `get_storage_size`实测压缩分配，互斥分类求和等于文件大小；不是解码数组大小：

|分项|实际bytes|
|---|---:|
|pre/post打包RGB-D|3,210,663,605|
|重复obs/depth_m|415,017,841|
|pre/post depth mask|4,359,398|
|触觉与其时间|1,934,844|
|FK/指关节采样（含对应诊断）|13,194,278|
|动作/执行/来源|7,561,573|
|状态/扰动诊断|13,562,036|
|三份RNG诊断|68,118,353|
|其他传感/生命周期字段|8,211,004|
|容器/索引/属性等开销|66,612,374|

旧image单个压缩chunk混有RGB与Depth，**无法把已压缩chunk准确拆成RGB/Depth各多少**；表中诚实合列。字段级shape、dtype、compression、chunk、logical/storage bytes在`storage_audit.json`。新格式分开数组，可单独计量。绝大部分有效post图等于下一pre图；depth还在image第4通道及depth_m中重复。

比较真实成功258步、失败38步、预算8步：

|样本|旧H5 bytes|选用H5＋双观看MP4|FFV1双无损视频＋仅其他数据H5|
|---|---:|---:|---:|
|成功|139,923,082|43,011,928|26,370,641|
|失败|20,330,072|6,669,891|4,162,830|
|预算中止|4,633,931|1,852,942|1,255,449|

FFV1 bgr0真实逐帧逐像素无损，已验证；不是因为视频不能无损而拒绝。暂选HDF5正式RGB便于现有Dataset精确随机切片、顺序回放和避免训练读视频seek依赖，代价比FFV1方案大。选用方案额外观看RGB副本分别3,310,635 /491,668 /121,429bytes。无新依赖安装，PyAV为现有环境依赖；无FFmpeg系统可执行程序假设。

一条本轮基准记录的三个主文件：

|文件|角色|bytes|
|---|---|---:|
|data.h5|正式无损双RGB、float32米制depth、mask、全部非图字段及metadata|25,961,183|
|camera_0.mp4|global_oblique观看派生H264|861,025|
|camera_1.mp4|local_opposite观看派生H264|1,293,206|

总28,115,414bytes。17条主文件577,176,219bytes；其中13条扫描432,176,005bytes。整个新目录（含旧样本转换比较、日志、审计和预览）约0.684GiB；已记录转换前峰值805,171,515bytes（0.750GiB），远低于10GiB。精确最终目录统计见run_status/final_stop.json；不是持续采样的精确峰值。每次预留2GiB最坏2400步原始＋转换空间，实际每步增长有界检查；旧源从未清理。

## 数据调用链与尚未接通的训练环节

`collect → EpisodeWriter完整原始记录 → validate_completion → convert_episode逐字段精确回读＋双视频 → 再次全审计 → manifest原子提交 → 清理该次新临时源 → ManifestReader / ManifestSequenceDataset → encode_episode_chunks → Nominal.encoder/predict_from_condition`。

- `EpisodeFile`兼容旧storage0与新storage1，重构原schema2图像字段；感知2、memory2、trace2独立，不重新解释旧checkpoint。
- Dataset默认成功scan；失败与异常完整保留。分割须按pair_id/seed，`assert_group_disjoint`拒绝同起点泄漏。当前scan只有一个独立起点，无法构造有意义的训练/验证起点划分。
- 输入仅nominal allowlist：双RGB-D/valid、58Dproprio、类别task ID/合法goal mask、真实自身预测字段（本纯专家记录为false）。专家clip前/后、delay后、target、执行结果分别追溯；监督用`command_clipped_pre_delay`，horizon(12,29)及未来有效mask。专家过去命令另列teacher-forcing，没冒充自身计划。
- 120Hz触觉/FK、实际wrench及物体/成功状态诊断完整保存，不进入Nominal。未运行Nominal/Correction，computed mask false；没有假造预测或参数训练。
- **未来BC trainer仍需最小衔接**：确定显式无自身计划baseline（训练/部署一致）或用已识别模型做完整因果旁路/回放；按组划分、只训练集归一化、memory2顺序验证loss、manifest hash绑定checkpoint。现有train.py未完成这些，显式拒绝manifest正式训练；纯专家自身计划训练也拒绝。不能说“目前一个命令就能正式训练新数据集”。
- 原checkpoint rollout加载接口仍存在；本轮没有新BC checkpoint，也未实测新正式BC训练→闭环。Correction监督目标未定，纯专家数据缺真实Nominal计划，训练入口继续拒绝。

## 实际验证

- 旧成功/失败各139字段、预算112字段：逐项dtype/shape/value完全一致；RGB float32确为uint8/255可逆，Depth保持float32；预算保留T+1末post，terminal保留-1缺失引用。
- 新17条转换均139字段完全相同；全图行数值/深度mask、相机时钟、120Hz触觉/FK时钟、非reset终止状态、delay索引及原arm增量/hand绝对目标公式全审计通过；manifest三个文件SHA256复核。
- 258步旧/新模型输入与标签精确一致，chunk7/step1最大差5.36e−7，reset重复精确；新manifest两episode411步相同检查最大差5.07e−7，跨episode联合读取与各自零初始读取精确相等。两次参数及buffer前后SHA256均`8eda8d1d90a73d712a69849b0c85c1d2ff86030a39b3180df24654a586fd2d7a`，parameter_updates=0，有限预测(1,12,29)。不是只做形状检查。
- 测试覆盖无损/边界/坏RGB拒绝/时间错位拒绝、fresh/resume共同完成门槛、OBB旋转/接触/穿入、manifest校验/分组/失败筛选/混配置拒绝。全套35 passed；旧requests依赖warning仍在，未升级共享环境。
- 真实legacy回归220步16组数组逐位一致；world_hold方向验证如上。只渲染CPU场景，没使用已知无效的CPU无渲染路径；原控制/资产未改。
- CLI误用检查只运行应当失败的预检，exit2、无输出目录、无optimizer构造/step；未用train --smoke-only替代零更新测试。

## 下一阶段数量与资源建议（没有执行授权）

先解决起点有效性协议及Kit稳定性，再考虑**400条有效成功**作为第一阶段目标；不建议现在直接批量生产。2/3已检查新独立起点非法，且旧20/40也重叠，必须先区分保留原分布的实验统计与另立配置的合法初始示范生产。不能擅自抬高物体、重采姿态或改物理后继续混算。保留本次停止结果，下一阶段由用户选定协议。

本轮合法尝试平均199.15控制步、33.24MB三主文件，启动到转换审计提交平均45.01s。仅有一个独立合法起点，13/13不是生产成功率估计。以下是**假设情景**，不套用旧75%，保留所有合法失败：

|假设合法尝试成功率|为400成功所需尝试|文件GiB（按当前均长）|串行采集＋转换审计小时|
|---|---:|---:|---:|
|25%|1600|49.54|20.01|
|50%|800|24.77|10.00|
|75%|534|16.53|6.68|

另留2GiB转换空间和日志/检查余量，实际失败可能更长，需分段重新测容量。表不含非法初始化、启动错误与最终分析开销；初始化接受率未知，不宜外推总wall预算。物理60Hz模拟时钟不等于本机60Hz实时采集。

ACT/ALOHA [2304.13705](https://arxiv.org/abs/2304.13705)、DP [2303.04137](https://arxiv.org/abs/2303.04137)、TacDiffusion [2409.11047](https://arxiv.org/abs/2409.11047)原论文入口本轮可访问；任务/硬件/示范条件不同，没有统一数据阈值。HDF5、Zarr/episode_ends、Parquet＋视频的真实代码对比与固定commit见HISTORY_AND_TRACE；本轮无新增第三方代码复制。400/800/1200后续应由独立起点覆盖及获授权后的学习曲线决定，而非文献数字。
