# StateDP / ImageDP 抓取失败归因诊断

## 审阅结论：Arm的离线误差尚未胜过零运动基线

**最有力的解释是接触前运动拟合已经不足，随后自主历史中的误差继续累积；尚不能把全部失败归因于某一个机制。**
专家处理后target经过相同绝对接口2/2成功，已排除“该接口根本无法执行专家任务”的解释。原两模型正式结果仍各0/12。

在完全相同validation40、8670个有效tick和first12口径下：

|指标|保持当前target|StateDP-10000|ImageDP-10000|
|---|---:|---:|---:|
|Arm RMSE rad|0.01554|0.02066|0.01842|
|Hand RMSE rad|0.19648|0.11193|0.11388|
|移动Arm方向cosine|不定义|0.654|0.682|
|移动Arm幅值比|0|1.170|1.109|

Arm误差比保持target更大的验证轨迹：State 37/40，Image 29/40。Hand优于保持target：State 40/40，Image 40/40。
这只是同口径离线零运动参照，不声称保持target的机器人能完成任务。复用现有模型验证结果，未新增推理或训练。
不能只看0.02rad绝对数值说“已经拟合得很好”；Arm相对本任务的精细运动尺度仍不够，Hand能降低平均误差也不保证正确空间构型下的稳定抓取。

### 最小修复与消融决定

- StateDP：仅精度修复 0/2 完整成功；确定性K0 0/1、K4 0/1。具体有效覆盖率和软件故障见下表，缺失不能算失败。
- ImageDP：仅精度修复 0/2 完整成功；确定性K0 0/1、K4 0/1。具体有效覆盖率和软件故障见下表，缺失不能算失败。

9条新诊断均无完整成功，其中2条K4前缀不匹配，单列为非严格因果结果。舍入修复已通过严格数值合同，但未解决闭环；K4即使降低跳变，也没有恢复稳定任务能力。
因此不把这一小数值修复或K4默认推广为“抓取修复”，旧默认和权重继续保留。

**严格消融验收：StateDP PASS，ImageDP FAIL。** State确定性K0/K4前12tick六类字段逐值相同；共同235tick内手指边界RMS从0.07588降至0.03025rad，235tick fall变601tick timeout，仍无首子目标或完整成功。
Image的固定输入独立进程前向/DINO检查逐值相同，但真实Kit首计划已差0.0032769rad（早于K4）；初始Actor/物理导出一致。原始live RGB-D/在线DINO/condition没有保存，无法唯一定位到渲染或网络运行期计算。
保留Image原生timeout结果，不把它作为K4收益。停止新增Kit，不用剩余单次启动去凑严格一对。Image下一轮优先记录并逐值对照原始图像、DINO、condition及随机流；不得直接用本次未匹配结果调参。

持续接触丢失靠近重规划（±3tick）的次数：State 5/10，Image 4/7，成功专家8/14；约7/12的控制tick天然处于这种邻域。仅凭时间邻近无法证明因果。
模型接触丢失窗口均有部分joint-limit clipping，但这是常见现象；专家比较的是post-limit target，不能把它的零额外clipping当作未发生上游限位的证据。
**是否改网络/损失：当前均不改，也不重训。** 下一轮最有价值的是固定前12tick、只改变当前target锚定/动作增量连续性的诊断，并比较匹配新鲜专家观测下的学生动作与离轨后的专家shadow纠正。
若证实专家输入下仍无法胜过移动基线，优先提出相对动作/移动阶段监督的有界对照；若主要是离轨后无恢复能力，再提出roll-in纠正数据。两者都不等于授权从头重训两组10000。
[审阅机器证据](evidence/grasp_diagnosis_20261104_01/review.json) · [validation40保持target基线](evidence/grasp_diagnosis_20261104_01/validation_hold_baseline.json)。

---
**原正式结果保持 StateDP-10000 0/12、ImageDP-10000 0/12。没有新训练，也没有使用 test40。**

本轮 `grasp_diagnosis_20261104_01`；当前 `completed_with_unmatched_pairs`。24条原闭环和12条同初态成功在线专家均只读。

|检查|判定|实际证据与边界|
|---|---|---|
|专家29D处理后target经DP绝对位置接口独立回放|PASS（复用）|2/2完整成功，168/153ticks；提交目标/执行器误差0，14类记录逐字节一致；原生RL指令回放没有替代此项。|
|回放接口仍适用于当前模型|PASS|旧执行函数AST和环境/原控制核心SHA与当前一致；旧session初始commit过早，采用实际runtime amendment文件SHA对应c7fcbb0。|
|24条模型与12条在线专家初态配对|PASS|可观察初态、环境RNG和所有已导出字段相同；不声称PhysX隐藏状态完全相同。|
|标签、29D顺序/rad、动作索引、60/120Hz、反归一化|PASS|复用400数据审计；新完整轨迹标签第0项/时间误差0、逆变换最大1.20e-7rad、全部24条实际target到执行器误差0。|
|原部署输入严格数值一致性|FAIL（已保留）|归一化proprio反算q与训练原始q差3.58e-7rad；标准化后5.09e-5超过预登记2e-5。最终计划差≤2.87e-6rad，不能据此解释任务失败。|
|新可选精度修复后端到端一致性|PASS|只从已有当前target减tracking-error恢复编码器q；不读取物体/触觉/未来。两条完整专家轨迹、两最终EMA，阈值未放宽。|
|图像、DINO、深度、相机时钟|PASS（所测两条完整轨迹）|原始RGB逐字节可逆；全部28个anchor的DINO、depth、timestamp差0；不是只抽首尾帧。|
|GRU与条件/输出|PASS（修复后）|每tick实际Controller和完整训练encode对照；同一顺序DDIM随机流。不是用teacher forcing声明闭环成功。|
|边界跳变是唯一/最早根因|FAIL（证据不支持）|24/24在专家持续接触前已偏离；22/24在第一次重规划tick12之前已偏离。|

## 动作链：四种量不能混用

|量|含义与本轮检查|
|---|---|
|原生RL指令|29D、无量纲；经过原delay queue。Arm是增量，Hand映射到关节范围，随后原EMA与限位。|
|专家29D target|`target_after_rad[t]`，canonical顺序、rad；已包含delay/EMA/限位，是当前模型的监督标签。`target_before[t]`才是发动作前能观察到的当前目标。|
|实际actuator target|DP预测或专家target回放经同一绝对位置接口、真实限位、canonical→Lab后，由原控制器在两个120Hz物理子步施加；没有第二次delay/EMA。24条旧模型记录和两条专家位置回放的提交/执行器最大误差均0。|
|真实关节位置q|动力学与接触作用后的测量，不能当作actuator target。成功专家的target/q RMS约0.0854/0.1063rad，绝对接口回放逐值复现。|

[位置动作合同与既有回放证据](POSITION_ACTION_CONTRACT.md#实际位置回放验收)。本轮H24/执行12tick不改变原60Hz控制、120Hz物理；旧合同中的H32是早期模型配置，不是本轮最终checkpoint配置。

## 先偏离，再接触失败

使用既有阈值：link7末端5mm、arm0.01rad、hand0.05rad、object5mm/5°，连续3ticks；同控制tick公共前缀，不做时间伸缩。
State末端首次偏离ticks：`[9, 3, 10, 6, 3, 6, 7, 4, 4, 4, 2, 5]`；Image：`[6, 13, 6, 4, 3, 5, 8, 7, 6, 5, 12, 6]`。
全部在专家持续接触前；无法把这些初始偏差归因于接触后滑移或第一次重规划。
对照专家均完整成功，但原生lift标志与subgoal reset相关，不是强制单调阶段漏斗。

## 手指与专家同口径对比

以下是12条各自公共前缀指标的中位数，避免模型601tick超时尾段和较短成功专家混算。单位rad。

|模型|模型边界跳变|专家同tick边界|模型片段内跳变|专家片段内|模型跟踪RMS|专家跟踪RMS|
|---|---:|---:|---:|---:|---:|---:|
|StateDP|0.06515|0.03573|0.02924|0.03711|0.05546|0.10246|
|ImageDP|0.05790|0.03573|0.03091|0.03711|0.05868|0.10246|

边界跳变确实偏大，但成功专家的目标/实际手指差反而更大：它可能体现接触承载，不能等同于控制器故障。
模型raw target经真实joint limits裁剪，最大手指裁剪State 0.2341rad、Image 0.3666rad。
专家比较对象是已经经过native限位的target，其额外position-interface裁剪为0；这不代表专家上游native映射从未限位。
每次持续接触丢失前后±3tick的跳变、clipping、tracking及是否跨边界，和专家对应统计保存在analysis.json；同步出现仍不是因果证据。

![train配对](evidence/grasp_diagnosis_20261104_01/paired_00.png)

![validation配对](evidence/grasp_diagnosis_20261104_01/paired_04.png)

## 有界、单因素实测

先只修复proprio浮点重构并复评旧checkpoint，再在同一修复版本上只加K4拼接。固定train ordinal0和validation ordinal4、两个模型；旧K0对照直接复用已完成结果。
K4只把旧计划尚未执行的[12:16]和新计划[0:4]按1/4、1/2、3/4、1融合；没有新延迟/限速、参数更新、expert takeover或控制器增益修改。
修订计划9条新episode、最多10次启动/2次软件恢复；两次连续同阶段故障停止；沿用本轮最初4小时截止，不重置。
原先4条K4计划在发现严格一致性FAIL后、任何新Kit启动之前扩展为4条精度修复对照+4条K4；两次变化分开比较。

|版本|模型|成功/有效|计划|接触|lift|首子目标|软件缺失|
|---|---|---:|---:|---:|---:|---:|---:|
|precision_only|StateDP|0/2|2|2|1|0|0|
|precision_only|ImageDP|0/2|2|1|0|0|0|
|precision_K4|StateDP|0/1|1|1|1|0|0|
|precision_K4|ImageDP|0/0|0|0|0|0|0|
|deterministic_precision_only|StateDP|0/1|1|1|1|0|0|
|deterministic_precision_only|ImageDP|0/1|1|1|0|0|0|
|deterministic_precision_K4|StateDP|0/1|1|1|1|0|0|
|deterministic_precision_K4|ImageDP|0/1|1|1|0|0|0|

K4必须与精度修复对照的前12tick command/target/joint/object/actor/contact逐值相同才接受单因素比较；无结果项不是任务失败。
## 归因可信程度与是否重训

- 高：当前绝对位置接口本身能复现完整专家任务；逐tick实际执行器target没有额外delay/EMA或单位/顺序错配。
- 高：早期目标/轨迹误差先于稳定接触和首次重规划；整体离线RMSE低不保证初始接近轨迹、抓取构型与动作幅值足够准确。
- 中：自主观测/过去动作历史偏离会继续累积误差；离线使用专家历史，与自主闭环分布不同。存在支持证据，尚未通过干预单独建立因果。
- 中：跨规划不连续与限位裁剪会改变手指请求，但不能从相关性断言它们主导失败；以K4实测判断收益。
- 低：proprio舍入是完整失败主因；量级微小，修复后的闭环是针对这一假设的检验。
当前不修改网络或损失，不启动重训。下一轮优先考察前12tick的动作增量/方向与专家差异，以及接近阶段小偏差后的恢复能力；先做有限诊断，再决定是否需要roll-in数据或改变监督。
不以这两个诊断初态挑选最终模型，不访问test40，也不宣称少量消融证明统计优势。

## 产物与完整性

[完整数值和每次故障](evidence/grasp_diagnosis_20261104_01/summary.json) · [24条逐例分析](evidence/grasp_diagnosis_20261104_01/analysis.json) · [原严格数值FAIL](evidence/grasp_diagnosis_20261104_01/parity.json) · [修复后数值](evidence/grasp_diagnosis_20261104_01/parity_precision.json)。
原始CSV、逐tick模型对照、视频和日志：`outputs/visuotactile/native_dp_v2/grasp_diagnosis_20261104_01`。原24条与所有旧checkpoint不改动；大文件不推送。
新数值审计早期两次脚本错误（旧source_commit标签、只读lease缺显式deadline）已定位并保留日志，均发生在仿真前；不计机器人任务失败。
新增5项定向CPU测试通过：单因素拼接/随机流、K0精确旁路、公共前缀/事件窗口、当前编码器字段、配对失败仍持久保存有效原生结果；未重复旧batch或Epsilon测试。

## 完成后的实测对照

本表边界RMS为各自完整episode，长度不同；严格单因素跳变比较另用下方共同235tick前缀。

|模型/初态|版本|结果|原生lift|首次末端偏离tick|全episode手指边界RMS rad|专家接触前实际FK RMS mm|
|---|---|---|---:|---:|---:|---:|
|StateDP/0|original|fall|1|9|0.08142|10.47|
|StateDP/0|precision_only|fall|1|9|0.09100|10.47|
|StateDP/0|precision_K4|fall|1|9|0.03609|10.88|
|StateDP/0|deterministic_precision_only|fall|1|9|0.07588|10.47|
|StateDP/0|deterministic_precision_K4|timeout|1|9|0.03318|10.88|
|StateDP/4|original|timeout|0|3|0.08171|29.35|
|StateDP/4|precision_only|timeout|0|3|0.08283|29.35|
|ImageDP/0|original|timeout|0|6|0.07761|59.95|
|ImageDP/0|precision_only|timeout|0|6|0.07372|59.67|
|ImageDP/0|deterministic_precision_only|timeout|0|6|0.07085|59.71|
|ImageDP/0|deterministic_precision_K4|timeout|0|6|0.03777|59.36|
|ImageDP/4|original|timeout|0|3|0.06878|26.01|
|ImageDP/4|precision_only|timeout|0|3|0.07002|26.28|

确定性K4严格匹配1/2对，其中1对降低共同前缀手指边界RMS；9条新诊断完整成功共0次。Image配对失败，不能作因果差值。
单因素差值和前12tick逐值验证已保存；小样本只用于机制诊断，不代表总体成功率提升。

## 为什么绝对位置离线误差不能代表抓取能力

下表使用相同原始示范、相同顺序DDIM seed以及共同tick范围；离线列读取专家历史，闭环列读取自己的历史。单位rad。
|模型/初态|首步Arm target误差|专家首步Arm变化|离线Arm公共前缀RMSE|自主Arm公共前缀RMSE|离线Hand RMSE|自主Hand RMSE|
|---|---:|---:|---:|---:|---:|---:|
|StateDP/0|0.01114|0.00247|0.01215|0.03403|0.06067|0.24935|
|ImageDP/0|0.01575|0.00247|0.01182|0.20483|0.08698|0.46971|
|StateDP/4|0.01833|0.00239|0.01651|0.08765|0.08593|0.39893|
|ImageDP/4|0.01303|0.00239|0.01561|0.15429|0.09165|0.30367|

首步Arm误差约为专家本步动作变化的4.5–7.7倍；绝对位置RMSE看起来小，仍可相对精细接近运动很大。
专家条件下移动anchor的12tick Arm方向cosine：State train/validation约0.876/0.746，Image约0.830/0.638；幅值比约0.93–1.04。方向和时间结构仍有误差，不是单纯把手指幅度整体缩小就能证明解决。
初始q、proprio、目标和物体尺度一致，但原4lane示范与新独立仿真Actor噪声不同：qd最大差0.269/0.482rad/s，noisy keypoint约0.02m。干净palm/指尖仅浮点差。
这解释了为什么“同物理初态”不等于“与原始示范同网络输入”。State使用这条原生带噪Actor分支，Image不使用Actor；不能把Image失败归因于Actor差异。在线专家与模型的配对初态/观测噪声流仍完全相同。
不从聚合RMSE相减声称误差可因果分解；来源四lane隐藏状态限制仍保留。
证据：[离线/闭环同口径数据](evidence/grasp_diagnosis_20261104_01/offline_closed_gap.json)、[完整性](evidence/grasp_diagnosis_20261104_01/final_integrity.json)、[单因素结果](evidence/grasp_diagnosis_20261104_01/ablation_comparisons.json)。

下一轮最有价值的是固定前12tick、接触前的小偏差恢复诊断：比较新鲜成功专家观测下的学生动作与自主偏离后同一专家的shadow纠正方向。若确认只在离轨状态无恢复能力，再提出有界roll-in/纠正数据训练；若专家输入下首段已明显错，先定位移动阶段监督与静止/位置偏置。
**本轮没有证据要求立即更换网络或再次从头训练两组10000；也没有证明新增某项loss即可解决。**
所有视频索引：`outputs/visuotactile/native_dp_v2/grasp_diagnosis_20261104_01/VIDEOS.md`。修订计划9条，实际有效9条，启动9次，软件缺失0次。
SDK fast exit0与正常SDK完整清理仍明确区分；无共享SDK改动。各worker资源回收和旧权重/源码/原始结果SHA复核已完成。

## 版本、结束状态与未解决项

[执行清单：源码与checkpoint SHA、全部计数、服务结束状态](evidence/grasp_diagnosis_20261104_01/execution_manifest.json)。前批运行源码`72c060d`，确定性修订`5e73c8a`；203项运行源码按冻结清单校验。三个脚本仅在全部worker退出后修正结果保存/后处理，运行版Git内容与冻结SHA一致，审阅版SHA单列。60个原结果文件及两个10000-update权重均未改变。

9次启动获得9个有效原生结果，0个完整成功、0个启动/运行/关闭软件缺失；其中2个K4结果不具备严格配对资格。旧计划3条未启动项已保留取消记录。最终状态为`completed_with_unmatched_pairs`。原owner因Image严格配对失败而结束，systemd仍保留failed状态和错误，未冒充门控通过；所有服务MainPID均0，worker资源回收通过。

监测拼接记录有975.17秒缺口，最大间隔446.92秒；已记录完整60秒窗口GPU最高41.20%，但**不能据此声称全会话连续监测或资源完全合规**。所有阶段退出时自有进程/GPU释放证据保留。SDK fast shutdown exit0与完整正常SDK析构有别；本轮未修改共享SDK。

未解决：Image真实Kit首计划不一致，缺少对应原始RGB-D、在线DINO与condition，故无法定位唯一原因；下一轮应先补齐这些记录，再做Image单因素因果比较。当前证据足以反对盲目重训，尚不足以唯一决定网络或损失改法。所有原始失败、两条未匹配结果及九条已解码视频均保留。
