# 冻结专家与窄分布StateDP数据

## 2026-11-06 09:29 UTC+8 实测：后续已停止

有效新数据仍为200条/201尝试。后续于02:13在GPU历史编码一致性审核失败后停止，新增采集0、追加更新0；没有进入扩采或新模型对照。原400及200条全部保留。完整原闭环、CPU复核和原因边界见[统一State结果](STATE_DP_RESULTS.md)。下方为历史实施/启动记录。


## 2026-11-06 更新

原直接窄域v2已达到200条完整成功（201次尝试，1次真实任务失败）、60家族44/8/8、43,621个控制步、17,972,841,587bytes，DATA_READY通过并已进入正式StateDP训练。下文7/7、DATA_READY=false是历史启动证据。

用户新授权一次条件化扩充，须等原闭环完成、训练曲线支持平台/泛化差距且数据/执行审核通过，才由独立root执行。新32家族目标、24/4/4预分割、最多96次真实速度轨迹，总量260–296；不改变窄域范围、不插值旧传感器、不将旧test家族加入train。原200和原400均保留。是否实际启动、增加数量和效果以新root机器状态和[State结果](STATE_DP_RESULTS.md)为准。

## 当前运行：直接窄域 v2（短监督通过，后台生产运行中）

最新用户授权已扩展到新数据验收后的自动随机StateDP训练和有界闭环。新root为`outputs/visuotactile/run/narrow_state_20261105_02`，当前生产服务`narrow-state-data-20261105-02-production.service`（第一阶段服务已在完整边界退出）；准备commit与实际运行commit分别记录在session.json及launch_revision.json，所有运行Python源码另存完整SHA清单。**短监督已通过；最新启动核验7/7条完整成功，DATA_READY=false，正式训练0更新。后台继续运行，数量以status.json为准。**

实际60/68候选准入（88.24%，8个保守机器人网格包围盒拒绝，均非任务失败），split44/8/8；4条pilot共1075控制步、436,376,287bytes，199/157原速与455/264慢速全部成功，154字段hash验收、两物理子步target误差0、同家族初态8组误差0。一个慢速初态旧query仍错误命中1项，新几何准入一致通过。实测时长比2.2864/1.6815，中位1.9840。

GPU一次性随机模型前后向/AdamW/EMA/保存恢复通过，正式0更新；在线/完整前缀条件误差9.53674e-7、尾部7个有效target、未来干预不影响过去。首次GPU smoke在优化前因deadline未显式传入的TypeError退出，证据保留；修复后唯一补充检查成功。46项CPU回归通过。

首worker实测含启动/家族门控29.74条/小时，4episode墙钟分别39.48/134.08/86.98/186.40s，受共享负载影响，不能保证200条。生产前在完整边界暂停并将原16h内的阶段额度改为采集6h/训练最多9h/末1h评估；首次开始21:57:39与绝对截止2026-11-06 13:57:39 UTC+8均不变。最低150及覆盖门槛不降低，临近采集截止预留20min完成当前episode和数据审核。

首worker整卡均值43.69%、最大完整60s52.64%、缺测0、最低可用RAM25.66GiB；执行的是shared_recent_v1而非旧50%硬门限，不称整卡始终≤50%。暂停修订和独立smoke另记监测窗口。[轻量机器证据](evidence/narrow_v2_startup.json)。

### 初始化矛盾与修复范围

本机Isaac Lab2.3.2、Omni PhysX107.3.26。冷启动重复reset一致；完成真实rollout后恢复e01_p000144，q/qd、物体/桌/孔及全部机器人link位姿一致，overlap从0变为11个手部命中。flush_changes、update_articulations_kinematic、USD输出和同姿态root重写都未消除误报，不能把它们写成有效修复。

检查原始CollisionAPI对应的34个碰撞网格（包括Xform下的实际Mesh、fixed-joint合并及局部变换），用实时link pose变换其保守包围盒，额外膨胀1mm后仍至少相隔108.62mm。故该case的overlap命中与实际几何不符。新增`live_collision_mesh_bounds_v1`以这些包围盒和原object/table/hole box SAT作保守准入；有相交可能即拒绝。旧query继续保存为诊断，不再作为v2准入。没有改SDK、模型、实际碰撞几何或物理积分；这不宣称恢复旧四lane隐藏状态。

依据为本机安装源码及对应版本[PhysX Python API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html)。官方flush说明用于处理缓冲变更，并不承诺修复本例；本机forward实际更新kinematics/fabric。SDK内部具体pruner缺陷未作二进制定位，不把缓存实现假设冒称已修复SDK。真正修复的是研究准入对不可靠查询的依赖。

有界诊断保留3次前动作软件错误（布尔差值、Path序列化、Xform碰撞父节点未处理）及4条自然成功native轨迹；冷/重复reset与全部字段原始证据在narrow_reset_query_20261105_02–07，均已退出，不并入新数据或学生成功分母。新gate的live采集验收以下一段机器状态为准。

### 新分布、数量和阶段

直接采样全新60家族目标，最多80候选、至少50准入；不复用旧恢复初态，也不宽分布反复拒绝。沿既存密集中心：object XY±25mm，hole−object XY±35mm（与原孔范围直接求交），旋转球20°，table z[.376,.384]m，物体高于table原点[.244,.256]m；native arm/hand reset lerp .03、初速±.15rad/s。所有随机类别保留变化；摩擦原为固定，本轮不加随机化。实际范围与拒绝率由initial_coverage.json给出。

先冻结初态与split（预计44/8/8），同家族所有速度同split；两train原速/.58真实校准，唯一允许将慢端上调至.65。生产alpha每episode稳定，原速加连续中/慢区间，约220个预注册尝试，目标200/最低150条成功且覆盖门槛通过。alpha不是时间倒数，实际时长比另报。保留完整三路RGB-D、触觉、控制目标和真实运动；成功校准数据按同一合同计入，失败和部分文件单列。旧隔离3条始终不进入新manifest。

普通owner自动转移：family→速度采集→数据/hash/时序/历史/split/train统计验收→一次性GPU前后向/恢复→新随机StateDP→train4及预注册unseen12闭环。H24/commit12及网络/loss不变。新固定16h含等待，采集6h，训练最多9h、保留最后1h闭环；75GiB产物上限、60GiB磁盘余量、6GiB VRAM/12GiB RAM准入、shared_recent_v1目标40%。每阶段最多2次审查后启动，无自动retry。没有重置旧4h预算。

```bash
R=outputs/visuotactile/run/narrow_state_20261105_02
bash visuotactile/run_cpu.sh -m tools.narrow_collection status --root "$R"
bash visuotactile/run_cpu.sh -m tools.narrow_collection stop --root "$R"
# 仅在审查阻断原因后使用；不清除预算或已有attempt
bash visuotactile/run_cpu.sh -m tools.narrow_collection resume --root "$R"
```

日志在R/workers/<stage>/attempt_*/worker.log；status/heartbeat/manifest记录真实进度；DATA_READY.json仅在完整验收后生成。训练R/training/{progress.json,training.jsonl,validation.jsonl,curves.png,last.pt,best.pt}，最终R/evaluation/result.json。短监督通过后由普通程序继续，不靠LLM长轮询。

### 旧 v1 阻断事实（永久保留）

narrow_state_20261105_01已退出、quarantined=true，原4h截止22:11:24及两次启动均不延长。15旧+9补充家族，360宽分布reset仅9通过；6次尝试中3次pre-action query拒绝（不能称真实穿透或任务失败）、3次完整成功（171/210原速、366ticks alpha.58，时长比1.743）。154字段hash、目标/两子步一致，仍因当时准入不可靠而隔离；无生产放行、无DATA_READY、正式0更新。一次性CPU smoke通过，不计正式训练。旧监测均值14.18%、最大完整60s19.56%，202.11s停机缺测保留。

本轮已核对原删除ledger的文件身份、SHA、同UID打开句柄与已分类session daemon，删除完成的一次性CPU/GPU checkpoint共2份，实际释放2306101248bytes。原400及metadata、冻结专家、依赖缓存、未知引用和所有物理失败证据保留；未重复计入旧126.673GiB清理。

## 历史冻结专家实验（以下方案已结束，不是当前启动指令）


本轮是独立的专家执行研究。原State150随机评估已按用户新指令在完整rollout边界提前暂停：143有效/原计划200，历史2/100、新初态0/43，32动作前碰撞拒绝；源码、checkpoint、随机清单、全部结果和恢复位置保留；[原评估报告](https://github.com/mao455/bcrl_visualTactile/blob/feature/native-dp-training-v2/research_docs/STATE_DP_RESULTS.md)仍在原feature分支逐次更新。新工作在唯一独立工作树`bc+rl_RSS_expert_speed`和分支`feature/expert-slow-execution`，复用现有环境/数据，不按尝试复制源码。

## 配置审核

|核查|实证|结论|
|---|---|---|
|任务与机器人|官方`ff2fc62`的tight_insertion→Lpeg_matchedmass.tol0p5mm；KUKA iiwa14+Sharpa，7+22关节；原生任务/机器人URDF字节一致|PASS，同发布任务|
|RL checkpoint|217700467 bytes，SHA256 `7e61015dad660430ab07bab4f715c90c6f76b84cc358da6be81319a6aee09990`，与官方Hugging Face assembly文件LFS SHA相同|PASS，非Play-only|
|推理|实测actor配置LSTM1024+MLP1024/1024/512/512，checkpoint RMS，140观测+组标识50；确定性输出截断[-1,1]|PASS；专家是LSTM，StateDP才是GRU|
|动作时序|物理120Hz、推理60Hz；原生0–3tick delay一次；Arm增量与Hand绝对映射后各EMA0.1及限位|PASS，原生目标Arm上限1.5×0.1=0.15rad/s；不是实际速度硬上限|
|几何与间隙|主杆250×30×20mm、侧杆20×80×20mm，质量57.8g；孔21×31mm，对应插入截面20×30mm|PASS，与官方资产相同；**单边0.5mm、总差1mm**|
|SDF/尺度|此任务物体2个box、孔5个box，无SDF标记；6.25/0.75/0.5是尺寸观测编码，不是放大25倍的碰撞几何|PASS；不能套用其他任务的SDF说法|
|执行器与碰撞|原生PD/限位未改；配置及smoke运行期实际值存机器审核。指尖摩擦1.5、其余主要材质0.5，保留原生PhysX参数|PASS，本轮不调增益/摩擦/质量|
|fork差异|相对官方基点原生文件仅action_utils增加只读trace和显式可选delay重放；普通路径100tick目标、队列和RNG逐值一致|PASS，普通控制数学一致|
|额外研究层|CPU物理、摄像头/接触记录、明确world_hold外力帧封装、采集四lane和本轮单lane不同|已披露，不声称与裸官方SDK逐位等价|
|成功与计时|两子目标顺序累计各10合格tick、最终7.5mm与指尖平均距离>0.1m；首次动作前计时，成功tick立即停止|保持发布原生定义；物理时间不包含Kit启动/墙钟等待|

官方基点`ff2fc62873f6c6cd93164123c8358e9ce2d65c9b`；审核最新upstream `621459224dd3618456de273871e507e380d4dedb`仅作对照，**没有升级到新upstream**。最新版增加其他机器人/任务和配置项，不能把“官方当前main”当作历史实验源码。原生数学对照与资产SHA见[机器审核](evidence/slow_expert/configuration_audit.json)。

来源：[官方任务与权重入口](https://github.com/kushal2000/play2perfect/tree/ff2fc62873f6c6cd93164123c8358e9ce2d65c9b)、[官方权重](https://huggingface.co/Kushal20/play2perfect-checkpoints/tree/main/tight_insertion)、[L-peg资产](https://github.com/kushal2000/play2perfect/blob/ff2fc62873f6c6cd93164123c8358e9ce2d65c9b/assets/urdf/peg_in_hole/lpeg_matchedmass/lpeg_matchedmass.urdf)。

## 3.4秒与论文真机6–11秒

原400条成功的3.412s均值/3.267s中位是仿真物理时间，并且**以成功为条件**。复用已完成的400条逐tick审核，不重新跑400专家。它不是整个初态分布的无条件完成时间，也不能证明官方通常应当3–4秒完成。

论文Table1是成功真机试验的全任务时间，涵盖接近、抓取、搬运、装配；10/2/0.5mm分别6.8/9.4/11.1s。论文按孔与插入件截面**尺寸之差**定义间隙，因此当前资产的1mm总差不能冒充论文0.5mm条件。论文120/60Hz与本地一致，真机物体跟踪30Hz；论文文字判据与发布状态机细节也有差异。本轮始终按发布原生成功计数。[论文及附录](https://arxiv.org/html/2606.26428v1)

另外，官方交互/离线eval默认关闭动作/观测延迟、外力等DR，本地原采集配置保留。正式A/B都保持本地采集配置，避免一边减速一边偷偷关闭DR。[官方交互入口](https://github.com/kushal2000/play2perfect/blob/ff2fc62873f6c6cd93164123c8358e9ce2d65c9b/evaluation/eval_isaacsim.py)、[离线入口](https://github.com/kushal2000/play2perfect/blob/ff2fc62873f6c6cd93164123c8358e9ce2d65c9b/evaluation/eval_offline.py)

真机完整PD、安全限速、设备等待和计时排除项未公开，不能断言时间差由某一种滤波或初始化耗时造成。官方部署只提供策略推理核心，不含完整硬件驱动。原URDF注释还提到质量调整的历史，不能从当前资产相同推断checkpoint训练时所有硬件参数已确认。[官方部署说明](https://github.com/kushal2000/play2perfect/blob/621459224dd3618456de273871e507e380d4dedb/docs/deployment.md)

## 已实现的有限实验

保持120/60Hz、原生delay/EMA/限位、奖励/终止与RL权重。Arm/Hand参数独立，第一轮共同取1.0/0.8/0.67/0.5；没有新EMA状态。实验A在一次原生动作处理后，按`已执行目标 + 系数×(本tick原生候选目标−已执行目标)`推进，写回真正将执行的目标，专家下一tick观察真实反馈。Arm有效目标上限降为0.15×系数rad/s；Hand是目标推进比例，**不承诺实际关节速度按比例下降**。系数1直接旁路所有算术，必须完整物理对照通过。

实验B使用原400中真实post-delay/EMA的`target_after_rad`，以初始`target_before_rad`作为q(0)，新第k步请求q(k/s)，s=1/1.25/1.5/2。通过既有29D位置接口执行，不再添加delay/EMA；保留原接口的随机流占位抽样。序列用尽后保持最后目标，直到原生成功/失败/超时，单列保持tick；不把序列用尽当成功，不插值状态/视频。新随机扩展时先生成同初态原专家成功参考，再物理回放；没有成功参考的样本单列缺失，不能让分母偷偷变小。

2026-11-05用户调整优先级：原State随机评估改为提前停止的探索性结果，不再作为200条必须完成的依赖。新研究首轮只用已冻结10初态中低/中/高时长代表（原索引0、5、9），先原专家3条和系数1严格等价1条；至少2/3原生成功才继续。候选A080/A067/A050各最多3条，B100未拉伸回放至少2/3成功才检查B125/B150/B200；A都缺乏成功时才检查原生首子目标后的C067。每候选连续2次失败且无首子目标即停，保存真实分母，不把小样本筛选当可靠性证明。

扩展候选须3条有效、至少2成功、成功中位4–8s、双方成功配对时长比至少1.2。先优先成功时长中位落在5–7s的合格候选，再按成功数及接近6s选最多1组；只给有希望者补齐10个原生基线/候选配对。正式可靠性门槛仍为原生基线≥8/10、慢速≥8/10、成功中位5–7s、配对时长比1.4–2.2；累计3次失败导致8/10不可达则停止。选一组到20个新随机初态，至少17/20且原生≥16/20才进入6条真实采集；出现4次失败立即停止无望扩展。这是自适应探索，不作为无偏固定样本成功率估计；所有实际尝试和软件/非法初态独立列出。

当前已完成的3初态配对：native3/3（2.317/3.067/3.767s）、A0803/3（3.133/3.567/4.367s）、A0673/3（3.383/4.250/5.567s）、A0503/3（9.000/11.633/9.400s）。系数1额外验收成功，18个控制/物理字段逐值相同。旧400目标的B100仅1/3成功：一条首子目标后掉落、一条601tick超时；因此不继续拉伸B125/B150/B200，不把未执行组合叫失败。历史PhysX隐藏状态缺失仍是开环基线的限制，不将该结果归因为时间插值数学错误。

A067和A050都3/3成功，但中位分别低于5s、高于7s，故基于已有数据新增唯一A060（Arm/Hand均0.60）3例区间筛选；这是明确的数据驱动候选，不冒称预注册确认性检验。A060实测2/3成功（成功6.400、6.083s，失败为首子目标后掉落），符合目标时间带，但可靠性尚未确认。仅优先扩展成功时间中位已落在5–7s的一个合格候选；A0673/3更快的对照保留，不以3比2的小样本差异宣布成功率显著优劣。正式10例仍须≥8次成功，20例仍须≥17次成功，未放宽。版本在当前B100 rollout完整提交后通过空目录边界暂停切换，所有原始结果/异常/预算保留，恢复不重复任何已提交样本。

[配对运动统计](evidence/slow_expert/paired_motion_screen.json)：A067相对native的每条轨迹p95再做配对比值，中位Arm实际速度0.658、Hand速度0.705、Hand加速度0.656、Hand跟踪误差0.840，接触力p95比值0.989。速度确实下降，但接触峰值没有同步明显下降；这是完整轨迹汇总，受停留时长影响，不能据此宣布接触稳定性的因果根因。

全新扩展仍以20条合法初态为目标，预先固定最多40个seed，仅动作前非法初态可用后续seed补足；合法原生任务失败必须保留并继续同初态候选对照，不按任务结果换seed。40名单仍不足时如实报告缺额。此处利用旧32/75非法初态的实证避免再次把非法初始化耗尽当作慢速任务失败；此前尚未执行任何扩展样本。

原生600tick是**每个子目标阶段**的计数上限，完成子目标后计数重置；故A050的11.633s完整成功没有扩展原生预算。首次完整成功仍立即停止。

原生阶段600tick预算保留（实际边界为601）；没有启用延时预算诊断。沿用155次硬启动上限，但不要求耗尽组合。**24小时执行预算从依赖释放后的首次执行阶段开始**，此前等待约3.5小时不扣执行预算；execution_budget.json首次写入后恢复不重置。用户此次授权替代旧“等待也计入24h”的约束。旧预算、冻结源码和切换证据保留在maintenance/priority_change_20261105。软件故障仍保留，有限处理，不掩盖为任务失败。

旧owner没有边界暂停钩子，本次仅预留下一次attempt的空目录，利用既有排他mkdir在启动下一个worker之前停止调度；当前worker自然完成、结果验收提交、监控正常退出。原始FileExistsError控制边界记录归档，pause_receipt.json明确是用户要求的paused而非软件实验失败。完整校验checkpoint、已提交结果/种子SHA、进程/显存/锁和打开的写文件；原源码不变。恢复须显式要求，移除记录的空屏障后从random_fresh_075继续，不重复已提交样本。

耗时审核（179次已结束启动含smoke）：墙钟平均102.28s（其中143条正式有效rollout平均117.91s、32条非法初始化平均33.90s）；App启动含准入/节流13.85s，场景与专家准备含等待8.53s；DP推理146条均值7.75s；退出回收1.60s。资源日志176条覆盖的准入均值0.59s、节流均值29.76s，最长节流374.63s；组件存在重叠，不能直接相加。旧日志无法拆分物理/渲染、RL shadow和视频编码，新worker用只读主机计时分别记录这些项，不新增物理步、同步或随机抽样。原整卡均值28.62%，最大完整60s均值62.89%，遥测缺失0；不是全程≤50%，且包含其他用户任务。本次不通过降低动力学精度或变更控制频率提速，暂不采用未经验证的Kit复用。

21项队列回归通过，覆盖显式暂停后接续、原证据漂移拒绝、预算不重复开始、两次无进展失败早停，以及成功却未实际减速不扩大。慢速物理结果仍以下方实时区为准。

恢复审查发现并修复了初筛计数范围错误：累计7条native导致固定3条门槛误判，新增物理试验为0；并非真实基线失败。原始错误gate/退出保留在maintenance/resume_screen_scope。恢复只用初筛identity计算筛选门槛，正式可靠性仍使用全部10/20条；已提交23条全部复用。

## 数据与训练门槛

只有随机20通过才录制6条真正重新执行的RGB-D、本体、模型均值（player裁剪前）、原始RL指令、未减速候选目标、实际执行目标、物体/目标、接触、时间和delay历史；使用原CompactEpisodeStream无损存储并逐字段SHA重建验收。成功与失败文件均保留证据，只有原生成功进入独立`slow_expert_pilot_v1`清单。监督仍为实际issued `target_after_rad`，绝不替换成到达关节位置。

这6条是数据合同验收样本，**不会自动被称为完整训练集**。其后的State离线/live输入配对、独立train/validation覆盖、原数据混合微调需依据真实采集结果继续审核；现有训练器与150checkpoint采样状态绑定，不能将新数据集静默塞进原续训。当前无慢速数据、无微调更新、无Image启动。只有慢速物理可靠性与新数据合同实测通过才进入后续扩展/小规模微调，不提前声明改善。

## 当前结论与后续决定

### 视觉—触觉修正的最小后续方案（条件性建议，未启动）

不再以完成旧State200为前置条件。慢速可靠性通过后，6条真实传感器小样本先审核连续时序、执行目标/真实状态、delay和past-only历史；除无损源字段SHA外，仍须逐tick核对训练/在线特征，特别是Image三视角RGB量化、depth validity、DINO固定revision/批规则与历史。此前strict live Image配对问题不能跳过。

合同通过后，建议先建立16训练+4验证的独立初态小集，混入原train320，State从150权重开始至多200更新的微调探针，保持H24/12/29D。新数据使用显式的新优化器/采样版本，不悄悄替换旧续训状态；此预算只验收可学性。Image先合同再小预算拟合，视觉nominal不读取触觉或特权几何。

随后优先做冻结视觉nominal、已训练触觉修正、同结构屏蔽触觉修正的三组对照。残差标签由因果nominal预测与真实专家执行目标产生，专家历史明确标记teacher forcing，部署只读过去真实提交/执行反馈；未训练修正不得控制机器人。第一轮6个固定合法新初态、三组共18条诊断，分别看完整成功、抓持/子目标和连续性；不启动几百次旧baseline评估。只有触觉与屏蔽对照出现可复核差异才扩大。本段是待数据实测支持的最小建议，不是已完成训练或自动启动授权记录。

任务/权重与官方发布基点一致；当前3.4秒与真机时长不是同一条件。哪种慢速方法可靠、手指/末端/物体实际速度如何变化，均待配对物理结果。软件已记录各tick原候选/执行/到达状态，原生事件、接触和运动统计，成功率分母包含全部有效失败；软件故障独立记录。若没有5–7秒且可靠成功的配置，将停在诊断，不强行生成数据或重训。

33项相关CPU回归通过：原生动作链100tick与官方基点逐值/RNG一致、1.0精确旁路、Arm/Hand独立缩放、一次调用和真实target反馈、时间插值起点/终点、未改位置接口/成功审核/现有抽样入口。CPU通过不等于真实物理成功；本轮未取得Kit结果前不作成功声明。

新增验收器也已只读复核一条保留的真实专家记录：成功/终止145tick一致，新增物体/末端运动指标可正确提取。这是新统计代码的兼容性验证，未重复启动Kit，也不算本轮新baseline样本。

每个新专家worker在推理前后校验模型全部参数/缓冲指纹，确认没有优化更新。`run/resume`复用已提交stage；`release`对本研究拒绝，不能误放行旧训练。

最新受监督现场快照（2026-11-05T16:12:40.411388+08:00）：服务已实际运行到`pilot_native_09`，已启动26次、已验收25条；正式扩展选择`A060`。旧State保持paused。[机器快照](evidence/slow_expert/live_status.json)与[各执行版本范围](evidence/slow_expert/runtime_revisions.json)保留真实PID、预算、资源与source SHA。下方生成区是最近一次完整rollout落盘时的统计，不以发出启动命令代替运行验收。

<!-- SLOW_EXPERT_RESULTS_START -->
## 实测进度

状态：`blocked`，阶段`capture_A060_00`；已尝试88/155。软件缺失1，验收失败0，非法初态14。

|阶段|配置|原生成功/有效|95% Wilson区间|成功均值/中位(s)|fall/timeout|
|---|---|---:|---|---|---|
|pilot|native|10/10|72.2%–100.0%|3.308/3.233|0/0|
|pilot|identity|1/1|20.7%–100.0%|2.317/2.317|0/0|
|pilot|A080|3/3|43.9%–100.0%|3.689/3.567|0/0|
|pilot|A067|3/3|43.9%–100.0%|4.400/4.250|0/0|
|pilot|A050|3/3|43.9%–100.0%|10.011/9.400|0/0|
|pilot|B100|1/3|6.1%–79.2%|3.350/3.350|1/1|
|pilot|A060|9/10|59.6%–98.2%|5.567/5.717|1/0|
|expansion|native|20/20|83.9%–100.0%|3.487/3.367|0/0|
|expansion|A060|18/20|69.9%–97.2%|5.053/5.067|1/1|

![原生成功与真实完成时间](evidence/slow_expert/speed_comparison.png)
- [pilot_native_00：成功，139ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_native_00/attempt_1/output/r931005_1793209376_e02_p000132/rollout.mp4)
- [pilot_identity_00：成功，139ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_identity_00/attempt_1/output/r931005_1793209376_e02_p000132/rollout.mp4)
- [pilot_native_01：成功，184ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_native_01/attempt_1/output/r931005_1793209376_e00_p000063/rollout.mp4)
- [pilot_A080_00：成功，188ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_A080_00/attempt_1/output/r931005_1793209376_e02_p000132/rollout.mp4)
- [pilot_A080_01：成功，214ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_A080_01/attempt_1/output/r931005_1793209376_e00_p000063/rollout.mp4)
- [pilot_A067_00：成功，203ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_A067_00/attempt_1/output/r931005_1793209376_e02_p000132/rollout.mp4)
- [pilot_A067_01：成功，255ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_A067_01/attempt_1/output/r931005_1793209376_e00_p000063/rollout.mp4)
- [pilot_A050_00：成功，540ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_A050_00/attempt_1/output/r931005_1793209376_e02_p000132/rollout.mp4)
- [pilot_A050_01：成功，698ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_A050_01/attempt_1/output/r931005_1793209376_e00_p000063/rollout.mp4)
- [pilot_B100_00：失败，243ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_B100_00/attempt_1/output/r931005_1793209376_e02_p000132/rollout.mp4)
- [pilot_B100_01：成功，201ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_B100_01/attempt_1/output/r931005_1793209376_e00_p000063/rollout.mp4)
- [pilot_B100_02：失败，601ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_B100_02/attempt_1/output/r931005_1793209376_e00_p000040/rollout.mp4)
- [pilot_A060_00：失败，384ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_A060_00/attempt_1/output/r931005_1793209376_e02_p000132/rollout.mp4)
- [pilot_A060_01：成功，384ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_A060_01/attempt_1/output/r931005_1793209376_e00_p000063/rollout.mp4)
- [pilot_A060_02：成功，365ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/pilot_A060_02/attempt_1/output/r931005_1793209376_e00_p000040/rollout.mp4)
- [expansion_A060_25：失败，899ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/expansion_A060_25/attempt_1/output/fresh_23020026/rollout.mp4)
- [expansion_A060_33：失败，768ticks](/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/stages/expansion_A060_33/attempt_1/output/fresh_23020034/rollout.mp4)

旧State评估按用户要求提前暂停：有效143/原计划200，未完成57；属于探索性提前停止，原checkpoint与恢复位置保留，资源回收后独立接续本研究。

本地原始记录：`/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/expert_speed_20261105_01`。首次成功与正式终止、阶段时刻、运动统计及视频见[机器结果](evidence/slow_expert/results.json)。
成功耗时仅对成功样本统计，不能忽略失败或据此声称固定倍率。无训练更新；Image未启动。
<!-- SLOW_EXPERT_RESULTS_END -->
