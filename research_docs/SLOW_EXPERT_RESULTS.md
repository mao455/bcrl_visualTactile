# 冻结Play2Perfect专家：慢速执行与真实示范

本轮是独立的专家执行研究。原State150历史100+全新100继续运行，源码、checkpoint、统计和队列保持；[原评估报告](https://github.com/mao455/bcrl_visualTactile/blob/feature/native-dp-training-v2/research_docs/STATE_DP_RESULTS.md)仍在原feature分支逐次更新。新工作在唯一独立工作树`bc+rl_RSS_expert_speed`和分支`feature/expert-slow-execution`，复用现有环境/数据，不按尝试复制源码。

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

先以原train320按成功时长分位选10个固定历史初态：原专家10条、系数1额外严格对照1条、A三配置各10、B四配置各10。历史PhysX隐藏状态不可完全恢复，所以必须先确认未减速专家本轮至少8/10成功、10条有效，再比较减速。所有历史初态仅称可观察重建；每次独立Kit，无复用或跨episode状态。若A三组均不足8/10，再做C的10条诊断：仅原生首子目标之后系数0.67，其余阶段保持1；不会发明新的成功阶段。

扩大门槛预先固定：10条均有效、成功≥80%、成功中位5–7s、双方成功配对耗时中位比1.4–2.2。B还须s=1参考至少8/10。优先成功数、再接近6s选择一组，进行20个新随机初态的原专家/候选配对；至少17/20候选成功且原专家至少16/20，耗时仍达标，才采集6条全新传感器小样本。比例只对双方成功统计，须同时看失败率与置信区间；小样本不等于部署可靠性证明。

原生阶段600tick预算保留（实际边界为601）；没有启用延时预算诊断。最大155次启动、固定24小时包括等待；软件故障保留并停止自动盲重启。仅使用一个持久owner和统一`tools.anchor_interim_eval`/`tools.anchor_continue_worker`；原200完成并资源回收前不启动Kit。

## 数据与训练门槛

只有随机20通过才录制6条真正重新执行的RGB-D、本体、模型均值（player裁剪前）、原始RL指令、未减速候选目标、实际执行目标、物体/目标、接触、时间和delay历史；使用原CompactEpisodeStream无损存储并逐字段SHA重建验收。成功与失败文件均保留证据，只有原生成功进入独立`slow_expert_pilot_v1`清单。监督仍为实际issued `target_after_rad`，绝不替换成到达关节位置。

这6条是数据合同验收样本，**不会自动被称为完整训练集**。其后的State离线/live输入配对、独立train/validation覆盖、原数据混合微调需依据真实采集结果继续审核；现有训练器与150checkpoint采样状态绑定，不能将新数据集静默塞进原续训。当前无慢速数据、无微调更新、无Image启动。只有慢速物理可靠性与新数据合同实测通过才进入后续扩展/小规模微调，不提前声明改善。

## 当前结论与后续决定

任务/权重与官方发布基点一致；当前3.4秒与真机时长不是同一条件。哪种慢速方法可靠、手指/末端/物体实际速度如何变化，均待配对物理结果。软件已记录各tick原候选/执行/到达状态，原生事件、接触和运动统计，成功率分母包含全部有效失败；软件故障独立记录。若没有5–7秒且可靠成功的配置，将停在诊断，不强行生成数据或重训。

33项相关CPU回归通过：原生动作链100tick与官方基点逐值/RNG一致、1.0精确旁路、Arm/Hand独立缩放、一次调用和真实target反馈、时间插值起点/终点、未改位置接口/成功审核/现有抽样入口。CPU通过不等于真实物理成功；本轮未取得Kit结果前不作成功声明。

新增验收器也已只读复核一条保留的真实专家记录：成功/终止145tick一致，新增物体/末端运动指标可正确提取。这是新统计代码的兼容性验证，未重复启动Kit，也不算本轮新baseline样本。

每个新专家worker在推理前后校验模型全部参数/缓冲指纹，确认没有优化更新。`run/resume`复用已提交stage；`release`对本研究拒绝，不能误放行旧训练。
