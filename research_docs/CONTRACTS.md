# 当前StateDP合同与数据准入

## 2026-11-06 数据可信度与StateDP变更审计（已完成）

审计已完成，独立root `dataset_trust_20261106_01` 为audit_complete/await_user_decision；60/60轨迹用尽，所有自有GPU进程退出，0训练/0学生rollout/0正式扩采。State数值275条、三视角75条准入，原400/275、权重及失败证据保留。不得自动继续任何旧runner或新训练/采集；下一步由用户决定。详见SLOW_EXPERT_RESULTS、STATE_DP_RESULTS和evidence/dataset_trust，历史授权段落不是当前待办。

先重读研究目标与冻结历史，逐条审核275条原始数据全部模态、分布、减速、成功/失败及采样偏差；基点84cbb98至0c209b9做真实Git/AST/运行SHA核查。静态通过后独立分层专家/逐tick target/H24回放，root dataset_trust_20261106_01，固定3h含准备等待、最多60轨迹启动、8GiB新增/60GiB磁盘保留、共享目标35%及6GiB VRAM/12GiB RAM。仅CPU审核和冻结RL专家/记录命令，正式数据新增0、学生rollout0、optimizer更新0。所有原400/275、manifest、正式checkpoint和失败证据只读。审计报告和准入建议为终点，用户决定下一阶段；不自动扩采/重训/学生误差调参/Image/nominal/触觉。H24/12、5/60/120Hz、target_after_rad而非q_after、29D、网络/损失/归一化/历史/原生物理保持。新脚本独立，原始验收标记不改写。

已完成的A/B/C机制诊断及旧预算仅为历史。其tiny多窗口采样和overlap4不属于正式baseline默认。本轮不继续该研究方向。

## 正式模型与输入

StateDP为普通state-based diffusion policy：Actor141、本体robot178和特权geometry85，GRU256持续历史；不读取图像或触觉。固定物理120Hz、底层60Hz、规划5Hz；H24预测未来24个60Hz目标，每12tick重新规划并执行前12步。29D绝对关节位置弧度（7Arm+22Hand），不是XYZ、速度增量或geometric nominal监督。

保持`position_rad_v1`/version3、sample/x0、Loss B及既有position/rotation/fingertip/temporal辅助权重，不改Arm/Hand权重。DDIM100训练噪声步、推理16步、eta0、clip_sample=false；正式overlap_steps=0。CUDA matmul/CuDNN TF32均关闭，与实际训练一致。Actor使用冻结专家RMS140 clip5和任务embedding32；robot逐tick、geometry每12tick按训练split拟合mean/std，动作每关节训练范围仿射归一化（常量维单独处理）。测试数据不拟合统计。

原200正式train148/val26/test26条、60家族44/8/8。扩充275正式train207/val34/test34条、92家族68/12/12；速度版本随初态家族同split。275全部State-valid，只有新增75条Multi-view-valid，旧200固定相机空白不改写；触觉为实际记录，首tick无效及后续有效标记必须保留。未来Image/触觉公平比较只能用相同有效模态子集，不能把数据量差异当架构优势。

## 标签、时间和执行

观察t发生在动作前。标签`transition/target_after_rad[t]`是本tick实际送入执行器的处理后绝对目标；`obs.actuator_target_rad[t] == target_before_rad[t] == target_after_rad[t-1]`（t>0）。`q_before[t]`和`q_after[t]`是真实到达关节位置，与目标不同，不能用q_after冒充动作标签。全275条逐tick验证q_after[t]=q_before[t+1]，两物理子步执行器target都等于当tick标签。

原生专家command经过原delay/Arm增量与Hand绝对映射/EMA/限位后产生target。StateDP已学习处理后的target，反归一化后按真实关节限位裁剪、canonical→Lab，仅写原目标缓冲；不能再次施加delay/EMA，不能强制真实q/物体状态。保留原delay随机抽样以保持后续环境随机流调用顺序，该抽样不再施加到位置目标。原动力学、资产、接触、成功/终止条件均不改。

robot178的116:145为严格过去请求位置，145:174为严格过去应用位置，174:176为age、176为valid；初态清零。训练使用明确标识的过去专家target teacher forcing；部署使用自身实际请求/应用反馈。current/future target仅作监督。每个真实60Hz观察更新GRU，replan不清空记忆，仅真实episode改变时reset；不得用子目标会重置的episode_length_buf判定新episode。

## 一次optimizer更新的真实含义

一个episode是一条完整变长轨迹，tick=1/60秒；候选anchor为0,12,24,…<长度。正式`AnchorSampler`先洗牌episode deck，每次最多64个不同episode，每episode从自己的无放回anchor deck取1个。275数据每轮实际batch64/64/64/15，平均51.75；200数据为64/64/20。尾batch不丢、不复制。它是episode-balanced，并非全体窗口等概率；较长episode的单anchor更少被抽到，某家族的速度版本越多则家族总权重越大。

每个anchor使用当前权重从真实episode开头重算历史：anchor所在96tick块之前no_grad得到hidden，此块至当前tick参与梯度；不是随机窗口零hidden，也不是固定长度96滑窗。短期robot13个native tick、geometry3个replan时刻与完整因果GRU共同形成condition[B,512]。

target[B,24,29]、valid[B,24]；尾部不足24只零填充且mask无效。每4个anchor做一次前向/反向，按本完整batch的有效target总数加权base/FK/rotation/finger项，按有效相邻对总数加权temporal项；全部microbatch后才一次grad clip1、AdamW和EMA更新。不能把microbatch4等同于effective batch4。

275 best4500累计232875次anchor抽样/4245=54.8587 anchor等效epoch，last5000累计258750/4245=60.9541；独立覆盖都是4245。原200 best5735累计282956/2829=100.0198。这些是重复曝光，不是新增样本数，也不是完整episode反传epoch。真实逐anchor访问、阶段和家族权重见state_mechanism轻量统计。

B诊断单独使用全anchor洗牌，多窗口可来自同一episode：family_066_repair_a055的25个anchor每轮16/9、micro4；完整因果历史、mask和损失保持。沿用275 train-only归一化及专家RMS/embedding，但参数、优化器、EMA和RNG全新。其checkpoint绝不替换正式275 best4500。

## A与C的诊断语义

A的native专家在线每tick反馈；H24/commit12 Oracle则回放该专家产生的处理后target，另回放一条已存慢速target。回放为开环，不冒称持续在线纠正；记录源target→请求→执行器→真实q及块边界。精确目标映射是门控，隐藏PhysX状态/轨迹是否完全相同不是门控；序列耗尽未成功记censored，不伪装原生timeout。只补足未执行的H24尾预测，不额外执行填充帧。

C固定275 best4500，开发初态为两个TRAIN家族、两扩散seed，比较overlap0/4。overlap4在新plan前4目标中使用旧plan对齐的未来12:16，权重1/4、1/2、3/4、1，属于明确额外控制变化，不改正式默认。统计分别报告raw预测、实际请求/执行器target与真实关节速度；相同seed/可观测初态不等于隐藏物理轨迹严格配对。

## 已完成历史与来源

200→275的单次数据对照已于2026-11-06 15:19结束：旧/新train0/4、unseen0/10有效，各2初态拒绝；275模型5000步平台早停、best4500。旧200未续训，全部数据和模型保留。本轮新授权不重置任何旧deadline或统计。

旧H32/H64、400条320/40/40、150 anchor等效epoch、历史专家/位置接口验证、已停止的旧runner预算均为历史，不是当前执行入口；完整事实保留于[5604249时的合同](https://github.com/mao455/bcrl_visualTactile/blob/5604249/research_docs/CONTRACTS.md)及STATE_DP_RESULTS历史段落。不能通过改metadata给旧checkpoint切换合同。
