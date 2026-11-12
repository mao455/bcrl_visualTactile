# Tight-Insertion：完整策略的成功筛选接管学习

**2026-11-12性能修订**：控制匹配run已实际启动，当前局部吞吐优化已测试、待安全边界接入；详情见 [DAGGER_PERFORMANCE](DAGGER_PERFORMANCE.md)。它保留下面的学习算法、标签与曝光，只允许第2轮两组DP计算并行，随后缓存/后端/评估顺序；不把局部1.228×/1.489×/1.935×写成整轮加速或任务成功。原run与旧审计快照均保留其时间范围。

本文区分三层证据：已经执行的历史协议、2026-11-12 控制/监督审计，以及新登记的公平 continuation。新算法实现、CPU验收、实际启动和闭环收益分别以机器凭据为准；尚未完成的实验不填成功率。当前结果入口为 [STATE_DP_RESULTS](STATE_DP_RESULTS.md)，动作及几何细节见 [CONTRACTS](CONTRACTS.md)。

## 1. 研究对象与完整学生

比较对象是两套完整部署策略，不是两个独立裸DP：

|组件|A / QG01|B / G01|共同条件|
|---|---|---|---|
|U-Net DP预测目标|专家执行后的真实配置 $q^+_t$|专家最终执行命令 $u^*_t$|29D canonical rad，H24/commit12，5Hz规划|
|前端观测|RGB-D、robot178|相同|冻结DINO；可训练投影、历史网络和U-Net继续学习|
|额外前端目标|既定手部软几何损失|无此损失|掌部FK监督按各自目标语义计算|
|Transformer|名义计划＋真实触觉/执行反馈→修正|相同结构|2层，d128，4头，FF256，756,581参数；60Hz更新|
|最终监督|组合后的 $u^*_t$|组合后的 $u^*_t$|共同命令normalizer，120Hz原生物理|

A的配置预测只是名义计划，必须经过已训练Transformer与最终执行适配器才能作为完整策略控制。后端缺失、加载失败或与前端SHA不配对时显式失败。历史采集的数值链审计没有发现静默裸DP回退；本次不把已有正确路径宣称为新修复。

设 $o_{\le t}$ 为真正经历过的因果观测，$h_t$ 为持续历史，$b_{t:t+23}=f_\theta(o_{\le t},h_t)$ 为当前名义计划，$c_t$ 为commit cursor。后端使用当前计划、过去反馈及触觉：

$$
\delta_t=r_\phi(b_{t:t+23},c_t,h_t,\mathrm{tactile}_{\le t}),\qquad
\bar u_t=b_{t+c_t}+\delta_t/s_u,\qquad
\hat u_t=\mathcal P(\bar u_t;p_t,\mathcal L).
$$

$s_u$ 为固定命令仿射的scale，$p_t$ 为当前动作之前真实执行器目标，$\mathcal L$ 为原关节限位。历史协议的 $\mathcal P$ 仅裁剪关节限位；新协议增加与专家相同的可达目标集合。各自前端保留不同的 $q/u$ normalizer；不能因为两者同为29维而互换。

## 2. 从真实学生状态连续接管

每条轨迹由登记的训练初态家族/seed开始，学生先自主执行完整策略。专家在每个真实观测上以recurrent shadow连续更新其隐藏态，其动作在学生前缀不控制环境。接管不会reset、回到理想姿态，或从另一条专家轨迹拼接状态。相同spec/seed只定义配对的可观测初态协议，不声称隐藏PhysX状态逐位一致。

接管时刻从预登记的有限控制tick集合分配，跨越早期接近、接触和较晚操作时间。计时覆盖不是语义阶段保证：历史第3轮A/B接管点分别有55/48个属于观测无接触，只有1/6个已有lift或子目标。应同时报告真实接管阶段，不能把八个时间点写成八种成功恢复机制。历史队列根据已完成轮的接管成功/阶段进展，有限地选择更早、更晚或较宽时间集合；新对照两组使用同一规则、同一当轮时间集合。

专家建议按原生规则裁剪到 $[-1,1]$，进入独立持续shadow队列。位置执行器保留原有一次delay索引抽样，专家使用此索引选取已保存建议；不额外抽样、也不破坏原native队列。随后进行native arm delta/hand affine/EMA/限位映射及已有Arm0.5、Hand1、Hand目标2rad/s处理。

队列有3槽，接管最初两tick仍可能含学生控制期间的专家建议。当前方案要求整个队列干净才监督，而不只检查本次抽中的槽。switch安排为 $12k-2$，第一个合法专家transition位于规划anchor $12k$。真实控制权、来源tick、delay index、queue clean、segment及命令都保留。

## 3. 什么是有效纠偏标签

令 $e_t$ 表示当前transition实际由专家控制，$d_t$ 表示延迟队列已清洁，$v_t$ 表示记录/时序完整。合法监督mask为 $m_t=e_t\land d_t\land v_t$。只有最终原生任务成功且通过记录准入的接管轨迹，才进入当前成功纠偏池。该成功筛选是episode级规则，不改变每tick实际来源。

|轨迹部分|历史输入|B和后端命令标签|A配置标签|
|---|---|---|---|
|学生前缀|完整保留|当前正式训练mask=0；shadow建议仅诊断|mask=0，学生q_after不能冒充专家q_reached|
|接管队列过渡|完整保留|mask=0|mask=0|
|干净专家实际执行后缀|完整保留|实际施加的最终命令 $u^*_t$|同transition积分后的真实 $q^+_t$|
|失败接管|完整保留诊断|未自动作为成功纠偏入池|未自动作为成功纠偏入池|

未来专家命令、未来实际配置、物体/接触安排不进入策略观测。几何所需后态上下文单独进入loss。未执行的专家建议不能提供反事实 $q^+$；现有delayed shadow建议也不能混称即时无延迟专家标签。若以后引入学生状态逐步标注，必须新建明确监督合同，重建recurrent历史并分别处理命令和实际配置标签；本次未凭空补这些标签。

DP监督窗口始于12的倍数，H24遇无效mask、控制段切换或尾部即截断；不能跨reset拼接。历史从episode起点持续推进，学生前缀也输入GRU；TBPTT detach与监督mask不清空运行历史。

## 4. 控制一致性审计与修复边界

历史 $u^*_t$ 已位于减速、delay和native映射之后。直接再把学生输出乘Arm0.5或再加delay，会改变训练目标与部署位置。因此“学生没有再次减速”本身不是重复处理bug。

实际差异在于：旧学生只受关节位置限位，仍可发出专家在相同 $p_t$ 下不可能产生的目标跃变。327条round2/3数值轨迹的标签/历史核对通过；学生前缀手部目标变化超过2rad/s的关节分量比例约11%–16%。该比例描述控制差异，不证明它是零成功的唯一原因。逐关节、接触阶段及真实运动对照见 [审计数值](evidence/control_consistency_20261112/summary.json) 和 [分阶段CSV](evidence/control_consistency_20261112/joint_stage_metrics.csv)。

新适配器以固定原生参数计算给定真实 $p_t$ 的可达集合：

$$
\mathcal R(p_t)=\{F(a,p_t):a\in[-1,1]^{29}\},\qquad
\mathcal P(\bar u;p_t,\mathcal L)=\operatorname{clip}(\bar u,\min\mathcal R(p_t),\max\mathcal R(p_t)).
$$

$F$ 是原生单关节单调映射和既有速度处理。通过两个端点即可得到各关节区间，无优化求解。在远离限位时，arm每tick上界为 $0.5\times0.1\times1.5/60=0.00125$ rad；hand还受原生EMA可达区间及 $2/60$ rad变化上限共同限制。这里限制目标变化，不改PD或真实关节速度。

同一纯torch函数在后端训练的最终命令loss、离线预测及实际部署中使用；参考量必须是robot178的58:87实际pre-action target，不能用DP重预测或过去名义计划。已合法的专家最终命令应在浮点容差内保持不变。新身份显式写入checkpoint配置，旧checkpoint保持旧数学行为。新适配器的CPU等价/梯度/真实标签读回检查与首批物理采集检查分别记录；不能仅靠CPU结果宣称物理轨迹完全等价或成功率改善。

新损失仍监督执行后的组合命令，而非把residual当完整动作。可达投影外的clip区间有零梯度；应报告实际饱和与有效梯度，不能假设加入投影必然改善学习。

CPU读回覆盖564条非test轨迹、169,853个合法专家tick：专家映射与日志最终命令最大差为0，投影改变专家标签的tick为0；端点FP32舍入最大约1.49e-8rad，固定1e-6rad容差只处理该数值边界。实际已训练A/B模型各用一个Core窗口和一个恢复初段窗口检查新组合loss，四个数据拟合梯度L2为0.001636、0.001742、0.015742、0.001816，均非零；投影生效的关节分量约46%–57%，不能隐去其零梯度区域。见[执行验收](evidence/control_consistency_20261112/execution_acceptance.json)及[真实模型梯度](evidence/control_consistency_20261112/execution_model_acceptance.json)。这些检查没有optimizer更新、GPU或新Kit启动。

历史来源可追溯到直接位置执行路径 `c7fcbb0e2d27e21fa5c818171014413d2b067d0a`、专家手部目标限速 `024cdaabf420b26565fb173ea2c917ddff7e98f2`，以及保留学生直接目标接口的DAgger专家桥 `01ce8d6ba785f73e8737f6598f4ceb4e13515b35`。受影响的是学生可提交目标范围与相关自主结果的控制条件解释，未发现这三个路径把合法专家标签改错。原数据、模型和负结果保留；新身份下双后端先适配，随后重新收集及更新两个模块，不把新结果追加进旧分母。

![学生前缀逐关节诊断](evidence/control_consistency_20261112/student_prefix_joint_metrics.png)
![实际恢复覆盖](evidence/control_consistency_20261112/recovery_coverage.png)

## 5. 聚合、两阶段更新与公平预算

原Core train166、全部合格旧纠偏和本轮新纠偏持续累积，按内容SHA去重；correction-dev在整episode采集计划中预登记并永久隔离。原validation30用于开发，test22不参与采集、训练、选模或本次评估。同轮A/B读同一不可变快照，采集率不同如实保留，拒绝或失败不补seed。

训练按Core/历史纠偏/本轮纠偏 $0.5/0.2/0.3$ 分层。前端在纠偏层内部约一半anchor来自首48个连续合法专家tick，其余仍覆盖完整后缀；后端对该初段最终命令loss加权3倍，按真实权重总和归一化。加权质量和重复抽样不是新增物理tick或新增恢复起点。历史第3轮A新纠偏层初段anchor实际占约50.18%，因此“恢复初段完全没被训练”不符合记录。

每轮先更新DP全部原可训练部分，DINO仍冻结；选定新前端后，用真实完整前缀刷新所有依赖它的H24计划/cursor/mask、可训练视觉读出、持续hidden与previous_nominal，再更新Transformer。真实issued/applied、q/qd、触觉和时间戳保持日志值。只复用图像、backbone、内容SHA均匹配的冻结DINO原始特征；DP相关缓存必须绑定新权重、数据、统计和mask身份。

新可比序列采用以下已登记设计，实际完成量另列：

|项目|新协议|
|---|---|
|前置适配phase0|两组各冻结已有前端，用相同旧池快照，将后端适配到共同Arm0.2/Hand0.9幅度与新执行器；6有效tick遍，训练完成才部署；不称作一个完整DAgger轮次|
|完整DAgger轮次|最多2轮，A/B每轮各48机会；配对家族、seed、接管规则与dev分区|
|前端训练预算|每组每轮6有效anchor等效遍；共同分层抽样与种子，完整batch取整差异明示|
|后端训练预算|每组每轮12有效专家监督tick等效遍；完整episode组取整差异明示|
|部署与选模|使用共同曝光endpoint配套；另留validation-best作诊断，不按闭环lift给一组多轮或回退另一组后继续比较|
|自主评估|每组每轮固定10个已登记开发机会；不根据中间表现扩缩，不补非法seed|
|有限墙钟|新continuation正式admission起至多24h，CPU准备、phase0、采集、缓存、训练、评估与等待均计入；旧run及其13:24截止不改|

两组初始已训练权重仍有历史曝光差异，无法通过新命名抹除。共同新协议和残差幅度消除了后续对应工程变量，但不构成从同一随机初始化出发的严格单变量研究；论文必须披露历史来源。A的几何开销与各自模型真实有效监督人口、实际消耗单独报告。若某组阶段失败或预算不足，保留部分证据，不单独让另一组进入更多轮后声称配对完成。

统一幅度有明确动机：B历史手部修正幅度仅0.1，第2轮恢复初段48tick中约33.47%的专家所需修正超出该界，第3轮约17.72%；A相应比例低于1.1%。这是表达范围不足的离线证据，不等于模型实际tanh饱和率，也不证明放大就能成功。phase0从各自已训练权重继续，双方均在共同范围上训练，不能只换B的buffer后直接部署。

时间估算依据历史第3轮：每组96次采集约2.13–2.22h、800 DP更新约1.27h、564条前端预测缓存约31min、12后端tick遍约28min。新每组每轮48次采集约1.1h、6 anchor遍随人口增长约2–2.4h、缓存35–45min、后端30–40min及10次评估约20min；phase0加两轮顺序安排预计20–24h，实际以吞吐与剩余预算为准。24h是本次明确有限上限；阶段超时、双组曝光不匹配或不足以完成下一对时停止并如实发布部分结果。

```mermaid
flowchart LR
  P[配套完整学生 A/B] --> C[配对训练初态 rollout]
  C --> E[同状态 recurrent 专家连续接管]
  E --> V[成功筛选与逐 tick 合法 mask]
  V --> D[原 Core 加历轮纠偏和新纠偏]
  D --> S[冻结共同快照]
  S --> F[分别更新 DP]
  F --> K[新 DP 刷新参数相关缓存]
  K --> R[冻结新 DP 更新 Transformer]
  R --> T[配套纯学生固定开发评估]
  T -->|有限下一轮| P
```

```text
Input: existing paired models for A/B; Core; qualified recovery pool; fixed budgets
Adapt both backends to the common execution contract and bounds on the same old snapshot
for k = 1,...,2 within the registered absolute deadline:
    for g in [A,B]:
        for each preregistered paired training start and takeover time:
            run complete student; advance expert shadow on every true observation
            switch control continuously; retain real command/queue provenance
            save all outcomes; admit only successful, valid expert suffixes
    freeze one shared snapshot; keep development episodes isolated
    if admitted_performance_revision and k >= 2:
        train both DPs in parallel for common anchor exposure, each with its own label semantics
        wait for both registered DP jobs to finish
    for g in [A,B]:
        if not parallel_frontends: train DP for common exposure using g's own labels
        freeze new endpoint DP; refresh every DP-dependent backend cache
        train backend for common valid-tick exposure against final expert commands
        save coupled endpoint; run 10 fixed pure-student development attempts
    continue only after both complete pipelines resolve; never reset the deadline
export all actual denominators, provenance, curves, failures and local artifact indices
```

## 6. 实验表与解释边界

可复现实验数据在 [逐轮表](evidence/control_consistency_20261112/experiments.csv)、[训练曝光](evidence/control_consistency_20261112/training_exposure.csv)、[独立恢复案例](evidence/control_consistency_20261112/recovery_cases.csv) 与 [自主逐例结果](evidence/control_consistency_20261112/autonomous_cases.csv)。采集机会、合法起点、真实接管、成功纠偏、初段有效tick/anchor、训练实际曝光和自主分母分别列出；未完成字段保持空值。

以下是控制审计生成时的旧run快照，不是当前matched run结果。该时刻历史round2 A/B各0/4自主完整成功；round3 A亦0/4，全601tick超时，无lift/子目标，B仍在训练。round3采集A/B各96机会、93合法、90条成功纠偏；180新纠偏聚合后池共368条（333train/35dev），加Core后训练499条。专家恢复率高并不表示学生能自主完成任务。

与经典逐状态标注DAgger相比，本方法让学生产生接管前状态，但训练监督集中于**成功恢复后的实际专家执行后缀**，不是对所有学生访问状态都采集训练标签。经典算法聚合学生诱导分布上的专家标签；其保证不能直接转用于当前成功筛选方案。[Ross等，AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html)。

因此应称为“带数据聚合的成功筛选专家接管恢复学习”。失败接管被排除会产生选择偏差，较晚恢复后缀可能稀释最初错误状态；早期前缀没有专家配置反事实标签，不能通过复制q_after解决。当前证据支持这些机制限制，以及学生实际目标越界差异；尚不支持“数据量不足是唯一失败原因”“触觉/几何带来因果收益”或“新执行器已改善自主成功”。离线命令MSE、目标FK和观测接触阶段都不等于实际任务成功或严格配对物理反事实。

## 7. 代码与恢复入口

控制映射：[position_control.py](../visuotactile/position_control.py)；采集/完整策略数值验收：[state_benchmark.py](../visuotactile/state_benchmark.py)；监督/mask与聚合：[narrow_data.py](../visuotactile/narrow_data.py)；训练及新执行器组合loss：[anchor_continue.py](../visuotactile/anchor_continue.py)；队列/共同曝光/后台接续：[tight_dagger.py](../tools/tight_dagger.py)；CPU数值与图表导出：[tight_diagnostics.py](../tools/tight_diagnostics.py)。

保持已有统一pipeline。新版本在隔离工作树准备；旧owner退出并完成发布后才部署。恢复使用既有 `dagger_resume --root <continuation_root>`，按持久阶段/SHA跳过完整工作，训练恢复保留optimizer/EMA/RNG；同run不重设截止。数据、权重、视频留本地，公开摘要、源版本、数值、图表和索引。实际服务、root、启动/停止/发布凭据以[交接](SESSION_HANDOFF.md)为准。

CPU复现命令如下，输出另写临时目录，不能覆盖运行准入已绑定SHA的凭据。第一个命令使用固定旧A-r03/B-r02自主记录；第二个命令按当时已存在的旧run报告导出，因此后续新增B-r03评估时需保留生成时间。

```bash
bash visuotactile/run_cpu.sh -m visuotactile.position_control \
  --execution-audit \
  --source outputs/visuotactile/run/tight_dagger_pool_20261111_01/round_03/QG01/dp/state_data \
  --rollout-root outputs/visuotactile/run/tight_dagger_pool_20261111_01/round_03/QG01/backend/QG01/evaluation \
  --rollout-root outputs/visuotactile/run/tight_dagger_pool_20261111_01/round_02/G01/backend/G01/evaluation \
  --model-root outputs/visuotactile/run/tight_dagger_pool_20261111_01/round_03/QG01/backend/QG01 \
  --model-root outputs/visuotactile/run/tight_dagger_pool_20261111_01/round_02/G01/backend/G01 \
  --output outputs/visuotactile/tmp/control_audit_reproduction

bash visuotactile/run_cpu.sh -m tools.tight_diagnostics control_audit_publish \
  --root outputs/visuotactile/run/tight_dagger_pool_20261111_01 \
  --output outputs/visuotactile/tmp/control_audit_reproduction/figures
```
