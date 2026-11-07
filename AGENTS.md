# 当前 Tight-Insertion：Core/Auxiliary合同接续

本轮从干净e26e3d92e7daa39bc695a9bef3bce64b49ca7538继续，仅修改准入/队列。使用`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`。用户明确授权新版Core/Auxiliary合同，旧65%严格产出率停机不再约束新生产；旧formal_05及10/20统计仍只读。当前运行root为`outputs/visuotactile/run/tight_state_20261107_01/composition_06`，实际状态须读status.json，运行源码完整HEAD为4d43a3975c9bd153ff90d7c13e03ca9a0e6aa917；2026-11-07 22:30由tight-state-composition-20261107-06.service启动，首条新增Core通过，尚未冻结/训练/评估。

固定Arm.50/Hand1/cap2rad/s，不开发新控制器；Aux bounds在新采集前固定1.35–3.30，Core1.5–3；最多8家族轮次、440慢速计划身份含旧20，复用55原速与10完整Core。旧2条数值Aux无RGB-D不能复用。原55家族/split/seed公式/范围及09:08采集、次日03:08整体deadline保持。Core优先，Aux≤100且≤Core/2、每家族≤2且≤本家族Core。Core-only训练；详见CONTRACTS。

已完成针对性29D动作/物理时间确定性检查；当前GRU持续完整prefix，并非64tick截断历史。H24未来标签从anchor开始，在线plan取tick%12；未发现需修模型的动作/时序错位。源码未变的168/510tick等工程验收继续有效。61项定向/准入/复用队列回归通过后已开始采集。原400/275/专家/资产/成功模型/失败分母保护，不重开旧审计或重试旧案例。

固定模型、资源、执行与真实交付要求以[当前合同](research_docs/CONTRACTS.md)为准；[当前结果](research_docs/STATE_DP_RESULTS.md)保留历史实验及本轮状态。
