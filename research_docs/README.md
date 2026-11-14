# Tight-Insertion 当前研究入口

本轮以根[AGENTS](../AGENTS.md)和用户授权为准：约30条不同初态完整成功示范重新训练；每60Hz真实三视角RGB-D进入学生，5Hz DP规划；持续开展完整A/B DAgger与自主评估。A预测下一控制步实际构型并配几何与后端，B预测专家绝对目标并配同类后端。

初始train30/validation6已完成从零训练，后续五轮共加入54条合法专家纠偏，训练人口84条、仍30初态family，test22隔离。第五轮完整自主A/B各0/4插入成功，A2/4、B1/4曾有既定拿稳窗口；没有原生对孔或子目标。最近后端输出起点对照四cell共16次自主均无拿稳/成功，虽然上一目标起点候选离线误差较低，实际主要卡在拿起，因此未采用。

正在执行同来源完整模型的视觉辅助监督对照：同辅助头/共同权重，对照detach读出、候选允许视觉梯度；物体相对掌位姿仅作训练标签。训练集尺度校准与真实CUDA验收通过，四cell各5遍DP已完成，当前刷新全部依赖缓存，随后各5遍后端和16次完整自主。辅助监督改善了位姿读出，动作验证没有优于第五轮来源；完整行为结果尚未完成，未晋升模型。主Agent持续分析阶段结果并衔接下一步。

实际进度集中于[交接](SESSION_HANDOFF.md)，证据见[evidence](evidence/tight_fresh_20261114/round05_anchor_study.json)；[合同](CONTRACTS.md)、[感知](PERCEPTION.md)、[方法](DAGGER_METHOD.md)、[性能](DAGGER_PERFORMANCE.md)、[变更](CHANGELOG.md)维护对应事实。源码和精简证据推origin/feature/tight-state-foundation，raw数据/权重留研究机。历史段落保留实验边界，不覆盖当前授权，不重启或延长旧run。
