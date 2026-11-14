# Tight-Insertion 当前研究入口

本轮以根[AGENTS](../AGENTS.md)和用户授权为准：约30条不同初态完整成功示范重新训练；每60Hz真实三视角RGB-D进入学生，5Hz DP规划；持续开展完整A/B DAgger与自主评估。A预测下一控制步实际构型并配几何与后端，B预测专家绝对目标并配同类后端。

初始train30/validation6已完成从零训练，后续五轮共加入54条合法专家纠偏，训练人口84条、仍30初态family，test22隔离。第五轮完整自主A/B各0/4插入成功，A2/4、B1/4曾有既定拿稳窗口；没有原生对孔或子目标。最近后端输出起点对照四cell共16次自主均无拿稳/成功，虽然上一目标起点候选离线误差较低，实际主要卡在拿起，因此未采用。

视觉辅助对照已完成四cell及16次完整自主，全部0插入/对孔/子目标；离线位姿读出改善没有转化为行为收益，不采用，保留第五轮完整A/B。已登记第六批8训练family×A/B16次持续接触抬升接管机会，允许掌中转动；前两次动作前超时已保留，第三次已完成初始化并真实接管。实际救回数量与后段采样决策以完整16次审查为准。

实际进度集中于[交接](SESSION_HANDOFF.md)，证据见[evidence](evidence/tight_fresh_20261114/round05_anchor_study.json)；[合同](CONTRACTS.md)、[感知](PERCEPTION.md)、[方法](DAGGER_METHOD.md)、[性能](DAGGER_PERFORMANCE.md)、[变更](CHANGELOG.md)维护对应事实。源码和精简证据推origin/feature/tight-state-foundation，raw数据/权重留研究机。历史段落保留实验边界，不覆盖当前授权，不重启或延长旧run。
