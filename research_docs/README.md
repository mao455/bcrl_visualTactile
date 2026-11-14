# Tight-Insertion 当前研究入口

本轮以根[AGENTS](../AGENTS.md)和用户授权为准：约30条不同初态完整成功示范重新训练；每60Hz真实三视角RGB-D进入学生，5Hz DP规划；持续开展完整A/B DAgger与自主评估。A预测下一控制步实际构型并配几何与后端，B预测专家绝对目标并配同类后端。

初始train30/validation6已完成从零训练，后续五轮共加入54条合法专家纠偏，训练人口84条、仍30初态family，test22隔离。第五轮完整自主A/B各0/4插入成功，A2/4、B1/4曾有既定拿稳窗口；没有原生对孔或子目标。最近后端输出起点对照四cell共16次自主均无拿稳/成功，虽然上一目标起点候选离线误差较低，实际主要卡在拿起，因此未采用。

视觉辅助对照已完成四组及16次完整自主，全部0插入/对孔/子目标，未采用。第六至八批已新增34条成功纠偏、9005合法专家步，最近完整训练池118条、42个训练family；原6验证family与test22保持隔离。第八轮完整自主A0/7、B0/8，少数初态持物延长，但仍不能转正、对孔或插入，保留A05/B06回退。新采集选A08+B06仅为获得互补的真实偏差状态，不是整体性能晋升。

第九批已启动64个新训练family、A/B共128次机会，CPUstandard16/M8批量视觉与网络、有界异步写盘和并行校验。首波6条专家恢复已入池，其中2条来自延后重查的持物状态；全部机会、后续训练与自主结果仍待完成。真实持久owner在原session06截止内自动核验采集，再接完整DP→依赖刷新→后端→自主；正式使用过和仅候选的加速项分别记录，不把独立benchmark当整轮收益。

实际进度集中于[交接](SESSION_HANDOFF.md)，证据见[evidence](evidence/tight_fresh_20261114/round05_anchor_study.json)；[合同](CONTRACTS.md)、[感知](PERCEPTION.md)、[方法](DAGGER_METHOD.md)、[性能](DAGGER_PERFORMANCE.md)、[变更](CHANGELOG.md)维护对应事实。源码和精简证据推origin/feature/tight-state-foundation，raw数据/权重留研究机。历史段落保留实验边界，不覆盖当前授权，不重启或延长旧run。
