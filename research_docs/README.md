# Tight-Insertion 当前研究入口

2026-11-14开始新一轮：约30条不同初态成功示范、DP/后端从零训练、每60Hz真实三视角视觉入网，5Hz规划。A预测下一控制步实际构型并配几何与后端；B预测专家目标并配同类后端。研究结论以完整学生“拿起、拿稳、转正、对孔、插入”的实际结果为准。

有效授权集中于[AGENTS](../AGENTS.md)，实际进度见[交接](SESSION_HANDOFF.md)与[结果](STATE_DP_RESULTS.md)。[合同](CONTRACTS.md)、[感知](PERCEPTION.md)、[方法](DAGGER_METHOD.md)、[性能](DAGGER_PERFORMANCE.md)、[变更](CHANGELOG.md)分别维护事实。下层文档的旧日期段落是历史证据，不重新授权旧run或恢复旧截止。

首批30训练/6验证已完成从零A/B训练与每步真实视觉验收。首轮11条合法专家纠偏也已完成DP→全缓存刷新→后端→自主复测；A/B各1/4拿稳、各0/4完整成功，仍无对孔或子目标。第二轮24机会已采到10条成功纠偏；训练在04:40因几何监督mask遗漏失败，09:27发现后修复，恢复与结果见最新交接。新源码与精简证据持续推origin/feature/tight-state-foundation，原始数据/权重留本机。
