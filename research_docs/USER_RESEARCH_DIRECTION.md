# 用户研究方向与当前授权

长期目标仍是视觉名义策略与独立触觉Correction的分工。普通专家动作模仿不是去预加载或“纯几何意图”，本阶段也没有证明触觉或RL收益。未训练Correction不得控制机器人。

当前用户优先要求解释StateDP为何离线误差低但闭环难以抓取。允许基于现有checkpoint修正采样/训练预算，已实施64不同episode anchors、完整历史和可恢复续训。此次完成150 anchor等效epoch评估，比较10000/16584/20918，并增加train初态拟合诊断；不允许再从头训练、更换网络/损失/执行接口或盲目延长。

用户明确新增全面清理：检查实际依赖后删除过时源代码、checkpoint、缓存、大量旧失败数据及重复文档；保留400原始成功、当前数据与完整恢复状态、未完成比较和精简科研证据，不建几十GB备份。当前维护一套实现与配置化入口、统一结果和清理报告；审查后提交push origin feature。

原资产、物理、控制、reset/DR/success、其他任务和共享SDK仍受保护。State允许既有声明的几何状态；Image不读特权几何。输入/标签来源、严格过去历史、软件错误与任务失败必须如实区分。

详细合同见[CONTRACTS](CONTRACTS.md)，当前实测见[STATE_DP_RESULTS](STATE_DP_RESULTS.md)。2026-10-27至11-04的完整授权原文保存在[清理前历史](https://github.com/mao455/bcrl_visualTactile/blob/48a3516/research_docs/USER_RESEARCH_DIRECTION.md)，历史待办不自动变成当前授权。
