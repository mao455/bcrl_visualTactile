# 交付规则

当前维护分支`feature/native-dp-training-v2`，remote `origin`为mao455/bcrl_visualTactile。用户已授权本轮审查后的代码、文档、精简JSON/图表提交和push，无需再次确认。

不push upstream，不force push，不改写历史；不混入其他worktree的未提交修改。原始HDF5、权重、视频、巨型日志、凭证和本地环境留在ignored outputs。删除过时实现不等于删除Git历史。

活动任务引用的源码保持冻结。组织整理在独立临时审查worktree完成，等相关worker退出、checkpoint完整且回归通过后切入当前feature，并记录冻结manifest迁移before/after SHA。最后移除已合入且干净的临时审查worktree。
