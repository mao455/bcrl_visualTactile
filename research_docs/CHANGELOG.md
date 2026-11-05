# 维护变更

完整旧阶段记录保存在[48a3516 CHANGELOG](https://github.com/mao455/bcrl_visualTactile/blob/48a3516/research_docs/CHANGELOG.md)。此处只保留当前变更，不复制历史巨型日志。

## 2026-11-05 — StateDP150评估与整理

- 原150 epoch trainer继续，不新建训练系统；既有评估入口增加milestone/cohort参数及证据审查hold，固定validation4+train4独立统计。
- 统一端点评估10000/16584/20918，train320/validation40，前12/24/precontact增量、hold基线、FK方向/幅值和累计误差；原始模型与控制保持。
- 从历史runner提取完整checkpoint恢复、进程I/O、视频、轨迹/FK统计；保留函数源码AST等价证据。Native环境、controller、数据时序和网络参数不重设计。
- 删除依赖图不可达旧实验实现与其专属测试/启动服务；将历史阶段文档合并为当前合同、结果、清理和精简历史索引。
- 清理按显式保护清单、文件身份/SHA和进程引用验证；硬链接按最后一条链接的实际分配块计数。完整ledger本地，摘要公开。400原始数据、当前缓存、统计、checkpoint和必要专家/模型视频保留。
- 本轮具体删除数量、释放空间、已运行测试与后续端点评估实测必须见[CLEANUP_AUDIT](CLEANUP_AUDIT.md)与[STATE_DP_RESULTS](STATE_DP_RESULTS.md)，准备或授权不算完成。

## 2026-11-05 — 整理后在线初始化修复

150阶段20918更新、train4/validation4与统一train320/validation40端点比较已经完成。原8次均有效，0软件缺失，未混入额外验收。

整理后的额外验收在模型首次动作前因`native_data.object_dimensions`缺少ElementTree导入失败，原日志及blocked状态保留。补回原导入；新增真实资产/缓存尺寸及在线Controller初始化回归。网络、损失、checkpoint、原生控制与已完成结果不变。只分配一次修复后的独立验收，不修改原失败计数或原评估期限；修复后78测试通过；唯一补充Kit启动在场景前FileThread SIGABRT（−6），保留且停止自动重试。实际回归与限制写统一报告。
