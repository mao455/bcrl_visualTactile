# 当前交接：formal_05正式生产准入率停止

实际运行HEAD `5b3bd48ef720f71d860b2c76a32b39b4e27d8dc1`，干净启动，工作树`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`，分支`feature/tight-state-foundation`。源码无变更；报告更新见当前提交。IDE原树既存改动不动。

root `outputs/visuotactile/run/tight_state_20261107_01/formal_05`；service `tight-state-formal-20261107-05.service`。2026-11-07 20:08:38–21:22:04，全部自有进程已退出。75登记尝试：55原速=54成功+1初态拒绝，20慢速=20成功/10准入。第20次累计准入50%<65%触发预登记停止，任务100%≥85%通过；剩余200次未启动。worker/owner exit1是门控退出，不是Kit/OOM/143。容量已恢复，不再是当前阻断。

10条新合格记录覆盖10家族（train8/validation2/test0），保留完整载荷，仍在pending manifest。14条开发合格轨迹独立保留；原400/275只读保护。没有正式manifest/统计/cache、新随机训练或新模型闭环，不得把它们写成完成或0%成功。不要清STOP/重启、补seed、扩大范围或继续恒定alpha/cap搜索来绕过门槛。

已接受168测试/510tick等基础证据继续有效，不重审。固定Arm.50/Hand1/cap2rad/s实现实际减速，但闭环路径及阶段时长改变，生产yield不足。最小下一机制只提出采集侧参考路径进度/Arm-Hand同步时间重参数化，尚未实现或授权为自动新实验；不得改专家/PD/物理/DR/任务/学生。

B混速State best4500独立结果保留：train1/4，新初态0/10合法+2拒绝；旧同权重canonical1/7对legacy0/7有正向案例，无稳定泛化结论。无旧seed复跑或新训练。

本次10份已拒绝RGB-D载荷清理2.705GiB；此前7.510GiB不重复计数，失败numeric/分母完整保留。207源码/计划、6保护文件、64新HDF5、10拒绝numeric SHA核对通过；原400/275全部存在且大小正确。完整结果和75次表见[当前结果](STATE_DP_RESULTS.md)，执行合同见[当前合同](CONTRACTS.md)，清理见[CLEANUP_AUDIT](CLEANUP_AUDIT.md)。
