# 当前交接：直接窄域新运行准备/启动验收

最新用户已授权数据就绪后自动随机初始化StateDP训练及有界闭环。工作树`bc+rl_RSS_narrow_state`，分支`feature/narrow-state-data`。新root`outputs/visuotactile/run/narrow_state_20261105_02`。实际启动与最新数字见该root的status.json；本段准备文字不代表DATA_READY或训练完成。

旧narrow_state_20261105_01的3条数据保持隔离，4h预算不延长；旧State143队列保持暂停。初始化查询矛盾已复现：q/qd及全部PhysX link poses相同，完成rollout后出现11个overlap命中；flush/kinematic/USD输出/同姿态root重写均无效。实际34个碰撞网格包围盒加1mm仍与物体相隔至少108.62mm。新版本使用实时碰撞网格包围盒保守准入，保留原query为诊断，不修改SDK或物理。

直接60家族目标、44/8/8预计split，约220真实多速度尝试、目标200/最低150成功。数据完整验收后自动GPU smoke→随机StateDP训练→train4+unseen12闭环。合同和预算详见CONTRACTS；结果见SLOW_EXPERT_RESULTS及STATE_DP_RESULTS。当前正式更新仍为0。

46项相关CPU回归通过。一次性CPU checkpoint已按原ledger核对身份/hash/打开句柄后删除，释放1,153,052,672分配字节；原400、冻结专家、旧隔离轨迹和未知文件保留。
