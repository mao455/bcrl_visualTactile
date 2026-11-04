# 当前交接

本轮：StateDP150 anchor等效epoch评估与全面清理。当前事实以[统一结果](STATE_DP_RESULTS.md)、[清理报告](CLEANUP_AUDIT.md)及机器状态为准。

- 工作分支`feature/native-dp-training-v2`，共享outputs真实目录属于原bc+rl_RSS项目。
- 原训练root `outputs/visuotactile/native_dp_v2/anchor_continue_20261104_01`；评估root `anchor_assessment_20261105_01`。原trainer自然完成20918后，owner被保持暂停，禁止未审查直接继续。
- 本次8个冻结初态：先validation4 ordinal4–7，后train4 ordinal0–3，分别统计；最多9启动。不重复原10000/16584闭环，也不使用test40。
- 后台分析在Kit资源释放后依次评估10000/16584/20918的完整train320、validation40，保存每checkpoint数值和逐episode目标。模型、损失、归一化、控制器未改。
- 当前/最佳完整checkpoint、400原始数据和索引缓存保护。old-directory依赖仍保留，不能再按年代整目录删。全删除ledger与保留原因在评估root/maintenance；Git只放摘要。
- 旧code/log保留规则已被本轮明确清理授权覆盖；不可删除未知或活动依赖。清理前Git版本48a3516可检索所有旧实现。
- 严格Image live输入配对仍未通过。不能因State离线误差下降自动启动Image大训练。

所有代码/结果的最终切入、回归及push SHA必须按实际完成更新；本交接不把授权或准备代码写成已完成评估。
