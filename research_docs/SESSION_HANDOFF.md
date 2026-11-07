# 当前有限实验交接

最新授权从1b70d3b6继续，A=Hand cap最小实测、B=独立canonical State诊断。入口仍tools.tight_finish，root `outputs/visuotactile/run/tight_state_20261107_01/hand_cap_04`；先6次固定2rad/s cap，再16次canonical best4500。只有A改善才条件启动4独立家族×3event可靠性验证；规则和种子已经登记，见CONTRACTS及root plans。旧工程证据冻结；新增33受影响回归与21条6765tick关闭限速恒等通过。尚无本轮物理结果，不把启动当完成。无新训练计划，理由和旧曝光/平台在合同。

## 上一轮已完成记录



从362c630继续，A运行代码991db6a1c02af140ef12fb1f3a35f159c7ba2193，root为`outputs/visuotactile/run/tight_state_20261107_01/baseline_03`。24次登记尝试全部结束，自有owner/worker/monitor均退出；旧275混速State数据best4500同权重legacy/canonical输入对照：train均0/4，未见初态旧0/7、canonical1/7，各另1次动作前初态拒绝。unseen_05 canonical178tick完整成功，legacy601timeout；数值修复有正向案例，未证明稳定改善。新训练更新0，无checkpoint复制。

B只读分析33capture（21慢+12原速），新采样0：B00耗时增加主要伴随更长插入路径；B01/B02 Hand插入速度仍接近原速、Arm路径缩短。旧Hand更低参数也呈非单调时间及最终插入timeout。没有可靠统一方案，不生产新正式慢速集。最小提案仅在采集侧Hand目标增加因果速度上限，尚未实施；不得把A成功当B过关。

原168项/510tick等工程证据有效，本轮24受影响回归通过。运行后只读分析器补入实际两子步外力统计，33条重算一致；学生/执行代码未再修改。完整结果、逐条表、CSV/图和机器证据均在既有STATE_DP_RESULTS及evidence/tight_state_20261107.json。原400/275/专家/资产、旧6+新2开发合格记录保持；本轮清理0。当前无需重开基础审计或自动扩展队列。
