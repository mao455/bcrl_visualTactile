# 自适应阶段DAgger：执行登记（尚未完成）

本轮从`9417461`的完整A130-50/B130-50继续，按用户附件授权执行两轮真实学生前缀、专家从当前状态纠偏、共同累计池、四组件充分训练、闭环比较及最后冻结测试。独立分支`feature/tight-adaptive-dagger-20261119`；源码工作树位于主目录outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS，原工作区和历史报告不修改。

新run：`outputs/visuotactile/run/tight_adaptive_dagger_20261119_01`。每轮目标24–40条合法纠偏、最多56次机会、A/B均参与；真实无触发和专家失败全部保留。接触停滞、真实持物搬运、近孔停滞、早期失稳和首子目标后五类因果触发只用于采集，不改变学生输入、物理、成功或监督合同。连续无恢复收益的阶段在后续机会中前移。

每轮DP与Transformer均预登记60完整deck，所有原数据累计，原30示范每deck3访，其余各1访。各自DP从父EMA初始化、新AdamW与EMA，学习率1.5e-5；对应后端继承各自父权重、新AdamW1e-4。选中新DP之后重建各自nominal缓存，四个实际选中checkpoint均须通过新增轨迹访问≥50、逐合法tick不同已提交optimizer step≥50及事务日志重放验收，祖先曝光不计。只在达标模型中按原validation6选择。

每轮比较父/新完整A/B及72历史模型，原训练8/已打开开发8另加8个新开发初态。两轮闭合后先冻结模型，再一次打开28839000–007共8个新测试初态；测试打开后无训练或选模。不自动晋升。

原72→130丢失成功：A训练28817004、B开发28826000。原始命令从tick0已分叉；扣除各lane基座平移后，物体首次超过1mm分叉分别66、60tick。这是实际成对记录的观测，非反事实专家因果证明。B开发初态不得用于训练采集。五类屏幕在16条训练初态历史上的命中分别8/14/8/14/5，不能当作专家恢复率。

142项相关CPU检查通过（110项接管/历史/评估/统一入口与32项曝光/覆盖/B2）；真实4例计划准入验证完整父pair、训练family、原生10cm布局和无固定tick兜底。尚未证明本轮新纠偏或自主收益。

[登记配置](../configs/experiments/adaptive_dagger/experiment.json)、[保护SHA](evidence/tight_adaptive_dagger_20261119_01/protected.json)、[历史逐例诊断](evidence/tight_adaptive_dagger_20261119_01/baseline_diagnosis.json)。后续仅以真实run的逐轮采集、训练曝光与自主结果更新本报告。
