# Tight-Insertion：两轮自适应DAgger实验（执行中）

本轮接续用户指定的 `9417461`，独立实验分支 `feature/tight-adaptive-dagger-20261119`，源码工作树位于主目录 `outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS`。主工作区和另一会话的文档整理保持隔离。实际run为 `outputs/visuotactile/run/tight_adaptive_dagger_20261119_01`，原有限session截止不变。本文只记录本轮，未改README或历史研究报告。

**尚未完成训练与闭环验证，当前没有新的学生自主收益结论。** Round 1采集已经闭合，A-DP正在运行；四组件的最终选中checkpoint必须各自通过本轮新增逐轨≥50访问、逐合法tick≥50不同已提交optimizer step的独立重建。父模型曝光不补本轮计数，未达标的早期验证最优不能入选。

## Round 1：采集与数据准入

| 学生 | 完整机会 | 真实接管 | 验收纠偏/专家救回 | 新增训练合法tick |
|---|---:|---:|---:|---:|
| A130-50 | 24 | 14 | 13 | 2,271 |
| B130-50 | 24 | 17 | 16 | 2,584 |
| 合计 | 48 | 31 | 29 | 4,855 |

共同池由130增至159条，原130逐条保留，仍使用原30训练family及固定6个验证family；A/B人口一致，DP目标和对应nominal缓存独立。原归一化不变。48条完整机会之外，原4lane资源中断的未闭合前缀单独保留，不混入有效采集分母；另一次准备失败没有物理记录。8个原已完成的机会因准入类别错误曾被拒绝，修复后只做CPU重新准入，没有重跑这8个物理机会。

阶段采用真实已完成学生历史触发，无固定tick回退。每阶段各12个完整机会：接触停滞7次接管/7救回，搬运8/8，对准7/6，支撑不稳9/8。阶段之间没有随机化对照，触发率与救回率不能当作因果优劣排序。

所有实际接管位于93–377 tick，覆盖12tick规划周期的多个相位，不把接管时刻取整至规划边界。31个实际交接的首合法标签均为switch+2；学生prefix与队列交接均mask0。48条记录的命令、关节、时间、actuator反馈和观察时刻连续性最大误差均为0；这只证明所记录可观测量连续，不证明隐藏仿真状态可确定重放。

A在family28817010的对准接管241 tick后未完成；B在family28817023的第一子目标后支撑不稳接管377 tick后未完成。两条失败后缀共792个原始合法专家tick完整保留为失败证据，未加入训练池。未触发的17机会中仅一条原生自主成功；其余未触发不作为专家救回失败。专家救回绝不计入学生自主成功率。

## 训练登记与当前验收

四组件预登记60完整deck，原30示范每deck重复3次，其余轨迹1次；各自从完整130-50父pair继续，fresh AdamW/EMA，DP学习率1.5e-5、后端1e-4。DP先完成，从选中DP重建各自后端nominal。4GiB冻结视觉hotset显式启用，所有实际历史保留。训练准入、资源守卫、恢复checkpoint和逐更新日志继续使用现有统一入口。

对A-DP durable epoch2/update220已独立从实际selected anchors与原始mask重算全部逐步贡献，ledger和轨迹访问完全一致：159条/32,363个合法tick，本轮最低轨迹访问2、tick不同更新2，**未达50、不可选中**。该核验只有CPU、0额外策略更新。

另补后端独立验收器，从原始HDF5合法mask、seeded重复deck、microbatch4及32tick/128合法tick窗口重建每次真实optimizer提交，不仅重放训练自己报告的贡献集合。已在父A后端实际选中50/16174上核验：130轨/27,508合法tick，全部日志逐项一致，最低访问/tick步骤50/50；这只是验收程序的真实历史检查，不能算本轮新曝光。正式本轮组件仍须分别通过。

当前训练owner未中断。协调器已在训练期间移交至独立分析边界脚本；原owner按原登记完成训练后，再做四组件曝光重建与完整配对评测，随后停在Round 1诊断边界。第二轮尚未采集，必须在第一轮真实闭环诊断后登记调整后的触发与配额，使用第一轮新完整学生。新终测28839000–007保持未打开。

## 基线诊断与评测合同

原72→130丢失成功为A训练28817004和B已打开开发28826000。原命令从tick0分叉，物体基座相对位置首次超过1mm分叉分别66/60 tick；这不是反事实专家因果真值。开发family不允许用于新采集。既有142项相关CPU检查通过；资源恢复另有51项接管、队列、记录与曝光回归。首个4lane资源中断时各lane只有40个学生tick、0接管、0合法纠偏，原失败保留。

48条实际学生前缀的专家shadow建议已单独分析，未变成监督标签。按当时真实前命令和关节限位重算可达区间后，A/B有22/20条在tick48后出现连续6tick、arm RMS超过完整可达跨度1/4的差异，首次中位67/61.5tick；原始弧度阈值对受限arm过大，原空结果保留。两条未救回案例A10/B23的归一化arm首次差异在164/220，分别先于接管77/157tick；hand在157/208，先于接管84/169tick。阈值只是描述性屏幕、不是首次有害错误真值，也不能从未执行的专家建议推出可恢复性。它提供后续更早真实接管的候选依据，最终Round 2调度仍要等待R1新学生闭环。

每轮预登记24个初态（原训练8、已打开开发8、新开发8），比较对应父/新完整A/B；72历史A/B在独立物理波检查。父/新A/B同一4lane波实际可观测初态检查，不把分波的72历史对照声称同波，也不声称隐藏状态精确重放。最终冻结后只打开8个新终测初态，测试后不再训练或选模，不自动晋升。

## 证据与复现

- `configs/experiments/adaptive_dagger/experiment.json`：预算、family隔离、初态、模型来源、有限session及资源合同。
- `research_docs/evidence/tight_adaptive_dagger_20261119_01/round01/collection_{summary,manifest}.json`：48完整机会、29条训练准入及数据SHA。
- `round01/combined_analysis/{records,summary}.json`：逐机会控制相位、mask、真实反馈、持物诊断及失败；路径相对同一证据根。
- `round01/learning_registration.json`：159共同池、原统计不变及四组件曝光门。
- `round01/partial_A_dp_audit.json`：早期durable checkpoint独立逐tick核验，明确未达标。
- `round01/shadow_diagnosis/`：完整学生前缀与实际shadow专家mapper命令的原始/可达跨度归一化差异、阶段统计及解释限制。
- `audit_backend_masks.py`、`backend_mask_replay_smoke.json`：从原始mask与seeded完整deck重建后端每次窗口提交，历史父A实际验收通过；不增加本轮训练曝光。
- `baseline_offline/{summary,per_episode}.csv`：父完整模型的全量真实历史离线误差，0CUDA，A/B各自目标空间解释。
- `late_alignment_failure.json`、`media_review.json`：真实三视角视频与已查看帧；视频原件在run的media目录。
- `round1_boundary.py`：等待既有训练owner完成后验收与评测，不重跑原采集或优化更新；失败会保留独立回执。

运行环境为既有isaaclab环境、2个Torch/BLAS线程；完整训练与物理执行继续走 `python -m tools.tight_project`，必须提供显式job与本轮有效session。运行中的训练源文件未热改。四组件最终曝光表、闭环逐例结果、曲线、两轮比较与最终结论将在真实完成后填入，不能把本执行中报告作为完成交付。
