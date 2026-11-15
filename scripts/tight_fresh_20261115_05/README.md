# 抓持优先：在现有时段内接续第十五批

当前状态见 `research_docs/SESSION_HANDOFF.md`。本包实现、CPU检查已完成；新采集是否开始、专家是否救回及新学生行为，以运行回执为准。

第十四批已完成32机会/23成功/6415合法专家步，26.258分钟，52.555成功/h；其中16条是fixed1补充示范、7条为持物相关。306条累计轨迹来自162训练family。已有前端训练继续，冻结3d148de不改。

第十五批采用A11/B09完整配对，继承第十四批的数据池。重新实际运行第十三批held0遗漏的10个训练family（12个角色失败），A/B共20机会；当前空间与时间优先用于这些失败，新增训练family为0。`preregistered_training_revisit_v2`允许重访此前登记但从未成功入池的训练family，不伪造既有成功身份，也不迁移验证/test身份。事件种子29104000起；重新仿真，不恢复几个位置冒充精确物理重放。

`early_contact_recovery_v1`从已完成学生transition读取接触与高度：参考仍为25–48步高度中位数；接触后12步仍缺少双指、近期双指接触丢失2步、开始升高但接触不足、96步仍无接触、150步仍无接触抬升时，下一控制步接管。允许非12倍数，不以旋转或旧拿稳窗失效当掉落。这只是可恢复状态的候选规则，不保证专家救回。原30专家首次接触55–90步/中位68，接触抬升中位119；12个失败历史回看触发63–92步，0条新成功。

DP16遍、新纠偏repeat4；选中DP后重建commit12计划与可训练历史，再单独训练Transformer12遍，也对新纠偏整条repeat4。后端预算按重复后的合法专家tick人口计算，记录每条实际visit与监督步；完整前缀每visit按当前权重重算，prefix与队列过渡仍mask0。平均损失归一化不变，不重复此前首12步加权对照。初始/selected后端记录新纠偏全部合法、首90合法、全局150步前误差；selected与source前端用相同真实新轨迹报告误差及早期误差，A/B标签含义不变。

固定开发8/角色继续CPU standard8；之后GPU standard16评估10个困难训练family及24个额外未训练开发family/角色。新开发seed29500000起，与原始train/val/test及历次训练登记的seed/构型排除重叠；原test22不读取。后者是开发集，后续需保持隔离。

新持物诊断：60步/1秒始终高于参考2cm，至少90%步的两个120Hz子步均≥2指对物体接触>0.1N，连续接触缺口最多3步；不限制掌内转动。另报告120步与原生lift、near、alignment、subgoal、insertion。A11为4/7，B09与B13各2/8，不是90%稳定。候选采用条件包含固定组抓持不降与原困难训练初态改善；观察到90%还需额外独立批确认，不自动转向后段或覆盖回退。

本包**继承04固定截止2026-11-16 02:05:20.389**。等待旧owner和monitor正常退出后才启动新root，至少保留10000秒；不足就明确停止，不延长截止、不重跑旧实验。新root只是源码/数据身份隔离。普通owner负责阶段衔接、失败记录与清理；`after_current.py`观察首批已关闭真实结果。完整新流程耗时和行为结果仍待实际运行。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv_isaacsim/bin/python scripts/tight_fresh_20261115_05/check.py
# 在新冻结worktree中，用实际完整提交SHA运行；后台由systemd限定原截止：
.venv_isaacsim/bin/python scripts/tight_fresh_20261115_05/after_current.py \
  --worker-tree "$TIGHT_GRASP_TREE" --expected-commit "$TIGHT_GRASP_COMMIT" \
  --ledger /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_grasp_pending_20261115_05
```

统一检查6项通过，4.165秒（pytest1.64秒）；更早冷启动检查79.07秒也保留说明。无新增GPU检查或仿真。`retire.py`已按固定SHA清单删除15个旧非best中间/末尾权重：分配17,462,333,440字节，实时空闲增加17,313,419,264字节（同时存在训练写盘）；所有原始数据、best及当前冻结特征保留。**不要重跑清理。** 中间优化器恢复点不可再用，旧小型结果与最终best保留。
