第七批完成16次真实学生前缀：7次触发接管，6次专家恢复并插入、通过校验入池；1次专家续接超时，9次没有满足持续接触抬高条件。A入池3条/837合法专家步，B入池3条/523步，共1360步。失败专家的27合法控制步另行保留，未入训练。学生自主插入为0，专家成功不能计为学生成功。

本批8个原训练初态家族各运行A/B一次；6条入池轨迹来自5个已有家族，不是6个新初态。接管使用第五轮完整A/B，contact-height0/12，无fallback。所有学生前缀原生抬升0/16，旧拿稳窗口1/16，持续双指接触抬高7/16，原生near/alignment/首子目标均为0。原生抬升与允许转动的持物筛选是不同判据，不能相互替代。

六条成功纠偏均从持续持物但方向仍错的真实状态接起：切换在127–167步，物体高于沉降参考49–79mm，root距首目标147–191mm，原生最大关键点距离284–372mm，姿态角差1.199–2.608rad。它们补到了早期持物后的转向过程，仍没有补到学生自己接近孔口后产生的精细偏差。六条切换时均不满足旧拿稳窗口，这不等于已掉落。B/formal007直到572步才接管，专家只剩29步即达到原601步超时；这是晚触发后的恢复失败，不能据此单独断言该物理状态不可恢复。

实际加入的1360个专家动作中，529步处于已完成的持续持物窗口；其中267步存在过去12步掌中相对转动超过0.2rad，333步同时处于同一当前目标关键点60mm内。原生near127步、alignment12步、首子目标之后344步，6条均由专家完成原任务。各计数是可重叠的诊断掩码，不是互斥阶段；相对运动不证明有意调整或打滑。抓取前/首次接触仍在真实学生历史里，但这批该阶段合法专家动作数为0。新增数据尚未参加优化器训练。

| 完整owner口径 | 串行第六批 | 批量第七批 |
|---|---:|---:|
| 尝试 / 触发 / 入池 | 16 / 9 / 9 | 16 / 7 / 6 |
| 入池家族数（已有训练家族） | 6 | 5 |
| 入池合法专家控制步 | 1850 | 1360 |
| 全部实际控制步 | 7245 | 8229 |
| 完整墙钟秒（含启动/收尾/准入） | 1218.417 | 1059.375 |
| 尝试每小时 | 47.274 | 54.372 |
| 入池轨迹每小时 | 26.592 | 20.389 |
| 入池合法专家步每小时 | 5466.110 | 4621.594 |
| 总控制步每秒 | 5.946 | 7.768 |

总控制吞吐增加30.6%、机会吞吐增加15.0%，但有效入池轨迹吞吐减少23.3%、合法专家步吞吐减少15.5%。**正式批量采集已跑通，尚不能宣布有效采集速度提升。** 本批触发/恢复比例下降，且轨迹时长不同；相同登记初态和模型不意味着批量RNG、隐藏物理状态或接管状态精确配对。两批还同时改变Kit线程4→16、视觉microbatch与批量调度、首计划取证开启→关闭、同步→异步写入；这个比值不是某一优化的独立因果收益。

第七批两个wave共1003.360秒，owner1059.375秒。初始化20.797秒；worker中初始化和wave外的31.536秒同时包含准入、加载和汇总，旧冻结代码未单列准入耗时，不能全部称为准入。下一实现已单列`wave.admission_s`。单writer累计compact追加367.640秒、校验371.297秒；主提交等待92.337秒、最终等待268.329秒。它们存在重叠，不能相加为墙钟或比例。两波payload峰值536.506/536.675MB，接近512MiB上限，队列峰值36项、最终保留0字节；这不是进程RSS。并行校验新实现尚未用于第七批，不能提前计作实测提速。

`pool07_untrained`已经由CPU构造为99条：原30示范＋历史54纠偏＋第六批9条＋第七批6条，共仍30个训练家族，6验证条目不变。A/B旧数据entry、所有旧numeric tensor、原30统计和split均精确保留；新监督掩码均为1360步，未把失败27步或学生前缀入监督。没有跑图像缓存或训练。这个池的continuation忠实记录采集来源A05/B05，不能称为A05/B06权重；下一独立时段必须分别绑定选定A05/B06策略与这个累计数据池，完成105条train+validation的冻结DINO缓存后再推进。

现有`existing_identities`只用于原训练身份，B06四个开发初态不能转作训练。near60的[事前设计](near60_design.json)及[CPU重放](near60_replay.json)只取其诊断证据；下一批用事前新注册训练家族，独立新dev初态另行保留。原四个开发轨迹继续纵向追踪，不能重复称为独立泛化。

[collection06.json](collection06.json)、[collection07.json](collection07.json)保存逐case当前/专家权限覆盖、接管边界、原生终止和SHA；[throughput.json](throughput.json)保存完整分母、计时字段来源和对比限制；[pool07_binding.json](pool07_binding.json)保存累计池精确检查。CPU复核两批32条真实numeric的时钟、相邻状态、触发、队列冲刷及标签；两套冻结源码分别179/193文件全部SHA匹配。源记录的全部字段/相机验收仍由原采集准入保留，本次未重复解压全部RGB-D，不声称再次全量重算图像校验。

研究机CPU复现，输出必须用新文件名：

```bash
PY=/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python
BASE=/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_05
$PY -m tools.tight_batch_collection_review --collection "$BASE/takeover_07_batch8_m8_01" --output /path/to/new_collection07.json
$PY -m tools.tight_batch_collection_review --collection "$BASE/takeover_06_tasking4_01" --output /path/to/new_collection06.json
$PY -m tools.tight_batch_throughput_report --serial-root "$BASE/takeover_06_tasking4_01" --batch-root "$BASE/takeover_07_batch8_m8_01" --output /path/to/new_throughput.json
```

本次已经执行的CPU聚合命令（目标已存在，禁止原址重跑）：

```bash
$PY -m tools.tight_fresh_dagger aggregate --root "$BASE/round_06" --collection "$BASE/takeover_07_batch8_m8_01" --output "$BASE/pool07_untrained"
```

零新仿真、零优化器更新、零测试轨迹访问。大权重、HDF5和缓存留在研究机。
