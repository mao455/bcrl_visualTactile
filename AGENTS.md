# Play2Perfect / Tight-Insertion：现行工作约定

先读 [研究方向](research_docs/USER_RESEARCH_DIRECTION.md)、[合同](research_docs/CONTRACTS.md)、[交接](research_docs/SESSION_HANDOFF.md) 和 [实际结果](research_docs/STATE_DP_RESULTS.md)。当前授权与实现配置位于 `configs/tight_state.json` 的 `dagger`；历史结果不构成重启旧队列的指令。

## 当前覆盖 — 2026-11-11 多轮共享纠偏池

最新用户授权覆盖以下历史单轮规则。在隔离 `bc+rl_RSS_qg_report` 复用现有模型、训练器、collector和普通owner，实现最多3轮、总18h（含等待，最后10min发布预留）的成功筛选接管式DAgger，优先QG01、再G11。起点是已完成 `tight_dagger_20261111_01` 各自选定的完整前后端配对。当前实现/CPU验收不代表新队列已启动GPU；注册、实际起点与绝对deadline以新root机器凭据为准，恢复不得重置。

两组共用持续去重池和每轮不可变快照，已有21条合法成功纠偏全属历史train。每轮每组默认32机会（有限规则可取24），其中4个整episode预登记为correction_dev；3轮总最多192次机会、每组最多96，失败不补种子。Core、历史纠偏、本轮纠偏按 .5/.2/.3 分层；空纠偏层权重移至另一纠偏层，记录实际抽样/重复/唯一覆盖。原validation30与新增correction_dev分开，test22不读取。

每轮先微调现有U-Net前端，再按选定前端刷新全train/validation/correction_dev名义缓存，最后微调Transformer后端。DP最多10/最低6有效anchor等效遍，约每2遍验证；后端最多12/最低6有效监督tick等效遍，每3遍验证；LR分别1e-5、3e-4，吞吐短探针不另缩正式LR日程。后端以完整episode组取整，实际超出tick公开。选模含真实update0，旧val/纠偏dev/纠偏首24有效tick分别按其update0归一化，再按 .5/.25/.25 合成。保留上轮完整pair回退，禁止拼接不同前端下各自best后端。

自主开发先固定4例，有>=1成功、>=1子目标或>=2原生lift才按原预登记队列扩至10；不以0成功自动停止有界下一轮。当前Transformer无失触立即硬清零；原幅度/限位保留，不增加新门控、新网络或改变物理。旧GRU门控只属旧checkpoint合同。

恢复入口：`bash visuotactile/run_cpu.sh -m tools.tight_finish dagger_resume --root <root>`。已完成且SHA核验的阶段跳过；中断训练从last完整状态恢复。STOP标记在owner/root及具体训练目录，恢复只清理本任务登记的STOP，不重启无关进程；未持久化loss行另存uncommitted日志，正式曝光按持久化计数。新root实际启动由owner完成一次核验后交给普通脚本，不由模型轮询守候。

以下各节是上一轮授权与完成前快照；冲突时以上述覆盖、当前CONTRACTS和最新用户指令为准，原始证据保留。

## 当前授权与隔离

用户已授权当前 `tight_transformer_20261110_02` 正常结束后，自动进行一轮成功筛选的专家接管式 DAgger：G11、QG01 各最多12次训练分布采集，聚合后先更新现有 U-Net 前端，再冻结新前端、刷新预测缓存并更新 Transformer 后端，最后各最多10次固定开发自主评估。最新更正覆盖附件最初“只更新后端、冻结前端”的要求。

旧实验继续使用 `bc+rl_RSS_palm_eval` 的冻结源码、数据、预算和模型；准备工作只在隔离 `bc+rl_RSS_dagger`。不得热改旧运行树、关闭源码哈希检查或抢先启动新 GPU/Kit。普通后继程序核验旧 owner/自有 worker 实际退出、相关选定 checkpoint 有效，再接续；单纯报告收尾失败不能冒充权重损坏。G11 复用既有 G10，QG01 复用上一轮 QG00；模型来源和 SHA 必须明确。

一轮新阶段上限12小时，包含其采集、探针、训练、缓存、评估与等待；使用实际新起点，不继承旧绝对截止。两组共享接管规则和机会，不能追加尝试凑成功。普通脚本负责后台推进；注册后短查实际状态，Codex 不轮询守候。脚本存在不等于已注册，注册不等于已启动 GPU。

## 数据、训练与评估

- 学生从训练初态自主执行，专家 shadow 随真实观察连续推进；接管不重置、不拼接 episode。原生专家动作经过已有映射与明确限速后送同一29D位置执行接口。
- 只有原生完整成功的接管 episode 可进纠正池。学生前缀与切换队列未清空段保留真实历史，监督为零；专家有效后缀才有标签。提前学生成功单独统计。
- 前端标签按语义区分 `u_expert` 与 `q_reached`；后者只允许实际专家执行后的配置，不能把学生 q_after 当反事实标签。H24 不跨无效段、控制权切换或 reset；尾部显式 mask。
- 保持 actual issued/applied、q/qd、触觉与时钟；`previous_nominal` 不替代执行反馈。持续记忆从完整真实前缀推进，监督 mask 不清空记忆。
- 原 Core train166 加新成功纠正；原 validation30 固定选模，test22 不用于本轮采集、调参或评估。原仿射统计与后端幅度界限冻结，不在聚合后重拟合。
- 前端最多20有效 anchor 等效遍历、10后允许验证停止；后端最多30有效监督 tick 遍历、10后允许验证停止。探针决定有限墙钟，完整恢复与正式更新分开计数；硬截止优先，不称低曝光为收敛。
- 新前端选定后，全部 train/validation 的名义预测与持续历史重新生成并绑定 DP、数据和统计 SHA；只有冻结 DINO 特征可逐 SHA 复用。
- 最终学生自主评估无专家动作替换，采用原生完整成功；接管成功不能计入自主成功，G11/QG01 两组比较不能隔离证明几何因果收益。

## 工程与保护

沿用 `native_data/narrow_data/state_history/anchor_sampling/anchor_continue/state_benchmark`；`tools.tight_finish` 是公共入口，`tools.tight_dagger` 只协调后继与阶段，不另造模型、采集器或训练器。H24/commit12、29D Arm7+Hand22绝对 rad、5/60/120Hz、原生物理/资产/PD/任务成功不变；特权几何只进入专家或明确的 loss context。

可充分使用空闲 GPU，保留6GiB显存、14GiB RAM、60GiB磁盘；不操作无关进程。只对自有进程实施退出与恢复。每次实质变更更新现有 CHANGELOG/合同/状态，不新增重复 MD，不把 CPU/mock 验收写成真实接管能力。

用户已授权按实际依赖清理旧代码、权重、缓存与报告。保护原400与仍用 Core、专家/资产、当前和已排队依赖、初始/选定/可恢复模型及必要比较分母；确认无引用后才删除。清理记录文件数、实际释放空间和依赖核验，不搬进归档冒充删除。

审查后只推 `origin/feature/tight-state-foundation`，不 force、不改共享历史、不推 upstream；旧 owner 先完成其发布，再整合新工作树。大轨迹、权重、视频留本地。必要历史已收敛到 STATE_DP_RESULTS 与既有小型结果记录，不沿用过时的“无自动下一阶段”限制。
