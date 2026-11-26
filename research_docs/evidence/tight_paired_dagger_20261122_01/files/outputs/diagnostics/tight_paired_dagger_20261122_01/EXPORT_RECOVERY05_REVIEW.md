# Recovery05 轻量终态证据导出器：未激活

`export_final_evidence_recovery05.py`已实现，26项隔离CPU fixture通过。当前真实输入检查仍为`blocked: Final assembler draft_ready required`，没有创建任何`research_docs/evidence`目标，没有运行GPU/Kit/训练或修改运行中代码。当前只读stat清单823文件、28,456,754字节，最大2,935,335字节；此时没有最终评估/新轮训练产物，不能当作最终清单或最终完成。

默认只检查门槛并给出dry inventory。实际导出必须显式`--apply`、指定新目标与`--selected-ledgers`，并且已存在`draft_ready_for_root_review`组装回执及绑定SHA的64家族/256四套学生结果。导出器重算全部配对与逐rollout成功一致性、绝对成功数，保留软件未完成行及全体分母，要求正式summary退出0。主执行者仍需审阅科学结论；工具不提交Git或推送。

允许清单是代码中的有限精确路径/文件名、有限8组件与60deck循环、预登记cases[*].name、注册wave路径和专家尝试台账；没有递归复制源树。包括10项实验配置、父本/预算/数据配置、匹配nominal缓存身份、部署/训练/曝光曲线、初态/rollout/漏斗/配对CSV、源身份登记、隔离恢复/中断/失败/闭合回执、计算开销和既有历史报告。生产Python仅从最终评估`source_sha256`取路径且严格核对冻结SHA；AUDIT编排/恢复源码依独立source closure提议中的实际冻结SHA核对，未绑定者标明仅保存当前非运行源码。测试、导出工具本身也保存，所有实际复制字节均登记源相对路径、SHA256、长度和类别。

拒绝HDF5、视觉/特征/缓存、权重/张量/RNG.pt、pickle/npz/npy、视频、logs/jsonl、压缩包、其他后缀，以及项目外路径和逃逸软链接。个体16MiB，总量256MiB（含生成MANIFEST和README）；先预检查全部源大小，再读有界字节核对SHA，并再次核对最终门槛，全部通过后才创建目标。只允许`research_docs/evidence`的一个不存在的直接子目录；独占创建且不覆盖，文件444、目录555。意外写入失败会保留部分目标并抛错，不删除失败证据，也不下次覆盖。

正式闭合并主执行者审阅后，可由root显式调用（本次未执行）：

```bash
python outputs/diagnostics/tight_paired_dagger_20261122_01/export_final_evidence_recovery05.py \
  --assembly /绝对路径/AUDIT/新的已完成组装目录 \
  --selected-ledgers /绝对路径/AUDIT/新的已完成8组件ledger提取目录

python outputs/diagnostics/tight_paired_dagger_20261122_01/export_final_evidence_recovery05.py \
  --assembly /绝对路径/AUDIT/新的已完成组装目录 \
  --selected-ledgers /绝对路径/AUDIT/新的已完成8组件ledger提取目录 \
  --destination /home/CNS2025331827/docs/bc+rl_RSS/research_docs/evidence/tight_paired_dagger_20261122_01 \
  --apply
```

26测试覆盖：未闭合绝不创建目标、dry无写入、少一rollout拒绝、family重复拒绝、assembler SHA变更拒绝、256软件失败全部保留、SHA逐文件一致、只读输出和不覆盖、禁止后缀及伪装后缀、项目外/软链接逃逸、payload目录、个体及总量限制、冻结源码SHA拒绝/成功、实际cases[*].name、真实缓存绑定/deck文件名、注册wave与case元数据，以及未登记文件不复制。

当前清单与机器可读限制见同名JSON。root还需确认正式闭合后的实际audit输出文件名和最终检查回执均在静态清单内；可审阅后补充精确路径，不应扩大到递归复制。原大载荷、历史权重及所有失败保留本机，本轻量包不声称能独立执行原始训练/物理重放。

新增必需门槛：独立`extract_selected_ledgers_recovery05.py`闭合后产出passed的`extraction_status.json`、`selected_ledger_summary.json`和8个组件JSON。导出器只读这些小JSON，核验8实际选中checkpoint SHA/epoch/update与assembler一致，逐合法tick成功step、轨迹访问和DP原生入口每槽均≥50，合法mask SHA、初始12tick网格人口、数据/本组DP→rear与A/B预算对应。10个提取文件进入manifest；任何缺项均在目标创建前拒绝。DP_PIPELINE_FEASIBILITY05及REPRODUCTION_PACKAGE_REVIEW05也已加入精确清单。

真实schema联调：使用提取器自己的隔离CPU fake-state fixture调用真实collect函数，生成10个JSON，再调用真实导出器selected_ledger_gate，26项全通过。没有打开本轮任一真实checkpoint。测试需isaaclab环境中的torch；普通系统python缺torch时该项跳过。


本次追加：实际 recovery05 终态回执与主流程闭合门槛已实现，仍未激活。导出器31项测试、提取器10项测试合计41项及86个子测试通过（3.78秒）；只有既有requests依赖警告。全部实际导出只发生在临时fixture；没有读取真实checkpoint、原始载荷，未启动GPU/Kit，也未创建research_docs/evidence目标。

强制清单新增实际collection_integrity_recovery05_actual01的integrity.json与CPU noise_audit.json、collection_noise_recovery05_actual01的最终CUDA noise_audit.json、主driver状态/storage就绪、waiter状态/reviewed_main退出，以及ROOT_EVIDENCE_TOOLS_REVIEW05和EXTRACT_SELECTED_LEDGERS_REVIEW05三份审阅文件。原有有限清单保留；现在同时纳入冻结main实际15个阶段的退出回执。storage_gate本来没有阶段退出文件，未制造该路径；waiter的aggregate_and_publish_once回执仅在实际产生时收入。

导出前要求main_stages_complete、reviewed_main_closed、主进程退出0且owned helper已完成回收；15阶段均退出0。main_cache、main_train与development/test owner都需注册job/commands数量对应成功project_complete，并children_reaped=true，出现本attempt failure则拒绝。Integrity必须passed并保留47通过/1未核验；CPU和CUDA噪声文件按各自真实schema核验96机会、48配对、认证/不可认证端点分母和匹配case集合。不会要求所有96端点都可认证，软件失败继续保留。所有实际闭合回执SHA进入gate，并在目标mkdir前再次核对。

新增fixture逐一删除37份必需回执，验证均在建目标前拒绝；另验证17种未闭合/失败/矛盾、实际manifest覆盖、条件聚合回执以及准备字节期间回执变化。未改生产源码、冻结driver/waiter/assembler、注册文件或活跃回执。机器审阅中的旧stat inventory是原历史快照，本次未重新扫描。ROOT_EVIDENCE_TOOLS_REVIEW05仍由root拥有，所载旧导出器SHA需要root在最终激活前更新；本次未冒充新的独立root审阅。
