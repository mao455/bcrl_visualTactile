# 最新用户更正：保留前端并复用 G00/G10

本轮从 f352ac8 停止记录接续新的授权。保留 U-Net、现有编码器与持续历史；直接复用 G00/G10 selected EMA，本轮两前端 optimizer 更新数必须为0。G01/G11 使用现有756581参数后端Transformer（2层、d128、4头、FF256、过去32tick与持续GRU），不替换前端。
Q00/QG00 共用旧 G00 EMA 参数作为初始化，分别使用相同 train166 的 q_reached affine、新优化器/EMA计数/采样器和学习率日程；这是新监督适配，不把旧 u_expert 权重称作已经学好的配置策略。H24/commit12、29D绝对目标、5/60/120Hz、166/30/22划分及原生成功判据不变。q_after标签、加噪clean sample与掌部FK监督一致切换，未来物体/接触安排只进loss。所有后端最终仍监督 u_expert，统一命令affine，并使用真实冻结DDIM16前端预测。

读 research_docs/CONTRACTS.md 最新段和 STATE_DP_RESULTS.md；旧18/2/20/60/144小时限制不适用于本轮。各分支按实测有限预算独立推进，test22不用；短查真实更新和checkpoint后交普通后台，不依赖模型轮询。旧run tight_transformer_20261110_01正式更新0，FAILED不重启；新root _02。

以下为历史授权，冲突以本段及当前用户为准。

# 2026-11-10 当前授权：匹配Transformer与实际配置监督

用户附件明确授权原位重构、训练/评估、依赖核对后实际清理和origin feature/tight-state-foundation推送。起点8e67fb7，实际工作树bc+rl_RSS_palm_eval；其它工作树未提交修改保护。读CONTRACTS当前段、STATE_DP_RESULTS及最新SESSION_HANDOFF。
复用现有native_dp/native_data/anchor_continue/tight_finish，G00/G10重新匹配训练，G01/G11冻结新基础并用Transformer；Q00/Q01实际配置监督，QG仅真实几何门通过准入。没有完整动作依赖力模型/接触校准时内力明确BLOCKED。
run outputs/visuotactile/run/tight_transformer_20261110_01；actual status/launch/budget/physics gate优先。200规划anchor遍历与200全控制tick遍历不混称；test22只能在模型/评价规则冻结后，不能选模。
首GPU gate起3h验收和144h绝对上限包含等待，不重置；probe测得预算封存后才能正式训练，匹配曝光不足停止依赖阶段。原始400/Core、旧首次成功/分母/selected weights保护；允许删除已证实可重建无引用派生物。
充分使用空闲GPU，6GiB VRAM/14GiB RAM/60GiB磁盘；不动无关进程。冻结运行源码后不得修改正在使用代码。普通owner自动阶段转换/退出；Codex短查启动后返回，不持续轮询等待。
本段授权不是实现或训练完成证据；以下全部为已完成历史记录。

# 2026-11-10 已完成：G01/G11结果发布

用户要求上传GitHub。本次只发布c3386b6已经完成的结果：G01 0/10，G11 1/10，均无拒绝/中断，test22未用。
两组98610更新/200接触tick等效遍历，固定validation选epoch110/180；原基础冻结。12:01:02+08:00正常结束，所有自有进程退出。
4视频完整解码，包括G11 seed27150000原生完整成功279tick；该例native_lift=false，不能改写指标或反推抬升。
原始400/275、218Core、Aux、权重、缓存、视频与所有历史失败记录保护。无自动下一轮/新推理/Kit/补录/删除。
只扩展现有CPU publish入口、当前结果/README/合同，公开源码/MD/9图/CSV/JSON与压缩数值指标，origin feature/tight-state-foundation。
正式资源窗口最大采样间隔1.1382s；继承monitor包含验收重启间隙807.409s，须分开披露。退出前4条PhysX错误影响未专项审计。
读STATE_DP_RESULTS及evidence/tight_residual_20261110/summary.json。下面训练授权为已执行历史，不授权重启。

# 2026-11-10 当前授权：G01/G11 60Hz触觉残差

用户明确授权下一阶段，覆盖下文历史“不启动G01/G11”限制。实际工作树bc+rl_RSS_palm_eval，参照4164f83。
G01冻结G00 EMA4356acbc；G11冻结G10 EMA5f3460ab；两基础不重训，原0/10结果及视频和FAILED退出凭据全部保留。
复用3e11d62的60Hz GRU残差，不采用10Hz Transformer；适配当前H24/commit12/29D绝对执行器目标。
两组相同结构、字段、300接触tick等效遍历上限、>=200才允许平台停止、开发10个固定spec/种子；最多2视频/组。
离线nominal必须是冻结基础完整DDIM16真实预测；过去专家执行历史明确teacher forcing。闭环反馈实际混合目标。
持续历史只在真episode重置，基础5Hz、残差60Hz、触觉120Hz；不改数据166/30/22、控制、任务成功、物理、资产。
独立普通owner复用tools.tight_finish residual_run：冻结预测缓存<=1.5h，训练<=9h，离线<=1h，评估<=1h，总<=12.5h含等待。
允许空闲GPU充分使用；实测并行/串行择优，仅管理自己子进程，保留6GiB VRAM/14GiB RAM/60GiB磁盘。
验收和提交推送后启动一次；Codex只短确认实际推进后退出，不轮询、不另开agent守候。普通owner最终更新STATE_DP_RESULTS。
下一轮不得自动启动；原始专家数据、400/275、Core/Aux、checkpoint、视频均保护。公开仅源码/配置/MD/小指标，origin feature/tight-state-foundation。
实际就绪/进度以本轮run的residual_acceptance.json/status.json/launch.json为准，授权不表示训练或成功已完成。

# 2026-11-10 当前阶段：发布G00/G10既有实验资料

用户明确要求把代码和实验资料提交GitHub；沿用代码/MD/图表/小型原始指标公开，大checkpoint/源轨迹/视频本地保留的范围。
读STATE_DP_RESULTS。G00/G10各10合法/0成功，原生lift1/3、首子目标3/1，2视频完整解码；20例已完成且自有进程全部退出。
训练10132/6929更新、119.541/81.753 anchor epoch，固定墙钟停止；各38离线完成，test22未使用，曝光不等，不能作触觉因果结论。
接续队列在所有trial落盘后因退出函数要求root/result.json顶层software_complete而报错，实际使用evaluation_complete.json；原FAILED完整保留。
本次只扩展现有CPU指标导出与绘图，公开目录research_docs/evidence/tight_vision_20261110；不修改历史状态、不重训、不再评估或补录。
只推origin feature/tight-state-foundation，不force、不推upstream；不要把前轮有限接续授权当成自动重试权限。

# 2026-11-10 已结束授权：修复视觉评估调用并接续

用户明确要求“那改好，继续”。98dfafe训练已完成有限预算：G00 10132更新、G10 6929更新，最终各38条离线完成。
旧闭环在首次act前漏传canonical q/qd而失败，0控制transition/0完整trial；原FAILED、partial视频与checkpoint全部保留。
仅修state_benchmark调用：G00/G10明确传入当前joint_position_rad/joint_velocity_rad_s两字段，不传其他诊断真值。
复用原tools.tight_finish owner的vision_resume，root为tight_vision_20261109_01/evaluation_resume_01；只接续原10×2身份、各首1视频。
不重训、不重算离线、不改变模型/采样/控制，不自动第二次重试；原评估截止2026-11-10 02:08:51 +08:00不重置。
6GiB VRAM/14GiB RAM/60GiB磁盘储备、完整GPU授权不变；先提交推送后冻结，启动短查≤60–90s即退出。
实际状态以接续root/status.json为准；原root/status.json继续保留历史FAILED，不混作新状态。

# 2026-11-09 历史授权：G00/G10视觉训练

本轮用户明确授权218 Core(166/30/22)上的G00/G10新视觉策略；不是继续StateDP。
实际源码工作树bc+rl_RSS_palm_eval，起点72a9256，现有tools.tight_finish vision_run与configs/tight_state.json vision_pair。
读CONTRACTS与STATE_DP_RESULTS。无Actor141/Geometry85真值；共用冻结DINOv2-S/14缓存与共同随机初始化；G10联合训练触觉直接融合。
完整GPU使用覆盖旧45/50%上限；6GiB VRAM/14GiB RAM/60GiB磁盘储备保持。已正常停止明确识别的fingereye-async-production.service，数据保留。
缓存1.5h、共享训练墙钟9h、离线合计1h、单Kit评估合计1h，总12.5h；每组25428更新≈300epoch、原200后平台规则。
固定validation选模；各最多10同spec开发trial，首1视频；test22不用于选择。G01/G11本轮不训练，不新增数据/自动调参/失败重试。
只编辑现有源码/配置/MD，先提交推origin feature再启动并冻结；Codex启动后短监督≤60–90s即退出，普通owner收尾，无LLM轮询。
本轮root outputs/visuotactile/run/tight_vision_20261109_01；机器status/launch与实际产物优先于历史叙述。

# 2026-11-09 已完成StateDP结案与发布范围

训练root tight_palm_20261108_01 COMPLETE：19865更新/234.374 anchor等效epoch，validation平台停止；
评估root tight_palm_eval_20261109_01/remaining_01 COMPLETE：10合法/0成功、8timeout/2fall，3视频完整解码。
全部owned worker/monitor已退出；不自动重训、重录或扩大评估。实际指标/资源越界/源码身份见STATE_DP_RESULTS。
用户明确要求推送代码、MD、图表、小型原始指标；完整checkpoint/源轨迹/视频保留本地。
清理范围明确为tight_state_20261107_01，需逐项清单；当前218数据中的208仍引用该root，不得整体删除。
已删除一个重复可重建numeric缓存，13日志无损压缩，回收289.15MiB；旧best、统计、轨迹和全部分母保留。
清单及恢复说明见CLEANUP_AUDIT与evidence/tight_palm_20261109。仅推origin feature/tight-state-foundation，不force、不推upstream。
下文是已经完成的阶段授权与证据，不代表仍有待运行队列。

# 2026-11-09 用户明确接续：GPU采样偏高先等待（已完成）

用户在收到“启动4秒整卡均值59.76%触发终止、物理0次”的结果后明确要求“不要因为采样太多停止，继续”。
授权同一19865更新EMA/同10个未启动seed的一次有界接续；root为原tight_palm_eval_20261109_01/remaining_01。
复用现有VideoLease的exceeded_mean_action=wait_for_original_admission：负载偏高先等待，不因启动短样本立即失败；
整卡50%上限、45%准入/35%目标、6GiB VRAM/14GiB RAM/60GiB磁盘不放宽，实际越界仍留账。
重新登记一次≤1h墙钟（含初始化/资源等待），不追改已过期旧deadline，不自动开启更多轮次。
不改模型、采样器、任务/控制/物理，不重训；前3个原身份录制，原FAILED证据和0次分母保留。
当前代码已经支持该策略，无需新增源码或调度器；冻结run配置后普通后台执行，Codex只短监督。

# 2026-11-09 最新授权：平台后小队列闭环与视频

用户明确授权“接近收敛可以尝试成功率实验并录制视频”，仅覆盖此前无新学生闭环限制。
原训练运行于35485d7工作树，保持其代码、预算、数据、优化器和评估完全不动。
本分支/独立工作树复用tools.tight_videos与configs/tight_failure_videos.json的conditional_evaluation；
普通CPU程序等待原训练及最终38条离线诊断正常结束、全部owned进程退出，再检查原≥200 anchor等效epoch/最后4次validation分数改善<1%、无严重退化条件。
未满足写SKIPPED，不称预算耗尽为收敛；失败则终止，不能放宽阈值来凑视频。
满足才使用最终validation4选定EMA，冻结10个新初态seed27150000–27150009，前3身份录制；拒绝不补seed/补录。
最多1h GPU评估，单Kit串行；零成功不扩100，有成功也不扩本次10例。保持29D/H24/commit12/5/60/120Hz与原生完整成功判据。
沿用50%整卡批次/60s上限、35%目标、45%准入、6GiB VRAM/14GiB RAM/60GiB磁盘；实测越界停止自己的录制worker，保留中断分母。
等待截止为原整体deadline+120s，评估绝对截止再加1h；不重置原训练截止、不重试、不新训。
root outputs/visuotactile/demos/tight_palm_eval_20261109_01。机器状态/README为准；Codex启动短查后退出，无LLM守候。

# 当前 Tight-Insertion：掌部几何损失与有限扩充的新训练

最新显式任务授权一个全新StateDP：只原位改已有文件，补齐输入审查，commit12掌部位置/完整旋转辅助loss，
原36训练家族有限新增目标50/最多60Core，最多120物理尝试/2h；原val30/test22、Aux不混入。
一次fresh训练目标300 anchor等效epoch，通常≥200才平台早停，训练墙钟9h优先，最后固定train8/validation30离线诊断≤1h；无自动学生闭环或下一轮。
读CONTRACTS与STATE_DP_RESULTS，配置configs/tight_state.json，入口tools.tight_finish。
root `outputs/visuotactile/run/tight_palm_20261108_01`；读取实际status判断是否启动/结束，不从本段推断训练计数。
采用严格Lease整卡50%上限/35%目标/45%准入，6GiB VRAM/14GiB RAM/60GiB磁盘储备；全流程≤12.5h。
冻结源码后不改运行代码，普通后台完成转换/训练/报告，Codex不驻守；结束后不自动追加。

## 已完成 Core/Auxiliary 基线

使用`/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup`、`feature/tight-state-foundation`；读research_docs/CONTRACTS.md、STATE_DP_RESULTS.md和SESSION_HANDOFF.md。root为outputs/visuotactile/run/tight_state_20261107_01/composition_06，2026-11-08 13:21:54 complete/evaluated，所有owned服务/worker/monitor已退出，不重启旧队列。

采集源码4d43a3975c9bd153ff90d7c13e03ca9a0e6aa917；训练/评估fdb7ea94a1457de57ff2a9fc21d1a9486742d5ab。208Core/50家族＋15Aux/11家族冻结，Core-only fresh8500更新/100.21 anchor等效epoch，完整恢复及有限充分性条件通过；闭环train0/10、validation0/7、新初态0/10，学生仍无完整成功。没有自动追加实验、采样、Aux模型或100例评估。

保持command-based target_after_rad、Arm7+Hand22 rad绝对目标、持续GRU、原网络/损失/权重、H24/commit12、5/60/120Hz、DDIM/overlap0与原生任务/PD/物理/DR/专家。当前不是64tick截断历史。源码未变的168/510tick等基础证据继续有效，不重开全仓审计；本轮61项定向/准入＋1成员回归通过。

旧20严格10/20与旧275混速结果独立保留。原400/275、专家/资产、开发/成功证据及失败分母保护。135拒绝RGB-D+1重复checkpoint已清理36.191GiB，不重复删除；best原字节保留，last完整恢复硬链接。当前数据/统计/cache路径与225合格载荷SHA清理后通过。

资源实况存在整卡60s59.632%及空闲VRAM4154MiB越界，不能称全程合规；遥测/失败记录保留。v1 SIGSTOP造成比例等待重复计时，已改v2原gpu.lock计算段外等待，trainer未重启、算法未变，guard已退出。不得修改无关进程。

不上传数据/checkpoint、不重写共享Git历史、不新增重复MD。后续显式任务覆盖旧阶段授权限制；本文件记录完成边界，不把数据/训练通过冒称学生成功。
