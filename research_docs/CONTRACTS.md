## 2026-11-15 held0与统一新训练实施边界

contact_height_recheck_takeover_v1允许显式min_delay_ticks=0，首个12步持续抬起/双指接触窗口完成计数N后，由action index N真实接管；自然终止优先，无固定fallback。旧60/120不改，条件消失不等同原生掉落。held_root60的rigid-root距离仍只是采集调度，不称对孔或孔口距离。

新有限loop按登记使用每轮新训练family、完整DP+后端、全部真实60Hz视觉历史；20遍B2曝光、随机合法起点与半数几何须有独立配置身份。后端继续模拟commit12真实计划，不采用每tick重新规划。GPU采集/CPU评估分别登记，不宣称两后端隐藏轨迹一致。每轮必须完整采集/训练/自主结束后才建立下一来源；失败、STOP、空间或剩余时间不足保留现场退出。实际后台运行须等主Agent观察一整轮及下一轮交接，脚本准备本身不是运行证据。

# 2026-11-15 显式随机训练时刻与轮换几何

training_sampling.json精确登记fresh_random_visit_training_v1。随机起点来自supervision_mask为真的全部60Hz动作时刻，每visit独立CPU SHA种子绑定identity/epoch/visit_index，不消费diffusion/deck RNG；数量=min(合法tick数,max(1,旧commit12有效anchor数))，无放回且排序。每轨history仍从真实tick0完整重算一次；未来truth只在loss。label尾部/控制segment合法mask不变。单轨及B2均支持；部署/后端输入准备仍12步更新，中间执行原计划。

A几何legal_tick_half_phase_v1从该visit所选anchor的commit12合法构型按(identity,tick)去重；同tick多个预测窗口仅随机选一个槽。按真实train诊断阶段人口近比例抽约半数，非空阶段至少一个，实际率明确记录且不称无偏总体均值；cap只限制每visit成本。loss是所选不同构型均值，不因抽样增多放大。phase标签仅监督采样；独立seed、每visit实际ticks、累计不同位置及阶段counts保存进日志/checkpoint。原物理容差/几何权重暂不改变，待统一实测梯度后选择。旧无profile路径用于复现。

capture-audit-memo-v1只复用同进程冻结文件完整SHA与模型契约匹配的元数据；stat/expected SHA绑定、进程变化清空、文件变化重验/拒绝。数据每case动作、history、真实相机和成功判据继续验收，缓存不是跳过新轨迹验收。pending16只增加等待校验的路径/元数据槽，不无界累积图像。

## B2恢复算术定位探针

`trajectory_recovery_arithmetic_probe_v1`只读取既有失败B gate的第一更新完整checkpoint及同一train singleton，严格绑定原失败/来源/两helper/统计与train-only缓存。默认算术同对象首跑与重复、默认新对象、确定性新对象两次，共最多5个可丢弃更新；每个分支先exact验证完整恢复边界。裁剪后/Adam前复制逐参数梯度，比较模型/EMA/Adam/RNG的exact及描述性数值差，不添加更宽通过门槛，不更改正式trainer。

必须由新进程启动前设置CUBLAS_WORKSPACE_CONFIG=:4096:8；默认分支也有这一环境差别和梯度CPU拷贝，不能称旧gate逐位重放。确定性分支显式deterministic algorithms/cuDNN true、benchmark false，unsupported即保留失败、不重试；候选算术不自动进入训练。570秒内界与外层<=600秒回收、原session/STOP同时生效；更新计时含另行列出的梯度拷贝，短探针不代表完整epoch速度。

## held_root60 持物接近当前目标根部的独立采集入口

`held_root_near_goal_then_delay_v1`沿用held12：12个已完成60Hz transition都抬高至少2cm，每tick两个120Hz样本均有至少两指物体接触力>0.1N；高度参考是completed25–48的中位数。当前transition第二个物理子步后的world rigid-root到当前目标rigid-root距离<=60mm，当前transition前后goal pose与stage完全一致，下一动作接管、delay0、无超时fallback。不是COM/keypoint距离，不加方向角门槛；角度和原生关键点误差只记诊断。真值只用于采集触发和审核，不进学生条件。

旧`near60`仍要求原生keypoint<=60mm，两个入口身份独立。当前第一目标root在孔root上方186mm，因此root接近不等于到孔口或原生near/alignment；成功标准和时钟保持。末帧刚满足却没有下一动作时不计实际接管。串行、batch和关闭HDF回放共用因果读取，逐lane独立状态。A52历史265tick回放只是候选覆盖证据，0真实新专家动作；当前冻结87577e8的128计划不变。

## 显式完整轨迹B2优化合同

`full_trajectory_batch_v1`以原登记deck相邻两个visit为一batch；同ID出现两次仍各自完整重建历史，尾单轨保留，不跨epoch拼批。逐轨损失在实际batch内取均值，paired每轨系数1/2、singleton系数1；每batch只做一次clip/AdamW/EMA，学习率和epoch数不自动缩放。因此与逐轨连续AdamW不是同一优化方法，必须同时登记episode visits、optimizer updates、EMA updates。显式profile与数据/实际multiplicity/预算写入cycle、checkpoint与finished；缺flag不能把已登记B2根静默恢复为单轨。

`full_trajectory_batch_learning_v1`每更新一行包含有序visits与局部anchors/合法H24槽/完整帧数/损失系数、deck序号和SHA，顶层保存累计visits/updates。读者按visit展开，不能把一行当一条轨迹；重复visit不是新数据。恢复只在完整epoch checkpoint，以原deck文件内容和顺序、逐轨次数、累计剂量、AdamW/EMA及全部RNG核验，未持久化尾日志归档。旧单轨格式保持。当前版本拒绝paired_episode_io_v1、已有EpisodePrefetch与visual auxiliary组合；不缓存可训练隐藏状态。多文件epoch收尾中断仍可能缺selected/validation凭据，须保全现场修复，不承诺任意位置自动恢复。

`trajectory_batch_two_training_recovery_gate_v1`只对两个train ID的三次visit作有限诊断。serial均值梯度与B2各两次更新，磁盘完整checkpoint恢复后一次singleton，最多5实际诊断optimizer更新；原source AdamW超参数、2e-5比较门槛与exact恢复检查固定。它不是正式训练候选/自主评估，实际GPU通过也不能替代完整cycle吞吐和行为验收；570秒内部工作检查须配<=600秒外层硬回收及原session截止。

## 显式B前端组件诊断

`batched_trained_B_frontend_diagnostic_v1`只加载已完成训练的B u_expert/H24/G00/every-frame best EMA，不加载rear。正式研究默认不调用；组件入口仅projection模式，使用与reachable_nominal_centered_linear_v1零提案相同的锚点与最终执行投影。每lane有独立完整历史和seed7091生成器；新controller初始种子一致，episode reset本身不重置随机流。真实submitted与applied目标分别确认，terminal步仍必须ack，真值物体/目标不进模型。

`B_frontend_projection_component_gate_v1`先用已登记train-only原始前缀进行同输入/历史/投影准入，离线明确用真实记录目标teacher forcing；线上两臂不使用该覆盖路径。两臂保留原8devspec、batch seed/event和自然成功/失败/timeout，独立Kit不是隐藏物理精确配对；所有行标diagnostic_only、非正式评估资格、非训练数据，不据无rear结果称完整学生。有限1800秒含清理、原session截止优先，失败不自动重试。

## 显式逐角色历史自主来源

`per_role_development_sources_v1`分别引用A/B原resource root、development目录和冻结源码tree。它不是新执行计划或共同运行的混合pair；只按角色读取其实际模型、case spec、wave/lane、执行配置与完成证据，并与候选该角色的合法交集比较。source-preflight无需候选完成，不生成候选计划/结果。旧单来源CLI保留；可见初态相等也不代表隐藏PhysX精确相等。

## 显式候选：独立完整轨迹合批

`independent_full_trajectory_batch_v1` 保留每条真实完整观察历史、原始每轨32帧视觉处理、独立GRU状态和原生5Hz anchors；结束轨迹不再更新，不填充虚假观察。合并去噪计算后仍按每轨原监督mask分别归一化，再等权平均；A每轨保留原最多两项几何选择，不能把整批预算缩成两项。噪声按原逐轨randn后randint顺序产生。一次合批一次optimizer/EMA更新，与连续逐轨更新不是同一优化过程，采用时必须显式新身份和剂量。已实现独立opt-in训练路径；当前冻结第九轮未启用。

`trajectory_batch_two_gradient_gate_v1` 固定两条已知train、原best EMA，同参数serial梯度平均对照B2，前向/梯度atol/rtol=2e-5、离散mask/anchors/RNG末态严格一致、0参数更新；原session截止与有限工作边界同时有效。CPU通过不等于CUDA或整轮吞吐通过。

## 2026-11-14：同session来源视图与离线批量计划的显式身份

`same_session_policy_data_selection_v1`仅分离采集可继承的数据池和已经完整自主执行的A/B来源，原checkpoint配置、统计与其nominal cache保持真实绑定。当前资源session逐字继承、截止/STOP不重置，plan中的source_selection必须在运行和聚合时重验；非法初态必须有零动作碰撞凭据，不替换或算自主失败。

`offline_planning`是前端metrics的显式配置；`cache_offline_planning_profile`是单独的后端缓存配置，两者互不隐含启用。前者先重建每条真实完整历史再批量5Hz条件；后者仍逐60Hz重算视觉/条件，之后只批量无反馈的DDIM。均保留从tick0含学生prefix的seed7091逐行(1,24,29)抽样、合法mask与29维绝对目标。后端config/index/每条payload profile一致，旧路径缺省不添加键；不同profile不能混用或悄悄resume。数值门控不意味着不同batch浮点逐位一致或模型选中epoch必相同。

阶段曝光审计的H24 target slots包括未来窗口重复和实际采样重复；不等于独立专家动作、loss权重或梯度强度。各诊断阶段mask重叠。selected与last、当前新增与源已见、当前训练与继承训练分别报告。

# 显式当前持物重查接管与并行后端候选

`contact_height_recheck_after_min_delay_v1`配置仅version、min_delay_ticks=60或120，fallback必须NEVER_FALLBACK。已完成学生transition数N满足首次12步持物后最少延时且当前12步仍持物时，才安排zero-based action N接手。高度参考仍completed25–48的中位数+2cm，双物理子步至少两指>0.1N。触发与原生成功/掉落判定独立；跨goal等待只记before/after/current及变化，不限制距离。所有训练历史来自真实学生/专家记录，动作监督仍合法专家mask。

`--parallel-backends`只改变独立A/B后端链的调度，不改变每链依赖、配置、数据或更新量。两前端完整后联合fresh telemetry准入；不足则serial fallback。两链都完成才允许development。systemd KillMode=control-group为继承的leader退出后子孙清理边界提供外部兜底，独立CLI不能凭leader退出宣称没有子孙。

`offline_contiguous_anchor_batch_v1`仅独立离线候选，不进入现有train/cache/online默认路径。输入条件由对应模型完整因果历史生成，H24/commit12/16step DDIM/no clipping/host scheduler；逐行相同seed7091的(1,24,29)噪声，包括未监督prefix/gaps。连续段按显式max_anchors切分，不能跨缺失anchor拼接；批形状浮点差须实际CUDA固定2e-5rad验收。目标度量是相同合法commit12槽，非自主成功。

# 显式训练合集阶段拟合v3

`train_only_selected_frontend_stage_fit_v3`仅由stage_fit prepare的重复`--include-collection`启用，主collection仍精确等于当前latest IDs；新增collection按immutable source SHA绑定当前train entries与真实成功接管凭据，验证family隔离与全历史/合法mask一致。source seen/unseen按源选中checkpoint实际train manifest的同SHA判定，source阶段访问数与当前selected访问数分开，不补造累计继承剂量。报告all/latest/每批/seen/unseen重叠分层，各层累加平方误差/合法槽数后求RMSE；空层为0计数/未知误差。每个模型独立完整重建记忆、同一5Hz anchor保持基线，旧v1/v2比较身份不变。

# 显式候选：自然结束环境的standard渲染停用

`inactive_render_profile=standard_inactive_render_products_v1`（完整字段由模块常量给出）必须在prepare登记，限定CPU物理/standard三视角；缺省原路径。只通过已有Hydra产品的updates_enabled切换，不创建/拆卸sensor或annotator，不改变物理、控制、reset和成功标准。每波reset前全部恢复；初态非法/未分配lane或原生自然结束且末步已提交后才能停用，禁止为了速度裁剪有效专家后段。全局finally恢复并保存清理凭据；失败不发布complete。

活动观测验证真实新drawable事件、当前host physics clock、CameraParams与renderer view一致、research/physics/time分别推进1/2/1/60秒；停用lane的camera_valid=false/time与frame=-1，图像不进入控制或训练。此为软件渲染/时钟关联证据，不是RGB/depth buffer与Hydra frame直接同一性或硬件曝光证明。诊断门控允许显式36tick内人工退休，始终diagnostic_only/accepted_for_training=false，不能传播为正式采集截断。

# 显式候选：当前观测视觉尾块

`vision_tail_profile={"version":"current_observation_exact_tail_v1"}` 只改变同一控制时刻 DINO 最后一组的计算形状：M8 不足8个活动环境时处理实际 `3N` 张图，不复制填满。缺省仍使用原固定填充合同；不隐式改变旧计划或检查点。真实三视角、60Hz观测、各环境历史、5Hz规划、29维目标及模型参数不变。正式采集、批量评估、扩展初态采集和 cycle 的评估均必须在 prepare 时登记；运行不能临时切换。

候选报告实际 backbone 核形状计数、真实/处理图数与各环境历史消费次数；非零填充或计数不符拒绝。CPU检查不代表CUDA数值或吞吐验收；须先用相同真实观测比较完整学生，再经正式流程测量。当前第八批和第八轮仍冻结原M8配置，未采用此候选。

## 2026-11-14 阶段拟合的因果保持基线

train_only_selected_frontend_stage_fit_v2沿用新增train-only投影与独立source/selected完整历史前向，在实际规划anchor取robot[0:29]的q并保持commit12；B额外取robot[58:87]的动作前执行器target保持。两基线与DP严格共用同一合法label槽；不能用每个未来控制tick刷新后的观测冒充5Hz预测。source/selected的保持值和分母必须完全一致。v1旧compare仍保留身份，不补造基线结果。这些是预测误差参照，不是已经部署过的保持策略。

## 2026-11-14 显式Tiled位姿候选与失败后渲染诊断

execution_backend_v3必须CUDA+tiled+tiled_fixed_camera_usd_sync=true，固定已审安装源码SHA，仅修改两个固定相机现有view实例的USD镜像布尔；腕部不改，class/module/shared安装不改。仅证明配置readback，不代表实际外参或画面正确。旧v1/v2保持。failure_render_calls默认0，只有max_control_ticks=1且失败control_tick0可启用至多4次sim.render；保留原坏包/异常，不调用step/reset/controller/camera.update，不替代正式60Hz观测。每次可观察物理、目标、队列、时钟、packet/frame与已初始化RNG变化则停止；不宣称完整隐藏PhysX相等。

train_only_selected_frontend_stage_fit_v1绑定来源continuation EMA与完成训练选中best EMA，分进程用各自冻结trainer、完整历史与合法专家mask做前端commit12误差。诊断几何只作重叠阶段掩码，不进神经输入；不同阶段动作可见次数不等于优化器target槽曝光。数值混合容器仅完整性SHA/mmap元数据，新增训练tensor复制至512MiB内独立投影，运行不读取heldout视觉/原始数据。此结果不测后端或自主成功。

## 2026-11-14 可扩展执行与显式策略/数据来源

当前正式batch使用正整数lane数、按同权重分波、逐lane独立历史/接管/终止/合法专家mask；每个60Hz时刻的真实三视角均进入学生。vision_microbatch显式登记，填充复制当前lane并丢弃，不增加观察或监督。原120/60/5Hz及29维目标不变；普通CPU/standard默认仍兼容，新的execution_backend_v1/v2可登记CPU或CUDA、standard或tiled、Kit/PhysX线程及v2渲染描述符容量，实际读回不等于真实物理/传感通过。没有未经证据的4环境上限。

independent_followup_binding_v3分别绑定每角色已完成自主的完整DP/后端来源与共同数据池。较大池须严格追加已准入训练纠偏、保留旧条目和验证/测试隔离、原30统计、真实numeric与完整冻结DINO缓存。纯数据池无需已经训练；已选权重的原数据/cache身份不得改写为较大池。新时段只在所有来源owner退出后创建，旧deadline/现场保持。

异步采集始终先完成真实执行、再对关闭文件进行全部源字段校验/完整hash、最后通过原控制/相机/成功审核入池。execution_complete、record_verified、accepted_for_training三者分开；软件partial与未触发不是新的成功数据。计时以完整owner总wall和实际入池轨迹/合法专家步为主，异步重叠时间不相加。

## 2026-11-14 near60 持物接近当前目标的采集合同

`held_near_goal_then_delay_v1`显式固定60mm、delay0、无fallback哨兵1,000,000,000；它只筛选采集状态，不改变原生15mm关键点成功、掉落或终止。已完成学生transition须满足原contact-height连续12tick（两物理子步均双指>0.1N，物体高于已完成25–48tick中位高2cm，允许掌中转动），且当前transition的原生最大关键点距离<=60mm。该距离对应动作前当前目标，因此目标pose与stage在transition前后必须完全一致；目标切换时拒绝触发。完成计数N合格后由下一动作index N接管，只用当前与过去信息，真值只用于调度，不进入学生条件。

串行、批量与HDF5审计通过同一`observe_transition`读取真实已完成记录；逐lane独立窗口与历史。mixed expansion使用显式`near60` schedule，与fixed40/80、contact0/12分别记账，禁止阈值或fallback漂移；旧stable/contact入口不静默改语义。原生成功、真实专家恢复、完整校验、实际入池及优化器曝光分别报告。B06历史回看选择60mm的事前证据保留为near60_design_01.json；回看的后续持物长度只解释设计，在线触发器不读取未来。实现及CPU回读通过不等于真实新批专家恢复或自主行为提升。

## 2026-11-14 latest_takeover_repeat_v1 前端访问合同

可选sampling_contract={version:latest_takeover_repeat_v1, latest_repeat:1/2/4}仅用于真实DAgger aggregate的本次新增训练episode。原30统计不重算，旧84等历史每遍一次，latest按登记次数完整重放当前权重历史；student prefix与队列flush仍无动作标签，A几何和DP监督共用原合法mask。前端每次episode访问一次optimizer更新，记录episode次数、合法anchor/目标槽，不把步份额称精确梯度份额。每个epoch的实际访问总数随repeat改变，A/B预算相同。

此合同目前仅接入前端；后端仍均匀完整episode访问，必须分别报告其实际合法步与episode曝光。无等预算对照时，不将完整新数据+接续训练的行为变化单独归因于重采样。

## 2026-11-14 可选完整周期双DP并行

`tools.tight_fresh_cycle --parallel-frontends`只重叠独立A/B前端进程；联合启动余量22GiB显存/24GiB可用RAM与新鲜遥测不足时明确降为串行。每组仍保留固定完整episode预算、独立随机流/optimizer、全真实历史和原标签mask，前端全部完成后才允许后端依赖刷新及训练。每stage记录PID启动身份/结束码/完成SHA；并行不是新数据或额外训练配额，不承诺浮点逐位等价或未测加速。

阶段审计按登记的stable或contact-height合同选择原触发器和文件；姿态窗口、接管资格、掉落及原生成功各自独立。报告兼容修复不改变收集或学生输入。

## 2026-11-14 有界Kit启动调度诊断准备

两次contact06启动均卡在仿真初始化之前，未开始任何case；独立owner保持180秒启动界限，SIGTERM无响应再回收自有进程组。资源遥测无等待原因，native main futex等待/全进程约200%CPU；gdb取栈被系统ptrace限制拒绝。去掉本次额外PXR/OPENBLAS环境设置未解决，根因尚未确认。

tools/tight_state增加显式TIGHT_KIT_TASKING_THREADS=2/4，缺省与原Kit参数字符串逐字相同；仅carb.tasking调度线程可选4，PhysX配置仍2，原任务/资产/120Hz物理/60Hz控制/5Hz规划/模型不变。请求参数与Kit实际配置readback分别写eval_startup/kit_started，后者不等于完成物理初始化。7项参数/拒绝注入测试及18项collection_owner检查通过。拟在新冻结源码/新目录、同16机会/pair/deadline下仅做一次4线程启动诊断；成功不证明饥饿是根因，也不声称隐藏轨迹精确配对。失败则保留现场做更具体调查，不遍历线程数或重置截止。

## 2026-11-14 可选接触与抬升采集触发（尚未真实执行）

`contact_height_then_delay_v1`是新的采集调度合同，不替换拿稳诊断或原生成功条件：学生连续12个真实控制transition、每个transition两个物理子步均至少两指物体接触>0.1N、物体root高度均超过已完成25–48步median z至少2cm，即可在该完成计数后0/12步接管。掌中位移和转动不作为该采集条件，但触发时仍记录原拿稳窗真值，不能把旋转中的接触称为已经拿稳。仅观察已完成学生动作，标签和专家未来状态不进触发器或学生条件。

`contact_height_only_takeover_v1`沿用原train30唯一family、完整A/B学生、无固定时刻fallback、专家实际新观察后段、队列冲刷及合法mask。新旧trigger与stage身份严格配对，原stable合同行为不变；无触发、已触发但接管前自然结束、专家救回/未救回分别计数。每次机会保存全部真实学生前缀，即使无合法标签。采集不修改任务/资产/物理/控制频率，下一轮先看完整视觉对照结果再固定来源模型和机会。

新增`--first-plan-forensics`可在新登记机会保存首tick真实RGB-D/条件/独立采样器状态与计划指纹。记录不调用额外模型、不抽随机数；CPU复制可能同步CUDA并增加首步耗时，单独披露。它不使相同seed或可见初态成为精确物理重放。当前活动视觉对照继续冻结6780cea，不热改或补录。

## 2026-11-14 视觉辅助监督对照合同（实际阶段见交接）

`visual_relative_pose_aux_v1`仅将同一动作前时刻的掌坐标物体几何中心位置与朝向前两列作为独立训练标签，严格绑定当前物体URDF、真实60Hz时钟和原始authority。学生前缀与队列过渡可提供视觉标签，动作监督仍只来自合法专家后段；两种有效mask不得互相替代。新增v7权重显式保存训练辅助头，部署不调用该头、不读真值位姿，不静默修改v6权重合同。三视角每tick真实输入、5Hz规划、最终绝对目标和原任务保持。

来源固定第五轮验证选中的完整A/B，保留原后端全部参数，不清零head。四cell为A/B各一组detached_readout控制和shared_vision候选：同随机辅助头、同共同权重、同84训练轨迹/30初态family、原30统计、同shuffle与扩散随机流、共同fresh AdamW和EMA。控制只训练辅助头，不让其梯度改变视觉；候选允许辅助梯度更新共享视觉。策略参数与辅助头分别clip norm1，避免控制被辅助头梯度裁剪间接改变。完整训练需先通过真实CUDA的条件/动作/梯度/一步参数增量对照，不放宽预先固定容差。

共同loss权重只用按排序前6训练family及其实际纠偏、A/B共享的梯度范数中位数比一次校准，目标0.1，零optimizer更新；参考是原DP base loss对共享视觉的梯度，不含几何项，不解释为每次更新恒定10%。CUDA验收另做4次一次性更新且不保存可部署权重；该验收必须绑定同份源码/校准/数据/统计和LR3e-5。正式各5遍DP、刷新90条各自当前DP计划和可训练视觉缓存，再各5遍后端及4次完整开发机会，合计16次，仍用固定6验证原动作指标选模。全部阶段在独立session完成；每个旧session deadline和产物保持。

结果须分开报告动作loss、辅助loss、固定验证动作误差/位姿读出、selected与last学习曝光、完整学生拿起/拿稳/转正/对孔/插入和吞吐。相同数据上的进一步拟合或视觉读出改善不能替代自主收益；不自动晋升候选。真实校准/CUDA验收通过，四前端已完成5遍；后端与自主结果未完整，实际收据及选模见SESSION_HANDOFF。

## 2026-11-14 输出起点对照已选择，等待独立时段启动

第五轮完整结果审查后选择下述同预算对照，来源改为第五轮选中完整pair、84训练轨迹/90缓存。每cell15后端遍、原4开发机会/组，合计16。该选择不等于启动/完成；机器回执与SESSION_HANDOFF记实际状态。原有坐标/共同head-reset/无自动晋升/来源trunk偏置合同均保持。

## 2026-11-14 可选实际上一目标锚点对照（代码准备，尚未训练）

新增显式reachable_previous_actual_target_linear_v1：后端proposal为动作前实际执行器目标robot[58:87]加可达区间span乘raw，最终仍走原生共同关节/速度投影。它不是实测q、上一网络请求或专家未来标签。名义DP完整计划、每60Hz视觉/历史、触觉与5Hz规划仍输入后端；A/B前端定义不变。原reachable_nominal_centered_linear_v1默认和旧checkpoint执行路径保留。两个映射可达范围与内部导数相同，新映射改变零输出先验，不称已确认控制错误修复。

待选实验共同继承登记来源的完整已评估后端，仅将最后policy.2的weight/bias清零，fresh optimizer，同一登记数据/精确缓存/固定6验证/相同实际曝光（最初第四轮CPU验收为83轨迹/89缓存），全部后端参数可训练。新head-reset初始化有独立合同；旧权重不能直接切锚点部署，reset后update0不算训练完成。控制组也同样reset，防止把初始化差异混入比较；继承trunk曾学习名义坐标，结论仅适用于此接续条件。当前只准备代码与CPU验收，是否运行及来源选择须依据第五轮完整DAgger结果；目前未选择或运行。

若执行，固定每cell15遍，四cell合计16个原开发机会，共享Kit并平衡A/B、control/previous执行顺序。结果必须同时报告未加权验证、已学纠偏首12/随后12/其余段、保持上一目标基线、真实命令增量及完整自主拿起/拿稳/转正/对孔/插入。开发机会已重复使用，不是未触碰终测，也不宣称隐藏物理状态精确配对。若控制组重置头后未恢复原验证水平，对这个预算的解释受限；不因低raw范数或训练误差下降自动采用候选。

## 2026-11-14 独立后续时段

用户持续研究授权允许主Agent审查完整上一阶段结果后建立新session；不得更改、续期或重启旧run。tight_fresh_followup_session只接受完整自主A/B来源与一致模型/统计/数据/cache，确认旧归属worker已退出，记录原截止与源码/来源hash；新root使用独立监控/资源文件，只读链接原资产，新采集/训练必须写新子目录。过期的旧时段不是授权阻塞，历史PID须结合当前命令/cwd核验以免复用误判。大raw和冻结视觉分片本入口绑定既有manifest SHA及当前文件stat，不宣称重新全量hash；小依赖和当前后端cache实际hash。24项边界/集成测试及round_04、control的真实只读检查通过。报告/阶段采集/session与已有trigger、DAgger合计63项CPU测试通过。

## 2026-11-14 恢复初段对照已完成与可选无fallback阶段采集

下方“尚未真实训练”是实现当时快照；实际weight1/3各A/B三遍、490更新及完整自主已完成，见最新SESSION_HANDOFF。新stable_only_takeover_v1仅明确原train30唯一family，既有因果拿稳窗后0/12控制tick触发专家；never-fallback哨兵必须大于原生完整episode保守上界，运行前逐例检查，不改任务timeout。未触发者保留完整自主轨迹、零专家标签、不进纠偏训练；触发后仍走原实际expert bridge/队列冲刷/新观察与合法mask。未拿稳、拿稳后未到切换即结束、专家救回/未救回分别记录。源码和CPU边界验收通过，不把其视为真实Kit验收。

## 2026-11-14 可选fresh恢复初段loss权重（尚未真实训练）

fresh_recovery_weighting={version: fresh_actual_takeover_onset_weighting_v1, ticks:12, backend_weight:1或3}只能显式用于fresh30_actual_takeover_aggregate_v1，固定128未加权tick窗口；拒绝与旧recovery_prefix、sampling_strata同时存在。以normalization_train_ids识别原30，其权重恒1；额外episode必须真实expert_takeover_v1，仅首个连续合法专家segment初12tick可加权，学生prefix/队列过渡仍无监督。实际曝光、episode顺序与完整历史不变；权重质量单独记账，fixed128分母保持。无此字段的历史路径不变。权重1 CPU实际trainer与无权重逐参数/梯度/顺序一致，真实完整模型对照尚未开始。

## 2026-11-14 可选批量执行合同

neural_batch只改变推理调度：每lane逐60Hz真实视觉/历史、独立episode/step/计划cursor/执行反馈/DP随机流不变；DDIM仅合并当前需要5Hz重规划的lane，后端合并当前active lane。reset lane零/invalid历史不影响继续lane，终止lane不更新。无权重/统计/标签/执行器语义变更。默认关闭，真实输入对照通过后才用于独立工程评估；批量算子数值容差不等于PhysX轨迹精确重放。

## 2026-11-14 显式后端增量初始化

新fresh aggregate可选--continue-rear，绑定其continuation.json中的previous_backend与继承的旧frontend SHA，使用全部已选后端参数且新optimizer。原30命令统计、执行域、history/loss/output与buffers不变；新前端、新数据每episode计划和可训练visual全部重算，不能复用旧参数依赖cache。默认仍为显式随机后端，旧实验不改写；准备记录random_initialization/inherited_student_parameters及来源。

## 2026-11-14 可选因果阶段接管合同

stable_grasp_then_delay_v1仅为采集调度诊断。参考高是已完成ticks25–48的median z；稳定窗12tick、每tick两个120Hz样本均至少2指object-contact >0.1N、整窗高于参考2cm、物体掌中相对位姿漂移<=1cm/0.2rad。稳定窗完成计数N，调度zero-based action N+delay，最迟登记fallback；无稳定窗或提前自然结束单独记录。它只观察已经实际执行的学生transition，不读专家后态，不影响网络输入或原生成功。

bridge仅在submit之前且未进入专家authority时允许一次向前调度。result保留fallback、首次稳定窗、实际调度及实际source switch；前缀trace的takeover_tick表示当时登记的未来schedule，可能在触发后缩短。最终训练entry.takeover_tick来自真实source切换，而非fallback。合法标签仍要求实际专家queue完全冲洗，A用真实q[t+1]，B/rear用真实专家target。初始/固定时刻接管合同不改变。

## 2026-11-14 fresh30的真实专家后段聚合

新身份fresh30_actual_takeover_aggregate_v1保留原30 affine和6验证family，额外数据必须expert_takeover_v1。所有控制步更新真实视觉/本体/触觉/命令历史，仅合法专家后段做监督。显式继续自当前DP EMA、fresh optimizer/EMA；恢复起点窗口保持原生12tick规划，队列冲刷/无效洞之前不生成标签。两组共享成功后段；后端依赖当前新DP重新缓存并同seed重训。原fresh初始和历史合同不静默改变。

## 2026-11-14 新后端与共同train30统计

新后端准备入口`tools.tight_fresh_backend`只接受明确fresh30/every-native-frame身份，保持原历史166条入口。A/B后端随机同种子初始化，共用B的train30命令normalizer，绑定各自当前DP SHA、全部逐帧缓存、数据与统计哈希；验证覆盖6个独立family。现有Transformer/每步实际反馈/当前权重前缀重算继续复用。

本轮选用已实现的可达区间中心化线性提案和span FY损失，以避免硬裁剪外零梯度及关节间极不均衡的尺度；这是工程选择，尚未证明任务收益。DP预测的完整计划仍进入后端；提案锚点限制到真实上一控制目标对应的专家可达区间，最终绝对目标只投影一次，不重复延迟或平滑。不得把锚点称为最终稳态构型。

86项归一化、输入合同、残差、span投影、中心化及tick-window测试通过。A已完成首次5遍/150更新；B同预算在运行，尚未生成新后端或完整学生闭环结论。活动DP使用独立冻结d45f11b源码，本次改动不热改worker。

## 2026-11-14 当前重启：逐帧视觉与train30

本轮授权替代旧截止、旧初始化和只启动后返回要求；当前指令集中在根AGENTS.md。研究机CNS202533182702；远端feature/tight-state-foundation核验为506d185，旧主工作区3e11d62及其未提交文档未改。新隔离树continuous_vision_20261114。

确认旧视觉每12tick一次，后端重复旧visual。新独立version6/visual_stride1合同把每60Hz真实RGB-D送入训练、在线视觉GRU和后端缓存；DP仍5Hz。旧合同保留。q_reached为transition后的q(t+1)、间隔1/60秒，不是稳定姿态；u_expert是当步实际执行目标。新模型尚未训练，不宣称任务改善。

原Core有36训练family；一家族一条中位时长完整成功，再按实际初态标准化最远点选择30条，原validation独立family取6条，test22未打开。36条共12,460tick，原文件SHA和q_after/q_before时序逐条通过；统计只用train30，未加载旧学生权重。原始数据与专家保留。详见evidence/tight_fresh_20261114/selection.json；标准化混合坐标距离只用于选取与去重，不作物理精确重放声明。

实现当前处于接口验收/全帧缓存准备。初始完整A/B训练、后端、真实多环境吞吐及DAgger研究循环尚未完成，不能把准备记为科研成功。新本地root：outputs/visuotactile/run/tight_fresh_20261114_01。

---

以下为历史记录，当前状态以上文和机器收据为准。

### 2026-11-13 22:12 有限共同曝光与冻结参考

`rear_span_dose_02`只在原run内、原绝对截止下追加共同2后端人口遍；原span一遍reference新增0更新，所有继承曝光明确区分。A/B相同数据mask/排序seed/公共处理/optimizer重启/预算/选模机会，前端和配套cache冻结且SHA绑定；无DAgger或新采集。每arm固定10同seed/family/spec机会，初态拒绝不替换；各组历史初始化不同，跨arm剂量与optimizer重启不同。真实parser和参考CPU入口只验证工程前置，candidate须真实完整训练及offline后准入；旧失败保留，不热改运行源码。固定阶段23:47:57，计算23:55:15/绝对次日00:05:15，独立cgroup到时保存退出并回交真实Agent。

### 2026-11-13 21:18 只读offline证据接续

评估恢复不得缩小注册组分母或伪造offline/result.json。复用必须complete=true、test_used=false、checkpoint SHA等于实际选定best，且state_data/nominal实际指向原验证依赖；不同既有结果拒绝覆盖。存在任何已启动trial/失败标记/部分轨迹时禁止整队重新登记执行。`rear_span_metric_eval_recovery_02`只续原40固定机会，原阶段和绝对截止不变，训练/离线推理新增0；实际入口CPU检查与真实仿真验收分开。

### 2026-11-13 20:53 显式可达比例监督metric

`ideal_box_span_fy_roundoff_v1`固定multiplier2、原1e−6rad理想化容差、regularizer0、minimum_span_rad1e−6；FY元素除以max(upper−lower,minimum_span_rad)²。旧身份和缺省路径保持。centered输出、最终执行链、normalizer和所有buffer、物理/PD/时钟/native成功不变。A仍实际q_reached+既定几何，B和两后端最终标签仍各自既定u_expert。无STE/重复delay，真实executed-MSE继续原metric。新loss写checkpoint，显式continuation允许、同attempt恢复身份必须匹配。研究`rear_span_metric_01`仅训练候选rear，参考训练只读复用；A/B准备与完成审计、test22隔离、原截止仍为硬约束。

## 2026-11-13 19:20 固定后端采样分布诊断

`rear_replay_balance_01`只训练后端，不采集、不改变冻结DP/commit12配套cache，不称DAgger。原配比与人口比例均对A/B生效，每cell同组同初始化tensor、新AdamW、共同学习率/loss/窗口/历史，1有效tick遍后固定最终endpoint评估；保留validation-best。每cell固定10相同新开发机会、共40，原生成功和自然终止，不结果驱动扩展。跨arm整episode取整/更新数不保证相等，真实账本必须披露；arm内A/B实际曝光/数据mask/公共修复/选模和机会必须匹配。frozen-frontend fairness验证冻结SHA及新DP0更新，准备合同不是完成证据。源数据/所有旧checkpoint及结果保护。阶段截止21:48:09不替代原23:55:15计算、次日00:05:15绝对截止；资源保护与test22隔离不变。

## 2026-11-13 18:45 当前contact A/B复验分母补齐

只在新`ab_replication_completion_01`运行contact A对应的20固定完整自主机会，原B contact20与previous20均不重复。完全复用B contact计划/seed/reset与评估代码，禁止修改pair、补拒绝、结果驱动扩展或混入旧10例分母；previous保持B版本诊断。训练增量公平审计启动前/完成后均保留，完成收据核对真实机会和合法分母；不匹配结果保留并标注，禁止强行通过。A事后补齐和历史曝光不等仍是局限。1h子预算、独立systemd cgroup、原计算/绝对截止、资源保护与test22隔离不变，无额外训练或采集。

## 2026-11-13 17:48 冻结完整B复验登记

contact周期A0/10、B1/10后不机械追加DAgger。`success_replication_01`固定当前/上一B完整pair各20同规格新开发机会，两批反向顺序；不训练、不更改loss/执行/物理/成功标准、不补拒绝/软件失败、不按结果增加尝试、test22隔离。普通owner只执行登记，研究解释交真实Agent。原始数据、checkpoint/cache/源码保留；新隔离树、full_gpu_v1储备不变，最长2h且不越原23:55:15计算/次日00:05:15绝对截止。

## 2026-11-13 13:12 完整tick-window结果与接触恢复新登记

本次新身份`adaptive_contact_recovery_01`仅登记一个有限完整双模块恢复实验，不改旧活动源码。固定实际tick-window完整pair采集每组12机会(58/70/82tick)，成功筛选、共享不可变池快照、DP2 anchor遍/新cache/rear1 tick遍/完整开发10次；新DP使用已审查recovery_onset_commit12_v1。Core/合法全部历史保留，后端使用centered/FY与fixed128tick，真实当前权重因果prefix，无stale-state缓存。所有原执行/标签/几何/test22约束保留。6h子截止不重置原23:55:15/次日00:05:15；不足完整阶段预留则交回真实Agent，不自动补机会或开启下一轮。

### 2026-11-13 恢复起点DP窗口（新训练显式启用）

`target_window_contract=recovery_onset_commit12_v1`是训练loss窗口身份，非模型输入或在线控制合同。设首个合法专家tick为s，额外允许a=floor(s/12)*12的原生规划点；仅s>a时新增，a..s−1全mask，s以后只取同一连续合法segment直至首个洞/段变化，最多H24。更早规划点不跨无效段获取监督。A目标仍实际q_reached/B仍u_expert，历史保持完整真实prefix；当前/未来标签不得进入条件。commit12几何分母、邻接/target数按实际valid位置，新增anchor归初段抽样。旧字段缺省行为不变；原run配置/optimizer恢复禁止改变该字段，必须新登记前端continuation。缓存仍完整t0重放、每12tick predict、逐tick旧计划cursor，后端标签mask不变。

### 2026-11-13 12:24 显式tick-window训练协议（准备）

`fixed_supervised_tick_window_v1`仅在新登记continuation启用，缺字段维持逐chunk旧行为。梯度按固定128未加权合法tick分母累积，恢复权重只进分子；达到阈值的完整32tick块或完整episode-group尾部提交优化器。尾部不缩分母、不丢监督；物理曝光与优化器更新分别记录。参数未变时使用相同权重下detached历史，参数更新后重放真实完整prefix；不是stale-hidden缓存。仍为原FY真实梯度与共同最终执行器，A/B前端标签、后端final u_expert、test22、资源及截止均不变。该试验不构成新DAgger。

### 2026-11-13 12:16 padding及输出诊断修订

无效padding槽的执行previous可以替换为合法joint lower占位，实际有效tick仍使用原58:87字段并严格校验；该占位不进入模型历史、不变更监督和控制数学。输出contract为线性/线性尾部时，旧`residual_tanh_fraction`原始字段仅是|delta|/历史scales，新汇总以`legacy_normalized_amplitude`标注，`tanh_saturation.applicable=false`；旧trace和旧默认bounded-tanh统计保持。该修订不表示原loss或真实执行梯度改变。

### 2026-11-13 10:40 最新自主科研覆盖

下文固定更新数、双门槛、候选和两轮上限均是历史实验设计，最新用户授权允许按实际证据自主修订，不作为停止研究的机械条件。当前训练与完整闭环优先完成；独立事件监督只监测和交接，真实Agent决定后续实验。原始数据/历史证据、无关任务、冻结源码、物理执行与成功标准、A=q_reached＋既定几何/B=u_expert/后端最终u_expert、test22隔离、资源储备和原计算/绝对截止仍为硬约束。参见[DAGGER_METHOD](DAGGER_METHOD.md#execution-learning-repair)及[最新交接](SESSION_HANDOFF.md)。

### 2026-11-13 自主修复补充：提案尾部与有界接续

新 `output_contract=reachable_tanh_linear_tail_v1` 独立于loss/history/执行器版本。阈值c=atanh(.95)，中心z_c=clip(z,-c,c)，rad斜率k=max((scales/action_scale)*(1-.95²), upper-lower)。delta=scales*tanh(z_c)+action_scale*k*(z-z_c)。中心区数学及Tensor输出保持旧行为；尾部是真实线性导数，没有STE。提案域不再被旧归一化幅度截断，最终joint clamp和expert_reachable_postprocessed_v1完全保留。A/B同原则；历史缺字段仍用旧tanh。旧权重可显式热启动，output身份变更须实际训练后才能作开发验证，不作为旧模型静默解释。现有loss版本允许明确记录0到1e-4的残差正则权重，其他FY常量保持；缺字段旧行为不变。

200更新是检查点，原门槛仍如实报告；允许有证据的后续800/1600更新与优化调整。新的完整数据适配来自原已训练配套pair，短探针权重不部署。完整自主仍每组10机会；必要时最多两轮定向纠偏，每轮12机会/组，均更新DP→重建其缓存→训练后端→完整学生开发。原24h绝对截止不变，任何等待/发布占用同一预算，test22隔离。所有候选与旧证据保留；开发候选没有覆盖受保护正式权重。

## 2026-11-13 执行学习修复覆盖（独立新任务）

新授权覆盖已失败lambda1阶段不再探索的限制，旧matched run及原截止继续保护。数学和实际证据集中在[DAGGER_METHOD 4.2](DAGGER_METHOD.md#execution-learning-repair)。

新增后端显式loss `ideal_box_fy_roundoff_v1`：两倍稳定分段FY、共同action_scale²与Arm/Hand等权、原合法mask/恢复初48权重3、独立1e-4 delta正则。执行v1的1e-6rad容差分支不变，loss声明理想box近似；合法标签只允许≤1e-6rad端点舍入，超界报错，真实执行MSE仍对原标签。无STE、无第二delay/减速。

历史沿用独立`current_weight_prefix_v1`，参数更新前重算实际完整prefix，prefix no_grad、仅当前32tick反传；旧checkpoint缺字段保持旧数学。output仍原有bounded tanh与容量，execution/head/history/loss身份分别记录；新增loss不套A的q_reached前端。训练/加载/恢复/离线/部署沿同一TargetResidual契约。

新任务从2026-11-13 00:05:15起最多24h，23:55:15停止计算、11月14日00:05:15绝对终止；含准备等待，不重置旧截止。只登记单一候选的A/B各旧/新200更新。双组优化门槛通过后可依据实测吞吐登记共同1/3/6有效tick遍后端适配，再每组固定10完整学生开发机会；本轮新DAgger轮/采集预算为0。旧run最新完整共同轮的前后端/数据/缓存SHA分别绑定；不拼装不同DP后端、不删有效纠偏，test22不读取。

普通后台等待原owner/worker和终局发布退出后才使用GPU/推送；数学通过不表示短训通过或闭环已完成。CPU原像分类明确区分执行不可达、极限/浮点饱和与非饱和稳定表达，不能再把raw命令超范围计数当执行不可达。

# Tight-Insertion 现行合同：成功筛选的专家接管式 DAgger

## 2026-11-12 梯度快检的显式候选身份（未正式采用）

原位支持后端 `loss_contract=exec_preclip_equal_arm_hand_v1`（lambda_pre明确数值）和独立 `history_contract=current_weight_prefix_v1`；checkpoint顶层版本与configuration不一致拒绝加载/恢复。两字段缺失完全保留旧行为。前者在所有硬裁剪前的绝对组合命令增加同命令尺度/Arm-Hand等权/mask/恢复权重监督，执行投影不改；后者按当前权重重放t0至当前块，不能跨optimizer更新复用旧hidden。固定DINO/DP缓存输入可复用，rear hidden不可当固定数据缓存。

此次A/B各原/新损失200更新探针均使用完整当前权重前缀，DP冻结仅属诊断。lambda_pre=1虽修通梯度，但两组开发执行误差及Core退化，未通过采用门槛；正式matched run维持原冻结源码、损失及历史训练近似，不更换任何pair、不重置截止。探针checkpoint均disposable，不允许普通owner加载为正式模型。数学和证据集中于[DAGGER_METHOD](DAGGER_METHOD.md#41-硬裁剪梯度快检与未采用的最小候选2026-11-12)。


## 2026-11-12 性能修订覆盖（已测试，待安全边界接入）

性能实测、范围和未完成项集中于 [DAGGER_PERFORMANCE](DAGGER_PERFORMANCE.md)。同一matched run保留绝对截止与已完成阶段；只在第2轮共同DP训练阶段允许两个独立learner并行，其后缓存、Transformer和评估顺序。第一轮A/B不在半轮改变数值设置，B继续micro4；第二轮共同micro16。每个worker保持2CPU/2线程和nice19，双DP用不同核。A第一轮已准备的缓存不动，B第一轮新准备时可用逐位等价的新缓存路径。显式 `fp32_legacy_rng_microbatch_v1` 保持effective64、原micro4 noise/timestep抽样顺序和全部监督人口，下一轮计算micro16；只能缓存不可变原始DINO/depth/时间输入，可训练投影、GRU和名义预测仍随新DP刷新。新缓存路径必须保持实际执行历史及31字段同值，真实checkpoint/data/normalizer身份继续绑定。

未来采集的 `collection.capture_compression='lzf'` 仅改无损编码；与gzip4逐值等价，184源字段SHA全部验证，不改图像分辨率、相机数或有效mask。增加磁盘开销须纳入原reserve。Transformer候选已撤回，2线程/FP32保持；无多env、AMP/TF32/compile。性能probe权重均disposable，不能部署或计入正式DAgger更新。

## 2026-11-12 新控制/配对协议覆盖

`matched_control_ab_v1` 是独立有限continuation，集中数学、算法与预算见 [DAGGER_METHOD](DAGGER_METHOD.md)。原run按已登记截止完成，新owner admission开始最多24h（含CPU准备和等待），不改旧历史。

新 `expert_reachable_postprocessed_v1` 显式绑定checkpoint，在后端训练/离线/部署共享最终命令可达投影，参考真实robot58:87 pre-action actuator target。原native-mapped目标只做一次速度处理；最终u_expert不重复减速或delay。固定float32边界容差1e-6rad，全169,853合法专家tick投影改变0，CPU端点最大漂移1.49e-8rad。A q_reached/B u_expert语义、原normalizer、native物理/PD不变；旧checkpoint缺该identity仍为legacy。

两组先在phase0从原Transformer参数适配至共同Arm.2/Hand.9及新执行器，6有效tick遍完成后才部署；只有这项显式适配允许变scales buffer，其余参数初值/normalizer沿原身份。随后最多2轮，每组48配对机会、共享快照、DP6anchor遍、rear12tick遍、固定10开发机会、共同曝光endpoint。validation-best诊断单独保留，未完成或实际曝光不配对不进入后续轮。历史初始化差异仍披露。以下为旧协议保留细节，冲突项仅对新身份由本节覆盖，不热改旧运行。

本页约束 `complete_strategy_ab_v1`：只调度A、B完整学生，每轮依次更新前端与后端；以原run的持久状态、安全边界和绝对截止推进。历史 `_02`、单轮DAgger和本run首轮的原始结果保留。实现、CPU验证、真实采集与研究效果分别报告，授权不等于验收。[实际结果](STATE_DP_RESULTS.md)。

## 范围、来源与执行顺序

配置在[tight_state.json](../configs/tight_state.json)的`dagger`，底层池版本仍`successful_takeover_pool_v1`，范围修订为`complete_strategy_ab_v1`。

|完整策略/内部来源标识|实际前端模态|前端训练标签|后端与最终标签|
|---|---|---|---|
|A / QG01|RGB-D＋robot178，G00模态，无前端直接触觉|q_reached，保留手部软几何|756,581参数Transformer；组合最终u_expert|
|B / G01|相同G00模态|u_expert|同容量Transformer；组合最终u_expert|

A从本run首轮选中前端804更新、后端12212更新的完整pair继续；B从既有G00/G01有效完整pair继续（前端SHA`4356acbc…`、后端SHA`0e9ec397…`，完整身份在机器索引）。G11此前引入前端直接触觉，现退出活动队列；其754次持久化前端更新、选定201更新、全部模型/合法数据和历史结果保留。裸Q/QG前端只训练/离线预测验证，不独立DAgger采集或任务成功率考核；不启第三条完整多轮队列，G10辅助基线也无默认新增预算。

共享监督映射为G00/G10类→u_expert，Q00/QG00类→q_reached，所有后端→组合后的u_expert。由显式`supervision`选择字段、normalizer与损失，不按29维形状或临时组名判断。两组前端模态、历史、频率和后端容量匹配；保留已训练残差归一化幅度A Arm0.2/Hand0.9、B全0.1及各自历史曝光，不能声称严格单变量或同预算对照。

每轮顺序：完整配对学生采集→成功专家有效后缀入持续池→冻结快照→微调现有U-Net可训练部分→联合开发选前端→固定新前端并刷新所有依赖缓存→从原Transformer继续学习最终命令→配套自主评估/回退。冻结DP只用于本轮后端训练。数据不足须报告，禁止仅训后端后宣称完成两部分DAgger；没有端到端物理rollout/BOL/PPO新增目标。

必须核验checkpoint SHA、真实训练来源、前后端配对和各自统计。不可部署disposable或随机update0；选中update0须有已训练源权重SHA链。旧命令v4可显式声明为v5 u_expert而保持参数/统计逐tensor不变，不把Q模型重新转作普通命令。新round用新优化器/日程，同round恢复完整状态。

## 有限调度与一次条件追加

仍用`outputs/visuotactile/run/tight_dagger_pool_20261111_01`，原起点2026-11-11 13:24:31不变。第2轮按原配置完成：A最多96/B最多48真实机会，共同快照，分别更新DP、缓存、Transformer并进行纯学生评估。追加准备位于隔离工作树，不能热改其冻结运行源码。

仅当第2轮A/B各自完整新candidate的已分配纯学生评估全部实际完成、合法且无软件错误、配对身份正确，并且两组完整成功都为0，才追加第3轮。原4例→满足预登记进展条件再扩10例的协议保持；未启动的未分配6例不是失败。任一组有成功即不追加；缺失、不完整、非法或软件错误不能替代零成功凭据。发布器只呈现owner判定，不自主触发或选模。

- 追加最多一轮，第3轮两组各最多96真实机会，来源必须是本轮新candidate配套前后端，不能因旧开发回退规则继续用旧学生。每组记录checkpoint与normalizer身份；只改变下一轮采集选择，不改写第2轮旧选择证据。
- Core与全部合格旧/新纠偏持续累积，两组同轮使用共同快照。采集量按剩余时间与两组完整训练预留裁减，192机会是上限，不承诺全部完成，不用补seed凑成功。
- 前端仍增量最大800更新，400后可按验证趋势停止，每200验证、保留共同曝光checkpoint；几何与配置/命令监督分别保留。新DP刷新全部依赖缓存后，后端最多12/最少6有效监督tick遍，3遍验证。不得取消任一模块更新。
- 仅在完整零成功条件触发时，原绝对截止2026-11-12 07:24:31最多延长21600秒至13:24:31，计算13:14:31停止；原起点不变，不按实际后继启动再加6小时。追加后无论结果如何退出，无递归或新run预算。
- `conditional_extension_request.json`为用户条件授权；`conditional_extension.json`及可选独立decision记录实际评估门控与新pair、旧/新截止；status/service记录后继启动阶段。`started=true`只表示后继进程启动，真实采集还需实际首批轨迹验收；任何一层不能替代下一层凭据。无触发时未调度轮次不计失败。

增加真实恢复覆盖属于有限探索，不能仅凭零成功断言数据不足；A/B历史曝光与残差幅度差异仍需披露。所有新评估仍是开发集，test22不读取。

选模仍包含真实update0，旧validation/纠偏dev/最初24合法tick分别相对其update0归一化后按.5/.25/.25合成。恢复训练强化的48tick与此24tick开发指标分别记名，不混为同一人口。test22不使用。

## 原生执行与专家接管

H24/commit12、sample/x0、DDIM16/seed7091、overlap0；5Hz规划、60Hz控制、120Hz物理。动作是 canonical Arm7+Hand22 的29D绝对关节目标 rad，经原关节限位和位置执行接口提交；不把归一化原生 expert action 直接当 rad。原资产、PD、物理、任务成功、timeout、reset 与 DR 不改。

每次从登记训练家族初态运行学生。专家 recurrent shadow 在每个真实观察上连续推进；按冻结的本轮计划接管：新计划switch=12k−2，原3项delay队列清洁后缀从12k规划anchor开始；默认switch为46/70/94/118/154/190/238/298tick，后续仅按登记有限规则调整，不在终止后接管、不恢复理想姿态或拼接 reset。专家动作经原生映射后施加已有 Arm=.5、Hand=1、Hand目标变化上限2rad/s；学生目标不再重复这一限速。两者通过同一位置执行器，并记录两个物理子步实际目标。

专家独立建议队列从学生前缀开始更新，复用位置执行器原有的一次 delay RNG draw，不额外抽样或修改共享 native queue。接管时全队列尚含学生控制期建议的过渡段不监督；干净专家实际执行后缀才有有效标签。连续 shadow、映射和时序需新阶段真实采集验收，CPU/mock 通过不等于物理接管通过。

采集入口必须实际加载配套DP＋Transformer并运行短验证，记录两个checkpoint SHA、nominal、residual、组合issued及真实applied。缺后端、SHA/合同不配或加载失败显式报错，禁止fallback到裸DP；不要求每一步修正非零。已发生轨迹按真实日志核验，路径原本正确则报告“未发现裸DP采集问题”，不宣称修复不存在的bug；若有受影响记录只隔离对应范围。验收及独立起点/覆盖计数嵌入正常采集，不增加长诊断队列。

只有原生完整成功的接管 episode、且内容/时序/标签有效，才能加入成功纠正主池。学生提前自主成功、接管失败、超时和软件中断各自保留分母；它们不自动变为专家训练数据。不会为了凑成功补 seed 或扩大机会。

## 记录、标签与历史的分离

标准 HDF5 `transition/target_after_rad`、`diagnostics_before/after`、issued/applied 和观测时钟保留实际执行。额外 `transition/dagger/` 包含专家 raw/clipped/delayed action、来源 tick、control_source、segment_id、expert_label_valid、switch_clean、previous_issued、previous_nominal/valid、u_expert、q_reached。原始 q_after 可保存全段；它只有在专家真正执行且 mask 有效时才是专家配置监督。

|共享字段或语义|来源与用途|约束|
|---|---|---|
|`robot` / `command`|完整真实 q/qd、目标和过去实际执行；command保持实际 target_after|学生前缀也真实保留；不被零标签或重预测替代|
|`clean_sample`|前端按 supervision 类型选 u_expert 或 q_reached|只用于加噪 clean sample 和同语义目标/FK loss|
|`expert_command`|后端最终命令标签 u_expert|不因前端采用 q_reached 而改变|
|`supervision_mask[T]`|映射 expert_label_valid；原 Core 全 true|学生/切换过渡为 false；是 loss/exposure mask|
|`control_segment[T]`|映射 segment_id|控制权或有效性切换递增；H24不能跨段|
|`inputs.valid`|真实历史行存在位|保留学生前缀；不能与监督 mask 混用|
|`previous_nominal`|上一 tick 的名义预测|与严格过去 issued/applied 独立；首tick无效|
|`loss_context`|真实后态物体/接触等几何监督|独立文件；不进入学生 encoder、接管触发或成功 gate|

`u_expert` 是专家实际最终执行目标；`q_reached` 是同一专家 transition 积分后的真实配置。学生执行后的 q_after 不能作为专家建议的反事实配置，也不能从原示范“相同 tick”取标签覆盖偏离状态。

规划 anchor 是12的倍数。H24 从 anchor 当前 transition 开始，遇首次无效、segment变化或 episode尾部即截断余下标签；填充为零且 mask=false。训练只抽有专家监督的 anchor，曝光按实际有效人口计。观察历史可跨接管边界，GRU仍从真正 episode起点推进；96tick TBPTT或后端32tick detach只限制梯度，不重置记忆。未来观察/动作只作允许的标签，不作策略输入。

## 聚合、统计与缓存

原Core train166加两组/各轮共享成功纠偏train；已有21条旧纠偏均保留为历史train，新采集的correction_dev在collection_plan整episode预登记并永久隔离。validation30和test22身份保持，test22 本轮不读取轨迹、不评估、不选模。原 Core 的55,392训练tick、4,690规划anchor是旧人口；聚合后的分母由实际 mask 重新计数，不能用历史 epoch 或学生前缀冒充新增监督。

`normalization_train_ids` 固定为原训练人口；`statistics.pt` 与 G10 的 `tactile_statistics.pt` 按原字节复用。Q 前端保持 q affine；G 前端保持 u affine；所有后端保持共同 `u_expert_train166_affine_v1`。不从纠正数据/validation/test重新拟合。G10六个、G00四个观测/动作统计 buffer 在同语义续训时逐 tensor 校验。

池的全局身份包括run/group/round/source与内容SHA；同内容去重，禁止改变既定cohort。每轮snapshot冻结之后再训练，同轮实际调度组读同一快照。本轮层必须同时匹配run_id与round_index，不能把旧run相同轮号混入本轮。采样按Core .5、历史纠偏 .2、本轮纠偏 .3，空纠偏层权重移到另一纠偏层；DP按有效anchor，后端按有效监督tick分配完整episode。记录实际各层比例、重复样本、唯一episode/anchor/tick覆盖与等效曝光，不把重复抽样称为新增数据。

恢复初段从首个合法专家tick起最多48个连续有效tick，遇mask无效或控制segment变化立即截断，不跳过缺口续数；学生和队列过渡不能补入。前端在既有纠偏层内将约一半有效anchor抽样分配到初段，其余仍覆盖全部合法后缀；后端对合法初段最终命令loss施加3倍权重，归一化与实际加权曝光公开。原Core/历史/本轮.5/.2/.3混合层保持。记录原始独立接管起点、来源pair版本、训练家族、实际接管阶段/偏离描述、初段合法tick/anchor人口以及实际采样曝光；重复采样不计新恢复案例，阶段/偏离统计不解释为已证明根因。

前端保持现有 Conditional U-Net 和持续历史；DINO backbone 冻结，既有视觉投影/深度/历史/策略参数可训练。每轮先恢复这些前端参数参与微调，再固定本轮选定前端训练后端；不能把整个 DAgger 轮次称为“冻结前端”。前后端分别优化，没有端到端联合反传。

后端仍为 `target_residual_transformer60_v1`：756,581参数，2层d128/4头/FF256，过去32tick与持续GRU、当前H24计划/cursor/mask和独立上一名义输出。无失触立即清零门，既定残差幅度与关节限位不变。报告所需修正超界比例、实际幅度/饱和和观测失触→再接触；后者不证明已学会恢复。

新前端选定后，对全部原Core/纠正train、validation和correction_dev重放完整实际历史，重新计算持续隐藏态、视觉读出、H24/cursor/mask及 previous_nominal。不能复用旧 DP 的预测或可训练表征。每个后端缓存绑定 `base_checkpoint_sha256/statistics_sha256/data_sha256/residual_statistics_sha256`，index另绑定实际监督mask+segment SHA。只允许复用源轨迹、backbone与内容SHA都一致的冻结DINO特征；新纠正图像重新提特征。

<a id="qg-loss-details"></a>

## QG 数学目标、量纲与梯度

以下是已完成离线 QG00 与本轮 DAgger QG01 前端实际沿用的 `hand_surface_soft_hinge_v1` 目标。Q00 使用相同配置监督但没有手部几何项；G00/G10 保持命令监督。`loss_id=palm_command_sample_commit12` 是历史标识字符串，不能据此把 QG 的真实 `q_reached` 标签解释成 `u_expert`。实际标签由合同中的 `supervision/clean_sample_field/normalization_identity` 决定。

令 $q^*$ 为有效专家 transition 完成后的配置（rad），$z^*=s_q\odot q^*+b_q$ 为其固定 train166 affine，$\hat z$ 为 U-Net 对加噪 $z^*$ 的 sample/x0 预测，$\hat q=(\hat z-b_q)/s_q$。clean sample、加噪输入和拟合目标都使用这一配置语义。基础拟合覆盖全部有效 H24 目标：

$$
L_{\rm fit}=\frac12\operatorname{mean}_{V,\,j\in Arm7}(\hat z_j-z_j^*)^2
+\frac12\operatorname{mean}_{V,\,j\in Hand22}(\hat z_j-z_j^*)^2.
$$

$V$ 排除 padding、学生前缀和切换过渡；窗口在首次无效监督或控制段边界处截断。两个关节组各占一半，不能按29个关节直接混合求均值。掌部项仅覆盖 $C=V\cap\{0,\ldots,11\}$，用还原后的真实弧度做 FK：

$$
L_{\rm pos}=\operatorname{mean}_{C}\|p(\hat q)-p(q^*)\|_2^2,\qquad
L_{\rm rot}=\frac12\operatorname{mean}_{C}\|R(\hat q)-R(q^*)\|_F^2.
$$

位置项单位是 **m²**，旋转矩阵项无量纲，不能把毫米或角度误差直接代入这些训练项。这里的掌部位姿是预测关节配置的 FK，不是实际自主轨迹偏离。代码见 [sample/x0 与掌部目标](../visuotactile/state_image_loss.py#L59)、[Arm/Hand 等权拟合](../visuotactile/dp_objectives.py#L8)、[有效窗口](../visuotactile/anchor_sampling.py#L371)。

对抽中的配置，$d_c$ 是手部碰撞表面与物体之间的有符号距离（m，分离为正、穿透为负），$d_f$ 是指定手指实际末节/弹性体碰撞表面的最小距离，$m_f$ 是要求接触的五指 mask。令 $[x]_+=\max(x,0)$：

$$
\ell_{\rm geom}=
\operatorname{mean}_{c}\left(\frac{[-d_c-0.001]_+}{0.001}\right)^2
+\frac{\sum_f m_f\left(\frac{[d_f-0.002]_+}{0.002}\right)^2}{\max(1,\sum_fm_f)}.
$$

穿透容许1mm、要求接触的表面间距容许2mm；平方 hinge 除以各自容差平方，因此几何项与其乘子均无量纲。接触安排来自同 transition 最后120Hz采样的 object-filtered force，范数超过0.1N的指头才有 $m_f=1$。接近、释放及未要求接触的指头没有吸附项。after-state 物体位姿、接触 mask 和阶段只进入 loss，不进入策略观测；它们与 `q_reached` 对齐。[软几何实现](../visuotactile/state_image_loss.py#L162) · [几何标签时序](../visuotactile/state_image_loss.py#L354)

每个完整 optimizer update 最多抽两个合法 commit 配置，有候选时接触/非接触各一个，否则从剩余候选补齐。$L_{\rm geom}$ 是这两个配置的等权均值；分母在整个 update 内固定，各微批直接累加贡献，不能误乘成每微批两个配置。该分层抽样均值**不是整个目标人口的无偏估计**；基础拟合仍使用全部合法目标，困难几何样本不按可行性剔除。[抽样](../visuotactile/anchor_continue.py#L125) · [全 update 归一化](../visuotactile/state_image_loss.py#L113) · [微批累加](../visuotactile/anchor_sampling.py#L319)

实际总损失与权重为：

$$
L=L_{\rm fit}+g_p(k)\,[0.3342396121814294L_{\rm pos}
+0.029946394754032265L_{\rm rot}]
+g_g(k)\,2.355370821894758\times10^{-6}L_{\rm geom},
$$

其中 $k$ 是本训练阶段从1开始的 optimizer update，$g_p(k)=\min(1,k/500)$，$g_g(k)=\operatorname{clip}((k-100)/400,0,1)$。前100次更新跳过几何距离计算，第101次开始渐入，第500次达到全权重；掌部权重独立渐入，手部几何不会再重复乘 $g_p$。位置权重对应上面的 m² 标度。[实际训练累加与调度](../visuotactile/anchor_continue.py#L717)

几何权重经过训练集六状态检查和**一次完整 update**校准：基础手部输出梯度范数0.001015357876518895，未加权几何手部输出梯度范数86.21639251708984，最终加权比0.2，手臂几何输出梯度为0。这个0.2是该次校准的输出梯度比，既不是模型参数梯度比，也不是后续每次更新的保证；没有使用 validation/test 调权重。[原始软几何验收](evidence/tight_reuse_20261110/geometry_soft_acceptance.json) · [吞吐与拟合短测](evidence/tight_reuse_20261110/training_readiness.json)

距离计算复用 `HandCollisionGeometry.distances()` 的局部手关节 Jacobian；碰撞 witness 不反向传播，arm/object 上下文 detach，几何梯度只接到输出的 Hand22。共享网络参数会收到这些手部梯度，不能声称共享参数更新绝不会影响手臂预测。当前正式训练不调用 SLSQP。此前 `projection_v1` 则是求局部可行投影后回归投影手部目标，只有可行样本贡献；两者数学目标与权重单位不同，不能把旧投影权重直接沿用或混称同一方法。精确投影代码只留诊断。**内部力项未启用**，软惩罚也不保证几何可行或任务成功。

几何训练日志记录的是抽样 x0 输出在记录物体场景下的距离与局部梯度；`maximum_penetration_violation_m` 是**超过1mm后的违约量**，绝对穿透深度应取 $\max(0,-\min_c d_c)$。离线几何诊断使用专家观测前缀下的完整 DDIM 预测。两者都不能冒充自主 rollout 的实际物理穿透或失败因果证据。

## 后端目标与验证选择

所有 Transformer 后端都监督最终 `u_expert`，包括接在 Q00/QG00 后面的 Q01/QG01。设共同命令 affine 的 scale 为 $s_u$，$\hat u$ 为限位后的组合命令，$\Delta$ 为归一化命令空间残差，则每个合法监督 tick：

$$
\ell_{\rm back}=\frac12\operatorname{mean}_{Arm7}[s_u\odot(\hat u-u^*)]^2
+\frac12\operatorname{mean}_{Hand22}[s_u\odot(\hat u-u^*)]^2
+10^{-4}\left(\frac12\operatorname{mean}_{Arm7}\Delta^2+\frac12\operatorname{mean}_{Hand22}\Delta^2\right).
$$

当前恢复强化目标为每个optimizer update内的加权均值 $L_{\rm back}=\sum_t w_t\ell_t/\max(1,\sum_t w_t)$：纠偏首段连续48tick取 $w_t=3$，其余合法监督取1，无效监督取0；原未启用恢复配置的路径仍为普通均值。实际物理tick曝光按Boolean mask计数，权重质量不能冒充额外专家tick。接触/非接触都参与；学生前缀仍更新历史但不监督。后端没有新增 QG 几何项，也不把命令 affine 换成前端配置 affine。组合为 `base + delta / action_scale` 后施加关节限位，残差幅度沿用既有校准。[组合与后端目标](../visuotactile/models/native_dp.py#L549)

前端按固定 validation4、真实完整前缀、部署 batch1 DDIM16/seed7091 的 commit12 误差选择最低分 EMA：

$$
S=\frac14\left[(\mathrm{ArmRMSE}_{rad}/0.01)^2+
(\mathrm{HandRMSE}_{rad}/0.1)^2+
(\mathrm{PalmRMSE}_{mm}/10)^2+
(\mathrm{MeanAngle}_{deg}/5)^2\right].
$$

四项物理尺度固定，不是训练loss或几何可行率。H24全长误差另报。前端旧验证、纠偏dev及其每episode最初24个有效commit目标沿用同一物理分数；后端使用命令归一化Arm/Hand等权MSE，不加残差正则。学生前缀与跨控制段窗口均不计误差，仍完整推进历史。

两阶段都真实评估update0；三部分各自除以对应update0分数（数值分母下限1e-12），按旧validation .5、correction_dev .25、纠偏初24 .25加权。较差的训练checkpoint可以全部不入选；选中update0不表示未进行训练，正式完成凭据另报告真实新增更新。保存baseline、各项原分数/归一化比、best和last。后端checkpoint始终绑定产生其缓存的选定前端SHA。新pair若自主表现劣化，回退上轮完整配对；不能任意拼装两个独立best。

## 有限预算、停止与自主评估

|项目|本次登记|
|---|---|
|round/总时限|原root从第2轮接续；2026-11-12 07:24:31截止，计算07:14:31结束；resume不重设deadline|
|采集机会|第2轮A最多96/B最多48，第3轮A最多96；按剩余完整阶段预留裁减，整episode dev预登记，无失败补seed|
|最低新增监督|成功纠偏episode、至少48有效专家tick和4个有效DP anchor；新增数据不足如实保留原pair，不偷用dev或退为仅后端|
|前端|固定增量最多800更新，400后可验证停止，每200验证；LR峰值1e-5；共同400/800 checkpoint；实际anchor曝光另报|
|后端|最多12有效监督tick等效遍，6后可验证停止，每3遍验证；LR3e-4；每组每轮最多.75h；完整episode组取整的超出tick明示|
|开发自主评估|先固定4例；>=1原生成功、>=1子目标或>=2原生lift时才按同一登记队列扩至10；每组1视频；拒绝/中断不补，始终无专家替换|
|资源|充分使用空闲GPU；至少6144MiB空闲显存、14GiB RAM、60GiB磁盘；只管理自有进程|

正式LR日程先固定，短探针只提前结束且明确disposable；不把少量验证当作收敛。DP与backend使用各自新增更新/有效曝光；重复轮次不重做已有SHA凭据完整的阶段。0自主成功本身不取消已授权有限下一轮；数据错误、持续工程错误、资源/绝对时限仍会停止。旧结果可在相同模型SHA、协议和初态spec下引用，不能重跑挑选有利结果。接管成功、自主成功和软件有效分母分开，早停理由及真实耗时均公开。

## 复用的模块与新增职责

|职责|原位复用/新增|本轮变化|
|---|---|---|
|配置与入口|现有 `configs/tight_state.json`、`tools.tight_finish`|增加 dagger 配置与子命令，不按组复制脚本|
|后继协调|新增小模块 `tools.tight_dagger`|持久共享池/每轮snapshot、登记/锁/阶段、有限决策和恢复凭据；不实现模型或优化循环|
|采集/动作|现有 `state_benchmark.trial`、`position_control`、recording/compact|opt-in expert bridge和同episode接管字段|
|数据与缓存|现有 `narrow_data/native_data/anchor_sampling`|同loader聚合、两监督类型、mask/segment、DINO共享与新DP缓存绑定|
|前端与后端|现有 `state_history/native_dp/native_controller`|严格同语义EMA/后端参数续训；网络种类不变|
|优化/评估/发布|现有 `anchor_continue`、`tight_finish`、`tight_diagnostics`|有效曝光与恢复、无接管评估、当前结果汇总|

官方 Diffusion Policy 的 image workspace 分别构造 policy、dataset/validation、normalizer、optimizer/EMA 和 env runner；本项目据此保持数据、模型、训练循环与运行协调的职责分离，没有迁移其框架或套用其任务动作语义。[官方 workspace 源码](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py)

robomimic 的 SequenceDataset 将 HDF5 观察键、动作键、序列长度、缓存与padding mask作为数据层参数；这里沿用统一 episode 读取和显式 mask，额外保证接管控制段不被H24跨越。[官方 SequenceDataset](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/dataset.py)

imitation 将交互收集、轨迹保存与后续 BC 训练分离；其 InteractiveTrajectoryCollector 可在学生实际执行时仍保存专家建议。本轮只监督成功接管后的专家实际执行后缀，明确是成功筛选的接管变体，不声称与标准全学生状态标注 DAgger 等价。[官方算法说明](https://imitation.readthedocs.io/en/latest/algorithms/dagger.html) · [官方 collector/trainer 源码](https://imitation.readthedocs.io/en/latest/_modules/imitation/algorithms/dagger.html)

## 接续、发布与清理

准备在隔离 `bc+rl_RSS_qg_report`，旧树/旧产物不动。普通owner核验源冻结、原配对SHA与自有worker实际退出后推进。恢复命令：`bash visuotactile/run_cpu.sh -m tools.tight_finish dagger_resume --root <root>`；完整阶段按凭据/SHA跳过，中断训练从last的optimizer/EMA/RNG/deck/cursor恢复。STOP只请求本任务在optimizer或完整episode组边界保存；未持久化日志保留为`*.uncommitted.jsonl`，不重复纳入正式计数。恢复不重设总deadline。注册/状态/真实进程共同证明启动，仅有脚本或CPU测试不构成GPU成果；不得由LLM持续轮询。

只推 `origin/feature/tight-state-foundation`；新小型结果写现有文档，大数据/权重/视频留本地。按实际依赖清理不再使用的旧代码、checkpoint、缓存和报告，记录文件数与实际释放量；保护原400/仍用Core、专家/资产、活动/排队依赖、选定及恢复模型和必要比较结果。不能整树删除仍被manifest引用的历史run，不通过归档搬家冒充删除。

当前已完成的CPU数据与接口验收见 [SESSION_HANDOFF](SESSION_HANDOFF.md)。它不构成真实接管成功、正式训练完成或自主成功证据。
# 2026-11-13 11:25 中心化提案显式契约

`reachable_nominal_centered_linear_v1`仅为opt-in后端提案：实际前动作target定义理想可达box，投影nominal为锚点、box宽度乘raw；最终joint clamp/position adapter不变。原affine/scales缓冲及参数保留，但新输出不再以旧scales限制提案；历史幅度比不解释为tanh饱和。FY reg0保留真实非平坦梯度；非法/不可达标签仍拒绝，学生前缀不监督，test22不读。`adaptive_centered_01`只诊断后端，非DAgger，无采集、无前端更新；完整DP＋Transformer仍用于所有任务评估。新源码在隔离树，旧worker/supervisor冻结保护，原deadline不重置。
## 2026-11-14 显式批量执行候选

`whole_wave_complete_student_v1`将同完整权重的机会按role分组、整批自然执行，环境数为正整数且由实际资源/吞吐验收选择；默认旧串行路径保持。每lane的DP噪声生成器、全部持续历史、专家队列、接管触发和输出目录独立；本批结束前不复用已经自然终止的lane。子目标不清空历史。每个active lane在每60Hz tick消费自己的真实三视角RGB-D，DP仍每12tick规划。自然结束后停止该lane记录，不将自动reset图像附给终帧。原生事件随机流为本批所有lane共享的向量化流，不声称不同N或后端隐藏状态逐位配对。

`batched_expert_takeover_position_wave_v1`只包装原单lane桥，复用同一次原生N维delay draw；逐lane的实际专家建议队列、source-step、source/valid segment和完整queue flush保持原语义。inactive lane保持其实际当前原生目标，`bridge_active=False`且无专家标签，不进入旧episode。学生在专家后段仍按新真实观测处理历史，实际issued/applied写回下一tick；学生前缀始终mask0。成功、触发、专家救回、相机/标签验收及入训练分别统计。

新`SharedCaptureWriter`由唯一写线程拥有可写HDF5句柄，主线程提交独立CPU快照；共同字节上限覆盖队列、正在写入和保留的行缓冲，压缩临时工作区另列。`verification_workers=0`保留原线程内完整验证；显式登记`2`及`verification_max_pending=4`时，`shared_async_capture_v2`先关闭该lane全部文件，仅把路径与小型凭据交给有界spawn进程池。上限包含排队及正在验证的任务，不跨进程传RGB数组；各验证进程独立持有只读HDF5锁，逐块解码内存另计，不纳入512MiB写入payload上限。默认和正式计划的参数均校验，不向已运行批次热替换。

验证子进程继续调用原`verify_stream_fields`重建并比对全部源字段SHA（batch07为185项），再计算完整compact文件SHA；验证成功后才原子发布`capture_complete.json`并完成ticket。`verification_queued.json`、`verification_started.json`保留关闭文件路径和真实验证PID；文件关闭、验证完成、相机/控制/动作来源验收、最终入训练是不同阶段。任一child异常同时传到ticket、writer.check和close，唤醒容量等待，保留失败输入及`capture_error.json`；正常close等待并回收进程。正式`python -m tools.tight_fresh_batch_run`的Kit入口处于`__main__`保护内，spawn的`__mp_main__`导入不启动Kit；此启动路径已做CPU检查，实际Kit吞吐仍须新批次验收。

`tight_batch_owner`在启动时绑定自己创建的worker PID/start_ticks/PGID/SID/UID；即使native异常使leader先退出，finally仍清理同一有据可查的组内存活后代并保存`owner_group_cleanup.json`。检测到PID重用、session/UID不一致即拒绝发信号。旧session、冻结worker与其他owner的通用停止行为保持。全部源字段重建SHA/相机内容/控制来源验收仍是入训练前置；软件中断保留partial和完成case清单，不把异步提交返回当完成落盘。吞吐以完整batch含启动、drain、验收与退出的墙钟为准；lane/写入/验证计时重叠不能相加，短CPU测试不是实际采集加速证据。

仿真候选另登记`execution_backend`，参见[后端合同](EXECUTION_BACKEND_SCALING.md)。上述为实现合同；真实Kit与完整研究周期是否通过必须读新run结果，不能由CPU测试推断。

`preregistered_training_families_v1`在结果发生前固定新增train family、seed及float32请求初态hash，明确排除已有和原验证/测试配置；不代表扩大质量/摩擦域。`registered_training_expansion_takeover_v1`允许事前明确的fixed40/80、contact0/12或near60机会；固定时刻、接触触发与持物接近目标分开审核，真实学生前缀没有专家标签。registry、初态hash和来源身份贯穿metadata/entry/aggregate；缺省未知family仍拒绝。六验证family、test22、原统计不变；不读取测试轨迹。固定接管也必须专家实际救回及完整来源/传感验收，不能将登记机会算成新数据。

`frozen_episode_prefetch_v1`只对冻结CPU原特征做有界精确clone，不缓存可学习GRU；每次完整历史按当前权重重算。训练前显式登记，已开始不可热替换；吞吐必须含等待及验证，不能把重叠预取和计算相加。短吞吐探针使用独立engineering身份，控制上限截断单列，不计自然失败、成功或新增轨迹。
