# Tight-Insertion command StateDP 当前合同

2026-11-07 用户授权完整执行基础链路整改、必要慢速重采、随机初始化训练、闭环评估和分类清理。验收通过后自动进入下一阶段，历史“等待授权/审计后停止”不再是当前限制。任何真实门控失败必须保留证据并阻断其依赖阶段。审查基点642f5a2，当前入口与实际状态见[README](README.md)和[结果](STATE_DP_RESULTS.md)。

## 本轮：Hand限速最小验证与独立StateDP推进（基点1b70d3b6fce8052300a54f5361f4092da27cba5f）

本轮A指Hand限速，B指StateDP，与下方上一轮A/B编号不同。运行root为`outputs/visuotactile/run/tight_state_20261107_01/hand_cap_04`，仍使用tools.tight_finish监督入口；此前工程验收保持，新增33项受影响回归通过。登记并非实际完成，结果以root下summary/status为准。

A先读取B00/B01/B02既有记录，用实际60Hz前后状态计算Arm/Hand平均每关节路径、物理palm末端和物体的平移/旋转路径。速度定义path/阶段时间，按T=L/v作对称秒数分解；这是描述性归因，不是固定路径因果实验，也不恢复120Hz内部振荡。旧B00两次插入物体路径3.29/3.88倍，均速1.01/.93倍；不能把旧时长合格等同于无绕路慢速。

唯一候选Arm.50/Hand1+Hand各关节2rad/s目标限速，只在采集native分支、原delay/EMA/限位后应用一次；None关闭返回原target对象，学生已处理位置路径直接绕过。原专家、PD、物理、DR、限位和成功/timeout不变。2rad/s依据6原速参照主要活动关节P90约3–5rad/s而低运动轴多<.32rad/s；不按家族拟合，不增加第二候选。

预登记6次新慢速（B00/B01/B02各2个已有initial/event seed，复用6原速证据），1h pilot窗。完整成功≥5/6、加强后的逐条准入≥4/6且每家族≥1才算改善。原逐条完整成功、真实总时长1.5–3、各实际运动阶段和三模态门槛保持；额外要求Hand整体/插入、palm和object插入平均实际速度≤原速.90，三者插入路径≤1.25。25%为预先声明的保守筛查阈值，防止旧2–4倍绕行；不要求每阶段都慢1.5–3。没有改善则停止恒定alpha/Hand cap搜索。

只有pilot通过才启动事先登记的4个独立开发家族×3event（12慢+12原速），固定同一参数：慢速成功≥11/12、全部准入≥10/12、每家族≥2成功且≥2准入，原速成功≥11/12；阈值不可达即停，不替换seed。通过才进入既有正式生产/全量准入/冻结/新StateDP训练合同。机制开发总窗3h，原root不重置；新root后续生产/训练使用启动时登记的36h总界限与资源储备，不把到期叫收敛。

B决定本轮不随机重训：canonical只修在线构造，原275离线训练输入已canonical；原4000–5000步selection MSE约.00417平台，同时训练loss下降而验证loss上升。换初始化不能隔离修复效果，也没有当前训练标签改变的依据；不声称旧模型全局收敛。保留同一best4500完整状态（54.86anchor等效epoch；历史末步5000约60.95），固定canonical，独立评估后4个字典序训练家族005/008/010/011和12个全新初态31171001–12，共16次上限，3h窗，无旧7seed复跑、无重试/补seed/自动扩评/最终test选择。历史275仍是原始混速State baseline，不是严格慢速多模态baseline；不启ImageDP或触觉训练。

## 本轮实际验收与生产配置冻结

Hand cap pilot5/6成功、4/6准入；独立4家族×3事件12/12成功、10/12准入，覆盖3/3/2/2，预登记门槛全部通过。唯一当前配置已固定Arm.50/Hand1/Hand cap2rad/s与physical_motion_gate(.90/1.25)，development_revision4；配置中的结果数字仅作来源标记，不能替代真实gate receipt。正式入口不再从开发hand_cap_plan读取运行参数；拒绝错方案、缺实际路径证据或用旧标量alpha入口冒充当前方案。

本轮有14条7家族train-only合格开发数据，可在来源/配置复核后复用；尚非正式数据集。State独立评估train1/4，新初态0/10合法（12尝试、2拒绝）；不随机重训的理由及完整失败分母见结果。可靠性过关后，实际容量门控因保留60GiB后只有约8GiB、复用开发数据仍至少约需31GiB新增空间而停止。没有正式物理采集、数据冻结或训练。空间到位后继续已有生产/训练合同，不重跑本轮开发队列、不重置既有预算/截止时间；若旧窗口到期，应如实登记未完成及新的实际授权范围，不能冒称收敛。

## 上一轮已完成A/B独立范围

任务A允许原275条State-valid原始混合速度数据的受控baseline诊断，不授予1.5–3慢速多模态资格。原200空白相机不进入视觉模型。现存best4500版本/数据/统计/EMA加载兼容，通过显式evaluation-only配置分别运行legacy_packet_inverse_v1与canonical_joint_state_v1，不修改源checkpoint或网络/损失/权重；仅此任务允许旧checkpoint的声明式输入对照，不静默升级。

A在启动前登记4个字典序训练家族、8个新初态，每例两种输入，共24物理启动上限、2h总窗；相同初态/事件/DDIM seed、交替执行顺序，非法初态保留不替换，软件/执行器错误或deadline停止。原275实际初态采样域为XY±25mm/相对±35mm、20°、joint lerp.03、qd±.15，按保存的家庭spec和实际旧采样函数确认；与B进一步收窄域区分。未见初态是此次新seed，训练家族是旧训练split来源；不以最终test选择权重。不因B失败抹掉A结果。

本次A已经完成全部24尝试：训练家族旧/canonical均0/4，未见初态旧0/7、canonical1/7，各另1次初态拒绝，未扩大队列。canonical出现完整原生成功并保留全记录；这只是有限同权重输入对照，不授权把原275当合格慢速数据或宣称稳定成功率。无新参数更新。

任务B仅分析已完成旧21capture（15慢+6参考）和新12capture（6慢+6参考），不重采。新的有限测试须有具体证据，保持逐条正式慢速标准；无统一可靠方案则明确最小采集侧修改提案，不擅改专家/PD/物理/学生算法。未来不同慢速数据必须另做matched-data baseline。

## 固定模型与控制

监督为transition/target_after_rad，29D绝对关节目标弧度，7Arm+22Hand；不是q_after，不是nominal。Actor141、robot178、geometry85及GRU256持续历史、原网络、sample/x0 Loss B和辅助权重保持。H24/commit12，5Hz规划/60Hz执行/120Hz物理；DDIM100训练噪声步、推理16、eta0、clip_sample=false、正式overlap=0。FP32关闭matmul和CuDNN TF32。

不改专家权重、资产、PD、原生观测噪声/观测延迟/动作延迟/外力参数或分布、任务成功/终止定义。不进入ImageDP、nominal、触觉模型或DAgger。StateDP使用声明的仿真几何，不冒称视觉估计。触觉/RGB-D记录只作数据准入，不是StateDP输入。

## 输入—标签—执行

obs[t]是动作前观察；target_after_rad[t]是两物理子步的真实执行器目标。target_before_rad[t]等于obs.actuator_target_rad[t]，t>0还等于target_after_rad[t-1]。实际q_after[t]等于下一tick q_before，不把实际q当目标。

新checkpoint显式canonical_joint_state_v1：robot的q/qd直接取与HDF5相同的canonical观察值，避免normalized-proprio反算丢位；旧checkpoint继续legacy_packet_inverse_v1，不静默改输入。

robot178的116:145为严格过去请求位置、145:174为严格过去应用位置、174:176为age、176为valid；首tick清零。训练历史是明确teacher-forced的过去专家目标，部署使用实际学生请求/应用反馈；当前/未来标签不可进入输入。真实每60Hz观察更新历史，replan/子目标不reset；只以真实episode ID切换清空。

已处理学生目标经反归一化和原关节限位裁剪，canonical→Lab后直接写原目标缓冲。不得再次alpha、delay、EMA，不写真实关节/物体状态。仅保留原delay随机抽样调用次序，其结果不再作用于处理后位置目标。必须核对实际两物理子步目标，不能只核对网络输出。

geometry为既有base-frame85D构造；四元数wxyz和q/-q处理保持。Actor使用冻结专家RMS140/clip5与embedding32；robot统计每native tick、geometry统计每12tick、动作仿射统计仅训练split拟合。三split按家族隔离；原速参考不得入慢速训练。开发示范仅在最终参数/来源一致时可入train，绝不进入独立validation/test；当前旧6条因Hand参数不同未复用。

## 唯一当前配置与数据门控

正式慢速生产配置为[configs/tight_state.json](../configs/tight_state.json)，运行时只读该配置、专家和资产，不读旧session/calibration/checkpoint作为新训练配置。原400及现有275源HDF5/manifest只读。旧200空白固定相机不改、不补图、不准入完整多模态集。

开发范围已在同一deadline内收窄一次：当前物体XY±2mm、相对孔位XY±3mm、旋转半径1.5723103°；台面z[.379,.381]m、物体距台面原点[.248,.252]m、原生关节lerp .0075、初速度±.03rad/s；中心完整值见唯一配置。最初±8/10mm、6°仅是保留的开发证据。初态采样与观测噪声/延迟、动作延迟、外力分别记录；只调整声明初始化参数，原生DR保持。direct_configuration实际读取每项配置。

已结束的前段开发登记（不限制最新明确授权的续行）：最初3家族×(原速+.55/.50/.45)=12次；一次收窄后的3家族×(原速+.55)=6次；最后同三家族.525共3次。合计21专家启动，该前段已结束。另1次240tick执行器映射oracle独立计数。旧all(admission)门槛未过不能解释成可靠性零容错要求；续行按下面独立登记的可靠性、产出和覆盖门槛判断。

每条慢速示范必须原生完整任务成功、同家族同配置原速参考、真实时长比[1.5,3]、抓取/抬升运输/插入三个实际运动阶段减速、全State/三路RGB-D/触觉时钟和有效位通过。运动阶段及分阶段实际joint-speed RMS是诊断代理，另记录object速度；不代替原生成功定义，也不主张轨迹逐点相同。alpha及中位数不能替代逐条时长；不通过成功后等待、帧复制或时间戳修改凑比例。

固定相机必须明确配置。RGB-D验收逐帧逐视角读取真实像素、depth与mask、时间及frame counter；camera_valid全True不能替代内容验收。触觉首tick无效、后续有效及120Hz两子步必须保持真实。原速参考与未达标成功记录仅作参考/历史，不进入慢速manifest。

未执行的生产预算为55家族、每家族4次慢速尝试，41/7/7家族train/validation/test；规模是软目标，不扩大范围补数量。所有attempt在物理启动前写账，task failure、非法初态、software/sensor fault分别统计。正式manifest/统计/cache只有逐条门控通过后才冻结；未知partial不删。

## 已完成有界续行的方案门槛

从e67bfc6修复工作树继续，固定Arm0.50/Hand1.00、phase=all，当前收窄域不再改。3个已有开发家族×4个新post-reset event seed，12对/最多24次物理尝试；不重复原seed事件案例。初始状态及reset抽取的DR概率由原initial seed保留；仅切换后续环境随机流，不改分布或抽样顺序。

启动前登记：慢速完整成功≥11/12、个体准入≥9/12，每家族≥3成功且≥2准入；原速参考≥11/12成功。可靠性、产出与覆盖分别输出，不再用all(admission)代替三者。阈值已无法达到则提前停止，不补seed、不扩参数网格。正式逐条门槛仍为原完整成功/实际总时长1.5–3/真实运动减慢/三模态有效，不要求每个关节或子阶段都1.5–3。

原5连续任务失败生产停止规则保持；新增预注册每20慢速attempt检查累计任务成功率≥.85、个体准入率≥.65。新2h验证窗口由本次续行授权设立，原整体36h和生产18h绝对deadline不重置。使用原pipeline与systemd用户service，Restart=no；未变工程实现复用原receipt和源码SHA，新正式数据本身仍验收并只拟合train统计。

实际结束：原速6/6、慢速6/6完整成功，慢速仅2/6逐条合格；3家族各2个新事件seed。准入最多2+6=8<9触发预登记早停，剩余6对未启动。完整可靠性和覆盖未验完，当前方案未获准生产；没有数据/训练/评估阶段通过。

## 训练与评估

模型、optimizer、EMA、scheduler、RNG、计数全新；不加载旧学生。原episode-balanced AnchorSampler每次最多64个不同episode，各取一个无放回anchor；短尾不丢不复制，micro4按全batch有效target数及有效相邻对分别加权，所有microbatch后才clip1/AdamW/EMA一次。以当前参数从episode开头重算历史，原96tick TBPTT边界保持；不使用随机零状态窗口。

预算按真实anchor数登记：初始120、最低80、上限200 anchor等效epoch；每500更新验完整validation。仍改善时按预注册40 anchor等效epoch阶段继续完整状态，不能重置优化器/EMA；最终test不选模型。达到固定更新或墙钟上限只能称预算停止，不能称收敛。GPU完整轨迹验收逐tick比较真实HistoryController与离线prefix的原始/normalized输入、hidden、condition、固定噪声动作；保存恢复必须包含优化器矩、EMA、采样和noise/Torch/CUDA RNG及schedule计数。

分别报告train家族、validation与冻结后新初态。先10个新初态；只有真实非零成功才扩到100；若扩展队列前50有效例全失败则停止，不补齐100或改分母。软件错误和非法初态单列，合法失败保留。充分训练仍失败也不自动换网络、损失、频率或无限加实验。

## 资源、清理、证据

共享GPU目标35%，保留50%整卡批次/完整60s约束与6GiB VRAM/12GiB RAM/60GiB磁盘余量。监控从启动至退出；实际超限/缺口如实报告。普通脚本管理有界阶段，模型在本次任务内复核完成结果，不把后台启动当作完成。

先解除依赖、检查活跃进程/打开句柄，再按路径、inode、size和必要SHA删除。原400/275、专家、资产、未知partial与新当前产物受保护；失败分母以精简结果表保留，不建巨型备份。旧历史完整文档/代码由Git642f5a2追溯，不重写共享Git历史。仅发布已审查代码和小证据，不上传数据/checkpoint。

本次B仅完成33条既有capture的阶段/运动/DR分析，新增物理采样0。当前固定参数仍不准生产；Hand因果目标速度上限只作为提案记录，未实施，未改变任何当前执行算法。
