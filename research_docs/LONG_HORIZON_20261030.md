# G00-new / G10-new：长预测预训练视觉候选

## 最终结果 — 2026-10-31 03:41:43 UTC+8完成

两组训练、各一次held-out test、六条自主诊断及两条视频完整解码均已完成；
自有owner、trainer、仿真和monitor均已退出。10:00训练截止和12:00最终截止前完成，
不自动追加epoch、重测test或增加诊断。下文“运行中”是历史快照，以本节和最终小统计为准。

| 项目 | G00-new | G10-new |
| --- | --- | --- |
| 完整train320遍历 / 更新 | 40 / 33,280 | 40 / 33,280 |
| validation选定EMA checkpoint | epoch40 | epoch40 |
| 训练完成时间（UTC+8） | 03:32:24 | 03:33:16 |
| test40绝对参考arm / hand RMSE（rad） | 0.052432 / 0.530495 | 0.053575 / 0.528676 |
| test40还原native命令arm / hand RMSE | 1.145378 / 0.712621 | 1.154378 / 0.706728 |
| test首12步命令RMSE / 饱和比例 | 0.645846 / 43.39% | 0.654622 / 42.93% |
| 自主诊断任务成功 / 子目标 | 0/3 / 0 | 0/3 / 0 |

每遍5,547个anchor、301,333个有效未来时刻（各29D）；40遍累计221,880 anchors、
12,053,320个有效未来时刻、12,800次episode遍历，独立训练episode仍为320。
小示范pilot不接入main、不加到部署checkpoint更新数。test均用40条、固定RNG7093，仅运行一次。
test的copy-current-q参考基线arm/hand为0.079729/0.609985rad，新候选参考误差更低；
但零命令arm基线0.955774优于两组的新arm命令误差。hand命令则低于零命令0.921256。
因此不能把角度误差下降等同于有效控制学好，也不能把新离线协议与旧B0/B1直接作同口径排名。

### 六条自主诊断

| seed | 组 | 原生步数 / 终止 | 传感器接触步数 | 原生抬升 / 最高高度变化 |
| --- | --- | --- | --- | --- |
| 981001 | G00-new | 601 / timeout | 20 | 否 / −0.002044m |
| 981001 | G10-new | 601 / timeout | 156 | 否 / −0.002044m |
| 981002 | G00-new | 601 / timeout | 19 | 否 / −0.002044m |
| 981002 | G10-new | 601 / timeout | 16 | 否 / −0.002044m |
| 981004 | G00-new | 601 / timeout | 65 | 否 / −0.002044m |
| 981004 | G10-new | 600 / fall | 248 | 是 / +0.107093m |

接触指任一原120Hz子步、任一指端力模长>0.1N；没有碰撞对象身份，不能据此确认抓住目标。
六条均无子目标；G10的981004抬升后掉落，稳定抓握与任务完成未建立。其余五条未抬升且反复接近/退离。
G00三条计划边界命令RMSE为0.5286/0.5557/0.5573，G10为0.5097/0.5116/0.5115；
闭环命令饱和比例G00约36.45–41.84%、G10约34.21–38.30%，仍有较大切换和跟踪误差。
与旧无抬升诊断相比出现一个新的行为信号，但不是任务成功改善；只有三个初态、同时更改多项架构，
不能宣称触觉因果收益或方法收敛。三个种子的全部保存初态字段逐值精确匹配；不声称图像逐位相等。
本六条只属新候选开发诊断，原B0/B1正式各0/50及旧残差分母不变。

仅预登记首seed981001各录一条代表视频，两条均601帧/60fps，完整解码与当前SHA复核通过；
它们展示timeout，不能用来观看未录制的981004抬升。路径：

- `outputs/visuotactile/run/long_horizon_20261030_v2/closed_loop/G00-new_s981001.mp4`
- `outputs/visuotactile/run/long_horizon_20261030_v2/closed_loop/G10-new_s981001.mp4`

### 最终资源与产物核验

自适应节流训练段02:20:54–03:33:16共4,341.111秒，整卡加权均值37.8641%，达到约40%的目标范围。
替代阶段完整monitor含缓存、pilot、等待、训练及诊断，均值16.7201%，完整60秒最大41.3339%，
超50%窗口0、缺测0、最大采样间隔1.7852秒；最低空闲显存18,865MiB。
两组恢复后worker峰值RSS约3.399/3.437GiB，CUDA allocated峰值约1.645/1.652GiB，各自单独统计。
全程最低可用RAM仍为9.825729GiB，有一次低于10GiB；更早无效初版还有监测缺口，
因此整个研究会话不满足“全程资源合规”。不能用后段均值掩盖此前事件，也不能把monitor RSS当作总RAM。
report中的elapsed/work/wait/pacing与timings只覆盖恢复后worker生命周期；完整主训练窗口约90.06分钟。

最终只读CPU产物审查通过：6条trace所有数值有限，H64/每12步重规划/逐步reference对应、
原生1/60秒时钟、实际提交与clip命令、接触统计及native lift一致；3对初态精确匹配；
validation-selected best、test和部署SHA相同；运行记录的6个源文件hash未漂移；
原manifest和划分来源B0 best SHA未变。原controller/研究sim wrapper相对接手提交无修改。
视频只复核已完成解码记录与当前hash，未重跑策略或test；本次仅更新文档和小统计，不重复全套代码测试。
既有95项全套和调度后16项定向有重叠，不能相加。

前次实际清理410文件/4,350,141,356 bytes（约4.05GiB）保持不变，本次不额外删除产物。
最终小统计与权重/轨迹hash见[evidence/long_horizon_final_20261031.json](evidence/long_horizon_final_20261031.json)。
训练及限定诊断已交付，任务成功、稳定抓握和触觉收益仍未得到证明；不能自动扩训练或试验预算。

## 最新运行修订 — 2026-10-31 GPU目标40%

用户明确要求平均GPU约40%。sample pilot各3300更新已完成并通过完整示范拟合门槛：
G00/G10训练arm命令RMSE0.401126/0.419831、hand0.025333/0.026953，参考arm0.013626/0.014306rad；
这些仅是四条训练示范的拟合证据，不是40条test或自主结果。
调度交接时全训练已进入第4遍。固定2.5倍休眠时GPU利用率偏低，因此改为共享GPU锁内的反馈节流，
10秒时间加权均值每5秒调整倍率，目标40%±2%；完整60秒及整批均值45%准入，原50%上限不变。
至少2秒的重型调用仍采用不小于2.5倍休眠，启动/仿真边界不假设短块安全性；6/14GiB内存准入保留。
缺测不填0，全部worker共享同一状态，不会各自根据低均值同时加速；原120/60Hz与5Hz规划不变。

为让新调度实际生效，只有本任务owner被终止，两个trainer完整episode边界保存2845/2852更新后退出；
monitor未停止。显式--resume-training归档旧状态并用新日志接续同一训练，模型、EMA、optimizer、RNG、
epoch/episode游标与33280更新schedule不变，只有resource源码hash变化；未从零训练、未重复增加完成更新。
gpu_schedule.json配置目标，gpu_pacing_state.json/work.jsonl记录反馈；16项定向检查通过。
下方早期pilot仍运行、固定节流和历史资源数值均为时间点快照；新的实测窗口继续追加，不能称已训练完成。

02:26:29 UTC+8实测：恢复后的334.787秒整卡加权均值36.5095%，末180秒38.4496%，
最新完整60秒38.4583%、恢复后完整60秒最大39.2677%，超50%窗口0、缺测0。
调整前单个完整60秒为17.5645%；当前累计均值9.8122%含早期空闲，不以额外工作补偿历史均值。
恢复段最低空闲显存18,889MiB、可用RAM34,744,274,944 bytes（32.3581GiB）。
G00/G10已达4950/4975更新，均进入第6遍（完成5遍），各目标33280更新/40遍不变；
status中的秒/更新约0.1545/0.1541，调整前快照0.3666/0.3655，阶段不同，不作受控加速比结论。
上述是调度验收，训练/test/六诊断仍待完成。早期缺测尝试与01:45的RAM低于10GiB事实继续保留，
不能称整个研究会话资源合规。小型统计见`evidence/gpu_pacing_20261031.json`。

授权2026-10-30，截止2026-10-31 12:00 Asia/Singapore。开始核对本地与origin功能分支均为
`a4998e8bbd79c973e4b3a2ef6c059b6cec0e07cc`，初始工作树干净。原控制器/研究sim wrapper未改。
本轮同时改变视觉、历史、预测长度、动作表示；任何改善不能单项归因。

## 状态与无效尝试

当前开发中，正式全划分训练/闭环尚未交付。初步`long_horizon_20261030`有当前命令泄漏、错误划分、
5Hz而非60Hz持续状态、未固定DINO源码/224²resize等问题；400条缓存及G00一轮822更新后被发现并停止。
其模型/结果无效，不能算本轮合法训练、离线改善或可部署checkpoint；未开始G10和闭环。
初步尝试监测没有连续覆盖启动到退出，低采样均值不能证明本轮全程资源合规。

替代root `outputs/visuotactile/run/long_horizon_20261030_v2`，cache/checkpoint版本2拒绝初步产物。
原B0 best中原320/40/40列表直接复用；test只在选定EMA best后独立一次。
训练截止10:00（UTC+8），最终12:00；重试不延长截止，预留两小时验证/诊断/交付。

## 实际输入与输出

三相机固定顺序fixed_high/fixed_side/wrist_rgbd，448×252原记录60Hz；新视觉每12tick编码一次（5Hz）。
RGB uint8/255、ImageNet mean [.485,.456,.406]/std [.229,.224,.225]，不resize/crop。
官方DINOv2 ViT-S/14，`dinov2_vits14_pretrain.pth`；源码pin
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`，权重SHA由cache_index保存。
来源 https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8 （Apache-2.0）。
冻结全部22,056,576个backbone参数；缓存每帧3×384×18×32完整patch特征，FP16存储。
深度/valid固定14×14块预处理为有效米值平均/2与coverage，3×2×18×32，不缓存训练模块输出。
可训练RGB384→32、depth2→16卷积融合3×5网格→128/视图；固定视图身份拼接384D。

| 输入 | 原张量与显式跨度 | 编码 |
| --- | --- | --- |
| 视觉 | 11个5Hz时刻，跨度2s，真实sample time/age/valid | 384+3age+3valid→masked GRU128 |
| 本体 | 121个60Hz编码器q(rad)/qd(rad/s)，跨度2s，时间/valid | 58+时间→masked GRU64 |
| 动作 | 121条过去已提交/实际延迟采用29D命令、两个source age/valid；最新为t−1 | 61→masked GRU64 |
| 触觉（G10） | 242个120Hz五指world/local法向力，age/valid，原样本顺序 | signed log1p力+age/valid共32→masked GRU64 |
| 持续记忆 | 每60Hz当前本体/执行反馈/持有视觉与age，G10加两按序力样本 | native GRU128，episode才reset |

窗口不足左padding，打包GRU跳过无效前缀；显式窗口和持续状态共存，融合512D条件。
固定维度状态不是原始历史无损保存。非刷新tick保留旧视觉真实time/age，不伪装新图像。
过去专家执行明确teacher forcing；当前/未来专家轨迹只用于扩散监督，干净未来不进条件，无旧计划输入。
部署仅实际学生ack；物体/阶段/reward仅诊断；G00无触觉gate，二组无残差，controller target只在输出适配。

每次输出64×29绝对参考，覆盖64/60=1.0667s、首末63/60=1.05s，逐tick执行12个不同参考再重规划。
canonical前7arm/22hand；Lab手指顺序不同，记录字段已canonical，env.step仍做原映射。
arm标签r=p_before+speed×dt×u_clipped，不提前限位；hand标签用原limits绝对映射。
逐tick用当前controller target逆变换到命令再clip[-1,1]，原delay/平滑/限位/物理不变。
不使用target_after二次积分、不直接写目标/关节、不把参考当未来真实q或保证延迟后按时到位。

## 官方依据和训练协议

DP pin `5ba07ac6661db573af695b419a7947ecb704690f`（MIT），已有vendor许可证/来源保留。
ConditionalUnet1D [256,512,1024]、kernel5、time128、groups8、FiLM scale/bias；cosine100 DDPM，
DDIM16 eta0。初步合法pilot使用epsilon；10月31日拟合复核后独立sample目标pilot见下方，不切换旧权重语义。
适配：H64/29D/512D历史、三路RGB-D/DINO、60Hz持续状态、尾部mask、绝对参考。
关闭scheduler sample clipping，防止暗裁超限arm参考；normalizer按官方limits affine规则train320拟合min/max，
常数维保留可逆性，无标签clamp。AdamW1e-4/betas(.95,.999)/wd1e-6，cosine+500update warmup，
EMA inv_gamma1/power.75/max.9999。首先4条完整train/4条val拟合；pilot不接续main。
main两组同预算独立权重/optimizer/RNG，每native96tick TBPTT含8anchors，跨块状态保留，epoch只打乱整episode。
EMA validation固定RNG7091，以arm/hand等权采样native command MSE选best；test用7093，一次。
同时报告reference/command RMSE、前12步、饱和、计划边界变化及copy-current-q基线。
离线还原使用专家target_before，不冒充学生未来target。last含model/EMA/optimizer/RNG/schedule/episode游标。

## 资源与验证口径

独立1Hz monitor从替代缓存启动前到最终退出；全部GPU worker共享fcntl锁，batch准入38%、完整60s准入35%，
每短块后实测工作墙钟2.5倍让出。资源余量4GiB VRAM/10GiB可用RAM/12,000,000,000 bytes磁盘。
01:55复核后，后续新worker准入提高到空闲VRAM6GiB/可用RAM14GiB，为模型和CPU快照临时分配留余量；
正在运行的pilot仍按其启动hash使用旧4/10准入，已应用快照回收修复，当前余量须看实时监测。
软件采样不是硬件quota，实际超限照实记。初步未连续监测的尝试仍单列，不能用后段覆盖。
patch缓存预估9,297,690,624 bytes，每次最多2个采样帧，每episode有界。内存统一MiB/GiB，磁盘另列bytes。
删除前sha/size/type/reason清单，实际unlink才记已清理；原400/manifest/split/best/旧0/50及逐条摘要保留。

`bash visuotactile/run_cpu.sh -m pytest -q tests/test_long_horizon.py` 6通过；加action_history和training_resources
合计15通过，包含前6，不能相加。初步19项是旧action_history/command_training回归，不等于新模型验收；
该19项未用run_cpu入口，后续统一使用。requests既有warning未改。
新测试覆盖当前/未来标签变动不影响当前输入、整段/分块/逐步native状态一致、短轨迹/tail、padding、
严格组allowlist、限位附近arm不裁剪与非同一手指排列往返。最终真实数据/闭环证据待追加。

## 当前已完成的真实验收与预算声明

400条逐条oracle参考→command最大误差2.384185791e−6，native随机delay+限位+平滑公式重建target_after最大误差0。
实际delay索引0/1/2分别27,063/27,568/27,263次；真实记录中无arm参考越限，补充单元测试覆盖限位附近。
手指canonical/Lab非同序已核对；测试调用原action_utils，固定同RNG验证32次原链路适配等价。
真实25tick训练/在线反馈重放条件最大差G00 2.384e−6/G10 1.431e−6，memory最大差4.172e−7/5.402e−7。
该重放是记录专家ack，不是自主student。新测试增加原native随机delay后7项；完整tests共93通过，
命令`bash visuotactile/run_cpu.sh -m pytest -q tests`，包含前述15/7，不能相加。

缓存400条完成，9,456,515,110 bytes（约8.81GiB）；权重SHA
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`。
参数：G00可训练71,166,285+冻结DINO22,056,576=93,222,861；G10可训练71,242,445+
冻结DINO22,056,576=93,299,021；共同denoiser70,369,437，EMA/optimizer副本不算神经参数。

最初各330update的四完整示范pilot仅早期学习信号，采样仍高饱和，未当充分拟合验收。
固定追加独立fit3300：同四条完整train/4条val，各300遍/3,300updates，不接续main，不用test。
全量预算已于pilot真实更新后写入training_budget.json：每组40完整遍历、33,280updates（832/遍），
每遍5,547anchors、301,333有效未来目标，native96tick TBPTT/8anchors。Cosine总量固定33,280/warmup500。
两组权重/optimizer/RNG独立seed20261031/20261032，缓存共享，GPU互斥交错。
早期并发pilot约0.5–0.6s/update/worker含频繁四episode保存/验证；全量实测仍需单列，不能冒称完成。

实际清理410文件、4,350,141,356 bytes（约4.05GiB）：404个无效初版缓存/checkpoint、6条旧失败MP4。
保留旧正式100trace、best、400示范、manifest/split/归一化、各0/50逐条摘要；有效恢复文件不删。
删除清单见[evidence/long_horizon_cleanup_20261030.json](evidence/long_horizon_cleanup_20261030.json)。

## 2026-10-31接续审查：拒绝仅凭角度下降放行

前一轮对话结束时没有启动main，两个epsilon强pilot后来完成。两组各300次四完整episode遍历、
3,300更新；这不是320条训练集的300个epoch。G00/G10训练集参考arm RMSE分别0.228596/0.218865rad，
hand0.586308/0.589422rad；native arm1.355884/1.362114，hand0.785598/0.787535。
copy-current-q的参考arm基线0.074390rad，零命令arm基线0.952017。故不能称充分拟合通过，main未被放行。

冻结同一pilot，四条完整train上检查raw/EMA、DDIM16/100，不访问test：raw与EMA几乎相同；
DDIM100的arm参考误差达到121.92/120.35rad，饱和约99.7%。在无sample clipping的epsilon反演中，
接近终端噪声的误差放大确实可见，但未证明它是全部欠拟合的唯一原因，增加采样步数不是已验证修复。
记录在各pilot的`sampling_review.json`。保留这些有效但未通过拟合的checkpoint，不执行它们。

固定DP源码的`diffusion_unet_hybrid_image_policy.py`同时支持epsilon和sample目标。
新增显式`prediction_type`，旧配置缺省仍epsilon，不静默改变旧checkpoint。
新`sample3300`独立权重、同seed/4完整train/4val/300遍/3,300更新、同结构和DDIM16；
sample直接监督干净归一化绝对参考，无额外controller target输入，无sample clipping。
启动前固定准入：两组分别要求arm/hand native命令RMSE均低于零命令、参考RMSE均低于copy-q，
命令误差相对初始化下降，且真实因果验收通过。不能以仅reference下降的旧`improved`布尔值替代。
这只是少量完整示范的可拟合性门槛，不证明任务成功或主训练收敛。主预算40遍/33,280更新不缩减。

审查同时修复：采样/训练H2D及checkpoint D2H统一在GPU lease内，CPU写盘在锁外；
checkpoint只保留一次CPU副本并及时回收，新增worker自身VmHWM/CUDA allocator峰值；
完成时status不再遗留training；resume拒绝配置不一致之前不覆盖config；test使用最终deadline，
而不是10:00训练截止。首25次主训练吞吐另存`first25.json`，不能靠最新status倒推。
监测进程的RSS仍仅代表monitor；worker峰值从work.jsonl/status/report读取，不冒充总显存占用。

`run_long`是可后台执行的单一进程owner：等待已启动sample pilot，验收后启动独立main，
完整或截止时用已验证best各test一次，再运行预登记981001/981002/981004×两组。
主训练错误/准入失败不会自动重跑，Kit退出0必须同时有完整report才能算诊断完成；
不替换非法种子。全部子进程结束才停止monitor。CPU完整解码新视频后另存验证记录。
尚未实际跑过的新仿真入口必须标为待闭环验证；代码存在不等于六条已完成。

最新回归：`bash visuotactile/run_cpu.sh -m pytest -q tests`，95通过（含新9项），
是在sample目标/准入与调度审查后的版本，覆盖旧93及定向集合，不能累加；既有requests warning保留。

01:45左右发现CPU checkpoint快照的RSS累积；在可用RAM仍大于10GiB时，定向SIGTERM两个自有pilot，
完整episode边界分别保存1,115/1,122更新后退出。修复单次CPU复制及保存后gc回收，显式runtime-only恢复：
旧/新trainer、resource hash和更新数保存在`resume_runtime_*.json`；model/data/action源码hash完全一致，
模型/EMA/optimizer/RNG/epoch游标原样加载，未重复计入已完成更新，未更改3,300更新schedule。
恢复后RSS回落约3.4GiB/worker，须继续查看后续峰值，不能以单次下降证明永久稳定。
本次13项定向回归包含long_horizon与training_resources，不与95全套相加。

进一步用真实148步示范的前145个连续观察跨过两秒窗口边界，保留原25步证据；
新`causal_acceptance_145.json`的G00/G10条件最大差1.669e−6/1.848e−6，memory最大差5.662e−7/5.066e−7。
仍是记录专家反馈重放，不是自主控制。400个缓存逐文件bytes/SHA复验通过，manifest及原split来源best SHA未变。

后台状态入口（只读）：

```bash
bash visuotactile/run_cpu.sh -m visuotactile.run_long --root outputs/visuotactile/run/long_horizon_20261030_v2 --status
```

实时`pipeline.json`区分waiting_pilots/training/test/closed_loop/complete/interrupted，
各组status/report保留实际更新数；`first25.json`是主训练首批吞吐，`test.json`与`closed_loop/report.json`
才是后续结果。`complete`只表示流程结束，不能替代task success。
提交时sample pilot约1650/3300更新，main/test/六条诊断仍待验收后接续；这是进行中快照。
公开小报告见[evidence/long_horizon_review_20261031.json](evidence/long_horizon_review_20261031.json)，
不上传缓存、checkpoint、原始轨迹、视频或日志。

### 资源不合规订正（01:55 UTC+8核对）

替代阶段在2026-10-31 01:45:08.447985出现一个1Hz采样点，可用RAM10,550,296,576 bytes=
9.825729GiB，低于10GiB要求。该点发生在旧CPU快照路径停止/恢复之前；不能把“发出停止时仍有余量”
写成整个停止过程始终满足余量。整卡GPU均值约8.16%、最大完整60秒25.63%，监测缺失0，
但RAM约束已实际违反，且更早无效尝试还有连续监测缺口，因此本轮绝不标为全程资源合规。
快照回收修复后worker RSS约3.4GiB，当前可用RAM约35GiB；新主训练将使用6/14GiB保守准入。
采样不是硬配额，后续仍按真实记录报告，不以最终均值覆盖这次低余量。
