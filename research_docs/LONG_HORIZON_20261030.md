# G00-new / G10-new：长预测预训练视觉候选

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
