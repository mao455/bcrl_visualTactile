# 三组 Native-command DP：实施与结果登记

## 最新运行（late-gate recovery，2026-11-01）

用户紧急授权已被机器记录为 `native_20261101_02/late_gate_recovery.json`：原
`gate_deadline`、`training_deadline` 和总 `deadline` 均未修改，没有新建预算或清除
blocked 历史。恢复前置检查因 Bash 路径映射阻断一次，未启动 GPU；随后使用本机
bundled `imageio-ffmpeg`/PyAV 后端重新启动。

当前恢复进程已用 `setsid + nohup` 脱离会话运行，主 stage 为
`late_gate_preflight`，`status.json` 实测 `waiting_resources`。Kit preflight
尚未写入 `result.json`，因此不能称为补验通过；没有新的训练更新或评估结果。
后台继续由原 run 的资源监控、总截止和阶段 manifest 管理，Codex不驻守轮询。

## 当前状态（2026-11-01核验）

**未完成，当前blocked，无后台训练worker。** 主run为`native_20261101_02`，
source commit `8a147f8`。证据见[evidence/native_dp_status_20261101.json](evidence/native_dp_status_20261101.json)。
旧run01保留；run03重复启动已在GPU前停止，未启用新预算。下述实测均属于run02。

- 数据Gate：400/400，320/40/40，normalizer往返最大误差0。
- Oracle base和instrumented：各220步原生成功，12项控制/物理字段逐位相等；
  诊断传感器与录像的行为中性在这条参考轨迹上得到支持。
- S与Official真实network Gate：condition/memory/action一致性误差均0；target
  干预确实改变condition并有有效梯度。尚未执行History真实GPU Gate。
- 正式训练0更新，test40未执行，正式60条闭环未执行；不能填写正式成功率。

| 组 | 可训练参数 | 已完成拟合Gate | arm/hand RMSE | 当前未完成部分 |
| --- | ---: | --- | --- | --- |
| S-NativeDP | 71,300,509 | fit1 400更新；fit4 500更新；pilot20更新 | fit1 .02709/.04888；fit4 .09278/.16097 | 正式训练及评估 |
| I-OfficialDP | 82,811,613 | fit1 400更新 | .03336/.06122 | fit1自主诊断、fit4、pilot、正式训练及评估 |
| I-HistoryDP | 72,770,109 | 未执行GPU Gate | 未测 | 全部GPU Gate及后续阶段 |

以上RMSE来自EMA前12步原生命令、真实完整train轨迹；两组fit1和S fit4均达到原阈值
及相对零命令基线至少30%改善。Official last checkpoint在CPU重新读回：optimizer
232组状态均为step400，EMA400，scheduler400，数据cursor0，RNG存在；best/last hash
与完成阶段manifest一致。这证明实际更新和保存，不构成正式训练完成或恢复等价性证据。

S的两条fit自主诊断均从固定首train初态执行：fit1在168步fall，fit4在402步fall，
均有目标接触、稳定抓持诊断及native lift，但无子目标、最终成功0/2。它们与正式test
分开，不能扩展成模型泛化结论。规划超过200ms分别1/5次。

## 阻断原因与预算纠正

Official fit1 diagnostic attempt1在Kit初始化期间报告`std::system_error: Invalid argument`，
没有形成轨迹或结果；`testSetWatchesEnabled(false)`确实已记录，但不能据两个Oracle
成功就声称后续启动完全修复。现有worker日志没有提供FileThread栈，所指Kit日志现已
不可读，根因仍未唯一确定。systemd journal的cgroup inotify ENOSPC属实，但早于
多个成功的Kit阶段；换用setsid并未证明能修复这次Kit故障，不作为已验证方案。

时间均为北京时间：首GPU为11月1日10:38:19；Gate截止 **11月1日13:38:19**；
训练截止11月2日07:38:19；总截止 **11月2日10:38:19**。15:51的attempt2被Gate
截止阻断，并非第二次原生崩溃。先前摘要所称“24小时已过”已更正。

本轮曾据该摘要以setsid启动run03数据复核；发现与session冲突后已显式停止，
PID退出，GPU时钟仍为null，source commit实际5032af3。保留此失误和停止记录，
不重启它来延长授权预算。现有run02原deadline和历史成果未修改。

流水线已修正：worker启动前校验具体截止；Gate/总截止/monitor退出分别报告；
失败attempt保留退出码；预启动拒绝不会覆盖历史attempt。没有改控制器、物理、数据、
网络、阈值或实验预算。继续需验证新的Kit修复方向，并明确处理已经耗尽的Gate窗口。

## 现有产物与资源

root：`outputs/visuotactile/native_dp_3way/native_20261101_02/`。
专家视频：`stages/oracle_instrumented/attempt_1/01_expert_success.mp4`；
checkpoint：`stages/{group}_{fit1,fit4,pilot}/attempt_1/{best,last}.pt`（仅列实际已有组/阶段）。
视频为H.264/yuv420p/60fps、220帧、3.666667秒，PyAV和ffmpeg全片解码、ffprobe
元数据检查均通过。默认PATH缺工具，已找到本机isaac环境可执行副本；只调用二进制，
不切换Python环境。流水线增加GPU前工具/动态库检查及显式路径环境变量，路径/hash
写入video_tools.json；本次核验路径/hash见证据JSON。未安装或更改共享环境。
三个学生正式视频和正式训练曲线尚无产物。

首次活动监测区间均值27.8459%、完整60秒最大46.4103%、缺测样本0。恢复后的总表
均值28.0244%，另记录停机期间17311.88秒无采样；不能抹掉间隔或声称连续无缺测。
现存protected清单逐项复核保持，未清理400数据、权重、失败日志或未知文件。

## 保留的实验合同与尚待执行项

History数量不含独立冻结DINO；最终Gate另记录它的实际参数量和缓存复现误差。
这三组同时改变输入权限、视觉编码与历史机制，是共同预算系统baseline，不能解释为单变量因果消融。

共同配置：H64/commit12，29维原生pre-delay命令，condition512，U-Net[256,512,1024]/kernel5/time128，
sample/x0，100步cosine训练噪声，16步DDIM，归一化空间arm7/hand22各50%的Loss B。
AdamW lr1e−4、betas(.95,.999)、wd1e−6，warmup500、cosine、clip1、官方EMA power.75。
当前采用FP32，未以未经验证的混合精度替换它。

Gate顺序为400条数据与缓存身份、220步native-request重放、诊断/录像行为中性、真实网络梯度、
完整episode在线/离线一致性、每组1条及4条完整轨迹拟合及各一条自主诊断。
单条arm/hand RMSE≤.20/.10；四条≤.35/.20；均须比零命令基线改善至少30%。
失败保留原因，不自动降阈值、不继续正式训练。通过离线拟合不等于自主任务成功。

首个GPU Gate前开启持续监测并固定24小时deadline；Gate至第3小时，训练至第21小时。
三组pilot各20更新并完整val40，用实测更新/验证时间计算共同上限，保留15%，向下取整到1000；
上限20000，若不足5000则阻断。每1000更新完整val40，最终EMA best才做一次test40。
至少5000更新且20遍训练后，连续6次验证相对改善不足.5%才可早停。

正式闭环前固定4条train初态/模型。正式test按排序前20 ID，每模型一次；不按结果更换初态或继续调参。
三个学生主视频来自第一条正式test实际评估，专家为独立seed17/220步参考；两者初态不同。
接触由工件过滤的真实法向力记录，稳定抓持诊断与native lift/subgoal/success分别统计。

资源目标整卡40%、准入45%、批次和完整60秒上限50%，持续记录功耗/进程显存/缺测与真实超限。
采样准入不是硬GPU配额，不能阻止其他进程突发；不操作其他任务。当前保守准入为6GiB VRAM/14GiB RAM，
底线4GiB/10GiB，磁盘12GB；所有等待计入既定deadline。

机器可读入口为每个新root的session、manifest、status以及completion或blocked JSON。
Codex只做有界启动验收，之后退出交互；持久后台脚本负责执行；当前服务已退出。最终详细解释、清理和Git推送在结果产生后审核。
不能假设后台服务会自动唤醒Codex，也不能把启动登记当作训练完成。

本次21项Native CPU测试通过，覆盖过期恢复零worker启动、原attempt保留、退出码、
monitor分类及视频工具/动态库检查。历史验证：14项新CPU测试通过；与旧H64、T0和资源合同合并35项测试通过，compileall和git diff --check通过。
这包含目标输入有效梯度、过去动作隔离、H64/commit12与尾mask、全episode/分段条件一致、
当前权重前缀、History两种更新速率、完整视野深度梯度、接触稳定窗与产物hash验收。
真实GPU/Kit/Gate拟合结果如上；本轮仍无正式训练结果。存在环境既有requests依赖版本警告，未修改共享环境。

研究问题的当前答案：三组命令接口已实现，controller target存在有效条件梯度；
自主成功率、DINO历史收益、是否支持RSS视觉BC、后续残差基座均尚未验证。
若仍全失败，必须分别报告接近/目标接触/稳定抓持/lift/对齐/子目标；不得直接归因视觉或历史容量。

进度只读命令（不启动任务）：

```bash
bash visuotactile/run_cpu.sh -m visuotactile.native_pipeline status \
  --root outputs/visuotactile/native_dp_3way/native_20261101_02
```

未来经预算重新确认后运行时，使用已验证的进程级工具路径：
`NATIVE_DP_FFPROBE=/home/CNS2025331827/miniconda3/envs/isaac/bin/ffprobe`、
`NATIVE_DP_FFMPEG=/home/CNS2025331827/miniconda3/envs/isaac/bin/ffmpeg`。
设置路径不构成恢复授权；run02仍受原截止和源码身份约束。
