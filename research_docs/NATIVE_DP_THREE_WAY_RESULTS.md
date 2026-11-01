# 三组 Native-command DP：实施与结果登记

## 新run（进行中）

旧 `native_20261101_01` 的 Oracle 阻断已单独保留。新 run
`native_20261101_02` 使用 source commit `8a147f8`，已通过独立最小 Kit 探针：
在现有近满 inotify 环境中调用 OmniClient `testSetWatchesEnabled(false)` 后，Kit
正常启动并退出。正式流水线服务 `native-dp-20261101-02.service` 已脱离会话运行，
当前实查阶段为 `data`，400条数据复核尚未写入完成标记；GPU deadline 尚未开始。
训练、checkpoint、闭环和视频仍不能填写结果，后续以该 root 的机器可读产物为准。

状态：实现已完成，训练前数据Gate通过，但 Oracle Gate 阻断。正式训练、test40、60条闭环与四视频尚未验收，不能填写成功率。
本次最后实查：root `outputs/visuotactile/native_dp_3way/native_20261101_01`，
独立用户服务 `native-dp-20261101-01-resume1.service` 已退出，`MainPID=0`、退出码2。
数据Gate attempt2已完成400/400；`oracle_base` attempt1返回-6，未发生GPU正式训练、optimizer更新或checkpoint读回。
日志显示 `libomniclient` `provider_file::FileThread` 抛出 `std::system_error: Invalid argument` 并触发原生 abort。
进程级 `/app/extensions/fsWatcherEnabled=false` 已传入；当前用户级 inotify 实查约524199/524288个watch，仍不足以证明底层监听失败之外的唯一根因。
该run的3小时Gate截止已过，24小时deadline保持原值，未重置、未自动重试、未启动训练。
本轮以 a905bca 为起点，采用用户确认的24小时总窗口、固定20条保留初态/模型。
原400数据、320/40/40划分和旧0/50结果保持独立。

| 组 | 可训练参数（CPU/meta构建实数） | 视觉/状态 | 长期记忆 | 当前结果 |
| --- | ---: | --- | --- | --- |
| S-NativeDP | 71,300,509 | robot13tick＋特权几何3帧 | 无 | Oracle Gate前阻断 |
| I-OfficialDP | 82,811,613 | 完整ResNet18/GN＋32关键点/视角＋depth | 无 | Oracle Gate前阻断 |
| I-HistoryDP | 72,770,109 | 固定DINO patch＋6×8空间token/视角 | visual256＋native256 GRU | Oracle Gate前阻断 |

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
Codex只做有界启动验收，之后退出交互；普通systemd脚本继续。最终详细解释、清理和Git推送在结果产生后审核。
不能假设后台服务会自动唤醒Codex，也不能把启动登记当作训练完成。

已验证：14项新CPU测试通过；与旧H64、T0和资源合同合并35项测试通过，compileall和git diff --check通过。
这包含目标输入有效梯度、过去动作隔离、H64/commit12与尾mask、全episode/分段条件一致、
当前权重前缀、History两种更新速率、完整视野深度梯度、接触稳定窗与产物hash验收。
真实GPU/Kit/训练结果以运行产物为准；本轮无训练结果。存在环境既有requests依赖版本警告，未修改共享环境。

研究问题的当前答案：三组命令接口已实现，controller target存在有效条件梯度；
自主成功率、DINO历史收益、是否支持RSS视觉BC、后续残差基座均尚未验证。
若仍全失败，必须分别报告接近/目标接触/稳定抓持/lift/对齐/子目标；不得直接归因视觉或历史容量。
