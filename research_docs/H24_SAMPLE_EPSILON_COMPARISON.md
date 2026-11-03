# H24 Sample / Epsilon 对照

## 真实恢复现场（2026-11-03，北京时间）

本轮恢复已有 `h24_parameterization_20261103_01`，没有新建替代实验。
工作树 `bc+rl_RSS_native_v2`，分支 `feature/native-dp-training-v2`，恢复前 HEAD `8ef2a16`。
此前连接中断时，Epsilon 代码、审计及 owner/report 脚本已保存但未提交，
`status.json` 为 `gate_completed`；无本轮存活训练/Kit/monitor，无 Epsilon checkpoint，
也没有 `setup_complete.json`。因此尚未开始 Epsilon 更新，不能称为训练中断后重训。

Sample 的原始 `h24_train/milestone_3000.pt` 为 1,153,275,969 字节，SHA256
`70783fcf67344f91a68ff9dec66e42bb0125b87e8dca5192c827c49dd29b25c0`。
CPU 读取确认 updates/AdamW steps/EMA 均为 3000，832 项 schedule 的 cursor 为 176；
optimizer、model、EMA、CPU/CUDA RNG、schedule/noise RNG、history 均在。
当前 LR 为 `9.55614245194068e-05`，沿原 19000-update cosine，不重置。
原文件不写入；续训副本使用干净的冻结工作树 `bc+rl_RSS_h24_sample_pin`，
commit `8ef2a1646c54c47d695b0ca92ec7830f0b4abd03`。

上次 39 项数值/回归测试、Sample 新旧逐值一致性及真实尾段 28 个 padding 梯度为零
作为既有验证沿用，本次不重跑；原 batch 审计不重做。恢复新增测试只覆盖 owner
断点、结果幂等性、日志回退及已有选择/报告接口。实际启动另用状态和已提交更新核验。

## 固定训练合同与预算

- train320 / validation40 / test40，test 不参与本轮选择；832 个至多 96-tick 片段，
  每次更新采样 2 段，平均 13.3267 个 anchor，两个片段均贡献梯度；batch 不变。
- H24 / commit12，原 29D 绝对位置 target_after 弧度动作、State 观察与持续 GRU，
  原生 120 Hz 物理 / 60 Hz 控制。统计仅来自 train320。
- Arm7/Hand22 归一化空间组内 MSE 各 50%，有效 mask；AdamW LR1e-4，warmup500，
  cosine19000，原初始化 seed20261102、EMA、beta schedule / 100 timesteps。
- DDIM16、eta0、clip=False，policy seed7091；唯一学习差异为 Sample/x0 与 Epsilon/noise
  监督及对应 DDIM 重建公式。无 Temporal/FK 辅助损失、无架构或采样改动。
- Epsilon 先 1000 checkpoint 再累计 3000；公平性核对配置、每步 LR、采样顺序/游标、
  EMA count、noise/schedule RNG。两组 3000 各评估后仅选一组到累计 5000、8000、10000。
  每个里程碑保存 optimizer/EMA/RNG/sampler 和指标、执行同口径闭环；0 成功本身不停训。
- 沿用既有固定端点：2026-11-03 21:02:12 setup Gate、11-04 05:02:12 训练端点、
  11-04 07:02:12 总端点（UTC+8），没有因本次恢复延长预算。

## 配对评估和选择

离线统一比较重建 clean action 的原单位 Arm/Hand RMSE，valid positions 加权；
同时报告目标 FK 末端 mm、motion cosine/amplitude、关节变化与重规划边界跳变。
训练诊断为固定 train16、验证为 val40；原 diffusion loss 可追溯，但两类 loss 的绝对值不比较。

使用上次登记的同一 train4 + validation8 初态/seed，各条独立 Kit 启动；每组每里程碑至多12条。
统计完整任务 success、接触、lift、首子目标、alignment、fall、timeout 和实际接触前末端偏差。
观察初态逐字段核验，未证明全部 PhysX 隐状态完全相同。软件故障/未启动不计任务失败，
只在共同完成的合法 validation IDs 上配对选择；train4 作为诊断单列。

沿用 `run_plan.json` 预登记保守排序：完整 validation 成功与退化保护、子目标/lift、
接触前目标 FK、balanced prefix12 MSE、边界跳变、规划耗时。另报告方向/幅值与全部曲线。
单种策略 seed、小样本、冲突或缺失证据下的选择始终标为暂定，不宣称统计显著优势；
一次成功或单项略好不足以得出模型明显更好。两组 3000 权重均保留。
若两次连续已分类的 pre-scene 启动故障，则该 cohort 剩余项记软件缺失，
后续训练/里程碑继续原预算；不反复重试同一条、不改种子、不启用进程复用。

## Kit 与恢复机制

已有4个完整专家 episode，4/4成功；原始 inotify ENOSPC、失败启动和 strace 全保留。
成功启动不证明 FileThread 全由 inotify 引起，也不证明 SDK 根因修复。
进程内 reset 轨迹一致性失败；正式评估强制 `isolated_singleton`。
脚本路径展开修正已落盘；新的固定 shell 入口避免嵌套 shell 字符串中的变量展开错误。
已存结果后隔离退出/父进程回收，durable worker error 后保存证据再退出，防止 SDK cleanup 挂住。

普通 systemd 服务 `h24-comparison-20261103-01.service` 管理 owner、monitor 和单 GPU worker。
每个阶段有独立目录、PID/launch/exit、日志和 SHA 完成标记；已完成结果直接接续。
真实训练中断最多恢复一次，保留失败目录及完整日志，从完整 checkpoint 恢复，
仅将超出 checkpoint 的未提交日志行归档后回退；数值发散、配置冲突和未分类异常阻断。
状态明确区分 running/completed/blocked/software_missing；报告错误另记 `report_error.json`。
共享 GPU 延续既有近期负载调速、约50%目标、6GiB VRAM/12GiB RAM/12GB disk余量，
不更改其他任务、系统 inotify 限额、原生控制和冻结运行源码。

## 结果、曲线与产物

自动结果区块由后台更新，代表实测产物，最终研究结论仍需复核。
[轻量机器报告](evidence/h24_parameterization_20261103/summary.json)保存
配置/源码 SHA、checkpoint SHA、原始指标、选择依据及软件缺失；
[训练/验证曲线](evidence/h24_parameterization_20261103/training_validation.png)按累计 updates 绘制。
尚未完成的 3000 比较、选型及 5000/8000/10000 结果不预填。
失败阶段只能由实际接触/抬升/子目标与终止原因描述；不从离线 MSE 下降推断闭环成功或因果根因。

本地产物根：`outputs/visuotactile/native_dp_v2/h24_parameterization_20261103_01`。
Epsilon：`h24_epsilon_train/`；Sample 续训：`sample_resume/h24_train/`。
GitHub 仅代码、Markdown、小 JSON/CSV/PNG；数据、视频、权重、大日志及失败 strace 均留本机。
本次恢复不删除任何已有实验产物。后台只写报告，不自动提交/推送未经复核的未来结果。

<!-- h24-results:start -->
当前恢复前状态：`gate_completed`。Epsilon 尚未开始，配对比较和后续里程碑待执行。
<!-- h24-results:end -->

本次恢复新增12项owner/选型/报告回归通过（CPU，0新Kit）；原39项数值验证沿用。
