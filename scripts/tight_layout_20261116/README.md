# 仅平移插销起点的完整任务验证

原位、远移5cm、远移10cm九个登记机会全部关闭。10cm三例合法且原生全任务成功；原位3/3，5cm2/2合法成功和一次保守起点拒绝。真实抓前碰撞几何支持消除部分落桌时的基座干扰，学生收益尚待完整训练/自主评估。视频保留在原pilot各case的rollout.mp4；详细结果和限制见原有TIGHT_TRAINING_DIAGNOSIS_20261116.md。

当前数据root为`outputs/visuotactile/run/tight_layout10_20261116_01_recovery02`，冻结9b0c0a3；原两条成功来自`..._01`，未重复执行。`..._01_recovery01`保留启动前资源路径错误，0新物理机会。原17:02的24小时总界、3小时采集界均不重置。采集owner仅接续尚未开始的预登记初态，保存所有拒绝/失败。

学习使用`.../tight_layout10_20261116_learning01`，使用本次冻结提交；`tight-layout10-learning02-20261116.service`先等采集owner/monitor与数据审核完成，再串行冻结视觉、实际输入gate、A/B前端、刷新各自后端输入、后端、配对完整任务评估。服务启动不代表训练开始或成功。A/B共同train30/validation6，从随机学生权重开始。当前架构、随机合法规划时刻/半数几何/B2、原生120/60/5Hz及最终命令语义保持。开发28820000–003独立于训练28817000–039、验证28818000–005与最终测试28819000–007。

只读状态：`bash visuotactile/run_cpu.sh -m scripts.tight_layout_20261116.readout`。先核对真实服务/PID和`failure.json`，不要重启已有root。正式数目只读闭合collection_results/finished/result，独立测试不提前使用。首轮完整结果由主Agent审阅后继续共享池DAgger；后台脚本不作模型有效性宣称。

`cleanup.py`对应已执行的单次清单，净释放135318011904bytes；不得重算或重跑删除。原400、所有有效成功源、依赖manifest源、完整最好与必要回退、专家/通用视觉权重保持。完整逐文件清单在原pilot root/cleanup，公开精简凭据在research_docs/evidence/tight_layout_20261116。
