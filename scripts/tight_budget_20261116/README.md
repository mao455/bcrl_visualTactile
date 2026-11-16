# 固定训练分布的预算诊断

当前入口 `run.py prepare`（仅一次）、`run.py train`。使用项目Isaac Python；先启动同root的`python -m visuotactile.state_resources --root ROOT`监控。新根tight_budget_20261116_01，不改旧19。所有阶段心跳/命令/PID在logs；六小时固定截止，异常保存failure.json，重启显式复用last且最多两次attempt，不自动采集。

独立验证6初态；348train保持同分布，最新4条repeat12；DP从19 last8完整恢复到至多24，后端固定新DP重新生成输入，以原来源相同参数/种子/新Adam初始化，保留第8轮best和末尾best；不存在后端EMA。独立验证四次无0.5%改善则LR减半，之后四次且至少16遍则停。所有细节/结果维护于现有TIGHT_TRAINING_DIAGNOSIS_20261116.md。

源检查点和有效数据不删；只新增最后/最佳/后端第8轮比较快照及新DP后端缓存，不保存每轮全模型。mapped冻结视觉无新缓存副本。31聚焦测试通过；是否已运行看root实际状态。

正式训练冻结树76352bd，root session记录固定截止。`align_frontends.py`只调度严格命令绑定的两个自有DP进程，在同deck上保持至多6个batch进度差（2时恢复），复用OS页而不改参数/数据/随机流。每次暂停最长10分钟，finally恢复；SIGTERM转异常走finally。首次32KiB尾读取不够容纳A几何报告已退出，无signals；正式调度用4MiB完整行尾部。服务内存软/硬34/40GiB，研究实际10GiB RAM余量检查保留。`evaluate.py`在训练与离线结果完整后执行16环境×最多3组短闭环，最多48次，原8轮完整模型/新DP+rear前8轮best/新DP+后端best；重复权重组省略，不采数据。`report.py`只画实测点。

当前恢复入口：`recover_serial.py`（一次性，先核对原owner退出与完整last11/2156）。正式训练仍76352bd，A/B顺序GPU；`read_ahead_serial.py`使用单个16MiB buffer预读当前/下一B2的冻结文件，仅暖OS页、不生成缓存副本。第12轮原并行OOM和各140丢弃更新保留；本次没有全程守住显存余量。源failure.json是历史失败，当前看serial_recovery_started/failed/complete、logs各attempt及实际systemd。不要重启原并行train入口。完整截止未改，第二次失败不自动重启。
