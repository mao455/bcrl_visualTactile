# 2026-11-15 新研究时段启动包

这是一份待启动的静态配置，不是已经创建的 session、验收或训练结果。只在完成第九轮完整训练与自主执行、旧 owner/所有旧 Kit 和 trainer 退出、旧 monitor 写出 `monitor_exit.json` 后使用。脚本不停止旧进程，不创建冻结工作树，不自动重试任何已有身份。

从 root 审核并冻结后的工作树运行。`FROZEN_TREE` 和 `FULL_COMMIT` 必须替换为实际路径与完整 40 位提交 SHA；不要把开发树当前 HEAD 当成未来冻结版本。

```bash
FROZEN_TREE=/absolute/path/to/reviewed/frozen/tree
FULL_COMMIT=replace_with_reviewed_40_character_commit
"$FROZEN_TREE/.venv_isaacsim/bin/python" \
  "$FROZEN_TREE/scripts/tight_fresh_20261115_01/launch.py" \
  --worker-tree "$FROZEN_TREE" --expected-commit "$FULL_COMMIT"
```

上面的默认操作只校验静态配置并显示准确 argv，不创建 session，不检查尚未完成的第九轮为成功。确认准备启动时，在相同命令末尾加入 `--launch`。这一个入口按顺序执行已有 `tight_fresh_followup_session --hours 12 --allow-illegal-starts`、新 monitor、loop prepare 和 systemd owner；不会因失败重新计时。

来源固定为 `tight_fresh_20261114_06/round_09` 及其 `development`。必须保留并接受已核验的零动作非法初态记录，不替换初态。新根为 `tight_fresh_20261115_01`，12 小时从 followup 真正创建 session 时起算；旧时段截止不改变。首轮以完整 A09/B09 为研究继续来源，之后使用上轮完整新候选；这不是宣布其优于保留的 A05/B06 回退模型。

配置安排三个有限轮次 10–12，每轮 32 个新训练初态家族、A/B 共 64 次机会。新训练 seeds 分别为 29000000–29000031、29000032–29000063、29000064–29000095；event seeds 为 29100000、29101000、29102000。采集循环为 `fixed80` 四次、`held0` 两次、`held_root60`、`contact0`。验证原六家族保持不变；自主开发 cohort 每角色仍为 28500000–28500007。

采集使用已测 GPU16 standard 后端、Kit 16 线程/物理 2 线程、M8、512 MiB 有界写入、2 校验 worker/pending16、capture audit memo。此 GPU 配置目前只有前 180 步的接触/控制/传感证据，完整专家救回由正式采集逐条接受原生成功与完整 capture 审计。没有虚设的“两个成功后自动放行”条件，没有附加 benchmark。任务资产、120/60/5 Hz 和最终 29 维绝对关节目标不变；GPU 与旧 CPU 的 batch RNG 不同，不承诺逐位重复或公平加速比。

每轮前端 20 epochs、完整双轨迹 B2、独立 A/B 参数与真实完整历史；训练采用随机合法 anchor 加真实构型约半采样。后端 continuation 15 epochs；复用来源的冻结视觉 index，只生成缺失的新条目。离线前后端计划批量均为 M32。A/B 前端顺序执行，双轨迹 B2 不变：当前最长实际训练轨迹为 865 帧，重复采样时同批可达 1730 帧，超过已测 761 帧组合；现有实现没有按下一批激活峰值准入，因此先避免两套前端共同放大内存峰值，这不是已测吞吐更快的结论。后端并行仍使用既有资源准入，可回退串行。冻结视觉特征显式采用 frozen_visual_eager_v1，直接从文件顺序载入原生张量；只保留当前 B2 的两条轨迹，离线评估与后端输入准备各保留当前一条，不增加 clone、常驻 LRU 或可学习历史缓存。该 I/O 身份和输入文件 SHA 随 cycle 登记，统一入口对一条真实冻结特征做 CPU 字节比较后再冻结启动；不改训练目标、采样次数或模型身份。自主评估保持 CPU standard、Kit16、8 lane/M8，对每角色八次 cohort 无 16 lane 填充。没有使用 TiledCamera、inactive-render 或 exact-tail 候选。

每轮入场需要 110 GiB 可用盘和至少 10800 秒；采集最多 7200 秒，并保留至少 7200 秒给完整 cycle。沿用原生任务自然结束与既有至少八条成功纠偏的训练准入；剩余时间、有效数据或磁盘不足就保留现场停止，不要求强行完成三轮。整个新 session 的模型/数据原始资源规则与 60 GiB 硬保留继承不变。

启动历史写入新根同级 `tight_fresh_20261115_01_launch/`，包括完整命令、日志、源提交、bundle SHA、旧进程退出证据和 systemd 实际属性；新根 `launcher/` 保存最终展开配置与截止绑定。`research_loop/` 记录每轮每阶段状态、结果和失败。owner 和 monitor 分别用新 systemd unit，均 `Restart=no`、`KillMode=control-group`、`LimitNOFILE=65536`，实际 `RuntimeMaxSec` 只取新截止剩余时间。owner 结束后 `ExecStopPost` 仅写新根 `STOP_MONITOR`，新 monitor 正常收尾；不会杀旧作业或无关进程。启动脚本最多等待 monitor 45 秒、owner 30 秒，只确认启动，不将后台已启动视为工作完成。

若发生失败，不直接重跑此入口。先读 launch/status、research_loop/owner_finished 和对应阶段日志；原记录、全部失败轨迹及部分检查点保留。后续恢复需要依据现场证据建立显式新身份，不能删除目录、改 session 时间或跳过完成条件。
