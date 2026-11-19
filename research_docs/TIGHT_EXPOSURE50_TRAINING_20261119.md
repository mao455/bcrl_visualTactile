# A/B四组件逐合法tick充分曝光：已登记，尚未完成

本任务从943913c接续，原72完整A/B分别初始化，固定130训练轨迹（原30＋100纠偏）和独立validation6。禁止新encoder、DAgger、RL、任务/控制/成功标准修改。新root为`outputs/visuotactile/run/tight_exposure50_20261119_01`，源实验身份不复用。此文初稿不是完成报告。

四组件均新增训练，不累计父模型曝光。预登记60完整deck，原30各3访、其他100各1访，共190visits/epoch；DP B2仍每遍95更新。DP保留原grasp随机抽样，再用最早未覆盖合法tick的H24窗口补齐；仅真实合法标签、原segment/window mask，不新增反事实监督。每次优化后对实际loss窗口中的identity/tick取并集，计数一次。后端沿用完整历史、32tick截断和固定128合法tick量级累积，按实际正权loss_mask收集，optimizer提交后去重计数。零误差也属于正权监督参与；不是每tick非零梯度的声明。

A仍q_reached及原几何项，B仍u_expert。补齐窗口改变采样人口与既有几何抽样候选数量，但不改几何算法/权重、动作接口或未来信息边界。DP3e-5，后端1e-4，fresh AdamW；DP从各自72 EMA初始化并重建新EMA，后端从对应72后端初始化，无后端EMA。每组件最多一次学习率减半：40遍后验证连续8次未提升0.5%。60遍前不早停。只在本轮最少访问≥50且每合法tick不同optimizer step计数≥50的checkpoint中，按原validation6等权normalized arm/hand MSE最小选择，平分保留较早者。不会把49遍模型写成最终候选。

先训练两个DP，再各自刷新136条nominal缓存、训练相应后端；不得混用父或另一DP缓存。固定原30统计。实际全池预检A/B均27,508合法专家tick，全部可被合法自身时刻输入窗口覆盖，零额外排除。逐轨masked索引、标签/父权重/统计身份见registration.json。合成含gap/段界测试、去重/恢复/49次拒绝测试和相关回归67项通过；尚不代表GPU完整训练完成。

DP每完整epoch保存模型/EMA/Adam/随机数/曝光；后端每完整轨迹组保存。未持久化尾更新不计恢复模型，原尾日志另存。修复正式hotset配置在restore比对之后才加入导致的配置误判，数值路径不变。

存储预检最初约12.85GiB空闲，无法容纳四组件恢复断点。36个已索引、无活动句柄、当前136缓存以外的旧视觉缓存无损Zstd归档，节省1.4238GiB；另24个已完成历史训练日志无损归档，节省5.4619GiB，全部完整解压SHA验证后替换为压缩档。原400、当前130、任何模型/optimizer checkpoint均未删除或改写。日志清单26行含两处旧路径别名重复，首批9个后中断，按真实路径去重继续，证据保留。归档的历史日志/缓存需显式恢复后用于旧run恢复或审计，不能宣称旧路径仍可直接读取。

```bash
python -m tools.tight_project train --job configs/experiments/exposure50/train.json
python -m tools.tight_cache_archive restore --manifest research_docs/evidence/tight_exposure50_20261119_01/cache_archive_manifest.json
python -m tools.tight_cache_archive restore --manifest research_docs/evidence/tight_exposure50_20261119_01/training_log_archive_manifest.json
```

训练作业是一次性已登记身份，不能重复启动。故障后保留原job回执，以新owner attempt、相同根/截止/模型状态恢复。24小时总资源session为执行边界，不把deadline当曝光完成条件；未达标必须报告实际缺口和恢复入口。保留12GiB磁盘、10GiB RAM、4GiB VRAM余量，整卡full_gpu_v1，保护其他任务。

完整四组件达到门槛之后，登记原训练8＋开发8、每family四完整策略（72A/B、新130A/B）共64自主机会；保持原生成功，hold30/60/120仅诊断。未开始闭环，不从历史分母拼接结果。预登记视频初态28817004、28826004；失败样本不删除。正式状态以后续真实checkpoint、逐步计数、结果回执为准。

实际启动：tight-exposure50-20261119-01.service执行统一训练队列，finish服务在四组件训练全部完成并通过曝光重放门槛后才登记/执行64例评估；失败则保留并退出，不自动放宽门槛。136原始HDF5全SHA和A/B标签逐值检查通过。A-DP epoch3/update285实际断点计数与成功更新日志独立重放逐项相等，最低tick曝光3，尚不达标。全部状态仍以现场文件为准。
