# 三组 Native-command DP代码交接

从[NATIVE_DP_THREE_WAY_RESULTS](NATIVE_DP_THREE_WAY_RESULTS.md)确认实际状态，再读控制合同与历史架构。
本轮构建独立入口，保留既有采集、controller、DR、奖励/终止、T0证据和原400。

| 文件 | 关键入口与职责 |
| --- | --- |
| native_data.py | prepare核对400来源与split，数值cache，train统计；Episode只按需读取既有RGB/DINO |
| models/native_dp.py | 三组条件编码、Memory生命周期、encode_segment当前权重因果预热、官方U-Net调用 |
| models/vendor/ema_model.py | 固定官方DP来源的EMA；其step计数进入checkpoint |
| native_training.py | network_gate、train、metrics、common_budget；真实优化更新及完整恢复 |
| native_controller.py | 原生tick输入、真实执行ack、直接提交chunk前12命令 |
| sim/native_diagnostics.py | 只读工件过滤接触与native几何诊断，原wrapper/控制数学不改 |
| native_evaluation.py | 220步Oracle、行为中性、固定初态闭环、同次正式轨迹视频 |
| native_resources.py | 复用StudyLease/ResourceGuard，连续整卡均值/功耗/进程记录，磁盘等待 |
| native_pipeline.py | initialize/run/status/resume/stop，子进程/依赖/尝试/hash/退出码/最终产物核验 |
| native_sources.py | 固定revision的小体积公开源码审计，不安装其他方法的训练框架 |

数据调用链：原manifest→HDF5观测/transition→native数值cache＋只读图像来源→
完整episode顺序/96tick段→前缀预热→最多8锚点→四段梯度累积→AdamW/EMA→原子last/best。
没有无界DataLoader或后台prefetch；一条episode图像按需读取，缓存的DINO特征不更新。

checkpoint保存model、EMA、AdamW、schedule update、CPU/CUDA RNG、epoch、段游标、累计锚点与更新数，
以及模型/动作/观察合同、normalizer、split、source/data hash。只从原子last的已完成更新恢复；
未完成梯度累积不提交，恢复重建前缀。退出码0但缺result.json不算完成，防Kit关闭掩盖异常。

```bash
bash visuotactile/run_cpu.sh -m visuotactile.native_pipeline initialize --root outputs/visuotactile/native_dp_3way/<run_id>
systemd-run --user --unit=native-dp-<run_id> --property=Restart=no --property=KillMode=control-group --working-directory="$PWD" bash visuotactile/run_cpu.sh -m visuotactile.native_pipeline run --root outputs/visuotactile/native_dp_3way/<run_id>
bash visuotactile/run_cpu.sh -m visuotactile.native_pipeline status --root outputs/visuotactile/native_dp_3way/<run_id>
bash visuotactile/run_cpu.sh -m visuotactile.native_pipeline stop --root outputs/visuotactile/native_dp_3way/<run_id>
```

显式resume使用同一root和同一deadline；已完成阶段先复验hash再跳过；失败尝试保留新attempt。
2026-11-01修复：截止检查在新attempt创建、checkpoint复制及worker启动之前执行；
Gate截止、总截止和monitor退出分别报告，预启动拒绝保留原attempt原因及退出码。
源码/配置不符、NaN、Gate失败、Kit异常不自动重试。修改源码后不能绕过身份检查恢复正式权重。
禁止以新目录重置同一授权运行窗口；需要修复时先核对是否仍在正式冻结前，并记录具体变更。

首GPU前检查`ffprobe`/`ffmpeg`可执行及动态库，并登记`video_tools.json`中的路径/hash。
PATH没有工具时允许进程级`NATIVE_DP_FFPROBE`和`NATIVE_DP_FFMPEG`指定绝对路径；
本机已验证副本为`/home/CNS2025331827/miniconda3/envs/isaac/bin/{ffprobe,ffmpeg}`。
systemd启动时用`--setenv=NATIVE_DP_FFPROBE=<path>`与`--setenv=NATIVE_DP_FFMPEG=<path>`
传入；不激活该环境或修改共享库。只读工具检查不代表Kit或训练恢复通过。
当前run02因Gate窗口已过而blocked，run03重复启动已在GPU前停止，不照上述模板重开预算。

资源监测贯穿GPU启动至全部owned workers退出。stop只作用于本服务创建的子进程组，不操作无关任务。
systemd用户服务必须实际验证Linger和服务状态；后台脚本没有唤醒Codex的能力。
一次有界启动检查后停止模型监督，待用户重新调用后读取completion/blocked、完整日志与产物。

原生120Hz/60Hz、随机延迟、arm/hand非同义控制、平滑及终止详见控制合同。
恢复的是可观测初态，未完整恢复PhysX内部接触/solver或历史并行随机消耗；不能称精确训练场景复现。
工件接触无法取得时阻断对象抓持验收，绝不以手指距离替代。

正式结果尚不能决定触觉残差基座。不得自动追加H32、残差、SAC或新数据收集。
