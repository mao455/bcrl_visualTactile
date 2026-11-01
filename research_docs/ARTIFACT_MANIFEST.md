# Native-command产物登记

正式新root统一为 `outputs/visuotactile/native_dp_3way/<run_id>/`。当前状态见结果报告，
未完成的训练、checkpoint、视频不填写虚构路径、大小或下载链接。大型产物仅本机保留，不上传Git。

当前有效run_id为`native_20261101_02`；原run01的FileThread失败及误启动run03均保留，run03不续跑。
恢复运行源码commit为`902b4b1`，准确文件身份由session逐文件hash和pre_freeze_source_reviews保存。
数据、两个Oracle、S/Official各自Gate与late-gate preflight已有16个完成stage；有Gate checkpoint、
自主诊断与专家视频，正式训练/测试产物尚未产生。阶段最新状态以结果报告与实际manifest为准。

| 产物 | 位置与验证方式 |
| --- | --- |
| 运行合同/源码/保护对象 | session.json；实际Git HEAD与逐文件SHA256同时保存 |
| 阶段、命令、PID、attempt、退出码 | manifest.json；result和完整数量核对后才complete |
| 400数据/固定split/缓存来源 | data.json；原HDF5、DINO缓存hash与新数值cache逐项记录 |
| train-only normalizer | statistics.pt；hash进入data与checkpoint合同 |
| Native冻结视觉数值合同 | native_perception.json；原cache hash、400份复现结果、独立奇数尾帧overlay索引 |
| DINO修复证据 | dino_probe_20261101.json及dino_causal_probe_20261101_v2.json；原计算对照与完整首train采样 |
| 正式best/last | stages/<group>_formal/attempt_N；原子写入，恢复不改旧attempt |
| 验证选模冻结 | selected.json；三组路径及SHA256 |
| 离线/闭环结果 | 各stage result.json，trace.npz，plans_h64.npy，关键帧 |
| 专家主视频 | oracle_instrumented当前成功attempt/01_expert_success.mp4 |
| 三个学生主视频 | formal_evaluation首个预登记test ID/student.mp4 |
| 曲线 | verify成功attempt/training_curves.png及svg |
| 资源 | resources/resources.jsonl、power_processes.jsonl、resource_incidents.jsonl、monitor_exit.json |
| 完成/阻断证据 | completion.json或blocked.json；不以PID存在替代产物验收 |
| 最终逐文件清单 | verify成功attempt/artifact_manifest.json，真实路径/bytes/hash/视频probe |

引用保护：原400 manifest及HDF5、B0/B1/G00/G10关键best与正式报告、T0新220步成功源及历史必要证据。
公开代码不带原始轨迹、权重、视频、日志、环境或私有配置。后续收尾报告记录真实commit与远端hash。
本轮公开小证据为`research_docs/evidence/native_dp_dino_recovery_20261101.json`；新尾帧overlay属于
本机派生缓存，保持ignored，不能代替或覆盖旧实验的DINO缓存。
