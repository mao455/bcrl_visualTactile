# Native-command产物登记

正式新root统一为 `outputs/visuotactile/native_dp_3way/<run_id>/`。当前状态见结果报告，
未完成的训练、checkpoint、视频不填写虚构路径、大小或下载链接。大型产物仅本机保留，不上传Git。

已登记本轮run_id为`native_20261101_01`；实验登记时source commit为`a905bcaa3a2ecc444681c9074c5864c5855441b6`。
当前工作树包含本轮实现和文档，提交身份以本次审查后的commit为准；运行源码的准确身份由session逐文件hash及
pre_freeze_source_reviews保存。数据阶段已完成，`oracle_base`因原生 FileThread abort 阻断，未产生训练checkpoint、闭环或视频。

| 产物 | 位置与验证方式 |
| --- | --- |
| 运行合同/源码/保护对象 | session.json；实际Git HEAD与逐文件SHA256同时保存 |
| 阶段、命令、PID、attempt、退出码 | manifest.json；result和完整数量核对后才complete |
| 400数据/固定split/缓存来源 | data.json；原HDF5、DINO缓存hash与新数值cache逐项记录 |
| train-only normalizer | statistics.pt；hash进入data与checkpoint合同 |
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
