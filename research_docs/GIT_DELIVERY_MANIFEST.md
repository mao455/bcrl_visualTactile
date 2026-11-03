# Git delivery manifest

## 2026-11-03 StateDP恢复与已有运动证据

当前feature/native-dp-training-v2基准739ad84。新增tools/state_motion_recovery.py和
tests/test_state_motion_recovery.py；171项冻结训练/runtime源码及checkpoint身份保持不变。
34项CPU测试通过，实际启动核验H24首阶段250/1000updates；不声称后续训练完成。
12tick记录器错误及24tick场景前中止均保留为missing；没有追加启动重试或延长deadline。

同步现有AGENTS/STATE_TRAINING_RESULTS/SESSION_HANDOFF/CHANGELOG，不另建研究框架。
evidence/state_motion_20261103新增recovery_startup_review.json、已完成C0/C2及sampling/
overlap聚合CSV/PNG/summary、资源曲线和单milestone训练点。summary为14:16停止快照，
后续14:23运行核验单列；公开路径去标识、LF换行及单点marker变换有hash记录。
只交付小型报告/源码；所有权重、HDF5、NPZ、视频、失败attempt与本机注册留原位置。
后台后续自动输出不自动提交，必须下次核验实际状态和产物后再发布。

## 2026-11-03 H32位置：最终证据交付

基准f1953c7，沿用feature/native-dp-training-v2。实现提交
`c7fcbb0e2d27e21fa5c818171014413d2b067d0a`，启动快照817d885，短对照结果
`c5b0fa747887a93f1f3551ce5f051077f3968025`均已推送origin；它们不代表最终研究成功。
本节所在Git提交交付04:45:10UTC+8结束的实际结果；最终提交SHA由交付回复及Git历史给出，
运行源码仍以session登记hash为准，报告提交不追溯改变实验身份。

NativeH32/PositionH32完成1000/19000 updates，最终各0/12。位置2lift、1train首子目标，
没有完整自主成功。Image未触发，H24未执行，无新BC。服务和全部自有进程已退出。
核心文档：POSITION_ACTION_CONTRACT、KIT_RELIABILITY、STATE_TRAINING_RESULTS、本清单；
AGENTS/SESSION_HANDOFF/CHANGELOG同步状态，不新增另一份结果报告。

最终小证据在`evidence/position_20261103/results`：完整累计训练指标、各cohort结果、
冻结后test40、注册配置、接口审计、87条trace诊断摘要、15视频解码hash和两张曲线。
重复的累计训练快照无需再提交，原始outputs完整保留；通用旧图把位置误差标作Native RMSE，
不交付该图，使用注明profile单位的position_curves.png。completed_review.json核验
170源码/119stage/2best/15视频hash；前次89测试通过，本次无源码修改及重复测试。

权重、原400、MP4/HDF5/NPZ和outputs链接不上传。全部best/milestone与原400仍保留；
先前依赖审计只删4份逻辑重复last和1份旧可重建cache，共2,512,265,377bytes，
逐文件SHA与保留对象见cleanup.json。本次不追加实验载荷删除。


## 2026-11-02 full State execution delivery

Implementation `c2e9895e4b17259f08648074417f531b19f474c2` is pushed and matched to
origin/feature/native-dp-training-v2. Adds state_resources, state_full_training,
state_full_run and targeted tests; extends shared benchmark/report without changing
model/normalizer/native task. Prior failed root02 final small results are preserved.
Current service: `state-full-20261102-01.service`; root `state_full_20261102_01`.
Registered start17:12:42UTC+8; training cutoffNov3 14:12:42; absolute endpointNov3 17:12:42.
The initial malformed systemd command failed argument parsing before a worker existed;
its journal is retained. Corrected service uses the same root/deadline, no trial repeated.
Initial verification: active owner, singleton expert stages advancing, telemetry owns
one GPU worker and recognizes the other project separately. This is not new student success.

```bash
bash visuotactile/run_cpu.sh -m visuotactile.state_full_run run --root outputs/visuotactile/native_dp_v2/state_full_20261102_01
```

The run command locks the owner, verifies source/config hashes, skips completed stages,
and permits one legal trainer checkpoint recovery. Do not re-initialize an existing root.
The owner exports curves, first divergences and video decode results at completion into
`evidence/state_shared_20261102/results` and updates STATE_TRAINING_RESULTS locally.
Future generated results require review before a separate commit/push; weights/raw videos remain local.


工作树：`bc+rl_RSS_native_v2`，branch `feature/native-dp-training-v2`，基线及启动时origin均
`ec84bd3ac7283c79da068c19506ed8d7bb20a364`。原`bc+rl_RSS`工作树保持独立，不提交其文件。

## 必要运行源码

| 新文件 | 用途 |
|---|---|
| `visuotactile/state_history.py` | State v2 Actor/GRU/完整前缀/online controller |
| `visuotactile/environment_rng.py` | opt-in环境CPU RNG隔离 |
| `visuotactile/state_benchmark.py` | 初态真实shape检查、专家/学生共享原生step及trace |
| `visuotactile/state_history_audit.py` | 全400Actor与真实train/live接口数值审计 |
| `visuotactile/state_history_training.py` | 受Gate限制的小BC/DP训练与validation选模 |
| `visuotactile/state_history_run.py` | 登记、固定预算、监控与阶段owner |
| `tests/test_state_history.py` | 真实历史/随机流/失效查询拒绝回归 |
| `tools/state_history_report.py` / `state_history_finish.py` | 只读小结果/曲线/视频校验与owner结束后自动整理 |

依赖均使用已有版本：torch、diffusers、h5py、numpy、Isaac Sim/Lab、rl_games及repo已有vendor U-Net/EMA；
没有新外部训练服务或共享环境升级。完整必要旧依赖已在repo中；`.gitignore`未排除新增源码、测试和五份MD。
`outputs`和`.venv_isaacsim`为本机untracked symlink，禁止git-add；大型HDF5/checkpoint/视频/NPZ留本地。
`.gitignore`新增根路径规则，目录规则原先不能覆盖symlink；当前两者已明确ignored。

外部专家：`pretrained_assembly/tight_insertion/model.pth`，SHA256
`7e61015dad660430ab07bab4f715c90c6f76b84cc358da6be81319a6aee09990`，使用原项目released expert获取路径，
新模型仅复用其固定输入RMS和静态group embedding，不导入LSTM hidden。数据沿用原400采集清单和320/40/40 split。
源`data.json` SHA256 `abe7a725ae1be6af6fefba9d3a92809e0667d95137da565447afeeb22576ca64`；
`statistics.pt` SHA256 `65030eef5ae5c89cec4992af828e27d760c422f922be73fd52ee6743ca5cc7e9`。
实际资产/运行库hash见[evidence runtime_assets](evidence/state_history_20261102/runtime_assets.json)，
包含安装Lab源码commit、200个core源码hash、PhysX插件hash、完整相关资产树。
外部Lab工作树非干净：相关DirectRLEnv有两个optional diagnostic callback hook，当前研究env未定义它们；
相关10行patch作为[小型外部依赖差异](evidence/state_history_20261102/isaaclab_optional_diagnostic_hooks.patch)保存，
用于复现已安装文件hash，不要求更改当前共享安装；
不以git commit单独代表实际安装，也不上传该工作树其他项目的修改。Lab安装metadata版本0.54.3，与先前文档2.3.2称呼区分。
专家获取：HuggingFace `Kushal20/play2perfect-checkpoints`，`python download_checkpoints.py --what assembly --problems tight_insertion`；
下载工具未固定远端revision，因此必须按上述SHA验收，不能接受同名更新权重。
新结果小证据在最终整理时补入同一evidence目录；不上传权重或复制上游大源码树。

## 运行入口

```bash
bash visuotactile/run_cpu.sh -m pytest -q tests/test_state_history.py
bash visuotactile/run_cpu.sh -m visuotactile.state_history_run initialize --root <fresh-root>
bash visuotactile/run_cpu.sh -m visuotactile.state_history_run run --root <fresh-root>
```

旧root不能直接重启；复现实验需新独立身份，不代表扩大本轮预算。当前先提交已审查实现与明确pending的报告，最终实验结果仍待资源释放；
普通finish服务结束后只更新本地报告/小JSON，不未经审查自动提交。
最终实验结果review/commit/push与远端一致性仍待本轮结果收尾，
不将base commit冒充最终交付commit。所有实际launch保存source SHA，commit后的身份不会追溯覆盖运行时源码。

## 已推送实现身份

实现commit `89b933462f72e0ec46f2d3ab603ae27448126744` 已推送origin/feature/native-dp-training-v2，并核对远端SHA相同。
当前实验仍因资源准入等待，尚未完成闭环研究交付；后续自动产物须人工复核后另行提交。
本记录后的文档提交以Git历史为准；不把该实现commit称为最终实验完成commit。
