# Git delivery manifest

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
