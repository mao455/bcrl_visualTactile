# Native DP 全链路执行审计（2026-11-01）

本报告记录 `native_20261101_02` 的一次有界只读审计。审计快照时间为
2026-11-01 20:05（UTC+8）；后台服务继续独立运行，未停止或重启当前 worker，原
Gate、训练和总截止时间均未改变。报告中的“当前”是快照事实，不代表最终实验结果。

## 当前事实

| 项目 | 实测值 |
| --- | --- |
| run / service | `native_20261101_02` / `native-dp-20261101-02-dino-recovery.service` |
| 当前阶段 | `I-OfficialDP_formal`，PID `4163572` |
| 正式更新 | `3130 / 19000`（快照后仍会增长） |
| 最近100次速度 | 约 `1.25 update/s`；长期阶段字段约 `0.75 s/update` |
| 当前阶段保守剩余 | 约 `3.5 h`，按最近100次估计 |
| 已完成正式组 | `S-NativeDP`：`19000/19000`，耗时 `3628.2 s` |
| 已完成 Gate | 数据、两个 Oracle、三组 network/fit/diagnostic/pilot、History perception/network |
| 尚未开始 | Official/History formal 完成前的后续 test40、60 条闭环和学生视频 |
| 训练截止 / 总截止 | 2026-11-02 07:38:19 / 10:38:19（UTC+8） |

`S-NativeDP` 已产生 `best.pt`、`last.pt`、验证曲线和结果文件；Official 当前
`last.pt` 已持续更新，`best.pt` 在最近一次验证后存在。没有依据把当前 loss 或
离线 RMSE称为任务成功率。

## 耗时分解

下表取自各阶段 manifest 的 `started/completed`，不是估算。自主诊断和验证耗时包含
在对应 stage 内；其余 Python 文件 I/O 没有独立计时，因此不伪造更细的比例。

| 部分 | 已耗时 | 当前主要问题 | 修复后实测 |
| --- | ---: | --- | --- |
| 数据及 Oracle | `628.2 s`（数据390.1、Oracle238.1） | 已完成，未重跑 | 无需修改 |
| S Gates | `1162.4 s`（含两个自主诊断） | 已完成 | 无需修改 |
| Official Gates | `2202.5 s`（含失败后的有界恢复） | 诊断/Kit启动曾产生长等待 | 当前阶段已通过，未重跑 |
| History Gates | `539.0 s`（含 perception80.1） | DINO缓存首帧/奇数尾帧已修复并复验 | `400/400` perception、网络 parity 通过 |
| GPU 主动休眠 | 当前 Official worker `138.5 s` | `Lease.block` 在持锁期间等待及pacing；长计算块强制 `max(2.5, ratio)` | 未在冻结 run 中改动 |
| 资源等待 | 当前 Official worker `79.3 s` | 由 VRAM/整卡准入触发；82条历史 `vram_below_4gib` 事件保留 | 未在冻结 run 中改动 |
| Kit 启动及崩溃 | 已记录一次 Kit 初始化失败及有界重试 | 根因未唯一确定；不会无限重试 | 当前服务启动正常 |
| checkpoint / I/O | 由 `work.jsonl` 总计间接覆盖；未单独计时 | 每100更新保存、每次验证保存，且每段反复加载 Episode | 当前 run 不改数学语义 |
| 正式训练 | S `3628.2 s`；Official 当前约 `39 min` | Official 图像编码及重复 I/O 速度约 `0.75 s/update` | 以同一冻结配置继续 |

当前 Official worker 的累计计时为：计算/lease work `2051.0 s`、资源等待
`79.3 s`、主动 pacing `138.5 s`。整卡资源监视器快照为加权均值 `33.10%`、完整
60 秒最大均值 `46.41%`、缺测 `0`；历史监测间隔 `23814.78 s` 未被填零，不能
声称整个墙钟窗口连续监测。当前 RAM 可用约 `35.3 GiB`，项目进程峰值 RSS 约
`5.73 GiB`、CUDA allocator 峰值约 `2.04 GiB`（不含渲染/驱动分配）。

## 已确认的代码问题

以下结论直接来自冻结源码，不是推测。

1. `native_training.train` 在每个四段更新批次内重新构造 `Episode`；构造函数会
   `torch.load` numeric record，并对 History 再次读取 perception cache。Official 的
   `Episode.frame` 还会在每个视觉 tick重新打开 HDF5。按当前 H64/96-tick合同，单次
   更新至少有4次 numeric load，并有多次 HDF5 open；这是可安全做只读 identity cache
   的性能机会，但当前 run 已冻结 source hash，不能在运行中替换实现。
2. `encode_segment` 对 History 每个 segment 将 `first=0`，从 episode 起点用当前权重
   重建因果前缀。该语义是正确的，不能用旧权重 memory shortcut；但 Python 循环和
   CPU-to-GPU 逐tick拷贝构成已确认的计算成本。
3. `long_resources.Lease.block` 在持有 `gpu.lock` 时执行资源等待；退出计算后对
   大于两秒的块强制使用至少 `2.5 * work` 的 pacing。当前 Official 主要块短于两秒，
   因而观测到的 pacing 不是强制上限的完整体现；锁持有范围仍会串行化等待。
4. `native_training` 在 test40 读取 checkpoint 以前就创建 `<group>_test_started.json`。
   若技术故障发生在 marker 之后，现有实现会把同一固定测试 cohort 误判为已开始，缺少
   “checkpoint identity + cohort + completed episodes” 的安全恢复状态。当前 test40
   尚未启动，因此没有污染本 run 的正式结果。
5. 现有 Kit 自动恢复只对有明确 `initializing` 生命周期证据的单次原生崩溃开放一次
   重试；运行期 worker 异常、test 中途退出和闭环逐 episode 恢复仍以 stage 失败为主，
   需要后续 source review，不能在冻结 run 中临时改写。

## 本次安全处理

- 没有停止、重启、发送信号或修改当前训练服务。
- 新增 `tools/native_dp_status.py`，只读汇总 stage、optimizer 更新/速度、checkpoint
  时间、work/wait/pacing、资源均值、截止剩余时间和 stall 提示；它不在冻结源码哈希
  的 `visuotactile/` 目录内。
- 新增一个临时目录状态夹具测试，固定数据下验证速率、ETA、资源和 work 分解。
- DINO、History prefix、checkpoint 读回、ffprobe 等既有 Gate 证据继续沿用；没有放宽
  阈值、改动作合同、改物理频率、改变随机种子或新增训练预算。

## 预算判断

冻结配置记录共同正式预算 `19000` updates。Pilot 仅用于三组共同预算计算，不能替代
正式速度；实际 S 已在 `3628.2 s` 完成，Official 当前阶段按约 `0.75 s/update`
保守外推仍可在训练截止前给 History 留出时间。是否能在总截止前完成 test40、60 条
闭环和四视频，取决于 Official/History 实际速度及 Kit 启动，不在本快照承诺。若后续
预算不足，流水线必须保留真实 checkpoint 并写 `blocked` 或 `incomplete`，不能伪造
完成或降低 `5000` 更新门槛。

## 查询与复现

只读阶段命令：

```bash
bash visuotactile/run_cpu.sh -m visuotactile.native_pipeline status \
  --root outputs/visuotactile/native_dp_3way/native_20261101_02
python tools/native_dp_status.py \
  --root outputs/visuotactile/native_dp_3way/native_20261101_02
```

后台服务由 user systemd 管理，不依赖 Codex 会话。最终结果只在出现
`completion.json` 或 `blocked.json` 后读取；在此之前不把进程存活当作实验完成。
