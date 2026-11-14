# A/B 同轨迹读取窗口：可回退训练 I/O 候选

`paired_episode_io_v1` 只让两组独立前端在同一条训练轨迹开始、结束时会合，缩短它们读取同一冻结视觉文件的时间差。默认关闭，尚未取得真实训练加速结果。它不合并模型、不减少训练次数，也不缓存可学习的视觉投影或 GRU 隐藏状态。

## 已有证据与选择理由

只读检查源为 `tight_fresh_20261114_05/round_06`；完整小型结果见 [CPU I/O 检查](evidence/tight_fresh_20261114/paired_io/feasibility.json)。检查读取 8 条训练数值轨迹及冻结分片元数据，没有读取验证/测试原始画面，没有启动 CUDA。

- 93 个训练分片合计 48,061,829,069 字节（44.76 GiB），A/B 的 93 个对应文件已经全部共享 inode。因此它们本来就共享操作系统页缓存，不是两份独立物理缓存。
- 两组实际 1,020 次 episode 访问顺序完全相同：每轮 93 条旧/新轨迹，加 9 条新纠偏的额外访问，共 102 次，10 轮。同步不生成新的训练顺序。
- 每组按完整分片大小计算的 10 轮逻辑访问量为 521,146,801,280 字节。这不是实测磁盘读取量，也不是可以节约的确定字节数。该旧轮日志没有新版本的进程 I/O 计数，不能补造逐轨迹物理读盘归因。
- A/B 的训练日志累计 episode 耗时分别为 1,151.04/897.29 秒，完整前端阶段分别为 1,519.60/1,424.67 秒。进度错开可能降低页缓存复用；会合也可能让较快的 B 等待 A，因此必须实测总完成时间。
- 冻结 RGB patch 为 FP16 `[T,3,384,18,32]`。投影中的 `1×1 Conv → GroupNorm → SiLU → pool` 可学习，不能把提前池化的冻结 patch 声称为等价替换。一个 522 tick 轨迹仅 FP32 patch 输入就约 1.29 GiB；跨轨迹训练同时保留完整计算图还需要更多显存。

真正的多轨迹模型 batch 暂未实现。它需要每条轨迹独立的完整历史、padding mask、合法专家目标、几何抽样和逐轨迹 loss 归一化。两条轨迹合成一次 Adam 更新会改变更新次数、EMA 和参数路径；CPU 小例中这种改法已产生不同参数，不能称为原训练的等价提速。

## 接入及边界

在已有新周期命令上添加 `--parallel-frontends --paired-episode-io`。普通命令不添加第二个选项时沿用旧训练路径。选项登记在 cycle registration，并在视觉缓存完成、双前端共同资源准入通过后，写入输出根的 `paired_episode_io.json`。资源不足时明确失败，不能回退成带屏障的串行训练。

登记要求两组有完全相同的有序训练 ID、采样倍数、冻结缓存 SHA/来源/时间戳身份，并且对应数据文件的 device、inode、大小、mtime 一致；每次训练读取前再次检查该分片 stat。只读取小索引和 stat，不为了登记把整套 48 GB 缓存重新扫入 RAM。此检查依赖既有缓存完整性验收，不能替代缓存最初的 SHA 校验，也不是针对恶意保持大小/mtime 的文件篡改检测。

每组仍独立执行原有步骤：完整真实 episode 的视觉和 GRU 重算、合法专家 mask、每次 visit 一次 backward/Adam/EMA、自己的噪声 RNG 和模型参数。局部生成器只在登记时重建期望 deck SHA，不消费训练生成器或全局 Torch RNG。每个进程只为自己的 ready/done receipt 写文件；不新增大 tensor 缓存。现有可选 1 GiB 冻结 episode prefetch 仍使用自己的上限，两个选项可分开验收。

`paired_episode_io/{A,B}/status.json` 持久记录状态、PID、Linux process start ticks、worker token 和登记摘要。`visit_*/{ready,done}/{A,B}.json` 记录实际会合点，`{A,B}/waits.jsonl` 记录每条轨迹的等待时间。默认启动会合上限 300 秒，后续每次会合上限 600 秒，轮询 20 ms，同时受原 session 绝对截止约束。底层登记函数允许显式设置有限等待；不自动延长会话。

同伴显式失败、退出、成为 zombie、PID 被复用、deck/receipt 不匹配或等待超时均使当前进程失败。正在进行的 episode 计算不会被后台线程强杀；它在下一个会合点发现同伴错误，已完成的更新与日志保留。正常结束的同伴留下的精确最后 done receipt 可以被读取。最终验证失败仍会使外层双前端阶段失败，不能开始依赖训练。

当前版本明确拒绝配对训练自动恢复：已有配对目录、已开始训练或 checkpoint 不得自动跳过/重复 episode。失败后保留现场，由新的显式实验身份决定如何处理。该限制不改变未启用选项的旧恢复路径。

## 已验证及下一项实测

CPU 检查 26 项通过，包括两个真实 spawned 进程的完整 GRU 历史训练：与各自未同步参考相比，参数、Adam 状态、EMA、访问顺序、噪声/调度/全局 Torch RNG 逐项完全一致。另覆盖缺失同伴、异常退出、显式错误、活着但卡住、绝对截止、错误 deck、修改缓存、恢复拒绝、默认关闭和 cycle 准入后登记。这不是 CUDA 数值一致或真实模型吞吐验收。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv_isaacsim/bin/python -m pytest -q \
  tests/test_paired_episode_io.py tests/test_episode_io.py \
  tests/test_dagger_parallel_frontends.py tests/test_fresh_cycle.py
```

下一次有限真实比较需保持相同数据、起点、epoch/deck/更新预算，分别记录双前端共同完成 wall time、每组更新与曝光、epoch `process_io`、ready/done 等待、RSS/显存和可用 RAM。`learning.jsonl` 的 episode `seconds` 包含 ready 等待，不包含在该行落盘之后发生的 done 等待；全 epoch wall time 和单独 waits 记录包含后者。OS 页缓存初始冷热状态必须披露，不能把暖缓存改善单独归因于屏障。只有真实总吞吐和后续完整采集→训练→自主执行结果支持，才将候选作为正式默认。

## 有限真实比较入口（尚未运行）

`tools.tight_fresh_paired_io_study` 使用 round06 原来的 93 条 train、6 条 validation、0 条 test，以及原第五轮 continuation checkpoint。CPU prepare 对起点 checkpoint、数值数据和小型配置作 SHA 绑定，对已经验收的 99 个冻结视觉分片作索引 SHA 与 inode/stat 绑定；两 arm 链接同一只读输入目录，不复制缓存、不预热视觉 payload。准备时读取起点 checkpoint 进行 SHA 检查，本身会影响这些 checkpoint 的缓存冷热状态，不能称为系统完全冷缓存。

在包含本工具的审核后冻结工作树中执行下面两步。第一步仅 CPU 准备，第二步才启动 GPU 训练；主 Agent 统一调度第二步。`paired_io_compare01` 必须是全新目录，工具拒绝覆盖或恢复既有尝试。

```bash
.venv_isaacsim/bin/python -m tools.tight_fresh_paired_io_study prepare \
  --source /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_05/round_06 \
  --output /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_05/paired_io_compare01 \
  --max-seconds 720
.venv_isaacsim/bin/python -m tools.tight_fresh_paired_io_study run \
  --output /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_05/paired_io_compare01
```

普通双前端先运行，同轨迹同步双前端后运行。每 arm 各组一完整 epoch、102 次更新，共 204 次更新；两边都禁用 prefetch。owner 复用现有 `paired_frontends` 和正式 `tight_fresh_train train`，不复制训练循环。原 session05 截止不变；总计算/回收窗口为启动后至多 720 秒与原截止的较早者，其中预留最后 20 秒用于停止自有子进程。SIGTERM/SIGINT 或本次输出根的 `STOP` 会请求停止本次子进程。普通 arm 失败后不启动同步 arm，失败不自动重试。

`owner/<arm>/*_worker.json` 和阶段 receipts 保存真实 PID/start ticks、命令、退出原因与最后存活进程 I/O/RSS 采样。最后进程采样可能早于退出，不能作为精确全进程读取量。每组 epoch 的精确 `/proc/self/io` 差值、开始/结束 UTC 与共同 epoch 区间用于报告更新/秒；另报完整 arm wall time（含初始化、验证、保存和小摘要）。sum read_bytes 是两个 worker 的内核归因之和，不是独占设备 I/O 测量。

完整 epoch 的 `last.pt`、`best.pt` 和已有日志原样保留；若在 epoch 中断，只有先前已经保存的完整 checkpoint 可以恢复，部分更新日志不等于已保存参数。工具明确不恢复该诊断。生成 checkpoint 的大小/stat 与 trainer 已报告的 best SHA 会被登记，不在两 arm 间额外扫描数 GB checkpoint，以免改变下一个 arm 的页缓存状态。工程诊断产物标记 `deployable=false`，不进入自主成功率或正式模型选择。

新增 owner/input CPU 测试连同配对训练、预取和 cycle 测试共 32 项通过；其中实际启动一个 CPU 睡眠子进程，并验证有界终止、回收与失败凭据；另验证 owner 收到 SIGTERM 后保留完成 arm、阻止下一 arm 启动并恢复信号处理器。未启动真实训练，未证明收益。
