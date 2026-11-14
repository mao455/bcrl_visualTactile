# 完整学生训练的有界冻结特征前取

2026-11-14 的活动第六轮 A/B 使用原冻结源码，未重启。现场双 DP 曾合计约53MiB/s磁盘读取、各进程累计400GB以上；完整 frozen DINO 集合约48GB，而64GB RAM还需容纳训练进程与系统。相邻 episode 的实际用时可在0.34秒与十多秒之间变化。这与共享 page cache 命中状态、并行读取有关，但尚非受控的因果瓶颈证明。

新增可选 `frozen_episode_prefetch_v1` 在 **新训练身份** 的 `A/episode_io.json`、`B/episode_io.json` 登记；缺省仍沿用原 mmap 路径。示例：

正式cycle可显式传 `--prefetch-mib 1024`，在aggregate之后、任何trainer启动前给A/B登记相同配置；已有配置或任一角色已开始训练时禁止替换。活动第六轮没有此参数。

```json
{"version":"frozen_episode_prefetch_v1","max_bytes":1073741824,"minimum_available_bytes":10737418240}
```

一个后台 CPU 线程按已经固定的 episode deck 依次载入完整真实 frozen patches/depth/camera time/valid。当前和待用 episode 的 clone 共享总计最多1GiB预算；下一片放不下就等待当前访问完成释放，单片超预算或可用RAM不足则保留原 mmap 路径并记录原因。不会为了提前读取改变随机顺序、访问数、标签mask、DP噪声或optimizer预算。异常传给训练stage，不丢掉失败片、不补新片。没有 pin-memory 或 GPU 前取，也没有全库常驻cache。

缓存仍是原始 DINO dense patches。可学习 `1×1 conv → GroupNorm → SiLU → pool` 顺序未改；提前池化不等价，不在此优化中采用。每次访问仍按当前参数重建真实视觉和GRU历史，学生前缀仍只提供历史。前取scope结束后清空其tensor字典，必须已完成backward，调用方不得在scope外持有这些feature。

每次更新和每个训练epoch新增进程 `read_bytes/rchar/syscr`、minor/major faults、consumer等待、clone字节、预算等待、fallback和epoch wall统计。epoch报告在验证之前截取；后台前取与训练重叠，不能把各片时间简单相加当wall，也不能仅凭低read_bytes将OS共享缓存收益归为算法速度。1GiB限额针对本loader拥有的clone；OS文件page cache由内核管理，不能把它当成总RSS硬限额。

CPU测试使用真实mmap shard，验证clone的张量、`conv/GroupNorm/SiLU/pool`前向和梯度、随机状态逐位一致；另验证总预算、等待后释放、内存/单片回退、乱序拒绝和后台失败传播。尚未以此入口跑完真实训练周期，**尚无采用或实测加速结论**。

本轮尝试在不改活动worker的条件下建立一次只读单片预取：`session05/io_prefetch_attempt01` 先按原seed和sampling合同重演全部10遍deck，并核对1020条已完成学习记录；核验后发现A已完成全部更新，故helper明确跳过，没有额外大文件读取，也没有为了测速重复训练。`tools.tight_episode_io_probe`保留单片顺序读取/mmap clone/eager三种后续只读比较入口，不drop caches，不预热全库；暖缓存与并行作业混杂须单列。
