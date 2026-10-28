# 2026-10-29 夜间400条采集

## 本轮授权与时间

2026-10-29 01:19新加坡/上海时间检查开始，交付截止08:00（UTC 2026-10-29 00:00，epoch 1793232000）。此前60GB、40GB、CPU两环境和首次软件错误即停规则已被用户覆盖。仍禁止训练、改任务或操作其他任务。当前实时状态以`outputs/visuotactile/run/close400/status.json`、`progress.json`和data manifest为准，文档数字是快照。

## 已实现的最小采集链路

- `collect.py`：显式设备、线程、环境间距和持久采集入口；AppLauncher后加载环境，一次加载released expert。原诊断入口仍存在。
- `sim/environment.py::_reset_idx`：仅生产开启原生reset后的box合法性筛选；逐环境记录每次尝试，非法者仅重采自身，最多100次；不重设全局seed。研究episode ID随每次真实reset增长，子目标不触发该ID。
- `collect_success_batch.py::collect_persistent`：一个Kit，批量专家推理，独立episode文件、逐环境RNN状态清零、有界writer队列（4批次）、两个spawn CPU审核进程；成功经审核后原子移动并提交manifest。
- `recording.py::CompactEpisodeStream`：每8控制步批量写入原storage2布局，RGB uint8精确可逆、depth float32米制、mask和所有非图像字段无损，LZF压缩。保存每个原始字段的增量SHA256；审核重新构建全部143字段并逐字段验证hash、dtype与shape。省去重复raw和第二次转换。
- `finalize_collection.py::finalize_stream`：重新检查自然成功/终止、初态、动作延迟/映射、状态连续性、120Hz力/FK、三相机全帧/动态位姿以及原物理配置。未训练Nominal/Correction缺失标记保持原定义。
- `collection_resources.py`：实际剩余磁盘12GB底线、有界队列；30秒盘点、不再每步递归扫描。全卡一秒采样，时间加权批次均值和完整60秒均值；只在均值接近45%时暂停本任务子进程组，40%恢复，不因单一峰值暂停。
- `initialization_report.py::manifest_statistics`：从已经实测审核的初态/扰动数值更新CSV和JSON，世界位置减各自env origin。成功、非法初态、自然失败、启动/worker/审计错误分别计数，旧v3成功不计本目标。

保持原physics=120Hz、控制=60Hz，渲染/CPU墙钟耗时不改变dt、动作幅度或成功判据。保留标准Camera及每render product的动态CameraParams；没有冒充已验证TiledCamera/GPU物理。当前tight_insertion采用preInsertAndFinal、random_goal_fraction=0，两个插入目标使用原固定容差；原课程阈值3高于本任务2个目标，不通过长实例放宽判据。

## 本轮查出的真实问题

1. runtime cfg含tuple，存盘JSON为list；初次持久启动被严格配置比较误拒绝。已统一为JSON值表示，未改变配置数值。
2. released expert具有循环状态；torch.inference_mode下生成状态后在模式外按done清零会报错。改用torch.no_grad（同原collector），逐环境状态清零。第一条完整成功文件在错误退出后通过143字段/完整审核恢复，而非丢弃。
3. **原1.2m环境间距使env1侧相机看到邻居桌子。** 张量/动态外参测试不会发现该视觉问题；实际查看原图后停批。该配置已得到8条自然成功，但全部从400目标排除，轻量原因/统计保留；只留一条可复核的邻居入镜证据，删除其余已确认不合格载荷。目标重新为原两条s930014/016。生产候选仅将环境原点间隔改为8m，局部相机和任务几何不变，避免近邻碰撞和5m裁剪范围内的邻居。4环境/8m的真实审核结果随后追加。

## 腕部选择：保留v4

对当前上侧支架与一个下侧候选做同seed17、完整220步对比，固定两路相机保持v4坐标。128个非图像字段逐位相同，自然成功相同，下侧完整数据审核无错误。下侧光心在这条轨迹上的机器人网格AABB最小距离35.24mm；不是实体外壳/支架/线缆适配认证。

亲看逐25步接触图和完整帧序列：下侧更清楚地看到绿色指尖和红色物体接触，但初始物体大面积裁出，孔口接近侧面投影，在搬运/插入时覆盖不足。220步中孔口投影至少部分入画：当前150帧、下侧71帧；深度清晰像素>10：当前133、下侧48。这是诊断投影/深度证据，不单独等同可见性真值；图像同时支持选择。保留当前的理由是孔口覆盖及与原两条正式v4数据兼容，未称下侧没有研究价值。

当前固定局部eye=(.12,.16,.07)m，朝向局部target=(0,-.14,.24)，父刚体iiwa14_link_7；ROS+Z为光轴。它跟手腕旋转，并非始终世界向下：seed17的世界光轴z分量约-0.716到-0.391。视图中的“在手上方”是支架相对手的偏置，不是固定世界相机，也不每帧追踪孔座。图示/全帧比较：`outputs/visuotactile/demos/wrist_choice/upper_vs_lower_contact_sheet.png`、`upper_vs_lower_all_frames.mp4`。

一手资料（本轮实际读取）：DROID项目页https://droid-dataset.github.io/明确两外部ZED2＋腕部ZEDMini；SharpaDex https://sharpadex.sharpa.com.cn/明确头部双视图及左右腕部；Cloak https://tml.stanford.edu/cloak/setup/sharpa.html 的实体图与文字明确右SHARPA上方相机、Franka斜转接板与专用支架，并明确不能直接替换左手。网页读取器对Cloak失败，本机HTTP取得完整页面；未复制第三方代码/资产。资料证明类似安装存在，不证明KUKA左手支架已硬件验证。

## 验证记录（持续追加）

- 23项相关单元测试通过：stream双episode独立字段、逐行hash重建、终止/缺失帧、拒绝错位、旧compact读取、已有资源/初始化/试验检查。未重复网络梯度或训练。
- 两环境原间距试采：完整成功提交速度2.21条/分钟（54.32秒两条，启动另计）；全监测72.98秒GPU均值14.78%，结束完整60秒均值17.18%，峰值显存12136MiB；无节流。这是吞吐证据，**因随后发现邻居入镜而不构成正式数据通过**。
- 新LZF示例183步165.26MB、154步141.01MB、197步178.69MB；与旧gzip不同文件大小但143字段精确。400条量级约70–90GB，当前可用磁盘约190GB，12GB余量底线仍适用。

### 4环境/8m通过（01:38快照）

4条合格成功157/194/203/193步，含env3跨episode真实reset。另1条原生hand_far失败（6步）和非法初态独立记录；没有用失败凑数。ManifestReader全SHA验证、ManifestSequenceDataset逐条三视角chunk `(1,2,3,4,252,448)`、动作监督`(2,12,29)`读取通过；143原始字段hash重建及全帧数值/时钟/动态位姿审核通过。分环境原点为(±4,±4,0)，真实视图显示不同物体/孔座和动作，而非复制env0。初/中/末图已人工查看；侧视邻居桌子消失。

从采集循环开始97.15秒完整提交4条（2.47/min），含Kit启动和退出共116.84秒（2.05/min）；全卡GPU均值24.04%，结束完整60秒均值25.08%，峰值显存8189MiB，未节流。峰值只是本次采样值，不能用不同启动阶段推断环境数越大反而显存越小。4环境已满足当前400剩余量/时间的必要吞吐，停止继续测8或GPU/TiledCamera。按当前小样本不能保证400完成ETA，正式生产会滚动报告。

新正式生产PID/状态请使用下面命令；01:38累计6条（旧2＋合格4），400生产已启动。旧1.2m的8条已排除。正式原图视频在`demos/production_v4/three_view_demo_cn.mp4`，视场图`scene_camera_overview.png`。

## 日常命令（仓库根目录执行）

启动/恢复：
```bash
RESEARCH_THREADS=4 RESEARCH_CPUS=20-23 bash visuotactile/run_cpu.sh -m visuotactile.collect_success_batch start --num-envs 4 --threads 4 --target-successes 400
```
状态：
```bash
bash visuotactile/run_cpu.sh -m visuotactile.collect_success_batch status
```
只停止本任务，完成正在审核的文件后退出：
```bash
bash visuotactile/run_cpu.sh -m visuotactile.collect_success_batch stop
```

入口固定本轮绝对deadline；后续新授权才显式传`--deadline-epoch`。后台stdin关闭、独立session、日志文件输出，不依赖终端。supervisor锁防双启动；只管理自己创建的子进程组，STOP正常收尾，180秒有效运行时间无heartbeat视为卡住（自身GPU节流暂停不计）。已知native启动错误/卡住最多三次无进展重试；Python/审计错误保留现场等待工程修复，不能把无限重启当恢复。恢复会重新审核已结束但未提交的文件；未知partial保留、不给成功名额。

目录：`data/close_v4/manifest.json`是当前唯一正式索引；`run/close400/{owner,status,progress,resources}.json`与有限日志；`run/close400/statistics/{initial_states.csv,summary.json,attempts.json}`由审核数据自动生成；`tmp/<run>/<episode>/`只用于当前写盘/审核（4批writer、最多8条audit），成功移动到data、已确认失败/删失只留小统计。

### 目录整理与保留依据（01:58快照）

顶层现在只有`data/ demos/ run/ tmp/`。已核验删除4,541,529,281bytes（约4.54GB），其中1.608GB是排除的1.2m邻居入镜成功试采，其余为12条旧非法初态载荷、已淘汰相机候选、旧普通观看视频/截图、已确认的早期截断测试和重复PKL导出。每个删除项有路径/bytes/SHA/理由，未知损坏未盲删；已知两个partial先打开验证连续step与未完成属性，再保留原因清理。

旧17条双路正式数据和3条v3正式数据移动后重新用ManifestReader核验SHA；被删除的观看MP4从manifest显式取消声明，HDF5未改。旧配对实验的合法成功、原历史机制有效示例、旧nominal checkpoint、两条完整v4参考均保留分组。`data/INDEX.json`和`run/path_relocations.json`提供新读取入口；原HDF5内历史绝对路径是当时来源，不为“看起来整洁”改写。当前400生产只有`data/close_v4/manifest.json`，其他成功不混计。

实际容量快照：data17.885GB、demos11.51MB、run20.38MB、tmp490MB，正式数据已71条；容量随后台提交增加。旧脚本重建/读取需使用新的数据路径或迁移映射。

### 运行语义与边界补充

持续实例中的seed是整个run的seed，不把每条episode伪称独立重新设seed；唯一身份是run ID＋env index＋native episode ID。metadata继承的`initial_torch_rng_sha256`指run首次reset后时刻，各episode真正决策时的CPU RNG位于首行及逐行`transition/torch_rng_before_inference`；失败/非法初始化事件保留其真实初态，不能由未来成功补填。没有在运行中重新设seed或修改RNG。

world坐标以每个环境真实origin存储；所有相机局部标定仍同v4。初始腕部world外参随初态不同不是标定不兼容，比较使用固定局部安装变换，actual parent prim仍按对应env保存。每个CameraParams来自该相机自身render product，原始全帧姿态与FK父刚体同步审核。动态位姿继续是诊断/标定，未新增模型输入。旧两条s930014/016与新数据可在同一ManifestSequenceDataset读取，CPU多环境导致随机序列/浮点轨迹变化，未声称与旧单环境逐位相同。

完整episode初始状态、命令与目标、关节/力/时间字段保持现有schema2/trace2/storage2；新增writer provenance和per-env metadata。文件不是通过降低图像/深度精度变小。非终止截止的最后真实post帧被另存；自然终止图仍显式缺失，不能把reset图归到上条。

剩余风险：共享inotify watch压力未全局修改；持久实例减少启动次数但不能证明native根因消失。若未来native启动/worker卡顿，supervisor有限恢复并保留已提交数据；Python/审计异常需要定位后恢复，不无限重启。真实KUKA左手实体支架、视觉/触觉策略能力、去预加载和Correction训练目标仍未验证/未决定。本轮零参数训练。

### 02:08 长批真实快照

累计112/400（剩288），当前run106成功、88非法起点、3原生任务失败，4条正在运行/2条审核中；另外6条是旧2＋本轮8m验收4。循环4.23条/分钟，含启动4.18条/分钟，动态ETA约68分钟。GPU最大完整60秒46.22%，全批约30.8%，无遥测缺失。进程2117854/2117864均实查存活。源码commit80e703f已成功推送origin/main，10个实际运行文件hash均匹配。此处是快照，是否400完成必须看实时manifest/status。

可以使用更短的默认命令：`bash visuotactile/run_cpu.sh -m visuotactile.collect_success_batch start`；默认即4环境、4线程、目标400和本轮08:00截止。status/stop同上。
