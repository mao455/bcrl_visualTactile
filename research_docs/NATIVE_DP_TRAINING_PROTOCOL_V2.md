# State-first Native DP protocol V2

授权来源为2026-11-02完整任务附件；不延长旧run02截止，不改变原400、划分、checkpoint或原生任务。
新目录、独立worktree、独立服务。以下区分已实现、启动后才有的实验与尚被Gate锁住的工作。

## 输入、输出和控制合同

三组继续使用 `native_data.CONTRACT` 的原生命令推理接口；新训练metadata增加protocol2。
旧formal恢复仍用原合同，不能将旧optimizer状态解释成新随机anchor训练的无损续训。

| 输入/输出 | 维度 | 因果来源 |
|---|---|---|
| q / qd / controller target / target−q |各29 | 当前动作前encoder/真实target，rad、rad/s |
| previous issued / delayed |各29 | 严格上一步实际执行反馈；训练为明确标注的expert teacher forcing |
| 两种命令年龄 / history valid / episode elapsed |2+1+1 | 可由控制时钟获得，不使用未来命令 |
| robot合计 |178 | 不是触觉或未来controller target |
| recent robot |13×180=2340 | 13个native60Hz时刻；每行加年龄+valid；初期左padding0 |
| S几何 |85 | robot-base下object/goal/hole/table/palm pose共35，object/goal速度12，五指pose35，物体尺寸3 |
| S近期几何 |3×87=261 | 每12tick读取当前几何；各加年龄+valid；没有长GRU |
| S condition |512 | robot MLP256 + geometry MLP256 → projection512 |
| DP输出 |64×29 | sample/x0预测，经train-only affine反归一化，clip[-1,1]，执行前12 |
| StateBC诊断 |2601→256→256→1856 | 两层Mish MLP，独立确定性chunk回归；相同输入/labels/normalizer，独立BC合同 |

StateBC仅诊断，不是第四个正式baseline，也不代替StateDP成功率。
Official/History完整输入沿用 `HISTORY_AND_OBSERVATION_ARCHITECTURE.md`；Official ResNet18/GN从零训练，
History冻结DINO+空间注意力+双GRU；本轮不引入预训练替换、触觉残差或大模型。

底层仍canonical29→原生动作排列→原生随机delay→arm增量/hand范围映射→原平滑/限位→两120Hz物理步。
不改controller、task、DR、reset、reward、success、timeout；replay专用模式仍使用已有opt-in记录delay/wrench接口。

## Gate定义

每个Gate固定输出 `infrastructure_passed / offline_fit_passed / closed_loop_success / acceptance_passed / blocked_reason`。
未知写null，false不得由程序正常返回覆盖。软件stage另用 `software_complete`，没有含混的统一passed。

| Gate | 验收/失败处理 |
|---|---|
| G0 数据和环境 |400数据/normalizer/native函数一致性；8个不同历史示范，8/8原生完整成功且命令、delay、target、wrench容差通过。物理逐场景差异另报，不冒称精确hidden state。失败阻断全部新optimizer更新 |
| G1 网络与学习能力 |形状/有限梯度、真实窗在线离线一致；保留旧State fit1/fit4作为历史小样本证据，另运行新StateBC≤1000updates。旧fit不能当新闭环成功 |
| G2 State训练初态 |冻结8个不同train初态；至少2个原生完整成功才通过；全部原生终止、软件错误另列。零成功先分析，不因offline改善扩大预算 |
| G3 State验证初态 |冻结8个不同validation初态；同样至少2完整成功；test不参与选择 |
| G4 视觉启动 |G0–G3通过，并在至少2个validation初态上另一个diffusion seed重复完整成功；新checkpoint须自己的重复证据，旧模型重复不能嫁接 |
| G5 正式评估 |配置/validation选模冻结、G4和视觉小规模验收通过之后才能发起test40及固定sorted first20×3；此前test使用另记，不宣称全新测试集 |

2/8是本轮最小证据门槛，绝不是稳定泛化成功率目标。只有28或32条诊断的实际结果才可填充闭环字段。
本轮没有放宽原数据/控制一致性阈值，也不因失败事后选择较容易初态。

## 固定诊断与顺序

1. 重读400原文件SHA、数值时序/因果历史、train-only统计；检查原生末尾success/retract、长度分布、重复。
2. 在train和validation各自按长度四分位端点选择4条，再以初始几何/q标准化距离补至8条。
   样本身份写入不可变session；没有使用模型表现或test选择。8 replay取上述train4+val4，覆盖119–539tick。
3. 等旧owner、trainer、evaluator和monitor退出；普通后台脚本等待，保护旧任务。
4. 在一个持久单env Kit中执行8 replay，然后冻结旧best的train8/val8、两组各首2条第二seed敏感性。
   另检查last；EMA逐tensor字节相同则复用并明确标注，否则再做train2/val2，最多32 starts。
5. 对旧best/last做train8和validation40离线3seed诊断。保留原单seed旧验证指标，
   新batch32诊断是另一个固定测量协议，不把新数值与旧.6695/.4612强行当完全配对。
6. G0失败时保留冻结诊断并停止学习，不自动重放更多轨迹、改任务或重新采集。
7. G0通过才训练StateBC，≤1000updates、train4，完整validation40；独立train8/val8自主诊断。
   BC无训练初态成功则停止并分析公共数据/状态/闭环基础，不盲目新DP。
8. 有依据的新StateDP从零建立训练身份：固定32 anchors，shuffle无放回；
   milestones1000/3000/6000/10000/19000，各自offline多seed及固定train8/val8闭环。
   三个≥6000的检查点窗口无success且stage进展停滞则停；早期零成功不直接判无效。
9. G4未过，视觉与V2正式test不启动。当前高成本视觉阶段不沿用旧三组直接跑满的组织方式。
   新State通过后仍须补其重复seed证据和视觉小规模真实验收；这些未发生前不写成完成。

## 新State训练设置与效率

模型/U-Net、Loss B、sample、H64/commit12/DDIM16、AdamW1e−4/.95/.999/wd1e−6、warmup500/cosine、EMA.75、clip1保持。
唯一首选训练对照为随机anchor batch与等价数值缓存：每遍5547 anchors，batch32，最后batch11，174更新/遍。
这改变batch相关性与每次更新有效目标分布，因此必须新目录，不能伪称只改变速度而完全同训练轨迹。

`StateWindows`只缓存原始标准化因果输入窗与标签，不缓存encoder输出；梯度仍穿过所有原State可训练模块。
不重复打开HDF5或回放无RNN的13行窗口。real-data测试比较完整原online条件，DP和BC均通过。
保留无效tail mask，Loss B按整个batch有效位置数归一化，不给短末段虚构标签。
训练日志记录optimizer steps、实际anchors、完整遍历数/分数、effective batch、LR、grad norm、loss及load/work/wait/pacing/checkpoint耗时。
“episode_traversals_equivalent”是覆盖折算，不把混合anchor batch冒充顺序整episode执行。

History当前权重的GRU前缀不能跨optimizer step缓存成精确等价结果。未来可按整episode chunked BPTT顺序运行，
但会改变梯度/更新边界，必须独立合同和真实online parity验证。当前保持旧已验证实现并由G4禁止昂贵重训。
Official可缓存原始选中帧/开持久HDF5句柄；不能缓存其正在训练的ResNet特征。DINO冻结特征可继续使用已核对的cache+尾帧overlay。
这些视觉优化当前为后续受Gate约束的工作，不宣称已经实现或测速。

## Checkpoint、选模和恢复

- 新训练保存last（含model/EMA/optimizer/RNG/cursor/epoch）、`best_validation_offline.pt`、不可变milestone。
- `best_validation_rollout.json`按validation完整成功数、阶段得分、较早update排序，引用milestone和SHA；不复制大权重也不读test。
- offline以固定3seed前12步arm/hand均权MSE均值选模，同时记录标准差、H64误差、diffusion objective及clip前饱和率。
- 单条命令run/resume/status。训练恢复要求配置/data/statistics/实现SHA一致，游标从原子checkpoint续；
  已完成阶段按结果hash复用。中断过的实际rollout cohort拒绝自动重复；需先审阅已有attempt，不能重复记分母。
- 当前实现不自动启动未验收视觉阶段或自动修改源码恢复Kit；所有失败日志保留。
- 诊断GPU窗口4h含等待；若G0允许学习，训练采用首次GPU开始时预注册的总20h绝对截止（4+16），不重置起点。
  等待旧run另有固定10h上限；从新session注册起总计最多24h（包括等待），所有GPU阶段也受该绝对截止约束。旧run的24h截止完全不变。

## 资源、并行和视频

两CPU线程、nice19、既有22/23亲和性；GPU整卡时间加权与完整60s≤50%，目标40%、45%准入，4GiB/10GiB硬余量，
6GiB/14GiB准入余量，12GB磁盘reserve；持续monitor从新GPU启动至所有worker结束。
V2 lock只包GPU计算/同步或必须的环境操作，CPU数值加载、checkpoint写盘、等待/pacing在锁外；
记录work/wait/pacing分别计时，整卡测量不冒称本进程归因。

不改物理频率来加速。多env一致性没有通过，当前只用持久单env，避免反复初始化Kit；
首条replay和首条train/val学生录完整60fps视频，其余只数值，不让录制拖慢全部样本。
shadow推理计时与真正控制推理分开；视频/serialization也单列。缺失terminal图不替换reset画面。
现有已验证220步专家成功视频作为参考成功案例，不将其与历史回放或学生当同初态配对。

## 入口与状态

在独立worktree `bc+rl_RSS_native_v2`：

```bash
bash visuotactile/run_cpu.sh -m visuotactile.native_v2_pipeline initialize \
  --root outputs/visuotactile/native_dp_v2/state_first_20261102_01
bash visuotactile/run_cpu.sh -m visuotactile.native_v2_pipeline run \
  --root outputs/visuotactile/native_dp_v2/state_first_20261102_01
bash visuotactile/run_cpu.sh -m visuotactile.native_v2_pipeline status \
  --root outputs/visuotactile/native_dp_v2/state_first_20261102_01
```

首次initialize只执行一次。长任务由独立服务运行；Codex只核验一次启动，不以对话循环监督。
显式`resume`将旧STOP/blocked归档后继续，不重置deadline、不重复中断的轨迹；`report`由现有文件重建小统计和曲线。
原400/所有正式权重/旧日志保留；本轮清理清单的deleted为空。
