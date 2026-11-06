# 当前交接：200条StateDP及条件化后续

<!-- NARROW_REPAIR_START -->
## 2026-11-06：视觉与GPU历史修复、单次慢速数据对照

实测状态 `blocked` / `slow_collection`；独立run `narrow_repair_20261106_01`，10h固定总窗，旧run预算和失败保持。代码和排队不代表训练完成。
白屏已定位：direct fresh入口遗漏固定相机配置，两路停在默认相同位姿，RGB恒定/有效深度0；同初态配置前后物理读数完全相同，配置后三视角有真实内容。首条199帧专家成功，起/中/尾画面目视通过，全帧RGB-D、时钟、固定位姿和mask检查通过；单lane持久Kit，不声明多lane已验证。
GPU差异来自未统一的CuDNN TF32：旧默认condition最大2.52545e-4；关闭TF32后最大1.07288e-6、前12目标1.13249e-6rad、FK0.000229mm。原2e-5门限未放宽；完整序列/逐tick/训练prefix/reset均检查，无优化更新。

12:35首次生产停止：第50条原速专家真实掉落，末帧物体离开侧视野；81帧RGB-D和数值合同通过。成功示范的目标可见性检查错误地先于任务失败分类，导致少记1次尝试/掉落。复核49条成功文件哈希后，保留原错误和计数快照，补记为50次/49成功/1任务失败，失败轨迹不入训练。修复仅分离自然任务失败与传感器错误，成功数据门槛不变；8项CPU测试通过。本次复核后的接续入口只运行原冻结清单剩余26条，不重跑已执行条目或门控，不重置计数、17:12采集/21:12总截止；独立阶段reviewed_collection_completion仍只允许一次新训练。实际接续进度以上方机器状态和当前计数为准。

实际扩采/训练/评估结果见[StateDP统一记录](STATE_DP_RESULTS.md)和[机器统计](evidence/narrow_repair.json)，不把准备写成成功。
<!-- NARROW_REPAIR_END -->

启动核验：b0e1bd0源码的production服务已运行，随机GPU smoke完成1次可丢弃更新并完整恢复model/AdamW/EMA，history最大1.073e-6；正式新训练仍0。一个199帧完整专家成功与后续学生复核均已实际落盘，owner/worker/monitor存活、记录持续推进；实际后续phase看机器status，慢速扩采/正式训练不提前声称完成。旧学生14条Arm重规划边界目标跳变RMS中位0.02429rad，专家200条0.00203rad，块内分别0.01388/0.00203；这是不同轨迹的描述统计，不是速度或失稳因果证明。新owner自动门控后续阶段，最终只更新生成区并推送轻量结果；不再LLM轮询。一次零物理步软件失败保持，新修复不重置原总预算。


## 2026-11-06 10:06 UTC+8：代码已发布，4条失败视频完成

源码/新数据生成入口已推送至origin/feature/narrow-state-data（b119470、fde65f2、26e0320及观看脚本ade4f1f）。新观看队列独立固定1h/单worker/4starts，已正常完成，owner/worker/monitor均退出：train_family_000/004、unseen_00/04全部合法超时，步数601/824/601/601，仅004达到首子目标。实际best update5735、H24/commit12/推理参数未改，未训练或扩采；旧预算、原400+200的manifest、best/last和正式结果hash保持。

本地索引 `outputs/visuotactile/demos/narrow_failures_20261106_01/README.md` 包含4个原三视角MP4和4个便于观看的腕部裁剪版本。均完整CPU解码、帧数一致、60fps/1x/自然终止。原可观测初态8组字段全等，首plan最大差约3.67e-4rad，物理轨迹不相同；004原718步、复跑824步，不称精确回放，不加入正式0/4和0/10分母。source ade4f1f，CPU裁剪代码及文件SHA另记。Kit worker在结果落盘后使用既有fast-exit(0)，不声称SDK常规析构已验证；进程确已回收。资源整卡均值17.177%、最大完整60s27.013%、缺测0。[完整轻量证据](evidence/narrow_failure_videos.json)、[预览](evidence/narrow_failure_videos_preview.jpg)。

画面发现两固定相机空白；200条新示范首帧各200/200为RGB238常量，腕部200/200非恒定，另首尾两示范各首/中/尾也复核。未改相机/原始文件，未完成全帧和深度有效性审查、原因未定位。State不读RGB，但ImageDP前必须修复并验收。见[相机证据](evidence/narrow_camera_review.json)。原后续仍在GPU历史一致性审查处blocked，无自动重试或续训/扩采。

## 2026-11-06 09:29 UTC+8 实测：后续已停止

原200条训练及闭环已完成：5735更新，train0/4；unseen0/10合法运行，另2次准入拒绝。后续服务于02:13接手后在GPU历史一致性断言失败处停止，所有自有进程已退出。未续训、未扩采、未重启；原门限和新截止10:13:16未改。CPU只读复核通过不能代替GPU验收。具体失败、证据和限制见[统一State结果](STATE_DP_RESULTS.md)及[evidence](evidence/narrow_followup.json)。下方为先前启动快照。


2026-11-06：原run实际200/200成功、201尝试、43,621控制步、17,972,841,587bytes，60家族44/8/8。原服务已完成5,735/5,735更新并进入闭环；检查时前2条train均601tick超时，整体train4/unseen12尚未完成。best/last完整checkpoint已在CPU读回，EMA/176组AdamW step均5735、LR3e-6；不能把后续授权写成闭环完成。原绝对截止13:57:39 UTC+8不变。

用户新授权的独立源码在`bc+rl_RSS_narrow_followup`，独立root为`outputs/visuotactile/run/narrow_state_followup_20261106_01`。`tools.narrow_followup run`只等待原run正常complete、自有进程全部退出和owner锁释放，再启动新monitor/数据和已记录控制审核；依据实际结果选择最多一次续训、一次新增约60–96条真实数据、一次独立随机训练或结束。新执行期固定8h，等待原任务不使用GPU且有原deadline+15min上限。停止方式为在新root创建STOP；不删除预算、attempt或失败证据，不自动重启。

现有末段证据（3000→4500更新）验证MSE改善1.44%，train8改善42.27%，未见首块幅度约3.11倍、hold误差比1.56；只是平台/泛化差距的预警。是否扩采须等完整闭环和数据/在线控制审核；数据不足不是已证明原因。新的最终16个未见初态预先固定，不进入调度；原unseen12标记开发用途。

新owner实现普通后台等待、固定预算/单次启动/资源监控、阶段转换、最终停止和轻量自动发布，服务名`narrow-state-followup-20261106-01.service`。`tools.narrow_delivery`仅更新现有主文档生成区及一个JSON/曲线，发布前拒绝HEAD、人工正文或工作树变化；只push origin同feature。所有400+200及正式模型、失败证据保留，暂无可确认废弃的新增checkpoint可删。51项CPU回归通过；真实已记录学生tick0/12/95的本体读取误差2.38419e-7，未启动额外GPU/Kit。启动实证见evidence/narrow_followup_startup.json及新root机器状态，不能以此宣称后续阶段执行完成。

下面为原run启动时历史证据，不是当前进度。

分支`feature/narrow-state-data`，工作树`bc+rl_RSS_narrow_state`，root`outputs/visuotactile/run/narrow_state_20261105_02`。已准入60/68直接候选，44/8/8；4条真实pilot全部成功，199/157原速、455/264步alpha.58，实际时长比2.286/1.682。完整154字段hash/120–60Hz对齐及同家族初态一致通过。新几何准入使用34个实际碰撞网格包围盒和实时body pose，旧query误报继续留证。

一次性GPU随机模型优化1次、AdamW/EMA保存恢复、完整/在线历史与未来隔离通过；正式训练0更新，DATA_READY=false。最初GPU smoke在优化前的deadline参数错误保留，唯一补充检查通过。46项CPU回归通过。

普通owner将在生产结束且数据门控通过后自动GPU完整数据smoke→随机StateDP→train4/unseen12闭环。总截止2026-11-06 13:57:39 UTC+8不变；按实测共享负载在生产前重分配采集6h/训练最多9h/末1h评估。达到150–180条且覆盖合格可按预算结束，不无限补数。完整边界暂停/源码切换证据在maintenance/production_cutover；旧v1及旧State143队列不恢复。

首worker整卡均值43.69%、最大完整60s52.64%，缺测0，最低RAM25.66GiB；不称满足旧50%硬门限。资源用shared_recent_v1目标40%、6GiB/12GiB准入。原400逐文件存在/大小和manifest SHA通过，仍70,186,972,525bytes。

状态命令：`bash visuotactile/run_cpu.sh -m tools.narrow_collection status --root outputs/visuotactile/run/narrow_state_20261105_02`。真实后台是否继续、目前条数/更新数以该JSON为准。实验设置和日志/stop/resume见SLOW_EXPERT_RESULTS，训练合同见CONTRACTS。

生产源码b119470；当前服务`narrow-state-data-20261105-02-production.service`已实际完成额外2条自然成功，最近快照7/7、DATA_READY=false、正式0更新。owner/worker/monitor实际在运行。两个production样例已有推理、环境/渲染、numeric、compact和finalize耗时；详细机器证据见evidence/narrow_v2_startup.json。后续由普通程序接续，不再LLM轮询。清理共释放2,306,101,248bytes（2.148GiB），原数据/依赖保留。
