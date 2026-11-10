# 2026-11-11：原实验继续，准备自动接续 DAgger

旧运行树是 `/home/CNS2025331827/docs/bc+rl_RSS_palm_eval`，源码 e9385bc，root `outputs/visuotactile/run/tight_transformer_20261110_02`。本次只在隔离 `bc+rl_RSS_dagger` 修改；旧源码/数据/预算不变。文档整理时只读状态仍为 RUNNING / independent_dependency_queue：G01/G11 COMPLETE，Q00/QG00 RUNNING，Q01/QG01 PENDING。这是状态快照，不是最后结果；后续须读原root/status及实际凭据。

用户已经授权旧队列实际结束后自动进行一轮成功筛选的专家接管式 DAgger，最新更正包含前端和后端。配置 `configs/tight_state.json.dagger`：G11/QG01各12采集机会、各10固定开发自主评估，新阶段12h硬上限；前端20/后端30有效遍历上限，平台停止最低10遍，硬墙钟优先。test22不使用，不追加旧轮离线训练。

## 当前已完成与尚未完成

|事项|当前证据边界|
|---|---|
|隔离准备|共享数据/缓存/训练/接管接口与普通后继模式已实现，仍按最终集成验收状态推进|
|真实Core CPU聚合|196条train/validation读回；train166=55,392tick、4,690有效规划anchor，H24累计108,792有效窗口目标；原统计字节相同，test源读取0|
|数据/缓存测试|新增7项测试通过：实际执行与u/q标签分离、H24切断、监督/历史mask分离、固定统计聚合、真实G10 EMA续训、DINO无GPU复用、重预测因果历史/previous_nominal与四SHA|
|最终CPU集成验收|88项联合通过（40.00s）；新增后继等待保护测试随9项定向回归通过（0.81s），去重共89项，覆盖接管/聚合、两阶段续训与恢复、缓存身份、owner依赖、发布、清理及相关历史兼容；一个既有requests依赖版本warning|
|真实接管/GPU|本次文档整理未启动新GPU/Kit，未获得DAgger纠正数据、正式增量更新或自主后测|
|后继注册|必须用实际服务/进程及registration/status凭据补录；脚本和配置存在不能称已排队|

原Core全196条监督mask+segment SHA为 `6601d614380010cc37b7ffb031cab35405859b0bcb7e1ad7652ef27166466e80`；加入纠正后按实际数据重新计算。临时CPU聚合产物由临时目录释放，未写旧run。

## 接续核验

公共入口仍为 `bash visuotactile/run_cpu.sh -m tools.tight_finish`，子命令为 `dagger_prepare` 与 `dagger_follow`，并提供 `--root`。准备登记与普通后台follow必须都实际完成；离开前短查一次WAITING/真实推进，不用模型守候。

接手时核验后继的配置/采集计划SHA、registration、实际服务或PID身份、源冻结与依赖路径。follow依据旧owner/自有worker实际退出和可用选定模型推进，不能只看预计结束时刻，也不能忽略仍在发布的旧owner。单纯报告异常可独立修复，不把它等同模型不可用。新阶段时间预算从实际新起点建立，不复用旧绝对deadline。

G11初始前端为旧G10 selected EMA；QG01初始前端为旧轮QG00 selected EMA。各自后端必须是同一前端SHA配套的所选Transformer；新轮先训练前端，再重建全部train/validation名义缓存并训练后端。采集起点、专家、统计和最终选定模型的确切路径/SHA由 `initial_pair.json` 与各训练凭据报告，不从历史简称猜测。

监督只取成功接管episode的有效专家后缀，学生/切换前缀保留实际历史且mask=0。H24不跨段；G11的u、QG01的q与后端u区分。原统计与残差幅度界限固定。无新成功纠正或有效DP监督不足时报告不足，不改成仅后端训练。

## 发布与历史

旧owner先完成其结果发布；后继在指定origin分支整合后再发布新变化，避免两个工作树同时推送。新小型结果集中在 [STATE_DP_RESULTS](STATE_DP_RESULTS.md)，清理集中在 [CLEANUP_AUDIT](CLEANUP_AUDIT.md)，不新增重复报告MD。

必要历史：StateDP 19,865更新/开发0/10；G00/G10 10,132/6,929更新/各0/10；旧GRU残差G01/G11各98,610更新/0/10、1/10。旧G11成功native_lift=false，旧视觉队列FAILED收尾与PhysX/资源问题如实保留。当前Transformer与旧GRU同时改变网络、监督范围和门控，不能据历史单例声称改善。

新前端以旧队列最近10次更新参考，20个原训练人口anchor等效遍历约1.8–1.9h/组；聚合量、串行资源和前端模态会影响实际速度，后继仍必须用真实输入probe分配时间。DP/后端、缓存与评估实际耗时独立记录，总12h截止不变。

本页的CPU实现状态不代替新阶段物理验收或成果。现行详细合同见 [CONTRACTS](CONTRACTS.md)，过时阶段授权已从当前文档入口去重。
