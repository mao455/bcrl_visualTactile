# 2026-11-11：多轮共享纠偏池，当前实现与交接

当前工作树 `/home/CNS2025331827/docs/bc+rl_RSS_qg_report`。`tight_transformer_20261110_02` 与单轮 `tight_dagger_20261111_01` 均已有COMPLETE凭据；后者源码 `dc1ed4afd51343679a57bd2f71a8a1d6a42e7550`、两组COMPLETE且owned_workers_exited=true。旧“等待原实验/尚未采集”快照已过期。历史训练量、0成功等结果由 [STATE_DP_RESULTS](STATE_DP_RESULTS.md) 及原机器证据说明。

最新授权是QG01优先、G11随后，最多3轮、总18h含等待与最后10min发布预留。每轮每组默认32采集机会、其中4个整episode纠偏dev（预算规则取24次时保留其中3个预登记编号），最多192机会总量；不是成功数目标，不补失败种子。新队列 `outputs/visuotactile/run/tight_dagger_pool_20261111_01` 已于北京时间2026-11-11 13:24:31启动，冻结运行源码 `593f1097216c0934c52fbd238080532dc77d6b20`，普通服务 `tight-dagger-pool-20261111-01.service`。原绝对截止为2026-11-12 07:24:31（计算07:14:31停止，最后10min收尾），不因中期报告提交重置。

17:01:29阶段快照：第一轮QG01采集27/32成功接管，G11为32/32；QG01前端804次、后端12212次正式更新已完成，自主开发0/10成功、2次原生lift、0子目标、全部601tick超时，按既定规则选中完整新pair。G11前端109/1002、loss有限；后端/自主评估未发生。两组本轮真实CUDA输入、梯度及保存恢复检查均通过。新结果见[结果页首段](STATE_DP_RESULTS.md)与[快照](evidence/tight_dagger_pool_20261111_01/interim_status.json)，后续以root机器进度为准，不用此静态计数推断实时状态。139项历史合并CPU测试仍保留；这次只作CPU报告导出与核对，没有改动264个冻结运行文件或触发额外实验。

## 数据与当前验收

- 旧21条与本轮59条合法成功接管合并为80条纠偏：73条训练、7条整episode开发；另加原Core166共239条训练episode。新数据按run/group/round/source与内容SHA去重，第一轮共同snapshot已冻结，两组共享，开发数据不进入optimizer。
- 旧21条每条最初10个干净专家tick早于首合法12tick规划anchor，共210tick未用于DP目标；后端仍有这210个有效监督tick。原标签与结果不改写。真实逐条边界在 `outputs/visuotactile/data/tight_dagger_pool/acceptance/readback.json`。新switch=12k−2，按原3项delay队列清空后从12k进入干净后缀；不能偷偷跨无效段补旧标签。
- Core166/validation30与原统计保留，test22不打开。原Core train55,392tick/4,690anchor是旧人口，新预算按实际mask/分层人口记录。学生前缀完整推进真实历史，监督0；u_expert/q_reached分离，后端始终监督u。
- 当前Transformer没有失触立即清零；旧GRU的门控只属旧合同。残差幅度/限位不改，失触→再接触计数不证明恢复能力。
- 训练/采样/离线定向CPU测试42项通过：稀少纠偏层占比、可恢复decks、初段与跨控制段mask、真实小型optimizer更新、update0保留、开发集不进optimizer、完整组中断恢复后的参数/Adam/RNG/曝光精确一致及probe正式LR。此为CPU验收，不是新真实接管或闭环效果证据。

## 下一轮的固定合同

|项|当前约定|
|---|---|
|起点|`tight_dagger_20261111_01`的各自selected完整pair，逐SHA确认；QG01先、G11后|
|分层|Core .5 / 历史纠偏 .2 / 本轮纠偏 .3；空纠偏层权重转移，实际比例、重复与唯一覆盖公开|
|前端|现有U-Net及历史，最多10/最低6有效anchor遍；约2遍验证；LR峰值1e-5；每组每轮最多2h|
|后端|新选定前端全train/validation/correction_dev缓存刷新后训练；最多12/最低6有效tick遍；3遍验证；LR3e-4；每组每轮最多.75h|
|选模|实际测量update0；旧val/纠偏dev/首24有效tick各自相对update0，再按.5/.25/.25合成；update0可入选，正式阶段仍须报告新增更新|
|自主评估|固定4例；>=1成功/子目标或>=2原生lift才按同队列扩至10；没有专家替换；旧pair真实同spec结果可复用|
|回退|上轮完整pair保留；新前端与对应训练/验证过的后端作为单一候选，禁止拼装独立best|

短探针提前结束但不缩正式LR日程。backend完整episode组取整可能超过名义tick目标，`excess_supervised_ticks`明示；墙钟停止时曝光和采样比例只报实际值。旧val与纠偏dev分开，绝不使用test22选模。

## 恢复与退出

```bash
bash visuotactile/run_cpu.sh -m tools.tight_finish dagger_resume --root <root>
```

按原source freeze、snapshot、配置和绝对deadline恢复，不能新建同义root重置时间。owner跳过有完整结果与SHA凭据的阶段；中断DP保留optimizer/EMA/RNG/anchor decks，backend保留完整episode组cursor和计数。训练STOP请求在DP optimizer边界或backend完整组边界保存并退出，不写完成receipt。backend支持root/STOP与training/STOP；DP支持其训练root/STOP。恢复只清理本任务记录的STOP标记，不操作无关worker。

崩溃后loss日志晚于last.pt的行保留在 `*.uncommitted.jsonl`，正式计数与日志按持久化update重建，不双计曝光。checkpoint后尚未提交的计算可能重算，不能称其为新增独立数据。已完成训练即使低曝光或best=update0也不从头重训；真正异常/缺失凭据单独处理。模型只核验一次启动后交给普通脚本，不持续轮询。

只发布审查后的源码、既有文档、小型数值/曲线到origin指定feature分支；大轨迹/模型/视频留本地。保护原400、Core、专家/资产、当前/上轮配对、last及未结束阶段依赖；清理由实际SHA/引用审计驱动。原报告中的失败、软件错误与分母不因新轮改变。

启动核验与有限计划：[启动凭据](evidence/tight_dagger_pool_20261111_01/startup_verification.json) · [实测吞吐与时间估计](evidence/tight_dagger_pool_20261111_01/time_estimate.json)。停止：在该root创建`STOP`；恢复使用已有`dagger_resume`入口，原截止不变。owner日志为该root的`owner.log`，逐阶段log和`active_stage.json`保留。
