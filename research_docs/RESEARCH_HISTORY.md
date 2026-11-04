# 已结束研究阶段：精简证据索引

以下是历史结果，不是当前待执行队列。旧代码与完整报告留在[清理前Git版本48a3516](https://github.com/mao455/bcrl_visualTactile/tree/48a3516/research_docs)。按2026-11-05明确授权，已淘汰的模型、冗余缓存和大批原始诊断载荷允许删除；原始400、当前依赖与必要证据保留。

|阶段|关键条件与实测|结论边界/旧报告|
|---|---|---|
|400示范采集|400自然成功，固定320/40/40，三视角448×252 RGB-D，原120/60Hz|不等于学生能力；原始数据全部保留；NIGHT_COLLECTION|
|B0/B1 command基线|各50 epochs，validation选epoch49；50配对合法初态，各0/50，全部601tick timeout，无首子目标|没有触觉收益证据；BASELINE_TRAINING|
|独立Correction|B/C各5 epochs；12分配启动，11有效timeout，1切换验收中断；自主各0/2|局部roll-in与自主分列，未建立RL/触觉因果收益；CORRECTION_TRAINING|
|H64长时策略|G00/G10各40遍33280updates，各0/3；一条lift后fall|低离线误差未转化完整成功；LONG_HORIZON_20261030|
|Loss A/B与Transformer correction|四DP线各9984新增updates；两个correction各3730；六组各0/3诊断|Loss B轻微离线改善，masked也有改善；不能宣称触觉收益；DP_LOSS_AND_TRANSFORMER_RESIDUAL_20261031|
|历史T0/控制回放|历史四laneobservable可恢复但物理不精确；fresh seed17专家与native-request replay均220步成功，比较字段逐字节相同|不可把历史T0失败直接解释为泛化差；T0_REPLAY_AND_DP_CONTROL_FIX|
|绝对29D位置接口|专家target独立回放2/2成功，168/153ticks，14类字段逐值一致，执行器误差0|通过DP同一位置接口，不用原生RL command回放冒充；保留foundation证据|
|State/Image最终10000|原冻结train4+validation8各0/12；State接触10、lift1，Image接触5、lift0；软件错误另计|完整成功率均0%，contact不等于稳定抓持；EVALUATION历史记录|
|抓取失败诊断|24/24在专家接触前偏离，22/24在首次replan12前；Image live严格配对未通过|单纯边界平滑不是已证实根因；保留GRASP_FAILURE_DIAGNOSIS与简明数值|
|State100 anchor等效epoch|16584；validation4 0/4成功、4/4contact、0lift/子目标；偏离ticks3/5/7/5|与旧同4初态0/4、3/4contact比较，无实质任务进展；保留interim数值与必要视频|

Kit的ENOSPC、启动FileThread SIGABRT和关闭SIGSEGV分别记录。当前独立worker隔离与有限恢复不是修好共享SDK或证明graceful shutdown；未改共享SDK。旧资源违规仍成立：部分历史session完整60s整卡均值超过50%，不能用新窗口或删除大日志抹去。各历史报告原文、SHA和精简证据可由Git链接与本地maintenance/history_retention.json追溯。

当前150评估结果仅见[统一结果](STATE_DP_RESULTS.md)，不从旧状态文本推断进度。后续不自动扩展训练或试验。
