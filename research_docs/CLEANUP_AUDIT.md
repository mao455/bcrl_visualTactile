# 当前清理与保护结果

## 2026-11-07追加清理：已执行，容量仍阻断

按用户最新授权，从b85738ee694b00b1f287b108fb46510caabf46eb清理无用记录与重复入口。本次新增删除1867个产物文件，释放8,063,660,032bytes（7.510GiB）；不累计此前22.7412GiB。212个.pt路径主要是202个感知缓存尾状态、7个接口/训练smoke权重、2个对比探针及1个重复last，不是212个训练模型。

旧State200 best/last逐字段完整状态一致，合并为同inode的两个恢复路径；best原SHA不变，原last序列化SHA与映射保留。State/Image10000、100/150epoch、State200与当前best4500等实际对照保留。已删9条旧未准入慢速RGB-D载荷，33条旧capture的numeric、DR、阶段分析及全部成功/失败分母保留，旧8条合格轨迹和本轮30条capture完整保留。另删144个已结束失败诊断目录的numeric/plan/trace重型载荷，逐例result、初态、seed、模型SHA和正式175尝试/143合法/2成功/32拒绝的原队列结果保留；144是清理目录数，不能当作新评估分母。

删除4个过期独立owner/发布脚本，4个旧流水线文件仅保留对照/拒绝回归仍需的纯函数与兼容入口；移除旧自动改写MD/推送分支的尾部调用。当前唯一tools.tight_finish入口、采集/学生/训练源码不变。8份字节相同的旧报告去重，保留位置与SHA见evidence/tight_state_20261107.json的additional_cleanup_20261107；历史receipt保留原文，可按映射追溯。没有删除不同实验对照来冒充去重。

清理后54项受影响回归通过；5225个受保护文件identity不变，原400/275 manifest、best4500、统计SHA及14条合格开发payload SHA通过。当前4个CLI、依赖、唯一配置和已接受基础receipt有效；未重跑基础工程/物理试验。

清理后可用71.46GiB，保留60GiB后约11.46GiB；复用全部14条开发数据仍按原最乐观预算需30.86GiB，至少还缺19.41GiB。剩余保护清单之外的本项目内容总共仅约10.2GiB且含必要证据，继续删本项目也无法填满缺口。其他项目不在此次清理范围。正式采集仍0次，manifest/统计/cache未冻结，新StateDP训练未执行；原deadline不重置。

本地逐文件inode/SHA/删除日志：outputs/visuotactile/run/tight_state_20261107_01/cleanup_20261107_02。GitHub仅通过当前origin feature分支普通提交删除过期文件，不重写历史、不上传原始数据或checkpoint。

## 前次已完成清理（独立计数）

当前run `tight_state_20261107_01`。以下只计本轮，历史126.673GiB及其他清理不重复累计。全部删除前检查已完成run、依赖、原始保护目录、同UID打开句柄、inode/mtime/size与必要SHA；不建巨型备份、不删除未知partial。

|ledger|实际文件数|释放分配bytes|内容|
|---|---:|---:|---|
|cleanup_runtime_ledger.json|5409|9,021,136,896|两个已结束旧run的私有Kit临时文件；未知symlink保留|
|cleanup_unused_ledger.json|491|9,631,641,600|400个退休DINO特征cache＋91个已失败观看视频；不是400个模型|
|cleanup_checkpoint_ledger.json|3|3,459,284,992|tiny失败模型best/last及旧275非选中last；best4500保留|
|cleanup_diagnostic_ledger.json|2|2,306,113,536|本轮一次性GPU验收checkpoint；完整恢复receipt和失败诊断保留|
|合计|5905|24,418,177,024|22.741199GiB，按最后硬链接的allocated blocks计数|

账本和每文件删除journal保存在当前root。91条失败视频对应的精简结果、seed、真实分母、终止原因、numeric trace和模型SHA保留于retained_failure_table.json与原结果JSON。旧checkpoint配置/SHA/计数在retired_checkpoint_metadata.json；有成功历史模型与复现配置保留。当前没有正式模型，因而没有可合并的本轮last/best；一次性诊断不冒称正式权重。

源码删除tools/narrow_video_export.py、tools/narrow_videos.py；重复结果文档SLOW_EXPERT_RESULTS.md、GRASP_FAILURE_DIAGNOSIS.md合并入当前结果/历史表。旧全文仍在Git642f5a2，不重写共享历史。其他schema/环境/初始化附录改为当前解释，清除旧阶段待办和冲突授权。

清理后检查：原400全部payload SHA及现275全部payload SHA通过；专家和123个原生Python源码SHA未变；独立入口、坏输入拒绝、自动阶段分支、SIGTERM清理、Markdown链接及依赖检查见机器证据。历史best4500真实model/EMA/optimizer/schedule RNG的CPU读回通过，0优化更新。新随机诊断的完整GPU恢复及下一随机计算通过后，才删除其一次性权重。正式训练/学生闭环没有运行，不宣称这些阶段完成。

首次清理因不可读renderer zombie句柄目录被拒绝，尚未删除文件；仅增加Linux确认Z状态的已退出进程豁免后执行。活进程不豁免，真实zombie/live回归已覆盖。没有对外部进程发信号。

[当前结果](STATE_DP_RESULTS.md) · [精简机器结果](evidence/tight_state_20261107.json)
