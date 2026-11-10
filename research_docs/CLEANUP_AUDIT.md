## 2026-11-11 DAgger 准备与延期清理

当前运行树保持原样，新实现位于隔离工作树。现阶段实际删除旧产物 **0 个**；14个明确旧权重候选由普通后继在两轮自有进程退出、初始化依赖与打开文件复查后处理。源数据、有效DINO缓存、G00/G10 selected best以及新轮初始化/恢复依赖保留；不按目录年龄整树删除。最终逐项SHA、跳过原因、硬链接实际释放空间写本轮 `cleanup.json`，并由CPU发布器公开。

本次职责整理前/后：自有源码脚本91→92，测试42→48，配置4→4，文档18→18，上游/第三方656→656，原数值图像637→637。唯一新增源码模块 `tools/tight_dagger.py` 负责阶段与依赖协调；模型、优化、数据读取和评估仍走已有实现。八个现有文档去除过时授权与重复合同，没有新增MD。字节分类与验证见 [准备凭据](evidence/tight_dagger_20261111_01/preparation.json)；新生成的小型发布JSON单列，不伪装成源码减量。

# 当前清理与保护结果

## 2026-11-10 当前复用调整：555.359375MiB

依最新精确授权，删除失败 `tight_transformer_20261110_01` 内 `command/common_initial.pt`、
`configuration/common_initial.pt` 两个随机初始化副本，以及新 `_02/configuration/common_initial.pt` 指向后者的残留软链接。
**本次仅2个普通文件＋1个软链接，释放582,336,512 allocated bytes＝555.359375MiB。**
两个权重文件的逻辑大小合计582,326,912 bytes；不把共享磁盘空闲量变化当成本次释放量，不累计此前398文件的清理。

command副本与保留的 `tight_vision_20261109_01/common_initial.pt` 文件SHA相同，可逐字节复制重建。
configuration副本可从该保留文件加载，再用仍保留的 `_01/configuration/state_data/statistics.pt` 替换
`robot_mean`、`robot_std`、`action_scale`、`action_offset` 四个buffer；删除前195个张量逐项完全相等。
后者仅两个action buffer数值不同，不承诺重新序列化后的文件SHA相同。

当前Q00/QG00明确从旧G00选定EMA热启动，fresh训练和验收的warm-start分支均不再读取或哈希随机初始化副本。
删除前核对实际session依赖、同UID打开句柄、候选SHA/inode/size/mtime与单硬链接；清理计划先落盘。
删除后8项关键保留文件的identity和SHA未变，失败root其余239个既有文件路径的identity未变。
旧G00/G10 selected best、旧G01/G11 best/last、原vision随机初始化、q统计、全部失败日志/probe/data均保留；原400/Core不在删除清单内，未重复读取其载荷做全SHA审计。

[逐文件SHA、allocated bytes、依赖与实际删除凭据](evidence/tight_reuse_20261110/cleanup.json) ·
[权重/缓存复用及历史训练口径](evidence/tight_reuse_20261110/reuse.json)。本次没有GPU、Kit、参数更新或新增归档。

## 2026-11-10 前次重构清理：823.789063MiB

本轮依用户最新授权，实际删除完成的 `tight_residual_20261110_01` 内 **398个明确文件**：
392个旧G01/G11冻结名义预测cache（每组train166+validation30），以及6个验收用可重绘PNG。
**实际释放863,805,440 allocated bytes＝823.789063MiB；未删除训练checkpoint，未创建archive。**
通用本地ledger的`deleted_checkpoints=392`按`.pt`后缀计数，实际全是逐episode派生预测cache，不是392个模型。

旧G00/G10 selected best及G01/G11 best/last、Core218源轨迹与当前DINO/numeric/tactile共享cache、统计、nominal index/scales均保留。
全部20次开发成功/失败的numeric/trace/result及4视频、首次G11成功的完整目录、历史错误与分母均保留。
此前清理时计划重训基础；最新任务已改为直接复用旧G00/G10，0更新。已删除的旧GRU nominal payload目前仍缺失，保留索引不能代替真实预测。
历史c3386b6缓存入口和保留模型/原始数据可重新计算，未实际重建，也不承诺重建序列化字节SHA相同。

删除前逐文件核对index SHA、inode/size/mtime、同UID打开句柄和保护依赖；计划先落盘，之后按已审核清单执行。
删除后406个非候选文件identity和SHA不变，6个保留checkpoint真实CPU读回；原始400和Core218每条载荷存在、size及HDF签名检查通过。
沿用既有完整性审计，未重复读取全部源载荷做全SHA扫描，不把metadata检查冒称完整重审。
`tools.tight_finish`/诊断入口未删除；少数仍有对照用途的权重和共享cache保留，不按日期整目录删。

[逐文件SHA、原因、实际释放与读回机器证据](evidence/tight_refactor_20261110/cleanup.json)。
原机before/request/deletions/after账本位于该run的`cleanup_20261110*`；无新GPU/Kit或训练。

下文为此前独立清理，释放空间不重复累计。

## 2026-11-09 指定上一轮Core目录：289.15MiB

用户明确选择清理 `outputs/visuotactile/run/tight_state_20261107_01` 并要求逐项清单。
本次移除14个原文件路径：1个可重建重复数值cache，13个日志先无损gzip归档再删除原文件。
**实际释放303,198,208 allocated bytes＝289.152344MiB**，未累计历史释放量；不是删除14条实验轨迹。

|处理|旧root内路径|结果|
|---|---|---|
|删除重复cache|`composition_06/state_data/numeric.pt`|122,160,231逻辑bytes；208条records与当前218条cache对应记录逐字段相等|
|无损归档13日志|具体路径见[删除清单CSV](evidence/tight_palm_20261109/cleanup_files.csv)|同目录`.gz`；逐文件解压SHA与原文件SHA一致|
|保留|208条当前继承轨迹、Aux、原速参照、旧best/last、统计、manifest/split、试验结果及失败分母|旧root不能整体删除|
|保留开发cache|`diagnostic_data/numeric.pt`|部分开发RGB-D源已在历史清理中删除，不能假定可重建，故保留|

删除前检查同UID进程没有打开候选文件，逐项核对SHA、size及单硬链接；清理后9002个非候选/当前关键文件inode/size/mtime检查通过。
当前manifest全部218载荷存在且size匹配；未重新读取50GB以上源轨迹做无关全量审计，不把metadata检查冒充全SHA重审。
删除清单和压缩SHA见[机器记录](evidence/tight_palm_20261109/cleanup.json)，原执行账本在当前run的`cleanup_previous_core.json`。
没有删除唯一旧checkpoint、改变新训练数据统计、处理其他实验root、修改无关进程或重启训练。

日志原路径需要时用 `gzip -dk <原路径>.gz` 恢复，解压后SHA应与清单一致。
旧Core数值cache可在CPU上从当前 `tight_palm_20261108_01/state_data/numeric.pt` 按旧 `composition_06/state_data/data.json` 的entries提取208条records，
保留旧contract和data_sha256，经既有 `visuotactile.checkpointing.save` 写回旧numeric路径；旧统计无需重算。
也可从未改的原HDF5重建。重写序列化不保证原文件字节SHA，需记录新cache校验；不得伪造旧冻结输入验收或自动重跑旧队列。

下文为历史清理记录，当时“当前/cache有效”等表述只代表当时验收状态。

## 2026-11-08 composition_06：36.191101GiB

全部owned进程实际退出后，复用tools.project_maintenance的分类计划、PID/打开句柄、文件identity/SHA预检和逐文件journal，删除135份新未准入RGB-D载荷与1份重复checkpoint，共136文件、38,859,898,880分配bytes。原400/275及历史root在允许范围之外；225合格Core/Aux载荷（含2未选Aux）、135拒绝numeric、27学生失败轨迹、全部433数据尝试及失败分母保留。

best/last完整状态逐字段相同，保留best SHA6940b066cf8296d8e90203c0aa714f2ad780d5440c6128b89ddc5698bbbaa4da；last原SHAab119825270213c17556a13e5a00cce361b06073a49461b6c69addd8caaa8b45留账，恢复路径原子切为best硬链接，旧重复inode再删除，没有巨型备份。未删不同实验对照或成功历史。

清理后211执行源码/计划、6保护SHA、225合格payload SHA、135拒绝numeric SHA、冻结manifest/stat/cache指纹、3CLI及last完整状态读回通过；8500计数/442012draws保持，无GPU再训练或基础重审。当前可用约884.57GiB。root内cleanup_ledger_20261108.json、deletions journal、checkpoint_cleanup_mapping.json及post_cleanup_verification.json可复查。原400/275、专家、资产未改；旧7.510/2.705GiB与更早清理均不重复计算。

下文仅为既往清理记录，其中当时“当前/未训练/容量阻断”等不代表本轮最终状态。

## formal_05本次生产清理：2.705GiB，未重复计算

本次75登记尝试后按预登记yield停止。采集器仅对10条已完整写盘、State/RGB-D/触觉验收完成但逐条慢速准入被拒绝的新记录，核对原payload SHA并确认numeric证据存在后删除其RGB-D大载荷；完整账本为formal_05/collection_cleanup.jsonl。实际释放2,904,580,096分配bytes（2.705101GiB）。不是删除10次失败分母：10条均任务成功，但不符合慢速数据标准，numeric、初态、参数、事件、终止、模态验收与原payload SHA全部保留。

10条合格新慢速和54条成功原速HDF5完整保留；64个SHA及10个拒绝numeric SHA复核通过。207执行源码/计划、6保护文件SHA不变；原400/275全部payload存在且大小匹配，原全SHA审计沿用。当前模型/完整状态、专家、资产不动，没有新checkpoint可清理。全部本run自有进程已退出；未再删除其他项目或重建旧缓存。

此前1867路径/7.510GiB及更早22.7412GiB独立计数，不重复累计。本次启动前空间1018.29GiB、结束约1004.25GiB；容量已解除，停止原因是慢速准入10/20低于65%，不是磁盘。仅小型审查结果发布到origin feature，不上传数据/checkpoint或重写历史。

## 2026-11-07前次追加清理（当时容量阻断，现已解除）

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
