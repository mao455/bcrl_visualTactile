# Tight-Insertion 工程统一（2026-11-18）

当前唯一研究代码入口为 `/home/CNS2025331827/docs/bc+rl_RSS`，分支 `feature/tight-state-foundation`。本轮从现场最新451f591接续，未训练、未增加DAgger、未改变方法或晋升130候选。72阶段完整A/B仍为保护基准。

## 全目录实查与迁移

[机器可读131路径清单](evidence/project_unification_20261118/directories.json)逐项列出性质、HEAD/分支、文件/代码数、磁盘占用、大文件、独有提交、原未提交内容、依赖、迁移结果和新位置。完整remotes、逐文件Git差异、忽略文件、路径引用、内容hash、进程cwd/fd/maps/环境变量名均在本机 `outputs/project_archive/20261118/audit/`；不公开环境变量值。

实查结果是1个主仓库和130个共享同一Git对象库的工作树，不是131份独立clone。多数工作树对应有限实验冻结源；仅凭名称、提交或工作树元数据无法可靠认定具体由哪个Codex agent创建，因此清单明确标为无法归属，没有猜测“agent临时目录可删”。移动前复查活动进程cwd/fd，没有移动正在被研究worker使用的树。

原目录HEAD为3e11d62，本地CHANGELOG和HANDOFF修改先保存为私有分支 `archive/unify-20261118-original` 的720201c；`narrow_followup`脏文档也保存到 `archive/unify-20261118-state-foundation`，释放研究分支后，原目录切换研究分支并快进至451f591。没有将旧源覆盖新源，也未采用强制reset。迁移命令回执见[main_migration.json](evidence/project_unification_20261118/main_migration.json)，完整原始patch保存在本机audit。

130旧树均用 `git worktree move` 移入 `outputs/project_archive/20261118/worktrees/<旧名称>`，再 `git worktree lock`。旧docs路径保留兼容软链接，因此历史checkpoint、manifest、服务和冻结源码路径继续解析；这些链接不是新的当前代码入口。原主目录及其outputs位置不变，没有复制数百GB训练资产。

| 独有内容/冲突 | 处置 |
|---|---|
| continuous_vision的最新研究代码451f591 | 已通过Git快进进入原主目录；旧树只读归档 |
| 原主目录与narrow_followup脏研究说明 | 私有分支与完整patch保留；不拼入当前方法说明 |
| dagger_continue独有4ac5128（旧条件追加DAgger实现/测试） | 私有archive tag及完整树保留；未合并旧方法/自动追加逻辑 |
| native_v2独有50c06a1（旧State评估暂停及结果） | 私有tag与完整树保留；未修改当期实验结果 |
| control_audit、dagger_perf脏报告及未跟踪证据 | 原路径内容/status保持，归档保留；不相互覆盖同名报告 |
| visual_report未跟踪tight_fresh_visual_aux_report.py | 与当前主线逐字节相同；归档仍保留原文件 |
| 未分类/忽略文件、失败证据、模型与缓存 | 全部保留；没有按目录名称删除 |

所有旧HEAD均有本地 `archive/unify-20261118/<目录名>` tag保护；这些私有历史与原始资产不随本次源码推送公开。还保留130个注册的归档worktree，因为它们有独立冻结历史/路径依赖；不是宣称已删除所有Git工作树。后续新实验只登记配置和outputs/run，不再新增docs平行项目。

## 内部实现整理与保留边界

[源码依赖审计](evidence/project_unification_20261118/implementation_audit.json)记录226处日期脚本或anchor_continue引用。本次不是仅把相似函数一律删除：

| 类别 | 实查结论与处理 |
|---|---|
| paired_eval与video_eval共有配对安装器 | AST一致；抽到visuotactile/paired_execution.py，历史入口转发 |
| video_eval完整策略视频执行 | 抽到tools/tight_evaluate.py，旧模块转发，保留原视频与退出钩子行为 |
| 各日期owner/monitor/recover/launcher | 历史deadline、故障恢复和采样条件不同；保留冻结复现。当前入口统一为tight_project显式队列和单owner监控 |
| tight_fresh_train、tight_fresh_backend、tight_finish residual_cache、anchor_continue | DP训练、后端准备、缓存和Transformer训练承担不同职责；统一调度，保留数值实现 |
| A/B、普通/重复deck、早期曝光/geometry | 数学与监督合同不同，不能因外观相似合并；由既有配置显式选择 |
| 日期模块动态import及运行时函数安装 | 当前配对/视频不再依赖日期实现；其他历史隐式依赖仍保留并登记。配对和视频钩子仍在隔离Kit worker安装，不声称已清除所有monkeypatch |
| 历史控制/几何模块 | 仍有复现或间接依赖，未证明完全无依赖，未删除或替换其数学行为 |

因此本轮落实了单一当前入口和已证明等价的公共实现提取，没有进行未经验证的全仓核心改写。保留历史代码不是将其当作新的当前训练入口。

## 统一目录与命令

```text
bc+rl_RSS/
  configs/tight_project.json         当前路径、模型阶段和接口登记
  tools/tight_project.py             唯一当前作业入口/owner
  tools/tight_evaluate.py            完整策略评估worker
  tools/tight_fresh_*.py             当前登记worker与保留历史组件
  visuotactile/                      DP、Transformer、历史、标签、控制、缓存
  scripts/                          冻结历史实验，不再复制为新实验入口
  assets/、预训练资产及现有环境链接   保持内容与解析位置
  research_docs/                    合同、当前交接、公开小证据
  outputs/visuotactile/
    data/close_v4/                  原400
    run/tight_long_20261117_01/round05/learning/  72完整A/B
    run/tight_recovery_20261118_01/learning/      130池与候选
    run/、demos/                    当前/历史实验资产，不进Git
  outputs/project_archive/20261118/
    worktrees/                      130锁定旧树
    audit/                          原始盘点/依赖/迁移/去重回执
    validation/                     推理载荷、恢复检查、Kit验收
```

在主目录中使用已安装环境：

```bash
cd /home/CNS2025331827/docs/bc+rl_RSS
.venv_isaacsim/bin/python -m tools.tight_project status
.venv_isaacsim/bin/python -m tools.tight_project train --job /ABSOLUTE/registered_train.json
.venv_isaacsim/bin/python -m tools.tight_project dagger --job /ABSOLUTE/registered_dagger.json
.venv_isaacsim/bin/python -m tools.tight_project evaluate --job /ABSOLUTE/registered_evaluate.json
```

三个执行命令需要具体登记job，不会自行选择新预算或启动历史实验。本轮不准备任何自动生效的新训练/DAgger授权。加 `--dry-run` 可仅检查展开后的argv。

job格式如下（为已完成真实验收作业的结构示例；原作业再次执行会被deadline/已尝试身份拦截）：

```json
{
  "version": "tight_project_job_v1",
  "operation": "evaluate",
  "resource_root": "/ABSOLUTE/finite_session",
  "stages": [
    {"entry": "evaluate", "args": ["--output", "/ABSOLUTE/prepared_evaluation"]}
  ]
}
```

`train`允许frontend、backend_prepare、backend_cache、backend_train、backend_offline；`prepare`允许aggregate、visual_cache、dagger_prepare；`dagger`只执行已准备的dagger；`evaluate`只执行已准备的完整策略评估。每阶段args沿用worker真实CLI，模型、数据、预算、精度与监督条件来自已登记的配置，不能靠新入口默认值替换旧合同。先用prepare/既有配置生成器准备工件，再执行有序队列。DP更新后仍需刷新匹配后端输入。

job.resource_root与各worker配置的resource/session root必须指向同一有效有限session。统一owner不创建/重设session截止、不自动重试；记录job与source commit、单owner锁、持续监控、deadline/STOP退出和子进程回收，SIGTERM/SIGHUP也进入清理。registry是目录登记，不会静默重写历史SHA绑定配置或提供跨盘透明重定位。原outputs已是共享资产根，无需再次移动大载荷。

## 验收证据与实际限制

| 检查 | 结果/范围 |
|---|---|
| 当前模块导入 | 当前调度worker及模型/配对模块从统一主目录导入；真实Kit完整链路已运行 |
| 72阶段完整A/B | 两份DP、两份匹配Transformer、归一化/配置及必要缓存实际加载；未替换为130候选 |
| 同记录迁移前后推理 | A/B各25tick×2lane，从真实前缀重建完整历史；最大绝对差0。对原记录self replay误差A6.85e-7/B8.87e-7rad，不宣称历史GPU回放逐位一致 |
| 标签/历史/mask/控制 | 聚焦CPU回归通过，日志与文件清单见test_validation.json；网络、loss、控制数学未改 |
| 原400与当前池 | 原400全部HDF5 SHA通过；当前train130+val6全量SHA通过；72+val6校验及必要特征路径通过，共536个不同原始载荷SHA核验 |
| 模型/配置保护 | 16个保护文件迁移前后SHA相同，见protected.json；模型/数据/优化器未删除 |
| 八份旧断点恢复 | 72与130各A/B×DP/后端，在CPU调用实际恢复函数，model/EMA/Adam/sampler/RNG通过；0优化器更新 |
| 真实仿真 | 一次2lane Kit，A/B各61控制步，真实三RGB-D视角、120/60/5Hz、完整DP+Transformer和29维目标链路；软件正常完成退出，122步属于截断工程验收，不计成功率 |
| 当前路径/归档 | 130别名解析、Git worktree注册/lock、原dirty状态保留；原扫描25398链接新增失效0 |

恢复验收修复了实际旧缺陷：130轨迹使用190次重复访问deck，旧后端恢复误要求order长度等于130。现在按数据验证的每轨迹访问次数逐项核对Counter，精确保留已保存顺序与cursor，不重抽样、不改采样权重。普通和stratified历史合同不改变。真实模型CPU恢复不代表授权重启已截止的旧run；冻结source/config identity仍会拒绝未经登记的源码漂移，不绕过这些保护。

**未消除的历史缺失：迁移前已有18494个失效软链接，主要指向历史已缺失的缓存、训练副本或旧输出。** 原始逐链接清单在 `outputs/project_archive/20261118/audit/preexisting_broken_links.json`，本次保留，没有伪造空文件或删除链接来制造“全通过”。因此全仓任意历史实验可无条件复现、所有历史路径零失效两项不能声称成立。当前受保护的400/72/130依赖已单独核验；没有新失效链接。未知缺失载荷无法由工程迁移自动重建。

归档read-only范围是受跟踪普通文件及Git worktree lock；未知/未跟踪文件保持原内容和权限，目录不设不可逆immutable属性。归档硬链接文件不能原地解锁编辑；需要复用时复制到当前工作区。当前可写项目不与归档去重文件共享新增硬链接。

## 磁盘、RAM与显存

实查原131树（不跟随外部软链接）分配占用合计975,332,921,344字节（908.35GiB）；归档后主项目含全部旧树，按唯一inode计941,347,536,896字节（876.70GiB）。两次扫描间净减少31.65GiB，包含新增验收载荷、审计文件和Git变化。

实际去重48842个SHA256一致、模式兼容的归档受跟踪大文件；回收分配空间34,713,960,448字节（32.33GiB），通过归档内部只读硬链接共享内容实现。未删除checkpoint/训练数据/缓存/视频/失败证据；不把硬链接路径逻辑大小相加误当实际磁盘占用。清理是增量归档过程中进行，最终再完成一致性与保护核验；不是声称先完成全部验收才开始去重。

终态文件系统空闲36,922,535,936字节（34.39GiB），会随其他任务变化。磁盘不是RAM/显存：迁移后推理worker峰值RSS约3.03GiB，CUDA allocator allocated约0.61GiB/reserved约0.64GiB；不含驱动/renderer，不能冒充全卡峰值。Kit资源逐样本保存在validation/simulation/resources，最终自有研究worker候选0、project_exit.children_reaped=true。

公开交付只包含源码、配置、文档和小回执；原始载荷、私有归档分支/tag保持本机。提交推送仅origin/feature/tight-state-foundation。
