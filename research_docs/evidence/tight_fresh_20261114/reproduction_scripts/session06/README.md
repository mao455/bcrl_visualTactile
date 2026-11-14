# Session06 实际审计脚本归档

这里保留原研究机上实际使用的 **10 个脚本原字节**：最初核对的8个，以及随后明确追加的2个已完成B16审计脚本。它们支持已公开的持物接管、目标可观察性、近目标根部距离分析，以及已报告的磁盘预算；不会将回看可接管状态称作新恢复数据。

[manifest.json](manifest.json) 逐文件记录原绝对路径、SHA256、输入、原输出报告SHA、代码依赖、输出覆盖风险及复现限制。归档仅包含源码与小型登记信息；没有复制原始轨迹、权重、DINO缓存、图像、详细私有报告，也没有重新执行这些脚本。

| 脚本 | 支持的结论 / 访问范围 |
| --- | --- |
| [held_recheck_retrospective_01/review.py](held_recheck_retrospective_01/review.py) | 第6/7/8批真实学生前缀的持物窗口与被实际接管截断的未来；只读已关闭numeric |
| [collection09_census_01/audit_closed_first_wave.py](collection09_census_01/audit_closed_first_wave.py) | 第9批首个A波的6个已关闭持物案例 |
| [collection09_A64_review_01/audit.py](collection09_A64_review_01/audit.py) | A64完整角色切片，24个持物时序与合法专家动作；不代表整个128机会完成 |
| [target_observability_01/audit_numeric.py](target_observability_01/audit_numeric.py) | 原生目标/阶段/剩余时钟；另读取同一训练轨迹的两个RGB时刻作相等性检查 |
| [target_observability_01/export_frames.py](target_observability_01/export_frames.py) | 公开可见性判断所用的私有真RGB导出程序；会生成12张原像素PNG与4张拼图，图像仍只留本机 |
| [near60_root_orientation_01/audit.py](near60_root_orientation_01/audit.py) | 8个near60真实前缀与6个独立已成功持物纠偏的距离/角度分析 |
| [near60_root_orientation_01/hypothetical_root60.py](near60_root_orientation_01/hypothetical_root60.py) | 历史前缀第265步满足新条件；没有执行专家恢复、没有新增入池数据 |
| [disk_projection09_01/audit.py](disk_projection09_01/audit.py) | 小JSON与文件stat构成的当时磁盘估算；重新查询不会恢复历史文件系统快照 |
| [collection09_B16_review_01/audit.py](collection09_B16_review_01/audit.py) | 已关闭B首16与相同初态家族A16；227步持物后合法专家动作及865步长恢复的783步合法标签 |
| [collection09_B16_review_01/summarize.py](collection09_B16_review_01/summarize.py) | 上述闭合切片的解释、阶段计数和实际初始状态差异；只读JSON及日志哈希 |

采集源码与分析源码是两项不同依赖。第9批实际采集使用 `bc+rl_RSS_fresh_collect09_20261114`、提交 `87577e84f5f9183ba739e483b34729029629bb99`；当时CPU分析从 `bc+rl_RSS_continuous_vision_20261114` 导入报告工具。旧采集冻结树没有后来加入的 `tight_collection_summary`，也没有 `held_root60` 候选，不能直接用它执行全部分析。

清单给出可重建的分析代码参考提交：历史持物回看依赖SHA匹配 `9e8c408`，后续报告工具参考 `4dfd850`，新root60与B16参考 `3ee97c5`。这些参考来自已记录文件SHA或所需API；没有记录的原运行Git HEAD、shell argv保持未知，不从当前HEAD反推。目标可观察性审计还会对原机的采集冻结目录进行源码哈希核对。已知的原源码SHA匹配范围逐项写入清单。

**不要直接运行此归档里的脚本，也不要覆盖旧证据。** 多数输出位置由 `__file__` 决定，少数仅以 `assert` 检查输出不存在再写入；帧导出程序的PNG保存可以覆盖同名文件。不得使用 `python -O` 绕过这些断言。

如需复算，先在Git目录之外建立新的私有审计身份，把所需脚本原字节复制进去，并按清单准备只读的原输入与小型先决报告。保持每个脚本要求的相邻目录关系：near60分析读取其他审计目录；B16分析把脚本目录的上两级作为session根部。确认所有输出文件名都不存在，然后从相应参考代码checkout设置导入路径，以CPU Python执行新的私有副本。若必须调整绝对路径或输出位置，应另存适配版本并记录新SHA，不修改本归档。RGB导出的新图像仍留私有目录。

重新运行的时间戳、输出路径、增长中的结果索引及文件系统统计可能改变，因此新报告是新测量，不保证与历史JSON逐字节一致。复现结论仍需原研究机保留的已关闭numeric/raw输入；本归档不是可脱离数据运行的示例。

[launch_verification.json](launch_verification.json) 核对第9批实际systemd服务的 `ExecStart`、工作目录、13958秒硬上限、30秒清理、文件句柄与线程参数；与[已公开启动注册](../../collection09_registration/startup.json)和原机注册逐项相同，没有遗漏的配置需要另行备份。那里已经包含采集与后续cycle的完整argv。**登记了cycle命令不表示它已经执行或完成。** 已消失的第8批临时systemd单元只能依靠保留的原阶段/owner收据，不使用当前默认属性冒充历史配置。

归档验证仅做了AST语法解析、10份原文件与归档字节/SHA比较、已记录脚本SHA核对和登记参数只读对照；没有启动GPU、Kit、训练或任何旧审计程序。
