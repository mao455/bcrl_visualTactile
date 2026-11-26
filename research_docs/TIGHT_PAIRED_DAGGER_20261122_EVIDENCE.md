# A/B 配对 DAgger：证据索引

**2026-11-26更正：** 原登记的10 cm布局仅写入元数据，96个新家族的实际reset没有施加100 mm位移。旧轻量包保持不可变，原四组0/64不替换；关于实际分布、权重关系、故障与接管的新证据以[原机复核](TIGHT_MACHINE_REVIEW_20261126.md)及其[证据索引](evidence/tight_machine_review_20261126_01/README.md)为准。

[短报告](TIGHT_PAIRED_DAGGER_20261122.md)为最终科学结论；主包内DRAFT_REPORT是保留的机器草稿，未包含全部最终限定，不替代短报告。原始失败及其时间顺序均保持。

主包 [MANIFEST](evidence/tight_paired_dagger_20261122_01/MANIFEST.json) 固定 6734 个文件、178,149,499 字节（约169.90 MiB，另加manifest和README）；`files/`保留项目相对路径。仅包含小型JSON/CSV、源码、Markdown和8张训练曲线图。两份精确缓存pipeline元数据例外已列入清单，其余权重、HDF5、视觉／特征、缓存、RNG张量、视频及普通原始日志留本机。

辅助区间及一次缺少PYTHONPATH的组装失败在[独立补充包](evidence/tight_paired_dagger_20261122_01_supplement21/README.md)，主包保持不可变。补充区间使用主包逐初态CSV的SHA与事前准备的方法脚本；它保留原bootstrap和软件失败编码。

当前运行已闭合；64个初态均已消耗机会，只有55波软件完整。`primary_summary.json`的`final_execution_complete=false`指软件轨迹未全完成，不表示仍有待派发队列。`observed_initial_pair_failures=8`在这里包括8个无初始化记录的波，实际56波核验通过，不能把缺测称为观测不匹配。

## 自主评估

| 证据 | 文件 |
| --- | --- |
| 封存64家族与四套完整策略身份 | [registration.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/evaluation/test/registration.json) |
| 主结果、阶段全分母与运行状态 | [primary_summary.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/evaluation/test/summary/primary_summary.json) |
| 逐初态配对CSV（64行） | [per_initial_state.csv](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/evaluation/test/summary/per_initial_state.csv) |
| 逐rollout CSV（256行） | [rollouts.csv](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/evaluation/test/summary/rollouts.csv) |
| 自主权限与295生产源码核验 | [AUTONOMY_CONTRACT_REVIEW17.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/AUTONOMY_CONTRACT_REVIEW17.json) |
| 仅开发轨迹的失败与命令链诊断 | [DEVELOPMENT_DIAGNOSIS17.md](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/DEVELOPMENT_DIAGNOSIS17.md) |

## 数据与采集

| 证据 | 文件 |
| --- | --- |
| 总实验合同、采样分布、固定家族与阶段规则 | [experiment.json](evidence/tight_paired_dagger_20261122_01/files/configs/experiments/paired_dagger_20261122/experiment.json) |
| 总冻结登记 | [registration.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/registration.json) |
| 新增家族登记与历史使用去重 | [training_expansion_registry_admission01.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/training_expansion_registry_admission01.json) |
| 碰撞合法性导致的家族替换修订 | [training_family_admission_amendment.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/training_family_admission_amendment.json) |
| 原始采集规则与机会登记 | [registration.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/collection/registration.json) |
| 采集汇总（保留96机会） | [collection_summary.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/collection/collection_summary.json) |
| 全部专家示范尝试与失败 | [results.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/experts_shared/results.json) |
| 共享24条专家示范准入 | [result.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/experts_shared/result.json) |
| 完整历史、mask及合法标签审计 | [integrity.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/collection_integrity_recovery05_actual01/integrity.json) |
| 采集噪声审计 | [noise_audit.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/collection_noise_recovery05_actual01/noise_audit.json) |
| 接管时held12、阶段覆盖与采集分母事实 | [REPORT_FACTS10.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/REPORT_FACTS10.json) |

## 训练与监督

| 证据 | 文件 |
| --- | --- |
| 实际重建父本训练年龄 | [actual_parent_ages.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/actual_parent_ages.json) |
| 采集父本配套身份 | [parent_pairs.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/parent_pairs.json) |
| 本轮固定第60轮／等更新／采样预算 | [budget_registration.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/learning/budget_registration.json) |
| 逐组件优化配置同一性 | [ROOT_PAIRED_OPTIMIZER_CONFIG_CHECK17.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/ROOT_PAIRED_OPTIMIZER_CONFIG_CHECK17.json) |
| A数据、划分、四类采样与归一化 | [data.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/learning/A/state_data/data.json) |
| B数据、划分、四类采样与归一化 | [data.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/learning/B/state_data/data.json) |
| 终末8载荷实际曝光提取状态 | [extraction_status.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/selected_ledgers_recovery21_final01/extraction_status.json) |
| 8组件曝光、原生入口槽与配套SHA | [actual_exposure_and_age.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/final_assembly_recovery21_final02/actual_exposure_and_age.json) |
| 逐阶段／家族训练与验证趋势解释 | [TRAINING_FACTS11.md](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/TRAINING_FACTS11.md) |
| 逐阶段／家族训练与验证完整数值 | [TRAINING_FACTS11.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/TRAINING_FACTS11.json) |
| 未实施编码器读出的可行性证据 | [encoder_probe_feasibility.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/encoder_probe_feasibility.json) |

8组件的`family_phase_curves.csv/png`、每轮真实采样deck、训练结束／deployment配置和逐轨曝光表也全部包含在主包。曲线位置遵循`files/`后的原运行路径，例如：

- [A DP 曲线](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/learning/A/training/family_phase_curves.png)
- [A Transformer 曲线](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/learning/A/backend/A/training/family_phase_curves.png)
- [B DP 曲线](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/learning/B/training/family_phase_curves.png)
- [B Transformer 曲线](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/learning/B/backend/B/training/family_phase_curves.png)

## 运行与复查

| 证据 | 文件 |
| --- | --- |
| 真实成本：认证时间并集、失败、弃用更新 | [COMPUTE_RECORD_RECOVERY15.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/COMPUTE_RECORD_RECOVERY15.json) |
| 实际队列机会消耗与子进程回收 | [project_complete.json](evidence/tight_paired_dagger_20261122_01/files/outputs/visuotactile/run/tight_paired_dagger_20261122_01/owners/test_continuation15/project_complete.json) |
| 最终driver闭合 | [closure.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/driver_test_continuation15/closure.json) |
| 主watcher实际关闭与110项恢复 | [WATCHER15_ACTUAL_CLOSED.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/WATCHER15_ACTUAL_CLOSED.json) |
| 第58波异常后自有进程清理的独立核验 | [ROOT_POST_EXCEPTION58_CLEANUP20_REVIEW.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/ROOT_POST_EXCEPTION58_CLEANUP20_REVIEW.json) |
| 第60波启动卡住的只读诊断（根因未知） | [STARTUP60_READONLY21.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/STARTUP60_READONLY21.json) |
| 主只读监控中断／未知退出 | [ROOT_MONITOR15_INTERRUPTION_REVIEW21.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/ROOT_MONITOR15_INTERRUPTION_REVIEW21.json) |
| 启动调度辅助16的实际恢复 | [STARTUP_AFFINITY16_ACTUAL_CLOSURE_REVIEW17.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/STARTUP_AFFINITY16_ACTUAL_CLOSURE_REVIEW17.json) |
| 未写入的辅助18失败 | [HOST_INIT18_ACTUAL_FAILURE_CLOSURE_REVIEW.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/HOST_INIT18_ACTUAL_FAILURE_CLOSURE_REVIEW.json) |
| 辅助19实际恢复及47波独立快照限制 | [HOST_INIT19_ACTUAL_CLOSURE_REVIEW.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/HOST_INIT19_ACTUAL_CLOSURE_REVIEW.json) |
| 未启用清理方案的时钟门阻断 | [FINAL_ERROR_CLEANUP20_ROOT_CLOCK_BLOCKER21.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/FINAL_ERROR_CLEANUP20_ROOT_CLOCK_BLOCKER21.json) |
| 不可变源文件／元数据登记 | [FINAL_RECOVERY15_BINDINGS.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/FINAL_RECOVERY15_BINDINGS.json) |
| 历史失败退出码的精确21修订 | [FINAL_RECOVERY21_AMENDMENT.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/FINAL_RECOVERY21_AMENDMENT.json) |
| 21适配器独立源码／8项CPU边界审阅 | [FINAL_RECOVERY21_INDEPENDENT_SOURCE_REVIEW.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/FINAL_RECOVERY21_INDEPENDENT_SOURCE_REVIEW.json) |
| 原15提取在权重读取前失败的回执 | [extraction_status.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/selected_ledgers_recovery15_final01/extraction_status.json) |
| 成功终态组装：0额外载荷读取／8表复用 | [assembly_status.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/final_assembly_recovery21_final02/assembly_status.json) |
| 68项相关检查回执 | [ROOT_FOCUSED_CHECK13.json](evidence/tight_paired_dagger_20261122_01/files/outputs/diagnostics/tight_paired_dagger_20261122_01/ROOT_FOCUSED_CHECK13.json) |

完整本机路径：`/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_paired_dagger_20261122_01`；审计根为同工作区的`outputs/diagnostics/tight_paired_dagger_20261122_01`。历史基线、原始训练池、所有科学候选、失败及归档恢复清单均保留。第58波异常后的清理只作用于登记的自有进程；不将其写成自然退出、有效评估或启动超时。历史collection退出回执、主只读monitor退出时刻均缺失，不能补造。

旧A/B entry前后端训练年龄不齐，经共同180重建父本；八组件第60轮身份和更新数均已核验。名为`best.pt`的部署文件按固定轮次选取，不是本轮开发胜出checkpoint。新A/B对应池不同，不能借nominal损失给不同目标空间排名。

辅助16/19只记录实际自有线程调度与恢复，外部进程未操作；19第47波恢复依赖协议与持久回执，缺少独立同期TID快照。清理方案20的合成测试虽通过，实际时钟门不符，始终未启用。没有零干扰、持续加速或隐藏状态精确同态证明。

以下只验证已导出小文件的身份，无需启动仿真或加载模型。请在项目根运行：

```bash
python - <<'PY'
from pathlib import Path
import hashlib, json
base = Path('research_docs/evidence/tight_paired_dagger_20261122_01')
manifest = json.loads((base / 'MANIFEST.json').read_text())
assert manifest['status'] == 'export_complete'
for row in manifest['files']:
    raw = (base / row['destination']).read_bytes()
    assert len(raw) == row['bytes']
    assert hashlib.sha256(raw).hexdigest() == row['sha256']
print('verified', len(manifest['files']), 'files')
PY
```

主包中的闭合验证／训练／采集入口仅用于审阅，不应直接重启已消耗实验。复现完整模型前按归档清单恢复所需缓存，并使用新的独立run与封存初态；本轮不自动重训或重测。
