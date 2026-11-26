# Final15 闭合后执行命令与补充绑定建议

本文件只依据已审源码、recipe和已有非结局事实稿编写；没有执行下列命令，没有构造实际binding，也没有读取实际生命周期、评估结局、权重或缓存。冻结final15和运行源码均未修改。这里的输出名称是一个明确的新输出方案；root实际执行前须确认未使用过，不能覆盖旧尝试。

执行前置条件：root已确认原47波队列、driver/native/monitor及原两项尾部helper真实闭合，完成watcher15自己的stop/certify和每个动态backup的精确恢复；已停止并核验启动辅助16的实际身份、回收和已尝试波的恢复证据；已创建并审核实际`FINAL_RECOVERY15_BINDINGS.json`及其`ROOT_EVIDENCE_TOOLS_REVIEW15.json`。source-only签核不替代这些实际事实。后续只读取冻结结果，不再训练、选模、重试波或变更成功判据。

在一个新bash中使用以下共同参数。只给本轮离线工具屏蔽CUDA；不会修改外部进程的环境或亲和性。详细8载荷提取和后续步骤顺序运行，避免并发重复读同一权重。所有路径保持权威工作区，不从候选快照目录执行。

```bash
set -euo pipefail
set -o noclobber
PROJECT='/home/CNS2025331827/docs/bc+rl_RSS'
PY='/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python'
AUDIT="$PROJECT/outputs/diagnostics/tight_paired_dagger_20261122_01"
RUN="$PROJECT/outputs/visuotactile/run/tight_paired_dagger_20261122_01"
LEDGERS="$AUDIT/selected_ledgers_recovery15_final01"
ASSEMBLY="$AUDIT/final_assembly_recovery15_final01"
COMPUTE="$AUDIT/COMPUTE_RECORD_RECOVERY15.json"
DEST="$PROJECT/research_docs/evidence/tight_paired_dagger_20261122_01"
DRY_RECEIPT="$AUDIT/FINAL15_EXPORT_DRY17.json"
APPLY_RECEIPT="$AUDIT/FINAL15_EXPORT_APPLY17.json"
cd "$PROJECT"
final15_python() {
  env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PY" "$@"
}
test ! -e "$LEDGERS"
test ! -L "$LEDGERS"
test ! -e "$ASSEMBLY"
test ! -L "$ASSEMBLY"
test ! -e "$COMPUTE"
test ! -e "$DEST"
test ! -L "$DEST"
test ! -e "$DRY_RECEIPT"
test ! -e "$APPLY_RECEIPT"
```

先且只先执行一次详细8组件载荷提取。它核验selected权重SHA、CPU mmap、实际成功更新曝光及本组DP/后端配套身份，写8个组件JSON、summary和status。退出码0还应对应`status=passed`、`checkpoint_reads=8`、`cuda_initialized=false`。这次读取独立于已有presealed guard的8次；不能写成全实验总共只读8次。若失败，保留它已经创建的输出和实际读取计数；不要仅为下游重试而自动再提取8载荷。

```bash
final15_python "$AUDIT/extract_selected_ledgers_recovery15.py" \
  --root "$RUN" \
  --audit "$AUDIT" \
  --output "$LEDGERS"
```

然后生成实际计算清单；此输出basename是冻结exporter要求的精确名称，不能仅另存一个带17后缀的文件来代替。它只读小元数据，认证区间保留失败/等待/重叠并求并集，历史collection结束未知，GPU小时保持null。

```bash
final15_python "$AUDIT/final_compute_recovery15.py" \
  --root "$RUN" \
  --audit "$AUDIT" \
  --output "$COMPUTE"
```

再用同一个详细ledger目录组装报告。它读取全部64初态/256机会及两个CSV并交叉核验、复用8个已提取表而不再加载权重，写`DRAFT_REPORT.md`、`assembly_status.json`、实际年龄/曝光、曲线趋势、池分区和耗时清单。退出码0应对应`draft_ready_for_root_review`，并登记`actual_checkpoint_reads=0`、`actual_selected_ledger_tables_reused=8`。该目录必须是audit直接的新子目录。

```bash
final15_python "$AUDIT/final_report_assembler_recovery15.py" \
  --root "$RUN" \
  --audit "$AUDIT" \
  --output "$ASSEMBLY" \
  --selected-ledgers "$LEDGERS"
```

root在此审阅草稿开头、完整分母、绝对成功率、两组新−父和A−B的配对区间、保留/新增/丢失、阶段全分母及软件未完成；把`REPORT_FACTS10.md`和`TRAINING_FACTS11.md`中的非结局事实用于简洁定稿。不要按开发领先换checkpoint，不以不同目标空间的nominal损失排名。已有事实说明稳定持物近孔never-entry对准接管仍为0；53次never-entry接管和51次专家恢复不能代替该子群覆盖。训练拟合继续改善、固定验证未一致改善，与独立自主泛化分别回答。最终报告还须据真实记录披露辅助16的尝试、停止和任何恢复证据缺口；冻结assembler不会自动把后来的host辅助解释加入正文。若编辑`DRAFT_REPORT.md`作科学措辞定稿，应在dry-run之前完成，并保留原组装版本及根审阅记录。

接着只做dry inventory。完整CLI没有`--audit`或`--root`选项；exporter由`--project`推导固定run/audit。它的main对`status=blocked`未统一返回非零退出码，因此不能只看shell成功，必须检查JSON中的准确状态。下面把JSON输出排他保存在新文件；这不是再次提取载荷。

```bash
final15_python "$AUDIT/export_final_evidence_recovery15.py" \
  --project "$PROJECT" \
  --assembly "$ASSEMBLY" \
  --destination "$DEST" \
  --selected-ledgers "$LEDGERS" > "$DRY_RECEIPT"
final15_python - "$DRY_RECEIPT" <<'PY'
import json,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_bytes())
if value.get('status')!='dry_inventory_ready' or value.get('destination_created') is not False:
    raise SystemExit('Final15 dry inventory failed: '+str(value.get('reason',value.get('status'))))
print(json.dumps({key:value.get(key) for key in ('status','file_count','total_bytes','max_file_bytes','max_total_bytes')},indent=2))
PY
```

root检查dry inventory中的精确文件及SHA，确认逐初态表、数据/训练配置、失败和历史基线轻量证据都在清单，且没有权重、HDF5、视觉、缓存载荷、RNG tensor、视频或原始log。冻结16MiB单文件/256MiB总量和不覆盖规则保持。确认具体清单后完成已授权的本地导出：

```bash
final15_python "$AUDIT/export_final_evidence_recovery15.py" \
  --project "$PROJECT" \
  --assembly "$ASSEMBLY" \
  --destination "$DEST" \
  --selected-ledgers "$LEDGERS" \
  --apply > "$APPLY_RECEIPT"
final15_python - "$APPLY_RECEIPT" "$DEST" <<'PY'
import json,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_bytes())
destination=Path(sys.argv[2])
if value.get('status')!='export_complete' or not (destination/'MANIFEST.json').is_file():
    raise SystemExit('Final15 export did not complete')
print(json.dumps({key:value.get(key) for key in ('status','file_count','total_bytes','immutable_destination')},indent=2))
PY
```

导出目录本身不代表已推GitHub。任何后续发布按已有用户授权和项目分支约定独立处理；不重跑已完成实验。`FINAL15_EXPORT_DRY17.json`和`FINAL15_EXPORT_APPLY17.json`为后生成运行回执，不应为了加入同一已冻结binding而回写旧binding或循环重导出；它们可留本机并由后续交付索引单列SHA。

无需变更冻结工具的补充小文件绑定方式：在root第一次写实际binding和review之前，把下列已经闭合、已经审过且未被builder收集的小文件SHA加入返回映射`small_file_sha256`，保留原映射所有项。`reviewed_binding`逐项验SHA，final15 exporter已有遍历该映射的路径；`.py/.md/.json/.txt`且≤16MiB、位于项目内的普通规范路径可沿用同一总量限制。不要向`local_retained_log_sha256`额外追加一般log；它只接受冻结三边界精确并集及mandatory failed13 helper。不要修改36个继承依赖、runtime注册或六工具SHA。

| 目的 | 最小补充文件 | 作用与边界 |
| --- | --- | --- |
| watcher15关闭方法 | `close_and_certify_watcher15.py`、`FINAL15_CLOSURE_INDEPENDENT_REVIEW.md` | 绑定实际certificate内`script_sha256`所指方法及其独立审阅；不能仅凭源码签核声称真实闭合 |
| watcher15关闭链 | `TEST_CONTINUATION15_ACTIVATED.json`、`ROOT_LIVE15_IDENTITIES.json`、`ROOT_WATCHER15_STOP_INTENT.json`、`owners/test_continuation15/monitor_exit.json` | certificate已由builder绑定；这些是其实际activation、历史snapshot、排他marker身份和monitor结束的直接支撑。已被builder包含的文件不重复添加 |
| 停止的host aid16 | `startup_affinity16.py`、`test_startup_affinity16.py`、`test_independent_startup_affinity16_review.py`、`STARTUP_AFFINITY16_INDEPENDENT_SOURCE_REVIEW.md`、`ROOT_STARTUP_AFFINITY16_SOURCE_REVIEW.json`、`startup_affinity16_actual01/started.json`、`startup_affinity16_actual01/exit.json` | 保留实际执行过的源码、作者与独立测试、先前root审阅及未通过的独立发现，避免把源码缺陷隐藏或改称通过；不将auxiliary工作加入模型训练年龄或学生机会数 |
| aid16前置有界probe | `ROOT_AFFINITY16_BOUNDED.json` | 按root要求绑定已有实际probe；保留其范围和非因果解释，本文没有打开该实际记录 |
| aid16逐波真实尝试 | 仅从原33–63有限范围按root实际intent census枚举的`waveXX_intent.json`、已有`waveXX_exit.json` | 不递归加入任意文件；每个intent都必须在root审计中有结论。缺失exit保持缺失并明确恢复未知，不补造或删除intent |
| aid16实际停止/恢复证明 | `STARTUP_AFFINITY16_ACTUAL_CLOSURE_REVIEW17.json`及`.md`，仅在独立实际签核完成后添加；再加root实际枚举的activation/stop/identity小JSON | 这是root指定的计划文件名，非本文证明它已经存在或通过。核验应记录精确PID/birth、自己的STOP路径/字节/时刻/身份、服务或进程已退出、逐intent恢复/退出/缺口与全部实际元数据SHA。保留完整activated/stopped辅助证据，不只选成功的逐波回执 |

关闭工具的`test_close_and_certify_watcher15.py`、`test_independent_closure15_review.py`、独立review及此命令稿也补充绑定，以便离线复核。裸`STOP`/`STOP_RESOURCE_WATCH15`无允许后缀，不直接塞入small-file映射；由已绑定intent/certificate记录其精确SHA、设备/inode和路径。helper/aid的普通stdout日志仍留本地，由root审计记录SHA和保留路径即可。不要把未经观测的`signals_sent=0`、`restored`预期字段当作已验证事实。

源码/CLI核对结论：四个CLI的必需参数已经全部列出；提取、组装各要求新输出目录，compute要求新JSON，export要求`research_docs/evidence`直接新子目录。canonical audit目录须同时保留六工具和全部36个继承源/fixture依赖，不能只复制六文件运行。8个组件的`family_phase_curves.csv`、epoch60部署/finished元数据、own-DP缓存身份与真实选中ledger均是已有科学门，不可因为缺文件而降级。当前未发现必需CLI缺失。

一个条件性源接口缺口已先报告root、未修改：`build_binding`把三边界中`.log`放入独立local-retained映射，`reviewed_binding`接受相同并集；但`failed12_gate`对`ROOT_TEST_FRAGMENT13_BOUNDARY.json/artifact_sha256`仍只查询`small_file_sha256`。若某root13边界包含`.log`，合法builder输出会在该门被拒绝。root随后明确反馈本次实际root13边界有10项artifact且log集合为空，因此本次不命中，无须更改冻结源码。本文只记root提供的核对结论，没有打开实际boundary；不能通过把raw log混入公开小文件映射绕过此门。
