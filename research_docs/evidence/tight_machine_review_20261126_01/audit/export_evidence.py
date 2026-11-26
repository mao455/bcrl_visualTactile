"""Lightweight review receipt; raw sensor/model payloads remain local."""
from pathlib import Path
import json,shutil,hashlib,time,subprocess,os
R=Path.cwd();A=R/'outputs/diagnostics/tight_machine_review_20261126_01';ROOT=R/'outputs/visuotactile/run/tight_machine_review_20261126_01'
B=R/'research_docs/evidence/tight_machine_review_20261126_01'
assert not B.exists();B.mkdir(parents=True)
rows=[]
def copy(p,dest):
    out=B/dest;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
    raw=out.read_bytes();rows.append(dict(source=str(p.relative_to(R)),destination=str(dest),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
for p in sorted(A.rglob('*')):
    if not p.is_file() or p.is_symlink():continue
    if p.suffix not in ('.json','.jsonl','.csv','.gz','.py','.md','.patch','.log'):continue
    if p.stat().st_size>16*2**20:continue
    copy(p,Path('audit')/p.relative_to(A))
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or p.is_symlink() or any(k in p.parts for k in ('state_data','resources')):continue
    if p.suffix not in ('.json','.jsonl','.csv','.log'):continue
    if p.stat().st_size>16*2**20:continue
    copy(p,Path('runs')/p.relative_to(ROOT))
for folder in ('tools','visuotactile'):
    for p in sorted((R/folder).rglob('*.py')):copy(p,Path('source')/p.relative_to(R))
for p in sorted((R/'configs/experiments/machine_review_20261126').glob('*.json')):copy(p,Path('config')/p.name)
for name in ('tests/test_machine_review.py','tests/test_training_expansion.py','research_docs/TIGHT_MACHINE_REVIEW_20261126.md',
             'outputs/maintenance_20261123/README_MACHINE_REVIEW_20261126.md'):
    p=R/name;copy(p,Path('context')/name)
text='''# 原机复核：证据索引

[短报告](../../TIGHT_MACHINE_REVIEW_20261126.md)为结论。原机运行根为
`outputs/visuotactile/run/tight_machine_review_20261126_01`，审计根为
`outputs/diagnostics/tight_machine_review_20261126_01`。不重启旧session。

| 内容 | 可离线复查文件 |
| --- | --- |
| 实际100mm布局差异与最早来源 | [layout_contract_comparison.json](audit/layout_contract_comparison.json) |
| 四组256逐例、原时间/恢复worker/原因 | [evaluation256_status.csv](audit/evaluation256_status.csv)、[wave64_status.json](audit/wave64_status.json) |
| 模型年龄与16载荷SHA | [model_identity_normalization.json](audit/model_identity_normalization.json) |
| 实际归一化buffer | [normalizer_buffer_fingerprints.json](audit/normalizer_buffer_fingerprints.json) |
| 恢复的更新/EMA/优化器/RNG/deck | [resume_boundary_audit.json](audit/resume_boundary_audit.json) |
| 正控逐次及同seed实际初态差异 | [positive_controls.json](audit/positive_controls.json) |
| 训练/开发前缀逐类完整表 | [train_development_stage_census.csv](audit/train_development_stage_census.csv)、[old_pool_prefix_stage_supervision.json](audit/old_pool_prefix_stage_supervision.json) |
| 原96采集机会（不重复恢复目录） | [all_collection_opportunity_stage_audit.json](audit/all_collection_opportunity_stage_audit.json) |
| 历史13条实际交权即满足屏幕的救回 | [historical_exact_never_gate_rescues.csv](audit/historical_exact_never_gate_rescues.csv) |
| 新26机会、含中断39合法tick | [diagnostic_branch_results.csv](audit/diagnostic_branch_results.csv) |
| 真实分支在交接前已分叉 | [branch_state_pairing.json](audit/branch_state_pairing.json) |
| 两个真实救回的实际冻结配置/回执 | [registration.json](runs/never_gate_pilot07_B72_training/registration.json)、[complete](runs/never_gate_pilot07_B72_training/rollouts/result.json) |
| 同历史输出、nominal/残差/投影/最终命令 | [fresh_rescue_action_chain.csv](audit/fresh_rescue_action_chain.csv) |
| 源B72自重放逐槽误差 | [source_self_replay.json](audit/source_self_replay.json) |
| 专家首合法动作与随后位姿变化 | [fresh_rescue_geometry.json](audit/fresh_rescue_geometry.json) |
| 旧已训练救回与新后缀拟合误差 | [旧](runs/fixed_history_fit01/inference/summary.json)、[新](runs/fresh_rescue_history02/inference01/inference/summary.json) |
| 原实际池/400保护依赖 | [retained_actual_dependencies.json](audit/retained_actual_dependencies.json) |
| 逐文件删除：SHA/inode/birth/mtime/实际释放 | [retirement_deletions.jsonl](audit/retirement_deletions.jsonl) |
| round2同期变化、剩余大户与消费者 | [disk_final_accounting.json](audit/disk_final_accounting.json)、[逐文件变化gz](audit/file_changes_since_round2.csv.gz) |
| 最终模型身份、固定记录字段一致、进程/时间 | [final_verification.json](audit/final_verification.json) |
| 本机/公开60e7b1f/修改前后源码SHA | [source_public_comparison_final.json](audit/source_public_comparison_final.json) |
| 完整方法和未判定限制 | [METHOD_AND_LIMITS.md](audit/METHOD_AND_LIMITS.md) |

`source/`为终核主源码，`config/`为本次配置，`runs/`保留逐次成功/失败/未派发身份及退出回执。
两个主检查分别80通过/2跳过及门精度/隔离5通过，互有重叠，不相加为独立测试总数。
准备失败、拼错的测试路径和启动/资源中断同样保留。主科学模型未更新。

400原专家、当前261唯一原始池/特征、正式历史/父/新模型、两个真救回完整观测/特征以及39合法tick中断分支留本机。
退休nominal、last、无标签传感副本和指定视频已按账本删除，不能声称仍可直接读取；所需核心数值/原始输入保留。
原四组64终测分母不可替换，不能用终测挑点或选模。附件文件未向本会话暴露，未认证附件身份。

`MANIFEST.json`对包内每个文件记录SHA、大小与原机相对路径；所有引用为真实已有结果。
全项目分配块快照包含SQLite自身的排除规则，索引/报告/轻量导出在快照之后生成；`delivery_accounting.json`单列交付体积，不能把它当作又一次清理释放。
完整逐文件birth/mtime快照本机为`audit根/disk_before.sqlite`与`disk_after.sqlite`；round2基线为`outputs/maintenance_20261123/round2`。
'''
(B/'README.md').write_text(text)
raw=(B/'README.md').read_bytes();rows.append(dict(source='generated index',destination='README.md',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
(B/'MANIFEST.json').write_text(json.dumps(dict(utc=time.time(),status='export_complete',files=rows,
    source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    scope='Only lightweight receipts/source/config; original immutable paired package retained',
    raw_payloads_included=False),indent=2)+'\n')
seen=set();allocated=0
for p in B.rglob('*'):
    s=p.lstat();key=s.st_dev,s.st_ino
    if key not in seen:seen.add(key);allocated+=s.st_blocks*512
receipt=dict(utc=time.time(),export_files=len(rows)+1,export_logical_bytes=sum(x['bytes'] for x in rows)+(B/'MANIFEST.json').stat().st_size,
    export_allocated_bytes_excluding_this_receipt=allocated,
    post_disk_snapshot=True,freed_bytes=0,manifest_sha256=hashlib.sha256((B/'MANIFEST.json').read_bytes()).hexdigest())
(B/'delivery_accounting.json').write_text(json.dumps(receipt,indent=2)+'\n')
(A/'delivery_accounting.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
