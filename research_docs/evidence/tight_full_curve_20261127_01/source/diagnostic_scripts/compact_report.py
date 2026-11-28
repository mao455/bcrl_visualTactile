"""Keep the public result report short; retain full methods in its evidence pack."""
from pathlib import Path
import datetime
import json

P = Path.cwd()
D = P / 'outputs/diagnostics/tight_full_curve_20261127_01'
B = P / 'research_docs/evidence/tight_full_curve_20261127_01'
report = P / 'research_docs/TIGHT_FULL_CURVE_20261127.md'
read = lambda p: json.loads(p.read_text())
assert (D / 'audits_complete.json').exists()
interpretation = read(D / 'final_interpretation.json')
assert interpretation['reviewed_actual_closed_results'] is True
assert interpretation['coverage_fit_generalization']
full = report.read_text()
assert '本轮补采正确100mm布局完整专家24条' in full
B.mkdir(parents=True, exist_ok=True)
(B / 'METHODS.md').write_text(full.replace('](evidence/tight_full_curve_20261127_01/', '](').replace('# Tight-Insertion：正确布局完整示范与固定架构训练曲线', '# 完整方法、运行中断与数值口径', 1))
lead = full.split('阶段计数全部使用整批分母；')[0].rstrip()
# Full phase numerators remain in the small CSV and detailed evidence report.
lead = lead.split('| 末段完整学生 / 集合 |')[0].rstrip()
runtime = read(D / 'runtime_accounting.json')
stamp = lambda x: datetime.datetime.fromtimestamp(x, datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
lines = [lead, '', interpretation['coverage_fit_generalization'], '',
    '补采24条正确100mm布局完整专家示范、7698合法tick，30次实际reset含6次动作前碰撞拒绝。原请求24个family中22个通过，以2个替补补足；旧池60个family变为62个，独立family只增加2个。旧未位移24条明确标记保留，原400、全部合法池及历史模型保持。本轮无新增学生纠偏。', '',
    '固定网络、DINO、传感输入、归一化、标签和原生120/60/5Hz。两条件各从本组同一完整起点训练；四个DP各12360成功更新，末段后端各48720，中段配套后端各24360。30/60点均固定末轮，后端分别绑定对应DP；监督ledger核验末段每合法tick与DP纠偏原生入口槽至少60个不同成功更新步。阶段/家族误差、采样权重与模型SHA见证据。', '',
    '每次reset在首动作前核验实际插入件/孔XY及有符号100mm位移；逐例可观测初态与CUDA采样器消费通过。全程自主、无专家权限。原64终测保持封存；独立16在四套末段完整模型冻结后打开，之后未按其结果训练或选模。同seed不承诺隐藏物理状态及逐位视觉计算一致。', '',
    '主表保留整批分母；配对区间按family整对bootstrap 200000次，seed20261122，未经多重比较校正。全零配对的[0,0]为重采样退化，不能证明等效。所有首次技术中断与重试均保留，只重试未完成机会，物理失败不替换。详见逐初态、全部尝试与阶段全体分母表。', '',
    f"实际登记{stamp(runtime['start_utc'])}至终核{stamp(runtime['utc'])}，墙钟{runtime['wall_seconds']/3600:.2f}小时；可认证owner区间并集{runtime['certified_union_seconds']/3600:.2f}小时，含准备、等待和失败，非GPU小时。按可用容量共享，未操作外部测试进程。原机运行根`outputs/visuotactile/run/tight_full_curve_20261127_01`。执行优化、恢复一致性及弃用更新逐项留证。", '',
    '发布仅写大仓mao455/bcrl_visualTactile既有feature/tight-state-foundation；第二个后来新建仓库持续冻结，本轮零写入。本轮无新增退役路径，历史30项删除不重复计；本轮未删数据/权重，不重复计旧清理释放。', '',
    '[证据索引](evidence/tight_full_curve_20261127_01/README.md) · [逐初态配对表](evidence/tight_full_curve_20261127_01/audit/per_initial_state.csv) · [主结果与区间](evidence/tight_full_curve_20261127_01/audit/primary_summary.json) · [完整方法与阶段分子](evidence/tight_full_curve_20261127_01/METHODS.md)']
report.write_text('\n'.join(lines) + '\n')
print('Short report and full methods separated, with identical actual results')
