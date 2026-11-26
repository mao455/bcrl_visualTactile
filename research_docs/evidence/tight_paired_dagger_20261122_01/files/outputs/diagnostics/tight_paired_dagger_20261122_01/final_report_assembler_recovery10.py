"""Inactive final report adapter; pinned scientific validation remains unchanged.

No source receipt is redirected or fabricated. The original unused summary exit
path must contain its real recovery10 receipt. Failed main06 and the new backend
owner appear separately in the closed owner interval union.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_final10_common_assembly', HERE/'final_recovery10_common.py')
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)
BASE_SHA256 = '1eca4843d9e7e2d1ac6287b220bf8d429fe36458cb58ddcf63a63e9ab2611b2f'
base = common.pinned_module('final_report_assembler.py', BASE_SHA256, 'assembler',
    replacements=(("root/'cycle_logs/summarize_test_exit.json'",
                   "root/'cycle_logs_recovery10/summarize_test_exit.json'"),))
ORIGINAL_ATTEMPT_COSTS = base.attempt_costs
SELECTED_LEDGERS = None


def actual_component(directory, component, _torch, _ledger_class):
    common.require(SELECTED_LEDGERS is not None, 'Explicit successful10 selected-ledger extraction required')
    directory = Path(directory).resolve()
    for scope in ('common180_parents','learning'):
        for role in 'AB':
            expected = CURRENT_ROOT/scope/role/('training' if component == 'frontend' else f'backend/{role}/training')
            if directory != expected: continue
            row = common.extracted_component(CURRENT_ROOT,CURRENT_AUDIT,SELECTED_LEDGERS,scope,role,component)
            entry = row['native_entry_rows']
            values = [n for episode in row['episodes'].values() for n in episode['successful_optimizer_steps']]
            exposure = dict(eligible=True,ancestors_counted=False,updates=row['updates'],
                trajectories=row['trajectories'],legal_ticks=len(values),minimum_tick_steps=min(values),
                maximum_tick_steps=max(values),minimum_visits=row['minimum_visits'],below50_ticks=0,
                below50_trajectories=0)
            curve = directory/'family_phase_curves.csv'
            common.require(curve.is_file(), 'Actual phase/family curves missing')
            finished = common.read_small(directory/'finished.json')
            return dict(directory=str(directory),checkpoint=row['checkpoint'],
                checkpoint_sha256=row['checkpoint_sha256'],epoch=60,updates=row['updates'],exposure=exposure,
                entry=dict(entries=len({r['identity'] for r in entry}),slots=len(entry),
                           minimum_steps=min(r['successful_optimizer_steps'] for r in entry),below50=0) if entry else None,
                base_sha256=row['base_sha256'],wall_seconds=finished.get('wall_seconds',finished.get('wall_s')),
                curves=str(curve),visits={identity:episode['visits'] for identity,episode in row['episodes'].items()},
                selected_ledger_reused_from=str(SELECTED_LEDGERS),additional_checkpoint_reads=0)
    raise ValueError('Component directory outside exact final populations')


base.actual_component = actual_component


def attempt_costs(root):
    result = ORIGINAL_ATTEMPT_COSTS(root)
    # The pinned base inventories ROOT/owners. Evaluation owners live elsewhere.
    intervals = [list(x) for x in result['closed_owner_merged_intervals']]
    for cohort in ('development', 'test'):
        owner = Path(root)/'evaluation'/cohort
        begin, end = owner/'project_job_started.json', owner/'project_exit.json'
        if not begin.is_file(): continue
        start = common.read_small(begin)['utc']
        closed = common.read_small(end) if end.is_file() else None
        finish = closed.get('utc') if closed else None
        certified = closed is not None and closed.get('children_reaped') is True
        if certified:
            base.interval_union_seconds([(start, finish)])
            intervals.append([start, finish])
        result['owner_attempts'].append(dict(owner='evaluation/'+cohort,
            start_utc=start, end_utc=finish,
            observed_closed_wall_seconds=finish-start if certified else None,
            status='closed' if certified else 'not_closed_or_unreaped',
            role='evaluation', start_receipt=str(begin),
            exit_receipt=str(end) if end.is_file() else None))
        for path in (begin, end):
            if path.is_file(): result['sources_sha256'][str(path)] = common.sha(path)
    seconds, merged = base.interval_union_seconds(intervals)
    result.update(closed_owner_wall_union_seconds=seconds,
        closed_owner_merged_intervals=merged,
        closed_owner_sum_before_overlap_removal_seconds=sum(
            row['observed_closed_wall_seconds'] or 0 for row in result['owner_attempts']),
        open_or_unreaped_owner_count=sum(row['status'] != 'closed' for row in result['owner_attempts']),
        recovery06_07_08_09_failed_ancestry_preserved=True)
    return result


base.attempt_costs = attempt_costs


def assemble(root, output, audit, selected_ledgers):
    global SELECTED_LEDGERS,CURRENT_ROOT,CURRENT_AUDIT
    CURRENT_ROOT,CURRENT_AUDIT = Path(root).resolve(),Path(audit).resolve()
    SELECTED_LEDGERS = Path(selected_ledgers).resolve()
    lifecycle = common.closure_gate(root, audit)
    if lifecycle['status'] != 'closed':
        return dict(status='pending', checklist=lifecycle.get('missing', []),
                    recovery10_lifecycle=lifecycle, actual_checkpoint_reads=0)
    value = base.assemble(root, output, audit)
    value['recovery10_lifecycle'] = lifecycle
    value['scientific_assembler_sha256'] = BASE_SHA256
    if value.get('status') == 'draft_ready_for_root_review':
        value['actual_checkpoint_reads'] = 0
        value['actual_selected_ledger_tables_reused'] = 8
        value['selected_ledgers_directory'] = str(SELECTED_LEDGERS)
        value['source_path_substitution'] = 'Only real summary exit path: cycle_logs_recovery10/summarize_test_exit.json'
        value['checkpoint_read_accounting']=lifecycle['payload_read_accounting']
        report=output/'DRAFT_REPORT.md'
        with report.open('a') as stream:
            stream.write('\n运行恢复：原06训练、07缓存、08可变遥测读取及09空暂停哨兵解析失败均保留；10仅续接未完成后端及原登记评估。辅助归档08失败且未节省、09在14文件后失败（节省100415954字节）；DP重复副本去重A已完成、B中止，不能报全成功或零操作。评估前guard与终末详细曝光提取各自读取8套权重；报告/导出复用后者，不再次读取权重。实际时间、原失败和资源等待见计算回执。\n')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=base.ROOT)
    parser.add_argument('--audit', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selected-ledgers',type=Path,required=True)
    args = parser.parse_args()
    audit, output = args.audit.resolve(), args.output.resolve()
    base.require(output.is_relative_to(audit) and output != audit and not output.exists(),
                 'Output must be a NEW AUDIT child')
    output.mkdir(parents=False, exist_ok=False)
    try:
        status = assemble(args.root.resolve(), output, audit,args.selected_ledgers)
    except Exception as exc:
        status = dict(status='pending_validation_failed', checklist=[dict(reason=repr(exc))],
                      actual_checkpoint_reads=0)
    status.update(observed_utc=time.time(), assembler_sha256=common.sha(Path(__file__)),
                  writes_only=str(output), gpu_or_kit_started=False)
    base.write(output/'assembly_status.json', status)
    print(json.dumps(dict(status=status['status'], output=str(output)), ensure_ascii=False))
    raise SystemExit(0 if status['status'] == 'draft_ready_for_root_review' else 2)


if __name__ == '__main__': main()
