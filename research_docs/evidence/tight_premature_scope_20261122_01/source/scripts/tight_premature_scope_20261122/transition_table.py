"""Read-only native-stage transitions from the completed frozen 24-state test.

No simulator, policy inference, optimization, collection, or trajectory decoding.
The categories describe recorded task outcomes; they do not identify causes.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


PROJECT = Path('/home/CNS2025331827/docs/bc+rl_RSS')
SOURCE = PROJECT / 'research_docs/evidence/tight_entry_final24_20261122_01'
OUTPUT = PROJECT / 'research_docs/evidence/tight_premature_scope_20261122_01'
CATEGORIES = ('no_subgoal', 'subgoal_without_completion', 'complete_success')
LABELS = ('未达首子目标', '达首子目标但未完成', '完整成功')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def category(result):
    assert result['legal'] and result['software_complete']
    reached = int(result['native_subgoals']) >= 1
    if result['success']:
        assert reached, 'A complete native success must include its first subgoal'
        return 'complete_success'
    return 'subgoal_without_completion' if reached else 'no_subgoal'


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def analyze():
    primary = read(SOURCE / 'primary_summary.json')
    freeze = read(SOURCE / 'registration.json')
    assert primary['complete'] and primary['episodes'] == 96
    assert primary['software_incomplete'] == 0 and primary['all_legal']
    assert freeze['reserved8'] == list(range(28851000, 28851008))
    assert freeze['fresh16'] == list(range(28852000, 28852016))
    assert len(freeze['seeds']) == len(set(freeze['seeds'])) == 24
    rollouts_path = SOURCE / 'rollouts.csv'
    rollouts = list(csv.DictReader(rollouts_path.open()))
    assert len(rollouts) == 96
    lookup = {(row['role'], int(row['seed'])): row for row in rollouts}
    assert len(lookup) == 96
    hashes = {str(path): sha(path) for path in (
        SOURCE / 'primary_summary.json', SOURCE / 'registration.json', rollouts_path)}
    by_seed = []
    for group in 'AB':
        for seed in freeze['seeds']:
            results, source_rows = {}, {}
            for version, suffix in (('original', '_130_50'), ('entry', '_entry')):
                role = group + suffix
                row = lookup[role, seed]
                path = Path(row['result_path'])
                result = read(path)
                assert result['seed'] == seed
                assert result['complete_student_policy'] == freeze['pairs'][role]
                assert str(bool(result['success'])) == row['success']
                assert int(result['native_subgoals']) == int(row['native_subgoals'])
                assert result['steps'] == int(row['steps'])
                assert row['legal'] == row['software_complete'] == 'True'
                hashes[str(path)] = sha(path)
                results[version], source_rows[version] = result, row
            assert source_rows['original']['cohort'] == source_rows['entry']['cohort']
            item = dict(group=group, cohort=source_rows['original']['cohort'], seed=seed)
            for version in ('original', 'entry'):
                result, row = results[version], source_rows[version]
                item.update({version + '_category': category(result),
                    version + '_success': bool(result['success']),
                    version + '_native_subgoals': int(result['native_subgoals']),
                    version + '_steps': result['steps'],
                    version + '_legal': bool(result['legal']),
                    version + '_software_complete': bool(result['software_complete']),
                    version + '_result_path': row['result_path'],
                    version + '_result_sha256': hashes[row['result_path']]})
            by_seed.append(item)
    assert len(by_seed) == 48
    cells = []
    tables = {}
    for group in 'AB':
        group_rows = [row for row in by_seed if row['group'] == group]
        tables[group] = []
        for original_category in CATEGORIES:
            table_row = []
            for entry_category in CATEGORIES:
                seeds = [row['seed'] for row in group_rows
                    if row['original_category'] == original_category
                    and row['entry_category'] == entry_category]
                table_row.append(len(seeds))
                cells.append(dict(group=group, original_category=original_category,
                    entry_category=entry_category, count=len(seeds), seeds=';'.join(map(str, seeds))))
            tables[group].append(table_row)
        assert sum(map(sum, tables[group])) == 24
        for version, suffix in (('original', '_130_50'), ('entry', '_entry')):
            funnel = next(row for row in primary['funnels']
                if row['role'] == group + suffix and row['cohort'] == 'all24')
            assert sum(row[version + '_category'] != 'no_subgoal' for row in group_rows) == funnel['first_subgoal']
            assert sum(row[version + '_success'] for row in group_rows) == funnel['success_after_subgoal']
    original_successes = [row for row in by_seed
        if row['group'] == 'A' and row['original_category'] == 'complete_success']
    entry_post_failures = [row for row in by_seed
        if row['group'] == 'A' and row['entry_category'] == 'subgoal_without_completion']
    assert len(original_successes) == len(entry_post_failures) == 5
    assert all(row['entry_category'] == 'no_subgoal' for row in original_successes)
    assert all(row['original_category'] == 'no_subgoal' for row in entry_post_failures)
    assert {row['seed'] for row in original_successes}.isdisjoint(
        row['seed'] for row in entry_post_failures)
    summary = dict(version='frozen24_native_stage_transitions_v1',
        source=str(SOURCE), different_initial_states=24, paired_rows=48,
        categories=list(CATEGORIES), category_labels=list(LABELS),
        rows='130-50 native outcome', columns='entry native outcome', tables=tables,
        A_original_success_destinations={key:sum(row['entry_category'] == key
            for row in original_successes) for key in CATEGORIES},
        A_entry_post_subgoal_failure_origins={key:sum(row['original_category'] == key
            for row in entry_post_failures) for key in CATEGORIES},
        A_original_success_seeds=[row['seed'] for row in original_successes],
        A_entry_post_subgoal_failure_seeds=[row['seed'] for row in entry_post_failures],
        selected_sets_overlap=0, software_incomplete=0,
        optimizer_updates=0, new_dagger_trajectories=0, new_physics_steps=0,
        policy_inferences=0,
        interpretation='Recorded native stages only. Does not locate the earliest physical error or attribute premature insertion/release as a cause. A original successes lost before native subgoal confirmation and A entry failures after confirmation are disjoint sets.')
    return summary, by_seed, cells, original_successes, entry_post_failures, hashes


def report_text(summary, original_successes, entry_post_failures):
    lines = ['# 冻结24初态的原生阶段转移', '',
        '只读取已完成终测的原生 `success`、`native_subgoals`、`legal` 与 `software_complete`，并对照96份原始结果和冻结模型身份。以下各表均为24个初态；行是130-50，列是入口修复版。三个类别互斥且完备：首子目标未达；首子目标已达但最终未完成；原生完整成功。', '']
    for group in 'AB':
        lines += [f'**{group}：原版 → 入口版**', '',
            '| 原版类别 \\ 入口版类别 | 未达首子目标 | 达首子目标但未完成 | 完整成功 | 行合计 |',
            '| --- | ---: | ---: | ---: | ---: |']
        table = summary['tables'][group]
        for label, row in zip(LABELS, table):
            lines.append(f'| {label} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |')
        totals = [sum(row[index] for row in table) for index in range(3)]
        lines += [f'| 列合计 | {totals[0]} | {totals[1]} | {totals[2]} | 24 |', '']
    lines += ['A原版5例完整成功，在入口版中**5/5均未达原生首子目标**，0例进入“已达首子目标但未完成”，0例保留完整成功。', '',
        '| 初态seed | 原版 | 入口版 | 原版/入口版原生首子目标数 |',
        '| --- | --- | --- | ---: |']
    for row in original_successes:
        lines.append(f"| {row['seed']} | 完整成功 | 未达首子目标 | {row['original_native_subgoals']}/{row['entry_native_subgoals']} |")
    lines += ['', '入口版A的5例“已达首子目标但未完成”，其原版**5/5均未达首子目标**；与上述丢失的5例原成功初态完全不重叠。', '',
        '| 初态seed | 原版 | 入口版 | 原版/入口版原生首子目标数 |',
        '| --- | --- | --- | ---: |']
    for row in entry_post_failures:
        lines.append(f"| {row['seed']} | 未达首子目标 | 达首子目标但未完成 | {row['original_native_subgoals']}/{row['entry_native_subgoals']} |")
    lines += ['', '因此，不能用入口版A这5例“过首子目标后未完成”解释原版5例成功的丢失。原成功的回退发生在**原生首子目标确认之前**这一分类范围内。该分类没有解析物理轨迹，不能判断是抓持、对准、提前下插/松手、阶段确认时序还是其他原因；尤其不能据此把这5例直接归因于某个开发案例中的提前推进。', '',
        'B的原版唯一成功在入口版中同样未达首子目标；入口版4例新完整成功中，1例原版未达首子目标、3例原版已达首子目标但未完成。这里只描述配对转移，不据24对推断因果改善。', '',
        '本分析软件未完成0，训练更新0、新DAgger0、新物理步0、策略推理0。结果仅覆盖本次固定权重、固定推理噪声方案下已完成的24个初态。', '',
        '[48对逐初态转移](transitions_by_seed.csv) · [18格计数与seed](transition_counts.csv) · [A原成功去向](A_original_success_transitions.csv) · [A入口版过首子目标后失败来源](A_entry_post_subgoal_failure_origins.csv) · [JSON汇总](summary.json) · [输入SHA](input_hashes.json)', '',
        '复核：在项目根目录执行 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/CNS2025331827/miniconda3/envs/isaaclab/bin/python -m scripts.tight_premature_scope_20261122.transition_table --verify-only`。该命令重读原始结果核对已写产物，不运行策略或仿真。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    summary, by_seed, cells, original_successes, entry_post_failures, hashes = analyze()
    json_text = lambda value: json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    files = {'summary.json':json_text(summary), 'input_hashes.json':json_text(hashes),
        'transitions_by_seed.csv':csv_text(by_seed), 'transition_counts.csv':csv_text(cells),
        'A_original_success_transitions.csv':csv_text(original_successes),
        'A_entry_post_subgoal_failure_origins.csv':csv_text(entry_post_failures),
        'REPORT.md':report_text(summary, original_successes, entry_post_failures)}
    if args.verify_only:
        for name, content in files.items():
            assert (OUTPUT / name).read_text() == content, name
        print('Verified all native-stage transition artifacts against the 96 frozen raw results.')
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        for name, content in files.items():
            (OUTPUT / name).write_text(content)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
