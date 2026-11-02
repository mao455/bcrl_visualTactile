"""Final budget-sensitivity evidence and local video index, without raw uploads."""
import argparse
import json
from pathlib import Path

import numpy as np

from .compact import sha256
from .native_v2_repair_export import public_value
from .native_v2_repair_report import summarize_rows
from .training_resources import atomic_json
from .viewing_report import check_video, resource_intervals


def report(root, output):
    session = json.loads((root/'session.json').read_text())
    parent = Path(session['parent_root'])
    status = json.loads((root/'status.json').read_text())
    owner = json.loads((root/'owner_exit.json').read_text())
    if status['state'] == 'running' or not owner['owned_children_exited']:
        raise ValueError('Supplement owner must exit before final report')
    output.mkdir(parents=True, exist_ok=True)
    learning = json.loads((root/'training/result.json').read_text())
    result = dict(status=status, session=session, training=learning, primary_starts=0,
                  monitor=json.loads((root/'monitor_exit.json').read_text()),
                  over50_intervals=resource_intervals(root/'resources/resources.jsonl'), videos=[])
    outcome = root/'saved_initials/result.json'
    if outcome.exists():
        observed = json.loads(outcome.read_text())
        result['trials'] = observed['trials']
        result['software_complete'] = observed['software_complete']
        result['primary_starts'] = len(observed['trials'])
        result['cohorts'] = {name: summarize_rows([row for row in observed['trials'] if row['cohort'] == name])
                             for name in ('train', 'validation')}
        result['initial_plan_checks'] = []
        for row in observed['trials']:
            plans = np.load(root/'saved_initials'/row['trial_name']/'plans.npz')
            difference = plans['student'][0, :12].clip(-1, 1)-plans['expert_history_shadow'][0, :12]
            result['initial_plan_checks'].append(dict(identity=row['identity'], cohort=row['cohort'],
                clipped_plan_max_error=float(np.abs(difference).max()), first_divergence=row['first_divergence']))
        for path in sorted((root/'saved_initials').glob('*/student.mp4')):
            steps = json.loads((path.parent/'result.json').read_text())['steps']
            result['videos'].append(dict(check_video(path, steps, set()), path=str(path.relative_to(root))))
    failures = root/'preaction_failure_1'
    if failures.exists():
        result['preaction_failure'] = dict(status=json.loads((failures/'status.json').read_text()),
            lifecycle=json.loads((failures/'saved_initials/kit_lifecycle.json').read_text()),
            log_sha256=sha256(failures/'saved_initials.log'), simulated_actions=0,
            recovery=json.loads((root/'recovery.json').read_text()))
    original = json.loads((parent/'report/summary.json').read_text())
    if (root/'initial_command_fit.json').exists():
        result['initial_command_fit'] = json.loads((root/'initial_command_fit.json').read_text())
    result['total_primary_starts'] = sum(row['trials'] for row in original['stages'].values())+result['primary_starts']
    result['total_optimizer_updates'] = {kind: learning['models'][kind]['history'][-1]['updates'] for kind in ('state', 'actor')}
    atomic_json(output/'fit_extension.json', public_value(result))
    for name in ('fresh_bytewise', 'initial_box_geometry', 'saved_initial_box_geometry', 'foundation_provenance'):
        atomic_json(output/(name+'.json'), public_value(json.loads((parent/'report'/(name+'.json')).read_text())))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for kind in ('state', 'actor'):
        history = original['bc_learning']['models'][kind]['history']+learning['models'][kind]['history']
        for axis, part in zip(axes, ('arm', 'hand')):
            for split, style in (('train', '-'), ('validation', '--')):
                axis.plot([row['updates'] for row in history], [row[split][part+'_rmse'] for row in history],
                          style, label=kind+' '+split)
            axis.set_yscale('log'); axis.set_xlabel('Total optimizer updates')
            axis.set_ylabel(part+' native command RMSE'); axis.legend()
    figure.tight_layout(); figure.savefig(output/'bc_budget_sensitivity.png', dpi=150); plt.close(figure)
    original_videos = json.loads((parent/'report/video_verification.json').read_text())
    index = ['# 本轮真实诊断视频', '',
             '完整原生终止，1×模拟时间；不是额外试验或专家接管。终止后的reset图像不加入视频。',
             '首条fresh三栏的fixed_high缺乏有效内容，侧视/腕视可用；State/Actor诊断未使用图像输入。', '',
             '| 轨迹 | 帧数 | 原生完整成功 | 文件 |', '|---|---:|---|---|']
    for path in sorted(parent.glob('*/seed_*/rollout.mp4'))+sorted(parent.glob('bc_saved/*/student.mp4')):
        stored = json.loads((path.parent/'result.json').read_text())
        row = stored['trials'][0] if 'trials' in stored else stored
        relative = path.relative_to(parent)
        index.append(f"| {relative.parent} | {row['steps']} | {row['success']} | [{path.name}]({relative}) |")
    for path in sorted((root/'saved_initials').glob('*/student.mp4')):
        row = json.loads((path.parent/'result.json').read_text())
        relative = Path('..')/root.name/path.relative_to(root)
        index.append(f"| BC4000/{path.parent.name} | {row['steps']} | {row['success']} | [{path.name}]({relative}) |")
    index += ['', f"原52条视频{len(original_videos)}份，补充视频{len(result['videos'])}份；全部逐帧解码。",
              '预登记身份的首条视频，不根据结果选择；视频仅本地保存。']
    (parent/'README.md').write_text('\n'.join(index)+'\n')
    print(json.dumps(dict(total_starts=result['total_primary_starts'], status=status,
                         cohorts=result.get('cohorts'), videos=len(result['videos'])), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.root, arguments.output)
