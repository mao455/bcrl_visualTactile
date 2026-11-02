"""Summarize only committed native outcomes and preserve attribution limits."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from .native_v2_forensics import configuration_diff
from .training_resources import atomic_json


def wilson(successes, count):
    if not count:
        return None
    quantile = 1.959963984540054; proportion = successes/count; denominator = 1+quantile*quantile/count
    center = (proportion+quantile*quantile/(2*count))/denominator
    radius = quantile*((proportion*(1-proportion)+quantile*quantile/(4*count))/count)**.5/denominator
    return [center-radius, center+radius]


def summarize_rows(rows):
    successes = sum(bool(row['success']) for row in rows)
    return dict(trials=len(rows), successes=successes, success_rate=successes/len(rows) if rows else None,
        wilson95=wilson(successes, len(rows)), native_lift=sum(bool(row.get('native_lift')) for row in rows),
        native_subgoal=sum(row.get('native_subgoals', row.get('subgoals', 0)) > 0 for row in rows),
        native_alignment=sum(bool(row.get('native_alignment')) for row in rows),
        outcomes=dict(Counter('actions_exhausted_censored' if row.get('recorded_actions_exhausted') else
                      '+'.join(name for name, value in row.get('reasons', {}).items() if value) for row in rows)),
        steps=[row['steps'] for row in rows])


def report(root):
    output = root/'report'; output.mkdir(exist_ok=True)
    summary = dict(status=json.loads((root/'status.json').read_text()), stages={},
                   gates={path.stem: json.loads(path.read_text()) for path in (root/'gates').glob('*.json')})
    names = ('expert1', 'expert4', 'fresh_replay', 'historical_world', 'old_state', 'bc_state', 'bc_actor', 'bc_saved')
    for name in names:
        path = root/name/'result.json'
        if path.exists():
            result = json.loads(path.read_text())
            summary['stages'][name] = summarize_rows(result['trials'])
            if name in ('fresh_replay', 'historical_world'):
                summary['stages'][name]['comparisons'] = [row['comparison'] for row in result['trials']]
            if name == 'bc_saved':
                summary['stages'][name]['cohorts'] = {cohort: summarize_rows([row for row in result['trials'] if row['cohort'] == cohort])
                                                     for cohort in ('train', 'validation')}
    if (root/'bc_compare/result.json').exists():
        summary['bc_learning'] = json.loads((root/'bc_compare/result.json').read_text())
    if (root/'monitor_exit.json').exists() and summary['status']['state'] != 'running':
        summary['resources'] = json.loads((root/'monitor_exit.json').read_text())
    elif (root/'resources/resources.json').exists():
        summary['resources_partial'] = json.loads((root/'resources/resources.json').read_text())
    summary['fresh_initial_comparisons'] = {}
    for name in ('old_state', 'bc_state', 'bc_actor'):
        comparisons = []
        for first in sorted((root/'expert1').glob('seed_*/initial_observable.json')):
            second = root/name/first.parent.name/'initial_observable.json'
            if not second.exists():
                continue
            before = json.loads(first.read_text()); after = json.loads(second.read_text())
            errors = {key: float(np.abs(np.asarray(value)-np.asarray(after[key])).max())
                      for key, value in before.items() if np.asarray(value).dtype != bool}
            comparisons.append(dict(seed=first.parent.name, errors=errors, exact_observable=all(value == 0 for value in errors.values()),
                                    hidden_PhysX_state_equal=None))
        summary['fresh_initial_comparisons'][name] = comparisons
    single = root/'expert1/environment.json'; parallel = root/'expert4/environment.json'
    if single.exists() and parallel.exists():
        summary['single_four_config_diff'] = configuration_diff(json.loads(single.read_text())['config'],
                                                                json.loads(parallel.read_text())['config'])
        first = next(iter(sorted((root/'expert1').glob('seed_*/initial_export.pt'))), None)
        second = next(iter(sorted((root/'expert4').glob('seed_*/initial_export.pt'))), None)
        if first is not None and second is not None:
            one = torch.load(first, map_location='cpu', weights_only=False)['physics']
            four = torch.load(second, map_location='cpu', weights_only=False)['physics']
            properties = {}
            for body in one:
                for key, value in one[body].items():
                    if key != 'root_state_w':
                        properties[body+'/'+key] = dict(single_shape=list(value.shape), four_shape=list(four[body][key].shape),
                            max_error=float((four[body][key]-value).abs().max())
                            if value.shape[1:] == four[body][key].shape[1:] else None)
            summary['single_four_actual_physics_properties'] = properties
    atomic_json(output/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if summary['stages']:
        figure, axis = plt.subplots(figsize=(10, 5))
        labels = list(summary['stages']); scores = [summary['stages'][name]['success_rate'] for name in labels]
        axis.bar(labels, scores)
        for index, name in enumerate(labels):
            row = summary['stages'][name]; axis.text(index, scores[index]+.02, f"{row['successes']}/{row['trials']}", ha='center')
        axis.set_ylim(0, 1.12); axis.set_ylabel('Native complete task success')
        axis.tick_params(axis='x', rotation=25); figure.tight_layout(); figure.savefig(output/'native_outcomes.png', dpi=150); plt.close(figure)
    if 'bc_learning' in summary:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for kind in ('state', 'actor'):
            rows = [json.loads(line) for line in (root/'bc_compare'/kind/'train.jsonl').read_text().splitlines()]
            axes[0].plot([row['updates'] for row in rows], [row['loss'] for row in rows], label=kind)
            for split, style in [('train', '-'), ('validation', '--')]:
                history = summary['bc_learning']['models'][kind]['history']
                axes[1].plot([row['updates'] for row in history], [row[split]['selection_mse'] for row in history], style, label=kind+' '+split)
        axes[0].set_yscale('log'); axes[1].set_yscale('log')
        axes[0].set_ylabel('Loss B (training)'); axes[1].set_ylabel('Native command MSE (teacher forcing)')
        for axis in axes:
            axis.set_xlabel('Optimizer updates'); axis.legend()
        figure.tight_layout(); figure.savefig(output/'bc_learning.png', dpi=150); plt.close(figure)
    videos = sorted(root.glob('*/seed_*/rollout.mp4')) + sorted(root.glob('bc_saved/*/student.mp4'))
    atomic_json(output/'videos.json', [dict(path=str(path.resolve()), bytes=path.stat().st_size) for path in videos])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args(); report(arguments.root)
