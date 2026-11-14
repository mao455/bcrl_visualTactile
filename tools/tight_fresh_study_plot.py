"""Plot comparable unweighted metrics from the audited paired study report."""
import argparse
import json
from pathlib import Path


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads(report.read_text())
    if not data['complete'] or data['test_used']:
        raise ValueError('A complete non-test comparison is required')
    figure, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    candidate = 'previous' if data.get('version') == 'matched_anchor_study_report_v1' else 'weighted'
    cohort = 'all_corrections' if candidate == 'previous' else 'all53'
    colors = {'control': '#3972ba', candidate: '#c46c32'}
    for row, role in enumerate(('A', 'B')):
        result = data['roles'][role]
        for col, split in enumerate(('train', 'validation')):
            axis = axes[row, col]
            for arm in ('control', candidate):
                history = result['arms'][arm]['history']
                axis.plot([r['epoch'] for r in history],
                          [r[split]['all']['hand_rmse_rad'] for r in history],
                          marker='o', color=colors[arm], label=arm)
            axis.set(title=f'{role}: fixed {split} command error', xlabel='Additional rear epoch',
                     ylabel='Hand RMSE (rad)')
            axis.grid(alpha=.2)
            axis.legend()
        axis = axes[row, 2]
        arms = ('source', 'control', candidate)
        population = result['onset_fitting']['source'][cohort]['episodes']
        axis.bar(arms, [result['onset_fitting'][arm][cohort]['bins']['first12']['hand_rmse_rad']
                        for arm in arms], color=['#888888', colors['control'], colors[candidate]])
        axis.set(title=f'{role}: first 12 legal recovery ticks', ylabel='Hand RMSE (rad)')
        axis.text(.5, -.19, f'{population} learned corrections; same recorded histories', transform=axis.transAxes,
                  ha='center', fontsize=8)
    figure.suptitle('Matched additional training; unweighted errors, not autonomous success')
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.report, args.output)
