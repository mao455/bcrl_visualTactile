"""Plot measured adjacent-stage learning curves from bound reports."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--updated', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    initial, updated, baseline = map(read, (args.initial, args.updated, args.baseline))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for role, color in [('A', '#1166aa'), ('B', '#d46a18')]:
        old, new = initial['roles'][role], updated['roles'][role]
        for column, part in enumerate(('arm', 'hand')):
            key = part + '_rmse_rad'
            axes[0, column].plot([0] + [r['epoch'] for r in new['history']],
                [new['initial_metrics']['validation'][key]] + [r['validation'][key] for r in new['history']],
                color=color, marker='.', label=role + (' DP q[t+1]' if role == 'A' else ' DP expert target'))
            axes[0, column].plot([0] + [r['epoch'] for r in new['history']],
                [new['initial_metrics']['train'][key]] + [r['train'][key] for r in new['history']],
                color=color, linestyle=':', label=role + ' DP train diagnostic2')
            for label, report, style in [('prior stage', old, '--'), ('current stage', new, '-')]:
                history = report['backend_history']
                axes[1, column].plot([r['epoch'] for r in history],
                    [r['validation']['all'][key] for r in history], color=color, linestyle=style,
                    label=role + ' rear ' + label)
            axes[1, column].plot([r['epoch'] for r in new['backend_history']],
                [r['train']['all'][key] for r in new['backend_history']], color=color, linestyle=':',
                label=role + ' rear train diagnostic')
            if role == 'A':
                row = next(r for r in baseline['roles'][role]['metrics']
                    if r['split'] == 'validation' and r['phase'] == 'all' and r['group'] == part)
                axes[1, column].axhline(row['previous_target_rmse_rad'], color='#555555', linestyle=':',
                    label='Hold actual previous target')
    for column, part in enumerate(('Arm', 'Hand')):
        axes[0, column].set(title=part + ' DP validation', xlabel='Recovery DP epochs', ylabel='RMSE (rad)')
        axes[1, column].set(title=part + ' final-command validation', xlabel='Backend epoch within shown stage', ylabel='RMSE (rad)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Fixed six-family validation on recorded histories; not autonomous performance\nDP labels differ: compare within role. Rear targets share the same command definition.', fontsize=10)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
