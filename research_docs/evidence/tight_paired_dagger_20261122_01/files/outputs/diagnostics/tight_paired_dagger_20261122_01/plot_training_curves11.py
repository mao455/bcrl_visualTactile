"""Render existing completed curve CSVs on CPU; no experiment mutation."""
from pathlib import Path
import csv
import hashlib
import json
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
OUT = HERE / 'training_curves11'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    inputs = {}
    panels = []
    for role in 'AB':
        for component, relative in (
            ('DP', f'{role}/training/family_phase_curves.csv'),
            ('Transformer', f'{role}/backend/{role}/training/family_phase_curves.csv'),
        ):
            path = ROOT / 'learning' / relative
            if path.stat().st_size > 8 * 2**20:
                raise ValueError('Unexpected large curve CSV')
            inputs[str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            rows = [row for row in rows if 1 <= int(row['epoch']) <= 60]
            panels.append((role, component, rows))

    for kind in ('overall', 'validation_phase', 'validation_family'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for axis, (role, component, rows) in zip(axes.flat, panels):
            if kind == 'overall':
                groups = [(split, [r for r in rows if r['split'] == split
                                   and r['family'] == '__all__' and r['phase'] == '__all__'])
                          for split in ('train_evaluation', 'validation')]
            elif kind == 'validation_phase':
                selected = [r for r in rows if r['split'] == 'validation'
                            and r['family'] == '__all__' and r['phase'] != '__all__']
                groups = [(phase, [r for r in selected if r['phase'] == phase])
                          for phase in sorted({r['phase'] for r in selected})]
            else:
                selected = [r for r in rows if r['split'] == 'validation'
                            and r['family'] != '__all__' and r['phase'] == '__all__']
                groups = [(family, [r for r in selected if r['family'] == family])
                          for family in sorted({r['family'] for r in selected})]
            for label, points in groups:
                points.sort(key=lambda r: int(r['epoch']))
                epochs = [int(r['epoch']) for r in points]
                if epochs != list(range(1, 61)):
                    raise ValueError(f'Incomplete or duplicate curve: {role}/{component}/{label}')
                values = [float(r['normalized_equal_mse']) for r in points]
                if values[0] <= 0:
                    raise ValueError('Positive normalization reference required')
                axis.plot(epochs, [100 * value / values[0] for value in values],
                          label=label.replace('layout10_validation_', ''), linewidth=1.4)
            axis.set_title(f'{role} {component}')
            axis.set_xlabel('Epoch (deployment fixed at 60)')
            axis.set_ylabel('Own metric / own epoch 1 (%)')
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
        fig.suptitle('Existing fixed histories; train panel 2 trajectories, validation 6 families\n'
                     'Each curve has its own reference; A/B nominal target spaces differ', fontsize=11)
        fig.savefig(OUT / (kind + '.png'), dpi=160)
        plt.close(fig)
    outputs = {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(OUT.glob('*.png'))}
    manifest = dict(utc=time.time(), script_sha256=digest(Path(__file__)), inputs=inputs, outputs=outputs,
                    policy_updates=0, gpu_or_kit_started=False, outcome_reads=0,
                    normalization='Each curve normalized independently to its own epoch1 equal-weight arm/hand MSE',
                    limits='Fixed train2 panel is not the full pool. Repeated validation6 is not independent autonomy.')
    with (OUT / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(dict(output=str(OUT), plots=len(outputs))))


if __name__ == '__main__':
    main()
