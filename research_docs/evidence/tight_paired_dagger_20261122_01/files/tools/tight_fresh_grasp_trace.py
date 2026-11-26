"""Plot a repeated autonomous opportunity using recorded physical diagnostics.

Root-origin proximity and quaternion angle are descriptive; native keypoint
criteria and actual subgoals determine task progress. No model forward or Kit.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from tools.tight_fresh_phases import ROOT_POSE_SEMANTICS, case, stable_windows
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(path.read_text())


def trace(evaluation, name):
    plan = read(evaluation / 'plan.json')
    cases = [c for c in plan['cases'] if c['name'] == name]
    if (plan.get('kind') != 'autonomous' or plan.get('test_used') is not False
            or len(cases) != 1 or cases[0]['spec'].get('split') != 'development'
            or not read(evaluation / 'result.json').get('software_complete')):
        raise ValueError('One registered complete autonomous development opportunity required')
    registered = cases[0]
    pair = plan['pairs'][registered['role']]
    directory = evaluation / name
    result = read(directory / 'result.json')
    indexed = [r['result'] for r in read(evaluation / 'results.json') if r['case'] == name]
    if (indexed != [result] or not result.get('legal') or not result.get('software_complete')
            or result.get('dagger') is not None or result.get('censored')
            or result.get('worker_error') or result['residual_ticks'] != result['steps']
            or result['checkpoint_sha256'] != pair['backend_sha256']
            or result['simulation_step_hz'] != 60 or result['plan_hz'] != 5
            or any(result[k] != v for k, v in registered['spec'].items())):
        raise ValueError('Recorded full-student outcome differs from its registration')
    for key in ('frontend', 'backend'):
        if sha256(pair[key]) != pair[key + '_sha256']:
            raise ValueError('Registered selected model changed')
    phase = case(directory)
    with h5py.File(directory / 'numeric.h5') as f:
        t = f['episode/transition']
        d = t['diagnostics_after']
        obj, goal, palm = [d[k][:] for k in ('object_pose_w', 'goal_pose_w', 'palm_body_pose_w')]
        force = t['object_contact_force_w'][:]
        times = t['time_after_s'][:] - t['time_before_s'][0]
        subgoals = d['successes'][:]
        if (len(obj) != result['steps'] or len(obj) < 48 or (subgoals != 0).any()
                or not np.allclose(np.diff(times), 1 / 60, rtol=0, atol=1e-9)):
            raise ValueError('This diagnostic requires one unchanged first goal at native60Hz')
        if 'dagger/control_source' in t and t['dagger/control_source'][:].any():
            raise ValueError('Expert transitions cannot enter an autonomous trace')
        settled = float(np.median(obj[24:48, 2]))
        stable, contact = stable_windows(obj, palm, force, settled)
        arrays = dict(time_s=times, height_mm=1000 * (obj[:, 2] - settled),
            root_distance_mm=1000 * np.linalg.norm(obj[:, :3] - goal[:, :3], axis=-1),
            angle_rad=2 * np.arccos(np.clip(abs((obj[:, 3:] * goal[:, 3:]).sum(-1)), 0, 1)),
            keypoint_distance_mm=1000 * t['native_keypoints_max_dist'][:],
            tolerance_mm=1000 * t['native_success_diagnostic/insertion_tolerance_m'][:],
            min_contact_fingers=contact.sum(-1).min(-1), stable=stable.astype(int))
    if not all(np.isfinite(v).all() and len(v) == result['steps'] for v in arrays.values()):
        raise ValueError('Finite aligned actual trace values required')
    if int(stable.sum()) != phase['grasp_progress']['stable_ticks']:
        raise ValueError('Plotted stable-window count differs from phase report')
    return arrays, dict(evaluation=str(evaluation.resolve()), case=name, spec=registered['spec'],
        role=registered['role'], pair=pair, phase=phase,
        plan_sha256=sha256(evaluation/'plan.json'), results_sha256=sha256(evaluation/'results.json'),
        initial_observable=read(directory/'initial_observable.json'))


def run(before, after, name, output, figure):
    if output.exists() or figure.exists():
        raise FileExistsError('Preserve existing diagnostic evidence')
    rows = {label: trace(path, name) for label, path in [('before', before), ('after', after)]}
    a, b = (rows[k][1] for k in ('before', 'after'))
    if a['spec'] != b['spec'] or a['role'] != b['role']:
        raise ValueError('Repeated registered opportunity differs')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 1, figsize=(10, 11), sharex=True, constrained_layout=True,
                             gridspec_kw={'height_ratios':[1,1,1,1,1,.45]})
    panels = [('height_mm', 'Root height above\nsettled (mm)'),
              ('root_distance_mm', 'Object-to-goal\nroot distance (mm)'),
              ('angle_rad', 'Goal quaternion\nangle (rad)'),
              ('keypoint_distance_mm', 'Native keypoint\ndistance (mm)'),
              ('min_contact_fingers', 'Minimum contact\nfingers / tick')]
    for label, color in [('before', '#3972ba'), ('after', '#c46c32')]:
        values, info = rows[label]
        for ax, (key, title) in zip(axes, panels):
            ax.plot(values['time_s'], values[key], color=color, label=label, linewidth=1.2)
            ax.set_ylabel(title, fontsize=9)
            ax.grid(alpha=.2)
        ticks = values['stable'].astype(bool)
        axes[-1].scatter(values['time_s'][ticks], np.full(ticks.sum(), 0 if label=='before' else 1),
            s=5, color=color)
    axes[3].plot(rows['after'][0]['time_s'], rows['after'][0]['tolerance_mm'], 'k--', linewidth=1,
                 label='Native insertion tolerance')
    axes[-2].set_yticks(range(6))
    axes[-1].set_yticks([0,1],['before','after']); axes[-1].set_ylim(-.5,1.5)
    axes[-1].set_ylabel('Past12 stable\nwindow endpoints',fontsize=9)
    axes[-1].set_xlabel('Simulated seconds; native60Hz control')
    for ax in axes[:-1]: ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(name + f'\nBefore: {before.parent.name}; after: {after.parent.name}'
        '\nSame registered opportunity; no exact hidden-state replay claim', fontsize=11)
    fig.savefig(figure, dpi=160); plt.close(fig)
    observable_equal = a.pop('initial_observable') == b.pop('initial_observable')
    atomic_json(output, dict(complete=True, cases={k:v[1] for k,v in rows.items()},
        recorded_initial_observable_equal=observable_equal, pose_reference_semantics=ROOT_POSE_SEMANTICS,
        interpretation='Selected descriptive repeated opportunity, not an independent test or population estimate. '
            'Contact count is the minimum over two physics samples per tick. Stable markers denote past12-window '
            'endpoints, not total grasp duration. Native criteria and subgoals remain authoritative.',
        figure_sha256=sha256(figure), source_sha256=sha256(__file__),
        dependencies_sha256={p:sha256(Path(__file__).resolve().parents[1]/p) for p in
            ('tools/tight_fresh_phases.py','visuotactile/takeover_trigger.py')},
        model_forward_calls=0, optimizer_updates=0, simulator_starts=0, test_used=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for key in ('before', 'after', 'output', 'figure'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--case', required=True); args = p.parse_args()
    run(args.before, args.after, args.case, args.output, args.figure)
