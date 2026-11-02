"""Read-only export of small State-history evidence; never starts a simulator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def first(values, threshold):
    hits = np.flatnonzero(values > threshold)
    return int(hits[0]) if len(hits) else None


def summarize(root, output):
    output.mkdir(parents=True, exist_ok=True)
    result = dict(root=str(root.resolve()), stages={}, trace_diagnostics=[], videos=[],
        resource_paused=(root/'RESOURCE_PAUSE').exists())
    for name in ('session.json', 'status.json', 'foundation.json', 'owner_exit.json'):
        path = root/name
        if path.exists():
            value = json.loads(path.read_text()); write(output/name, value)
            result[name] = dict(sha256=digest(path), value=value if name != 'session.json' else 'registration copied')
    stages = sorted(set(('preservation', 'parity', 'freeze', 'expert', 'replay', 'old', 'bc_train', 'bc', 'dp_train', 'dp')) |
        {p.parent.name for pattern in ('*/result.json', '*/progress.json') for p in root.glob(pattern)
            if '_single_' not in p.parent.name})
    for stage in stages:
        path = root/stage/'result.json'
        partial = not path.exists()
        if partial:
            path = root/stage/'progress.json'
        if not path.exists():
            result['stages'][stage] = dict(complete=False)
            continue
        value = json.loads(path.read_text()); write(output/(stage+'.json'), value)
        result['stages'][stage] = dict(complete=not partial and value.get('software_complete', False),
            partial=partial, sha256=digest(path))
        if 'trials' in value and value['trials']:
            result['stages'][stage]['cohorts'] = {cohort: dict(
                attempted=sum(r['cohort'] == cohort for r in value['trials']),
                legal=sum(r['cohort'] == cohort and r.get('legal', False) for r in value['trials']),
                successes=sum(r['cohort'] == cohort and r.get('success', False) for r in value['trials']),
                lifts=sum(r['cohort'] == cohort and r.get('native_lift', False) for r in value['trials']),
                subgoals=sum(r['cohort'] == cohort and r.get('native_subgoals', 0) > 0 for r in value['trials']))
                for cohort in sorted({r['cohort'] for r in value['trials']})}
    for stage in [s for s in stages if s.startswith(('old', 'bc', 'dp','native_h32','position_h32','image_')) and 'train' not in s and 'offline' not in s]:
        for path in sorted((root/stage).glob('*/trace.npz')):
            identity = path.parent.name; reference = root/('expert_formal' if stage.endswith('_formal') else 'expert')/identity/'trace.npz'
            if not reference.exists():reference=root/'expert_learning'/identity/'trace.npz'
            if not reference.exists():
                continue
            with np.load(path) as actual, np.load(reference) as expert:
                count = min(len(actual['command']), len(expert['command']))
                position=stage.startswith(('position_h32','image_'))
                error = actual['command'][:count]-expert['target' if position else 'command'][:count]
                arm = np.sqrt(np.mean(error[:, :7]**2, -1)); hand = np.sqrt(np.mean(error[:, 7:]**2, -1))
                joint = np.max(np.abs(actual['joint'][:count]-expert['joint'][:count]), -1)
                target = np.max(np.abs(actual['target'][:count]-expert['target'][:count]), -1)
                obj = np.linalg.norm(actual['object'][:count, :3]-expert['object'][:count, :3], axis=-1)
                contact = np.linalg.norm(expert['object_contact'][:count], axis=-1).reshape(count, -1).max(-1) > .1
                lift = expert['lifted'][:count].reshape(count, -1).any(-1)
                goals = expert['subgoals'][:count].reshape(count, -1).max(-1)
                masks = dict(before_object_contact=np.cumsum(contact) == 0,
                    after_contact_before_lift=(np.cumsum(contact) > 0) & ~lift & (goals == 0),
                    lifted_before_subgoal=lift & (goals == 0), after_first_subgoal=goals > 0)
                shadow = None if position else actual['command']-actual['expert_shadow']
                jumps = np.sqrt(np.mean(np.diff(actual['command'], axis=0)**2, axis=-1))
                boundary = np.arange(1, len(actual['command'])) % 12 == 0
                phase = {name: dict(ticks=int(mask.sum()), arm_rmse=float(np.sqrt(np.mean(arm[mask]**2))),
                    hand_rmse=float(np.sqrt(np.mean(hand[mask]**2)))) for name, mask in masks.items() if mask.any()}
                row = dict(stage=stage, identity=identity, compared_ticks=count,
                    first_contact_expert=first(contact.astype(float), .5),
                    command_arm_first_over_0p1=first(arm, .1), command_hand_first_over_0p1=first(hand, .1),
                    joint_first_over_1mrad=first(joint, .001), target_first_over_1mrad=first(target, .001),
                    object_first_over_1mm=first(obj, .001), reset_arm_rmse=float(arm[0]), reset_hand_rmse=float(hand[0]),
                    reset_actor_max=float(np.max(np.abs(actual['actor'][0]-expert['actor'][0]))),
                    action_unit='rad' if position else 'native_command',
                    tracking_arm_rmse_rad=float(np.sqrt(np.mean((actual['joint'][:,:7]-actual['target'][:,:7])**2))),
                    tracking_hand_rmse_rad=float(np.sqrt(np.mean((actual['joint'][:,7:]-actual['target'][:,7:])**2))),
                    actual_trajectory_expert_shadow_arm_rmse=None if shadow is None else float(np.sqrt(np.mean(shadow[:, :7]**2))),
                    actual_trajectory_expert_shadow_hand_rmse=None if shadow is None else float(np.sqrt(np.mean(shadow[:, 7:]**2))),
                    chunk_boundary_command_jump_rms=float(np.sqrt(np.mean(jumps[boundary]**2))) if boundary.any() else None,
                    within_chunk_command_jump_rms=float(np.sqrt(np.mean(jumps[~boundary]**2))) if (~boundary).any() else None,
                    phases=phase, trace_sha256=digest(path),
                    scope='Expert reference phases; shadow has its own LSTM over actual student observations, never controls')
                result['trace_diagnostics'].append(row)
    legal = []
    for path in sorted((root/'freeze').glob('*/initial_collision.json')):
        value = json.loads(path.read_text()); legal.append(dict(identity=path.parent.name, **value))
    write(output/'initial_collision.json', legal)
    for path in sorted(root.glob('*/*/*.mp4')):
        if '_single_' in path.parts[-3]:
            continue
        result['videos'].append(dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest(path)))
    if (root/'resources/resources.json').exists():
        write(output/'resources.json', json.loads((root/'resources/resources.json').read_text()))
    write(output/'summary.json', result)
    plot(root, output, result)
    resource_plot(root, output)
    return result


def resource_plot(root, output):
    source = root/'resources/shared.jsonl'
    if not source.exists():
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if not rows:
        return
    time = [(r['utc_epoch']-rows[0]['utc_epoch'])/60 for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for key, name in [('gpu_util_percent', 'whole card'), ('rolling_60s_percent', 'recent1m'), ('rolling_300s_percent', 'recent5m')]:
        axes[0].plot(time, [r.get(key) for r in rows], label=name, linewidth=1)
    axes[0].set(ylabel='GPU core utilization %'); axes[0].legend()
    axes[1].plot(time, [(r['gpu_free_mib'] or 0)/1024 for r in rows], label='free GPU GiB')
    axes[1].plot(time, [r['ram_available_bytes']/1024**3 for r in rows], label='available RAM GiB')
    axes[1].set(xlabel='minutes', ylabel='GiB'); axes[1].legend()
    fig.tight_layout(); fig.savefig(output/'resource_curves.png', dpi=150); plt.close(fig)


def plot(root, output, summary):
    kinds = sorted(p.name[:-6] for p in root.glob('*_train') if (p/'metrics.json').exists())
    if not any((root/(kind+'_train')/'metrics.json').exists() for kind in kinds):
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for kind in kinds:
        path = root/(kind+'_train')/'metrics.json'
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        for part, ax in zip(('arm_rmse', 'hand_rmse'), axes):
            for split, style in (('train', '-o'), ('validation', '--o')):
                ax.plot([r['update'] for r in rows], [r[split][part] for r in rows], style, label=kind+' '+split)
    for title, ax in zip(('Native arm RMSE', 'Native hand RMSE'), axes):
        ax.set(title=title, xlabel='Optimizer updates', ylabel='Teacher-forced command RMSE'); ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'learning_curves.png', dpi=160); plt.close(fig)
    if not any('h32' in kind for kind in kinds):return
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for kind in kinds:
        rows=json.loads((root/(kind+'_train')/'metrics.json').read_text())
        logs=root/(kind+'_train')/'train.jsonl'
        if logs.exists():
            updates=[json.loads(line) for line in logs.read_text().splitlines()]
            if updates:
                window=25
                x=[r['update'] for r in updates]
                y=np.convolve([r['loss'] for r in updates],np.ones(min(window,len(updates)))/min(window,len(updates)),mode='valid')
                axes[0,0].plot(x[len(x)-len(y):],y,label=kind+' train')
        axes[0,0].plot([r['update'] for r in rows],[r['validation'].get('loss',float('nan')) for r in rows],'o--',label=kind+' validation')
    diagnostic=summary['trace_diagnostics']
    for part in ('arm','hand'):
        axes[0,1].plot(range(len(diagnostic)),[r['tracking_'+part+'_rmse_rad'] for r in diagnostic],'.',label=part)
    cohorts=[(stage,cohort,counts) for stage,v in summary['stages'].items() for cohort,counts in v.get('cohorts',{}).items() if stage.startswith(('position_h32','native_h32','image_'))]
    for field,label in [('successes','complete native success'),('lifts','native lift'),('subgoals','first subgoal')]:
        axes[1,0].plot(range(len(cohorts)),[counts[field]/max(1,counts['legal']) for _,_,counts in cohorts],'o-',label=label)
    axes[1,0].set_xticks(range(len(cohorts)),[s.replace('_h32','')+':'+c for s,c,_ in cohorts],rotation=90,fontsize=6)
    for part in ('arm_rmse','hand_rmse'):
        for kind in kinds:
            rows=json.loads((root/(kind+'_train')/'metrics.json').read_text())
            axes[1,1].plot([r['update'] for r in rows],[r['validation'][part] for r in rows],'o-',label=kind+' '+part)
    for ax,title in zip(axes.flat,['Diffusion loss (25-update train mean)','Actual target tracking RMSE, rad','Completed legal trials only; missing listed in JSON','Validation first12 action error (units by profile)']):
        ax.set_title(title,fontsize=9);ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'position_curves.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    report = summarize(a.root, a.output)
    print(json.dumps(report['stages'], indent=2))
