"""Plot comparable unweighted metrics from the audited paired study report."""
import argparse
import json
import math
from pathlib import Path


def autonomous_panels(data, role):
    """Keep historical source and new study denominators separate in the figure."""
    result=[]
    for arm in ('source','control','previous'):
        cohort=data['source_autonomous'] if arm=='source' else data['autonomous']
        totals=cohort['totals'][role] if arm=='source' else cohort['totals'][arm][role]
        cases=[row for row in cohort['cases'] if row['role']==role and row['arm']==arm]
        if len(cases)!=totals['opportunities'] or sum(bool(row['legal']) for row in cases)!=totals['legal']:
            raise ValueError('Autonomous chart denominator differs from audited cases')
        turns=[]
        for row in cases:
            snapshot=row['phase_snapshots']['first_goal_after_hold_best_orientation'] if row['legal'] else None
            if snapshot is not None:
                angle=snapshot['root_quaternion_angle_rad']
                if not math.isfinite(angle) or not 0<=angle<=math.pi+1e-6:
                    raise ValueError('Invalid recorded orientation diagnostic')
                turns.append(dict(case=row['case'],degrees=math.degrees(angle),stable=bool(snapshot['stable_window'])))
        result.append(dict(arm=arm,label=data['arm_semantics'][arm]['label'],totals=totals,turns=turns))
    return result


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads(report.read_text())
    if not data['complete'] or data['test_used']:
        raise ValueError('A complete non-test comparison is required')
    candidate = 'previous' if data.get('version') == 'matched_anchor_study_report_v1' else 'weighted'
    behavior=candidate=='previous' and data.get('report_schema_revision',1)>=2
    columns=5 if behavior else 3
    figure, axes = plt.subplots(2, columns, figsize=(4*columns, 7 if behavior else 6), constrained_layout=True)
    cohort = 'all_corrections' if candidate == 'previous' else 'all53'
    colors = {'source':'#888888','control': '#3972ba', candidate: '#c46c32'}
    labels={arm:data.get('arm_semantics',{}).get(arm,{}).get('label',arm) for arm in ('source','control',candidate)}
    tick_labels={arm:label.replace(' (','\n(').replace(' anchor','\nanchor') for arm,label in labels.items()}
    for row, role in enumerate(('A', 'B')):
        result = data['roles'][role]
        for col, split in enumerate(('train', 'validation')):
            axis = axes[row, col]
            scopes=[]
            for arm in ('control', candidate):
                summary=result['arms'][arm];history = summary['history']
                scopes.append(summary.get('selected_train_scope',{}).get('identities') if split=='train'
                    else summary.get('validation_identities'))
                axis.plot([r['epoch'] for r in history],
                          [r[split]['all']['hand_rmse_rad'] for r in history],
                          marker='o', color=colors[arm], label=labels[arm])
            if scopes[0]!=scopes[1] or (scopes[0] is not None and len(scopes[0])!=len(set(scopes[0]))) or (behavior and scopes[0] is None):
                raise ValueError('Fixed fitting-curve populations differ')
            count=str(len(scopes[0])) if scopes[0] is not None else ''
            axis.set(title=f'{role}: fixed {split}{count} command error', xlabel='Additional rear epoch',
                     ylabel='Hand RMSE (rad)')
            axis.grid(alpha=.2)
            axis.legend()
        axis = axes[row, 2]
        arms = ('source', 'control', candidate)
        population = result['onset_fitting']['source'][cohort]['episodes']
        axis.bar([tick_labels[arm] for arm in arms], [result['onset_fitting'][arm][cohort]['bins']['first12']['hand_rmse_rad']
                        for arm in arms], color=['#888888', colors['control'], colors[candidate]])
        axis.set(title=f'{role}: first 12 legal recovery ticks', ylabel='Hand RMSE (rad)',
            xlabel=f'{population} learned corrections; same recorded histories')
        axis.tick_params(axis='x',labelsize=8);axis.xaxis.label.set_size(8)
        if behavior:
            panels=autonomous_panels(data,role);axis=axes[row,3]
            stages=[('native_lift_cases','Lift\n(native)'),('settled_stable_cases','Hold\n(12 ticks)'),
                ('near_hole_cases','Near\n(keypoints)'),('first_subgoal_cases','Pre-insert\nsubgoal'),('successes','Insert\ncomplete')]
            for ordinal,panel in enumerate(panels):
                totals=panel['totals'];count=totals['opportunities']
                axis.bar([i+(ordinal-1)*.25 for i in range(len(stages))],
                    [totals[key] for key,_ in stages],width=.25,color=colors[panel['arm']],
                    label=f"{panel['label']}: {count} starts, {totals['legal']} legal")
            axis.set(title=f'{role}: actual autonomous milestones',ylabel='Cases',xticks=range(len(stages)),
                xticklabels=[label for _,label in stages],ylim=(0,max(p['totals']['opportunities'] for p in panels)+.6))
            axis.tick_params(axis='x',labelsize=8)
            axis.legend(fontsize=7);axis.grid(axis='y',alpha=.2)
            axis=axes[row,4]
            for ordinal,panel in enumerate(panels):
                for tick,turn in enumerate(panel['turns']):
                    jitter=(tick-(len(panel['turns'])-1)/2)*.05
                    axis.scatter(ordinal+jitter,turn['degrees'],edgecolors=colors[panel['arm']],
                        facecolors=colors[panel['arm']] if turn['stable'] else 'none')
                axis.text(ordinal,186,f"{len(panel['turns'])}/{panel['totals']['opportunities']} with prior hold",ha='center',fontsize=7)
            axis.set(title=f'{role}: best target angle after first hold',ylabel='Root quaternion angle (degrees)',
                xticks=range(3),xticklabels=[tick_labels[p['arm']] for p in panels],ylim=(-3,200),xlim=(-.5,2.5),
                xlabel='No hold = missing; hollow = no stable window\nat the angle minimum')
            axis.tick_params(axis='x',labelsize=8);axis.grid(axis='y',alpha=.2);axis.xaxis.label.set_size(7)
    title='Matched additional training; unweighted errors, not autonomous success'
    if behavior:
        title='Offline fitting and actual autonomous behavior; source is a separate earlier cohort\nRoot poses are reference origins; raw orientation ignores symmetry. Milestone counts are not a nested funnel.'
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.report, args.output)
