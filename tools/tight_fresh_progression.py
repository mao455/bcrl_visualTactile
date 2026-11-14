"""Compare fixed validation and complete autonomous development across rounds.

Reads existing metric reports and actual evaluation receipts, never test data or
model tensors. Repeated development seeds are not independent test trials.
"""
import argparse
import json
from pathlib import Path

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def summarize(learning, phases, plan, results):
    require(plan['kind'] == 'autonomous' and plan['test_used'] is False
            and learning['test_used'] is False, 'Only non-test complete autonomous cohorts')
    require(phases['planned'] == phases['completed'] == len(phases['cases']) == len(plan['cases']) == len(results),
            'Incomplete or changed development denominator')
    phase_rows = {row['case']: row for row in phases['cases']}
    result_rows = {row['case']: row for row in results}
    names = [case['name'] for case in plan['cases']]
    require(len(names) == len(set(names)) == len(phase_rows) == len(result_rows)
            and set(names) == set(phase_rows) == set(result_rows), 'Duplicate or missing cases')
    roles = {}
    for role in ('A', 'B'):
        learned, pair = learning['roles'][role], plan['pairs'][role]
        require(learned['frontend_complete'] and learned['backend_offline']['complete'],
                'Incomplete selected student')
        require(learned['checkpoint_sha256'] == pair['frontend_sha256']
                and learned['backend_cache']['base_sha256'] == pair['frontend_sha256']
                and learned['backend_selected']['checkpoint_sha256'] == pair['backend_sha256'],
                'Learning report and deployed complete pair differ')
        require(learned['metric_cohorts']['validation'] == learned['backend_actual_validation_ids'],
                'Frontend/backend validation populations differ')
        cases = [case for case in plan['cases'] if case['role'] == role]
        require(bool(cases), 'Missing autonomous student role')
        selected = []
        for case in cases:
            row, phase = result_rows[case['name']], phase_rows[case['name']]
            result = row['result']
            require(row['role'] == role and result['legal'] and result['software_complete']
                    and not result.get('censored') and not result.get('worker_error')
                    and case.get('dagger') is None and result.get('dagger') is None
                    and result['checkpoint_sha256'] == pair['backend_sha256']
                    and ('complete_student_policy' not in result or result['complete_student_policy'] == pair)
                    and result['simulation_step_hz'] == 60 and result['plan_hz'] == 5
                    and result['residual_ticks'] == result['steps']
                    and result['action_control_mode'] == 'absolute_joint_position_target_rad',
                    'Invalid full-student autonomous result')
            require(phase['legal'] and phase['steps'] == result['steps']
                    and phase['success'] == result['success'], 'Phase and result disagree')
            selected.append(phase)
        backend = learned['backend_selected_metrics']
        roles[role] = dict(
            training_episodes=len(learned['actual_episode_counts']),
            frontend_selected_epoch=learned['selected_epoch'],
            frontend_new_updates=learned['actual_optimizer_episodes'],
            backend_selected_epoch=learned['backend_selected']['epoch'],
            backend_new_updates=learned['backend_finished']['updates'],
            validation_ids=learned['metric_cohorts']['validation'],
            train_diagnostic_ids=learned['metric_cohorts']['train'],
            frontend_validation=learned['selected_validation']['selection'],
            frontend_train_diagnostic=learned['selected_train']['selection'],
            backend_validation_hand_rad=backend['validation']['all']['hand_rmse_rad'],
            backend_train_hand_rad=backend['train']['all']['hand_rmse_rad'],
            backend_validation_arm_rad=backend['validation']['all']['arm_rmse_rad'],
            opportunities=len(cases), successes=sum(r['success'] for r in selected),
            native_lift_cases=sum(r['native_lift'] for r in selected),
            stable_window_cases=sum(r['settled_height_stable_grasp'] for r in selected),
            native_alignment_cases=sum(r['native_alignment_first_tick'] is not None for r in selected),
            first_subgoal_cases=sum(r['native_subgoals'] > 0 for r in selected),
            frontend_sha256=pair['frontend_sha256'], backend_sha256=pair['backend_sha256'])
    return roles


def report(specification):
    rounds, bindings, reference = [], {}, None
    require(len(specification['rounds']) >= 2, 'At least two complete rounds required')
    require(len({r['name'] for r in specification['rounds']}) == len(specification['rounds']),
            'Duplicate round names')
    for spec in specification['rounds']:
        learning_path, phase_path = Path(spec['learning']), Path(spec['phases'])
        evaluation = Path(spec['evaluation'])
        paths = [learning_path, phase_path, *(evaluation / n for n in ('plan.json', 'result.json', 'results.json'))]
        learning, phases, plan, finished, results = map(read, paths)
        require(finished.get('software_complete') is True, 'Development not complete')
        require(phases['source_plan_sha256'] == sha256(evaluation / 'plan.json'), 'Phase plan changed')
        for row in phases['cases']:
            path = evaluation / row['case'] / 'result.json'
            require(sha256(path) == row['result_sha256'], 'Phase result changed')
            indexed = next((r['result'] for r in results if r['case'] == row['case']), None)
            require(read(path) == indexed, 'Result index disagrees with actual receipt')
            bindings[str(path)] = sha256(path)
        roles = summarize(learning, phases, plan, results)
        comparison = dict(
            cohorts={role: (value['validation_ids'], value['train_diagnostic_ids']) for role, value in roles.items()},
            statistics={role: learning['roles'][role]['statistics_sha256'] for role in roles},
            development=[(case['name'], case['role'], case['spec']) for case in plan['cases']],
            stable_rule=phases['stable_rule'], settled_reference=phases['settled_reference'])
        if reference is None:
            reference = comparison
        require(comparison == reference, 'Fixed cohorts or diagnostic definitions changed across rounds')
        bindings.update({str(path): sha256(path) for path in paths})
        rounds.append(dict(name=spec['name'], roles=roles, evaluation_source_commit=plan['source_commit']))
    return dict(version='fixed_cohort_research_progression_v1', rounds=rounds, input_sha256=bindings,
        source_sha256=sha256(__file__),
        test_used=False, binding_scope='Existing metric reports and actual per-case evaluation receipts; model/raw numeric hashes are inherited, not rehashed here.',
        interpretation='Fixed validation6 and train2 diagnostics, selected complete DP+backend weights. Repeated development opportunities are compared per round, never pooled as independent trials. Stable windows are brief diagnostic events, not task success. A/B DP labels differ; compare frontend errors only within role. Data and optimizer exposure change together across DAgger rounds, so this is descriptive, not a causal attribution.')


def plot(value, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rounds = value['rounds']
    denominators = {r['roles'][role]['opportunities'] for r in rounds for role in ('A', 'B')}
    require(len(denominators) == 1, 'Plot requires equal development opportunity counts')
    denominator = denominators.pop()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for role, color in [('A', '#1166aa'), ('B', '#d46a18')]:
        rows = [r['roles'][role] for r in rounds]
        x = list(range(len(rows)))
        axes[0].plot(x, [r['backend_validation_hand_rad'] for r in rows], 'o-', color=color, label=role + ' validation6')
        axes[0].plot(x, [r['backend_train_hand_rad'] for r in rows], ':', color=color, label=role + ' train2')
        for ax, key in zip(axes[1:], ('stable_window_cases', 'successes')):
            ax.plot(x, [r[key] for r in rows], 'o-', color=color, label=role)
    for ax in axes:
        ax.set_xticks(range(len(rounds)), [r['name'] for r in rounds])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set(title='Selected backend hand-command error', ylabel='RMSE (rad)')
    for ax, title in zip(axes[1:], ('Any past12-tick stable window', 'Autonomous native task success')):
        ax.set(title=title, ylabel=f'Cases / {denominator} repeated development opportunities',
               ylim=(-.1, denominator+.1), yticks=range(denominator+1))
    fig.suptitle('Learning on recorded histories versus complete autonomous behavior; repeated development, not held-out test', fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--specification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figure', type=Path)
    args = p.parse_args()
    value = report(read(args.specification))
    value['specification_sha256'] = sha256(args.specification)
    atomic_json(args.output, value)
    if args.figure:
        plot(value, args.figure)


if __name__ == '__main__':
    main()
