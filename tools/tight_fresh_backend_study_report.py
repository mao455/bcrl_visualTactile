"""Bind a completed rear-weight comparison to actual learning and autonomous results."""
import argparse
import json
from pathlib import Path

import torch

from tools.tight_fresh_phases import case as phase_case
from tools.tight_fresh_rollouts import completed_pair
from tools.tight_fresh_study_onset import aggregate
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


EXPOSURE_KEYS = ('update', 'epoch', 'supervised_ticks', 'contact_ticks', 'history_ticks',
                 'window_chunks', 'initial_recovery_supervised_ticks', 'remaining_supervised_ticks',
                 'fixed_denominator', 'group_tail')


def read(path):
    return json.loads(path.read_text())


def check_exposure(control, weighted, weight):
    if len(control) != len(weighted) or not control:
        raise ValueError('Optimizer window counts differ or are empty')
    for update, (a, b) in enumerate(zip(control, weighted), 1):
        if a['update'] != update or any(a[k] != b[k] for k in EXPOSURE_KEYS):
            raise ValueError('Actual learning schedule or exposure differs')
        if (a['loss_weight_mass'] != a['supervised_ticks']
                or b['loss_weight_mass'] != b['supervised_ticks'] + (weight - 1) * b['initial_recovery_supervised_ticks']):
            raise ValueError('Actual onset loss weight mass differs')
    return dict(updates=len(control), supervised_ticks=sum(r['supervised_ticks'] for r in control),
        initial_recovery_ticks=sum(r['initial_recovery_supervised_ticks'] for r in control),
        control_loss_weight_mass=sum(r['loss_weight_mass'] for r in control),
        weighted_loss_weight_mass=sum(r['loss_weight_mass'] for r in weighted), loss_weight_mass_is_exposure=False)


def rear_report(root, role, expected_frontend):
    rear = root / role / 'backend' / role / 'training'
    finished = read(rear / 'finished.json')
    selected = read(rear / 'selected.json')
    if not finished['target_reached'] or selected['checkpoint_sha256'] != sha256(rear / 'best.pt'):
        raise ValueError('Incomplete or different selected rear')
    if selected['base_checkpoint_sha256'] != expected_frontend:
        raise ValueError('Frontend changed during the rear-only comparison')
    history = read(rear / 'validation.json')
    chosen = next(r for r in history if r['update'] == selected['update'])
    if (chosen['epoch'] != selected['epoch'] or chosen['selection_score'] != selected['validation_score']
            or chosen['validation']['all']['normalized_equal_mse'] != selected['validation_score']
            or selected['validation_score'] != min(r['selection_score'] for r in history)):
        raise ValueError('Selected checkpoint does not match fixed unweighted validation selection')
    losses = [json.loads(line) for line in (rear / 'loss.jsonl').read_text().splitlines()]
    last = torch.load(rear / 'last.pt', map_location='cpu', weights_only=False)
    if (finished['updates'] != len(losses) or sum(r['supervised_ticks'] for r in losses) != finished['supervised_ticks']
            or last['updates'] != finished['updates'] or last['supervised_ticks'] != finished['supervised_ticks']
            or last['control_ticks'] != finished['control_ticks']
            or last['configuration'] != read(rear / 'configuration.json')):
        raise ValueError('Training completion differs from actual optimizer receipts')
    return dict(selected=selected, selected_train=chosen['train'], selected_validation=chosen['validation'],
        history=history, finished=finished, loss_sha256=sha256(rear / 'loss.jsonl'),
        selected_sha256=sha256(rear / 'selected.json'), validation_sha256=sha256(rear / 'validation.json'),
        data_sha256=sha256(root / role / 'state_data/data.json'),
        cache_index_sha256=sha256(root / role / 'backend' / role / 'nominal/index.json'),
        actual_episode_counts=last['episode_counts'],
        selected_train_scope=dict(identities=last['train_ids'][:2],
            meaning='Fixed first two original training episodes; not the aggregate correction fitting error.'),
        validation_identities=last['validation_ids']), losses, last


def check_training_configuration(study, role, control, weighted, registration):
    normalized = []
    for arm, checkpoint in (('control', control), ('weighted', weighted)):
        config = checkpoint['configuration']
        expected_weight = registration['control_weight' if arm == 'control' else 'weighted_weight']
        onset = config['fresh_recovery_weighting']
        population = config['fresh_recovery_population']
        audit = config['continuation_audit']
        if (config['continuation_from']['sha256'] != registration['source_pairs'][role]['backend_sha256']
                or config['base_sha256'] != registration['source_pairs'][role]['frontend_sha256']
                or onset['ticks'] != registration['first_legal_recovery_ticks']
                or onset['backend_weight'] != expected_weight or population['contract'] != onset
                or population['loss_weight_mass'] != population['supervised_ticks'] +
                    (expected_weight - 1) * population['initial_recovery_supervised_ticks']
                or audit['optimizer_state_restored'] or not audit['statistics_and_delta_buffers_unchanged']
                or audit['scale_adaptation'] is not None or audit['new_optimizer_updates'] != 0):
            raise ValueError('Starting rear, frozen frontend, fresh optimizer or onset contract differs')
        current = json.loads(json.dumps(config).replace(str(study / arm), '<candidate>'))
        current['fresh_recovery_weighting']['backend_weight'] = 1.
        current['fresh_recovery_population']['contract']['backend_weight'] = 1.
        current['fresh_recovery_population']['loss_weight_mass'] = population['supervised_ticks']
        normalized.append(current)
        states = checkpoint['optimizer']['state']
        if not states or any(int(state['step']) != checkpoint['updates'] for state in states.values()):
            raise ValueError('Actual optimizer step counters differ from update receipts')
    if normalized[0] != normalized[1]:
        raise ValueError('Comparison configuration changed more than onset weight and candidate paths')
    if control['optimizer']['param_groups'] != weighted['optimizer']['param_groups']:
        raise ValueError('Optimizer hyperparameters or parameter groups differ')
    return dict(same_selected_source_sha256=registration['source_pairs'][role]['backend_sha256'],
        same_configuration_except_onset_weight_and_paths=True, fresh_optimizer_each=True,
        optimizer_step_counters_match_actual_updates=True)


def autonomous(root):
    directory = root / 'development'
    plan = read(directory / 'plan.json')
    if not read(directory / 'result.json')['software_complete']:
        raise ValueError('Autonomous cohort is incomplete')
    rows = read(directory / 'results.json')
    if len(rows) != len(plan['cases']) or {r['case'] for r in rows} != {c['name'] for c in plan['cases']}:
        raise ValueError('Autonomous denominator differs from the registered cohort')
    pairs = {role: completed_pair(root, role) for role in ('A', 'B')}
    if plan['kind'] != 'autonomous' or plan.get('test_used') is not False or plan['pairs'] != pairs:
        raise ValueError('Autonomous plan differs from current selected complete pairs')
    cases = []
    for registered in plan['cases']:
        name = registered['name']
        role = registered['role']
        row = next(r for r in rows if r['case'] == name)
        result = read(directory / name / 'result.json')
        started = read(directory / name / 'started.json')
        if (row['role'] != role or row['result'] != result or not result['software_complete'] or not result['legal']
                or started['spec'] != registered['spec'] or started['checkpoint'] != pairs[role]['backend']
                or any(result[key] != value for key, value in registered['spec'].items())):
            raise ValueError('Incomplete, illegal or inconsistent autonomous result')
        # Ordinary autonomous trial results record their rear hash, while the
        # plan binds the complete pair. complete_student_policy is takeover-only.
        if (result['checkpoint_sha256'] != pairs[role]['backend_sha256'] or result['expert_shadow_computed']
                or result['residual_version'] != pairs[role]['residual_version']):
            raise ValueError('Autonomous controller identity or expert computation differs')
        for key in ('dino_rgb', 'learned_visual_and_history', 'residual_forward_compose'):
            if result['native_inference']['summary'][key]['wall_s']['count'] != result['steps']:
                raise ValueError('Missing real native-frame processing')
        phases = phase_case(directory / name)
        if (phases['steps'] != result['steps'] or phases['authority']['expert_ticks'] != 0
                or result['simulation_step_hz'] != 60 or result['physics_hz'] != 120 or result['plan_hz'] != 5):
            raise ValueError('Autonomous authority, record length or native rates differ')
        cases.append(dict(role=role, **phases))
    totals = {}
    for role in ('A', 'B'):
        selected = [r for r in cases if r['role'] == role]
        seconds = sum(r['timing']['total_seconds'] for r in rows if r['role'] == role)
        totals[role] = dict(cases=len(selected), successes=sum(r['success'] for r in selected),
            native_lift_cases=sum(r['native_lift'] for r in selected),
            settled_stable_cases=sum(r['settled_height_stable_grasp'] for r in selected),
            alignment_cases=sum(r['native_alignment_first_tick'] is not None for r in selected),
            near_hole_cases=sum(r['native_near_goal_first_tick'] is not None for r in selected),
            first_subgoal_cases=sum(r['native_first_subgoal_tick'] is not None for r in selected),
            steps=sum(r['steps'] for r in selected), seconds=seconds,
            control_ticks_per_wall_second=sum(r['steps'] for r in selected) / seconds)
    return dict(cases=cases, totals=totals, plan_sha256=sha256(directory / 'plan.json'),
                results_sha256=sha256(directory / 'results.json')), plan


def check_onset_binding(candidate, reference, role, arm, registration_sha256, expected_rear):
    current = candidate['roles'][role]
    source = reference['roles'][role]
    if (not candidate['complete'] or candidate['test_used'] or candidate['device'] != 'cpu'
            or candidate['arm'] != arm or candidate.get('cuda_initialized') is not False
            or candidate.get('error_contract') != 'physical_command_rmse_first_contiguous_legal12_next12_remaining_v1'
            or candidate.get('source_sha256') != reference['source_sha256']
            or candidate.get('dependency_sha256') != reference['dependency_sha256']
            or candidate['registration_sha256'] != registration_sha256
            or any(current[key] != source[key] for key in
                ('frontend_sha256', 'data_sha256', 'numeric_sha256', 'cache_index_sha256'))
            or [(r['identity'], r['cache_sha256'], r['legal_mask_sha256'], r['bin_mask_sha256']) for r in current['cases']]
            != [(r['identity'], r['cache_sha256'], r['legal_mask_sha256'], r['bin_mask_sha256']) for r in source['cases']]):
        raise ValueError('Onset comparison uses different records or precision/device scope')
    if current['backend_sha256'] != expected_rear:
        raise ValueError('Onset diagnostic rear differs from the evaluated selected rear')
    return current


def run(study, output, *, onset_suffix='_onset_cpu'):
    if output.exists():
        raise FileExistsError('Preserve earlier report')
    torch.set_num_threads(2)
    registration = read(study / 'registration.json')
    if not read(study / 'finished.json')['software_complete']:
        raise ValueError('Both comparison arms must complete before reporting')
    for path, digest in registration['protected'].items():
        if sha256(path) != digest:
            raise ValueError('Protected comparison dependency changed')
    roles = {}
    for role in ('A', 'B'):
        reports, losses, checkpoints = {}, {}, {}
        for arm in ('control', 'weighted'):
            reports[arm], losses[arm], checkpoints[arm] = rear_report(study / arm, role,
                registration['source_pairs'][role]['frontend_sha256'])
        a, b = checkpoints['control'], checkpoints['weighted']
        initialization = check_training_configuration(study, role, a, b, registration)
        for key in ('train_ids', 'validation_ids', 'episode_counts', 'episode_order'):
            if a[key] != b[key]:
                raise ValueError('Actual checkpoint schedule differs: ' + key)
        if not torch.equal(a['schedule_rng'], b['schedule_rng']):
            raise ValueError('Actual schedule RNG differs')
        for key in ('data_sha256', 'cache_index_sha256'):
            if reports['control'][key] != reports['weighted'][key]:
                raise ValueError('Comparison inputs differ: ' + key)
        fairness = check_exposure(losses['control'], losses['weighted'], registration['weighted_weight'])
        if fairness['supervised_ticks'] != reports['control']['finished']['supervised_ticks']:
            raise ValueError('Fairness exposure differs from finished training')
        for arm in ('control', 'weighted'):
            checkpoint = checkpoints[arm]
            finished = reports[arm]['finished']
            exposure = finished['fresh_recovery_exposure']
            if (any(n != registration['epochs'] for n in checkpoint['episode_counts'].values())
                    or exposure['supervised_ticks'] != fairness['supervised_ticks']
                    or exposure['initial_recovery_supervised_ticks'] != fairness['initial_recovery_ticks']
                    or exposure['loss_weight_mass'] != fairness[arm + '_loss_weight_mass']):
                raise ValueError('Finished cumulative exposure differs from actual window receipts')
        def validation_population(report):
            return [(r['epoch'], r['update'], {split: {phase: metrics['ticks']
                for phase, metrics in r[split].items()} for split in ('train', 'validation')}) for r in report['history']]
        if validation_population(reports['control']) != validation_population(reports['weighted']):
            raise ValueError('Validation schedule or metric denominators differ')
        roles[role] = dict(arms=reports, matched_initialization=initialization, matched_actual_exposure=fairness)
    outcomes, plans = {}, {}
    for arm in ('control', 'weighted'):
        outcomes[arm], plans[arm] = autonomous(study / arm)
    if plans['control']['cases'] != plans['weighted']['cases']:
        raise ValueError('Registered autonomous opportunities differ')
    onset_paths = {arm: study / (arm + onset_suffix + '.json') for arm in ('source', 'control', 'weighted')}
    onset = {arm: read(path) for arm, path in onset_paths.items()}
    for role in ('A', 'B'):
        for arm in ('source', 'control', 'weighted'):
            expected = (registration['source_pairs'][role]['backend_sha256'] if arm == 'source'
                        else roles[role]['arms'][arm]['selected']['checkpoint_sha256'])
            current = check_onset_binding(onset[arm], onset['source'], role, arm,
                                          sha256(study / 'registration.json'), expected)
            for cohort, chosen in (('all53', current['cases']),
                                   ('new12', [r for r in current['cases'] if r['current_correction']])):
                summary = current['summaries'][cohort]
                if (summary['episodes'] != len(chosen) or len(chosen) != (53 if cohort == 'all53' else 12)
                        or summary['bins'] != {name: aggregate([row['bins'][name] for row in chosen])
                                               for name in ('first12', 'next12', 'remaining')}):
                    raise ValueError('Onset summary differs from physical per-case squared errors')
        roles[role]['onset_fitting'] = {arm: onset[arm]['roles'][role]['summaries'] for arm in onset}
    atomic_json(output, dict(complete=True, registration_sha256=sha256(study / 'registration.json'),
        source_commit=registration['source_commit'], source_sha256=sha256(__file__), roles=roles,
        autonomous=outcomes, onset_reports_sha256={arm: sha256(path) for arm, path in onset_paths.items()},
        test_used=False, limits='Matched rear starting weights, data, cache and actual update schedule. '
        'Weighted losses have different meanings; compare unweighted fixed validation and physical onset errors. '
        'Onset metrics use trained histories. Autonomous seeds identify matched opportunities, not exact simulator replay. '
        'Four development starts per role/arm do not establish robust success or a general A/B mechanism advantage.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--onset-suffix', default='_onset_cpu')
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve(), onset_suffix=args.onset_suffix)
