"""Read-only A/B comparison gate for the existing complete-policy round layout."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit(round_root, *, complete=True):
    root = Path(round_root)
    checks = {}
    groups = {}
    for group in ('QG01', 'G01'):
        job = root / group
        groups[group] = dict(
            dp=read(job / 'dp/training/configuration.json'),
            rear=read(job / 'backend' / group / 'training/configuration.json'),
            data=read(job / 'dp/state_data/data.json'),
            cache=read(job / 'backend' / group / 'nominal/index.json'),
            plan=read(job / 'backend/evaluation_plan.json'))
    a, b = groups.values()

    def equal(section, keys):
        for key in keys:
            checks[f'{section}.{key}'] = (key in a[section] and key in b[section]
                                         and a[section][key] == b[section][key])

    equal('data', ['splits', 'sampling_strata', 'snapshot_id'])
    equal('dp', ['seed', 'target', 'effective_batch', 'microbatch', 'lr_peak',
                 'lr_floor', 'warmup', 'sampler', 'incremental_schedule', 'selection',
                 'diagnostic_cohorts', 'dino_sha256', 'target_window_contract',
                 'training_performance', 'sampling_weights', 'recovery_prefix', 'correction_dev_ids'])
    equal('rear', ['initialization_seed', 'microbatch', 'learning_rate', 'weight_decay',
                   'maximum_control_tick_epochs', 'minimum_control_tick_epochs',
                   'validation_epoch_interval', 'selection', 'loss_contract', 'output_contract',
                   'history_contract', 'execution_adapter', 'tick_window_contract',
                   'supervision_mask_sha256', 'sampling_strata', 'sampling_weights',
                   'recovery_prefix', 'architecture', 'residual_statistics_sha256',
                   'residual_normalization_identity', 'cache_definition',
                   'residual_hz', 'horizon', 'commit', 'tbptt_ticks'])
    equal('cache', ['sampler', 'inputs', 'supervision_mask_sha256',
                    'residual_statistics_sha256', 'residual_normalization_identity'])
    # Plans embed different group names; the externally specified opportunity must agree.
    plans = [[{'identity': row['identity'], 'spec': row['spec']} for row in g['plan']]
             for g in (a, b)]
    checks['evaluation.same_opportunities'] = bool(plans[0]) and plans[0] == plans[1]
    # residual_setup loads residual_statistics into TargetResidual; statistics is the frontend.
    for group, g in groups.items():
        stats = Path(g['rear']['residual_statistics_path'])
        checks[f'{group}.actual_command_statistics'] = (
            hashlib.sha256(stats.read_bytes()).hexdigest() == g['rear']['residual_statistics_sha256'])
        cache_path = root / group / 'backend' / group / 'nominal/index.json'
        checks[f'{group}.cache_manifest_binding'] = (
            hashlib.sha256(cache_path.read_bytes()).hexdigest() == g['rear']['cache_index_sha256']
            and all(g['cache'].get(k) == g['rear'].get(k) for k in
                ('base_sha256', 'data_sha256', 'statistics_sha256', 'residual_statistics_sha256',
                 'supervision_mask_sha256')))
        checks[f'{group}.frontend_statistics_binding'] = (
            g['rear']['statistics_sha256'] == g['dp']['statistics_sha256'])
    checks['labels.intentional_semantics'] = (a['dp']['teacher_supervision'] == 'q_reached'
        and b['dp']['teacher_supervision'] == 'u_expert'
        and a['rear']['teacher_supervision'] == b['rear']['teacher_supervision'] == 'u_expert')
    counters = {}
    if complete:
        for group, g in groups.items():
            job = root / group
            g['dp_done'] = read(job / 'dp/training/finished.json')
            g['rear_done'] = read(job / 'backend' / group / 'training/finished.json')
            g['eval'] = read(job / 'backend' / group / 'evaluation_summary.json')
            g['receipt'] = read(job / 'backend/evaluation_complete.json')
            checks[f'{group}.selected_frontend_pair'] = (
                Path(g['dp_done']['best_checkpoint']).resolve() == Path(g['rear']['base_checkpoint']).resolve())
            counters[group] = dict(dp_updates=g['dp_done']['updates'],
                dp_target_slots=g['dp_done']['supervised_target_ticks'],
                rear_updates=g['rear_done']['updates'], rear_ticks=g['rear_done']['supervised_ticks'],
                completed=g['eval']['completed'], successes=g['eval']['successes'])
            checks[f'{group}.complete'] = (g['dp_done']['target_reached']
                and g['rear_done']['target_reached'] and g['eval']['complete']
                and g['receipt']['complete'] and g['eval']['software_failures'] == 0
                and g['eval']['completed'] == len(g['plan']))
            checks[f'{group}.autonomous_non_test'] = (g['eval']['expert_takeover'] is False
                and g['eval']['formal_test22_used'] is False)
            actual = [(r['identity'], r['seed'], r['narrow_reset']) for r in g['eval']['results']]
            expected = [(r['identity'], r['spec']['seed'], r['spec']['narrow_reset']) for r in g['plan']]
            checks[f'{group}.actual_opportunities'] = actual == expected
        equal('dp_done', ['updates', 'supervised_target_ticks', 'coverage'])
        equal('rear_done', ['updates', 'supervised_ticks', 'coverage'])
        checks['evaluation.same_valid_denominator'] = a['eval']['valid'] == b['eval']['valid']
    return dict(version='matched_increment_ab_audit_v1', phase='completed' if complete else 'prepared',
        matched_increment_passed=all(checks.values()), checks=checks,
        failed=[k for k, v in checks.items() if not v], actual=counters,
        same_plan_digest=digest(plans[0]) if plans[0] == plans[1] else None,
        evidence_scope='Declared configs/cache bindings plus actual completed receipts and coverage counts; not independently replayed sampler order, optimizer tensors or hidden simulator bitwise parity.',
        strict_all_history_equality=False,
        intentional_differences=['A q_reached plus established geometry; B u_expert',
                                 'Target-specific frontend normalizers and trained initial weights'],
        limitations=['Historical initialization and exposure differ; only the incremental protocol is matched.',
                     'B-only old/new replication is not an A/B comparison and must not be pooled into this cohort.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared', action='store_true')
    args = parser.parse_args()
    try:
        result = audit(args.round, complete=not args.prepared)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = dict(matched_increment_passed=False, failed=['missing_or_invalid_evidence'],
                      actual={}, error=type(exc).__name__)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['matched_increment_passed', 'failed', 'actual']}))
    raise SystemExit(0 if result['matched_increment_passed'] else 2)


if __name__ == '__main__':
    main()
