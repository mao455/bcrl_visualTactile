"""Bounded research decisions from validation and diagnostic cohorts, never final test.

Thresholds are scheduling heuristics, not task acceptance criteria. Every decision
retains the measurements and can stop without claiming convergence or success.
"""
import math
import statistics


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(values) if values else None


def curve_summary(rows):
    if not rows:
        raise ValueError('No validation evidence')
    # A short final interval must not masquerade as a full validation interval.
    spaced = []
    for row in reversed(rows):
        if not spaced or spaced[-1]['update'] - row['update'] >= 400:
            spaced.append(row)
    window = list(reversed(spaced[:4]))
    first, last = window[0], window[-1]
    gain = lambda a, b: (a-b)/max(abs(a), 1e-12)
    early = lambda r, cohort, key: mean(x[key] for x in r['early'][cohort])
    hold = lambda r, cohort: mean(x['arm_rmse_rad']/max(x['hold_arm_rmse_rad'], 1e-8)
                                 for x in r['early'][cohort])
    return dict(points=len(window),span_updates=last['update']-first['update'],update=last['update'],
        validation_gain=gain(first['metrics']['selection_mse'],last['metrics']['selection_mse']),
        train_gain=gain(first['train8']['selection_mse'],last['train8']['selection_mse']),
        arm_gain=gain(first['metrics']['arm_rmse'],last['metrics']['arm_rmse']),
        fk_gain=gain(first['metrics']['ee_distance_rmse_mm'],last['metrics']['ee_distance_rmse_mm']),
        gap=last['metrics']['selection_mse']/max(last['train8']['selection_mse'],1e-12),
        first_gap=first['metrics']['selection_mse']/max(first['train8']['selection_mse'],1e-12),
        train_early_hold_ratio=hold(last,'train'),validation_early_hold_ratio=hold(last,'validation'),
        early_direction=early(last,'validation','direction_cos'),
        early_amplitude=early(last,'validation','amplitude_ratio'),
        early_hold_regression=hold(last,'validation')/max(hold(first,'validation'),1e-8)-1,
        latest_validation=last['metrics']['selection_mse'],latest_train8=last['train8']['selection_mse'],
        best_update=min(rows,key=lambda r:r['metrics']['selection_mse'])['update'])


def decide(rows, cohorts, *, extension_used=False, audit_passed=False):
    c = curve_summary(rows)
    evidence = dict(curve=c,cohorts=cohorts,extension_used=extension_used,
                    final_test_used=False,heuristics_are_not_success_thresholds=True)
    train, unseen = cohorts['train'], cohorts['unseen']
    if train['valid'] < 3 or unseen['valid'] < 8:
        return dict(evidence,action='stop',reason='Insufficient legal closed-loop evidence; no outcome replacements')
    tr = train['successes']/train['valid']; ur = unseen['successes']/unseen['valid']
    if ur >= .5 and tr >= .5:
        return dict(evidence,action='final_validate',reason='Meaningful small-cohort success; independent final validation before ending')
    improving = (c['points'] >= 3 and c['validation_gain'] > .03 and
                 max(c['arm_gain'],c['fk_gain']) > .01 and c['early_hold_regression'] < .05 and
                 c['gap']/c['first_gap'] < 1.6)
    if improving and not extension_used:
        return dict(evidence,action='extend',reason='Validation and motion still improve without rapidly widening gap')
    plateau = c['points'] >= 3 and c['validation_gain'] <= .03
    coverage = c['gap'] >= 2.5 and c['train_early_hold_ratio'] < .8 and c['validation_early_hold_ratio'] > 1.
    if plateau and coverage and audit_passed:
        return dict(evidence,action='expand',reason='Validation plateau plus fitted train motion and unseen-state gap; audited coverage hypothesis')
    return dict(evidence,action='stop',reason='Evidence does not justify more updates or one data expansion; preserve for method review')


def early_stop(rows, minimum_update):
    if len(rows) < 4 or rows[-1]['update'] < minimum_update:
        return False
    c = curve_summary(rows)
    return c['points'] >= 4 and c['validation_gain'] < .01 and (c['gap'] > 2 or c['validation_gain'] < -.03)


def continuation_rate(step, original):
    from .anchor_sampling import continuation_lr
    # Continue the original cosine to its existing floor. No rewarm, stretching,
    # optimizer/EMA reset or modification of any historical schedule value.
    total = original['target']
    return continuation_lr(min(step,total),total,original['warmup'],original['lr_peak'],original['lr_floor'])


def failure_stage(result):
    if not result.get('legal'): return 'illegal_initialization'
    if result.get('success'): return 'complete'
    if result.get('native_subgoals',0) >= 2: return 'retraction_after_ordered_insertion'
    if result.get('native_subgoals',0) >= 1: return 'second_insertion'
    if result.get('native_alignment'): return 'insertion_before_first_subgoal'
    if result.get('native_lift'): return 'post_lift_alignment_or_loss'
    if result.get('approach_proxy'): return 'grasp_or_lift_unresolved'
    return 'approach'
