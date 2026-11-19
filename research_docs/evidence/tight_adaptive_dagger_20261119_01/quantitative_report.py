"""Summarize completed evidence without policy inference or physics.

Offline parent comparisons join the exact trajectory, phase and component.
Current-round cohort labels define both sides; older cohort labels may differ.
All aggregate RMSE values use summed squared errors and joint/tick counts.
"""
import argparse
import collections
import csv
import math
from pathlib import Path
import statistics

from tools.tight_adaptive_dagger import PUBLIC, read, write
from visuotactile.compact import sha256


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def key(row):
    return tuple(row[k] for k in ('role', 'identity', 'stage', 'component'))


def compare(current, parent):
    """Only compare actual matching legal populations, never episode means."""
    lookup = {key(row): row for row in parent}
    assert len(lookup) == len(parent), 'Duplicate parent measurement'
    assert len({key(row) for row in current}) == len(current)
    totals = collections.defaultdict(lambda: collections.Counter())
    identities = collections.defaultdict(set)
    missing = collections.Counter()
    for child in current:
        group = tuple(child[k] for k in ('role', 'cohort', 'stage', 'component'))
        before = lookup.get(key(child))
        if before is None:
            missing[group] += 1
            continue
        ticks = int(child['ticks'])
        assert ticks == int(before['ticks']) and ticks > 0, 'Mask population changed'
        identities[group].add(child['identity'])
        total = totals[group]
        total['ticks'] += ticks
        for label, row in (('parent', before), ('new', child)):
            for part in ('arm', 'hand'):
                value = float(row[part + '_squared_sum'])
                assert math.isfinite(value) and value >= 0
                total[label + '_' + part] += value
    result = []
    for group, total in sorted(totals.items()):
        out = dict(zip(('role', 'cohort', 'stage', 'component'), group))
        out.update(trajectories=len(identities[group]), ticks=total['ticks'],
                   missing_parent_trajectories=missing[group])
        for part, dimensions in (('arm', 7), ('hand', 22)):
            for label in ('parent', 'new'):
                out[label + '_' + part + '_rmse_rad'] = math.sqrt(
                    total[label + '_' + part] / total['ticks'] / dimensions)
            old = out['parent_' + part + '_rmse_rad']
            out[part + '_relative_change_percent'] = (
                100 * (out['new_' + part + '_rmse_rad'] / old - 1) if old else None)
        result.append(out)
    return result, [dict(zip(('role', 'cohort', 'stage', 'component'), group),
                         missing_parent_trajectories=count)
                    for group, count in sorted(missing.items())]


def save(path, values):
    assert values, 'No actual measurements'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def collection_stages(records):
    result = []
    for role in ('all', 'A', 'B'):
        for stage in sorted({r['stage'] for r in records}):
            subset = [r for r in records if r['stage'] == stage and
                      (role == 'all' or r['role'] == role)]
            if not subset:
                continue
            taken = [r for r in subset if r['takeover_tick'] is not None]
            ticks = [r['takeover_tick'] for r in taken]
            phases = ','.join(map(str, sorted({r['takeover_phase'] for r in taken})))
            result.append(dict(role=role, stage=stage, opportunities=len(subset),
                takeovers=len(taken), expert_rescues=sum(r['expert_rescue'] for r in taken),
                accepted=sum(r['accepted'] for r in subset),
                accepted_legal_ticks=sum(r['legal_expert_ticks'] for r in subset if r['accepted']),
                excluded_expert_ticks=sum(r['legal_expert_ticks'] for r in subset if not r['accepted']),
                takeover_min=min(ticks) if ticks else None,
                takeover_median=statistics.median(ticks) if ticks else None,
                takeover_max=max(ticks) if ticks else None,
                planning_phases=phases))
    return result


def export(round_index):
    public = PUBLIC / f'round{round_index:02d}'
    assert read(public / 'offline/scope.json')['complete']
    current_path = public / 'offline/per_episode.csv'
    parent_path = PUBLIC / ('baseline_offline/per_episode.csv' if round_index == 1
                            else 'round01/offline/per_episode.csv')
    current, parent = rows(current_path), rows(parent_path)
    compared, missing = compare(current, parent)
    save(public / 'offline/shared_parent_comparison.csv', compared)
    # A current student's own-role captured parent predictions on new expert
    # histories used online noise7093. Offline noise7091 is different; keep
    # these descriptive comparisons separate from matched-noise measurements.
    captured_path = public / 'captured_parent_fit/per_episode.csv'
    captured, _ = compare(current, rows(captured_path))
    save(public / 'offline/captured_parent_comparison.csv', captured)
    completion = read(public / 'training_completion.json')['components']
    curves = rows(public / 'training_curves.csv')
    training = []
    for component in completion:
        subset = sorted((r for r in curves if r['model'] == component['model']),
                        key=lambda r: int(r['epoch']))
        selected = next(r for r in subset if int(r['epoch']) == component['selected_epoch'])
        first_eligible = next(r for r in subset if r['eligible'] == 'True')
        assert int(first_eligible['minimum_tick_steps']) >= 50
        train = dict(model=component['model'], first_epoch=int(subset[0]['epoch']),
                     selected_epoch=component['selected_epoch'], last_epoch=int(subset[-1]['epoch']),
                     first_eligible_epoch=int(first_eligible['epoch']))
        for label, row in (('first', subset[0]), ('first_eligible', first_eligible),
                           ('selected', selected), ('last', subset[-1])):
            for measure in ('train', 'validation'):
                train[label + '_' + measure] = float(row[measure])
        start = train['first_eligible_validation']
        train['eligible_to_last_validation_change_percent'] = (
            100 * (train['last_validation'] / start - 1))
        training.append(train)
    save(public / 'training_fit_summary.csv', training)
    evaluated = read(public / 'evaluation/per_case.json')
    failed = []
    for role, cohort in sorted({(r['role'], r['cohort']) for r in evaluated}):
        subset = [r for r in evaluated if r['role'] == role and r['cohort'] == cohort]
        stages = collections.Counter('success' if r['success'] else
            'after_first_subgoal' if r['native_subgoals'] else
            'after_lift' if r['native_lift'] else 'before_lift' for r in subset)
        failed.append(dict(model=role, cohort=cohort, opportunities=len(subset),
            **{stage: stages[stage] for stage in
               ('success', 'before_lift', 'after_lift', 'after_first_subgoal')}))
    save(public / 'evaluation/failure_stages.csv', failed)
    stages = collection_stages(read(public / 'combined_analysis/records.json'))
    save(public / 'collection_stage_comparison.csv', stages)
    write(public / 'quantitative_summary.json', dict(round=round_index,
        shared_parent_comparisons=compared, missing_parent_populations=missing,
        captured_parent_comparisons=captured, training=training, failure_stages=failed,
        collection_stages=stages,
        source_sha256={str(p): sha256(p) for p in (current_path, parent_path, captured_path)},
        scope='Read-only evidence aggregation; exact identity/tick count joins; tick-weighted RMSE; no updates or rollouts',
        captured_parent_limitation='Own-role captured parent only; online noise7093 versus offline7091; descriptive, not exact matched-noise effect',
        training_limitation='train is fixed train2 diagnosis, not all-pool fitting; eligible tail change is descriptive, not convergence proof'))
    return training


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, choices=(1, 2), required=True)
    print(export(parser.parse_args().round))
