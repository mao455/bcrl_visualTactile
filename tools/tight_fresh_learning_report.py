"""Export comparable metrics and actual learning exposure without opening test data."""
import argparse
from collections import Counter
import json
from pathlib import Path
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(path.read_text())


def compact_metric(value):
    return {k: v for k, v in value.items() if k != 'episodes'}


def compact_epoch(row):
    return {k: compact_metric(v) if k in ('train', 'validation', 'correction') else v
            for k, v in row.items()}


def exposure_rows(learning, training_ids, corrections):
    counts = Counter()
    totals = {key: Counter(dict(optimizer_episodes=0, visual_frames=0, anchors_seen=0, target_ticks=0))
              for key in ('original', 'correction')}
    previous = dict(anchors_seen=0, target_ticks=0)
    for expected_update, row in enumerate(learning, 1):
        if row['episode'] not in training_ids or row['updates'] != expected_update:
            raise ValueError('Unexpected episode or noncontiguous optimizer receipt')
        counts[row['episode']] += 1
        bucket = totals['correction' if row['episode'] in corrections else 'original']
        bucket['optimizer_episodes'] += 1
        bucket['visual_frames'] += row['visual_frames']
        for key in previous:
            change = row[key] - previous[key]
            if change <= 0:
                raise ValueError('Nonpositive learning exposure increment')
            bucket[key] += change
            previous[key] = row[key]
    return counts, {k: dict(v) for k, v in totals.items()}


def role_report(root, role):
    import torch
    directory = root / role
    training = directory / 'training'
    data = read(directory / 'state_data/data.json')
    corrections = set(data.get('aggregation', {}).get('new_training_ids', []))
    history = [json.loads(s) for s in (training / 'validation.jsonl').read_text().splitlines()]
    learning = [json.loads(s) for s in (training / 'learning.jsonl').read_text().splitlines()]
    counts, exposure = exposure_rows(learning, set(data['splits']['train']), corrections)
    best = training / 'best.pt'
    saved = torch.load(best, map_location='cpu', weights_only=False)
    selected = next(r for r in history if r['epoch'] == saved['epoch'])
    complete = (training / 'finished.json').exists()
    if complete:
        finished = read(training / 'finished.json')
        for key in ('updates', 'anchors_seen', 'target_ticks'):
            if finished[key] != history[-1][key] or finished[key] != learning[-1][key]:
                raise ValueError('Finished exposure differs from actual learning receipts')
        if finished['checkpoint_sha256'] != sha256(best):
            raise ValueError('Finished selected frontend identity differs')
    result = dict(frontend_complete=complete, checkpoint_sha256=sha256(best),
        selected_epoch=saved['epoch'], selected_updates=saved['updates'],
        selected_validation=selected['validation'], selected_train=selected['train'],
        latest_updates=history[-1]['updates'], latest_epoch=history[-1]['epoch'],
        actual_optimizer_episodes=len(learning), actual_episode_counts=dict(counts),
        actual_exposure_by_source=exposure,
        anchors_seen=history[-1]['anchors_seen'], target_slots=history[-1]['target_ticks'],
        history=[compact_epoch(r) for r in history],
        metric_cohorts=dict(train=data['splits']['train'][:2], validation=data['splits']['validation'],
            correction=sorted(corrections)),
        source_configuration_sha256=sha256(training / 'configuration.json'),
        learning_receipts_sha256=sha256(training / 'learning.jsonl'),
        validation_receipts_sha256=sha256(training / 'validation.jsonl'),
        data_sha256=sha256(directory / 'state_data/data.json'),
        statistics_sha256=sha256(directory / 'state_data/statistics.pt'), test_used=False)
    if (training / 'initial_metrics.json').exists():
        result['initial_metrics'] = read(training / 'initial_metrics.json')
        before = result['initial_metrics']['validation']['selection']
        result['selected_validation_relative_change'] = selected['validation']['selection'] / before - 1
    if 'correction' in history[-1]:
        result['last_epoch_correction_metrics'] = history[-1]['correction']
        result['correction_metric_checkpoint_scope'] = 'Last EMA epoch, not necessarily validation-selected best'
        result['correction_metrics_match_selected'] = history[-1]['epoch'] == saved['epoch']
    rear = directory / 'backend' / role
    if (rear / 'training/finished.json').exists():
        result['backend_finished'] = read(rear / 'training/finished.json')
        chosen = read(rear / 'training/selected.json')
        if chosen['checkpoint_sha256'] != sha256(rear / 'training/best.pt'):
            raise ValueError('Selected backend identity differs')
        result['backend_selected'] = chosen
        rear_history = read(rear / 'training/validation.json')
        result['backend_history'] = rear_history
        result['backend_selected_metrics'] = next(r for r in rear_history if r['update'] == chosen['update'])
        result['backend_actual_validation_ids'] = data['splits']['validation']
        result['backend_legacy_scope_note'] = 'Historical validation4 description is stale; actual fixed validation has six independent families.'
    if (rear / 'offline/result.json').exists():
        offline = read(rear / 'offline/result.json')
        if offline['checkpoint_sha256'] != sha256(rear / 'training/best.pt') or offline['test_used']:
            raise ValueError('Offline identity or cohort differs')
        result['backend_offline'] = {k: offline[k] for k in ('complete', 'checkpoint_sha256', 'scope', 'test_used')}
        result['backend_offline']['metrics'] = [r for r in offline['metrics'] if r['motion'] == 'all']
        result['backend_offline']['source_sha256'] = sha256(rear / 'offline/result.json')
    if (rear / 'nominal/index.json').exists():
        index = read(rear / 'nominal/index.json')
        if index['base_sha256'] != sha256(best):
            raise ValueError('Backend cache was not refreshed for selected DP')
        result['backend_cache'] = {k: v for k, v in index.items() if k != 'episodes'}
        result['backend_cache']['episodes'] = len(index['episodes'])
        result['backend_cache']['index_sha256'] = sha256(rear / 'nominal/index.json')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reports = {role: role_report(a.root, role) for role in ('A', 'B')}
    atomic_json(a.output, dict(roles=reports,
        interpretation='Frontend A/B labels differ: compare within-role before/after. Backend command errors share one original30 affine. Offline errors do not establish autonomous success.',
        test_used=False))


if __name__ == '__main__':
    main()
