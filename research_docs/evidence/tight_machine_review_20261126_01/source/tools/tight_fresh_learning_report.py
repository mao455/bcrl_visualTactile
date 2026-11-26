"""Export comparable metrics and actual learning exposure without opening test data."""
import argparse
from collections import Counter
import json
from pathlib import Path
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from visuotactile.training_receipts import BATCH_ROW_VERSION, iter_visits, learning_counters


def read(path):
    return json.loads(path.read_text())


def compact_metric(value):
    return {k: v for k, v in value.items() if k != 'episodes'}


def compact_epoch(row):
    return {k: compact_metric(v) if k in ('train', 'validation', 'correction') else v
            for k, v in row.items()}


def exposure_rows(learning, training_ids, corrections, *, source_groups=None):
    counts = Counter()
    if source_groups is None:
        source_groups = {i: 'correction' if i in corrections else 'original' for i in training_ids}
        keys = ('original', 'correction')
    else:
        keys = sorted(set(source_groups.values()))
    if set(source_groups) != set(training_ids):
        raise ValueError('Learning source groups must cover the complete training population')
    batched = any(row.get('version') == BATCH_ROW_VERSION for row in learning)
    if batched and not all(row.get('version') == BATCH_ROW_VERSION for row in learning):
        raise ValueError('Mixed single-trajectory and B2 optimizer receipts')
    visit_key = 'episode_visits' if batched else 'optimizer_episodes'
    totals = {key: Counter({visit_key: 0, 'visual_frames': 0, 'anchors_seen': 0, 'target_ticks': 0})
              for key in keys}
    for visit in iter_visits(learning):
        if visit['episode'] not in training_ids:
            raise ValueError('Unexpected episode or noncontiguous optimizer receipt')
        counts[visit['episode']] += 1
        bucket = totals[source_groups[visit['episode']]]
        bucket[visit_key] += 1
        bucket['visual_frames'] += visit['visual_frames']
        bucket['anchors_seen'] += visit['anchors']
        bucket['target_ticks'] += visit['target_ticks']
    return counts, {k: dict(v) for k, v in totals.items()}


def sampled_visit_coverage(rows, profile, *, limit=None):
    """Read actual selected times; no grid inference or image/numeric payload I/O."""
    import hashlib
    from visuotactile.fresh_visit_sampling import validate_profile, Coverage
    profile=validate_profile(profile)
    selected_rows=rows if limit is None else rows[:limit]
    has_receipts=any('anchor_sampling' in row or any('anchor_sampling' in v for v in row.get('visits',[]))
        for row in selected_rows)
    if profile is None:
        if has_receipts:raise ValueError('Random anchor receipts require their registered sampling profile')
        return None
    coverage=Coverage()
    for visit in iter_visits(rows,limit=limit):
        receipt=visit.get('anchor_sampling');ticks=receipt.get('selected_anchors') if isinstance(receipt,dict) else None
        if (not isinstance(ticks,list) or not ticks or len(ticks)!=visit['anchors'] or
                any(type(t) is not int or not 0<=t<visit['visual_frames'] for t in ticks) or
                ticks!=sorted(set(ticks)) or receipt.get('selected_count')!=len(ticks) or
                receipt.get('identity')!=visit['episode'] or receipt.get('epoch')!=visit.get('epoch') or
                receipt.get('visit_index')!=visit.get('episode_index') or
                any(type(receipt.get(k)) is not int or receipt[k]<1 for k in ('epoch','visit_index')) or
                receipt.get('non_native_grid_count')!=sum(t%12!=0 for t in ticks)):
            raise ValueError('Actual random anchor receipt differs from its training visit')
        context=dict(version=profile['anchors']['version'],seed=profile['anchors']['seed'],
            identity=visit['episode'],epoch=receipt['epoch'],visit_index=receipt['visit_index'])
        digest=hashlib.sha256(json.dumps(context,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (receipt.get('version')!=profile['anchors']['version'] or receipt.get('seed_context_sha256')!=digest or
                receipt.get('derived_seed')!=int(digest[:16],16)%(2**63) or
                receipt.get('selected_anchors_sha256')!=hashlib.sha256(json.dumps(ticks,separators=(',',':')).encode()).hexdigest()):
            raise ValueError('Random anchor visit seed or selected-time digest differs')
        coverage.add(visit['episode'],receipt,visit.get('geometry_sampling'))
    return coverage


def batch_checkpoint_exposure(saved, rows):
    """Reconcile complete B2 deck visits independently of Adam/EMA updates."""
    from visuotactile.trajectory_training import validate_profile, validate_exposure
    config = saved['configuration']
    profile = validate_profile(config.get('full_trajectory_batch'))
    if profile is None:
        if any(row.get('version') == BATCH_ROW_VERSION for row in rows):
            raise ValueError('B2 learning rows lack an explicit checkpoint profile')
        return None
    if not all(row.get('version') == BATCH_ROW_VERSION and
            row.get('full_trajectory_batch') == profile for row in rows):
        raise ValueError('B2 checkpoint and actual learning profiles differ')
    multiplicities = config['episode_multiplicities']
    if (set(multiplicities) != set(saved['train_ids']) or
            any(type(n) is not int or n < 1 for n in multiplicities.values()) or
            config['episodes_per_epoch'] != sum(multiplicities.values()) or
            config['optimizer_updates_per_epoch'] != (sum(multiplicities.values())+1)//2):
        raise ValueError('B2 checkpoint registered visit deck differs')
    validate_exposure(saved.get('sampling_exposure'), multiplicities,
        completed_epochs=saved['epoch'], updates=saved['updates'], visits_seen=saved.get('visits_seen'))
    counters = learning_counters(rows, limit=saved['updates'])
    if any(saved[key] != counters[key] for key in counters):
        raise ValueError('B2 checkpoint counters differ from actual learning receipts')
    counts = Counter(v['episode'] for v in iter_visits(rows, limit=saved['updates']))
    if dict(counts) != saved['sampling_exposure']:
        raise ValueError('B2 checkpoint per-episode visits differ from actual learning receipts')
    return counters


def learning_sources(data):
    training = set(data['splits']['train'])
    current = set(data.get('aggregation', {}).get('new_training_ids', []))
    original = set(data.get('normalization_train_ids', training - current))
    if not original <= training or not current <= training or original & current:
        raise ValueError('Original/current correction populations overlap or leave training')
    corrections = training - original
    groups = {i: ('original' if i in original else 'current_correction' if i in current
                  else 'historical_correction') for i in training}
    return corrections, current, groups


def role_report(root, role):
    import torch
    directory = root / role
    training = directory / 'training'
    data = read(directory / 'state_data/data.json')
    corrections, current_corrections, source_groups = learning_sources(data)
    history = [json.loads(s) for s in (training / 'validation.jsonl').read_text().splitlines()]
    learning = [json.loads(s) for s in (training / 'learning.jsonl').read_text().splitlines()]
    counts, exposure = exposure_rows(learning, set(data['splits']['train']), corrections)
    _, detailed_exposure = exposure_rows(learning, set(data['splits']['train']), corrections,
        source_groups=source_groups)
    best = training / 'best.pt'
    saved = torch.load(best, map_location='cpu', weights_only=False)
    sampling=saved['configuration'].get('training_sampling')
    selected_coverage=sampled_visit_coverage(learning,sampling,limit=saved['updates'])
    latest_coverage=sampled_visit_coverage(learning,sampling)
    if selected_coverage is not None and selected_coverage.state_dict()!=saved.get('visit_sampling_coverage'):
        raise ValueError('Selected checkpoint random sampling coverage differs from actual visits')
    selected_batch_exposure = batch_checkpoint_exposure(saved, learning)
    selected = next(r for r in history if r['epoch'] == saved['epoch'])
    if selected_batch_exposure is not None and any(selected.get(key) != value
            for key, value in selected_batch_exposure.items()):
        raise ValueError('Selected B2 validation row differs from actual checkpoint exposure')
    complete = (training / 'finished.json').exists()
    if complete:
        finished = read(training / 'finished.json')
        for key in ('updates', 'anchors_seen', 'target_ticks'):
            if finished[key] != history[-1][key] or finished[key] != learning[-1][key]:
                raise ValueError('Finished exposure differs from actual learning receipts')
        if finished['checkpoint_sha256'] != sha256(best):
            raise ValueError('Finished selected frontend identity differs')
        if latest_coverage is not None and (finished.get('training_sampling')!=sampling or
                finished.get('visit_sampling_coverage')!=latest_coverage.summary()):
            raise ValueError('Finished random sampling coverage differs from actual visits')
        if selected_batch_exposure is not None:
            from visuotactile.trajectory_training import validate_exposure
            actual = learning_counters(learning)
            if (finished.get('full_trajectory_batch') != saved['configuration']['full_trajectory_batch'] or
                    finished.get('visits_seen') != actual['visits_seen'] or
                    history[-1].get('visits_seen') != actual['visits_seen'] or
                    finished.get('sampling_exposure') != dict(counts)):
                raise ValueError('Finished B2 visit exposure differs from actual learning receipts')
            validate_exposure(dict(counts), saved['configuration']['episode_multiplicities'],
                completed_epochs=finished['epochs'], updates=actual['updates'], visits_seen=actual['visits_seen'])
    result = dict(frontend_complete=complete, checkpoint_sha256=sha256(best),
        selected_epoch=saved['epoch'], selected_updates=saved['updates'],
        selected_validation=selected['validation'], selected_train=selected['train'],
        latest_updates=history[-1]['updates'], latest_epoch=history[-1]['epoch'],
        actual_optimizer_episodes=len(learning), actual_episode_counts=dict(counts),
        actual_exposure_by_source=exposure,
        actual_exposure_by_correction_age=detailed_exposure,
        anchors_seen=history[-1]['anchors_seen'], target_slots=history[-1]['target_ticks'],
        history=[compact_epoch(r) for r in history],
        metric_cohorts=dict(train=data['splits']['train'][:2], validation=data['splits']['validation'],
            correction=sorted(current_corrections)),
        source_populations={key: sorted(i for i, group in source_groups.items() if group == key)
            for key in sorted(set(source_groups.values()))},
        source_configuration_sha256=sha256(training / 'configuration.json'),
        learning_receipts_sha256=sha256(training / 'learning.jsonl'),
        validation_receipts_sha256=sha256(training / 'validation.jsonl'),
        data_sha256=sha256(directory / 'state_data/data.json'),
        statistics_sha256=sha256(directory / 'state_data/statistics.pt'), test_used=False)
    if selected_batch_exposure is not None:
        result.pop('actual_optimizer_episodes')
        result.update(full_trajectory_batch=saved['configuration']['full_trajectory_batch'],
            actual_optimizer_updates=len(learning), actual_episode_visits=sum(counts.values()),
            selected_episode_visits=selected_batch_exposure['visits_seen'],
            selected_sampling_exposure=saved['sampling_exposure'],
            exposure_semantics='One optimizer/EMA update per one/two full-trajectory visits; source buckets count visits and observed target slots, not separable optimizer steps or gradient weight.')
    if selected_coverage is not None:
        result.update(training_sampling=sampling,selected_visit_sampling_coverage=selected_coverage.summary(),
            latest_visit_sampling_coverage=latest_coverage.summary(),
            sampling_semantics='Distinct episode/tick starts and geometry configurations from actual visit receipts; target slots use observed increments, never original-grid population times visits.')
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
        last_rear = torch.load(rear / 'training/last.pt', map_location='cpu', weights_only=False)
        numeric = torch.load(directory / 'state_data/numeric.pt', map_location='cpu', weights_only=False)['records']
        rear_exposure = {key: dict(episode_draws=0, supervised_ticks=0, history_ticks=0) for key in ('original', 'correction')}
        rear_detailed = {key: dict(episode_draws=0, supervised_ticks=0, history_ticks=0)
            for key in sorted(set(source_groups.values()))}
        if set(last_rear['episode_counts']) != set(data['splits']['train']):
            raise ValueError('Backend episode population differs')
        for identity, count in last_rear['episode_counts'].items():
            record = numeric[identity]
            ticks = len(record['time'])
            supervised = int(record['supervision_mask'].sum()) if 'supervision_mask' in record else ticks
            for bucket in (rear_exposure['correction' if identity in corrections else 'original'],
                    rear_detailed[source_groups[identity]]):
                bucket['episode_draws'] += count
                bucket['supervised_ticks'] += count * supervised
                bucket['history_ticks'] += count * ticks
        if sum(v['supervised_ticks'] for v in rear_exposure.values()) != result['backend_finished']['supervised_ticks']:
            raise ValueError('Backend actual episode use does not match legal tick exposure')
        result['backend_actual_episode_counts'] = last_rear['episode_counts']
        result['backend_actual_exposure_by_source'] = rear_exposure
        result['backend_actual_exposure_by_correction_age'] = rear_detailed
        result['backend_actual_validation_ids'] = last_rear['validation_ids']
        if result['backend_actual_validation_ids'] != data['splits']['validation']:
            raise ValueError('Actual selected validation population differs')
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
