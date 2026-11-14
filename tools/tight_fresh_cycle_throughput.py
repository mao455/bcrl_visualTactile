"""Read completed cycle JSON receipts on CPU; never load model, image or trajectory payloads.

Counts describe actual work, not causal speedups. Missing stages remain incomplete.
All rates name their wall denominator; concurrent worker durations are never added
to obtain pipeline elapsed time.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


VERSION = 'fresh_cycle_throughput_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), 'Invalid ' + name)
    return value


def count(value, name):
    require(type(value) is int and value >= 0, 'Invalid count: ' + name)
    return value


def union_seconds(intervals):
    """Elapsed coverage of intervals, including nested/overlapping workers once."""
    ordered = []
    for start, end in intervals:
        number(start, 'interval start'); number(end, 'interval end')
        require(end >= start, 'Reversed interval')
        ordered.append((start, end))
    end = None; total = 0.
    for start, stop in sorted(ordered):
        if end is None or start > end:
            total += stop - start
        else:
            total += max(0., stop - end)
        end = max(stop, end) if end is not None else stop
    return total


class Sources:
    def __init__(self):
        self.bindings = {}

    def read(self, path, *, lines=False):
        path = Path(path).resolve(); raw = path.read_bytes()
        self.bindings[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        if lines:
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        return json.loads(raw)


def stage_receipts(directory, sources):
    finished = directory / 'finished.json'
    final = sources.read(finished) if finished.exists() else None
    if final is not None:
        records = final.get('stages', [])
    else:
        records = [sources.read(path) for path in sorted((directory / 'ledger').glob('*.json'))
                   if not path.name.endswith('_worker.json')]
        records = [row for row in records if all(k in row for k in ('stage', 'returncode', 'seconds'))]
    names = set(); output = []
    for row in records:
        require(row['stage'] not in names, 'Duplicate stage receipt'); names.add(row['stage'])
        start = number(row['started'], 'stage start')
        seconds = number(row['seconds'], 'stage seconds', positive=True)
        complete = row['returncode'] == 0 and row.get('stop_reason') is None
        if complete:
            path = Path(row['completion']).resolve()
            sources.read(path)
            require(sources.bindings[str(path)]['sha256'] == row['completion_sha256'],
                    'Completion receipt changed: ' + row['stage'])
        output.append(dict(stage=row['stage'], start=start, end=start + seconds,
                           seconds=seconds, complete=complete, returncode=row['returncode']))
    complete = final is not None and final.get('software_complete') is True
    if complete:
        required = {'aggregate', 'visual_cache', 'frontend_A', 'frontend_B', 'development_prepare', 'development'}
        required |= {'backend_' + stage + '_' + role for stage in ('prepare', 'cache', 'train', 'offline') for role in 'AB'}
        require(required <= names and all(row['complete'] for row in output), 'Incomplete finished stage ledger')
        number(final['utc'], 'cycle completion UTC')
        require(final['utc'] + .01 >= max(row['end'] for row in output), 'Cycle ended before its stages')
    return output, final, complete


def interval_summary(rows):
    intervals = [(row['start'], row['end']) for row in rows]
    worker_seconds = sum(row['seconds'] for row in rows)
    covered = union_seconds(intervals)
    return dict(stage_worker_seconds=worker_seconds, stage_interval_union_seconds=covered,
                overlapping_worker_seconds=max(0., worker_seconds - covered))


def manifest_counts(data):
    splits = data['splits']; entries = data['entries']
    seen = set(); result = {}
    for split, ids in splits.items():
        require(len(set(ids)) == len(ids) and not seen.intersection(ids), 'Overlapping or duplicate split IDs')
        seen.update(ids)
        require(set(ids) <= set(entries), 'Missing manifest entries')
        result[split] = dict(trajectories=len(ids), families=len({entries[i]['family'] for i in ids}),
                            full_history_ticks=sum(count(entries[i]['length'], 'episode length') for i in ids))
    return result


def population_change(cycle, registration, sources, accepted_ids):
    populations = {}; manifests = {}
    for role in 'AB':
        old = sources.read(Path(registration['source_root']) / role / 'state_data/data.json')
        new = sources.read(cycle / role / 'state_data/data.json')
        before, after = manifest_counts(old), manifest_counts(new)
        added = set(new['splits']['train']) - set(old['splits']['train'])
        require(set(old['splits']['train']) <= set(new['splits']['train']), 'Old training episodes removed')
        require(added == set(new['aggregation']['new_training_ids']) == set(accepted_ids),
                'New training IDs differ from accepted collection entries')
        for identity in set(old['splits']['train']) | set(old['splits'].get('validation', [])):
            require(all(old['entries'][identity].get(k) == new['entries'][identity].get(k)
                        for k in ('family', 'length', 'source_sha256')), 'Old population binding changed')
        require(old['splits'].get('validation') == new['splits'].get('validation'), 'Validation population changed')
        require(not new['splits'].get('test'), 'This cycle report expects no test population')
        populations[role] = dict(before=before, after=after, added_training_trajectories=len(added),
            added_training_families=after['train']['families'] - before['train']['families'], validation_unchanged=True)
        manifests[role] = new
    require(populations['A'] == populations['B'], 'A/B data population differs')
    require(manifests['A']['splits'] == manifests['B']['splits'], 'A/B split IDs differ')
    for identity in manifests['A']['entries']:
        require(all(manifests['A']['entries'][identity].get(k) == manifests['B']['entries'][identity].get(k)
                    for k in ('family', 'length', 'source_sha256')), 'A/B episode provenance differs')
    return populations['A'], manifests


def collection_summary(root, sources):
    root = Path(root).resolve(); result_path = root / 'result.json'; owner_path = root / 'owner_finished.json'
    plan = sources.read(root / 'plan.json')
    require(plan.get('kind') == 'takeover' and plan.get('test_used') is False, 'Non-test collection required')
    result = sources.read(result_path) if result_path.exists() else None
    owner = sources.read(owner_path) if owner_path.exists() else None
    complete = bool(result and owner and result.get('software_complete') is True
        and result.get('worker_error') is None and owner.get('software_complete') is True and owner.get('returncode') == 0)
    output = dict(root=str(root), complete=complete, registered_attempts=len(plan['cases']),
                  registered_families=len({case['spec']['family'] for case in plan['cases']}),
                  batch=plan.get('batch'), backend=plan.get('execution_backend'))
    if not complete:
        output.update(status='incomplete', completed_attempts=result.get('cases') if result else None,
                      throughput=None, interval=None)
        return output, None
    from tools.tight_batch_throughput_report import population
    rows = sources.read(root / 'results.json'); entries = sources.read(root / 'correction_entries.json')
    counts = population(root, plan, rows, entries)
    # Existing population's triggered_attempts includes scheduled fixed takeovers.
    # Rename it here; do not manufacture a causal trigger rate for mixed schedules.
    counts['actual_takeovers'] = counts.pop('triggered_attempts')
    require(result['cases'] == counts['attempts'], 'Collection completion count differs')
    started = sources.read(root / 'owner_started.json')
    plan_sha = sources.bindings[str((root / 'plan.json').resolve())]['sha256']
    require(started.get('plan_sha256') == owner.get('plan_sha256') == plan_sha, 'Collection owner plan binding differs')
    seconds = number(owner['seconds'], 'collection owner seconds', positive=True)
    start = number(started['started'], 'collection start'); end = number(owner['ended'], 'collection end')
    require(abs(end - start - seconds) < .01, 'Collection owner interval differs')
    counts['accepted_families'] = len({entries[i]['family'] for i in entries})
    counts['attempted_families'] = output['registered_families']
    output.update(counts=counts, owner_seconds=seconds,
        worker_elapsed_seconds=number(result['elapsed_seconds'], 'collection worker seconds', positive=True),
        interval=dict(start=start, end=end), accepted_ids=sorted(entries),
        throughput=dict(attempts_per_hour=counts['attempts'] * 3600 / seconds,
            accepted_trajectories_per_hour=counts['accepted_trajectories'] * 3600 / seconds,
            accepted_valid_expert_ticks_per_second=counts['accepted_valid_expert_ticks'] / seconds,
            aggregate_control_ticks_per_second=counts['all_control_ticks'] / seconds),
        denominator='collection owner start through completion, including startup and output validation')
    return output, entries


def frontend_summary(directory, sources):
    final = sources.read(directory / 'training/finished.json')
    require(final.get('complete') is True, 'Frontend not complete')
    rows = sources.read(directory / 'training/learning.jsonl', lines=True)
    validation = sources.read(directory / 'training/validation.jsonl', lines=True)
    require(rows and validation, 'Missing frontend learning records')
    for field in ('updates', 'anchors_seen', 'target_ticks'):
        values = [count(row[field], field) for row in rows]
        require(all(b >= a for a, b in zip(values, values[1:])), 'Non-monotonic training counter: ' + field)
        require(values[-1] == final[field] == validation[-1][field], 'Frontend final counter differs: ' + field)
    require([row['updates'] for row in rows] == list(range(1, final['updates'] + 1)),
            'Missing, duplicated or reordered optimizer updates')
    require([row['epoch'] for row in validation] == list(range(1, final['epochs'] + 1)), 'Incomplete epoch log')
    batch_profile = final.get('full_trajectory_batch')
    visit_count = len(rows)
    if batch_profile is not None:
        from visuotactile.trajectory_training import validate_profile, validate_exposure
        from visuotactile.training_receipts import learning_counters, iter_visits
        from collections import Counter
        validate_profile(batch_profile)
        require(all(row.get('full_trajectory_batch') == batch_profile for row in rows + validation),
                'B2 learning or validation profile differs')
        totals = learning_counters(rows)
        visit_count = totals['visits_seen']
        require(visit_count == final['visits_seen'], 'B2 final visit count differs')
        require(final['ema_updates'] == final['updates'], 'B2 EMA/update count differs')
        validate_exposure(final['sampling_exposure'], final['episode_multiplicities'],
            completed_epochs=final['epochs'], updates=final['updates'], visits_seen=visit_count)
        require(dict(Counter(v['episode'] for v in iter_visits(rows))) == final['sampling_exposure'],
                'B2 per-episode visit exposure differs')
        for row in validation:
            actual = learning_counters(rows, limit=row['updates'])
            require(all(actual[key] == row[key] for key in actual), 'B2 epoch exposure differs')
    else:
        require(not any('visits' in row or 'full_trajectory_batch' in row for row in rows),
                'B2 learning receipts require explicit finished profile')
    seconds = number(final['wall_seconds'], 'frontend wall seconds', positive=True)
    epochs = []; io_totals = {}; prefetch = {}
    for row in validation:
        io = row.get('training_epoch_io')
        if io is None:
            continue
        elapsed = number(io['wall_seconds'], 'epoch training seconds', positive=True)
        start, end = io['started_utc'], io['ended_utc']
        require(abs(end - start - elapsed) < .01, 'Training epoch clock differs')
        for key, value in io.get('process_io', {}).items():
            io_totals[key] = io_totals.get(key, 0) + count(value, 'process I/O ' + key)
        for key, value in (io.get('prefetch') or {}).items():
            if key in ('visits', 'clone_bytes', 'consumer_wait_s', 'load_s', 'budget_wait_s', 'mmap_fallbacks'):
                prefetch[key] = prefetch.get(key, 0) + number(value, 'prefetch ' + key)
            elif key == 'peak_owned_clone_bytes':
                prefetch[key] = max(prefetch.get(key, 0), count(value, key))
        epochs.append(dict(epoch=row['epoch'], start=start, end=end, seconds=elapsed,
            process_read_bytes=io.get('process_io', {}).get('read_bytes'), prefetch=io.get('prefetch'),
            paired_episode_io=io.get('paired_episode_io'),
            post_training_to_report_seconds=max(0., row['utc'] - end)))
    training = sum(row['seconds'] for row in epochs) if len(epochs) == final['epochs'] else None
    require(training is None or training <= seconds + .01, 'Epoch timings exceed frontend wall')
    offline = []
    for row in validation:
        for split in ('train', 'validation'):
            metric = row[split]
            if 'offline_planning' in metric:
                offline.append(dict(epoch=row['epoch'], split=split, **metric['offline_planning'],
                    encoding_seconds=sum(v['offline_planning']['encoding_seconds'] for v in metric['episodes']),
                    planning_seconds=sum(v['offline_planning']['planning_seconds'] for v in metric['episodes'])))
    counters = dict(optimizer_updates=final['updates'], anchors=final['anchors_seen'],
        legal_horizon_target_slots=final['target_ticks'],
        full_history_frame_visits=sum(count(row['visual_frames'], 'visual frame visits') for row in rows),
        episode_visits=visit_count)
    result = dict(complete=True, epochs=final['epochs'], counters=counters, wall_seconds=seconds,
        rates_per_worker_wall_second={key: value / seconds for key, value in counters.items()},
        training_epoch_seconds=training,
        other_frontend_seconds=seconds - training if training is not None else None,
        other_frontend_scope='Initial/validation/final-correction metrics, checkpointing, setup and waits; not pure DDIM.',
        epoch_io=epochs, process_io_totals=io_totals, prefetch_totals=prefetch,
        process_io_epoch_coverage=dict(expected=final['epochs'], observed=len(epochs),
            read_bytes_observed=sum(row['process_read_bytes'] is not None for row in epochs)),
        prefetch_scope='Worker/background timing attribution; overlapping load/wait/clone components are not additive.',
        paired_episode_io=final.get('paired_episode_io'), offline_planning=final.get('offline_planning'),
        timed_metrics=offline or None,
        metric_timing_scope='Only explicit train/validation timings when logged; excludes initial and final-correction metrics.',
        dose_scope='Completed training, not only selected checkpoint. Target slots include horizon overlap and repeat visits; not unique action ticks.')
    if batch_profile is not None:
        result.update(full_trajectory_batch=batch_profile, ema_updates=final['ema_updates'],
            optimization_scope='One optimizer/EMA update per actual B2 batch or singleton; equal visits are not equal optimizer budgets or equivalent sequential AdamW updates.')
    return result


def backend_summary(directory, role, sources):
    cache = sources.read(directory / 'backend/cache_complete.json')
    index = sources.read(directory / 'backend' / role / 'nominal/index.json')
    require(cache.get('complete') is True and cache['episodes'] == len(index['episodes']), 'Backend cache incomplete')
    cache_seconds = number(cache['wall_s'], 'backend cache seconds', positive=True)
    cache_counts = dict(episodes=len(index['episodes']), native_history_ticks=sum(count(v['ticks'], 'cache ticks') for v in index['episodes'].values()),
                        plans=sum(count(v['plans'], 'cache plans') for v in index['episodes'].values()))
    trained = sources.read(directory / 'backend' / role / 'training/finished.json')
    require(trained.get('target_reached') is True and trained.get('run_kind') == 'formal', 'Backend training not complete')
    seconds = number(trained['wall_s'], 'rear wall seconds', positive=True)
    training = number(trained['training_wall_s'], 'rear training seconds', positive=True)
    counts = {key: count(trained[key], key) for key in ('new_optimizer_updates', 'supervised_ticks', 'control_ticks',
        'control_tick_population', 'supervised_tick_population', 'train_episode_count')}
    return dict(cache=dict(complete=True, counters=cache_counts, wall_seconds=cache_seconds,
            rates_per_worker_wall_second={key: value / cache_seconds for key, value in cache_counts.items()},
            planning_profile=index.get('cache_offline_planning_profile'), scope='Train plus validation dependency cache; no parameter updates.'),
        training=dict(complete=True, counters=counts, wall_seconds=seconds, training_wall_seconds=training,
            setup_seconds=trained.get('setup_wall_s'), control_tick_epochs=trained['control_tick_epochs'],
            legal_expert_ticks_per_worker_wall_second=counts['supervised_ticks'] / seconds,
            legal_expert_ticks_per_training_wall_second=counts['supervised_ticks'] / training,
            scope=trained.get('timing_scope')))


def development_summary(root, sources):
    final = sources.read(root / 'result.json'); plan = sources.read(root / 'plan.json')
    rows = sources.read(root / 'results.json'); waves = sources.read(root / 'wave_profiles.json')
    require(final.get('software_complete') is True and final.get('worker_error') is None, 'Development incomplete')
    cases = {case['name']: case for case in plan['cases']}
    require(len(cases) == len(rows) == final['cases'] and {row['case'] for row in rows} == set(cases), 'Development population differs')
    require(plan.get('kind') == 'autonomous' and plan.get('test_used') is False, 'Registered development autonomy required')
    per_role = {}
    for row in rows:
        value = row['result']; role = row['role']
        require(value.get('software_complete') is True and role == cases[row['case']]['role'], 'Incomplete development lane')
        require(not value.get('censored') and not value.get('diagnostic_only', False), 'Diagnostic/censored lane cannot count as complete autonomy')
        steps = value.get('steps', 0)
        require('steps' in value or not value.get('legal'), 'Legal development lane lacks its clock count')
        count(steps, 'development steps')
        counters = per_role.setdefault(role, dict(attempts=0, legal_attempts=0, successes=0, aggregate_control_ticks=0))
        counters['attempts'] += 1; counters['legal_attempts'] += bool(value.get('legal'))
        counters['successes'] += bool(value.get('success') and value.get('legal'))
        counters['aggregate_control_ticks'] += steps
    ticks = sum(row['aggregate_control_ticks'] for row in per_role.values())
    require(sum(w['control_ticks'] for w in waves) == ticks, 'Development wave clock counts differ')
    seconds = number(final['elapsed_seconds'], 'development worker seconds', positive=True)
    return dict(complete=True, per_role=per_role, aggregate_control_ticks=ticks,
        families=len({case['spec']['family'] for case in plan['cases']}), worker_elapsed_seconds=seconds,
        aggregate_control_ticks_per_worker_second=ticks / seconds,
        legal_evaluations_per_hour=sum(v['legal_attempts'] for v in per_role.values()) * 3600 / seconds,
        scope='Complete repeated development cohort, not new test generalization; invalid starts are separate from legal failures.',
        waves=[{k: w[k] for k in ('role', 'wave', 'seconds', 'control_ticks', 'controller_s', 'environment_s') if k in w} for w in waves])


def visual_storage(cycle, manifests, sources):
    unique = {}; logical_bytes = 0; missing = []; outputs = {}; shared = {}
    for role in 'AB':
        index = sources.read(cycle / role / 'cache_index.json')
        require(index.get('complete') is True, 'Frozen visual cache incomplete')
        require(index['data_sha256'] == sources.bindings[str((cycle / role / 'state_data/data.json').resolve())]['sha256'],
                'Frozen visual index belongs to another data manifest')
        ids = set().union(*(set(v) for v in manifests[role]['splits'].values()))
        require(ids <= set(index['entries']), 'Missing frozen cache entries')
        frames = 0; role_bytes = 0
        for identity in sorted(ids):
            value = index['entries'][identity]; path = Path(value['path']); frames += count(value['frames'], 'cached frames')
            require(value['source_sha256'] == manifests[role]['entries'][identity]['source_sha256'], 'Cached raw source differs')
            if not path.exists():
                missing.append(str(path)); continue
            info = path.stat(); inode = (info.st_dev, info.st_ino)
            logical_bytes += info.st_size; role_bytes += info.st_size; unique[inode] = info.st_size
            if identity in shared:
                require(shared[identity] == (inode, value['sha256']), 'A/B frozen inputs are not shared identical files')
            shared[identity] = (inode, value['sha256'])
        outputs[role] = dict(episodes=len(ids), frames=frames, logical_bytes=role_bytes)
    return dict(per_role=outputs, combined_logical_bytes=logical_bytes,
        unique_file_bytes=sum(unique.values()), unique_files=len(unique), missing_paths=sorted(set(missing)),
        measurement='Small JSON indexes and filesystem stat only; no payload reread or content revalidation.',
        cache_temperature='Unknown. Existing cache files and prior execution do not measure resident OS page-cache warmth.')


def summarize(cycle_root, *, collection_root=None, stages_root=None, outer_registration=None):
    cycle = Path(cycle_root).resolve(); sources = Sources()
    stages = Path(stages_root).resolve() if stages_root else cycle.parent / (cycle.name + '_stages')
    outer = sources.read(outer_registration) if outer_registration else None
    if outer:
        require(Path(outer['output']).resolve() == cycle, 'Outer registration output differs')
        if collection_root is not None:
            require(Path(collection_root).resolve() == Path(outer['collection']).resolve(), 'Outer collection binding differs')
        collection_root = outer['collection']
    path = stages / 'registration.json'; registration = sources.read(path) if path.exists() else None
    if registration:
        require(Path(registration['output']).resolve() == cycle and registration.get('test_used') is False, 'Cycle registration differs')
        if collection_root is not None:
            require(Path(collection_root).resolve() == Path(registration['collection']).resolve(), 'Cycle collection binding differs')
        collection_root = registration['collection']
        if outer:
            require(Path(outer['data_source']).resolve() == Path(registration['source_root']).resolve(), 'Outer source pool differs')
    require(collection_root is not None, 'Need collection root or actual cycle/outer registration')
    collection, entries = collection_summary(collection_root, sources)
    if outer:
        plan_path = str((Path(collection_root) / 'plan.json').resolve())
        require(sources.bindings[plan_path]['sha256'] == outer['collection_plan_sha256'], 'Outer collection plan changed')
    receipts, final, ledger_complete = stage_receipts(stages, sources)
    completed = {row['stage']: row for row in receipts if row['complete']}
    output = dict(version=VERSION, created_utc=time.time(), cycle_root=str(cycle), stages_root=str(stages),
        generator=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        complete=False, collection=collection, timeline=dict(stages=receipts, **interval_summary(receipts)),
        configuration={key: registration.get(key, False if key in ('paired_episode_io', 'parallel_backends', 'payload_audit_memo') else None)
            for key in ('source_commit', 'parallel_frontends', 'parallel_backends', 'paired_episode_io', 'payload_audit_memo',
                'frontend_offline_planning', 'backend_offline_planning', 'frontend_epochs', 'backend_epochs',
                'frontend_latest_repeat', 'backend_latest_repeat', 'episode_io', 'evaluation_lanes', 'vision_microbatch',
                'execution_backend', 'reuse_visual_index')} if registration else None,
        registered_future_command=outer.get('cycle_command') if outer and registration is None else None,
        limitations=['Multiple implementation settings, data volume, learning difficulty and cache temperature can differ; rates are descriptive, not causal speedup evidence.',
            'Process read_bytes is per-process kernel accounting; shared I/O attribution cannot be equated with independent physical disk traffic.',
            'An anchor, overlapping legal horizon target slot, full-history observation and legal rear action tick are distinct units.',
            'The collection-to-evaluation envelope may include other registered autonomous/gate work and human review; unclassified gaps are not scheduling losses.',
            'JSON receipt and source bindings are verified here; model, DINO, image and trajectory payloads are not read.'])
    if registration and 'full_trajectory_batch' in registration:
        from visuotactile.trajectory_training import validate_profile
        require(validate_profile(registration['full_trajectory_batch']) is not None, 'Missing B2 profile')
        output['configuration']['full_trajectory_batch'] = registration['full_trajectory_batch']
        output['version'] = 'fresh_cycle_throughput_batched_training_v1'
    if registration and entries is not None and 'aggregate' in completed:
        correction_path = str((Path(collection_root) / 'correction_entries.json').resolve())
        require(sources.bindings[correction_path]['sha256'] == registration['collection_entries_sha256'], 'Accepted collection binding changed')
        populations, manifests = population_change(cycle, registration, sources, entries)
        output['population'] = populations
        if 'visual_cache' in completed:
            output['frozen_visual_storage'] = visual_storage(cycle, manifests, sources)
    output['frontends'] = {role: frontend_summary(cycle / role, sources) for role in 'AB' if 'frontend_' + role in completed}
    output['backends'] = {role: backend_summary(cycle / role, role, sources) for role in 'AB'
                          if {'backend_cache_' + role, 'backend_train_' + role} <= set(completed)}
    if 'development' in completed:
        output['development'] = development_summary(cycle / 'development', sources)
    for label, prefix in (('frontends', 'frontend_'), ('backend_cache', 'backend_cache_'), ('backend_training', 'backend_train_')):
        selected = [r for r in receipts if r['stage'].startswith(prefix)]
        output['timeline'][label] = interval_summary(selected)
    if len(output['frontends']) == 2:
        merged = {key: sum(v['counters'][key] for v in output['frontends'].values())
                  for key in output['frontends']['A']['counters']}
        seconds = output['timeline']['frontends']['stage_interval_union_seconds']
        output['frontend_aggregate'] = dict(counters=merged, stage_interval_union_seconds=seconds,
            rates_per_stage_union_second={key: value / seconds for key, value in merged.items()},
            training_epoch_process_read_bytes=(sum(v['process_io_totals']['read_bytes'] for v in output['frontends'].values())
                if all(v['process_io_epoch_coverage']['read_bytes_observed'] == v['epochs'] for v in output['frontends'].values()) else None),
            observed_training_epoch_process_read_bytes=(sum(v['process_io_totals'].get('read_bytes', 0) for v in output['frontends'].values())
                if any(v['process_io_epoch_coverage']['read_bytes_observed'] for v in output['frontends'].values()) else None))
        epoch_intervals = [(epoch['start'], epoch['end']) for worker in output['frontends'].values() for epoch in worker['epoch_io']]
        if all(worker['training_epoch_seconds'] is not None for worker in output['frontends'].values()):
            covered = union_seconds(epoch_intervals)
            output['frontend_aggregate'].update(training_epoch_interval_union_seconds=covered,
                rates_per_training_epoch_union_second={key: value / covered for key, value in merged.items()})
    if len(output['backends']) == 2:
        output['backend_aggregate'] = {}
        for label, part in (('backend_cache', 'cache'), ('backend_training', 'training')):
            # Population sizes describe unique data and must not be added across A/B.
            keys = ('episodes', 'native_history_ticks', 'plans') if part == 'cache' else ('new_optimizer_updates', 'supervised_ticks', 'control_ticks')
            merged = {key: sum(v[part]['counters'][key] for v in output['backends'].values()) for key in keys}
            seconds = output['timeline'][label]['stage_interval_union_seconds']
            output['backend_aggregate'][part] = dict(counters=merged, stage_interval_union_seconds=seconds,
                rates_per_stage_union_second={key: value / seconds for key, value in merged.items()})
    complete = bool(ledger_complete and collection['complete'])
    if complete:
        require('population' in output and len(output['frontends']) == len(output['backends']) == 2
                and 'development' in output, 'Complete cycle lacks measured stage artifacts')
        begin, end = completed['aggregate']['start'], completed['development']['end']
        require(end > begin, 'Invalid aggregate-to-development interval')
        pipeline_rows = [row for row in receipts if begin <= row['start'] and row['end'] <= end + .01]
        require(len(pipeline_rows) == len(receipts), 'Stage lies outside complete cycle envelope')
        cycle_seconds = end - begin; covered = union_seconds([(r['start'], r['end']) for r in receipts])
        collect = collection['interval']; envelope = end - collect['start']
        envelope_union = union_seconds([(r['start'], r['end']) for r in receipts] + [(collect['start'], collect['end'])])
        output['timeline'].update(cycle_start_utc=begin, cycle_end_utc=end, cycle_wall_seconds=cycle_seconds,
            cycle_definition='aggregate owner start through development stage owner exit; concurrent workers counted once',
            cycle_unclassified_or_other_work_seconds=max(0., cycle_seconds - covered),
            collection_to_evaluation_envelope_seconds=envelope,
            observed_collection_and_cycle_interval_union_seconds=envelope_union,
            envelope_unclassified_or_other_work_seconds=max(0., envelope - envelope_union),
            end_of_collection_to_aggregate_seconds=begin - collect['end'],
            cycle_ledger_finalization_seconds=final['utc'] - end)
        output['complete'] = True
    else:
        output['status'] = 'incomplete; no full-cycle throughput or future completion inferred'
    output['bindings'] = sources.bindings
    return output


def comparison(baseline, candidate):
    if not (baseline['complete'] and candidate['complete']):
        return dict(complete=False, ratios=None, reason='Both actual complete cycles are required')
    fields = ('cycle_wall_seconds', 'collection_to_evaluation_envelope_seconds')
    return dict(complete=True, elapsed_candidate_over_baseline={k: candidate['timeline'][k] / baseline['timeline'][k] for k in fields},
        same_population_counts=baseline['population']['after'] == candidate['population']['after'],
        baseline_population=baseline['population']['after'], candidate_population=candidate['population']['after'],
        scope='Observed whole-cycle elapsed ratios, not isolated optimization causality or equal-work benchmark.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cycle-root', type=Path, required=True)
    parser.add_argument('--collect-root', type=Path)
    parser.add_argument('--stages-root', type=Path)
    parser.add_argument('--outer-registration', type=Path)
    parser.add_argument('--baseline-cycle-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve previous report; choose a new output')
    value = summarize(args.cycle_root, collection_root=args.collect_root, stages_root=args.stages_root,
                      outer_registration=args.outer_registration)
    if args.baseline_cycle_root:
        baseline = summarize(args.baseline_cycle_root)
        value = dict(version=VERSION, baseline=baseline, candidate=value, comparison=comparison(baseline, value))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False); file.write('\n')


if __name__ == '__main__':
    main()
