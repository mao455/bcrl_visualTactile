"""Read-only file-stat/JSON disk projection; no payload reads, cleanup or job changes."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time

SESSION = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06')
TREE = Path('/home/CNS2025331827/docs/bc+rl_RSS_continuous_vision_20261114')
GIB = 1024**3
bindings = {}


def read(path):
    raw = path.read_bytes(); bindings[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def kind(path):
    parts = path.parts
    if 'perception' in parts: return 'frozen_dino'
    if 'nominal' in parts: return 'rear_dependency_cache'
    if path.suffix == '.pt' and 'training' in parts:
        return 'rear_checkpoints' if 'backend' in parts else 'frontend_checkpoints'
    if 'state_data' in parts: return 'numeric_and_geometry'
    if parts[0] == 'development': return 'development'
    if 'demonstration' in parts: return 'raw_capture_including_partial'
    if path.name == 'numeric.h5': return 'numeric_diagnostics'
    return 'other'


def scan(root):
    groups = {}; files = {}; seen = set(); unique_allocated = 0
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in names:
            path = Path(parent) / name; rel = path.relative_to(root); info = path.lstat()
            row = groups.setdefault(kind(rel), dict(bytes=0, allocated=0, files=0, symlinks=0))
            if path.is_symlink(): row['symlinks'] += 1; continue
            row['bytes'] += info.st_size; row['allocated'] += info.st_blocks * 512; row['files'] += 1
            files[str(rel)] = info.st_size
            inode = (info.st_dev, info.st_ino)
            if inode not in seen: unique_allocated += info.st_blocks * 512; seen.add(inode)
    return dict(categories=groups, file_bytes=sum(v['bytes'] for v in groups.values()),
                unique_allocated_bytes=unique_allocated, file_sizes=files)


def main():
    began = time.time(); disk_before = shutil.disk_usage(SESSION)
    collection = SESSION / 'new_family_09'; previous = SESSION / 'round_08'
    plan = read(collection / 'plan.json'); rows = read(collection / 'results.json')
    entries = read(collection / 'correction_entries.json'); progress = read(collection / 'batch_progress.json')
    session = read(SESSION / 'session.json'); bounds = sorted(collection.glob('wave_*_native_bound.json'))
    bound = read(bounds[-1]); registered = {v['name'] for v in plan['cases']}
    assert {r['case'] for r in rows} <= registered
    assert len(rows) == len({r['case'] for r in rows})
    old = scan(previous); current = scan(collection); source = read(previous / 'A/state_data/data.json')
    cache_index = read(previous / 'A/cache_index.json')
    accepted_frames = sum(e['length'] for e in entries.values())
    resolved_steps = sum(r['result'].get('steps', 0) for r in rows)
    resolved_bytes = sum(b for name, b in current['file_sizes'].items() if name.split('/')[0] in {r['case'] for r in rows})
    raw_rows = []
    for row in rows:
        n = row['result'].get('steps', 0)
        names = [row['case'] + '/demonstration/' + file for file in ('data.h5', 'data.h5.partial')]
        b = sum(current['file_sizes'].get(name, 0) for name in names)
        if n and b: raw_rows.append(dict(ticks=n, bytes=b, bytes_per_tick=b / n))
    raw_bytes_per_tick = sum(r['bytes'] for r in raw_rows) / sum(r['ticks'] for r in raw_rows)
    raw_max_bytes_per_tick = max(r['bytes_per_tick'] for r in raw_rows)
    raw_overhead_bytes_per_tick = max(0., (resolved_bytes - sum(r['bytes'] for r in raw_rows)) / resolved_steps)
    old_new_ids = source['aggregation']['new_training_ids']
    dino_sizes = [(cache_index['entries'][i]['frames'], Path(cache_index['entries'][i]['path']).stat().st_size) for i in old_new_ids]
    dino_mean = sum(b for n, b in dino_sizes) / sum(n for n, b in dino_sizes)
    dino_max = max(b / n for n, b in dino_sizes)
    old_frames = sum(e['length'] for e in source['entries'].values())
    base_frames = sum(source['entries'][i]['length'] for ids in source['splits'].values() for i in ids)
    rear_per_frame = old['categories']['rear_dependency_cache']['bytes'] / old_frames
    numeric_per_frame = old['categories']['numeric_and_geometry']['bytes'] / old_frames
    ckpt = old['categories']['frontend_checkpoints']['bytes']
    largest_front_ckpt = max(b for name, b in old['file_sizes'].items() if name.endswith('.pt') and 'training/' in name and '/backend/' not in name)
    checkpoint_peak = ckpt + 2 * largest_front_ckpt
    other_fixed = (old['categories']['rear_checkpoints']['bytes'] + old['categories']['development']['bytes']
                   + old['categories']['other']['bytes'] + GIB)
    disk_after = shutil.disk_usage(SESSION); available = disk_after.free; reserve = max(60 * GIB, session['reserve_bytes'])
    remaining = len(registered) - len(rows)
    assumptions = [dict(name='observed_A_prefix_extrapolation_not_forecast', future_ticks=resolved_steps / len(rows),
        future_acceptance=len(entries) / len(rows), future_accepted_ticks=accepted_frames / max(1, len(entries)), conservative=False),
        dict(name='every_unresolved_case_accepted_601_ticks_scenario', future_ticks=601, future_acceptance=1., future_accepted_ticks=601, conservative=True),
        dict(name='every_unresolved_case_accepted_732_ticks_observed_old_tail_stress', future_ticks=732, future_acceptance=1., future_accepted_ticks=732, conservative=True),
        dict(name='registered_loop_bound_not_expected_duration', future_ticks=bound['collector_max_control_ticks'], future_acceptance=1.,
             future_accepted_ticks=bound['collector_max_control_ticks'], conservative=True)]
    projections = []
    for assumption in assumptions:
        future = remaining * assumption['future_ticks']
        new_frames = accepted_frames + remaining * assumption['future_acceptance'] * assumption['future_accepted_ticks']
        raw_rate = raw_max_bytes_per_tick if assumption['conservative'] else raw_bytes_per_tick
        estimated_collection = resolved_bytes + future * (raw_rate + raw_overhead_bytes_per_tick) + GIB
        additional_collection = max(0., estimated_collection - current['file_bytes'])
        dino = new_frames * (dino_max if assumption['conservative'] else dino_mean)
        rear = (base_frames + new_frames) * rear_per_frame * 1.1
        numeric = (base_frames + new_frames) * numeric_per_frame * 1.1
        training_peak = dino + rear + numeric + checkpoint_peak + other_fixed
        additional = additional_collection + training_peak
        free_at_peak = available - additional
        projections.append(dict(**assumption, new_accepted_history_frames=new_frames,
            estimated_collection_final_bytes=math.ceil(estimated_collection),
            additional_collection_bytes=math.ceil(additional_collection), new_shared_dino_bytes=math.ceil(dino),
            full_pool_rear_cache_bytes=math.ceil(rear), full_pool_numeric_bytes=math.ceil(numeric),
            checkpoint_peak_bytes=checkpoint_peak, other_fixed_and_slack_bytes=other_fixed,
            additional_through_cycle_peak_bytes=math.ceil(additional), estimated_free_at_peak_bytes=math.floor(free_at_peak),
            reserve_margin_bytes=math.floor(free_at_peak - reserve), preserves_60GiB_reserve=free_at_peak >= reserve))
    for value in (old, current): value.pop('file_sizes')
    value = dict(version='round09_disk_projection_v1',started_utc=began,ended_utc=time.time(),
        read_only_measurement=True,no_files_removed=True,no_jobs_changed=True,
        disk=dict(total_bytes=disk_after.total,free_before_bytes=disk_before.free,free_after_bytes=available,
                  minimum_reserve_bytes=reserve,headroom_above_reserve_bytes=available-reserve),
        current_collection=dict(snapshot_incomplete=True,root=str(collection),registered_attempts=len(registered),
            resolved_results=len(rows),accepted_entries=len(entries),accepted_history_frames=accepted_frames,
            resolved_control_ticks=resolved_steps,remaining_unresolved_attempts=remaining,
            batch_progress=progress,storage=current),
        round08_storage=old,measured_rates=dict(raw_mean_bytes_per_tick=raw_bytes_per_tick,
            raw_max_bytes_per_tick=raw_max_bytes_per_tick,raw_non_capture_overhead_bytes_per_tick=raw_overhead_bytes_per_tick,
            frozen_dino_mean_bytes_per_frame=dino_mean,frozen_dino_max_bytes_per_frame=dino_max,
            both_rear_caches_bytes_per_full_history_frame=rear_per_frame,both_numeric_bytes_per_full_history_frame=numeric_per_frame,
            checkpoint_largest_frontend_bytes=largest_front_ckpt,source_pool_history_frames_including_validation=base_frames),
        projections=projections,bindings=bindings,
        code_evidence=dict(frozen_cache_reuse='visuotactile/native_data.py: cache_visual_pair symlinks matching previous frozen shards; B shares A index',
            checkpoints='visuotactile/checkpointing.py: save uses same-directory .partial.pt then replace; two DP workers can each need one temporary full snapshot',
            nominal_cache='Each new selected DP needs all pool per-tick residual features/plans; old DP cache is not reused',
            reserve_guard='visuotactile/state_resources.py: Lease.check checks current free bytes against session reserve; not a future-stage capacity guarantee'),
        limitations=['Snapshot during an active collection; file-stat counters and filesystem free space are not one atomic filesystem snapshot.',
            'First resolved waves are A only; their acceptance and duration do not predict remaining A or B outcomes.',
            '601 is a per-goal native timeout scale, not a whole-trajectory bound. 732 occurred in round08; neither stress scenario is a worst-case proof.',
            'The registered 2408 control loop bound is deliberately conservative, not an expected episode duration.',
            'Raw compression and future trajectory length vary. Future whole-pool caches add 10% to measured per-frame bytes; other/slack adds 1 GiB plus another 1 GiB collection allowance.',
            'Temporary checkpoint peak assumes two DP writers concurrently; optional diagnostics, other jobs and unclassified extra artifacts are excluded.',
            'Current reserve guards cannot pre-reserve all future output or prevent a single atomic write crossing the threshold; check actual accepted frames at collection end before a capacity conclusion.',
            'No image, DINO, model or trajectory payload was opened; old cleanup of 135 last/intermediate artifacts was not repeated.'])
    value['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = Path(__file__).with_name('report.json')
    with out.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(report=str(out),free_gib=available/GIB,resolved=len(rows),accepted=len(entries),
        collection_gib=current['file_bytes']/GIB,scenarios=[dict(name=r['name'],additional_gib=r['additional_through_cycle_peak_bytes']/GIB,
            free_at_peak_gib=r['estimated_free_at_peak_bytes']/GIB,margin_gib=r['reserve_margin_bytes']/GIB) for r in projections]),indent=2))


if __name__ == '__main__': main()
