"""Snapshot round13 receipts and actual prefetch timing; never open raw/model payloads."""
import argparse
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


TREE = Path(__file__).resolve().parents[2]


def summarize(root, previous):
    bindings = {}

    def read(path, lines=False):
        payload = path.read_bytes()
        bindings[str(path)] = hashlib.sha256(payload).hexdigest()
        return [json.loads(line) for line in payload.splitlines()] if lines else json.loads(payload)

    def metrics(value):
        return {key: value[key] for key in ('arm_rmse_rad', 'hand_rmse_rad', 'selection')}

    def io_summary(rows):
        epochs = [row['training_epoch_io'] for row in rows]
        batches = [epoch['batch_prefetch'] for epoch in epochs if epoch.get('batch_prefetch')]
        return dict(
            epochs=len(epochs), training_epoch_wall_seconds=sum(x['wall_seconds'] for x in epochs),
            process_read_bytes=sum(x['process_io']['read_bytes'] for x in epochs),
            batch_prefetch={key: sum(x[key] for x in batches) for key in
                            ('batches', 'prefetch_hits', 'admission_fallbacks', 'consumer_wait_s', 'load_s')},
            maximum_next_batch_bytes=max((x['maximum_next_batch_bytes'] for x in batches), default=None),
            timing_semantics='Epoch wall excludes validation. Background load overlaps computation; do not add load_s to wall. OS caching and workload differ across rounds.')

    module_path = TREE / 'scripts/tight_fresh_20261115_01/report_collection11.py'
    spec = importlib.util.spec_from_file_location('collection_receipts', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    collection = module.summarize(root, round_index=13)
    held = [collection['schedules'][key] for key in ('held0', 'held_root60')]
    collection['holding_related'] = {key: sum(row[key] for row in held) for key in held[0]}
    collection['holding_related']['accepted_per_hour'] = (
        collection['holding_related']['accepted'] * 3600 / collection['collection_seconds'])

    candidate = root / 'round_13'
    stages = root / 'round_13_stages'
    data = read(candidate / 'A/state_data/data.json')
    latest = data['aggregation']['new_training_ids']
    roles = {}
    for role in ('A', 'B'):
        training = candidate / role / 'training'
        if not (training / 'finished.json').exists():
            progress = read(training / 'progress.json')
            roles[role] = dict(complete=False, progress={key: progress.get(key) for key in
                               ('epoch', 'batch_index', 'updates', 'visits_seen', 'utc')})
            continue
        done = read(training / 'finished.json')
        assert done['complete']
        rows = read(training / 'validation.jsonl', lines=True)
        selected = min(rows, key=lambda row: row['validation']['selection'])
        initial = read(training / 'initial_metrics.json')
        roles[role] = dict(
            complete=True, epochs=done['epochs'], optimizer_updates=done['updates'],
            ema_updates=done['ema_updates'], episode_visits=done['visits_seen'],
            stage_seconds=read(stages / ('frontend_' + role + '.json'))['seconds'],
            training_finished_wall_seconds=done['wall_seconds'],
            selected_epoch=selected['epoch'], selected_checkpoint_sha256=done['checkpoint_sha256'],
            selection_source='First minimum of completed validation receipts; checkpoint tensors not reopened.',
            selected_new_correction_visits_histogram=dict(Counter(selected['sampling_exposure'][key] for key in latest)),
            last_new_correction_visits_histogram=dict(Counter(done['sampling_exposure'][key] for key in latest)),
            initial_validation=metrics(initial['validation']), selected_validation=metrics(selected['validation']),
            validation_selection_relative_reduction=1 - selected['validation']['selection'] / initial['validation']['selection'],
            initial_new_correction=metrics(initial['correction']),
            last_new_correction=metrics(rows[-1]['correction']),
            selected_new_correction=metrics(selected['correction']) if 'correction' in selected else None,
            selected_correction_scope='Absent selected-epoch cohort metrics remain null; last is not selected.',
            sampling_coverage=done['visit_sampling_coverage'], actual_epoch_io=io_summary(rows))

    before_done = read(previous / 'round_12/A/training/finished.json')
    before_rows = read(previous / 'round_12/A/training/validation.jsonl', lines=True)
    before_seconds = read(previous / 'round_12_stages/frontend_A.json')['seconds']
    before_io = io_summary(before_rows)
    after = roles['A']
    comparison = dict(
        previous_round=12, candidate_round=13, role='A',
        previous_stage_seconds=before_seconds, candidate_stage_seconds=after['stage_seconds'],
        previous_updates=before_done['updates'], candidate_updates=after['optimizer_updates'],
        stage_wall_relative_reduction=1 - after['stage_seconds'] / before_seconds,
        observed_updates_per_stage_second_ratio=(after['optimizer_updates'] / after['stage_seconds']) / (before_done['updates'] / before_seconds),
        previous_epoch_io=before_io,
        epoch_wall_relative_reduction=1 - after['actual_epoch_io']['training_epoch_wall_seconds'] / before_io['training_epoch_wall_seconds'],
        interpretation='Actual formal frontend stage measurements. Data pool, new-suffix repeat count and update workload differ; not an isolated causal prefetch benchmark or completed-cycle speedup.')
    services = {}
    for name in ('tight-fresh-prefetch-20261115-03.service', 'tight-fresh-monitor-20261115-03.service'):
        result = subprocess.run(['systemctl', '--user', 'show', name,
                                 '--property=ActiveState,SubState,MainPID,ExecMainStatus'],
                                text=True, capture_output=True, timeout=10)
        services[name] = dict(returncode=result.returncode,
                              properties=dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line))
    evaluation = candidate / 'development/result.json'
    return dict(
        version='round13_progress_and_prefetch_actual_v1',
        observed_at=datetime.datetime.now().astimezone().isoformat(),
        collection=collection, train_trajectories=len(data['splits']['train']),
        training_families=len({data['entries'][key]['family'] for key in data['splits']['train']}),
        validation_trajectories=len(data['splits']['validation']), roles=roles,
        frontend_A_observed_comparison=comparison,
        visual_cache_seconds=read(stages / 'visual_cache.json')['seconds'],
        active_stage=read(stages / 'status.json'), owner_status=read(root / 'research_loop/owner_status.json'),
        services=services, autonomous_complete=read(evaluation).get('software_complete') is True if evaluation.exists() else False,
        models_opened=0, raw_payloads_opened=0, learning_jsonl_opened=0,
        interpretation='Successful expert recovery is not student success. Held collection conditions are distinct from native lift/alignment. Complete DP plus refreshed backend and autonomous evaluation remain required.',
        input_sha256=bindings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root.resolve(), args.previous.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(dict(observed_at=result['observed_at'], collection=result['collection']['total'],
                         holding_related=result['collection']['holding_related'],
                         train=result['train_trajectories'], families=result['training_families'],
                         active_stage=result['active_stage'], B=result['roles']['B']), ensure_ascii=False))
