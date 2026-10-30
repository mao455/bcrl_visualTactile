"""Decode every viewing frame and publish only small, sanitized statistics."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .compact import sha256
from .residual_report import execution_summary, paired_trace_summary, resource_summary, switch_physical_summary
from .training_resources import atomic_json


def check_video(path, expected_frames, preview_steps, preview_path=None):
    import av
    from PIL import Image
    selected = []
    times = []
    sizes = set()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        rate = float(stream.average_rate)
        for ordinal, frame in enumerate(container.decode(video=0)):
            times.append(float(frame.pts*frame.time_base))
            sizes.add((frame.width, frame.height))
            if ordinal in preview_steps:
                selected.append(frame.to_image())
    if len(times) != expected_frames or rate != 60 or sizes != {(1344, 316)}:
        raise ValueError(f'Video count/rate/three-view geometry mismatch: {path.name}')
    if not np.allclose(np.diff(times), 1/60, rtol=0, atol=1e-9) or times[0] != 0:
        raise ValueError(f'Video timestamps are not consecutive native observations: {path.name}')
    preview = Image.new('RGB', (1344, 316*len(selected)))
    for ordinal, frame in enumerate(selected):
        preview.paste(frame, (0, ordinal*316))
    if preview_path is not None:
        preview.save(preview_path, quality=90)
    return dict(file=path.name, sha256=sha256(path), bytes=path.stat().st_size,
        decoded_frames=len(times), fps=rate, width=1344, height=316,
        duration_s=len(times)/60, last_frame_time_s=times[-1], full_decode=True,
        pts_uniform=True, preview_steps=sorted(preview_steps),
        terminal_image_available=False, reset_frames_included=False)


def check_trace(path, row):
    if sha256(path) != row['trace_sha256']:
        raise ValueError('Viewing trace hash changed')
    with np.load(path) as trace:
        length = len(trace['command'])
        student = trace['student'].astype(bool)
        bypass = student & ~trace['gate'].astype(bool)
        checks = dict(native_ack=np.array_equal(trace['command'].clip(-1, 1), trace['clipped']),
            causal_feedback=np.array_equal(trace['previous_command'][1:], trace['clipped'][:-1]),
            memory_continuous=np.array_equal(trace['memory_updates'], np.arange(1, length+1)),
            cursor_continuous=np.array_equal(trace['cursor'], np.arange(length)%6),
            generation_steps=np.array_equal(trace['generation_step'], np.arange(length)//6*6),
            no_contact_exact_base=np.array_equal(trace['command'][bypass], trace['base'][bypass]),
            expected_steps=length==row['steps'])
        if row['variant'] == 'a':
            checks['zero_residual_exact_base'] = np.array_equal(trace['command'][student], trace['base'][student])
        if not all(checks.values()):
            raise ValueError(f'Causal trace check failed: {row["attempt"]}: {checks}')
        return checks


def resource_intervals(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    intervals = []
    active = None
    for row in rows:
        rolling = row['rolling_60s_percent']
        if rolling is not None and rolling > 50:
            if active is None:
                active = dict(first_sample_epoch=row['utc_epoch'], last_sample_epoch=row['utc_epoch'],
                              max_complete_60s_mean_percent=rolling, samples=0)
            active['last_sample_epoch'] = row['utc_epoch']
            active['max_complete_60s_mean_percent'] = max(active['max_complete_60s_mean_percent'], rolling)
            active['samples'] += 1
        elif active is not None:
            intervals.append(active)
            active = None
    if active is not None:
        intervals.append(active)
    return intervals


def preservation_checks():
    baseline = Path('outputs/visuotactile/run/dp_command_20261029')
    previous = Path('outputs/visuotactile/run/residual_20261030_01')
    expected = json.loads((previous / 'preservation.json').read_text())
    audit = json.loads(Path('outputs/visuotactile/run/audit_20261030_01/traces.json').read_text())
    for row in audit['trials']:
        path = baseline / 'evaluation' / f'{row["variant"]}_s{row["seed"]}.npz'
        if sha256(path) != row['trace_sha256']:
            raise ValueError('Original formal trace identity changed')
    identities = {name:sha256(baseline / name / 'best.pt') for name in ('b0', 'b1')}
    for name, digest in identities.items():
        if digest != expected[name+'_sha256']:
            raise ValueError('Original best checkpoint identity changed')
    manifest = sha256(Path('outputs/visuotactile/data/close_v4/manifest.json'))
    formal = sha256(baseline / 'evaluation/report.json')
    if manifest != expected['manifest_sha256'] or formal != expected['formal_report_sha256']:
        raise ValueError('Original manifest/formal report changed')
    prior_report = json.loads((previous / 'public_report.json').read_text())
    checked = 0
    for row in prior_report['diagnostic_trials']:
        if 'trace_sha256' in row:
            if sha256(previous / row['trace_directory'] / (row['attempt']+'.npz')) != row['trace_sha256']:
                raise ValueError('Prior residual diagnostic trace changed')
            checked += 1
    return dict(original_formal_trace_hashes_checked=len(audit['trials']), prior_residual_trace_hashes_checked=checked,
        all_match=True, checkpoint_sha256=identities, manifest_sha256=manifest, formal_report_sha256=formal)


def worker_resource_window(path, started, finished):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    area, measured, missing = 0., 0., 0.
    for previous, current in zip(rows, rows[1:]):
        duration = max(0., min(current['utc_epoch'], finished)-max(previous['utc_epoch'], started))
        if previous['gpu_util_percent'] is None or current['elapsed_s']-previous['elapsed_s'] > 15:
            missing += duration
        else:
            area += duration*previous['gpu_util_percent']
            measured += duration
    complete = [row['rolling_60s_percent'] for row in rows
                if started+60 <= row['utc_epoch'] <= finished and row['rolling_60s_percent'] is not None]
    return dict(started_epoch=started, finished_epoch=finished, measured_seconds=measured, missing_seconds=missing,
        time_weighted_mean_percent=area/measured if measured else None,
        max_complete_60s_mean_percent=max(complete) if complete else None,
        scope='Recorder launch through final trial report, before Kit teardown; includes admission/pacing waits')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    session = json.loads((root / 'session.json').read_text())
    report = json.loads((root / 'viewing/report.json').read_text())
    if session.get('purpose') != 'viewing_rerun' or not report['complete'] or len(report['trials']) != 6:
        raise ValueError('Final viewing report requires the six allocated naturally terminated runs')
    launch = json.loads((root / 'viewing/launch.json').read_text())
    monitor = json.loads((root / 'monitor_final.json').read_text())
    if Path('/proc', str(launch['pid'])).exists() or Path('/proc', str(session['monitor_pid'])).exists():
        raise ValueError('Recorder and continuous monitor must exit before final aggregation')
    trials, videos, traces = [], {}, {}
    for row in report['trials']:
        trial = {name:value for name,value in row.items() if name not in ('causal_events', 'geometry')}
        trials.append(trial)
        switch = row['switch_step']
        previews = {0, 120, 360, row['steps']-1} if row['mode']=='autonomous' else {
            0, switch if switch is not None else 120,
            min((switch if switch is not None else 120)+20, row['steps']-1), row['steps']-1}
        path = root / 'viewing' / row['video']['file']
        videos[row['attempt']] = check_video(path, row['steps'], previews, path.with_name(path.stem+'_preview.jpg'))
        if videos[row['attempt']]['sha256'] != row['video']['sha256']:
            raise ValueError('Recorded video hash mismatch')
        traces[row['attempt']] = check_trace(root / 'viewing' / (row['attempt']+'.npz'), row)
    baseline = {}
    old_checks = json.loads((args.baseline_root / 'video_checks.json').read_text())
    prior_videos = {row['file']:row for row in old_checks['videos']}
    for path in sorted(args.baseline_root.glob('*.mp4')):
        baseline[path.stem] = check_video(path, 601, {0, 120, 360, 600})
        if baseline[path.stem]['sha256'] != prior_videos[path.name]['sha256']:
            raise ValueError('Reused baseline video changed')
    resources = resource_summary(root / 'resources/resources.jsonl')
    resources['over_50_complete_window_intervals'] = resource_intervals(root / 'resources/resources.jsonl')
    resources['both_gpu_mean_limits_met'] = bool(resources['time_weighted_mean_percent'] <= 50 and
        resources['max_complete_60s_mean_percent'] <= 50 and resources['missing_seconds'] == 0)
    waits = [{name:value for name,value in json.loads(path.read_text()).items() if name!='pid'}
             for path in root.glob('waits_*.json')]
    result = dict(report_version=1, purpose='viewing reruns, not new formal or original diagnostic samples',
        launch_source_commit=launch['source_commit'], source_sha256={Path(name).name:value
            for name,value in launch['source_sha256'].items()}, trials=trials,
        new_videos=videos, reused_baseline_videos=baseline, causal_trace_checks=traces,
        previous_baseline_video_checks_sha256=sha256(args.baseline_root / 'video_checks.json'),
        original_preservation=preservation_checks(),
        execution=execution_summary(root, trials), paired_traces=paired_trace_summary(root/'viewing', trials),
        switch_physical_differences=switch_physical_summary(root, trials), resources=resources, waits=waits,
        recorder_resource_window=worker_resource_window(root/'resources/resources.jsonl',
            launch['started_epoch'], report['finished_epoch']),
        session_elapsed_s=monitor['ended_epoch']-session['started_epoch'], session_limit_s=7200,
        final_disk_free_bytes=shutil.disk_usage(root).free, disk_minimum_scope='12 GB checked before work; continuous minimum unknown',
        owned_processes=dict(recorder_exited=True, monitor_exited=True, training_started=False),
        limitations=['Aggregate normal forces do not identify contact partners.',
            'Matching seeds or switch physics does not guarantee pixel-identical or identical plan replay.',
            'C retains the real force gate; not a strict tactile-free control.',
            'Whole-card telemetry includes other workloads; gate cannot guarantee a hard quota.'])
    atomic_json(root / 'video_checks.json', result)
    print(json.dumps(dict(videos=len(videos), baseline_videos=len(baseline), resources=resources), indent=2))


if __name__ == '__main__':
    main()
