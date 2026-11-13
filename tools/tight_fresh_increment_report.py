"""Compare saved backend errors to the causal previous-actuator-target baseline."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def report(root, role):
    directory = root / role
    data_path = directory / 'state_data/data.json'
    data = json.loads(data_path.read_text())
    offline = directory / 'backend' / role / 'offline'
    result = json.loads((offline / 'result.json').read_text())
    if not result['complete'] or result['test_used']:
        raise ValueError('Expected complete non-test offline evaluation')
    with (offline / 'sampled_ticks.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    records = {}
    bindings = {}
    for identity in sorted({row['identity'] for row in rows}):
        if identity not in data['splits']['train'] + data['splits']['validation']:
            raise ValueError('Unexpected evaluation population')
        entry = data['entries'][identity]
        bindings[identity] = entry['source_sha256']
        with h5py.File(entry['path']) as stream:
            group = stream['episodes'][entry['episode']]
            target = group['transition/target_after_rad'][:].reshape(-1, 29)
            previous = group['obs/actuator_target_rad'][:].reshape(-1, 29)
            if len(target) > 1 and not np.array_equal(target[:-1], previous[1:]):
                raise ValueError('Recorded actual actuator target continuity differs')
            records[identity] = (target - previous).astype(np.float64)
    buckets = defaultdict(list)
    seen = set()
    for row in rows:
        identity, tick = row['identity'], int(row['tick'])
        key = (row['split'], identity, tick)
        if key in seen:
            raise ValueError('Duplicated evaluation tick')
        seen.add(key)
        delta = records[identity][tick]
        for phase in ('all', row['phase']):
            for part, section in (('arm', slice(0, 7)), ('hand', slice(7, 29))):
                buckets[row['split'], phase, part].append(
                    (float(row[part + '_mse_rad2']), float(np.square(delta[section]).mean())))
    metrics = []
    for (split, phase, part), values in sorted(buckets.items()):
        mean = np.mean(values, axis=0)
        metrics.append(dict(split=split, phase=phase, group=part, ticks=len(values),
            backend_rmse_rad=float(np.sqrt(mean[0])), previous_target_rmse_rad=float(np.sqrt(mean[1])),
            backend_to_previous_target_rmse_ratio=float(np.sqrt(mean[0] / mean[1])) if mean[1] else None))
    return dict(metrics=metrics, checkpoint_sha256=result['checkpoint_sha256'],
        offline_result_sha256=sha256(offline / 'result.json'), sampled_ticks_sha256=sha256(offline / 'sampled_ticks.csv'),
        data_sha256=sha256(data_path), recorded_source_hashes=bindings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, dict(roles={role: report(args.root, role) for role in ('A', 'B')},
        scope='Same saved supervised ticks and actual expert history; baseline holds actual pre-transition actuator target. No model inference, parameter updates or autonomous success claim.',
        test_used=False))


if __name__ == '__main__':
    main()
