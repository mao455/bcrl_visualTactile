"""Read-only final hashes of original data, caches, formal weights and results."""
import argparse
import json
from pathlib import Path
import time

from .compact import sha256
from .training_resources import atomic_json


def audit(root, output):
    output.mkdir(parents=True, exist_ok=False)
    session = json.loads((root/'session.json').read_text())
    source = Path(session['source_root'])
    data = json.loads((source/'data.json').read_text())
    prior = json.loads((Path(session['previous_root'])/'data_audit.json').read_text())
    checks = dict(data=sha256(source/'data.json') == prior['source_data_sha256'],
                  statistics=sha256(source/'statistics.pt') == prior['statistics_sha256'])
    began = time.time()
    for index, (identity, entry) in enumerate(data['entries'].items(), start=1):
        checks['raw/'+identity] = sha256(entry['path']) == entry['source_sha256']
        checks['numeric/'+identity] = sha256(source/'numeric'/(identity+'.pt')) == entry['numeric_sha256']
        if index % 20 == 0:
            atomic_json(output/'progress.json', dict(episodes=index, all_match=all(checks.values())))
    manifest = json.loads((source/'manifest.json').read_text())
    for group in ('S-NativeDP', 'I-OfficialDP', 'I-HistoryDP'):
        for suffix in ('formal', 'test', 'formal_evaluation', 'train_diagnostic'):
            for path, expected in manifest['stages'][group+'_'+suffix]['artifacts'].items():
                checks['old/'+group+'_'+suffix+'/'+Path(path).name] = sha256(path) == expected
    native = manifest['stages']['S-NativeDP_formal']['source_hashes']
    for path, expected in native.items():
        if path.startswith('isaacsimenvs/tasks/play/') or path == 'visuotactile/sim/environment.py':
            checks['source/'+path] = sha256(path) == expected
    atomic_json(output/'result.json', dict(software_complete=True, passed=all(checks.values()),
        checks=checks, raw_episodes=len(data['entries']), elapsed_s=time.time()-began,
        scope='Full raw400 SHA256, numeric400, old three formal best/last/results/test/evaluation and native task/wrapper'))
    if not all(checks.values()):
        raise ValueError('Protected identity mismatch')
    print(json.dumps(dict(passed=True, checks=len(checks), elapsed_s=time.time()-began)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.root, arguments.output)
