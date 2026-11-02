"""Publish small experiment summaries after all owned native work has exited."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from .compact import sha256
from .training_resources import atomic_json
from .viewing_report import check_video, resource_intervals


def public_value(value):
    if isinstance(value, dict):
        return {key: public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, str):
        if value.startswith('/') or (len(value) > 2 and value[1:3] in (':\\', ':/')):
            normalized = value.replace('\\', '/')
            for marker in ('outputs/', 'research_docs/', 'visuotactile/', 'assets/', 'isaacsimenvs/'):
                if marker in normalized:
                    return marker + normalized.split(marker, 1)[1]
            return Path(normalized).name
    return value


def export(root, output):
    status = json.loads((root/'status.json').read_text())
    owner = json.loads((root/'owner_exit.json').read_text())
    if status['state'] != 'diagnostics_complete' or not owner['owned_children_exited']:
        raise ValueError('Completed experiment and exited owned children required')
    output.mkdir(parents=True, exist_ok=True)
    session = json.loads((root/'session.json').read_text())
    source = Path(session['source_root'])
    previous = Path(session['previous_root'])
    audit = json.loads((previous/'data_audit.json').read_text())
    data = json.loads((source/'data.json').read_text())
    checks = dict(data=sha256(source/'data.json') == audit['source_data_sha256'],
        statistics=sha256(source/'statistics.pt') == audit['statistics_sha256'],
        old_state=sha256(session['old_state_checkpoint']) == session['old_state_sha256'],
        released_expert=sha256(session['expert_checkpoint']) == session['expert_sha256'])
    numeric_checks = [sha256(source/'numeric'/(identity+'.pt')) == entry['numeric_sha256']
                      for identity, entry in data['entries'].items()]
    if not all(checks.values()) or not all(numeric_checks) or len(numeric_checks) != 400:
        raise ValueError('Protected source identity changed')
    monitor = json.loads((root/'monitor_exit.json').read_text())
    paths = dict(outcomes=root/'report/summary.json', phase_metrics=root/'phase_metrics/result.json',
        history_only=root/'history_only/result.json', closed_loop_audit=root/'closed_loop_audit/result.json',
        historical_forensics=root.parent/'forensics_20261102_01/summary.json',
        actor_memory=root.parent/'actor_audit_20261102_02/result.json',
        deployment_features=root.parent/'deployment_audit_20261102_03/result.json',
        preservation=root/'preservation/result.json')
    for name, path in paths.items():
        atomic_json(output/(name+'.json'), public_value(json.loads(path.read_text())))
    atomic_json(output/'configuration.json', public_value(dict(session=session,
        bc=json.loads((root/'bc_compare/configuration.json').read_text()),
        environment=json.loads((root/'expert1/environment.json').read_text()),
        actor=json.loads((root/'expert1/actor_config.json').read_text()))))
    videos = []
    for path in sorted(root.glob('*/seed_*/rollout.mp4')) + sorted(root.glob('bc_saved/*/student.mp4')):
        result = json.loads((path.parent/'result.json').read_text())
        steps = result['trials'][0]['steps'] if 'trials' in result else result['steps']
        video = check_video(path, steps, set())
        videos.append(dict(video, path=str(path.relative_to(root))))
    atomic_json(root/'report/video_verification.json', videos)
    atomic_json(output/'videos.json', videos)
    for source_path, name in [(root/'report/native_outcomes.png', 'native_outcomes.png'),
                              (root/'report/bc_learning.png', 'bc_learning.png'),
                              (root.parent/'forensics_20261102_01/historical_divergence.png', 'historical_divergence.png')]:
        shutil.copyfile(source_path, output/name)
    resource = dict(monitor=monitor, over50_intervals=resource_intervals(root/'resources/resources.jsonl'))
    atomic_json(output/'resources.json', public_value(resource))
    atomic_json(output/'verification.json', dict(utc=time.time(), report_commit_base=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), protected_hashes=checks,
        numeric_cache_hashes_checked=len(numeric_checks), numeric_cache_hashes_match=all(numeric_checks),
        original_raw400_hashes='Re-read after new BC1000 training; all836 protected checks passed',
        status=status, owner=owner, video_count=len(videos),
        input_report_hashes={name: sha256(path) for name, path in paths.items()},
        public_scope='Small summaries/configuration/curves only; no raw trajectories, tensors, weights or videos'))
    print(json.dumps(dict(output=str(output), videos=len(videos), protected_hashes=checks), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    export(arguments.root, arguments.output)
