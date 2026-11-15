"""One read of real round14 control progress; no polling, simulation or model loading."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def run(root, output):
    bindings = {}

    def read(path):
        payload = path.read_bytes()
        bindings[str(path)] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    ledger = root.with_name(root.name + '_launch')
    launch = read(ledger / 'status.json')
    assert launch['launch_complete']
    session = read(root / 'session.json')
    old = Path(launch['configuration']['old_session'])
    prior = read(old / 'session.json')
    assert bindings[str(old / 'session.json')] == launch['old_session_sha256']
    assert session['deadline'] == session['started_utc'] + 21600
    assert session['reserve_bytes'] == 32*1024**3
    plan = read(root / 'new_family_14/plan.json')
    progress = read(root / 'new_family_14/batch_progress.json')
    assert progress['wave'] > 0 or progress['tick'] >= 25
    schedules = dict(Counter(case['expansion_schedule'] for case in plan['cases']))
    assert schedules == dict(fixed1=16, held0=8, held_root60=8)
    assert plan['batch']['verification_workers'] == 8
    data = read(root / 'A/state_data/data.json')
    services = {}
    for key in ('owner_unit', 'monitor_unit'):
        name = launch['configuration'][key]
        command = subprocess.run(['systemctl', '--user', 'show', name,
            '--property=ActiveState,SubState,MainPID,ExecMainStatus'], capture_output=True, text=True, timeout=10)
        assert command.returncode == 0
        services[name] = dict(line.split('=', 1) for line in command.stdout.splitlines() if '=' in line)
        assert services[name]['ActiveState'] == 'active'
    result = dict(version='round14_actual_startup_v1', observed_at=datetime.datetime.now().astimezone().isoformat(),
        source_commit=launch['source_commit'], worker_tree=launch['worker_tree'],
        new_started=session['started_utc'], new_deadline=session['deadline'],
        old_deadline=prior['deadline'], old_session_preserved=True,
        schedules=schedules, verification_workers=plan['batch']['verification_workers'],
        source_pairs=plan['pairs'], source_train_trajectories=len(data['splits']['train']),
        source_train_families=len({data['entries'][key]['family'] for key in data['splits']['train']}),
        actual_progress=progress, owner=read(root / 'research_loop/owner_status.json'), services=services,
        interpretation='Actual control advancement observed after round13 closed. No new recovery count, training result, throughput gain or autonomous benefit established yet.',
        input_sha256=bindings)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in ('observed_at', 'new_started', 'new_deadline', 'schedules', 'actual_progress')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
