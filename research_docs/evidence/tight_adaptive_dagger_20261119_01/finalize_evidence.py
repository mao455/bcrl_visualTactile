"""Audit completed evidence and retained assets, without training or simulation."""
import collections
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

import psutil

PUBLIC = Path(__file__).resolve().parent
PROJECT = PUBLIC.parents[2]
CANONICAL = Path('/home/CNS2025331827/docs/bc+rl_RSS')
ROOT = CANONICAL / 'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def export():
    started = time.time()
    session = read(ROOT / 'session.json')
    cycle = read(ROOT / 'cycle_complete.json')
    assert cycle['rounds'] == 2 and cycle['final_test_complete']
    assert cycle['utc'] < session['deadline'] and not cycle['automatic_promotion']
    freeze = read(ROOT / 'final_freeze.json')
    registration = read(ROOT / 'final_test/registration.json')
    assert registration['seeds'] == freeze['seeds'] == list(range(28839000, 28839008))
    cases = read(PUBLIC / 'final_test/per_case.json')
    assert len(cases) == 32
    assert len({(r['role'], r['seed']) for r in cases}) == 32
    assert all(r['legal'] and r['software_complete'] for r in cases)
    assert len(registration['waves']) == 16
    for raw in registration['waves']:
        wave = Path(raw)
        plan = read(wave / 'plan.json')
        assert plan['pairs'] == freeze['pairs'] and plan['test_used']
        assert plan['holdout_contract'] == 'layout_final_test_v1'
        assert len(plan['cases']) == plan['batch']['lanes'] == 2
        assert plan['cases'][0]['spec'] == plan['cases'][1]['spec']
        assert read(wave / 'paired_initial_check.json')['passed']
        assert read(wave / 'result.json')['software_complete']
        assert read(wave / 'run_started.json')['utc'] >= freeze['utc']
    execution = read(PUBLIC / 'final_test/execution_audit/receipt.json')
    assert execution['passed'] and execution['cases'] == 32
    hashes = dict(read(PUBLIC / 'protected.json'))
    for pair in freeze['pairs'].values():
        for component in ('frontend', 'backend'):
            hashes[pair[component]] = pair[component + '_sha256']
    raw_assets, visual_assets, populations = {}, {}, []
    for index, size in ((1, 159), (2, 180)):
        learning = ROOT / f'round{index:02d}/learning'
        components = read(PUBLIC / f'round{index:02d}/training_completion.json')['components']
        assert len(components) == 4
        for item in components:
            assert item['eligible'] and item['minimum_visits'] >= 50 and item['minimum_tick_steps'] >= 50
            role = item['model'][0]
            relative = 'training/best.pt' if item['model'].endswith('DP') else f'backend/{role}/training/best.pt'
            hashes[str(learning / role / relative)] = item['selected_sha256']
        previous = None
        for role in 'AB':
            data = read(learning / role / 'state_data/data.json')
            assert len(data['splits']['train']) == size and len(data['splits']['validation']) == 6
            assert not data['splits'].get('test', [])
            if previous is not None:
                assert data['splits'] == previous['splits'] and data['entries'] == previous['entries']
                assert data['normalization_train_ids'] == previous['normalization_train_ids']
            previous = data
            for identity, entry in data['entries'].items():
                assert not any(str(seed) in str(entry['family']) for seed in freeze['seeds'])
                path = str(Path(entry['path']).resolve(strict=True))
                expected = entry['source_sha256']
                assert raw_assets.get(path, expected) == expected
                raw_assets[path] = expected
            for entry in read(learning / role / 'cache_index.json')['entries'].values():
                path = str(Path(entry['path']).resolve(strict=True))
                expected = entry['sha256']
                assert visual_assets.get(path, expected) == expected
                visual_assets[path] = expected
            populations.append(dict(round=index, role=role, train=size, validation=6, test=0))
    rows = []
    for kind, mapping in [('protected_or_selected', hashes), ('raw', raw_assets), ('frozen_visual', visual_assets)]:
        print(json.dumps(dict(stage=kind, files=len(mapping), bytes=sum(Path(p).stat().st_size for p in mapping))), flush=True)
        for ordinal, (raw, expected) in enumerate(mapping.items()):
            path = Path(raw)
            actual = digest(path)
            assert actual == expected, str(path)
            rows.append(dict(kind=kind, path=str(path), sha256=actual, bytes=path.stat().st_size))
            if ordinal % 20 == 0:
                print(json.dumps(dict(stage=kind, verified=ordinal + 1, total=len(mapping))), flush=True)
    manifest = CANONICAL / 'outputs/visuotactile/data/close_v4/manifest.json'
    original = read(manifest)
    assert len(original['episodes']) == 400
    old_count = 0
    for episode in original['episodes']:
        for name, value in episode['files'].items():
            path = manifest.parent / episode['directory'] / name
            assert path.is_file() and path.stat().st_size == value['bytes'], str(path)
            old_count += 1
    active = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            command = process.info['cmdline'] or []
            if any(str(ROOT) in arg or str(PUBLIC) in arg for arg in command):
                if any(marker in ' '.join(command) for marker in ('tools.tight_project', 'tools.tight_adaptive_dagger', 'round2_disk_recovery04.py', 'supervise_resources.py')):
                    active.append(dict(pid=process.pid, command=command))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    assert not active, active
    duration = cycle['utc'] - session['started_utc']
    changes = read(PUBLIC / 'final_test/paired_changes.json')
    decision = dict(complete=True, automatic_promotion=False, promote=False,
        reason='No autonomous insertion improvement on the preregistered final test; old successes lost despite sufficient per-round exposure and better supervised training fit',
        test_population=8, test_parent='Initial complete130-50 A/B pairs', candidate='Round2 complete A/B pairs',
        final_test_counts=read(PUBLIC / 'final_test/counts.json'),
        paired_changes={role: {key: sum(v[key] for v in changes if v['role'] == role)
            for key in ('retained', 'lost', 'gained')} for role in 'AB'},
        final_test_opening_does_not_authorize_more_training=True,
        original_deadline=session['deadline'], cycle_finished_utc=cycle['utc'],
        elapsed_seconds=duration, populations=populations,
        optimizer_updates_this_audit=0, physics_updates_this_audit=0)
    write(PUBLIC / 'final_decision.json', decision)
    closure = dict(complete=True, utc=time.time(), local_time=datetime.now().astimezone().isoformat(),
        original_deadline_preserved=True, actual_cycle_seconds=duration,
        final_test_cases=32, final_test_legal=32, final_test_software_incomplete=0, actual_paired_start_checks=16,
        final_test_freeze_and_model_hashes_passed=True, exposure_components_passed=8,
        protected_and_selected_sha_files=len(hashes), raw_sha_files=len(raw_assets), frozen_visual_sha_files=len(visual_assets),
        original400_manifest_sha256=digest(manifest), original400_episodes=400, original400_files_size_checked=old_count,
        original400_scope='Existence and byte sizes only; no new full-byte content SHA audit of these400 historical episodes',
        active_owned_workers=active, remaining_disk_gib=shutil.disk_usage(ROOT).free / 2**30,
        active_dependencies_remain_protected=True, seconds=time.time()-started,
        scientific_decision=decision['reason'], promotion=False, evidence_files=rows)
    write(PUBLIC / 'final_closure.json', closure)
    print(json.dumps({k: v for k, v in closure.items() if k != 'evidence_files'}), flush=True)


if __name__ == '__main__':
    export()
