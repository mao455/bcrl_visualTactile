"""Resume the interrupted B DP; retain completed A and the original finite cycle."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import psutil

PUBLIC = Path(__file__).resolve().parent
ROOT = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_adaptive_dagger_20261119_01')
TARGET = ROOT / 'round02/learning'
REG = ROOT / 'round02/disk_recovery04/registration.json'


def load(name):
    spec = importlib.util.spec_from_file_location(name, PUBLIC / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load('round2_supervised_recovery')
read, write, digest = helpers.read, helpers.write, helpers.digest


def remaining_stages(target):
    stages = [dict(entry='frontend', args=['--root', str(target), '--role', 'B',
              '--epochs', '60', '--full-trajectory-batch', '--offline-anchor-batch', '64'])]
    for role in 'AB':
        backend = target / role / 'backend'
        stages.extend([
            dict(entry='backend_prepare', args=['--root', str(target), '--role', role,
                 '--epochs', '60', '--continue-rear', '--offline-anchor-batch', '64',
                 '--frozen-visual-io-json', str(target / 'frozen_visual_io.json')]),
            dict(entry='backend_cache', args=['--root', str(backend)]),
            dict(entry='backend_train', args=['--root', str(backend / role),
                 '--job', 'image', '--output', str(backend / role / 'training')]),
            dict(entry='backend_offline', args=['--root', str(backend / role),
                 '--job', 'image', '--output', str(backend / role / 'offline')])])
    return stages


def session_gate(registration=None):
    session = read(ROOT / 'session.json')
    assert time.time() < session['deadline'], 'Original deadline expired'
    assert not (ROOT / 'final_freeze.json').exists(), 'No learning after final freeze'
    assert not (ROOT / 'final_test').exists(), 'No learning after final test planning'
    if registration:
        assert registration['original_deadline'] == session['deadline']
    return session


def register():
    import torch
    from tools.tight_frozen_frontend import sources
    session = session_gate()
    assert not REG.parent.exists(), 'Preserve previous recovery identity'
    cleanup = ROOT / 'resource_cleanup_20261121/tinyobj_old_data/result.json'
    cleaned = read(cleanup)
    assert cleaned['complete'] and cleaned['clean_stat_identities_unchanged']
    previous = ROOT / 'round2_ram_recovery03_failure.json'
    assert previous.is_file()
    required = ('model', 'optimizer', 'ema', 'ema_optimization_step', 'torch_rng', 'cuda_rng',
                'noise_rng', 'schedule_rng', 'effective_exposure_state',
                'visit_sampling_coverage', 'sampling_exposure', 'visits_seen', 'configuration')
    checkpoint = TARGET / 'B/training/last.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert all(k in state for k in required)
    assert (state['epoch'], state['updates']) == (1, 120)
    assert state['effective_exposure_state']['updates'] == state['updates'] == state['ema_optimization_step']
    assert state['configuration'] == read(TARGET / 'B/training/configuration.json')
    assert state['configuration']['source_sha256'] == sources()
    identities = {}
    for role in 'AB':
        config = read(TARGET / role / 'training/configuration.json')
        for key, relative in (
                ('data_sha256', 'state_data/data.json'),
                ('statistics_sha256', 'state_data/statistics.pt'),
                ('cache_index_sha256', 'cache_index.json')):
            path = TARGET / role / relative
            assert digest(path) == config[key]
            identities[str(path)] = config[key]
        data = read(TARGET / role / 'state_data/data.json')
        assert len(data['splits']['train']) == 180 and len(data['splits']['validation']) == 6
        assert not (TARGET / role / 'backend').exists(), 'Backends were not started in predecessor'
    completed = read(TARGET / 'A/training/finished.json')
    assert completed['complete'] and (completed['epochs'], completed['updates']) == (60, 7200)
    assert completed['selected_effective_exposure']['eligible']
    for path in (previous, TARGET / 'A/training/best.pt', TARGET / 'A/training/last.pt',
                 TARGET / 'A/training/finished.json', cleanup):
        identities[str(path)] = digest(path)
    assert identities[str(TARGET / 'A/training/best.pt')] == completed['checkpoint_sha256']
    interrupted_sha = digest(checkpoint)
    REG.parent.mkdir(parents=True)
    pinned = REG.parent / 'B_interrupted_last.pt'
    os.link(checkpoint, pinned)  # Subsequent trainer writes replace last.pt atomically.
    assert digest(pinned) == interrupted_sha
    identities[str(pinned)] = interrupted_sha
    value = dict(version='round2_disk_recovery04_v1', utc=time.time(),
                 original_deadline=session['deadline'], identities=identities,
                 completed_A_skipped=dict(epochs=60, updates=7200,
                     selected=completed['selected_effective_exposure'],
                     selected_sha256=completed['checkpoint_sha256']),
                 B_durable=dict(epoch=state['epoch'], updates=state['updates'],
                     visits_seen=state['visits_seen'], required_state_keys=list(required),
                     preserved_checkpoint=str(pinned), checkpoint_sha256=interrupted_sha,
                     optimizer_state_entries=len(state['optimizer']['state'])),
                 predecessor_failure=str(previous), cleanup_receipt=str(cleanup),
                 exposure_minimum=50, budget_epochs=60, runtime_guards=session['resources'],
                 recovery='Exact B raw model/Adam/EMA/RNG/deck/exposure restore; A is skipped; each backend uses its own newly selected DP',
                 physical_opportunities_repeated=0, policy_source_changes=0,
                 promotion=False, final_test_opened=False)
    write(REG, value)
    write(PUBLIC / 'round02/disk_recovery04_registration.json', value)
    print(json.dumps(dict(registered=True, A_updates=7200, B_resume_update=120,
                         original_deadline=session['deadline'])), flush=True)


def validate_registration():
    registration = read(REG)
    session_gate(registration)
    for raw, expected in registration['identities'].items():
        assert digest(Path(raw)) == expected, 'Dependency changed: ' + raw
    assert digest(TARGET / 'B/training/last.pt') == registration['B_durable']['checkpoint_sha256']
    return registration


def run():
    validate_registration()
    with (REG.parent / 'started.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time(), utc=time.time()), stream)
    cycle = load('round2_cycle')
    experiment = cycle.experiment
    experiment.resource_wait = helpers.stable_admission
    write(ROOT / 'progress.json', dict(round=2, stage='B_DP_full_state_restore', utc=time.time()))
    experiment.execute('train', 'r2_train_disk_recovery04', remaining_stages(TARGET))
    experiment.audit_training(TARGET, 2)
    for role in 'AB':
        cycle.load('audit_backend_masks').audit(
            TARGET, role, PUBLIC / f'round02/{role}_backend_raw_mask_acceptance.json')
    cycle.load('exposure_phases').export(2)
    write(ROOT / 'progress.json', dict(round=2, stage='evaluate', utc=time.time()))
    cycle.evaluate(cycle.SOURCE, TARGET, 2)
    cycle.load('analyze_round').forks(2)
    write(ROOT / 'progress.json', dict(round=2, stage='offline', utc=time.time()))
    cycle.load('analyze_round').offline(TARGET, PUBLIC / 'round02/offline')
    write(ROOT / 'round02/complete.json', dict(
        collection=str(ROOT / 'round02/collection_combined'), learning=str(TARGET),
        evaluation=True, utc=time.time()))
    pairs = {f'{r}_parent': cycle.completed_pair(experiment.BASE, r) for r in 'AB'}
    pairs.update({f'{r}_new': cycle.completed_pair(TARGET, r) for r in 'AB'})
    write(ROOT / 'final_freeze.json', dict(
        pairs=pairs, seeds=read(experiment.CONFIG / 'experiment.json')['final_test_seeds'],
        utc=time.time(), selection='Own validation-eligible round2 complete pairs, frozen before any new final-test outcome'))
    write(ROOT / 'progress.json', dict(round=2, stage='final_test', utc=time.time()))
    cycle.evaluate(experiment.BASE, TARGET, 2, heldout=True)
    write(ROOT / 'cycle_complete.json', dict(rounds=2, final_test_complete=True,
                                             utc=time.time(), automatic_promotion=False))
    experiment.close_report()
    write(ROOT / 'progress.json', dict(round=2, stage='complete', utc=time.time()))


def supervise():
    supervisor = load('supervise_resources')
    registration = read(ROOT / 'round2_disk_recovery04_controller.json')
    with (REG.parent / 'supervision_started.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                       utc=time.time(), coordinator=registration, poll_seconds=30), stream)
    with (REG.parent / 'supervision.jsonl').open('x') as stream:
        while True:
            row = supervisor.snapshot(registration)
            row['final_test_frozen'] = (ROOT / 'final_freeze.json').exists()
            row['final_test_opened'] = any((ROOT / 'final_test').glob('paired*/results.json'))
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            write(REG.parent / 'supervision_latest.json', row)
            if row['state'] in ('complete', 'failed_or_coordinator_exited') or time.time() >= row['deadline']:
                write(REG.parent / 'supervision_stopped.json', row)
                return
            time.sleep(min(30, max(.1, row['deadline'] - time.time())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('register', 'run', 'supervise'))
    args = parser.parse_args()
    try:
        globals()[args.action]()
    except BaseException as error:
        if args.action == 'run':
            write(ROOT / 'round2_disk_recovery04_failure.json', dict(error=repr(error), utc=time.time()))
        raise
