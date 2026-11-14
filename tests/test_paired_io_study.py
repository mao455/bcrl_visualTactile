import json
import os
from pathlib import Path
import time

import pytest
import torch

from tools import tight_fresh_paired_io_study as study
from visuotactile.compact import sha256
from visuotactile.latest_takeover_sampling import episode_deck
from visuotactile.training_resources import atomic_json


def fixture_source(root):
    source = root / 'round_06'; source.mkdir()
    write = atomic_json
    session = dict(resource_policy='full_gpu_v1', deadline=time.time()+3600)
    write(source / 'session.json', session); write(source / 'gate.json', dict(passed=True))
    write(source / 'metadata.json', dict(preserved='objective'))
    write(source / 'shared_resources.json', dict(utc_epoch=time.time(), gpu_free_mib=32768,
                                              ram_available_bytes=40<<30))
    train = ['train_%03d' % i for i in range(93)]; validation = ['val_%d' % i for i in range(6)]
    for role in ('A', 'B'):
        directory = source / role; (directory / 'state_data/perception').mkdir(parents=True)
        entries = {}; shards = {}
        for identity in train + validation:
            path = directory / 'state_data/perception' / (identity + '.pt')
            if role == 'A': path.write_bytes(identity.encode())
            else: os.link(source / 'A/state_data/perception' / path.name, path)
            entries[identity] = dict(trajectory_kind='expert_takeover_v1')
            shards[identity] = dict(path=str(path), sha256=sha256(path), source_sha256='0'*64,
                                    frames=8, camera_time_sha256='1'*64)
        data = dict(visual_pair=True, visual_stride=1, entries=entries,
            splits=dict(train=train, validation=validation, test=[]), normalization_train_ids=train[:30],
            aggregation=dict(version='expert_takeover_aggregate_v1', student_prefix_supervised=False,
                             new_training_ids=train[-9:]))
        write(directory / 'state_data/data.json', data)
        for name in ('numeric.pt', 'statistics.pt', 'loss_context.pt'):
            (directory / 'state_data' / name).write_bytes((role+name).encode())
        checkpoint = root / 'round_05' / role / 'training/best.pt'; checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(('original-parameters-' + role).encode())
        write(directory / 'continuation.json', dict(path=str(checkpoint), sha256=sha256(checkpoint),
            statistics_sha256=sha256(directory / 'state_data/statistics.pt'),
            sampling_contract=dict(version='latest_takeover_repeat_v1', latest_repeat=2),
            target_window_contract='recovery_onset_commit12_v1'))
        write(directory / 'cache_index.json', dict(complete=True, stride=1, trainable_projection_cached=False,
            data_sha256=sha256(directory / 'state_data/data.json'), entries=shards))
    return source


def prepared(tmp_path, monkeypatch):
    source = fixture_source(tmp_path); output = tmp_path / 'comparison'
    monkeypatch.setattr(study, 'code_identity', lambda: dict(root='frozen', commit='test', sha256={'x': 'test'}))
    plan = study.prepare(source, output)
    return source, output, plan


def test_cpu_prepare_preserves_inputs_deadline_and_explicit_arms(tmp_path, monkeypatch):
    source, output, plan = prepared(tmp_path, monkeypatch)
    assert not torch.cuda.is_initialized()
    study.verify(output, plan)
    assert plan['max_seconds'] == 720 and plan['visits_per_role'] == 102
    for arm in study.ARMS:
        assert (output / arm / 'session.json').read_bytes() == (source / 'session.json').read_bytes()
        assert (output / arm / 'paired_episode_io.json').exists() == (arm == 'paired')
        for role in ('A', 'B'):
            assert not (output / arm / role / 'episode_io.json').exists()
            assert not (output / arm / role / 'training').exists()
            assert (output / arm / role / 'state_data').resolve() == source / role / 'state_data'
            assert (output / arm / role / 'continuation.json').read_bytes() == (source / role / 'continuation.json').read_bytes()
    with pytest.raises(FileExistsError): study.prepare(source, output)


@pytest.mark.parametrize('change', ['checkpoint', 'numeric', 'metadata', 'prefetch', 'paired_config', 'source_code'])
def test_input_or_configuration_drift_rejected(tmp_path, monkeypatch, change):
    source, output, plan = prepared(tmp_path, monkeypatch)
    if change == 'checkpoint': path = tmp_path / 'round_05/A/training/best.pt'
    elif change == 'numeric': path = source / 'A/state_data/numeric.pt'
    elif change == 'metadata': path = output / 'plain/metadata.json'
    elif change == 'paired_config': path = output / 'paired/paired_episode_io.json'
    elif change == 'prefetch': path = output / 'plain/A/episode_io.json'
    else:
        monkeypatch.setattr(study, 'code_identity', lambda: {'changed': True}); path = None
    if path is not None: path.write_text('{}')
    with pytest.raises(ValueError): study.verify(output, plan)


def synthesize_arm(root, plan):
    population = plan['source_inputs']['shared_train']['population']
    order = episode_deck(population['ids'], population['multiplicities'], torch.Generator().manual_seed(2026111402))
    for role in ('A', 'B'):
        directory = root / role / 'training'; directory.mkdir()
        for name in ('last.pt', 'best.pt'): (directory / name).write_bytes(b'preserved-checkpoint')
        rows = [dict(episode=identity, updates=index+1) for index, identity in enumerate(order)]
        (directory / 'learning.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        atomic_json(directory / 'finished.json', dict(complete=True, epochs=1, updates=102,
            anchors_seen=204, target_ticks=900, sampling_exposure=population['multiplicities'],
            checkpoint_sha256=sha256(directory / 'best.pt')))
        atomic_json(directory / 'validation.json', dict(training_epoch_io=dict(started_utc=10, ended_utc=20,
            wall_seconds=10, process_io=dict(read_bytes=123), paired_episode_io=None)))


def test_owner_sequences_exact_existing_train_dispatch_and_retains_receipts(tmp_path, monkeypatch):
    _, output, plan = prepared(tmp_path, monkeypatch); calls = []
    def worker(output_arg, arm, name, args, completion, deadline, **kwargs):
        assert output_arg == output and args == ['tools.tight_fresh_train', 'train', '--root', output / arm,
                                               '--role', name[-1], '--epochs', 1]
        assert deadline <= plan['session_deadline'] and deadline-time.time() <= 700
        calls.append((arm, name))
    monkeypatch.setattr(study, 'run_worker', worker)
    original = study.paired_frontends
    def paired(execute, root, epochs):
        assert epochs == 1
        result = original(execute, root, epochs)
        synthesize_arm(root, plan)
        return result
    monkeypatch.setattr(study, 'paired_frontends', paired)
    study.run(output)
    assert [arm for arm, name in calls] == ['plain', 'plain', 'paired', 'paired']
    report = study.read(output / 'result.json')
    assert report['software_complete'] and report['deployable'] is False
    assert report['arms']['plain']['aggregate_updates'] == 204
    assert report['arms']['plain']['summed_process_training_read_bytes'] == 246
    assert report['paired_over_plain_epoch_throughput'] == 1
    with pytest.raises(FileExistsError): study.run(output)


def test_owner_does_not_start_second_arm_after_failure_or_retry(tmp_path, monkeypatch):
    _, output, _ = prepared(tmp_path, monkeypatch); calls = []
    def failed(execute, root, epochs):
        calls.append(root.name)
        (root / 'A/training').mkdir(); (root / 'A/training/partial.json').write_text('{}')
        return dict(complete=False, errors={'A': 'injected'})
    monkeypatch.setattr(study, 'paired_frontends', failed)
    with pytest.raises(RuntimeError, match='arm failed'): study.run(output)
    assert calls == ['plain'] and (output / 'plain/A/training/partial.json').exists()
    assert study.read(output / 'status.json')['stage'] == 'failed'
    with pytest.raises(FileExistsError): study.run(output)


def test_owner_signal_preserves_finished_arm_and_stops_next_launch(tmp_path, monkeypatch):
    _, output, plan = prepared(tmp_path, monkeypatch); calls = []
    old_handler = study.signal.getsignal(study.signal.SIGTERM)
    def stopped(execute, root, epochs):
        calls.append(root.name); synthesize_arm(root, plan)
        study.signal.raise_signal(study.signal.SIGTERM)
        return dict(complete=True, errors={})
    monkeypatch.setattr(study, 'paired_frontends', stopped)
    with pytest.raises(InterruptedError, match='STOP before next arm'): study.run(output)
    assert calls == ['plain'] and (output / 'plain_result.json').exists()
    assert study.read(output / 'stop_requested.json')['signal'] == study.signal.SIGTERM
    assert study.signal.getsignal(study.signal.SIGTERM) == old_handler


def test_owned_real_cpu_child_is_stopped_at_bound_without_completion(tmp_path):
    started = time.time()
    with pytest.raises(RuntimeError, match='did not complete'):
        study.run_worker(tmp_path, 'plain', 'cpu_wait',
            ['timeit', '-n', '1', '-r', '1', 'import time; time.sleep(20)'],
            tmp_path / 'never_finished.json', time.time()+.2)
    assert time.time()-started < 3
    receipt = study.read(tmp_path / 'owner/plain/cpu_wait.json')
    assert receipt['stop_reason'] == 'bounded_study_training_cutoff' and receipt['returncode'] != 0
    assert not Path('/proc', str(receipt['pid'])).exists()


def test_existing_epoch_order_and_exposure_are_checked(tmp_path, monkeypatch):
    _, output, plan = prepared(tmp_path, monkeypatch)
    synthesize_arm(output / 'plain', plan)
    path = output / 'plain/A/training/learning.jsonl'; lines = path.read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]; path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError, match='exposure/order'): study.arm_summary(output, 'plain', plan)
