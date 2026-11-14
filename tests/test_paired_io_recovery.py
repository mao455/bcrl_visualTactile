import json
import os
from pathlib import Path
import signal
import time

import pytest
import torch

from tools import tight_fresh_paired_io_recovery as recovery
from tools import tight_fresh_paired_io_study as study
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from test_paired_io_study import prepared, synthesize_arm


def failed_study(tmp_path, monkeypatch):
    source, original, plan = prepared(tmp_path, monkeypatch)
    synthesize_arm(original / 'plain', plan)
    plain = study.arm_summary(original, 'plain', plan)
    plain.update(full_arm_wall_seconds=2, full_arm_aggregate_updates_per_second=102)
    atomic_json(original / 'plain_result.json', plain)
    atomic_json(original / 'run_started.json', dict(pid=99999999, proc_start_ticks=1,
        original_session_deadline=plan['session_deadline'], training_cutoff=time.time()-20,
        bounded_deadline=time.time()-10))
    atomic_json(original / 'failure.json', dict(error="RuntimeError('Joint frontend resource admission failed')",
        completed_arms=['plain'], original_deadline_unchanged=True))
    atomic_json(original / 'plain_workers.json', dict(complete=True, errors={}))
    atomic_json(original / 'paired_admission.json', dict(admitted=False, observed_gpu_free_mib=16389.,
        required_free_gpu_mib=22*1024, required_available_ram_bytes=24*1024**3))
    (original / 'owner/plain').mkdir(parents=True)
    for role in ('A', 'B'):
        receipt = dict(pid=99999998, proc_start_ticks=1, returncode=0, stop_reason=None,
            ended_utc=time.time()-3, completion_sha256=sha256(original / 'plain' / role / 'training/finished.json'))
        atomic_json(original / 'owner/plain' / ('frontend_' + role + '.json'), receipt)
        atomic_json(original / 'owner/plain' / ('frontend_' + role + '_worker.json'), receipt)
    monkeypatch.setattr(recovery, 'STUDY', original)
    monkeypatch.setattr(recovery, 'ORIGINAL_PLAN_SHA', sha256(original / 'plan.json'))
    monkeypatch.setattr(recovery, 'ORIGINAL_FAILURE_SHA', sha256(original / 'failure.json'))
    monkeypatch.setattr(recovery, 'SESSION_DEADLINE', plan['session_deadline'])
    monkeypatch.setattr(recovery, 'frozen_module', lambda plan: study)
    return source, original, plan


def small_snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*')
            if p.is_file() and not p.is_symlink()}


def test_prepare_keeps_original_failed_and_exact_93_102_frozen_inputs(tmp_path, monkeypatch):
    source, old, old_plan = failed_study(tmp_path, monkeypatch)
    before = small_snapshot(old); output = tmp_path / 'recovery'
    plan = recovery.prepare(old, output)
    recovery.verify(output, plan)
    assert not torch.cuda.is_initialized()
    assert plan['max_seconds'] == 360 and plan['admission_max_seconds'] == 30
    assert plan['session_deadline'] == old_plan['session_deadline']
    assert plan['original']['owner']['training_cutoff'] < time.time()
    assert not (output / 'plain').exists()
    assert len(plan['original']['plan']['source_inputs']['shared_train']['population']['ids']) == 93
    assert plan['visits_per_role'] == 102
    for role in ('A', 'B'):
        assert (output / 'paired' / role / 'state_data').resolve() == source / role / 'state_data'
        assert not (output / 'paired' / role / 'training').exists()
    assert study.read(output / 'paired/paired_episode_io.json')['attempt'] != study.read(old / 'paired/paired_episode_io.json')['attempt']
    assert small_snapshot(old) == before
    with pytest.raises(FileExistsError): recovery.prepare(old, tmp_path / 'another_recovery')


@pytest.mark.parametrize('problem', ['wrong_failure', 'old_training', 'old_sync', 'old_owner',
                                     'plain_owner', 'checkpoint', 'plain_exposure', 'orphan'])
def test_rejects_other_failure_started_pair_live_owner_and_input_drift(tmp_path, monkeypatch, problem):
    _, old, _ = failed_study(tmp_path, monkeypatch)
    if problem == 'wrong_failure':
        atomic_json(old / 'failure.json', dict(error='different', completed_arms=['plain']))
    elif problem == 'old_training':
        (old / 'paired/A/training').mkdir()
    elif problem == 'old_sync':
        (old / 'paired/paired_episode_io/A').mkdir()
    elif problem in ('old_owner', 'plain_owner'):
        path = old / ('run_started.json' if problem == 'old_owner' else 'owner/plain/frontend_B.json')
        atomic_json(path, dict(study.read(path), **recovery.process_identity(os.getpid())))
    elif problem == 'checkpoint':
        (tmp_path / 'round_05/A/training/best.pt').write_bytes(b'changed')
    elif problem == 'plain_exposure':
        path = old / 'plain/A/training/finished.json'
        atomic_json(path, dict(study.read(path), updates=101))
    else:
        monkeypatch.setattr(recovery, 'assert_no_worker', lambda root: (_ for _ in ()).throw(ValueError('orphan')))
    with pytest.raises(ValueError): recovery.prepare(old, tmp_path / 'recovery')
    assert not (tmp_path / 'recovery').exists()


def test_registered_inputs_and_budget_cannot_drift(tmp_path, monkeypatch):
    _, old, _ = failed_study(tmp_path, monkeypatch)
    output = tmp_path / 'recovery'; plan = recovery.prepare(old, output)
    path = output / 'paired/A/continuation.json'; path.write_text('{}')
    with pytest.raises(ValueError, match='metadata'): recovery.verify(output, plan)


def test_run_only_paired_once_original_plain_referenced_and_confound_disclosed(tmp_path, monkeypatch):
    _, old, old_plan = failed_study(tmp_path, monkeypatch)
    before = small_snapshot(old); output = tmp_path / 'recovery'
    plan = recovery.prepare(old, output); calls = []
    def worker(root, arm, name, args, completion, deadline, **kwargs):
        assert root == output and arm == 'paired'
        assert args == ['tools.tight_fresh_train', 'train', '--root', output / 'paired', '--role', name[-1], '--epochs', 1]
        assert deadline <= plan['session_deadline'] and deadline-time.time() <= 340
        calls.append(name)
    monkeypatch.setattr(study, 'run_worker', worker)
    paired = recovery.paired_frontends
    def run_pair(execute, root, epochs):
        receipt = paired(execute, root, epochs)
        synthesize_arm(root, old_plan)
        return receipt
    monkeypatch.setattr(recovery, 'paired_frontends', run_pair)
    recovery.run(output)
    report = study.read(output / 'result.json')
    assert sorted(calls) == ['frontend_A', 'frontend_B']
    assert report['old_study_remains_failed'] and report['independent_recovery_window']
    assert report['original_plain_result']['sha256'] == sha256(old / 'plain_result.json')
    assert report['confounds']['plain_metadata_scan_overlap_seconds'] == 134.76
    assert report['arms']['paired']['roles']['B']['updates'] == 102
    assert report['deployable'] is False and small_snapshot(old) == before
    with pytest.raises(FileExistsError): recovery.run(output)


def test_admission_waits_for_new_post_exit_sample_and_original_headroom(tmp_path, monkeypatch):
    source, old, _ = failed_study(tmp_path, monkeypatch)
    output = tmp_path / 'recovery'; plan = recovery.prepare(old, output)
    clock = [time.time()]; exited = plan['original']['plain_workers_exited_utc']; updates = []
    atomic_json(source / 'shared_resources.json', dict(utc_epoch=exited-1, gpu_free_mib=32768,
                                                      ram_available_bytes=40<<30))
    monkeypatch.setattr(recovery.time, 'time', lambda: clock[0])
    def sleep(seconds):
        clock[0] += seconds; updates.append(seconds)
        atomic_json(source / 'shared_resources.json', dict(utc_epoch=clock[0],
            gpu_free_mib=16389 if len(updates) == 1 else 22528, ram_available_bytes=24<<30))
    monkeypatch.setattr(recovery.time, 'sleep', sleep)
    gate = recovery.await_admission(output, plan, clock[0]+100)
    samples = study.read(output / 'paired_admission.json')['samples']
    assert gate['admitted'] and len(samples) == 3
    assert not samples[0]['after_plain_exit'] and not samples[1]['admitted']
    assert samples[2]['required_free_gpu_mib'] == 22*1024


def test_admission_bounded_timeout_never_starts_trainer(tmp_path, monkeypatch):
    source, old, _ = failed_study(tmp_path, monkeypatch)
    output = tmp_path / 'recovery'; recovery.prepare(old, output)
    clock = [time.time()]
    monkeypatch.setattr(recovery.time, 'time', lambda: clock[0])
    monkeypatch.setattr(recovery.time, 'sleep', lambda duration: clock.__setitem__(0, clock[0]+duration))
    atomic_json(source / 'shared_resources.json', dict(utc_epoch=clock[0], gpu_free_mib=16389,
                                                      ram_available_bytes=40<<30))
    monkeypatch.setattr(recovery, 'paired_frontends', lambda *args: pytest.fail('must not launch'))
    with pytest.raises(TimeoutError, match='admission'): recovery.run(output)
    assert study.read(output / 'failure.json')['seconds'] == 30
    assert not (output / 'paired/A/training').exists()


def test_signal_during_admission_sets_stop_without_launch_and_restores_handler(tmp_path, monkeypatch):
    _, old, _ = failed_study(tmp_path, monkeypatch)
    output = tmp_path / 'recovery'; recovery.prepare(old, output)
    previous = signal.getsignal(signal.SIGTERM)
    def stopped(output, plan, cutoff):
        signal.raise_signal(signal.SIGTERM)
    monkeypatch.setattr(recovery, 'await_admission', stopped)
    monkeypatch.setattr(recovery, 'paired_frontends', lambda *args: pytest.fail('must not launch'))
    with pytest.raises(InterruptedError, match='STOP'): recovery.run(output)
    assert study.read(output / 'stop_requested.json')['signal'] == signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == previous


def test_failed_pair_preserved_and_never_restarted(tmp_path, monkeypatch):
    _, old, _ = failed_study(tmp_path, monkeypatch)
    output = tmp_path / 'recovery'; recovery.prepare(old, output)
    def failed(execute, root, epochs):
        (root / 'A/training').mkdir(); (root / 'A/training/partial.json').write_text('{}')
        return dict(complete=False, errors={'A': 'injected'})
    monkeypatch.setattr(recovery, 'paired_frontends', failed)
    with pytest.raises(RuntimeError, match='paired arm failed'): recovery.run(output)
    assert (output / 'paired/A/training/partial.json').exists()
    assert study.read(output / 'failure.json')['automatic_retry'] is False
    with pytest.raises(FileExistsError): recovery.run(output)


@pytest.mark.parametrize('seconds', [True, 0, 59, 361, 720])
def test_window_cannot_expand(tmp_path, seconds):
    with pytest.raises(ValueError, match='window'):
        recovery.prepare(tmp_path / 'study', tmp_path / 'recovery', max_seconds=seconds)
