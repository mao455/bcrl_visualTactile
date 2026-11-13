import pytest

from tools import tight_span_metric as runner
from tools.runtime import read, write, sha


def source_fixture(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    recovery = tmp_path / 'recovery'
    monkeypatch.setattr(runner, 'ROOT', source)
    monkeypatch.setattr(runner, 'RECOVERY_ROOT', recovery)
    monkeypatch.setattr(runner.time, 'time', lambda: 1000.)
    write(source / 'registration.json', dict(stage_deadline=10000., started_utc=100.,
        compute_deadline=11000., absolute_deadline=12000., bindings={}))
    write(source / 'launch.json', dict(pid=12345, process=dict(start_ticks=678)))
    return source, recovery


def test_recovery_preserves_original_stage_and_identity(tmp_path, monkeypatch):
    source, recovery = source_fixture(tmp_path, monkeypatch)
    runner.prepare_recovery(recovery)
    registered = read(recovery / 'registration.json')
    original = read(source / 'registration.json')
    for key in ('stage_deadline', 'started_utc', 'compute_deadline', 'absolute_deadline'):
        assert registered[key] == original[key]
    assert registered['source_owner'] == dict(pid=12345, start_ticks=678)
    assert registered['new_optimizer_updates'] == 0
    with pytest.raises(ValueError, match='overwrite'):
        runner.prepare_recovery(recovery)


def test_recovery_refuses_late_admission(tmp_path, monkeypatch):
    _, recovery = source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner.time, 'time', lambda: 9000.)
    with pytest.raises(TimeoutError):
        runner.prepare_recovery(recovery)
    assert not recovery.exists()


def test_recovery_refuses_repeating_source_evaluations(tmp_path, monkeypatch):
    source, recovery = source_fixture(tmp_path, monkeypatch)
    runner.prepare_recovery(recovery)
    monkeypatch.setattr(runner, 'proc', lambda pid: None)
    write(source / 'failure.json', dict(error='Unexpected cross-metric training difference'))
    write(source / 'span/G01/backend/evaluation_complete.json', dict(complete=True))
    with pytest.raises(ValueError, match='repeating'):
        runner.recover(recovery)
    assert read(recovery / 'status.json')['state'] == 'FAILED'
    assert not (recovery / 'span').exists()


def offline_fixture(tmp_path):
    source = tmp_path / 'source'
    destination = tmp_path / 'destination'
    for name in ('training', 'nominal', 'state_data'):
        (source / name).mkdir(parents=True)
        destination.mkdir(exist_ok=True)
        (destination / name).symlink_to(source / name, target_is_directory=True)
    checkpoint = source / 'training/best.pt'
    checkpoint.write_bytes(b'fixed selected endpoint')
    write(source / 'offline/result.json', dict(complete=True, test_used=False,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint)))
    return source, destination


def test_exact_offline_reuse_does_not_run_inference(tmp_path):
    source, destination = offline_fixture(tmp_path)
    receipt = runner.reuse_selected_offline(destination)
    assert read(destination / 'offline/result.json') == read(source / 'offline/result.json')
    assert receipt['exact_reuse'] and not receipt['new_inference']
    assert receipt['new_optimizer_updates'] == 0
    assert runner.reuse_selected_offline(destination) == receipt


@pytest.mark.parametrize('field,value', [('complete', False), ('test_used', True),
    ('checkpoint_sha256', 'different checkpoint')])
def test_offline_reuse_rejects_wrong_evidence(tmp_path, field, value):
    source, destination = offline_fixture(tmp_path)
    result = read(source / 'offline/result.json')
    result[field] = value
    write(source / 'offline/result.json', result)
    with pytest.raises(ValueError):
        runner.reuse_selected_offline(destination)
    assert not (destination / 'offline/result.json').exists()


@pytest.mark.parametrize('name', ['nominal', 'state_data'])
def test_offline_reuse_rejects_mismatched_dependencies(tmp_path, name):
    _, destination = offline_fixture(tmp_path)
    (destination / name).unlink()
    (destination / name).mkdir()
    with pytest.raises(ValueError, match='identity differs'):
        runner.reuse_selected_offline(destination)


@pytest.mark.parametrize('suffix', ['evaluation_started/case.json',
    'evaluation_failures/case.json', 'evaluation/case/started.json',
    'evaluation_summary.json'])
def test_recovery_rejects_even_partial_started_attempt(tmp_path, suffix):
    write(tmp_path / 'reference/QG01/backend/QG01' / suffix, {})
    with pytest.raises(ValueError, match='repeating'):
        runner.require_unstarted_evaluation(tmp_path)


def test_second_recovery_preserves_deadlines_and_source(tmp_path, monkeypatch):
    _, first = source_fixture(tmp_path, monkeypatch)
    runner.prepare_recovery(first)
    second = tmp_path / 'second'
    monkeypatch.setattr(runner, 'OFFLINE_RECOVERY_ROOT', second)
    write(first / 'launch.json', dict(pid=23456, process=dict(start_ticks=789)))
    runner.prepare_recovery(second)
    registered = read(second / 'registration.json')
    original = read(first / 'registration.json')
    assert registered['recovery_source'] == str(first)
    for key in ('stage_deadline', 'started_utc', 'compute_deadline', 'absolute_deadline'):
        assert registered[key] == original[key]
