import pytest

from tools import tight_span_metric as runner
from tools.runtime import read, write


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
