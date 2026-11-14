import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import tight_state
from visuotactile import startup_diagnostics


def test_default_environment_does_not_enable_diagnostics(tmp_path, monkeypatch):
    seen = {}
    def fake_environment(root, evaluation, **kwargs):
        seen.update(kwargs)
        return 'unchanged'
    monkeypatch.setattr(tight_state, '_environment', fake_environment)
    monkeypatch.setattr(startup_diagnostics.faulthandler, 'dump_traceback_later',
        lambda *args, **kwargs: pytest.fail('Formal default must not arm a timer'))
    assert tight_state.environment(tmp_path, num_envs=8) == 'unchanged'
    assert seen['num_envs'] == 8 and 'startup_mark' not in seen
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('fail', [False, True])
def test_opt_in_stages_timer_cancel_and_original_error(tmp_path, monkeypatch, fail):
    timers = []
    monkeypatch.setattr(startup_diagnostics.faulthandler, 'dump_traceback_later',
        lambda *args, **kwargs: timers.append((args, kwargs)))
    monkeypatch.setattr(startup_diagnostics.faulthandler, 'cancel_dump_traceback_later', lambda: timers.append('cancel'))
    def fake_environment(root, evaluation, **kwargs):
        kwargs['startup_mark']('environment_constructor_started', num_envs=kwargs['num_envs'])
        if fail:
            raise RuntimeError('native constructor failure')
        kwargs['startup_mark']('environment_constructor_complete')
        return 'ready'
    monkeypatch.setattr(tight_state, '_environment', fake_environment)
    if fail:
        with pytest.raises(RuntimeError, match='native constructor failure'):
            tight_state.environment(tmp_path, num_envs=8, startup_diagnostics=True)
    else:
        assert tight_state.environment(tmp_path, num_envs=8, startup_diagnostics=True) == 'ready'
    assert timers[0][0] == (120,) and timers[0][1]['repeat'] is True
    assert timers[-1] == 'cancel' and timers[0][1]['file'].closed
    report = json.loads((tmp_path/'environment_initialization.json').read_text())
    assert report['current_stage'] == ('environment_startup_failed' if fail else 'environment_ready')
    assert report['events'][1]['num_envs'] == 8


def test_abrupt_worker_exit_leaves_last_stage_and_real_python_stack(tmp_path):
    # The timer runs in an isolated child so it cannot interfere with pytest's
    # own faulthandler. No simulator, GPU or controller is imported.
    code = '''import os, sys, time
from visuotactile.startup_diagnostics import StartupDiagnostics
with StartupDiagnostics(sys.argv[1], interval_s=.05) as probe:
    probe.mark('environment_constructor_started')
    time.sleep(.13)
    os._exit(17)
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=15)
    assert result.returncode == 17
    report = json.loads((tmp_path/'environment_initialization.json').read_text())
    assert report['current_stage'] == 'environment_constructor_started'
    assert not any(row['stage'] == 'environment_ready' for row in report['events'])
    trace = (tmp_path/'startup_tracebacks.log').read_text()
    assert 'Timeout' in trace and 'File "<string>", line 5' in trace
