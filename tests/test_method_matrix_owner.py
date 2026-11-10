"""Exercise owner failure/deadline cleanup without launching a process or GPU."""
import json

import pytest

from tools import tight_finish


@pytest.mark.parametrize('termination', ['budget', 'worker_failure'])
def test_matrix_gate_failure_stops_owned_work_and_never_trains(tmp_path, monkeypatch, termination):
    root = tmp_path/'matrix'; root.mkdir()
    monkeypatch.setattr(tight_finish, 'CODE', tmp_path)
    tight_finish.write(root/'configuration.json', dict(method_matrix=dict(maximum_hours=1/3600, acceptance_hours=1/3600)))
    tight_finish.write(root/'session.json', {})
    tight_finish.write(root/'shared_resources.json', dict(utc_epoch=0.))
    now = [0.]
    children = []
    reports = []
    class Child:
        def __init__(self, args, **kwargs):
            self.args = args
            self.pid = 1000+len(children)
            self.returncode = None
            children.append(self)
            if 'tools.reset_query_probe' in args:
                if termination == 'budget': now[0] = 2.
                else: self.returncode = 1
        def poll(self): return self.returncode
        def wait(self, timeout=None):
            self.returncode = 0 if self.returncode is None else self.returncode
            return self.returncode
    def git(args, **kwargs):
        if args[:3] == ['git', 'rev-parse', 'HEAD']: return 'reviewed-source\n'
        return ''
    def stop(child):
        child.returncode = -15 if child.returncode is None else child.returncode
    def no_sleep(*args, **kwargs):
        raise AssertionError('Early failed gate should not enter a polling wait')
    monkeypatch.setattr(tight_finish.subprocess, 'Popen', Child)
    monkeypatch.setattr(tight_finish.subprocess, 'check_output', git)
    monkeypatch.setattr(tight_finish.time, 'time', lambda: now[0])
    monkeypatch.setattr(tight_finish.time, 'sleep', no_sleep)
    monkeypatch.setattr(tight_finish, 'kill_owned', stop)
    monkeypatch.setattr(tight_finish, 'matrix_report', lambda path: reports.append(tight_finish.read(path/'status.json')))
    expected = TimeoutError if termination == 'budget' else ValueError
    with pytest.raises(expected): tight_finish.matrix_run(root)
    status = json.loads((root/'status.json').read_text())
    assert status['state'] == ('BUDGET_STOP' if termination == 'budget' else 'FAILED')
    assert status['owned_workers_exited']
    assert len(children) == 2 and all(child.poll() is not None for child in children)
    assert any('tools.reset_query_probe' in child.args for child in children)
    assert all('vision_run' not in child.args and 'residual_run' not in child.args for child in children)
    assert reports[-1]['owned_workers_exited']
    # No automatic restart can replace this failed root or reset its deadline.
    with pytest.raises(ValueError, match='relaunch'): tight_finish.matrix_run(root)
