"""Second independent review: synthetic fixtures, no live process/payload I/O."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


spec = importlib.util.spec_from_file_location('final13_capacity18', Path(__file__).with_name('host_init_capacity18.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Proc:
    pid = 123
    def create_time(self): return 100.
    def is_running(self): return True
    def status(self): return 'running'


@contextmanager
def probe_fixture():
    held = [False]
    writes, changes, restores = [], [], []
    p = Proc()
    @contextmanager
    def lock():
        assert not held[0]
        held[0] = True
        try: yield
        finally: held[0] = False
    def write(path, row):
        assert held[0]
        writes.append((Path(path).name, deepcopy(row)))
    def set_mask(token, proc, birth, cpus, row, **kwargs):
        assert held[0]
        row['mutation_attempted'] = True
        row['affinity_writes'] += 1
        row['expansion_writes'] += 1
        changes.append((token, set(cpus)))
        return True
    def restore(proc, birth, wide, row):
        assert held[0]
        restores.append(set(wide))
        row['restored'] = True
    with patch.object(m.psutil, 'Process', return_value=p), \
         patch.object(m, 'identity', return_value={}), \
         patch.object(m, 'source_guard'), \
         patch.object(m, 'stopped', return_value=False), \
         patch.object(m, 'ready', return_value=False), \
         patch.object(m, 'read', return_value={'current_stage': 'environment_constructor_started'}), \
         patch.object(m, 'spare_samples', return_value=([1, 2], [])), \
         patch.object(m, 'lifecycle_lock', side_effect=lock), \
         patch.object(m, 'masks', return_value=[((123, 10), m.OLD, 10)]), \
         patch.object(m, 'external_snapshot', return_value={}), \
         patch.object(m, 'same', return_value=True), \
         patch.object(m, 'write', side_effect=write), \
         patch.object(m, 'set_mask', side_effect=set_mask), \
         patch.object(m, 'restore', side_effect=restore), \
         patch.object(m.time, 'time', return_value=200.), \
         patch.object(m.time, 'monotonic', return_value=1.), \
         patch.object(m.time, 'sleep'):
        yield held, writes, changes, restores


def status():
    return dict(stage=26, pid=123, utc=100.1)


@pytest.mark.parametrize('states', [
    ['environment_constructor_started', 'other_nonready_stage'],
    ['environment_constructor_started', 'environment_constructor_started', 'other_nonready_stage'],
])
def test_constructor_drift_during_initial_or_locked_capacity_sample_is_noop(states):
    with probe_fixture() as (held, writes, changes, restores), \
         patch.object(m, 'read', side_effect=[{'current_stage': stage} for stage in states]):
        row = m.probe(status())
    assert changes == restores == []
    assert row['no_affinity_writes'] is True
    assert [name for name, _ in writes] == ['wave43_intent.json', 'wave43_exit.json']
    assert not held[0]


def test_constructor_drift_immediately_before_first_thread_write_is_noop():
    stages = ['environment_constructor_started']*3 + ['other_nonready_stage']
    with probe_fixture() as (_, writes, changes, restores), \
         patch.object(m, 'read', side_effect=[{'current_stage': stage} for stage in stages]):
        row = m.probe(status())
    assert changes == restores == [] and row['no_affinity_writes'] is True
    assert [name for name, _ in writes] == ['wave43_intent.json', 'wave43_expansion_intent.json', 'wave43_exit.json']


def test_constructor_drift_between_thread_writes_restores_first_write_under_lock():
    stages = ['environment_constructor_started']*4 + ['other_nonready_stage']*2
    with probe_fixture() as (held, writes, changes, restores), \
         patch.object(m, 'masks', return_value=[((123, 10), m.OLD, 10), ((124, 11), m.OLD, 10)]), \
         patch.object(m, 'read', side_effect=[{'current_stage': stage} for stage in stages]):
        row = m.probe(status())
    assert len(changes) == 1 and len(restores) == 1
    assert row['restored'] is True and writes[-1][0] == 'wave43_exit.json'
    assert row['placement_end_verified'] is True
    assert not held[0]


@pytest.mark.parametrize('chosen', [[0], [6], [True], [1, 2, 4]])
def test_unregistered_cpu_or_overbudget_fails_before_affinity_write(chosen):
    with probe_fixture() as (_, writes, changes, restores), \
         patch.object(m, 'spare_samples', return_value=(chosen, [])):
        with pytest.raises(ValueError, match='CPU whitelist/budget'):
            m.probe(status())
    assert changes == restores == []
    assert writes[-1][0] == 'wave43_exit.json'


def test_failed_restore_with_already_gone_worker_still_requests_global_abort(tmp_path):
    rows = []
    with patch.object(m, 'O', tmp_path), patch.object(m, 'D', tmp_path), \
         patch.object(m, 'same', return_value=False), \
         patch.object(m, 'write', side_effect=lambda p, row: rows.append((Path(p).name, row))):
        m.wait_failed_restore(Proc(), 100., dict(wave=str(m.TEST/'wave43')))
    assert (tmp_path/'STOP').exists()
    assert [name for name, _ in rows] == ['wave43_infrastructure_abort.json', 'wave43_abort_worker_exited.json']
    assert rows[0][1]['infrastructure_abort'] is True
    assert rows[1][1]['owner_stop_committed'] is True
    assert not (tmp_path/'startup_stops').exists()


def test_failed_exit_receipt_cannot_bypass_failed_restore_stop_and_birthbound_wait(tmp_path):
    held = [False]
    rows, seen, failed_exit_rows = [], [], []
    def write(path, row):
        assert held[0]
        if Path(path).name == 'wave43_exit.json':
            failed_exit_rows.append(deepcopy(row))
            raise OSError('injected exit I/O failure')
        rows.append((Path(path).name, deepcopy(row)))
    def live(proc, birth):
        assert held[0]
        seen.append(True)
        if len(seen) == 1:
            raise m.psutil.AccessDenied(proc.pid)
        return len(seen) < 3
    stages = ['environment_constructor_started']*4 + ['other_nonready_stage']
    with probe_fixture() as (held, _, __, ___), \
         patch.object(m, 'O', tmp_path), patch.object(m, 'D', tmp_path), \
         patch.object(m, 'write', side_effect=write), \
         patch.object(m, 'same', side_effect=live), \
         patch.object(m, 'read', side_effect=[{'current_stage': stage} for stage in stages]), \
         patch.object(m, 'restore', side_effect=ValueError('injected restore failure')):
        with pytest.raises(OSError, match='exit I/O'):
            m.probe(status())
    assert len(seen) == 3 and (tmp_path/'STOP').exists()
    assert rows[-1][0] == 'wave43_abort_worker_exited.json'
    assert rows[-1][1]['lock_releasable_after_exact_worker_exit'] is True
    assert rows[-1][1]['actual_placement_elapsed_seconds'] == 0.
    assert failed_exit_rows[0]['actual_placement_elapsed_seconds'] == 0.
    assert failed_exit_rows[0]['placement_end_verified'] is False
    assert not held[0]


def test_thread_nice_drift_causes_zero_affinity_syscalls():
    row = dict(mutation_attempted=False, affinity_writes=0, restoration_writes=0, expansion_writes=0)
    with patch.object(m, 'thread_token', return_value=(123, 10)), \
         patch.object(m.os, 'getpriority', return_value=0), \
         patch.object(m.os, 'sched_setaffinity') as change:
        with pytest.raises(ValueError, match='nice changed'):
            m.set_mask((123, 10), Proc(), 100., m.OLD|{1}, row)
    change.assert_not_called()
    assert row['mutation_attempted'] is False


def test_per_tid_starttick_is_bound_to_tgid_with_embedded_parenthesis():
    fields = ['0']*20
    fields[19] = '8181'
    with patch.object(m, 'same', return_value=True), \
         patch.object(Path, 'read_text', side_effect=['Name:\tworker\nTgid:\t123\n', '124 (worker (child)) '+' '.join(fields)]):
        assert m.thread_token(124, Proc(), 100.) == (124, 8181)


@pytest.mark.parametrize('index', [0, 12, 17, 42, 64, 100])
def test_probe_outside_registered_wave_range_performs_no_process_io(index):
    with patch.object(m.psutil, 'Process') as inspect, patch.object(m, 'stopped', return_value=False):
        assert m.probe(dict(stage=index-17, pid=123, utc=100.)) is None
    inspect.assert_not_called()


def test_signal_stop_flag_is_cooperative_and_has_no_syscall(tmp_path):
    with patch.object(m, 'STOP_REQUESTED', True), patch.object(m, 'D', tmp_path/'audit'), \
         patch.object(m, 'O', tmp_path/'owner'), patch.object(m.os, 'sched_setaffinity') as change:
        assert m.stopped() is True
    change.assert_not_called()


@pytest.mark.parametrize('post_intent', [False, True])
def test_main_preintent_race_can_continue_but_postintent_error_ends_helper(tmp_path, post_intent):
    audit = tmp_path/'audit'
    rows, attempts, sleeps = [], [], []
    def probe(row):
        attempts.append(row)
        if post_intent:
            (audit/'wave43_intent.json').write_text('{}')
        raise FileNotFoundError('synthetic dispatch race')
    def stopped():
        return len(sleeps) == 1
    def write(path, row):
        rows.append((Path(path).name, deepcopy(row)))
    with patch.object(m, 'D', audit), patch.object(m, 'source_guard'), \
         patch.object(m.Path, 'cwd', return_value=m.B), \
         patch.object(m, 'read', return_value=status()), patch.object(m, 'digest', return_value='fixture'), \
         patch.object(m.psutil, 'Process') as process, \
         patch.object(m, 'write', side_effect=write), patch.object(m, 'probe', side_effect=probe), \
         patch.object(m, 'stopped', side_effect=stopped), patch.object(m.time, 'time', return_value=200.), \
         patch.object(m.time, 'sleep', side_effect=lambda delay: sleeps.append(delay)), \
         patch.object(m.signal, 'signal'), patch('sys.argv', ['fixture', '--apply']):
        process.return_value.create_time.return_value = 100.
        process.return_value.cmdline.return_value = ['fixture', '--apply']
        process.return_value.cwd.return_value = str(m.B)
        if post_intent:
            with pytest.raises(FileNotFoundError, match='dispatch race'):
                m.main()
        else:
            m.main()
    assert len(attempts) == 1
    assert len(sleeps) == (0 if post_intent else 1)
    assert rows[-1][0] == 'exit.json'
    assert bool(rows[-1][1]['error']) is post_intent
    assert rows[-1][1]['durable_attempts'] == (['wave43_intent.json'] if post_intent else [])


def test_main_durable_noop_is_attempted_only_once_and_signal_handler_only_sets_flag(tmp_path):
    audit = tmp_path/'audit'
    probes, sleeps, handlers = [], [], {}
    row = status()
    def probe(value):
        probes.append(value)
        (audit/'wave43_intent.json').write_text('{}')
        return {'no_affinity_writes': True}
    def stopped():
        return len(sleeps) >= 2
    with patch.object(m, 'D', audit), patch.object(m, 'source_guard'), \
         patch.object(m.Path, 'cwd', return_value=m.B), \
         patch.object(m, 'read', return_value=row), patch.object(m, 'digest', return_value='fixture'), \
         patch.object(m.psutil, 'Process') as process, patch.object(m, 'write'), \
         patch.object(m, 'probe', side_effect=probe), patch.object(m, 'stopped', side_effect=stopped), \
         patch.object(m.time, 'time', return_value=200.), \
         patch.object(m.time, 'sleep', side_effect=lambda delay: sleeps.append(delay)), \
         patch.object(m.signal, 'signal', side_effect=lambda sig, handler: handlers.setdefault(sig, handler)), \
         patch.object(m, 'STOP_REQUESTED', False), patch.object(m.os, 'sched_setaffinity') as change, \
         patch('sys.argv', ['fixture', '--apply']):
        process.return_value.create_time.return_value = 100.
        process.return_value.cmdline.return_value = ['fixture', '--apply']
        process.return_value.cwd.return_value = str(m.B)
        m.main()
        assert len(probes) == 1
        assert set(handlers) == {m.signal.SIGTERM, m.signal.SIGHUP}
        handlers[m.signal.SIGTERM](m.signal.SIGTERM, None)
        assert m.STOP_REQUESTED is True
    change.assert_not_called()
