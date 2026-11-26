"""Independent CPU-only regressions; no actual process mutation or payload I/O."""
import copy
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location('independent_capacity18', Path(__file__).with_name('host_init_capacity18.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class P:
    pid = 123
    def create_time(self): return 100.
    def is_running(self): return True
    def status(self): return 'running'
    def cmdline(self): return [m.PYTHON, '-m', m.MODULE, 'worker', '--output', str(m.TEST/'wave43')]
    def cwd(self): return str(m.B)
    def nice(self): return 10


def status():
    return dict(pid=123, stage=26, utc=100.1, command=P().cmdline())


@contextmanager
def binding_fixture():
    """Pure synthetic lifecycle graph; every JSON/digest/process read replaced."""
    s = status(); wave = m.TEST/'wave43'
    command = P().cmdline()
    job = dict(resource_root=str(m.O), operation='evaluate', stages=[{} for _ in range(47)])
    job['stages'][26] = dict(entry='paired_evaluate_continuation15', args=command[3:])
    commands = [[] for _ in range(47)]; commands[26] = command
    rows = {
        m.REGISTRY: dict(owner_identity=dict(pid=10, created=90.)),
        m.REGISTRY15: dict(job_sha256='job', original_test_registration_sha256='freeze',
            eligible_indices=list(range(17, 64)), frozen_pairs={'A': 'a', 'B': 'b'}, original13_registry_sha256='reg13'),
        m.REVIEW15: dict(status='source_and_actual_missing47_review_passed_inactive', registration_sha256='reg15'),
        m.ACTIVATED15: dict(registry_sha256='reg15', root_review_sha256='review15'),
        m.JOB: job,
        m.O/'project_job_started.json': dict(pid=10, utc=90.1, job=job, commands=commands),
        m.O/'project_status.json': s,
        m.O/'launches/wave43.json': dict(stage=26, index=43, command=command, registration_sha256='reg15',
            dispatch_attempt=1, rerun=False, utc=99.9),
        m.O/'dispatches/wave43.json': dict(pid=123, index=43, wave=str(wave), dispatch_attempt=1,
            rerun=False, registration_sha256='reg15', original13_registration_sha256='reg13',
            launch_sha256='launch', plan_sha256='plan', utc=100.2),
        m.TEST/'registration.json': dict(wave_plan_sha256={str(wave): 'plan'}),
        wave/'plan.json': dict(pairs={'A': 'a', 'B': 'b'}, resource_root=str(m.TEST)),
    }
    hashes = {m.REGISTRY15: 'reg15', m.REVIEW15: 'review15', m.JOB: 'job',
              m.TEST/'registration.json': 'freeze', wave/'plan.json': 'plan',
              m.O/'launches/wave43.json': 'launch', m.O/'dispatches/wave43.json': 'dispatch'}
    processes = {
        10: dict(pid=10, created=90., ppid=1, command=[m.PYTHON, '-m', m.MODULE, 'project', 'evaluate', '--job', str(m.JOB)], cwd=str(m.B)),
        123: dict(pid=123, created=100., ppid=10, command=command, cwd=str(m.B)),
    }
    with patch.object(m, 'read', side_effect=lambda p: copy.deepcopy(rows[Path(p)])), \
         patch.object(m, 'digest', side_effect=lambda p: hashes[Path(p)]), \
         patch.object(m, 'process_row', side_effect=lambda p: copy.deepcopy(processes[p])):
        yield s, rows, processes


def test_independent_exact_synthetic_binding_passes():
    with binding_fixture() as (s, _, __):
        assert m.identity(s, P(), 100., m.TEST/'wave43')['index'] == 43


def test_independent_reused_worker_born_after_old_status_rejected():
    with binding_fixture() as (s, _, __):
        s['utc'] = 99.
        with pytest.raises(ValueError, match='Live worker/status'):
            m.identity(s, P(), 100., m.TEST/'wave43')


@pytest.mark.parametrize('field,value', [('launch_sha256', 'other'), ('pid', 124), ('dispatch_attempt', 2), ('rerun', True)])
def test_independent_dispatch_fence_change_rejected(field, value):
    with binding_fixture() as (s, rows, _):
        rows[m.O/'dispatches/wave43.json'][field] = value
        with pytest.raises(ValueError, match='dispatch fence'):
            m.identity(s, P(), 100., m.TEST/'wave43')


def test_independent_owner_pid_reuse_rejected():
    with binding_fixture() as (s, _, processes):
        processes[10]['created'] = 91.
        with pytest.raises(ValueError, match='owner ancestry'):
            m.identity(s, P(), 100., m.TEST/'wave43')


def test_independent_thread_tgid_mismatch_rejected():
    with patch.object(m, 'same', return_value=True), \
         patch.object(Path, 'read_text', side_effect=['Tgid:\t999\n', '123 (worker) '+ ' '.join(['0']*20)]):
        assert m.thread_token(123, P(), 100.) is None


def test_independent_reused_thread_token_makes_no_write():
    row = dict(mutation_attempted=False, affinity_writes=0, expansion_writes=0, restoration_writes=0)
    with patch.object(m, 'thread_token', return_value=(123, 999)), \
         patch.object(m.os, 'sched_setaffinity') as change:
        assert m.set_mask((123, 100), P(), 100., m.OLD|{1}, row) is False
    change.assert_not_called()
    assert row['mutation_attempted'] is False


@contextmanager
def probe_fixture(ready_values, restore=None):
    locked = [False]; writes = []; changes = []; p = P()
    @contextmanager
    def lock():
        assert not locked[0]
        locked[0] = True
        try: yield
        finally: locked[0] = False
    def change(token, proc, birth, cpus, row, **kwargs):
        assert locked[0]
        row['mutation_attempted'] = True
        row['affinity_writes'] += 1
        changes.append(set(cpus))
        return True
    def checked_restore(proc, birth, wide, row):
        assert locked[0]
        if restore: return restore(proc, birth, wide, row)
        row['restored'] = True
    with patch.object(m.psutil, 'Process', return_value=p), \
         patch.object(m, 'identity', return_value={}), patch.object(m, 'stopped', return_value=False), \
         patch.object(m, 'ready', side_effect=ready_values), \
         patch.object(m, 'read', return_value={'current_stage': 'environment_constructor_started'}), \
         patch.object(m, 'spare_samples', return_value=([1], [{1: 0.}])), \
         patch.object(m, 'lifecycle_lock', side_effect=lock), patch.object(m, 'source_guard'), \
         patch.object(m, 'masks', return_value=[((123, 1), m.OLD, 10)]), \
         patch.object(m, 'write', side_effect=lambda p, row: writes.append((Path(p).name, copy.deepcopy(row)))), \
         patch.object(m, 'set_mask', side_effect=change), patch.object(m, 'restore', side_effect=checked_restore), \
         patch.object(m, 'external_snapshot', return_value={}), patch.object(m, 'same', return_value=True), \
         patch.object(m.time, 'time', return_value=200.), patch.object(m.time, 'monotonic', return_value=1.):
        yield locked, writes, changes


def test_independent_readiness_race_zero_affinity_and_restore_writes():
    with probe_fixture([False, True]) as (locked, writes, changes):
        row = m.probe(status())
    assert changes == [] and row['no_affinity_writes'] is True
    assert [p for p, _ in writes] == ['wave43_intent.json', 'wave43_exit.json']
    assert locked == [False]


def test_independent_capacity_disappears_zero_affinity_and_restore_writes():
    with probe_fixture([False, False]) as (_, __, changes), \
         patch.object(m, 'spare_samples', side_effect=[([1], [{1: 0.}]), ([], [{1: 100.}])]):
        row = m.probe(status())
    assert changes == [] and row['no_affinity_writes'] is True


def test_independent_restore_failure_still_exit_receipt_under_lock():
    failures = []
    def fail_restore(*args): raise ValueError('injected restore failure')
    with probe_fixture([False, False, False, False, True], fail_restore) as (locked, writes, changes), \
         patch.object(m, 'wait_failed_restore', side_effect=lambda *args: failures.append(locked[0])):
        with pytest.raises(RuntimeError, match='Restore failed'):
            m.probe(status())
    assert len(changes) == 1 and failures == [True]
    assert writes[-1][0] == 'wave43_exit.json' and writes[-1][1]['restored'] is False
    assert locked == [False]


def test_independent_failed_restore_io_error_cannot_skip_global_stop_and_wait(tmp_path):
    seen = []
    def same(proc, birth):
        seen.append(True)
        return len(seen) == 1
    row = dict(wave=str(m.TEST/'wave43'))
    with patch.object(m, 'O', tmp_path), patch.object(m, 'D', tmp_path), \
         patch.object(m, 'write', side_effect=OSError('injected evidence I/O failure')), \
         patch.object(m, 'same', side_effect=same), patch.object(m.time, 'sleep'):
        try: m.wait_failed_restore(P(), 100., row)
        except OSError: pass  # Error may be reported after confirmed exit.
    assert (tmp_path/'STOP').exists(), 'Cooperative owner STOP must survive evidence failure'
    assert len(seen) >= 2, 'Unrestored worker must be observed gone before helper can unlock'


def test_independent_failed_restore_identity_error_cannot_unlock(tmp_path):
    seen = []
    def same(proc, birth):
        seen.append(True)
        if len(seen) == 1: raise m.psutil.AccessDenied(123)
        return False
    with patch.object(m, 'O', tmp_path), patch.object(m, 'D', tmp_path), \
         patch.object(m, 'write'), patch.object(m, 'sync_directory'), \
         patch.object(m, 'same', side_effect=same), patch.object(m.time, 'sleep'):
        try: m.wait_failed_restore(P(), 100., dict(wave=str(m.TEST/'wave43')))
        except m.psutil.AccessDenied: pass
    assert len(seen) >= 2, 'Identity inspection failure is not confirmed worker exit'


def test_independent_external_new_pid_or_reused_birth_retracts():
    assert m.external_busy({(123, 100.): 1.}, {(123, 200.): 1.}, 1.) is True
    assert m.external_busy({(123, 100.): 1.}, {(123, 100.): 1.3}, 1.) is True
    assert m.external_busy({(123, 100.): 1.}, {(123, 100.): 1.1}, 1.) is False


def test_independent_partial_expansion_failure_still_restores_under_lock():
    checks = []; count = []
    def restored(proc, birth, wide, row):
        checks.append((row['mutation_attempted'], row['affinity_writes']))
        row['restored'] = True
    def partial(token, proc, birth, cpus, row):
        row['mutation_attempted'] = True
        count.append(token)
        if len(count) == 2: raise OSError('injected second-thread expansion failure')
        row['affinity_writes'] += 1
        return True
    with probe_fixture([False]*5, restored) as (locked, writes, _), \
         patch.object(m, 'masks', return_value=[((123, 1), m.OLD, 10), ((124, 1), m.OLD, 10)]), \
         patch.object(m, 'set_mask', side_effect=partial):
        with pytest.raises(OSError, match='second-thread'):
            m.probe(status())
    assert len(count) == 2 and checks == [(True, 1)]
    assert writes[-1][0] == 'wave43_exit.json' and writes[-1][1]['restored'] is True
    assert locked == [False]


def test_independent_cap_checks_between_thread_writes_and_records_actual_elapsed():
    ticks = iter([0., 0., 0., 41., 41., 42.])
    with probe_fixture([False]*4) as (locked, writes, changes), \
         patch.object(m, 'masks', return_value=[((123, 1), m.OLD, 10), ((124, 1), m.OLD, 10)]), \
         patch.object(m.time, 'monotonic', side_effect=lambda: next(ticks)):
        row = m.probe(status())
    assert len(changes) == 1 and row['actual_placement_elapsed_seconds'] == 42.
    assert row['cap_seconds'] == 40. and row['restored'] is True
    assert locked == [False]


def test_independent_restore_covers_thread_created_after_expansion():
    old = m.OLD; wide = old|{1}; changes = []
    snapshot = [((123, 1), old, 10), ((124, 2), wide, 10)]
    restored = [((123, 1), old, 10), ((124, 2), old, 10)]
    row = dict(mutation_attempted=True, affinity_writes=1, restoration_writes=0, expansion_writes=1)
    def changed(token, proc, birth, mask, receipt, **kwargs):
        assert kwargs == {'restore': True}
        changes.append((token, mask))
        return True
    with patch.object(m, 'same', return_value=True), \
         patch.object(m, 'masks', side_effect=[snapshot, restored]), patch.object(m, 'set_mask', side_effect=changed):
        m.restore(P(), 100., wide, row)
    assert changes == [((124, 2), old)] and row['restored'] is True


def test_independent_source_guard_pin_drift_fails_without_any_affinity_write():
    hashes = {str(Path(m.__file__).resolve()): 'self', **{str(p): h for p, h in m.PINNED.items()}}
    artifacts = {str(p): 'binding-'+p.name for p in (m.REGISTRY15, m.REVIEW15, m.ACTIVATED15, m.JOB, m.TEST/'registration.json')}
    registration = dict(version='host_init_capacity18_v1', eligible_indices=m.ELIGIBLE, deadline=m.DEADLINE,
        audit=str(m.D), candidate_cpus=sorted(m.CANDIDATES), maximum_extra_cpus=m.MAX_EXTRA,
        cap_seconds=m.CAP, minimum_worker_age_seconds=m.MIN_AGE, physics_initialization_overlap_allowed=True,
        source_sha256=dict(hashes), binding_sha256=artifacts, owner_identity=dict(pid=10, created=90.))
    digests = {**hashes, **artifacts, str(m.REGISTRY): 'reg18'}
    review = dict(status='inactive_root_reviewed_for_activation', registration_sha256='reg18')
    with patch.object(m, 'read', side_effect=lambda p: registration if p == m.REGISTRY else review), \
         patch.object(m, 'digest', side_effect=lambda p: digests[str(p)]), \
         patch.object(m.os, 'sched_setaffinity') as mutation:
        assert m.source_guard() == registration
        registration['source_sha256'][str(Path(m.__file__).resolve())] = 'changed'
        with pytest.raises(ValueError, match='source pin changed'):
            m.source_guard()
    mutation.assert_not_called()


@pytest.mark.parametrize('birth,utc', [(True, 1.), (100., True), (float('nan'), 100.), (100., float('inf')), (0., 1.), (100., 99.9)])
def test_independent_invalid_birth_window_rejected(birth, utc):
    assert m.receipt_birth(birth, utc) is False
