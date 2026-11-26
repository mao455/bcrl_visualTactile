"""CPU-only synthetic host-init tests; every process/affinity operation mocked."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

spec = importlib.util.spec_from_file_location('host_init19_author', Path(__file__).with_name('host_init_capacity19.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Clock:
    value = 1000.
    def time(self): return self.value
    def monotonic(self): return self.value-1000.
    def sleep(self, seconds): self.value += seconds


class Proc:
    pid = 123
    birth = 900.
    live = True
    def create_time(self): return self.birth
    def is_running(self): return self.live
    def status(self): return 'sleeping'
    def nice(self): return 10
    def cmdline(self): return [m.PYTHON, '-m', m.MODULE, 'worker', '--output', str(m.TEST/'wave43')]
    def cwd(self): return str(m.B)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    clock = Clock()
    p = Proc()
    monkeypatch.setattr(m, 'D', tmp_path/'audit'); m.D.mkdir()
    monkeypatch.setattr(m, 'O', tmp_path/'owner'); m.O.mkdir()
    monkeypatch.setattr(m.time, 'time', clock.time)
    monkeypatch.setattr(m.time, 'monotonic', clock.monotonic)
    monkeypatch.setattr(m.time, 'sleep', clock.sleep)
    monkeypatch.setattr(m.psutil, 'Process', lambda pid: p)
    events, writes = [], []
    state = SimpleNamespace(locked=False, masks={(123, 10): set(m.OLD), (124, 11): set(m.OLD)},
                            ready=lambda: False, source_calls=0, writes=writes, events=events, clock=clock, proc=p)
    @contextmanager
    def lock():
        assert not state.locked
        state.locked = True
        events.append('lock')
        try: yield
        finally:
            events.append('unlock')
            state.locked = False
    def guard():
        state.source_calls += 1
    def observe(proc, birth):
        return [(token, set(cpus), 10) for token, cpus in state.masks.items()]
    def setmask(token, proc, birth, cpus, row, *, restore=False):
        assert state.locked
        row['mutation_attempted'] = True
        row['affinity_writes'] += 1
        row['restoration_writes' if restore else 'expansion_writes'] += 1
        events.append(('restore' if restore else 'expand', token))
        state.masks[token] = set(cpus)
        return True
    def write(path, row):
        assert state.locked
        writes.append((path.name, deepcopy(row)))
    monkeypatch.setattr(m, 'identity', lambda *args: {'index': 43})
    monkeypatch.setattr(m, 'source_guard', guard)
    monkeypatch.setattr(m, 'same', lambda proc, birth: p.live and p.birth == birth)
    monkeypatch.setattr(m, 'ready', lambda wave: state.ready())
    monkeypatch.setattr(m, 'read', lambda path: {'current_stage': 'environment_constructor_started'})
    monkeypatch.setattr(m, 'spare_samples', lambda **kw: ([1, 4], [{1: 0., 2: 20., 4: 1., 5: 30.}]))
    monkeypatch.setattr(m, 'external_snapshot', lambda *args: {(888, 500.): 0.})
    monkeypatch.setattr(m, 'lifecycle_lock', lock)
    monkeypatch.setattr(m, 'masks', observe)
    monkeypatch.setattr(m, 'set_mask', setmask)
    monkeypatch.setattr(m, 'write', write)
    state.status = dict(stage=26, pid=123, utc=900., command=p.cmdline())
    return state


def test_successful_extension_lock_cap_and_restore(harness):
    row = m.probe(harness.status)
    assert row['restored'] is True
    assert row['actual_placement_elapsed_seconds'] == pytest.approx(40.)
    assert row['chosen'] == [1, 4]
    assert row['expansion_writes'] == row['restoration_writes'] == 2
    assert harness.events[0] == 'lock' and harness.events[-1] == 'unlock'
    assert all(cpus == m.OLD for cpus in harness.masks.values())
    assert [name for name, _ in harness.writes] == ['wave43_intent.json', 'wave43_expansion_intent.json', 'wave43_exit.json']
    assert harness.writes[1][1]['chosen'] == [1, 4]
    assert len(harness.writes[1][1]['original_threads']) == 2


def test_readiness_race_has_zero_affinity_writes(harness):
    calls = iter([False, True])
    harness.ready = lambda: next(calls)
    row = m.probe(harness.status)
    assert row['no_affinity_writes'] is True
    assert row['affinity_writes'] == 0
    assert not any(isinstance(event, tuple) for event in harness.events)
    assert [name for name, _ in harness.writes] == ['wave43_intent.json', 'wave43_exit.json']


def test_capacity_race_has_zero_affinity_writes(harness, monkeypatch):
    monkeypatch.setattr(m, 'spare_samples', Mock(side_effect=[([1, 4], []), ([], [])]))
    row = m.probe(harness.status)
    assert row['affinity_writes'] == 0 and row['no_affinity_writes'] is True
    assert row['no_change_reason'] == 'capacity_stop_or_readiness_race'


def test_aux_stop_after_lock_has_zero_affinity_writes(harness, monkeypatch):
    monkeypatch.setattr(m, 'stopped', Mock(side_effect=[False, True]))
    row = m.probe(harness.status)
    assert row['affinity_writes'] == 0


def test_source_drift_after_lock_has_exit_and_zero_writes(harness, monkeypatch):
    monkeypatch.setattr(m, 'source_guard', Mock(side_effect=ValueError('pin drift')))
    with pytest.raises(ValueError, match='pin drift'):
        m.probe(harness.status)
    assert harness.writes[-1][0] == 'wave43_exit.json'
    assert harness.writes[-1][1]['affinity_writes'] == 0


def test_source_drift_immediately_before_mutation_has_no_writes(harness, monkeypatch):
    monkeypatch.setattr(m, 'source_guard', Mock(side_effect=[None, None, ValueError('late pin drift')]))
    with pytest.raises(ValueError, match='late pin drift'):
        m.probe(harness.status)
    assert harness.writes[-1][1]['affinity_writes'] == 0


def test_partial_expansion_failure_restores_and_writes_exit(harness, monkeypatch):
    original = m.set_mask
    count = 0
    def broken(token, proc, birth, cpus, row, *, restore=False):
        nonlocal count
        result = original(token, proc, birth, cpus, row, restore=restore)
        if not restore:
            count += 1
            if count == 1:
                raise OSError('partial mutation')
        return result
    monkeypatch.setattr(m, 'set_mask', broken)
    with pytest.raises(OSError, match='partial mutation'):
        m.probe(harness.status)
    assert all(cpus == m.OLD for cpus in harness.masks.values())
    assert harness.writes[-1][1]['restored'] is True
    assert harness.writes[-1][0] == 'wave43_exit.json'


def test_restore_failure_writes_exit_stop_and_holds_lock_until_exit(harness, monkeypatch):
    monkeypatch.setattr(m, 'restore', Mock(side_effect=ValueError('restoration failed')))
    sleeps = 0
    def sleep(seconds):
        nonlocal sleeps
        harness.clock.value += seconds
        if (m.O/'STOP').exists():
            assert harness.locked
            sleeps += 1
            if sleeps == 2:
                harness.proc.live = False
    monkeypatch.setattr(m.time, 'sleep', sleep)
    with pytest.raises(RuntimeError, match='Restore failed'):
        m.probe(harness.status)
    assert (m.O/'STOP').exists()
    assert not (m.O/'startup_stops').exists()
    names = [name for name, _ in harness.writes]
    assert names[-3:] == ['wave43_exit.json', 'wave43_infrastructure_abort.json', 'wave43_abort_worker_exited.json']
    assert harness.writes[-3][1]['restored'] is False
    assert sleeps == 2 and harness.events[-1] == 'unlock'


def test_newly_born_thread_is_restored(harness, monkeypatch):
    original = m.external_snapshot
    def snapshot(*args):
        harness.masks[(129, 99)] = m.OLD | {1, 4}
        return original(*args)
    monkeypatch.setattr(m, 'external_snapshot', snapshot)
    row = m.probe(harness.status)
    assert row['restored'] is True
    assert harness.masks[(129, 99)] == m.OLD
    assert row['restored_thread_count'] == 3


def test_worker_exit_requires_no_restoration(harness, monkeypatch):
    calls = 0
    def snapshot(*args):
        nonlocal calls
        calls += 1
        if calls > 1:
            harness.proc.live = False
        return {}
    monkeypatch.setattr(m, 'external_snapshot', snapshot)
    row = m.probe(harness.status)
    assert row['worker_exited'] is True and row['restored'] is None
    assert row['restoration_writes'] == 0


def test_external_cpu_rise_retracts_early(harness, monkeypatch):
    monkeypatch.setattr(m, 'external_snapshot', Mock(side_effect=[{(888, 500.): 0.}, {(888, 500.): 1.}]))
    row = m.probe(harness.status)
    assert row['retracted_for_external_cpu_activity'] is True
    assert row['actual_placement_elapsed_seconds'] < 1.
    assert row['restored'] is True


def test_cap_includes_individual_expansion_writes(harness, monkeypatch):
    original = m.set_mask
    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        if not kwargs.get('restore'):
            harness.clock.value += 40.1
        return result
    monkeypatch.setattr(m, 'set_mask', slow)
    row = m.probe(harness.status)
    assert row['expansion_writes'] == 1
    assert row['actual_placement_elapsed_seconds'] == pytest.approx(40.1)
    assert row['restored'] is True


def test_failed_first_affinity_syscall_still_enters_restore(monkeypatch):
    p = Proc()
    row = dict(mutation_attempted=False, affinity_writes=0, expansion_writes=0, restoration_writes=0)
    monkeypatch.setattr(m, 'thread_token', lambda *args: (123, 10))
    monkeypatch.setattr(m.os, 'getpriority', lambda *args: 10)
    setter = Mock(side_effect=OSError('kernel affinity failed'))
    monkeypatch.setattr(m.os, 'sched_setaffinity', setter)
    with pytest.raises(OSError):
        m.set_mask((123, 10), p, 900., m.OLD | {1}, row)
    assert row['mutation_attempted'] is True
    assert row['affinity_writes'] == 0


@pytest.mark.parametrize('token', [None, (123, 11), (999, 10)])
def test_recycled_or_foreign_tid_is_never_written(monkeypatch, token):
    monkeypatch.setattr(m, 'thread_token', lambda *args: token)
    setter = Mock()
    monkeypatch.setattr(m.os, 'sched_setaffinity', setter)
    row = dict(mutation_attempted=False, affinity_writes=0, expansion_writes=0, restoration_writes=0)
    assert m.set_mask((123, 10), Proc(), 900., m.OLD, row, restore=True) is False
    setter.assert_not_called()
    assert not row['mutation_attempted']


@pytest.mark.parametrize('mapped,utc', [
    (True, 100.), ({'lower':100., 'upper':100.01}, True),
    ({'lower':float('nan'), 'upper':100.01},100.), ({'lower':100., 'upper':float('inf')},100.),
    ({'lower':-1., 'upper':100.},100.), ({'lower':101., 'upper':101.01},100.),
    ({'lower':1., 'upper':1.01},130.)])
def test_invalid_or_reused_pid_birth_window_rejected(mapped, utc):
    assert m.receipt_birth(mapped, utc) is False


def test_mapped_receipt_quantization_and_launch_chronology():
    mapped = dict(lower=100., upper=100.01)
    assert m.receipt_birth(mapped,100.005)
    assert m.receipt_birth(mapped,99.99)
    assert not m.receipt_birth(mapped,99.95)
    assert m.launch_birth(mapped,100.005)
    assert m.launch_birth(mapped,100.02)
    assert not m.launch_birth(mapped,100.06)


def test_external_snapshot_aggregate_and_new_identity_retract():
    assert m.external_busy({(1, 1.): 0.}, {(1, 1.): .3}, 1.)
    assert m.external_busy({(1, 1.): 0.}, {(1, 1.): .2, (2, 2.): 0.}, 1.)
    assert m.external_busy({(1, 1.): 0., (2, 2.): 0.}, {(1, 1.): .2, (2, 2.): .2}, 1.)
    assert not m.external_busy({(1, 1.): 0.}, {(1, 1.): .1}, 1.)
    with pytest.raises(ValueError):
        m.external_busy({(1, 1.): 1.}, {(1, 1.): 0.}, 1.)


@pytest.fixture
def bound_identity(monkeypatch, tmp_path):
    paths = dict(B=tmp_path, A=tmp_path/'audit', R=tmp_path/'run')
    paths.update(O=paths['R']/'owners/test_continuation15', TEST=paths['R']/'evaluation/test')
    paths.update(REGISTRY=paths['A']/'reg18.json', REGISTRY15=paths['A']/'reg15.json', REVIEW15=paths['A']/'review15.json',
                 ACTIVATED15=paths['A']/'active15.json', JOB=paths['A']/'job.json')
    for name, path in paths.items(): monkeypatch.setattr(m, name, path)
    wave = m.TEST/'wave43'
    wave.mkdir(parents=True)
    m.A.mkdir()
    (m.O/'launches').mkdir(parents=True); (m.O/'dispatches').mkdir()
    p = Proc()
    status = dict(pid=123, utc=900.1, stage=26, command=p.cmdline())
    owner = dict(pid=111, created=800., ppid=1, command=[m.PYTHON, '-m', m.MODULE, 'project', 'evaluate', '--job', str(m.JOB)], cwd=str(m.B))
    def put(path, value): path.write_text(json.dumps(value))
    job = dict(resource_root=str(m.O), operation='evaluate', stages=[{} for _ in range(47)])
    job['stages'][26] = dict(entry='paired_evaluate_continuation15', args=status['command'][3:])
    put(m.JOB, job)
    put(wave/'plan.json', dict(resource_root=str(m.TEST), pairs=['frozen']))
    put(m.TEST/'registration.json', dict(wave_plan_sha256={str(wave): m.digest(wave/'plan.json')}))
    put(m.REGISTRY15, dict(job_sha256=m.digest(m.JOB), original_test_registration_sha256=m.digest(m.TEST/'registration.json'),
         eligible_indices=list(range(17, 64)), frozen_pairs=['frozen'], original13_registry_sha256='13'*32))
    put(m.REVIEW15, dict(status='source_and_actual_missing47_review_passed_inactive', registration_sha256=m.digest(m.REGISTRY15)))
    put(m.ACTIVATED15, dict(registry_sha256=m.digest(m.REGISTRY15), root_review_sha256=m.digest(m.REVIEW15)))
    put(m.REGISTRY, dict(owner_identity={k:owner[k] for k in ('pid', 'created')}))
    commands = [[] for _ in range(47)]; commands[26] = status['command']
    put(m.O/'project_job_started.json', dict(pid=111, utc=800.1, job=job, commands=commands))
    put(m.O/'project_status.json', status)
    launch_path = m.O/'launches/wave43.json'
    put(launch_path, dict(stage=26, index=43, command=status['command'], registration_sha256=m.digest(m.REGISTRY15),
                         dispatch_attempt=1, rerun=False, utc=899.9))
    put(m.O/'dispatches/wave43.json', dict(pid=123, index=43, wave=str(wave), dispatch_attempt=1, rerun=False,
         registration_sha256=m.digest(m.REGISTRY15), original13_registration_sha256='13'*32,
         launch_sha256=m.digest(launch_path), plan_sha256=m.digest(wave/'plan.json'), utc=900.1))
    monkeypatch.setattr(m.psutil, 'Process', lambda pid: SimpleNamespace(pid=pid))
    monkeypatch.setattr(m, 'mapped_birth', lambda proc, raw: dict(lower=raw, upper=raw+.01))
    monkeypatch.setattr(m, 'process_row', lambda pid: owner if pid == 111 else dict(pid=123, created=p.birth, ppid=111, command=p.cmdline(), cwd=str(m.B)))
    return SimpleNamespace(p=p, status=status, wave=wave, put=put, owner=owner)


def test_full_identity_fixture_passes(bound_identity):
    x = bound_identity
    assert m.identity(x.status, x.p, x.p.birth, x.wave)['index'] == 43


@pytest.mark.parametrize('case', ['status_future_birth', 'owner_reused', 'launch_birth', 'launch_command', 'dispatch_pid',
                                 'dispatch_launch', 'dispatch_plan', 'dispatch_rerun', 'plan_pairs', 'status_stage',
                                 'status_pid_bool', 'owner_command', 'job_stage', 'job_started', 'source_registration'])
def test_exact_binding_rejects_each_identity_drift(bound_identity, case):
    x = bound_identity
    target = None
    if case == 'status_future_birth': x.p.birth = x.status['utc']+1.
    elif case == 'owner_reused': x.owner['created'] += 1.
    elif case == 'status_stage': x.status['stage'] = True
    elif case == 'status_pid_bool': x.status['pid'] = True
    elif case == 'owner_command': x.owner['command'][-1] = 'other'
    else:
        path, key, value = {
            'launch_birth': (m.O/'launches/wave43.json', 'utc', 901.),
            'launch_command': (m.O/'launches/wave43.json', 'command', ['other']),
            'dispatch_pid': (m.O/'dispatches/wave43.json', 'pid', 999),
            'dispatch_launch': (m.O/'dispatches/wave43.json', 'launch_sha256', 'bad'),
            'dispatch_plan': (m.O/'dispatches/wave43.json', 'plan_sha256', 'bad'),
            'dispatch_rerun': (m.O/'dispatches/wave43.json', 'rerun', True),
            'plan_pairs': (x.wave/'plan.json', 'pairs', ['unfrozen']),
            'job_stage': (m.JOB, 'stages', []),
            'job_started': (m.O/'project_job_started.json', 'job', {}),
            'source_registration': (m.REVIEW15, 'registration_sha256', 'bad'),
        }[case]
        value_row = m.read(path); value_row[key] = value; x.put(path, value_row)
    with pytest.raises((ValueError, IndexError, KeyError)):
        m.identity(x.status, x.p, x.p.birth, x.wave)


@pytest.fixture
def source_fixture(monkeypatch, tmp_path):
    me = str(Path(m.__file__).resolve())
    dependency = tmp_path/'frozen.py'
    binding_paths = [tmp_path/name for name in ['reg15', 'review15', 'active15', 'job', 'freeze']]
    for key, path in zip(['REGISTRY15', 'REVIEW15', 'ACTIVATED15', 'JOB'], binding_paths[:4]):
        monkeypatch.setattr(m, key, path)
    monkeypatch.setattr(m, 'TEST', tmp_path/'test')
    binding_paths[-1] = m.TEST/'registration.json'
    monkeypatch.setattr(m, 'PINNED', {dependency:'dep_hash'})
    registry = dict(version='host_init_capacity19_v1', eligible_indices=m.ELIGIBLE, deadline=m.DEADLINE,
                    audit=str(m.D), candidate_cpus=sorted(m.CANDIDATES), maximum_extra_cpus=m.MAX_EXTRA,
                    cap_seconds=m.CAP, minimum_worker_age_seconds=m.MIN_AGE, physics_initialization_overlap_allowed=True,
                    source_sha256={me:'self_hash', str(dependency):'dep_hash'},
                    binding_sha256={str(path):'binding_hash' for path in binding_paths},
                    owner_identity=dict(pid=111, created=800.))
    review = dict(status='inactive_root_reviewed_for_activation', registration_sha256='reg_hash')
    def get(path):
        if path == m.REGISTRY: return registry
        if path == m.REVIEW: return review
        raise AssertionError('unexpected science or lifecycle read '+str(path))
    hashes = {me:'self_hash', str(dependency):'dep_hash', str(m.REGISTRY):'reg_hash',
              **{str(path):'binding_hash' for path in binding_paths}}
    monkeypatch.setattr(m, 'read', get)
    monkeypatch.setattr(m, 'digest', lambda path: hashes[str(path)])
    return SimpleNamespace(registry=registry, review=review, hashes=hashes, me=me, dependency=dependency)


def test_source_guard_exact_self_dependency_and_lifecycle_pins_pass(source_fixture):
    assert m.source_guard() is source_fixture.registry


@pytest.mark.parametrize('case', ['self_drift', 'dep_drift', 'missing_self', 'extra_source', 'binding_drift',
                                 'review_unbound', 'wrong_indices', 'wrong_owner', 'no_physics_overlap'])
def test_source_guard_rejects_pin_or_policy_drift(source_fixture, case):
    x = source_fixture
    if case == 'self_drift': x.hashes[x.me] = 'changed'
    elif case == 'dep_drift': x.hashes[str(x.dependency)] = 'changed'
    elif case == 'missing_self': del x.registry['source_sha256'][x.me]
    elif case == 'extra_source': x.registry['source_sha256']['extra'] = 'bad'
    elif case == 'binding_drift': x.hashes[str(m.JOB)] = 'changed'
    elif case == 'review_unbound': x.review['registration_sha256'] = 'unbound'
    elif case == 'wrong_indices': x.registry['eligible_indices'] = [43]
    elif case == 'wrong_owner': x.registry['owner_identity']['pid'] = True
    elif case == 'no_physics_overlap': x.registry['physics_initialization_overlap_allowed'] = False
    with pytest.raises(ValueError): m.source_guard()


def test_any_failed_restore_requests_global_stop_even_if_target_already_gone(harness, monkeypatch):
    harness.proc.live = False
    with m.lifecycle_lock():
        m.wait_failed_restore(harness.proc, harness.proc.birth, dict(wave=str(m.TEST/'wave43')))
    assert (m.O/'STOP').exists()
    assert [name for name, _ in harness.writes] == ['wave43_infrastructure_abort.json', 'wave43_abort_worker_exited.json']


def test_lock_wait_aux_stop_never_acquires_or_mutates(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'O', tmp_path)
    monkeypatch.setattr(m, 'stopped', lambda: True)
    monkeypatch.setattr(m.fcntl, 'flock', Mock(side_effect=BlockingIOError))
    with pytest.raises(InterruptedError):
        with m.lifecycle_lock():
            pytest.fail('Must not acquire lock after STOP')


def test_integer_btime_fraction_is_mapped_without_arbitrary_launch_slack(monkeypatch):
    p = Proc()
    raw = 1795505158.41
    kernel = dict(pid=p.pid, starttick=256493441, hz=100, integer_btime=1792940224)
    sample = dict(wall_before=1795506000., boottime=2565775.1065443, wall_after=1795506000.00001,
                  offset_lower=1792940224.8934557, offset_upper=1792940224.8934657)
    monkeypatch.setattr(m, 'same', lambda *args: True)
    monkeypatch.setattr(m, 'kernel_birth', lambda pid: kernel)
    monkeypatch.setattr(m, 'clock_sample', lambda: sample)
    mapped = m.mapped_birth(p, raw)
    assert mapped['raw_birth'] == raw
    assert mapped['lower'] == pytest.approx(1795505159.3034557, abs=.000001)
    assert mapped['upper']-mapped['lower'] == pytest.approx(.01001, abs=.000001)
    assert m.launch_birth(mapped, 1795505159.3052104)
    assert m.receipt_birth(mapped, 1795505159.33)
    assert not m.receipt_birth(mapped, 1795505159.0)
    assert not m.launch_birth(mapped, 1795505159.36)


@pytest.mark.parametrize('raw', [900., True, float('nan'), float('inf'), -1.])
def test_kernel_raw_birth_disagreement_or_invalid_birth_rejected(monkeypatch, raw):
    monkeypatch.setattr(m, 'same', lambda *args: True)
    monkeypatch.setattr(m, 'kernel_birth', lambda pid: dict(pid=123, starttick=10045, hz=100, integer_btime=1000))
    monkeypatch.setattr(m, 'clock_sample', lambda: dict(offset_lower=1000.8, offset_upper=1000.80001))
    with pytest.raises(ValueError, match='Raw psutil birth'):
        m.mapped_birth(Proc(), raw)


@pytest.mark.parametrize('field,value', [('pid',999), ('hz',True), ('hz',0), ('starttick',True),
                                          ('starttick',0), ('integer_btime',False), ('integer_btime',-1)])
def test_kernel_identity_or_hz_invalid_rejected(monkeypatch, field, value):
    kernel = dict(pid=123, starttick=10045, hz=100, integer_btime=1000)
    kernel[field] = value
    monkeypatch.setattr(m, 'same', lambda *args: True)
    monkeypatch.setattr(m, 'kernel_birth', lambda pid: kernel)
    monkeypatch.setattr(m, 'clock_sample', lambda: dict(offset_lower=1000.8, offset_upper=1000.80001))
    with pytest.raises(ValueError, match='kernel birth identity'):
        m.mapped_birth(Proc(), 1100.45)


def test_raw_pid_birth_reuse_during_mapping_rejected(monkeypatch):
    monkeypatch.setattr(m, 'same', Mock(side_effect=[True, False]))
    monkeypatch.setattr(m, 'kernel_birth', lambda pid: dict(pid=123, starttick=10045, hz=100, integer_btime=1000))
    monkeypatch.setattr(m, 'clock_sample', lambda: dict(offset_lower=1000.8, offset_upper=1000.80001))
    with pytest.raises(ValueError, match='during clock mapping'):
        m.mapped_birth(Proc(), 1100.45)


@pytest.mark.parametrize('before,boot,after', [(1000.,100.,999.), (1000.,100.,1000.06),
    (float('nan'),100.,1000.), (1000.,float('nan'),1000.), (1000.,True,1000.), (1000.,-1.,1000.)])
def test_clock_sample_reversed_wide_or_invalid_rejected(monkeypatch, before, boot, after):
    monkeypatch.setattr(m.time, 'time', Mock(side_effect=[before,after]))
    monkeypatch.setattr(m.time, 'clock_gettime', lambda clock: boot)
    with pytest.raises(ValueError): m.clock_sample()


def test_clock_sample_records_measured_uncertainty(monkeypatch):
    monkeypatch.setattr(m.time, 'time', Mock(side_effect=[1000.,1000.00001]))
    monkeypatch.setattr(m.time, 'clock_gettime', lambda clock: 100.)
    row=m.clock_sample()
    assert row['offset_lower']==900.
    assert row['offset_upper']==pytest.approx(900.00001)


def test_kernel_starttick_hz_btime_parser_with_parentheses(monkeypatch):
    fields=['0']*20;fields[19]='10045'
    monkeypatch.setattr(m.Path,'read_text',Mock(side_effect=['123 (worker (kernel)) '+' '.join(fields),
                                                          'cpu 1 2 3\nbtime 1000\nprocesses 9\n']))
    monkeypatch.setattr(m.os,'sysconf',lambda key:100)
    assert m.kernel_birth(123)==dict(pid=123,starttick=10045,hz=100,integer_btime=1000)
