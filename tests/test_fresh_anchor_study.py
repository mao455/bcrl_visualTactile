import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import tight_fresh_anchor_study as study
from visuotactile.compact import sha256
from visuotactile.residual_output import MATCHED_FINAL_HEAD_RESET, REACHABLE_PREVIOUS_TARGET_OUTPUT


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def change(path, function):
    value = study.read(path)
    function(value)
    write(path, value)


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    output = tmp_path / 'study'
    session = dict(deadline=study.time.time() + 3600, domain={'fixed':'narrow'},
        resource_root=str(root), expert_checkpoint='expert.pt', visual_stride=1,
        control_hz=60, planning_hz=5, owned_worker_roots=[str(root)])
    write(root / 'session.json', session)
    write(root / 'shared_resources.json', {'owner':'original'})
    pairs = {}
    for role in ('A', 'B'):
        base = root / role
        write(base / 'state_data/data.json', dict(splits=dict(train=['train'], validation=['val'], test=[])))
        write(base / 'state_data/statistics.pt', {'affine':role})
        write(base / 'cache_index.json', {'frozen':'vision'})
        write(base / 'training/best.pt', {'dp':role})
        write(base / 'backend' / role / 'training/best.pt', {'rear':role})
        pairs[role] = dict(frontend=str(base / 'training/best.pt'),
            frontend_sha256=sha256(base / 'training/best.pt'),
            backend=str(base / 'backend' / role / 'training/best.pt'),
            backend_sha256=sha256(base / 'backend' / role / 'training/best.pt'),
            residual_version='target_residual_transformer60_v1')
    for role in ('A', 'B'):
        base = root / role
        write(base / 'backend' / role / 'nominal/index.json', dict(
            base_sha256=pairs[role]['frontend_sha256'], data_sha256=sha256(base / 'state_data/data.json'),
            statistics_sha256=sha256(base / 'state_data/statistics.pt'),
            residual_statistics_sha256=sha256(root / 'B/state_data/statistics.pt')))
        write(base / 'backend' / role / 'nominal/scales.json', {'scales':[1.]})
    cases = [dict(name=role + '_trial', role=role, spec=dict(identity='trial', seed=7))
             for role in ('A', 'B')]
    evaluation = root / 'development'
    write(evaluation / 'plan.json', dict(kind='autonomous', pairs=pairs, test_used=False,
        cases=cases, resource_root=str(root), source_commit='reviewed', source_sha256={'source':'bound'}))
    write(evaluation / 'run_started.json', {'plan_sha256':sha256(evaluation / 'plan.json')})
    rows = []
    for case in cases:
        result = dict(case['spec'], legal=True, software_complete=True, steps=601,
            residual_ticks=601, success=False, checkpoint_sha256=pairs[case['role']]['backend_sha256'],
            residual_version=pairs[case['role']]['residual_version'], simulation_step_hz=60,
            plan_hz=5, action_control_mode='absolute_joint_position_target_rad')
        write(evaluation / case['name'] / 'result.json', result)
        rows.append(dict(case=case['name'], role=case['role'], result=result))
    write(evaluation / 'results.json', rows)
    write(evaluation / 'result.json', dict(software_complete=True, worker_error=None, cases=2))
    calls = []
    def backend(candidate, role, epochs, **kwargs):
        calls.append(dict(arm=candidate.name, role=role, epochs=epochs, **kwargs))
        rear = candidate / role / 'backend' / role
        (rear / 'nominal').mkdir(parents=True)
        write(rear / 'session.json', study.read(candidate / 'session.json'))
        write(rear / 'configuration.json', dict(epochs=epochs, **kwargs))
    monkeypatch.setattr(study, 'completed_pair', lambda candidate, role:copy.deepcopy(pairs[role]))
    monkeypatch.setattr(study, 'prepare_backend', backend)
    monkeypatch.setattr(study, 'sources', lambda:{})
    monkeypatch.setattr(study.subprocess, 'check_output', lambda *a, **k:'commit\n')
    return SimpleNamespace(root=root, output=output, session=session, pairs=pairs,
                           evaluation=evaluation, calls=calls)


def test_cases_pair_four_complete_students_without_sharing_mutable_reset(monkeypatch):
    from visuotactile import narrow_data
    monkeypatch.setattr(narrow_data, 'direct_configuration', lambda domain, seed:dict(domain=domain, seed=seed))
    cases = study.development_cases({'range':[1, 2]}, 4, 123)
    assert len({case['name'] for case in cases}) == 16
    for index in range(4):
        block = cases[index * 4:(index + 1) * 4]
        assert {c['role'] for c in block} == {'control_A', 'control_B', 'previous_A', 'previous_B'}
        assert all(c['spec'] == block[0]['spec'] for c in block)
        assert all(c['video'] is (index == 0) for c in block)
        assert block[0]['spec']['event_seed'] == 100123 + index
    assert [c['role'] for c in cases[4:8]] == list(reversed([c['role'] for c in cases[:4]]))
    cases[0]['spec']['narrow_reset']['domain']['range'][0] = 99
    assert cases[1]['spec']['narrow_reset']['domain']['range'] == [1, 2]


@pytest.mark.parametrize('fault', ['takeover', 'test', 'plan_drift', 'case_drift', 'model',
    'partial_student', 'censored', 'dagger', 'missing_case', 'worker_error'])
def test_source_requires_registered_complete_autonomous_full_student_evidence(source, fault):
    x = source
    if fault in ('takeover', 'test', 'plan_drift'):
        key, value = {'takeover':('kind', 'takeover'), 'test':('test_used', True),
                      'plan_drift':('source_commit', 'changed')}[fault]
        change(x.evaluation / 'plan.json', lambda v:v.update({key:value}))
    elif fault == 'case_drift':
        change(x.evaluation / 'A_trial/result.json', lambda v:v.update(steps=3))
    elif fault == 'missing_case':
        change(x.evaluation / 'results.json', lambda v:v.pop())
    elif fault == 'worker_error':
        change(x.evaluation / 'result.json', lambda v:v.update(worker_error='failed'))
    else:
        key, value = {'model':('checkpoint_sha256', 'other'), 'partial_student':('residual_ticks', 0),
                      'censored':('censored', True), 'dagger':('dagger', {'switch':4})}[fault]
        change(x.evaluation / 'A_trial/result.json', lambda v:v.update({key:value}))
        change(x.evaluation / 'results.json', lambda v:v[0]['result'].update({key:value}))
    with pytest.raises(ValueError):
        study.prepare(x.root, x.output, 2, 4, 123)
    assert not x.output.exists()


def test_prepare_preserves_source_and_binds_shared_cache_matched_reset_and_deadline(source):
    x = source
    before = {str(p):sha256(p) for p in x.root.rglob('*') if p.is_file()}
    study.prepare(x.root, x.output, 3, 4, 123)
    config = study.read(x.output / 'registration.json')
    assert config['deadline'] == x.session['deadline']
    assert config['source_pairs'] == x.pairs
    assert len(x.calls) == 4
    for call in x.calls:
        assert call['epochs'] == 3 and call['continue_rear']
        assert call['continuation_head_reset'] == MATCHED_FINAL_HEAD_RESET
        assert call['output_contract'] == (REACHABLE_PREVIOUS_TARGET_OUTPUT
                                          if call['arm'] == 'previous' else None)
        base = x.output / call['arm'] / call['role']
        assert (base / 'training').resolve() == x.root / call['role'] / 'training'
        assert (base / 'backend' / call['role'] / 'nominal').resolve() == (
            x.root / call['role'] / 'backend' / call['role'] / 'nominal')
        assert study.read(base / 'backend/cache_complete.json')['recomputed'] is False
    assert study.check_session_binding(x.output, config)['resource_root'] == str(x.root)
    assert before == {str(p):sha256(p) for p in x.root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault', ['source_session', 'study_session', 'backend_session', 'arm_session',
                                 'registered_deadline', 'telemetry'])
def test_run_rejects_session_or_monitor_drift_before_launch(source, monkeypatch, fault):
    x = source
    study.prepare(x.root, x.output, 2, 4, 123)
    if fault == 'registered_deadline':
        change(x.output / 'registration.json', lambda v:v.update(deadline=v['deadline'] + 300))
    elif fault == 'telemetry':
        link = x.output / 'previous/shared_resources.json'
        link.unlink()
        write(link, {'another_owner':True})
    else:
        path = {'source_session':x.root / 'session.json', 'study_session':x.output / 'session.json',
                'arm_session':x.output / 'control/session.json',
                'backend_session':x.output / 'control/A/backend/A/session.json'}[fault]
        change(path, lambda v:v.update(domain={'changed':True}))
    monkeypatch.setattr(study.subprocess, 'Popen', lambda *a, **k:pytest.fail('No child should start'))
    with pytest.raises(ValueError):
        study.run(x.output)
    assert not (x.output / 'run_started.json').exists()


def test_development_keeps_four_group_identity_and_original_monitor(source, monkeypatch):
    from visuotactile import narrow_data
    x = source
    study.prepare(x.root, x.output, 2, 4, 123)
    monkeypatch.setattr(narrow_data, 'direct_configuration', lambda domain, seed:dict(seed=seed))
    monkeypatch.setattr(study, 'completed_pair', lambda candidate, role:dict(group=candidate.name + '_' + role))
    study.prepare_development(x.output)
    plan = study.read(x.output / 'development/plan.json')
    assert set(plan['pairs']) == {'control_A', 'control_B', 'previous_A', 'previous_B'}
    assert all(pair['group'] == key for key, pair in plan['pairs'].items())
    assert {c['role'] for c in plan['cases']} == set(plan['pairs'])
    assert plan['resource_root'] == str(x.root) and plan['test_used'] is False
    assert study.read(x.output / 'development/session.json')['deadline'] == x.session['deadline']
    with pytest.raises(FileExistsError):
        study.prepare_development(x.output)


@pytest.mark.parametrize('epochs,count,seed', [(True, 4, 1), (2, 4.5, 1), (2, 4, True),
    (2, 4, -1), (2, 4, 2**32), (21, 4, 1), (2, 9, 1)])
def test_invalid_budgets_do_not_create_attempt(source, epochs, count, seed):
    with pytest.raises(ValueError):
        study.prepare(source.root, source.output, epochs, count, seed)
    assert not source.output.exists()


def test_dangling_attempt_is_preserved(source):
    source.output.symlink_to(source.root / 'missing')
    with pytest.raises(FileExistsError):
        study.prepare(source.root, source.output, 2, 4, 123)
    assert source.output.is_symlink()


@pytest.mark.parametrize('incomplete', [False, True])
def test_owned_stage_chain_uses_original_deadline_and_requires_positive_completion(source, monkeypatch, incomplete):
    from visuotactile import narrow_data
    x = source
    study.prepare(x.root, x.output, 2, 4, 123)
    monkeypatch.setattr(narrow_data, 'direct_configuration', lambda domain, seed:dict(seed=seed))
    commands, waits = [], []
    def spawn(command, **kwargs):
        commands.append(command)
        destination = Path(command[command.index('--output') + 1])
        training = 'residual-train' in command
        value = {} if incomplete else dict(target_reached=True) if training else (
            dict(complete=True) if 'residual-offline' in command else dict(software_complete=True))
        write(destination / ('finished.json' if training else 'result.json'), value)
        assert kwargs['start_new_session'] is True
        assert kwargs['env']['OMP_NUM_THREADS'] == kwargs['env']['MKL_NUM_THREADS'] == '2'
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(study.subprocess, 'Popen', spawn)
    def wait(worker, deadline, **kwargs):
        waits.append(dict(deadline=deadline, **kwargs))
    monkeypatch.setattr(study, 'wait_owned', wait)
    if incomplete:
        with pytest.raises(RuntimeError, match='Incomplete stage receipt'):
            study.run(x.output)
        assert len(commands) == 1
        assert study.read(x.output / 'status.json')['stage'] == 'failed'
        assert not (x.output / 'finished.json').exists()
    else:
        study.run(x.output)
        assert len(commands) == 9
        assert [Path(c[c.index('--root') + 1]).parts[-4:-2] for c in commands[:8:2]] == [
            ('control', 'A'), ('previous', 'B'), ('previous', 'A'), ('control', 'B')]
        assert all(row['deadline'] == x.session['deadline'] for row in waits)
        assert all(row['startup_seconds'] is None for row in waits[:8])
        assert waits[-1]['startup_seconds'] == 180
        assert waits[-1]['startup_marker'] == x.output / 'development/initialization.json'
        assert study.read(x.output / 'finished.json')['software_complete'] is True
        with pytest.raises(FileExistsError, match='duplicate'):
            study.run(x.output)


def test_ordinary_autonomous_worker_routes_all_four_groups_without_expert_control(source, monkeypatch):
    import torch
    from tools import tight_fresh_rollouts as rollouts, tight_state
    from visuotactile import narrow_data, native_evaluation, state_benchmark, state_resources
    x = source
    study.prepare(x.root, x.output, 2, 2, 123)
    monkeypatch.setattr(narrow_data, 'direct_configuration', lambda domain, seed:dict(seed=seed))
    def pair(root, role):
        value = copy.deepcopy(x.pairs[role])
        rear = root / role / 'backend' / role / 'training/best.pt'
        write(rear, dict(group=root.name + '_' + role))
        value.update(backend=str(rear), backend_sha256=sha256(rear))
        return value
    monkeypatch.setattr(study, 'completed_pair', pair)
    study.prepare_development(x.output)
    lease = SimpleNamespace(wait_s=0.)
    env = SimpleNamespace(sim=SimpleNamespace(step=lambda:None, render=lambda:None),
                          scene=SimpleNamespace(update=lambda:None))
    monkeypatch.setattr(torch, 'load', lambda path, **kwargs:study.read(path))
    monkeypatch.setattr(state_resources, 'lease_for', lambda root, deadline:lease)
    monkeypatch.setattr(tight_state, 'environment', lambda *a, **k:(None, env, None, lease))
    calls = []
    def trial(env, lease, directory, spec, previous, player, checkpoint, **kwargs):
        directory.mkdir()
        assert kwargs['position'] is True
        assert kwargs['expert_shadow'] is False and kwargs['capture_dataset'] is False
        assert kwargs['dagger'] is None and previous == {}
        group = kwargs['saved']['group']
        assert directory.name.startswith(group + '_')
        assert kwargs['video_title'].startswith(group + ' ')
        calls.append((group, spec['seed']))
        return dict(legal=True, software_complete=True, steps=3, residual_ticks=3, success=False)
    monkeypatch.setattr(state_benchmark, 'trial', trial)
    monkeypatch.setattr(native_evaluation, 'exit_completed_kit_worker', lambda output:None)
    rollouts.run(x.output / 'development')
    assert len(calls) == 8 and len(set(calls)) == 8
    assert {group for group, seed in calls} == {'control_A', 'control_B', 'previous_A', 'previous_B'}
    assert study.read(x.output / 'development/result.json')['cases'] == 8
