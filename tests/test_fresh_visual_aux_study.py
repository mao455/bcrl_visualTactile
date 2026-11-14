import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import tight_fresh_visual_aux_study as study
from visuotactile.position_control import continuous_image_contract
from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT, REACHABLE_PREVIOUS_TARGET_OUTPUT
from visuotactile.visual_aux_training import CALIBRATION_VERSION, ESTIMATOR, calibration_ids


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def change(path, **updates):
    write(path, dict(study.read(path), **updates))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path/'source'; output = tmp_path/'study'
    calibration = tmp_path/'calibration.json'; acceptance = tmp_path/'acceptance.json'
    session = dict(deadline=study.time.time()+3600, domain={'narrow':'registered'}, resource_root=str(source),
        visual_stride=1, control_hz=60, planning_hz=5, resource_policy='full_gpu_v1', expert_checkpoint='expert.pt')
    write(source/'session.json', session); write(source/'shared_resources.json', {'owner':'original'})
    write(source/'metadata.json', {'task':'unchanged'}); write(source/'gate.json', {'passed':True})
    original = [f'original{i:02d}' for i in range(30)]; validation = [f'val{i:02d}' for i in range(6)]
    data = dict(normalization_train_ids=original, splits=dict(train=original+['correction'], validation=validation, test=[]),
        entries={i:dict(family=i) for i in original+validation})
    data['entries']['correction'] = dict(family=original[0]); pairs = {}; audits = {}
    for role in ('A', 'B'):
        directory = source/role; front = directory/'training/best.pt'; rear = directory/f'backend/{role}/training/best.pt'
        write(directory/'state_data/data.json', data); write(directory/'state_data/statistics.pt', {'stats':role})
        write(directory/'state_data/numeric.pt', {'numeric':role}); write(directory/'cache_index.json', {'raw_only':True})
        contract = continuous_image_contract('q_reached' if role == 'A' else 'u_expert')
        config = dict(continuation={'target_window_contract':'recovery_onset_commit12_v1'})
        write(directory/'training/configuration.json', config)
        torch.save(dict(contract=contract, updates=10, configuration=config), front)
        rear.parent.mkdir(parents=True)
        torch.save(dict(contract=contract, residual_output_version=REACHABLE_CENTERED_OUTPUT['version'],
            configuration={'output_contract':REACHABLE_CENTERED_OUTPUT}), rear)
        pairs[role] = dict(frontend=str(front), frontend_sha256=study.sha256(front),
            backend=str(rear), backend_sha256=study.sha256(rear))
        files = {name:dict(path=str(directory/name), sha256=study.sha256(directory/name))
            for name in ('state_data/data.json', 'state_data/statistics.pt', 'state_data/numeric.pt',
                'cache_index.json', 'training/configuration.json')}
        audits[role] = dict(files=files, splits=data['splits'], normalization_train_ids=original)
    receipt = dict(version=CALIBRATION_VERSION, passed=True, roles=['A','B'],
        routes=['detached_readout','shared_vision'], shared_weight=True, head_seed=71,
        gradient_ratio_target=.1, estimator=ESTIMATOR, loss_weight=.01,
        data_sha256={r:study.sha256(source/r/'state_data/data.json') for r in pairs},
        frontend_sha256={r:pairs[r]['frontend_sha256'] for r in pairs},
        statistics_sha256={r:study.sha256(source/r/'state_data/statistics.pt') for r in pairs},
        target_window_contract={r:'recovery_onset_commit12_v1' for r in pairs}, subset_ids=calibration_ids(data),
        optimizer_updates=0, simulator_starts=0, validation_used=False, test_used=False, source_sha256={})
    write(calibration, receipt)
    write(acceptance, dict(version='visual_pose_aux_acceptance_v1', passed=True, learning_rate=3e-5,
        formal_training_updates=0, validation_used=False, test_used=False, device='cuda:0',
        calibration_sha256=study.sha256(calibration), source_sha256={}, optimizer_updates=4, source_unchanged=True,
        roles=['A','B'],routes=list(study.ROUTES.values()),
        **{key:receipt[key] for key in ('head_seed','loss_weight','frontend_sha256','data_sha256','statistics_sha256','target_window_contract')}))
    evidence_file = source/'development/evidence.json'; write(evidence_file, {'actual':'completed source fixture'})
    evidence = dict(root=str(source), audit=dict(files={'result':dict(path=str(evidence_file),
        sha256=study.sha256(evidence_file))}, case_results={}))
    monkeypatch.setattr(study, 'sources', lambda:{})
    monkeypatch.setattr(study, 'source_evidence', lambda *args:copy.deepcopy(evidence))
    monkeypatch.setattr(study, 'audit_role', lambda root, role, pair, session:copy.deepcopy(audits[role]))
    monkeypatch.setattr(study, 'completed_pair', lambda root, role:copy.deepcopy(pairs[role]))
    monkeypatch.setattr(study.subprocess, 'check_output', lambda *args, **kwargs:'reviewed_commit\n')
    return SimpleNamespace(source=source, output=output, calibration=calibration, acceptance=acceptance,
        pairs=pairs, session=session, data=data)


def prepare(x):
    study.prepare(x.source, x.output, x.calibration, x.acceptance, 3, 4, 2, 123)


def test_prepare_preserves_source_and_shares_only_raw_cache_and_matching_inputs(fixture):
    x = fixture
    before = {str(p):study.sha256(p) for p in x.source.rglob('*') if p.is_file()}
    prepare(x); config = study.read(x.output/'registration.json')
    assert config['frontend_epochs'] == 3 and config['backend_epochs'] == 4
    assert config['deadline'] == x.session['deadline'] and config['test_used'] is False
    assert config['normalization_train_ids'] == x.data['normalization_train_ids']
    assert study.check_binding(x.output, config) == x.session
    for role in ('A', 'B'):
        specs = []; continuations = []
        for arm in study.ARMS:
            root = x.output/arm; directory = root/role
            assert (directory/'state_data').resolve() == x.source/role/'state_data'
            assert (directory/'cache_index.json').resolve() == x.source/role/'cache_index.json'
            assert not (directory/'training').exists() and not (directory/'backend').exists()
            specs.append(study.read(directory/'visual_auxiliary.json'))
            continuations.append(study.read(directory/'continuation.json'))
            assert specs[-1]['route'] == study.ROUTES[arm]
            assert continuations[-1]['learning_rate'] == 3e-5
        specs[0]['route'] = specs[1]['route']
        assert specs[0] == specs[1] and continuations[0] == continuations[1]
    assert before == {str(p):study.sha256(p) for p in x.source.rglob('*') if p.is_file()}


def test_distinct_active_session_owns_deadline_without_changing_expired_source_evidence(fixture, tmp_path):
    x = fixture; active = tmp_path/'active'
    change(x.source/'session.json', deadline=1)
    session = dict(x.session, resource_root=str(active))
    write(active/'session.json', session);write(active/'shared_resources.json', {'new':'owner'})
    study.prepare(x.source,x.output,x.calibration,x.acceptance,3,4,2,123,session_root=active)
    config = study.read(x.output/'registration.json')
    assert config['session_binding']['session_root'] == str(active)
    assert config['deadline'] == x.session['deadline']
    assert study.read(x.source/'session.json')['deadline'] == 1
    assert study.check_binding(x.output,config)['resource_root'] == str(active)
    assert (x.output/'control/A/shared_resources.json').resolve() == active/'shared_resources.json'


@pytest.mark.parametrize('fault', ['acceptance_false', 'acceptance_no_version', 'accepted_other_source',
    'accepted_other_calibration', 'formal_updates', 'calibration_updates', 'calibration_validation',
    'calibration_source', 'calibration_weight', 'calibration_statistics', 'calibration_window', 'previous_anchor',
    'acceptance_cpu', 'acceptance_lr'])
def test_no_formal_preparation_without_exact_admission(fixture, fault):
    x = fixture
    if fault.startswith('calibration_'):
        key, value = {'calibration_updates':('optimizer_updates', 1), 'calibration_validation':('validation_used', True),
            'calibration_source':('source_sha256', {'changed':'source'}), 'calibration_weight':('loss_weight', -1.),
            'calibration_statistics':('statistics_sha256', {'A':'a'*64,'B':'b'*64}),
            'calibration_window':('target_window_contract', {'A':None,'B':None})}[fault]
        change(x.calibration, **{key:value})
        change(x.acceptance, calibration_sha256=study.sha256(x.calibration))
    elif fault == 'previous_anchor':
        path = Path(x.pairs['A']['backend']); saved = torch.load(path, weights_only=False)
        saved['configuration']['output_contract'] = REACHABLE_PREVIOUS_TARGET_OUTPUT
        saved['residual_output_version'] = REACHABLE_PREVIOUS_TARGET_OUTPUT['version']; torch.save(saved, path)
        x.pairs['A']['backend_sha256'] = study.sha256(path)
    else:
        key, value = {'acceptance_false':('passed',False), 'acceptance_no_version':('version','unknown'),
            'accepted_other_source':('source_sha256',{'different':'code'}),
            'accepted_other_calibration':('calibration_sha256','wrong'), 'formal_updates':('formal_training_updates',1),
            'acceptance_cpu':('device','cpu'), 'acceptance_lr':('learning_rate',1e-4)}[fault]
        change(x.acceptance, **{key:value})
    with pytest.raises(ValueError): prepare(x)
    assert not (x.output/'registration.json').exists()


@pytest.mark.parametrize('fault', ['source', 'calibration', 'spec', 'session', 'telemetry', 'deadline'])
def test_mutation_after_registration_stops_before_worker(fixture, monkeypatch, fault):
    x = fixture; prepare(x)
    if fault == 'source': change(x.source/'A/state_data/data.json', mutated=True)
    elif fault == 'calibration': change(x.calibration, loss_weight=.2)
    elif fault == 'spec': change(x.output/'control/A/visual_auxiliary.json', route='shared_vision')
    elif fault == 'session': change(x.output/'control/session.json', deadline=0)
    elif fault == 'deadline': change(x.output/'registration.json', deadline=x.session['deadline']+3600)
    else:
        path = x.output/'auxiliary/shared_resources.json';path.unlink();write(path, {'new':'owner'})
    monkeypatch.setattr(study.subprocess, 'Popen', lambda *args, **kwargs:pytest.fail('No worker may start'))
    with pytest.raises(ValueError): study.run(x.output)
    assert not (x.output/'run_started.json').exists()


def test_interleaved_four_cells_share_opportunities_and_keep_independent_specs(monkeypatch):
    from visuotactile import narrow_data
    monkeypatch.setattr(narrow_data, 'direct_configuration', lambda domain,seed:dict(domain=domain,seed=seed))
    cases = study.development_cases({'x':[1,2]}, 2, 99)
    assert len(cases) == len({r['name'] for r in cases}) == 8
    assert [r['role'] for r in cases[4:]] == list(reversed([r['role'] for r in cases[:4]]))
    assert {r['role'] for r in cases} == {'control_A','control_B','auxiliary_A','auxiliary_B'}
    for block in (cases[:4], cases[4:]):
        assert all(row['spec'] == block[0]['spec'] for row in block)
    cases[0]['spec']['narrow_reset']['domain']['x'][0] = 9
    assert cases[1]['spec']['narrow_reset']['domain']['x'] == [1,2]


@pytest.mark.parametrize('incomplete', [False,True])
def test_owned_chain_refreshes_each_updated_frontend_cache_before_rear_and_evaluation(fixture, monkeypatch, incomplete):
    from visuotactile import narrow_data
    x = fixture; prepare(x); commands = []; waits = []
    monkeypatch.setattr(narrow_data, 'direct_configuration', lambda domain,seed:dict(seed=seed))
    def spawn(command, **kwargs):
        commands.append(command); root = Path(command[command.index('--root')+1]); module = command[2]
        if module == 'tools.tight_fresh_train':
            destination = root/command[command.index('--role')+1]/'training/finished.json'; proof = {'complete':True}
        elif module == 'tools.tight_fresh_backend':
            destination = root/command[command.index('--role')+1]/'backend/prepared.json'
            proof = {'inherited_student_parameters':True}
            assert '--continue-rear' in command
        elif module == 'tools.tight_finish':
            destination = root/'cache_complete.json';proof = {'complete':True}
        else:
            destination = Path(command[command.index('--output')+1])
            training = 'residual-train' in command
            destination /= 'finished.json' if training else 'result.json'
            proof = {'target_reached':True} if training else {'software_complete':True} if module.endswith('rollouts') else {'complete':True}
        write(destination, {} if incomplete else proof)
        assert kwargs['start_new_session'] is True and kwargs['env']['OMP_NUM_THREADS'] == '2'
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(study.subprocess, 'Popen', spawn)
    monkeypatch.setattr(study, 'wait_child', lambda worker, deadline, **kwargs:waits.append(dict(deadline=deadline,**kwargs)))
    if incomplete:
        with pytest.raises(ValueError, match='Incomplete stage receipt'): study.run(x.output)
        assert len(commands) == 1 and study.read(x.output/'status.json')['stage'] == 'failed'
        assert not (x.output/'finished.json').exists()
        return
    study.run(x.output)
    assert len(commands) == 21 and all(row['deadline'] == x.session['deadline'] for row in waits)
    assert all(command[2] == 'tools.tight_fresh_train' for command in commands[:4])
    for offset, (arm, role) in enumerate(study.ORDER):
        block = commands[4+4*offset:8+4*offset]
        assert [row[2] for row in block] == ['tools.tight_fresh_backend','tools.tight_finish',
            'visuotactile.anchor_continue','visuotactile.anchor_continue']
        assert block[1][3] == 'residual_cache' and block[2][3] == 'residual-train'
        assert Path(block[1][block[1].index('--root')+1]) == x.output/arm/role/'backend'
    assert waits[-1]['startup_marker'] == x.output/'development/initialization.json'
    assert study.read(x.output/'finished.json')['software_complete'] is True
    plan = study.read(x.output/'development/plan.json')
    assert len(plan['cases']) == 8 and plan['resource_root'] == str(x.source)
    assert plan['test_used'] is False and plan['kind'] == 'autonomous'
    with pytest.raises(FileExistsError, match='duplicate'):study.run(x.output)


def test_stop_before_launch_and_owned_child_stop_deadline_and_startup(fixture, monkeypatch, tmp_path):
    x = fixture; prepare(x); (x.output/'STOP').touch()
    monkeypatch.setattr(study.subprocess, 'Popen', lambda *args, **kwargs:pytest.fail('No worker may start'))
    with pytest.raises(InterruptedError):study.run(x.output)
    assert not (x.output/'run_started.json').exists()
    stopped = []; process = SimpleNamespace(poll=lambda:None)
    monkeypatch.setattr(study, 'stop_owned', lambda child:stopped.append(child))
    assert study.wait_child(process, study.time.time()+30, stop_paths=[x.output/'STOP'], stop_requested=lambda:False) == 'requested_stop'
    assert study.wait_child(process, 0, stop_paths=[], stop_requested=lambda:False) == 'original_session_deadline'
    ticks = iter([0,181]);monkeypatch.setattr(study.time, 'time', lambda:next(ticks))
    assert study.wait_child(process, 300, stop_paths=[], stop_requested=lambda:False,
        startup_marker=tmp_path/'not_initialized') == 'pre_control_startup_timeout'
    assert stopped == [process,process,process]


@pytest.mark.parametrize('epochs,rear,count,seed', [(True,2,2,1),(2,0,2,1),(2,2,0,1),(2,2,2,-1),(2,2,2,2**32)])
def test_invalid_budget_never_prepares(fixture, epochs, rear, count, seed):
    x = fixture
    with pytest.raises(ValueError):study.prepare(x.source,x.output,x.calibration,x.acceptance,epochs,rear,count,seed)
    assert not x.output.exists()
