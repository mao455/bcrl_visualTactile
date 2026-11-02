"""Scientific admission and the causal diagnostic input comparison."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from visuotactile.native_v2_bc_compare import ActorBC, ActorController, ACTOR_CONTRACT, actor_history
from visuotactile.native_v2_forensics import first_difference, contact_start, configuration_diff
from visuotactile.native_v2_gates import (Gate, data_gate, historical_replay_gate, infrastructure_gate,
                                        online_expert_gate, require_training_foundation)
from visuotactile.native_v2_expert import lane_values


def expert_rows():
    return [dict(identity=str(index), policy='released_online_expert', success=True,
                 checkpoint_sha256='expert', environment_sha256='environment', fresh_native_reset=True,
                 software_complete=True, native_termination=True) for index in range(8)]


def test_historical_failure_does_not_veto_valid_online_expert():
    rows = [dict(identity=str(index), success=index < 3,
                 comparison={'errors': {key: {'first_exceeded': None} for key in
                             ('command', 'delayed', 'delay_index', 'target', 'force', 'torque')}})
            for index in range(8)]
    historical = historical_replay_gate({'infrastructure_passed': True}, rows, [str(index) for index in range(8)])
    assert historical.name == 'G0-C' and historical.acceptance_passed
    assert historical.closed_loop_success['successes'] == 3
    numeric = data_gate({'infrastructure_passed': True, 'failures': []}, {'passed': True, 'episodes': 400})
    online = online_expert_gate(expert_rows(), [str(index) for index in range(8)], 'expert', 'environment')
    assert infrastructure_gate(numeric, online).acceptance_passed


@pytest.mark.parametrize('field,value', [('success', False), ('fresh_native_reset', False),
    ('policy', 'recorded_replay'), ('checkpoint_sha256', 'other'), ('environment_sha256', 'other'),
    ('native_termination', False), ('identity', 'duplicate')])
def test_online_failure_or_provenance_mismatch_blocks(field, value):
    rows = expert_rows()
    for row in rows[:2]:
        row[field] = value
    gate = online_expert_gate(rows, [str(index) for index in range(8)], 'expert', 'environment')
    assert not gate.acceptance_passed


def test_old_combined_gate_cannot_bypass_new_foundation(tmp_path):
    (tmp_path/'gates').mkdir()
    (tmp_path/'gates/G0.json').write_text(json.dumps(Gate('G0', True, None, None, True, None).to_dict()))
    with pytest.raises(FileNotFoundError):
        require_training_foundation(tmp_path)


def test_exact_first_difference_is_distinct_from_tolerance():
    source = np.zeros((5, 2, 3)); actual = source.copy()
    actual[1, 0, 2] = 1e-7; actual[3, 1, 0] = .02
    assert first_difference(actual, source)['first'] == dict(tick=1, component=[0, 2], source=0., evaluation=1e-7, absolute_error=1e-7)
    assert first_difference(actual, source, .001)['first']['tick'] == 3
    assert contact_start(np.zeros((5, 2, 5, 3))) is None


def test_multi_lane_recording_does_not_duplicate_lane_zero():
    value = dict(obs=torch.arange(12).reshape(4, 3), transition={'delay': torch.arange(4)})
    selected = lane_values(value, 3)
    assert torch.equal(selected['obs'], torch.tensor([[9, 10, 11]]))
    assert selected['transition']['delay'].item() == 3


def test_actor_history_online_matches_causal_training_and_masks_future():
    torch.set_num_threads(2)
    model = ActorBC(); saved = dict(contract=ACTOR_CONTRACT, updates=1, ema=model.state_dict())
    controller = ActorController(saved, device='cpu')
    values = torch.randn(25, 141); times = torch.arange(25, dtype=torch.float64)/60
    with torch.no_grad():
        for tick in range(25):
            packet = dict(step_id=torch.tensor([tick]), time_s=times[tick:tick+1])
            command = controller.act(values[tick:tick+1], packet)
            if tick % 12 == 0:
                condition = actor_history(values, times, tick, model.actor_mean, model.actor_std)[None]
                expected = model.predict(condition).clamp(-1, 1)
                assert torch.equal(command, expected[:, 0])
                changed = values.clone(); changed[tick+1:] = 1e6
                assert torch.equal(condition[0], actor_history(changed, times, tick, model.actor_mean, model.actor_std))
    assert actor_history(values, times, 0, model.actor_mean, model.actor_std).reshape(13, 143)[:-1].count_nonzero() == 0


def test_configuration_comparison_preserves_missing_fields():
    assert configuration_diff({'scene': {'num_envs': 4}}, {'scene': {'num_envs': 1}}) == [
        dict(field='scene.num_envs', source=4, evaluation=1)]


def test_registered_expert_minimum_includes_failures_without_replacement():
    rows = expert_rows(); rows[2]['success'] = False; rows[5]['success'] = False
    registered = [str(index) for index in range(8)]
    gate = online_expert_gate(rows, registered, 'expert', 'environment', minimum_successes=6)
    assert gate.acceptance_passed and gate.closed_loop_success['successes'] == 6
    assert gate.closed_loop_success['trials'] == 8
    assert not online_expert_gate(rows[:7], registered, 'expert', 'environment', 6).acceptance_passed


def test_native_outcome_report_keeps_censored_and_fall_separate():
    from visuotactile.native_v2_repair_report import summarize_rows, wilson
    rows = [dict(success=False, steps=180, reasons={'timeout': False}, recorded_actions_exhausted=True),
            dict(success=False, steps=334, reasons={'fall': True}),
            dict(success=True, steps=170, reasons={'max_successes': True})]
    result = summarize_rows(rows)
    assert result['successes'] == 1 and result['trials'] == 3
    assert result['outcomes'] == {'actions_exhausted_censored': 1, 'fall': 1, 'max_successes': 1}
    assert wilson(6, 8)[0] < .5 < wilson(6, 8)[1]


def test_actual_diagnostic_state_input_roundtrip_matches_feature_budget():
    from pathlib import Path
    path = Path('outputs/visuotactile/native_dp_v2/deployment_audit_20261102_03/result.json')
    if not path.exists():
        pytest.skip('Local immutable deployment audit absent')
    result = json.loads(path.read_text())
    assert result['episodes'] == 360 and result['anchors'] == 6286 and result['features'] == 2601
    error = np.load(path.parent/'per_feature_max_error.npy')
    assert np.isfinite(error).all() and error.max() < 6e-5
    assert not error[:2340].reshape(13, 180)[:, 116:178].any()
    assert not error[2340:].any()


def test_public_export_removes_absolute_machine_paths_but_keeps_numeric_sources():
    from visuotactile.native_v2_repair_export import public_value
    record = dict(path='/private/work/outputs/run/trace.npz', source=0.5,
                  configuration={'asset': 'C:\\private\\assets\\object.urdf'})
    assert public_value(record) == dict(path='outputs/run/trace.npz', source=0.5,
                                       configuration={'asset': 'assets/object.urdf'})


def test_shadow_metrics_separate_arm_and_hand_and_preserve_command_scale():
    from visuotactile.native_v2_closed_loop_audit import error_metrics
    expected = np.zeros((2, 29)); actual = expected.copy()
    actual[:, :7] = .5; actual[:, 7:] = 1.
    assert error_metrics(actual, expected) == dict(commands=2, arm_rmse=.5, hand_rmse=1.)


def test_fit_extension_keeps_original_deadline_and_caps(tmp_path, monkeypatch):
    import time
    from visuotactile.native_v2_fit_extension import initialize
    monkeypatch.chdir(tmp_path)
    parent = tmp_path/'parent'; parent.mkdir(); (parent/'gates').mkdir()
    deadline = time.time()+100
    original = dict(started_epoch=time.time()-1000, deadline=deadline, source_root='source',
                    bc_train_ids=['train'], bc_validation_ids=['val'])
    (parent/'session.json').write_text(json.dumps(original))
    for name in ('G0-A', 'G0-B', 'G0'):
        (parent/'gates'/(name+'.json')).write_text(json.dumps(dict(name=name, acceptance_passed=True)))
    for kind in ('state', 'actor'):
        path = parent/'bc_compare'/kind; path.mkdir(parents=True)
        (path/'milestone_1000.pt').write_bytes(b'immutable-test-checkpoint')
    initialize(tmp_path/'extension', parent)
    session = json.loads((tmp_path/'extension/session.json').read_text())
    assert session['deadline'] == deadline == session['training_deadline']
    assert session['started_epoch'] == original['started_epoch']
    assert session['additional_updates'] == 3000 and session['maximum_additional_starts'] == 8
    assert session['test_calls'] == session['automatic_trajectory_retries'] == 0
    assert not session['automatic_further_training']


@pytest.mark.parametrize('target', ['data.json', 'statistics.pt'])
def test_foundation_rejects_audit_source_drift_even_with_copied_gate(tmp_path, target):
    from visuotactile.compact import sha256
    from visuotactile.native_v2_foundation import verify_foundation_provenance
    root = tmp_path/'root'; root.mkdir()
    source = tmp_path/'source'; source.mkdir()
    prior = tmp_path/'prior'; prior.mkdir()
    (source/'statistics.pt').write_bytes(b'original-train-statistics')
    digest = sha256(source/'statistics.pt')
    (source/'data.json').write_text(json.dumps(dict(statistics_sha256=digest)))
    (prior/'data_audit.json').write_text(json.dumps(dict(source_data_sha256=sha256(source/'data.json'),
                                                        statistics_sha256=digest)))
    (prior/'control_audit.json').write_text('{}')
    (root/'session.json').write_text(json.dumps(dict(source_root=str(source), previous_root=str(prior))))
    if target == 'data.json':
        (source/target).write_text(json.dumps(dict(statistics_sha256=digest, changed=True)))
    else:
        (source/target).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        verify_foundation_provenance(root)


def test_actual_fresh_replay_matches_source_bytes_for_all_recorded_physical_fields():
    from pathlib import Path
    path = Path('outputs/visuotactile/native_dp_v2/repair_20261102_01/report/fresh_bytewise.json')
    if not path.exists():
        pytest.skip('Local immutable fresh expert replay evidence absent')
    report = json.loads(path.read_text())
    assert report['all_fields_bytewise']
    assert [row['steps'] for row in report['rows']] == [170, 493]
    required = {'command', 'delayed', 'delay_index', 'target', 'joint', 'velocity', 'object', 'contact',
                'force', 'torque', 'substep_joint', 'substep_velocity', 'substep_object', 'subgoals', 'retract'}
    for row in report['rows']:
        assert set(row['fields']) == required
        assert all(value['bytes_equal'] and value['dtype_equal'] for value in row['fields'].values())


def test_frozen_protocol_refusal_does_not_mutate_old_evidence(tmp_path, monkeypatch):
    from visuotactile.native_v2_pipeline import Runner, main
    session = dict(protocol_version=2)
    (tmp_path/'session.json').write_text(json.dumps(session))
    (tmp_path/'status.json').write_text('{"state":"original-frozen"}')
    (tmp_path/'blocked.json').write_text('{"reason":"original-frozen"}')
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    runner = object.__new__(Runner); runner.session = session
    with pytest.raises(RuntimeError, match='frozen'):
        runner.run()
    monkeypatch.setattr('sys.argv', ['pipeline', 'resume', '--root', str(tmp_path)])
    with pytest.raises(RuntimeError, match='frozen'):
        main()
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


def test_extension_recovery_refuses_started_trajectory(tmp_path):
    import time
    from visuotactile.native_v2_fit_extension import recover_preaction
    (tmp_path/'session.json').write_text(json.dumps(dict(deadline=time.time()+100)))
    attempt = tmp_path/'saved_initials'; attempt.mkdir()
    (attempt/'kit_lifecycle.json').write_text(json.dumps(dict(phase='environment_ready')))
    (attempt/'train_0').mkdir()
    with pytest.raises(ValueError, match='pre-environment'):
        recover_preaction(tmp_path)
    assert (attempt/'train_0').is_dir() and not (tmp_path/'preaction_failure_1').exists()


def test_gate_a_never_ignores_numeric_failures():
    assert not data_gate({'infrastructure_passed': True, 'failures': ['wrong delayed label']},
                         {'passed': True, 'episodes': 400}).acceptance_passed
    assert not data_gate({'infrastructure_passed': True, 'failures': []},
                         {'passed': True, 'episodes': 399}).acceptance_passed
