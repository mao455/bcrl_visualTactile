import json
import os

import h5py
import numpy as np
import pytest
import torch

from tools.state_motion_recovery import append_mode_numeric, correct_takeover_result, shutdown_stall
from tools.state_motion_recovery import RecoveryRunner, FAILED_STAGE, STARTUP_FAILED_STAGE
from visuotactile.compact import sha256
from visuotactile.replay_fresh import append_numeric


def test_switch_preserves_original_values_and_marks_absent_fields(tmp_path):
    with h5py.File(tmp_path/'record.h5', 'w') as stream:
        group = stream.create_group('episode')
        append_mode_numeric(group, dict(transition=dict(position_requested_rad=torch.tensor([[.5, .6]]),
            target_after_rad=torch.tensor([[.5, .6]]))), 0)
        append_mode_numeric(group, dict(transition=dict(command_clipped_pre_delay=torch.tensor([[-.2, .1]]),
            target_after_rad=torch.tensor([[.49, .61]]), control_trace=dict(effective_lag_steps=torch.tensor([2])))), 1)
        assert np.array_equal(group['transition/position_requested_rad'][0], np.array([.5, .6], np.float32))
        assert np.isnan(group['transition/position_requested_rad'][1]).all()
        assert np.isnan(group['transition/command_clipped_pre_delay'][0]).all()
        assert np.array_equal(group['recorded_field_valid/transition/position_requested_rad'][:], [True, False])
        assert np.array_equal(group['recorded_field_valid/transition/command_clipped_pre_delay'][:], [False, True])
        assert np.array_equal(group['recorded_field_valid/transition/control_trace/effective_lag_steps'][:], [False, True])
        assert group['transition/control_trace/effective_lag_steps'][1] == 2
        assert group.attrs['completed_steps'] == 2
        with pytest.raises(ValueError):
            append_mode_numeric(group, dict(transition=dict(target_after_rad=torch.ones(1, 3))), 2)
        assert len(group['transition/target_after_rad']) == 2
        with pytest.raises(ValueError):
            append_mode_numeric(group, {}, 3)


def test_takeover_metric_never_compares_native_command_to_position(tmp_path):
    with h5py.File(tmp_path/'numeric.h5', 'w') as stream:
        group = stream.create_group('episode')
        append_mode_numeric(group, dict(transition=dict(position_requested_rad=torch.tensor([[.5]]),
            target_after_rad=torch.tensor([[.5]]))), 0)
        append_mode_numeric(group, dict(transition=dict(command_clipped_pre_delay=torch.tensor([[-1.]]),
            target_after_rad=torch.tensor([[.4975]]))), 1)
    np.savez_compressed(tmp_path/'trace.npz', command=np.array([[.5], [-1.]]))
    original = dict(takeover=dict(student_ticks=1), submitted_target_max_error=1.4975)
    result = correct_takeover_result(tmp_path/'result.json', original)
    assert result is original
    assert result['submitted_target_max_error'] == 0
    assert np.array_equal(np.load(tmp_path/'trace.npz')['command_is_position'], [True, False])
    assert np.array_equal(np.load(tmp_path/'trace.npz')['command'], [[.5], [-1.]])


def test_watchdog_leaves_active_work_and_resource_waits_alone(tmp_path):
    (tmp_path/'lifecycle.json').write_text(json.dumps(dict(state='running', utc=1)))
    assert shutdown_stall(tmp_path, 10000) is None
    (tmp_path/'lifecycle.json').write_text(json.dumps(dict(state='closing', utc=100)))
    assert shutdown_stall(tmp_path, 129) is None
    assert 'thirty' in shutdown_stall(tmp_path, 130)
    error = tmp_path/'worker_error.json'; error.write_text('{}')
    os.utime(error, (100, 100))
    assert 'five' in shutdown_stall(tmp_path, 106)


def test_single_mode_numeric_values_match_frozen_recorder(tmp_path):
    with h5py.File(tmp_path/'record.h5', 'w') as stream:
        before = stream.create_group('original'); after = stream.create_group('recovery')
        for tick in range(3):
            values = dict(transition=dict(target_after_rad=torch.arange(29.)[None]+tick,
                time_after_s=torch.tensor([tick/60], dtype=torch.float64),
                control_trace=dict(valid=torch.tensor([True]), delay_index=torch.tensor([tick]))))
            append_numeric(before, values, tick); append_mode_numeric(after, values, tick)
        def compare(name, obj):
            if isinstance(obj, h5py.Dataset):
                assert np.array_equal(obj[:], after[name][:])
                assert obj.dtype == after[name].dtype
                assert after['recorded_field_valid/'+name][:].all()
        before.visititems(compare)


def test_failed_takeover_stays_missing_without_new_start(tmp_path, monkeypatch):
    runner = object.__new__(RecoveryRunner); runner.root = tmp_path
    monkeypatch.setattr('tools.state_motion_recovery.verify_registration', lambda root: {})
    checkpoint = tmp_path/'model.pt'; checkpoint.write_bytes(b'preserved')
    specs = [dict(identity='train0', seed=1, cohort='train')]
    inference = dict(inference_steps=16)
    (tmp_path/(FAILED_STAGE+'_job.json')).write_text(json.dumps(dict(specs=specs,
        checkpoint_sha256=sha256(checkpoint), inference=inference)))
    result = runner.evaluate('single_takeover_12', 'dp', checkpoint, specs,
        inference=inference, takeover_steps=12)
    assert not result['software_complete'] and result['trials'] == []
    assert result['missing'][0]['native_outcome'] is None
    assert not (tmp_path/FAILED_STAGE).exists()
    with pytest.raises(ValueError):
        runner.evaluate('single_takeover_12', 'dp', checkpoint, specs,
            inference=inference, takeover_steps=24)


def test_startup_failure_is_missing_without_consuming_another_attempt(tmp_path, monkeypatch):
    runner = object.__new__(RecoveryRunner); runner.root = tmp_path
    monkeypatch.setattr('tools.state_motion_recovery.verify_registration', lambda root: {})
    output = tmp_path/STARTUP_FAILED_STAGE; output.mkdir()
    (output/'runtime_controls.json').write_text('{}')
    (output/'lifecycle.json').write_text(json.dumps(dict(state='launching')))
    (tmp_path/(STARTUP_FAILED_STAGE+'.log')).write_text(
        "terminate called after throwing an instance of 'std::system_error'\n  what():  Invalid argument")
    (tmp_path/(STARTUP_FAILED_STAGE+'_exit.json')).write_text(json.dumps(
        dict(error=STARTUP_FAILED_STAGE+' failed: exit=-6')))
    (tmp_path/(STARTUP_FAILED_STAGE+'_process_exit.json')).write_text(json.dumps(dict(pid=-1, reaped=True)))
    checkpoint = tmp_path/'model.pt'; checkpoint.write_bytes(b'preserved')
    specs = [dict(identity='train0', seed=1, cohort='train')]
    (tmp_path/(STARTUP_FAILED_STAGE+'_job.json')).write_text(json.dumps(dict(specs=specs,
        checkpoint_sha256=sha256(checkpoint), inference={})))
    result = runner.evaluate('single_takeover_24', 'dp', checkpoint, specs, inference={}, takeover_steps=24)
    assert result['missing'][0]['classification'] == 'pre_scene_native_fault'
    assert result['missing'][0]['native_outcome'] is None
    assert result['trials'] == [] and not result['software_complete']
    assert not (tmp_path/'startup_recoveries').exists()
    (output/'numeric.h5').touch()
    with pytest.raises(ValueError, match='pre-scene'):
        runner.evaluate('single_takeover_24', 'dp', checkpoint, specs, inference={}, takeover_steps=24)


def test_missing_diagnostics_prevent_selection_and_test(tmp_path, monkeypatch):
    from visuotactile import state_full_run
    runner = object.__new__(RecoveryRunner); runner.root = tmp_path
    folder = tmp_path/'single_takeover_24'; folder.mkdir()
    (folder/'result.json').write_text(json.dumps(dict(software_complete=False)))
    original_write = state_full_run.atomic_json
    def finish(owner):
        state_full_run.atomic_json(owner.root/'motion_selection.json', dict(group='H24'))
        pytest.fail('Held-out test must be unreachable')
    monkeypatch.setattr(state_full_run.Runner, 'run_motion', finish)
    with pytest.raises(RuntimeError, match='selection/test withheld'):
        runner.run_motion()
    assert not (tmp_path/'motion_selection.json').exists()
    assert json.loads((tmp_path/'motion_selection_withheld.json').read_text())['incomplete'] == ['single_takeover_24']
    assert state_full_run.atomic_json is original_write


def test_early_termination_before_handback_has_only_position_units(tmp_path):
    with h5py.File(tmp_path/'numeric.h5', 'w') as stream:
        append_mode_numeric(stream.create_group('episode'), dict(transition=dict(
            position_requested_rad=torch.tensor([[.5]]), target_after_rad=torch.tensor([[.5]]))), 0)
    np.savez_compressed(tmp_path/'trace.npz', command=np.array([[.5]]))
    result = correct_takeover_result(tmp_path/'result.json', dict(takeover=dict(student_ticks=24)))
    assert result['submitted_target_max_error'] == 0
    assert np.load(tmp_path/'trace.npz')['command_is_position'].all()
