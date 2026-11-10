"""Actual after-state labels, guarded local source audit, and target identities."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from visuotactile.anchor_sampling import labels
from visuotactile.native_data import limits_affine
from visuotactile.narrow_data import actual_configuration_labels
from visuotactile.position_control import image_contract

SOURCE = Path('outputs/visuotactile/run/tight_vision_20261109_01/state_data/data.json')


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def audit_actual_sources():
    """Read train/validation numeric HDF5 fields only; test source access denied."""
    data = json.loads(SOURCE.read_text())
    ids = data['splits']['train']+data['splits']['validation']
    forbidden = {str(Path(data['entries'][i]['path']).resolve()) for i in data['splits']['test']}
    allowed = {str(Path(data['entries'][i]['path']).resolve()) for i in ids}
    assert allowed.isdisjoint(forbidden)
    file_opener = h5py.File
    opened = []
    def train_validation_only(path, *args, **kwargs):
        resolved = str(Path(path).resolve())
        assert resolved in allowed and resolved not in forbidden
        opened.append(resolved)
        return file_opener(path, *args, **kwargs)
    records, reports = {}, {}
    from unittest.mock import patch
    with patch('visuotactile.narrow_data.h5py.File', side_effect=train_validation_only):
        for identity in ids:
            records[identity], reports[identity] = actual_configuration_labels(data['entries'][identity])
    train = torch.cat([records[i] for i in data['splits']['train']])
    scale, offset = limits_affine(train)
    ticks = sum(r['ticks'] for r in reports.values())
    result = dict(source_data_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        train_episodes=len(data['splits']['train']), validation_episodes=len(data['splits']['validation']),
        inspected_episodes=len(opened), inspected_ticks=ticks, test_source_reads=0,
        after_before_exact=all(r['after_before_exact'] for r in reports.values()),
        terminal_after_preserved=all(r['terminal_after_preserved'] for r in reports.values()),
        command_reached_max_abs_rad=max(r['command_reached_max_abs_rad'] for r in reports.values()),
        command_reached_arm_rmse_rad=(sum(r['ticks']*r['command_reached_arm_rmse_rad']**2 for r in reports.values())/ticks)**.5,
        command_reached_hand_rmse_rad=(sum(r['ticks']*r['command_reached_hand_rmse_rad']**2 for r in reports.values())/ticks)**.5,
        normalization_identity='q_reached_train166_affine_v1', normalization_train_ids=data['splits']['train'],
        normalization_ticks=len(train), action_scale_sha256=tensor_sha(scale), action_offset_sha256=tensor_sha(offset),
        action_scale=scale.tolist(), action_offset=offset.tolist(),
        q_reached_field='transition/diagnostics_after/joint_position_rad',
        u_expert_field='transition/target_after_rad',
        future_state_used_as_input=False)
    return result, records, data


@pytest.fixture(scope='module')
def real_audit():
    if not SOURCE.exists(): pytest.skip('Original numeric evidence remains local')
    return audit_actual_sources()


def test_all_train_validation_actual_labels_preserve_timing_and_terminal_boundary(real_audit):
    report, records, data = real_audit
    assert report['train_episodes'] == 166 and report['validation_episodes'] == 30
    assert report['inspected_episodes'] == 196 and report['test_source_reads'] == 0
    assert report['after_before_exact'] and report['terminal_after_preserved']
    assert report['command_reached_max_abs_rad'] > .01
    assert report['command_reached_arm_rmse_rad'] > 0
    assert report['command_reached_hand_rmse_rad'] > 0
    assert set(records) == set(data['splits']['train']+data['splits']['validation'])


def test_reached_affine_statistics_use_only_training_extrema(real_audit):
    report, records, data = real_audit
    train_ids = data['splits']['train']
    low = torch.stack([records[i].amin(0) for i in train_ids]).amin(0)
    high = torch.stack([records[i].amax(0) for i in train_ids]).amax(0)
    expected_scale, expected_offset = limits_affine(torch.stack((low, high)))
    assert tensor_sha(expected_scale) == report['action_scale_sha256']
    assert tensor_sha(expected_offset) == report['action_offset_sha256']
    # Held-out mutation cannot enter the exact population used by preparation.
    replaced = dict(records)
    for i in data['splits']['validation']: replaced[i] = torch.full_like(records[i], 1000.)
    scale, offset = limits_affine(torch.cat([replaced[i] for i in train_ids]))
    assert tensor_sha(scale) == report['action_scale_sha256']
    assert tensor_sha(offset) == report['action_offset_sha256']


def test_clean_sample_changes_diffusion_labels_but_preserves_command_and_padding():
    command = torch.full((3, 29), .7)
    reached = torch.arange(3*29).reshape(3, 29).float()/100
    record = dict(time=torch.arange(3), command=command, clean_sample=reached)
    value, valid = labels(record, torch.tensor([1]), 24)
    assert torch.equal(value[0, :2], reached[1:])
    assert valid.sum() == 2 and torch.count_nonzero(value[0, 2:]) == 0
    assert torch.equal(record['command'], command)
    old, _ = labels({k:v for k,v in record.items() if k != 'clean_sample'}, torch.tensor([1]), 24)
    assert torch.equal(old[0, :2], command[1:]) and not torch.equal(old, value)


def test_command_and_reached_contracts_cannot_share_normalization_identity():
    old = image_contract('I-HistoryDP', 24, 'G00')
    command = image_contract('I-HistoryDP', 24, 'G00', 'u_expert')
    reached = image_contract('I-HistoryDP', 24, 'G00', 'q_reached')
    assert old != command != reached and old['version'] == 4 and reached['version'] == 5
    assert command['normalization_identity'] != reached['normalization_identity']
    assert reached['clean_sample_field'] == 'transition/diagnostics_after/joint_position_rad'
    assert command['action'] == reached['action'] == 'absolute_joint_position_target_rad'
    assert reached['horizon'] == 24 and reached['commit'] == 12


@pytest.mark.parametrize('corruption', ['reset', 'clock', 'terminal', 'before_clock', 'duplicate_clock'])
def test_actual_label_reader_rejects_reset_clock_or_terminal_substitution(tmp_path, corruption):
    path = tmp_path/'numeric.h5'
    q = np.arange(87, dtype=np.float32).reshape(3, 29)/100
    before = np.concatenate((np.zeros((1, 29), np.float32), q[:-1]))
    time = np.arange(3, dtype=np.float64)/60
    with h5py.File(path, 'w') as f:
        g = f.create_group('episodes/example'); g.attrs['complete'] = True
        g['transition/diagnostics_after/joint_position_rad'] = q
        g['transition/diagnostics_before/joint_position_rad'] = before
        g['transition/target_after_rad'] = q+.1
        g['transition/time_after_s'] = time+1/60
        g['transition/time_before_s'] = time
        g['obs/time_s'] = time; g['obs/step_id'] = np.arange(3)
        g['frames/post_index'] = [1, 2, -1]
        if corruption == 'reset': g['transition/diagnostics_before/joint_position_rad'][1] = 0
        if corruption == 'clock': g['transition/time_after_s'][1] += .01
        if corruption == 'terminal': g['frames/post_index'][-1] = 0
        if corruption == 'before_clock': g['transition/time_before_s'][1] += .01
        if corruption == 'duplicate_clock':
            g['obs/time_s'][1] = 0
            g['transition/time_before_s'][1] = 0
            g['transition/time_after_s'][1] = 1/60
    with pytest.raises(ValueError): actual_configuration_labels(dict(path=str(path), episode='example', length=3))
