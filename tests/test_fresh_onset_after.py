import copy

import h5py
import numpy as np
import pytest
import torch

from tools.tight_fresh_onset_after import aggregate, comparison_bins, validate_record
from tools.tight_fresh_onset_report import BINS, PARTS, array_sha, finish


def fixture_record(tmp_path):
    n = 40
    mask = np.arange(n) >= 2
    target = np.arange(n * 29, dtype=np.float32).reshape(n, 29) / 1000
    previous = target - np.float32(.01)
    segment = mask.astype(np.int64)
    path = tmp_path / 'numeric.h5'
    with h5py.File(path, 'w') as f:
        f['episode/transition/dagger/expert_label_valid'] = mask
        f['episode/transition/dagger/u_expert'] = target
        f['episode/transition/dagger/segment_id'] = segment
        f['episode/obs/actuator_target_rad'] = previous
    binding = {'base_checkpoint_sha256': 'a' * 64, 'data_sha256': 'b' * 64}
    robot = torch.zeros(n, 178)
    robot[:, 58:87] = torch.from_numpy(previous)
    record = dict(identity='case', cache_identity=binding, robot=robot,
        history_source='strictly_past_actual_executed_targets_including_student_prefix; never regenerated nominal labels',
        target=torch.from_numpy(target), supervision_mask=torch.from_numpy(mask),
        control_segment=torch.from_numpy(segment))
    reference = dict(case='case', legal_mask_sha256_uint8=array_sha(mask.astype(np.uint8)),
        legal_indices_sha256_int64_le=array_sha(np.flatnonzero(mask).astype('<i8')))
    return record, binding, reference, path


def test_record_preserves_complete_history_and_exact_legal_rows(tmp_path):
    record, binding, reference, path = fixture_record(tmp_path)
    indices, target, previous = validate_record(record, binding, reference, path)
    assert indices.tolist() == list(range(2, 40))
    assert target.shape == previous.shape == (40, 29)


@pytest.mark.parametrize('field', ['supervision_mask', 'target', 'robot', 'control_segment', 'cache_identity'])
def test_changed_cache_provenance_is_rejected(tmp_path, field):
    record, binding, reference, path = fixture_record(tmp_path)
    changed = copy.deepcopy(record)
    if field == 'supervision_mask':
        changed[field][2] = False
    elif field == 'target':
        changed[field][2, 0] += .01
    elif field == 'robot':
        changed[field][0, 58] += .01  # Even the unsupervised prefix must stay real.
    elif field == 'control_segment':
        changed[field][3] += 1
    else:
        changed[field]['data_sha256'] = 'c' * 64
    with pytest.raises(ValueError):
        validate_record(changed, binding, reference, path)


def reference_bins(target, previous, indices, error):
    result = {}
    for name, rows in zip(BINS, (indices[:12], indices[12:24], indices[24:])):
        sums = {part: [float(np.square(np.full_like(target[rows, section], error)).sum()),
                       float(np.square(previous[rows, section] - target[rows, section]).sum())]
                for part, section in PARTS.items()}
        result[name] = finish(sums, len(rows))
    return result


def test_comparison_and_aggregation_sum_squared_errors():
    cases = []
    for n, error in [(26, .1), (38, .2)]:
        target = np.zeros((n, 29), dtype=np.float64)
        previous = np.ones_like(target) * .3
        indices = np.arange(2, n)
        old = reference_bins(target, previous, indices, .4)
        bins = comparison_bins(target + error, target, previous, indices, old)
        cases.append(dict(bins=bins))
    grouped = aggregate(cases)
    assert grouped['first12']['after']['ticks'] == 24
    assert grouped['after24']['after']['ticks'] == 12
    assert grouped['first12']['after']['hand']['shadow_rmse_rad'] == pytest.approx(np.sqrt((.1**2 + .2**2) / 2))
    assert grouped['first12']['after']['hand']['shadow_rmse_rad'] != pytest.approx(.15)
    assert grouped['after24']['after']['hand']['shadow_rmse_rad'] == pytest.approx(.2)
    assert grouped['first12']['improved_cases'] == {'arm': 2, 'hand': 2}


def test_changed_baseline_or_nonfinite_prediction_rejected():
    target = np.zeros((40, 29), dtype=np.float64)
    previous = np.ones_like(target)
    indices = np.arange(2, 40)
    old = reference_bins(target, previous, indices, .4)
    with pytest.raises(ValueError, match='baseline'):
        comparison_bins(target, target, previous * .9, indices, old)
    prediction = target.copy()
    prediction[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        comparison_bins(prediction, target, previous, indices, old)
