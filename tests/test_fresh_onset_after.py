import copy
from contextlib import nullcontext
import json

import h5py
import numpy as np
import pytest
import torch

from tools.tight_fresh_onset_after import aggregate, comparison_bins, run, validate_record
from tools.tight_fresh_onset_report import BINS, PARTS, array_sha, finish, fit_eligible, read_case, report
from visuotactile.compact import sha256


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


def write_onset_case(collection, name='case', role='A', *, student_ticks=40,
                     first_legal=None, success=False):
    """A complete real-history-shaped trace, including absent/queue-only labels."""
    n = 40
    collection.mkdir(exist_ok=True)
    pair = {}
    for part in ('frontend', 'backend'):
        path = collection / (role + '_' + part + '.pt')
        path.write_bytes((role + part).encode())
        pair[part] = str(path)
        pair[part + '_sha256'] = sha256(path)
    case = dict(name=name, role=role)
    directory = collection / name
    directory.mkdir()
    source = (np.arange(n) >= student_ticks).astype(np.uint8)
    mask = np.zeros(n, dtype=np.uint8)
    if first_legal is not None:
        mask[first_legal:] = 1
    actual = np.arange(n * 29, dtype=np.float32).reshape(n, 29) / 1000
    previous = np.concatenate([np.zeros((1, 29), dtype=np.float32), actual[:-1]])
    with h5py.File(directory / 'numeric.h5', 'w') as f:
        prefix = 'episode/transition/dagger/'
        f[prefix + 'expert_label_valid'] = mask
        f[prefix + 'control_source'] = source
        f[prefix + 'segment_id'] = source + mask
        f[prefix + 'switch_clean'] = mask
        f[prefix + 'expert_source_is_expert_issued'] = source
        f[prefix + 'student_proposed_target_rad'] = actual + np.float32(.1)
        f[prefix + 'u_expert'] = actual
        f['episode/obs/actuator_target_rad'] = previous
        f['episode/transition/target_after_rad'] = actual
    expert_ticks = n - student_ticks
    result = dict(identity=name, split='train', family='family_' + name,
        legal=True, software_complete=True, success=success, steps=n,
        complete_student_policy=pair, checkpoint_sha256=pair['backend_sha256'],
        residual_forward_ticks=n, dagger=dict(student_ticks=student_ticks,
            expert_ticks=expert_ticks, expert_valid_ticks=int(mask.sum()),
            expert_switch_masked_ticks=expert_ticks - int(mask.sum()),
            takeover_triggered=expert_ticks > 0,
            expert_takeover_success=bool(success and expert_ticks > 0),
            student_autonomous_success=bool(success and expert_ticks == 0)))
    (directory / 'result.json').write_text(json.dumps(result))
    return case, pair


@pytest.mark.parametrize('student_ticks,success,cohort', [
    (40, False, 'no_expert_takeover'),
    (40, True, 'student_success_without_takeover'),
    (38, False, 'no_legal_expert_labels'),
    (38, True, 'no_legal_expert_labels'),
])
def test_zero_labels_remain_complete_opportunities_without_fitting(tmp_path, student_ticks, success, cohort):
    case, pair = write_onset_case(tmp_path, student_ticks=student_ticks, success=success)
    row = read_case(tmp_path, case, pair)
    assert row['cohort'] == cohort
    assert row['student_ticks'] == student_ticks
    assert row['masked_expert_transition_ticks'] == 40 - student_ticks
    assert row['legal_expert_ticks'] == 0
    assert row['label_vs_actual_max_abs_rad'] is None
    assert not fit_eligible(row)
    if student_ticks == 40:
        assert not row['expert_takeover_success']
    for item in row['bins'].values():
        assert item['ticks'] == 0
        assert item['first_completed_tick'] is item['last_completed_tick'] is None
        for part in PARTS:
            assert item[part]['shadow_rmse_rad'] is None
            assert item[part]['previous_target_rmse_rad'] is None
            assert item[part]['shadow_to_previous_target_rmse_ratio'] is None


@pytest.mark.parametrize('change', ['student_label', 'source_switch_back', 'nonfinite_history', 'false_success'])
def test_zero_label_support_retains_execution_checks(tmp_path, change):
    case, pair = write_onset_case(tmp_path)
    if change == 'false_success':
        path = tmp_path / 'case/result.json'
        result = json.loads(path.read_text())
        result['dagger']['expert_takeover_success'] = True
        path.write_text(json.dumps(result))
    else:
        with h5py.File(tmp_path / 'case/numeric.h5', 'r+') as f:
            if change == 'student_label':
                f['episode/transition/dagger/expert_label_valid'][0] = 1
            elif change == 'source_switch_back':
                f['episode/transition/dagger/control_source'][0] = 1
            else:
                f['episode/obs/actuator_target_rad'][0, 0] = np.nan
    with pytest.raises(ValueError):
        read_case(tmp_path, case, pair)


def write_mixed_collection(collection):
    plan = dict(kind='takeover', test_used=False, source_commit='fixture', cases=[], pairs={})
    for i in range(12):
        role = 'A' if i % 2 == 0 else 'B'
        case, pair = write_onset_case(collection, name=f'case{i}', role=role,
            student_ticks=4 if i == 11 else 40, first_legal=6 if i == 11 else None,
            success=i == 11)
        plan['cases'].append(case)
        plan['pairs'][role] = pair
    (collection / 'plan.json').write_text(json.dumps(plan))
    (collection / 'results.json').write_text(json.dumps([{'case': case['name']} for case in plan['cases']]))
    return plan


def test_mixed_collection_keeps_twelve_opportunities_and_one_fitting_suffix(tmp_path):
    write_mixed_collection(tmp_path)
    result = report(tmp_path)
    assert result['completed_cases'] == result['population']['completed_opportunities'] == 12
    assert result['population']['zero_label_cases'] == 11
    assert result['population']['takeover_cases'] == result['population']['successful_legal_suffix_cases'] == 1
    assert result['population']['legal_expert_ticks'] == 34
    assert result['roles']['A']['population']['completed_opportunities'] == 6
    assert result['roles']['A']['expert_failure']['cases'] == 0
    assert result['roles']['A']['no_expert_takeover']['cases'] == 6
    assert result['roles']['A']['expert_success']['bins']['first12']['hand']['shadow_rmse_rad'] is None
    assert result['roles']['B']['expert_success']['cases'] == 1
    assert result['roles']['B']['expert_success']['bins']['after24']['ticks'] == 10
    assert result['roles']['B']['no_expert_takeover']['cases'] == 5


@pytest.mark.parametrize('preflight_only', [True, False])
def test_after_validates_both_pairs_but_only_runs_successful_own_role(tmp_path, monkeypatch, preflight_only):
    import tools.tight_fresh_onset_after as after
    import visuotactile.anchor_continue as anchor
    import visuotactile.models.native_dp as native_dp
    import visuotactile.precision as precision
    import visuotactile.state_resources as resources

    collection = tmp_path / 'collection'
    write_mixed_collection(collection)
    before_path = tmp_path / 'before.json'
    before_path.write_text(json.dumps(report(collection)))
    root = tmp_path / 'round'
    root.mkdir()
    (root / 'session.json').write_text(json.dumps({'deadline': 1e20}))
    verified_roles, loaded_roles, predicted_cases = [], [], []

    def role_inputs(current_root, role, plan, references):
        assert current_root == root
        verified_roles.append(role)
        names = [c['name'] for c in plan['cases'] if c['role'] == role and fit_eligible(references[c['name']])]
        index_path = root / role / 'index.json'
        index_path.parent.mkdir()
        binding = {'base_checkpoint_sha256': role * 64}
        index, data = {'episodes': {}}, {'entries': {}}
        for name in names:
            numeric = collection / name / 'numeric.h5'
            with h5py.File(numeric) as f:
                robot = torch.zeros(40, 178)
                robot[:, 58:87] = torch.from_numpy(f['episode/obs/actuator_target_rad'][:])
                record = dict(identity=name, cache_identity=binding, robot=robot,
                    history_source='strictly_past_actual_executed_targets_including_student_prefix; never regenerated nominal labels',
                    supervision_mask=torch.from_numpy(f['episode/transition/dagger/expert_label_valid'][:]),
                    control_segment=torch.from_numpy(f['episode/transition/dagger/segment_id'][:]),
                    target=torch.from_numpy(f['episode/transition/dagger/u_expert'][:]))
            cache_path = index_path.parent / (name + '.pt')
            torch.save(record, cache_path)
            index['episodes'][name] = {'sha256': sha256(cache_path)}
            data['entries'][name] = {'path': str(numeric), 'source_sha256': sha256(numeric)}
        return {'role': role}, data, names, index_path, index, binding, {'role': role}

    def load_residual(saved, device):
        loaded_roles.append(saved['role'])
        return torch.nn.Identity()

    def residual_prediction(model, record, lease):
        predicted_cases.append(record['identity'])
        return record['target']

    class Lease:
        def block(self):
            return nullcontext()

    monkeypatch.setattr(after, 'role_inputs', role_inputs)
    monkeypatch.setattr(native_dp, 'load_residual', load_residual)
    monkeypatch.setattr(anchor, 'residual_prediction', residual_prediction)
    monkeypatch.setattr(precision, 'state_fp32', lambda: None)
    monkeypatch.setattr(resources, 'lease_for', lambda *args, **kwargs: Lease())
    output = tmp_path / 'after'
    run(root, collection, before_path, output, preflight_only=preflight_only)
    result = json.loads((output / ('preflight.json' if preflight_only else 'result.json')).read_text())
    assert verified_roles == ['A', 'B']
    assert loaded_roles == ([] if preflight_only else ['B'])
    assert predicted_cases == ([] if preflight_only else ['case11'])
    assert result['population']['completed_opportunities'] == 12
    assert result['population']['zero_label_cases'] == 11
    assert result['fitting_cases'] == 1
    a, b = result['roles']['A'], result['roles']['B']
    assert a['comparison_status'] == 'no_successful_legal_own_role_suffix'
    assert not a['model_loaded']
    assert a['fitting_cases'] == a['selected_legal_ticks'] == 0
    assert len(a['excluded_cases']) == 6
    assert b['fitting_cases'] == 1 and b['selected_legal_ticks'] == 34
    assert len(b['excluded_cases']) == 5
    for bin_value in a['bins'].values():
        for when in ('before', 'after'):
            assert bin_value[when]['ticks'] == 0
            assert bin_value[when]['hand']['shadow_rmse_rad'] is None
    if not preflight_only:
        assert b['bins']['first12']['after']['hand']['shadow_rmse_rad'] == 0
