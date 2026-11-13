import copy

import pytest

from tools.tight_ab_replication import checked_root, compare_batches, verify_bindings
from tools.runtime import sha


def batches():
    result = []
    for batch in (0, 1):
        plan = [dict(identity=f'case_{index}', spec=dict(seed=index, narrow_reset={'seed': index}))
                for index in range(batch * 10, (batch + 1) * 10)]
        rows = [dict(identity=row['identity'], seed=row['spec']['seed'], narrow_reset=row['spec']['narrow_reset'],
                legal=True, success=False, native_lift=True, native_subgoals=0) for row in plan]
        summary = dict(complete=True, software_failures=0, completed=10, expert_takeover=False,
                       formal_test22_used=False, results=rows)
        result.append((plan, copy.deepcopy(summary), copy.deepcopy(summary)))
    return result


def test_fixed_paired_cohort():
    report = compare_batches(batches())
    assert report['opportunities_matched']
    assert all(values['attempts'] == 20 for values in report['totals'].values())
    assert all(values['successes'] == 0 for values in report['totals'].values())
    assert not report['test22']


@pytest.mark.parametrize('field,value', [('complete', False), ('software_failures', 1),
    ('completed', 9), ('expert_takeover', True), ('formal_test22_used', True)])
def test_invalid_summary_rejected(field, value):
    values = batches()
    values[0][1][field] = value
    with pytest.raises(ValueError):
        compare_batches(values)


@pytest.mark.parametrize('field,value', [('seed', -1), ('narrow_reset', {}), ('identity', 'other')])
def test_mismatched_sample_rejected(field, value):
    values = batches()
    values[0][1]['results'][0][field] = value
    with pytest.raises(ValueError):
        compare_batches(values)


def test_rejection_preserved_without_fairness_pass():
    values = batches()
    values[0][1]['results'][0].update(legal=False, success=True)
    report = compare_batches(values)
    assert report['totals']['QG01']['attempts'] == 20
    assert report['totals']['QG01']['valid'] == 19
    assert report['totals']['QG01']['successes'] == 0
    assert not report['opportunities_matched']


def test_missing_and_duplicate_batch_rejected():
    values = batches()
    for invalid in (values[:1], [values[0], values[0]]):
        with pytest.raises(ValueError):
            compare_batches(invalid)


def test_binding_guard(tmp_path):
    path = tmp_path / 'evidence.json'
    path.write_text('original')
    registration = dict(bindings={str(path): sha(path)})
    verify_bindings(registration)
    path.write_text('changed')
    with pytest.raises(ValueError):
        verify_bindings(registration)


def test_unique_root_guard(tmp_path):
    with pytest.raises(ValueError):
        checked_root(tmp_path)
