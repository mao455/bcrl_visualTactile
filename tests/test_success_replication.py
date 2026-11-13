import copy

import pytest

from tools.tight_success_replication import ORDER, aggregate, checked_root, plans


def summaries():
    result = []
    for batch, label in ORDER:
        rows = [dict(identity=f'fresh_{index}', seed=index, narrow_reset={'fixed': index},
            legal=True, success=label == 'contact' and index == 3, native_lift=True,
            native_subgoals=int(label == 'contact' and index == 3))
            for index in range(batch * 10, (batch + 1) * 10)]
        result.append(dict(label=label, summary=dict(complete=True, software_failures=0,
            completed=10, results=rows)))
    return result


def test_paired_denominators():
    report = aggregate(summaries())
    assert report['totals']['contact']['successes'] == 1
    assert report['totals']['previous']['successes'] == 0
    assert report['totals']['contact']['attempts'] == 20
    assert len(report['paired']) == 20
    assert not report['test22']


@pytest.mark.parametrize('change', ['incomplete', 'duplicate', 'seed', 'reset'])
def test_invalid_comparison_rejected(change):
    values = copy.deepcopy(summaries())
    if change == 'incomplete':
        values[0]['summary']['complete'] = False
    elif change == 'duplicate':
        values[0]['summary']['results'][0]['identity'] = 'fresh_1'
    elif change == 'seed':
        values[0]['summary']['results'][0]['seed'] = -1
    else:
        values[0]['summary']['results'][0]['narrow_reset'] = {}
    with pytest.raises(ValueError):
        aggregate(values)


def test_rejected_cases_not_replaced():
    values = summaries()
    values[0]['summary']['results'][3]['legal'] = False
    report = aggregate(values)
    assert report['totals']['contact']['attempts'] == 20
    assert report['totals']['contact']['valid'] == 19
    assert report['totals']['contact']['successes'] == 0


def test_plan_is_frozen_and_balanced(monkeypatch):
    monkeypatch.setattr('tools.tight_success_replication.direct_configuration', lambda domain, seed: {'seed': seed})
    first = plans({})
    assert first == plans({})
    assert list(map(len, first)) == [10, 10]
    assert len({case['spec']['seed'] for batch in first for case in batch}) == 20
    assert all(case['spec']['split'] == 'new' for batch in first for case in batch)
    assert ORDER == ((0, 'contact'), (0, 'previous'), (1, 'previous'), (1, 'contact'))


def test_root_rejects_new_budget(tmp_path):
    with pytest.raises(ValueError):
        checked_root(tmp_path)
