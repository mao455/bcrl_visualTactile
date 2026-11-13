import copy

import pytest

from tools.tight_replay_balance import population_weights, checked_root, ROOT, ORDER


def test_population_weighting_removes_current_overrepresentation():
    coverage = dict(strata={name: dict(supervised_tick_population=count) for name, count in
        [('core', 55392), ('historical_correction', 133628), ('current_correction', 4500)]})
    before = copy.deepcopy(coverage)
    weights = population_weights(coverage)
    assert sum(weights.values()) == pytest.approx(1)
    assert weights['current_correction'] == pytest.approx(4500 / 193520)
    assert weights['historical_correction'] > .69
    assert coverage == before


def test_rejects_missing_population_and_wrong_root(tmp_path):
    with pytest.raises(ValueError):
        population_weights(dict(strata={}))
    with pytest.raises(ValueError):
        checked_root(tmp_path)
    assert checked_root(ROOT) == ROOT
    assert len(set(ORDER)) == 4
    assert {group for arm, group in ORDER if arm == 'population'} == {'QG01', 'G01'}
