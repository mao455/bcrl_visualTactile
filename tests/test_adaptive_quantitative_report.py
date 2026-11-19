"""Prevent false fitting gains from changed populations or episode weighting."""
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / 'research_docs/evidence/tight_adaptive_dagger_20261119_01/quantitative_report.py'
SPEC = importlib.util.spec_from_file_location('adaptive_quantitative', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def point(identity, ticks, arm, cohort='original30'):
    return dict(role='A', identity=identity, stage='legal', component='final',
                cohort=cohort, ticks=str(ticks), arm_squared_sum=str(arm),
                hand_squared_sum=str(arm * 22 / 7))


def test_exact_population_tick_weighting_and_current_cohort():
    before = [point('short', 1, 7, 'this_round'), point('long', 9, 63, 'this_round')]
    after = [point('short', 1, 28), point('long', 9, 63)]
    result, missing = MODULE.compare(after, before)
    assert missing == [] and len(result) == 1
    row = result[0]
    assert row['cohort'] == 'original30' and row['trajectories'] == 2 and row['ticks'] == 10
    assert row['parent_arm_rmse_rad'] == pytest.approx(1)
    assert row['new_arm_rmse_rad'] == pytest.approx((91 / 70) ** .5)


def test_missing_parent_and_mask_changes_are_not_improvement():
    result, missing = MODULE.compare([point('new', 2, 0)], [point('old', 2, 7)])
    assert result == [] and missing[0]['missing_parent_trajectories'] == 1
    with pytest.raises(AssertionError, match='Mask population changed'):
        MODULE.compare([point('same', 3, 0)], [point('same', 2, 7)])


def test_failed_expert_suffix_is_excluded_from_training_ticks():
    records = [dict(role='A', stage='alignment', takeover_tick=120, takeover_phase=0,
                    expert_rescue=True, accepted=True, legal_expert_ticks=50),
               dict(role='B', stage='alignment', takeover_tick=241, takeover_phase=1,
                    expert_rescue=False, accepted=False, legal_expert_ticks=80)]
    row = MODULE.collection_stages(records)[0]
    assert row['role'] == 'all' and row['takeovers'] == 2
    assert row['expert_rescues'] == 1 and row['accepted_legal_ticks'] == 50
    assert row['excluded_expert_ticks'] == 80 and row['planning_phases'] == '0,1'
