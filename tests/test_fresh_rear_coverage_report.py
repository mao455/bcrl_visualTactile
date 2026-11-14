import copy

import pytest

from tools.tight_fresh_rear_coverage_report import compact_report, coverage_fractions, summary


def row(category, ticks, *, accepted=True):
    stable = category != 'no_completed_stable_grasp'
    return dict(accepted=accepted, expert_takeover_success=accepted,
        student_phase_category=category, expert_supervision={'legal_ticks':ticks},
        student=dict(student_contact_ever=True, student_two_fingers_ever=True,
            student_native_lift_ever=stable, student_rise_2cm_with_two_fingers=stable,
            student_ever_stable=stable, stable_at_takeover=category=='stable_at_takeover',
            student_native_near_goal_ever=False, student_native_alignment_ever=False))


def test_coverage_distinguishes_prior_stability_from_actual_handover_and_excludes_failed_labels():
    rows = [row('no_completed_stable_grasp', 100),
        row('formerly_stable_now_unstable', 20), row('stable_at_takeover', 5),
        row('stable_at_takeover', 1000, accepted=False)]
    result = coverage_fractions(rows)
    assert result['ever_student_stable_successes'] == pytest.approx(2/3)
    assert result['stable_at_handover_successes'] == pytest.approx(1/3)
    assert result['stable_at_handover_source_legal_ticks'] == pytest.approx(5/125)
    assert result['ever_student_stable_source_legal_ticks'] == pytest.approx(25/125)
    totals = summary(rows)
    assert totals['opportunities'] == 4 and totals['accepted'] == 3
    assert totals['source_legal_ticks'] == 125
    assert totals['source_legal_ticks_by_category']['stable_at_takeover'] == 5
    assert totals['student_stage_counts']['stable_at_takeover'] == 2
    assert totals['student_stage_counts']['student_native_alignment_ever'] == 0


def test_no_accepted_data_has_no_defined_source_fraction():
    assert all(value is None for value in coverage_fractions([]).values())
    assert all(value is None for value in coverage_fractions([
        row('no_completed_stable_grasp', 100, accepted=False)]).values())


def test_compact_output_keeps_completed_and_selected_exposure_separate():
    item = row('stable_at_takeover', 5)
    first = dict(stage='round_03', frontend_episode_updates=10,
        frontend_planning_anchor_exposures=80, frontend_legal_target_slot_exposures=1700,
        backend_completed_legal_ticks=150, backend_selected_legal_ticks=140)
    second = dict(first, stage='round_04', backend_completed_legal_ticks=160, backend_selected_legal_ticks=155)
    item.update(case='one', numeric_sha256='trace_digest', source_population={'large':'redundant'},
        actual_training_exposure={'A':[first, second]})
    original = copy.deepcopy(item)
    report = compact_report(dict(cases=[item], stable_student_cases=[item]))
    case = report['cases'][0]
    assert case['actual_training_exposure_totals']['A']['backend_completed_legal_ticks'] == 310
    assert case['actual_training_exposure_totals']['A']['backend_selected_legal_ticks'] == 295
    assert case['actual_training_exposure_totals']['A']['frontend_episode_updates'] == 20
    assert case['numeric_sha256'] == 'trace_digest' and 'source_population' not in case
    assert 'actual_training_exposure' not in case
    assert report['stable_student_cases'][0]['actual_training_exposure'] == original['actual_training_exposure']
    assert item == original
