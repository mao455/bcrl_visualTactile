import numpy as np
import pytest

from tools import tight_fresh_next_collection_coverage as report


def arrays(n=84):
    obj = np.zeros((n, 7), np.float32)
    obj[:, 2] = .6; obj[48:, 2] = .7; obj[:, 3] = 1
    palm = np.zeros_like(obj); palm[:, 3] = 1
    force = np.zeros((n, 2, 5, 3), np.float32); force[:, :, :2, 2] = 1
    goal = obj.copy(); goal[:, 0] = .2
    return dict(object=obj, palm=palm, force=force, goal_after=goal,
        subgoals_after=np.zeros(n, int), near=np.zeros(n, bool), lifted=np.arange(n) >= 48)


def test_delay_uses_actual_posttransition_state_and_counts_each_new_contact_gap():
    a = arrays(); a['force'][65:68, 1, 1, 2] = .1
    windows, height = report.causal_contact_height(a['object'], a['force'])
    start = report.first(windows)
    assert start == 60
    immediate = report.delay_state(a, windows, height, start, 0, 84)
    later = report.delay_state(a, windows, height, start, 12, 84)
    assert immediate['proposed_action_index'] == 60 and immediate['post_trigger_contact_gap_ticks'] == 0
    assert later['proposed_action_index'] == 72 and later['preceding_completed_tick'] == 72
    assert later['post_trigger_contact_gap_ticks'] == 3
    assert later['two_contact_fingers_both_substeps']
    assert not later['contact_height_window_at_action_state']


def test_actual_expert_intervention_censors_hypothetical_delayed_states():
    a = arrays(70)
    windows, height = report.causal_contact_height(a['object'], a['force'])
    value = report.delay_state(a, windows, height, 60, 12, 100)
    assert value['scope'] == 'delay_censored_by_actual_expert_switch'
    assert 'object_to_active_goal_angle_rad' not in value
    assert not value['action_state_observed']


def test_native_terminal_after_state_is_not_another_action_opportunity():
    a = arrays(72)
    windows, height = report.causal_contact_height(a['object'], a['force'])
    value = report.delay_state(a, windows, height, 60, 12, 72)
    assert value['scope'] == 'delay_reaches_or_exceeds_native_termination'
    assert not value['action_state_observed']


def test_candidate_uses_new_active_goal_when_previous_transition_completed_a_subgoal():
    a = arrays()
    a['goal_after'][71, :3] = a['object'][71, :3]
    a['subgoals_after'][71] = 1
    windows, height = report.causal_contact_height(a['object'], a['force'])
    value = report.delay_state(a, windows, height, 60, 12, 84)
    assert value['object_to_active_goal_root_distance_m'] == 0
    assert not value['active_goal_unchanged_since_trigger']
    assert value['native_subgoals'] == 1


def test_relative_rotation_cancels_common_world_rotation():
    a = arrays()
    theta = .7
    a['object'][71, 3:] = [np.cos(theta/2), 0, 0, np.sin(theta/2)]
    a['palm'][71, 3:] = a['object'][71, 3:]
    windows, height = report.causal_contact_height(a['object'], a['force'])
    value = report.delay_state(a, windows, height, 60, 12, 84)
    assert value['object_world_rotation_since_trigger_rad'] == pytest.approx(theta, abs=1e-6)
    assert value['palm_relative_rotation_since_trigger_rad'] == pytest.approx(0, abs=1e-6)


def test_proposal_has_sixteen_distinct_family_role_opportunities_and_no_delay24_or_formal029():
    ids, delays = zip(*report.PROPOSAL)
    assert len(ids) == len(set(ids)) == 8
    assert set(delays) == {0, 12}
    assert all(not value.startswith('formal_029_') for value in ids)
    families = {name.rsplit('_slow_', 1)[0]:{} for name in ids}
    rows = report.proposal_rows(families)
    assert sum(row['planned_opportunities'] for row in rows) == 16
    assert len({(row['family'], role) for row in rows for role in row['roles']}) == 16


def test_delay_summary_keeps_unknown_states_out_of_contact_denominator():
    a = arrays(70)
    windows, height = report.causal_contact_height(a['object'], a['force'])
    value = report.delay_state(a, windows, height, 60, 12, 100)
    summary = report.delay_summary([dict(delay_comparison={'12':value})], 12)
    assert summary['qualifying_prefixes'] == 1
    assert summary['observed_action_states'] == summary['two_contact_fingers_still_true'] == 0


def test_failed_expert_suffix_accounting_is_not_training_admission():
    raw = dict(steps=601, success=False, dagger=dict(expert_valid_ticks=239,
        recovery_coverage=dict(population=dict(dp_anchors=20))))
    indexed = dict(raw, accepted_for_training=False, actual_exposure=dict(episodes=1,
        history_ticks=601, supervised_ticks=239, supervised_anchors=20))
    report.validate_historical_result(raw, indexed, False)
    with pytest.raises(ValueError, match='admission'):
        report.validate_historical_result(raw, indexed, True)
    with pytest.raises(ValueError, match='Historical result'):
        report.validate_historical_result(raw, dict(indexed, success=True), False)
