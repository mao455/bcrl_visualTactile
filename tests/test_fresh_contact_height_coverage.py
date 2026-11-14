import numpy as np
import pytest

from tools import tight_fresh_contact_height_coverage as audit


def prefix(n=84):
    pose = np.zeros((n, 7), np.float32)
    pose[:, 2] = .6
    pose[48:, 2] = .7
    pose[:, 3] = 1
    force = np.zeros((n, 2, 5, 3), np.float32)
    force[:, :, :2, 2] = 1
    return pose, force


def test_height_reference_and_twelve_full_completed_transitions_are_causal():
    pose, force = prefix()
    windows, height = audit.causal_contact_height(pose, force)
    assert height == pytest.approx(.6)
    assert audit.first(windows) == 60
    assert windows.sum() == 25
    short, height = audit.causal_contact_height(pose[:47], force[:47])
    assert not short.any() and height is None


def test_strict_point_one_and_both_substeps_are_required_for_every_tick():
    pose, force = prefix()
    force[59, 1, 1, 2] = .1
    windows, _ = audit.causal_contact_height(pose, force)
    assert not windows[59:71].any()
    assert audit.first(windows) == 72


def test_rotation_is_not_part_of_contact_height_collection_screen():
    pose, force = prefix()
    palm = np.zeros_like(pose)
    palm[:, 3] = 1
    theta = np.arange(len(pose))*.1
    pose[:, 3], pose[:, 5] = np.cos(theta/2), np.sin(theta/2)
    windows, height = audit.causal_contact_height(pose, force)
    old, _ = audit.current.stable_windows(pose, palm, force, height)
    assert windows.any() and not old.any()
    assert np.array_equal(windows, audit.current.contact_height_windows(pose, force, height))


def test_expert_suffix_cannot_supply_the_student_height_or_windows():
    pose, force = prefix()
    # A rescue after completed48 does not create a window in its student prefix.
    windows, _ = audit.causal_contact_height(pose[:48], force[:48])
    assert not windows.any()
    assert audit.causal_contact_height(pose, force)[0].any()


def test_schedule_counts_only_actual_control_states_and_marks_delay_censoring():
    assert audit.schedule_scope(60, 12, 84, 84)['scope'] == 'observed_student_action_state'
    assert audit.schedule_scope(72, 12, 84, 100)['scope'] == 'observed_actual_expert_switch_state'
    assert audit.schedule_scope(73, 12, 84, 100)['scope'] == 'delay_censored_by_actual_expert_switch'
    terminal = audit.schedule_scope(72, 12, 84, 84)
    assert terminal['scope'] == 'delay_reaches_or_exceeds_native_termination'
    assert not terminal['action_state_observed']
    assert audit.schedule_scope(None, 12, 84, 84)['proposed_action_index'] is None


def test_active_goal_snapshot_uses_pretransition_goal_and_exact_contact_window():
    pose, force = prefix()
    palm = pose.copy()
    goal = pose.copy()
    windows, height = audit.causal_contact_height(pose, force)
    value = audit.endpoint(60, pose, palm, force, goal, np.zeros(84, bool),
        np.zeros(84, int), np.zeros(84, bool), height)
    assert value['contact_both_substeps_all12']
    assert not value['frozen_stable']
    assert value['object_to_active_goal_root_distance_m'] == 0
    assert value['completed_tick'] == 60


def test_recorded_trigger_loader_fails_closed_on_changed_source(tmp_path):
    path = tmp_path/'visuotactile/takeover_trigger.py'
    path.parent.mkdir()
    path.write_text('VERSION = "original"\n')
    digest = audit.sha256(path)
    module, binding = audit.frozen_trigger(tmp_path, digest)
    assert module.VERSION == 'original' and binding['sha256'] == digest
    path.write_text('VERSION = "changed"\n')
    with pytest.raises(ValueError, match='Actual old trigger source changed'):
        audit.frozen_trigger(tmp_path, digest)


def test_aggregate_admission_does_not_hide_changed_physical_result():
    raw = dict(steps=512, success=True, dagger=dict(expert_valid_ticks=214,
        recovery_coverage=dict(population=dict(dp_anchors=18))))
    indexed = dict(raw, accepted_for_training=True, actual_exposure=dict(episodes=1,
        history_ticks=512, supervised_ticks=214, supervised_anchors=18, supervised_window_targets=412))
    audit.validate_result_index(raw, indexed, dict(accepted=True))
    with pytest.raises(ValueError, match='Completed index/case result'):
        audit.validate_result_index(raw, dict(indexed, success=False), dict(accepted=True))
    with pytest.raises(ValueError, match='learning-accounting'):
        audit.validate_result_index(raw, dict(indexed, actual_exposure=dict(indexed['actual_exposure'],
            supervised_ticks=512)), dict(accepted=True))
