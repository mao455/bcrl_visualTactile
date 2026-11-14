import copy

import h5py
import numpy as np
import pytest

from tools.tight_fresh_report import audit_trigger_case
from tools import tight_fresh_stage_takeover as stage
from visuotactile.takeover_trigger import (
    CONTACT_HEIGHT_VERSION, VERSION, ContactHeightTrigger, StableGraspTrigger,
    contact_height_windows, make_trigger, stable_windows,
)


def sample(tick, defect=None):
    # Realistic unit quaternions with deliberate relative motion: qualification
    # cannot accidentally depend on the original rigid-grasp constraint.
    angle = tick * .05
    pose = np.array([tick * .002, 0., .53 if tick >= 48 else .5,
        np.cos(angle / 2), 0., 0., np.sin(angle / 2)])
    palm = np.array([0., 0., .5, 1., 0., 0., 0.])
    force = np.zeros((2, 5, 3)); force[:, :2, 2] = .2
    if defect == 'one_substep': force[1, 1] = 0.
    if defect == 'height': pose[2] = .5
    if defect == 'threshold': force[:, 1, 2] = .1
    return pose, palm, force


@pytest.mark.parametrize('delay', [0, 12])
def test_moving_object_qualifies_without_relabeling_stable_grasp(delay):
    config = dict(version=CONTACT_HEIGHT_VERSION, delay_ticks=delay)
    trigger = make_trigger(config, stage.NEVER_FALLBACK)
    assert isinstance(trigger, ContactHeightTrigger)
    for tick in range(60 + delay):
        event = trigger.observe(tick, *sample(tick))
        assert event == (60 + delay if tick == 59 else None)
    receipt = trigger.receipt()
    assert receipt['first_contact_height_window_end_completed_tick'] == 60
    assert receipt['frozen_stable_window_at_trigger'] is False
    assert receipt['stable_grasp_claim'] is False
    assert receipt['native_success_changed'] is False
    assert 'first_stable_window_end_completed_tick' not in receipt
    result = dict(legal=True, software_complete=True, dagger=dict(
        expert_takeover_success=True, student_autonomous_success=False,
        expert_ticks=30, expert_valid_ticks=28))
    audit = dict(category='student_contact_height_then_takeover',
        actual_switch_tick=60 + delay, receipt=receipt)
    assert stage.classify(result, audit) == 'expert_rescue_after_contact_height'
    changed = copy.deepcopy(audit); changed['actual_switch_tick'] += 1
    with pytest.raises(ValueError, match='registered causal'):
        stage.classify(result, changed)
    with pytest.raises(ValueError): trigger.observe(60 + delay, *sample(60 + delay))


@pytest.mark.parametrize('defect', ['one_substep', 'height', 'threshold'])
def test_full_window_requires_height_and_two_contacts_in_both_substeps(defect):
    trigger = make_trigger(dict(version=CONTACT_HEIGHT_VERSION, delay_ticks=0), 100)
    for tick in range(100): assert trigger.observe(tick, *sample(tick, defect)) is None
    assert trigger.receipt()['first_contact_height_window_end_completed_tick'] is None
    result = dict(legal=True, software_complete=True, dagger=dict(
        expert_takeover_success=False, student_autonomous_success=False,
        expert_ticks=0, expert_valid_ticks=0))
    audit = dict(category='ended_before_takeover', actual_switch_tick=None, receipt=trigger.receipt())
    assert stage.classify(result, audit) == 'ended_without_student_contact_height_window'


def test_offline_screen_matches_online_without_changing_original_predicate():
    rows = [sample(t) for t in range(80)]
    pose, palm, force = [np.stack(x) for x in zip(*rows)]
    selected = contact_height_windows(pose, force, .5)
    original, _ = stable_windows(pose, palm, force, .5)
    assert np.flatnonzero(selected)[0] == 59 and not original.any()
    old = make_trigger(dict(version=VERSION, delay_ticks=0), 80)
    assert isinstance(old, StableGraspTrigger)
    for tick, row in enumerate(rows): assert old.observe(tick, *row) is None
    with pytest.raises(ValueError): StableGraspTrigger(dict(version=CONTACT_HEIGHT_VERSION, delay_ticks=0), 80)
    with pytest.raises(ValueError): ContactHeightTrigger(dict(version=VERSION, delay_ticks=0), 80)


def test_audit_recomputes_actual_prefix_and_rejects_receipt_or_authority_changes(tmp_path):
    config = dict(version=CONTACT_HEIGHT_VERSION, delay_ticks=12)
    trigger = make_trigger(config, stage.NEVER_FALLBACK)
    rows = [sample(t) for t in range(90)]
    for tick in range(72): trigger.observe(tick, *rows[tick])
    with h5py.File(tmp_path / 'numeric.h5', 'w') as f:
        t = f.create_group('episode/transition')
        pose, palm, force = [np.stack(x) for x in zip(*rows)]
        t['diagnostics_after/object_pose_w'] = pose
        t['diagnostics_after/palm_body_pose_w'] = palm
        t['object_contact_force_w'] = force
        t['dagger/control_source'] = (np.arange(90) >= 72).astype(int)
    result = dict(dagger=dict(configuration=dict(trigger=config, takeover_tick=stage.NEVER_FALLBACK),
        trigger_receipt=trigger.receipt(), recovery_coverage=dict(switch_tick=72)))
    assert audit_trigger_case(tmp_path, result)['category'] == 'student_contact_height_then_takeover'
    changed = copy.deepcopy(result); changed['dagger']['trigger_receipt']['frozen_stable_window_at_trigger'] = True
    with pytest.raises(ValueError, match='cannot be reproduced'): audit_trigger_case(tmp_path, changed)
    changed = copy.deepcopy(result); changed['dagger']['recovery_coverage']['switch_tick'] = 73
    with pytest.raises(ValueError, match='authority switch'): audit_trigger_case(tmp_path, changed)


def test_explicit_new_plan_rejects_version_mix_and_early_termination_is_not_failure():
    trigger = make_trigger(dict(version=CONTACT_HEIGHT_VERSION, delay_ticks=12), stage.NEVER_FALLBACK)
    for tick in range(62): trigger.observe(tick, *sample(tick))
    result = dict(legal=True, software_complete=True, dagger=dict(
        expert_ticks=0, expert_valid_ticks=0, student_autonomous_success=False))
    audit = dict(category='ended_before_takeover', actual_switch_tick=None, receipt=trigger.receipt())
    assert stage.classify(result, audit) == 'ended_after_contact_height_before_scheduled_takeover'
    plan = dict(stage_takeover_contract=stage.CONTACT_HEIGHT_CONTRACT, never_fallback=stage.NEVER_FALLBACK,
        cases=[dict(name='one', spec=dict(takeover_tick=stage.NEVER_FALLBACK),
            dagger=dict(trigger=dict(version=CONTACT_HEIGHT_VERSION, delay_ticks=12)))])
    stage.validate_plan(plan)
    plan['cases'][0]['dagger']['trigger']['version'] = VERSION
    with pytest.raises(ValueError, match='Registered trigger'): stage.validate_plan(plan)


def test_reference_is_causal_and_missing_or_repeated_transitions_fail():
    trigger = make_trigger(dict(version=CONTACT_HEIGHT_VERSION, delay_ticks=0), stage.NEVER_FALLBACK)
    # A moving/high object during initialization cannot trigger before the
    # preregistered reference interval has actually been observed.
    for tick in range(47):
        pose, palm, force = sample(tick); pose[2] = .8
        assert trigger.observe(tick, pose, palm, force) is None
        assert trigger.height is None
    with pytest.raises(ValueError, match='exactly once'): trigger.observe(48, *sample(48))
    with pytest.raises(ValueError, match='exactly once'): trigger.observe(46, *sample(46))
