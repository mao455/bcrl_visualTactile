import numpy as np
import torch
from visuotactile.adaptive_takeover import profile, AdaptiveTrigger
from visuotactile.batch_recording import recorded_observation, snapshot


def replay(*, never=False, observe=False, gate_tick=None):
    trigger=AdaptiveTrigger(profile('alignment',never_entered_only=never,observe_only=observe),10**9)
    for t in range(100):
        if trigger.scheduled_tick<=t:break
        trigger.observe(t,np.array([0.,0.,.60 if t<48 else .63,1.,0.,0.,0.]),
            np.array([0.,0.,.63,1.,0.,0.,0.]),np.ones((2,5,3)),
            keypoint_max_distance_m=.01 if t==gate_tick else .03,
            goal_index_before=0,goal_index_after=0,
            goal_pose_before=np.array([0.,0.,.64,1.,0.,0.,0.]),
            goal_pose_after=np.array([0.,0.,.64,1.,0.,0.,0.]))
    return trigger


def test_never_entry_uses_entire_completed_history():
    assert replay(never=True).scheduled_tick<10**9
    assert replay(never=True,gate_tick=20).scheduled_tick==10**9
    assert replay(gate_tick=20).scheduled_tick<10**9


def test_observe_only_keeps_student_authority_after_eligibility():
    t=replay(never=True,observe=True)
    assert t.first_eligible_completed_tick is not None
    assert t.scheduled_tick==10**9 and t.trigger_completed_tick is None
    assert t.receipt()['ever_entered_15mm'] is False


def test_history_gate_matches_native_inclusive_float32_boundary():
    for key, expected in ((np.float32(.015), True),
                          (np.nextafter(np.float32(.015), np.float32(1)), False)):
        trigger=AdaptiveTrigger(profile('alignment',never_entered_only=True),10**9)
        pose=np.array([0.,0.,.6,1.,0.,0.,0.])
        trigger.observe(0,pose,pose,np.zeros((2,5,3)),keypoint_max_distance_m=float(key),
            goal_index_before=0,goal_index_after=0,goal_pose_before=pose,goal_pose_after=pose)
        assert trigger.ever_entered_15mm is expected


def test_numeric_selection_preserves_all_written_fields_and_collection_images():
    packet={'joint':torch.randn(2,29),'image':torch.randn(2,3,3,4,4),
            'depth_m':torch.ones(2,3,1,4,4),'depth_valid':torch.ones(2,3,1,4,4),
            'step_id':torch.arange(2),'unused_string':'x'}
    old=snapshot({k:v for k,v in packet.items() if torch.is_tensor(v)})
    new=snapshot(recorded_observation(packet,capture=False))
    assert set(new)==set(old)-{'image','depth_m','depth_valid'}
    for k in new:np.testing.assert_array_equal(new[k],old[k])
    assert set(recorded_observation(packet,capture=True))==set(old)


def test_development_diagnostic_cannot_silently_become_training_capture():
    import pytest
    from visuotactile.position_control import ExpertPositionBridge
    from visuotactile.state_benchmark import dagger_configuration
    config=dict(version=ExpertPositionBridge.version,expert_speed=ExpertPositionBridge.speed_scheme,
        diagnostic_takeover=True,trigger=profile('alignment',never_entered_only=True))
    spec=dict(takeover_tick=10**9,diagnostic_only=True,split='diagnostic_development',cohort='fresh',
        family='development1',source_identity='prefix1',narrow_reset={},event_seed=3)
    for split in ('diagnostic_development','diagnostic_train'):
        assert dagger_configuration(config,dict(spec,split=split),{'splits':{'train':[]}})['diagnostic_takeover']
    with pytest.raises(ValueError):dagger_configuration(config,dict(spec,split='train'),{'splits':{'train':[]}})
    without=dict(config);del without['diagnostic_takeover']
    with pytest.raises(ValueError):dagger_configuration(without,spec,{'splits':{'train':[]}})
