import numpy as np
import pytest
from visuotactile.adaptive_takeover import AdaptiveTrigger, profile
from visuotactile.takeover_trigger import make_trigger, validate_trigger


def observation(t, *, held=False, fingers=3, goal=0, change=False, x=0., z=None, key=.06):
    return (np.array([x,0.,.5 if z is None else z,1.,0.,0.,0.]),
            np.array([0.,0.,.5,1.,0.,0.,0.]),
            np.pad(np.full((2,fingers,1),.2),((0,0),(0,5-fingers),(0,2))),
            dict(keypoint_max_distance_m=key,goal_index_before=goal,
                 goal_index_after=goal+int(change),
                 goal_pose_before=np.array([.04,0.,.5,1.,0.,0.,0.]),
                 goal_pose_after=np.array([.04,0.,.5,1.,0.,0.,0.]) if not change else np.array([.05,0.,.5,1.,0.,0.,0.])))


def push(trigger,t,**kwargs):
    pose,palm,force,extra=observation(t,**kwargs)
    return trigger.observe(t,pose,palm,force,**extra)


def test_contact_stall_cannot_use_old_fixed_times_or_round_to_plan():
    trigger=make_trigger(profile('contact'),10**9)
    for t in range(96):
        assert push(trigger,t,fingers=0) is None
    assert push(trigger,96,fingers=0)==97
    assert trigger.receipt()['trigger_completed_tick']==97
    assert trigger.scheduled_tick%12!=0
    with pytest.raises(ValueError,match='once'):push(trigger,97,fingers=0)


def test_transport_requires_real_held_history_then_travel():
    trigger=AdaptiveTrigger(profile('transport'),10**9)
    for t in range(100):
        assert push(trigger,t,fingers=3,z=.5 if t<60 else .53,x=0.) is None
    assert push(trigger,100,z=.53,x=.04) is None
    assert push(trigger,101,z=.53,x=.04)==102


def test_alignment_stall_requires_unchanged_goal_and_completed_window():
    trigger=AdaptiveTrigger(profile('alignment'),10**9)
    for t in range(70):
        assert push(trigger,t,z=.5 if t<60 else .53,change=t==69) is None
    for t in range(70,105):
        assert push(trigger,t,z=.53) is None
    assert push(trigger,105,z=.53) is None
    assert push(trigger,106,z=.53)==107


def test_instability_detects_before_complete_drop_and_replays_exactly():
    live=AdaptiveTrigger(profile('instability'),10**9)
    replay=AdaptiveTrigger(profile('instability'),10**9)
    for t in range(92):
        kwargs=dict(z=.5 if t<60 else .53 if t<90 else .524,fingers=3 if t<90 else 1)
        a=push(live,t,**kwargs);b=push(replay,t,**kwargs)
        assert a==b
    assert live.scheduled_tick==92
    assert live.receipt()==replay.receipt()
    assert live.at_trigger['fingers']==1


def test_post_subgoal_excludes_goal_change_boundary():
    trigger=AdaptiveTrigger(profile('post_subgoal'),10**9)
    for t in range(82):
        assert push(trigger,t,z=.5 if t<60 else .53,goal=0 if t<70 else 1,change=t==69) is None
    assert push(trigger,82,z=.53,goal=1)==83
    assert trigger.scheduled_tick==83



def test_no_stage_no_fallback_and_malformed_profiles_rejected():
    trigger=AdaptiveTrigger(profile('post_subgoal'),10**9)
    for t in range(250):assert push(trigger,t,fingers=0) is None
    assert trigger.scheduled_tick==10**9
    with pytest.raises(ValueError):validate_trigger(profile('contact'),240)
    bad=profile('contact');bad['earliest_contact_tick']=60
    with pytest.raises(ValueError):validate_trigger(bad,10**9)
