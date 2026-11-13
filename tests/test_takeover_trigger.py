import numpy as np
import pytest
from visuotactile.takeover_trigger import StableGraspTrigger, VERSION, stable_windows, validate_trigger


def sample(tick, defect=None):
    pose=np.array([0.,0.,.5 if tick<48 else .53,1.,0.,0.,0.])
    palm=pose.copy();palm[0]=.03
    force=np.zeros((2,5,3));force[:,:2,2]=.2
    if defect=='substep':force[1,1]=0
    if defect=='height':pose[2]=palm[2]=.5
    if defect=='drift':palm[0]=tick*.02
    return pose,palm,force


@pytest.mark.parametrize('delay',[0,24,120])
def test_causal_completed_window_then_future_action(delay):
    trigger=StableGraspTrigger(dict(version=VERSION,delay_ticks=delay),360)
    schedule=None
    for tick in range(60):
        result=trigger.observe(tick,*sample(tick))
        if result is not None:schedule=result
        if tick<59:assert result is None
    assert schedule==60+delay
    assert trigger.receipt()['first_stable_window_end_completed_tick']==60
    assert trigger.receipt()['settled_height_m']==.5
    # A future fall cannot change an already issued scheduling decision.
    if delay:
        assert trigger.observe(60,*sample(60,'height')) is None
        assert trigger.scheduled_tick==schedule


@pytest.mark.parametrize('defect',['height','substep','drift'])
def test_contact_alone_or_one_good_substep_does_not_trigger(defect):
    trigger=StableGraspTrigger(dict(version=VERSION,delay_ticks=0),80)
    for tick in range(80):assert trigger.observe(tick,*sample(tick,defect)) is None
    assert trigger.receipt()['reason']=='fallback_no_completed_stable_window'
    assert trigger.receipt()['first_stable_window_end_completed_tick'] is None
    with pytest.raises(ValueError,match='completed student'):trigger.observe(80,*sample(80))


def test_fallback_is_never_extended_and_no_duplicate_or_skipped_observations():
    trigger=StableGraspTrigger(dict(version=VERSION,delay_ticks=30),80)
    with pytest.raises(ValueError):trigger.observe(1,*sample(1))
    for tick in range(80):assert trigger.observe(tick,*sample(tick)) is None
    assert trigger.scheduled_tick==80
    assert trigger.receipt()['reason']=='fallback_before_requested_stable_delay'


@pytest.mark.parametrize('value',[dict(version='unknown',delay_ticks=0),dict(version=VERSION,delay_ticks=-1),
    dict(version=VERSION,delay_ticks=True),dict(version=VERSION,delay_ticks=0,threshold=0)])
def test_unknown_or_mutable_threshold_contract_rejected(value):
    with pytest.raises(ValueError):validate_trigger(value,360)


def test_offline_window_and_causal_online_agree_after_reference_available():
    rows=[sample(t) for t in range(120)]
    pose,palm,force=[np.stack(x) for x in zip(*rows)]
    offline,_=stable_windows(pose,palm,force,float(np.median(pose[24:48,2])))
    trigger=StableGraspTrigger(dict(version=VERSION,delay_ticks=30),110)
    for tick,row in enumerate(rows):
        event=trigger.observe(tick,*row)
        if event is not None:break
    first=int(np.flatnonzero(offline & (np.arange(120)>=47))[0]+1)
    assert trigger.first_stable_completed_tick==first==60 and event==first+30
