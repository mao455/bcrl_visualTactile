import math

import pytest

from visuotactile.long_resources import update_pacing


def sample(stamp, utilization, rolling=40.):
    return dict(utc_epoch=stamp,gpu_util_percent=utilization,rolling_60s_percent=rolling)


def test_shared_feedback_does_not_adjust_twice_for_same_sample():
    state = {}
    for stamp in range(11):
        state = update_pacing(state,sample(stamp,20,20))
    ratio = state['sleep_work_ratio']
    assert ratio < 1
    for worker in range(50):
        state = update_pacing(state,sample(10,20,20))
    assert state['sleep_work_ratio']==ratio
    assert len(state['observations'])==11


def test_feedback_targets_forty_and_reacts_to_increased_load():
    state = {}
    utilization = []
    rolling = []
    for stamp in range(360):
        current = 60/(1+state.get('sleep_work_ratio',1.))
        utilization.append(current)
        mean = sum(utilization[-60:])/len(utilization[-60:])
        rolling.append(mean)
        state = update_pacing(state,sample(stamp,current,mean))
    assert 37 <= rolling[-1] <= 43
    assert max(rolling[60:]) < 50
    old_ratio = state['sleep_work_ratio']
    for stamp in range(360,380):
        state = update_pacing(state,sample(stamp,55,46))
    assert state['sleep_work_ratio'] > old_ratio


def test_missing_short_window_never_becomes_zero_utilization():
    state = {}
    for stamp in range(11):
        state = update_pacing(state,sample(stamp,None if stamp==5 else 10,10))
    assert state['short_mean_percent'] is None
    assert state['sleep_work_ratio']==1.
    state = update_pacing(state,sample(30,10,10))
    assert state['short_mean_percent'] is None
    assert math.isfinite(state['sleep_work_ratio'])
    with pytest.raises(ValueError,match='below the 50'):
        update_pacing({},sample(0,0),target=50)
