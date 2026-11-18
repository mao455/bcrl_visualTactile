import numpy as np
import pytest
from visuotactile.deviation_takeover import profile,DeviationTrigger
from visuotactile.takeover_trigger import make_trigger

def step(t,i,*,z=.5,fingers=3,goal=0,dist=.1,x=0):
 force=np.zeros((2,5,3));force[:,:fingers,2]=1
 return t.observe(i,np.array([x,0,z,1,0,0,0]),np.array([0,0,.5,1,0,0,0]),force,keypoint_max_distance_m=dist,goal_index_before=goal,goal_index_after=goal,goal_pose_before=np.array([0,0,.7,1,0,0,0]),goal_pose_after=np.array([0,0,.7,1,0,0,0]))
def test_no_normal_relative_motion_takeover():
 t=make_trigger(profile(),10**9)
 for i in range(100):assert step(t,i,z=.5 if i<48 else .55,fingers=1 if i>=70 else 3,x=.003*(i-70) if i>=70 else 0) is None
 assert t.receipt()['trigger_completed_tick'] is None

def test_height_loss_triggers_next_action_without_grid():
 t=DeviationTrigger(profile(),10**9)
 for i in range(62):assert step(t,i,z=.5 if i<48 else .55) is None
 assert step(t,62,z=.535,fingers=1) is None
 assert step(t,63,z=.53,fingers=1)==64
 assert t.receipt()['first_regression_transition_tick']==62
 with pytest.raises(ValueError):step(t,64)

def test_supported_release_after_subgoal_does_not_trigger():
 t=DeviationTrigger(profile(),10**9)
 for i in range(90):assert step(t,i,z=.5 if i<48 else (.55 if i<62 else .53),fingers=3 if i<62 else 0,goal=1) is None

def test_lane_state_independent_and_no_fallback():
 a=make_trigger(profile(),10**9);b=make_trigger(profile(),10**9)
 for i in range(80):step(a,i)
 assert b.records==[] and a.scheduled_tick==10**9
 with pytest.raises(ValueError):make_trigger(profile(),72)

def test_new_trigger_classification_never_asks_for_delay_ticks():
 from tools.tight_fresh_stage_takeover import classify
 t=DeviationTrigger(profile(),10**9)
 for i in range(62):step(t,i,z=.5 if i<48 else .55)
 step(t,62,z=.535,fingers=1);step(t,63,z=.53,fingers=1)
 result=dict(legal=True,software_complete=True,dagger=dict(expert_takeover_success=True))
 audit=dict(receipt=t.receipt(),category='student_observable_regression_then_takeover',actual_switch_tick=64)
 assert classify(result,audit)=='expert_rescue_after_observable_regression'
