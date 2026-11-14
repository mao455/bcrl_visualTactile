import numpy as np
import pytest
from tools.tight_fresh_phases import authority_phases


def test_expert_grasp_and_subgoal_cannot_be_claimed_as_student_progress():
    source=np.array([0,0,0,1,1,1]);stable=np.array([False,True,False,False,True,True])
    near=np.array([False,False,False,True,True,True]);subgoals=np.array([0,0,0,0,1,2])
    value=authority_phases(source,stable,near,near,subgoals)
    assert value['student_first_stable_window_end_tick']==2
    assert not value['student_stable_window_at_last_tick']
    assert value['student_near_goal_first_tick'] is None
    assert value['student_max_subgoals']==0
    assert value['expert_first_stable_window_end_tick']==5
    assert value['student_ticks']==value['expert_ticks']==3


def test_authority_cannot_switch_back_silently():
    with pytest.raises(ValueError,match='prefix'):
        authority_phases(np.array([0,1,0]),np.ones(3,dtype=bool),np.ones(3,dtype=bool),
            np.ones(3,dtype=bool),np.zeros(3))


def test_grasp_progress_excludes_later_goals_and_counts_loss_of_stability():
    from tools.tight_fresh_phases import grasp_progress
    r=grasp_progress(np.array([0,0,1,1,0,1,0],bool),np.array([9.,8.,7.,6.,5.,4.,.1]),
        np.array([2.,1.8,1.6,1.4,1.2,1.,.01]),np.array([0,0,0,0,0,0,1]))
    assert r['first_tick']==3 and r['last_tick']==6 and r['stable_ticks']==3
    assert r['longest_continuous_stable_ticks']==2 and not r['stable_at_final_tick']
    assert r['first_stable_goal_distance_m']==7 and r['subsequent_first_goal_min_distance_m']==4
    assert r['subsequent_first_goal_min_angle_rad']==1 and r['first_goal_last_distance_m']==4
    assert grasp_progress(np.zeros(2,bool),np.ones(2),np.ones(2),np.zeros(2))['first_tick'] is None
