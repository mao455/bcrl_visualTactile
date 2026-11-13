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
