import numpy as np
import pytest
from tools.tight_fresh_command_chain import (angle, compare_recorded, compose_pose, direction,
    first_lost_window, grasp_motion, relative_pose, window_goal_identity)


def pose(position=(0,0,0), angle_z=0.):
    return np.array([*position,np.cos(angle_z/2),0,0,np.sin(angle_z/2)],dtype=np.float64)


def test_rigid_attachment_turn_separates_palm_motion_from_slip():
    palm0=pose(); obj0=pose((1,0,0)); palm1=pose((0,0,0),np.pi/2)
    obj1=compose_pose(palm1,relative_pose(obj0,palm0));goal=obj1.copy()
    assert np.allclose(relative_pose(obj1,palm1),relative_pose(obj0,palm0))
    motion=grasp_motion(obj0,palm0,obj1,palm1,goal)
    assert motion['relative_palm_translation_m']<1e-14
    assert motion['relative_palm_rotation_rad']==0
    assert motion['rigid_attachment_endpoint_window_start_goal_root_angle_rad']==0
    # Same observed palm turn with a stationary object demonstrates attachment
    # drift; the kinematic rigid estimate is not mistaken for the actual object.
    slipping=grasp_motion(obj0,palm0,obj0,palm1,goal)
    assert slipping['actual_window_start_goal_root_angle_rad'][1]==pytest.approx(np.pi/2)
    assert slipping['rigid_attachment_endpoint_window_start_goal_root_angle_rad']==0
    assert slipping['relative_palm_rotation_rad']==pytest.approx(np.pi/2)


def test_direction_denominators_do_not_count_tiny_or_zero_motion_as_opposite():
    report=direction(np.array([1.,1.,1e-8,0.]),np.array([-1.,0.,-1.,0.]))
    assert report['components']==4
    assert report['both_active_components']==1
    assert report['opposite_active_components']==1
    assert report['opposite_fraction_among_both_active']==1
    assert direction(np.zeros(4),np.zeros(4))['cosine'] is None


def test_normalized_quaternion_angle_and_first_failed_window_conditions():
    q=np.array([1.,0,0,0]);assert angle(q*(1-1e-7),-q)==0
    n=15;palm=np.tile(pose(),(n,1));obj=np.tile(pose((0,0,.03)),(n,1))
    force=np.ones((n,2,5,3));stable=np.zeros(n,bool);stable[11:14]=True
    force[14]=0
    out=first_lost_window(stable,obj,palm,force,0.)
    assert out['completed_tick']==15 and out['first_stable_completed_tick']==12
    assert out['all_two_contact_fingers_both_substeps'] is False
    assert out['all_above_settled_plus_2cm'] is True
    assert out['translation_condition'] and out['rotation_condition']


def test_recorded_chain_guard_rejects_wrong_alignment_and_changed_arrays():
    a=np.zeros((3,29),dtype=np.float32)
    assert compare_recorded(a,a.copy(),'target',exact=True)==0
    with pytest.raises(ValueError,match='differs'):
        compare_recorded(a,a+1e-4,'target')
    with pytest.raises(ValueError,match='shape'):
        compare_recorded(a,a[:2],'time')


def test_goal_switch_keeps_motion_reference_explicit_and_native_goals_separate():
    old=pose((0,0,0));new=pose((1,0,0),np.pi/2)
    before=np.array([old,old,new,new]);after=np.array([old,new,new,new])
    identity=window_goal_identity(before,after,0,4,subgoals_before=[0,0,1,1],subgoals_after=[0,1,1,1])
    assert identity['recorded_goal_pose_changed_within_window'] is True
    assert identity['native_subgoal_transition_completed_ticks']==[2]
    assert identity['native_subgoal_counter_start_before_end_after']==[0,1]
    assert identity['posttransition_goal_pose_change_completed_ticks']==[2]
    assert identity['preaction_goal_pose_change_completed_ticks']==[3]
    assert identity['native_keypoint_start_end_same_preaction_goal'] is False
    assert identity['goal_before_pose_sha256']!=identity['goal_after_pose_sha256']
    motion=grasp_motion(old,pose(),new,pose(),before[0])
    assert motion['goal_reference']=='window_start_preaction_goal_pose'
    assert motion['actual_window_start_goal_root_distance_m']==[0.,1.]
    assert motion['actual_window_start_goal_root_angle_rad'][1]==pytest.approx(np.pi/2)
    # The actual endpoint equals the NEW goal, but the reported endpoint
    # deliberately remains measured against the named OLD window-start goal.
    assert angle(new[3:],before[-1,3:])==0
    tail=window_goal_identity(before,after,2,4)
    assert not tail['recorded_goal_pose_changed_within_window']
    assert tail['native_keypoint_start_end_same_preaction_goal']
    assert tail['preaction_goal_pose_change_completed_ticks']==[]
    assert tail['posttransition_goal_pose_change_completed_ticks']==[]


def test_goal_quaternion_sign_change_is_not_a_native_subgoal():
    before=np.array([pose()]);after=before.copy();after[:,3:]*=-1
    result=window_goal_identity(before,after,0,1,subgoals_before=[0],subgoals_after=[0])
    assert result['recorded_goal_pose_changed_within_window']
    assert result['posttransition_goal_pose_change_completed_ticks']==[1]
    assert result['native_subgoal_transition_completed_ticks']==[]
    assert result['maximum_same_transition_goal_root_rotation_rad']==0
