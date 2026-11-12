"""Final-command identity and training/deployment parity without simulator claims."""
import copy
from types import SimpleNamespace as NS

import pytest
import torch

from visuotactile.position_control import (REACHABLE_POSITION_EXECUTION, POSITION_ACTION,
    processed_expert_target, project_processed_position_targets, reachable_position_bounds,
    validate_execution_environment, apply_position_targets)
from visuotactile.models.native_dp import TargetResidual, load_residual
from test_dagger_takeover import environment, native_pipeline


def test_same_native_target_and_history_expert_student_are_exact_after_single_mapping():
    env=environment(False); pipeline=native_pipeline(); rng=torch.Generator().manual_seed(701)
    adapter=REACHABLE_POSITION_EXECUTION
    limits=env.robot.data.joint_pos_limits[:,env._perm_lab_to_canon]
    for _ in range(100):
        previous=env._prev_targets[:,env._perm_lab_to_canon].clone()
        pipeline(env,torch.rand((1,29),generator=rng)*2-1)
        native=env._cur_targets[:,env._perm_lab_to_canon].clone()
        _,expert=processed_expert_target(previous,native,dict(arm=.5,hand=1.,hand_target_cap_rad_s=2.),1/60)
        # Student targets live after native/delay/speed processing, as u_expert
        # does. Going through the shared reachable set must not slow them again.
        student=project_processed_position_targets(expert,previous,limits,adapter)
        assert torch.equal(student,expert)
        # Both interpreted postprocessed requests use the actual shared actuator
        # entry. A native-mapped request first goes through the same one-time
        # conversion regardless of who proposed it; DP itself is postprocessed.
        raw_student=processed_expert_target(previous,native,
            dict(arm=.5,hand=1.,hand_target_cap_rad_s=2.),1/60)[1]
        assert torch.equal(raw_student,expert)
        expert_env=copy.deepcopy(env); student_env=copy.deepcopy(env)
        expert_applied,_=apply_position_targets(expert_env,expert)
        student_applied,_=apply_position_targets(student_env,student)
        assert torch.equal(expert_applied,student_applied)
        assert torch.equal(expert_env._cur_targets,student_env._cur_targets)
        env._prev_targets=expert[:,env._perm_canon_to_lab].clone()
        env._cur_targets=env._prev_targets.clone()


def test_common_reachable_set_clips_only_infeasible_targets_without_label_reslowing():
    limits=torch.tensor([[-2.,2.]]*29); previous=torch.zeros(3,29)
    bounds=reachable_position_bounds(previous,limits,REACHABLE_POSITION_EXECUTION)
    torch.testing.assert_close(bounds[:, :7,1],torch.full((3,7),.00125),atol=1e-9,rtol=0)
    torch.testing.assert_close(bounds[:, 7:,1],torch.full((3,22),2/60),atol=1e-8,rtol=0)
    requested=torch.stack((torch.zeros(29),torch.ones(29),-torch.ones(29)))
    applied=project_processed_position_targets(requested,previous,limits,REACHABLE_POSITION_EXECUTION)
    assert torch.equal(applied[0],requested[0])
    assert torch.equal(applied[1],bounds[1,:,1]) and torch.equal(applied[2],bounds[2,:,0])
    assert torch.equal(applied,project_processed_position_targets(applied,previous,limits,REACHABLE_POSITION_EXECUTION))
    # Repeating Arm .5 here would halve legitimate expert commands and is wrong.
    assert applied[1,0] != processed_expert_target(previous,applied,dict(arm=.5,hand=1.),1/60)[1][1,0]


def test_projection_gradient_is_target_only_and_composition_is_identical_after_reload():
    stats=dict(robot_mean=torch.zeros(178),robot_std=torch.ones(178),action_scale=torch.ones(29),action_offset=torch.zeros(29))
    model=TargetResidual(stats,[.2]*7+[.9]*22,execution_adapter=REACHABLE_POSITION_EXECUTION)
    saved=dict(residual_version=model.version,updates=2,model=model.state_dict(),
        configuration=dict(execution_adapter=REACHABLE_POSITION_EXECUTION))
    deployed=load_residual(saved)
    limits=torch.tensor([[-2.,2.]]*29); previous=torch.zeros(1,2,29,requires_grad=True)
    raw=torch.zeros(1,2,29,requires_grad=True); base=torch.zeros(1,2,29); gate=torch.ones(1,2,dtype=torch.bool)
    action,delta=model.compose(base,raw,gate,limits,previous=previous)
    actual,_=deployed.compose(base,raw,gate,limits,previous=previous)
    assert torch.equal(action,actual)
    loss=model.objective(action,torch.full_like(action,.0001),delta,gate);loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0
    assert previous.grad is None
    with pytest.raises(ValueError,match='real pre-action'):
        model.compose(base,raw,gate,limits)
    old=load_residual(dict(saved,configuration={}))
    assert old.execution_adapter is None
    old_action,_=old.compose(torch.ones_like(base),raw,gate,limits)
    new_action,_=model.compose(torch.ones_like(base),raw,gate,limits,previous=previous)
    assert torch.equal(old_action,torch.ones_like(base)) and not torch.equal(old_action,new_action)


def test_new_identity_requires_actual_environment_parameters():
    metadata=dict(action_control_mode=POSITION_ACTION,control_dt=1/60,
        config=dict(action=dict(dof_speed_scale=1.5,arm_moving_average=.1,hand_moving_average=.1)))
    validate_execution_environment(REACHABLE_POSITION_EXECUTION,metadata)
    changed=copy.deepcopy(metadata);changed['config']['action']['arm_moving_average']=.2
    with pytest.raises(ValueError,match='actual environment'):
        validate_execution_environment(REACHABLE_POSITION_EXECUTION,changed)
