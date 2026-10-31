import copy
import importlib.util
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import h5py
import pytest
import torch

from visuotactile.action_reference import command_to_reference, reference_to_command
from visuotactile.replay_events import RecordedEvents, bitwise_equal, error_summary, pose_error
from visuotactile.replay_fresh import append_numeric


def native_environment():
    return SimpleNamespace(num_envs=1,device='cpu',step_dt=1/60,
        cfg=SimpleNamespace(domain_randomization=SimpleNamespace(use_action_delay=True,action_delay_max=3),
            action=SimpleNamespace(dof_speed_scale=1.5,arm_moving_average=.1,hand_moving_average=.1)),
        _perm_canon_to_lab=torch.cat((torch.arange(7),torch.arange(28,6,-1))),
        episode_length_buf=torch.zeros(1,dtype=torch.long),_successes=torch.zeros(1,dtype=torch.long),
        _action_queue=torch.zeros(1,3,29),_prev_targets=torch.zeros(1,29),_cur_targets=torch.zeros(1,29),
        _arm_joint_ids=torch.arange(7),_hand_joint_ids=torch.arange(7,29),
        _arm_lower=torch.full((1,7),-1.),_arm_upper=torch.ones(1,7),
        _hand_lower=torch.full((1,22),-1.),_hand_upper=torch.ones(1,22))


def test_recorded_delay_uses_queue_preserves_draws_and_roundtrip_targets():
    pipeline=runpy.run_path('isaacsimenvs/tasks/play/utils/action_utils.py')['apply_action_pipeline']
    replay=native_environment()
    recovered=copy.deepcopy(replay)
    normal=copy.deepcopy(replay)
    traces=[]
    replay._record_action_trace=lambda command,index,start:traces.append((command.clone(),index.clone(),start.clone()))
    generator=torch.Generator().manual_seed(37)
    history=[]
    limits=torch.stack((-torch.ones(29),torch.ones(29)),-1)
    with torch.random.fork_rng(devices=[]):
        for step,index in enumerate((2,0,2,1,2,0,1,2)):
            command=torch.rand(1,29,generator=generator)*2-1
            if step==4:
                command=history[-1].clone()
                replay.episode_length_buf.zero_()
                replay._successes.fill_(1)
                recovered.episode_length_buf.zero_()
                recovered._successes.fill_(1)
            history.append(command)
            target=recovered._prev_targets[:,recovered._perm_canon_to_lab]
            reference=command_to_reference(command,target,limits)
            converted=reference_to_command(reference,target,limits)
            replay._recorded_delay_index=torch.tensor([index])
            recovered._recorded_delay_index=torch.tensor([index])
            rng=torch.get_rng_state()
            pipeline(normal,command)
            expected_rng=torch.get_rng_state()
            torch.set_rng_state(rng)
            pipeline(replay,command)
            assert torch.equal(torch.get_rng_state(),expected_rng)
            pipeline(recovered,converted)
            expected=history[max(0,step-index)][:,replay._perm_canon_to_lab]
            assert torch.equal(traces[-1][0],expected)
            assert int(traces[-1][1][0])==index
            assert bool(traces[-1][2][0])==(step==0)
            assert torch.allclose(replay._cur_targets,recovered._cur_targets,atol=3e-6,rtol=0)
            replay.episode_length_buf+=1
            recovered.episode_length_buf+=1
            normal.episode_length_buf+=1
    replay._recorded_delay_index=torch.tensor([3])
    with pytest.raises(ValueError,match='queue contract'):
        pipeline(replay,torch.zeros(1,29))


def test_event_boundaries_and_error_reporting():
    group={'transition/control_trace/delay_index':np.array([2,1]),
        'transition/applied_wrench/force_w_N':np.ones((2,2,1,3),dtype=np.float32),
        'transition/applied_wrench/torque_w_Nm':np.ones((2,2,1,3),dtype=np.float32)*.2,
        'transition/external_force_after_dr_w_N':np.ones((2,1,3),dtype=np.float32),
        'transition/external_torque_after_dr_w_Nm':np.ones((2,1,3),dtype=np.float32)*.2}
    events=RecordedEvents(group,'delay_wrench_fixed')
    assert events.at(1)['source_step']==1
    with pytest.raises(IndexError):
        events.at(2)
    extended=RecordedEvents(group,'delay_wrench_fixed',extend=True)
    assert extended.at(2)['cycle']==1 and extended.at(2)['source_step']==0
    error=error_summary(np.array([[0.,0.],[2.,0.]]),np.zeros((2,2)),.1)
    assert error['first_difference']==dict(step=1,flat_component=0,source=0.,replay=2.,error=2.)
    position,angle=pose_error(np.array([[0.,0.,0.,-1.,0.,0.,0.]]),np.array([[0.,0.,0.,1.,0.,0.,0.]]))
    assert position[0]==0 and angle[0]==0


def test_fresh_numeric_stream_preserves_native_batch_and_step_shapes(tmp_path):
    with h5py.File(tmp_path/'sample.h5','w') as stream:
        group=stream.create_group('episode')
        for step in range(2):
            append_numeric(group,dict(obs={'step_id':torch.tensor([step])},
                transition={'applied_wrench':{'force_w_N':torch.full((1,2,1,3),float(step))}}),step)
        assert group['obs/step_id'][:].tolist()==[0,1]
        assert group['transition/applied_wrench/force_w_N'].shape==(2,2,1,3)
        assert np.array_equal(group['transition/applied_wrench/force_w_N'][1],np.ones((2,1,3)))


def test_native_request_avoids_observed_world_body_roundtrip_error():
    location=Path(importlib.util.find_spec('isaaclab').origin).parent/'utils/math.py'
    spec=importlib.util.spec_from_file_location('replay_isaac_math',location)
    math_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_module)
    quaternion=torch.tensor([[0.6082516312599182,-0.10827945172786713,-0.7794876098632812,-0.10346260666847229]])
    request=torch.tensor([[-1.1757628917694092,-1.7539052963256836,-2.4367024898529053]])
    body=torch.tensor([[-2.162315845489502,-2.0937492847442627,1.1561675071716309]])
    assert torch.equal(math_module.quat_apply_inverse(quaternion,request),body)
    actual_world=math_module.quat_apply(quaternion,body)
    assert not torch.equal(math_module.quat_apply_inverse(quaternion,actual_world),body)


def test_bitwise_reporting_distinguishes_signed_zero_and_storage_dtype():
    positive=np.array([0.],dtype=np.float32)
    negative=np.array([-0.],dtype=np.float32)
    assert error_summary(positive,negative,0)['maximum']==0
    assert not bitwise_equal(positive,negative)
    assert not bitwise_equal(positive,positive.astype(np.float64))
    assert bitwise_equal(negative,negative.copy())
