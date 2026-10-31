import copy
import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from diffusers import DDPMScheduler

from visuotactile.diagnostic_recording import snapshot_before_policy, write_policy_step
from visuotactile.dp_objectives import recorded_objective
from visuotactile.evaluate_training_initials import initialize
from visuotactile.models.long_horizon import LongHorizonPolicy


def test_diagnostic_session_records_sources_before_simulator_start(tmp_path):
    evidence=tmp_path/'evidence'
    evidence.mkdir()
    (evidence/'dataset_distribution.json').write_text(json.dumps({'selected_train':['episode0']}))
    root=tmp_path/'diagnostics'
    before=torch.get_rng_state().clone()
    initialize(root,tmp_path/'checkpoints',evidence)
    session=json.loads((root/'session.json').read_text())
    assert session['diagnostic_input_version']==2
    assert session['selected_train']==['episode0']
    assert session['parameter_updates']==0
    assert 'visuotactile/diagnostic_recording.py' in session['source_sha256']
    assert all(Path(path).is_file() and len(value)==64 for path,value in session['source_sha256'].items())
    assert torch.equal(torch.get_rng_state(),before)
    with pytest.raises(FileExistsError,match='Fresh'):
        initialize(root,tmp_path/'checkpoints',evidence)


def test_formal_default_loss_balances_arm_hand_and_masks_tail():
    prediction=torch.nn.Parameter(torch.cat((torch.full((7,),2.),torch.ones(22))))
    model=SimpleNamespace(target_scale=torch.ones(29),target_offset=torch.zeros(29),
        cfg=SimpleNamespace(prediction_type='sample'),noise=DDPMScheduler(num_train_timesteps=100),
        denoiser=lambda noisy,timestep,global_cond:prediction.expand_as(noisy))
    target=torch.zeros(2,64,29)
    valid=torch.zeros(2,64,dtype=torch.bool)
    valid[0,:12]=True
    valid[1,:3]=True
    condition=torch.zeros(2,1)
    loss=LongHorizonPolicy.loss_from_condition(model,condition,target,valid)
    assert float(loss)==pytest.approx(2.5)
    loss.backward()
    assert float(prediction.grad[:7].sum())==pytest.approx(2.)
    assert float(prediction.grad[7:].sum())==pytest.approx(1.)
    historical=LongHorizonPolicy.loss_from_condition(model,condition,target,valid,objective='A')
    assert float(historical)==pytest.approx(50/29)
    with pytest.raises(ValueError,match='objective'):
        LongHorizonPolicy.loss_from_condition(model,condition,target,valid,objective='finger_equal')
    assert recorded_objective({})=='A'
    assert recorded_objective({'loss':'B'})=='B'


def test_native_hook_preserves_pinned_upstream_targets_queue_and_rng():
    path='isaacsimenvs/tasks/play/utils/action_utils.py'
    original=subprocess.check_output(['git','show',f'ff2fc62873f6c6cd93164123c8358e9ce2d65c9b:{path}'],text=True)
    namespace={}
    exec(compile(original,'pinned_upstream_action_utils.py','exec'),namespace)
    upstream=namespace['apply_action_pipeline']
    current=runpy.run_path(path)['apply_action_pipeline']
    permutation=torch.cat((torch.arange(7),torch.arange(28,6,-1)))
    env=SimpleNamespace(num_envs=4,device='cpu',step_dt=1/60,
        cfg=SimpleNamespace(domain_randomization=SimpleNamespace(use_action_delay=True,action_delay_max=3),
            action=SimpleNamespace(dof_speed_scale=1.5,arm_moving_average=.1,hand_moving_average=.1)),
        _perm_canon_to_lab=permutation,episode_length_buf=torch.tensor([0,0,2,0]),_successes=torch.tensor([0,1,0,0]),
        _action_queue=torch.zeros(4,3,29),_prev_targets=torch.full((4,29),.99),_cur_targets=torch.zeros(4,29),
        _arm_joint_ids=torch.arange(7),_hand_joint_ids=torch.arange(7,29),
        _arm_lower=torch.full((4,7),-1.),_arm_upper=torch.ones(4,7),
        _hand_lower=torch.full((4,22),-1.),_hand_upper=torch.ones(4,22))
    expected=copy.deepcopy(env)
    captured=[]
    env._record_action_trace=lambda actions,index,start:captured.append((actions.clone(),index.clone(),start.clone()))
    generator=torch.Generator().manual_seed(71)
    with torch.random.fork_rng(devices=[]):
        for tick in range(8):
            command=torch.rand(4,29,generator=generator)*2-1
            rng=torch.get_rng_state()
            upstream(expected,command)
            expected_rng=torch.get_rng_state()
            torch.set_rng_state(rng)
            current(env,command)
            assert torch.equal(torch.get_rng_state(),expected_rng)
            for name in ('_action_queue','_prev_targets','_cur_targets'):
                assert torch.equal(getattr(env,name),getattr(expected,name)),name
            assert torch.equal(captured[-1][0],env._action_queue[torch.arange(4),captured[-1][1]])
            env.episode_length_buf+=1
            expected.episode_length_buf+=1
    assert not captured[0][2][1]
    assert captured[0][2][0]


def test_diagnostic_snapshot_records_native_tick_without_rng_or_state_changes(tmp_path):
    packet=dict(episode_id=torch.tensor([9]),step_id=torch.tensor([1]),time_s=torch.tensor([1/60]),
        image=torch.zeros(1,3,4,2,2),depth_valid=torch.ones(1,3,1,2,2,dtype=torch.bool),
        contact_valid=torch.tensor([True]),contact_substeps_w_N=torch.ones(1,2,5,3),
        contact_time_s=torch.tensor([[1/120,1/60]]),finger_position_base_m=torch.ones(1,2,5,3),
        proprio=torch.zeros(1,58))
    env=SimpleNamespace(_action_queue=torch.zeros(1,3,29),_prev_targets=torch.ones(1,29),
        episode_length_buf=torch.tensor([1]),_successes=torch.tensor([0]),research_packet=packet)
    controller=SimpleNamespace(episode=9,state=torch.ones(1,1,128,dtype=torch.bfloat16),
        generator=torch.Generator().manual_seed(31),last_condition=torch.zeros(1,512),reference=torch.zeros(1,29),
        plan=torch.zeros(1,64,29),records={'action':[torch.ones(61)]})
    native_rng=torch.get_rng_state().clone()
    sampling_rng=controller.generator.get_state().clone()
    snapshot=snapshot_before_policy(env,controller)
    queue=env._action_queue.clone()
    state=controller.state.clone()
    with h5py.File(tmp_path/'trace.h5','w') as stream:
        write_policy_step(stream,1,packet,snapshot,torch.zeros(1,29),controller)
        write_policy_step(stream,12,packet,snapshot,torch.zeros(1,29),controller)
        assert 'contact_valid' in stream['1'] and 'finger_position_base_m' in stream['1']
        assert 'rgb_u8' not in stream['1'] and 'rgb_u8' in stream['12']
        assert np.array_equal(stream['1/before_policy/sampling_rng_before_policy'][:],sampling_rng.numpy())
        assert stream['1/policy/memory'].attrs['torch_dtype']=='torch.bfloat16'
    assert torch.equal(torch.get_rng_state(),native_rng)
    assert torch.equal(controller.generator.get_state(),sampling_rng)
    assert torch.equal(env._action_queue,queue)
    assert torch.equal(controller.state,state)
    snapshot['action_queue_lab'].fill_(3.)
    assert torch.equal(env._action_queue,queue)
    restored=torch.Generator().set_state(sampling_rng)
    assert torch.equal(torch.randn(1,64,29,generator=restored),torch.randn(1,64,29,generator=controller.generator))
