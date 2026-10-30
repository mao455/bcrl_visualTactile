"""Causal execution feedback, native configuration and version admission tests."""
import copy
from types import SimpleNamespace

import h5py
import pytest
import torch

from test_causal_memory import fixture_file
from test_perception_v2 import packet
from visuotactile.action_history import ExecutionHistory
from visuotactile.dataset import EpisodeSequenceDataset
from visuotactile.observations import nominal_keys
from visuotactile.rollout import load_command_controller, NominalController, TactileCommandController
from visuotactile.train_dp import _make_model, _condition, _save
from visuotactile.models.nominal import PolicyConfig


def transition(observation, command):
    return dict(episode_id=observation['episode_id'],step_id=observation['step_id'],
        command_clipped_pre_delay=command,time_before_s=observation['time_s'],
        time_after_s=observation['time_s']+1/60,
        control_trace=dict(valid=torch.ones(len(command),dtype=torch.bool),delayed_command=-command,
                           selected_command_time_s=observation['time_s']))


def test_offline_history_is_past_execution_across_chunks_and_label_mutation(tmp_path):
    path=tmp_path/'history.h5'
    fixture_file(path)
    with h5py.File(path,'a') as stream:
        group=next(iter(stream['episodes'].values()))
        group.create_dataset('transition/control_trace/delayed_command',data=-group['transition/command_clipped_pre_delay'][:])
        group.create_dataset('transition/control_trace/selected_command_time_s',data=torch.arange(4).double()/60)
        group.create_dataset('transition/control_trace/valid',data=[True]*4)
        group.create_dataset('transition/time_before_s',data=torch.arange(4).double()/60)
        group.create_dataset('transition/time_after_s',data=torch.arange(1,5).double()/60)
    dataset=EpisodeSequenceDataset(path,include_action_history=True)
    before=list(dataset.chunks(dataset.episodes[0],2))
    assert not before[0]['obs']['action_history_valid'][0,0]
    torch.testing.assert_close(before[1]['obs']['previous_issued_command'][0,0],torch.full((29,),.1))
    torch.testing.assert_close(before[1]['obs']['previous_delayed_command'][0,0],torch.full((29,),-.1))
    with h5py.File(path,'a') as stream:
        group=next(iter(stream['episodes'].values()))
        group['transition/command_clipped_pre_delay'][2:]=.8
    after=list(dataset.chunks(dataset.episodes[0],2))
    for name in nominal_keys(2,'issued'):
        torch.testing.assert_close(before[1]['obs'][name][:,0],after[1]['obs'][name][:,0],rtol=0,atol=0)
    assert not torch.equal(before[1]['action_target'][0],after[1]['action_target'][0])
    assert not torch.equal(before[1]['obs']['previous_issued_command'][:,1],after[1]['obs']['previous_issued_command'][:,1])


@pytest.mark.parametrize('variant',['b0','b1'])
def test_history_actual_controller_sequence_gradients_reset_and_version(tmp_path,variant):
    config=PolicyConfig(views=3,own_plan_mode='none',action_history_mode='issued')
    model=_make_model(variant,config).eval()
    controller=(NominalController if variant=='b0' else TactileCommandController)(model,2)
    packets=[]
    for step in range(8):
        observation=packet(t=step,views=3)
        command=controller.act(observation)
        packets.append(controller.last_packet)
        controller.observe_execution(transition(observation,command))
    observations={name:torch.stack([row[name] for row in packets],1) for name in nominal_keys(2,'issued')}
    observations['valid']=torch.ones(2,8,dtype=torch.bool)
    tactile_names=('contact_substeps_w_N','contact_substeps_local_N','finger_position_base_m',
                   'finger_rotation_base','contact_time_s')
    data=dict(obs=observations,tactile={name:torch.stack([row[name] for row in packets],1) for name in tactile_names},
              tactile_valid=torch.stack([row['contact_valid'] for row in packets],1))
    sequence,_=_condition(model,variant,data,None)
    torch.testing.assert_close(sequence[:,-1],controller.condition_rows(torch.ones(2,dtype=torch.bool)),atol=2e-6,rtol=2e-5)
    changed=copy.deepcopy(data)
    changed['obs']['previous_issued_command'][:,1]*=-1
    altered,_=_condition(model,variant,changed,None)
    torch.testing.assert_close(sequence[:,0],altered[:,0],rtol=0,atol=0)
    assert (sequence[:,1:]-altered[:,1:]).abs().max()>1e-6
    sequence[:,-1].square().sum().backward()
    encoder=model.encoder if variant=='b0' else model.visual
    assert encoder.action_history[0].weight.grad.norm()>0
    assert torch.isfinite(encoder.action_history[0].weight.grad).all()
    optimizer=torch.optim.AdamW(model.parameters())
    options=SimpleNamespace(variant=variant,chunk_size=2,batch_size=2,seed=41,amp='off')
    path=tmp_path/'history.pt'
    _save(path,model,optimizer,config,options,'fixture',(['a'],[],[]),1,[],torch.zeros(58),torch.ones(58),
          metadata={},optimizer_updates=1)
    restored=load_command_controller(path,2)
    restored_sequence,_=_condition(restored.model,variant,data,None)
    torch.testing.assert_close(sequence,restored_sequence,atol=0,rtol=0)
    reset=packet(t=8,views=3)
    reset['episode_id'][0]=1
    reset['step_id'][0]=0
    controller.act(reset)
    assert not controller.last_packet['action_history_valid'][0]
    assert controller.last_packet['action_history_valid'][1]
    payload=torch.load(path,weights_only=False)
    payload['command_checkpoint_version']=2
    torch.save(payload,path)
    with pytest.raises(ValueError,match='Incompatible'):
        load_command_controller(path,2)


def test_execution_acknowledgment_rejects_missing_mixed_duplicate_and_future():
    history=ExecutionHistory(2)
    with pytest.raises(ValueError,match='Missing'):
        history.context(packet(t=1))
    observation=packet()
    command=torch.full((2,29),.2)
    recorded=transition(observation,command)
    with pytest.raises(ValueError,match='mixed'):
        history.observe(recorded,-command,observation['episode_id'],observation['step_id'])
    history.observe(recorded,command,observation['episode_id'],observation['step_id'])
    with pytest.raises(ValueError,match='Duplicate'):
        history.observe(recorded,command,observation['episode_id'],observation['step_id'])
    with pytest.raises(ValueError,match='preceding'):
        history.context(packet(t=2))
    config=PolicyConfig(views=2,own_plan_mode='none',action_history_mode='issued')
    model=_make_model('b0',config)
    current=packet(t=1)
    current.update(history.context(current))
    current['previous_command_time_s']=current['time_s'].clone()
    obs={name:current[name][:,None] for name in nominal_keys(2,'issued')}
    obs['valid']=torch.ones(2,1,dtype=torch.bool)
    with pytest.raises(ValueError,match='precede'):
        model.encoder(obs)
    with pytest.raises(ValueError,match='separate'):
        PolicyConfig(action_history_mode='issued',own_plan_mode='private')


def test_native_contract_rejects_control_dr_reset_timeout_changes():
    from visuotactile.contracts import assert_execution_compatible
    metadata=dict(config=dict(action={'gain':1.5},assets={},domain_randomization={'delay':3},obs={},
        reset={'joint':.1},reward={},termination={},precise_assembly={},episode_length_s=10.,decimation=2,
        sim=dict(device='cpu',dt=1/120,gravity=[0,0,-9.81],physx={},physics_material={},use_fabric=True)),
        joint_order=list(range(29)),joint_order_lab=list(range(29)),joint_limits_rad=[[[-1,1]]*29])
    assert_execution_compatible(metadata,copy.deepcopy(metadata))
    for key in ('action','domain_randomization','reset','episode_length_s'):
        changed=copy.deepcopy(metadata)
        changed['config'][key]={} if isinstance(changed['config'][key],dict) else 20.
        with pytest.raises(ValueError,match='Native execution'):
            assert_execution_compatible(metadata,changed)
