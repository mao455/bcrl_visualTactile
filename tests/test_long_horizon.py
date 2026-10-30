"""Causality, boundary coverage and native representation tests for candidate2."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest

from visuotactile.action_reference import command_to_reference, reference_to_command
from visuotactile.long_data import shift_actions, inputs_for, supervision
from visuotactile.models.long_horizon import LongHorizonConfig, LongHorizonPolicy, MaskedHistory
from visuotactile.run_long import pilot_admission


def record(length=145):
    generator = torch.Generator().manual_seed(11)
    limits = torch.tensor([[-2.,2.]]*29)
    command = torch.rand(length,29,generator=generator)*2-1
    time = torch.arange(length,dtype=torch.float64)/60
    before = torch.randn(length,29,generator=generator)*.1
    frames = (length+11)//12
    return dict(version=2,time=time,proprio=torch.randn(length,58,generator=generator),
        command=command,target_before=before,target_after=before+.01,joint_after=before-.01,
        reference=command_to_reference(command,before,limits),
        action=shift_actions(command,-command,time,time,time),
        patches=torch.randn(frames,3,384,18,32,generator=generator).half(),
        depth=torch.zeros(frames,3,2,18,32,dtype=torch.float16),
        camera_time=time[::12,None].expand(-1,3).clone(),camera_valid=torch.ones(frames,3,dtype=torch.bool),
        tactile=torch.zeros(length,2,32))


def test_roundtrip_unclipped_arm_limits_and_nonidentity_hand_order():
    limits = torch.stack((torch.linspace(-2.,-.1,29),torch.linspace(.1,2.,29)),-1)
    commands = torch.stack((torch.zeros(29),torch.ones(29),-torch.ones(29)))
    before = limits[:,1].expand_as(commands).clone()
    refs = command_to_reference(commands,before,limits)
    assert (refs[1,:7] > limits[:7,1]).all()
    torch.testing.assert_close(reference_to_command(refs,before,limits),commands,atol=6e-6,rtol=0)
    permutation = torch.cat((torch.arange(7),torch.arange(28,6,-1)))
    permuted = command_to_reference(commands[:,permutation],before[:,permutation],limits[permutation])
    torch.testing.assert_close(permuted,refs[:,permutation])
    with pytest.raises(ValueError,match='finite'):
        reference_to_command(torch.full((1,29),float('nan')),before[:1],limits)


def test_current_and_future_commands_never_enter_conditions():
    source = record()
    anchor = 24
    original = inputs_for(source,0,25,anchors=torch.tensor([anchor]))
    changed = copy.deepcopy(source)
    changed['command'][anchor:] = .95
    changed['action'] = shift_actions(changed['command'],-changed['command'],source['time'],source['time'],source['time'])
    modified = inputs_for(changed,0,25,anchors=torch.tensor([anchor]))
    for name in original:
        assert torch.equal(original[name],modified[name]),name
    assert not torch.equal(supervision(source,torch.tensor([anchor]))['command'],
                           supervision(changed,torch.tensor([anchor]))['command'])


def test_short_episode_and_two_second_boundaries():
    source = record()
    beginning = inputs_for(source,0,1,anchors=torch.tensor([0]))
    assert beginning['window_valid'].sum() == 1
    assert beginning['window_action_valid'].sum() == 0
    assert beginning['visual_valid'].sum() == 1
    later = inputs_for(source,120,121,True,torch.tensor([120]))
    assert later['window_valid'].all()
    assert later['window_age'][0,0] == 2.
    assert later['visual_age'][0,0,0] == 2.
    tail = supervision(source,torch.tensor([144]))
    assert tail['valid'].sum() == 1
    assert not tail['reference'][:,1:].any()
    assert 'tactile' not in beginning
    assert later['window_tactile'].shape == (1,242,32)


def test_padding_does_not_change_masked_history():
    model = MaskedHistory(4,8)
    real = torch.randn(1,5,4)
    left = torch.cat((torch.randn(1,7,4)*100,real),1)
    encoded = model(real,torch.ones(1,5,dtype=torch.bool))
    padded = model(left,torch.tensor([[False]*7+[True]*5]))
    torch.testing.assert_close(encoded,padded,atol=0,rtol=0)
    assert not model(left,torch.zeros(1,12,dtype=torch.bool)).any()


def test_sample_prediction_preserves_unclipped_reference_at_terminal_noise():
    assert LongHorizonConfig().prediction_type == 'epsilon'
    model = LongHorizonPolicy(LongHorizonConfig(prediction_type='sample'))
    scheduler = model.sampler
    scheduler.set_timesteps(100)
    reference = torch.full((1,64,29),1.25)
    noisy = torch.randn_like(reference)
    for timestep in scheduler.timesteps:
        noisy = scheduler.step(reference,timestep,noisy,eta=0.).prev_sample
    torch.testing.assert_close(noisy,reference,atol=0,rtol=0)
    assert not scheduler.config.clip_sample
    with pytest.raises(ValueError,match='prediction type'):
        LongHorizonConfig(prediction_type='unknown')


def test_pilot_admission_rejects_angle_only_improvement(tmp_path):
    (tmp_path/'causal_acceptance.json').write_text(json.dumps({'passed':True}))
    (tmp_path/'cache_index.json').write_text(json.dumps({'splits':{'train':['a','b','c','d']}}))
    for group in ('G00-new','G10-new'):
        output = tmp_path/(group+'_pilot_sample3300')
        output.mkdir()
        trained = {}
        initial = {}
        for part in ('arm','hand'):
            trained.update({f'command_{part}_rmse':.5,f'zero_command_{part}_rmse':.9,
                            f'reference_{part}_rmse':.03,f'copy_q_{part}_rmse':.07})
            initial[f'command_{part}_rmse'] = 1.2
        (output/'fit.json').write_text(json.dumps({'trained':trained,'initial':initial}))
        (output/'report.json').write_text(json.dumps({'updates':3300,'completed_epochs':300}))
        (output/'config.json').write_text(json.dumps({'model':{'prediction_type':'sample'},
            'train_episodes':['a','b','c','d'],'source_hashes':{}}))
    assert pilot_admission(tmp_path)['passed']
    trained['command_arm_rmse'] = 1.3
    (output/'fit.json').write_text(json.dumps({'trained':trained,'initial':initial}))
    assert not pilot_admission(tmp_path)['passed']


def test_adapter_uses_unchanged_native_delay_and_smoothing():
    path = Path('isaacsimenvs/tasks/play/utils/action_utils.py')
    spec = importlib.util.spec_from_file_location('native_action_reference_test',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    permutation = torch.cat((torch.arange(7),torch.arange(28,6,-1)))
    limits = torch.stack((torch.linspace(-2.,-.2,29),torch.linspace(.2,2.,29)),-1)
    native_limits = limits[permutation]
    original = SimpleNamespace(cfg=SimpleNamespace(
        domain_randomization=SimpleNamespace(use_action_delay=True,action_delay_max=3),
        action=SimpleNamespace(dof_speed_scale=1.5,arm_moving_average=.1,hand_moving_average=.1)),
        device='cpu',step_dt=1/60,num_envs=1,_perm_canon_to_lab=permutation,
        episode_length_buf=torch.zeros(1,dtype=torch.long),_successes=torch.zeros(1,dtype=torch.long),
        _action_queue=torch.zeros(1,3,29),_prev_targets=native_limits[:,1][None].clone(),
        _cur_targets=native_limits[:,1][None].clone(),_arm_lower=native_limits[:7,0],
        _arm_upper=native_limits[:7,1],_hand_lower=native_limits[7:,0],_hand_upper=native_limits[7:,1],
        _arm_joint_ids=torch.arange(7),_hand_joint_ids=torch.arange(7,29))
    adapted = copy.deepcopy(original)
    generator = torch.Generator().manual_seed(51)
    commands = torch.rand(32,29,generator=generator)*2-1
    commands[0] = 0
    commands[1] = 1
    commands[2] = -1
    delays = set()
    original._record_action_trace = lambda action,index,reset: delays.add(int(index[0]))
    for command in commands:
        canonical_target = original._prev_targets[:,permutation]
        reference = command_to_reference(command[None],canonical_target,limits)
        recovered = reference_to_command(reference,adapted._prev_targets[:,permutation],limits)
        rng = torch.get_rng_state()
        module.apply_action_pipeline(original,command[None])
        torch.set_rng_state(rng)
        module.apply_action_pipeline(adapted,recovered)
        torch.testing.assert_close(original._cur_targets,adapted._cur_targets,atol=2e-6,rtol=0)
        original.episode_length_buf += 1
        adapted.episode_length_buf += 1
    assert delays == {0,1,2}


@pytest.mark.parametrize('group',['G00-new','G10-new'])
def test_whole_prefix_chunk_and_online_memory_match(group):
    source = record(25)
    model = LongHorizonPolicy(LongHorizonConfig(group=group)).eval()
    tactile = model.cfg.tactile
    with torch.no_grad():
        full,full_state = model.encode(inputs_for(source,0,25,tactile))
        state = None
        chunks = []
        for start,stop in ((0,12),(12,24),(24,25)):
            result,state = model.encode(inputs_for(source,start,stop,tactile),state)
            chunks.append(result)
        torch.testing.assert_close(full,torch.cat(chunks),atol=3e-6,rtol=3e-5)
        torch.testing.assert_close(full_state,state,atol=3e-6,rtol=3e-5)
        state = None
        for step in range(25):
            output,state = model.encode(inputs_for(source,step,step+1,tactile,torch.tensor([step])),state)
            if step%12 == 0:
                torch.testing.assert_close(output[0],full[step//12],atol=3e-6,rtol=3e-5)
        bad = inputs_for(source,0,1,tactile)
        bad['reward'] = torch.ones(1)
        with pytest.raises(ValueError,match='allowlist'):
            model.encode(bad)
